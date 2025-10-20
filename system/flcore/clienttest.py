import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client

class ClientPrototypeGenerator(nn.Module):
    """
    - 클래스별 personal embedding 테이블(학습 파라미터)
    - 배치 임베딩에서 shared(mean) 산출
    - full = shared + personal
    """
    def __init__(self, model: nn.Module, num_classes: int, dropout: float = 0.1):
        super().__init__()
        embed_dim = model.head.in_features
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 클래스별 개인화 임베딩 파라미터
        self.personal_table = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.zeros_(self.personal_table)

    def forward(self, emb_list: dict) -> dict:
        device = self.personal_table.device
        protos = {}

        for cls_id, embs_per_class in emb_list.items():
            if len(embs_per_class) == 0:
                continue

            embs = torch.stack(embs_per_class).to(device)
            embs = self.dropout(self.norm(embs))

            shared = embs.mean(dim=0, keepdim=True)
            personal = self.personal_table[cls_id].unsqueeze(0)
            full = shared + personal

            protos[cls_id] = {"shared": shared, "personal": personal, "full": full}

        return protos

class clienttest(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.id = id
        self.device = torch.device(args.device)
        self.global_shared = {}
        self.global_protos = {}
        self.personalized_protos = {}
        self.local_personal_protos = {}

        # ===== NEW: prototype-regularization hyperparams =====
        self.lambda_proto_pull = getattr(args, "lambda_proto_pull", 0.3)  # CE에 더하는 가중치
        self.lambda_proto_ce   = getattr(args, "lambda_proto_ce",   0.1)  # (옵션) 프로토타입 분류 보조 로스
        self.proto_tau         = getattr(args, "proto_tau", 10.0)         # distance→logit 스케일

        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs

        self.criterion_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

        # self-attention 모듈
        self.proto_gen = ClientPrototypeGenerator(self.model, num_classes=self.num_classes).to(self.device)
    
    # 추가
    def _merge_personalized(self, g, p):
        g, p = g.view(1,-1), p.view(1,-1)
        z = torch.cat([g, p, (g-p).abs(), g*p], dim=-1)
        alpha = torch.sigmoid(self.alpha_gate(z))  # [1,1]
        return alpha * g + (1 - alpha) * p

    
    def _get_proto_for_label(self, y_label: int):
        """
        personalized_protos > global_protos > None 순서로 프로토를 반환
        반환 텐서는 [D] or [1,D] 형태 모두 허용
        """
        dev = self.device
        if hasattr(self, "personalized_protos") and self.personalized_protos:
            p = self.personalized_protos.get(y_label)
            if isinstance(p, torch.Tensor):
                return p.to(dev).squeeze(0).detach()
        if isinstance(self.global_protos, dict) and len(self.global_protos) > 0:
            g = self.global_protos.get(y_label)
            if isinstance(g, torch.Tensor):
                return g.to(dev).squeeze(0).detach()
        return None

    def _build_proto_logits(self, reps: torch.Tensor, available_protos: dict):
        """
        reps: [B, D]
        available_protos: {cls_id: [1, D] or [D]}
        return: logits [B, C_avail] (C_avail = len(available_protos))
        방식: -||x - p_c||^2 * tau  (Prototypical classifier)
        """
        if not available_protos:
            return None, None
        dev = reps.device
        classes = sorted(available_protos.keys())
        protos = [available_protos[c].to(dev).squeeze(0).detach() for c in classes]  # [C, D]
        P = torch.stack(protos, dim=0)  # [C, D]
        # dist^2 = ||x||^2 + ||p||^2 - 2 x·p
        x2 = (reps**2).sum(dim=1, keepdim=True)        # [B,1]
        p2 = (P**2).sum(dim=1, keepdim=True).T         # [1,C]
        xp = reps @ P.T                                # [B,C]
        d2 = x2 + p2 - 2 * xp                          # [B,C]
        logits = - self.proto_tau * d2                 # [B,C]
        return logits, classes
    
    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                reps = self.model.base(x)           # [B, D]
                logits = self.model.head(reps)      # [B, K]

                # 1) 기본 CE
                ce_loss = self.criterion_ce(logits, y)

                # 2) (NEW) Prototype Pull Loss: 각 샘플을 (personalized/global) prototype 쪽으로 당김
                #    - cosine pull 또는 L2 pull 중 택1 (여기선 cosine 기반을 사용)
                pull_losses = []
                with torch.no_grad():
                    # 미리 배치의 target 프로토 모아두기 (존재하는 것만)
                    target_protos = []
                    valid_mask = []
                    for yy in y.tolist():
                        p = self._get_proto_for_label(yy)
                        if p is None:
                            valid_mask.append(False)
                            target_protos.append(torch.zeros_like(reps[0]))
                        else:
                            valid_mask.append(True)
                            target_protos.append(p)
                    target_protos = torch.stack(target_protos, dim=0)  # [B, D]
                    valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)

                if valid_mask.any():
                    # cosine pull: 1 - cos(rep, proto)
                    rep_n = F.normalize(reps[valid_mask], dim=-1)
                    proto_n = F.normalize(target_protos[valid_mask], dim=-1)
                    cos_pull = 1.0 - (rep_n * proto_n).sum(dim=-1)     # [Bv]
                    pull_loss = cos_pull.mean()
                    pull_losses.append(pull_loss)

                # 3) (OPTIONAL) Prototype-based auxiliary CE (protocluster classifier)
                aux_loss = torch.tensor(0.0, device=self.device)
                if self.lambda_proto_ce > 0 and hasattr(self, "personalized_protos") and self.personalized_protos:
                    # 현재 클라가 가진 personalized 프로토만으로 보조 분류
                    plogits, classes = self._build_proto_logits(reps, self.personalized_protos)
                    if plogits is not None:
                        # y를 해당 classes 인덱스로 매핑
                        class_to_idx = {c: i for i, c in enumerate(classes)}
                        y_mapped = []
                        for yy in y.tolist():
                            if yy in class_to_idx:
                                y_mapped.append(class_to_idx[yy])
                            else:
                                y_mapped.append(-1)  # 없는 클래스는 제외
                        y_mapped = torch.tensor(y_mapped, device=self.device)
                        mask = y_mapped >= 0
                        if mask.any():
                            aux_loss = self.criterion_ce(plogits[mask], y_mapped[mask])

                # 4) 최종 손실 결합
                total_loss = ce_loss
                if pull_losses:
                    total_loss = total_loss + self.lambda_proto_pull * torch.stack(pull_losses).mean()
                if self.lambda_proto_ce > 0 and aux_loss.requires_grad or aux_loss.item() != 0.0:
                    total_loss = total_loss + self.lambda_proto_ce * aux_loss

                # 5) 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if self.learning_rate_decay and hasattr(self, 'learning_rate_scheduler'):
                    self.learning_rate_scheduler.step()

        # 시간·메트릭 저장
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # print(f"Client {self.id}, time: {time.time() - start_time}")

    def set_protos(self, global_shared):
        """
        Attention 기반 personalized prototype 생성:
        global_protos (shared)와 local_personal_protos를
        라벨 일치하는 부분만 attention-weighted merge.
        """
        self.global_protos = global_shared
        self.personalized_protos = {}

        if not hasattr(self, "local_personal_protos") or not self.local_personal_protos:
            return

        # ✅ 공통 라벨만 대상으로 merge
        shared_labels = set(self.global_protos.keys()) & set(self.local_personal_protos.keys())
        if not shared_labels:
            print(f"[Warning][Client {self.id}] No shared labels between local and global protos.")
            return

        for lbl in shared_labels:
            g = self.global_protos[lbl].to(self.device)
            p = self.local_personal_protos[lbl].to(self.device)

            # 변경한 부분 cosine simillarity -> 게이팅
            alpha = self._merge_personalized(g,p)

            personalized = alpha * g + (1 - alpha) * p
            self.personalized_protos[lbl] = personalized.detach().cpu()

        # ✅ 나머지 (없는 클래스) 보정
        missing_labels = set(self.global_protos.keys()) - shared_labels
        for lbl in missing_labels:
            self.personalized_protos[lbl] = self.global_protos[lbl].detach().cpu()

    def collect_protos(self):
        self.model.eval()
        trainloader = self.load_train_data()
        emb_list = defaultdict(list)

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                reps = self.model.base(x)  # [B, D]
                for i, label in enumerate(y):
                    emb_list[int(label)].append(reps[i])

        # ✅ 이제 emb_list를 전달
        local_proto_dict = self.proto_gen(emb_list)
        
        # 구조 분리
        shared = {k: v["shared"] for k, v in local_proto_dict.items()}
        personal = {k: v["personal"] for k, v in local_proto_dict.items()}
        full = {k: v["full"] for k, v in local_proto_dict.items()}

        # 저장해두기
        self.local_personal_protos = personal
        self.protos = full  # 기본 full 구조 사용

        return {"full":full, "shared": shared, "personal": personal}
    
    def decompose_with_global(self, global_shared):
        """
        global_shared: 서버가 보낸 클래스별 global prototype
        local_personal_protos: 클라이언트의 raw local prototype
        """
        self.local_style = {}
        self.local_content = {}
        eps = 1e-6

        for lbl, local_p in self.local_personal_protos.items():
            if lbl not in global_shared:
                continue
            g = global_shared[lbl].to(self.device)
            p = local_p.to(self.device)

            # 1️⃣ Projection: style factor = residual orthogonal to global
            proj = (p @ g.T) / (g @ g.T + eps) * g
            style = p - proj

            # 2️⃣ Normalization
            style = style / (style.norm(dim=-1, keepdim=True) + eps)

            self.local_content[lbl] = g.detach()
            self.local_style[lbl] = style.detach()

        return self.local_content, self.local_style

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = [x[0].to(self.device, non_blocking=True)]
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if isinstance(self.global_protos, dict) and len(self.global_protos) > 0:
                    proto_new = rep.detach().clone()
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        p = self.global_protos.get(y_c)
                        if isinstance(p, torch.Tensor):
                            proto_new[i] = p.to(self.device)
                    loss += self.loss_mse(proto_new, rep)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return losses, train_num, train_correct
