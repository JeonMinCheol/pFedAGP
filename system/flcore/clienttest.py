import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from utils.loss import SupConLoss

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
        self.global_protos = {}

        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs
        self.ce_lambda = args.ce_lambda
        self.supcon_lambda = args.supcon_lambda

        self.criterion_ce = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(args.tau)
        self.loss_mse = nn.MSELoss()

        # self-attention 모듈
        self.proto_gen = ClientPrototypeGenerator(self.model, num_classes=self.num_classes).to(self.device)

        self.protos_shared = {}
        self.global_shared = {}

    def calculate_loss(self, external_protos, proto_labels):
        self.model.eval()
        proto_map = {lbl: i for i, lbl in enumerate(proto_labels)}

        total_loss = 0.0
        num_batches = 0

        for x, y in self.load_train_data():
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.no_grad():
                rep = self.model.base(x)  # [B, D]
                logits = self.model.head(rep)

            idx = [i for i, lbl in enumerate(y) if lbl.item() in proto_map]
            if not idx:
                continue

            reps = rep[idx]
            tix = [proto_map[y[i].item()] for i in idx]
            tgt = external_protos[tix]

            # CE + Proto Loss
            loss_cls = self.loss(logits[idx], y[idx])
            loss_proto = self.loss_mse(reps, tgt)
            total_loss += (loss_cls +  loss_proto).item()
            num_batches += 1

        if num_batches == 0:
            return 0.0

        mean_loss = total_loss / num_batches
        return mean_loss  
    
    def train(self, external_protos=None, proto_labels=None):
        trainloader = self.load_train_data()
        self.model.train()

        # proto grad 누적용 버퍼 (없으면 None)
        proto_grad_accum = None
        start_time = time.time()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                rep = self.model.base(x)
                logits = self.model.head(rep)

                # 1) CE는 기존대로 클라 모델만 업데이트
                ce_loss = self.criterion_ce(logits, y)
                self.optimizer.zero_grad()
                ce_loss.backward()   # ✅ 모델만 업데이트
                self.optimizer.step()

                # 2) 프로토타입 쪽: 모델 그래프 끊고, external_protos에 대한 grad만 계산
                if external_protos is not None and proto_labels is not None:
                    proto_map = {lbl: i for i, lbl in enumerate(proto_labels)}
                    valid_idx = [i for i, lbl in enumerate(y) if int(lbl.item()) in proto_map]
                    if valid_idx:
                        # 모델 경로는 끊고(rep.detach), external_protos로만 미분 걸리게
                        rep_detached = rep.detach()[valid_idx]                     # [Nv, D]
                        tix = [proto_map[int(y[i].item())] for i in valid_idx]
                        target = external_protos[tix]                               # [Nv, D] (requires_grad=True)

                        # proto loss는 external_protos로만 의존하도록 구성
                        proto_loss = F.mse_loss(rep_detached, target)

                        # external_protos에 대한 gradient를 직접 얻는다
                        grad_wrt_target = torch.autograd.grad(
                            proto_loss, target, retain_graph=True, create_graph=False
                        )[0]                                                        # [Nv, D]

                        # 이제 클래스별로 모아 C×D 형태로 평균 grad를 만든다
                        C = external_protos.shape[0]
                        D = external_protos.shape[1]
                        grad_c_d = torch.zeros(C, D, device=external_protos.device)

                        # 동일 클래스에 여러 샘플이 있으면 평균
                        for g, cls_idx in zip(grad_wrt_target, tix):
                            grad_c_d[cls_idx] += g
                        # 샘플 수로 나누기
                        count = torch.bincount(torch.tensor(tix, device=grad_c_d.device), minlength=C).clamp_min(1).float()
                        grad_c_d = grad_c_d / count.unsqueeze(-1)

                        # 누적
                        if proto_grad_accum is None:
                            proto_grad_accum = grad_c_d
                        else:
                            proto_grad_accum = proto_grad_accum + grad_c_d

                if self.learning_rate_decay:
                    if hasattr(self, 'learning_rate_scheduler'): self.learning_rate_scheduler.step()

        # 시간·메트릭 저장
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # print(f"Client {self.id}, time: {time.time() - start_time}")
        if proto_grad_accum is not None:
            proto_grad_accum = proto_grad_accum / float(self.local_epochs)
            return proto_grad_accum.detach()

        # 아무 것도 못 모았으면 None
        return None

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

            # 어텐션 기반 가중치 (cosine similarity)
            sim = F.cosine_similarity(g, p, dim=-1, eps=1e-8)
            alpha = torch.sigmoid(sim).unsqueeze(-1)  # [0,1]

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

        return {"shared": shared, "personal": personal}

    def decompose_protos(self, local_protos):
        """
        InstanceNorm 기반 프로토타입 분리:
        각 프로토타입 벡터 p를 평균 μ와 표준편차 σ로 정규화하여
        shared(μ), personal((p-μ)/σ)로 분리.
        """
        shared = {}
        personal = {}
        eps = 1e-6

        for lbl, proto in local_protos.items():
            # [D] 혹은 [1, D] 형태로 가정
            if proto.dim() == 2:
                proto = proto.squeeze(0)

            mean = proto.mean(dim=0, keepdim=True)
            std = proto.std(dim=0, keepdim=True) + eps

            # 공통(shared): 평균 구조
            shared[lbl] = mean.expand_as(proto)

            # 개인(personal): Instance-normalized residual
            personal[lbl] = (proto - mean) / std

        return shared, personal

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

                # incorporate prototype regularization
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
