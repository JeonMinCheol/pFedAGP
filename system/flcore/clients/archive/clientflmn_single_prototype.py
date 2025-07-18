import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=2)

    def forward(self, features, labels):
        """
        features: [B, n_views, D]
        labels:   [B]
        """
        device = features.device
        B, n_views, D = features.shape
        features = F.normalize(features, dim=2)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*n_views, D]
        anchor_feature = contrast_feature  # [B*n_views, D]

        # 🔥 확장된 라벨 만들기: [B * n_views]
        labels = labels.repeat(n_views)  # 원래 [B] → [B*n_views]

        labels = labels.view(-1)
        mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask.fill_diagonal_(False)

        # logits
        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )  # [B*n_views, B*n_views]

        eye = torch.eye(mask.shape[0], device=device, dtype=torch.bool)
        # ② 반전하여 self-contrast 제거용 마스크 생성
        logits_mask = ~eye
        # ③ AND 연산으로 self-contrast 위치 제거
        mask = mask & logits_mask

        # log-softmax
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

class ClientPrototypeGenerator(nn.Module):
    """
    Generate local prototypes via self-attention weighted sum, parallelized across classes.
    """
    def __init__(self, model: nn.Module, num_heads: int = 4, dropout: float = 0.4):
        super().__init__()
        embed_dim = model.head.in_features
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb_list: dict) -> dict:
        """
        단순 평균 기반 클래스 프로토타입 생성 (Self-Attention 제거)

        Args:
            emb_list (dict): mapping class_id -> list of embeddings (Tensor of shape [embed_dim])
        Returns:
            agg_protos (dict): mapping class_id -> average prototype Tensor
        """
        device = next(self.norm.parameters()).device
        prototype = {}

        for cls_id, emb_list_per_class in emb_list.items():
            if len(emb_list_per_class) == 0:
                continue

            embs = torch.stack(emb_list_per_class).to(device)  # [N_c, D]
            embs = self.dropout(self.norm(embs))               # normalize + dropout
            proto = embs.mean(dim=0, keepdim=True)             # [1, D]
            prototype[cls_id] = proto

        if not prototype:
            raise ValueError("No valid prototypes to aggregate.")
            
        return prototype  # [C, D]

class clientFLMN(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.device = torch.device(args.device)
        self.id = id
        self.protos = {}
        self.mask = None
        self._idx_map = None
        self.global_protos = None
        self.cl_lamda=args.cl_lamda
        self.cne_lamda=args.cne_lamda
        self.attn_weight = args.attn_weight

        # self-attention 모듈
        self.proto_gen = ClientPrototypeGenerator(self.model).to(self.device)
        self.loss_mse = nn.MSELoss()
        self.supcon_loss = SupConLoss(args.tau).to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        max_epochs = self.local_epochs
        if self.train_slow:
            max_epochs = np.random.randint(1, max_epochs//2)

        for epoch in range(max_epochs):
            for ii, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)

                # 1) Forward: representation + classification
                rep    = self.model.base(x)          # [B, D], requires_grad=True
                logits = self.model.head(rep)        # [B, C]
                loss_cl = self.loss(logits, y) * self.cl_lamda

                loss_cne = rep.new_zeros(1)
                if self.global_protos is not None and self.protos is not None:
                    # 배치 내에서 global_protos 에 존재하는 샘플만 골라내기
                    self._idx_map = {c:i for i,c in enumerate(self.protos.keys())}
                    self.mask = torch.tensor(
                        [int(lbl) in self._idx_map for lbl in y],
                        device=self.device
                    )

                    if self.mask.any():
                        rep_valid = rep[self.mask].detach()    # [B', D]
                        labels_valid= torch.tensor(
                            [self._idx_map[int(lbl)] for i,lbl in enumerate(y) if self.mask[i]],
                            device=self.device
                        )
                        proto_valid = torch.stack([self.protos[int(lbl)] for i,lbl in enumerate(y) if self.mask[i]], dim=0).to(self.device).detach()
                        # features: [B', 2, D]
                        features = torch.cat([ rep_valid.unsqueeze(1), proto_valid.unsqueeze(1) ], dim=1)
                        loss_cne = self.supcon_loss(features, labels_valid) * self.cne_lamda
                
                total_loss = loss_cl + loss_cne
                
                # 1) global update
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                rep_det = rep.detach()
                batch_emb = defaultdict(list)
                for i, lbl in enumerate(y):
                    batch_emb[int(lbl)].append(rep_det[i])

                self.prototype = self.proto_gen(batch_emb)

            # optional LR decay for model
            if self.learning_rate_decay:
                if hasattr(self, 'learning_rate_scheduler'): self.learning_rate_scheduler.step()

        # 시간·메트릭 저장
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.save_item(self.model.state_dict(), str(self.dirchlet), self.pt_path)
        print(f"Client {self.id}, time: {time.time() - start_time}")

    def set_protos(self, global_protos):
        self.global_protos = global_protos
        self._proto_mat = None
        self._idx_map = None

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        emb_list = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    emb_list[y_c].append(rep[i, :].detach().data)

        # generate self-attention based prototypes
        self.prototype = self.proto_gen(emb_list)

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