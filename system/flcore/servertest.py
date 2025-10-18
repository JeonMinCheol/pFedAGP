from flcore.servers.serverbase import Server
from flcore.clients.clienttest import clienttest
from flcore.servers.module.custom_transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict
from utils.loss import SupConLoss

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, dataset, num_classes, embed_dim, num_layers=2, num_heads=8, dropout=0.05, num_clients=10):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = min(num_heads, embed_dim)

        # head_dim 정합성 보정
        if embed_dim % self.num_heads != 0:
            while embed_dim % self.num_heads != 0 and self.num_heads > 1:
                self.num_heads -= 1
            print(f"[Warning] embed_dim {embed_dim} not divisible by num_heads, adjusted to {self.num_heads}")
        self.head_dim = embed_dim // self.num_heads

        self.client_embed = nn.Embedding(num_clients, embed_dim)
        self.class_embed  = nn.Embedding(num_classes, embed_dim)
        self.pos_proj     = nn.Linear(1, embed_dim)
        self.norm         = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        nn.init.normal_(self.pos_proj.weight, std=0.02)
        nn.init.zeros_(self.pos_proj.bias)

    def forward(self, CP: torch.Tensor, round_num: int, client_ids: torch.Tensor, all_labels):
        """
        CP: [M, C, D]
        client_ids: [M]
        all_labels: list[int]
        """
        device = CP.device
        M, C, D = CP.shape

        all_labels = [int(l) for l in (all_labels.tolist() if torch.is_tensor(all_labels) else list(all_labels))]

        assert len(all_labels) == C, f"len(all_labels)={len(all_labels)} vs C={C}"
        assert D == self.embed_dim, f"D={D}, embed_dim={self.embed_dim}"

        # 유효 마스크
        with torch.no_grad():
            mask = (CP.abs().sum(dim=-1) > 0)  # [M, C], bool

        # 기본 임베딩 구성
        id_embed = self.client_embed(client_ids.to(device)).unsqueeze(1).expand(M, C, D)
        class_index = torch.tensor(all_labels, device=device, dtype=torch.long)
        class_embed = self.class_embed(class_index).unsqueeze(0).expand(M, C, D)
        x = self.norm(CP + id_embed + class_embed)  # [M, C, D]

        # ---------------- Transformer Encoding ----------------
        # Transformer는 한 번에 모든 클래스 간 관계를 모델링 가능
        x_trans = self.transformer(x)  # [M, C, D]

        # personalized prototypes 반환
        personalized_list = []
        zero_vec = torch.zeros(D, device=device)
        for i in range(M):
            d = {}
            for j, lbl in enumerate(all_labels):
                d[lbl] = x_trans[i, j] if mask[i, j] else zero_vec
            personalized_list.append(d)

        # 어텐션 엔트로피 계산 (Transformer attention에서 추출 불가하므로, surrogate로 variance 사용)
        ent_loss = self.compute_variance_entropy(x_trans, mask)

        return personalized_list, ent_loss

    def compute_variance_entropy(self, x_trans, mask):
        """
        Transformer는 attention map을 직접 반환하지 않으므로,
        출력 embedding의 분산을 surrogate entropy로 사용
        """
        valid_mask = mask.unsqueeze(-1)
        x_valid = x_trans * valid_mask
        var = x_valid.var(dim=1, unbiased=False).mean()
        entropy_like = -torch.log(var + 1e-8)
        return entropy_like
    
    def compute_attention_entropy(self, attn_weights_list):
        if len(attn_weights_list) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ents = []
        for w in attn_weights_list:
            # 보편 처리: [B, H, S, S] 또는 [H, 1, S, S] 모두 대응
            if w.dim() == 4:
                if w.size(0) == 1:       # [1, H, S, S]
                    p = w.mean(dim=1).squeeze(0)  # [S, S]
                elif w.size(1) == 1:     # [H, 1, S, S]
                    p = w.mean(dim=0).squeeze(0)  # [S, S]
                else:
                    p = w.mean(dim=1).mean(dim=0)  # [S, S]로 수렴
            else:
                p = w
            p = p.mean(dim=0)                 # [S]
            p = p.clamp_min(1e-12)
            ent = -(p * p.log()).sum()
            ents.append(ent)
        return torch.stack(ents).mean()
    
    def log_grad_stats(self):
        print("---- [Aggregator Gradients] ----")
        for name, p in self.named_parameters():
            if p.grad is not None:
                print(f"{name:50s} : {p.grad.abs().mean().item():.6e}")
        print("--------------------------------------")

class test(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clienttest)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.global_protos = {}
        self.client_global_protos = {}
    
        embed_dim = self.clients[0].model.head.in_features
        self.aggregator = ServerPrototypeAggregator(self.dataset, self.num_classes, embed_dim, num_clients=self.num_clients, num_layers=args.num_layers).to(self.device)
        self.agg_opt = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.learning_rate_decay = args.learning_rate_decay
        self.loss_scale = 1
        self.l_ent=0.1
        self.agg_steps= 1
        self.prev_all_labels = None
        self.client_protos_input = None

        self.supcon_loss_fn = SupConLoss(temperature=0.07)
        self.supcon_use_class_mean = False
        self.lambda_supcon = 1  # 예시 가중치

    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate personalized models")
                self.evaluate(round_num)

            ent_losses, client_ids, shared, all_labels = [], [], [], []
            
            if round_num > 0:
                start_time = time.time()
                selected_clients = self.select_clients()

                for client in selected_clients:
                    client.train()
                    cilent_prototype = client.collect_protos()
                    shared.append(cilent_prototype["shared"])
                    client_ids.append(client.id)
                    all_labels.extend(list(cilent_prototype["shared"].keys()))
                    
                client_ids = torch.tensor(client_ids, device=self.device)
                all_labels = sorted(set(all_labels))
                if not all_labels and self.prev_all_labels:
                    all_labels = self.prev_all_labels[:]

                M = len(shared)
                D = list(shared[0].values())[0].shape[-1]
                C = len(all_labels)

                # 클라이언트 프로토타입 텐서 구성
                CP = torch.zeros(M, C, D, device=self.device)
                for i, d in enumerate(shared):
                    for j, lbl in enumerate(all_labels):
                        if lbl in d:
                            CP[i, j] = d[lbl].to(self.device) 

                self.client_protos_input = CP.detach()
                self.prev_all_labels = all_labels[:]

                # -------------------- Step 2. Aggregator 학습 --------------------
                self.aggregator.train()
                self.agg_opt.zero_grad()

                for client in selected_clients:
                    # SupConLoss 계산 (어텐션 representation 학습)
                    personalized_batches = []
                    for client in selected_clients:
                        personalized_list, ent_loss = self.aggregator(self.client_protos_input, round_num, client_ids, self.prev_all_labels)
                        ent_losses.append(ent_loss)
                        for pd in personalized_list:
                            personalized_batches.append(torch.stack(
                                [pd[lbl] for lbl in self.prev_all_labels]))
                    agg_output = torch.stack(personalized_batches, 0)
                    total_loss = self.supcon_loss_fn(agg_output, torch.arange(agg_output.size(0)).to(self.device))
                    total_loss.backward()
                    self.agg_opt.step()
                    torch.cuda.empty_cache()
                    ent_losses.clear()

                with torch.no_grad():
                    mean_shared = CP.mean(dim=0).detach().cpu()
                    self.global_protos = {lbl: mean_shared[j].unsqueeze(0) for j, lbl in enumerate(all_labels)}

                for client in self.clients:
                    client.set_protos(self.global_protos)

                round_time = time.time() - start_time
                self.Budget.append(round_time)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * round_time

                print('-' * 50)
                print(f"[Round {round_num}] time: {round_time:.2f}s | "f"SupCon: {float(total_loss):.4f} | Ent: {float(ent_loss):.4f}")
                print('-' * 50)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            torch.cuda.empty_cache()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def _aggregate_shared(self, uploaded):
        agg = defaultdict(list)
        for d in uploaded:
            for label, proto in d.items():
                agg[label].append(proto)
        return {lbl: torch.stack(p).mean(dim=0) for lbl, p in agg.items()}

    def receive_protos(self, selected_clients):
        uploaded = []
        for c in selected_clients:
            uploaded.append(c.collect_protos())
        return uploaded
    
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
