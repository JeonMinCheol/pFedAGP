from flcore.servers.serverbase import Server
from flcore.clients.clienttest import clienttest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import seaborn as sns 
# from utils.loss import SupConLoss

from flcore.servers.module.content_style import *
from flcore.servers.module.disentangler import *

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, dataset, num_classes, embed_dim, num_heads=4, dropout=0.05):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.embed_dim = int(embed_dim)
        self.num_heads = max(1, int(num_heads))
        self.head_dim = self.embed_dim // self.num_heads

        self.norm = nn.LayerNorm(self.embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        self.global_query = nn.Parameter(torch.randn(num_classes, self.embed_dim))

    def forward(self, client_protos: torch.Tensor, round_num: int, client_idx: int, all_labels):
        M, C, D = client_protos.shape
        device = client_protos.device
        assert D % self.num_heads == 0, f"embed_dim {D} must be divisible by num_heads {self.num_heads}"

        # (1) Transformer로 클래스별 contextual encoding
        protos_norm = self.norm(client_protos.view(-1, D)).view(M, C, D)
        protos_trans = torch.zeros_like(protos_norm)
        for c in range(C):
            class_seq = protos_norm[:, c, :].unsqueeze(0)        # [1, M, D]
            class_trans = self.transformer(class_seq).squeeze(0) # [M, D]
            protos_trans[:, c, :] = class_trans

        # (2) Attention 계산
        q_all = self.global_query[all_labels].view(C, self.num_heads, self.head_dim).to(device)
        kv_all = protos_trans.permute(1, 0, 2).contiguous().view(C, M, self.num_heads, self.head_dim)

        scores = torch.einsum('chd,cmhd->chm', q_all, kv_all) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        attn_out = torch.einsum('chm,cmhd->chd', attn_weights, kv_all).reshape(C, D)  

        attn_entropy = -(attn_weights * (attn_weights.clamp_min(1e-8).log())).sum(dim=-1).mean()
        attn_loss = attn_entropy

        output = {int(lbl): attn_out[j] for j, lbl in enumerate(all_labels)}
        for lbl in all_labels:
            if lbl not in output or not isinstance(output[lbl], torch.Tensor):
                output[lbl] = torch.zeros(self.embed_dim, device=device)

        return output, attn_loss


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

        self.disentangler = ProtoDisentangler(embed_dim).to(self.device)
        self.disentangler.configure_heads(num_classes=self.num_classes, num_domains=self.num_clients)
        self.disent_opt = torch.optim.Adam(self.disentangler.parameters(), lr=self.learning_rate)
        self.disentangler.to(self.device)

        self.aggregator = ServerPrototypeAggregator(self.dataset, self.num_classes, self.disentangler.k).to(self.device)
        self.agg_opt = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.learning_rate_decay = args.learning_rate_decay
        self.client_id_to_idx = {cid: idx for idx, cid in enumerate(sorted([c.id for c in self.clients]))}

    def train(self):
        client_protos_input = None
        zs_cache = None  
        for round_num in range(self.global_rounds+1):
            start_time = time.time()
            selected_clients = self.select_clients()

            if round_num%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(round_num)

            # ----------------- Step 1: 클라 학습/집계기 gradient 수집 -----------------
            proto_grads_list = []
            personalized_protos_list = []
            
            self.aggregator.train()
            self.agg_opt.zero_grad()

            # (A) client_protos_input 이 있을 경우 → 집계기 출력으로 proto_grad 수집
            for idx, client in enumerate(selected_clients):
                if client_protos_input is not None:
                    personalized_dict, ent_loss = self.aggregator(client_protos_input, round_num, idx, all_labels)
                    # dict -> [C,D]
                    perC = torch.stack([personalized_dict[lbl] for lbl in all_labels])   # [C, latent_dim]
                    zs_i = zs_cache[idx]                                                 # [C, latent_dim]
                    decoded = self.disentangler.decode(perC, zs_i)                       # [C, D_original]
                    personalized_tensor = decoded

                    proto_grad = client.train(external_protos=personalized_tensor, proto_labels=all_labels)
                    if proto_grad is not None:
                        proto_grads_list.append(proto_grad)
                        personalized_protos_list.append(personalized_tensor)
                else:
                    client.train()

            # (B) 집계기 업데이트 (평균 grad → 한 번만 backward)
            if proto_grads_list:
                avg_grad = torch.stack(proto_grads_list).mean(dim=0)   # [C,D]
                target_tensor = personalized_protos_list[0]            # [C,D], requires_grad=True
                self.agg_opt.zero_grad()
                torch.autograd.backward(target_tensor, grad_tensors=avg_grad)
                self.agg_opt.step()

            # ----------------- Step 2: 클라 로컬 프로토 수집 → CP -----------------
            uploaded = self.receive_protos(selected_clients)
            all_labels = sorted(set().union(*[set(d["shared"].keys()) for d in uploaded]))
            if len(all_labels) == 0:
                continue

            M, C = len(uploaded), len(all_labels)
            D = list(uploaded[0]["shared"].values())[0].shape[-1]

            CP = torch.zeros(M, C, D, device=self.device)
            for i, d in enumerate(uploaded):
                for j, lbl in enumerate(all_labels):
                    if lbl in d:
                        CP[i, j] = d[lbl].to(self.device)

            selected_domain_ids = torch.tensor([self.client_id_to_idx[c.id] for c in selected_clients], device=self.device)
            dis_loss = disentangle_train_step(
                model=self.disentangler,
                optimizer=self.disent_opt,
                CP=CP,
                domain_ids=selected_domain_ids,
                steps=1,
                round_idx=round_num
            )

            self.disentangler.eval()
            with torch.no_grad():
                zc, zs, _ = self.disentangler(CP.view(-1, D))
                zc = zc.view(M, C, -1)
                zs = zs.view(M, C, -1)

            # === 기존 client_protos_input 을 zc 로 교체 ===
            client_protos_input = zc.detach()
            zs_cache = zs.detach()  # 나중에 decode에 사용

            # ----------------- Step 4: 각 클라 personalized 생성/전송 -----------------
            for idx, client in enumerate(selected_clients):
                # 집계기는 그대로 사용(고정). 입력만 zc로 바뀜.
                personalized, ent_loss = self.aggregator(zc, round_num, idx, all_labels)  # dict: {class: [D]}

                with torch.no_grad():
                    # 최종 내려보낼 때만 style 결합하여 decode
                    # personalized(dict) → [C,D]
                    perC = torch.stack([personalized[lbl] for lbl in all_labels])     # [C,D]
                    zs_i = zs_cache[idx]                                              # [C,D]
                    decoded = self.disentangler.decode(perC, zs_i)                    # [C,D]

                    protos = {}
                    for j in range(C):
                        protos[all_labels[j]] = decoded[j].cpu()
                    client.set_protos(protos)
                    self.client_global_protos[client.id] = protos

            # (2) 공통(global) shared 생성 — 모든 클라 평균 (원래대로)
            with torch.no_grad():
                mean_shared = CP.mean(dim=0).detach().cpu()  # [C, D]
                self.global_protos = {
                    lbl: mean_shared[j].unsqueeze(0)
                    for j, lbl in enumerate(all_labels)
                }

            for client in self.clients:
                if client not in selected_clients:
                    client.set_protos(self.global_protos)

            round_time = time.time() - start_time
            self.Budget.append(round_time)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * round_time

            print('-' * 50)
            print(f"[Round {self.current_round}]  disentangle_loss={dis_loss:.4f} attn_loss={ent_loss.item():.6f}, time={self.Budget[-1]:.2f}s")
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
