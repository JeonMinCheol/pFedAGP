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
from utils.loss import SupConLoss

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, dataset, num_classes, embed_dim, supcon_lambda, num_heads=4, dropout=0.05):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.supcon_lambda = supcon_lambda
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, client_protos: torch.Tensor, round_num: int, client_idx: int, all_labels):
        M, C, D = client_protos.shape
        device = client_protos.device

        assert D % self.num_heads == 0, f"embed_dim {D} must be divisible by num_heads {self.num_heads}"

        protos_norm = self.norm(client_protos.view(-1, D)).view(M, C, D)
        protos_trans = torch.zeros_like(protos_norm)

        for c in range(C):
            class_seq = protos_norm[:, c, :].unsqueeze(0)         # [1, M, D]
            class_trans = self.transformer(class_seq).squeeze(0)  # [M, D]
            protos_trans[:, c, :] = class_trans

        q_all = self.global_query[all_labels].view(C, self.num_heads, self.head_dim).to(device) # [C, self.num_heads, self.head_dim]
        kv_all = protos_trans.permute(1, 0, 2).contiguous().view(C, M, self.num_heads, self.head_dim) # [C, M, self.num_heads, self.head_dim]

        # Scaled dot-product
        scores = torch.einsum('chd,cmhd->chm', q_all, kv_all) / math.sqrt(self.head_dim)  
        attn_weights = F.softmax(scores, dim=-1)                               
        attn_out = torch.einsum('chm,cmhd->chd', attn_weights, kv_all).reshape(C, D)           

        # logging
        if False:
            if round_num % 10 == 0:
                os.makedirs(f"attention_heatmaps/{self.dataset}", exist_ok=True)

                _, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(attn_weights[:, 0, :].detach().cpu().numpy(), cmap="YlGnBu", cbar=True, ax=ax, vmin=0, vmax=1)
                ax.set_title(f'Attention Heatmap (Round {round_num}, Client {client_idx})')
                ax.set_xlabel('Client Index (M)')
                ax.set_ylabel('Class Index (C)')
                plt.tight_layout()
                plt.savefig(f'attention_heatmaps/{self.dataset}/round{round_num}_client{client_idx}.png')
                plt.close()

        output = {c: attn_out[c] for c in range(C)}
        for lbl in all_labels:
            if lbl not in output:
                output[lbl] = torch.zeros(self.embed_dim, device=device)

        return output

class test(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clienttest)

        self.agg_steps = args.agg_steps

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.global_protos = {}
        self.client_global_protos = {}
        self.uploaded_memories = []
        self.download_memories = []

        embed_dim = self.clients[0].model.head.in_features
        self.aggregator = ServerPrototypeAggregator(self.dataset, self.num_classes, embed_dim, args.supcon_lambda).to(self.device)
        self.agg_opt = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.learning_rate_decay = args.learning_rate_decay

    def train(self):
        client_protos_input = None
        for round_num in range(self.global_rounds+1):
            start_time = time.time()
            selected_clients = self.select_clients()

            if round_num%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(round_num)

            # ----------------- Step 1: 클라이언트 로컬 학습 및 그래디언트 수신 -----------------
            proto_grads_list = []
            personalized_protos_list = []
            
            # aggregator 학습 모드
            self.aggregator.train()
            self.agg_opt.zero_grad()

            for idx, client in enumerate(selected_clients):
                # 첫 라운드가 아닐 때만 aggregator를 통해 개인화 프로토타입 생성
                if client_protos_input is not None:
                    # 집계기가 개인화 프로토타입 생성 (그래프 유지)
                    personalized_dict = self.aggregator(client_protos_input, round_num, idx, all_labels)
                    
                    # 딕셔너리를 정렬된 텐서로 변환하여 클라이언트에 전달
                    personalized_tensor = torch.stack([personalized_dict[lbl] for lbl in all_labels])

                    # 클라이언트 학습 및 프로토타입에 대한 그래디언트 수신
                    proto_grad = client.train(external_protos=personalized_tensor, proto_labels=all_labels)
                    if proto_grad is not None:
                        proto_grads_list.append(proto_grad)
                        personalized_protos_list.append(personalized_tensor)
                else:
                    # 첫 라운드는 aggregator 학습 없이 로컬 학습만 진행
                    client.train()

             # ----------------- Step 2: Aggregator 업데이트 -----------------
            if proto_grads_list:
                # 여러 클라의 grad 평균
                avg_grad = torch.stack(proto_grads_list).mean(dim=0)           # [C, D]

                # personalized_tensor도 하나로 정해 사용 (여기선 첫 번째 것으로)
                target_tensor = personalized_protos_list[0]                     # [C, D], requires_grad=True

                # 🔥 서버에서 단 한 번 autograd.backward 실행
                self.agg_opt.zero_grad()
                torch.autograd.backward(target_tensor, grad_tensors=avg_grad)  # ✅ 여기서만 그래프 소비
                self.agg_opt.step()

            uploaded = self.receive_protos(selected_clients)
            all_labels = sorted(set().union(*[set(d["shared"].keys()) for d in uploaded]))
            M, C = len(uploaded), len(all_labels)
            D = list(uploaded[0]["shared"].values())[0].shape[-1]

            CP = torch.zeros(M, C, D, device=self.device)
            for i, d in enumerate(uploaded):
                for j, lbl in enumerate(all_labels):
                    if lbl in d:
                        CP[i, j] = d[lbl].to(self.device)

            client_protos_input = CP.detach() # 다음 라운드를 위해 저장

            # 각 클라별 personalized shared 생성
            for idx, client in enumerate(selected_clients):
                for step in range(self.agg_steps):
                    self.agg_opt.zero_grad()
                    personalized = self.aggregator(CP, round_num, idx, all_labels) # client_protos: torch.Tensor, round_num: int, client_idx: int, all_labels):

                with torch.no_grad():
                    protos = {}
                    for j in range(C):
                        protos[all_labels[j]] = personalized[j].cpu()
                    client.set_protos(protos)
                    self.client_global_protos[client.id] = protos

            # (2) 공통(global) shared 생성 — 모든 클라 평균
            with torch.no_grad():
                mean_shared = CP.mean(dim=0).detach().cpu()  # [C, D]
                self.global_protos = {
                    lbl: mean_shared[j].unsqueeze(0)  # [1, D] 형태로 유지
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
            print(f"[Round {self.current_round}]  attn_loss={avg_grad.item():.6f}, time={self.Budget[-1]:.2f}s")
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
