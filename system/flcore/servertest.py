import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from flcore.servers.serverbase import Server
from flcore.clients.clienttest import clienttest
from utils.loss import SupConLoss

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, dataset, num_classes, embed_dim, num_layers=1, num_heads=8, dropout=0.05, num_clients=10):
        super().__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = min(num_heads, embed_dim)
        self.client_embed = nn.Embedding(num_clients, embed_dim)
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, CP, client_ids, all_labels, return_all=False):
        device = CP.device
        M, C, D = CP.shape
        id_embed = self.client_embed(client_ids).unsqueeze(1).expand(M, C, D)
        class_embed = self.class_embed(torch.tensor(all_labels, device=device)).unsqueeze(0).expand(M, C, D)
        x = self.norm(CP + id_embed + class_embed)
        x_trans = self.transformer(x)  # [M, C, D]
        if return_all:
            return x_trans
        global_proto = x_trans.mean(dim=0)  # [C, D]
        return {lbl: global_proto[j] for j, lbl in enumerate(all_labels)}
    
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

        embed_dim = self.clients[0].model.head.in_features
        self.aggregator = ServerPrototypeAggregator(
            self.dataset, self.num_classes, embed_dim,
            num_clients=self.num_clients, num_layers=args.num_layers
        ).to(self.device)

        self.supcon_loss_fn = SupConLoss(temperature=0.07)
        self.agg_opt = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.global_protos = {}
        self.prev_all_labels = None

    def train(self):
        for round_num in range(self.global_rounds + 1):
            if round_num % self.eval_gap == 0:
                print(f"\n[Round {round_num}] Evaluate models")
                self.evaluate(round_num)

            selected_clients = self.select_clients()
            shared_list, client_ids, all_labels = [], [], []

            # -------------------- Step 1. collect shared prototypes --------------------
            for client in selected_clients:
                client.train()
                cp = client.collect_protos()
                shared_list.append(cp["shared"])
                client_ids.append(client.id)
                all_labels.extend(list(cp["shared"].keys()))

            all_labels = sorted(set(all_labels))
            self.prev_all_labels = all_labels
            M = len(shared_list)
            D = list(shared_list[0].values())[0].shape[-1]
            C = len(all_labels)

            CP = torch.zeros(M, C, D, device=self.device)
            for i, d in enumerate(shared_list):
                for j, lbl in enumerate(all_labels):
                    if lbl in d:
                        CP[i, j] = d[lbl].to(self.device)

            client_ids = torch.tensor(client_ids, device=self.device, dtype=torch.long)

            # -------------------- Step 2. aggregate shared prototypes --------------------
            self.aggregator.train()
            self.agg_opt.zero_grad()

            x_trans = self.aggregator(CP, client_ids, all_labels, return_all=True)  # [M, C, D]

            # (2) SupConLoss 계산 (client-level contrast)
            # flatten: (M*C, D)
            B = x_trans.shape[0] * x_trans.shape[1]
            emb = x_trans.view(B, -1).unsqueeze(1)
            # label per client: repeat each client id per class
            labels = torch.arange(C).repeat(M).to(self.device)

            supcon_loss = self.supcon_loss_fn(emb, labels)
            total_loss = supcon_loss
            total_loss.backward()
            if round_num % 10 == 0:
                self.aggregator.log_grad_stats()

            self.agg_opt.step()

            # (3) global prototype 업데이트
            with torch.no_grad():
                weights = F.softmax((self.aggregator.client_embed(client_ids) @ self.aggregator.client_embed.weight.T).mean(dim=-1), dim=0)
                global_shared = (weights.view(M, 1, 1) * x_trans).sum(dim=0)
                self.global_protos = {lbl: global_shared[j].detach().cpu() for j, lbl in enumerate(all_labels)}
            
            # -------------------- Step 3. broadcast global prototypes --------------------
            for client in selected_clients:
                client.set_protos(self.global_protos)
                client.decompose_with_global(self.global_protos)

            print(f"[Round {round_num}] Aggregation done. Loss: {supcon_loss.item()}")
