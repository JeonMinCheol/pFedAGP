from flcore.servers.serverbase import Server
from flcore.clients.clientpFedAGP import clientpFedAGP
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from collections import defaultdict
from matplotlib import pyplot as plt
import math
import seaborn as sns  # 예쁜 히트맵 시각화를 위해 seaborn 사용

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, n_views, D]
        device = features.device
        B, n_views, D = features.shape

        # Normalize
        features = F.normalize(features, dim=2)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*n_views, D]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        contrast_count = n_views
        anchor_feature = contrast_features
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_features.T),
            self.temperature
        )  # [B*n_views, B*n_views]

        # mask out self-comparisons
        logits_mask = torch.scatter(
            torch.ones_like(anchor_dot_contrast),
            1,
            torch.arange(B * n_views).view(-1, 1).to(device),
            0
        )
        mask = mask.repeat(anchor_count, contrast_count)
        mask = mask * logits_mask

        # compute log prob
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

class ServerPrototypeAggregator(nn.Module):
    def __init__(self, num_classes, embed_dim, ent_lambda, supcon_lambda, ent_threshold, init_temp,
                 num_heads=4, dropout=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ent_lambda = ent_lambda
        self.supcon_lambda = supcon_lambda
        self.ent_threshold = ent_threshold

        # Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Transformer encoder (shared across all classes)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )

        # Learnable temperature
        self.attn_temperature = nn.Parameter(torch.tensor(init_temp))

        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Global Query via embedding table
        self.global_query = nn.Parameter(torch.randn(num_classes, embed_dim))

        # Contrastive loss
        self.supcon_loss = SupConLoss(0.07)

    def forward(self, client_protos: torch.Tensor, round_num: int, client_idx: int, all_labels):
        """
        client_protos: [M, C, D]
        client_idx: index into batch (the row in client_protos)
        """
        M, C, D = client_protos.shape
        device = client_protos.device
        H = self.attn.num_heads
        Hdim = D // H

        assert D % H == 0, f"embed_dim {D} must be divisible by num_heads {H}"

        ### Normalize
        protos_norm = self.norm(client_protos.view(-1, D)).view(M, C, D)

        ### Transformer per class
        protos_trans = torch.zeros_like(protos_norm)

        for c in range(C):
            class_seq = protos_norm[:, c, :].unsqueeze(0)         # [1, M, D]
            class_trans = self.transformer(class_seq).squeeze(0)  # [M, D]
            protos_trans[:, c, :] = class_trans

        ### --------- Vectorized Multi-Head Scaled Dot-Product Attention ----------
        temp = torch.clamp(self.attn_temperature, min=0.1)

        # Prepare heads
        q_all = self.global_query[all_labels].to(device) / temp  # [C_round, D]
        q_all = q_all.view(C, H, Hdim)                            # [C, H, Hdim]

        kv_all = protos_trans.permute(1, 0, 2)                     # [C, M, D]
        kv_all = kv_all.view(C, M, H, Hdim)                        # [C, M, H, Hdim]

        # Scaled dot-product
        scores = torch.einsum('chd,cmhd->chm', q_all, kv_all) / math.sqrt(Hdim)  # [C, H, M]
        attn_weights = F.softmax(scores, dim=-1)                                # [C, H, M]

        # Weighted sum
        attn_out = torch.einsum('chm,cmhd->chd', attn_weights, kv_all)           # [C, H, Hdim]
        attn_out = attn_out.reshape(C, D)                                        # [C, D]
        # ------------------------------------------------------------------------

        ### Entropy Regularization
        attn_probs = attn_weights + 1e-8
        entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean()
        temp_ent = entropy

        if entropy < self.ent_threshold:
            entropy = torch.tensor(0.0, device=device)

        ### Contrastive Loss
        features_all = []
        labels_all = []

        for c in range(C):
            anchor = protos_trans[client_idx, c].unsqueeze(0)     # [1, D]
            positive = attn_out[c].unsqueeze(0)                   # [1, D]
            features = torch.stack([anchor, positive], dim=0)     # [2, D] 이런 식으로 하는 이유??
            features_all.append(features.unsqueeze(0))            # [1, 2, D]
            labels_all.append(c)

        features_all = torch.cat(features_all, dim=0).squeeze(2)  # [C, 2, D]
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        loss_con = self.supcon_loss(features_all, labels_all)
        loss_total = self.supcon_lambda * loss_con + self.ent_lambda * entropy

        ### Debug Logging
        # if round_num % 5 == 0:
        #     with torch.no_grad():
        #         pos_sim_list = []
        #         neg_sim_list = []

        #         for c in range(C):
        #             anchor = protos_trans[client_idx, c]
        #             positive = attn_out[c]
        #             sim_pos = F.cosine_similarity(anchor, positive, dim=0)
        #             pos_sim_list.append(sim_pos.item())

        #             negatives = torch.stack([attn_out[i] for i in range(C) if i != c], dim=0)
        #             if negatives.shape[0] > 0:
        #                 sim_negs = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=1)
        #                 neg_sim_list.extend(sim_negs.cpu().tolist())

        #     print(f"[Debug] mean cosine(sim_pos): {sum(pos_sim_list)/len(pos_sim_list):.4f}, "
        #         f"mean cosine(sim_neg): {sum(neg_sim_list)/len(neg_sim_list):.4f}")
        #     print(f"loss_con = {loss_con.item():.4f}, attention entropy = {entropy.item():.4f}, "
        #         f"attn_temperature = {self.attn_temperature.data:.4f}, loss_total = {loss_total.item():.4f}\n")

        ### Return

        # head 차원 평균 → [C, M]
        if round_num % 10 == 0:
            entropy_val = temp_ent.item()

            os.makedirs("attention_heatmaps", exist_ok=True)
            os.makedirs("attention_entropy_logs", exist_ok=True)

            _, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(attn_weights[:, 0, :].detach().cpu().numpy(), cmap="YlGnBu", cbar=True, ax=ax, vmin=0, vmax=1)
            ax.set_title(f'Attention Heatmap (Round {round_num}, Client {client_idx}) Entropy: {entropy_val:.4f}')
            ax.set_xlabel('Client Index (M)')
            ax.set_ylabel('Class Index (C)')
            plt.tight_layout()
            plt.savefig(f'attention_heatmaps/round{round_num}_client{client_idx}_entropy{entropy_val:.4f}.png')
            plt.close()

            # 엔트로피 로그 저장
            with open(f"attention_entropy_logs/client{client_idx}.txt", "a") as f:
                f.write(f"{round_num},{entropy_val:.6f}\n")

        return {c: attn_out[c] for c in range(C)}, loss_total

class pFedAGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientpFedAGP)

        self.agg_steps = args.agg_steps

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.global_protos = {}
        self.client_global_protos = {}
        self.uploaded_memories = []
        self.download_memories = []

        embed_dim = self.clients[0].model.head.in_features
        self.aggregator = ServerPrototypeAggregator(self.num_classes, embed_dim, args.ent_lambda, args.supcon_lambda, args.ent_threshold, args.init_temp).to(self.device)
        self.aggregator_optimizer = torch.optim.SGD(self.aggregator.parameters(), lr=self.learning_rate)
        self.aggregator_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.aggregator_optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def train(self):

        for self.current_round in range(0, self.global_rounds):
            torch.cuda.empty_cache()
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {self.current_round}-------------")
            if self.current_round%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(self.current_round)
            
            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.global_protos = self.proto_aggregation()
            self.send_protos()
            
            torch.cuda.empty_cache()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def proto_aggregation(self):
        all_labels = sorted(set().union(*[lp.keys() for lp in self.uploaded_protos]))
        valid_protos = [lp for lp in self.uploaded_protos if lp]
        if not valid_protos:
            raise ValueError("No valid prototypes received from clients.")

        M, C = len(self.uploaded_protos), len(all_labels)
        D = list(self.uploaded_protos[0].values())[0].shape[-1]

        CP = torch.zeros(M, C, D, device=self.device)
        for i, lp in enumerate(self.uploaded_protos):
            for j, label in enumerate(all_labels):
                if label in lp:
                    proto = lp[label]
                    if proto.dim() == 2:
                        proto = proto.squeeze(0)  # [1, D] -> [D]
                    CP[i, j] = proto.to(self.device).detach()

        # 여러 스텝 중 마지막 스텝에만 backward 수행
        for idx, client in enumerate(self.selected_clients):
            CP_client = CP.clone()

            for step in range(self.agg_steps):
                self.aggregator_optimizer.zero_grad()
                personalized_protos, loss = self.aggregator(CP_client, self.current_round, idx, all_labels)
                loss.backward(retain_graph=True)
                self.aggregator_optimizer.step()

                # learning rate decay
                if self.learning_rate_decay:
                    self.aggregator_learning_rate_scheduler.step()

            # 최종 global_protos 업데이트 (detach & cpu)
            with torch.no_grad():
                self.client_global_protos[client.id] = {
                label: personalized_protos[j].detach().cpu()
                for j, label in enumerate(all_labels)
            }
        
        return self.global_protos
   
    def send_protos(self):
        assert (len(self.clients) > 0)

        # 참여하지 않은 클라이언트는 평균 전송
        average_global_protos = proto_aggregation(self.uploaded_protos)

        for client in self.clients:
            start_time = time.time()

            if client.id in self.uploaded_ids:
                client.set_protos({
                    c: p.to(self.device)
                    for c, p in self.client_global_protos[client.id].items()
                })
            else:
                client.set_protos(average_global_protos)
                
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

        return self.global_protos
    
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
