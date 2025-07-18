from flcore.clients.clientpcl import clientPCL
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
import time
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F

class FedPCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientPCL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.proto_dim = None
        self.global_protos = {k: None for k in range(self.num_classes)}
        self.client_protos_set = [None for _ in range(self.num_clients)]

    def train(self):
        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(i)

            for client in self.selected_clients:
                client.train()

            self.receive_protos()

            self.global_protos, self.proto_dim = proto_aggregation(
                self.uploaded_protos,
                self.global_protos,
                self.num_classes,
                self.proto_dim,
                self.device
            )

            self.prototype_padding()
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def send_protos(self):
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos, self.client_protos_set)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.client_protos_set[client.id] = client.protos

    def prototype_padding(self):
        for cid in range(self.num_clients):
            if self.client_protos_set[cid] is None:
                self.client_protos_set[cid] = self.global_protos
            else:
                for k in range(self.num_classes):
                    if k not in self.client_protos_set[cid]:
                        self.client_protos_set[cid][k] = self.global_protos[k]


def proto_aggregation(local_protos_list, previous_global_protos, num_classes, proto_dim, device):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        if local_protos is None:
            continue
        for label, proto in local_protos.items():
            if proto is not None:
                agg_protos_label[label].append(proto)

    if proto_dim is None:
        for proto_list in agg_protos_label.values():
            if len(proto_list) > 0:
                proto_dim = proto_list[0].shape[0]
                print(f"✅ [Server] Auto-inferred proto_dim = {proto_dim}")
                break
        if proto_dim is None:
            raise ValueError("No prototypes received to infer proto_dim!")

    new_global_protos = {}
    for label in range(num_classes):
        proto_list = agg_protos_label.get(label, [])
        if len(proto_list) == 0:
            # 아무도 안 보낸 클래스는 유지
            if previous_global_protos[label] is None:
                new_global_protos[label] = torch.zeros(proto_dim, device=device)
                print(f"⚠️ No proto ever for label {label}. Initialized zeros.")
            else:
                new_global_protos[label] = previous_global_protos[label].clone()
        else:
            stacked = torch.stack(proto_list)
            new_global_protos[label] = stacked.mean(dim=0)

    return new_global_protos, proto_dim
