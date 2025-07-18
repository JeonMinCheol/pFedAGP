import torch.nn.functional as F
import torch
import numpy as np
import time
from collections import defaultdict
from flcore.clients.clientbase import Client

class clientPCL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.protos = None
        self.global_protos = None
        self.client_protos_set = None
        self.tau = args.tau

    def train(self):
        if self.protos is not None:
            trainloader = self.load_train_data()
            start_time = time.time()
            self.model.train()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            my_labels = set(self.protos.keys())

            for step in range(max_local_epochs):
                # 🔥 Only keep global protos for my classes
                global_protos_emb = []
                label_list = []
                for k in self.global_protos.keys():
                    if k in my_labels:
                        global_protos_emb.append(self.global_protos[k])
                        label_list.append(k)
                global_protos_emb = torch.stack(global_protos_emb)

                client_protos_embs = []
                for client_protos in self.client_protos_set:
                    client_protos_emb = []
                    for k in label_list:
                        client_protos_emb.append(client_protos[k])
                    client_protos_emb = torch.stack(client_protos_emb)
                    client_protos_embs.append(client_protos_emb)

                for i, (x, y) in enumerate(trainloader):
                    x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    rep = self.model(x)
                    rep = F.normalize(rep, dim=1)

                    # 🔥 Filter only my local labels
                    mask = torch.tensor([label in my_labels for label in y], device=self.device)
                    if mask.sum() == 0:
                        continue
                    rep = rep[mask]
                    y = y[mask]
                    y_remap = torch.tensor([label_list.index(label.item()) for label in y], device=self.device)

                    similarity = torch.matmul(rep, global_protos_emb.T) / self.tau
                    L_g = self.loss(similarity, y_remap)

                    L_p = 0
                    for client_protos_emb in client_protos_embs:
                        similarity = torch.matmul(rep, client_protos_emb.T) / self.tau
                        L_p += self.loss(similarity, y_remap) / len(self.client_protos_set)

                    loss = L_g + L_p

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()

            self.collect_protos()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
        else:
            self.collect_protos()

    def set_protos(self, global_protos, client_protos_set):
        self.global_protos = global_protos
        self.client_protos_set = client_protos_set

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model(x)
                rep = F.normalize(rep, dim=1)
                for i, yy in enumerate(y):
                    protos[yy.item()].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
