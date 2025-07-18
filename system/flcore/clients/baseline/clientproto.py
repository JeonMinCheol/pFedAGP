from collections import defaultdict
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device, non_blocking=True)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                    
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        # self.collect_protos()
        self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.save_item(self.model.state_dict(), str(self.dirchlet), self.pt_path)


    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
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
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
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
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return losses, train_num, train_correct

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos