import torch
from flcore.clients.clientbase import Client
import numpy as np
import time
import math
import copy

class clientMTL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.omega = None
        self.W_glob = None
        self.idx = 0
        self.itk = args.itk
        self.lamda = 1e-4

        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model = self.load_model('model')
        # self.model.to(self.device, non_blocking=True)
        self.model.train()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.W_glob[:, self.idx] = flatten(self.model)
                loss_regularizer = 0
                loss_regularizer += self.W_glob.norm() ** 2

                # for i in range(self.W_glob.shape[0] // self.itk):
                #     x = self.W_glob[i * self.itk:(i+1) * self.itk, :]
                #     loss_regularizer += torch.sum(torch.sum((x*self.omega), 1)**2)
                loss_regularizer += torch.sum(torch.sum((self.W_glob*self.omega), 1)**2)
                f = (int)(math.log10(self.W_glob.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss += loss_regularizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        self.omega = None
        self.W_glob = None

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.personalized_params = self.optimizer.param_groups[-1]['params']
        self.save_item(self.model.state_dict(), str(self.dirchlet), self.pt_path)

    def set_parameters(self, W_glob, omega, idx):
        self.omega = torch.sqrt(omega[0][0])
        self.W_glob = copy.deepcopy(W_glob)
        self.idx = idx

def flatten(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)