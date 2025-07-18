from sklearn.preprocessing import label_binarize
import torch
import numpy as np
import time
import copy
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F

class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.ditto_mu
        self.plocal_steps = args.plocal_steps

        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )


    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device, non_blocking=True)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.save_item(self.model.state_dict(), str(self.dirchlet), self.pt_path)

        
    def ptrain(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device, non_blocking=True)
        self.model_per.train()

        max_local_epochs = self.plocal_steps
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
                output = self.model_per(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                output = self.model_per(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num, train_correct

    def test_metrics(self, test_loader=None):
        if test_loader is None:
            testloaderfull = self.load_test_data()
        else:
            testloaderfull = test_loader
            
        self.model_per.eval()

        y_prob = []
        y_true = []

        test_correct = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)

                y = y.to(self.device, non_blocking=True)
                output = self.model_per(x)

                p = F.softmax(output, dim=1).detach().cpu().numpy()
                t = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))

                test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(p)
                y_true.append(t)
                
        # self.model.cpu()
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        return test_correct, test_num, y_prob, y_true
