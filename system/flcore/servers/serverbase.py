import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from torch.utils.data import DataLoader
from sklearn import metrics

from utils.data_utils import read_client_data
from utils.dlg import DLG

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.aggregation_times = []
        self.aggregation_memories = []

        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.learning_rate_decay = args.learning_rate_decay
        self.auto_break = args.auto_break
        self.agg_steps = args.agg_steps
        self.dirchlet = args.dirchlet
        self.attn_lr = args.attn_lr
        self.goal = args.goal
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.Budget = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_fscore = []
        self.rs_test_ece = []

        self.times = times
        self.start_time = time.time()
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch
        self.current_round = 0

        self.test_loader = self.set_test_data()
        self.csv_dir = f"{args.dataset}/{str(args.dirchlet)}/{self.algorithm}_time_{str(self.start_time)}"

        with open(f"./results/{self.csv_dir}.csv", "w") as f: 
            f.write("round, train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, std_accs, std_aucs, std_fscores, std_eces\n")

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, alpha=self.dirchlet)
            test_data = read_client_data(self.dataset, i, is_train=False, alpha=self.dirchlet)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
            
    def set_test_data(self):
        test_data = []
        for i in range(self.num_clients):
            test_data.extend(read_client_data(self.dataset, i, is_train = False))
            
        return DataLoader(test_data, self.batch_size, drop_last=True, shuffle=True)
        
    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        rnd = np.random.RandomState(seed=42 + self.current_round)  # 내부 상태 기반 시드

        if self.random_join_ratio:
            self.current_num_join_clients = rnd.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        selected_clients = list(rnd.choice(
            self.clients, self.current_num_join_clients, replace=False
        ))

        self.current_round += 1

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models",str(self.dirchlet),self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models",str(self.dirchlet),self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + str(self.dirchlet) + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_fscore', data=self.rs_test_fscore)
                hf.create_dataset('rs_test_ece', data=self.rs_test_ece)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        pass

    def test_metrics(self, test_loader=None):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        all_y_prob = []
        all_y_true = []

        client_aucs = []
        client_accs = []
        client_fscores = []

        global_num = 0
        global_correct = 0

        for c in self.clients:
            correct_c, num_c, y_prob_c, y_true_c = c.test_metrics(test_loader)
            if num_c == 0:
                continue

            if num_c == 1e-5:
                return 0, 0, 0, 0, 0, 0, 0, 0 

            # 1-a) per-client accuracy
            acc_i = correct_c / num_c
            client_accs.append(acc_i)

            global_correct += correct_c
            global_num += num_c

            # 1-b) per-client AUC / F1
            auc_i = metrics.roc_auc_score(y_true_c, y_prob_c, average='micro')
            f1_i = metrics.f1_score(
                np.argmax(y_true_c, axis=1),
                np.argmax(y_prob_c, axis=1),
                average='weighted'
            )

            client_aucs.append(auc_i)
            client_fscores.append(f1_i)

            all_y_true.append(y_true_c)
            all_y_prob.append(y_prob_c)

        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_prob = np.concatenate(all_y_prob, axis=0)

        # 3) 클라이언트별 표준편차
        acc_std = float(np.std(client_accs))
        auc_std = float(np.std(client_aucs))
        f1_std  = float(np.std(client_fscores))
        ece_list = []
        ece_error = []

        for c in range(self.num_classes):
            probs = all_y_prob[:, c]
            trues = all_y_true[:, c]

            bins = np.linspace(0.0, 1.0, 11)
            bin_ids = np.digitize(probs, bins) - 1

            ece_c = 0.0

            for i in range(10):
                bin_samples = (bin_ids == i)
                if np.sum(bin_samples) > 0:
                    acc = np.mean(trues[bin_samples])
                    conf = np.mean(probs[bin_samples])
                    error = np.abs(acc - conf)
                    ece_error.append(error)

                    ece_c += (np.sum(bin_samples) / len(probs)) * error

            ece_list.append(ece_c)
            
        global_ece_mean = np.mean(ece_list)
        ece_std   = np.std(ece_error)
        global_acc = global_correct / global_num
        global_auc = metrics.roc_auc_score(all_y_true, all_y_prob, average='micro')
        global_fscore = metrics.f1_score(np.argmax(all_y_true, axis=1), np.argmax(all_y_prob, axis=1), average='weighted')

        return global_acc, global_auc, global_fscore, global_ece_mean, acc_std, auc_std, f1_std, ece_std

    def _print(self, train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std):
        if train_loss is not None:
            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Train acc: {:.4f}".format(train_acc))

        if test_acc is not None:
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Averaged Test F1: {:.4f}".format(test_fscore))
            print("Averaged Test ECE: {:.4f}".format(test_ece))

            print("Std Test Accurancy: {:.4f}".format(acc_std))
            print("Std Test AUC: {:.4f}".format(auc_std))
            print("Std Test F1: {:.4f}".format(f1_std))
            print("Std Test ECE: {:.4f}".format(ece_std))

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0], [0]
        
        num_samples = []
        losses = []
        accuracies = []
        for c in self.clients:
            ls, ns, tc = c.train_metrics()
            num_samples.append(ns)
            losses.append(ls*1.0)
            accuracies.append(tc*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, accuracies

    def evaluate(self, round, acc_list=None, loss_list=None):
        test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std = self.test_metrics()
        stats_train = self.train_metrics()

        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_acc = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        if acc_list == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc_list.append(test_acc)
        
        if loss_list == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss_list.append(train_loss)

        self.rs_test_auc.append(test_auc)
        self.rs_test_fscore.append(test_fscore)
        self.rs_test_ece.append(test_ece)
        
        self._print(train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std)

        with open(f"./results/{self.csv_dir}.csv", "a") as f: 
            f.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(round, train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std))

        return train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std