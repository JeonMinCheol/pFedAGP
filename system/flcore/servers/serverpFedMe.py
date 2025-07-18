import os
import copy
import time
import h5py
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

from sklearn import metrics

class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpFedMe)

        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")
            if i%self.eval_gap == 0:
                # print("\nEvaluate global model")
                # self.evaluate(acc=None, loss=None, global_model=True)
                print("\nEvaluate personalized model")
                self.evaluate(i)

            if i%(4*self.eval_gap) == 0:
                print("\nEvaluate domain generalization")
                with open(f"./results/generalization/{self.dataset}/{self.dirchlet}/{self.algorithm}.csv", "a") as f: 
                    test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std = self.test_metrics(self.test_loader)
                    f.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(i, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std))
                    self._print(None, None,test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std)

            for client in self.selected_clients:
                client.train()
            
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break

        # print("\nBest global accuracy.")
        # # self.print_(max(self.rs_test_acc), max(
        # #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(i)

    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
    
    def test_metrics(self, test_loader=None):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        all_y_prob = []
        all_y_true = []

        client_aucs = []
        client_accs = []
        client_fscores = []

        for c in self.clients:
            correct_c, num_c, y_prob_c, y_true_c = c.test_metrics(test_loader)

            # 1-a) per-client accuracy
            acc_i = correct_c / num_c
            client_accs.append(acc_i)

            # 1-b) per-client AUC / F1
            auc_i = metrics.roc_auc_score(y_true_c, y_prob_c, average='micro')
            f1_i = metrics.f1_score(
                np.argmax(y_true_c, axis=1),
                np.argmax(y_prob_c, axis=1),
                average='macro'
            )

            client_aucs.append(auc_i)
            client_fscores.append(f1_i)

            all_y_true.append(y_true_c)
            all_y_prob.append(y_prob_c)

        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_prob = np.concatenate(all_y_prob, axis=0)
        
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

        # 3) 클라이언트별 표준편차
        acc_std = float(np.std(client_accs))
        auc_std = float(np.std(client_aucs))
        f1_std  = float(np.std(client_fscores))    
        ece_std = np.std(ece_error)

        global_ece_mean = np.mean(ece_list)
        global_acc = (all_y_true.argmax(1) == all_y_prob.argmax(1)).mean()
        global_auc = metrics.roc_auc_score(all_y_true, all_y_prob, average='micro')
        global_fscore = metrics.f1_score(np.argmax(all_y_true, axis=1), np.argmax(all_y_prob, axis=1), average='macro')

        return global_acc, global_auc, global_fscore, global_ece_mean, acc_std, auc_std, f1_std, ece_std

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0], [0]
        
        num_samples = []
        losses = []
        accuracies = []
        for c in self.clients:
            loss, train_num, train_acc = c.train_metrics()
            num_samples.append(train_num)
            losses.append(loss*1.0)
            accuracies.append(train_acc*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, accuracies

    # evaluate selected clients
    def evaluate(self, acc_list=None, loss_list=None):
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

        with open(f"./results/personal/{self.dataset}/{self.dirchlet}/{self.algorithm}.csv", "a") as f: 
            f.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(self.round, train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std))
            self.round = self.round + 1