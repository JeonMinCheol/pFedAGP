import copy
import numpy as np
import time
import torch
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread

from sklearn import metrics

class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")

            if i%self.eval_gap == 0:
                # Personalization 성능 측정
                print("\nEvaluate personalized models")
                self.evaluate(i)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

            torch.cuda.synchronize()
            overall_start = time.perf_counter()

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            torch.cuda.synchronize()
            overall_end = time.perf_counter()

            aggregation_time = (overall_end - overall_start) * 1000
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            
            self.aggregation_times.append(aggregation_time)
            self.aggregation_memories.append(peak_mem)

            if i == 100:
                for client in self.clients:
                    client.visualize_embeddings()
            
            # pthreads = [Thread(target=client.ptrain)
            #            for client in self.selected_clients]
            # [t.start() for t in pthreads]
            # [t.join() for t in pthreads]

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print(f"Aggregation memory average size: {np.mean(self.aggregation_memories):.2f}")
        print(f"Aggregation time average size: {np.mean(self.aggregation_times):.2f} ms")

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate(i)

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
            
        ece_std   = np.std(ece_error)
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
            ls, ns, tc = c.train_metrics()
            num_samples.append(ns)
            losses.append(ls*1.0)
            accuracies.append(tc*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, accuracies

    # evaluate selected clients
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

        with open(f"./results/{self.dataset}/{self.dirchlet}/{self.algorithm}.csv", "a") as f: 
            f.write("{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(round, train_loss, train_acc, test_acc, test_auc, test_fscore, test_ece, acc_std, auc_std, f1_std, ece_std))