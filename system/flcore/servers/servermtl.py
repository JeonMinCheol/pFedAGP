import time
import torch
from flcore.clients.clientmtl import clientMTL
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

class FedMTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.dim = len(self.flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=args.device)
        self.device = args.device

        I = torch.ones((self.num_join_clients, self.num_join_clients))
        i = torch.ones((self.num_join_clients, 1))
        omega = (I - 1 / self.num_join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMTL)
        self.Budget = []
            
        print(f"\nJoin clients / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")


    def train(self):
        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            

            self.aggregate_parameters()

            

            
            print(f"\n-------------Round number: {i}-------------")
            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(i)
                    
            for idx, client in enumerate(self.selected_clients):
                start_time = time.time()
                
                client.set_parameters(self.W_glob, self.omega, idx)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
            overall_start = time.perf_counter()

            for client in self.selected_clients:
                client.train()

            torch.cuda.synchronize()
            overall_end = time.perf_counter()

            aggregation_time = (overall_end - overall_start) * 1000
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            
            self.aggregation_times.append(aggregation_time)
            self.aggregation_memories.append(peak_mem)
            
            print(f"✅ Total proto_aggregation time: {aggregation_time:.2f} ms")
            print(f"✅ GPU memory peak usage: {peak_mem:.2f} KB")
        
            torch.cuda.empty_cache()
            
            if i == 100:
                for client in self.clients:
                    client.visualize_embeddings()
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print(f"Aggregation memory average size: {np.mean(self.aggregation_memories):.2f}")
        print(f"Aggregation time average size: {np.mean(self.aggregation_times):.2f} ms")

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # self.save_global_model()

    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)

    def aggregate_parameters(self):
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=self.device)
        for idx, client in enumerate(self.selected_clients):
            self.W_glob[:, idx] = self.flatten(client.model)
