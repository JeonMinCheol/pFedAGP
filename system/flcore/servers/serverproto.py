from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import torch

class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.uploaded_memories = []
        self.download_memories = []

    def train(self):
        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")
            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate(i)

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
            
            if i == 100:
                for client in self.clients:
                    client.visualize_embeddings()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


            self.receive_protos()

            
            self.global_protos = self.proto_aggregation(self.uploaded_protos)
            

            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        # print(f"upload memory average size: {np.mean(self.uploaded_memories)}")
        # print(f"download memory average size: {np.mean(self.download_memories)}")
        print(f"Aggregation memory average size: {np.mean(self.aggregation_memories):.2f}")
        print(f"Aggregation time average size: {np.mean(self.aggregation_times):.2f} ms")
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            # 전달 시 Tensor를 dict로 다시 매핑
            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

        total_bytes = 0
        for client_protos in self.uploaded_protos:
            for proto in client_protos.values():
                total_bytes += proto.numel() * proto.element_size()

        total_KB = total_bytes / (1024)
        self.uploaded_memories.append(total_KB)
        print(f"📦 uploaded_protos 총 메모리 사용량: {total_KB:.2f} KB")

    def proto_aggregation(self, local_protos_list):
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

        total_bytes = sum(p.numel() * p.element_size() for p in agg_protos_label.values())
        total_KB = total_bytes / (1024)
        print(f"📦 Global Prototype size: {total_KB:.2f} KB")
        self.download_memories.append(total_KB)

        return agg_protos_label

