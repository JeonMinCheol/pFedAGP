import os
import csv
import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch.nn as nn
import time
from collections import defaultdict

class HeadAggregatorWrapper(nn.Module):
    """
    Dummy aggregator for profiling FedPAC-like head aggregation.
    Performs weighted sum of [M, 1, D] input.
    """
    def __init__(self, weights):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights).view(-1, 1, 1), requires_grad=False)
        self.norm = nn.Identity()         # Dummy

    def forward(self, CP):
        """
        CP: [M, 1, D] - Stacked flattened head parameters
        returns: [1, D] - Aggregated head
        """
        assert CP.dim() == 3 and CP.size(1) == 1, "Expected input of shape [M, 1, D]"
        weighted = CP * self.weights.to(CP.device)  # [M, 1, D]
        agg = weighted.sum(dim=0)  # [1, D]
        return agg

class FedAvgAggregatorWrapper(nn.Module):
    """
    Wrapper for simple prototype averaging (FedAvg-style aggregation).
    Assumes input is [M, C, D] and performs mean over M.
    """
    def forward(self, x):
        # x: [M, C, D]
        return x.mean(dim=0)  # [C, D]

class FullPathAggregatorWrapper(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator

    def forward(self, x):
        M, C, D = x.shape
        protos_norm = self.aggregator.norm(x.view(-1, D)).view(M, C, D)
        global_proto_tensor = torch.empty((C, D), device=x.device)

        for c in range(C):
            p_norm = protos_norm[:, c, :]
            trans_out = self.aggregator.transformer(p_norm.unsqueeze(0)).squeeze(0)

            attn_out, attn_weights = self.aggregator.attn(
                trans_out.unsqueeze(0),
                trans_out.unsqueeze(0),
                trans_out.unsqueeze(0)
            )
            agg = attn_out.squeeze(0)

            total = attn_weights.sum(dim=1).squeeze(0)
            diag = torch.diag(attn_weights.squeeze(0))
            target = (total - diag) / (total + 1e-8)

            pred = self.aggregator.domain_regressor(p_norm).squeeze(1)
            alpha = self.aggregator.gate(p_norm).squeeze(1)
            t_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
            inv = alpha * t_norm + (1 - alpha) * (1 - pred)

            combined = self.aggregator.dropout(agg + trans_out)
            global_proto_tensor[c] = (combined * inv.unsqueeze(1)).sum(dim=0) / (inv.sum() + 1e-8)

        return global_proto_tensor

class LogitAggregatorWrapper(torch.nn.Module):
    """
    Wrapper to make logit_aggregation callable via nn.Module for FLOPs profiling.
    Assumes input is List[Dict[int, Tensor]] (client logits) → returns Dict[int, Tensor]
    """
    def __init__(self, aggregation_fn):
        super().__init__()
        self.aggregation_fn = aggregation_fn

    def forward(self, client_logits):
        return self.aggregation_fn(client_logits)

def flatten_model(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)

def log_flops_memory_latency(
    aggregator, CP, logs_dir, current_round,
    num_clients, num_classes, topk=3, filename="flops_detailed_log.csv"
):
    wrapper = FullPathAggregatorWrapper(aggregator).eval().to(CP.device)

    with torch.no_grad():
        inputs = (CP.clone(),)
        flops = FlopCountAnalysis(wrapper, inputs)
        total_flops = flops.total() / 1e6  # MFLOPs

        torch.cuda.reset_peak_memory_stats()
        _ = wrapper(CP.clone())
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = wrapper(CP.clone())
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        print(f"[FLOPs] Total (Full Path): {total_flops:.2f} MFLOPs")
        print(f"[Memory] Peak: {peak_memory:.2f} MB")
        print(f"[Latency] Inference time: {latency_ms:.2f} ms")

        layer_flops = sorted(flops.by_module().items(), key=lambda x: x[1], reverse=True)[:topk]
        layer_names = [name for name, _ in layer_flops]
        layer_vals = [f"{val / 1e6:.2f}" for _, val in layer_flops]

        csv_path = os.path.join(logs_dir, filename)
        os.makedirs(logs_dir, exist_ok=True)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                header = [
                    "Round", "NumClients", "NumClasses", "EmbeddingDim",
                    "TotalFLOPs(M)", "PeakMemory(MB)", "Latency(ms)"
                ]
                for i in range(topk):
                    header.extend([f"TopModule{i+1}", f"FLOPs{i+1}(M)"])
                writer.writerow(header)

            row = [
                current_round,
                num_clients,
                num_classes,
                CP.shape[-1],
                f"{total_flops:.2f}",
                f"{peak_memory:.2f}",
                f"{latency_ms:.2f}"
            ]
            for i in range(topk):
                if i < len(layer_names):
                    row.extend([layer_names[i], layer_vals[i]])
                else:
                    row.extend(["-", "-"])
            writer.writerow(row)

def profile_logit_aggregation(
    aggregation_fn,
    uploaded_logits,
    logs_dir,
    current_round,
    topk=5,
    filename="flops_detailed_log.csv"
):
    """
    Measure latency, peak memory, and flops during logit aggregation.
    Args:
        aggregation_fn: callable, e.g., logit_aggregation
        uploaded_logits: input list to aggregation_fn
        logs_dir: where to save logs
        current_round: current training round
        topk: top-k modules to list
        filename: CSV filename
    """

    device = uploaded_logits[0][list(uploaded_logits[0].keys())[0]].device

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    _ = aggregation_fn(uploaded_logits)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    print(f"[Logit Aggregation] Round {current_round}")
    print(f"⏱️ Latency: {latency:.4f}ms | 📈 Peak Memory: {peak_memory:.2f} MB")

    # CSV 저장
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["Round", "Latency(ms)", "PeakMemory(MB)"])
        writer.writerow([current_round, f"{latency:.4f}", f"{peak_memory:.2f}"])

def profile_aggregation_fedkd(global_model, uploaded_models, logs_dir, round_id, filename="fedkd_aggregation_log.csv"):
    """
    FedKD의 aggregate_parameters() 이후 성능 측정: latency, memory, 간이 FLOPs 추정

    Args:
        global_model: 현재 server의 global model (np.ndarray dict)
        uploaded_models: 클라이언트 모델 리스트 (np.ndarray dict list)
        logs_dir: 결과 저장 폴더
        round_id: 현재 라운드
        filename: 저장할 CSV 이름
    """

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 레이턴시 측정
    start_time = time.time()
    temp_model = {}
    for k in global_model.keys():
        temp_model[k] = np.zeros_like(global_model[k])

    for client_model in uploaded_models:
        for k in temp_model.keys():
            temp_model[k] += client_model[k] / len(uploaded_models)
    latency = (time.time() - start_time) * 1000

    # 메모리 측정
    memory = torch.cuda.memory_allocated() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    # 간단한 FLOPs 추정: element-wise add + scale → count 요소 수 × 클라이언트 수
    total_flops = 0
    for k in global_model.keys():
        if isinstance(global_model[k], np.ndarray):
            total_flops += global_model[k].size * len(uploaded_models)
    total_flops /= 1024**2  # MFLOPs

    print(f"\n[FedKD Aggregation Profiling] Round {round_id}")
    print(f"🔹 Latency: {latency:.4f} ms")
    print(f"🔹 Peak Memory: {peak_memory:.2f} MB")
    print(f"🔹 Estimated Aggregation FLOPs: {total_flops:.2f} MFLOPs")

    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Round", "Latency(ms)", "PeakMemory(MB)", "FLOPs(M)"])
        writer.writerow([round_id, f"{latency:.4f}", f"{peak_memory:.2f}", f"{total_flops:.2f}"])

def profile_aggregation_fedmtl(selected_clients, dim, device, logs_dir, round_id, filename="fedmtl_aggregation_log.csv"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    W = torch.zeros((dim, len(selected_clients)), device=device)
    for idx, client in enumerate(selected_clients):
        W[:, idx] = flatten_model(client.model)

    memory = torch.cuda.memory_allocated(device) / 1024**2
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2

    total_flops = dim * len(selected_clients) / 1024**2  # MFLOPs

    print(f"\n[FedMTL Aggregation Profiling] Round {round_id}")
    print(f"🔹 Peak Memory: {peak_memory:.2f} MB")
    print(f"🔹 Estimated FLOPs: {total_flops:.2f} MFLOPs")

    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)

    return round_id, peak_memory, total_flops, selected_clients, csv_path
    
def profile_proto_aggregation(local_protos_list, logs_dir, round_id, filename="proto_aggregation_log.csv"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = next(iter(local_protos_list[0].values())).device
    torch.cuda.synchronize()
    start_time = time.time()

    # 원래의 aggregation 로직
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label, proto in local_protos.items():
            agg_protos[label].append(proto)

    for label, proto_list in agg_protos.items():
        if len(proto_list) > 1:
            agg = 0 * proto_list[0]
            for p in proto_list:
                agg += p
            agg_protos[label] = agg / len(proto_list)
        else:
            agg_protos[label] = proto_list[0]

    torch.cuda.synchronize()
    latency = (time.time() - start_time) * 1000  # 🔄 milliseconds
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    # FLOPs 추정: 평균 연산 → add + div
    num_labels = len(agg_protos)
    D = proto_list[0].numel()
    M = len(local_protos_list)
    total_flops = num_labels * D * M / 1e6  # MFLOPs

    print(f"\n[Proto Aggregation Profiling] Round {round_id}")
    print(f"🔹 Latency: {latency:.2f} ms")
    print(f"🔹 Peak Memory: {peak_memory:.2f} MB")
    print(f"🔹 Estimated FLOPs: {total_flops:.2f} MFLOPs")

    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Round", "Latency(ms)", "PeakMemory(MB)", "FLOPs(M)", "NumClients", "NumClasses", "Dim"])
        writer.writerow([round_id, f"{latency:.2f}", f"{peak_memory:.2f}", f"{total_flops:.2f}", M, num_labels, D])

    return agg_protos

def profile_aggregation_parameters(uploaded_models, logs_dir, round_id, filename="aggregation_log.csv"):
    device = next(uploaded_models[0].parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.time()

    # 평균 모델 초기화
    avg_model = [p.clone().detach() for p in uploaded_models[0].parameters()]
    for i in range(1, len(uploaded_models)):
        for j, p in enumerate(uploaded_models[i].parameters()):
            avg_model[j] += p.data
    for j in range(len(avg_model)):
        avg_model[j] /= len(uploaded_models)

    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000  # ms
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    # FLOPs (대략적인 연산량 추정)
    total_flops = sum(p.numel() for p in avg_model) * len(uploaded_models) / 1e6  # MFLOPs

    print(f"\n[Aggregation Profiling] Round {round_id}")
    print(f"⏱️ Latency: {latency:.2f} ms")
    print(f"📦 Peak Memory: {peak_memory:.2f} MB")
    print(f"⚙️ Estimated FLOPs: {total_flops:.2f} MFLOPs")

    # 로그 저장
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["Round", "Latency(ms)", "PeakMemory(MB)", "FLOPs(M)", "NumClients"])
        writer.writerow([round_id, f"{latency:.2f}", f"{peak_memory:.2f}", f"{total_flops:.2f}", len(uploaded_models)])

def profile_flmn_proto_aggregation(aggregator, CP, logs_dir, round_id, filename="proto_aggregation_log_flmn.csv", verbose=True, topk=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = CP.device

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        _ = aggregator(CP)  # Forward only
    torch.cuda.synchronize()
    latency = (time.time() - start_time) * 1000  # ms
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    # FLOPs 분석
    try:
        flops = FlopCountAnalysis(aggregator, CP)
        total_flops = flops.total() / 1e6  # MFLOPs
        top_flops = sorted(flops.by_module().items(), key=lambda x: x[1], reverse=True)[:topk]
    except Exception as e:
        print(f"[⚠️ FLOPs 측정 실패]: {e}")
        total_flops = 0.0
        top_flops = []

    if verbose:
        print(f"\n[FLMN Proto Aggregation Profiling] Round {round_id}")
        print(f"⏱️ Latency: {latency:.2f} ms")
        print(f"📦 Peak Memory: {peak_memory:.2f} MB")
        print(f"⚙️ Total FLOPs: {total_flops:.2f} MFLOPs")
        if top_flops:
            print(flop_count_table(flops))

    # CSV 기록
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, filename)
    is_new = not os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        header = ["Round", "Latency(ms)", "PeakMemory(MB)", "FLOPs(M)", "NumClients", "NumClasses", "Dim"]
        for i in range(topk):
            header.extend([f"TopModule{i+1}", f"FLOPs{i+1}(M)"])
        if is_new:
            writer.writerow(header)

        M, C, D = CP.shape
        row = [round_id, f"{latency:.2f}", f"{peak_memory:.2f}", f"{total_flops:.2f}", M, C, D]
        for i in range(topk):
            if i < len(top_flops):
                row.extend([top_flops[i][0], f"{top_flops[i][1]/1e6:.2f}"])
            else:
                row.extend(["-", "-"])
        writer.writerow(row)

def log_head_aggregation_metrics(aggregator, CP_heads, logs_dir, current_round, num_clients, filename):
    """
    Measures the latency, memory usage, and parameter count of a custom head aggregator
    used for FedPAC-style head aggregation.

    Parameters:
    - aggregator: Aggregator wrapper that accepts CP_heads and outputs a new head.
    - CP_heads: Tensor of shape [M, 1, D], where M is the number of clients.
    - logs_dir: Directory to save the CSV log.
    - current_round: Round number for logging.
    - num_clients: Number of client heads involved.
    - filename: CSV file name to write the profiling info.
    """

    # Create logs directory if not exists
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, filename)

    device = CP_heads.device

    # Memory before
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)

    # Measure latency (ms)
    start_time = time.time()
    output_proto = aggregator(CP_heads)  # forward pass
    latency = (time.time() - start_time) * 1000  # ms

    # Memory after
    end_mem = torch.cuda.memory_allocated(device)
    memory_used_kb = (end_mem - start_mem) / 1024  # KB

    # Parameter count
    total_params = sum(p.numel() for p in aggregator.parameters())

    # Save to CSV
    header = ['round', 'num_clients', 'param_count', 'memory_kb', 'latency_ms']
    row = [current_round, num_clients, total_params, memory_used_kb, latency]
    return header, row, log_path
