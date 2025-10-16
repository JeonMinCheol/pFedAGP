# system/flcore/trainmodel/content_style_pro.py
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def robust_rankk_svd(CP: torch.Tensor,
                     client_losses: torch.Tensor = None,
                     k: int = 8,
                     tau: float = 1.5,
                     huber_delta: float = 1.0,
                     chunk: int = 4):
    """
    CP: [M, C, D]  (client prototypes)
    client_losses: [M] or None (loss-based 가중치)
    k: rank of content subspace per class
    tau: loss temperature for weights (softmax(-tau*zscore(loss)))
    huber_delta: 허버 로버스트 중심 계산에 사용
    return:
      MU:     [C, D]      (class mean; robust)
      V:      [C, k, D]   (content bases per class; row-orthonormal)
      ALPHA:  [M, C, k]   (per-client coefficients)
      S_res:  [M, C, D]   (style residuals)
    """
    M, C, D = CP.shape
    device = CP.device
    dtype = CP.dtype

    # client loss → 가중치 w_i
    if client_losses is None:
        w = torch.ones(M, device=device, dtype=dtype) / M
    else:
        z = (client_losses - client_losses.mean()) / (client_losses.std() + 1e-8)
        w = torch.softmax(-tau * z, dim=0)  # loss 낮을수록 가중 ↑
    w = w.clamp_min(1e-8)

    MU = torch.zeros(C, D, device=device, dtype=dtype)
    V  = torch.zeros(C, k, D, device=device, dtype=dtype)
    ALPHA = torch.zeros(M, C, k, device=device, dtype=dtype)
    S_res = torch.zeros(M, C, D, device=device, dtype=dtype)

    chunk = max(1, int(chunk))
    for s in range(0, C, chunk):
        e = min(C, s + chunk)
        Pc = CP[:, s:e, :]            # [M, c, D]
        # robust center (Huber) per class
        # 초기 평균
        mu = (w.view(M, 1, 1) * Pc).sum(0) / w.sum()
        # 1~2번 허버 보정
        for _ in range(2):
            r = Pc - mu  # [M, c, D]
            # 허버 가중
            r2 = (r ** 2).sum(-1)  # [M, c]
            scale = torch.clamp(r2.sqrt() / huber_delta, min=1.0)  # >1이면 down-weight
            hw = (w.view(M, 1) / scale).clamp_min(1e-6)            # [M, c]
            hw_exp = hw.unsqueeze(-1)  # [M, c, 1]
            denom = hw.sum(0).unsqueeze(-1) + 1e-6  # [c,1]
            mu = (hw_exp * Pc).sum(0) / denom  # [c,D]

        MU[s:e] = mu  # [c, D]

        # 가중 중심화
        X = Pc - mu      # [M, c, D]
        Xw = (w.view(M, 1, 1).sqrt() * X).transpose(0, 1)  # [c, M, D]

        # 각 클래스별 SVD (비용↑ 허용)
        for j in range(e - s):
            Xm = Xw[j]  # [M, D]
            # full SVD 대신 top-k: torch.linalg.svd로 충분 (M,D가 수천이면 chunk 조절)
            U, S, VT = torch.linalg.svd(Xm, full_matrices=False)
            vk = VT[:k]                     # [k, D]
            V[s + j, :vk.shape[0]] = vk
            # 계수: (P_i - mu) @ vk^T
            a = (X[:, j] @ vk.T)           # [M, k]
            ALPHA[:, s + j, :a.shape[1]] = a
            # 잔차
            recon = (a @ vk) + mu[j]       # [M, D]
            S_res[:, s + j] = Pc[:, j] - recon

        del Pc, mu, X, Xw, U, S, VT, vk, a, recon
        torch.cuda.empty_cache()

    return MU, V, ALPHA, S_res


@torch.no_grad()
def assemble_content_from_rankk(MU, V, ALPHA):
    """
    MU: [C,D], V:[C,k,D], ALPHA:[M,C,k]  →  P_content_clients:[M,C,D]
    """
    M, C, k = ALPHA.shape
    D = MU.shape[-1]
    P = torch.zeros(M, C, D, device=MU.device, dtype=MU.dtype)
    for c in range(C):
        vk = V[c]            # [k,D]
        a  = ALPHA[:, c, :]  # [M,k]
        P[:, c] = (a @ vk) + MU[c]
    return P  # [M,C,D]


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

def grad_reverse(x, lam=1.0):
    return _GradReverse.apply(x, lam)

# === Nullspace Projector ===
class NullspaceProjector(nn.Module):
    """
    도메인 정보(클라/도메인 분포)를 제거하는 투영기 P.
    - 선형 투영 W: R^D -> R^D (차원 유지, 도메인 방향 억제)
    - 정규화: ||W^T W - I|| 로 직교 근사
    """
    def __init__(self, dim, ortho_weight=1e-3):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=False)  # 차원 유지형 투영
        nn.init.eye_(self.proj.weight)               # 초깃값: 항등
        self.ortho_weight = ortho_weight

    def forward(self, x):
        # x: [..., D]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        z_flat = self.proj(x_flat)
        z = z_flat.reshape(*x.shape[:-1], D)
        return z

    def ortho_reg(self):
        W = self.proj.weight  # [D, D]
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        return (W.t() @ W - I).pow(2).mean()

@torch.no_grad()
def proto_rpca(CP: torch.Tensor, iters: int = 3, chunk: int = 8):
    """
    Proto-RPCA (랭크-1 근사)
    CP: [M, C, D]  (M clients, C classes, D dim)
    return:
      C_dir:   [C, D]     # 클래스별 공통 방향 벡터 v_c (정규화)
      S_style: [M, C, D]  # 스타일(잔차) = P_i,c - alpha_i,c * v_c
      Alpha:   [M, C]     # 각 클라의 콘텐츠 스칼라 계수 alpha_i,c
    """
    M, C, D = CP.shape
    device = CP.device
    C_dir   = torch.zeros(C, D, device=device, dtype=CP.dtype)
    S_style = torch.zeros_like(CP)
    Alpha   = torch.zeros(M, C, device=device, dtype=CP.dtype)

    chunk = max(1, int(chunk))
    for s in range(0, C, chunk):
        e = min(C, s + chunk)
        Pc = CP[:, s:e, :]                        # [M, chunk, D]

        # 파워 메서드로 v_c 근사 (초경량 / 안정)
        v = torch.randn(e - s, D, device=device, dtype=CP.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        for _ in range(max(1, iters)):
            # Pv: 각 클라가 현 v에 투영된 스칼라
            Pv = (Pc * v.unsqueeze(0)).sum(-1, keepdim=True)   # [M, chunk, 1]
            # 새 v: 클라별 가중합
            v  = (Pv * Pc).sum(0)                              # [chunk, D]
            v  = v / (v.norm(dim=1, keepdim=True) + 1e-8)

        alpha = (Pc * v.unsqueeze(0)).sum(-1)                  # [M, chunk]
        resid = Pc - alpha.unsqueeze(-1) * v.unsqueeze(0)      # [M, chunk, D]

        C_dir[s:e]     = v
        S_style[:, s:e] = resid
        Alpha[:, s:e]   = alpha

        # 메모리 관리
        del Pc, v, alpha, resid, Pv
        torch.cuda.empty_cache()

    return C_dir, S_style, Alpha


@torch.no_grad()
def compute_style_gate(losses: torch.Tensor, k: float = 2.0,
                       beta_min: float = 0.0, beta_max: float = 1.0, normalize: bool = True):
    """
    losses: [M]  (클라별 평균 loss; 값이 클수록 콘텐츠만으로는 안 맞음)
    beta = sigmoid(k * zscore(loss)) ∈ (0,1) → 스타일 잔차 가중치
    """
    z = losses
    if normalize:
        z = (z - z.mean()) / (z.std() + 1e-8)
    beta = torch.sigmoid(k * z)
    if beta_min > 0.0 or beta_max < 1.0:
        beta = beta.clamp(beta_min, beta_max)
    return beta  # [M]
