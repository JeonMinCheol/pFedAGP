import torch
import torch.nn as nn
import torch.nn.functional as F

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None
def grl(x, lam=1.0): return _GRL.apply(x, lam)

class ProtoDisentangler(nn.Module):
    """
    E(x)->(z_c,z_s),  D(z_c,z_s)->x_hat
    - class_head(z_c) : 클래스 예측 유지
    - domain_head(grl(z_c)) : 도메인 정보 제거
    - style_domain(z_s) : 도메인 정보 담기
    - style_class(grl(z_s)) : 클래스 정보 제거
    - swap reconstruct: (z_c^i, z_s^j)-> x_i
    """
    def __init__(self, D, k=16, s=16, h=512):
        super().__init__()
        self.D, self.k, self.s = D, k, s
        self.enc = nn.Sequential(
            nn.Linear(D, h), nn.ReLU(inplace=True),
            nn.Linear(h, h), nn.ReLU(inplace=True),
        )
        self.to_c = nn.Linear(h, k)
        self.to_s = nn.Linear(h, s)
        self.dec = nn.Sequential(
            nn.Linear(k+s, h), nn.ReLU(inplace=True),
            nn.Linear(h, h), nn.ReLU(inplace=True),
            nn.Linear(h, D),
        )
        self.class_head  = nn.Linear(k, 0)  # placeholder, set out_features externally
        self.domain_head = nn.Linear(k, 0)
        self.style_dom   = nn.Linear(s, 0)
        self.style_cls   = nn.Linear(s, 0)

    def configure_heads(self, num_classes, num_domains):
        self.class_head  = nn.Linear(self.k, num_classes)
        self.domain_head = nn.Linear(self.k, num_domains)
        self.style_dom   = nn.Linear(self.s, num_domains)
        self.style_cls   = nn.Linear(self.s, num_classes)

    def encode(self, x):
        h = self.enc(x)
        zc = self.to_c(h)
        zs = self.to_s(h)
        return zc, zs

    def decode(self, zc, zs):
        return self.dec(torch.cat([zc, zs], dim=-1))

    def forward(self, x):
        zc, zs = self.encode(x)
        xh = self.decode(zc, zs)
        return zc, zs, xh

def disentangle_train_step(model, optimizer, CP, domain_ids, steps=1, batch_size=4096,
                           lam_recon=1.0, lam_swap=0.5, lam_dom=0.3, lam_cls=1.0, lam_adv=0.3,
                           round_idx=0, warmup_rounds=10):

    model.train()
    device = CP.device
    M, C, D = CP.shape

    # 🔹 normalization
    CP = (CP - CP.mean(dim=-1, keepdim=True)) / (CP.std(dim=-1, keepdim=True) + 1e-6)

    X  = CP.reshape(M*C, D).detach()
    yC = torch.arange(C, device=device).repeat(M)
    yD = domain_ids.view(M,1).repeat(1,C).reshape(-1)

    # 🔹 adversarial warm-up λ
    lambda_adv = min(1.0, round_idx / warmup_rounds)

    B = min(batch_size, X.shape[0])
    total = 0.0
    for _ in range(steps):
        for s in range(0, X.shape[0], B):
            xb, yc, yd = X[s:s+B], yC[s:s+B], yD[s:s+B]

            zc, zs, xh = model(xb)
            loss_rec = F.mse_loss(xh, xb)

            # swap reconstruction
            perm = torch.randperm(xb.shape[0], device=device)
            x_swap = model.decode(zc, zs[perm])
            loss_swap = F.mse_loss(x_swap, xb)

            logits_cls = model.class_head(zc)
            logits_dom = model.domain_head(grl(zc, lam=lambda_adv))
            logits_sdm = model.style_dom(zs)
            logits_scl = model.style_cls(grl(zs, lam=lambda_adv))

            loss_cls = F.cross_entropy(logits_cls, yc)
            loss_adv = F.cross_entropy(logits_dom, yd)
            loss_sdm = F.cross_entropy(logits_sdm, yd)
            loss_scl = F.cross_entropy(logits_scl, yc)

            # 🔹 안정화된 총손실
            loss = (lam_recon * loss_rec +
                    lam_swap  * loss_swap +
                    lam_cls   * loss_cls +
                    lam_adv   * loss_adv +
                    0.5 * lam_dom * (loss_sdm + loss_scl))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item())

    return total
