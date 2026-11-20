import torch

from quant_attention import _relevant_zero_reconstruct


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(20260909)

B, H, T = 2, 3, 16
K = 4
tau = 6.0

causal = torch.ones(
    T, T,
    dtype=torch.bool,
    device=device,
).tril()

att0 = (
    torch.randn(
        B, H, T, T,
        device=device,
        dtype=torch.float32,
    ) * 3.0
)


# ------------------------------------------------------------
# 1. Forward row-shift invariance
# ------------------------------------------------------------

shift = torch.randn(
    B, H, T, 1,
    device=device,
)

w0, *_ = _relevant_zero_reconstruct(
    att0,
    causal,
    K,
    "nearest",
    tau=tau,
    range_grad="full",
)

w1, *_ = _relevant_zero_reconstruct(
    att0 + shift,
    causal,
    K,
    "nearest",
    tau=tau,
    range_grad="full",
)

err = (w0 - w1).abs().max().item()

print(
    "forward row-shift max error:",
    f"{err:.3e}",
)

assert err < 2e-6


# ------------------------------------------------------------
# 2. Full-gradient row-shift symmetry
# ------------------------------------------------------------

att = att0.detach().clone().requires_grad_(True)

w, *_ = _relevant_zero_reconstruct(
    att,
    causal,
    K,
    "nearest",
    tau=tau,
    range_grad="full",
)

probe = torch.randn_like(w)

loss = (w * probe).sum()
loss.backward()

g = att.grad

valid = causal.view(
    1, 1, T, T
)

g_valid = g.masked_fill(
    ~valid,
    0.0,
)

row_sum = g_valid.sum(
    dim=-1
)

max_row_sum = (
    row_sum.abs().max().item()
)

print(
    "full-grad row-sum max error:",
    f"{max_row_sum:.3e}",
)

assert max_row_sum < 2e-5


# ------------------------------------------------------------
# 3. Tail is exactly zero
# ------------------------------------------------------------

scores = torch.zeros(
    1, 1, 4, 4,
    device=device,
    dtype=torch.float32,
)

scores[0, 0, 3, 0] = 0.0
scores[0, 0, 3, 1] = -1.0
scores[0, 0, 3, 2] = -6.0
scores[0, 0, 3, 3] = -6.01

c4 = torch.ones(
    4, 4,
    dtype=torch.bool,
    device=device,
).tril()

w, *_ = _relevant_zero_reconstruct(
    scores,
    c4,
    K=4,
    approx="nearest",
    tau=6.0,
    range_grad="full",
)

print(
    "boundary weight z=-6 :",
    w[0, 0, 3, 2].item(),
)

print(
    "tail weight z=-6.01 :",
    w[0, 0, 3, 3].item(),
)

assert w[0, 0, 3, 2] > 0
assert w[0, 0, 3, 3].item() == 0.0


# ------------------------------------------------------------
# 4. Backward is finite
# ------------------------------------------------------------

assert torch.isfinite(g).all()

print("ALL RELEVANT-ZERO SANITY TESTS PASSED")
