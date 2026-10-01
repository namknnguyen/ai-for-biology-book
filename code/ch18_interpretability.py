"""Chapter 18: attribution methods with ground truth, one-hot gradient correction, CKA, sparse autoencoder, conformal coverage."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); rng = np.random.default_rng(0)
L, MOTIF = 40, [3, 0, 2, 1, 3, 2]                                   # motif T A G C T G
k = len(MOTIF)

# ---------------------------------------------------------------------------------------------
# A hand-built "trained model" whose logic we know: logit = max_t motif_score_t - 5; output = sigmoid(logit).
# Ground truth: the bases of the motif occurrence(s) matter.
# ---------------------------------------------------------------------------------------------
W = torch.full((1, 4, k), -1.0)
for j, b in enumerate(MOTIF): W[0, b, j] = 1.0                      # +1 for matching base, -1 otherwise
def model(x):                                                       # x: (B, 4, L) one-hot (or any real extension of it)
    return torch.sigmoid(F.conv1d(x, W).amax(-1).squeeze(-1) - 5.0)

def make_seq(copies):
    s = torch.randint(0, 4, (L,)); pos = []
    for p in ([5] if copies == 1 else [5, 25]):                     # 1 copy, or 2 redundant copies of the motif
        s[p:p + k] = torch.tensor(MOTIF); pos += list(range(p, p + k))
    return F.one_hot(s, 4).T.float(), pos                           # (4, L)

def ism(x):                                                         # max |change in output| over the 3 alternative bases
    base = model(x[None]).item(); out = np.zeros(L)
    for i in range(L):
        a0 = int(x[:, i].argmax())
        for b in range(4):
            if b == a0: continue
            xm = x.clone(); xm[:, i] = 0; xm[b, i] = 1
            out[i] = max(out[i], abs(model(xm[None]).item() - base))
    return out

def grad_x_input(x):
    xx = x[None].clone().requires_grad_(True); model(xx).sum().backward()
    g = xx.grad[0]; g = g - g.mean(0, keepdim=True)                 # one-hot correction: remove the arbitrary common component
    return (g * x).sum(0).abs().numpy()

def integrated_gradients(x, steps=64):
    base = torch.full_like(x, 0.25)                                 # uniform-background reference
    tot = torch.zeros_like(x)
    for a in (torch.arange(steps).float() + 0.5) / steps:
        xx = (base + a * (x - base))[None].clone().requires_grad_(True); model(xx).sum().backward(); tot += xx.grad[0]
    return ((x - base) * tot / steps).sum(0).abs().numpy()

def motif_mass(attr, pos):
    return attr[pos].sum() / (attr.sum() + 1e-12)                  # fraction of attribution mass on the true motif bases

print("fraction of total |attribution| falling on the true motif positions (random-attribution baseline = "
      f"{k}/{L} = {k / L:.2f} for one copy, {2 * k / L:.2f} for two):")
for copies in (1, 2):
    res = {"ISM (single mutations)": [], "gradient x input (corrected)": [], "integrated gradients": []}
    outs = []
    for _ in range(20):
        x, pos = make_seq(copies); outs.append(model(x[None]).item())
        res["ISM (single mutations)"].append(motif_mass(ism(x), pos))
        res["gradient x input (corrected)"].append(motif_mass(grad_x_input(x), pos))
        res["integrated gradients"].append(motif_mass(integrated_gradients(x), pos))
    print(f"  {copies} motif copy(ies) (model output ~ {np.mean(outs):.3f}): " + "; ".join(f"{n}: {np.mean(v):.2f}" for n, v in res.items()))

# the one-hot gradient problem: two networks that agree on one-hot inputs but differ off the simplex
x, _ = make_seq(1)
class Shifted(nn.Module):                                           # same function on one-hot inputs, different extension
    def forward(s, z): return model(z + 0.7 * (z.sum(1, keepdim=True) - 1))   # the added term vanishes when channels sum to 1
xx1 = x[None].clone().requires_grad_(True); model(xx1).sum().backward()
xx2 = x[None].clone().requires_grad_(True); Shifted()(xx2).sum().backward()
g1, g2 = xx1.grad[0], xx2.grad[0]
print(f"\none-hot gradient artefact: same predictions ({abs(model(x[None]).item() - Shifted()(x[None]).item()):.0e} apart) but raw gradients differ by "
      f"{(g1 - g2).abs().max().item():.3f}; after mean-centring over the 4 channels they differ by {((g1 - g1.mean(0)) - (g2 - g2.mean(0))).abs().max().item():.1e}")

# ---------------------------------------------------------------------------------------------
# Linear CKA: invariant to orthogonal transformations and isotropic scaling, not to invertible linear maps.
# ---------------------------------------------------------------------------------------------
def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return np.linalg.norm(Y.T @ X, "fro") ** 2 / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
Xr = rng.normal(size=(500, 20)); Q, _ = np.linalg.qr(rng.normal(size=(20, 20))); M = rng.normal(size=(20, 20)) * 3
print(f"\nCKA(X, XQ) with Q orthogonal = {cka(Xr, Xr @ Q):.3f}; CKA(X, 3 X) = {cka(Xr, 3 * Xr):.3f}; "
      f"CKA(X, X M) with an invertible M = {cka(Xr, Xr @ M):.3f}; CKA(X, independent) = {cka(Xr, rng.normal(size=(500, 20))):.3f}")

# ---------------------------------------------------------------------------------------------
# Superposition and sparse autoencoders: sparse features squeezed into d = 8 dimensions.
# ---------------------------------------------------------------------------------------------
def sae_experiment(n_feat, d, p_active, m, lam, steps=5000, n=30000):
    Wtrue = torch.randn(d, n_feat); Wtrue /= Wtrue.norm(dim=0, keepdim=True)
    fz = (torch.rand(n, n_feat) < p_active).float() * (0.5 + 0.5 * torch.rand(n, n_feat)); H = fz @ Wtrue.T   # sparse features -> d-dim activations
    enc = nn.Linear(d, m); dec = nn.Linear(m, d, bias=False)
    with torch.no_grad(): dec.weight /= dec.weight.norm(dim=0, keepdim=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 2e-3)
    for _ in range(steps):
        hb = H[torch.randint(0, n, (1024,))]; f = F.relu(enc(hb))
        loss = ((dec(f) - hb) ** 2).sum(1).mean() + lam * f.sum(1).mean()                # reconstruction + L1 sparsity
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): dec.weight /= dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
    cos = (Wtrue.T @ dec.weight.detach()).abs()                                          # true directions vs learned dictionary
    with torch.no_grad(): f = F.relu(enc(H)); rec = ((dec(f) - H) ** 2).sum(1).mean().item()
    print(f"  {n_feat:2d} features in {d} dims ({n_feat / d:.1f}x compression), SAE width {m}: recovered {int((cos.max(1).values > 0.9).sum()):2d}/{n_feat} "
          f"true directions (cosine > 0.9); mean best cosine {cos.max(1).values.mean():.2f}; active units/sample {(f > 1e-3).float().sum(1).mean():.1f}; recon MSE {rec:.4f}")
print("\nsuperposition and sparse autoencoders:")
sae_experiment(12, 8, 0.03, 12, 0.05); sae_experiment(16, 8, 0.03, 32, 0.05); sae_experiment(24, 8, 0.02, 24, 0.05)

# ---------------------------------------------------------------------------------------------
# Split conformal prediction: distribution-free coverage under exchangeability, and its failure under shift.
# ---------------------------------------------------------------------------------------------
def data(n, shift=0.0):
    x = rng.uniform(-3, 3, n) + shift; y = np.sin(x) + (0.1 + 0.2 * np.abs(x)) * rng.normal(size=n); return x, y
xtr, ytr = data(2000); coef = np.polyfit(xtr, ytr, 7)
pred = lambda x: np.polyval(coef, x)
xc, yc = data(1000); scores = np.abs(yc - pred(xc)); alpha = 0.1
q = np.sort(scores)[int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1]                 # finite-sample-corrected quantile
for name, sh in (("exchangeable test data", 0.0), ("covariate-shifted test data (x + 2.5)", 2.5)):
    xt, yt = data(20000, sh); cover = np.mean(np.abs(yt - pred(xt)) <= q)
    print(f"conformal 90% interval (half-width {q:.2f}): empirical coverage on {name} = {cover:.3f}")
