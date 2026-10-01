"""Chapter 10: which grammar can a CNN express? Motif spacing, receptive fields, and RC weight tying.

Task: a sequence is positive iff motif A (TGACTCA) is followed by motif B (CACGTG) with a gap of 8-12 bp.
Negatives: A and B too far apart, only A, only B, or neither.
  M1: conv -> ReLU -> global max pool   (can detect each motif, cannot measure spacing)
  M2: conv -> ReLU -> conv(25) -> ReLU -> global max pool   (second layer sees A and B together if the gap is short)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); rng = np.random.default_rng(0)
A_MOTIF, B_MOTIF, L = "TGACTCA", "CACGTG", 120
enc = {c: i for i, c in enumerate("ACGT")}
def put(seq, motif, pos): seq[pos:pos + len(motif)] = [enc[c] for c in motif]

def make(n):
    X = rng.integers(0, 4, (n, L)); y = np.zeros(n); kind = np.empty(n, dtype=object)
    for i in range(n):
        k = rng.choice(["pos", "far", "onlyA", "onlyB", "none"], p=[0.4, 0.2, 0.15, 0.15, 0.1]); kind[i] = k
        if k == "pos":
            gap = rng.integers(8, 13); pa = rng.integers(0, L - 7 - gap - 6); put(X[i], A_MOTIF, pa); put(X[i], B_MOTIF, pa + 7 + gap); y[i] = 1
        elif k == "far":
            gap = rng.integers(30, 60); pa = rng.integers(0, L - 7 - gap - 6); put(X[i], A_MOTIF, pa); put(X[i], B_MOTIF, pa + 7 + gap)
        elif k == "onlyA": put(X[i], A_MOTIF, rng.integers(0, L - 7))
        elif k == "onlyB": put(X[i], B_MOTIF, rng.integers(0, L - 6))
    oh = F.one_hot(torch.tensor(X), 4).permute(0, 2, 1).float()          # (n, 4, L)
    return oh, torch.tensor(y).float(), kind

Xtr, ytr, _ = make(8000); Xte, yte, kte = make(3000)

class M1(nn.Module):
    def __init__(s): super().__init__(); s.c1 = nn.Conv1d(4, 32, 9); s.fc = nn.Linear(32, 1)
    def forward(s, x): return s.fc(F.relu(s.c1(x)).amax(-1)).squeeze(-1)
class M2(nn.Module):
    def __init__(s): super().__init__(); s.c1 = nn.Conv1d(4, 32, 9); s.c2 = nn.Conv1d(32, 32, 25); s.fc = nn.Linear(32, 1)
    def forward(s, x): return s.fc(F.relu(s.c2(F.relu(s.c1(x)))).amax(-1)).squeeze(-1)

def fit(m, steps=700):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    for t in range(steps):
        idx = torch.randint(0, len(Xtr), (128,))
        loss = F.binary_cross_entropy_with_logits(m(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m

def rf_of(conv_layers):                                                   # receptive field of stacked stride-1 convs
    return 1 + sum(k - 1 for k in conv_layers)

hard = np.isin(kte, ["pos", "far"])                                       # the only distinction here is spacing
print(f"receptive fields: M1 = {rf_of([9])} bp, M2 = {rf_of([9, 25])} bp; a positive spans 7+gap+6 = 21-25 bp, a 'far' negative spans >= 43 bp")
for name, M in (("M1 (1 conv layer)", M1), ("M2 (2 conv layers)", M2)):
    m = fit(M());
    with torch.no_grad(): s = m(Xte).numpy()
    print(f"{name:20s} AUROC all test = {roc_auc_score(yte.numpy(), s):.3f}; "
          f"AUROC positives vs 'far' negatives only = {roc_auc_score(yte.numpy()[hard], s[hard]):.3f}; params = {sum(p.numel() for p in m.parameters())}")
    if name.startswith("M2"): m2 = m

# first-layer filters recover the planted motifs (best normalised correlation between filter and motif one-hot)
W = m2.c1.weight.detach().numpy()                                          # (32, 4, 9)
def best_match(motif):
    M = np.zeros((4, len(motif)));
    for j, c in enumerate(motif): M[enc[c], j] = 1
    M = M - M.mean(0, keepdims=True); best = 0
    for f in range(W.shape[0]):
        Wf = W[f] - W[f].mean(0, keepdims=True)
        for off in range(W.shape[2] - len(motif) + 1):
            seg = Wf[:, off:off + len(motif)]
            best = max(best, float((seg * M).sum() / (np.linalg.norm(seg) * np.linalg.norm(M) + 1e-9)))
    return best
print(f"first-layer filter vs planted motif correlation: A = {best_match(A_MOTIF):.2f}, B = {best_match(B_MOTIF):.2f}")

# ---------------------------------------------------------------------------------------------
# Reverse-complement weight tying: pairing every filter with its reverse complement makes the layer RC-equivariant.
# ---------------------------------------------------------------------------------------------
def rc_x(x): return x.flip(-1)[:, [3, 2, 1, 0], :]
class RCConv(nn.Module):
    def __init__(s, n_filters=8, k=7):
        super().__init__(); s.w = nn.Parameter(torch.randn(n_filters, 4, k) * 0.3)
    def forward(s, x):
        w_rc = s.w.flip(-1)[:, [3, 2, 1, 0], :]                            # reverse complement of each filter
        return F.conv1d(x, torch.cat([s.w, w_rc], 0))                      # (B, 2F, L-k+1)
layer = RCConv(); x = F.one_hot(torch.randint(0, 4, (4, 50)), 4).permute(0, 2, 1).float()
out, out_rc = layer(x), layer(rc_x(x)); nf = layer.w.shape[0]
expected = torch.cat([out[:, nf:], out[:, :nf]], 1).flip(-1)                # swap filter/RC-filter channels, reverse positions
print(f"RC-tied conv layer: max |f(rc(x)) - T(f(x))| = {(out_rc - expected).abs().max().item():.1e}  (equivariant)")
