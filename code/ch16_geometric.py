"""Chapter 16: permutation equivariance, 1-WL limits, E(n)-equivariant message passing, over-smoothing, degree bias."""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Message passing with sum aggregation is permutation-equivariant; graph readout by sum is invariant.
# ---------------------------------------------------------------------------------------------
class MPNN(nn.Module):
    def __init__(s, d=16, layers=3):
        super().__init__(); s.inp = nn.Linear(1, d)
        s.msg = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.ReLU()) for _ in range(layers))
        s.upd = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, d), nn.ReLU()) for _ in range(layers))
    def forward(s, A, x):                                            # A: (n,n) 0/1 adjacency, x: (n,1) node features
        h = s.inp(x)
        for msg, upd in zip(s.msg, s.upd):
            n = len(h); hi, hj = h[:, None, :].expand(n, n, -1), h[None, :, :].expand(n, n, -1)
            m = (A[:, :, None] * msg(torch.cat([hi, hj], -1))).sum(1)    # sum over neighbours j of message(h_i, h_j)
            h = upd(torch.cat([h, m], -1))
        return h
net = MPNN()
n = 8; A = (torch.rand(n, n) < 0.35).float(); A = ((A + A.T) > 0).float(); A.fill_diagonal_(0); x = torch.randn(n, 1)
P = torch.eye(n)[torch.randperm(n)]
h, hp = net(A, x), net(P @ A @ P.T, P @ x)
print(f"permutation equivariance: max |f(PAP^T, Px) - P f(A, x)| = {(hp - P @ h).abs().max().item():.1e}; "
      f"sum-readout invariance: {(hp.sum(0) - h.sum(0)).abs().max().item():.1e}")

# ---------------------------------------------------------------------------------------------
# 2. Expressivity: message passing cannot tell a 6-cycle from two disjoint triangles (both 2-regular, 6 nodes).
# ---------------------------------------------------------------------------------------------
def cycle(k, offset, n):
    A = torch.zeros(n, n)
    for i in range(k): A[offset + i, offset + (i + 1) % k] = A[offset + (i + 1) % k, offset + i] = 1
    return A
hexagon, two_triangles = cycle(6, 0, 6), cycle(3, 0, 6) + cycle(3, 3, 6)
ones = torch.ones(6, 1)
g1, g2 = net(hexagon, ones).sum(0), net(two_triangles, ones).sum(0)
print(f"1-WL limit: ||embedding(C6) - embedding(2 x C3)|| = {(g1 - g2).norm().item():.1e}  (identical, though the graphs differ)")

# ---------------------------------------------------------------------------------------------
# 3. EGNN layer: E(n)-equivariant coordinate updates built from invariant distances.
# ---------------------------------------------------------------------------------------------
class EGNNLayer(nn.Module):
    def __init__(s, d=16):
        super().__init__()
        s.phi_e = nn.Sequential(nn.Linear(2 * d + 1, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        s.phi_x = nn.Linear(d, 1); s.phi_h = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
    def forward(s, h, x):                                            # h: (n,d) invariant features, x: (n,3) coordinates
        n = len(h); diff = x[:, None, :] - x[None, :, :]; d2 = (diff ** 2).sum(-1, keepdim=True)   # (n,n,1) invariant
        m = s.phi_e(torch.cat([h[:, None].expand(n, n, -1), h[None].expand(n, n, -1), d2], -1))      # messages from invariants only
        x_new = x + (diff * s.phi_x(m) * (1 - torch.eye(n))[..., None]).sum(1) / (n - 1)                # move along x_i - x_j
        return s.phi_h(torch.cat([h, m.sum(1)], -1)), x_new
layer = EGNNLayer(); h0, x0 = torch.randn(7, 16), torch.randn(7, 3)
Q, _ = torch.linalg.qr(torch.randn(3, 3)); Rrot = Q * torch.det(Q).sign()      # random proper rotation
shift = torch.randn(1, 3)
h1, x1 = layer(h0, x0); h2, x2 = layer(h0, x0 @ Rrot.T + shift)
print(f"EGNN: coordinates equivariant: max |x'(Rx+t) - (Rx'+t)| = {(x2 - (x1 @ Rrot.T + shift)).abs().max().item():.1e}; "
      f"features invariant: {(h2 - h1).abs().max().item():.1e}")
mlp = nn.Sequential(nn.Linear(21, 32), nn.ReLU(), nn.Linear(32, 3))
f = lambda xx: mlp(xx.flatten()[None]).view(3)                       # a coordinate-input MLP is not equivariant
print(f"naive MLP on raw coordinates: ||f(Rx) - R f(x)|| = {(f(x0[:7] @ Rrot.T)[:3] - Rrot @ f(x0[:7])).norm().item():.2f}  (not equivariant)")
# reflections: EGNN is E(3)-equivariant (includes mirror images); proteins are chiral, so SE(3) is the right group for them
refl = torch.diag(torch.tensor([-1., 1., 1.]))
h3, _ = layer(h0, x0 @ refl.T)
print(f"EGNN invariant features under a mirror image: max difference = {(h3 - h1).abs().max().item():.1e} (it cannot see chirality)")

# ---------------------------------------------------------------------------------------------
# 4. Over-smoothing: repeated neighbourhood averaging drives all node features together.
# ---------------------------------------------------------------------------------------------
n = 200; pts = rng.random((n, 2)); D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
Aadj = ((D < 0.2) & (D > 0)).astype(float); At = Aadj + np.eye(n)                  # geometric graph + self loops
Dm = np.diag(1 / np.sqrt(At.sum(1))); S = Dm @ At @ Dm                                # GCN propagation matrix
lam2 = np.sort(np.abs(np.linalg.eigvalsh(S)))[-2]
H = rng.normal(size=(n, 8)); stds = []
for k in range(0, 41, 10):
    Hk = np.linalg.matrix_power(S, k) @ H; stds.append(np.std(Hk / np.sqrt(np.diag(At))[:, None], axis=0).mean())
print(f"over-smoothing: feature spread across nodes after k = 0,10,20,30,40 propagations = {np.round(stds, 4).tolist()}  (second eigenvalue {lam2:.3f})")

# ---------------------------------------------------------------------------------------------
# 5. Degree bias in link prediction: a preferential-attachment network, 10% of edges held out.
# ---------------------------------------------------------------------------------------------
N, m = 2000, 3
deg = np.ones(N); edges = set()
for v in range(m, N):
    targets = rng.choice(v, size=m, replace=False, p=deg[:v] / deg[:v].sum())
    for u in targets: edges.add((int(u), v)); deg[u] += 1; deg[v] += 1
edges = list(edges); rng.shuffle(edges); n_test = len(edges) // 10
test, train = edges[:n_test], edges[n_test:]
dtr = np.zeros(N)
for u, v in train: dtr[u] += 1; dtr[v] += 1
def score(u, v): return max(dtr[u], dtr[v])                                             # a preferential-attachment-style score
def auc_for(negatives):
    s_pos = [score(u, v) for u, v in test]; s_neg = [score(u, v) for u, v in negatives]
    return roc_auc_score([1] * len(s_pos) + [0] * len(s_neg), s_pos + s_neg)
train_nbrs = {i: set() for i in range(N)}
for u, v in train + test: train_nbrs[u].add(v); train_nbrs[v].add(u)
uniform_neg = [tuple(rng.integers(0, N, 2)) for _ in range(n_test)]
bins = np.floor(np.log2(dtr + 1)).astype(int)
matched_neg = []
for u, v in test:                                                                        # keep u, replace v by a node of the same degree bin
    pool = np.nonzero((bins == bins[v]) & (np.arange(N) != u))[0]
    matched_neg.append((u, int(rng.choice(pool))))
print(f"link prediction by degree alone: AUROC with uniformly random negatives = {auc_for(uniform_neg):.3f}; "
      f"with degree-matched negatives = {auc_for(matched_neg):.3f}")
