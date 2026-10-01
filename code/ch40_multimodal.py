"""Chapter 40: aligning two modalities of the same cells, in a simulation with known structure.
Cells carry a 6-dimensional latent state z with 6 clusters (cell types).  Modality X (60 features, 'RNA') sees z[0:2] (X-private) and z[4:6] (shared); modality Y (40 features, 'chromatin or protein') sees
z[2:4] (Y-private, through a saturating nonlinearity) and z[4:6]; each has its own nuisance variables (4 dimensions, standard deviation 1.5) and measurement noise.  Questions:
1. Paired data: how many paired cells does cross-modal retrieval need, for a linear method (CCA) and a contrastive (InfoNCE) dual encoder?
2. What does an alignment objective keep?  Cell-type accuracy from X alone, Y alone, both, and from shared embeddings.
3. Unpaired data: Gromov-Wasserstein matching when the two datasets have the same versus different cell-type composition; value of a few anchor pairs."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
K, DX, DY = 6, 60, 40
centers = rng.normal(0, 2.0, (K, 6)); Ax = rng.normal(0, 1, (DX, 4)) / 2; Cx = rng.normal(0, 1, (DX, 4)) / 2; By = rng.normal(0, 1, (DY, 4)) / 1.5; Cy = rng.normal(0, 1, (DY, 4)) / 2
def sample(n, probs=None, seed=0):
    r = np.random.default_rng(seed); k = r.choice(K, n, p=probs); z = centers[k] + 0.7 * r.standard_normal((n, 6))
    X = np.c_[z[:, 0:2], z[:, 4:6]] @ Ax.T + 1.5 * r.standard_normal((n, 4)) @ Cx.T + 0.8 * r.standard_normal((n, DX))
    Y = np.tanh(np.c_[z[:, 2:4], z[:, 4:6]]) @ By.T * 2 + 1.5 * r.standard_normal((n, 4)) @ Cy.T + 0.8 * r.standard_normal((n, DY))
    return X.astype(np.float32), Y.astype(np.float32), k
def sample_full(n, probs=None, seed=0):
    """Control: both modalities see the full 6-d state (no modality-private structure)."""
    r = np.random.default_rng(seed); k = r.choice(K, n, p=probs); z = centers[k] + 0.7 * r.standard_normal((n, 6)); Af = np.random.default_rng(5).normal(0, 1, (DX, 6)) / 2; Bf = np.random.default_rng(6).normal(0, 1, (DY, 6)) / 1.5
    return (z @ Af.T + 0.8 * r.standard_normal((n, DX))).astype(np.float32), (np.tanh(z) @ Bf.T * 2 + 0.8 * r.standard_normal((n, DY))).astype(np.float32), k
Xte, Yte, kte = sample(500, seed=999); Xtr_all, Ytr_all, ktr_all = sample(5000, seed=1)
def retrieval(Zx, Zy):
    Zx = Zx / (np.linalg.norm(Zx, axis=1, keepdims=True) + 1e-9); Zy = Zy / (np.linalg.norm(Zy, axis=1, keepdims=True) + 1e-9); S = Zy @ Zx.T          # row i: query y_i against all x_j
    rank = (S > np.diag(S)[:, None]).sum(1); return (rank == 0).mean(), (rank < 5).mean()
class Enc(nn.Module):
    def __init__(self, d_in, d=6): super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)
def contrastive(X, Y, steps=600, temp=0.1):
    ex, ey = Enc(DX), Enc(DY); opt = torch.optim.Adam(list(ex.parameters()) + list(ey.parameters()), 2e-3, weight_decay=1e-4); Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        idx = torch.randint(0, len(Xt), (min(256, len(Xt)),)); a, b = ex(Xt[idx]), ey(Yt[idx]); logits = a @ b.T / temp; lab = torch.arange(len(idx))
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2; opt.zero_grad(); loss.backward(); opt.step()
    return ex.eval(), ey.eval()
import os
if os.environ.get('SKIP12') != '1':
    print("== 1. Paired cells needed for cross-modal retrieval (500 held-out cells; chance top-1 = 0.002, top-5 = 0.010) ==")
    print("paired cells    CCA (6 components) top-1 / top-5     contrastive dual encoder top-1 / top-5")
    for n in (50, 200, 1000, 5000):
        Xp, Yp = Xtr_all[:n], Ytr_all[:n]
        try:
            cca = CCA(n_components=6, max_iter=2000, scale=True).fit(Xp, Yp) if n > 100 else None
            c = retrieval(*[a for a in cca.transform(Xte, Yte)]) if cca is not None else (float("nan"), float("nan"))
        except Exception: c = (float("nan"), float("nan"))
        ex, ey = contrastive(Xp, Yp, steps=700)
        with torch.no_grad(): d = retrieval(ex(torch.tensor(Xte)).numpy(), ey(torch.tensor(Yte)).numpy())
        print(f"{n:8d}        {c[0]:6.3f} / {c[1]:6.3f}                      {d[0]:6.3f} / {d[1]:6.3f}")
    print("(CCA is not fit with fewer than 100 pairs: the covariance estimates are rank-deficient)")

    print("\n== 2. What an alignment objective keeps: cell-type accuracy (6 types; chance 0.17) with a logistic classifier trained on 1,000 labeled cells ==")
    def clf_acc(Ftr, Fte): return LogisticRegression(max_iter=2000).fit(Ftr, ktr_all[:1000]).score(Fte, kte)
    ex, ey = contrastive(Xtr_all, Ytr_all, steps=1200)
    with torch.no_grad(): sx_tr, sy_tr, sx_te, sy_te = [f(torch.tensor(a)).numpy() for f, a in ((ex, Xtr_all[:1000]), (ey, Ytr_all[:1000]), (ex, Xte), (ey, Yte))]
    cca = CCA(n_components=6, max_iter=3000).fit(Xtr_all, Ytr_all); cx_tr, cy_tr = cca.transform(Xtr_all[:1000], Ytr_all[:1000]); cx_te, cy_te = cca.transform(Xte, Yte)
    rows = [("X alone (RNA-like, raw)", Xtr_all[:1000], Xte), ("Y alone (raw)", Ytr_all[:1000], Yte), ("X and Y concatenated (raw)", np.c_[Xtr_all[:1000], Ytr_all[:1000]], np.c_[Xte, Yte]),
            ("shared embedding from X (contrastive, 6-d)", sx_tr, sx_te), ("shared embedding from Y (contrastive, 6-d)", sy_tr, sy_te), ("shared embeddings of X and Y concatenated (contrastive)", np.c_[sx_tr, sy_tr], np.c_[sx_te, sy_te]),
            ("CCA, X side (6-d)", cx_tr, cx_te), ("CCA, X and Y sides concatenated", np.c_[cx_tr, cy_tr], np.c_[cx_te, cy_te])]
    for name, a, b in rows: print(f"{name:58s} {clf_acc(a, b):.3f}")
    print("reading: the cell types are separable using X-private, Y-private, and shared dimensions; shared embeddings keep only what both modalities see")


print("\n== 3. Unpaired alignment by entropic Gromov-Wasserstein (400 cells per modality, no pairs) ==")
def gw(D1, D2, eps=0.01, outer=40, inner=80):
    n, m = len(D1), len(D2); p, q = np.ones(n) / n, np.ones(m) / m; D1 = D1 / D1.max(); D2 = D2 / D2.max(); T = np.outer(p, q)
    c1, c2 = (D1 ** 2) @ p, (D2 ** 2) @ q
    for _ in range(outer):
        cost = c1[:, None] + c2[None, :] - 2 * D1 @ T @ D2.T; Kmat = np.exp(-(cost - cost.min()) / eps); u = np.ones(n) / n
        for _ in range(inner): v = q / (Kmat.T @ u + 1e-300); u = p / (Kmat @ v + 1e-300)
        T = u[:, None] * Kmat * v[None, :]
    return T
def pdist(A): sq = (A ** 2).sum(1); return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0))
def standardize(A): return (A - A.mean(0)) / (A.std(0) + 1e-6)
# only the SHARED structure is comparable across modalities: use each modality's top principal components of a shared-aware projection (CCA directions from 0 anchors are unavailable), so use PCA
from sklearn.decomposition import PCA
def unpaired_acc(probs_x, probs_y, seed, n_anchor=0, full=False):
    smp = sample_full if full else sample; X, _, kx = smp(400, probs_x, seed=seed); _, Y, ky = smp(400, probs_y, seed=seed + 50)
    Px = PCA(6).fit_transform(standardize(X)); Py = PCA(6).fit_transform(standardize(Y)); T = gw(pdist(Px), pdist(Py))
    match = T.argmax(1); acc0 = (ky[match] == kx).mean()
    if n_anchor == 0: return acc0
    # anchors: a few cells known to be the same TYPE in both modalities (e.g., from marker genes); fit CCA on the 6-d PCA coordinates of the anchors, then match every X cell to its nearest Y cell in the CCA space
    r = np.random.default_rng(seed); ax = [r.choice(np.flatnonzero(kx == k), 1)[0] for k in range(K) for _ in range(max(1, n_anchor // K))]; ay = [r.choice(np.flatnonzero(ky == kx[a]), 1)[0] for a in ax]
    cca = CCA(n_components=5, max_iter=2000).fit(Px[ax], Py[ay]); ux, uy = cca.transform(Px, Py); ux, uy = standardize(ux), standardize(uy)
    nn_ = np.argmin(((ux[:, None, :] - uy[None, :, :]) ** 2).sum(2), 1); return (ky[nn_] == kx).mean()
equal = np.ones(K) / K; shifted_y = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03]); shifted_x = shifted_y[::-1]
print("matching accuracy = fraction of X cells matched to a Y cell of the same cell type (chance = 0.17 for equal composition; 0.26 for the different composition, where matching everything to the commonest type would give the share of that type)")
print("modalities                       composition of the two datasets          GW, no anchors    with 12 anchor pairs   with 60 anchor pairs   (mean of 6 seeds)")
for mod, full in (("modality-private structure", False), ("fully shared geometry", True)):
    for name, px, py in (("same (equal proportions in both)", equal, equal), ("different (40% / 3% extremes, reversed)", shifted_x, shifted_y)):
        r0 = np.mean([unpaired_acc(px, py, s, 0, full) for s in range(6)]); r6 = np.mean([unpaired_acc(px, py, s, 12, full) for s in range(6)]); r30 = np.mean([unpaired_acc(px, py, s, 60, full) for s in range(6)])
        print(f"{mod:32s} {name:40s} {r0:6.3f}           {r6:6.3f}              {r30:6.3f}")
