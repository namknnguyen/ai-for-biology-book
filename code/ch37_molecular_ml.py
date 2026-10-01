"""Chapter 37: molecular machine learning for drug discovery, on real data (BACE-1 inhibitor potency, 1,513 molecules; Chapter 24).
Compares a random forest on Morgan count fingerprints, a nearest-neighbour similarity model, and a small graph convolutional network (3 layers, trained from scratch, ensemble of 3) under
a random split and a cluster split ('a new chemical series'), with the metrics that matter in a campaign: enrichment of potent compounds among the top-ranked, hits found for a fixed
synthesis budget, and whether predictive uncertainty tells you when to distrust the model.  5-fold cross-validation; all numbers are means over folds."""
import os, urllib.request, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore"); torch.set_num_threads(2); torch.manual_seed(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bace.csv")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv", path)
df = pd.read_csv(path)[["mol", "pIC50"]].dropna().reset_index(drop=True)
mols = [Chem.MolFromSmiles(s) for s in df.mol]; ok = [m is not None for m in mols]; df = df[ok].reset_index(drop=True); mols = [m for m in mols if m is not None]
y = df.pIC50.values.astype(np.float32); n = len(y)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.stack([gen.GetCountFingerprintAsNumPy(m).astype(np.float32) for m in mols]); fps = [gen.GetFingerprint(m) for m in mols]
sim = np.stack([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps]).astype(np.float32)
dist = []
for i in range(1, n): dist += list(1 - sim[i, :i])
cl = np.zeros(n, int)
for k, c in enumerate(Butina.ClusterData(dist, n, 0.6, isDistData=True)): cl[list(c)] = k
ncl = cl.max() + 1
# ---------------------------------------------------------------- graph featurization (dense, padded)
ELEM = ["C", "N", "O", "F", "S", "Cl", "Br"]; HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
def atom_feat(a):
    f = [a.GetSymbol() == e for e in ELEM] + [a.GetSymbol() not in ELEM] + [a.GetDegree() == d for d in range(6)] + [a.GetFormalCharge() == c for c in (-1, 0, 1)]
    f += [a.GetIsAromatic()] + [a.GetTotalNumHs() == h for h in range(5)] + [a.GetHybridization() == h for h in HYB] + [a.IsInRing()]
    return np.array(f, np.float32)
NMAX = max(m.GetNumAtoms() for m in mols); D_IN = len(atom_feat(mols[0].GetAtomWithIdx(0)))
H0 = np.zeros((n, NMAX, D_IN), np.float32); A = np.zeros((n, NMAX, NMAX), np.float32); mask = np.zeros((n, NMAX), np.float32)
for i, m in enumerate(mols):
    na = m.GetNumAtoms(); mask[i, :na] = 1
    for a in m.GetAtoms(): H0[i, a.GetIdx()] = atom_feat(a)
    for b in m.GetBonds(): u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); A[i, u, v] = A[i, v, u] = 1
    A[i, np.arange(na), np.arange(na)] = 1                                                   # self-loops
deg = A.sum(2); Anorm = A / np.sqrt(deg[:, :, None] * deg[:, None, :] + 1e-9)                # D^-1/2 (A+I) D^-1/2
H0t, At, Mt = torch.tensor(H0), torch.tensor(Anorm), torch.tensor(mask)
class GCN(nn.Module):
    def __init__(self, d=96):
        super().__init__(); self.inp = nn.Linear(D_IN, d); self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, h, a, m):                                                              # h: (B, N, F), a: (B, N, N), m: (B, N)
        x = torch.relu(self.inp(h))
        for lin in self.layers: x = x + torch.relu(lin(a @ x))                               # residual graph convolution: (B, N, d)
        x = x * m[:, :, None]; mean = x.sum(1) / m.sum(1, keepdim=True); mx = x.max(1).values
        return self.head(torch.cat([mean, mx], 1))[:, 0]
def train_gcn(tr, seed, epochs=80):
    torch.manual_seed(seed); mu, sd = y[tr].mean(), y[tr].std(); net = GCN(); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-5)
    yt = torch.tensor((y - mu) / sd); tr = np.array(tr)
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(seed * 1000 + ep).permutation(len(tr))
        for b in range(0, len(tr), 64):
            idx = tr[perm[b:b + 64]]; loss = ((net(H0t[idx], At[idx], Mt[idx]) - yt[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net, mu, sd
def predict_gcn(net, mu, sd, te):
    with torch.no_grad(): return net(H0t[te], At[te], Mt[te]).numpy() * sd + mu
def knn_pred(tr, te, k=5):
    s = sim[np.ix_(te, tr)]; idx = np.argsort(-s, 1)[:, :k]; w = np.take_along_axis(s, idx, 1); return (w * y[tr][idx]).sum(1) / (w.sum(1) + 1e-9), s.max(1)

def folds(split, seed=0):
    r = np.random.default_rng(seed)
    if split == "random": perm = r.permutation(n); return [(np.setdiff1d(np.arange(n), perm[k::5]), perm[k::5]) for k in range(5)]
    order = r.permutation(ncl); gid = {c: i % 5 for i, c in enumerate(order)}; g = np.array([gid[c] for c in cl])
    return [(np.flatnonzero(g != k), np.flatnonzero(g == k)) for k in range(5)]
ACTIVE = 8.0; base_rate = (y >= ACTIVE).mean(); BUDGET = 0.05
print(f"{n} molecules, {ncl} Butina clusters; 'potent' = pIC50 >= {ACTIVE} ({base_rate:.1%} of the data); atoms padded to {NMAX}; GCN: 3 residual graph-convolution layers, width 96, ensemble of 3, 80 epochs (fixed in advance)")
print("metrics per model: Pearson r | RMSE | enrichment factor of potent compounds in the top 10% by prediction (1 = random; maximum %.1f) | potent hits among the 5%% of the test set chosen by prediction, as a multiple of random choice" % (1 / base_rate))
results = {}
for split in ("random", "cluster"):
    acc = {m: [] for m in ("RF", "kNN (Tanimoto, k=5)", "GCN (1 net)", "GCN ensemble (3)", "RF + GCN ensemble")}; unc = []
    for tr, te in folds(split):
        rf = RandomForestRegressor(300, max_features=0.3, n_jobs=2, random_state=0).fit(X[tr], y[tr]); trees = np.stack([t.predict(X[te]) for t in rf.estimators_]); p_rf = trees.mean(0)
        p_knn, maxsim = knn_pred(tr, te)
        nets = [train_gcn(tr, s) for s in range(3)]; P = np.stack([predict_gcn(*nt, te) for nt in nets]); p_g1, p_g3 = P[0], P.mean(0)
        for name, p in [("RF", p_rf), ("kNN (Tanimoto, k=5)", p_knn), ("GCN (1 net)", p_g1), ("GCN ensemble (3)", p_g3), ("RF + GCN ensemble", (p_rf + p_g3) / 2)]:
            top10 = np.argsort(-p)[: max(1, len(te) // 10)]; ef = (y[te][top10] >= ACTIVE).mean() / max((y[te] >= ACTIVE).mean(), 1e-9)
            k5 = np.argsort(-p)[: max(1, int(BUDGET * len(te)))]; hits = (y[te][k5] >= ACTIVE).mean() / max((y[te] >= ACTIVE).mean(), 1e-9)
            acc[name].append((pearsonr(p, y[te])[0], np.sqrt(np.mean((p - y[te]) ** 2)), ef, hits))
        unc.append((te, p_g3, P.std(0), trees.std(0), p_rf, maxsim))
    results[split] = (acc, unc)
    print(f"\n-- {split} split --")
    print("model                       Pearson r    RMSE     EF@10%    hits in top 5% (x random)")
    for name, v in acc.items():
        a = np.mean(v, 0); print(f"{name:26s}   {a[0]:6.3f}   {a[1]:6.3f}   {a[2]:6.2f}      {a[3]:6.2f}")
print("\n-- does predictive uncertainty tell you when to distrust a prediction? (Spearman correlation between predicted SD and |error|; coverage of mean +/- 1.645 SD intervals, nominal 0.90) --")
print("split     model                  rho(SD, |error|)   coverage of the 90% interval   rho(1 - max Tanimoto to training, |error|)")
for split in ("random", "cluster"):
    unc = results[split][1]
    for name, which in (("GCN ensemble (3)", 2), ("RF (tree spread)", 3)):
        rho, cov = [], []
        for te, pg, sg, sr, prf, ms in unc:
            p, s = (pg, sg) if which == 2 else (prf, sr); err = np.abs(p - y[te]); rho.append(spearmanr(s, err)[0]); cov.append(np.mean(err <= 1.645 * np.maximum(s, 1e-6)))
        rs = np.mean([spearmanr(1 - ms, np.abs((pg if which == 2 else prf) - y[te]))[0] for te, pg, sg, sr, prf, ms in unc])
        print(f"{split:8s}  {name:21s}  {np.mean(rho):8.3f}                  {np.mean(cov):8.3f}                       {rs:8.3f}")
