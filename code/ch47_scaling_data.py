"""Chapter 47: learning curves, ceilings, and the value of data diversity, on real data (BACE-1 inhibitor potency; Chapter 24).
1. Learning curves of a random forest on Morgan fingerprints under a random split and a cluster split; power-law fits with a floor.
2. Diversity versus volume: the same number of training molecules drawn from few versus many chemical clusters, tested on held-out clusters.
3. What a fitted curve predicts for a 10x larger dataset, and what the noise ceiling allows."""
import os, urllib.request, warnings
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bace.csv")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv", path)
df = pd.read_csv(path)[["mol", "pIC50"]].dropna().reset_index(drop=True)
mols = [Chem.MolFromSmiles(s) for s in df.mol]; ok = [m is not None for m in mols]; df = df[ok].reset_index(drop=True); mols = [m for m in mols if m is not None]
y = df.pIC50.values.astype(float); n = len(y)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.stack([gen.GetCountFingerprintAsNumPy(m).astype(np.float32) for m in mols]); fps = [gen.GetFingerprint(m) for m in mols]
dist = []
for i in range(1, n): dist += [1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
clusters = Butina.ClusterData(dist, n, 0.6, isDistData=True); cl = np.zeros(n, int)
for k, c in enumerate(clusters): cl[list(c)] = k
ncl = len(clusters); print(f"{n} molecules, {ncl} Butina clusters (Tanimoto >= 0.4 to the centroid); sd of pIC50 = {y.std():.2f}")
def rmse(tr, te, seed=0):
    rf = RandomForestRegressor(200, max_features=0.3, n_jobs=2, random_state=seed).fit(X[tr], y[tr]); return np.sqrt(np.mean((rf.predict(X[te]) - y[te]) ** 2))

# ---------------------------------------------------------------- 1. learning curves
print("\n== 1. Learning curves: RMSE (pIC50 units) of a random forest versus training-set size ==")
sizes = [50, 100, 200, 400, 800]
curves = {"random split": [], "cluster split": []}
for name in curves:
    for m in sizes:
        e = []
        for rep in range(6):
            r = np.random.default_rng(rep)
            if name == "random split":
                perm = r.permutation(n); te = perm[:400]; pool = perm[400:]
            else:
                order = r.permutation(ncl); te_c = set(order[: max(1, ncl // 4)]); te = np.flatnonzero(np.isin(cl, list(te_c))); pool = np.flatnonzero(~np.isin(cl, list(te_c)))
            tr = r.choice(pool, min(m, len(pool)), replace=False); e.append(rmse(tr, te, rep))
        curves[name].append((np.mean(e), np.std(e) / np.sqrt(len(e))))
print("n train    random split RMSE (SE)    cluster split RMSE (SE)")
for i, m in enumerate(sizes): print(f"{m:6d}     {curves['random split'][i][0]:.3f} ({curves['random split'][i][1]:.3f})            {curves['cluster split'][i][0]:.3f} ({curves['cluster split'][i][1]:.3f})")
def pl(nn, a, b, c): return a * nn ** (-b) + c
print("\npower-law fits  RMSE = a * n^-b + c")
for name in curves:
    yv = np.array([c[0] for c in curves[name]])
    try:
        p, _ = curve_fit(pl, np.array(sizes, float), yv, p0=[2.0, 0.5, 0.5], bounds=([0, 0.01, 0.0], [50, 3, 1.5]), maxfev=20000)
        print(f"{name:14s}: a = {p[0]:.2f}, b = {p[1]:.2f}, floor c = {p[2]:.3f};  predicted RMSE at n = 1,200: {pl(1200, *p):.3f}; at 12,000: {pl(12000, *p):.3f}; at 120,000: {pl(120000, *p):.3f}")
    except Exception as ex: print(name, "fit failed", ex)
print("noise ceiling check: a single-measurement noise SD of 0.54 pIC50 (Chapter 24) is an RMSE floor of 0.54 for any model")

# ---------------------------------------------------------------- 2. diversity vs volume
NTR = 100
print(f"\n== 2. The same number of training molecules ({NTR}) from few vs many clusters; test = molecules from held-out clusters ==")
sizes_cl = np.array([(cl == k).sum() for k in range(ncl)])
print("clusters used   training molecules   test RMSE (SE over 15 draws)")
for k_cl in [1, 2, 5, 10, 20, 40]:
    e = []
    for rep in range(15):
        r = np.random.default_rng(100 + rep); order = r.permutation(ncl); te_c = order[: ncl // 4]; te = np.flatnonzero(np.isin(cl, te_c))
        cand = [c for c in order[ncl // 4:] if sizes_cl[c] >= max(3, NTR // k_cl // 2)]
        for _ in range(200):                                                      # draw clusters until their union holds enough molecules
            tc = list(r.choice(cand, k_cl, replace=False)); pool = np.flatnonzero(np.isin(cl, tc))
            if len(pool) >= NTR: break
        else: continue
        # equalize the contribution of clusters as far as possible: sample round-robin
        tr = []; groups = [list(r.permutation(np.flatnonzero(cl == c))) for c in tc]
        while len(tr) < NTR and any(groups):
            for g in groups:
                if g and len(tr) < NTR: tr.append(g.pop())
        e.append(rmse(np.array(tr), te, rep))
    print(f"{k_cl:8d}        {NTR:8d}            {np.mean(e):.3f} ({np.std(e) / np.sqrt(len(e)):.3f})   [{len(e)} valid draws]")
