"""Chapter 24: molecular representation, binding free energy, and the evaluation of molecular models on real data.
Datasets (public, MoleculeNet): BACE-1 inhibitor pIC50 (Subramanian et al. 2016) and ESOL aqueous solubility (Delaney 2004)."""
import os, urllib.request, warnings
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, GroupKFold
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
def fetch(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve(BASE + name, path)
    return pd.read_csv(path)

# ---------------------------------------------------------------- 1. representation
print("== 1. One molecule, many strings; one graph, two molecules ==")
aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
rand = {Chem.MolToSmiles(aspirin, doRandom=True, canonical=False) for _ in range(2000)}
back = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in rand}
print(f"aspirin: canonical = {Chem.MolToSmiles(aspirin)};  distinct valid SMILES strings in 2000 random draws = {len(rand)};  all map back to {len(back)} canonical string")
caff = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
print(f"caffeine (24 atoms with H, 14 heavy): distinct SMILES in 2000 draws = {len({Chem.MolToSmiles(caff, doRandom=True, canonical=False) for _ in range(2000)})}")
r_carv, s_carv = Chem.MolFromSmiles("CC1=CC[C@H](CC1=O)C(C)=C"), Chem.MolFromSmiles("CC1=CC[C@@H](CC1=O)C(C)=C")
fp2d = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)                       # no chirality
fp3d = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
t2 = DataStructs.TanimotoSimilarity(fp2d.GetFingerprint(r_carv), fp2d.GetFingerprint(s_carv))
t3 = DataStructs.TanimotoSimilarity(fp3d.GetFingerprint(r_carv), fp3d.GetFingerprint(s_carv))
print("carvone enantiomers: CIP labels", Chem.FindMolChiralCenters(r_carv), Chem.FindMolChiralCenters(s_carv),
      f"| Tanimoto (Morgan r=2, chirality ignored) = {t2:.2f}; (chirality-aware) = {t3:.2f}")
print("same molecular formula and the same 2D graph, identical standard descriptors:",
      Descriptors.MolWt(r_carv) == Descriptors.MolWt(s_carv), Crippen.MolLogP(r_carv) == Crippen.MolLogP(s_carv))

# ---------------------------------------------------------------- 2. BACE data
bace = fetch("bace.csv")[["mol", "pIC50"]].dropna().reset_index(drop=True)
mols = [Chem.MolFromSmiles(s) for s in bace.mol]; keep = [m is not None for m in mols]
bace = bace[keep].reset_index(drop=True); mols = [m for m in mols if m is not None]
y = bace.pIC50.values.astype(float); n = len(y)
RT = 0.593                                                                    # kcal/mol at 298 K
print(f"\n== 2. BACE-1 inhibitors: n = {n}, pIC50 mean {y.mean():.2f}, sd {y.std():.2f}, range [{y.min():.1f}, {y.max():.1f}] ==")
dG = -RT * np.log(10) * y                                                      # kcal/mol, IC50 ~ Kd approximation
print(f"binding free energy  dG = -RT ln(10) pIC50:  1 pIC50 unit = {RT*np.log(10):.2f} kcal/mol; dataset range {dG.min():.1f} to {dG.max():.1f} kcal/mol")
ha = np.array([m.GetNumHeavyAtoms() for m in mols]); LE = -dG / ha
print(f"heavy atoms: mean {ha.mean():.1f}; ligand efficiency LE = -dG/N_heavy: mean {LE.mean():.2f}, 95th pct {np.percentile(LE,95):.2f} kcal/mol/atom")
print(f"corr(pIC50, heavy-atom count) = {np.corrcoef(y, ha)[0,1]:.2f}  -> size alone explains R^2 = {np.corrcoef(y, ha)[0,1]**2:.2f}")

gen_c = rdFingerprintGenerator.GetMorganGenerator(radius=2)
gen_f = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def counts(m):
    return gen_f.GetCountFingerprintAsNumPy(m).astype(np.float32)
X = np.stack([counts(m) for m in mols])
bits = [gen_f.GetFingerprint(m) for m in mols]
unfolded = [set(gen_c.GetSparseCountFingerprint(m).GetNonzeroElements().keys()) for m in mols]
allenv = set().union(*unfolded)
print(f"distinct Morgan (r<=2) environments in {n} molecules: {len(allenv)}; folded to 2048 bits, mean bit occupancy {np.mean(X>0):.3f}; ", end="")
col = {}
for e in allenv: col.setdefault(e % 2048, []).append(e)
print(f"{np.mean([len(v) > 1 for v in col.values()]):.2f} of occupied bits are shared by >1 distinct environment")

# ---------------------------------------------------------------- 3. similarity-property principle & activity cliffs
print("\n== 3. Similarity vs. property difference (all pairs) ==")
sim = np.zeros((n, n), np.float32)
for i in range(n): sim[i] = DataStructs.BulkTanimotoSimilarity(bits[i], bits)
iu = np.triu_indices(n, 1); S = sim[iu]; D = np.abs(y[iu[0]] - y[iu[1]])
print("Tanimoto bin   pairs      mean |dpIC50|   P(|dpIC50|>=1)   P(|dpIC50|>=2)")
for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
    m = (S >= lo) & (S < hi)
    print(f"[{lo:.1f}, {min(hi,1):.1f}{')' if hi<1 else ']'}   {m.sum():9d}   {D[m].mean():8.2f}        {np.mean(D[m]>=1):6.3f}           {np.mean(D[m]>=2):6.3f}")
m = S >= 0.7
print(f"pairs with Tanimoto >= 0.7: {m.sum()};  of these, {np.sum(D[m]>=2)} differ by >= 100-fold in IC50 (activity cliffs) and {np.sum(D[m]>=1)} by >= 10-fold")


m1 = S >= 0.999                                                               # identical chirality-blind fingerprints
fpc = [fp3d.GetFingerprint(m) for m in mols]
I1, J1 = iu[0][m1], iu[1][m1]
split_by_chi = np.mean([DataStructs.TanimotoSimilarity(fpc[a], fpc[b]) < 1 for a, b in zip(I1, J1)])
print(f"pairs with identical chirality-blind fingerprints (Tanimoto = 1): {m1.sum()}; mean |dpIC50| = {D[m1].mean():.2f}, "
      f"{np.mean(D[m1]>=1):.2f} differ by >= 10-fold; a chirality-aware fingerprint separates {split_by_chi:.2f} of them")

# ---------------------------------------------------------------- 4. splits
print("\n== 4. Generalization depends on how the test set is chosen (RF on Morgan counts, 5-fold CV) ==")
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
sg = pd.factorize(np.array(scaf))[0]
dist = [1 - x for i in range(1, n) for x in sim[i, :i]]                      # Butina expects condensed lower triangle
cl = Butina.ClusterData(dist, n, 0.6, isDistData=True)                      # clusters with Tanimoto >= 0.4 to the centroid
cg = np.zeros(n, int)
for k, c in enumerate(cl): cg[list(c)] = k
print(f"{len(set(sg))} Bemis-Murcko scaffolds, {len(cl)} Butina clusters (distance cutoff 0.6)")
def cv(split, groups=None, seed=0):
    folds = KFold(5, shuffle=True, random_state=seed).split(X) if groups is None else GroupKFold(5).split(X, groups=groups)
    pred = np.zeros(n); maxsim = np.zeros(n)
    for tr, te in folds:
        rf = RandomForestRegressor(300, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
        pred[te] = rf.predict(X[te]); maxsim[te] = sim[np.ix_(te, tr)].max(1)
    return pred, maxsim
res = {}
for name, g in [("random", None), ("scaffold", sg), ("cluster", cg)]:
    p, ms = cv(name, g); res[name] = (p, ms)
    print(f"{name:9s}: Pearson r = {np.corrcoef(p, y)[0,1]:.3f}, RMSE = {np.sqrt(np.mean((p-y)**2)):.3f} pIC50, mean max-Tanimoto test->train = {ms.mean():.2f}")
print(f"baseline predicting the mean: RMSE = {y.std():.3f}")
print("random-split RF, error by similarity to the nearest training molecule:")
p, ms = res["random"]
for lo, hi in [(0, .5), (.5, .65), (.65, .8), (.8, .95), (.95, 1.01)]:
    mm = (ms >= lo) & (ms < hi)
    if mm.sum(): print(f"   nearest-train Tanimoto in [{lo:.2f},{min(hi,1):.2f}): n = {mm.sum():4d}, RMSE = {np.sqrt(np.mean((p[mm]-y[mm])**2)):.3f}")

# ---------------------------------------------------------------- 5. activity-cliff prediction
print("\n== 5. Can the (random-split) model rank the members of a cliff pair? ==")
p = res["random"][0]; I, J = iu
sel = (S >= 0.7) & (D >= 1.5)
dp = p[I[sel]] - p[J[sel]]; dt = y[I[sel]] - y[J[sel]]
print(f"{sel.sum()} pairs with Tanimoto >= 0.7 and |dpIC50| >= 1.5; model gets the sign of the difference right for {np.mean(np.sign(dp)==np.sign(dt)):.2f}; "
      f"mean |predicted difference| = {np.abs(dp).mean():.2f} vs true {np.abs(dt).mean():.2f}")

# ---------------------------------------------------------------- 6. measurement noise ceiling
sd_rep = 0.54                                                                   # noise SD of one published potency value (Kramer et al. 2012; see text)
print(f"\n== 6. Noise ceiling ==\nif a single reported pIC50 has noise SD {sd_rep}, the best possible R^2 against such labels is {1 - sd_rep**2/y.var():.2f} (r <= {np.sqrt(1 - sd_rep**2/y.var()):.2f}), RMSE >= {sd_rep:.2f}")

# ---------------------------------------------------------------- 7. ESOL: inductive bias and shift
es = fetch("delaney-processed.csv"); es = es.rename(columns={"measured log solubility in mols per litre": "logS", "smiles": "smi"})
em = [Chem.MolFromSmiles(s) for s in es.smi]; ey = es.logS.values.astype(float); ne = len(ey)
def desc(m):
    heavy = m.GetNumHeavyAtoms(); arom = sum(a.GetIsAromatic() for a in m.GetAtoms())
    return [Crippen.MolLogP(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRotatableBonds(m), arom / heavy]
Xd = np.array([desc(m) for m in em]); Xe = np.stack([counts(m) for m in em])
esc = pd.factorize(np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in em]))[0]
print(f"\n== 7. ESOL aqueous solubility, n = {ne}: four physical descriptors vs. 2048-d fingerprint ==")
def cv2(model_fn, Xm, groups, seed=0):
    folds = KFold(5, shuffle=True, random_state=seed).split(Xm) if groups is None else GroupKFold(5).split(Xm, groups=groups)
    pr = np.zeros(ne)
    for tr, te in folds:
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
        pr[te] = model_fn().fit((Xm[tr]-mu)/sd, ey[tr]).predict((Xm[te]-mu)/sd)
    return np.sqrt(np.mean((pr - ey) ** 2))
for split, g in [("random", None), ("scaffold", esc)]:
    r_lin = cv2(lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)), Xd, g)
    r_rf = cv2(lambda: RandomForestRegressor(300, max_features=0.3, n_jobs=-1, random_state=0), Xe, g)
    r_rfd = cv2(lambda: RandomForestRegressor(300, n_jobs=-1, random_state=0), Xd, g)
    print(f"{split:9s} RMSE (logS): linear on 4 descriptors {r_lin:.3f} | RF on 4 descriptors {r_rfd:.3f} | RF on fingerprint {r_rf:.3f}   (sd of logS = {ey.std():.2f})")
