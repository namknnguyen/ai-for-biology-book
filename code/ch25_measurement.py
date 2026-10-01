"""Chapter 25: measurement models for single-cell, spatial, and perturbation data (synthetic, mechanistic).
Every simulation isolates one property of the measurement process; none claims to reproduce a specific real dataset."""
import numpy as np
from scipy import stats
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)

def nb_counts(mean, phi, size=None):
    """Gamma-Poisson (Chapter 19): variance = mean + phi*mean^2."""
    mean = np.asarray(mean, float)
    if phi <= 0: return rng.poisson(mean, size)
    return rng.poisson(rng.gamma(1 / phi, mean * phi, size=size if size is not None else mean.shape))

# ---------------------------------------------------------------- 1. capture: thinning an NB gives an NB; zeros are sampling, not "dropout"
G, N, P_CAP, PHI = 5000, 2000, 0.10, 0.2
mu = rng.lognormal(0, 1.8, G); mu *= 13.0 / mu.mean()                   # true mean mRNA molecules per cell per gene (~200k molecules / 15k genes)
m = nb_counts(mu[None, :], PHI, (N, G))                                  # true molecules, bursty
y = rng.binomial(m, P_CAP)                                               # capture: binomial thinning with probability 0.1
ybar = y.mean(0); var = y.var(0)
print("== 1. Binomial thinning of a Gamma-Poisson gives a Gamma-Poisson with the SAME dispersion ==")
print(f"empirical variance/(mean + phi*mean^2) over genes with mean>1: {np.median(var[ybar>1]/(ybar[ybar>1]+PHI*ybar[ybar>1]**2)):.3f}  (theory: 1)")
print("mean UMI/cell      genes   observed zero frac   NB prediction   Poisson prediction   per-cell reliability (true signal var / total var)")
edges = [0.02, 0.2, 0.5, 1, 3, 10, 1e9]
true_signal = (P_CAP * mu)                                              # E[y | gene] is constant across cells here; bursty heterogeneity is the signal
sig_var = (P_CAP ** 2) * PHI * mu ** 2                                    # variance of the cell-to-cell *true* expectation p*m
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (P_CAP * mu >= lo) & (P_CAP * mu < hi)
    if s.sum() == 0: continue
    pm = P_CAP * mu[s]
    z_obs = (y[:, s] == 0).mean(); z_nb = ((1 + PHI * pm) ** (-1 / PHI)).mean(); z_po = np.exp(-pm).mean()
    rel = (sig_var[s] / (pm + PHI * pm ** 2)).mean()
    print(f"[{lo:5.2f},{min(hi,99):5.1f})   {s.sum():5d}        {z_obs:.3f}              {z_nb:.3f}           {z_po:.3f}                {rel:.3f}")

# ---------------------------------------------------------------- 2. compositionality of library-size normalization
print("\n== 2. A few genes go up; library-size normalization makes everything else go down ==")
G2 = 5000
mu2 = rng.lognormal(0, 1.8, G2); mu2 *= 13.0 / mu2.mean()
order = np.argsort(mu2)[::-1]; k_top = int(np.searchsorted(np.cumsum(mu2[order]) / mu2.sum(), 0.30)) + 1
top = order[:k_top]; share = mu2[top].sum() / mu2.sum()
muB = mu2.copy(); muB[top] *= 3.0
yA = rng.binomial(nb_counts(mu2[None], 0.2, (300, G2)), 0.1); yB = rng.binomial(nb_counts(muB[None], 0.2, (300, G2)), 0.1)
a, b = yA.sum(0) + 0.5, yB.sum(0) + 0.5                                   # pseudobulk
rest = np.ones(G2, bool); rest[top] = False; keep = rest & (yA.sum(0) + yB.sum(0) > 200)
cpm = np.log2((b / b.sum()) / (a / a.sum()))
mor = np.log2(b / a); mor = mor - np.median(mor[keep])                    # median-ratio normalization (assumes most genes unchanged)
print(f"the {k_top} most abundant genes carry {100*share:.0f}% of molecules and are tripled in condition B; the other {keep.sum()} well-detected genes are unchanged")
print(f"apparent log2 fold-change of the unchanged genes: CPM normalization median {np.median(cpm[keep]):+.2f} (true 0.00); median-ratio normalization {np.median(mor[keep]):+.2f}")
print(f"fraction of unchanged genes with apparent log2FC < -0.5: CPM {np.mean(cpm[keep] < -0.5):.2f}; median-ratio {np.mean(mor[keep] < -0.5):.2f}")

# ---------------------------------------------------------------- 3. sampling cells: how many to see a rare type
print("\n== 3. Cells needed to observe at least 20 cells of a type with frequency f, with probability 0.95 ==")
for f in [0.05, 0.01, 0.001, 0.0001]:
    n = 100
    while stats.binom.sf(19, n, f) < 0.95: n = int(n * 1.03) + 1
    print(f"  f = {f:<7g}: about {n:>9,d} cells")

# ---------------------------------------------------------------- 4. doublets create fake intermediate states
print("\n== 4. Doublets masquerade as an intermediate cell state ==")
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
Gd, n_real = 1000, 4000
base = rng.lognormal(0, 1.2, Gd); base *= 1.5 / base.mean()
isA = np.zeros(Gd); isA[:100] = 1; isB = np.zeros(Gd); isB[100:200] = 1
def cells_at(t):                                                           # a continuum A -> B: marker set A falls and set B rises with t
    mean = base[None, :] * 1.8 ** ((1 - t)[:, None] * isA[None, :] + t[:, None] * isB[None, :])
    mean = mean * rng.lognormal(0, 0.4, len(t))[:, None]                    # cell-to-cell variation in size and capture efficiency
    return rng.poisson(rng.gamma(3, mean / 3))
t_real = rng.beta(0.3, 0.3, n_real)                                        # U-shaped: most cells sit near the two ends, few are truly intermediate
X_real = cells_at(t_real)
nd = int(0.06 * n_real)                                                    # 6% doublets, formed from an A-end and a B-end cell: counts add
X_dbl = cells_at(rng.uniform(0, 0.15, nd)) + cells_at(rng.uniform(0.85, 1, nd))
X = np.vstack([X_real, X_dbl]); is_dbl = np.r_[np.zeros(n_real, bool), np.ones(nd, bool)]; t_all = np.r_[t_real, np.full(nd, np.nan)]
Z = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
pca = PCA(10, random_state=0).fit(Z - Z.mean(0)); P = pca.transform(Z - Z.mean(0))
ends = P[np.r_[t_real, np.zeros(nd)] < 0.1][:2000].mean(0), P[np.r_[t_real, np.ones(nd)] > 0.9][:2000].mean(0)
axis = ends[1] - ends[0]; pos = ((P - ends[0]) @ axis) / (axis @ axis)          # position along the A -> B axis; 0 = A end, 1 = B end
mid = (pos > 0.35) & (pos < 0.65)
print(f"{len(X)} droplets, {nd} ({100*nd/len(X):.0f}%) are A+B doublets; true intermediate cells (0.35<t<0.65): {np.sum((t_real>0.35)&(t_real<0.65))}")
print(f"cells that look intermediate in the expression embedding (position 0.35-0.65 on the A-B axis): {mid.sum()}; of these {np.mean(is_dbl[mid]):.0%} are doublets; {np.mean(mid[is_dbl]):.0%} of all doublets land there")
lt = np.log(X.sum(1))
print(f"total UMI per droplet, doublets / singlets: {np.exp(lt[is_dbl].mean() - lt[~is_dbl].mean()):.2f}x;  AUROC for doublet detection from total UMI alone = {roc_auc_score(is_dbl, lt):.3f}")
ia, ib = rng.integers(0, len(X), len(X)), rng.integers(0, len(X), len(X))   # Scrublet-style: add random pairs of observed droplets to make synthetic doublets
Xs = X[ia] + X[ib]; Zs = np.log1p(Xs / Xs.sum(1, keepdims=True) * 1e4)
Ps = pca.transform(Zs - Z.mean(0))
nbr = NearestNeighbors(n_neighbors=31).fit(np.vstack([P, Ps])).kneighbors(P)[1][:, 1:]
score = (nbr >= len(P)).mean(1)                                            # fraction of a droplet's neighbors that are synthetic doublets
print(f"synthetic-doublet neighbor score: AUROC = {roc_auc_score(is_dbl, score):.3f} over all droplets; within the 'intermediate' region only, AUROC = {roc_auc_score(is_dbl[mid], score[mid]):.3f}")
lam = 0.05
print(f"Poisson loading with mean {lam} cells/droplet: doublet fraction among occupied droplets = {(1-np.exp(-lam)-lam*np.exp(-lam))/(1-np.exp(-lam)):.3f} (~lambda/2 = {lam/2:.3f})")

# ---------------------------------------------------------------- 5. donors, not cells, are the replicates
print("\n== 5. Pseudoreplication in single-cell differential expression: 3 vs 3 donors, 400 cells each ==")
Gx, D, C = 1000, 6, 400
base = rng.lognormal(0, 1.3, Gx); base *= 1.5 / base.mean()
de = np.zeros(Gx, bool); de[:50] = True
fc = np.where(de, 2 ** 0.6, 1.0)                                          # 50 truly DE genes, log2FC = 0.6, case donors only
meta = np.repeat(np.arange(D), C); grp = (meta >= 3)
don = np.exp(rng.normal(0, 0.35, (D, Gx)))                                 # donor-level random effect on every gene (genetic background, batch)
size = rng.lognormal(0, 0.25, D * C)
lam_c = base[None, :] * don[meta] * np.where(grp[:, None], fc[None, :], 1.0) * size[:, None]
Y = rng.poisson(rng.gamma(5, lam_c / 5))
L = np.log1p(Y / Y.sum(1, keepdims=True) * Y.sum(1).mean())
p_cell = stats.ttest_ind(L[grp], L[~grp], equal_var=False).pvalue
pb = np.stack([L[meta == d].mean(0) for d in range(D)])
p_pb = stats.ttest_ind(pb[3:], pb[:3], equal_var=False).pvalue
for name, p in [("cells as replicates (Welch t on 1,200 vs 1,200 cells)", p_cell), ("donors as replicates (pseudobulk, 3 vs 3)", p_pb)]:
    print(f"  {name:55s}: false-positive rate at p<0.05 on null genes = {np.mean(p[~de] < .05):.3f}; power on DE genes = {np.mean(p[de] < .05):.3f}")
print(f"  (null genes called at p<0.001 with cells as replicates: {np.mean(p_cell[~de] < 1e-3):.2f}; with donors: {np.mean(p_pb[~de] < 1e-3):.3f})")

# ---------------------------------------------------------------- 6. spatial spots are mixtures: deconvolution and its failure modes
print("\n== 6. Spatial spots as mixtures: NNLS deconvolution, and what breaks it ==")
Gs, K = 300, 5
S = rng.gamma(0.4, 1.0, (Gs, K)) + 0.02
for k in range(K): S[k * 40:(k + 1) * 40, k] += rng.gamma(3, 2.0, 40)      # 40 marker genes per cell type
S /= S.sum(0)                                                            # each column: fraction of a cell type's RNA by gene
rna = np.array([1.0, 1.0, 2.5, 0.6, 1.3])                                 # RNA content per cell by type
def simulate_spots(n_spots, depth, mismatch, hidden):
    est_rna, tru_rna, est_cell, tru_cell = [], [], [], []
    bias = np.exp(rng.normal(0, 0.5, Gs)) if mismatch else np.ones(Gs)    # platform-specific gene-wise efficiency (reference built on another technology)
    for _ in range(n_spots):
        cells = rng.multinomial(rng.integers(3, 15), rng.dirichlet(np.ones(K) * 0.6))
        w = cells * rna; mix = (S * w).sum(1) * bias; mix /= mix.sum()
        ysp = rng.poisson(depth * mix)
        use = K - 1 if hidden else K                                       # 'hidden': the reference lacks cell type K-1
        coef, _ = nnls(S[:, :use], ysp / max(ysp.sum(), 1)); coef = coef / max(coef.sum(), 1e-12)
        full = np.zeros(K); full[:use] = coef
        t_rna = w / w.sum(); t_cell = cells / cells.sum()
        c_hat = full / rna; c_hat = c_hat / max(c_hat.sum(), 1e-12)
        est_rna.append(full); tru_rna.append(t_rna); est_cell.append(c_hat); tru_cell.append(t_cell)
    e = lambda a, b_: np.mean(np.abs(np.array(a) - np.array(b_)))
    return e(est_rna, tru_rna), e(est_cell, tru_cell), e(est_rna, tru_cell)
print("                                       MAE RNA-fraction   MAE cell-fraction   (RNA-fraction estimate read as cell fraction)")
for depth in [200, 1000, 5000]:
    for name, mis, hid in [("matched reference", False, False), ("reference from another platform", True, False), ("reference missing a cell type", False, True)]:
        a_, b_, c_ = simulate_spots(300, depth, mis, hid)
        print(f"  UMI/spot = {depth:5d}, {name:33s}: {a_:.3f}              {b_:.3f}              {c_:.3f}")

# ---------------------------------------------------------------- 7. Perturb-seq: power and reliability
print("\n== 7. Perturb-seq: how many cells per perturbation? (one gene, baseline mean 2 UMI, phi = 0.3, Welch t-test on log1p, alpha = 1e-3) ==")
def power(n, lfc, q=1.0, reps=1500, alpha=1e-3, base=2.0, phi=0.3):
    ctrl = np.log1p(nb_counts(np.full((reps, n), base), phi))
    eff = np.where(rng.random((reps, n)) < q, 2 ** lfc, 1.0)               # fraction q of cells carry an effective perturbation
    pert = np.log1p(nb_counts(base * eff, phi))
    return np.mean(stats.ttest_ind(pert, ctrl, axis=1, equal_var=False).pvalue < alpha)
ns = [25, 50, 100, 200, 500, 1000]
print("cells per perturbation:      " + "  ".join(f"{n:>5d}" for n in ns))
for lfc in [0.25, 0.5, 1.0]:
    print(f"log2FC {lfc:4.2f}, all cells effective: " + "  ".join(f"{power(n, lfc):5.2f}" for n in ns))
print(f"log2FC 0.50, 60% of cells effective: " + "  ".join(f"{power(n, 0.5, 0.6):5.2f}" for n in ns))

print("\nsplit-half reliability of the observed mean-shift vector (2,000 genes; log2 shift estimates from independent halves)")
Gp = 2000
b0 = rng.lognormal(0, 1.8, Gp); b0 *= 1.3 / b0.mean()
def half(mean_vec, n): return nb_counts(mean_vec[None, :], 0.3, (n, Gp)).mean(0)
print("true DE genes   cells/half   r(all genes)   r(true DE genes only)   reliability 2r/(1+r) of the full-sample estimate -> ceiling r_max = sqrt(reliability)   [means of 8 repeats]")
for s_de in [0, 5, 50, 500]:
    for n in [100, 1000]:
        r_all, r_de = [], []
        for _ in range(8):
            shift = np.zeros(Gp); idx = rng.choice(Gp, s_de, replace=False); shift[idx] = rng.normal(0, 0.5, s_de)
            meanP = b0 * 2 ** shift
            est = [np.log2((half(meanP, n) + 0.05) / (half(b0, 4 * n) + 0.05)) for _ in range(2)]
            r_all.append(np.corrcoef(est[0], est[1])[0, 1])
            if s_de > 2: r_de.append(np.corrcoef(est[0][idx], est[1][idx])[0, 1])
        r = max(np.mean(r_all), 0.0); rel = 2 * r / (1 + r)
        rde = f"{np.mean(r_de):6.3f}" if r_de else "   n/a"
        print(f"{s_de:8d}        {n:6d}        {np.mean(r_all):6.3f}         {rde}                 {rel:6.3f} -> {np.sqrt(rel):.3f}")
