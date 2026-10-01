"""Chapter 26: statistical genetics simulations: confounding and LD score regression, LMM = ridge, fine-mapping,
polygenic-score portability across LD structures, and Mendelian randomization. All data are synthetic; each block isolates one mechanism."""
import numpy as np
from scipy import stats
from scipy.special import ndtri

rng = np.random.default_rng(0)

def sim_geno(N, M, rho, maf, block=50, group=None, maf_group=None):
    """Diploid genotypes (0/1/2) with AR(1) LD inside blocks of `block` SNPs: a latent Gaussian chain thresholded at the allele frequency."""
    X = np.zeros((N, M), np.uint8); rho_b = np.broadcast_to(rho, (-(-M // block),))   # rho may differ by block
    for bi, b in enumerate(range(0, M, block)):
        rho = float(rho_b[bi]); sl = slice(b, min(b + block, M)); k = sl.stop - sl.start
        for _ in range(2):                                                  # two haplotypes
            z = np.empty((N, k), np.float32); z[:, 0] = rng.standard_normal(N)
            for j in range(1, k): z[:, j] = rho * z[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(N)
            thr = ndtri(maf[sl])[None, :] if group is None else ndtri(maf_group[group][:, sl])
            X[:, sl] += (z < thr).astype(np.uint8)
    return X

def standardize(X):
    X = X.astype(np.float32); X -= X.mean(0); sd = X.std(0); sd[sd == 0] = 1; return X / sd

def marginal(X, y):
    """Per-SNP marginal regression of standardized y on standardized genotypes: z-scores."""
    n = len(y); b = X.T @ y / n; se = np.sqrt((1 - b ** 2) / (n - 2)); return b, b / se

# ---------------------------------------------------------------- 1. LD, confounding, LD score regression
print("== 1. Stratification inflates GWAS statistics; LD score regression separates it from polygenicity ==")
M, N, H2, BLK = 6000, 8000, 0.40, 20
RHO = rng.uniform(0.3, 0.98, M // BLK)                                       # blocks differ in LD strength, so LD scores vary across the genome
maf = rng.uniform(0.1, 0.5, M)
Fst = 0.03
delta = rng.normal(0, np.sqrt(maf * (1 - maf) * Fst / (1 - Fst)), M)          # allele-frequency difference between two ancestry groups
maf_g = np.clip(np.stack([maf - delta / 2, maf + delta / 2]), 0.02, 0.98)
causal = rng.random(M) < 0.10; beta = np.where(causal, rng.normal(0, np.sqrt(H2 / causal.sum()), M), 0.0)

def ld_scores(X):
    n = X.shape[0]; R = (X.T @ X) / n; r2 = R ** 2 - (1 - R ** 2) / (n - 2)      # unbiased r^2
    l = np.zeros(X.shape[1])
    for b in range(0, X.shape[1], BLK):                                         # LD is block-diagonal by construction; sum within a 3-block window
        lo, hi = max(0, b - BLK), min(X.shape[1], b + 2 * BLK); l[b:b + BLK] = r2[b:b + BLK, lo:hi].sum(1)
    return l

def gwas_stats(strat, env_shift, pc_adjust=False):
    if strat:
        g = rng.integers(0, 2, N); X = sim_geno(N, M, RHO, maf, block=BLK, group=g, maf_group=maf_g)
    else:
        g = np.zeros(N, int); X = sim_geno(N, M, RHO, maf, block=BLK)
    Xs = standardize(X); gen = Xs @ beta
    e = rng.standard_normal(N) * np.sqrt(1 - H2)
    y = gen + e + env_shift * (g - 0.5)                                        # environment differs between ancestry groups
    y = (y - y.mean()) / y.std()
    if pc_adjust:
        u, s, vt = np.linalg.svd(Xs[:, ::10] - 0, full_matrices=False)           # leading principal component of genotypes
        pc = u[:, 0] * s[0]; pc = (pc - pc.mean()) / pc.std()
        y = y - pc * (pc @ y) / N; Xr = Xs - np.outer(pc, pc @ Xs / N)
        Xr = Xr / Xr.std(0); y = y / y.std(); b, z = marginal(Xr, y)
    else:
        b, z = marginal(Xs, y)
    return z, ld_scores(Xs)

def report(name, z, l):
    chi2 = z ** 2; lam = np.median(chi2) / 0.4549
    slope, intercept = np.polyfit(l, chi2, 1)
    print(f"{name:42s}: mean chi2 {chi2.mean():5.2f}, lambda_GC {lam:4.2f}, LDSC intercept {intercept:5.2f}, h2 from slope {slope * M / N:5.3f}")

z0, l0 = gwas_stats(False, 0.0); report("homogeneous population (polygenic, h2 = 0.40)", z0, l0)
z1, l1 = gwas_stats(True, 0.30); report("two ancestry groups, uncorrected", z1, l1)
z2, l2 = gwas_stats(True, 0.30, pc_adjust=True); report("two ancestry groups, leading PC as covariate", z2, l2)
print(f"(true h2 = {H2}; LDSC intercept near 1 means no confounding; polygenicity inflates lambda_GC through the slope, confounding through the intercept)")

# ---------------------------------------------------------------- 2. LMM = ridge (random effect on all SNPs)
print("\n== 2. A linear mixed model is ridge regression ==")
n, m = 400, 1500
Xs = standardize(sim_geno(n, m, 0.5, rng.uniform(0.1, 0.5, m)))
yv = Xs @ rng.normal(0, np.sqrt(0.4 / m), m) + rng.normal(0, np.sqrt(0.6), n); yv = yv - yv.mean()
sg2, se2 = 0.4, 0.6; lam = se2 * m / sg2 / m * m / m                             # ridge penalty lam = sigma_e^2 / sigma_b^2 with sigma_b^2 = sg2/m
lam = se2 / (sg2 / m)
b_ridge = Xs.T @ np.linalg.solve(Xs @ Xs.T + lam * np.eye(n), yv)                    # ridge solution in the dual form
K = Xs @ Xs.T / m
g_blup = sg2 * K @ np.linalg.solve(sg2 * K + se2 * np.eye(n), yv)                    # BLUP of the genetic value under the LMM y = g + e, g ~ N(0, sg2 K)
print(f"max |X beta_ridge - g_BLUP| = {np.max(np.abs(Xs @ b_ridge - g_blup)):.2e}  (same estimator; n = {n}, SNPs = {m})")

# ---------------------------------------------------------------- 3. fine-mapping a locus
print("\n== 3. Fine-mapping one causal variant among 100 SNPs in LD (single-effect model, Wakefield approximate Bayes factors) ==")
def finemap(rho, N, beta_c, reps=400, Msnp=100, W=0.05 ** 2):
    idx = np.arange(Msnp); R = rho ** np.abs(idx[:, None] - idx[None, :]); Lc = np.linalg.cholesky(R)
    V = 1.0 / N; top_causal = 0; cs_size = []; cover = 0; pip_c = []; rank = []
    for _ in range(reps):
        c = rng.integers(Msnp); z = np.sqrt(N) * beta_c * R[:, c] + Lc @ rng.standard_normal(Msnp)   # summary statistics: z ~ N(sqrt(N) R beta, R)
        lbf = 0.5 * np.log(V / (V + W)) + 0.5 * z ** 2 * W / (V + W)
        pip = np.exp(lbf - lbf.max()); pip /= pip.sum()
        order = np.argsort(-pip); cum = np.cumsum(pip[order]); k = np.searchsorted(cum, 0.95) + 1
        cs = order[:k]; cs_size.append(k); cover += c in cs; pip_c.append(pip[c])
        top_causal += np.argmax(np.abs(z)) == c; rank.append(int(np.argsort(np.argsort(-np.abs(z)))[c]) + 1)
    return top_causal / reps, np.median(cs_size), cover / reps, np.mean(pip_c), np.median(rank)
print("rho(adjacent)  N        effect (z at causal)   lead SNP = causal   median 95% credible set   coverage   mean PIP(causal)   median rank by |z|")
for rho in [0.7, 0.9, 0.97]:
    for N, bc in [(20000, 0.03), (100000, 0.03)]:
        t, cs, cov, pp, rk = finemap(rho, N, bc)
        print(f"{rho:6.2f}        {N:7d}   {np.sqrt(N)*bc:5.1f}                {t:5.2f}                {cs:6.0f}                  {cov:5.2f}      {pp:5.2f}              {rk:4.0f}")

# ---------------------------------------------------------------- 4. polygenic scores and portability
print("\n== 4. Polygenic score portability: same causal effects, different LD ==")
Mp, Ntr, Nte, H2p, Mc = 2000, 20000, 5000, 0.5, 100
mafp = rng.uniform(0.1, 0.5, Mp)
caus = rng.choice(Mp, Mc, replace=False); bt = np.zeros(Mp); bt[caus] = rng.normal(0, np.sqrt(H2p / Mc), Mc)
mafp2 = np.clip(mafp + rng.normal(0, 0.05, Mp), 0.05, 0.95)                          # allele frequencies differ between populations
X1 = sim_geno(Ntr + Nte, Mp, 0.8, mafp); X2 = sim_geno(Nte, Mp, 0.4, mafp2)          # population 2: weaker LD, different frequencies
def pheno(X):
    Xs = standardize(X); g = Xs @ bt; y = g + rng.standard_normal(len(g)) * np.sqrt(1 - H2p); return Xs, (y - y.mean()) / y.std()
Xs1, y1 = pheno(X1); Xs2, y2 = pheno(X2)                                                  # note: effects are on each population's own standardized scale
Xtr, ytr, Xte1, yte1 = Xs1[:Ntr], y1[:Ntr], Xs1[Ntr:], y1[Ntr:]
def fit_methods(cols):
    """Fit C+T and ridge using only the SNPs in `cols` (so, if the causal SNPs are excluded, effects are carried by tags)."""
    Xt = Xtr[:, cols]; bh, zh = marginal(Xt, ytr); Mq = len(cols)
    sel = np.zeros(Mq, bool)
    for b0 in range(0, Mq, 50):
        j = b0 + np.argmax(np.abs(zh[b0:b0 + 50]))
        if abs(zh[j]) > stats.norm.isf(5e-4): sel[j] = True
    w_ct = np.zeros(Mp); w_ct[cols[sel]] = bh[sel]
    w_rd = np.zeros(Mp); w_rd[cols] = np.linalg.solve(Xt.T @ Xt + Mp * (1 - H2p) / H2p * np.eye(Mq), Xt.T @ ytr)
    return w_ct, w_rd, int(sel.sum())
def r2(score, y): return np.corrcoef(score, y)[0, 1] ** 2
all_cols = np.arange(Mp); array_cols = np.setdiff1d(all_cols, caus)
print(f"GWAS sample size {Ntr}; M = {Mp} SNPs ({Mc} causal), h2 = {H2p}; population 1 has adjacent-SNP LD 0.8, population 2 has 0.4 (r^2 0.64 vs 0.16)")
print("predictors available                  method                     R2 in population 1 (test)   R2 in population 2   ratio")
for label, cols in [("causal SNPs genotyped (oracle)", all_cols), ("causal SNPs NOT genotyped (tags only)", array_cols)]:
    w_ct, w_rd, ns = fit_methods(cols)
    for name, w in [(f"clump & threshold ({ns} loci)", w_ct), ("ridge on all available SNPs", w_rd)]:
        a1, a2 = r2(Xte1 @ w, yte1), r2(Xs2 @ w, y2)
        print(f"{label:37s} {name:26s} {a1:8.3f}                   {a2:8.3f}            {a2 / a1:5.2f}")

# ---------------------------------------------------------------- 5. Mendelian randomization
print("\n== 5. Mendelian randomization: confounded OLS vs instrumental variables (true causal effect theta = 0.20) ==")
def mr_sim(gamma_sd, pleio_mean, J=30, N=200000, theta=0.20, seed=1):
    r = np.random.default_rng(seed)
    f = r.uniform(0.1, 0.5, J); G = r.binomial(2, f, (2 * N, J)).astype(float); G = (G - G.mean(0)) / G.std(0)
    gam = r.normal(0, gamma_sd, J); alpha = (pleio_mean * np.sign(gam) + r.normal(0, 0.01, J)) if pleio_mean else np.zeros(J)    # alpha: direct effect on Y aligned with the exposure effect (directional pleiotropy)
    U = r.standard_normal(2 * N)                                                                       # unmeasured confounder
    Xe = G @ gam + 0.8 * U + r.standard_normal(2 * N)
    Y = theta * Xe + G @ alpha + 0.8 * U + r.standard_normal(2 * N)
    ols = np.polyfit(Xe[:N], Y[:N], 1)[0]
    bx = np.array([np.polyfit(G[:N, j], Xe[:N], 1)[0] for j in range(J)]); sx = np.array([stats.linregress(G[:N, j], Xe[:N]).stderr for j in range(J)])
    by = np.array([np.polyfit(G[N:, j], Y[N:], 1)[0] for j in range(J)]); sy = np.array([stats.linregress(G[N:, j], Y[N:]).stderr for j in range(J)])
    w = 1 / sy ** 2; ivw = np.sum(w * bx * by) / np.sum(w * bx ** 2)
    s = np.sign(bx); X_ = np.c_[np.ones(J), np.abs(bx)]; Wd = np.diag(w)
    egg = np.linalg.solve(X_.T @ Wd @ X_, X_.T @ Wd @ (by * s))                                     # MR-Egger: weighted regression with intercept
    F = np.mean((bx / sx) ** 2)
    return ols, ivw, egg[1], egg[0], F
print("scenario                                      OLS     IVW     MR-Egger slope   Egger intercept   mean instrument F")
for name, gs, pm in [("strong instruments, no pleiotropy", 0.10, 0.0), ("weak instruments, no pleiotropy", 0.008, 0.0), ("strong, directional pleiotropy (0.005 per SNP)", 0.10, 0.005)]:
    o, i, e, ei, F = mr_sim(gs, pm)
    print(f"{name:42s}  {o:6.3f}  {i:6.3f}  {e:6.3f}          {ei:8.4f}          {F:6.1f}")
