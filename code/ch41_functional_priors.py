"""Chapter 41: what a functional annotation (for example the score of a sequence-based variant-effect model) is worth in statistical genetics, in simulation.
Annotation score for SNP j:  s_j ~ N(mu * c_j, 1) with c_j = 1 if the SNP is causal, so the annotation separates causal from non-causal SNPs with AUROC = Phi(mu / sqrt 2).
1. Fine-mapping a locus with one causal SNP in strong LD (Wakefield approximate Bayes factors): uniform prior versus an annotation-informed prior, including an over-confident prior.
2. Polygenic prediction: uniform ridge (the LMM), lasso, and a ridge whose per-SNP prior variance is informed by the annotation.
Genotypes are standardized Gaussian vectors with block LD (as in Chapter 26)."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LassoCV
rng = np.random.default_rng(0)
auroc = lambda mu: norm.cdf(mu / np.sqrt(2))
# ---------------------------------------------------------------- 1. fine-mapping
print("== 1. Fine-mapping: 100 SNPs in one LD block (correlation 0.3-0.95), one causal SNP explaining 1% of variance, n = 5,000; 3,000 loci per row ==")
def wakefield_log_abf(z, n, W=0.2 ** 2):
    V = 1.0 / n; return 0.5 * np.log(V / (V + W)) + 0.5 * z ** 2 * W / (V + W)
def finemap_trial(mu, gamma, n=5000, M=100, rho=None):
    rho = rho if rho is not None else rng.uniform(0.3, 0.95)
    f = rng.standard_normal(n)[:, None]; X = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, M)); c = rng.integers(M)
    y = 0.1 * X[:, c] + np.sqrt(1 - 0.01) * rng.standard_normal(n); r = (X * (y - y.mean())[:, None]).mean(0) / (X.std(0) * y.std()); z = np.sqrt(n) * r
    s = rng.standard_normal(M) + mu * (np.arange(M) == c)
    lw = wakefield_log_abf(z, n) + gamma * s; w = np.exp(lw - lw.max()); pip = w / w.sum(); order = np.argsort(-pip); cs = np.searchsorted(np.cumsum(pip[order]), 0.95) + 1
    return order[0] == c, cs, pip[c], (order[:cs] == c).any()
print("annotation AUROC   prior used                          top-PIP SNP is causal   95% credible set: mean size   coverage   mean PIP of the causal SNP")
for mu in (0.0, 1.0, 2.0, 3.0):
    for name, gamma in (("uniform prior", 0.0), ("annotation prior (correct strength)", mu), ("annotation prior (twice too confident)", 2 * mu)):
        if mu == 0 and gamma != 0: continue
        out = np.array([finemap_trial(mu, gamma) for _ in range(3000)], float)
        print(f"{auroc(mu):10.2f}         {name:36s}    {out[:, 0].mean():6.3f}               {out[:, 1].mean():8.1f}               {out[:, 3].mean():6.3f}      {out[:, 2].mean():6.3f}")

# ---------------------------------------------------------------- 2. polygenic prediction
print("\n== 2. Polygenic prediction: 2,000 SNPs in 20 LD blocks, 40 causal SNPs, heritability 0.4; test R^2 in 4,000 held-out individuals; mean of 6 repetitions ==")
M, NB, NC, H2 = 2000, 20, 40, 0.4
def make_geno(n, rhos, seed):
    r = np.random.default_rng(seed); X = np.zeros((n, M))
    for b in range(NB):
        f = r.standard_normal((n, 1)); X[:, b * 100:(b + 1) * 100] = np.sqrt(rhos[b]) * f + np.sqrt(1 - rhos[b]) * r.standard_normal((n, 100))
    return X
def weighted_ridge(X, y, w, lam):
    """Posterior mean of beta ~ N(0, tau2 * w_j): dual form, n x n solve.  lam = sigma2 / tau2."""
    Xw = X * np.sqrt(w)[None, :]; K = Xw @ Xw.T; a = np.linalg.solve(K + lam * np.eye(len(y)), y - y.mean()); return np.sqrt(w) * (Xw.T @ a)
def run(mu, n_tr, seed):
    r = np.random.default_rng(seed); rhos = r.uniform(0.3, 0.95, NB); Xtr, Xte = make_geno(n_tr, rhos, seed * 10 + 1), make_geno(4000, rhos, seed * 10 + 2)
    causal = r.choice(M, NC, replace=False); beta = np.zeros(M); beta[causal] = r.standard_normal(NC); g_tr, g_te = Xtr @ beta, Xte @ beta; sc = np.sqrt(H2 / g_tr.var())
    beta *= sc; ytr = Xtr @ beta + np.sqrt(1 - H2) * r.standard_normal(n_tr); yte = Xte @ beta + np.sqrt(1 - H2) * r.standard_normal(4000)
    s = r.standard_normal(M) + mu * np.isin(np.arange(M), causal)
    res = {}
    lam0 = (1 - H2) / (H2 / M)
    res["ridge, uniform prior (LMM)"] = np.corrcoef(Xte @ weighted_ridge(Xtr, ytr, np.ones(M), lam0), yte)[0, 1] ** 2
    las = LassoCV(cv=3, n_alphas=15, max_iter=3000, n_jobs=1).fit(Xtr, ytr); res["lasso (sparse, no annotation)"] = np.corrcoef(Xte @ las.coef_, yte)[0, 1] ** 2
    for name, gam in (("ridge, annotation-informed (correct strength)", mu), ("ridge, annotation-informed (twice too confident)", 2 * mu)):
        w = np.exp(gam * s - gam ** 2 / 2); w = w / w.mean()                                                     # prior variance proportional to the likelihood ratio, normalised to mean 1
        res[name] = np.corrcoef(Xte @ weighted_ridge(Xtr, ytr, w, lam0), yte)[0, 1] ** 2
    return res
print("annotation AUROC   training n    ridge, uniform (LMM)   lasso     ridge, annotation-informed   ridge, annotation twice too confident   oracle upper bound (h2 = 0.40)")
for mu in (0.0, 1.0, 2.0, 3.0):
    for n_tr in (1000, 3000):
        rs = [run(mu, n_tr, s) for s in range(6)]; m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"{auroc(mu):10.2f}        {n_tr:6d}        {m['ridge, uniform prior (LMM)']:8.3f}           {m['lasso (sparse, no annotation)']:6.3f}         {m['ridge, annotation-informed (correct strength)']:8.3f}                      {m['ridge, annotation-informed (twice too confident)']:8.3f}                          0.400")
