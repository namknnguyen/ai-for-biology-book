"""Chapter 44: what observational data can and cannot say about causal regulation, in a linear structural causal model with a latent confounder.
40 genes, a sparse directed acyclic network B (x = Bx + lambda*u + e), a hidden cell-state confounder u, observational cells and knock-down experiments.
Task: rank ordered gene pairs (i, j) by evidence that perturbing i changes j (|total effect| > 0.15)."""
import numpy as np
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
G = 40

def make_network(conf):
    B = np.zeros((G, G))
    for j in range(1, G):
        parents = rng.choice(j, size=min(j, rng.integers(0, 3)), replace=False)
        for i in parents: B[j, i] = rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)             # B[j, i]: effect of gene i on gene j (i < j: topological order)
    lam = rng.normal(0, conf, G)                                                              # loadings of the hidden confounder (cell state)
    return B, lam
def simulate_obs(B, lam, n, noise=0.5):
    u = rng.standard_normal((n, 1)); e = noise * rng.standard_normal((n, G))
    return (u @ lam[None, :] + e) @ np.linalg.inv(np.eye(G) - B).T
def simulate_do(B, lam, i, n, c=-2.0, noise=0.5):
    Bi = B.copy(); Bi[i] = 0                                                                  # cut the equation of gene i
    u = rng.standard_normal((n, 1)); e = noise * rng.standard_normal((n, G)); e[:, i] = c + 0.2 * rng.standard_normal(n)
    return (u @ lam[None, :] + e) @ np.linalg.inv(np.eye(G) - Bi).T
def total_effects(B):
    return np.linalg.inv(np.eye(G) - B)                                                       # T[j, i] = effect on x_j of setting x_i by one unit

def run(conf, n_obs, n_int, frac_perturbed, seed):
    global rng; rng = np.random.default_rng(seed)
    B, lam = make_network(conf); T = total_effects(B)
    X = simulate_obs(B, lam, n_obs)
    pert = rng.random(G) < frac_perturbed
    shift = np.zeros((G, G)); mask = np.zeros((G, G), bool)                                  # shift[i, j]: measured mean shift of gene j when gene i is knocked down
    for i in np.flatnonzero(pert):
        Y = simulate_do(B, lam, i, n_int); shift[i] = (Y.mean(0) - X.mean(0)) / (-2.0); mask[i] = True
    C = np.cov(X.T); sd = np.sqrt(np.diag(C)); R = C / np.outer(sd, sd)
    P = np.linalg.inv(C + 1e-3 * np.eye(G)); PC = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
    slope = C / np.diag(C)[:, None]                                                           # slope[i, j]: regression of x_j on x_i
    pos = np.abs(T.T - np.eye(G)) > 0.15                                                      # pos[i, j]: truly perturbing i moves j (excluding i itself)
    off = ~np.eye(G, dtype=bool)
    out = {}
    scores = {"correlation |r|": np.abs(R), "regression slope of j on i": np.abs(slope), "partial correlation": np.abs(PC)}
    for k, s in scores.items(): out[k] = roc_auc_score(pos[off], s[off])
    pm = mask & off
    out["slope, restricted to perturbed genes"] = roc_auc_score(pos[pm], np.abs(slope)[pm]) if pm.any() else np.nan
    out["interventional shift (perturbed genes)"] = roc_auc_score(pos[pm], np.abs(shift)[pm]) if pm.any() else np.nan
    # direction: among truly causal pairs with i->j detected, how often does the observational slope rank i->j above j->i?
    both = pos & ~pos.T & off
    out["orientation: P(|slope i->j| > |slope j->i|), true pairs"] = np.mean(np.abs(slope)[both] > np.abs(slope).T[both]) if both.any() else np.nan
    return out
def avg(conf, n_obs, n_int, frac, seeds=range(8)):
    rs = [run(conf, n_obs, n_int, frac, s) for s in seeds]; return {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}

print("== AUROC for finding pairs (i, j) such that knocking down gene i changes gene j (|total effect| > 0.15); 40 genes; mean of 8 random networks ==")
print("scenario                                                     correlation   slope j~i   partial corr   slope (perturbed i only)   interventional (perturbed i only)   orientation correct")
for name, conf, n_obs, n_int, frac in [("no hidden confounder, 5,000 cells", 0.0, 5000, 100, 0.5), ("hidden confounder (loading SD 0.7), 5,000 cells", 0.7, 5000, 100, 0.5),
                                       ("strong confounder (SD 1.5), 5,000 cells", 1.5, 5000, 100, 0.5)]:
    r = avg(conf, n_obs, n_int, frac)
    print(f"{name:60s} {r['correlation |r|']:8.3f}     {r['regression slope of j on i']:8.3f}    {r['partial correlation']:8.3f}         {r['slope, restricted to perturbed genes']:8.3f}                   {r['interventional shift (perturbed genes)']:8.3f}                    {r['orientation: P(|slope i->j| > |slope j->i|), true pairs']:8.3f}")
print("\nInterventional AUROC versus cells per knock-down (strong confounder, half of the genes perturbed):")
print("cells per perturbation   interventional AUROC   observational slope AUROC on the same pairs")
for n_int in [10, 30, 100, 300]:
    r = avg(1.5, 5000, n_int, 0.5); print(f"{n_int:12d}              {r['interventional shift (perturbed genes)']:8.3f}                {r['slope, restricted to perturbed genes']:8.3f}")
print("\nObservational sample size does not repair confounding (strong confounder):")
print("observational cells   correlation   slope j~i   partial corr")
for n_obs in [500, 5000, 50000]:
    r = avg(1.5, n_obs, 100, 0.5, seeds=range(4)); print(f"{n_obs:12d}          {r['correlation |r|']:8.3f}     {r['regression slope of j on i']:8.3f}    {r['partial correlation']:8.3f}")
