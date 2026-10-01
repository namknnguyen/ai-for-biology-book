"""Chapter 46: experimental design as part of the model.
1. Which perturbations to measure: random vs D-optimal vs uncertainty sampling vs diversity, with clustered (imbalanced) perturbation features.
2. Cells per perturbation versus number of perturbations at a fixed total of cells.
3. Designing edits to identify collinear effects (Chapter 31's composite element).
4. Closed-loop optimization of a rugged landscape: random vs greedy vs bootstrap-Thompson batches."""
import numpy as np
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. choosing perturbations
print("== 1. Learning a response map from k measured perturbations (features d = 12, clustered and imbalanced; 600 candidate perturbations, 500 genes) ==")
d, P, Gn = 12, 600, 500
def make_world(seed):
    r = np.random.default_rng(seed)
    centers = r.normal(0, 2.0, (6, d)); sizes = np.array([0.45, 0.25, 0.15, 0.08, 0.05, 0.02]); lab = r.choice(6, P, p=sizes)
    Z = centers[lab] + 0.6 * r.normal(0, 1, (P, d)); Theta = r.normal(0, 1, (d, Gn)) / np.sqrt(d)
    Y = Z @ Theta + 0.5 * r.normal(0, 1, (P, Gn))                                              # noisy measured response vector of each perturbation
    return Z, Y, lab
def ridge_fit(Z, Y, lam=1.0): return np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ Y)
def select(Z, k, method, r):
    n = len(Z)
    if method == "random": return list(r.choice(n, k, replace=False))
    chosen = [int(r.integers(n))]
    for _ in range(k - 1):
        if method == "D-optimal (greedy)":
            M = Z[chosen].T @ Z[chosen] + 1e-3 * np.eye(d); Mi = np.linalg.inv(M); gain = np.einsum("ij,jk,ik->i", Z, Mi, Z)       # leverage = increase in log det
        elif method == "uncertainty sampling":
            M = Z[chosen].T @ Z[chosen] + 1.0 * np.eye(d); gain = np.einsum("ij,jk,ik->i", Z, np.linalg.inv(M), Z)               # predictive variance (same score; ridge-regularized)
        elif method == "diversity (farthest point)":
            gain = np.min(np.linalg.norm(Z[:, None, :] - Z[chosen][None], axis=2), axis=1)
        gain[chosen] = -np.inf; chosen.append(int(np.argmax(gain)))
    return chosen
methods = ["random", "D-optimal (greedy)", "diversity (farthest point)"]
print("budget k   method                       held-out R2 (all)   held-out R2 on the rarest cluster (2% of perturbations)")
for k in [12, 24, 48, 96]:
    res = {m: [] for m in methods}
    for seed in range(20):
        Z, Y, lab = make_world(seed); r = np.random.default_rng(1000 + seed)
        for m in methods:
            idx = select(Z, k, m, r); Th = ridge_fit(Z[idx], Y[idx]); te = np.setdiff1d(np.arange(P), idx)
            pred = Z[te] @ Th; truthY = Y[te]; mu = truthY.mean()
            r2_all = 1 - np.sum((pred - truthY) ** 2) / np.sum((truthY - mu) ** 2)
            rare = te[lab[te] == 5]
            r2_rare = 1 - np.sum((Z[rare] @ Th - Y[rare]) ** 2) / np.sum((Y[rare] - Y[te].mean()) ** 2) if len(rare) > 2 else np.nan
            res[m].append((r2_all, r2_rare))
    for m in methods:
        a = np.nanmean(np.array(res[m]), 0); print(f"{k:6d}     {m:28s} {a[0]:8.3f}              {a[1]:8.3f}")

# ---------------------------------------------------------------- 2. cells per perturbation vs number of perturbations
print("\n== 2. A fixed budget of 200,000 cells: many perturbations with few cells, or few with many? (d = 12 features, per-gene noise SD 1.0 per cell) ==")
print("perturbations P   cells each n   error in predicting 2,000 held-out perturbations (RMSE of response, mean over genes)")
r = np.random.default_rng(5); d2, G2 = 12, 200
centers = r.normal(0, 2.0, (6, d2)); sizes = np.array([0.45, 0.25, 0.15, 0.08, 0.05, 0.02])
def world(n_p):
    lab = r.choice(6, n_p, p=sizes); return centers[lab] + 0.6 * r.normal(0, 1, (n_p, d2))
Theta = r.normal(0, 1, (d2, G2)) / np.sqrt(d2) * 1.0; Zt = world(2000); Yt = Zt @ Theta
for n_p in [20, 50, 200, 1000, 4000, 10000]:
    n = 200000 // n_p; errs = []
    for rep in range(8):
        Zp = world(n_p); Ym = Zp @ Theta + r.normal(0, 1.0 / np.sqrt(n), (n_p, G2))                       # mean over n cells has noise SD 1/sqrt(n)
        Th = np.linalg.solve(Zp.T @ Zp + 1.0 * np.eye(d2), Zp.T @ Ym); errs.append(np.sqrt(np.mean((Zt @ Th - Yt) ** 2)))
    print(f"{n_p:12d}      {n:8d}      {np.mean(errs):8.3f}")

# ---------------------------------------------------------------- 3. designing edits to identify collinear effects
print("\n== 3. Identifying two motif effects (true +1.0 and -0.8) whose occurrences are perfectly collinear in reference sequences ==")
print("experiment design (n measurements)                         n     mean |error| in effect A    in effect B    (noise SD 0.3)")
def design(kind, n, r):
    if kind == "observational reference loci":
        a = (r.random(n) < 0.5).astype(float); b = a.copy()
    elif kind == "random mutagenesis (each motif disrupted with prob 0.15)":
        a = (r.random(n) < 0.5).astype(float); b = a.copy(); a = a * (r.random(n) > 0.15); b = b * (r.random(n) > 0.15)
    elif kind == "targeted single-motif deletions (half A, half B)":
        a = np.ones(n); b = np.ones(n); half = np.arange(n) % 2 == 0; a[half] = 0; b[~half] = 0
    elif kind == "targeted: A, B, both, neither (balanced factorial)":
        a = (np.arange(n) % 2).astype(float); b = ((np.arange(n) // 2) % 2).astype(float)
    return a, b
for kind in ["observational reference loci", "random mutagenesis (each motif disrupted with prob 0.15)", "targeted single-motif deletions (half A, half B)", "targeted: A, B, both, neither (balanced factorial)"]:
    for n in [20, 100, 400]:
        eA, eB = [], []
        for rep in range(300):
            a, b = design(kind, n, rng); y = 1.0 * a - 0.8 * b + 0.3 * rng.standard_normal(n); X = np.c_[a, b]
            beta = np.linalg.lstsq(X + 1e-9 * rng.standard_normal(X.shape), y, rcond=None)[0] if np.linalg.matrix_rank(X) == 2 else np.linalg.pinv(X) @ y
            eA.append(abs(beta[0] - 1.0)); eB.append(abs(beta[1] + 0.8))
        print(f"{kind:58s} {n:4d}       {np.mean(eA):8.3f}            {np.mean(eB):8.3f}")

# ---------------------------------------------------------------- 4. closed-loop optimization of a rugged landscape
print("\n== 4. Design-build-test-learn: 12-position, 4-letter sequences, pairwise-epistatic landscape; 40 random starts, 8 rounds of batches of 20 ==")
Lp, q = 12, 4
def landscape(seed):
    r = np.random.default_rng(seed); h = r.normal(0, 0.5, (Lp, q)); J = np.zeros((Lp, Lp, q, q))
    for i in range(Lp):
        for j in range(i + 1, Lp):
            if r.random() < 0.35: M = r.normal(0, 0.7, (q, q)); J[i, j] = M
    def f(S):
        e = h[np.arange(Lp)[None, :], S].sum(1)
        for i in range(Lp):
            for j in range(i + 1, Lp): e = e + J[i, j][S[:, i], S[:, j]]
        return 1 / (1 + np.exp(-(e - 1.0)))                                                      # saturating readout (global epistasis)
    return f
_pairs = np.random.default_rng(0).choice(Lp * q * (Lp * q - 1) // 2, 300, replace=False)
_iu = np.triu_indices(Lp * q, 1); _pa, _pb = _iu[0][_pairs], _iu[1][_pairs]
def feats(S):
    X = np.zeros((len(S), Lp * q)); X[np.arange(len(S))[:, None], np.arange(Lp)[None, :] * q + S] = 1
    return np.c_[X, X[:, _pa] * X[:, _pb]]
def run(strategy, seed, rounds=8, batch=20, n0=40):
    f = landscape(seed); r = np.random.default_rng(seed + 99); pool = r.integers(0, q, (100000, Lp)); fp = f(pool)
    idx = list(r.choice(len(pool), n0, replace=False)); Xp = feats(pool); best = [fp[idx].max()]
    for _ in range(rounds):
        cand = np.setdiff1d(np.arange(len(pool)), idx)
        if strategy == "random": new = r.choice(cand, batch, replace=False)
        else:
            y = fp[idx]; Xi = Xp[idx]
            if strategy == "greedy (ridge)":
                w = np.linalg.solve(Xi.T @ Xi + 1.0 * np.eye(Xi.shape[1]), Xi.T @ y); new = cand[np.argsort(-(Xp[cand] @ w))[:batch]]
            else:                                                                                  # bootstrap Thompson: each batch member from a bootstrap-refit model
                new = []
                for _b in range(batch):
                    bi = r.integers(0, len(idx), len(idx)); Xb, yb = Xi[bi], y[bi]
                    w = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ yb); sc = Xp[cand] @ w; sc[np.isin(cand, new)] = -np.inf; new.append(cand[int(np.argmax(sc))])
                new = np.array(new)
        idx += list(new); best.append(fp[idx].max())
    return np.array(best), fp.max(), np.quantile(fp, 0.999)
print("strategy                 best fitness found after rounds 0, 2, 4, 8 (as a fraction of the best in a 100,000 pool)      P(reach top 0.1% of the pool)  [200 random evaluations would succeed with probability 0.18]")
for strat in ["random", "greedy (ridge)", "bootstrap Thompson"]:
    curves, top = [], 0; n_runs = 20
    for seed in range(n_runs):
        b, mx, q99 = run(strat, seed); curves.append(b / mx); top += b[-1] >= q99
    c = np.mean(curves, 0); print(f"{strat:24s} {c[0]:.3f}  {c[2]:.3f}  {c[4]:.3f}  {c[8]:.3f}                                                  {top / n_runs:.2f}")
