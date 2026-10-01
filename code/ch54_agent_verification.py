"""Chapter 54: two ways an automated research loop manufactures confidence, in simulation.
1. Forking paths: an agent that tries many analysis variants on one null dataset and reports the best.
2. Judge drift: a hypothesis-evolution loop whose selection signal (a judge's score) is only partly aligned with truth.
Everything is synthetic; the point is the arithmetic of selection, not a claim about any particular system."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. forking paths
n, P, reps = 60, 20, 1500
print("== 1. One dataset, many analysis variants; the agent reports the variant with the smallest p-value ==")
print(f"n = {n} samples, {P} candidate predictors (pairwise correlation 0.3), 1 binary covariate; variants = predictor x subgroup (5) x transform (2) = {P*5*2}")
def variants_p(X, y, g):
    """p-values of the Pearson correlation of each predictor with y, for each subgroup and each transform. Returns array (P, 5, 2)."""
    n = len(y); half = np.arange(n) < n // 2
    groups = [np.ones(n, bool), g == 0, g == 1, half, ~half]; out = np.ones((X.shape[1], 5, 2))
    for s, m in enumerate(groups):
        for t in range(2):
            Xm, ym = X[m], y[m]
            if t == 1: Xm = stats.rankdata(Xm, axis=0); ym = stats.rankdata(ym)
            Xc = Xm - Xm.mean(0); yc = ym - ym.mean(); r = (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()); k = m.sum() - 2
            tt = r * np.sqrt(k / (1 - r ** 2)); out[:, s, t] = 2 * stats.t.sf(np.abs(tt), k)
    return out.reshape(-1)
C = 0.3 * np.ones((P, P)) + 0.7 * np.eye(P); Lc = np.linalg.cholesky(C)
nv = P * 10; order_cache = [rng.permutation(nv) for _ in range(reps)]
Ms = [1, 5, 20, 50, 100, nv]; fp = {M: 0 for M in Ms}; fp_bonf = {M: 0 for M in Ms}; fp_perm = 0; minp_all = []; true_hits = {M: 0 for M in Ms}; true_called = {M: 0 for M in Ms}
# datasets: half null; half with a real effect of predictor 0 on y (beta = 0.45). A 'hit' counts as real if the dataset contains the effect, even when the reported variant is a correlated predictor.
raw = []; pre = []
for rep in range(reps):
    X = rng.standard_normal((n, P)) @ Lc.T; g = rng.integers(0, 2, n); y = rng.standard_normal(n)
    kind = "null" if rep % 2 == 0 else "real"
    if kind == "real": y = y + 0.45 * X[:, 0]
    p0 = variants_p(X, y, g); pre.append((kind, p0[0])); p = p0[order_cache[rep]]; raw.append((kind, p, order_cache[rep]))
    minp_all.append(p.min())
print("variants tried M   P(report p < 0.05) on NULL data   with Bonferroni over M   fraction of reported hits that are real (data are 50% null / 50% real effect)")
for M in Ms:
    null_hit = np.mean([p[:M].min() < 0.05 for kind, p, _ in raw if kind == "null"])
    null_bonf = np.mean([p[:M].min() * M < 0.05 for kind, p, _ in raw if kind == "null"])
    real_hit = np.mean([p[:M].min() < 0.05 for kind, p, _ in raw if kind == "real"])
    prec = real_hit / (real_hit + null_hit)
    print(f"{M:12d}       {null_hit:12.3f}                      {null_bonf:10.3f}                 {prec:8.3f}")
# permutation-calibrated minimum p over the full set of logged variants (the correct correction for 'the best of everything tried')
Bperm = 200; Xp = rng.standard_normal((n, P)) @ Lc.T; gp = rng.integers(0, 2, n); yp = rng.standard_normal(n)
nullmins = np.array([variants_p(Xp, rng.permutation(yp), gp).min() for _ in range(Bperm)])
thr = np.quantile(nullmins, 0.05)
cal_fp = np.mean([p.min() < thr for kind, p, _ in raw if kind == "null"]); cal_pow = np.mean([p.min() < thr for kind, p, _ in raw if kind == "real"])
print(f"\nmin-p calibrated against its own null distribution over all {nv} logged variants (threshold {thr:.2g}, not 0.05): false-positive rate {cal_fp:.3f}, power for the real effect {cal_pow:.3f}")
print(f"for comparison, ONE pre-specified test (predictor 0, all samples, raw): false-positive rate {np.mean([p < 0.05 for k, p in pre if k == 'null']):.3f}, power {np.mean([p < 0.05 for k, p in pre if k == 'real']):.3f}")

# ---------------------------------------------------------------- 2. judge drift
print("\n== 2. A hypothesis-evolution loop selected by a judge: the judge's score rises faster than the truth ==")
print("hypothesis i has true value v_i ~ N(0,1) (probability-it-is-true-and-useful on a logit-like scale) and a 'persuasiveness' f_i ~ N(0,1) unrelated to truth;")
print("the judge sees J = v + w*f + noise (noise SD 1.0 per comparison, 20 comparisons per hypothesis); each round, every survivor is mutated 5 ways:")
print("a mutation changes v by N(0, 0.15^2) (truth is hard to move) and f by N(0, 0.6^2) (persuasiveness is easy to move); the top 20% by tournament rank survive and are re-expanded.")
def evolve(w, rounds=8, N=100, seed=0):
    r = np.random.default_rng(seed); v = r.standard_normal(N); f = r.standard_normal(N); hist = []
    for rd in range(rounds + 1):
        J = v + w * f
        hist.append((J.mean() - v.mean() * 0, v.mean(), f.mean(), J[np.argsort(-(J + r.standard_normal(N) * 0.5))[:5]].mean(), v[np.argsort(-(J + r.standard_normal(N) * 0.5))[:5]].mean()))
        if rd == rounds: break
        # noisy tournament: each hypothesis plays 20 random opponents; score = wins
        wins = np.zeros(N)
        for _ in range(20):
            opp = r.permutation(N); d = (J - J[opp]) + r.standard_normal(N) * 1.0; wins += (d > 0); wins[opp] += (d <= 0)
        keep = np.argsort(-wins)[: N // 5]; v0, f0 = v[keep], f[keep]
        v = np.repeat(v0, 5) + r.normal(0, 0.15, 5 * len(keep)); f = np.repeat(f0, 5) + r.normal(0, 0.6, 5 * len(keep))
    return np.array(hist)
print("judge weight on persuasiveness w   round   mean judge score of the pool   mean TRUE value of the pool   mean persuasiveness")
for w in [0.0, 1.0]:
    H = np.mean([evolve(w, seed=s) for s in range(30)], 0)
    for rd in [0, 2, 4, 8]: print(f"{w:20.1f}                  {rd:5d}   {H[rd,0]:20.2f}   {H[rd,1]:24.2f}   {H[rd,2]:16.2f}")
print("\nselecting 3 hypotheses for experiments out of N candidates with the judge (w = 1.0, no evolution): mean TRUE value of the selected three, 400 repetitions")
print("N candidates   judge-selected   random   oracle (best 3 by true value)")
for N in [10, 30, 100, 300, 1000]:
    a = []
    for rep in range(400):
        v = rng.standard_normal(N); f = rng.standard_normal(N); J = v + 1.0 * f + rng.standard_normal(N) * 1.0; a.append((v[np.argsort(-J)[:3]].mean(), v.mean(), np.sort(v)[-3:].mean()))
    a = np.mean(a, 0); print(f"{N:10d}       {a[0]:8.2f}       {a[1]:6.2f}   {a[2]:6.2f}")
