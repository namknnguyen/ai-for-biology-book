"""Chapter M8: estimators and bias, MLE, confidence intervals, p-values, multiple testing, bootstrap, regression, Simpson, power."""
import math

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Bias of an estimator: dividing the sum of squares by n underestimates the variance; n - 1 is unbiased.
# ---------------------------------------------------------------------------------------------
n, reps, true_var = 5, 200000, 4.0
data = rng.normal(0, math.sqrt(true_var), size=(reps, n))
print(f"variance estimators, n = {n}, true variance {true_var}: mean of MLE (divide by n) = {data.var(1, ddof=0).mean():.3f}"
      f" (theory {(n - 1) / n * true_var:.2f});  mean of unbiased (divide by n-1) = {data.var(1, ddof=1).mean():.3f}")

# ---------------------------------------------------------------------------------------------
# 2. Confidence intervals: coverage of the nominal 95% interval for a mean, z-based vs t-based.
# ---------------------------------------------------------------------------------------------
print("\ncoverage of 95% confidence intervals for a mean (true mean 10, sd 3), 20000 repeated experiments:")
for n in (5, 10, 30, 100):
    x = rng.normal(10, 3, size=(20000, n)); m = x.mean(1); se = x.std(1, ddof=1) / math.sqrt(n)
    z_cov = np.mean(np.abs(m - 10) < 1.96 * se)
    t_cov = np.mean(np.abs(m - 10) < stats.t.ppf(0.975, n - 1) * se)
    print(f"  n = {n:>3}: z-interval covers {z_cov:.3f},  t-interval covers {t_cov:.3f}")

# ---------------------------------------------------------------------------------------------
# 3. p-values are uniform under the null. A thousand genes with no effect at all.
# ---------------------------------------------------------------------------------------------
G = 10000
a = rng.normal(size=(G, 10)); b = rng.normal(size=(G, 10))
pvals = stats.ttest_ind(a, b, axis=1).pvalue
hist = np.histogram(pvals, bins=10, range=(0, 1))[0]
print(f"\n{G} genes, no true differences, 10 vs 10 samples: p-value histogram (10 bins) = {hist.tolist()}  (flat = uniform); p < 0.05 for {np.sum(pvals < 0.05)} genes (expected {int(0.05 * G)})")

# ---------------------------------------------------------------------------------------------
# 4. Multiple testing: 10,000 genes, 500 truly differential (effect 1.5 sd, n = 10 per group).
# ---------------------------------------------------------------------------------------------
true_de = np.zeros(G, dtype=bool); true_de[:500] = True
a = rng.normal(size=(G, 10)); b = rng.normal(size=(G, 10)); b[true_de] += 1.5
p = stats.ttest_ind(a, b, axis=1).pvalue

def bh(p, q):
    order = np.argsort(p); ranked = p[order]; m = len(p)
    thresh = q * np.arange(1, m + 1) / m
    ok = np.nonzero(ranked <= thresh)[0]
    k = ok.max() + 1 if len(ok) else 0
    sel = np.zeros(m, dtype=bool); sel[order[:k]] = True
    return sel

def report(name, sel):
    tp = np.sum(sel & true_de); fp = np.sum(sel & ~true_de)
    print(f"  {name:<30} discoveries {sel.sum():>5}   true {tp:>4}   false {fp:>4}   false discovery proportion {fp / max(sel.sum(), 1):.3f}   power {tp / 500:.3f}")
print(f"\nmultiple testing, {G} genes, 500 truly differential:")
report("p < 0.05 (uncorrected)", p < 0.05)
report("Bonferroni (p < 0.05/10000)", p < 0.05 / G)
report("Benjamini-Hochberg, FDR 5%", bh(p, 0.05))

# ---------------------------------------------------------------------------------------------
# 5. Bootstrap and permutation test for a small sample.
# ---------------------------------------------------------------------------------------------
x = np.array([2.1, 2.5, 3.0, 3.2, 3.9, 4.1, 4.8, 5.5, 9.7, 12.4])     # skewed measurements
boots = np.array([np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(20000)])
print(f"\nbootstrap 95% interval for the median of 10 skewed values: [{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]  (sample median {np.median(x):.2f})")
g1 = np.array([5.1, 6.0, 5.8, 6.4, 7.2]); g2 = np.array([6.9, 7.5, 7.1, 8.3, 7.8])
obs = g2.mean() - g1.mean(); pooled = np.concatenate([g1, g2]); cnt = 0; trials = 50000
for _ in range(trials):
    rng.shuffle(pooled)
    cnt += abs(pooled[5:].mean() - pooled[:5].mean()) >= abs(obs)
print(f"permutation test, difference of means {obs:.2f}: p = {cnt / trials:.4f};  t-test p = {stats.ttest_ind(g1, g2).pvalue:.4f}")

# ---------------------------------------------------------------------------------------------
# 6. Regression to the mean / winner's curse: select the 'top hits' of a noisy screen and re-measure them.
# ---------------------------------------------------------------------------------------------
true_effect = rng.normal(0, 1, size=20000)
m1 = true_effect + rng.normal(0, 1, size=20000)           # first screen: effect plus measurement noise of the same size
m2 = true_effect + rng.normal(0, 1, size=20000)           # independent replicate
top = m1 > np.quantile(m1, 0.99)
print(f"\nwinner's curse: top 1% of a noisy screen: mean first measurement {m1[top].mean():.2f}, mean re-measurement {m2[top].mean():.2f}, mean true effect {true_effect[top].mean():.2f}")
slope = np.cov(m1, true_effect)[0, 1] / np.var(m1)
print(f"  the best linear prediction of the true effect from one measurement shrinks it by the factor {slope:.2f} (= signal variance / total variance = 1/2)")

# ---------------------------------------------------------------------------------------------
# 7. Simpson's paradox: a treatment looks harmful overall and helpful within every batch.
# ---------------------------------------------------------------------------------------------
#                  batch 1 (easy cases)   batch 2 (hard cases)
rows = {"treated":   [(81, 87), (192, 263)],      # (successes, total) in batch 1, batch 2
        "untreated": [(234, 270), (55, 80)]}
for k, v in rows.items():
    tot_s, tot_n = sum(s for s, _ in v), sum(n_ for _, n_ in v)
    print(f"  {k:<10} success rates by batch: " + ", ".join(f"{s / n_:.2f}" for s, n_ in v) + f"   overall {tot_s / tot_n:.2f}")
print("  -> treated does better in each batch, but worse overall, because most treated cases are in the hard batch (confounding)")

# ---------------------------------------------------------------------------------------------
# 8. Power: probability of detecting a true standardized effect d = 0.5 with a two-sample t-test at alpha = 0.05.
# ---------------------------------------------------------------------------------------------
print("\npower of a two-sample t-test, true effect 0.5 standard deviations, alpha = 0.05 (20000 simulated experiments each):")
for n in (10, 20, 64, 128):
    a = rng.normal(size=(20000, n)); b = rng.normal(0.5, 1, size=(20000, n))
    print(f"  n = {n:>3} per group: power {np.mean(stats.ttest_ind(a, b, axis=1).pvalue < 0.05):.3f}")
print("  rule of thumb: n per group = 16 / d^2 for 80% power at alpha = 0.05 -> d = 0.5 gives", int(16 / 0.5 ** 2))
