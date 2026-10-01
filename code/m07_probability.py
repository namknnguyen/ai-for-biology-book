"""Chapter M7: Bayes' rule, independence vs correlation, distributions, the law of large numbers, the CLT, Monte Carlo."""
import math

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Bayes' rule for a screening test: the base-rate effect.
# ---------------------------------------------------------------------------------------------
def ppv(prev, sens, spec):
    return prev * sens / (prev * sens + (1 - prev) * (1 - spec))

print("screening test, sensitivity 95%, specificity 95%: P(disease | positive) by prevalence")
for prev in (0.5, 0.1, 0.01, 0.001):
    print(f"  prevalence {prev:>6.1%}: P(disease | positive) = {ppv(prev, 0.95, 0.95):.3f}")
N = 1_000_000
dis = rng.random(N) < 0.01
pos = np.where(dis, rng.random(N) < 0.95, rng.random(N) < 0.05)
print(f"simulation of 1,000,000 people at 1% prevalence: P(disease | positive) = {dis[pos].mean():.3f};  {pos.sum():,} positives of which {dis[pos].sum():,} are true")

# ---------------------------------------------------------------------------------------------
# 2. Uncorrelated does not mean independent.
# ---------------------------------------------------------------------------------------------
x = rng.normal(size=200000); y = x ** 2
print(f"\nY = X^2 with X ~ N(0,1): correlation(X, Y) = {np.corrcoef(x, y)[0, 1]:+.4f} (uncorrelated), "
      f"but E[Y] = {y.mean():.3f} while E[Y | |X| > 2] = {y[np.abs(x) > 2].mean():.3f} (strongly dependent)")

# ---------------------------------------------------------------------------------------------
# 3. Distributions: binomial, Poisson as its rare-event limit, over-dispersion, Gaussian, exponential.
# ---------------------------------------------------------------------------------------------
n, p = 1000, 0.003
binom = np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(8)])
lam = n * p
pois = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(8)])
print(f"\nBinomial(n=1000, p=0.003) vs Poisson(3), k = 0..7:\n  binomial {np.round(binom, 4)}\n  poisson  {np.round(pois, 4)}   max |difference| = {np.abs(binom - pois).max():.1e}")
mu_nb, r_nb = 5.0, 2.0
lam_cells = rng.gamma(shape=r_nb, scale=mu_nb / r_nb, size=100000)          # the rate varies from cell to cell
counts = rng.poisson(lam_cells)
print(f"Poisson counts: mean = variance (Fano factor 1). A gamma-mixed Poisson (negative binomial): mean {counts.mean():.2f}, variance {counts.var():.2f}, "
      f"Fano {counts.var() / counts.mean():.2f}  (theory 1 + mu/r = {1 + mu_nb / r_nb:.2f})")
z = rng.normal(size=1_000_000)
print(f"Gaussian: fraction within 1, 2, 3 standard deviations = {np.mean(np.abs(z) < 1):.4f}, {np.mean(np.abs(z) < 2):.4f}, {np.mean(np.abs(z) < 3):.4f}  (theory 0.6827, 0.9545, 0.9973)")
T = rng.exponential(scale=4.0, size=1_000_000)                              # waiting times with mean 4
s_, t_ = 3.0, 2.0
print(f"exponential waiting times are memoryless: P(T > {t_}) = {np.mean(T > t_):.4f},  P(T > {s_ + t_} | T > {s_}) = {np.mean(T[T > s_] > s_ + t_):.4f}  (theory {math.exp(-t_ / 4):.4f})")

# ---------------------------------------------------------------------------------------------
# 4. Law of large numbers and the central limit theorem, starting from a very skewed distribution (exponential).
# ---------------------------------------------------------------------------------------------
print("\nsample mean of n exponential(1) draws: LLN (the mean approaches 1) and CLT (standardized mean approaches N(0,1))")
print("      n   mean of means   std of means   1/sqrt(n)   skewness   P(standardized mean < 1.96)  (normal: 0.975)")
for nn in (1, 5, 30, 200):
    m = rng.exponential(size=(100000, nn)).mean(1)
    zs = (m - 1) * math.sqrt(nn)
    skew = np.mean(((m - m.mean()) / m.std()) ** 3)
    print(f"  {nn:>5}   {m.mean():.4f}        {m.std():.4f}         {1 / math.sqrt(nn):.4f}      {skew:+.3f}     {np.mean(zs < 1.96):.4f}")

# ---------------------------------------------------------------------------------------------
# 5. Monte Carlo: estimate pi from random points; error shrinks like 1/sqrt(n) whatever the dimension.
# ---------------------------------------------------------------------------------------------
print("\nMonte Carlo estimate of pi (fraction of random points in the unit square falling inside the quarter circle, times 4):")
for nn in (100, 10_000, 1_000_000):
    errs = []
    for _ in range(200):
        pts = rng.random((nn, 2))
        errs.append(abs(4 * np.mean((pts ** 2).sum(1) < 1) - math.pi))
    print(f"  n = {nn:>9,}: typical |error| = {np.mean(errs):.4f}   (theory: 1.31/sqrt(n) = {1.31 / math.sqrt(nn):.4f})")

# ---------------------------------------------------------------------------------------------
# 6. Joint, marginal, conditional: genotype at two linked loci.
# ---------------------------------------------------------------------------------------------
joint = np.array([[0.40, 0.10],     # rows: allele at locus 1 (A, a); columns: allele at locus 2 (B, b)
                  [0.05, 0.45]])
m1, m2 = joint.sum(1), joint.sum(0)
cond = joint / m1[:, None]
cov = (joint * np.outer([1, 0], [1, 0])).sum() - m1[0] * m2[0]
print(f"\njoint table of two linked loci (rows A/a, columns B/b):\n{joint}\n  marginals: P(A) = {m1[0]:.2f}, P(B) = {m2[0]:.2f};  P(B | A) = {cond[0, 0]:.3f}, P(B | a) = {cond[1, 0]:.3f}")
print(f"  independence would give P(A,B) = {m1[0] * m2[0]:.3f}; observed 0.400 -> covariance of the indicators {cov:.3f} (linkage disequilibrium D)")

# variance of a sum: Var(X+Y) = Var X + Var Y + 2 Cov(X, Y)
a = rng.normal(size=100000); b = 0.6 * a + 0.8 * rng.normal(size=100000)
print(f"\nVar(X+Y) = {np.var(a + b):.3f}; Var X + Var Y + 2 Cov = {np.var(a) + np.var(b) + 2 * np.cov(a, b)[0, 1]:.3f};  if independent it would be {np.var(a) + np.var(b):.3f}")
