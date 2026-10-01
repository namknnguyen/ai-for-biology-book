"""Chapter 4: counts, pseudoreplication, multiple testing, and Bayesian bookkeeping."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Negative binomial = Gamma-Poisson mixture:  Var(Y) = mu + mu^2 / r
# ---------------------------------------------------------------------------------------------
mu, r, n = 5.0, 2.0, 400_000
rate = rng.gamma(shape=r, scale=mu / r, size=n)       # cell-to-cell variation in the true rate
y = rng.poisson(rate)                                  # sampling noise given the rate
print(f"Gamma-Poisson: mean = {y.mean():.3f} (mu = {mu}), var = {y.var():.3f} "
      f"(mu + mu^2/r = {mu + mu**2 / r:.3f}; Poisson alone would give {mu})")

# ---------------------------------------------------------------------------------------------
# 2. Pseudoreplication: testing cells instead of donors inflates false positives.
#    No true condition effect exists; donors differ (random effect), cells within a donor are noisy.
# ---------------------------------------------------------------------------------------------
def false_positive_rate(n_donors=5, cells=200, sd_donor=1.0, sd_cell=1.0, sims=2000, alpha=0.05):
    fp_cell = fp_donor = 0
    for _ in range(sims):
        data = []
        for cond in (0, 1):
            donor_effect = rng.normal(0, sd_donor, size=n_donors)
            x = donor_effect[:, None] + rng.normal(0, sd_cell, size=(n_donors, cells))  # (donors, cells)
            data.append(x)
        p_cell = stats.ttest_ind(data[0].ravel(), data[1].ravel(), equal_var=False).pvalue
        p_donor = stats.ttest_ind(data[0].mean(1), data[1].mean(1), equal_var=False).pvalue  # "pseudobulk"
        fp_cell += p_cell < alpha
        fp_donor += p_donor < alpha
    return fp_cell / sims, fp_donor / sims

fpr_cell, fpr_donor = false_positive_rate()
rho = 1.0 / (1.0 + 1.0)                                # intra-donor correlation sd_d^2/(sd_d^2+sd_c^2)
n_eff = 1000 / (1 + (200 - 1) * rho)                   # effective n per condition from the equicorrelation formula
print(f"\npseudoreplication (5 donors x 200 cells per condition, NO true effect): "
      f"false positive rate, cell-level test = {fpr_cell:.2f}; donor-level (pseudobulk) test = {fpr_donor:.2f}")
print(f"  intra-donor correlation rho = {rho:.2f};  effective n per condition ~ {n_eff:.1f} (nominal 1000)")

# ---------------------------------------------------------------------------------------------
# 3. Multiple testing: 20,000 genes, 10% truly different. Raw p<0.05 vs Bonferroni vs Benjamini-Hochberg.
# ---------------------------------------------------------------------------------------------
m, frac_true = 20_000, 0.10
is_true = rng.random(m) < frac_true
z = rng.normal(size=m) + np.where(is_true, 3.0, 0.0)  # effect of 3 standard errors for true genes
p = 2 * stats.norm.sf(np.abs(z))

def report(name, rej):
    fd = np.sum(rej & ~is_true)
    print(f"  {name:22s} rejections = {rej.sum():5d}, false discoveries = {fd:5d}, "
          f"FDP = {fd / max(rej.sum(), 1):.3f}, power = {np.sum(rej & is_true) / is_true.sum():.2f}")

def benjamini_hochberg(p, q=0.05):
    order = np.argsort(p)
    thresh = q * np.arange(1, len(p) + 1) / len(p)
    below = p[order] <= thresh
    k = np.max(np.nonzero(below)[0]) + 1 if below.any() else 0
    rej = np.zeros(len(p), dtype=bool)
    rej[order[:k]] = True
    return rej

print("\nmultiple testing (20,000 genes, 2,000 truly different):")
report("raw p < 0.05", p < 0.05)
report("Bonferroni (FWER 0.05)", p < 0.05 / m)
report("Benjamini-Hochberg q=.05", benjamini_hochberg(p))

# ---------------------------------------------------------------------------------------------
# 4. Bayesian bookkeeping: posterior odds = prior odds x likelihood ratios (ACMG/AMP-style, Tavtigian 2018).
# ---------------------------------------------------------------------------------------------
prior = 0.10
OP_very_strong = 350.0
levels = {"supporting": OP_very_strong ** (1 / 8), "moderate": OP_very_strong ** (1 / 4),
          "strong": OP_very_strong ** (1 / 2), "very strong": OP_very_strong}
print("\nBayesian variant classification (prior P(pathogenic) = 0.10): odds of pathogenicity per evidence level")
print("  " + ", ".join(f"{k} = {v:.2f}" for k, v in levels.items()))
odds = prior / (1 - prior) * levels["moderate"] * levels["supporting"] * levels["supporting"]
print(f"  one moderate + two supporting -> posterior odds = {odds:.2f}, P(pathogenic) = {odds / (1 + odds):.2f}")

# Ioannidis-style positive predictive value of a 'significant' finding
for prior_odds in (1.0, 0.1, 0.01):
    power, alpha = 0.8, 0.05
    ppv = power * prior_odds / (power * prior_odds + alpha)
    print(f"  prior odds that a tested hypothesis is true = {prior_odds:5.2f}: PPV of a significant result = {ppv:.2f}")

# ---------------------------------------------------------------------------------------------
# 5. Winner's curse / regression to the mean when selecting the best of many noisy measurements.
#    True effects theta ~ N(0, tau^2); measurement = theta + N(0, s^2). Pick the top 10 of 1000 candidates.
# ---------------------------------------------------------------------------------------------
tau, s, n_cand, k_top = 1.0, 1.0, 1000, 10
theta = rng.normal(0, tau, n_cand)
meas = theta + rng.normal(0, s, n_cand)
top = np.argsort(meas)[-k_top:]
shrink = tau ** 2 / (tau ** 2 + s ** 2)                # posterior-mean shrinkage factor (normal-normal model)
print(f"\nwinner's curse: top {k_top} of {n_cand}: mean measured = {meas[top].mean():.2f}, "
      f"mean TRUE = {theta[top].mean():.2f}, shrunken estimate (x{shrink:.2f}) = {shrink * meas[top].mean():.2f}")
