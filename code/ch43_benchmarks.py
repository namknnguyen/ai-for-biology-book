"""Chapter 43: how benchmarks mislead, in simulation.
1. Winner's curse and rank reproducibility: many models of (nearly) equal true quality evaluated on a finite test set.
2. Adaptive reuse of a test set: selecting the best of K variants on the same test set.
3. Metric choice with rare positives: the same classifier under AUROC, AUPRC, and precision at fixed recall as prevalence falls.
4. Resolution: test items needed to distinguish two models near a noise ceiling."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. winner's curse and rank reproducibility
print("== 1. 100 models whose true accuracies differ by SD 0.01 around 0.80; two independent test sets of n items ==")
print("test items n   SE of one accuracy   mean accuracy of the model ranked #1 on test A   its accuracy on fresh test B   true accuracy of that model   Spearman(rank A, rank B)   P(#1 on A is in top 10 on B)")
M = 100
for n in [200, 1000, 5000, 50000]:
    inflate, fresh, true_, rho, top10 = [], [], [], [], 0
    reps = 400
    for _ in range(reps):
        true = np.clip(rng.normal(0.80, 0.01, M), 0, 1)
        a = rng.binomial(n, true) / n; b = rng.binomial(n, true) / n
        i = a.argmax(); inflate.append(a[i]); fresh.append(b[i]); true_.append(true[i]); rho.append(stats.spearmanr(a, b)[0]); top10 += (b[i] >= np.sort(b)[-10])
    print(f"{n:9d}       {np.sqrt(0.8*0.2/n):8.4f}              {np.mean(inflate):8.4f}                                   {np.mean(fresh):8.4f}                       {np.mean(true_):8.4f}                    {np.mean(rho):6.2f}                      {top10/reps:5.2f}")

# ---------------------------------------------------------------- 2. adaptive reuse of one test set
print("\n== 2. A researcher tries K variants (true improvement 0 over a baseline with accuracy 0.800) and reports the best on the SAME test set (n = 2,000) ==")
n = 2000
print("variants tried K   reported improvement over baseline (mean)   improvement on a fresh test set (mean)   P(reported improvement > 1 SE of a paired difference)")
se_pair = np.sqrt(2 * 0.8 * 0.2 / n * 0.5)                                    # paired difference with correlation 0.5 between models
for K in [1, 5, 20, 100, 1000]:
    rep, fresh, sig = [], [], 0; reps = 1500
    for _ in range(reps):
        base = rng.binomial(n, 0.8) / n; var = rng.binomial(n, 0.8, K) / n; j = var.argmax()
        d_rep = var[j] - base; rep.append(d_rep); fresh.append(rng.binomial(n, 0.8) / n - rng.binomial(n, 0.8) / n); sig += d_rep > se_pair
    print(f"{K:12d}         {np.mean(rep):+10.4f}                                  {np.mean(fresh):+9.4f}                                 {sig/reps:6.2f}")

# ---------------------------------------------------------------- 3. metrics with rare positives
print("\n== 3. One classifier (scores N(1.5,1) for positives, N(0,1) for negatives), varying the positive rate ==")
print("positive rate   AUROC    AUPRC    precision at 50% recall    expected true hits among the top 100   (hits if chance)")
for prev in [0.5, 0.05, 0.005, 0.0005]:
    n_ = 400000; y = rng.random(n_) < prev; s = np.where(y, rng.normal(1.5, 1, n_), rng.normal(0, 1, n_))
    thr = np.quantile(s[y], 0.5); prec = y[s >= thr].mean(); top = np.argsort(-s)[:100]
    print(f"{prev:11.4f}   {roc_auc_score(y, s):.3f}    {average_precision_score(y, s):.3f}       {prec:6.3f}                      {y[top].sum():6.0f}                          {100*prev:6.2f}")

# ---------------------------------------------------------------- 4. resolution near the ceiling
print("\n== 4. Test items needed to detect an accuracy difference (two-sided alpha 0.05, power 0.8; paired models with error correlation 0.5) ==")
print("accuracy near   difference   items needed")
z = stats.norm.isf(0.025) + stats.norm.isf(0.2)
for p in [0.60, 0.80, 0.90, 0.95]:
    row = []
    for d in [0.05, 0.02, 0.01]:
        var = 2 * p * (1 - p) * (1 - 0.5)                                       # variance of the paired difference per item
        row.append(int(np.ceil(var * z ** 2 / d ** 2)))
    print(f"{p:8.2f}      0.05 -> {row[0]:6d}    0.02 -> {row[1]:6d}    0.01 -> {row[2]:7d}")
