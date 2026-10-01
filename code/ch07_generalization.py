"""Chapter 7: double descent, genetic architecture vs regulariser, selection leakage, metrics."""
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Double descent for minimum-norm least squares. n=40 training points, D=200 candidate features,
#    use the first p features. The test error peaks near p = n (interpolation threshold), then falls again.
# ---------------------------------------------------------------------------------------------
n, D, noise, reps = 40, 200, 0.5, 60
beta = rng.normal(size=D); beta /= np.linalg.norm(beta)          # dense true signal, unit norm
print("double descent (min-norm least squares): test MSE vs number of features p, n = 40")
rows = []
for p in (5, 10, 20, 30, 38, 40, 42, 50, 80, 120, 200):
    errs = []
    for _ in range(reps):
        X = rng.normal(size=(n, D)); y = X @ beta + noise * rng.normal(size=n)
        w = np.linalg.pinv(X[:, :p]) @ y                          # min-norm solution (OLS when p<n)
        Xt = rng.normal(size=(2000, D)); yt = Xt @ beta + noise * rng.normal(size=2000)
        errs.append(np.mean((Xt[:, :p] @ w - yt) ** 2))
    rows.append((p, np.median(errs)))
print("  " + "  ".join(f"p={p}:{e:.2f}" for p, e in rows))

# ---------------------------------------------------------------------------------------------
# 2. Genetic architecture decides which regulariser wins (p >> n).
# ---------------------------------------------------------------------------------------------
def r2(y, yhat): return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
def trial(sparse, n=300, p=1000, h2=0.5, k=20):
    X = rng.normal(size=(n + 2000, p))
    b = np.zeros(p)
    idx = rng.choice(p, k, replace=False) if sparse else np.arange(p)
    b[idx] = rng.normal(size=len(idx))
    g = X @ b
    y = g + rng.normal(size=len(g)) * np.sqrt(g.var() * (1 - h2) / h2)
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    lasso = LassoCV(cv=5, n_jobs=1, max_iter=5000).fit(Xtr, ytr)
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Xtr, ytr)
    return r2(yte, lasso.predict(Xte)), r2(yte, ridge.predict(Xte))
print("\nR^2 on held-out data (p=1000, heritability 0.5 so the ceiling is 0.5), mean of 4 simulations:")
for n_train in (300, 1500):
    for name, sp in (("sparse (20 causal features)", True), ("dense (all 1000 small effects)", False)):
        res = np.array([trial(sp, n=n_train) for _ in range(4)]).mean(0)
        print(f"  n = {n_train:4d}  {name:32s} LASSO = {res[0]:5.2f}   ridge = {res[1]:5.2f}")

# ---------------------------------------------------------------------------------------------
# 3. Selection leakage: choosing features on ALL data before cross-validation (labels are pure noise).
# ---------------------------------------------------------------------------------------------
n, p, reps = 60, 4000, 40
wrong, right = [], []
for rep in range(reps):
    X = rng.normal(size=(n, p)); y = rng.integers(0, 2, n)        # no relationship between X and y at all
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    top = np.argsort(np.abs(((X - X.mean(0)) * (y - y.mean())[:, None]).sum(0)))[-20:]   # WRONG: used every label
    wrong.append(cross_val_score(LogisticRegression(C=1.0, max_iter=1000), X[:, top], y, cv=cv).mean())
    acc = []                                                      # RIGHT: select inside each training fold
    for tr, te in cv.split(X, y):
        sel = np.argsort(np.abs(((X[tr] - X[tr].mean(0)) * (y[tr] - y[tr].mean())[:, None]).sum(0)))[-20:]
        acc.append(LogisticRegression(C=1.0, max_iter=1000).fit(X[tr][:, sel], y[tr]).score(X[te][:, sel], y[te]))
    right.append(np.mean(acc))
print(f"\nselection leakage on pure noise (n=60, p=4000, {reps} datasets): CV accuracy with selection on all data = "
      f"{np.mean(wrong):.2f} +/- {np.std(wrong):.2f}; selection inside folds = {np.mean(right):.2f} +/- {np.std(right):.2f} (truth: 0.50)")

# ---------------------------------------------------------------------------------------------
# 4. AUROC = P(score_pos > score_neg) (Mann-Whitney U / rank formula); AUPRC depends on prevalence.
# ---------------------------------------------------------------------------------------------
from scipy.stats import rankdata
for prev in (0.5, 0.01, 0.001):
    N = 400_000
    y = rng.random(N) < prev
    s = rng.normal(size=N) + 1.8 * y                              # same score separation at every prevalence
    ranks = rankdata(s); n_pos, n_neg = y.sum(), (~y).sum()
    auc_rank = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)   # P(pos score > neg score)
    print(f"prevalence {prev:6.3f}: AUROC = {roc_auc_score(y, s):.3f} (rank formula {auc_rank:.3f}), "
          f"AUPRC = {average_precision_score(y, s):.3f} (random-guess AUPRC = {prev:.3f})")

# ---------------------------------------------------------------------------------------------
# 5. Between-gene vs within-gene correlation: a predictor that only knows each gene's mean.
# ---------------------------------------------------------------------------------------------
genes, indiv = 2000, 100
gene_mean = rng.normal(0, 2.0, genes)                             # large between-gene variance
indiv_effect = rng.normal(0, 0.2, (genes, indiv))                 # small between-individual variance
y = gene_mean[:, None] + indiv_effect
pred = np.repeat(gene_mean[:, None], indiv, axis=1) + 0.0         # perfect gene mean, no individual information
print(f"\nbetween/within: Pearson over all (gene, individual) points = {np.corrcoef(y.ravel(), pred.ravel())[0, 1]:.3f}; "
      f"mean within-gene Pearson across individuals = "
      f"{np.nanmean([np.corrcoef(y[g], pred[g] + 1e-9 * rng.normal(size=indiv))[0, 1] for g in range(200)]):.3f}")
