"""Chapter 45: distribution shift, in four controlled forms.
(1) covariate shift, (2) label (prevalence) shift, (3) concept shift (the mechanism differs), (4) a spurious shortcut (batch-like feature).
For each: what breaks, what is fixable without target labels, how many target labels are needed, and what happens to calibration and conformal coverage."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(0)
D = 10
w = rng.normal(0, 0.8, D)
def label(X, w_=None, inter=1.2):
    w_ = w if w_ is None else w_
    z = X @ w_ + inter * X[:, 0] * X[:, 1]
    return (rng.random(len(X)) < 1 / (1 + np.exp(-z))).astype(int)
def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
def conformal_cov(model, Xc, yc, Xt, yt, alpha=0.1, wt=None):
    """Split-conformal prediction sets for binary labels: score = 1 - p(true class)."""
    pc = model.predict_proba(Xc); s = 1 - pc[np.arange(len(yc)), yc]
    q = np.quantile(s, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
    pt = model.predict_proba(Xt); inset = (1 - pt[np.arange(len(yt)), yt]) <= q
    return inset.mean(), np.mean((1 - pt[:, 0] <= q).astype(int) + (1 - pt[:, 1] <= q).astype(int))
def report(name, m, Xt, yt, extra=""):
    p = m.predict_proba(Xt)[:, 1]; print(f"{name:46s} acc {np.mean((p > 0.5) == yt):.3f}  AUROC {roc_auc_score(yt, p):.3f}  AUPRC {average_precision_score(yt, p):.3f}  ECE {ece(p, yt):.3f} {extra}")

# ------------------------------------------------------------ 1. covariate shift
print("== 1. Covariate shift: P(x) changes, P(y|x) fixed; the true logit has an interaction x1*x2 that a linear model misses ==")
Xs = rng.standard_normal((6000, D)); ys = label(Xs)
mu = np.zeros(D); mu[:3] = [1.5, 1.5, -1.0]
Xt = rng.standard_normal((6000, D)) * np.r_[1.3, 1.3, np.ones(D - 2)] + mu; yt = label(Xt)
Xc = rng.standard_normal((2000, D)); yc = label(Xc)                           # source calibration set
print(f"source prevalence {ys.mean():.2f}; target prevalence {yt.mean():.2f}")
lr = LogisticRegression(max_iter=2000).fit(Xs, ys); gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xs, ys)
for nm, m in [("linear model, source-trained", lr), ("boosted trees, source-trained", gb)]:
    p_s = m.predict_proba(Xc)[:, 1]
    print(f"{nm:46s} source acc {np.mean((p_s > .5) == yc):.3f} AUROC {roc_auc_score(yc, p_s):.3f} | ", end=""); report("target", m, Xt, yt)
    cov, size = conformal_cov(m, Xc, yc, Xt, yt); cov0, _ = conformal_cov(m, Xc, yc, Xc[:1000], yc[:1000]); print(f"    90% conformal sets: coverage on source (exchangeable) {cov0:.3f}; on shifted target {cov:.3f}")
# importance weighting with a domain classifier on quadratic features
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(2, include_bias=False); dom = LogisticRegression(max_iter=3000, C=1.0).fit(pf.fit_transform(np.vstack([Xs, Xt])), np.r_[np.zeros(len(Xs)), np.ones(len(Xt))])
pr = dom.predict_proba(pf.transform(Xs))[:, 1]; iw = np.clip(pr / (1 - pr), 0, 20); print(f"importance weights: mean {iw.mean():.2f}, max {iw.max():.1f}, effective sample size {iw.sum() ** 2 / (iw ** 2).sum():.0f} of {len(iw)}")
lr_w = LogisticRegression(max_iter=2000).fit(Xs, ys, sample_weight=iw); gb_w = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xs, ys, sample_weight=iw)
report("linear model, importance-weighted", lr_w, Xt, yt); report("boosted trees, importance-weighted", gb_w, Xt, yt)
oracle = LogisticRegression(max_iter=2000).fit(np.c_[Xt, Xt[:, 0] * Xt[:, 1]], yt)                    # reference: a model with the right features trained on target labels
p = oracle.predict_proba(np.c_[Xt, Xt[:, 0] * Xt[:, 1]])[:, 1]; print(f"{'reference: right features, target labels':46s} acc {np.mean((p > 0.5) == yt):.3f}  AUROC {roc_auc_score(yt, p):.3f}")

# ------------------------------------------------------------ 2. label shift
print("\n== 2. Label (prevalence) shift: P(y) changes from 0.50 to 0.05 with P(x|y) fixed ==")
def gen(n, prev):
    y = (rng.random(n) < prev).astype(int); X = rng.standard_normal((n, D)) + y[:, None] * np.r_[0.9, 0.7, 0.5, np.zeros(D - 3)]; return X, y
Xs2, ys2 = gen(6000, 0.5); Xt2, yt2 = gen(20000, 0.05)
m = LogisticRegression(max_iter=2000).fit(Xs2, ys2); report("source-trained (prior 0.50)", m, Xt2, yt2)
pt = m.predict_proba(Xt2)[:, 1]; pi = 0.5                                                             # EM prior correction (Saerens et al.) using unlabeled target data
for _ in range(50):
    post = pt * (pi / 0.5) / (pt * (pi / 0.5) + (1 - pt) * ((1 - pi) / 0.5)); pi = post.mean()
adj = pt * (pi / 0.5) / (pt * (pi / 0.5) + (1 - pt) * ((1 - pi) / 0.5)); print(f"{'EM prevalence estimate (true 0.05)':46s} {pi:.3f}")
print(f"{'prior-corrected posteriors':46s} acc {np.mean((adj > 0.5) == yt2):.3f}  AUROC {roc_auc_score(yt2, adj):.3f}  AUPRC {average_precision_score(yt2, adj):.3f}  ECE {ece(adj, yt2):.3f}")
print(f"{'precision at the 0.5 threshold: uncorrected':46s} {np.mean(yt2[pt > .5]):.3f}; corrected {np.mean(yt2[adj > .5]):.3f}")

# ------------------------------------------------------------ 3. concept shift and the value of target labels
print("\n== 3. Concept shift: the mechanism differs in the target (35% of weights change sign or size); accuracy vs number of target labels ==")
w_t = w.copy(); chg = rng.random(D) < 0.35; w_t[chg] = -0.5 * w[chg] + rng.normal(0, 0.4, chg.sum())
def lab_t(X): return label(X, w_t, inter=0.4)
Xs3 = rng.standard_normal((6000, D)); ys3 = label(Xs3); Xt3 = rng.standard_normal((8000, D)); yt3 = lab_t(Xt3)
base = LogisticRegression(max_iter=2000).fit(Xs3, ys3); print(f"source-trained model, no target labels: target acc {base.score(Xt3, yt3):.3f}")
print("target labels   target-only   pooled source + target   source model fine-tuned (shrunk toward source)")
for n_t in [10, 30, 100, 300, 1000]:
    accs = []
    for rep in range(10):
        i = rng.choice(len(Xt3), n_t, replace=False); Xl, yl = Xt3[i], yt3[i]; te = np.setdiff1d(np.arange(len(Xt3)), i)
        a = LogisticRegression(max_iter=2000, C=0.5).fit(Xl, yl).score(Xt3[te], yt3[te]) if len(set(yl)) > 1 else 0.5
        wts = np.r_[np.ones(len(Xs3)) * 0.05, np.ones(n_t)]; b = LogisticRegression(max_iter=2000).fit(np.vstack([Xs3, Xl]), np.r_[ys3, yl], sample_weight=wts).score(Xt3[te], yt3[te])
        # fine-tuning: logistic regression on residual features with an offset equal to the source logit and an L2 penalty on the change
        off = base.decision_function(Xl); Xaug = np.c_[Xl, off]; c = LogisticRegression(max_iter=2000, C=0.3).fit(Xaug, yl).score(np.c_[Xt3[te], base.decision_function(Xt3[te])], yt3[te]) if len(set(yl)) > 1 else base.score(Xt3[te], yt3[te])
        accs.append((a, b, c))
    a, b, c = np.mean(accs, 0); print(f"{n_t:8d}        {a:8.3f}        {b:8.3f}                {c:8.3f}")

# ------------------------------------------------------------ 4. spurious shortcut
print("\n== 4. A shortcut feature (batch signature) correlates with the label in the source (r about 0.9) and not in the target ==")
def gen4(n, shortcut_corr):
    X = rng.standard_normal((n, D)); y = label(X, inter=0.0); s = np.where(rng.random(n) < shortcut_corr, 2 * y - 1, rng.choice([-1, 1], n)) + 0.3 * rng.standard_normal(n)
    return np.c_[X, s], y
Xs4, ys4 = gen4(6000, 0.9); Xt4, yt4 = gen4(6000, 0.0); Xs4v, ys4v = gen4(2000, 0.9)
erm = LogisticRegression(max_iter=2000).fit(Xs4, ys4)
print(f"ERM with the shortcut: source-like test acc {erm.score(Xs4v, ys4v):.3f}; target acc {erm.score(Xt4, yt4):.3f}; weight on shortcut {erm.coef_[0, -1]:.2f} vs mean |biological weight| {np.abs(erm.coef_[0, :-1]).mean():.2f}")
no_s = LogisticRegression(max_iter=2000).fit(Xs4[:, :-1], ys4); print(f"shortcut known and removed:           source-like acc {no_s.score(Xs4v[:, :-1], ys4v):.3f}; target acc {no_s.score(Xt4[:, :-1], yt4):.3f}")
Xa = Xs4.copy(); Xa[:, -1] = rng.permutation(Xa[:, -1]); aug = LogisticRegression(max_iter=2000).fit(np.vstack([Xs4, Xa]), np.r_[ys4, ys4]); print(f"augmentation (shuffle the shortcut):  source-like acc {aug.score(Xs4v, ys4v):.3f}; target acc {aug.score(Xt4, yt4):.3f}")
Xm, ym = gen4(300, 0.0); both = LogisticRegression(max_iter=2000).fit(np.vstack([Xs4, Xm]), np.r_[ys4, ym]); print(f"300 target-like examples added:       source-like acc {both.score(Xs4v, ys4v):.3f}; target acc {both.score(Xt4, yt4):.3f}")
