"""Chapter 8: EM, identifiability, graphical-model zeros, clusters in a continuum, reparameterisation."""
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. EM for a Gaussian mixture; the log-likelihood increases monotonically.
# ---------------------------------------------------------------------------------------------
def gauss_logpdf(X, mu, S):
    d = X.shape[1]; diff = X - mu
    sol = np.linalg.solve(S, diff.T).T
    return -0.5 * (np.sum(diff * sol, 1) + np.linalg.slogdet(S)[1] + d * np.log(2 * np.pi))

def em_gmm(X, K, iters=60, seed=0):
    r = np.random.default_rng(seed); N, d = X.shape
    mu = X[r.choice(N, K, replace=False)]; S = np.stack([np.cov(X.T)] * K); pi = np.full(K, 1 / K)
    ll_hist = []
    for _ in range(iters):
        logp = np.stack([np.log(pi[k]) + gauss_logpdf(X, mu[k], S[k]) for k in range(K)], 1)   # (N, K)
        m = logp.max(1, keepdims=True); lse = m + np.log(np.exp(logp - m).sum(1, keepdims=True))
        ll_hist.append(lse.sum())
        R = np.exp(logp - lse)                                       # E-step: responsibilities p(z|x)
        Nk = R.sum(0)                                                # M-step: expected complete-data MLE
        pi = Nk / N; mu = (R.T @ X) / Nk[:, None]
        S = np.stack([((R[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + 1e-6 * np.eye(d) for k in range(K)])
    return pi, mu, S, np.array(ll_hist)

true_mu = np.array([[0, 0], [4, 0], [2, 3.5]])
X = np.vstack([rng.normal(true_mu[k], 0.8, size=(300, 2)) for k in range(3)])
pi, mu, S, ll = em_gmm(X, 3)
print(f"EM: log-likelihood after iter 1 = {ll[0]:.1f}, final = {ll[-1]:.1f}, monotone non-decreasing = {bool(np.all(np.diff(ll) > -1e-8))}")
sk = GaussianMixture(3, covariance_type="full", random_state=0, n_init=5).fit(X)
print(f"  sklearn best-of-5 log-likelihood = {sk.score(X) * len(X):.1f}; recovered means (sorted by x): "
      f"{np.round(mu[np.argsort(mu[:, 0])], 2).tolist()}")

# ---------------------------------------------------------------------------------------------
# 2. Non-identifiability of factor-analysis latents: any rotation gives the same likelihood.
# ---------------------------------------------------------------------------------------------
D, k = 8, 3
W = rng.normal(size=(D, k)); Psi = np.diag(rng.uniform(0.2, 0.5, D))
R, _ = np.linalg.qr(rng.normal(size=(k, k)))                       # random orthogonal rotation of latent axes
cov1, cov2 = W @ W.T + Psi, (W @ R) @ (W @ R).T + Psi
print(f"\nfactor analysis: max |Cov(W) - Cov(WR)| = {np.abs(cov1 - cov2).max():.1e}  "
      f"(different loadings, identical data distribution) | max |W - WR| = {np.abs(W - W @ R).max():.2f}")

# ---------------------------------------------------------------------------------------------
# 3. Gaussian graphical model: marginal correlation vs conditional independence (chain A -> B -> C).
# ---------------------------------------------------------------------------------------------
n = 200_000
A = rng.normal(size=n); B = 0.9 * A + 0.5 * rng.normal(size=n); C = 0.9 * B + 0.5 * rng.normal(size=n)
data = np.stack([A, B, C], 1)
corr = np.corrcoef(data.T); prec = np.linalg.inv(np.cov(data.T))
partial = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
print(f"\nchain A->B->C: corr(A,C) = {corr[0, 2]:.3f}, but partial corr(A,C | B) = {partial[0, 2]:.3f} "
      f"(precision entry {prec[0, 2]:.3f} ~ 0 means conditionally independent)")

# ---------------------------------------------------------------------------------------------
# 4. A mixture model will find 'clusters' in a continuum; BIC keeps rewarding more components.
# ---------------------------------------------------------------------------------------------
t = rng.uniform(0, 3 * np.pi, 1500)
Z = np.stack([t * np.cos(t), t * np.sin(t)], 1) / 3 + 0.15 * rng.normal(size=(1500, 2))   # one continuous spiral
bics = {K: GaussianMixture(K, random_state=0, n_init=2).fit(Z).bic(Z) for K in (1, 2, 3, 5, 8, 12, 16)}
best = min(bics, key=bics.get)
print("\nspiral (a single continuous trajectory, no discrete types): BIC by K = "
      + ", ".join(f"{K}:{v:.0f}" for K, v in bics.items()) + f"  -> best K = {best}")

# ---------------------------------------------------------------------------------------------
# 5. Reparameterisation vs score-function gradient estimators for d/dmu E_{z~N(mu,s^2)}[z^2] = 2 mu.
# ---------------------------------------------------------------------------------------------
mu0, s0, M = 1.5, 1.0, 200_000
eps = rng.normal(size=M); z = mu0 + s0 * eps
g_reparam = 2 * z                                                  # d f(mu+s*eps)/d mu = f'(z)
g_score = (z ** 2) * (z - mu0) / s0 ** 2                           # f(z) * d log q(z)/d mu
print(f"\ngradient of E[z^2] wrt mu (true value {2 * mu0:.2f}): reparameterisation mean {g_reparam.mean():.3f}, sd {g_reparam.std():.2f}; "
      f"score-function mean {g_score.mean():.3f}, sd {g_score.std():.2f}")
# closed-form KL(N(m,s^2) || N(0,1)) = 0.5 (m^2 + s^2 - 1 - log s^2) vs Monte Carlo
m, s = 0.7, 0.6
zz = m + s * rng.normal(size=1_000_000)
logq = -0.5 * ((zz - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi); logp = -0.5 * zz ** 2 - 0.5 * np.log(2 * np.pi)
print(f"KL(N({m},{s}^2) || N(0,1)): closed form {0.5 * (m**2 + s**2 - 1 - np.log(s**2)):.4f}, Monte Carlo {np.mean(logq - logp):.4f}")
