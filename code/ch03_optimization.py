"""Chapter 3: gradient descent, conditioning, momentum, Adam, and implicit regularisation."""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Gradient check: the softmax cross-entropy gradient w.r.t. the logits is p - y.
# ---------------------------------------------------------------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def xent(z, y_idx):
    return -np.log(softmax(z)[y_idx])

z = rng.normal(size=5); y_idx = 2
analytic = softmax(z) - np.eye(5)[y_idx]
numeric = np.array([(xent(z + 1e-6 * np.eye(5)[i], y_idx) - xent(z - 1e-6 * np.eye(5)[i], y_idx)) / 2e-6
                    for i in range(5)])
print(f"softmax-xent gradient check: max |analytic - numeric| = {np.abs(analytic - numeric).max():.2e}")

# ---------------------------------------------------------------------------------------------
# 2. Quadratic f(w) = 1/2 w^T A w with condition number kappa. GD, heavy-ball momentum, and Adam.
# ---------------------------------------------------------------------------------------------
kappa, d = 100.0, 50
eigs = np.linspace(1.0, kappa, d)                  # eigenvalues from mu=1 to L=kappa
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
A = Q @ np.diag(eigs) @ Q.T
w0 = rng.normal(size=d)
tol = 1e-6

def run(update, steps=20000):
    w, state = w0.copy(), {}
    for t in range(1, steps + 1):
        g = A @ w
        w = update(w, g, state, t)
        if np.linalg.norm(w) < tol * np.linalg.norm(w0):
            return t
    return steps

mu, L = eigs.min(), eigs.max()
gd = lambda w, g, s, t: w - (2 / (mu + L)) * g                 # optimal constant step for GD
rate_gd = (kappa - 1) / (kappa + 1)

def heavy_ball(w, g, s, t):
    # Polyak momentum with the optimal parameters for a quadratic
    lr = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    v = beta * s.get("v", 0) - lr * g
    s["v"] = v
    return w + v

def adam(w, g, s, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * s.get("m", 0) + (1 - b1) * g
    v = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    s["m"], s["v"] = m, v
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(vh) + eps)

print(f"\nquadratic, kappa = {kappa:.0f}: iterations to shrink ||w|| by 1e{int(np.log10(tol))}")
print(f"  gradient descent (optimal step): {run(gd):6d}   theory ~ ln(1/tol)/ln(1/rho) = "
      f"{np.log(1 / tol) / np.log(1 / rate_gd):.0f}  (rho = (k-1)/(k+1) = {rate_gd:.3f})")
print(f"  heavy-ball momentum            : {run(heavy_ball):6d}   (rate improves from ~kappa to ~sqrt(kappa))")
print(f"  Adam (lr=0.1, no decay)        : {run(adam):6d}   (may oscillate; shown for contrast)")

# ---------------------------------------------------------------------------------------------
# 3. Implicit regularisation: GD on underdetermined least squares, started at 0,
#    converges to the minimum-norm interpolating solution (the pseudo-inverse solution).
# ---------------------------------------------------------------------------------------------
n, p = 30, 200                                      # many more features than samples
X = rng.normal(size=(n, p)); y = rng.normal(size=n)
w = np.zeros(p)
lr = 1.0 / np.linalg.norm(X, 2) ** 2
for _ in range(20000):
    w -= lr * X.T @ (X @ w - y)
w_minnorm = np.linalg.pinv(X) @ y
print(f"\nunderdetermined LS (n={n}, p={p}): train residual = {np.linalg.norm(X @ w - y):.1e}, "
      f"||w_GD - w_pinv|| = {np.linalg.norm(w - w_minnorm):.1e}, ||w_GD|| = {np.linalg.norm(w):.3f}")
w_other = w_minnorm + (np.eye(p) - np.linalg.pinv(X) @ X) @ rng.normal(size=p)  # another interpolating solution
print(f"another interpolating solution: residual = {np.linalg.norm(X @ w_other - y):.1e}, "
      f"||w|| = {np.linalg.norm(w_other):.3f}  (larger norm; GD did not choose it)")
