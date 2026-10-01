"""Chapter M4: gradients, Jacobians, the multivariable chain rule, Hessians and saddles, gradient descent, Lagrange multipliers."""
import math

import numpy as np

rng = np.random.default_rng(0)

def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

# ---------------------------------------------------------------------------------------------
# 1. Gradient of f(x, y) = x^2 y + sin(y): partial derivatives, and the gradient points uphill.
# ---------------------------------------------------------------------------------------------
f = lambda v: v[0] ** 2 * v[1] + math.sin(v[1])
grad_f = lambda v: np.array([2 * v[0] * v[1], v[0] ** 2 + math.cos(v[1])])
p = np.array([1.0, 2.0])
print("f(x,y) = x^2 y + sin y at (1, 2): f =", round(f(p), 5))
print("  analytic gradient", np.round(grad_f(p), 5), "  numeric gradient", np.round(num_grad(f, p), 5))
g = grad_f(p); u = g / np.linalg.norm(g)
rand_dirs = [(lambda d: d / np.linalg.norm(d))(rng.normal(size=2)) for _ in range(200)]
rates = [(f(p + 1e-5 * d) - f(p)) / 1e-5 for d in rand_dirs]
print(f"  rate of change along the gradient direction: {(f(p + 1e-5 * u) - f(p)) / 1e-5:.4f}  (= ||grad|| = {np.linalg.norm(g):.4f});"
      f" best of 200 random directions: {max(rates):.4f}")

# ---------------------------------------------------------------------------------------------
# 2. Chain rule for a composition: loss = (w . x + b - y)^2 passed through a sigmoid and a log.
#    L(w) = -log sigmoid(w.x)  for one example with label 1;  dL/dw = -(1 - sigmoid(w.x)) x.
# ---------------------------------------------------------------------------------------------
sig = lambda z: 1 / (1 + np.exp(-z))
x = rng.normal(size=5); w = rng.normal(size=5)
L = lambda w_: -np.log(sig(w_ @ x))
analytic = -(1 - sig(w @ x)) * x
print("\nchain rule for L(w) = -log sigmoid(w.x): max |analytic - numeric| =", f"{np.abs(analytic - num_grad(L, w)).max():.2e}")

# Jacobian of a vector-valued map F(x) = (x0*x1, x0 + x1^2, sin(x0)):  3 outputs, 2 inputs -> 3 x 2 matrix
F = lambda v: np.array([v[0] * v[1], v[0] + v[1] ** 2, math.sin(v[0])])
J_analytic = lambda v: np.array([[v[1], v[0]], [1.0, 2 * v[1]], [math.cos(v[0]), 0.0]])
v0 = np.array([0.7, -1.2])
J_num = np.column_stack([(F(v0 + 1e-6 * np.eye(2)[i]) - F(v0 - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
print("Jacobian of F: shape", J_num.shape, " max |analytic - numeric| =", f"{np.abs(J_analytic(v0) - J_num).max():.1e}")
# chain rule for Jacobians: J_(G o F) = J_G J_F, with G(u) = (u0 + u1 + u2, u0 u2)
G = lambda u_: np.array([u_[0] + u_[1] + u_[2], u_[0] * u_[2]])
JG = lambda u_: np.array([[1.0, 1.0, 1.0], [u_[2], 0.0, u_[0]]])
comp = lambda v: G(F(v))
Jc_num = np.column_stack([(comp(v0 + 1e-6 * np.eye(2)[i]) - comp(v0 - 1e-6 * np.eye(2)[i])) / 2e-6 for i in range(2)])
print("chain rule J_(G o F) = J_G J_F: max difference", f"{np.abs(JG(F(v0)) @ J_analytic(v0) - Jc_num).max():.1e}", " (shapes (2x3)(3x2) -> 2x2)")

# ---------------------------------------------------------------------------------------------
# 3. Critical points and the Hessian: f(x,y) = x^2 - y^2 has a saddle at 0 (eigenvalues +2, -2);
#    x^2 + 3y^2 has a minimum (eigenvalues 2, 6).
# ---------------------------------------------------------------------------------------------
for name, H in (("x^2 - y^2", np.array([[2.0, 0], [0, -2.0]])), ("x^2 + 3y^2", np.array([[2.0, 0], [0, 6.0]]))):
    ev = np.linalg.eigvalsh(H)
    kind = "minimum" if (ev > 0).all() else "maximum" if (ev < 0).all() else "saddle"
    print(f"  Hessian of {name}: eigenvalues {ev} -> {kind} at the origin")

# ---------------------------------------------------------------------------------------------
# 4. Gradient descent on a quadratic f(w) = 1/2 w^T A w with condition number kappa = lambda_max / lambda_min.
#    Steps needed to reduce ||w|| by 1e-6 with the best fixed step size scale like kappa.
# ---------------------------------------------------------------------------------------------
print("\ngradient descent on 1/2 w^T A w, d = 20, best fixed step 2/(lmin + lmax):")
for kappa in (1, 10, 100, 1000):
    eigs = np.linspace(1.0, float(kappa), 20) if kappa > 1 else np.ones(20)
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    A = Q @ np.diag(eigs) @ Q.T
    w = rng.normal(size=20); n0 = np.linalg.norm(w)
    lr = 2 / (eigs.min() + eigs.max())
    steps = 0
    while np.linalg.norm(w) > 1e-6 * n0 and steps < 200000:
        w = w - lr * (A @ w); steps += 1
    print(f"  kappa = {kappa:>5}: {steps:>6} steps   (theory ~ (kappa/2) ln(1e6) = {kappa / 2 * math.log(1e6):.0f})")
A = np.diag([1.0, 10.0]); w = np.array([1.0, 1.0])
for lr in (0.05, 0.19, 0.21):
    ww = w.copy()
    for _ in range(100): ww = ww - lr * (A @ ww)
    print(f"  learning rate {lr}: ||w|| after 100 steps = {np.linalg.norm(ww):.3e}  (diverges above 2/lambda_max = 0.2)")

# ---------------------------------------------------------------------------------------------
# 5. Least squares from the gradient: grad ||y - Xw||^2 = -2 X^T (y - Xw). Setting it to zero = normal equations.
# ---------------------------------------------------------------------------------------------
X = rng.normal(size=(200, 5)); w_true = np.array([1.0, -2.0, 0.0, 0.5, 3.0]); y = X @ w_true + 0.1 * rng.normal(size=200)
w_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("\nnormal-equation solution", np.round(w_hat, 3), " gradient at the solution: max |g| =", f"{np.abs(-2 * X.T @ (y - X @ w_hat)).max():.1e}")
w_gd = np.zeros(5)
for _ in range(500): w_gd = w_gd + 0.001 * (X.T @ (y - X @ w_gd))
print("gradient descent after 500 steps agrees:", np.allclose(w_gd, w_hat, atol=1e-3))

# ---------------------------------------------------------------------------------------------
# 6. Lagrange multipliers: maximize entropy -sum p_i log p_i subject to sum p_i = 1 and sum p_i E_i = <E>.
#    Calculus gives p_i proportional to exp(-beta E_i) (the Boltzmann / softmax form). Verify numerically.
# ---------------------------------------------------------------------------------------------
E = np.array([0.0, 1.0, 2.0, 3.0, 4.0]); target = 1.2
def boltz(beta):
    z = np.exp(-beta * E); return z / z.sum()
lo, hi = -50.0, 50.0
for _ in range(200):
    mid = (lo + hi) / 2
    if boltz(mid) @ E > target: lo = mid
    else: hi = mid
beta = (lo + hi) / 2
p_star = boltz(beta)
ent = lambda p: -np.sum(p * np.log(p))
best = -1
for _ in range(20000):                       # random feasible competitors: p >= 0, sum 1, mean energy = target
    q = rng.dirichlet(np.ones(5))
    Ec = E - E.mean()                         # shifting along Ec keeps sum(q) = 1 and sets the mean energy to the target
    q = q + (target - q @ E) * Ec / (Ec @ Ec)
    if (q > 1e-9).all() and abs(q.sum() - 1) < 1e-9:
        best = max(best, ent(q))
print(f"\nmax-entropy with mean energy {target}: Lagrange solution p = {np.round(p_star, 4)}, beta = {beta:.4f}, entropy {ent(p_star):.4f} nats")
print(f"  best entropy among {20000} random feasible distributions that satisfied the constraints: {best:.4f}  (never exceeds the Boltzmann solution)")

# ---------------------------------------------------------------------------------------------
# 7. A double integral: the 2D standard Gaussian density integrates to 1; integrate on a grid.
# ---------------------------------------------------------------------------------------------
g1 = np.linspace(-6, 6, 1201); dx = g1[1] - g1[0]
XX, YY = np.meshgrid(g1, g1)
dens = np.exp(-(XX ** 2 + YY ** 2) / 2) / (2 * math.pi)
print(f"\n2D Gaussian integrates to {dens.sum() * dx * dx:.6f};  P(x^2 + y^2 < 1) = {dens[XX ** 2 + YY ** 2 < 1].sum() * dx * dx:.4f} (exact 1 - e^-0.5 = {1 - math.exp(-0.5):.4f})")
