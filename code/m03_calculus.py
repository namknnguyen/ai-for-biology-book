"""Chapter M3: derivatives, finite-difference error, Taylor series, Riemann sums, an ODE for mRNA, and Newton's method."""
import math

import numpy as np

# ---------------------------------------------------------------------------------------------
# 1. Derivative as a limit of slopes, and the numerical derivative's U-shaped error:
#    truncation error shrinks like h, but floating-point cancellation grows like eps/h.
# ---------------------------------------------------------------------------------------------
f, df = math.exp, math.exp          # d/dx e^x = e^x, so the true derivative at x=1 is e
x0 = 1.0
print("forward difference (f(x+h)-f(x))/h  vs  central difference (f(x+h)-f(x-h))/(2h), f = exp at x = 1")
print("      h     forward error   central error")
best_f = best_c = (1e9, None)
for e in range(1, 17):
    h = 10.0 ** (-e)
    err_f = abs((f(x0 + h) - f(x0)) / h - df(x0))
    err_c = abs((f(x0 + h) - f(x0 - h)) / (2 * h) - df(x0))
    if err_f < best_f[0]: best_f = (err_f, h)
    if err_c < best_c[0]: best_c = (err_c, h)
    if e in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        print(f"  1e-{e:02d}   {err_f:12.3e}   {err_c:12.3e}")
print(f"best forward: error {best_f[0]:.1e} at h = {best_f[1]:.0e};  best central: error {best_c[0]:.1e} at h = {best_c[1]:.0e}")

# ---------------------------------------------------------------------------------------------
# 2. Taylor series: the best polynomial approximation near a point. sigma(x) = 1/(1+e^-x) near 0:
#    sigma(x) ~ 1/2 + x/4 - x^3/48.
# ---------------------------------------------------------------------------------------------
sig = lambda x: 1 / (1 + math.exp(-x))
print("\nTaylor approximations of the sigmoid at 0:  sigma(x) vs 1/2 + x/4  vs  1/2 + x/4 - x^3/48")
for x in (0.1, 0.5, 1.0, 2.0, 4.0):
    t1 = 0.5 + x / 4
    t3 = 0.5 + x / 4 - x ** 3 / 48
    print(f"  x = {x:3.1f}: exact {sig(x):.5f}   linear {t1:.5f} (err {abs(t1 - sig(x)):.1e})   cubic {t3:.5f} (err {abs(t3 - sig(x)):.1e})")
print("e^x with 1+x+x^2/2+x^3/6 at x=1:", round(1 + 1 + 0.5 + 1 / 6, 5), "vs", round(math.e, 5))
print("log(1+x) ~ x for small x: x = 0.01 ->", round(math.log1p(0.01), 6), "(and 1 - x ~ exp(-x): 0.99 vs", round(math.exp(-0.01), 6), ")")

# ---------------------------------------------------------------------------------------------
# 3. Integral as accumulated change: Riemann sums converge to the integral; integral of exp(-x^2) over R is sqrt(pi).
# ---------------------------------------------------------------------------------------------
print("\nRiemann sums for the integral of x^2 on [0,1] (exact 1/3):")
for n in (4, 16, 64, 1024):
    xs = np.linspace(0, 1, n + 1)
    left = np.sum(xs[:-1] ** 2) / n
    mid = np.sum(((xs[:-1] + xs[1:]) / 2) ** 2) / n
    print(f"  n = {n:>4}: left-endpoint {left:.6f}  (error {abs(left - 1 / 3):.1e})   midpoint {mid:.6f}  (error {abs(mid - 1 / 3):.1e})")
xs = np.linspace(-8, 8, 160001)
gauss = np.sum(np.exp(-xs ** 2)) * (xs[1] - xs[0])
print(f"integral of exp(-x^2) over the real line: numeric {gauss:.8f}, sqrt(pi) = {math.sqrt(math.pi):.8f}")

# ---------------------------------------------------------------------------------------------
# 4. A differential equation for gene expression: dm/dt = alpha - delta * m  (production minus degradation).
#    Exact solution m(t) = alpha/delta + (m0 - alpha/delta) exp(-delta t). Compare Euler's method.
# ---------------------------------------------------------------------------------------------
alpha, delta, m0 = 20.0, 0.25, 0.0          # molecules/h, 1/h (half-life ln2/0.25 = 2.77 h)
exact = lambda t: alpha / delta + (m0 - alpha / delta) * math.exp(-delta * t)
print(f"\nmRNA ODE: steady state alpha/delta = {alpha / delta:.1f}; half-life of approach = {math.log(2) / delta:.2f} h")
for dt in (2.0, 0.5, 0.1, 0.01):
    m, t = m0, 0.0
    while t < 12 - 1e-12:
        m += dt * (alpha - delta * m)
        t += dt
    print(f"  Euler dt = {dt:5.2f}: m(12) = {m:8.4f}   exact {exact(12):8.4f}   error {abs(m - exact(12)):.2e}")
print("  (halving dt roughly halves the error: Euler is first-order accurate; it is stable only if delta*dt < 2)")
m = m0
for _ in range(10):
    m += 10.0 * (alpha - delta * m)
print("  Euler with dt = 10 (delta*dt = 2.5), after 10 steps:", round(m, 1), "(the exact solution never exceeds 80: the scheme diverges)")

# ---------------------------------------------------------------------------------------------
# 5. Newton's method for a root, and for a minimum (set the derivative to zero): fast near the answer.
# ---------------------------------------------------------------------------------------------
g = lambda x: x ** 3 - 2 * x - 5
dg = lambda x: 3 * x ** 2 - 2
x = 2.0
print("\nNewton's method for x^3 - 2x - 5 = 0, starting at 2.0:")
for i in range(5):
    x = x - g(x) / dg(x)
    print(f"  step {i + 1}: x = {x:.12f}   |g(x)| = {abs(g(x)):.1e}")

# maximum-likelihood example: coin with 7 heads in 10 flips, L(p) = p^7 (1-p)^3, maximize log L by calculus
ps = np.linspace(0.001, 0.999, 9999)
logL = 7 * np.log(ps) + 3 * np.log(1 - ps)
print(f"\nlog-likelihood for 7 heads in 10 flips is maximized at p = {ps[np.argmax(logL)]:.3f}  (calculus: d/dp = 7/p - 3/(1-p) = 0  =>  p = 0.7)")

# Jensen's inequality: the mean of a log is below the log of a mean
rng = np.random.default_rng(0)
xs = rng.gamma(shape=0.8, scale=2.0, size=100000)
print(f"Jensen: mean(log(x)) = {np.mean(np.log(xs)):.3f}  <  log(mean(x)) = {np.log(np.mean(xs)):.3f}")
