"""Chapter 19: stochastic gene expression (bursting -> negative binomial) and a bistable gene circuit (cell states as attractors)."""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Bursty transcription: bursts arrive at rate k_b (Poisson), each burst makes a geometric number of mRNAs
#    (mean b); mRNA degrades at rate gamma.  Steady state is negative binomial with r = k_b/gamma, mean = r*b,
#    variance = mean * (1 + b)  (Fano factor 1 + b).   Compare with a constitutive (Poisson) gene of equal mean.
# ---------------------------------------------------------------------------------------------
def simulate_bursty(k_b, b, gamma, T=60.0, cells=4000):
    """Exact event-driven simulation of many independent cells, observed at time T (>> 1/gamma)."""
    counts = np.empty(cells, dtype=int)
    for c in range(cells):
        n_bursts = rng.poisson(k_b * T)                           # bursts in [0, T]
        t_burst = rng.uniform(0, T, n_bursts)
        sizes = rng.geometric(1 / (1 + b), n_bursts) - 1         # geometric on {0,1,..} with mean b
        survive = rng.random(sizes.sum()) < np.exp(-gamma * np.repeat(T - t_burst, sizes))   # each mRNA survives to time T
        counts[c] = survive.sum()
    return counts
k_b, b, gamma = 0.2, 8.0, 0.1                                      # 0.2 bursts/min, 8 mRNA/burst, half-life ~7 min
x = simulate_bursty(k_b, b, gamma)
r = k_b / gamma; mu_pred = r * b
print(f"bursting gene: mean = {x.mean():.2f} (theory r*b = {mu_pred:.2f}), variance = {x.var():.2f} (theory mu(1+b) = {mu_pred * (1 + b):.2f}), "
      f"Fano factor = {x.var() / x.mean():.2f} (theory {1 + b:.1f})")
pois = rng.poisson(mu_pred, 4000)
print(f"  Poisson gene with the same mean: Fano factor = {pois.var() / pois.mean():.2f};  fraction of cells with zero mRNA: bursting {np.mean(x == 0):.3f} vs Poisson {np.mean(pois == 0):.3f}")
# method-of-moments NB fit recovers r and mean
mu_hat = x.mean(); r_hat = mu_hat ** 2 / (x.var() - mu_hat)
print(f"  NB moment fit: r = {r_hat:.2f} (theory {r:.2f}), i.e. dispersion 1/r = {1 / r_hat:.2f}")

# ---------------------------------------------------------------------------------------------
# 2. Mutual repression (toggle switch): du/dt = a/(1+v^n) - u, dv/dt = a/(1+u^n) - v.  Bistable for a large enough.
#    Cell types as attractors; noise-driven switching.
# ---------------------------------------------------------------------------------------------
def fixed_points(a, n):
    """Find fixed points by Newton iteration from many starts; classify stability from the Jacobian eigenvalues."""
    F = lambda z: np.array([a / (1 + z[1] ** n) - z[0], a / (1 + z[0] ** n) - z[1]])
    J = lambda z: np.array([[-1, -a * n * z[1] ** (n - 1) / (1 + z[1] ** n) ** 2], [-a * n * z[0] ** (n - 1) / (1 + z[0] ** n) ** 2, -1]])
    pts = []
    for u0 in np.linspace(0.05, a, 25):
        for v0 in np.linspace(0.05, a, 25):
            z = np.array([u0, v0])
            for _ in range(60): z = z - np.linalg.solve(J(z), F(z))
            if np.linalg.norm(F(z)) < 1e-9 and (z > 0).all() and not any(np.linalg.norm(z - q) < 1e-4 for q, _ in pts):
                pts.append((z, np.linalg.eigvals(J(z)).real.max() < 0))
    return pts
for a in (1.5, 3.0):
    fp = sorted(fixed_points(a, 2), key=lambda t: t[0][0])
    print(f"toggle switch, a = {a}: " + "; ".join(f"(u={z[0]:.2f}, v={z[1]:.2f}) {'stable' if st else 'UNSTABLE (saddle)'}" for z, st in fp))

def simulate_switching(a=3.0, n=2, noise=0.35, steps=200_000, dt=0.02):
    u, v = 2.6, 0.4; state, switches = 0, 0                           # start in the u-high state
    for _ in range(steps):
        du = a / (1 + v ** n) - u; dv = a / (1 + u ** n) - v
        u = max(u + dt * du + noise * np.sqrt(dt) * rng.normal(), 0.0); v = max(v + dt * dv + noise * np.sqrt(dt) * rng.normal(), 0.0)
        if state == 0 and v - u > 1.2: state, switches = 1, switches + 1       # count only clear commitments to the other state
        elif state == 1 and u - v > 1.2: state, switches = 0, switches + 1
    return switches
for noise in (0.3, 0.6, 0.9):
    print(f"  noise sd {noise}: number of committed state switches in 4,000 time units = {simulate_switching(noise=noise)}")
