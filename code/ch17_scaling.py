"""Chapter 17: scaling-law fitting and extrapolation, compute-optimal allocation, LoRA, KL-regularised alignment."""
import numpy as np
from scipy.optimize import curve_fit, minimize

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Fitting L(N) = E + A N^-alpha: the irreducible term E matters enormously for extrapolation.
#    'True' world: E = 1.2 nats/token (entropy of the data), A = 8, alpha = 0.3.  Six model sizes, 1% noise.
# ---------------------------------------------------------------------------------------------
E0, A0, a0 = 1.2, 8.0, 0.30
N = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8]); L = (E0 + A0 * N ** -a0) * (1 + 0.01 * rng.normal(size=len(N)))
f = lambda n, E, A, a: E + A * n ** -a
popt, _ = curve_fit(f, N, L, p0=[1.0, 5.0, 0.25], maxfev=20000)
slope, icpt = np.polyfit(np.log(N), np.log(L), 1)                    # naive: assume E = 0 (pure power law on the whole loss)
print(f"free-E fit : E = {popt[0]:.2f}, alpha = {popt[2]:.3f} (truth E = {E0}, alpha = {a0});  "
      f"naive E=0 fit: alpha = {-slope:.3f}")
for n_big in (1e9, 1e10, 1e12):
    print(f"  extrapolate to N = {n_big:.0e}: truth {f(n_big, E0, A0, a0):.3f}   free-E fit {f(n_big, *popt):.3f}   "
          f"naive {np.exp(icpt) * n_big ** slope:.3f}")
# bootstrap uncertainty of the free-E extrapolation to N = 1e10
preds = []
for _ in range(300):
    Lb = (E0 + A0 * N ** -a0) * (1 + 0.01 * rng.normal(size=len(N)))
    try: pb, _ = curve_fit(f, N, Lb, p0=[1.0, 5.0, 0.25], maxfev=20000); preds.append(f(1e10, *pb))
    except RuntimeError: pass
print(f"  bootstrap 90% interval of the free-E prediction at N = 1e10: [{np.percentile(preds, 5):.3f}, {np.percentile(preds, 95):.3f}] (truth {f(1e10, E0, A0, a0):.3f})")

# ---------------------------------------------------------------------------------------------
# 2. Compute-optimal allocation for L(N, D) = E + A N^-alpha + B D^-beta with C = 6 N D.
#    Illustrative constants of the form fitted for text LMs (Hoffmann et al. 2022): E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28.
# ---------------------------------------------------------------------------------------------
E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
def optimum(C):
    best = minimize(lambda lg: E + A * np.exp(-al * lg[0]) + B * np.exp(-be * (np.log(C / 6) - lg[0])), x0=[np.log(1e9)], method="Nelder-Mead").x[0]
    Nopt = np.exp(best); return Nopt, C / (6 * Nopt)
G = (al * A / (be * B)) ** (1 / (al + be))
print("\ncompute-optimal allocation (numerical minimisation vs closed form N = G (C/6)^(beta/(alpha+beta))):")
for C in (1e21, 1e22, 1e23):
    Nn, Dn = optimum(C); Nc = G * (C / 6) ** (be / (al + be))
    print(f"  C = {C:.0e}: N_opt = {Nn:.2e} (closed form {Nc:.2e}), D_opt = {Dn:.2e}, tokens per parameter = {Dn / Nn:5.1f}")
print(f"  exponents: N ~ C^{be / (al + be):.2f}, D ~ C^{al / (al + be):.2f}")

# ---------------------------------------------------------------------------------------------
# 3. LoRA: W' = W + (alpha/r) B A with B initialised to zero: identical function at initialisation; update has rank <= r.
# ---------------------------------------------------------------------------------------------
d_out, d_in, r = 512, 512, 8
W = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in); Amat = rng.normal(size=(r, d_in)) / np.sqrt(d_in); Bmat = np.zeros((d_out, r))
x = rng.normal(size=d_in); same = np.allclose(W @ x, (W + (16 / r) * Bmat @ Amat) @ x)
Bmat = rng.normal(size=(d_out, r)); dW = (16 / r) * Bmat @ Amat
print(f"\nLoRA r={r}: function unchanged at init (B=0): {same}; rank of the update = {np.linalg.matrix_rank(dW)}; "
      f"trainable parameters {r * (d_out + d_in):,} vs full {d_out * d_in:,} ({100 * r * (d_out + d_in) / (d_out * d_in):.1f}%)")

# ---------------------------------------------------------------------------------------------
# 4. KL-regularised reward maximisation: pi* = pi0 exp(r/beta) / Z  (the Gibbs variational principle behind RLHF).
# ---------------------------------------------------------------------------------------------
K = 6; pi0 = rng.dirichlet(np.ones(K)); r_vec = rng.normal(size=K)
for beta in (0.2, 1.0, 5.0):
    objective = lambda logits: -(softmax := np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()) @ r_vec + beta * np.sum(softmax * np.log(softmax / pi0))
    sol = minimize(objective, np.zeros(K), method="BFGS").x; pi_num = np.exp(sol - sol.max()); pi_num /= pi_num.sum()
    pi_star = pi0 * np.exp(r_vec / beta); pi_star /= pi_star.sum()
    print(f"beta = {beta:3.1f}: max |numerical optimum - pi0 exp(r/beta)/Z| = {np.abs(pi_num - pi_star).max():.1e}; "
          f"expected reward = {pi_star @ r_vec:.3f} (prior {pi0 @ r_vec:.3f}); KL from prior = {np.sum(pi_star * np.log(pi_star / pi0)):.3f}")
