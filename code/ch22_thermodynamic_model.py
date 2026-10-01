"""Chapter 22: a thermodynamic model of transcription-factor binding and regulatory logic."""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Binding to a site with energy E (in units of kT above the consensus site): K_d = K_0 exp(E),
#    occupancy = c / (c + K_d)  (a logistic function of log concentration).
# ---------------------------------------------------------------------------------------------
K0 = 1.0
for E in (0, 2.5, 5.0, 10.0):
    print(f"site with {E:4.1f} kT penalty: Kd = {K0 * np.exp(E):9.1f}; occupancy at c = Kd(consensus) = {1 / (1 + np.exp(E)):.2e}")

# ---------------------------------------------------------------------------------------------
# 2. The specificity problem (Chapter 5, motif information): a 10-bp motif (20 bits at most) in a genome of 10^6 positions,
#    additive mismatch energy eps = 2.5 kT per mismatched base.  Where does the bound TF actually sit?
# ---------------------------------------------------------------------------------------------
def specificity(genome_len, motif_len=10, eps=2.5, c=1.0, seed=0):
    r = np.random.default_rng(seed)
    mism = r.binomial(motif_len, 0.75, genome_len)                       # number of mismatches at each random position
    E = mism * eps; occ = c / (c + np.exp(E))                            # occupancy of each position (K0 = 1)
    consensus_occ = c / (c + 1.0)                                        # one planted consensus site
    background = occ.sum()
    return consensus_occ, background, consensus_occ / (consensus_occ + background)
print("\nspecificity problem (10-bp motif, 2.5 kT per mismatch, TF concentration c = Kd of the consensus site):")
for G in (1e4, 1e6, 1e8):
    cons, bg, frac = specificity(int(G))
    print(f"  genome of {G:8.0e} positions: occupancy of the consensus site = {cons:.2f}; summed occupancy of all other positions = {bg:9.2f}; "
          f"fraction of bound TF at the consensus site = {frac:.3f}")
# the analytic expectation: sum_i 1/(1+exp(E_i)) ~ G * E[exp(-E)] for large E, per-position E[exp(-E)] = (1/4 + 3/4 exp(-eps))^L
per_pos = (0.25 + 0.75 * np.exp(-2.5)) ** 10
print(f"  analytic: expected summed background occupancy ~ G * (1/4 + 3/4 e^-eps)^L = {1e6 * per_pos:.1f} for G = 10^6")

# ---------------------------------------------------------------------------------------------
# 3. Cooperative binding at a promoter with two sites (statistical-weights model).
#    States: empty (weight 1), A bound (a), B bound (b), both bound (omega * a * b), with a = [A]/K_A, b = [B]/K_B.
#    AND logic: active only when both are bound; OR logic: active when at least one is bound.
# ---------------------------------------------------------------------------------------------
def p_active(a, b, omega, logic):
    Z = 1 + a + b + omega * a * b
    return (omega * a * b / Z) if logic == "AND" else ((a + b + omega * a * b) / Z)
def hill_coefficient(omega, logic):
    """Same TF at two sites (a = b = x = [TF]/K): the response to the TF concentration x; apparent Hill coefficient from the 10-90% rise."""
    x = np.logspace(-4, 4, 8001); p = p_active(x, x, omega, logic)
    lo, hi = np.interp([0.1 * p.max(), 0.9 * p.max()], p, x)
    return np.log(81) / np.log(hi / lo)                                 # n_H = ln(81) / ln(x_90 / x_10)
print("\nsame TF at two promoter sites, AND logic (a = b = x): apparent Hill coefficient of expression vs TF concentration:")
for omega in (1, 10, 100, 1000, 1e5):
    print(f"  cooperativity omega = {omega:7.0f}: apparent Hill coefficient n_H = {hill_coefficient(omega, 'AND'):.2f}")
a_grid = np.array([0.1, 1, 10]); b_grid = np.array([0.1, 1, 10])
print("expression P(active) under AND logic with omega = 100  (rows: [A]/K_A = 0.1, 1, 10; columns: [B]/K_B = 0.1, 1, 10):")
for a in a_grid: print("   ", "  ".join(f"{p_active(a, b, 100, 'AND'):.3f}" for b in b_grid))
