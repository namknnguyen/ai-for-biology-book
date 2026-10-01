"""Chapter 21: Wright-Fisher drift and selection, Fst, coalescent, LD decay, and Jukes-Cantor, each checked against theory."""
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Wright-Fisher drift (N diploids = 2N gene copies): Var(dp) = p(1-p)/(2N); heterozygosity decays as (1 - 1/(2N))^t.
# ---------------------------------------------------------------------------------------------
N, reps = 100, 200_000
p = np.full(reps, 0.5)
p1 = rng.binomial(2 * N, p) / (2 * N)
print(f"drift: one-generation Var(dp) = {np.var(p1 - 0.5):.5f} (theory p(1-p)/2N = {0.25 / (2 * N):.5f})")
p = np.full(reps, 0.5); H0 = 2 * 0.25
for t in range(1, 51):
    p = rng.binomial(2 * N, p) / (2 * N)
H50 = np.mean(2 * p * (1 - p))
print(f"heterozygosity after 50 generations: simulated {H50:.4f}, theory H0 (1-1/2N)^t = {H0 * (1 - 1 / (2 * N)) ** 50:.4f}")

# neutral fixation probability and conditional time to fixation
p0 = 0.1; p = np.full(reps, p0); t = np.zeros(reps); alive = np.ones(reps, bool); fixed = np.zeros(reps, bool)
for g in range(1, 4000):
    p[alive] = rng.binomial(2 * N, p[alive]) / (2 * N)
    done = alive & ((p == 0) | (p == 1)); fixed |= done & (p == 1); t[done] = g; alive &= ~done
    if not alive.any(): break
tbar = -4 * N * (1 - p0) / p0 * np.log(1 - p0)            # Kimura-Ohta: expected time to fixation, conditional on fixation
print(f"neutral allele starting at p0 = {p0}: fixation probability = {fixed.mean():.4f} (theory p0 = {p0}); "
      f"mean time to fixation (given fixation) = {t[fixed].mean():.0f} generations (Kimura-Ohta {tbar:.0f})")

# ---------------------------------------------------------------------------------------------
# 2. Fixation of a NEW mutation under selection (haploid WF, N = 100 copies): u = (1 - e^{-2s}) / (1 - e^{-2Ns}).
# ---------------------------------------------------------------------------------------------
def fixation_prob(s, N=100, reps=400_000):
    k = np.ones(reps, int); alive = np.ones(reps, bool)
    for g in range(3000):
        pk = k[alive] * (1 + s) / (k[alive] * (1 + s) + (N - k[alive]))     # selection: fitness-weighted sampling
        k[alive] = rng.binomial(N, pk)
        alive[alive] = (k[alive] > 0) & (k[alive] < N)
        if not alive.any(): break
    return np.mean(k == N)
print("fixation probability of a new mutation, N = 100 (neutral expectation 1/N = 0.0100):")
for s in (0.0, 0.02, -0.02):
    th = 1 / 100 if s == 0 else (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * 100 * s))
    print(f"  s = {s:+.2f} (N s = {100 * s:+.0f}): simulated {fixation_prob(s):.4f}, theory {th:.4f}")

# ---------------------------------------------------------------------------------------------
# 3. Population divergence: F_ST after t generations of drift in two isolated populations of size N.
# ---------------------------------------------------------------------------------------------
N2, t2, L = 500, 50, 40000
p_anc = rng.uniform(0.2, 0.8, L); pa, pb = p_anc.copy(), p_anc.copy()
for _ in range(t2):
    pa = rng.binomial(2 * N2, pa) / (2 * N2); pb = rng.binomial(2 * N2, pb) / (2 * N2)
pbar = (pa + pb) / 2
fst = np.mean((pa - pb) ** 2 / 2) / np.mean(pbar * (1 - pbar))
print(f"\nF_ST after {t2} generations at N = {N2}: simulated {fst:.4f}, theory 1 - exp(-t/2N) = {1 - np.exp(-t2 / (2 * N2)):.4f}")

# ---------------------------------------------------------------------------------------------
# 4. Coalescent: E[T_MRCA] = 2(1 - 1/n), E[tree length] = 2 sum 1/i, E[S] = theta sum 1/i, SFS E[xi_i] = theta / i  (time in units of 2N generations).
# ---------------------------------------------------------------------------------------------
def coalescent(n, theta, reps=20000):
    tmrca, length, S, sfs = [], [], [], np.zeros(n - 1)
    for _ in range(reps):
        lineages = [1] * n; T = 0.0; L_tot = 0.0; mutations = np.zeros(n - 1)
        while len(lineages) > 1:
            k = len(lineages); dt = rng.exponential(1 / (k * (k - 1) / 2)); T += dt; L_tot += k * dt
            for size in lineages:                                    # mutations on each branch during this interval
                mutations[size - 1] += rng.poisson(theta / 2 * dt) if size < n else 0
            i, j = rng.choice(k, 2, replace=False)
            merged = lineages[i] + lineages[j]; lineages = [x for idx, x in enumerate(lineages) if idx not in (i, j)] + [merged]
        tmrca.append(T); length.append(L_tot); S.append(mutations.sum()); sfs += mutations
    return np.mean(tmrca), np.mean(length), np.mean(S), sfs / reps
n, theta = 10, 4.0
tm, ln, S, sfs = coalescent(n, theta)
a_n = sum(1 / i for i in range(1, n))
print(f"\ncoalescent, n = {n}: mean T_MRCA = {tm:.3f} (theory {2 * (1 - 1 / n):.3f}); mean tree length = {ln:.3f} (theory {2 * a_n:.3f}); "
      f"mean segregating sites = {S:.2f} (theory theta*a_n = {theta * a_n:.2f})")
print("  site-frequency spectrum, simulated vs theta/i : " + ", ".join(f"i={i}: {sfs[i - 1]:.2f} vs {theta / i:.2f}" for i in (1, 2, 3, 5, 9)))

# ---------------------------------------------------------------------------------------------
# 5. LD decay with recombination fraction c: D_t = (1 - c)^t D_0 (infinite population, random mating).
# ---------------------------------------------------------------------------------------------
pA, pB, D0, c = 0.4, 0.3, 0.1, 0.01
pAB = pA * pB + D0
for _ in range(50): pAB = (1 - c) * pAB + c * pA * pB
print(f"\nLD decay: D after 50 generations at c = {c}: simulated {pAB - pA * pB:.5f}, theory D0 (1-c)^t = {D0 * (1 - c) ** 50:.5f}")

# ---------------------------------------------------------------------------------------------
# 6. Jukes-Cantor: P_ii(t) = 1/4 + 3/4 exp(-4 mu t / 3); distance correction d = -3/4 ln(1 - 4p/3).
# ---------------------------------------------------------------------------------------------
mu = 1.0; Q = np.full((4, 4), mu / 3) - np.eye(4) * (mu / 3) * 4
tt = 0.5; P = expm(Q * tt)
print(f"Jukes-Cantor: P_ii(0.5) from matrix exponential = {P[0, 0]:.5f}, formula = {0.25 + 0.75 * np.exp(-4 * mu * tt / 3):.5f}")
for tt in (0.1, 0.5, 1.0, 2.0):
    p_diff = 0.75 * (1 - np.exp(-4 * mu * tt / 3)); d = -0.75 * np.log(1 - 4 * p_diff / 3)
    print(f"  true distance mu*t = {tt:.1f}: observed fraction different p = {p_diff:.3f}; JC-corrected distance = {d:.3f}")

# ---------------------------------------------------------------------------------------------
# 7. Phylogenetic non-independence: two traits evolving independently on the SAME tree (Brownian motion) look correlated
#    across species.  Naive regression vs phylogenetic GLS (whitening by the tree covariance).
# ---------------------------------------------------------------------------------------------
from scipy import stats
def random_tree_cov(n):
    """Kingman coalescent tree -> Brownian-motion covariance V_ij = T - tau_ij (shared path length from the root)."""
    groups = [[i] for i in range(n)]; tau = np.zeros((n, n)); T = 0.0
    while len(groups) > 1:
        k = len(groups); T += rng.exponential(1 / (k * (k - 1) / 2)); a, b = rng.choice(k, 2, replace=False)
        for i in groups[a]:
            for j in groups[b]: tau[i, j] = tau[j, i] = T
        merged = groups[a] + groups[b]; groups = [g for idx, g in enumerate(groups) if idx not in (a, b)] + [merged]
    V = T - tau; np.fill_diagonal(V, T)                                         # variance of a tip = root-to-tip length T
    return V
n_sp, trials, alpha = 40, 1500, 0.05
fp_naive = fp_pgls = 0
for _ in range(trials):
    V = random_tree_cov(n_sp); Lc = np.linalg.cholesky(V + 1e-9 * np.eye(n_sp))
    x, y = Lc @ rng.normal(size=n_sp), Lc @ rng.normal(size=n_sp)              # independent traits, shared ancestry
    fp_naive += stats.linregress(x, y).pvalue < alpha
    Xw = np.linalg.solve(Lc, np.column_stack([np.ones(n_sp), x])); yw = np.linalg.solve(Lc, y)     # whitened design (PGLS)
    beta, res, *_ = np.linalg.lstsq(Xw, yw, rcond=None); s2 = res[0] / (n_sp - 2) if len(res) else 1.0
    se = np.sqrt(s2 * np.linalg.inv(Xw.T @ Xw)[1, 1]); fp_pgls += 2 * stats.t.sf(abs(beta[1] / se), n_sp - 2) < alpha
print(f"\nphylogenetic non-independence ({n_sp} species, NO true relationship between traits): false-positive rate at 0.05 = "
      f"{fp_naive / trials:.2f} for naive regression vs {fp_pgls / trials:.2f} for phylogenetic GLS")
