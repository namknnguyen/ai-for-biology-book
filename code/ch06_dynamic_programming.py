"""Chapter 6: dynamic programming for sequences (alignment, HMM forward/Viterbi), verified against brute force."""
import itertools
import numpy as np

# ---------------------------------------------------------------------------------------------
# 1. Needleman-Wunsch global alignment with a linear gap penalty. O(L1 * L2) time and memory.
# ---------------------------------------------------------------------------------------------
def needleman_wunsch(a, b, match=2, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1), dtype=int)            # S[i, j] = best score aligning a[:i] with b[:j]
    S[:, 0] = gap * np.arange(n + 1)
    S[0, :] = gap * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            S[i, j] = max(diag, S[i - 1, j] + gap, S[i, j - 1] + gap)
    # traceback
    i, j, top, bot = n, m, [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and S[i, j] == S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            top.append(a[i - 1]); bot.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and S[i, j] == S[i - 1, j] + gap:
            top.append(a[i - 1]); bot.append("-"); i -= 1
        else:
            top.append("-"); bot.append(b[j - 1]); j -= 1
    return S[n, m], "".join(reversed(top)), "".join(reversed(bot))

score, top, bot = needleman_wunsch("GATTACA", "GCATGCU".replace("U", "A"))
print("NW alignment, score", score); print(" ", top); print(" ", bot)

def brute_force_alignment_score(a, b, match=2, mismatch=-1, gap=-2):
    """Enumerate every alignment path (feasible only for tiny inputs) to verify the DP."""
    best = -10 ** 9
    def rec(i, j, s):
        nonlocal best
        if i == len(a) and j == len(b):
            best = max(best, s); return
        if i < len(a) and j < len(b):
            rec(i + 1, j + 1, s + (match if a[i] == b[j] else mismatch))
        if i < len(a):
            rec(i + 1, j, s + gap)
        if j < len(b):
            rec(i, j + 1, s + gap)
    rec(0, 0, 0)
    return best

rng = np.random.default_rng(0)
ok = all(needleman_wunsch(x, y)[0] == brute_force_alignment_score(x, y)
         for x, y in (("".join(rng.choice(list("ACGT"), rng.integers(3, 7))),
                       "".join(rng.choice(list("ACGT"), rng.integers(3, 7)))) for _ in range(30)))
print("DP matches brute force on 30 random pairs:", ok)

# ---------------------------------------------------------------------------------------------
# 2. A two-state HMM (CpG-island-like): forward algorithm in log space, Viterbi, brute-force check.
# ---------------------------------------------------------------------------------------------
states = ["background", "island"]
log_pi = np.log([0.9, 0.1])
log_T = np.log([[0.95, 0.05],
                [0.10, 0.90]])
# emission probabilities over A,C,G,T: islands are GC-rich
log_E = np.log([[0.30, 0.20, 0.20, 0.30],
                [0.15, 0.35, 0.35, 0.15]])
code = {c: i for i, c in enumerate("ACGT")}

def logsumexp(v, axis=None):
    m = np.max(v, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True))).squeeze()

def forward_loglik(obs):
    alpha = log_pi + log_E[:, obs[0]]
    for o in obs[1:]:
        alpha = logsumexp(alpha[:, None] + log_T, axis=0) + log_E[:, o]   # sum over previous state
    return logsumexp(alpha)

def viterbi(obs):
    delta = log_pi + log_E[:, obs[0]]
    back = []
    for o in obs[1:]:
        cand = delta[:, None] + log_T            # (prev, next)
        back.append(cand.argmax(0))
        delta = cand.max(0) + log_E[:, o]
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return list(reversed(path)), delta.max()

seq = "ATATTAATGCGCGGCGCCGCGATATTAATA"
obs = [code[c] for c in seq]
short = obs[:14]                                          # brute force over 2^14 paths is feasible; 2^30 is not
brute = logsumexp(np.array([
    log_pi[p[0]] + log_E[p[0], short[0]] + sum(log_T[p[t - 1], p[t]] + log_E[p[t], short[t]] for t in range(1, len(short)))
    for p in itertools.product(range(2), repeat=len(short))]))
print(f"\nforward log-likelihood (first 14 bases) = {forward_loglik(short):.6f}, brute force over 2^14 paths = {brute:.6f}")
print(f"forward log-likelihood (all {len(obs)} bases) = {forward_loglik(obs):.4f}  [O(L K^2) instead of K^L paths]")
path, _ = viterbi(obs)
print("Viterbi path (I = island):")
print(" ", seq)
print(" ", "".join("I" if s else "." for s in path))
