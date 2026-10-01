"""Chapter 5: entropy, KL asymmetry, language models as compressors, motif information, data processing."""
import numpy as np
from itertools import product

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Forward vs reverse KL: fit one Gaussian q to a bimodal p.
# ---------------------------------------------------------------------------------------------
x = np.linspace(-12, 12, 4801); dx = x[1] - x[0]
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
p = 0.5 * npdf(x, -3, 0.7) + 0.5 * npdf(x, 3, 0.7)
def kl(a, b):
    return np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
m_fwd = np.sum(x * p) * dx; s_fwd = np.sqrt(np.sum((x - m_fwd) ** 2 * p) * dx)  # forward KL(p||q): moment matching
best = min(((kl(q := npdf(x, m, s), p), m, s) for m in np.linspace(-4, 4, 81) for s in np.linspace(0.3, 4, 75)))
print(f"bimodal p: forward KL(p||q) fit  -> mean {m_fwd:5.2f}, sd {s_fwd:4.2f}  (covers both modes, puts mass between them)")
print(f"           reverse KL(q||p) fit  -> mean {best[1]:5.2f}, sd {best[2]:4.2f}  (locks onto one mode)")

# ---------------------------------------------------------------------------------------------
# 2. A language model is a compressor: held-out bits per base of order-k Markov models.
# ---------------------------------------------------------------------------------------------
true_order, A = 3, 4
ctxs = list(product(range(A), repeat=true_order))
trans = {c: rng.dirichlet(np.full(A, 0.35)) for c in ctxs}        # peaked transitions -> compressible
def sample(n):
    seq = list(rng.integers(0, A, true_order))
    for _ in range(n - true_order):
        seq.append(rng.choice(A, p=trans[tuple(seq[-true_order:])]))
    return np.array(seq)
train, test = sample(120_000), sample(60_000)

# entropy rate of the source (stationary distribution over contexts, estimated from a long sample)
long = sample(400_000)
idx = lambda s, i, k: int("".join(map(str, s[i - k:i])) or "0", A) if k else 0
H_rate = 0.0
cnt = {}
for i in range(true_order, len(long)):
    c = tuple(long[i - true_order:i]); cnt[c] = cnt.get(c, 0) + 1
tot = sum(cnt.values())
for c, v in cnt.items():
    H_rate += (v / tot) * -(trans[c] * np.log2(trans[c])).sum()
print(f"\nsource entropy rate = {H_rate:.3f} bits/base (uniform coding would need 2.000)")

def order_k_cross_entropy(k, alpha=0.5):
    counts = {}
    for i in range(k, len(train)):
        c = tuple(train[i - k:i]); counts.setdefault(c, np.zeros(A))[train[i]] += 1
    bits, n = 0.0, 0
    for i in range(k, len(test)):
        c = tuple(test[i - k:i])
        cc = counts.get(c, np.zeros(A))
        bits += -np.log2((cc[test[i]] + alpha) / (cc.sum() + alpha * A)); n += 1
    return bits / n
for k in (0, 1, 2, 3, 4, 6, 8):
    print(f"  order-{k} model: held-out cross-entropy = {order_k_cross_entropy(k):.3f} bits/base")

# ---------------------------------------------------------------------------------------------
# 3. Information content of a motif (Schneider): R_seq = sum_positions (2 - H_position) bits.
# ---------------------------------------------------------------------------------------------
pwm = np.array([[0.90, 0.03, 0.04, 0.03],      # strongly conserved A
                [0.05, 0.05, 0.85, 0.05],      # strongly conserved G
                [0.25, 0.25, 0.25, 0.25],      # uninformative
                [0.10, 0.70, 0.10, 0.10],
                [0.60, 0.10, 0.20, 0.10],
                [0.03, 0.03, 0.03, 0.91]])
H = -(pwm * np.log2(pwm)).sum(1)
R = (2 - H).sum()
G = 3.1e9
print(f"\nmotif information per position (bits): {np.round(2 - H, 2)}; total R_seq = {R:.1f} bits")
print(f"  chance occurrences expected in a {G:.1e}-base genome (both strands) ~ {2 * G * 2 ** -R:,.0f}; "
      f"bits needed for a unique site ~ {np.log2(G):.1f}")

# ---------------------------------------------------------------------------------------------
# 4. Data processing inequality on a random discrete Markov chain X -> Y -> Z.
# ---------------------------------------------------------------------------------------------
def mutual_info(pxy):
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    mask = pxy > 0
    return (pxy[mask] * np.log2(pxy[mask] / (px @ py)[mask])).sum()
px = rng.dirichlet(np.ones(6))
Pyx = rng.dirichlet(np.ones(6) * 0.5, size=6)       # row x -> distribution of y
Pzy = rng.dirichlet(np.ones(6) * 0.5, size=6)
pxy = px[:, None] * Pyx
pxz = pxy @ Pzy
print(f"\ndata processing: I(X;Y) = {mutual_info(pxy):.3f} bits >= I(X;Z) = {mutual_info(pxz):.3f} bits")
