"""Chapter M10: NumPy arrays and broadcasting, vectorization, floating point, stable variance, random seeds, autograd preview."""
import math
import time

import numpy as np


# ---------------------------------------------------------------------------------------------
# 0. Plain Python for sequences: parse a FASTA record, GC content, reverse complement, translation, k-mer counts.
# ---------------------------------------------------------------------------------------------
from collections import Counter
fasta = """>toy_gene some description
ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG
"""
def read_fasta(text):
    records, name, chunks = {}, None, []
    for line in text.strip().splitlines():
        if line.startswith(">"):
            if name: records[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if name: records[name] = "".join(chunks)
    return records
seq = read_fasta(fasta)["toy_gene"]
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
revcomp = "".join(comp[b] for b in reversed(seq))
gc = sum(b in "GC" for b in seq) / len(seq)
bases = "TCAG"; aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
table = {a + b + c: aa[16 * i + 4 * j + k] for i, a in enumerate(bases) for j, b in enumerate(bases) for k, c in enumerate(bases)}
protein = "".join(table[seq[i:i + 3]] for i in range(0, len(seq) - 2, 3))
print(f"sequence {seq} (length {len(seq)}), GC content {gc:.3f}\nreverse complement {revcomp}\ntranslation (stop = *): {protein}")
print("3-mer counts, top 3:", Counter(seq[i:i + 3] for i in range(len(seq) - 2)).most_common(3), "\n")
# ---------------------------------------------------------------------------------------------
# 1. Arrays, shapes, broadcasting. A cells x genes matrix, per-cell scaling, per-gene centering.
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.poisson(2.0, size=(6, 4)).astype(float)                         # 6 cells x 4 genes of counts
depth = X.sum(1, keepdims=True)                                         # shape (6, 1): total counts per cell
Xn = X / depth * 1e4                                                    # (6,4) / (6,1) broadcasts the column across genes
print("X shape", X.shape, "| depth shape", depth.shape, "| normalized shape", Xn.shape, "| every cell now sums to", np.round(Xn.sum(1), 1))
gene_mean = Xn.mean(0, keepdims=True)                                   # shape (1, 4)
Xc = Xn - gene_mean                                                     # (6,4) - (1,4)
print("centered columns have mean", np.round(Xc.mean(0), 12), "| shape rule: align shapes from the right; each pair of sizes must be equal or 1")
a = np.arange(3).reshape(3, 1); b = np.arange(4).reshape(1, 4)
print("(3,1) + (1,4) ->", (a + b).shape, "; an accidental (3,) + (3,1) ->", (np.arange(3) + a).shape, "(a silent shape bug: 3 x 3 instead of 3)")
print("einsum 'bld,bkd->blk' style contraction:", np.einsum("bld,bkd->blk", rng.normal(size=(2, 5, 7)), rng.normal(size=(2, 4, 7))).shape)

# ---------------------------------------------------------------------------------------------
# 2. Vectorization: the same computation as a Python loop and as one array operation.
# ---------------------------------------------------------------------------------------------
n = 1_000_000
x = rng.normal(size=n); xl = x.tolist()                                  # the same numbers as a Python list of floats
def best(f, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
t_loop = best(lambda: sum(v * v for v in xl))
t_vec = best(lambda: float(np.sum(x * x)))
print(f"\nsum of squares of 1e6 numbers (best of 5): Python loop {t_loop * 1000:.0f} ms, NumPy {t_vec * 1000:.1f} ms -> {t_loop / t_vec:.0f}x faster (hardware dependent)")
A = rng.normal(size=(300, 300)); B = rng.normal(size=(300, 300))
t0 = time.perf_counter()
C = np.zeros((300, 300))
for i in range(60):                                                      # 60 of 300 rows with triple Python loops, scaled up
    for j in range(300):
        C[i, j] = sum(A[i, k] * B[k, j] for k in range(300))
t_py = (time.perf_counter() - t0) * 5
t0 = time.perf_counter(); C2 = A @ B; t_mm = time.perf_counter() - t0
print(f"300x300 matrix product: Python loops (est.) {t_py:.1f} s, NumPy matmul {t_mm * 1000:.1f} ms; same answer on the computed rows: {np.allclose(C[:60], C2[:60])}")

# ---------------------------------------------------------------------------------------------
# 3. Floating point: finite precision, rounding, cancellation, overflow, underflow.
# ---------------------------------------------------------------------------------------------
print("\n0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3, f"(0.1 + 0.2 = {0.1 + 0.2:.17f});  np.isclose:", np.isclose(0.1 + 0.2, 0.3))
print(f"machine epsilon (double) = {np.finfo(np.float64).eps:.3e}; float32 = {np.finfo(np.float32).eps:.3e}; float16 = {np.finfo(np.float16).eps:.3e}")
print("1 + 1e-16 == 1 ?", 1 + 1e-16 == 1, " | adding a small number to a large one: 1e16 + 1 - 1e16 =", 1e16 + 1 - 1e16)
with np.errstate(over="ignore"):
    big = np.exp(np.float64(710))
print("largest float64 ~", f"{np.finfo(np.float64).max:.2e}", "| exp(710) =", big, "(overflow) | exp(-800) =", np.exp(-800.0), "(underflow to 0)")
x16 = np.float16(2049.0); print("float16 cannot represent 2049: stored as", float(x16), "(integers above 2048 are rounded to even)")
# catastrophic cancellation: (1 - cos x) / x^2 for tiny x, and the algebraically equivalent stable form 2 sin^2(x/2) / x^2
for xv in (1e-2, 1e-5, 1e-8):
    naive = (1 - math.cos(xv)) / xv ** 2
    stable = 2 * math.sin(xv / 2) ** 2 / xv ** 2
    print(f"  x = {xv:.0e}: naive (1 - cos x)/x^2 = {naive:.10f}   stable form = {stable:.10f}   (true limit 0.5)")

# ---------------------------------------------------------------------------------------------
# 4. A numerically fragile statistic: the variance of data with a large offset, naive vs two-pass vs Welford.
# ---------------------------------------------------------------------------------------------
y = 1e9 + rng.normal(0, 1.0, size=100000)                                # true variance 1, huge mean
naive = np.mean(y ** 2) - np.mean(y) ** 2
two_pass = np.mean((y - y.mean()) ** 2)
mean_w = 0.0; M2 = 0.0
for i, v in enumerate(y, 1):                                              # Welford's streaming algorithm
    d = v - mean_w; mean_w += d / i; M2 += d * (v - mean_w)
print(f"\nvariance of data with mean 1e9 and true variance 1.0:  naive E[y^2] - E[y]^2 = {naive:.4f};  two-pass = {two_pass:.4f};  Welford = {M2 / len(y):.4f}")
print("with float32 inputs the naive formula gives", f"{np.mean(y.astype(np.float32) ** 2) - np.mean(y.astype(np.float32)) ** 2:.1f}", "(garbage, possibly negative)")

# ---------------------------------------------------------------------------------------------
# 5. Random numbers: seeds, reproducibility, and how much a result depends on chance.
# ---------------------------------------------------------------------------------------------
def experiment(seed):
    r = np.random.default_rng(seed)
    a = r.normal(0.3, 1, size=20); b = r.normal(0.0, 1, size=20)
    return a.mean() - b.mean()
print("\nsame seed, same result:", experiment(7) == experiment(7), "| different seeds:", np.round([experiment(s) for s in range(6)], 2).tolist())
vals = np.array([experiment(s) for s in range(2000)])
print(f"over 2000 seeds the 'effect' (true 0.3) has mean {vals.mean():.2f} and standard deviation {vals.std():.2f}; fraction of runs with the wrong sign: {np.mean(vals < 0):.2f}")

# ---------------------------------------------------------------------------------------------
# 6. Automatic differentiation in PyTorch: gradients of the logistic loss from M4, with no hand derivation.
# ---------------------------------------------------------------------------------------------
import torch
w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
xv = torch.tensor([1.0, 2.0, -0.5]); yv = torch.tensor(1.0)
loss = -torch.log(torch.sigmoid(w @ xv)) if yv == 1 else -torch.log(1 - torch.sigmoid(w @ xv))
loss.backward()
p = 1 / (1 + math.exp(-float(w.detach() @ xv)))
print(f"\nautograd gradient {np.round(w.grad.numpy(), 5).tolist()} | hand formula -(1 - sigmoid(w.x)) x = {np.round(-(1 - p) * xv.numpy(), 5).tolist()} | tensor shapes: w {tuple(w.shape)}, loss {tuple(loss.shape)}")

# ---------------------------------------------------------------------------------------------
# 7. Always look at the data: Anscombe's quartet, four datasets with the same summary statistics.
# ---------------------------------------------------------------------------------------------
x123 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
ys = [np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
      np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
      np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
      np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])]
print("\nAnscombe's quartet: mean x, mean y, var y, correlation, fitted slope and intercept")
for k, (xx, yy) in enumerate(zip([x123, x123, x123, x4], ys), 1):
    slope, icpt = np.polyfit(xx, yy, 1)
    print(f"  dataset {k}: {xx.mean():.2f}  {yy.mean():.2f}  {yy.var(ddof=1):.2f}  r = {np.corrcoef(xx, yy)[0, 1]:.3f}  y = {slope:.2f} x + {icpt:.2f}")
print("  (a straight line, a curve, an outlier, and a single leverage point: identical numbers, completely different data)")
