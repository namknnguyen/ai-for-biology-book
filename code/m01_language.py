"""Chapter M1: functions, counting, induction checks, and the birthday problem in single-cell barcodes."""
import itertools
import math
from collections import Counter, defaultdict

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. The genetic code is a function from the 64 codons to 21 outcomes (20 amino acids + stop).
#    A function can be onto without being one-to-one: that is "degeneracy", and it is why a
#    protein sequence does not determine the DNA that encoded it.
# ---------------------------------------------------------------------------------------------
bases = "TCAG"
aa_table = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
code = {a + b + c: aa_table[16 * i + 4 * j + k]
        for i, a in enumerate(bases) for j, b in enumerate(bases) for k, c in enumerate(bases)}
codons, outcomes = set(code), set(code.values())
preimage = defaultdict(list)
for codon, aa in code.items():
    preimage[aa].append(codon)
sizes = sorted(len(v) for v in preimage.values())
print(f"domain size {len(codons)}, codomain (image) size {len(outcomes)}")
print("is the map one-to-one?", len(outcomes) == len(codons), "| is it onto the 21 symbols?", len(outcomes) == 21)
print("preimage sizes (codons per outcome):", sizes)
print("number of DNA sequences that encode the 10-residue peptide MKTAYIAKQR:",
      math.prod(len(preimage[a]) for a in "MKTAYIAKQR"))

# ---------------------------------------------------------------------------------------------
# 2. Induction in one line of code: check, for many n, the identity 1 + 2 + ... + n = n(n+1)/2
#    (a check is not a proof, but it catches a wrong guess before you spend an hour proving it),
#    and the geometric-series identity used for mutation and decay models.
# ---------------------------------------------------------------------------------------------
ok_sum = all(sum(range(1, n + 1)) == n * (n + 1) // 2 for n in range(1, 2001))
r, N = 0.9, 50
ok_geo = abs(sum(r ** k for k in range(N + 1)) - (1 - r ** (N + 1)) / (1 - r)) < 1e-12
print(f"\nchecked n(n+1)/2 for n = 1..2000: {ok_sum};  geometric series identity (r=0.9, N=50): {ok_geo}")
print("the false guess 'sum of first n odd numbers = n^2 + 1' fails at n =",
      next(n for n in range(1, 50) if sum(2 * k - 1 for k in range(1, n + 1)) != n ** 2 + 1))

# ---------------------------------------------------------------------------------------------
# 3. Counting: sequence spaces explode, and the explosion is the reason brute force fails.
# ---------------------------------------------------------------------------------------------
print("\nnumber of DNA sequences of length L  (4^L):")
for L in (10, 20, 30, 100):
    print(f"  L = {L:>3}: 4^L = {4 ** L:.3e}")
print(f"number of proteins of length 100 (20^100): {20 ** 100:.3e}   (atoms in the observable universe ~ 1e80)")
print(f"ways to choose 5 mutated sites among 300 positions: C(300,5) = {math.comb(300, 5):,}")
print(f"ways to order 8 distinct genes on a chromosome arm: 8! = {math.factorial(8):,}")
# Stirling: n! ~ sqrt(2 pi n) (n/e)^n
for n in (5, 20, 100):
    stirling = math.sqrt(2 * math.pi * n) * (n / math.e) ** n
    print(f"  Stirling n={n:>3}: relative error {abs(stirling - math.factorial(n)) / math.factorial(n):.3%}")

# brute-force check that the number of length-L sequences with exactly k G/C bases is C(L,k) 2^L
L, k = 8, 3
brute = sum(1 for s in itertools.product("ACGT", repeat=L) if sum(ch in "GC" for ch in s) == k)
print(f"length-8 sequences with exactly 3 G/C: brute force {brute}, formula C(8,3)*2^8 = {math.comb(8, 3) * 2 ** 8}")

# ---------------------------------------------------------------------------------------------
# 4. The birthday problem as barcode collisions. In droplet single-cell sequencing each cell gets a
#    random barcode; two cells that draw the same barcode are merged into one "cell" (a doublet).
#    P(no collision among n cells with B possible barcodes) = prod_{i<n} (1 - i/B)  ~ exp(-n(n-1)/2B).
# ---------------------------------------------------------------------------------------------
def p_collision(n, B):
    return 1.0 - math.exp(sum(math.log1p(-i / B) for i in range(n)))

print("\nbarcode collisions (birthday problem): P(at least one pair of cells shares a barcode)")
for B, n in ((365, 23), (4 ** 8, 300), (4 ** 12, 5000), (4 ** 16, 10000)):
    approx = 1 - math.exp(-n * (n - 1) / (2 * B))
    print(f"  B = {B:>10,}  n = {n:>6,}: exact {p_collision(n, B):.4f}   exp-approx {approx:.4f}")
B, n, trials = 4 ** 8, 300, 4000
sim = np.mean([len(np.unique(rng.integers(0, B, size=n))) < n for _ in range(trials)])
print(f"  simulation, B = {B:,}, n = {n}: {sim:.3f} (exact {p_collision(n, B):.3f})")
n_half = math.ceil(1.1774 * math.sqrt(4 ** 12))
print(f"  cells needed for a 50% chance of some collision with 12-base barcodes: about {n_half:,} (= 1.18 sqrt(B))")
