"""Chapter 2: low-rank structure, PCA, and what the leading components actually measure."""
import numpy as np

rng = np.random.default_rng(1)

# ---------------------------------------------------------------------------------------------
# 1. Eckart-Young: truncating the SVD gives the best rank-k approximation in Frobenius norm,
#    and the error equals the sum of the squared discarded singular values.
# ---------------------------------------------------------------------------------------------
A = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 40)) + 0.1 * rng.normal(size=(60, 40))  # rank-5 + noise
U, S, Vt = np.linalg.svd(A, full_matrices=False)
for k in (3, 5, 8):
    Ak = (U[:, :k] * S[:k]) @ Vt[:k]
    err = np.linalg.norm(A - Ak, "fro") ** 2
    print(f"rank {k}: ||A-A_k||_F^2 = {err:9.3f}   sum of discarded sigma^2 = {np.sum(S[k:] ** 2):9.3f}")

# ---------------------------------------------------------------------------------------------
# 2. PCA of genotypes recovers population structure (Balding-Nichols model).
#    Two populations diverged from an ancestral population; allele frequency drift is Fst-sized.
# ---------------------------------------------------------------------------------------------
N_per, M, Fst = 100, 5000, 0.05
p_anc = rng.uniform(0.1, 0.9, size=M)
a = p_anc * (1 - Fst) / Fst                                  # Beta parameters of the BN model
b = (1 - p_anc) * (1 - Fst) / Fst
p_pop = np.stack([rng.beta(a, b), rng.beta(a, b)])           # (2, M) population allele frequencies
G = np.vstack([rng.binomial(2, p_pop[k], size=(N_per, M)) for k in (0, 1)]).astype(float)  # (2*N_per, M)
labels = np.repeat([0, 1], N_per)

p_hat = G.mean(0) / 2
keep = (p_hat > 0.05) & (p_hat < 0.95)                       # drop near-monomorphic SNPs
Z = (G[:, keep] - 2 * p_hat[keep]) / np.sqrt(2 * p_hat[keep] * (1 - p_hat[keep]))  # standardised (N, M')
N, Mk = Z.shape
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
evals = S ** 2 / Mk                                           # eigenvalues of the sample relatedness matrix ZZ^T/M
mp_edge = (1 + np.sqrt(N / Mk)) ** 2                          # Marchenko-Pastur upper edge for pure noise
print(f"\nPCA genotypes: N={N}, M={Mk}.  top eigenvalues: {np.round(evals[:4], 2)}, noise edge ~ {mp_edge:.2f}")
pc1 = U[:, 0] * S[0]
print(f"PC1 means: pop0 = {pc1[labels == 0].mean():7.2f}, pop1 = {pc1[labels == 1].mean():7.2f}, "
      f"within-pop sd = {pc1[labels == 0].std():.2f}")

# ---------------------------------------------------------------------------------------------
# 3. The same machinery on expression: the leading PC may be a batch, not biology.
# ---------------------------------------------------------------------------------------------
n_cells, n_genes = 400, 2000
celltype = rng.integers(0, 2, n_cells)
batch = rng.integers(0, 2, n_cells)
base = rng.normal(size=n_genes)
w_cell = rng.normal(size=n_genes) * (rng.random(n_genes) < 0.1)      # 10% of genes differ by cell type
w_batch = rng.normal(size=n_genes) * (rng.random(n_genes) < 0.6)     # 60% of genes shift with batch (technical)
X = base + 1.0 * np.outer(celltype, w_cell) + 1.0 * np.outer(batch, w_batch) + rng.normal(size=(n_cells, n_genes))
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
for j in range(3):
    pc = U[:, j] * S[j]
    print(f"PC{j+1}: |corr with cell type| = {abs(np.corrcoef(pc, celltype)[0, 1]):.2f}, "
          f"|corr with batch| = {abs(np.corrcoef(pc, batch)[0, 1]):.2f}")
