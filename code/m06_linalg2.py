"""Chapter M6: rank and subspaces, projection and least squares, eigenvalues (Jukes-Cantor, Markov chains), SVD and PCA."""
import math

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Rank, independence, and the rank-nullity theorem.
# ---------------------------------------------------------------------------------------------
B = rng.normal(size=(6, 3)); A = B @ rng.normal(size=(3, 8))            # 6 x 8 matrix built from only 3 independent directions
r = np.linalg.matrix_rank(A)
print(f"A is 6 x 8 but built as (6x3)(3x8): rank {r}; column space dim {r}, row space dim {r}, null space dim {8 - r} (= 8 - rank), left null space dim {6 - r} (= 6 - rank)")
u = rng.normal(size=4); cols = np.column_stack([u, 2 * u, rng.normal(size=4)])
print("three vectors in R^4 where the second is twice the first: rank =", np.linalg.matrix_rank(cols), "-> linearly dependent")

# ---------------------------------------------------------------------------------------------
# 2. Projection onto a column space = least squares. The residual is orthogonal to every column.
# ---------------------------------------------------------------------------------------------
X = rng.normal(size=(30, 3)); y = X @ np.array([1.0, -1.0, 2.0]) + 0.5 * rng.normal(size=30)
w = np.linalg.solve(X.T @ X, X.T @ y)
P = X @ np.linalg.solve(X.T @ X, X.T)                                  # projection ("hat") matrix
resid = y - X @ w
print(f"\nleast squares: w = {np.round(w, 3)}; X^T residual = {np.round(X.T @ resid, 10)} (orthogonal);  P is idempotent (P P = P): {np.allclose(P @ P, P)};  trace(P) = {np.trace(P):.3f} = number of columns")
print(f"Pythagoras: |y|^2 = {y @ y:.3f} = |Py|^2 + |y - Py|^2 = {np.sum((P @ y) ** 2) + np.sum(resid ** 2):.3f}")
Q, R = np.linalg.qr(X)
print("QR: Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(3)), "; same fitted values from Q Q^T y:", np.allclose(Q @ (Q.T @ y), X @ w))

# ---------------------------------------------------------------------------------------------
# 3. Eigenvalues and eigenvectors: A v = lambda v. Diagonalization makes powers easy.
# ---------------------------------------------------------------------------------------------
A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
lam, V = np.linalg.eigh(A2)
print(f"\nA = [[2,1],[1,2]]: eigenvalues {lam} (trace {np.trace(A2):.0f} = sum, det {np.linalg.det(A2):.0f} = product), eigenvectors (columns)\n{np.round(V, 4)}")
print("A^10 via eigen:", np.round(V @ np.diag(lam ** 10) @ V.T, 1).tolist(), " direct:", np.linalg.matrix_power(A2, 10).tolist())

# Jukes-Cantor DNA substitution model: rate matrix Q (rows sum to 0): each base changes to each of 3 others at rate alpha
alpha = 0.1
Qm = alpha * (np.ones((4, 4)) - 4 * np.eye(4))
ev = np.sort(np.linalg.eigvalsh(Qm))[::-1]
print(f"\nJukes-Cantor rate matrix eigenvalues: {np.round(ev, 4)}  (0 and -4*alpha = {-4 * alpha})")
for t in (1.0, 5.0, 20.0):
    Pt = expm(Qm * t)
    same = Pt[0, 0]
    formula = 0.25 + 0.75 * math.exp(-4 * alpha * t)
    print(f"  t = {t:5.1f}: P(same base) = {same:.5f}  (formula 1/4 + 3/4 e^(-4 alpha t) = {formula:.5f});  rows sum to {Pt.sum(1).round(6)[0]}")
print("  as t grows every row of P(t) tends to the stationary distribution (1/4,1/4,1/4,1/4):", np.round(expm(Qm * 200)[0], 4))

# ---------------------------------------------------------------------------------------------
# 4. A Markov chain: transition matrix T (rows sum to 1). Stationary distribution = eigenvector with eigenvalue 1;
#    the speed of convergence is set by the second-largest eigenvalue.
# ---------------------------------------------------------------------------------------------
T = np.array([[0.90, 0.08, 0.02],       # states: healthy cell, senescent cell, dead cell
              [0.10, 0.80, 0.10],
              [0.00, 0.00, 1.00]])
T2 = np.array([[0.90, 0.10, 0.00],      # a recurrent chain (no absorbing state): states quiescent, cycling, differentiated
                [0.20, 0.70, 0.10],
                [0.05, 0.15, 0.80]])
vals, vecs = np.linalg.eig(T2.T)
order = np.argsort(-vals.real); vals = vals.real[order]; vecs = vecs.real[:, order]
pi = vecs[:, 0] / vecs[:, 0].sum()
print(f"\nrecurrent Markov chain (states quiescent, cycling, differentiated): eigenvalues {np.round(vals, 4)}; stationary distribution {np.round(pi, 4)} (pi T = pi: {np.allclose(pi @ T2, pi)})")
p0 = np.array([1.0, 0.0, 0.0])
for n in (1, 5, 20, 60):
    pn = p0 @ np.linalg.matrix_power(T2, n)
    print(f"  after {n:>2} steps: {np.round(pn, 4)}  distance to stationary {np.abs(pn - pi).sum():.2e}   (|lambda_2|^n = {abs(vals[1]) ** n:.2e})")
print("absorbing chain T (state 3 never leaves): the long-run distribution from state 1 is", np.round(p0 @ np.linalg.matrix_power(T, 500), 4))

# ---------------------------------------------------------------------------------------------
# 5. Symmetric matrices: the spectral theorem, covariance matrices and positive definiteness.
# ---------------------------------------------------------------------------------------------
Z = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))
C = np.cov(Z, rowvar=False)
lam_c, Vc = np.linalg.eigh(C)
print(f"\ncovariance matrix of 4 variables: eigenvalues {np.round(lam_c, 3)} (all >= 0), eigenvectors orthonormal: {np.allclose(Vc.T @ Vc, np.eye(4))}; reconstruct C = V diag(lam) V^T: {np.allclose(Vc @ np.diag(lam_c) @ Vc.T, C)}")
v = rng.normal(size=4)
print(f"quadratic form v^T C v = {v @ C @ v:.3f} = variance of the projection Z v: {np.var(Z @ v, ddof=1):.3f} (>0, as positive definiteness requires)")

# ---------------------------------------------------------------------------------------------
# 6. SVD and PCA: a cells x genes matrix with 3 cell types has rank ~3 plus noise.
# ---------------------------------------------------------------------------------------------
n_per, G = 100, 60
types = np.repeat([0, 1, 2], n_per)
centers = rng.normal(size=(3, G)) * 1.5
Xc = centers[types] + rng.normal(size=(3 * n_per, G))
Xc = Xc - Xc.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
ev_frac = s ** 2 / np.sum(s ** 2)
print(f"\nSVD of a {Xc.shape[0]}x{Xc.shape[1]} centered matrix (3 cell types + noise): top singular values {np.round(s[:5], 1)}, then {np.round(s[5:8], 1)}...")
print(f"  variance fraction of PC1..PC4: {np.round(ev_frac[:4], 3)} -> two components for three types (centered: types span a 2D subspace)")
for k in (1, 2, 5, 20):
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    print(f"  rank-{k:>2} approximation: relative error {np.linalg.norm(Xc - Xk) / np.linalg.norm(Xc):.3f};   Eckart-Young prediction sqrt(sum of dropped s^2)/|X| = {math.sqrt(np.sum(s[k:] ** 2) / np.sum(s ** 2)):.3f}")
scores = U[:, :2] * s[:2]
means = np.array([scores[types == t].mean(0) for t in range(3)])
within = np.mean([scores[types == t].std(0).mean() for t in range(3)])
print(f"  PC1/PC2 centroids of the three types:\n{np.round(means, 1)}   typical within-type spread {within:.1f} -> clusters are well separated")
