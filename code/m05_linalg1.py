"""Chapter M5: vectors, matrix products, linear systems, Gaussian elimination, geometry of maps, conditioning, a stoichiometric null space."""
import math

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. Vectors: dot product, length, angle. Cosine similarity of two expression profiles.
# ---------------------------------------------------------------------------------------------
a = np.array([3.0, 4.0, 0.0]); b = np.array([4.0, 3.0, 5.0])
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"a.b = {a @ b:.0f}, |a| = {np.linalg.norm(a):.0f}, |b| = {np.linalg.norm(b):.3f}, cosine = {cos:.4f}, angle = {math.degrees(math.acos(cos)):.1f} degrees")
x1 = rng.gamma(2.0, 1.0, size=2000); x2 = 5 * x1 + rng.normal(size=2000)            # same profile, scaled 5x, plus noise
print(f"cosine similarity of a profile and its 5x-scaled noisy copy: {x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2)):.4f} "
      f"(Euclidean distance between them is large: {np.linalg.norm(x1 - x2):.0f})")

# ---------------------------------------------------------------------------------------------
# 2. Matrix product: three equivalent views (entries, columns, sum of outer products).
# ---------------------------------------------------------------------------------------------
A = rng.integers(-3, 4, size=(3, 4)).astype(float); B = rng.integers(-3, 4, size=(4, 2)).astype(float)
C1 = np.array([[A[i] @ B[:, k] for k in range(2)] for i in range(3)])                  # dot products of rows and columns
C2 = np.column_stack([A @ B[:, k] for k in range(2)])                                   # A applied to each column of B
C3 = sum(np.outer(A[:, j], B[j]) for j in range(4))                                     # sum of rank-one outer products
print("\nshapes (3x4)(4x2) -> (3x2); three views agree:", np.allclose(C1, A @ B) and np.allclose(C2, A @ B) and np.allclose(C3, A @ B))
print("AB == BA for 2x2 matrices?", np.allclose(np.array([[1, 2], [3, 4]]) @ np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]) @ np.array([[1, 2], [3, 4]])),
      "(matrix multiplication does not commute)")

# ---------------------------------------------------------------------------------------------
# 3. Gaussian elimination with partial pivoting, written out, and its cost (~ 2/3 n^3 operations).
# ---------------------------------------------------------------------------------------------
def gauss_solve(A, b):
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b); flops = 0
    for k in range(n):
        p = k + np.argmax(np.abs(A[k:, k]))                      # pivot: largest entry in the column
        if p != k:
            A[[k, p]] = A[[p, k]]; b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]; flops += 1
            A[i, k:] -= m * A[k, k:]; b[i] -= m * b[k]; flops += 2 * (n - k) + 2
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                                # back substitution
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]; flops += 2 * (n - i - 1) + 2
    return x, flops

M = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]); rhs = np.array([8.0, -11.0, -3.0])
x, _ = gauss_solve(M, rhs)
print("\nsolve Mx = b:  Gaussian elimination", x, " numpy", np.linalg.solve(M, rhs), " (exact solution is [2, 3, -1])")
for n in (50, 100, 200):
    Mn = rng.normal(size=(n, n)) + n * np.eye(n); bn = rng.normal(size=n)
    xn, fl = gauss_solve(Mn, bn)
    print(f"  n = {n:>3}: operations {fl:>10,} = {fl / (2 / 3 * n ** 3):.2f} x (2/3)n^3;  max |x - numpy| = {np.abs(xn - np.linalg.solve(Mn, bn)).max():.1e}")

# ---------------------------------------------------------------------------------------------
# 4. Matrices as geometry: rotation, scaling, shear act on the unit square; |det| is the area scale factor.
# ---------------------------------------------------------------------------------------------
th = math.radians(30)
Rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
Scale = np.diag([2.0, 1.5]); Shear = np.array([[1.0, 1.0], [0.0, 1.0]]); Sing = np.array([[1.0, 2.0], [2.0, 4.0]])
print("\nmap            det      area of image of unit square   invertible?")
for name, T in (("rotation 30deg", Rot), ("scale (2,1.5)", Scale), ("shear", Shear), ("singular", Sing)):
    print(f"  {name:<14} {np.linalg.det(T):6.3f}   {abs(np.linalg.det(T)):6.3f}                       {abs(np.linalg.det(T)) > 1e-12}")
print("  rotation is orthogonal: R^T R = I:", np.allclose(Rot.T @ Rot, np.eye(2)), "; it preserves lengths:", np.isclose(np.linalg.norm(Rot @ np.array([3.0, 4.0])), 5.0))
print("  singular map sends (2,-1) to", Sing @ np.array([2.0, -1.0]), "-> a whole line collapses to a point, so no inverse exists")

# ---------------------------------------------------------------------------------------------
# 5. A stoichiometric matrix: metabolites A, B, C; reactions R1 uptake->A, R2 A->B, R3 A->C, R4 B->C, R5 C->export.
#    Steady state S v = 0 is a linear system whose solutions form the null space of S.
# ---------------------------------------------------------------------------------------------
S = np.array([[1, -1, -1,  0,  0],      # A
              [0,  1,  0, -1,  0],      # B
              [0,  0,  1,  1, -1]],     # C
             dtype=float)
U, sv, Vt = np.linalg.svd(S)
rank = int((sv > 1e-10).sum())
null = Vt[rank:]                         # rows spanning the null space
print(f"\nstoichiometric matrix S (3 metabolites x 5 reactions): rank {rank}, null-space dimension {5 - rank}")
v = np.array([10.0, 4.0, 6.0, 4.0, 10.0])      # uptake 10, A->B 4, A->C 6, B->C 4, export 10
print("flux vector v = (10, 4, 6, 4, 10):  S v =", S @ v, "-> steady state: every metabolite is produced as fast as it is consumed")
print("every steady-state flux has the form v = v1*(1,0,1,0,1) + v2*(0,1,-1,1,0):",
      np.allclose(S @ (3 * np.array([1, 0, 1, 0, 1]) + 2 * np.array([0, 1, -1, 1, 0])), 0))
print("a flux that violates mass balance: S v =", S @ np.array([10.0, 4.0, 6.0, 4.0, 5.0]), "(metabolite C accumulates at 5 per hour)")

# ---------------------------------------------------------------------------------------------
# 6. Conditioning: solving Hx = b for the Hilbert matrix loses digits as n grows.
# ---------------------------------------------------------------------------------------------
print("\nHilbert matrix H_ij = 1/(i+j-1): condition number and error of solving H x = H x_true (x_true = ones)")
for n in (4, 8, 12):
    H = 1.0 / (np.arange(1, n + 1)[:, None] + np.arange(1, n + 1)[None, :] - 1)
    xs = np.linalg.solve(H, H @ np.ones(n))
    print(f"  n = {n:>2}: cond = {np.linalg.cond(H):.1e}   max error in x = {np.abs(xs - 1).max():.1e}   (rule of thumb: digits lost ~ log10(cond) = {math.log10(np.linalg.cond(H)):.0f})")
