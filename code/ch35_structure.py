"""Chapter 35: how much contact information does a fold need?  Real structure (HIV capsid C-terminal domain, PDB 1A8O, chain A, 66 residues).
Reconstruct C-alpha coordinates from distance restraints: chain connectivity + a chosen set of predicted contacts with a given precision, by distance geometry
(shortest-path bound smoothing, metric-matrix embedding) followed by gradient refinement of a restraint energy.
Metrics: RMSD and TM-score after Kabsch superposition (minimum over the structure and its mirror image), and lDDT-C-alpha (superposition-free).
This is the classical coevolution-to-structure pipeline (Chapter 29), and it exposes the handedness ambiguity of distance-only information (Chapter 16)."""
import os, urllib.request
import numpy as np, torch
from scipy.sparse.csgraph import floyd_warshall
from Bio.PDB import PDBParser
torch.set_num_threads(2); rng = np.random.default_rng(0); torch.manual_seed(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "1A8O.pdb")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/biopython/biopython/master/Tests/PDB/1A8O.pdb", path)
ch = next(PDBParser(QUIET=True).get_structure("x", path)[0].get_chains())
X = np.array([r["CA"].coord for r in ch if r.id[0] == " " and "CA" in r]); L = len(X)
Dm = np.linalg.norm(X[:, None] - X[None], axis=2)
sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None])
true_c = np.argwhere(np.triu((Dm < 8.0) & (sep >= 6), 1)); noncontact = np.argwhere(np.triu((Dm > 12.0) & (sep >= 6), 1))
print(f"{L} residues; {len(true_c)} long-range contacts (C-alpha distance < 8 A, separation >= 6); {len(true_c)/L:.2f} per residue")

def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0); U, S, Vt = np.linalg.svd(P.T @ Q); d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    R = U @ D @ Vt; Pr = P @ R; return np.sqrt(np.mean(np.sum((Pr - Q) ** 2, 1))), np.linalg.norm(Pr - Q, axis=1)
def tm_score(di, L):
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8; return np.mean(1 / (1 + (di / d0) ** 2))
def best_over_mirror(P, Q):
    r1, d1 = kabsch_rmsd(P, Q); r2, d2 = kabsch_rmsd(P * np.array([1, 1, -1]), Q); return (r1, d1, "native") if r1 <= r2 else (r2, d2, "mirror")
def lddt(P, Q, cutoff=15.0):
    dP = np.linalg.norm(P[:, None] - P[None], axis=2); dQ = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    m = (dQ < cutoff) & (sep > 0); diff = np.abs(dP - dQ); s = np.mean([(diff[m] < t).mean() for t in (0.5, 1, 2, 4)]); return s

RG_CAP = 1.2 * 2.2 * L ** 0.38                                                   # generic compactness prior for a globular chain of L residues (not tuned to this protein)
print(f"native radius of gyration {np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()):.1f} A; compactness prior caps it at {RG_CAP:.1f} A")
far_pool = np.argwhere(np.triu((Dm > 14.0) & (sep >= 6), 1))

def dg_init(contacts, use_ss, r):
    """Distance-geometry start: upper bounds from chain geometry and contacts, smoothed by shortest paths, embedded by classical MDS."""
    W = np.full((L, L), np.inf)
    for i in range(L - 1): W[i, i + 1] = W[i + 1, i] = 3.8
    for k in (2, 3, 4):
        for i in range(L - k):
            u = Dm[i, i + k] if use_ss else (7.0 if k == 2 else np.inf); W[i, i + k] = W[i + k, i] = min(W[i, i + k], u)
    for a, b in contacts: W[a, b] = W[b, a] = min(W[a, b], 8.0)
    D = floyd_warshall(np.where(np.isinf(W), 0, W), directed=False)
    D = D * (0.75 + 0.05 * r.standard_normal((L, L))); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    J = np.eye(L) - 1.0 / L; G = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(G); idx = np.argsort(-w)[:3]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-6)) + 0.3 * r.standard_normal((L, 3))

def fold(contacts, use_ss, far=None, restarts=4, steps=500, seed=0):
    r = np.random.default_rng(seed)
    pairs = torch.tensor(contacts, dtype=torch.long) if len(contacts) else torch.zeros((0, 2), dtype=torch.long)
    farp = torch.tensor(far, dtype=torch.long) if far is not None and len(far) else torch.zeros((0, 2), dtype=torch.long)
    i3 = torch.arange(L - 3); i4 = torch.arange(L - 4); i2 = torch.arange(L - 2); Dt = torch.tensor(Dm, dtype=torch.float32)
    nb = torch.tensor(np.argwhere(np.triu(sep >= 3, 1))); best = None
    for _ in range(restarts):
        Z = torch.tensor(dg_init(contacts, use_ss, r), dtype=torch.float32).requires_grad_(True); opt = torch.optim.Adam([Z], 0.1)
        for st in range(steps):
            d = torch.cdist(Z, Z) + torch.eye(L) * 1e3
            e = 20 * ((d[torch.arange(L - 1), torch.arange(1, L)] - 3.8) ** 2).sum()
            if use_ss: e = e + 3 * ((d[i2, i2 + 2] - Dt[i2, i2 + 2]) ** 2).sum() + 3 * ((d[i3, i3 + 3] - Dt[i3, i3 + 3]) ** 2).sum() + 3 * ((d[i4, i4 + 4] - Dt[i4, i4 + 4]) ** 2).sum()
            else: e = e + 5 * (torch.relu(d[i2, i2 + 2] - 7.0) ** 2 + torch.relu(5.2 - d[i2, i2 + 2]) ** 2).sum()
            if len(pairs): e = e + 2 * (torch.relu(d[pairs[:, 0], pairs[:, 1]] - 8.0) ** 2).sum()
            if len(farp): e = e + 2 * (torch.relu(12.0 - d[farp[:, 0], farp[:, 1]]) ** 2).sum()
            e = e + 3 * (torch.relu(4.2 - d[nb[:, 0], nb[:, 1]]) ** 2).sum()
            rg = torch.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()); e = e + 5 * torch.relu(rg - RG_CAP) ** 2
            opt.zero_grad(); e.backward(); opt.step()
        if best is None or e.item() < best[0]: best = (e.item(), Z.detach().numpy().copy())
    return best[1]

def pick(arr, k, r): return arr[r.choice(len(arr), min(k, len(arr)), replace=False)] if k > 0 else np.zeros((0, 2), int)
def trial(n_contacts, precision, use_ss, n_far, seed, full_map=False):
    r = np.random.default_rng(seed)
    if full_map:                                                                    # the complete binary contact map (all pairs within 8 A with separation >= 3), plus every pair beyond 12 A
        ct = np.argwhere(np.triu((Dm < 8.0) & (sep >= 3), 1)); cf = np.zeros((0, 2), int); far = np.argwhere(np.triu((Dm > 12.0) & (sep >= 3), 1))
    else:
        k_true = int(round(n_contacts * precision)); ct = pick(true_c, k_true, r); cf = pick(noncontact, n_contacts - k_true, r); far = pick(far_pool, n_far, r)
    Y = fold(np.vstack([ct, cf]), use_ss, far, seed=seed); rm, di, which = best_over_mirror(Y, X)
    return rm, tm_score(di, L), lddt(Y, X), which, Y
print("\nsecondary-structure   long-range contacts (precision)   far pairs   RMSD (A)   TM-score   lDDT-Calpha   (mean of 3 reconstructions)")
rows = [(False, 0, 1.0, 0), (True, 0, 1.0, 0), (True, 24, 1.0, 0), (True, 47, 1.0, 0), (True, 47, 0.7, 0), (True, 47, 0.5, 0), (True, 47, 0.3, 0), (True, 47, 1.0, 100), ("full", 0, 1.0, 0)]
for use_ss, nc, pr, nf in rows:
    full = use_ss == "full"; res = [trial(nc, pr, True if full else use_ss, nf, 100 + s, full_map=full) for s in range(3)]
    label = "full binary contact map (all pairs)" if full else f"{nc} ({pr:.1f})"
    print(f"{'yes' if use_ss else 'no':>19s}   {label:>32s}   {nf:9d}   {np.mean([x[0] for x in res]):8.2f}   {np.mean([x[1] for x in res]):8.3f}   {np.mean([x[2] for x in res]):8.3f}")
# handedness: distances are identical for a structure and its mirror image; the reconstruction picks one of them at random
res = [trial(0, 1.0, True, 0, 200 + s, full_map=True) for s in range(8)]
native_hand = sum(1 for r_ in res if r_[3] == "native")
print(f"\nhandedness: 8 independent reconstructions from the full distance information: {native_hand} have the native handedness, {8 - native_hand} are mirror images (median RMSD after choosing the better hand {np.median([x[0] for x in res]):.2f} A)")
