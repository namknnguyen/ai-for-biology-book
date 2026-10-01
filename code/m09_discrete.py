"""Chapter M9: graphs, trees, dynamic programming, de Bruijn assembly, k-mer hashing and counting complexity."""
import math
import itertools
from collections import Counter, defaultdict, deque
from functools import lru_cache

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. A small protein-interaction network: adjacency matrix, degrees, BFS, components, walks, triangles, Laplacian.
# ---------------------------------------------------------------------------------------------
names = list("ABCDEFGHIJKL")
edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("D", "F"),   # component 1: A-F
         ("G", "H"), ("H", "I"), ("G", "I"), ("I", "J"),                                         # component 2: G-J
         ("K", "L")]                                                                             # component 3: K-L
idx = {n: i for i, n in enumerate(names)}
Adj = np.zeros((12, 12), dtype=int)
for u, v in edges:
    Adj[idx[u], idx[v]] = Adj[idx[v], idx[u]] = 1
deg = Adj.sum(1)
print("degrees:", dict(zip(names, deg.tolist())), "| sum of degrees =", deg.sum(), "= 2 x number of edges =", 2 * len(edges))

def bfs(src):
    dist = {src: 0}; q = deque([src])
    while q:
        u = q.popleft()
        for v in names:
            if Adj[idx[u], idx[v]] and v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist

d = bfs("A")
print("BFS from A (shortest path lengths):", d, " -> nodes unreachable from A:", sorted(set(names) - set(d)))
seen, comps = set(), []
for n in names:
    if n not in seen:
        c = bfs(n); comps.append(sorted(c)); seen |= set(c)
print("connected components:", comps)
A2 = Adj @ Adj; A3 = A2 @ Adj
print(f"walks of length 2 from A to D: {A2[idx['A'], idx['D']]} (via C);  triangles = trace(A^3)/6 = {np.trace(A3) // 6}  (ABC, DEF, GHI)")
L = np.diag(deg) - Adj
ev = np.linalg.eigvalsh(L)
print(f"graph Laplacian eigenvalues (smallest 5): {np.round(ev[:5], 3)} -> {int(np.sum(np.abs(ev) < 1e-9))} zero eigenvalues = number of connected components")

# ---------------------------------------------------------------------------------------------
# 2. Trees: how many phylogenies? An unrooted binary tree on n leaves has 2n-3 edges; there are (2n-5)!! of them.
# ---------------------------------------------------------------------------------------------
def n_unrooted(n):
    return math.prod(range(1, 2 * n - 4, 2)) if n >= 3 else 1
print("\nnumber of distinct unrooted binary trees with n labelled leaves, (2n-5)!!:")
for n in (4, 5, 8, 10, 20, 50):
    print(f"  n = {n:>2}: {n_unrooted(n):.3e}   (edges in each tree: {2 * n - 3})")

# UPGMA on a small distance matrix (a rooted tree, assumes a molecular clock)
labels = ["human", "chimp", "gorilla", "orang", "macaque"]
D = np.array([[0, 2, 6, 14, 20],
              [2, 0, 6, 14, 20],
              [6, 6, 0, 14, 20],
              [14, 14, 14, 0, 20],
              [20, 20, 20, 20, 0]], dtype=float)
clusters = {i: ([labels[i]], 0.0, labels[i]) for i in range(5)}
dist = {(i, j): D[i, j] for i in range(5) for j in range(i + 1, 5)}
nxt = 5
while len(clusters) > 1:
    (i, j), dmin = min(dist.items(), key=lambda kv: kv[1])
    members = clusters[i][0] + clusters[j][0]
    height = dmin / 2
    newick = f"({clusters[i][2]}:{height - clusters[i][1]:.0f},{clusters[j][2]}:{height - clusters[j][1]:.0f})"
    new = {c: dist[tuple(sorted((c, i)))] * len(clusters[i][0]) + dist[tuple(sorted((c, j)))] * len(clusters[j][0]) for c in clusters if c not in (i, j)}
    new = {c: v / (len(clusters[i][0]) + len(clusters[j][0])) for c, v in new.items()}
    del clusters[i], clusters[j]
    dist = {k: v for k, v in dist.items() if i not in k and j not in k}
    for c, v in new.items():
        dist[tuple(sorted((c, nxt)))] = v
    clusters[nxt] = (members, height, newick); nxt += 1
print("UPGMA tree (Newick, branch lengths in distance units):", list(clusters.values())[0][2] + ";")

# ---------------------------------------------------------------------------------------------
# 3. Dynamic programming: edit distance. Naive recursion repeats work exponentially; the table is O(nm).
# ---------------------------------------------------------------------------------------------
calls = 0
def edit_naive(a, b):
    global calls
    calls += 1
    if not a: return len(b)
    if not b: return len(a)
    return min(edit_naive(a[1:], b) + 1, edit_naive(a, b[1:]) + 1, edit_naive(a[1:], b[1:]) + (a[0] != b[0]))

def edit_dp(a, b):
    n, m = len(a), len(b)
    T = np.zeros((n + 1, m + 1), dtype=int)
    T[:, 0] = np.arange(n + 1); T[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            T[i, j] = min(T[i - 1, j] + 1, T[i, j - 1] + 1, T[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return T[n, m]

s1, s2 = "GATTACA", "GCATGCU"
print(f"\nedit distance({s1}, {s2}) = {edit_dp(s1, s2)}")
print("naive recursion vs dynamic programming (equal-length random DNA strings):")
for L_ in (4, 6, 8, 9):
    a = "".join(rng.choice(list("ACGT"), L_)); b = "".join(rng.choice(list("ACGT"), L_))
    calls = 0; r1 = edit_naive(a, b)
    print(f"  length {L_}: distance {r1} (DP agrees: {edit_dp(a, b) == r1});  recursive calls {calls:>9,}   table cells {(L_ + 1) ** 2}")

# ---------------------------------------------------------------------------------------------
# 4. de Bruijn graph assembly: nodes are (k-1)-mers, each k-mer is an edge; reads -> Eulerian path -> genome.
#    Repeats longer than k - 1 make the reconstruction ambiguous.
# ---------------------------------------------------------------------------------------------
genome = "ATGCGATGCCGTAATGACG"
def debruijn(seq, k):
    g = defaultdict(list)
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]; g[km[:-1]].append(km[1:])
    return g

def eulerian_paths(g, start, total_edges, limit=100000):
    paths = set(); count = 0
    def walk(node, used, path):
        nonlocal count
        if count >= limit: return
        if len(path) == total_edges + 1:
            paths.add("".join([path[0]] + [p[-1] for p in path[1:]])); count += 1; return
        for j, nb in enumerate(g.get(node, [])):
            if (node, j) not in used:
                walk(nb, used | {(node, j)}, path + [nb])
    walk(start, frozenset(), [start])
    return paths

print(f"\ngenome of length {len(genome)}: {genome}")
for k in (3, 4, 5, 6, 8):
    g = debruijn(genome, k)
    total = len(genome) - k + 1
    sols = eulerian_paths(g, genome[:k - 1], total)
    print(f"  k = {k}: {len(g):>2} nodes, {total:>2} edges (k-mers), distinct reconstructions with the right start: {len(sols):>3}; original among them: {genome in sols}")

# ---------------------------------------------------------------------------------------------
# 5. Hashing and k-mer counting: the number of distinct k-mers observed, and the cost of finding duplicates.
# ---------------------------------------------------------------------------------------------
Lg, k = 100000, 8
g = "".join(rng.choice(list("ACGT"), Lg))
counts = Counter(g[i:i + k] for i in range(Lg - k + 1))
B = 4 ** k; n = Lg - k + 1
print(f"\nrandom genome of length {Lg:,}: {n:,} {k}-mers occurrences, {len(counts):,} distinct (of {B:,} possible); expected distinct B(1 - e^(-n/B)) = {B * (1 - math.exp(-n / B)):,.0f}")
print(f"  most common 8-mers appear {counts.most_common(1)[0][1]} times (the average is {n / B:.2f}); fraction of possible 8-mers never seen: {1 - len(counts) / B:.3f} (theory e^(-n/B) = {math.exp(-n / B):.3f})")
print("duplicate detection among n items: all-pairs comparisons n(n-1)/2 vs hashing (n insertions):")
for nn in (10 ** 3, 10 ** 5, 10 ** 7):
    print(f"  n = {nn:>10,}: pairwise {nn * (nn - 1) // 2:>18,}   hashing {nn:>12,}   ratio {nn * (nn - 1) // 2 / nn:,.0f}x")
