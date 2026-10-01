"""Chapter 42: ancestral sequence reconstruction (ASR) on simulated phylogenies with a known root, and the bias of maximum-likelihood reconstructions.
Evolution: 32 extant sequences of 300 sites from a random ultrametric (coalescent) tree; each site has its own preferred amino-acid distribution pi_i (Dirichlet, concentration 0.15) and a Gamma-distributed
rate (shape 0.5); substitution follows the F81-type process  P(t) = exp(-r_i t) I + (1 - exp(-r_i t)) 1 pi_i^T.  Root-to-tip distance T is varied.
Reconstructions of the root: Fitch parsimony; maximum likelihood (marginal posterior mode) under (a) the Jukes-Cantor-type model (uniform frequencies, one rate), (b) global amino-acid frequencies,
(c) the TRUE site-specific model (an oracle); the majority-rule consensus of the extant sequences; and sampling from the posterior.
Reported: accuracy at the root, and a typicality score  s = mean_i log pi_i(a_i)  (a stand-in for stability or fitness: higher means closer to the site-wise consensus)."""
import numpy as np
rng = np.random.default_rng(0)
A, L, NLEAF = 20, 300, 32
def make_tree(T, r):
    """Ultrametric coalescent tree.  Returns children dict, branch length to parent, root id, leaves."""
    nodes = list(range(NLEAF)); time = {i: 0.0 for i in nodes}; children = {}; t = 0.0; nid = NLEAF; active = nodes[:]
    while len(active) > 1:
        k = len(active); t += r.exponential(1.0 / (k * (k - 1) / 2)); a, b = r.choice(len(active), 2, replace=False); x, y = active[a], active[b]
        children[nid] = (x, y); time[nid] = t; active = [z for z in active if z not in (x, y)] + [nid]; nid += 1
    scale = T / time[nid - 1]; blen = {}
    for p, (x, y) in children.items(): blen[x] = (time[p] - time[x]) * scale; blen[y] = (time[p] - time[y]) * scale
    return children, blen, nid - 1
def simulate(children, blen, root, pi, rate, r):
    cum = np.cumsum(pi, 1); seq = {root: np.array([np.searchsorted(cum[i], r.random()) for i in range(L)]).clip(0, A - 1)}
    stack = [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            keep = r.random(L) < np.exp(-rate * blen[c]); new = np.array([np.searchsorted(cum[i], r.random()) for i in range(L)]).clip(0, A - 1)
            seq[c] = np.where(keep, seq[p], new); stack.append(c)
    return seq
def prune(children, blen, root, leaf_seq, pi, rate):
    """Felsenstein pruning with the F81 site model; returns the partial likelihood at the root, shape (L, A)."""
    Lk = {}
    def rec(n):
        if n in leaf_seq: v = np.zeros((L, A)); v[np.arange(L), leaf_seq[n]] = 1.0; return v
        out = np.ones((L, A))
        for c in children[n]:
            lc = rec(c); e = np.exp(-rate * blen[c])[:, None]; out *= e * lc + (1 - e) * (lc * pi).sum(1, keepdims=True)
        return out
    return rec(root)
def fitch(children, root, leaf_seq, r):
    def rec(n):
        if n in leaf_seq: s = np.zeros((L, A), bool); s[np.arange(L), leaf_seq[n]] = True; return s
        a, b = [rec(c) for c in children[n]]; inter = a & b; return np.where(inter.any(1, keepdims=True), inter, a | b)
    S = rec(root); return np.array([r.choice(np.flatnonzero(S[i])) for i in range(L)])
def sample_post(post, r):
    cum = np.cumsum(post, 1); return (r.random(L)[:, None] > cum).sum(1).clip(0, A - 1)
print("== Reconstructing the root of a 32-leaf tree (300 sites): accuracy, and the typicality score of the reconstruction compared with the TRUE root; mean of 30 trees per row ==")
print("root-to-tip   method                                   accuracy   typicality s   bias in s (reconstruction - true)")
for T in (0.3, 0.8, 1.5, 3.0):
    acc = {}; typ = {}; true_typ = []
    for rep in range(30):
        r = np.random.default_rng(100 * int(T * 10) + rep); pi = r.dirichlet(np.full(A, 0.15), L) + 1e-4; pi /= pi.sum(1, keepdims=True); rate = r.gamma(0.5, 1 / 0.5, L)
        children, blen, root = make_tree(T, r); seq = simulate(children, blen, root, pi, rate, r); leaves = {i: seq[i] for i in range(NLEAF)}; truth = seq[root]
        logpi = np.log(pi); s = lambda a: logpi[np.arange(L), a].mean(); true_typ.append(s(truth))
        glob = np.mean([np.bincount(leaves[i], minlength=A) for i in leaves], 0) / L + 1e-3; glob /= glob.sum()
        models = {"ML, uniform frequencies (JC-type)": (np.full((L, A), 1.0 / A), np.ones(L)), "ML, global amino-acid frequencies": (np.tile(glob, (L, 1)), np.ones(L)), "ML, true site-specific model (oracle)": (pi, rate)}
        recs = {"Fitch parsimony": fitch(children, root, leaves, r), "majority-rule consensus of the extant sequences": np.array([np.bincount([leaves[i][j] for i in leaves], minlength=A).argmax() for j in range(L)])}
        for name, (p_, r_) in models.items():
            Lroot = prune(children, blen, root, leaves, p_, r_); post = Lroot * p_; post /= post.sum(1, keepdims=True)
            recs[name] = post.argmax(1); recs[name + ", sampled from the posterior"] = sample_post(post, r)
        for name, a in recs.items(): acc.setdefault(name, []).append((a == truth).mean()); typ.setdefault(name, []).append(s(a))
    tt = np.mean(true_typ)
    for name in acc: print(f"{T:8.1f}      {name:50s}  {np.mean(acc[name]):7.3f}    {np.mean(typ[name]):8.3f}        {np.mean(typ[name]) - tt:+7.3f}")
    print(f"             (typicality of the TRUE root: {tt:.3f})")

# ---------------------------------------------------------------- 2. shared ancestry: effective sample size and false correlations among species
from scipy import stats
print("\n== 2. Shared ancestry makes 32 species far fewer than 32 independent observations ==")
def shared_cov(children, blen, root):
    """Brownian-motion covariance between leaves: the length of the root-to-ancestor path they share."""
    path = {root: 0.0}; anc = {root: [root]}; stack = [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()): path[c] = path[p] + blen[c]; anc[c] = anc[p] + [c]; stack.append(c)
    C = np.zeros((NLEAF, NLEAF))
    for i in range(NLEAF):
        for j in range(NLEAF): C[i, j] = path[[a for a in anc[i] if a in anc[j]][-1]]
    return C
neff, fp_ols, fp_gls = [], [], []; nrep = 3000
for rep in range(nrep):
    r = np.random.default_rng(5000 + rep); children, blen, root = make_tree(1.0, r); C = shared_cov(children, blen, root); one = np.ones(NLEAF)
    neff.append(NLEAF ** 2 * 1.0 / (one @ C @ one))
    Lc = np.linalg.cholesky(C + 1e-9 * np.eye(NLEAF)); x = Lc @ r.standard_normal(NLEAF); y = Lc @ r.standard_normal(NLEAF)               # two traits evolving independently (Brownian motion on the tree)
    fp_ols.append(stats.pearsonr(x, y)[1] < 0.05)
    Ci = np.linalg.inv(C); X = np.c_[one, x]; XtCi = X.T @ Ci; cov = np.linalg.inv(XtCi @ X); b = cov @ XtCi @ y; res = y - X @ b; s2 = res @ Ci @ res / (NLEAF - 2)
    t = b[1] / np.sqrt(s2 * cov[1, 1]); fp_gls.append(2 * stats.t.sf(abs(t), NLEAF - 2) < 0.05)
print(f"effective number of independent observations for estimating a trait mean, from {NLEAF} species on a random coalescent tree: mean {np.mean(neff):.2f} (median {np.median(neff):.2f}, 90% range {np.percentile(neff, 5):.2f}-{np.percentile(neff, 95):.2f})")
print(f"two traits that evolved independently (no true association), across {NLEAF} species, {nrep} random trees: nominal 5% test rejects in {np.mean(fp_ols):.3f} of trees by ordinary least squares, in {np.mean(fp_gls):.3f} with phylogenetic generalized least squares")
