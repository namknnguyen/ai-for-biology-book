"""Chapter 29: classical models, tested.
A. PWM scores and p-values depend on the background model (real JASPAR matrices scanned across a real genome).
B. A seven-state HMM gene finder trained on annotated chloroplast genes, versus an HMM-free per-base classifier.
C. Potts-model coevolution: mutual information vs mean-field DCA vs pseudolikelihood DCA, with independent vs phylogenetically related sequences."""
import os, urllib.request, warnings
import numpy as np
from Bio import SeqIO
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NC_000932.gb")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_000932.gb", path)
rec = SeqIO.read(path, "genbank"); seq = str(rec.seq).upper(); L = len(seq)
code = {c: i for i, c in enumerate("ACGT")}; arr = np.array([code[c] for c in seq], np.int64)
comp = str.maketrans("ACGT", "TGCA"); rc = lambda s: s.translate(comp)[::-1]

# ---------------------------------------------------------------- A. PWM scoring against different backgrounds
print("== A. A PWM threshold means nothing without a background model (JASPAR matrices scanned over the 154-kb chloroplast genome, both strands) ==")
from pyjaspar import jaspardb
jdb = jaspardb(release="JASPAR2024")
comp_bg = np.bincount(arr, minlength=4) / L
def score_dist(logodds, bg, scale=100):
    """Exact distribution of PWM scores (discretized) for iid background bg."""
    dist = {0: 1.0}
    for pos in range(logodds.shape[0]):
        new = {}
        for s, p in dist.items():
            for b in range(4):
                k = s + int(round(logodds[pos, b] * scale)); new[k] = new.get(k, 0.0) + p * bg[b]
        dist = new
    ks = np.array(sorted(dist)); pr = np.array([dist[k] for k in ks]); return ks / scale, pr
def scan(logodds, a):
    m = logodds.shape[0]; n = len(a) - m + 1; sc = np.zeros(n)
    for j in range(m): sc += logodds[j, a[j:j + n]]
    return sc
# order-2 Markov background fitted to the genome, used to simulate a realistic null
ctx = arr[:-2] * 4 + arr[1:-1]; nxt = arr[2:]; cnt = np.bincount(ctx * 4 + nxt, minlength=64).reshape(16, 4) + 0.5; tr = cnt / cnt.sum(1, keepdims=True)
sim = np.zeros(1_000_000, np.int64); sim[:2] = arr[:2]
u = rng.random(len(sim)); cumtr = np.cumsum(tr, 1)
for i in range(2, len(sim)): sim[i] = min(int(np.searchsorted(cumtr[sim[i - 2] * 4 + sim[i - 1]], u[i])), 3)
for mid, name in [("MA0139.1", "CTCF (GC-rich, 19 bp)"), ("MA0108.2", "TBP (AT-rich TATA-box-like)")]:
    m = jdb.fetch_motif_by_id(mid); cnts = np.array([m.counts[b] for b in "ACGT"]).T.astype(float)
    P = (cnts + 0.5) / (cnts + 0.5).sum(1, keepdims=True); lo = np.log2(P / 0.25)                 # log-odds against the UNIFORM background, as is conventional
    ic = (P * np.log2(P / 0.25)).sum()
    both = [arr, 3 - arr[::-1]]
    obs_sc = np.concatenate([scan(lo, a) for a in both]); sim_sc = np.concatenate([scan(lo, sim), scan(lo, 3 - sim[::-1])])
    ks_u, pr_u = score_dist(lo, np.full(4, 0.25)); ks_c, pr_c = score_dist(lo, comp_bg)
    tail = lambda ks, pr, t: pr[ks >= t].sum()
    print(f"\n{name}, matrix {mid}: width {lo.shape[0]}, information content {ic:.1f} bits; genome GC = {comp_bg[1] + comp_bg[2]:.3f}; {len(obs_sc):,} windows scanned")
    print("threshold chosen so that the p-value under a UNIFORM background is   expected hits (uniform)   expected hits (composition)   expected hits (order-2 Markov sim)   observed")
    for p_target in [1e-3, 1e-4, 1e-5]:
        t = ks_u[np.argmax(np.cumsum(pr_u[::-1])[::-1] <= p_target)]
        print(f"   p_uniform = {p_target:7.0e} (score >= {t:5.2f} bits)                   {len(obs_sc) * tail(ks_u, pr_u, t):10.1f}               {len(obs_sc) * tail(ks_c, pr_c, t):10.1f}                   {len(obs_sc) * np.mean(sim_sc >= t):10.1f}                    {np.sum(obs_sc >= t):6d}")

# ---------------------------------------------------------------- B. HMM gene finder
print("\n== B. A seven-state HMM gene finder, trained on annotated genes, 6-fold block cross-validation ==")
# per-base labels: 0 noncoding; 1-3 forward-strand coding phase 0,1,2; 4-6 reverse-strand coding phase 0,1,2 (phase advances along the + strand)
label = np.zeros(L, np.int64)
for f in rec.features:
    if f.type != "CDS": continue
    parts = list(f.location.parts); strand = f.location.strand
    tx = parts if strand == 1 else parts[::-1]; off = 0                                 # transcript order
    for part in tx:
        s0, e0 = int(part.start), int(part.end)
        xs = np.arange(s0, e0) if strand == 1 else np.arange(e0 - 1, s0 - 1, -1)
        o = off + np.arange(len(xs)); cp = o % 3
        if strand == 1: label[xs] = 1 + cp
        else: label[xs] = 4 + (2 - cp)
        off += len(xs)
USE = 128214                                                                           # exclude the second copy of the inverted repeat (an exact duplicate of the first)
seg = arr[:USE]; lab = label[:USE]
print(f"annotated: {np.mean(lab > 0):.3f} of the {USE:,} bases used are coding ({np.mean((lab >= 1) & (lab <= 3)):.3f} forward, {np.mean(lab >= 4):.3f} reverse)")
def fit_hmm(seg_idx):
    em = np.full((7, 16, 4), 0.5)
    x = seg[seg_idx]; y = lab[seg_idx]
    c = x[:-2] * 4 + x[1:-1]; n_ = x[2:]; ys = y[2:]
    for s in range(7):
        sel = ys == s
        em[s] += np.bincount(c[sel] * 4 + n_[sel], minlength=64).reshape(16, 4)
    em /= em.sum(2, keepdims=True)
    # segment lengths for transition probabilities
    runs_cod, runs_non = [], []; i = 0
    while i < len(y):
        j = i
        while j < len(y) and (y[j] > 0) == (y[i] > 0): j += 1
        (runs_cod if y[i] > 0 else runs_non).append(j - i); i = j
    p_exit = 3.0 / np.mean(runs_cod); p_enter = 1.0 / np.mean(runs_non) / 2
    return np.log(em), np.log(p_exit), np.log(1 - p_exit), np.log(p_enter), np.log(1 - 2 * p_enter)
def viterbi(x, em, lpe, lps, lpn, lpnn):
    n = len(x); S = 7; NEG = -1e18
    T = np.full((S, S), NEG)
    T[0, 0] = lpnn; T[0, 1] = lpn; T[0, 4] = lpn                                          # noncoding -> start of a forward or reverse gene
    T[1, 2] = 0.0; T[2, 3] = 0.0; T[3, 1] = lps; T[3, 0] = lpe                             # forward cycle 0->1->2->0 ; leave only after phase 2
    T[4, 5] = 0.0; T[5, 6] = 0.0; T[6, 4] = lps; T[6, 0] = lpe
    ctx = np.r_[0, 0, x[:-2] * 4 + x[1:-1]]; e = em[:, ctx, x]                            # (S, n)
    V = np.full((S, n), NEG); bp = np.zeros((S, n), np.int8); V[:, 0] = np.log(1.0 / S) + e[:, 0]
    for t in range(1, n):
        cand = V[:, t - 1][:, None] + T; bp[:, t] = cand.argmax(0); V[:, t] = cand.max(0) + e[:, t]
    path_ = np.zeros(n, np.int64); path_[-1] = V[:, -1].argmax()
    for t in range(n - 1, 0, -1): path_[t - 1] = bp[path_[t], t]
    return path_
folds = np.array_split(np.arange(USE), 6); conf = np.zeros((2, 2)); exact = tot_cod = 0; emit_conf = np.zeros((2, 2)); win_conf = np.zeros((2, 2)); base_phase_ok = 0
for k, te in enumerate(folds):
    tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
    # training excludes blocks adjacent to nothing special; a gene spanning a fold boundary appears partly in both: a minor leak at 5 boundaries
    em, lpe, lps, lpn, lpnn = fit_hmm(tr)
    x = seg[te]; y = lab[te]; pth = viterbi(x, em, lpe, lps, lpn, lpnn)
    conf += np.array([[np.sum((y == 0) & (pth == 0)), np.sum((y == 0) & (pth > 0))], [np.sum((y > 0) & (pth == 0)), np.sum((y > 0) & (pth > 0))]])
    exact += np.sum((y > 0) & (pth == y)); tot_cod += np.sum(y > 0)
    # HMM-free baseline: per-base argmax of the emission probability alone
    ctx = np.r_[0, 0, x[:-2] * 4 + x[1:-1]]; e = em[:, ctx, x]; am = e.argmax(0)
    # windowed log-likelihood-ratio classifier: best of the six coding frames vs noncoding over a 121-base window (GeneMark-style, no HMM)
    csum = lambda v: np.r_[0, np.cumsum(v)]
    tt = np.arange(len(x)); hyp = []
    for strand_base in (1, 4):
        for o in range(3): hyp.append(csum(e[strand_base + (tt + o) % 3, tt]))
    cn = csum(e[0]); lo_, hi_ = np.clip(tt - 60, 0, len(x)), np.clip(tt + 61, 0, len(x))
    best = np.max([hc[hi_] - hc[lo_] for hc in hyp], axis=0); win = (best - (cn[hi_] - cn[lo_])) > 0
    win_conf += np.array([[np.sum((y == 0) & ~win), np.sum((y == 0) & win)], [np.sum((y > 0) & ~win), np.sum((y > 0) & win)]])
    emit_conf += np.array([[np.sum((y == 0) & (am == 0)), np.sum((y == 0) & (am > 0))], [np.sum((y > 0) & (am == 0)), np.sum((y > 0) & (am > 0))]])
def prf(c):
    tp = c[1, 1]; fp = c[0, 1]; fn = c[1, 0]; prec = tp / (tp + fp); rec_ = tp / (tp + fn); return prec, rec_, 2 * prec * rec_ / (prec + rec_), (c[0, 0] + c[1, 1]) / c.sum()
print("method                               coding precision   coding recall   F1      accuracy")
cod_frac = np.mean(lab > 0)
print(f"predict everything coding                {cod_frac:7.3f}            1.000       {2*cod_frac/(1+cod_frac):.3f}   {cod_frac:.3f}")
p_, r_, f_, a_ = prf(emit_conf); print(f"per-base argmax of emissions (no HMM)    {p_:7.3f}            {r_:5.3f}       {f_:.3f}   {a_:.3f}")
p_, r_, f_, a_ = prf(win_conf); print(f"window LLR classifier, 121 bases (no HMM){p_:7.3f}            {r_:5.3f}       {f_:.3f}   {a_:.3f}")
p_, r_, f_, a_ = prf(conf); print(f"HMM, Viterbi path                        {p_:7.3f}            {r_:5.3f}       {f_:.3f}   {a_:.3f}")
print(f"of the truly coding bases, the Viterbi path has the exact strand and codon phase for {exact / tot_cod:.3f}")

# ---------------------------------------------------------------- C. coevolution with a Potts model
print("\n== C. Coevolution: contacts from sequence families generated by a known Potts model (L = 30 positions, q = 4 states) ==")
Lp, q = 30, 4
rs = np.random.default_rng(1)
h = rs.normal(0, 0.3, (Lp, q)); J = np.zeros((Lp, Lp, q, q)); pairs = []
for i in range(Lp):
    for j in range(i + 4, Lp):
        if rs.random() < 0.10:
            M_ = rs.normal(0, 1.0, (q, q)); J[i, j] = M_; J[j, i] = M_.T; pairs.append((i, j))
true_mask = np.zeros((Lp, Lp), bool)
for i, j in pairs: true_mask[i, j] = true_mask[j, i] = True
K = len(pairs); print(f"{K} true coupled pairs among {sum(1 for i in range(Lp) for j in range(i + 4, Lp))} candidate pairs (|i-j| >= 4)")
def gibbs(n_chains, sweeps, init=None, r=rs):
    s = r.integers(0, q, (n_chains, Lp)) if init is None else init.copy()
    for _ in range(sweeps):
        for i in r.permutation(Lp):
            lg = h[i][None, :] + J[i, np.arange(Lp)[None, :], :, s].sum(1)
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            s[:, i] = (r.random(n_chains)[:, None] > np.cumsum(p, 1)).sum(1).clip(max=q - 1)
    return s
def phylo(depth, muts_per_branch, r=rs):
    seqs = gibbs(1, 400, r=r)
    for _ in range(depth):
        seqs = np.repeat(seqs, 2, axis=0)
        for _m in range(muts_per_branch):
            pos = r.integers(0, Lp, len(seqs))
            lg = h[pos] + J[pos[:, None], np.arange(Lp)[None, :], :, seqs].sum(1)
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            seqs[np.arange(len(seqs)), pos] = (r.random(len(seqs))[:, None] > np.cumsum(p, 1)).sum(1).clip(max=q - 1)
    return seqs
def weights(S, thr=0.8):
    ident = (S[:, None, :] == S[None, :, :]).mean(2); return 1.0 / (ident >= thr).sum(1)
def freqs(S, w, lam=0.5):
    N = w.sum(); oh = np.eye(q)[S]                                                      # (n, L, q)
    fi = np.einsum("n,nia->ia", w, oh); fij = np.einsum("n,nia,njb->ijab", w, oh, oh)
    fi = (1 - lam) * fi / N + lam / q; fij = (1 - lam) * fij / N + lam / q ** 2
    for i in range(Lp): fij[i, i] = np.diag(fi[i]) * 0 + np.eye(q) * fi[i][:, None]
    return fi, fij
def apc(S_):
    S2 = S_.copy(); np.fill_diagonal(S2, 0); r_ = S2.mean(1); return S2 - np.outer(r_, r_) / S2.mean()
def score_mi(S, w):
    fi, fij = freqs(S, w); mi = np.zeros((Lp, Lp))
    for i in range(Lp):
        for j in range(Lp):
            if i != j: mi[i, j] = np.sum(fij[i, j] * np.log(fij[i, j] / np.outer(fi[i], fi[j])))
    return apc(mi)
def score_mf(S, w, lam=0.5):
    fi, fij = freqs(S, w, lam); idx = [(i, a) for i in range(Lp) for a in range(q - 1)]
    C = np.zeros((len(idx), len(idx)))
    for u, (i, a) in enumerate(idx):
        for v, (j, b) in enumerate(idx): C[u, v] = fij[i, j, a, b] - fi[i, a] * fi[j, b]
    Jm = -np.linalg.inv(C); out = np.zeros((Lp, Lp))
    for i in range(Lp):
        for j in range(Lp):
            if i != j: out[i, j] = np.linalg.norm(Jm[i * (q - 1):(i + 1) * (q - 1), j * (q - 1):(j + 1) * (q - 1)])
    return apc(out)
from sklearn.linear_model import LogisticRegression
def score_plm(S, w, lam_J=0.01):
    oh = np.eye(q)[S].reshape(len(S), -1); Neff = w.sum(); Jh = np.zeros((Lp, Lp, q, q))
    for r_ in range(Lp):
        cols = [j * q + b for j in range(Lp) if j != r_ for b in range(q)]
        clf = LogisticRegression(C=1.0 / (2 * lam_J * Neff), max_iter=300, tol=1e-4).fit(oh[:, cols], S[:, r_], sample_weight=w)
        W = np.zeros((q, len(cols)));  W[clf.classes_] = clf.coef_ if len(clf.classes_) > 2 else np.vstack([-clf.coef_[0] / 2, clf.coef_[0] / 2])
        js = [j for j in range(Lp) if j != r_]
        for u, j in enumerate(js): Jh[r_, j] = W[:, u * q:(u + 1) * q]
    Jh = (Jh + Jh.transpose(1, 0, 3, 2)) / 2
    Jh = Jh - Jh.mean(2, keepdims=True) - Jh.mean(3, keepdims=True) + Jh.mean((2, 3), keepdims=True)
    out = np.linalg.norm(Jh.reshape(Lp, Lp, -1), axis=2); return apc(out)
cand = np.array([[abs(i - j) >= 4 for j in range(Lp)] for i in range(Lp)]) & np.triu(np.ones((Lp, Lp), bool), 1)
def precision(score):
    sc = np.where(cand, score, -np.inf); top = np.dstack(np.unravel_index(np.argsort(-sc, axis=None)[:K], sc.shape))[0]
    return np.mean([true_mask[i, j] for i, j in top])
print("sequences (effective number after 80%-identity weighting)       precision of the top-K pairs:   MI+APC   mean-field DCA   pseudolikelihood DCA")
for N in [100, 500, 2000, 8000]:
    S = gibbs(N, 250); w = weights(S) if N <= 2000 else np.full(N, 1.0)
    print(f"independent samples, N = {N:5d} (N_eff = {w.sum():7.1f})                                     {precision(score_mi(S, w)):6.2f}      {precision(score_mf(S, w)):6.2f}            {precision(score_plm(S, w)):6.2f}")
for depth, mp in [(10, 1), (10, 3)]:
    S = phylo(depth, mp); N = len(S); w1 = np.ones(N); w2 = weights(S)
    print(f"tree, depth {depth}, {mp} mutation(s) per branch, N = {N}: no weighting (N_eff = {N}) {precision(score_mi(S, w1)):6.2f}      {precision(score_mf(S, w1)):6.2f}            {precision(score_plm(S, w1)):6.2f}")
    print(f"      with 80%-identity reweighting (N_eff = {w2.sum():7.1f})                             {precision(score_mi(S, w2)):6.2f}      {precision(score_mf(S, w2)):6.2f}            {precision(score_plm(S, w2)):6.2f}")
