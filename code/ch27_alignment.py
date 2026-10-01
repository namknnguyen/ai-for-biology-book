"""Chapter 27: alignment statistics, read mapping, and variant calling, tested on synthetic data.
1. Karlin-Altschul: the null distribution of local alignment scores is Gumbel with lambda solving sum p_i p_j exp(lambda s_ij) = 1.
2. E-values: false-positive counts in a database search match the prediction.
3. FM-index: backward search on the Burrows-Wheeler transform equals brute-force counting.
4. Mappability: repeats make short reads ambiguous.
5. Variant calling: genotype likelihoods, depth, reference bias, and correlated errors."""
import re
import numpy as np
from scipy.optimize import brentq
from Bio.Align import substitution_matrices

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. Karlin-Altschul statistics for ungapped local alignment (BLOSUM62)
B = substitution_matrices.load("BLOSUM62")
AA = "ARNDCQEGHILKMFPSTWYV"
bg = np.array([0.0825, 0.0553, 0.0406, 0.0545, 0.0137, 0.0393, 0.0675, 0.0707, 0.0227, 0.0595,
               0.0966, 0.0584, 0.0242, 0.0386, 0.0470, 0.0657, 0.0534, 0.0108, 0.0292, 0.0687]); bg /= bg.sum()   # Robinson & Robinson frequencies
S = np.array([[B[a][b] for b in AA] for a in AA], float)
f = lambda lam: (bg[:, None] * bg[None, :] * np.exp(lam * S)).sum() - 1
lam = brentq(f, 1e-3, 1.0)                                                  # the unique positive root
E_s = (bg[:, None] * bg[None, :] * S).sum()
H = lam * (bg[:, None] * bg[None, :] * S * np.exp(lam * S)).sum()            # relative entropy of target vs background, in nats
print("== 1. Karlin-Altschul theory for ungapped local alignment, BLOSUM62 ==")
print(f"expected score per aligned pair under the background model = {E_s:.3f} (must be negative); lambda = {lam:.4f}; relative entropy H = {H:.3f} nats = {H/np.log(2):.2f} bits per aligned pair")

def max_ungapped(m, n, P):
    """Maximum local ungapped score for P independent random sequence pairs of lengths m, n (diagonal Kadane DP, vectorized over pairs)."""
    a = rng.choice(20, (P, m), p=bg); b = rng.choice(20, (P, n), p=bg)
    best = np.zeros(P); h = np.zeros((P, n + 1))
    for i in range(m):
        s = S[a[:, i][:, None], b]                                           # (P, n) scores of residue i against every residue of the other sequence
        h_new = np.zeros_like(h); h_new[:, 1:] = np.maximum(0, h[:, :-1] + s)
        best = np.maximum(best, h_new.max(1)); h = h_new
    return best
m = n = 250; P = 4000
smax = max_ungapped(m, n, P)
# location of the Gumbel: mean = mu + gamma/lambda; the *analytic* lambda is used, K_eff absorbs edge effects of finite sequence lengths
mu = smax.mean() - 0.5772156649 / lam
K = np.exp(lam * mu) / (m * n)
print(f"{P} random pairs of length {m}: mean max score {smax.mean():.2f}; analytic lambda = {lam:.4f}; effective K fitted from the mean = {K:.3f}")
sd_emp, sd_th = smax.std(), np.pi / (lam * np.sqrt(6))
print(f"standard deviation of the max score: empirical {sd_emp:.2f}; Gumbel theory pi/(lambda sqrt 6) = {sd_th:.2f}  (this checks lambda without fitting it)")

# ---------------------------------------------------------------- 2. E-value calibration in a database search (fresh, independent comparisons)
Nq, Ndb = 20, 5000
print(f"\n== 2. E-value calibration: {Nq} queries x a database of {Ndb:,} random sequences (all length {m}), {Nq*Ndb:,} comparisons ==")
s2 = np.concatenate([max_ungapped(m, n, Ndb) for _ in range(Nq)])
print("score threshold S   predicted E-value per query (Ndb K m n e^{-lambda S})   observed false hits per query   (95% Poisson interval)")
for t in [36, 40, 44, 48, 52]:
    e_pred = Ndb * K * m * n * np.exp(-lam * t); k = np.sum(s2 >= t)
    lo, hi = (k - 1.96 * np.sqrt(max(k, 1))) / Nq, (k + 1.96 * np.sqrt(max(k, 1))) / Nq
    print(f"{t:10d}          {e_pred:14.3f}                                    {k / Nq:8.3f}                      ({max(lo, 0):.3f}, {hi:.3f})")

# ---------------------------------------------------------------- 3. FM-index: backward search on the Burrows-Wheeler transform
print("\n== 3. FM-index backward search ==")
def suffix_array(s):
    n = len(s); k = 1; rank = np.array(list(s), np.int64); sa = np.argsort(rank, kind="stable")
    while True:
        key2 = np.where(sa + k < n, rank[np.minimum(sa + k, n - 1)], -1)
        order = np.lexsort((key2, rank[sa])); sa = sa[order]
        r1, r2 = rank[sa], key2[order]
        newrank = np.zeros(n, np.int64); newrank[sa] = np.concatenate([[0], np.cumsum((r1[1:] != r1[:-1]) | (r2[1:] != r2[:-1]))])
        rank = newrank
        if rank.max() == n - 1: break
        k *= 2
    return sa
G = 200000
genome = rng.integers(1, 5, G).astype(np.int64)                              # 1..4 = A,C,G,T; 0 is the sentinel
text = np.concatenate([genome, [0]])
sa = suffix_array(text)
bwt = text[(sa - 1) % len(text)]
C = np.zeros(6, np.int64)                                                    # C[c] = number of symbols smaller than c
for c in range(1, 6): C[c] = np.sum(text < c)
occ = np.zeros((len(bwt) + 1, 5), np.int32)
for c in range(5): occ[1:, c] = np.cumsum(bwt == c)                          # Occ[i, c] = occurrences of c in bwt[:i]
def backward_search(pat):
    lo, hi = 0, len(bwt)
    for c in pat[::-1]:
        lo = C[c] + occ[lo, c]; hi = C[c] + occ[hi, c]
        if lo >= hi: return 0, lo, hi
    return hi - lo, lo, hi
gs = "".join("ACGT"[x - 1] for x in genome)
bad = 0
for _ in range(300):
    k = int(rng.integers(6, 14)); st = int(rng.integers(0, G - k)); pat = genome[st:st + k]
    pstr = "".join("ACGT"[x - 1] for x in pat)
    naive = len(re.findall("(?=" + pstr + ")", gs))                           # overlapping occurrences, brute force
    cnt_fm, lo, hi = backward_search(pat)
    bad += (cnt_fm != naive)
print(f"genome of {G:,} bases; suffix array + BWT + Occ table; 300 random patterns of 6-13 bases: mismatches between FM-index counts and brute force = {bad}")
print(f"a pattern of length m is located in O(m) rank operations independent of genome size; memory here {occ.nbytes/1e6:.1f} MB for the full Occ table (production indexes sample it)")

# ---------------------------------------------------------------- 4. mappability and repeats
print("\n== 4. Repeats make short reads ambiguous (300 kb genome with a 300-bp element in 300 copies diverged by 0.5-10%) ==")
L = 300000; gen = rng.integers(0, 4, L); is_rep = np.zeros(L, bool)
element = rng.integers(0, 4, 300)
for _ in range(300):
    pos = int(rng.integers(0, L - 300)); copy = element.copy()
    div = rng.uniform(0.005, 0.10)                                               # each copy has its own age: 0.5-10% divergence from the consensus
    mut = rng.random(300) < div; copy[mut] = rng.integers(0, 4, mut.sum())
    gen[pos:pos + 300] = copy; is_rep[pos:pos + 300] = True
Kmer = 20
def kmers(arr):
    v = np.zeros(len(arr) - Kmer + 1, np.int64)
    for j in range(Kmer): v = v * 4 + arr[j:len(arr) - Kmer + 1 + j]
    return v
gk = kmers(gen); index = {}
for i, kk in enumerate(gk): index.setdefault(int(kk), []).append(i)
def map_read(read):
    votes = {}
    for off, kk in enumerate(kmers(read)):
        for p in index.get(int(kk), ()):
            votes[p - off] = votes.get(p - off, 0) + 1
    if not votes: return None, 0.0
    order = sorted(votes.items(), key=lambda x: -x[1]); best = order[0]
    second = order[1][1] if len(order) > 1 else 0
    return best[0], best[1] / max(second, 0.5)                                # support ratio of the best position over the runner-up
print("read length   region     reads   mapped correctly & unambiguous (ratio>=2)   ambiguous   wrong or unmapped")
for rl in [50, 100, 250]:
    for name, inrep in [("unique", False), ("repeat", True)]:
        ok = amb = bad_ = tot = 0
        while tot < 400:
            st = int(rng.integers(0, L - rl))
            if inrep and not is_rep[st:st + rl].all(): continue
            if (not inrep) and is_rep[st:st + rl].any(): continue
            read = gen[st:st + rl].copy(); err = rng.random(rl) < 0.01; read[err] = (read[err] + rng.integers(1, 4, err.sum())) % 4
            pos, ratio = map_read(read); tot += 1
            if pos is None: bad_ += 1
            elif ratio < 2: amb += 1
            elif pos == st: ok += 1
            else: bad_ += 1
        print(f"{rl:8d}      {name:8s}  {tot:5d}       {ok / tot:6.2f}                                {amb / tot:6.2f}      {bad_ / tot:6.2f}")

# ---------------------------------------------------------------- 5. variant calling from a pileup
print("\n== 5. Genotype calling from a read pileup ==")
def call_sites(n_sites, depth, err, p_alt=0.15, ref_bias=0.0, site_err_sd=0.0):
    """Biallelic sites; genotype ~ HWE with alt frequency p_alt; reads: Poisson(depth); each base is wrong with probability e (e may vary by site)."""
    gt = rng.choice(3, n_sites, p=[(1 - p_alt) ** 2, 2 * p_alt * (1 - p_alt), p_alt ** 2])           # alt allele dose 0,1,2
    prior = np.log(np.array([(1 - p_alt) ** 2, 2 * p_alt * (1 - p_alt), p_alt ** 2]))
    e_site = np.clip(err * np.exp(rng.normal(0, site_err_sd, n_sites)), 1e-5, 0.3) if site_err_sd else np.full(n_sites, err)
    d = rng.poisson(depth, n_sites); calls = np.zeros(n_sites, int)
    for i in range(n_sites):
        if d[i] == 0: continue
        pa = np.array([0.0, 0.5, 1.0])[gt[i]]
        pa = pa * (1 - ref_bias) / (pa * (1 - ref_bias) + (1 - pa))               # alt-carrying reads map with probability (1 - bias)
        pobs = pa * (1 - e_site[i]) + (1 - pa) * e_site[i]                          # probability that a read shows 'alt'
        k = rng.binomial(d[i], pobs)                                                # observed alt count
        e_assumed = err                                                             # the caller assumes the nominal error rate and perfect mapping
        pa_g = np.array([e_assumed, 0.5, 1 - e_assumed])
        ll = k * np.log(pa_g) + (d[i] - k) * np.log(1 - pa_g) + prior
        calls[i] = np.argmax(ll)
    return gt, calls
print("depth   het sensitivity   hom-ref miscalled as variant (per Mb)   scenario")
for depth in [2, 5, 10, 30]:
    for name, kw in [("independent errors, e = 0.01", {}), ("reference bias 20%", {"ref_bias": 0.2}), ("site-specific error rates (log-SD 1.0)", {"site_err_sd": 1.0})]:
        gt, c = call_sites(60000, depth, 0.01, **kw)
        het = gt == 1; ref = gt == 0
        print(f"{depth:5d}   {np.mean(c[het] == 1):10.3f}        {1e6 * np.mean(c[ref] != 0):12.0f}                  {name}")
