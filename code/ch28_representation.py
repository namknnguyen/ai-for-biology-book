"""Chapter 28: representing DNA for models, on a real genome (Arabidopsis thaliana chloroplast, NC_000932, 154,478 bp, 85 CDS).
1. an inverted repeat found without alignment; 2. held-out entropy of Markov models, with and without leakage and RC augmentation;
3. how much does knowing the codon frame buy; 4. tokenization: BPE compression, shift sensitivity, codon alignment, masked-k-mer leakage;
5. reverse-complement symmetry of genomes and its breaking in genes."""
import os, urllib.request
import numpy as np
from Bio import SeqIO
from tokenizers import Tokenizer, models, trainers

rng = np.random.default_rng(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NC_000932.gb")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_000932.gb", path)
rec = SeqIO.read(path, "genbank"); seq = str(rec.seq).upper(); L = len(seq)
comp = str.maketrans("ACGT", "TGCA"); rc = lambda s: s.translate(comp)[::-1]
code = {c: i for i, c in enumerate("ACGT")}
arr = np.array([code[c] for c in seq], np.int8)
print(f"genome: {L:,} bp, GC = {np.mean((arr == 1) | (arr == 2)):.3f}, features: {sum(f.type=='CDS' for f in rec.features)} CDS, {sum(f.type=='tRNA' for f in rec.features)} tRNA, {sum(f.type=='rRNA' for f in rec.features)} rRNA")

# ---------------------------------------------------------------- 1. an inverted repeat from k-mer hashing
print("\n== 1. Finding the large inverted repeat with 25-mer hashing (no alignment) ==")
K = 25; fwd = {}
for i in range(L - K + 1): fwd.setdefault(seq[i:i + K], []).append(i)
has_partner = np.zeros(L - K + 1, bool)
for i in range(L - K + 1):
    for j in fwd.get(rc(seq[i:i + K]), ()):
        if abs(i - j) > 1000: has_partner[i] = True
pos = np.flatnonzero(has_partner); gaps = np.flatnonzero(np.diff(pos) > 200)
starts = np.r_[pos[0], pos[gaps + 1]]; ends = np.r_[pos[gaps], pos[-1]] + K
big = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a > 5000]
print(f"{has_partner.mean():.1%} of 25-mers have a reverse-complement copy elsewhere; large inverted-repeat blocks (start, end, length): " + "; ".join(f"({a:,}, {b:,}, {b-a:,})" for a, b in big))
IR_a, IR_b = big[0], big[-1]

# ---------------------------------------------------------------- 2. held-out entropy of order-k Markov models
def contexts(a, k):
    n = len(a); c = np.zeros(n - k, np.int64)
    for j in range(k): c = c * 4 + a[j:n - k + j]
    return c, a[k:].astype(np.int64)
def fit(train_list, k):
    cnt = np.zeros(4 ** (k + 1), np.int32)
    for t in train_list:
        if len(t) <= k: continue
        c, nxt = contexts(t, k); cnt += np.bincount(c * 4 + nxt, minlength=4 ** (k + 1)).astype(np.int32)
    return cnt.reshape(-1, 4)
def xent(cnt, test, k, alpha=0.5):
    c, nxt = contexts(test, k); num = cnt[c, nxt] + alpha; den = cnt[c].sum(1) + 4 * alpha
    return -np.mean(np.log2(num / den))
rcarr = lambda a: (3 - a)[::-1]
def split(test_lo, test_hi):
    tr = [arr[:test_lo], arr[test_hi:]]; te = arr[test_lo:test_hi]
    return tr, te
print("\n== 2. Held-out cross-entropy (bits per base) of order-k Markov models; the maximum is 2.0 ==")
print(f"inverted repeat copies at {IR_a} and {IR_b}")
leaky = split(IR_b[0], IR_b[1])                                            # test = one IR copy; the other copy (reverse-complemented) is in the training data
clean_lo = 20000; clean = split(clean_lo, clean_lo + (IR_b[1] - IR_b[0]))     # test = a unique region of the same length in the large single-copy region
print("order k   unique test region               | test = one copy of the inverted repeat")
print("          forward only   + RC augmentation | forward only   + RC augmentation")
for k in [0, 2, 4, 6, 8, 10]:
    row = []
    for (tr, te) in [clean, leaky]:
        row.append(xent(fit(tr, k), te, k)); row.append(xent(fit(tr + [rcarr(t) for t in tr], k), te, k))
    print(f"{k:5d}     {row[0]:7.3f}        {row[1]:7.3f}          | {row[2]:7.3f}        {row[3]:7.3f}")

# ---------------------------------------------------------------- 3. the codon frame
print("\n== 3. Knowing the codon frame (held-out genes; coding sequences on the sense strand) ==")
cds = []
for f in rec.features:
    if f.type == "CDS":
        s = str(f.extract(rec.seq)).upper()
        if len(s) % 3 == 0 and len(s) >= 300 and set(s) <= set("ACGT"): cds.append(np.array([code[c] for c in s], np.int8))
perm = rng.permutation(len(cds)); n_tr = int(0.7 * len(cds)); tr_g = [cds[i] for i in perm[:n_tr]]; te_g = [cds[i] for i in perm[n_tr:]]
print(f"{len(cds)} complete CDS (>=300 bp); {n_tr} training genes ({sum(map(len, tr_g)):,} bp), {len(te_g)} held-out genes ({sum(map(len, te_g)):,} bp); GC3 = {np.mean([np.mean(np.isin(g[2::3], [1, 2])) for g in cds]):.3f}")
def frame_xent(k, frame_aware):
    tabs = [np.zeros((4 ** k, 4)) for _ in range(3 if frame_aware else 1)]
    for g in tr_g:
        c, nxt = contexts(g, k)
        for ph in range(3 if frame_aware else 1):
            sel = (np.arange(k, len(g)) % 3 == ph) if frame_aware else slice(None)
            np.add.at(tabs[ph], (c[sel], nxt[sel]), 1)
    tot = 0.0; n = 0
    for g in te_g:
        c, nxt = contexts(g, k)
        for ph in range(3 if frame_aware else 1):
            sel = (np.arange(k, len(g)) % 3 == ph) if frame_aware else slice(None)
            t = tabs[ph]; p = (t[c[sel], nxt[sel]] + 0.5) / (t[c[sel]].sum(1) + 2.0)
            tot += -np.log2(p).sum(); n += len(p)
    return tot / n
print("order k   frame-agnostic   frame-aware (3 position-specific tables)   gain (bits/base)")
for k in [0, 1, 2, 3, 4]:
    a, b = frame_xent(k, False), frame_xent(k, True)
    print(f"{k:5d}     {a:8.3f}        {b:8.3f}                                 {a - b:6.3f}")

# ---------------------------------------------------------------- 4. tokenization
print("\n== 4. Tokenization ==")
cut = int(0.75 * L); train_txt = seq[:cut]; test_txt = seq[cut:]
lines = [train_txt[i:i + 1000] for i in range(0, len(train_txt), 1000)]
def train_bpe(V):
    tok = Tokenizer(models.BPE(unk_token="N")); tr = trainers.BpeTrainer(vocab_size=V, special_tokens=["N"], initial_alphabet=list("ACGT"), show_progress=False)
    tok.train_from_iterator(lines, tr); return tok
def boundaries(tokens):
    b = set(); p = 0
    for t in tokens: p += len(t); b.add(p)
    return b
tests = test_txt[:30000]
res = {}
print("tokenizer              bases/token on held-out DNA   mean token length   boundaries retained after a 1-base shift   tokens identical after a 1-base insertion (downstream)")
def report(name, toks_fn):
    toks = toks_fn(tests)
    B = boundaries(toks); toks_s = toks_fn(tests[1:]); Bs = {b + 1 for b in boundaries(toks_s)}
    keep = len(B & Bs) / len(Bs)
    mid = len(tests) // 2; ins = tests[:mid] + "A" + tests[mid:]
    ti = toks_fn(ins); p = 0; down_ins = []
    for t in ti:
        if p > mid + 1: down_ins.append((p - 1, t))
        p += len(t)
    p = 0; orig = {}
    for t in toks:
        orig[(p, t)] = 1; p += len(t)
    ident = np.mean([(pp, t) in orig for pp, t in down_ins]) if down_ins else float("nan")
    print(f"{name:22s} {len(tests) / len(toks):10.2f}                    {np.mean([len(t) for t in toks]):8.2f}            {keep:8.3f}                                {ident:8.3f}")
report("single nucleotide", lambda s: list(s))
for kk in [3, 6]: report(f"non-overlapping {kk}-mer", lambda s, kk=kk: [s[i:i + kk] for i in range(0, len(s) - kk + 1, kk)])
for V in [260, 1024, 4096]:
    tok = train_bpe(V); res[V] = tok
    report(f"BPE, vocabulary {V}", lambda s, tok=tok: [t for t in tok.encode(s).tokens])

def with_pos(toks):
    p = 0; out = []
    for t in toks: out.append((p, t)); p += len(t)
    return out
snp_sites = rng.integers(1000, len(test_txt) - 1000, 400)
print("\neffect of a single-base substitution (window of 2,000 bases around the site): tokens that change, and whether the token count changes")
print("tokenizer              mean tokens changed (ref side)   fraction of SNPs that change the number of tokens")
fns = [("single nucleotide", lambda s_: list(s_)), ("non-overlapping 6-mer", lambda s_: [s_[i:i + 6] for i in range(0, len(s_) - 5, 6)]),
       ("overlapping 6-mer (stride 1)", lambda s_: [s_[i:i + 6] for i in range(0, len(s_) - 5)]), ("BPE, vocabulary 4096", lambda s_: res[4096].encode(s_).tokens)]
for name, fn in fns:
    ch = []; cn = []
    for p0 in snp_sites:
        w = test_txt[p0 - 1000:p0 + 1000]; alt = "ACGT".replace(w[1000], "")[int(rng.integers(0, 3))]; w2 = w[:1000] + alt + w[1001:]
        a, b = fn(w), fn(w2); sa, sb = set(with_pos(a)), set(with_pos(b))
        ch.append(len(sa - sb)); cn.append(len(a) != len(b))
    print(f"{name:28s}  {np.mean(ch):8.2f}                           {np.mean(cn):8.3f}")

# codon alignment of token boundaries inside forward-strand CDS
fcds = [(int(f.location.start), int(f.location.end)) for f in rec.features if f.type == "CDS" and f.location.strand == 1 and len(f.location.parts) == 1 and (int(f.location.end) - int(f.location.start)) % 3 == 0]
def codon_frac(toks_full):
    b = boundaries(toks_full); good = tot = 0
    for s0, e0 in fcds:
        for x in b:
            if s0 < x < e0: tot += 1; good += ((x - s0) % 3 == 0)
    return good / tot
print(f"\nfraction of token boundaries inside forward-strand genes that fall on codon boundaries (chance = 1/3), {len(fcds)} genes:")
print(f"  non-overlapping 3-mers from the genome start: {codon_frac([seq[i:i+3] for i in range(0, L - 2, 3)]):.3f}  (all-or-nothing per gene: {sum(s0 % 3 == 0 for s0, e0 in fcds)} of {len(fcds)} genes happen to start in the tokenizer's frame)")
print(f"  non-overlapping 6-mers from the genome start: {codon_frac([seq[i:i+6] for i in range(0, L - 5, 6)]):.3f}")
tk = res[4096]; enc = tk.encode(seq)
print(f"  BPE (vocabulary 4096), tokenizing the whole genome: {codon_frac(enc.tokens):.3f}")

# overlapping k-mer tokens: a single masked token is determined by its neighbors
k = 6; N = 20000; pos_ = rng.integers(10, len(test_txt) - 20, N)
leak = []
for i in pos_:
    tok_prev, tok_next = test_txt[i - 1:i - 1 + k], test_txt[i + 1:i + 1 + k]         # overlapping 6-mer tokens centred around the masked one at i
    guess = tok_prev[1:2] + tok_next[:k - 1]                                           # base i from the previous token, bases i+1..i+5 from the next token
    leak.append(guess == test_txt[i:i + k])
print(f"\noverlapping {k}-mer MLM: mask ONE token; a rule that copies from the two neighboring tokens recovers the masked token exactly in {np.mean(leak):.3f} of {N} cases")
# contiguous masking of k tokens hides 2k-1 bases; predict them from a left flank with an order-4 Markov model
tr_arr = arr[:cut]; cnt4 = fit([tr_arr], 4)
te_arr = arr[cut:]; correct = tot_ = 0; major = np.bincount(tr_arr).argmax()
for i in rng.integers(10, len(te_arr) - 20, 3000):
    ctx = int(sum(int(te_arr[i - 4 + j]) * 4 ** (3 - j) for j in range(4)))
    for step in range(2 * k - 1):                                                     # greedy generation of the hidden stretch
        pred = int(np.argmax(cnt4[ctx]));  true = int(te_arr[i + step])
        correct += pred == true; tot_ += 1; ctx = (ctx * 4 + pred) % 4 ** 4
print(f"mask k = {k} contiguous tokens (hides {2*k-1} bases): greedy order-4 Markov accuracy per hidden base = {correct / tot_:.3f}; most-common-base baseline = {np.mean(te_arr == major):.3f}; chance = 0.250")

# ---------------------------------------------------------------- 5. reverse-complement symmetry
print("\n== 5. Reverse-complement symmetry: Chargaff's second rule, and where it breaks ==")
sense = "".join(str(f.extract(rec.seq)).upper() for f in rec.features if f.type == "CDS")
def kmer_counts(s, k):
    a = np.array([code[c] for c in s if c in code], np.int64); c, nxt = contexts(a, k - 1)
    return np.bincount(c * 4 + nxt, minlength=4 ** k).astype(float)
def rc_index(k):
    idx = np.arange(4 ** k); out = np.zeros_like(idx)
    for w in idx:
        digits = [(w // 4 ** (k - 1 - j)) % 4 for j in range(k)]; r = [3 - d for d in digits[::-1]]
        out[w] = sum(d * 4 ** (k - 1 - j) for j, d in enumerate(r))
    return out
print("k   whole genome, one strand: corr(count(w), count(rc(w)))   concatenated sense strands of genes only   mean |log2 ratio| genome / genes")
for k in [2, 4, 6]:
    ri = rc_index(k); g = kmer_counts(seq, k); s = kmer_counts(sense, k)
    lg = np.mean(np.abs(np.log2((g + 1) / (g[ri] + 1)))); ls = np.mean(np.abs(np.log2((s + 1) / (s[ri] + 1))))
    print(f"{k}   {np.corrcoef(g, g[ri])[0, 1]:8.4f}                                      {np.corrcoef(s, s[ri])[0, 1]:8.4f}                              {lg:6.3f} / {ls:6.3f}")
