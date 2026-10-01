"""Chapter 32: a small genomic language model on a real genome, compared with Markov baselines.
(1) held-out bits per base on unique DNA and on the reverse-complement copy of a training region (memorization);
(2) zero-shot variant scoring in held-out genes: can the log-likelihood change separate synonymous from nonsynonymous and stop-gain variants?
Data: Arabidopsis chloroplast genome NC_000932 (154,478 bp). Training sequence excludes the test regions."""
import os, urllib.request, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from Bio import SeqIO
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4); torch.manual_seed(0)
rng = np.random.default_rng(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NC_000932.gb")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_000932.gb", path)
rec = SeqIO.read(path, "genbank"); seq = str(rec.seq).upper(); L = len(seq)
code = {c: i for i, c in enumerate("ACGT")}; arr = np.array([code[c] for c in seq], np.int64)
IRb, IRa = (84170, 110434), (128214, 154478)
TEST_U = (60000, 84170)                                                     # unique region of the large single-copy section
TEST_LEAK = IRa                                                             # the reverse-complement copy of IRb, which is in training
train_segs = [arr[:TEST_U[0]], arr[IRb[0]:IRa[0]]]                          # 0-60,000 and the second half of the genome up to (not including) IRa
print(f"training bases: {sum(map(len, train_segs)):,}; test unique region {TEST_U[1]-TEST_U[0]:,} bp; test repeat copy {IRa[1]-IRa[0]:,} bp (reverse complement of a training region)")

# ------------------------------------------------------------ Markov baselines (order 4, with reverse-complement augmentation)
rc_arr = lambda a: (3 - a)[::-1]
def contexts(a, k):
    n = len(a); c = np.zeros(n - k, np.int64)
    for j in range(k): c = c * 4 + a[j:n - k + j]
    return c, a[k:]
def fit_markov(k):
    cnt = np.zeros((4 ** k, 4)) + 0.5
    for t in train_segs + [rc_arr(s) for s in train_segs]:
        c, nx = contexts(t, k); np.add.at(cnt, (c, nx), 1)
    return np.log2(cnt / cnt.sum(1, keepdims=True))
def markov_bits(lp, a, k):
    c, nx = contexts(a, k); return -lp[c, nx].mean()
mk = {k: fit_markov(k) for k in (2, 4, 6, 8)}
print("\n== 1. Held-out cross-entropy (bits per base) ==")
print("model                                   unique test region    repeat copy (RC of a training region)")
tu, tl = arr[TEST_U[0]:TEST_U[1]], arr[TEST_LEAK[0]:TEST_LEAK[1]]
for k in (2, 4, 6, 8): print(f"order-{k} Markov, RC augmentation          {markov_bits(mk[k], tu, k):8.3f}              {markov_bits(mk[k], tl, k):8.3f}")

# ------------------------------------------------------------ a small causal transformer LM over nucleotides
CTX = 128
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.att = nn.MultiheadAttention(d, h, batch_first=True); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        h = self.ln1(x); x = x + self.att(h, h, h, attn_mask=mask, need_weights=False)[0]; return x + self.ff(self.ln2(x))
class DNALM(nn.Module):
    def __init__(self, d=64, h=4, nl=3):
        super().__init__(); self.emb = nn.Embedding(5, d); self.pos = nn.Embedding(CTX, d); self.blocks = nn.ModuleList([Block(d, h) for _ in range(nl)])
        self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 4)
    def forward(self, x):                                                       # x: (B, T) with a start token 4 prepended by the caller
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.emb(x) + self.pos(torch.arange(T))[None]
        for b in self.blocks: h = b(h, mask)
        return self.out(self.ln(h))                                             # logits for the NEXT base at every position
model = DNALM(); print(f"\ntransformer: {sum(p.numel() for p in model.parameters()):,} parameters, context {CTX}, single-nucleotide tokens")
def batch(B):
    xs = []
    for _ in range(B):
        s = train_segs[rng.integers(len(train_segs))]; i = int(rng.integers(0, len(s) - CTX)); w = s[i:i + CTX]
        if rng.random() < 0.5: w = rc_arr(w)                                    # reverse-complement augmentation
        xs.append(w)
    return torch.tensor(np.stack(xs))
opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=0.01); steps = 2500
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
t0 = time.time()
for st in range(steps):
    x = batch(48); inp = torch.cat([torch.full((48, 1), 4), x[:, :-1]], 1)
    loss = F.cross_entropy(model(inp).reshape(-1, 4), x.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
print(f"trained {steps} steps in {time.time() - t0:.0f} s; final training loss {loss.item() / np.log(2):.3f} bits/base")
model.eval()
def lm_bits(a, stride=CTX // 2):
    """Mean bits per base over a sequence, scoring each base once with at least CTX/2 bases of left context where available."""
    tot = n = 0; i = 0
    while i < len(a) - 1:
        w = a[max(0, i - CTX // 2): max(0, i - CTX // 2) + CTX]; x = torch.tensor(w)[None]
        inp = torch.cat([torch.full((1, 1), 4), x[:, :-1]], 1)
        with torch.no_grad(): lp = F.log_softmax(model(inp), -1)[0]
        off = min(i, CTX // 2); lo = off if i > 0 else 0; hi = min(len(w), off + stride)
        tot += -lp[torch.arange(lo, hi), x[0, lo:hi]].sum().item() / np.log(2); n += hi - lo; i += hi - lo
    return tot / n
print(f"transformer LM, RC augmentation             {lm_bits(tu):8.3f}              {lm_bits(tl):8.3f}")

# ------------------------------------------------------------ 2. zero-shot variant scoring in held-out genes
print("\n== 2. Zero-shot variant scoring: log-likelihood change of a 96-base window, genes inside the held-out unique region ==")
from Bio.Data import CodonTable
tab = CodonTable.unambiguous_dna_by_id[11].forward_table; stops = set(CodonTable.unambiguous_dna_by_id[11].stop_codons)
variants = []                                                               # (position, alt base, class)
comp = str.maketrans("ACGT", "TGCA")
for f in rec.features:
    if f.type != "CDS" or len(f.location.parts) != 1: continue
    s0, e0, strand = int(f.location.start), int(f.location.end), f.location.strand
    if s0 < TEST_U[0] + 60 or e0 > TEST_U[1] - 60 or (e0 - s0) % 3: continue
    cds = seq[s0:e0] if strand == 1 else seq[s0:e0][::-1].translate(comp)
    for j in range(len(cds) - 3):                                           # exclude the stop codon itself
        cod = cds[3 * (j // 3):3 * (j // 3) + 3]; pos_in = j % 3
        for b in "ACGT":
            if b == cds[j]: continue
            new = cod[:pos_in] + b + cod[pos_in + 1:]
            cls = "stop-gain" if new in stops else ("synonymous" if tab.get(new) == tab.get(cod) else "missense")
            gpos = s0 + j if strand == 1 else e0 - 1 - j; galt = b if strand == 1 else b.translate(comp)
            variants.append((gpos, "ACGT".index(galt), cls))
print(f"{len(variants):,} single-nucleotide variants in held-out genes: " + ", ".join(f"{c} {sum(v[2]==c for v in variants):,}" for c in ("synonymous", "missense", "stop-gain")))
sub = []                                                                    # score a stratified subsample: all stop-gain, 2,500 synonymous, 2,500 missense
for c, k in (("stop-gain", None), ("synonymous", 2500), ("missense", 2500)):
    pool = [v for v in variants if v[2] == c]; sub += pool if k is None else [pool[i] for i in rng.choice(len(pool), min(k, len(pool)), replace=False)]
variants = sub; print(f"scoring a stratified subsample of {len(variants):,} variants")
W = 48
def window_logp_markov(a, k):
    c, nx = contexts(a, k); return mk[k][c, nx].sum() * 1.0
def window_logp_lm(Xw):
    inp = torch.cat([torch.full((len(Xw), 1), 4), Xw[:, :-1]], 1)
    with torch.no_grad(): lp = F.log_softmax(model(inp), -1)
    return lp.gather(2, Xw[:, :, None])[..., 0].sum(1).numpy()
# the transformer sees up to CTX bases of left context; both models score the SAME window of 2W+1 bases, using preceding context where needed
CTXL = CTX
def llr_lm(batch_v):
    refs, alts = [], []
    for (g, b, c) in batch_v:
        lo = g - CTXL + W + 1; w = arr[lo:lo + CTXL].copy(); refs.append(w.copy()); w[g - lo] = b; alts.append(w)
    R = window_logp_lm(torch.tensor(np.stack(refs))); A = window_logp_lm(torch.tensor(np.stack(alts))); return R - A     # drop in log-likelihood caused by the variant
def llr_markov(batch_v, k=4):
    out = []
    for (g, b, c) in batch_v:
        w = arr[g - W - k:g + W + 1].copy(); r = window_logp_markov(w, k); w[W + k] = b; out.append(r - window_logp_markov(w, k))
    return np.array(out)
S_lm = np.concatenate([llr_lm(variants[i:i + 256]) for i in range(0, len(variants), 256)]); S_mk = llr_markov(variants)
cls = np.array([v[2] for v in variants])
def auc(score, pos, neg):
    m = np.isin(cls, [pos, neg]); return roc_auc_score(cls[m] == pos, score[m])
print("AUROC for ranking a variant class above synonymous variants by the drop in log-likelihood (0.5 = no signal)")
print("class vs synonymous        order-4 Markov (no frame)    transformer LM")
for c in ("missense", "stop-gain"): print(f"{c:12s}               {auc(S_mk, c, 'synonymous'):8.3f}                  {auc(S_lm, c, 'synonymous'):8.3f}")
# frame-aware baseline that knows the codon table: uses the annotation (not available to the LMs)
print("(reference: a rule that knows the genetic code separates stop-gain from synonymous with AUROC 1.000 by construction)")
