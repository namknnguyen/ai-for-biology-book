"""Chapter 33: thermodynamic RNA folding as a classical baseline, tested on the 37 tRNA genes of a real genome (Arabidopsis chloroplast, NC_000932).
(1) Does the minimum-free-energy structure recover the universal cloverleaf (acceptor stem + three hairpins)?
(2) Is 'folds strongly' a detector of structured RNA? The answer depends on composition: raw MFE vs shuffle z-score vs GC-matched controls."""
import os, urllib.request
import numpy as np, RNA
from Bio import SeqIO
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NC_000932.gb")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_000932.gb", path)
rec = SeqIO.read(path, "genbank"); seq = str(rec.seq).upper(); L = len(seq)
trnas = []; occupied = np.zeros(L, bool)
for f in rec.features:
    if f.type in ("tRNA", "rRNA"): occupied[int(f.location.start):int(f.location.end)] = True
for f in rec.features:
    if f.type == "tRNA" and len(f.location.parts) == 1:
        s = str(f.extract(rec.seq)).upper().replace("T", "U")
        if 60 <= len(s) <= 95 and set(s) <= set("ACGU"): trnas.append(s)
print(f"{len(trnas)} tRNA genes with a single exon and length 60-95 nt (mean length {np.mean([len(t) for t in trnas]):.1f}, mean GC {np.mean([(t.count('G')+t.count('C'))/len(t) for t in trnas]):.3f})")

def fold(s): return RNA.fold_compound(s).mfe()
def groups(struct):                                                                # top-level balanced groups
    depth = 0; g = 0
    for ch in struct:
        if ch == "(":
            if depth == 0: g += 1
            depth += 1
        elif ch == ")": depth -= 1
    return g
def pairs(struct):
    st, p = [], {}
    for i, ch in enumerate(struct):
        if ch == "(": st.append(i)
        elif ch == ")": j = st.pop(); p[i] = j; p[j] = i
    return p
acc = clover = 0; mfes = []
for s in trnas:
    st, e = fold(s); mfes.append(e); p = pairs(st); n = len(s)
    ok = sum(1 for i in range(7) if i in p and p[i] >= n - 12); acc += ok >= 6
    k = 0
    while k in p and p[k] > k and (k == 0 or p[k] < p[k - 1]): k += 1                 # consecutive pairs of the terminal (acceptor) stem
    inner = st[k:p[k - 1]] if k >= 6 else ""
    clover += (k >= 6 and groups(inner) == 3)
print(f"\n== 1. MFE structures of the {len(trnas)} tRNAs ==")
print(f"acceptor stem recovered (>=6 of the 7 terminal base pairs): {acc}/{len(trnas)} = {acc/len(trnas):.2f}")
print(f"full cloverleaf topology (acceptor stem enclosing exactly three hairpins): {clover}/{len(trnas)} = {clover/len(trnas):.2f}")
print(f"mean MFE {np.mean(mfes):.1f} kcal/mol ({np.mean(mfes)/np.mean([len(t) for t in trnas]):.3f} per nucleotide)")

def markov_shuffle(s, n=100):
    idx = {c: i for i, c in enumerate("ACGU")}; T = np.full((4, 4), 0.5)
    for a, b in zip(s[:-1], s[1:]): T[idx[a], idx[b]] += 1
    T /= T.sum(1, keepdims=True); out = []
    for _ in range(n):
        x = [rng.integers(4)]
        for _ in range(len(s) - 1): x.append(rng.choice(4, p=T[x[-1]]))
        out.append("".join("ACGU"[i] for i in x))
    return out
def stats(s):
    e = fold(s)[1]; sh = [fold(t)[1] for t in markov_shuffle(s, 60)]; z = (e - np.mean(sh)) / (np.std(sh) + 1e-9)
    gc = (s.count("G") + s.count("C")) / len(s); return e / len(s), z, gc
pos = np.array([stats(s) for s in trnas])
# controls: 75-nt windows from sequence outside tRNA/rRNA genes (coding and intergenic), and a GC-matched subset
ctrl = []; tries = 0
while len(ctrl) < 1500 and tries < 40000:
    tries += 1; i = int(rng.integers(0, L - 75)); w = seq[i:i + 75]
    if occupied[i:i + 75].any() or set(w) - set("ACGT"): continue
    ctrl.append(w.replace("T", "U"))
neg = np.array([stats(s) for s in ctrl])
gc_t = pos[:, 2]; lo, hi = gc_t.min(), gc_t.max()
matched = neg[(neg[:, 2] >= lo) & (neg[:, 2] <= hi)]
print(f"\n== 2. Detecting tRNAs among {len(neg)} 75-nt windows outside tRNA/rRNA genes (coding and non-coding) (GC of windows {neg[:,2].mean():.3f}; tRNAs {gc_t.mean():.3f}); GC-matched subset: {len(matched)} windows ==")
def auroc(sc_pos, sc_neg): return roc_auc_score(np.r_[np.ones(len(sc_pos)), np.zeros(len(sc_neg))], np.r_[sc_pos, sc_neg])
print("detector                               AUROC vs all windows    AUROC vs GC-matched windows")
print(f"GC content                              {auroc(pos[:,2], neg[:,2]):8.3f}                 {auroc(pos[:,2], matched[:,2]):8.3f}")
print(f"MFE per nucleotide (more negative)      {auroc(-pos[:,0], -neg[:,0]):8.3f}                 {auroc(-pos[:,0], -matched[:,0]):8.3f}")
print(f"MFE z-score vs Markov shuffles          {auroc(-pos[:,1], -neg[:,1]):8.3f}                 {auroc(-pos[:,1], -matched[:,1]):8.3f}")
