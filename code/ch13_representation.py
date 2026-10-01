"""Chapter 13: the objective decides what a representation keeps.

Synthetic 'loci'.  Each ancestral locus carries a conserved functional signal s in {0,1,2,3}: one of four 6-mer
motif variants at positions 10-15.  The rest of the sequence is background with a locus-specific GC bias, which is
large nuisance variation.  Two 'orthologs' of a locus share s but have independently drifted backgrounds.
We learn k-dimensional representations from UNLABELED sequences with
  - a reconstruction autoencoder (objective: reproduce the sequence),
  - InfoNCE contrastive learning with orthologs as positives (objective: identify the ortholog among negatives),
and read them out with linear probes: the signal s (4-way accuracy, chance 0.25) and the nuisance GC content (R^2).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge

torch.manual_seed(0); rng = np.random.default_rng(0)
L = 60
MOTIFS = np.array([[3, 0, 2, 1, 3, 2], [0, 3, 1, 2, 0, 1], [2, 2, 3, 0, 1, 3], [1, 3, 0, 0, 2, 2]])   # four 6-mer variants

def background(n, gc):
    p = np.stack([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2], 1)               # A, C, G, T
    return (rng.random((n, L))[..., None] > np.cumsum(p, 1)[:, None, :]).sum(-1)

def make_pairs(n):
    s = rng.integers(0, 4, n); out = []
    for _ in range(2):                                                          # two orthologs of each locus
        gc = rng.uniform(0.1, 0.9, n); X = background(n, gc); X[:, 10:16] = MOTIFS[s]
        out.append((F.one_hot(torch.tensor(X), 4).float().reshape(n, -1), gc))
    return out, s

(tr, s_tr), (te, s_te) = make_pairs(6000), make_pairs(2000)
(XA, gcA), (XB, _), (Xte, gc_te) = tr[0], tr[1], te[0]

def encoder(k): return nn.Sequential(nn.Linear(4 * L, 256), nn.ReLU(), nn.Linear(256, k))

def train_ae(k, steps=800):
    enc = encoder(k); dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 4 * L))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 2e-3); Xall = torch.cat([XA, XB])
    for _ in range(steps):
        x = Xall[torch.randint(0, len(Xall), (256,))]
        loss = F.cross_entropy(dec(enc(x)).view(-1, 4), x.view(-1, 4).argmax(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return enc

def train_contrastive(k, steps=800, tau=0.2):
    enc = encoder(k); opt = torch.optim.Adam(enc.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(XA), (256,))
        za, zb = F.normalize(enc(XA[idx]), dim=-1), F.normalize(enc(XB[idx]), dim=-1)
        logits = za @ zb.T / tau                                                # diagonal = true ortholog pairs
        loss = 0.5 * (F.cross_entropy(logits, torch.arange(256)) + F.cross_entropy(logits.T, torch.arange(256)))
        opt.zero_grad(); loss.backward(); opt.step()
    return enc

def probe(enc):
    with torch.no_grad(): Ztr, Zte = enc(XA).numpy(), enc(Xte).numpy()
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8; Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    acc = LogisticRegression(max_iter=3000).fit(Ztr, s_tr).score(Zte, s_te)
    r2 = Ridge(alpha=1.0).fit(Ztr, gcA).score(Zte, gc_te)
    return acc, r2

raw_acc = LogisticRegression(max_iter=3000).fit(XA.numpy(), s_tr).score(Xte.numpy(), s_te)
print(f"raw one-hot input, linear probe for the conserved signal s: accuracy = {raw_acc:.3f} (chance 0.25)")
print(f"{'k':>3s} | {'autoencoder: signal acc / GC R^2':>34s} | {'contrastive: signal acc / GC R^2':>34s}")
for k in (1, 2, 4, 16):
    a_acc, a_r2 = probe(train_ae(k)); c_acc, c_r2 = probe(train_contrastive(k))
    print(f"{k:3d} | {a_acc:>16.3f} / {a_r2:>6.3f}       | {c_acc:>16.3f} / {c_r2:>6.3f}")
