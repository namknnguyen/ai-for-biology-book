"""Chapter 31: a sequence-to-function model can predict well and still get the effect of a single edit wrong.
Toy regulatory locus: motif A (activator, +1.0) and motif B (repressor, -0.8) co-occur with rate c (composite element); motif C (+0.5) is independent.
Training data are reference-like sequences; the test of interest is a *counterfactual*: destroy A alone or B alone.
We vary the co-occurrence c, the number of training loci, and add a small set of MPRA-style single-edit experiments."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
import sys; sys.stdout.reconfigure(line_buffering=True)
rng = np.random.default_rng(0); torch.manual_seed(0)
L, ML = 200, 8
MA, MB, MC = [rng.integers(0, 4, ML) for _ in range(3)]
W = {"A": 1.0, "B": -0.8, "C": 0.5}

def place(seq, motif, occupied):
    for _ in range(100):
        p = int(rng.integers(0, L - ML))
        if not any(abs(p - q) < ML + 2 for q in occupied):
            seq[p:p + ML] = motif; occupied.append(p); return p
    return None

def make_gene(c):
    seq = rng.integers(0, 4, L); occ = []
    a = rng.random() < 0.5; b = (rng.random() < c) if a else (rng.random() < 1 - c); cc = rng.random() < 0.5
    pa = place(seq, MA, occ) if a else None
    if b:
        if a and pa is not None and pa + ML + 10 < L - ML: seq[pa + ML + 2:pa + 2 * ML + 2] = MB; occ.append(pa + ML + 2)    # composite: B immediately after A
        else: place(seq, MB, occ)
    if cc: place(seq, MC, occ)
    y = W["A"] * a + W["B"] * b + W["C"] * cc + 0.3 * rng.standard_normal()
    return seq, y, a, b, cc
def make_set(n, c):
    out = [make_gene(c) for _ in range(n)]; return np.stack([o[0] for o in out]), np.array([o[1] for o in out])

def find(seq, motif):
    for p in range(L - ML + 1):
        if np.array_equal(seq[p:p + ML], motif): return p
    return None
def destroy(seq, motif):
    s = seq.copy(); p = find(s, motif)
    if p is None: return None
    new = rng.integers(0, 4, ML)
    while np.array_equal(new, motif): new = rng.integers(0, 4, ML)
    s[p:p + ML] = new; return s

def onehot(S): return torch.tensor(np.eye(4, dtype=np.float32)[S]).permute(0, 2, 1)
class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv1d(4, 32, 9, padding=4); self.c2 = nn.Conv1d(32, 32, 5, padding=2); self.fc = nn.Linear(64, 1)
    def forward(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); return self.fc(torch.cat([h.max(2)[0], h.mean(2)], 1)).squeeze(1)
def train(S, y, epochs=40):
    X = onehot(S); Y = torch.tensor(y, dtype=torch.float32); m = CNN(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]; loss = F.mse_loss(m(X[idx]), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
def counterfactual(m, c):
    # genes carrying both A and B; edit each motif alone
    both = []
    while len(both) < 400:
        s, y, a, b, cc = make_gene(c)
        if a and b: both.append(s)
    res = {}
    for name, mot, truth in [("A", MA, -W["A"]), ("B", MB, -W["B"])]:
        ref = [s for s in both if destroy(s, mot) is not None]; mut = [destroy(s, mot) for s in ref]
        with torch.no_grad(): d = (m(onehot(np.stack(mut))) - m(onehot(np.stack(ref)))).numpy()
        res[name] = (np.mean(np.sign(d) == np.sign(truth)), d.mean(), truth)
    return res
def mpra_set(n, c):
    """MPRA-style experiment: genes carrying A and B, with one of the two motifs destroyed and the expression measured."""
    S, Y = [], []
    while len(S) < n:
        s, y, a, b, cc = make_gene(c)
        if not (a and b): continue
        which = MA if len(S) % 2 == 0 else MB; s2 = destroy(s, which)
        if s2 is None: continue
        a2 = a - (len(S) % 2 == 0); b2 = b - (len(S) % 2 == 1)
        S.append(s2); Y.append(W["A"] * a2 + W["B"] * b2 + W["C"] * cc + 0.3 * rng.standard_normal())
    return np.stack(S), np.array(Y)

print(f"true effects of destroying a motif: A -> {-W['A']:+.1f}, B -> {-W['B']:+.1f} (destroying the repressor B raises expression); noise SD 0.3 (the RMSE floor).\n")
print("training loci   co-occurrence c   corr(A,B)   i.i.d. test RMSE (R2)   sign correct: A deletion   B deletion   mean predicted effect: A (true -1.0)   B (true +0.8)")
configs = [(6000, 0.5), (6000, 0.9), (6000, 0.99), (6000, 1.0), (40000, 0.99), (40000, 1.0)]
for n, c in configs:
    S, y = make_set(n, c); St, yt = make_set(2000, c)
    m = train(S, y, epochs=40 if n <= 6000 else 12)
    with torch.no_grad(): p = m(onehot(St)).numpy()
    r2 = 1 - np.mean((p - yt) ** 2) / np.var(yt); rmse = np.sqrt(np.mean((p - yt) ** 2)); cf = counterfactual(m, c)
    print(f"{n:9d}       {c:8.2f}         {2 * c - 1:6.2f}       {rmse:6.3f} ({r2:5.3f})          {cf['A'][0]:6.2f}                 {cf['B'][0]:6.2f}                {cf['A'][1]:+7.2f}                         {cf['B'][1]:+7.2f}")
# remedy: add a small number of single-edit experiments (MPRA-like) to the training set at c = 1.0
for n_mpra in [100, 400]:
    S, y = make_set(6000, 1.0); Sm, ym = mpra_set(n_mpra, 1.0); S = np.concatenate([S, Sm]); y = np.concatenate([y, ym])
    m = train(S, y, epochs=40); St, yt = make_set(2000, 1.0)
    with torch.no_grad(): p = m(onehot(St)).numpy()
    r2 = 1 - np.mean((p - yt) ** 2) / np.var(yt); rmse = np.sqrt(np.mean((p - yt) ** 2)); cf = counterfactual(m, 1.0)
    print(f"{6000:5d} + {n_mpra:3d} MPRA    1.00            1.00       {rmse:6.3f} ({r2:5.3f})          {cf['A'][0]:6.2f}                 {cf['B'][0]:6.2f}                {cf['A'][1]:+7.2f}                         {cf['B'][1]:+7.2f}")
