"""Chapter 48: mechanistic interpretability on a model whose true circuit is known.
Task: label = 1 iff motif A AND motif B are both present in a 50-base sequence.  Two further motifs are planted: S, a SHORTCUT (present in 95% of positives and 5% of negatives in the training
distribution) and D, a DISTRACTOR that co-occurs with motif A in the training distribution (95%) but is otherwise unrelated to the label.  A small CNN (16 filters of width 6, ReLU, global max pool, one hidden layer) is trained on the shortcut
distribution.  We then ask what each interpretability method says:  (1) behavior on shortcut-free data, (2) filter-to-motif matching, (3) ablation of each filter, (4) linear probes versus causal tests,
(5) activation patching on counterfactual pairs, (6) a sparse autoencoder on the pooled representation."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)
L, W_ = 50, 6; MOT = {"A": "TGACGT", "B": "GCCATA", "S": "AATTCC", "D": "CCGGAA"}; code = {c: i for i, c in enumerate("ACGT")}
enc = lambda s: np.array([code[c] for c in s])
def make(n, shortcut_rho, seed):
    r = np.random.default_rng(seed); X = r.integers(0, 4, (n, L)); y = r.integers(0, 2, n); pres = {k: np.zeros(n, bool) for k in MOT}
    for i in range(n):
        has = {"A": False, "B": False}
        if y[i]: has = {"A": True, "B": True}
        else:
            c = r.integers(0, 3); has = {"A": c == 0, "B": c == 1}                                              # negatives: only A, only B, or neither
        s_prob = (0.95 if y[i] else 0.05) if shortcut_rho else 0.5
        has["S"] = r.random() < s_prob; has["D"] = (r.random() < (0.95 if has["A"] else 0.05)) if shortcut_rho else r.random() < 0.5
        slots = r.permutation(8)[:4]; starts = dict(zip("ABSD", slots))
        for k in "ABSD":
            if has[k]: p = starts[k] * 6 + r.integers(0, 1); X[i, p:p + 6] = enc(MOT[k]); pres[k][i] = True
    return X, y, pres
Xtr, ytr, ptr = make(8000, True, 1); Xte, yte, pte = make(3000, True, 2); Xfr, yfr, pfr = make(3000, False, 3)                    # train, in-distribution test, shortcut-free test
class Net(nn.Module):
    def __init__(self, nf=16):
        super().__init__(); self.conv = nn.Conv1d(4, nf, W_); self.h = nn.Linear(nf, 16); self.out = nn.Linear(16, 2)
    def pooled(self, x): return F.relu(self.conv(F.one_hot(x, 4).float().transpose(1, 2))).max(2).values              # (B, nf): the motif-detector layer
    def hidden(self, p): return F.relu(self.h(p))
    def forward(self, x, patch=None):
        p = self.pooled(x)
        if patch is not None: p = patch(p)
        return self.out(self.hidden(p))
EPOCHS = 5
net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3); Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        idx = perm[i:i + 128]; loss = F.cross_entropy(net(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
net.eval(); acc = lambda X, y, f=None: (net(torch.tensor(X), patch=f).argmax(1).numpy() == y).mean()
print("== 1. Behavior: the model was trained where the shortcut S predicts the label 95% of the time and D co-occurs with A ==")
print(f"accuracy on in-distribution test data: {acc(Xte, yte):.3f};  on shortcut-free data (S independent of the label): {acc(Xfr, yfr):.3f}")
Xs1 = Xfr.copy(); m = pfr["S"] & (yfr == 0); print(f"among shortcut-free NEGATIVES, accuracy when S is present: {acc(Xfr[m], yfr[m]):.3f}, when S is absent: {acc(Xfr[~pfr['S'] & (yfr == 0)], yfr[~pfr['S'] & (yfr == 0)]):.3f}")
m = pfr["S"] == 0; print(f"among shortcut-free POSITIVES, accuracy when S is present: {acc(Xfr[pfr['S'] & (yfr == 1)], yfr[pfr['S'] & (yfr == 1)]):.3f}, when S is absent: {acc(Xfr[~pfr['S'] & (yfr == 1)], yfr[~pfr['S'] & (yfr == 1)]):.3f}")

# ---------------------------------------------------------------- 2. filter-to-motif matching and 3. ablation
with torch.no_grad(): P = net.pooled(torch.tensor(Xfr)).numpy()
print("\n== 2-3. What does each of the 16 filters respond to, and what happens when it is ablated (its pooled activation fixed at its mean)? ==")
print("filter   most associated motif (AUROC of its activation for motif presence)   accuracy shortcut-free after ablation   change")
base = acc(Xfr, yfr); rows = []
for f in range(16):
    aucs = {k: roc_auc_score(pfr[k], P[:, f]) for k in MOT}; k = max(aucs, key=lambda z: abs(aucs[z] - 0.5))
    mean_f = float(P[:, f].mean()); a = acc(Xfr, yfr, lambda p, f=f, mf=mean_f: torch.cat([p[:, :f], torch.full_like(p[:, f:f + 1], mf), p[:, f + 1:]], 1)); rows.append((f, k, aucs[k], a))
for f, k, au, a in sorted(rows, key=lambda r: r[3])[:8]: print(f"{f:5d}    {k}  ({au:.2f})                                                          {a:.3f}                            {a - base:+.3f}")
print(f"(the remaining 8 filters change accuracy by at most {max(abs(r[3] - base) for r in sorted(rows, key=lambda r: r[3])[8:]):.3f})")

# ---------------------------------------------------------------- 4. probes versus causal tests
print("\n== 4. Linear probes decode every motif; only some are used ==")
with torch.no_grad(): Ptr = net.pooled(torch.tensor(Xtr)).numpy(); Pte = net.pooled(torch.tensor(Xfr)).numpy()
print("motif   probe AUROC (shortcut-free data)   accuracy after erasing the probe direction: shortcut-free data | training distribution")
for k in "ABSD":
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Ptr, ptr[k]); au = roc_auc_score(pfr[k], clf.decision_function(Pte)); w = clf.coef_[0]; w = w / np.linalg.norm(w)
    wt = torch.tensor(w, dtype=torch.float32); a = acc(Xfr, yfr, lambda p, wt=wt: p - (p @ wt)[:, None] * wt[None, :])             # project out the probe direction
    a_in = acc(Xte, yte, lambda p, wt=wt: p - (p @ wt)[:, None] * wt[None, :])
    print(f"{k}       {au:.3f}                      {a:.3f} (change {a - base:+.3f})                         {a_in:.3f} (change {a_in - acc(Xte, yte):+.3f})")

# ---------------------------------------------------------------- 5. activation patching on counterfactual pairs
print("\n== 5. Activation patching: clean = positive sequence with A and B; corrupted = the same sequence with motif A replaced by random bases ==")
def corrupt(X, which):
    Xc = X.copy()
    for i in range(len(X)):
        for p in range(0, L - 5):
            if (Xc[i, p:p + 6] == enc(MOT[which])).all(): Xc[i, p:p + 6] = rng.integers(0, 4, 6)
    return Xc
sel = np.flatnonzero((yfr == 1) & pfr["A"] & pfr["B"])[:600]; Xc_ = Xfr[sel]; Xk = corrupt(Xc_, "A")
with torch.no_grad(): lc = net(torch.tensor(Xc_)); lk = net(torch.tensor(Xk)); pc = net.pooled(torch.tensor(Xc_))
ld = lambda l: (l[:, 1] - l[:, 0]); d_clean, d_corr = ld(lc).mean().item(), ld(lk).mean().item()
print(f"mean logit difference (positive minus negative): clean {d_clean:.2f}, corrupted {d_corr:.2f}  (corrupting motif A flips {np.mean(lk.argmax(1).numpy() == 0):.2f} of the predictions)")
print("patching one filter's pooled activation from the clean run into the corrupted run: fraction of the logit difference recovered")
rec = []
for f in range(16):
    with torch.no_grad(): lp = net(torch.tensor(Xk), patch=lambda p, f=f: torch.cat([p[:, :f], pc[:, f:f + 1], p[:, f + 1:]], 1))
    rec.append((ld(lp).mean().item() - d_corr) / (d_clean - d_corr))
for f in np.argsort(rec)[::-1][:4]: print(f"   filter {f} (matches {rows[f][1]}): recovered {rec[f]:.2f}")
print(f"   all remaining filters: at most {sorted(rec)[-5]:.2f}")

# ---------------------------------------------------------------- 6. sparse autoencoder on the pooled representation
print("\n== 6. A sparse autoencoder (64 latents, L1 penalty) on the 16 pooled activations ==")
Z = torch.tensor(Ptr, dtype=torch.float32); torch.manual_seed(1)
enc_ = nn.Linear(16, 64); dec_ = nn.Linear(64, 16, bias=False); optz = torch.optim.Adam(list(enc_.parameters()) + list(dec_.parameters()), 3e-3)
for st in range(3000):
    idx = torch.randint(0, len(Z), (256,)); z = F.relu(enc_(Z[idx])); loss = ((dec_(z) - Z[idx]) ** 2).mean() + 3e-3 * z.abs().mean(); optz.zero_grad(); loss.backward(); optz.step()
with torch.no_grad(): H = F.relu(enc_(torch.tensor(Pte))).numpy(); recon = ((dec_(torch.tensor(H)).numpy() - Pte) ** 2).mean() / Pte.var()
alive = (H > 1e-3).mean(0) > 0.01
def best_sel(M, pres, names):
    out = {}
    for k in names:
        aus = [abs(roc_auc_score(pres[k], M[:, j]) - 0.5) + 0.5 for j in range(M.shape[1]) if M[:, j].std() > 1e-6]; out[k] = max(aus)
    return out
bn, bs = best_sel(Pte, pfr, "ABSD"), best_sel(H[:, alive], pfr, "ABSD")
print(f"{alive.sum()} active latents, fraction of variance unexplained {recon:.3f}, mean L0 {(H > 1e-3).sum(1).mean():.1f}")
print("motif   best single NEURON AUROC (of 16)   best single SAE LATENT AUROC (of %d active)" % alive.sum())
for k in "ABSD": print(f"{k}       {bn[k]:.3f}                              {bs[k]:.3f}")

# ---------------------------------------------------------------- 7. control with real superposition: 24 independent motifs, only 8 filters
print("\n== 7. Superposition control: 24 independently planted 6-mers (each present with probability 0.04, about one per sequence), a model with only 8 filters trained to report which are present ==")
rs = np.random.default_rng(5); mots = ["".join(rs.choice(list("ACGT"), 6)) for _ in range(24)]
def make_multi(n, seed):
    r = np.random.default_rng(seed); X = r.integers(0, 4, (n, L)); Y = np.zeros((n, 24), np.float32)
    for i in range(n):
        slots = r.permutation(8)
        for j in range(24):
            if r.random() < 0.04 and (Y[i].sum() < 8):
                used = int(Y[i].sum()); p = slots[used] * 6; X[i, p:p + 6] = enc(mots[j]); Y[i, j] = 1
    return X, Y
Xm, Ym = make_multi(12000, 11); Xmt, Ymt = make_multi(3000, 12)
class MNet(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv1d(4, 8, W_); self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 24))
    def pooled(self, x): return F.relu(self.conv(F.one_hot(x, 4).float().transpose(1, 2))).max(2).values
    def forward(self, x): return self.mlp(self.pooled(x))
torch.manual_seed(2); mn = MNet(); om = torch.optim.Adam(mn.parameters(), 3e-3); Xt_, Yt_ = torch.tensor(Xm), torch.tensor(Ym)
for ep in range(40):
    perm = torch.randperm(len(Xt_))
    for i in range(0, len(Xt_), 128):
        idx = perm[i:i + 128]; loss = F.binary_cross_entropy_with_logits(mn(Xt_[idx]), Yt_[idx]); om.zero_grad(); loss.backward(); om.step()
with torch.no_grad(): out = torch.sigmoid(mn(torch.tensor(Xmt))).numpy(); Pm = mn.pooled(torch.tensor(Xm)).numpy(); Pmt = mn.pooled(torch.tensor(Xmt)).numpy()
au_model = np.mean([roc_auc_score(Ymt[:, j], out[:, j]) for j in range(24)])
Zm = torch.tensor(Pm, dtype=torch.float32)
def best_per_motif(M): return np.array([max(abs(roc_auc_score(Ymt[:, j], M[:, k]) - 0.5) + 0.5 for k in range(M.shape[1]) if M[:, k].std() > 1e-6) for j in range(24)])
bn_ = best_per_motif(Pmt)
print(f"the model reports the motifs with mean AUROC {au_model:.3f} through an 8-dimensional bottleneck (24 features in 8 dimensions)")
print(f"best single NEURON per motif: mean AUROC {bn_.mean():.3f} (motifs with a neuron above 0.9: {(bn_ > 0.9).sum()} of 24)")
print("sparse autoencoder with 96 latents, by L1 coefficient:  active latents   mean L0   unexplained variance   mean best-latent AUROC   motifs with a latent above 0.9")
for l1 in (1e-3, 1e-2, 1e-1, 3e-1, 1.0):
    torch.manual_seed(3); e2 = nn.Linear(8, 96); d2 = nn.Linear(96, 8, bias=False); o2 = torch.optim.Adam(list(e2.parameters()) + list(d2.parameters()), 3e-3)
    for st in range(4000):
        idx = torch.randint(0, len(Zm), (256,)); z = F.relu(e2(Zm[idx])); l2 = ((d2(z) - Zm[idx]) ** 2).mean() + l1 * z.abs().mean(); o2.zero_grad(); l2.backward(); o2.step()
    with torch.no_grad(): Hm = F.relu(e2(torch.tensor(Pmt))).numpy(); rec2 = ((d2(torch.tensor(Hm)).numpy() - Pmt) ** 2).mean() / Pmt.var()
    live = (Hm > 1e-3).mean(0) > 0.01; bs_ = best_per_motif(Hm[:, live])
    print(f"                                       l1 = {l1:<6g}      {live.sum():6d}       {(Hm > 1e-3).sum(1).mean():5.1f}          {rec2:8.3f}                {bs_.mean():8.3f}                  {(bs_ > 0.9).sum():2d} of 24")
