"""Chapter 38: a miniature single-cell foundation model against classical baselines, on simulated atlases with known cell types, studies, and a novel cell type.
Pretraining corpus: 6 studies (different depths and batch effects), 11 cell types.  Held-out NEW study: 5 of the known types and 1 NOVEL type, shallower sequencing, its own batch effect.
Mini-FM: gene tokens ordered by expression rank (Geneformer-style rank-value encoding after dividing by gene-wise pretraining medians), a 4-layer transformer, masked-gene pretraining on the
corpus only, mean-pooled cell embedding used ZERO-SHOT on the new study.  Baselines are fit on corpus + new study together (they see the new data; the FM does not):
PCA on log-normalised counts, and the negative-binomial VAE of Chapter 30 (no batch covariate).  Metrics: cell-type label transfer from corpus cells to the new study, detection of the novel type,
and how strongly the embedding separates studies within a cell type."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import importlib.util, os, sys
torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
here = os.path.dirname(os.path.abspath(__file__)); spec = importlib.util.spec_from_file_location("ch30", os.path.join(here, "ch30_single_cell_methods.py")); ch30 = importlib.util.module_from_spec(spec); sys.modules["ch30"] = ch30; spec.loader.exec_module(ch30)
G = 1000
# ---------------------------------------------------------------- simulation
K = 12; lineage = np.repeat(np.arange(4), 3)[:K]                                         # 4 lineages x 3 subtypes (types 0..11); type 11 is held out of the corpus
base = rng.lognormal(0, 1.4, G); base /= base.sum()
lin_eff = np.where(rng.random((4, G)) < 0.20, rng.normal(0, 1.0, (4, G)), 0.0); sub_eff = np.where(rng.random((K, G)) < 0.10, rng.normal(0, 0.8, (K, G)), 0.0)
type_eff = lin_eff[lineage] + sub_eff
def study(n, types, depth, bsd, seed):
    r = np.random.default_rng(seed); k = r.choice(types, n); bat = np.exp(r.normal(0, bsd, G))
    rho = base[None, :] * np.exp(type_eff[k]) * bat[None, :]; rho /= rho.sum(1, keepdims=True)
    lib = r.lognormal(np.log(depth), 0.3, n); mu = lib[:, None] * rho; return r.poisson(r.gamma(1 / 0.3, mu * 0.3)).astype(np.float32), k
corpus_types = list(range(11)); depths = [3000, 2500, 4000, 2000, 3500, 3000]
def corpus(n_per):
    Ys, ks, ss = [], [], []
    for s in range(6): Y, k = study(n_per, corpus_types, depths[s], 0.4, 10 + s); Ys.append(Y); ks.append(k); ss.append(np.full(n_per, s))
    return np.vstack(Ys), np.concatenate(ks), np.concatenate(ss)
NEW_TYPES = [3, 4, 5, 6, 7, 11]; Ynew, knew = study(2400, NEW_TYPES, 1200, 0.5, 99)                               # shallow, own batch effect, contains the novel type 11
Ycorp, kcorp, scorp = corpus(3300)                                                                                  # 19,800 cells
print(f"corpus: {len(Ycorp):,} cells, 6 studies, 11 cell types, mean UMI {Ycorp.sum(1).mean():.0f};  new study: {len(Ynew):,} cells, mean UMI {Ynew.sum(1).mean():.0f}, types {NEW_TYPES} (type 11 never seen in the corpus)")

# ---------------------------------------------------------------- mini foundation model
TOP = 96; PAD = G; MASK = G + 1
def rank_tokens(Y, med):
    Z = Y / Y.sum(1, keepdims=True) * 1e4 / med[None, :]; Z[Y == 0] = -1                                             # normalise by gene-wise corpus median of nonzero values; zeros ranked last
    order = np.argsort(-Z, 1)[:, :TOP]; nz = np.take_along_axis(Y, order, 1) > 0; return np.where(nz, order, PAD)
class MiniFM(nn.Module):
    def __init__(self, d=64, nl=4, h=4):
        super().__init__(); self.emb = nn.Embedding(G + 2, d); self.pos = nn.Embedding(TOP, d)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True); self.enc = nn.TransformerEncoder(layer, nl); self.out = nn.Linear(d, G + 2)
    def hidden(self, tok):
        pad = tok == PAD; x = self.emb(tok) + self.pos(torch.arange(TOP))[None]; return self.enc(x, src_key_padding_mask=pad), pad
    def embed(self, tok):
        h, pad = self.hidden(tok); keep = (~pad).float()[:, :, None]; return (h * keep).sum(1) / keep.sum(1).clamp(min=1)
def pretrain(Y, steps, seed=0):
    torch.manual_seed(seed); med = np.array([np.median(Y[Y[:, g] > 0, g] / Y[Y[:, g] > 0].sum(1) * 1e4) if (Y[:, g] > 0).sum() > 5 else 1.0 for g in range(G)]) + 1e-6
    tok = torch.tensor(rank_tokens(Y, med)); net = MiniFM(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
    for st in range(steps):
        idx = torch.randint(0, len(tok), (128,)); x = tok[idx]; m = (torch.rand(x.shape) < 0.15) & (x != PAD); inp = torch.where(m, torch.full_like(x, MASK), x)
        h, pad = net.hidden(inp); loss = F.cross_entropy(net.out(h)[m], x[m]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net, med, loss.item()
def fm_embed(net, med, Y):
    tok = torch.tensor(rank_tokens(Y, med)); out = []
    with torch.no_grad():
        for i in range(0, len(tok), 512): out.append(net.embed(tok[i:i + 512]).numpy())
    return np.vstack(out)

# ---------------------------------------------------------------- evaluation
ref_idx = np.flatnonzero(np.isin(kcorp, [3, 4, 5, 6, 7])); ref_idx = rng.choice(ref_idx, 3000, replace=False)           # labeled reference cells from the corpus
def evaluate(Zref, Znew, name):
    nnr = NearestNeighbors(n_neighbors=15, metric="cosine").fit(Zref); dist, idx = nnr.kneighbors(Znew)
    votes = np.array([np.bincount(kcorp[ref_idx][i], minlength=12).argmax() for i in idx]); shared = knew != 11
    acc = (votes[shared] == knew[shared]).mean(); auc = roc_auc_score(knew == 11, dist.mean(1))
    asw_type = silhouette_score(Znew, knew, metric="cosine", sample_size=1500, random_state=0)
    Zall = np.vstack([Zref, Znew]); lab = np.r_[kcorp[ref_idx], knew]; study_lab = np.r_[np.zeros(len(Zref)), np.ones(len(Znew))]; m = lab != 11
    asw_b = np.mean([abs(silhouette_score(Zall[m & (lab == t)], study_lab[m & (lab == t)], metric="cosine", sample_size=1000, random_state=0)) for t in [3, 4, 5, 6, 7]])
    print(f"{name:46s}   {acc:6.3f}               {auc:6.3f}                  {asw_type:6.3f}               {asw_b:6.3f}")
print("\nmethod                                         label transfer accuracy   novel-type detection AUROC   cell-type silhouette   study separation within type (0 = mixed)")
def lognorm(Y): return np.log1p(Y / Y.sum(1, keepdims=True) * 1e4)
pool = np.vstack([Ycorp[ref_idx], Ynew])                                                                              # PCA and VAE are fit on reference + new cells
hv = np.argsort(-lognorm(pool).var(0))[:300]; Zp = PCA(30, random_state=0).fit_transform(lognorm(pool)[:, hv]); evaluate(Zp[:len(ref_idx)], Zp[len(ref_idx):], "PCA, 300 variable genes, 30 PCs")
Zp2 = PCA(30, random_state=0).fit_transform(lognorm(pool)); evaluate(Zp2[:len(ref_idx)], Zp2[len(ref_idx):], "PCA, all genes, 30 PCs")
bats = np.r_[np.zeros(len(ref_idx), int), np.ones(len(Ynew), int)]; Zv = ch30.fit_vae(pool, bats, False, epochs=60); evaluate(Zv[:len(ref_idx)], Zv[len(ref_idx):], "NB-VAE (no batch covariate), fit on reference + new")
for n_pre, steps in [(2000, 400), (19800, 800)]:
    sel = rng.choice(len(Ycorp), n_pre, replace=False); net, med, ls = pretrain(Ycorp[sel], steps)
    Zr = fm_embed(net, med, Ycorp[ref_idx]); Zn = fm_embed(net, med, Ynew); evaluate(Zr, Zn, f"mini-FM zero-shot, {n_pre:,} pretraining cells, {steps} steps")
torch.manual_seed(1); rnd = MiniFM().eval(); Zr = fm_embed(rnd, np.ones(G), Ycorp[ref_idx]); Zn = fm_embed(rnd, np.ones(G), Ynew); evaluate(Zr, Zn, "mini-FM architecture, random weights (no pretraining)")
