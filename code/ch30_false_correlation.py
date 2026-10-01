"""Chapter 30: denoised (decoder-mean) expression induces spurious gene-gene correlations.
Within one homogeneous population (cell type 2 in batch 0), genes outside the trajectory program are independent given library size;
we compare gene-gene correlation among such null genes in log-normalized counts and in NB-VAE decoder means."""
import numpy as np, torch
import ch30_single_cell_methods as M
from scipy.stats import spearmanr
M.rng = np.random.default_rng(5); torch.manual_seed(5)
Y, types, bats, tt = M.simulate(3000)
Z = M.fit_vae(Y, bats, "additive"); model, X, B = M.fit_vae.last_model
den = model.denoised(X, B).numpy()
sel = (types == 2) & (bats == 0)
# null genes: expressed, not trajectory genes. The trajectory coefficient is not exported, so use genes with no association with the true time
t = tt[sel]; raw = M.lognorm(Y)[sel]; dn = np.log1p(den[sel] / den[sel].sum(1, keepdims=True) * 1e4)
mean_expr = Y[sel].mean(0)
cand = np.flatnonzero(mean_expr > 0.5)
assoc = np.array([abs(spearmanr(raw[:, g], t)[0]) for g in cand]); null = cand[assoc < 0.05]
rng = np.random.default_rng(0); pairs = [tuple(rng.choice(null, 2, replace=False)) for _ in range(4000)]
def cors(A): return np.array([spearmanr(A[:, i], A[:, j])[0] for i, j in pairs])
cr, cd = cors(raw), cors(dn)
print(f"{sel.sum()} cells of one type in one batch; {len(null)} expressed genes with no association to the true trajectory; {len(pairs)} random gene pairs")
print(f"log-normalized counts : mean |r| = {np.abs(cr).mean():.3f}; fraction of pairs with |r| > 0.2 = {np.mean(np.abs(cr) > 0.2):.3f}")
print(f"NB-VAE decoder means  : mean |r| = {np.abs(cd).mean():.3f}; fraction of pairs with |r| > 0.2 = {np.mean(np.abs(cd) > 0.2):.3f}")
print(f"(sampling noise alone gives |r| about {0.8 / np.sqrt(sel.sum()):.3f} for independent genes with {sel.sum()} cells)")
