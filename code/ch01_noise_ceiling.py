"""Chapter 1: what does "R^2 = 0.7" mean? A simulation of noise ceilings and split leakage.

Genes live in neighbourhoods (blocks). Expression depends on
  (i)   a real, transferable sequence feature (a promoter motif score),
  (ii)  a regional factor shared by every gene in a block (chromatin domain activity),
  (iii) independent biological variation and measurement noise.
Genes in the same block also share a sequence "signature" (e.g. repeat content), so a flexible
model can recognise the block from sequence alone and memorise its regional factor.
"""
import numpy as np

rng = np.random.default_rng(0)

n_blocks, genes_per_block, sig_dim = 300, 20, 32
n = n_blocks * genes_per_block
block = np.repeat(np.arange(n_blocks), genes_per_block)  # (n,)
chrom = block % 10                                         # 10 "chromosomes"

motif = rng.normal(size=n)                                  # transferable causal feature
region = rng.normal(size=n_blocks)[block]                   # shared regional factor
signature = (rng.normal(size=(n_blocks, sig_dim))[block]    # block-level sequence signature
             + 0.5 * rng.normal(size=(n, sig_dim)))         # (n, sig_dim)

bio_noise = 0.5 * rng.normal(size=n)                        # gene-specific biological variation
y_true = 1.0 * motif + 1.0 * region + bio_noise             # latent "true" expression level
tech = lambda: 0.7 * rng.normal(size=n)                     # technical (measurement) noise
y_rep1, y_rep2 = y_true + tech(), y_true + tech()           # two replicate measurements
y = y_rep1                                                  # the label the model is trained on


def r2(pred, target):
    return 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def linear_fit_predict(Xtr, ytr, Xte, lam=1.0):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (ytr - ytr.mean()))
    return (Xtr - mu) @ w + ytr.mean(), (Xte - mu) @ w + ytr.mean()


def knn_residual_predict(Str, rtr, Ste, k=10):
    """Predict a residual from the sequence signature by averaging the k nearest training genes."""
    d2 = (Ste ** 2).sum(1)[:, None] - 2 * Ste @ Str.T + (Str ** 2).sum(1)[None, :]  # (n_te, n_tr)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    return rtr[idx].mean(1)


# Noise ceiling: the best any model could do against the *noisy* label is limited by
# Var(signal)/Var(measurement). Estimate from replicates: corr(rep1, rep2) ~ ceiling on R^2.
ceiling = np.corrcoef(y_rep1, y_rep2)[0, 1]
print(f"replicate correlation = {ceiling:.3f}  (Var(signal)/Var(label) = {np.var(y_true) / np.var(y_rep1):.3f})"
      f"  -> no model can exceed R^2 of about {ceiling:.2f} on this label")

splits = {
    "random genes": rng.random(n) < 0.8,                     # train on 80% of genes at random
    "held-out chromosomes": chrom < 8,                       # train on chr 0-7, test on chr 8-9
}
for name, train in splits.items():
    test = ~train
    ptr, pte = linear_fit_predict(motif[train, None], y[train], motif[test, None])
    r_motif = r2(pte, y[test])                               # transferable feature only
    resid = y[train] - ptr                                   # what the motif does not explain
    p_flex = pte + knn_residual_predict(signature[train], resid, signature[test])
    r_flex = r2(p_flex, y[test])                             # + memorise regional residuals by sequence similarity
    base = r2(np.full(test.sum(), y[train].mean()), y[test])
    print(f"{name:22s} mean-baseline R^2 = {base:6.3f} | motif-only R^2 = {r_motif:.3f} "
          f"| motif + sequence-similarity memorisation R^2 = {r_flex:.3f}")
