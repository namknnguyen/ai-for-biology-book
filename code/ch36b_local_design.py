"""Chapter 36 (second experiment): designing beyond the data.  Training data are LOCAL, like a deep mutational scan around one wild type: 300 variants within 1-3 mutations of a reference sequence.
The designer wants sequences far from the reference.  The surrogate is accurate near the reference and unreliable far from it; designs are sampled from p*(x) ~ prior(x) exp(f_hat(x)/beta),
where the prior is the site-independent model of the training set (a trust region around the reference whose width beta controls).  We report the surrogate's prediction, the true fitness, and the
distance from the reference, and the same for an ensemble lower-confidence-bound surrogate.  Landscape and functions come from ch36_design_surrogate.py."""
import numpy as np
from ch36_design_surrogate import Lp, q, make_landscape, latent, feats, fit_ridge, sample_design, gibbs_prior_weights, fit_mlp
import ch36_design_surrogate as C
def local_set(ref, n, r):
    S = np.tile(ref, (n, 1)); k = r.integers(1, 4, n)
    for i in range(n):
        pos = r.choice(Lp, k[i], replace=False); S[i, pos] = (S[i, pos] + r.integers(1, q, k[i])) % q
    return S
def far_set(ref, n, r, k):
    S = np.tile(ref, (n, 1))
    for i in range(n):
        pos = r.choice(Lp, k, replace=False); S[i, pos] = (S[i, pos] + r.integers(1, q, k)) % q
    return S
n_land, N_train = 8, 300
settings = [("beta = 2.0", 2.0), ("beta = 1.0", 1.0), ("beta = 0.5", 0.5), ("beta = 0.25", 0.25), ("beta = 0.1 (near argmax)", 0.1)]
out = {}
print(f"== Training data: {N_train} variants within 1-3 mutations of a reference (the best of 700 natural-like sequences); {n_land} landscapes; 60 designs per setting ==")
for ls in range(n_land):
    h, J = make_landscape(ls); r0 = np.random.default_rng(1000 + ls); S_all = gibbs_prior_weights(h, J, 1.5, 700, 40, r0)
    ref = S_all[np.argmax(latent(h, J, S_all))]; r = np.random.default_rng(3000 + ls)
    S_tr = local_set(ref, N_train, r); y = latent(h, J, S_tr); S_near = local_set(ref, 300, r); S_far = far_set(ref, 300, r, 8)
    freq = np.stack([np.bincount(S_tr[:, i], minlength=q) + 1.0 for i in range(Lp)]); logprior = np.log(freq / freq.sum(1, keepdims=True))
    for kind in ("pairwise ridge", "MLP"):
        if kind == "pairwise ridge":
            C.ADDITIVE = False; wv, b0 = fit_ridge(feats(S_tr), y, 3.0); f = lambda S2, wv=wv, b0=b0: feats(S2) @ wv + b0; ens = []
            for _ in range(8):
                bi = r.integers(0, N_train, N_train); w_, b_ = fit_ridge(feats(S_tr)[bi], y[bi], 3.0); ens.append((w_, b_))
        else:
            wv = fit_mlp(S_tr, y, ls); b0 = None; f = wv; ens = None
        r2n = np.corrcoef(f(S_near), latent(h, J, S_near))[0, 1] ** 2; r2f = np.corrcoef(f(S_far), latent(h, J, S_far))[0, 1] ** 2
        out.setdefault((kind, "R2"), []).append((r2n, r2f, latent(h, J, ref[None])[0], latent(h, J, S_tr).mean()))
        for name, beta in settings:
            for lcb in ([None, 1.0] if kind == "pairwise ridge" else [None]):
                D = sample_design(wv, b0, logprior, beta, r, lcb=(ens, lcb) if lcb else None)
                lt = latent(h, J, D); pred = f(D); dist = (D != ref).sum(1)
                out.setdefault((kind, name + (", ensemble lower bound" if lcb else "")), []).append((pred.mean(), lt.mean(), dist.mean(), len(np.unique(D, axis=0)) / len(D), (lt > latent(h, J, ref[None])[0]).mean()))
R2 = {k[0]: np.mean(v, 0) for k, v in out.items() if k[1] == "R2"}
print(f"reference fitness (latent) {R2['pairwise ridge'][2]:.1f}; mean fitness of the training variants {R2['pairwise ridge'][3]:.1f}")
for k, v in R2.items(): print(f"surrogate {k:15s}: R^2 on variants like the training data {v[0]:.2f};  R^2 on variants 8 mutations from the reference {v[1]:.2f}")
print("\nsurrogate        design setting                                    surrogate's prediction   TRUE fitness   gap (pred - true)   mutations from reference   fraction unique   fraction better than the reference")
for k, v in out.items():
    if k[1] == "R2": continue
    a = np.mean(v, 0); print(f"{k[0]:15s}  {k[1]:48s}  {a[0]:12.2f}          {a[1]:10.2f}      {a[0]-a[1]:9.2f}              {a[2]:8.1f}                 {a[3]:8.2f}           {a[4]:8.2f}")
