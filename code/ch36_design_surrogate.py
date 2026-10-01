"""Chapter 36: designing against a learned surrogate. Optimization power versus reliability, and the KL-regularized optimum.
A hidden landscape f(x) (pairwise epistasis + saturating readout) over sequences of 16 positions, 4 letters.  'Natural' training sequences are drawn from a selected prior.
A ridge surrogate with single + pairwise features is fit on N training sequences; designs are then sampled from  p*(x) proportional to p_prior(x) exp(f_hat(x)/beta)
(beta = infinity: the prior; beta -> 0: argmax of the surrogate).  We measure the surrogate's prediction, the TRUE fitness, and the distance from the training data."""
import numpy as np
rng = np.random.default_rng(0)
Lp, q = 16, 4
def make_landscape(seed):
    r = np.random.default_rng(seed); h = r.normal(0, 0.6, (Lp, q)); J = np.zeros((Lp, Lp, q, q))
    for i in range(Lp):
        for j in range(i + 1, Lp):
            if r.random() < 0.3: J[i, j] = r.normal(0, 0.8, (q, q))
    return h, J
def latent(h, J, S):
    e = h[np.arange(Lp)[None, :], S].sum(1)
    for i in range(Lp):
        for j in range(i + 1, Lp): e = e + J[i, j][S[:, i], S[:, j]]
    return e
ADDITIVE = False
def feats(S):
    X = np.zeros((len(S), Lp * q)); X[np.arange(len(S))[:, None], np.arange(Lp)[None, :] * q + S] = 1
    return X if ADDITIVE else np.c_[X, X[:, _pa] * X[:, _pb]]
_iu = np.triu_indices(Lp * q, 1); keep = (_iu[0] // q) < (_iu[1] // q); _pa, _pb = _iu[0][keep], _iu[1][keep]            # pairs at different positions
def gibbs_prior_weights(h, J, T0=1.5, n=500, sweeps=60, r=None):
    """'Natural' sequences: Gibbs samples from p(x) ~ exp(latent(x)/T0) (a selected, family-like distribution)."""
    S = r.integers(0, q, (n, Lp))
    for _ in range(sweeps):
        for i in r.permutation(Lp):
            lg = np.zeros((n, q))
            for a in range(q):
                S2 = S.copy(); S2[:, i] = a; lg[:, a] = latent(h, J, S2)
            lg = lg / T0; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); S[:, i] = (r.random(n)[:, None] > np.cumsum(p, 1)).sum(1).clip(max=q - 1)
    return S
def fit_ridge(X, y, lam): return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (y - y.mean())), y.mean()
def sample_design(wvec, b0, logprior, beta, r, n_chains=60, sweeps=20, lcb=None):
    """Gibbs sampling from p*(x) ~ prior(x) * exp(f_hat(x)/beta); beta=None gives the prior itself."""
    S = r.integers(0, q, (n_chains, Lp))
    for _ in range(sweeps):
        for i in r.permutation(Lp):
            sc = np.zeros((n_chains, q))
            for a in range(q):
                S2 = S.copy(); S2[:, i] = a; sc[:, a] = logprior[i, a] + (0 if beta is None else score(S2, wvec, b0, lcb) / beta)
            p = np.exp(sc - sc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); S[:, i] = (r.random(n_chains)[:, None] > np.cumsum(p, 1)).sum(1).clip(max=q - 1)
    return S
def score(S, wvec, b0, lcb):
    if callable(wvec): return wvec(S)
    X = feats(S)
    if lcb is None: return X @ wvec + b0
    ws, lam = lcb; P = np.stack([X @ w + b for w, b in ws]); return P.mean(0) - lam * P.std(0)
import torch, torch.nn as nn
torch.set_num_threads(1)
def fit_mlp(S, y, seed, steps=600):
    torch.manual_seed(seed); Xo = torch.tensor(feats_oh(S)); mu, sd = y.mean(), y.std(); yt = torch.tensor(((y - mu) / sd).astype(np.float32))
    net = nn.Sequential(nn.Linear(Lp * q, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)); opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(steps): loss = ((net(Xo)[:, 0] - yt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    def f(S2):
        with torch.no_grad(): return net(torch.tensor(feats_oh(S2)))[:, 0].numpy() * sd + mu
    return f
def feats_oh(S):
    X = np.zeros((len(S), Lp * q), np.float32); X[np.arange(len(S))[:, None], np.arange(Lp)[None, :] * q + S] = 1; return X
def hamming_nn(S, train): return np.array([np.min((train != s).sum(1)) for s in S])

if __name__ == "__main__":
    settings = [("prior samples (beta = inf)", None, None), ("beta = 1.0", 1.0, None), ("beta = 0.25", 0.25, None), ("beta = 0.1 (near argmax)", 0.1, None), ("beta = 0.1, ensemble lower bound (mean - 1 sd)", 0.1, 1.0)]
    n_land = 8
    print(f"== Designs sampled from p*(x) ~ prior(x) exp(f_hat(x)/beta): {Lp} positions x {q} letters; {n_land} hidden landscapes x 60 designs per setting ==")
    print("'latent fitness' is the landscape's additive-plus-pairwise score (natural-like sequences are Gibbs samples from a prior selected on it)")
    configs = [(60, "pair"), (150, "pair"), (400, "pair"), (400, "add"), (150, "mlp"), (400, "mlp")]
    store = {c: {s[0]: [] for s in settings} for c in configs}; heldout = {c: [] for c in configs}; train_mean = []
    for ls in range(n_land):
        h, J = make_landscape(ls); r0 = np.random.default_rng(1000 + ls)
        S_all = gibbs_prior_weights(h, J, 1.5, 700, 40, r0); S_te = S_all[400:]
        for (N_train, kind) in configs:
            ADDITIVE = kind == "add"; r = np.random.default_rng(2000 + ls); S_train = S_all[:N_train]; y = latent(h, J, S_train); lam = 3.0
            if kind == "mlp":
                wv = fit_mlp(S_train, y, ls); b0 = None; pred_fn = wv; heldout[(N_train, kind)].append(np.corrcoef(wv(S_te), latent(h, J, S_te))[0, 1] ** 2); ens = None
            else:
                Xtr = feats(S_train); wv, b0 = fit_ridge(Xtr, y, lam); pred_fn = lambda S2, wv=wv, b0=b0: feats(S2) @ wv + b0
                heldout[(N_train, kind)].append(np.corrcoef(pred_fn(S_te), latent(h, J, S_te))[0, 1] ** 2)
                ens = []
                for _ in range(8):
                    bi = r.integers(0, N_train, N_train); w_, b_ = fit_ridge(Xtr[bi], y[bi], lam); ens.append((w_, b_))
            if N_train == 400 and kind == "pair": train_mean.append(y.mean())
            freq = np.stack([np.bincount(S_train[:, i], minlength=q) + 1.0 for i in range(Lp)]); logprior = np.log(freq / freq.sum(1, keepdims=True))
            for name, beta, lcbl in settings:
                if kind != "pair" and "ensemble" in name: continue
                D = sample_design(wv, b0, logprior, beta, r, lcb=(ens, lcbl) if lcbl else None)
                lt = latent(h, J, D); pred = pred_fn(D)
                store[(N_train, kind)][name].append((pred.mean(), lt.mean(), hamming_nn(D, S_train).mean(), len(np.unique(D, axis=0)) / len(D)))
    ADDITIVE = False
    print(f"natural-like training sequences have mean latent fitness {np.mean(train_mean):.1f} (400 sequences)\n")
    print("surrogate                          training sequences   held-out R^2   design setting                                      surrogate's prediction   TRUE fitness   gap (pred - true)   distance to nearest training seq.   fraction unique")
    names = {"pair": "pairwise ridge", "add": "additive ridge (misspecified)", "mlp": "MLP, 2 hidden layers (flexible)"}
    for c in configs:
        for name, _, _ in settings:
            if not store[c][name]: continue
            a = np.mean(store[c][name], 0)
            print(f"{names[c[1]]:33s}  {c[0]:8d}            {np.mean(heldout[c]):6.2f}        {name:50s}  {a[0]:9.2f}               {a[1]:8.2f}       {a[0]-a[1]:8.2f}            {a[2]:8.1f}                {a[3]:8.2f}")
