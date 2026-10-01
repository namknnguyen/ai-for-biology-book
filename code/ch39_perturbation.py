"""Chapter 39: predicting the effect of unseen perturbations, in a simulated gene network with a known generating process.
World: N = 100 genes, a sparse directed network W (spectral radius 0.9), steady state x = tanh(W x + b); a CRISPRi-like perturbation clamps one (or two) genes low.
Observed: the mean over n cells with noise.  Train on 70 single perturbations; test on (a) 30 UNSEEN single perturbations and (b) double perturbations of seen genes.
Gene embeddings carry partial knowledge of the network (as pretrained gene embeddings or knowledge graphs do): each true edge is visible with probability rho, and false edges are added at a small rate.
Methods: no-change baseline, mean shift, embedding-regression (ridge), additive (doubles), and an interaction model.  Metrics: delta-correlation, RMSE relative to the no-change baseline, the
inflated raw correlation, and the noise ceiling."""
import numpy as np
rng = np.random.default_rng(0)
N, CLAMP, NCELLS, SIGMA = 100, -3.0, 400, 1.0
def make_world(seed):
    r = np.random.default_rng(seed); W = (r.random((N, N)) < 0.08) * r.normal(0, 1.0, (N, N)); np.fill_diagonal(W, 0)
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W))); b = r.normal(0, 1.2, N); return W, b
def steady(W, b, clamp):
    x = np.zeros(N)
    for _ in range(300):
        x = np.tanh(W @ x + b)
        for g in clamp: x[g] = CLAMP
    return x
def embeddings(W, rho, seed):
    r = np.random.default_rng(seed); A = (np.abs(W) > 0).astype(float); seen = (r.random((N, N)) < rho) * A; false = (r.random((N, N)) < 0.01).astype(float)
    S = np.sign(W) * seen + np.where(false > 0, r.normal(0, 0.5, (N, N)), 0)                                   # visible signed edges, plus false edges
    return np.hstack([S.T, S])                                                                                  # row g: signed edges leaving g, then edges entering g (dimension 2N)
def ridge(X, Y, lam): return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
def pearson(a, b): a = a - a.mean(); b = b - b.mean(); return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

res = {k: [] for k in ("noise ceiling", "mean shift", "oracle first-order", "generic embedding regression, rho=0.0", "generic embedding regression, rho=1.0", "graph propagation, rho=0.3", "graph propagation, rho=0.6", "graph propagation, rho=1.0")}; rmse = {k: [] for k in res}; raw_corr = {k: [] for k in res}
add_err, dbl_scale, int_corr = [], [], {0.0: [], 0.6: [], 1.0: []}; add_vs_ctrl = []; two_ctrl = []
for seed in range(10):
    W, b = make_world(seed); r = np.random.default_rng(100 + seed); x0 = steady(W, b, [])
    perm = r.permutation(N); tr, te = perm[:70], perm[70:]
    D = {g: steady(W, b, [g]) - x0 for g in range(N)}                                                           # TRUE single-perturbation effects (gene g clamped; the clamped gene itself changes by definition)
    noise = lambda: r.normal(0, SIGMA / np.sqrt(NCELLS), N)
    obs = {g: D[g] + noise() for g in range(N)}; rep = {g: D[g] + noise() for g in range(N)}                     # two independent measurements of each perturbation
    mask = lambda g: np.arange(N) != g                                                                          # score the other genes (the clamped gene's own change is trivial)
    # (a) unseen single perturbations
    mean_shift = np.mean([obs[g] for g in tr], 0)
    first_order = {g: W[:, g] * (CLAMP - x0[g]) * (1 - x0 ** 2) for g in range(N)}                              # oracle: direct effects only, from the true network
    preds = {"mean shift": lambda g: mean_shift, "oracle first-order": lambda g: first_order[g]}
    for rho in (0.0, 1.0):
        E = embeddings(W, rho, 7 + seed); B = ridge(E[tr], np.stack([obs[g] for g in tr]), 3.0); preds[f"generic embedding regression, rho={rho:.1f}"] = (lambda g, E=E, B=B: E[g] @ B)
    for rho in (0.3, 0.6, 1.0):                                                                                  # a model whose structure is tied to the (partial) network: 1-hop and 2-hop propagation, two scalars fit on the training perturbations
        E = embeddings(W, rho, 7 + seed); Sv = E[:, N:].T * 0 + E[:, :N].T                                         # Sv[i, g] = visible signed edge g -> i
        feat1 = lambda g, Sv=Sv: Sv[:, g] * (CLAMP - x0[g]); feat2 = lambda g, Sv=Sv: Sv @ Sv[:, g] * (CLAMP - x0[g])
        Xf = np.stack([np.r_[feat1(g), feat2(g)] for g in tr]); Fm = np.stack([np.stack([feat1(g), feat2(g)], 1) for g in tr]).reshape(-1, 2); yv = np.stack([obs[g] for g in tr]).reshape(-1)
        al = np.linalg.lstsq(Fm, yv, rcond=None)[0]; preds[f"graph propagation, rho={rho:.1f}"] = (lambda g, al=al, f1=feat1, f2=feat2: al[0] * f1(g) + al[1] * f2(g))
    for name, f in list(preds.items()) + [("noise ceiling", lambda g: None)]:
        c, e, rc = [], [], []
        for g in te:
            m = mask(g); truth = obs[g][m]
            if name == "noise ceiling": p = rep[g][m]
            else: p = f(g)[m]
            c.append(pearson(p, truth)); e.append(np.sqrt(np.mean((p - truth) ** 2)) / np.sqrt(np.mean(truth ** 2)))
            rc.append(pearson(p + x0[m], truth + x0[m]))                                                           # the same comparison on raw expression (baseline included)
        res[name].append(np.mean(c)); rmse[name].append(np.mean(e)); raw_corr[name].append(np.mean(rc))
    # (b) double perturbations of seen genes: additive prediction and its error
    pairs = [tuple(r.choice(tr, 2, replace=False)) for _ in range(150)]
    Dd = {p: steady(W, b, list(p)) - x0 for p in pairs}; obs2 = {p: Dd[p] + noise() for p in pairs}
    inter = {p: Dd[p] - (D[p[0]] + D[p[1]]) for p in pairs}                                                      # TRUE interaction (non-additivity)
    mk = lambda p: (np.arange(N) != p[0]) & (np.arange(N) != p[1])
    a_rmse = np.mean([np.sqrt(np.mean((obs[p[0]][mk(p)] + obs[p[1]][mk(p)] - obs2[p][mk(p)]) ** 2)) for p in pairs]); c_rmse = np.mean([np.sqrt(np.mean(obs2[p][mk(p)] ** 2)) for p in pairs])
    n_rmse = np.mean([np.sqrt(np.mean((obs2[p][mk(p)] - (D[p[0]] + D[p[1]] + 0)[mk(p)]) ** 2)) for p in pairs])                    # additive using TRUE singles (no measurement noise in the inputs)
    ctrl_noise = SIGMA / np.sqrt(NCELLS)
    add_err.append(a_rmse); two_ctrl.append(c_rmse); dbl_scale.append(np.mean([np.sqrt(np.mean(inter[p][mk(p)] ** 2)) for p in pairs])); add_vs_ctrl.append(n_rmse)
    # interaction model: ridge on the elementwise product of the two genes' embeddings, trained on 100 doubles, tested on 50
    for rho in int_corr:
        E = embeddings(W, rho, 7 + seed); F = lambda p: np.r_[E[p[0]] * E[p[1]], np.abs(E[p[0]]) @ np.abs(E[p[1]]) * np.ones(1)]
        Xp = np.stack([F(p) for p in pairs]); Yp = np.stack([inter[p] for p in pairs]); Bi = ridge(Xp[:100], Yp[:100], 10.0)
        pred = Xp[100:] @ Bi; int_corr[rho].append(np.mean([pearson(pred[k][mk(pairs[100 + k])], Yp[100 + k][mk(pairs[100 + k])]) for k in range(50)]))
print(f"== Perturbation response prediction in a simulated network: {N} genes, {NCELLS} cells per perturbation (measurement SD {SIGMA / np.sqrt(NCELLS):.2f} per gene), 10 networks; 70 training singles; 30 UNSEEN singles ==")
print("\nunseen single perturbations: correlation of predicted and observed CHANGE across genes (delta-correlation); RMSE as a fraction of the no-change baseline's RMSE (1.0 = no better than predicting 'no effect'); and the correlation on RAW expression")
print("method                       delta-correlation   RMSE / baseline RMSE   raw-expression correlation")
for k in res: print(f"{k:26s}      {np.mean(res[k]):7.3f}              {np.mean(rmse[k]):7.3f}                 {np.mean(raw_corr[k]):7.3f}")
print(f"\ndelta-correlation normalised by the noise ceiling: " + ", ".join(f"{k} {np.mean(res[k]) / np.mean(res['noise ceiling']):.2f}" for k in res if k != "noise ceiling"))
print(f"\ndouble perturbations of seen genes (150 pairs): RMSE of the no-change baseline {np.mean(two_ctrl):.3f}; RMSE of the additive prediction from measured singles {np.mean(add_err):.3f}; additive prediction from noise-free singles {np.mean(add_vs_ctrl):.3f}")
print(f"RMS size of the true interaction (non-additivity) {np.mean(dbl_scale):.3f}; measurement noise per gene {SIGMA / np.sqrt(NCELLS):.3f}  -> the interaction is {'smaller' if np.mean(dbl_scale) < SIGMA / np.sqrt(NCELLS) else 'larger'} than the noise")
print("interaction model (ridge on products of gene embeddings, 100 training doubles): correlation between predicted and true interaction vectors on 50 held-out doubles")
for rho, v in int_corr.items(): print(f"   embedding recall rho = {rho:.1f}:  {np.mean(v):.3f}")
