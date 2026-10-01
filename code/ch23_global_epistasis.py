"""Chapter 23: marginal stability and global epistasis. Additive effects on a latent stability scale + a sigmoid
phenotype = apparent epistasis on the measured scale; a latent-scale model recovers it, an additive model does not."""
import numpy as np, torch

rng = np.random.default_rng(0); torch.manual_seed(0)
RT = 0.593                                                       # kcal/mol at 298 K

# ---------------------------------------------------------------------------------------------
# 1. Fraction folded vs folding free energy:  f = 1 / (1 + exp(dG_fold / RT))
# ---------------------------------------------------------------------------------------------
print("fraction folded vs folding free energy dG (kcal/mol; negative = stable):")
for dG in (-8, -5, -3, -1, 0, 1):
    print(f"  dG = {dG:+d}: fraction folded = {1 / (1 + np.exp(dG / RT)):.4f}")

# ---------------------------------------------------------------------------------------------
# 2-3. A synthetic protein: L sites x 20 amino acids, additive ddG on the latent stability scale (mostly destabilising).
#      Fit (a) additive on the measured scale, (b) additive latent + sigmoid, on singles + doubles; test on 3-6-fold mutants.
# ---------------------------------------------------------------------------------------------
L, A = 40, 20
r2 = lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def experiment(dG_wt, seed=1):
    global rng
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    ddG = np.where(rng.random((L, A)) < 0.15, -rng.exponential(0.4, (L, A)), rng.gamma(1.5, 0.8, (L, A)))   # 15% stabilising
    wt = rng.integers(0, A, L); ddG[np.arange(L), wt] = 0.0
    phenotype = lambda muts: 1 / (1 + np.exp((dG_wt + sum(ddG[i, a] for i, a in muts)) / RT))
    def random_mutants(n_mut, n):
        out = []
        for _ in range(n):
            sites = rng.choice(L, n_mut, replace=False); out.append([(int(s_), int((wt[s_] + rng.integers(1, A)) % A)) for s_ in sites])
        return out
    f_wt = phenotype([]); singles = [[(i, a)] for i in range(L) for a in range(A) if a != wt[i]]
    f1 = np.array([phenotype(m) for m in singles])
    pairs = random_mutants(2, 4000)
    eps = np.array([phenotype(p) - (phenotype([p[0]]) + phenotype([p[1]]) - f_wt) for p in pairs])
    print(f"\nwild-type dG = {dG_wt} kcal/mol: wild type fraction folded = {f_wt:.3f}; single mutants: median {np.median(f1):.3f}, "
          f"fraction below 0.5 = {np.mean(f1 < 0.5):.2f}")
    print(f"   pairs: apparent epistasis f_AB - (f_A + f_B - f_wt): mean {eps.mean():+.3f}, sd {eps.std():.3f}; "
          f"fraction with |epistasis| > 0.05 = {np.mean(np.abs(eps) > 0.05):.2f}  (latent-scale epistasis is exactly 0 by construction)")
    def design(muts_list):
        X = np.zeros((len(muts_list), L * A))
        for r_, m in enumerate(muts_list):
            for i, a in m: X[r_, i * A + a] = 1
        return X
    train = singles + random_mutants(2, 3000); test = sum([random_mutants(k, 600) for k in (3, 4, 5, 6)], [])
    ytr = np.array([phenotype(m) for m in train]) + 0.01 * rng.normal(size=len(train)); yte = np.array([phenotype(m) for m in test])
    Xtr, Xte = design(train), design(test)
    w_add = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], ytr, rcond=None)[0]; pred_add = np.c_[Xte, np.ones(len(Xte))] @ w_add
    theta = torch.zeros(L * A, requires_grad=True); b0 = torch.tensor(-1.0, requires_grad=True); scale = torch.tensor(1.0, requires_grad=True)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam([theta, b0, scale], lr=0.05)
    for step in range(1500):
        pred = torch.sigmoid(-(b0 + Xt @ theta) * scale)
        loss = ((pred - yt) ** 2).mean() + 1e-4 * (theta ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred_ge = torch.sigmoid(-(b0 + torch.tensor(Xte, dtype=torch.float32) @ theta) * scale).numpy()
    ks = np.array([len(t) for t in test])
    print("   held-out R^2 on 3-6-fold mutants (trained on singles and doubles only):  " + "; ".join(
        f"{k}-fold: additive {r2(yte[ks == k], pred_add[ks == k]):.2f} / global-epistasis {r2(yte[ks == k], pred_ge[ks == k]):.2f}" for k in (3, 4, 5, 6)))
    print(f"   pooled: additive on the measured scale = {r2(yte, pred_add):.3f}; additive latent + sigmoid = {r2(yte, pred_ge):.3f}")
for dG in (-4.0, -1.5): experiment(dG)
