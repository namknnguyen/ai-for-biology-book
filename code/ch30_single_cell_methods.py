"""Chapter 30: single-cell methods before foundation models, tested on simulated data with known cell types, a known trajectory, and batch effects.
Counts follow the measurement model of Chapter 25 (Gamma-Poisson with per-cell library size). Two batches have *different* cell-type compositions.
Methods: PCA on log-normalized counts; PCA after per-batch gene centering; a negative-binomial VAE (scVI-style) with and without batch as a decoder covariate."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

rng = np.random.default_rng(0); torch.manual_seed(0)
G, NB_ = 1000, 2500

def simulate(mean_umi, batch_sd=0.35):
    base = rng.lognormal(0, 1.4, G); base /= base.sum()
    type_eff = np.zeros((4, G)); diff = rng.random(G) < 0.30
    for k in range(4): type_eff[k] = np.where(diff, rng.normal(0, 0.9, G), 0.0)
    traj = np.zeros(G); traj[rng.choice(G, 100, replace=False)] = rng.normal(0, 1.2, 100)            # genes that change along the trajectory in type 2
    comp = {0: [0.4, 0.3, 0.3, 0.0], 1: [0.0, 0.3, 0.3, 0.4]}                                            # batch 0 lacks type 3; batch 1 lacks type 0
    bat_eff = [np.exp(rng.normal(0, batch_sd, G)) for _ in range(2)]
    types, bats, tt, Y = [], [], [], []
    for b in range(2):
        k = rng.choice(4, NB_, p=comp[b]); t = np.where(k == 2, rng.random(NB_), np.nan)
        rho = base[None, :] * np.exp(type_eff[k] + np.nan_to_num(t)[:, None] * traj[None, :]) * bat_eff[b][None, :]
        rho /= rho.sum(1, keepdims=True)
        lib = rng.lognormal(np.log(mean_umi * (1.0 if b == 0 else 1.6)), 0.3, NB_)                          # batch 1 is sequenced deeper
        mu = lib[:, None] * rho; y = rng.poisson(rng.gamma(1 / 0.3, mu * 0.3))
        types.append(k); bats.append(np.full(NB_, b)); tt.append(t); Y.append(y)
    return np.vstack(Y), np.concatenate(types), np.concatenate(bats), np.concatenate(tt)

def lognorm(Y):
    return np.log1p(Y / Y.sum(1, keepdims=True) * 1e4)

class NBVAE(nn.Module):
    def __init__(self, G, n_batch, d=10, h=128, use_batch=True):
        super().__init__(); self.use_batch = use_batch; self.nb = n_batch
        self.enc = nn.Sequential(nn.Linear(G, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()); self.mu = nn.Linear(h, d); self.lv = nn.Linear(h, d)
        self.dec = nn.Sequential(nn.Linear(d + (n_batch if use_batch is True else 0), h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, G))
        self.bias = nn.Parameter(torch.zeros(n_batch, G))                                              # used only by the additive variant
        self.logtheta = nn.Parameter(torch.zeros(G))
    def forward(self, x, b):
        h_ = self.enc(torch.log1p(x)); mu, lv = self.mu(h_), self.lv(h_).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        inp = torch.cat([z, F.one_hot(b, self.nb).float()], 1) if self.use_batch is True else z
        logits = self.dec(inp) + (self.bias[b] if self.use_batch == "additive" else 0.0)
        rho = F.softmax(logits, 1); lib = x.sum(1, keepdim=True); m = lib * rho + 1e-8
        th = torch.exp(self.logtheta)[None, :]                                                         # inverse dispersion
        ll = (torch.lgamma(x + th) - torch.lgamma(th) - torch.lgamma(x + 1) + th * (torch.log(th) - torch.log(th + m)) + x * (torch.log(m) - torch.log(th + m))).sum(1)
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - 1 - lv).sum(1)
        return (-ll + kl).mean(), mu
    def denoised(self, x, b):
        with torch.no_grad():
            mu = self.mu(self.enc(torch.log1p(x)))
            inp = torch.cat([mu, F.one_hot(b, self.nb).float()], 1) if self.use_batch is True else mu
            logits = self.dec(inp) + (self.bias[b] if self.use_batch == "additive" else 0.0)
            return (F.softmax(logits, 1) * x.sum(1, keepdim=True))
def fit_vae(Y, bats, use_batch, epochs=120):
    X = torch.tensor(Y, dtype=torch.float32); B = torch.tensor(bats, dtype=torch.long)
    m = NBVAE(G, 2, use_batch=use_batch); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]; loss, _ = m(X[idx], B[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): _, mu = m(X, B)
    fit_vae.last_model = (m, X, B)
    return mu.numpy()

def evaluate(Z, types, bats, tt, k_nn=30):
    shared = np.isin(types, [1, 2])
    nn_ = NearestNeighbors(n_neighbors=k_nn + 1).fit(Z); idx = nn_.kneighbors(Z)[1][:, 1:]
    own = np.mean(bats[idx][shared] == bats[shared][:, None])                                         # fraction of neighbours from the cell's own batch (0.5 = perfectly mixed)
    km = KMeans(4, n_init=10, random_state=0).fit_predict(Z); ari = adjusted_rand_score(types, km)
    tr = (bats == 0) & shared; te = (bats == 1) & shared                                               # cross-batch label transfer on the two shared types
    acc = KNeighborsClassifier(15).fit(Z[tr], types[tr]).score(Z[te], types[te])
    m2 = types == 2; pc = PCA(1).fit_transform(Z[m2])[:, 0]; rho_t = abs(spearmanr(pc, tt[m2])[0])      # trajectory recovery inside type 2 (both batches)
    uniq = np.isin(types, [0, 3]); pur = np.mean(types[idx][uniq] == types[uniq][:, None])             # purity of the two batch-specific types
    return np.array([own, acc, ari, rho_t, pur])

NAMES = ["PCA, log-normalized, uncorrected", "PCA after per-batch gene centering", "NB-VAE without batch covariate", "NB-VAE, batch as free decoder covariate", "NB-VAE, additive per-gene batch bias"]
SEEDS = [0, 1, 2]
if __name__ == "__main__":
    for mean_umi in [3000, 400]:
        res = {n: [] for n in NAMES}
        for seed in SEEDS:
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            Y, types, bats, tt = simulate(mean_umi)
            L = lognorm(Y); Lc = L.copy()
            for b in range(2): Lc[bats == b] -= Lc[bats == b].mean(0)                                      # per-batch centering (assumes equal composition)
            embs = [PCA(10, random_state=0).fit_transform(L - L.mean(0)), PCA(10, random_state=0).fit_transform(Lc),
                    fit_vae(Y, bats, False), fit_vae(Y, bats, True), fit_vae(Y, bats, "additive")]
            for n, Z in zip(NAMES, embs): res[n].append(evaluate(Z, types, bats, tt))
        print(f"\n== mean UMI per cell about {mean_umi} (batch 1 sequenced 1.6x deeper); {2 * NB_} cells, {G} genes; batch 0 lacks cell type 3, batch 1 lacks cell type 0; mean (sd) over {len(SEEDS)} simulated datasets ==")
        print(f"{'method':42s} own-batch neighbours   label transfer     ARI (4 types)     trajectory |rho|   batch-specific-type purity")
        for n in NAMES:
            r = np.array(res[n]); f = lambda j: f"{r[:, j].mean():5.2f} ({r[:, j].std():4.2f})"
            print(f"{n:42s} {f(0)}         {f(1)}      {f(2)}      {f(3)}        {f(4)}")
        print("(own-batch neighbour fraction: 0.50 = perfect mixing of the shared cell types, 1.00 = no mixing)")
