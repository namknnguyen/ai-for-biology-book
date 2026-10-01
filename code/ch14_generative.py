"""Chapter 14: VAE beta sweep, flow change-of-variables, GAN optimal discriminator, EBM gradient, typical set."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

torch.manual_seed(0); rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# 1. VAE on synthetic loci (conserved signal s in {0..3} at fixed positions + GC-biased background).
#    Sweep the KL weight beta: reconstruction vs KL, active latent dimensions, what the latent encodes.
# ---------------------------------------------------------------------------------------------
L = 60
MOTIFS = np.array([[3, 0, 2, 1, 3, 2], [0, 3, 1, 2, 0, 1], [2, 2, 3, 0, 1, 3], [1, 3, 0, 0, 2, 2]])
def make(n):
    s = rng.integers(0, 4, n); gc = rng.uniform(0.1, 0.9, n)
    p = np.stack([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2], 1)
    X = (rng.random((n, L))[..., None] > np.cumsum(p, 1)[:, None, :]).sum(-1); X[:, 10:16] = MOTIFS[s]
    return X, s, gc
Xtr, s_tr, gc_tr = make(8000); Xte, s_te, gc_te = make(2000)
oh = lambda X: F.one_hot(torch.tensor(X), 4).float()                # (n, L, 4)
Otr, Ote = oh(Xtr), oh(Xte)

class VAE(nn.Module):
    def __init__(s, k=8):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(4 * L, 256), nn.ReLU()); s.mu, s.logvar = nn.Linear(256, k), nn.Linear(256, k)
        s.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 4 * L))
    def forward(s, x):
        h = s.enc(x.flatten(1)); mu, logvar = s.mu(h), s.logvar(h)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()        # reparameterisation trick
        logits = s.dec(z).view(-1, L, 4)
        recon = F.cross_entropy(logits.reshape(-1, 4), x.argmax(-1).reshape(-1), reduction="none").view(len(x), -1).sum(1)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)            # closed-form KL(q||N(0,I)) per latent dimension
        return recon, kl, mu

def run_vae(beta, steps=1500):
    m = VAE(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        x = Otr[torch.randint(0, len(Otr), (256,))]; recon, kl, _ = m(x)
        loss = (recon + beta * kl.sum(1)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        recon, kl, mu = m(Ote)
    Z = mu.numpy(); active = int((Z.std(0) > 0.1).sum())            # latent dims that vary across inputs
    Ztr = m(Otr)[2].detach().numpy()
    acc = LogisticRegression(max_iter=3000).fit(Ztr, s_tr).score(Z, s_te); r2 = Ridge(alpha=1.0).fit(Ztr, gc_tr).score(Z, gc_te)
    return recon.mean().item(), kl.sum(1).mean().item(), active, acc, r2
print("VAE (8 latent dims) KL-weight sweep on held-out data:")
print(f"  {'beta':>5s} {'-log p(x|z) (nats)':>20s} {'KL (nats)':>10s} {'ELBO (nats)':>12s} {'active dims':>12s} {'signal acc':>11s} {'GC R^2':>7s}")
for beta in (0.1, 1.0, 4.0, 16.0):
    r, k, a, acc, r2 = run_vae(beta)
    print(f"  {beta:5.1f} {r:20.1f} {k:10.1f} {-(r + k):12.1f} {a:12d} {acc:11.3f} {r2:7.3f}")

# ---------------------------------------------------------------------------------------------
# 2. Normalising flows: change of variables, and the affine-coupling log-determinant checked against autograd.
# ---------------------------------------------------------------------------------------------
z = rng.normal(size=200_000); x = np.exp(z)                          # X = exp(Z), Z ~ N(0,1) -> log-normal
x0 = 1.7
p_cov = stats.norm.pdf(np.log(x0)) * abs(1 / x0)                     # p_X(x) = p_Z(f^-1(x)) |d f^-1 / dx|
print(f"\nchange of variables: p_X(1.7) via formula = {p_cov:.5f}; scipy lognorm = {stats.lognorm.pdf(x0, 1):.5f}")

class Coupling(nn.Module):                                            # RealNVP-style affine coupling on 4-d input
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))
    def forward(s, x):
        x1, x2 = x[..., :2], x[..., 2:]; ls, t = s.net(x1).chunk(2, -1)
        return torch.cat([x1, x2 * ls.exp() + t], -1), ls.sum(-1)     # (y, log|det J|)
    def inverse(s, y):
        y1, y2 = y[..., :2], y[..., 2:]; ls, t = s.net(y1).chunk(2, -1)
        return torch.cat([y1, (y2 - t) * (-ls).exp()], -1)
flow = Coupling(); xt = torch.randn(4)
y, logdet = flow(xt)
J = torch.autograd.functional.jacobian(lambda v: flow(v)[0], xt)
print(f"affine coupling: log|det J| formula = {logdet.item():.6f}; autograd Jacobian = {torch.linalg.slogdet(J)[1].item():.6f}; "
      f"inverse error = {(flow.inverse(y) - xt).abs().max().item():.1e}; Jacobian triangular: {bool(torch.allclose(J[:2, 2:], torch.zeros(2, 2)))}")

# ---------------------------------------------------------------------------------------------
# 3. GAN theory: optimal discriminator and the Jensen-Shannon identity V(D*) = 2 JS(p||q) - log 4.
# ---------------------------------------------------------------------------------------------
xs = np.linspace(-10, 11, 200001); dx = xs[1] - xs[0]
p, q = stats.norm.pdf(xs, 0, 1), stats.norm.pdf(xs, 1, 1)             # data and generator densities
D = p / (p + q)                                                       # optimal discriminator
V = np.sum(p * np.log(D) + q * np.log(1 - D)) * dx
m = 0.5 * (p + q); JS = 0.5 * np.sum(p * np.log(p / m)) * dx + 0.5 * np.sum(q * np.log(q / m)) * dx
print(f"GAN: V(D*) = {V:.5f}; 2 JS - log 4 = {2 * JS - math.log(4):.5f}")

# ---------------------------------------------------------------------------------------------
# 4. EBM: grad_theta log p(x) = -grad_theta E(x) + E_{x'~p}[grad_theta E(x')]  (checked by finite differences).
# ---------------------------------------------------------------------------------------------
states = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)   # 8 binary states
feats = lambda S: np.concatenate([S, S[:, [0]] * S[:, [1]], S[:, [1]] * S[:, [2]]], 1)   # unary + pairwise features
theta = rng.normal(size=5); xobs = np.array([1., 0., 1.])
def logp(th, x):
    E = lambda S: -(feats(S) @ th); logZ = np.log(np.exp(-E(states)).sum())
    return -E(x[None])[0] - logZ
Pm = np.exp(feats(states) @ theta); Pm /= Pm.sum()
analytic = feats(xobs[None])[0] - Pm @ feats(states)                  # = -dE(x) + E_p[dE]  with E = -theta.phi
numeric = np.array([(logp(theta + 1e-6 * e, xobs) - logp(theta - 1e-6 * e, xobs)) / 2e-6 for e in np.eye(5)])
print(f"EBM: max |analytic - numeric| log-likelihood gradient = {np.abs(analytic - numeric).max():.1e}")

# ---------------------------------------------------------------------------------------------
# 5. Typical set: samples from p are not the mode. Sequences of 300 iid Bernoulli(0.7) symbols.
# ---------------------------------------------------------------------------------------------
n_sym, p1 = 300, 0.7
S = rng.random((5000, n_sym)) < p1
nll = -(S * math.log(p1) + (~S) * math.log(1 - p1)).sum(1) / n_sym
H = -(p1 * math.log(p1) + (1 - p1) * math.log(1 - p1))
print(f"typical set: per-symbol NLL of sampled sequences = {nll.mean():.3f} +/- {nll.std():.3f} (entropy {H:.3f}); "
      f"the single most probable sequence (all 1s) has NLL {-math.log(p1):.3f}, and occurs with probability {p1 ** n_sym:.1e}")
