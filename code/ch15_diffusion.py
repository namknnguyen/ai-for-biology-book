"""Chapter 15: DDPM, flow matching, Tweedie's formula, and classifier-free guidance on a 2-D toy problem."""
import math
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0); rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------
# Exact identities of the DDPM forward process (checked numerically)
# ---------------------------------------------------------------------------------------------
T = 200
betas = torch.linspace(1e-4, 0.05, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
x0 = torch.tensor([1.3, -0.7]); n = 400_000
x = x0.expand(n, 2).clone()
for t in range(40):                                                 # simulate 40 forward steps one at a time
    x = alphas[t].sqrt() * x + betas[t].sqrt() * torch.randn_like(x)
print(f"forward marginal after 40 steps: simulated mean {x.mean(0).numpy().round(3)} var {x.var(0).numpy().round(3)}; "
      f"closed form mean {(abar[39].sqrt() * x0).numpy().round(3)} var {(1 - abar[39]).item():.3f}")

t = 40; xt = abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * torch.randn(2)
mu_post = (abar[t - 1].sqrt() * betas[t] / (1 - abar[t])) * x0 + (alphas[t].sqrt() * (1 - abar[t - 1]) / (1 - abar[t])) * xt
# brute-force posterior q(x_{t-1}|x_t, x_0) by Bayes on a 1-D grid (first coordinate)
g = torch.linspace(-6, 6, 400001).double(); s_prev, s_t = (1 - abar[t - 1]).double(), betas[t].double()
log_post = (-(g - abar[t - 1].double().sqrt() * x0[0].double()) ** 2 / (2 * s_prev)
            - (xt[0].double() - alphas[t].double().sqrt() * g) ** 2 / (2 * s_t))
w = torch.softmax(log_post, 0); print(f"posterior mean of x_(t-1): formula {mu_post[0].item():.5f}, numerical Bayes {(w * g).sum().item():.5f}")

# Tweedie: E[x0 | xt] = (xt + sigma^2 * score(xt)) for xt = x0 + sigma * eps   (Gaussian data, closed form)
s0, sig, xt_val = 1.5, 0.8, 2.3
score = -xt_val / (s0 ** 2 + sig ** 2)                               # d/dxt log N(xt; 0, s0^2 + sig^2)
print(f"Tweedie: posterior mean s0^2/(s0^2+sig^2)*xt = {s0 ** 2 / (s0 ** 2 + sig ** 2) * xt_val:.5f}; xt + sig^2*score = {xt_val + sig ** 2 * score:.5f}")

# ---------------------------------------------------------------------------------------------
# Toy data: four Gaussians at (+-2, +-2), class = which mode.  Train DDPM (noise prediction) and flow matching.
# ---------------------------------------------------------------------------------------------
centers = torch.tensor([[2., 2.], [-2., 2.], [-2., -2.], [2., -2.]]); SD = 0.3
def sample_data(n):
    c = torch.randint(0, 4, (n,)); return centers[c] + SD * torch.randn(n, 2), c

class Net(nn.Module):                                               # input: (x, time, class or 'null' class 4)
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(5, 16)
        s.f = nn.Sequential(nn.Linear(2 + 1 + 16, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(s, x, t, c): return s.f(torch.cat([x, t[:, None], s.emb(c)], -1))

def train(loss_fn, steps=5000):
    net = Net(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(steps):
        x0, c = sample_data(512); c = torch.where(torch.rand(512) < 0.15, torch.full_like(c, 4), c)   # drop label 15%: CFG training
        loss = loss_fn(net, x0, c); opt.zero_grad(); loss.backward(); opt.step()
    return net

def ddpm_loss(net, x0, c):
    t = torch.randint(0, T, (len(x0),)); eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    return ((net(xt, t.float() / T, c) - eps) ** 2).mean()           # simplified DDPM loss: predict the noise

def fm_loss(net, x1, c):                                             # flow matching: x_t = (1-t) x0 + t x1, target velocity x1 - x0
    x0 = torch.randn_like(x1); t = torch.rand(len(x1)); xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    return ((net(xt, t, c) - (x1 - x0)) ** 2).mean()

@torch.no_grad()
def sample_ddpm(net, n, cls, w=0.0):                                  # ancestral sampling with classifier-free guidance weight w
    x = torch.randn(n, 2); c = torch.full((n,), cls); null = torch.full((n,), 4)
    for t in reversed(range(T)):
        tt = torch.full((n,), t / T)
        eps = (1 + w) * net(x, tt, c) - w * net(x, tt, null)         # eps_guided = (1+w) eps(x|y) - w eps(x)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x

@torch.no_grad()
def sample_fm(net, n, cls, steps=50):                                 # Euler integration of dx/dt = v(x, t) from noise to data
    x = torch.randn(n, 2); c = torch.full((n,), cls)
    for k in range(steps): x = x + net(x, torch.full((n,), k / steps), c) / steps
    return x

def report(name, xs):
    d = torch.cdist(xs, centers); near = d.argmin(1)
    cover = torch.bincount(near, minlength=4).float() / len(xs)
    within = torch.stack([xs[near == k].std(0).mean() for k in range(4) if (near == k).sum() > 5]).mean()
    print(f"{name:34s} mode coverage {cover.numpy().round(2)}  within-mode sd {within:.3f} (true {SD})  mean dist to nearest mode {d.min(1).values.mean():.3f}")

net_d = train(ddpm_loss); net_f = train(fm_loss)
# unconditional samples: use the 'null' class (index 4)
report("DDPM (200 steps), unconditional", sample_ddpm(net_d, 4000, 4))
report("flow matching (50 Euler steps)", sample_fm(net_f, 4000, 4))
print("classifier-free guidance, DDPM, target mode 0 = (+2,+2):")
for w in (0.0, 1.0, 3.0, 6.0):
    xs = sample_ddpm(net_d, 3000, 0, w); d0 = (xs - centers[0]).norm(dim=1)
    print(f"  w = {w:3.1f}: fraction within 1.0 of target mode = {(d0 < 1.0).float().mean():.3f}; sd around the mode = {xs[d0 < 1.0].std(0).mean():.3f}")
