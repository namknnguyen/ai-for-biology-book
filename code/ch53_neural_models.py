"""Chapter 53: two measurement-and-identifiability questions for models of neural systems, in simulation.
1. A noise ceiling for neural response prediction: what fraction of the explainable variance does a 'digital twin' capture when single-trial R^2 looks poor?
2. Is a connectome enough? A rate network whose wiring (which neurons connect, with how many synapses, excitatory or inhibitory) is KNOWN but whose synaptic strengths are not;
   models constrained by the connectome are compared with an unconstrained model as the number of recorded stimuli and the heterogeneity of synaptic strengths vary."""
import numpy as np, torch
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. noise ceiling for single-trial response prediction
print("== 1. Predicting single-trial responses: 120 neurons, 300 stimuli, 8 repeats, Poisson spiking with shared gain fluctuations ==")
Nn, S, R = 120, 300, 8
stim = rng.standard_normal((S, 6)); Wt = rng.standard_normal((6, Nn)) * 0.5
rate = np.exp(0.6 + np.tanh(stim @ Wt) * 1.1)                                             # mean spike count per stimulus and neuron (about 1.8 to 15)
gain = np.exp(0.25 * rng.standard_normal((S, R, 1)))                                       # trial-to-trial gain shared across neurons
counts = rng.poisson(rate[:, None, :] * gain)                                              # (S, R, Nn)
train = np.arange(S) < 200; test = ~train
def r2(pred, y): return 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean(0)) ** 2)
# a flexible model: random-feature ridge regression on the stimulus, trained on trial-averaged counts of the training stimuli
F = lambda s: np.c_[np.tanh(s @ np.random.default_rng(1).standard_normal((6, 400)) / 2), np.ones(len(s))]
mu_tr = counts[train].mean(1); wr = np.linalg.solve(F(stim[train]).T @ F(stim[train]) + 1.0 * np.eye(401), F(stim[train]).T @ mu_tr); pred = F(stim[test]) @ wr
single = counts[test].reshape(-1, Nn); predrep = np.repeat(pred[:, None, :], R, 1).reshape(-1, Nn)
r2_single = r2(predrep, single)
# ceiling: predicting a held-out trial with the mean of the OTHER trials of the same stimulus (the best any stimulus-only model can do, up to finite-sample noise)
ceil = []
for k in range(R):
    others = np.delete(counts[test], k, axis=1).mean(1); ceil.append(r2(others, counts[test][:, k]))
r2_ceiling = np.mean(ceil)
true_mean = rate[test] * np.exp(0.25 ** 2 / 2)
print(f"single-trial R^2 of the model: {r2_single:.3f};   reliability ceiling from the other 7 trials: {r2_ceiling:.3f};   model / ceiling = {r2_single / r2_ceiling:.2f}")
print(f"R^2 of the model against the TRUE mean response (which the experimenter cannot see): {r2(pred, true_mean):.3f};   single-trial R^2 of the true mean itself: {r2(np.repeat(true_mean[:, None, :], R, 1).reshape(-1, Nn), single):.3f}")
sh = counts[test].sum(2, keepdims=True); resid = counts[test] - counts[test].mean(1, keepdims=True)
print(f"noise correlation between neurons (shared gain): mean pairwise correlation of residuals = {np.mean(np.corrcoef(resid.reshape(-1, Nn).T)[np.triu_indices(Nn, 1)]):.3f}")

# ---------------------------------------------------------------- 2. connectome-constrained networks
N = 60
def make_world(sigma, seed):
    r = np.random.default_rng(seed); conn = r.random((N, N)) < 0.12; np.fill_diagonal(conn, False)
    sign = np.where(np.arange(N) < int(0.75 * N), 1.0, -1.0)                              # Dale's law: the presynaptic neuron's sign (excitatory 75%) is known from the connectome
    counts = r.poisson(3.0, (N, N)) * conn + conn                                          # synapse counts (at least 1 where connected)
    g = np.exp(sigma * r.standard_normal((N, N)) - sigma ** 2 / 2)                         # per-connection synaptic strength: heterogeneity sigma, NOT observed
    W = sign[None, :] * counts * g
    W = W / (1.25 * np.max(np.abs(np.linalg.eigvals(W))))                                   # spectral radius 0.8: a stable network
    return conn, sign, counts, W
def steady(W, U, iters=60):
    r = torch.zeros(U.shape[0], N)
    for _ in range(iters): r = torch.tanh(r @ W.T + U)
    return r
def fit(mask, W_init, U, Y, steps=500, lr=0.01, l2=1e-4):
    """Fit W (restricted to 'mask') by gradient descent on the steady-state response error."""
    P = torch.tensor(W_init, dtype=torch.float32, requires_grad=True); M = torch.tensor(mask, dtype=torch.float32)
    opt = torch.optim.Adam([P], lr)
    for _ in range(steps):
        loss = ((steady(P * M, U) - Y) ** 2).mean() + l2 * (P * M).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return (P * M).detach()
def corr(a, b): a = a.flatten() - a.mean(); b = b.flatten() - b.mean(); return float((a @ b) / np.sqrt((a @ a) * (b @ b)))
print(f"\n== 2. Is the wiring diagram enough? {N}-neuron rate network; connectome known (who connects, synapse counts, signs); synaptic strengths unknown (log-normal, spread sigma) ==")
print("models: A = counts x sign (one global scale fit); B = connectome-constrained: weights fit freely on the known connections only, from small random values; C = unconstrained: all N^2 weights fit freely from small random values")
print("quantities: held-out response correlation r (new stimuli) and the correlation of predicted vs true changes in responses when one neuron is silenced ('ablation')")
print("sigma  stimuli   A: counts only        B: connectome-constrained     C: unconstrained        (response r / ablation r)")
for sigma in [0.0, 0.5, 1.0]:
    for K in [30, 150, 600]:
        res = {"A": [], "B": [], "C": []}
        for seed in range(4):
            conn, sign, counts, Wtrue = make_world(sigma, seed); Wtrue_t = torch.tensor(Wtrue, dtype=torch.float32); r = np.random.default_rng(100 + seed)
            U = torch.tensor(r.standard_normal((K, N)) * 1.0, dtype=torch.float32); Y = steady(Wtrue_t, U) + 0.05 * torch.randn(K, N)
            Ute = torch.tensor(r.standard_normal((100, N)), dtype=torch.float32); Yte = steady(Wtrue_t, Ute)
            abl = r.choice(N, 6, replace=False)
            def ablation_r(Wm):
                a, b = [], []
                for j in abl:
                    Wa = Wm.clone(); Wa[:, j] = 0; Wt_ = Wtrue_t.clone(); Wt_[:, j] = 0
                    keep = np.arange(N) != j; a.append((steady(Wa, Ute) - steady(Wm, Ute))[:, keep].numpy()); b.append((steady(Wt_, Ute) - Yte)[:, keep].numpy())
                return corr(np.concatenate(a), np.concatenate(b))
            W0 = (sign[None, :] * counts).astype(np.float32); W0 = W0 / (1.25 * np.max(np.abs(np.linalg.eigvals(W0)))); W0t = torch.tensor(W0, dtype=torch.float32)
            sc = min(np.linspace(0.3, 2.0, 18), key=lambda c: float(((steady(c * W0t, U) - Y) ** 2).mean())); WA = sc * W0t                                   # model A: one global scale
            init = (0.02 * np.random.default_rng(7).standard_normal((N, N))).astype(np.float32); WB = fit(conn, init, U, Y); WC = fit(np.ones((N, N)), init, U, Y)                          # B and C start from small random weights
            for k, Wm in zip("ABC", [WA, WB, WC]): res[k].append((corr(steady(Wm, Ute).numpy(), Yte.numpy()), ablation_r(Wm)))
        m = {k: np.mean(v, 0) for k, v in res.items()}
        print(f"{sigma:4.1f}   {K:5d}    {m['A'][0]:.3f} / {m['A'][1]:6.3f}         {m['B'][0]:.3f} / {m['B'][1]:6.3f}              {m['C'][0]:.3f} / {m['C'][1]:6.3f}")
