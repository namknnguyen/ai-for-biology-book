"""Chapter 34: why a model pretrained across protein families can help a new family, and when it cannot (a controlled toy).
Families are Potts models over L = 30 positions, q = 4 states. 'Related' families share most of their coupling structure (a common fold); 'unrelated' families do not.
A masked-position neural network is pretrained on 20 families (sequence only, no family label), then adapted to a NEW family with N sequences.
Zero-shot variant-effect benchmark: Spearman correlation between the model's log-probability change and the true energy change for all single mutants of 25 wild types."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2); torch.manual_seed(0)
L, q = 30, 4
def make_family(rs, shared=None, strength=0.0):
    h = rs.normal(0, 0.3, (L, q)); J = np.zeros((L, L, q, q))
    for i in range(L):
        for j in range(i + 4, L):
            if rs.random() < 0.10: M = rs.normal(0, 1.0, (q, q)); J[i, j] = M; J[j, i] = M.T
    if shared is not None:                                                     # blend with a shared fold: J = strength*shared + (1-strength)*own
        h = strength * shared[0] + (1 - strength) * h; J = strength * shared[1] + (1 - strength) * J
    return h, J
def gibbs(h, J, n, sweeps, rs):
    s = rs.integers(0, q, (n, L))
    for _ in range(sweeps):
        for i in rs.permutation(L):
            lg = h[i][None, :] + J[i, np.arange(L)[None, :], :, s].sum(1)
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            s[:, i] = (rs.random(n)[:, None] > np.cumsum(p, 1)).sum(1).clip(max=q - 1)
    return s
def energy(h, J, S): return h[np.arange(L)[None, :], S].sum(1) + 0.5 * sum(J[i, np.arange(L)[None, :], S[:, i][:, None], S].sum(1) for i in range(L))
def mutants(S0):
    out = []
    for s in S0:
        for i in range(L):
            for a in range(q):
                if a != s[i]: m = s.copy(); m[i] = a; out.append((s, i, a, m))
    return out

class MaskedNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(L * q + L, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, q))
    def forward(self, S, pos):                                                 # S: (B, L) ints, pos: (B,) masked position
        oh = F.one_hot(S, q).float(); oh[torch.arange(len(S)), pos] = 0; m = F.one_hot(pos, L).float()
        return self.net(torch.cat([oh.reshape(len(S), -1), m], 1))
def train_net(net, S, epochs, lr=2e-3):
    S = torch.tensor(S); opt = torch.optim.Adam(net.parameters(), lr)
    for ep in range(epochs):
        perm = torch.randperm(len(S))
        for b in range(0, len(S), 256):
            x = S[perm[b:b + 256]]; pos = torch.randint(0, L, (len(x),)); loss = F.cross_entropy(net(x, pos), x[torch.arange(len(x)), pos]); opt.zero_grad(); loss.backward(); opt.step()
    return net
def score_net(net, muts):
    s = torch.tensor(np.stack([m[0] for m in muts])); pos = torch.tensor([m[1] for m in muts]); a = torch.tensor([m[2] for m in muts])
    with torch.no_grad(): lp = F.log_softmax(net(s, pos), 1)
    wt = s[torch.arange(len(s)), pos]; return (lp[torch.arange(len(s)), a] - lp[torch.arange(len(s)), wt]).numpy()
def score_independent(S, muts):
    f = np.stack([np.bincount(S[:, i], minlength=q) + 0.5 for i in range(L)]); f = f / f.sum(1, keepdims=True)
    return np.array([np.log(f[m[1], m[2]]) - np.log(f[m[1], m[0][m[1]]]) for m in muts])
def score_potts(S, muts, C=0.05):
    oh = np.eye(q)[S].reshape(len(S), -1); lps = {}
    for i in range(L):
        cols = [j * q + b for j in range(L) if j != i for b in range(q)]
        clf = LogisticRegression(C=C, max_iter=300).fit(oh[:, cols], S[:, i]); lps[i] = (clf, cols)
    out = []
    for (s, i, a, mm) in muts:
        clf, cols = lps[i]; x = np.eye(q)[s].reshape(1, -1)[:, cols]; pr = np.full(q, 1e-3); pr[clf.classes_] = clf.predict_proba(x)[0]; out.append(np.log(pr[a]) - np.log(pr[s[i]]))
    return np.array(out)
def spearman_by_wt(score, truth, n_per):
    return np.mean([spearmanr(score[k * n_per:(k + 1) * n_per], truth[k * n_per:(k + 1) * n_per])[0] for k in range(len(score) // n_per)])

def experiment(relatedness, label):
    rs = np.random.default_rng(7)
    shared = make_family(rs)                                                    # the 'fold' shared by related families
    pre = [make_family(rs, shared, relatedness) for _ in range(20)]
    target = make_family(rs, shared, relatedness)
    S_pre = np.vstack([gibbs(h, J, 400, 200, rs) for h, J in pre])
    h, J = target; S_test = gibbs(h, J, 25, 200, rs); muts = mutants(S_test); n_per = L * (q - 1)
    truth = np.array([energy(h, J, m[3][None])[0] - energy(h, J, m[0][None])[0] for m in muts])
    base = MaskedNet(); train_net(base, S_pre, 25)
    print(f"\n-- {label}: pretraining on 20 families (8,000 sequences); target family adapted with N sequences --")
    print("N target     site-independent   Potts (pseudolikelihood)   network from scratch   pretrained, zero-shot   pretrained + light adaptation   pretrained + full-budget adaptation")
    for N in [20, 50, 200, 1000]:
        S_t = gibbs(h, J, N, 200, rs)
        r_ind = spearman_by_wt(score_independent(S_t, muts), truth, n_per); r_pot = spearman_by_wt(score_potts(S_t, muts), truth, n_per)
        sc = MaskedNet(); train_net(sc, np.vstack([S_t] * max(1, 4000 // N)), 8); r_scr = spearman_by_wt(score_net(sc, muts), truth, n_per)
        r_zero = spearman_by_wt(score_net(base, muts), truth, n_per)
        ft = MaskedNet(); ft.load_state_dict(base.state_dict()); train_net(ft, np.vstack([S_t] * max(1, 2000 // N)), 4, lr=5e-4); r_ft = spearman_by_wt(score_net(ft, muts), truth, n_per)
        f2 = MaskedNet(); f2.load_state_dict(base.state_dict()); train_net(f2, np.vstack([S_t] * max(1, 4000 // N)), 8, lr=1e-3); r_ft2 = spearman_by_wt(score_net(f2, muts), truth, n_per)
        print(f"{N:7d}         {r_ind:8.3f}               {r_pot:8.3f}                {r_scr:8.3f}              {r_zero:8.3f}                {r_ft:8.3f}                        {r_ft2:8.3f}")
experiment(0.8, "RELATED families (80% of the coupling structure shared)")
experiment(0.0, "UNRELATED families (no shared structure)")
