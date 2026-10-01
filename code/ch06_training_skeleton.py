"""Chapter 6: a minimal, testable PyTorch research skeleton for sequence models.

Task: classify DNA windows that contain a planted motif on either strand.
Includes the unit tests an experienced researcher writes before trusting any model:
shape asserts, overfit-one-batch, reverse-complement symmetry, shuffled-label control.
"""
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
BASES = "ACGT"
MOTIF = "TGACGTCA"                                       # a palindromic-ish CRE-like motif (illustrative)
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
revcomp = lambda s: "".join(comp[c] for c in reversed(s))

def one_hot(seqs):                                        # -> (B, 4, L)
    idx = torch.tensor([[BASES.index(c) for c in s] for s in seqs])
    return F.one_hot(idx, 4).permute(0, 2, 1).float()

def make_data(n, L=100, p_pos=0.5, g=None):
    seqs, y = [], []
    for _ in range(n):
        s = [BASES[i] for i in torch.randint(0, 4, (L,), generator=g)]
        label = int(torch.rand(1, generator=g) < p_pos)
        if label:
            m = MOTIF if torch.rand(1, generator=g) < 0.5 else revcomp(MOTIF)
            pos = int(torch.randint(0, L - len(m), (1,), generator=g))
            s[pos:pos + len(m)] = list(m)
        seqs.append("".join(s)); y.append(label)
    return seqs, torch.tensor(y).float()

class MotifCNN(nn.Module):
    def __init__(self, n_filters=16, width=8):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, width)        # (B,4,L) -> (B,F,L-w+1): F motif detectors
        self.head = nn.Linear(n_filters, 1)
    def forward(self, x):                                  # x: (B, 4, L)
        h = F.relu(self.conv(x)).amax(dim=-1)              # global max pool over positions -> (B, F)
        return self.head(h).squeeze(-1)                    # logits (B,)

def rc_tensor(x):                                          # reverse complement of one-hot (B,4,L): flip L, swap A<->T, C<->G
    return x.flip(-1)[:, [3, 2, 1, 0], :]

class RCSymmetric(nn.Module):
    """Wrap any model so that f(x) == f(revcomp(x)) exactly (symmetrisation)."""
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return 0.5 * (self.net(x) + self.net(rc_tensor(x)))

def train(model, X, y, steps=200, lr=1e-2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    for t in range(steps):
        logits = model(X)
        assert logits.shape == y.shape, (logits.shape, y.shape)   # shape discipline
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()

def accuracy(model, X, y):
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()

g = torch.Generator().manual_seed(0)
seqs, y = make_data(2000, g=g); X = one_hot(seqs)
Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]
print("shapes:", tuple(X.shape), tuple(y.shape), " params:", sum(p.numel() for p in MotifCNN().parameters()))

# Test 1: overfit one small batch (a failure here means a bug, not a scientific problem)
m = MotifCNN(); loss = train(m, Xtr[:32], ytr[:32], steps=300)
print(f"[test 1] overfit 32 examples: final loss = {loss:.4f}  (should be near 0)")

# Test 2: full training; held-out accuracy
m = MotifCNN(); train(m, Xtr, ytr, steps=300)
print(f"[test 2] held-out accuracy = {accuracy(m, Xte, yte):.3f}")

# Test 3: reverse-complement symmetry (plain model is not exactly symmetric; the wrapper is)
msym = RCSymmetric(m)
with torch.no_grad():
    d_plain = (m(Xte) - m(rc_tensor(Xte))).abs().max().item()
    d_sym = (msym(Xte) - msym(rc_tensor(Xte))).abs().max().item()
print(f"[test 3] max |f(x) - f(rc(x))|: plain = {d_plain:.4f}, symmetrised = {d_sym:.2e}")

# Test 4: shuffled-label control (the model must NOT generalise; if it does, there is leakage)
perm = torch.randperm(len(ytr), generator=g)
m = MotifCNN(); train(m, Xtr, ytr[perm], steps=300)
print(f"[test 4] shuffled-label control: held-out accuracy = {accuracy(m, Xte, yte):.3f}  (should be ~0.5)")
