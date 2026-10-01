"""Chapter 12: attention, multi-head attention, RoPE, online softmax (FlashAttention idea), linear attention.

Every tensor shape is annotated.  Notation: B batch, H heads, L length, d model width, dh = d / H head width.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------------------------
# 1. Why divide by sqrt(dk): the variance of a dot product of independent unit-variance vectors is dk.
# ---------------------------------------------------------------------------------------------
for dk in (16, 64, 256):
    q, k = torch.randn(20000, dk), torch.randn(20000, dk)
    s = (q * k).sum(-1)
    print(f"dk={dk:4d}: var(q.k) = {s.var().item():7.1f} (theory {dk});  var(q.k/sqrt(dk)) = {(s / math.sqrt(dk)).var().item():.2f}")
scores = torch.randn(8) * 8                                       # unscaled-looking logits: softmax saturates
print(f"  softmax of logits with std 8: max prob = {F.softmax(scores, -1).max().item():.3f} (nearly one-hot; gradients vanish)")

# ---------------------------------------------------------------------------------------------
# 2. Scaled dot-product attention and multi-head attention (from scratch).
# ---------------------------------------------------------------------------------------------
def attention(Q, K, V, causal=False):
    """Q,K,V: (B,H,L,dh) -> output (B,H,L,dh), weights (B,H,L,L)."""
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])       # (B,H,L,L)  = einsum('bhqd,bhkd->bhqk')
    if causal:
        L = Q.shape[-2]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))            # position i may not look at j > i
    w = F.softmax(scores, dim=-1)                                   # rows sum to 1
    return w @ V, w                                                 # (B,H,L,dh)

class MultiHeadAttention(nn.Module):
    def __init__(self, d, H):
        super().__init__(); self.H, self.dh = H, d // H
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Linear(d, d) for _ in range(4))
    def split(self, x):                                              # (B,L,d) -> (B,H,L,dh)
        B, L, d = x.shape
        return x.view(B, L, self.H, self.dh).transpose(1, 2)
    def forward(self, x, causal=False, rope=None):
        Q, K, V = self.split(self.Wq(x)), self.split(self.Wk(x)), self.split(self.Wv(x))
        if rope is not None: Q, K = rope(Q), rope(K)
        out, w = attention(Q, K, V, causal)
        B, H, L, dh = out.shape
        return self.Wo(out.transpose(1, 2).reshape(B, L, H * dh)), w   # concatenate heads, mix with Wo

B, L, d, H = 2, 12, 32, 4
mha = MultiHeadAttention(d, H); x = torch.randn(B, L, d)
y, w = mha(x)
ref = F.scaled_dot_product_attention(mha.split(mha.Wq(x)), mha.split(mha.Wk(x)), mha.split(mha.Wv(x)))
print(f"\nshapes: x {tuple(x.shape)} -> y {tuple(y.shape)}, attention weights {tuple(w.shape)}; "
      f"max diff vs F.scaled_dot_product_attention = {(attention(mha.split(mha.Wq(x)), mha.split(mha.Wk(x)), mha.split(mha.Wv(x)))[0] - ref).abs().max().item():.1e}")

# Permutation equivariance: without positional information, reordering the inputs just reorders the outputs.
perm = torch.randperm(L)
y_perm, _ = mha(x[:, perm])
print(f"permutation equivariance (no positional encoding): max |f(Px) - P f(x)| = {(y_perm - y[:, perm]).abs().max().item():.1e}")

# ---------------------------------------------------------------------------------------------
# 3. Rotary position embeddings: q_m . k_n depends only on the offset n - m.
# ---------------------------------------------------------------------------------------------
class RoPE:
    def __init__(self, dh, base=10000.0):
        self.theta = base ** (-torch.arange(0, dh, 2).float() / dh)  # (dh/2,) rotation frequencies
    def __call__(self, x, offset=0):                                  # x: (B,H,L,dh)
        L = x.shape[-2]; ang = (torch.arange(L) + offset)[:, None] * self.theta[None]   # (L, dh/2)
        cos, sin = ang.cos(), ang.sin(); x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x); out[..., 0::2] = x1 * cos - x2 * sin; out[..., 1::2] = x1 * sin + x2 * cos
        return out
rope = RoPE(16)
q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
vals = [(rope(q, offset=m) * rope(k, offset=m + 7)).sum().item() for m in (0, 5, 100, 1000)]
print(f"RoPE: q_m . k_(m+7) for m = 0, 5, 100, 1000 -> {[round(v, 5) for v in vals]}  (identical: depends only on the offset)")

# ---------------------------------------------------------------------------------------------
# 4. Online (blockwise) softmax attention: exact result without materialising the L x L matrix.
# ---------------------------------------------------------------------------------------------
def flash_like(Q, K, V, block=16):
    """Single head: Q,K,V (L,dh). Keep running max m, normaliser l, and weighted sum o for every query."""
    L, dh = Q.shape; m = torch.full((L,), float("-inf")); l = torch.zeros(L); o = torch.zeros(L, dh)
    for j in range(0, K.shape[0], block):
        s = Q @ K[j:j + block].T / math.sqrt(dh)                      # (L, block): only a tile of the scores exists
        m_new = torch.maximum(m, s.max(-1).values)
        p = torch.exp(s - m_new[:, None]); scale = torch.exp(m - m_new)   # rescale old accumulators to the new max
        l = scale * l + p.sum(-1); o = scale[:, None] * o + p @ V[j:j + block]; m = m_new
    return o / l[:, None]
Q1, K1, V1 = (torch.randn(200, 32) for _ in range(3))
print(f"online softmax (blockwise) vs full attention: max diff = {(flash_like(Q1, K1, V1) - attention(Q1[None, None], K1[None, None], V1[None, None])[0][0, 0]).abs().max().item():.1e}")

# ---------------------------------------------------------------------------------------------
# 5. Linear attention: replace softmax(q.k) by phi(q).phi(k); associativity gives O(L d^2) instead of O(L^2 d).
# ---------------------------------------------------------------------------------------------
phi = lambda x: F.elu(x) + 1                                          # positive feature map
Qf, Kf = phi(Q1), phi(K1)
quadratic = (Qf @ Kf.T) / (Qf @ Kf.T).sum(-1, keepdim=True) @ V1      # (phi(Q) phi(K)^T) V  : O(L^2 d)
kv = Kf.T @ V1; z = Kf.sum(0)                                         # sum_j phi(k_j) v_j^T (dh x dh), sum_j phi(k_j)
linear = (Qf @ kv) / (Qf @ z)[:, None]                                # phi(Q) (phi(K)^T V)  : O(L d^2)
print(f"linear attention by associativity: max |quadratic - linear| = {(quadratic - linear).abs().max().item():.1e}")

# ---------------------------------------------------------------------------------------------
# 6. Parameter and FLOP accounting for one transformer layer (pre-LN: attention + MLP with 4d hidden).
# ---------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, H):
        super().__init__(); self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, H); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x): x = x + self.attn(self.ln1(x))[0]; return x + self.mlp(self.ln2(x))
for d_ in (256, 1024):
    n = sum(p.numel() for p in Block(d_, 8).parameters())
    print(f"d={d_:5d}: parameters per layer = {n:,} ; 12 d^2 = {12 * d_ ** 2:,} ; forward FLOPs/token at L=4096: {24 * d_ ** 2 + 4 * 4096 * d_:,}")
