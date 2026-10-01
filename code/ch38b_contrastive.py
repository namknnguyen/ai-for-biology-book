"""Chapter 38 (second experiment): does the pretraining OBJECTIVE matter?  Same mini-FM, same 19,800 pretraining cells, same evaluation as ch38_sc_fm.py, with a contrastive term added to the
masked-gene loss: two views of the same cell, made by binomial thinning of the counts (each view keeps a random 30-100% of the UMIs) so that the embedding must be invariant to sequencing depth,
are pulled together (InfoNCE, temperature 0.2) and pushed away from other cells in the batch."""
import numpy as np, torch, torch.nn.functional as F
import ch38_sc_fm as B
from ch38_sc_fm import MiniFM, rank_tokens, fm_embed, evaluate, Ycorp, Ynew, G, PAD, MASK, ref_idx
torch.set_num_threads(2); rs = np.random.default_rng(11)
def thin(Y, r):
    p = r.uniform(0.3, 1.0, (len(Y), 1)); return r.binomial(Y.astype(np.int64), p).astype(np.float32)
def pretrain_contrastive(Y, steps, lam=1.0, temp=0.2, seed=0, use_mlm=True):
    torch.manual_seed(seed); med = np.array([np.median(Y[Y[:, g] > 0, g] / Y[Y[:, g] > 0].sum(1) * 1e4) if (Y[:, g] > 0).sum() > 5 else 1.0 for g in range(G)]) + 1e-6
    net = MiniFM(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
    for st in range(steps):
        idx = rs.integers(0, len(Y), 128); y1, y2 = thin(Y[idx], rs), thin(Y[idx], rs)
        t1, t2 = torch.tensor(rank_tokens(y1, med)), torch.tensor(rank_tokens(y2, med)); loss = 0.0
        if use_mlm:
            m = (torch.rand(t1.shape) < 0.15) & (t1 != PAD); inp = torch.where(m, torch.full_like(t1, MASK), t1); h, pad = net.hidden(inp); loss = loss + F.cross_entropy(net.out(h)[m], t1[m])
        z1, z2 = F.normalize(net.embed(t1), dim=-1), F.normalize(net.embed(t2), dim=-1); logits = z1 @ z2.T / temp; lab = torch.arange(len(idx))
        loss = loss + lam * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2; opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net, med
print("method                                         label transfer accuracy   novel-type detection AUROC   cell-type silhouette   study separation within type (0 = mixed)")
for name, kw in (("mini-FM, masked-gene loss only (from ch38_sc_fm.py)", None), ("mini-FM, masked-gene + contrastive (depth-invariance)", dict(use_mlm=True)), ("mini-FM, contrastive only", dict(use_mlm=False))):
    if kw is None:
        net, med, _ = B.pretrain(Ycorp, 800)
    else:
        net, med = pretrain_contrastive(Ycorp, 800, **kw)
    evaluate(fm_embed(net, med, Ycorp[ref_idx]), fm_embed(net, med, Ynew), name)
