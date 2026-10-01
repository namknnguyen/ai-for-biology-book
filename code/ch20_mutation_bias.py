"""Chapter 20: mutation bias vs selection. A frequency-based variant score confounds the two (Chapter 5, section 5.6.4)."""
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
S, LINEAGES, TAU = 6000, 800, 0.02                      # sites, independent lineages, evolutionary time (arbitrary units)
BASES = "ACGT"; TS = {0: 2, 2: 0, 1: 3, 3: 1}           # transition partner: A<->G, C<->T
# ancestral sequence with CpG depletion-free composition (random), so many CpG sites exist
anc = rng.integers(0, 4, S)
is_cpg_C = np.zeros(S, bool); is_cpg_G = np.zeros(S, bool)
for i in range(S - 1):
    if anc[i] == 1 and anc[i + 1] == 2: is_cpg_C[i] = True; is_cpg_G[i + 1] = True
constrained = rng.random(S) < 0.20                      # 20% of sites under purifying selection
purge = np.where(constrained, rng.uniform(0.3, 0.9, S), 0.0)   # fraction of mutations removed by selection at each site (graded)

# mutation rates per site and alternative base (arbitrary units): transitions 1.0, transversions 0.25 each (Ts/Tv = 2);
# at CpG, C->T and G->A (methylated cytosine deamination) are 10x faster.
rate = np.zeros((S, 4))
for i in range(S):
    b = anc[i]
    for a in range(4):
        if a == b: continue
        rate[i, a] = 1.0 if a == TS[b] else 0.25
    if (is_cpg_C[i] and b == 1) or (is_cpg_G[i] and b == 2): rate[i, TS[b]] *= 10.0

# evolve independent lineages; at constrained sites a graded fraction of mutations is purged (selection)
alt_count = np.zeros((S, 4))
for a in range(4):
    p = 1 - np.exp(-rate[:, a] * TAU)                                   # probability of the substitution on a lineage
    hit = rng.random((LINEAGES, S)) < p[None]
    hit &= rng.random((LINEAGES, S)) >= purge[None]
    alt_count[:, a] = hit.sum(0)
freq = alt_count / LINEAGES
ref_freq = 1 - freq.sum(1)

rows = []                                                                 # all (site, alt) substitutions
for i in range(S):
    for a in range(4):
        if a == anc[i]: continue
        kind = "CpG transition" if rate[i, a] >= 5 else ("transition" if a == TS[anc[i]] else "transversion")
        rows.append((i, a, kind, constrained[i], np.log((freq[i, a] + 1e-3) / (ref_freq[i] + 1e-3)), np.log(rate[i, a])))
kind = np.array([r[2] for r in rows]); cons = np.array([r[3] for r in rows]); llr = np.array([r[4] for r in rows]); logmu = np.array([r[5] for r in rows])

print("mean log-likelihood-ratio score  log(f_alt / f_ref)  for NEUTRAL substitutions, by mutation class:")
for k in ("CpG transition", "transition", "transversion"):
    m = (kind == k) & ~cons
    print(f"  {k:15s}: mean score {llr[m].mean():6.2f}  (mean substitution frequency {np.exp(llr[m]).mean():.3f})")
print(f"  constrained sites (all classes): mean score {llr[cons].mean():6.2f}")

def auc(score, mask=None):
    mask = np.ones(len(score), bool) if mask is None else mask
    return roc_auc_score(cons[mask], -score[mask])                        # low score (rare alt allele) => predicted constrained
print(f"\nAUROC for separating constrained from neutral substitutions:")
print(f"  raw score, all substitution classes pooled          : {auc(llr):.3f}")
corrected = llr - logmu                                                   # subtract the log neutral expectation from the mutation model
print(f"  score corrected for mutation rate, pooled           : {auc(corrected):.3f}")
for k in ("CpG transition", "transition", "transversion"):
    m = kind == k
    print(f"  raw score within {k:15s} only                  : {auc(llr, m):.3f}  (n = {m.sum()})")
