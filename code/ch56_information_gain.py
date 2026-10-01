"""Chapter 56: choosing among experiments by expected information gain, with a worked example.
Hypotheses (why do sequence models ignore distal enhancers?): H1 distal effects are genuinely weak/context-specific; H2 the training data do not identify them;
H3 the architecture does not use distal context. Experiments return noisy binary outcomes with the likelihoods below (illustrative, elicited, not measured)."""
import itertools, numpy as np

H = ["H1: biology (effects weak)", "H2: data (not identified)", "H3: architecture (context unused)"]
prior = np.array([0.30, 0.40, 0.30])
# P(outcome = 1 | hypothesis) for each experiment
E = {
    "CRISPRi screen of 200 enhancers (outcome 1 = many strong distal effects)":        (np.array([0.10, 0.85, 0.85]), 100.0),
    "reporter assay (MPRA) of enhancer-promoter pairs (1 = strong effects in reporter)": (np.array([0.20, 0.85, 0.85]), 25.0),
    "mask distal context at inference (1 = predictions change)":                       (np.array([0.20, 0.55, 0.05]), 1.0),
    "retrain with 10x single-edit data (1 = distal edit effects recovered)":            (np.array([0.10, 0.80, 0.15]), 40.0),
    "shuffle enhancer sequence in silico (1 = model response proportional to measured)":(np.array([0.10, 0.20, 0.10]), 2.0),
}
def entropy(p): p = p[p > 0]; return -(p * np.log2(p)).sum()
def eig(prior, lik):
    """Mutual information between hypothesis and the binary outcome."""
    p1 = (prior * lik).sum(); post1 = prior * lik / p1; post0 = prior * (1 - lik) / (1 - p1)
    return entropy(prior) - (p1 * entropy(post1) + (1 - p1) * entropy(post0)), p1, post0, post1
print(f"prior over hypotheses: {dict(zip(['H1','H2','H3'], [float(v) for v in prior]))}; prior entropy = {entropy(prior):.3f} bits (maximum {np.log2(3):.3f})\n")
print("experiment                                                                           cost (rel.)   EIG (bits)   EIG per unit cost   P(outcome=1)")
rows = []
for name, (lik, cost) in E.items():
    g, p1, _, _ = eig(prior, lik); rows.append((name, cost, g, g / cost, p1))
for r in sorted(rows, key=lambda r: -r[3]): print(f"{r[0]:84s} {r[1]:8.1f}      {r[2]:7.3f}      {r[3]:9.4f}          {r[4]:.2f}")

# greedy sequential design: after each experiment update by the more likely outcome path in expectation (report the expected posterior entropy)
print("\nsequential strategy: pick the best experiment per unit cost, update, repeat (expected entropy after each step; outcomes enumerated)")
def expected_entropy_after(prior_vec, seq):
    tot = 0.0
    for outcomes in itertools.product([0, 1], repeat=len(seq)):
        p = prior_vec.copy(); pr = 1.0
        for o, lik in zip(outcomes, seq):
            like = lik if o == 1 else 1 - lik; pr *= (p * like).sum(); p = p * like / (p * like).sum()
        tot += pr * entropy(p)
    return tot
chosen = []; names = list(E)
remaining = names.copy(); total_cost = 0
for step in range(3):
    best = None
    for nm in remaining:
        seq = [E[x][0] for x in chosen] + [E[nm][0]]
        gain = entropy(prior) - expected_entropy_after(prior, seq)
        marginal = gain - (entropy(prior) - expected_entropy_after(prior, [E[x][0] for x in chosen]) if chosen else 0)
        score = marginal / E[nm][1]
        if best is None or score > best[0]: best = (score, nm, gain, marginal)
    chosen.append(best[1]); remaining.remove(best[1]); total_cost += E[best[1]][1]
    print(f"step {step + 1}: {best[1][:70]:70s} cumulative information {best[2]:.3f} bits (marginal {best[3]:.3f}), cumulative cost {total_cost:.0f}")
print(f"\ncumulative information of all five experiments: {entropy(prior) - expected_entropy_after(prior, [E[x][0] for x in names]):.3f} bits (upper bound = prior entropy {entropy(prior):.3f} bits)")

# sensitivity of the ranking to the elicited likelihoods
rng = np.random.default_rng(0); wins = {k: 0 for k in E}
for _ in range(2000):
    best = None
    for nm, (lik, cost) in E.items():
        noisy = np.clip(lik + rng.normal(0, 0.1, 3), 0.02, 0.98); g = eig(prior, noisy)[0] / cost
        if best is None or g > best[0]: best = (g, nm)
    wins[best[1]] += 1
print("\nrobustness: fraction of 2,000 draws (likelihoods perturbed by N(0, 0.1^2)) in which each experiment has the highest EIG per unit cost")
for nm, w in sorted(wins.items(), key=lambda kv: -kv[1]): print(f"  {w / 2000:5.2f}  {nm}")
