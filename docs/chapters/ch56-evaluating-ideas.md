# Chapter 56. Evaluating Ideas: Novelty, Significance, Feasibility, Decisiveness

!!! abstract "Chapter at a glance"
    **Motivation.** Chapter 55 produces many ideas quickly. Most should be killed. The skill that separates productive from unproductive research programs is *triage*: deciding, before spending months, which ideas are novel, meaningful, feasible, and *decisive* (able to change what you believe whichever way they come out). This chapter turns the twelfth question of the research-skills list, "Is it novel, meaningful, and feasible?", into four tests with numbers: a prior-art search, a value-of-information argument, a resource calculation, and an expected-information-gain computation. It ends with kill criteria and a portfolio rule.
    **Prerequisites.** Chapters 1, 4, 5, 17, 55.
    **You will be able to:** (1) run a prior-art search that finds the same idea under a different name; (2) state what decision an idea changes and what a positive and negative result would do to your beliefs; (3) estimate compute, data, and time to a first signal; (4) compute the expected information gain of candidate experiments and rank them per unit cost; (5) write kill criteria and run a pre-mortem; (6) assemble a portfolio with an explicit explore/exploit balance.

---

## 56.1 Four tests

An idea passes when four questions have convincing answers.

| Test | Question | Evidence | Typical failure |
|---|---|---|---|
| **Novel** | Has this been done, under this or another name? | A documented prior-art search | "Nobody has done X" asserted after a 20-minute search |
| **Meaningful** | If true, what changes (a belief, a method, a decision)? | A named decision, a rung of the Claim Ladder, a quantified value | The result would be *interesting* but would change nothing |
| **Feasible** | Can we get the data, compute, expertise, and time? | A resource calculation and a fast path to first signal | Dependence on data that cannot be obtained or on an experiment nobody can run |
| **Decisive** | Will the result discriminate between explanations? | Expected information gain; a pre-registered decision rule | Experiments whose outcome fits every hypothesis |

The order matters. *Decisiveness* is evaluated last because it requires the other three, but it is the most important: an experiment that cannot change your mind is a demonstration, not a test.

---

## 56.2 Novelty: a prior-art search that works

Claims of novelty fail more often because of vocabulary than because of effort. The same idea appears under different names in different fields (Chapter 55's cross-domain attack in reverse): *coevolution analysis* = *inverse Ising/Potts* = *Boltzmann machine learning*; *Mendelian randomization* = *instrumental variables*; *Harmony-type batch correction* = *domain adaptation*; *global epistasis* = *generalized linear model with latent additive trait*.

**A procedure.**

1. **Write the idea as a mathematical structure** (inputs, outputs, objective, assumption changed), not as a sentence in your field's jargon.
2. **Generate three names**: your field's name; the neighboring field's name (statistics, physics, signal processing, ML, genetics); the *problem-structure* name (e.g., "inverse problem with a compositional constraint").
3. **Search with all three** across preprint servers, proceedings, and review articles; read the *related-work sections* of the two or three closest papers, which are usually a better index than the search engine.
4. **Classify each close hit** on a prior-art ladder: *identical* (stop), *equivalent under reparameterization* (stop, but note the translation), *special case* (your idea generalizes it: what do you gain?), *generalization* (can you recover your idea as a special case?), *analogue in another setting* (is the transfer the contribution?).
5. **Check recency and concurrency.** The field moves fast: search the last six months by author and keyword; inspect citations to the closest paper. Note concurrent work; it is a negative signal for priority but a *positive* signal for importance.
6. **Check negative results.** Ideas that failed are rarely published. Ask people; look in appendices and issue trackers; state what is different this time.

**What novelty is worth.** Novelty is *instrumental*: a novel idea that is not meaningful is a curiosity, and a non-novel idea with a decisive experiment can be the most valuable contribution (a replication under a stricter test, a well-powered negative result). The Claim Ladder (Chapter 1) shows why: the most valuable papers in AI for biology are often those that *move a claim up a rung with an honest evaluation*, not those that propose a new architecture.

---

## 56.3 Meaningfulness: what changes if it is true?

Use three levels.

1. **Scientific.** What belief changes? Write the sentence "Before, we thought X; now we think Y." If you cannot, the idea is not yet a hypothesis.
2. **Methodological.** What can others now do that they could not? Is it a better *baseline*, a new *diagnostic*, a new *dataset*, a new *estimator* (with proven properties)?
3. **Applied.** Which decision does it improve, by how much, for whom? In drug discovery, that is the number of cycles saved or the hit rate at fixed cost; in genomics, the fraction of variants resolved; in agriculture or biosecurity, a risk reduced (Chapter 59).

Then **place the idea on the Claim Ladder**: the *planned* evidence reaches which rung (C0–C4)? If the headline claim is C3 or C4 but the planned experiments reach only C1, either the claim must be weakened or the experiments strengthened. For a model-building idea, ask the **delta question**: *compared with the strongest baseline at equal cost, what improvement would make the community change what it uses?* A 2% gain on a benchmark that is already within 10% of its noise ceiling does not change practice (Chapter 24).

**The value-of-information view.** Let a decision have options $a\in\mathcal A$ with utility $U(a,\theta)$ for unknown $\theta$. The expected value of learning $\theta$ before deciding is
$$
\mathrm{EVPI}=\mathbb E_\theta\Big[\max_aU(a,\theta)\Big]-\max_a\mathbb E_\theta\big[U(a,\theta)\big]\ \ge 0 .
$$
It is zero when one option dominates for every $\theta$: *if the decision would not change, the experiment is not worth doing, however interesting*. For research, this is a useful test: write the decision (which target to pursue, which model to use, which perturbation set to run) and ask whether any plausible result would flip it.

---

## 56.4 Feasibility: resources and the fast path

**Data.** Does it exist (public, licensed, obtainable), at the needed size and quality? What is its noise ceiling and leakage structure (Chapters 1, 27, 28)? If it must be generated: who runs the experiment, at what cost and turnaround, and with which controls? *Time to first signal* matters more than total time.

**Compute.** For a dense neural model, training compute is approximately
$$
C\approx6\,N\,D\ \text{FLOPs}
$$
($N$ parameters, $D$ training tokens or examples; Chapter 17). Divide by the sustained throughput of your hardware, typically 30–50% of peak. Two reference computations (back-of-envelope; hardware figures are rough and rapidly improving):

* A 100-million-parameter model on $10^{10}$ tokens: $C=6\times10^{18}$ FLOPs. At an effective $1.25\times10^{14}$ FLOP/s (about 40% of a 312-TFLOP/s accelerator) that is $4.8\times10^4$ s, about **13 hours on one GPU**.
* A 7-billion-parameter model on $3\times10^{11}$ tokens: $C=1.3\times10^{22}$ FLOPs; at an effective $4\times10^{14}$ FLOP/s that is $3\times10^7$ GPU-seconds, about **360 GPU-days** (a modest cluster for days to weeks).
* A 40-billion-parameter model on $9\times10^{12}$ tokens (the scale reported for Evo 2): $C\approx2\times10^{24}$ FLOPs, about $1.5\times10^6$ GPU-hours at the same effective rate, which is why such models are built by consortia with dedicated clusters (the arithmetic is ours; the scale of the data and parameters is as reported in Chapter 32).

The first tells you that a single researcher can train many small models and run *systematic ablations*; the third that *replicating* a frontier model is out of reach, and that your experiment should *use* its published weights instead (probing, zero-shot scoring, fine-tuning), which is a legitimate research design.

**Expertise and access.** Who has the missing skill (a wet lab, a clinical collaborator, a statistician)? Which approvals are needed (human subjects, biosecurity review for dual-use tasks: Chapter 59)?

**The fast path.** Write the cheapest experiment that gives *any* signal: a simulation, a re-analysis of public data, a toy model. If the idea survives, escalate. A project with no fast path (a one-year experiment with a single yes/no readout) is a *high-variance bet* and should be treated as one (§56.7).

---

## 56.5 Decisiveness: expected information gain

Suppose there are competing hypotheses $H_1,\dots,H_K$ with prior probabilities $\pi_k$ and a candidate experiment with outcomes $o$ whose likelihood under hypothesis $k$ is $P(o\mid H_k)$. The **expected information gain** (EIG) of the experiment is the mutual information between hypothesis and outcome,
$$
\mathrm{EIG}=I(H;O)=H(\pi)-\mathbb E_o\big[H(\pi(\cdot\mid o))\big],
$$
the expected reduction in entropy of the posterior (Chapter 5; Lindley, 1956). It is zero if all hypotheses predict the same outcome distribution, and it is bounded by $H(\pi)$ (the experiment can at best settle the question). *EIG per unit cost* ranks experiments for a single step; sequential designs choose greedily or by lookahead (Chapter 46).

### 56.5.1 A worked computation

Chapter 55's attack on "why do sequence-to-function models ignore distal enhancers?" (Chapter 31) leaves three hypotheses: **H1 (biology)**: distal effects are weak or context-specific; **H2 (data)**: training data do not identify them; **H3 (architecture)**: the model does not use distal context. Prior: $(0.30,0.40,0.30)$, entropy 1.571 bits (maximum $\log_23=1.585$: the prior is nearly uninformative). Five experiments, with *elicited* likelihoods $P(\text{outcome}=1\mid H_k)$ and relative costs (illustrative numbers, not measurements; `code/ch56_information_gain.py`):

| Experiment (outcome 1 means...) | $P(1\mid H_1,H_2,H_3)$ | Cost | EIG (bits) | EIG per unit cost |
|---|---|---|---|---|
| **Mask distal context at inference** (predictions change) | 0.20, 0.55, 0.05 | 1 | 0.175 | **0.1755** |
| Reporter assay of enhancer–promoter pairs (strong effects in reporter) | 0.20, 0.85, 0.85 | 25 | 0.286 | 0.0114 |
| Retrain with 10× single-edit data (distal edit effects recovered) | 0.10, 0.80, 0.15 | 40 | 0.356 | 0.0089 |
| Shuffle enhancer sequence in silico (response proportional to measured) | 0.10, 0.20, 0.10 | 2 | 0.014 | 0.0070 |
| **CRISPRi screen of 200 enhancers** (many strong distal effects) | 0.10, 0.85, 0.85 | 100 | **0.387** | 0.0039 |

The *most informative* experiment per se is the expensive CRISPRi screen (0.387 bits), but *per unit cost* the cheap inference-time test is 45 times better, and it is the best choice in 99% of 2,000 random perturbations of the elicited likelihoods (each shifted by $\mathcal N(0,0.1^2)$): the ranking is robust. A greedy sequential plan chooses (1) the masking test (cumulative 0.175 bits at cost 1), then (2) the reporter assay (0.456 bits, cost 26), then (3) the single-edit retraining (0.703 bits, cost 66). *All five together carry 0.915 bits of the 1.571 available*: even this entire program leaves substantial uncertainty, because the experiments share a blind spot (H2 and H3 predict similar outcomes for several of them). **To tell H2 from H3 a different experiment is needed**: for instance, fine-tuning a model with distal context versus one without on the same single-edit data.

!!! lens "Research lens: what EIG teaches"
    (1) *Cheap, partly informative experiments first*; (2) *an experiment whose outcome is predicted equally by all hypotheses has EIG 0*, whatever its prestige; (3) *design experiments to separate the hypotheses that remain after the cheap ones*; (4) *likelihoods are elicited, so test sensitivity* (the ranking above survived); and (5) *state the outcome-to-decision map in advance*, so that the result changes the plan.

### 56.5.2 Beyond a single number

The likelihoods in a real analysis are subjective; EIG is a *discipline* rather than an oracle. Its value is in forcing you to write what each hypothesis *predicts* for each experiment. When two hypotheses make identical predictions for every experiment you can afford, they are not (currently) scientifically distinguishable and the right response is to *invent* an experiment, not to run the nearest available one.

### 56.5.3 The computation, verbatim

```python
--8<-- "code/ch56_information_gain.py"
```

```text
prior over hypotheses: {'H1': 0.3, 'H2': 0.4, 'H3': 0.3}; prior entropy = 1.571 bits (maximum 1.585)

experiment                                                                           cost (rel.)   EIG (bits)   EIG per unit cost   P(outcome=1)
mask distal context at inference (1 = predictions change)                                 1.0        0.175         0.1755          0.30
reporter assay (MPRA) of enhancer-promoter pairs (1 = strong effects in reporter)        25.0        0.286         0.0114          0.66
retrain with 10x single-edit data (1 = distal edit effects recovered)                    40.0        0.356         0.0089          0.40
shuffle enhancer sequence in silico (1 = model response proportional to measured)         2.0        0.014         0.0070          0.14
CRISPRi screen of 200 enhancers (outcome 1 = many strong distal effects)                100.0        0.387         0.0039          0.62

sequential strategy: pick the best experiment per unit cost, update, repeat (expected entropy after each step; outcomes enumerated)
step 1: mask distal context at inference (1 = predictions change)              cumulative information 0.175 bits (marginal 0.175), cumulative cost 1
step 2: reporter assay (MPRA) of enhancer-promoter pairs (1 = strong effects i cumulative information 0.456 bits (marginal 0.281), cumulative cost 26
step 3: retrain with 10x single-edit data (1 = distal edit effects recovered)  cumulative information 0.703 bits (marginal 0.247), cumulative cost 66

cumulative information of all five experiments: 0.915 bits (upper bound = prior entropy 1.571 bits)

robustness: fraction of 2,000 draws (likelihoods perturbed by N(0, 0.1^2)) in which each experiment has the highest EIG per unit cost
   0.99  mask distal context at inference (1 = predictions change)
   0.01  shuffle enhancer sequence in silico (1 = model response proportional to measured)
   0.00  reporter assay (MPRA) of enhancer-promoter pairs (1 = strong effects in reporter)
   0.00  retrain with 10x single-edit data (1 = distal edit effects recovered)
   0.00  CRISPRi screen of 200 enhancers (outcome 1 = many strong distal effects)
```

---

## 56.6 Risk, kill criteria, and pre-mortems

**Probability and payoff.** Estimate $p$ = probability that the idea works as described (a result that clears the pre-registered bar), $V$ = value if it does, $c$ = cost to a decisive answer, and the value of the *consolation prize* if it fails (an informative null; a reusable dataset or tool). A rough expected value is $pV+(1-p)W-c$ with $W$ the consolation value; projects with low $p$ and high $V$ are valuable only if $W$ is not zero or if they are small in $c$. Informative nulls require **power**: specify the effect size you can exclude.

**Kill criteria.** Before starting, write: *the result that would make me stop*, *the result that would make me change direction*, and *the date or cost at which I reassess*. A common pattern: a *threshold on the baseline-relative improvement in the cheapest test* (e.g., "if the toy does not show the predicted sign flip with a gap of at least 20 percentage points, stop").

**The pre-mortem.** Imagine the project finished and failed; write the five most likely reasons. For AI-for-biology projects the common ones are:

1. *The benchmark was saturated or leaky* (Chapters 24, 28, 43).
2. *The baseline was weak or mis-specified* (Chapter 29).
3. *The effect was within the noise ceiling* (Chapters 1, 25).
4. *The key variable was not identified by the data* (Chapter 31).
5. *The biology did not transfer* across cell type, species, ancestry (Chapter 45).
6. *The compute or data arrived late.*
7. *A competitor published first.*

For each reason, specify a **cheap mitigation now** (a ceiling computation; the strongest baseline; a simulation of identifiability) and put it *before* the main investment.

**Registered reports and pre-specification.** Write the analysis plan (data, metrics, baselines, success thresholds) before looking at test results. It guards against the "garden of forking paths" and lets negative results carry weight.

---

## 56.7 Portfolio: explore and exploit

An individual or lab should hold a **portfolio** with explicit balance:

* **Exploit** (about half the effort): extensions of working lines where $p$ is high and $V$ moderate; they provide publications, tools, and credibility.
* **Explore** (about a third): ideas with substantial $V$ and $p$ perhaps 10–30%, *each with a fast path and a kill criterion*.
* **Infrastructure** (the remainder): baselines, evaluation harnesses, data cleaning (reusable ceilings, splits, and null models), which raise $p$ for everything else.

Two rules. *Ideas should be able to fail quickly:* a project whose first signal requires six months should have been preceded by a two-week toy. *Keep a graveyard:* record each killed idea with its reason; patterns in the graveyard (leakage, saturated benchmarks, unidentified effects) tell you which parts of your process are weak.

---

## 56.8 A triage score

The following is a template, not a formula to obey. Score each idea 1–5 on each of the criteria, weight by your situation, and discuss the *disagreements* more than the sum.

| Criterion | 1 (low) | 5 (high) |
|---|---|---|
| Novelty after prior-art search | Equivalent to known work | New structure or a new test of an old claim |
| Meaning: decision changed | None | A named, high-value decision |
| Claim-ladder reach | C0–C1 | C3–C4 with prospective test |
| Feasibility | No data or access | All resources in hand |
| Time to first signal | Over 6 months | Under 2 weeks |
| Decisiveness (EIG per cost) | Outcomes predicted by all hypotheses | Separates the top hypotheses |
| Downside value (informative null) | Worthless if it fails | Reusable data/tool/baseline |
| Risk (ethics, biosecurity, reproducibility) | Serious unmitigated risk | None (see Chapter 59) |

---

## 56.9 Worked research examples

!!! example "Worked Research Example 56.1: Triaging the Attack Sheet of Chapter 55"
    **Situation.** Example 55.1 produced ten ideas for improving cross-individual variant-effect prediction. You have two researchers and three months.

    **Reasoning.**

    1. *Novelty.* Search the three names (allele-specific expression benchmark of sequence models; haplotype-aware deep scoring; personal-genome fine-tuning). Haplotype-aware personalized models and fine-tuning on personal-genome expression have been explored; the A5 stratified re-analysis of published model scores by LD and fine-mapping PIP is, to the best of the search, less explored [[P]]. (State the search log.)
    2. *Meaning.* The decision is *which scores to trust for prioritizing noncoding variants in clinical interpretation*. A stratified evaluation that identifies where models are reliable (e.g., near TSS, high PIP) is directly decision-relevant; a new architecture is not unless it beats the stratified baselines.
    3. *Feasibility.* A5 re-analysis: public eQTL/PIP resources and model weights: two to three weeks. A8 (ASE target): public ASE data exist for several tissues, one to two months. A4 (fine-tune on MPRA): needs MPRA data (public) and compute (days). A6 (megabase context): depends on model access. A10 (CRISPRi): requires collaborators.
    4. *Decisiveness.* Use EIG as in §56.5: the A5 stratification separates "model errs because of LD tags" from "model errs because of missing distal context", both of which are live hypotheses, at near-zero cost.
    5. *Portfolio.* *Exploit*: A5 re-analysis (weeks; high $p$). *Explore*: A4×A3 (single-edit fine-tuning with a within-gene loss; kill if sign accuracy gains less than 10 points on held-out fine-mapped eQTLs). *Infrastructure*: a reusable evaluation harness with ceilings and baselines. *Deferred*: A10 (needs a wet-lab partner).

    **Expert analysis.** The triage selected the *least glamorous* idea first because it had the highest decisiveness per cost and provided infrastructure for the others. This pattern is typical: **evaluation and re-analysis ideas front-load information.**

!!! example "Worked Research Example 56.2: \"Train a 7B single-cell foundation model on 100 million cells to predict perturbations\""
    **Situation.** A group proposes pretraining a 7-billion-parameter transformer on 100 million single-cell profiles and fine-tuning on perturbation datasets to build a virtual cell.

    **Evaluate it with the four tests.**

    1. **Novel?** Large single-cell models (scGPT, Geneformer, State, UCE and others; Chapters 38–39) exist; a prior-art search will find them under "virtual cell", "foundation model", "transcriptomic LLM". What is new must be stated as a *hypothesis*: for example, that scaling parameters from $10^8$ to $7\times10^9$ changes perturbation-prediction accuracy at fixed data, or that a specific objective (A3) closes a gap.
    2. **Meaningful?** Which decision does a better perturbation predictor improve (Chapter 25's reliability analysis: how many perturbations have real effects; Chapter 39's ceilings)? If most benchmark variance is noise and an additive or mean baseline is within the ceiling, no model can show a meaningful gain on that benchmark.
    3. **Feasible?** $C\approx6ND$: with 100 million cells at roughly 2,000 expressed genes as tokens, $D\approx2\times10^{11}$ tokens per epoch, and $N=7\times10^9$ gives $C\approx8\times10^{21}$ FLOPs, on the order of 200–250 GPU-days per epoch at the effective throughput above: feasible for a funded group, but a single run, so ablations must use small models. *Fast path*: scaling curves with 10M–500M-parameter models on 1–10 million cells.
    4. **Decisive?** Hypotheses: (a) *scale helps*: loss and downstream metrics follow a power law with a non-trivial exponent; (b) *scale does not help*, because the bottleneck is the measurement (reliability ceiling), the objective (reconstruction rather than intervention), or the data (observational atlases do not identify causal effects: Chapter 31's mechanism); (c) *helps zero-shot but not after fine-tuning*. The decisive experiment is the *scaling study on a ceiling-normalized, family-split benchmark*, with the additive/mean baseline and a ridge model on prior-knowledge features as controls.

    **Verdict.** Not the proposal as stated: the *large* run is an expensive way to test a hypothesis that a series of small runs can already test. Recommended: run the scaling curves and the ceiling analysis first; spend the large budget only if the curve is still steep at the largest small-model size, and the metric has headroom.

    **Expert analysis.** The proposal embeds an A6 attack without an identified limiting resource (Chapter 55). The four tests convert it into a staged plan with a kill criterion: *if doubling parameters yields less than the replicate noise of the metric, stop scaling and switch to A3 or A4.*

---

## 56.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: pre-mortem, kill criteria, and a decision log"
    **Setting.** You have a ranked list from the Attack Sheet.

    1. **Prior-art log.** For each of your top three ideas, record three names, five searches, the closest paper, and its rung on the ladder.
    2. **Decision statement.** One sentence: "If [result], I will [action]; if [opposite result], I will [other action]." If the two actions are the same, the experiment is not decisive.
    3. **Likelihood table.** For each of your two or three hypotheses write $P(\text{outcome}\mid H)$ for each candidate experiment; compute the EIG per cost (adapt `code/ch56_information_gain.py`); perturb the likelihoods and check the ranking.
    4. **Pre-mortem.** Five reasons for failure; one mitigation each, scheduled before the main work.
    5. **Kill criteria and checkpoints.** Dates, thresholds, and who decides.
    6. **Graveyard.** After each project, record the reason it ended and which of the seven failure reasons it was.

    **What it teaches.** Triage is itself a skill that improves with a record. The graveyard is data about your own process.

    **An open question to carry forward.** The EIG framework depends on elicited likelihoods that are themselves uncertain. How would you extend it so that the *value of reducing uncertainty in the likelihoods* (e.g., by a cheap pilot) is part of the calculation, and when does a hierarchical Bayesian treatment of "how reliable are our predictions about experiments" pay for its complexity? Test it on a set of past projects whose outcomes you know.

---

## 56.11 Connections

- **Backward:** the Four Gaps, Claim Ladder, and Expert Chain (Chapter 1); entropy and mutual information (Chapter 5); noise ceilings and baselines (Chapters 1, 29); scaling and compute (Chapter 17); counterfactual identifiability (Chapter 31); idea generation (Chapter 55).
- **Forward:** how to read papers critically (Chapter 57); reasoning without known answers (Chapter 58); from idea to publication, ethics, and biosecurity (Chapter 59); experimental design and active learning (Chapter 46).

!!! takeaways "Key takeaways"
    1. An idea passes four tests: **novel** (a three-name prior-art search), **meaningful** (a named decision; a Claim-Ladder rung), **feasible** (data, compute, expertise, a fast path), **decisive** (it can change your beliefs whichever way it comes out).
    2. **Novelty is instrumental**; a rigorous replication or an honest negative with power can be worth more than a novel method.
    3. **Value of information** is zero when no result would change the decision; write the decision first.
    4. Training compute is $\approx6ND$: 13 GPU-hours for a 100M model on $10^{10}$ tokens, about 360 GPU-days for 7B on $3\times10^{11}$; use published weights for frontier-scale questions.
    5. **EIG per unit cost** ranks experiments; in the worked example the cheapest inference-time test beat the most informative experiment (CRISPRi screen) by a factor of 45 per unit cost, in 99% of perturbed-likelihood draws, while all five experiments together carried only 0.915 of 1.571 bits.
    6. Write **kill criteria** and run a **pre-mortem** before the investment; schedule cheap mitigations first.
    7. Keep a **portfolio** with explore, exploit, and infrastructure shares, and a **graveyard** of killed ideas.
    8. Evaluation and re-analysis ideas front-load information and tend to rank first in triage.

---

## Further reading

- Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Ann. Math. Stat.* 27, 986–1005. Howard, R. A. (1966). Information value theory. *IEEE Trans. Syst. Sci. Cybern.* 2, 22–26. Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: a review. *Stat. Sci.* 10, 273–304.
- Platt, J. R. (1964). Strong inference. *Science* 146, 347–353. Klein, G. (2007). Performing a project premortem. *Harvard Business Review* 85, 18–19. Nosek, B. A. et al. (2018). The preregistration revolution. *PNAS* 115, 2600–2606. Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Med.* 2, e124.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*. Hoffmann, J. et al. (2022). Training compute-optimal large language models. *NeurIPS*.
