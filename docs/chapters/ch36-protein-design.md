# Chapter 36. Protein and Biomolecular Design

!!! abstract "Chapter at a glance"
    **Motivation.** Prediction asks what a sequence does; design asks which sequence does a given thing. Generative models of structure (diffusion over backbones), of sequence (inverse folding and language models), and of complexes (co-folding networks) have made design of binders, enzymes, and antibodies a routine computational activity, with reported experimental success rates between a few percent and, for some targets, tens of percent per tested design. The scientific problem is that *every* design pipeline optimizes a surrogate (a structure predictor's confidence, a language-model likelihood, a learned fitness model) in a region where the surrogate was not trained; its predictions are then least reliable exactly where the optimizer pushes. This chapter formulates design as a KL-regularized optimization (the same mathematics as preference tuning, Chapter 17), derives the optimum and the optimizer's curse, and tests both in two controlled experiments: one in which the training data cover the region (optimization works, but predicted fitness overstates true fitness, and diversity collapses), and one in which the data are local, like a deep mutational scan, and the designer wants to go far (a trust-region phase transition, where a surrogate-optimized design can be *worse* than the starting point). It then dissects the main design systems and their evidence, states what a credible design claim must report, and covers biosecurity.
    **Prerequisites.** Chapters 14, 15, 17, 23, 34, 35, 43, 46.
    **You will be able to:** (1) write the design loop and identify where each model sits in it; (2) derive $p^\star\propto p_0\,e^{f/\beta}$ and relate $\beta$ to a trust region; (3) predict the share of a surrogate's promised gain that is real from its accuracy; (4) read a design paper's success rate with its denominator, target difficulty, novelty, and diversity; (5) design a test-and-learn campaign with arms that isolate the generator from the filter; (6) apply a biosecurity framework to a design pipeline.

---

## 36.0 Design as an inverse problem

**Forward problem.** Given a sequence $x$, predict a property $y=f(x)$: stability, binding affinity, enzymatic activity, expression, immunogenicity. **Inverse problem.** Given a desired property (or a specification: bind this epitope, catalyze this reaction, fold into this shape), find $x$ with $f(x)$ high. The inverse problem is harder for four reasons:

1. **The space is enormous and the solutions are sparse.** For a 100-residue protein there are $20^{100}$ sequences; functional ones are a vanishing fraction of them, and *designable* ones (those that fold and function) are a structured subset.
2. **The oracle is expensive.** Each experimental evaluation costs from tens of dollars (a binding assay) to thousands (a crystal structure); a campaign tests tens to thousands of designs, so computational filters decide what is tested.
3. **The filter is a model.** Designs are selected with surrogates (structure-prediction confidence, language-model likelihood, energy functions, learned fitness models). An optimizer will find where the surrogate is wrong.
4. **There are many objectives.** Binding, specificity, stability, expression in the host, aggregation, immunogenicity, manufacturability: designs are multi-objective and the objectives conflict.

**The design loop.**

```text
specify target t (structure of an epitope, a reaction, a fold, a functional motif)
repeat (a campaign has several rounds):
    generate   x ~ q_phi(x | t)                    # backbone diffusion, inverse folding, sequence LM, hallucination
    filter     keep x with  s(x, t) > threshold    # structure-predictor confidence (pLDDT, ipAE/ipTM), energy, language-model score
    test       measure y(x) experimentally         # expression, binding (SPR/BLI), structure, activity
    learn      update q_phi, s, or the selection rule using (x, y)    # supervised heads, active learning (Chapter 46), reinforcement
until budget exhausted
report         successes (denominator!), novelty, diversity, structures
```

---

## 36.1 The generators, and the filters

**Backbone generators.** *RFdiffusion* (Watson et al., *Nature* 2023) fine-tunes the RoseTTAFold structure network as a denoising diffusion model over residue frames (Chapter 15), conditioned on a target, motif, or symmetry; its *all-atom* successors extend conditioning to small molecules and nucleic acids (RFdiffusion3 was open-sourced in December 2025). **Sequence design for a backbone.** *ProteinMPNN* (Dauparas et al., *Science* 2022) is a message-passing network (Chapter 16) that predicts a sequence from a backbone; on native backbones it recovers about 52% of native residues against about 33% for Rosetta's protocol, and it is the default sequence designer of most pipelines; *LigandMPNN* adds the context of ligands and nucleic acids. **Hallucination and relaxed-sequence optimization.** Backpropagating through a structure predictor to find a sequence whose predicted structure has a target property (trRosetta hallucination; *BindCraft*, Pacesa et al., *Nature* 2025, which optimizes binders against an AlphaFold2-multimer-based loss and then redesigns the sequence) uses the *predictor itself* as the filter and the generator. **Sequence language models.** ProGen, ESM3 and autoregressive models (Chapter 34) sample sequences conditioned on tags, families, or function annotations. **All-atom co-folding generators.** Chai-2 (Chai Discovery, 2025) and similar proprietary models generate antibodies and binders conditioned on an antigen. **Genome-scale.** Evo-based phage design (Chapter 32).

**Filters.** (i) *Self-consistency*: fold the designed sequence with an independent predictor; require the predicted structure to match the designed backbone (scRMSD, scTM). (ii) *Interface confidence*: ipAE, ipTM, interface pLDDT from co-folding networks; these correlate with experimental binding, with an AUROC that depends on the target class and is not close to 1. (iii) *Physics*: Rosetta energies, molecular-dynamics stability. (iv) *Developability*: predicted aggregation propensity, charge, solubility, immunogenicity, expression.

!!! lens "Research lens: assumptions and what the filters ignore"
    Structure-predictor confidence is trained on *natural* proteins and complexes. A designed sequence is, by construction, selected to make the predictor confident. This is the setting of §34.4 and §35.5: an optimizer can find sequences that the predictor *folds* confidently without the sequence in fact folding or binding (adversarial examples against the predictor), the failure mode of hallucination. What the filters ignore: the host (expression, post-translational modification, proteolysis), kinetics, off-target binding, conformational dynamics, and the assay.

---

## 36.2 The mathematics of optimizing against a surrogate

**KL-regularized optimization.** Let $p_0(x)$ be a reference distribution of plausible sequences (a language model, a family model, or the empirical distribution of homologs), $\hat f$ a surrogate of the objective, and $\beta>0$ a temperature. Choose the design distribution
$$
p^\star=\arg\max_{p}\ \Big\{\mathbb E_{x\sim p}[\hat f(x)]-\beta\,\mathrm{KL}(p\,\|\,p_0)\Big\}.
$$
!!! math "Derivation: the optimum"
    Add a Lagrange multiplier $\lambda$ for $\sum_x p(x)=1$. Then $\partial/\partial p(x)$ of $\sum_xp\hat f-\beta\sum_xp\log\frac{p}{p_0}-\lambda(\sum_xp-1)$ gives $\hat f(x)-\beta\log\frac{p(x)}{p_0(x)}-\beta-\lambda=0$, hence
    $$
    p^\star(x)=\frac{p_0(x)\,e^{\hat f(x)/\beta}}{Z},\qquad Z=\sum_xp_0(x)e^{\hat f(x)/\beta}.
    $$
    As $\beta\to\infty$, $p^\star\to p_0$ (no optimization); as $\beta\to0$, $p^\star$ concentrates on $\arg\max\hat f$ among sequences with $p_0>0$. The value at the optimum is the free energy $\beta\log Z$. $\square$

This is the optimum of RLHF-style tuning (Chapter 17) and of the "design then filter" pipeline in the limit where filtering is rejection sampling with acceptance $\propto e^{\hat f/\beta}$. It is also the *Boltzmann distribution* of statistical physics with energy $-\hat f$ and temperature $\beta$ (Chapter 22). $\beta$ is a *trust region*: the KL divergence from $p_0$ that the designer is willing to spend.

**The optimizer's curse.** Suppose the surrogate has errors, $\hat f=f+\varepsilon$ with $\varepsilon\sim\mathcal N(0,\sigma_\varepsilon^2)$ independent of $f\sim\mathcal N(\mu,\sigma_f^2)$ across candidates. Selecting the top candidate of $N$ by $\hat f$, the expected gain in $\hat f$ is $\sigma_{\hat f}\,\mathbb E[Z_{(N)}]$ with $\sigma_{\hat f}^2=\sigma_f^2+\sigma_\varepsilon^2$, of which the share that is *real* (in $f$) is
$$
\frac{\Delta f}{\Delta \hat f}=\frac{\sigma_f^2}{\sigma_f^2+\sigma_\varepsilon^2}=R^2_{\hat f\leftrightarrow f}\quad(\text{the squared correlation}),
$$
and the remainder $1-R^2$ of the promised gain is selection on error. (This is the same arithmetic as the winner's curse of Chapter 43 and the judge drift of Chapter 54.) The consequences: (i) *predicted improvement overstates true improvement by $1/R^2$* in this model; (ii) the overstatement grows with selection intensity ($\beta\downarrow$, $N\uparrow$); (iii) errors are not independent of $x$ in practice: they grow with the *distance from the training data*, so the optimizer is drawn to where they are largest.

!!! rhyme "Structural rhyme: design against a surrogate ↔ reward overoptimization in RLHF ↔ judge drift (Chapter 54)"
    In each, a proxy is optimized past the region where it tracks the truth. Gao et al. (2023) showed for reward models that the true reward first rises and then *falls* as the policy moves farther (in KL) from the reference, with the peak at a KL that grows with reward-model size and data. The design experiment below finds the same shape for sequences.

---

## 36.3 Experiment 1: surrogate design when the data cover the region

A hidden fitness landscape on 16 positions and 4 letters has additive terms and pairwise epistasis (30% of position pairs coupled) with a saturating readout. "Natural-like" training sequences are Gibbs samples from a distribution selected on that landscape (so training sequences are good but not optimal). Surrogates are fit on $N$ training sequences and used to design by sampling from $p^\star\propto p_0\,e^{\hat f/\beta}$ with $p_0$ the site-independent frequency model of the training data. Surrogates: pairwise ridge regression (the right model class), an additive-only ridge (misspecified), and a two-hidden-layer MLP (flexible). For each of 8 landscapes, 60 designs are drawn per setting; the table gives the surrogate's prediction, the *true* fitness (latent units; the training sequences average 13.2), the gap, the Hamming distance to the nearest training sequence, and the fraction of unique designs.

```python
--8<-- "code/ch36_design_surrogate.py"
```

```text
== Designs sampled from p*(x) ~ prior(x) exp(f_hat(x)/beta): 16 positions x 4 letters; 8 hidden landscapes x 60 designs per setting ==
'latent fitness' is the landscape's additive-plus-pairwise score (natural-like sequences are Gibbs samples from a prior selected on it)
natural-like training sequences have mean latent fitness 13.2 (400 sequences)

surrogate                          training sequences   held-out R^2   design setting                                      surrogate's prediction   TRUE fitness   gap (pred - true)   distance to nearest training seq.   fraction unique
pairwise ridge                           60              0.36        prior samples (beta = inf)                              12.86                   6.26           6.61                 6.4                    1.00
pairwise ridge                           60              0.36        beta = 1.0                                              18.31                  13.25           5.06                 4.9                    1.00
pairwise ridge                           60              0.36        beta = 0.25                                             22.69                  19.53           3.16                 3.3                    0.51
pairwise ridge                           60              0.36        beta = 0.1 (near argmax)                                23.16                  20.24           2.92                 3.0                    0.17
pairwise ridge                           60              0.36        beta = 0.1, ensemble lower bound (mean - 1 sd)          22.27                  18.55           3.72                 3.6                    0.26
pairwise ridge                          150              0.52        prior samples (beta = inf)                              12.44                   6.70           5.75                 5.8                    1.00
pairwise ridge                          150              0.52        beta = 1.0                                              18.92                  15.49           3.42                 4.1                    0.99
pairwise ridge                          150              0.52        beta = 0.25                                             22.86                  20.87           1.99                 2.6                    0.50
pairwise ridge                          150              0.52        beta = 0.1 (near argmax)                                23.23                  21.11           2.12                 2.5                    0.19
pairwise ridge                          150              0.52        beta = 0.1, ensemble lower bound (mean - 1 sd)          22.58                  21.38           1.20                 2.2                    0.14
pairwise ridge                          400              0.70        prior samples (beta = inf)                              10.74                   6.77           3.97                 5.3                    1.00
pairwise ridge                          400              0.70        beta = 1.0                                              19.54                  18.01           1.52                 3.2                    0.99
pairwise ridge                          400              0.70        beta = 0.25                                             23.44                  22.79           0.65                 2.3                    0.51
pairwise ridge                          400              0.70        beta = 0.1 (near argmax)                                23.55                  22.75           0.80                 2.4                    0.26
pairwise ridge                          400              0.70        beta = 0.1, ensemble lower bound (mean - 1 sd)          23.46                  23.01           0.45                 2.1                    0.21
additive ridge (misspecified)           400              0.51        prior samples (beta = inf)                              12.81                   6.77           6.03                 5.3                    1.00
additive ridge (misspecified)           400              0.51        beta = 1.0                                              18.00                  13.14           4.85                 4.1                    1.00
additive ridge (misspecified)           400              0.51        beta = 0.25                                             20.93                  18.07           2.86                 3.1                    0.86
additive ridge (misspecified)           400              0.51        beta = 0.1 (near argmax)                                21.48                  18.39           3.09                 3.0                    0.42
MLP, 2 hidden layers (flexible)         150              0.50        prior samples (beta = inf)                              12.18                   6.61           5.56                 5.9                    1.00
MLP, 2 hidden layers (flexible)         150              0.50        beta = 1.0                                              18.98                  14.29           4.68                 4.4                    1.00
MLP, 2 hidden layers (flexible)         150              0.50        beta = 0.25                                             22.23                  19.03           3.19                 3.0                    0.70
MLP, 2 hidden layers (flexible)         150              0.50        beta = 0.1 (near argmax)                                22.77                  19.14           3.63                 2.8                    0.30
MLP, 2 hidden layers (flexible)         400              0.66        prior samples (beta = inf)                              11.16                   6.76           4.40                 5.3                    1.00
MLP, 2 hidden layers (flexible)         400              0.66        beta = 1.0                                              19.12                  17.06           2.06                 3.4                    0.99
MLP, 2 hidden layers (flexible)         400              0.66        beta = 0.25                                             22.66                  22.09           0.57                 2.2                    0.59
MLP, 2 hidden layers (flexible)         400              0.66        beta = 0.1 (near argmax)                                23.01                  22.45           0.56                 2.0                    0.26
```

**Reading Experiment 1.**

1. **Optimization works when the data cover the region.** With 400 training sequences (surrogate $R^2=0.70$ on held-out natural-like sequences), the true fitness of designs rises from 6.8 (samples from the site-independent prior alone) to 18.0 ($\beta=1$) and 22.8 ($\beta=0.25$): *above* the natural training sequences (13.2). The optimizer finds better sequences than the data contain, which is the promise of design.
2. **The prior matters.** Samples from the site-independent prior alone have true fitness 6.8, about half that of the natural sequences that produced it: *a profile model that ignores epistasis is a poor generator*. A good $p_0$ (a family-aware model, a protein language model) supplies the epistatic structure that the surrogate alone must otherwise learn.
3. **Predicted gain overstates the true gain, by an amount set by surrogate quality.** At $\beta=0.1$ the prediction minus truth is 0.8 for the pairwise surrogate with 400 training sequences, 2.1 with 150, and 2.9 with 60 (held-out $R^2$ 0.70, 0.52, 0.36): the overstatement shrinks as the surrogate improves, as the arithmetic of §36.2 predicts.
4. **Model class matters more than data in this setting.** An additive-only surrogate trained on the same 400 sequences ($R^2=0.51$ instead of 0.70) produces designs with true fitness 18.4 instead of 22.8 at $\beta=0.1$ and a gap of 3.1: *the misspecified surrogate wastes about four fitness units, nearly half of the gain over natural sequences (22.8 against 13.2)*. The flexible MLP at 150 sequences has a gap of 3.6 (true 19.1), at 400 sequences 0.6 (true 22.5).
5. **Diversity collapses as optimization power rises.** The fraction of unique designs falls from 1.00 (prior, $\beta=1$) to 0.51 ($\beta=0.25$) to 0.17–0.26 ($\beta=0.1$): out of 60 samples at near-argmax, only 10–16 are distinct. A design campaign that tests the top $k$ by the surrogate will test near-duplicates; selection must include a *diversity* criterion (Chapter 46).
6. **Conservatism helps when it matters.** The ensemble lower bound (mean minus one standard deviation over bootstrap surrogates) narrows the gap (1.2 against 2.1 at $N=150$; 0.45 against 0.80 at $N=400$) and raises the true fitness slightly where the surrogate is decent (21.4 against 21.1; 23.0 against 22.8), but at $N=60$ it *lowers* the true fitness (18.6 against 20.2): with a poor surrogate, uncertainty estimates from a bootstrap of a poor model are themselves poor.

---

## 36.4 Experiment 2: designing far from local data

Engineering campaigns usually have *local* data: a deep mutational scan around one wild type, or a few hundred variants from a round of directed evolution. Here the training set is 300 variants within 1–3 mutations of a reference sequence (the best of 700 natural-like sequences), and the designer samples from $p^\star$ with $p_0$ the site-independent model fit to those variants, which is sharply peaked on the reference: the trust region of width $\beta$ is around the reference.

```python
--8<-- "code/ch36b_local_design.py"
```

```text
== Training data: 300 variants within 1-3 mutations of a reference (the best of 700 natural-like sequences); 8 landscapes; 60 designs per setting ==
reference fitness (latent) 23.7; mean fitness of the training variants 17.4
surrogate pairwise ridge : R^2 on variants like the training data 0.94;  R^2 on variants 8 mutations from the reference 0.40
surrogate MLP            : R^2 on variants like the training data 0.94;  R^2 on variants 8 mutations from the reference 0.34

surrogate        design setting                                    surrogate's prediction   TRUE fitness   gap (pred - true)   mutations from reference   fraction unique   fraction better than the reference
pairwise ridge   beta = 2.0                                               22.43               22.56          -0.13                   0.7                     0.41               0.05
pairwise ridge   beta = 2.0, ensemble lower bound                         22.22               22.30          -0.09                   0.8                     0.45               0.07
pairwise ridge   beta = 1.0                                               23.17               23.30          -0.13                   0.5                     0.27               0.11
pairwise ridge   beta = 1.0, ensemble lower bound                         23.22               23.34          -0.12                   0.5                     0.25               0.12
pairwise ridge   beta = 0.5                                               23.63               23.77          -0.14                   0.4                     0.13               0.22
pairwise ridge   beta = 0.5, ensemble lower bound                         23.62               23.76          -0.14                   0.4                     0.14               0.22
pairwise ridge   beta = 0.25                                              23.90               24.04          -0.14                   0.5                     0.09               0.43
pairwise ridge   beta = 0.25, ensemble lower bound                        23.72               23.68           0.04                   0.8                     0.12               0.43
pairwise ridge   beta = 0.1 (near argmax)                                 22.03               18.25           3.78                   4.9                     0.32               0.49
pairwise ridge   beta = 0.1 (near argmax), ensemble lower bound           21.24               15.55           5.69                   6.2                     0.40               0.51
MLP              beta = 2.0                                               22.47               22.50          -0.04                   0.7                     0.43               0.06
MLP              beta = 1.0                                               23.23               23.29          -0.06                   0.5                     0.26               0.11
MLP              beta = 0.5                                               23.63               23.74          -0.11                   0.4                     0.16               0.21
MLP              beta = 0.25                                              23.88               24.00          -0.13                   0.4                     0.08               0.37
MLP              beta = 0.1 (near argmax)                                 24.01               24.13          -0.12                   0.7                     0.06               0.59
```

**Reading Experiment 2.**

1. **The surrogate is accurate near the data and not far from them**: $R^2=0.94$ on variants like the training data, 0.40 (pairwise) and 0.34 (MLP) on variants eight mutations from the reference.
2. **Inside the trust region, designs are safe and unambitious.** For $\beta\ge0.25$ designs stay within 0.4–0.8 mutations of the reference; predictions match truth (gap about $-0.1$), and the true fitness is 22.3–24.0 against the reference's 23.7: *the designs are the reference, with a few variants that improve it* (the fraction better than the reference rises from 5% at $\beta=2$ to 43% at $\beta=0.25$).
3. **A phase transition at small $\beta$.** At $\beta=0.1$ the pairwise-ridge optimizer leaves the trust region (4.9 mutations from the reference): the surrogate predicts 22.0 and the true fitness is **18.3**, a gap of 3.8 and a true fitness 5.4 units *below the reference it started from*. The ensemble lower bound does not rescue it (gap 5.7, true fitness 15.5, 6.2 mutations): the ensemble members, trained on the same local data, agree with each other while being wrong in the same direction, a general feature of bootstrap uncertainty on a *shared blind spot*. The MLP never left the trust region at this $\beta$ (0.7 mutations).
4. **So the design problem is a trust-region problem.** The useful $\beta$ is the *smallest* that keeps the predicted–true gap small, and it can be found only by *measuring* gap against distance with a round of experiments at graded distances (Chapter 46), not by trusting the surrogate's own confidence. This is the practical content of "optimize the acquisition function with a trust region" in Bayesian optimization, and of the KL penalty in RLHF.

!!! lens "Research lens: assumptions of both experiments"
    A landscape with pairwise epistasis (no higher-order terms, no assay noise), small sequences, a prior that is a site-independent model, an optimizer that is MCMC from random starts (so it finds the optimum of $p^\star$ only approximately), and a surrogate trained by ridge regression or an MLP. Real design uses structure networks as filters, which are less smooth than ridge regression and can be exploited more violently; real data have assay noise (Chapter 1's ceiling) that reduces the achievable $R^2$. The qualitative results (overstatement $\propto 1/R^2$, model-class sensitivity, diversity collapse, a trust-region phase transition) are the transferable part.

---

## 36.5 Systems and evidence

!!! paper "Paper dissection: BindCraft (Pacesa et al., *Nature* 646, 2025)"
    **Problem.** One-shot design of protein binders to a target of known structure, with experimental success high enough that no high-throughput screening is needed.
    **Insight.** Use AlphaFold2-multimer *as the generator*: backpropagate through it to optimize a binder sequence for interface confidence, then redesign the sequence with ProteinMPNN, then filter by independent predictors and physical checks.
    **Evaluation.** Binders designed against cell-surface receptors, allergens, de novo designed proteins, and a multi-domain nuclease (Cas9), tested by binding assays, with functional readouts in some cases (reduced IgE binding to a birch allergen in patient-derived samples, modulation of Cas9 editing, reduced cytotoxicity of a bacterial enterotoxin).
    **Result (as reported).** Experimental success rates between 10% and 100% across targets, nanomolar affinities, no screening of large libraries.
    **Why it worked.** Strong structure prediction for complexes; a good filter stack; targets with accessible hydrophobic surfaces, where binder design is easier.
    **Limitations.** Success varies by target by an order of magnitude (a mean hides the range; the denominator per target matters); the targets were selected by the authors; the filter and the generator share a predictor, which invites the exploitation of §36.4; binding is not function.
    **What followed.** Open-source adoption, extensions to antibodies and small-molecule binders.

| System | What it designs | Reported evidence (denominators in the sources) | What to ask |
|---|---|---|---|
| RFdiffusion (2023) and successors | Backbones and binders, symmetric assemblies, motif scaffolds | Experimental hits across many targets; structures confirmed by crystallography and cryo-EM for selected designs | Hit rate per target and per design; novelty relative to PDB |
| ProteinMPNN (2022) / LigandMPNN (2025) | Sequences for a backbone, with ligand context | Native-sequence recovery (52% vs 33% for Rosetta); many experimentally validated designs | Recovery is not function; designability of the backbone |
| BindCraft (2025) | Protein binders by hallucination | 10–100% success across targets; functional demonstrations | Targets selected; target-wise rates |
| AlphaProteo (2024) | Binders to seven targets, including VEGF-A | Success rates from 9% to 88% across targets and 3- to 300-fold better affinity than the best earlier methods, as reported by the developers | Developer-reported; benchmarks chosen by the developers [[P]] |
| Chai-2 (2025) | Antibodies and nanobodies, zero-shot | About 16% of tested designs bound, at most 20 designs per target, 52 targets without an antibody or nanobody binder in the PDB; at least one binder for 50% of targets in one round; 68% success for miniproteins (preprint) | Preprint; the hit-rate definition; affinity and developability of the hits [[P]] |
| RFdiffusion antibodies (Bennett et al., *Nature* 2025) | De novo VHHs and scFvs with specified epitopes | Cryo-EM confirmation of designed VHH poses against influenza hemagglutinin and *C. difficile* toxin B; a high-resolution structure of a designed CDR loop conformation | Initial designs were reported at modest affinity and improved by maturation; commentaries questioning parts of the evaluation have appeared, and replication is expected [[S]] for the method, [[P]] for generality |
| ESM3 (2025), ProGen (2023), Evo phage (2026) | Sequences and genomes from language models | A fluorescent protein at 58% identity to natural ones; functional lysozymes; 16 viable phage genomes from about 300 | The denominators (Chapters 32, 34) |

**What the evidence supports** [[S]]: for *binders to well-behaved targets*, computational design now yields experimentally validated binders at rates (tens of percent for the best targets) that would have been unimaginable in 2020; structures of several designed complexes agree with the design to within a few angstroms. [[P]]: antibody-like design with atomic accuracy across diverse epitopes. [[H]]: de novo enzymes with activities comparable to natural ones; function in cells and in vivo; designs that are developable (stable, non-aggregating, non-immunogenic) without further optimization. [[X]]: general-purpose design of arbitrary specified functions.

---

## 36.6 What a design claim must report

1. **The denominator**: number tested, number expressing, number folded, number binding/active, with the *definition* of success (affinity threshold, activity above a baseline).
2. **The selection**: how designs were chosen from the pool (top by the filter score? a random sample? every design?).
3. **Target difficulty**: how many targets were attempted, how many succeeded, which were excluded and why; per-target rates, not the mean.
4. **Novelty**: sequence identity and structural similarity to the closest natural protein *and* to the training set; for binders, the epitope.
5. **Diversity**: how many distinct sequences/folds among successes.
6. **Controls**: the baseline generator (random mutagenesis of a known binder; sampling from the prior $p_0$) tested under the same assay.
7. **Orthogonal validation**: affinity by two methods; experimental structure for at least some; specificity (off-target panel); function in a relevant assay.
8. **Failures**: expression failures and aggregation count as failures.

---

## 36.7 Biosecurity

Design tools can lower the barrier to making harmful agents. Three layers apply (Chapter 59): *data* (exclude hazardous sequences from training; Evo 2 excluded eukaryotic viral genomes), *access* (tiered release, screening of requests, logging), and *downstream screening* (synthesis providers screen orders against databases of hazardous sequences). Wittmann et al. (*Science*, 2025) showed that generative protein design tools can *reformulate* known toxins into sequence variants predicted to keep function but that evade the homology-based screening software then in use, and that the screening could be patched (function-aware and structure-aware screening) with a coordinated disclosure to providers [[S]]. The lesson is the one of §36.2 applied to defense: a screening rule is a surrogate, and a generative model is an optimizer against it; **screening must be evaluated adversarially** and updated as generators improve. Responsible-design commitments by protein-design researchers (2024) and frameworks for structured access are under development; this is an active area where policy is changing faster than any text.

---

## 36.8 Worked research examples

!!! example "Worked Research Example 36.1: A paper reports a '50% success rate' for designed binders"
    **Situation.** An abstract states that 50% of designed binders were validated experimentally against 12 targets.

    **Question.** What do you need to know, and what could 50% mean?

    **Reasoning (Expert Chain).**

    1. **What is the denominator?** "50% of designs tested", "50% of targets with at least one binder", or "50% of experiments"? Chai-2's headline "16% hit rate" and "50% of targets" are different quantities from the same study.
    2. **What was the selection?** If the designs tested were the top 0.1% of 100,000 by an in-silico filter, then "50%" is the precision *at an extreme cutoff*; a fair comparison uses the number of designs generated, or the compute and cost per hit.
    3. **Which targets?** If the 12 are chosen among targets with hydrophobic, convex epitopes (easy), the rate is not representative; ask for per-target rates and for failed targets.
    4. **What is a success?** Binding at 10 µM in a single-concentration assay (a weak criterion with many false positives) differs from a 10 nM dissociation constant by SPR with a counter-screen.
    5. **Controls.** Baseline: mutagenesis of an existing binder, or screening a library of random sequences of matched composition under the same assay; a design method's value is the *enrichment* over the baseline per unit cost.
    6. **Novelty.** If the binders resemble known complexes in the training data of the structure predictor (§35.5), the method may be retrieving, not designing.
    7. **Claim.** After these checks, a defensible statement might be: "for the 9 of 12 targets with hydrophobic epitopes, 20–80% of the 24 designs per target bound with a $K_D$ below 1 µM, against 2% for random mutagenesis of a known binder".

    **Expert analysis.** A single success percentage hides four choices (denominator, selection, targets, threshold); each is where a field-wide comparison fails. The decision-relevant quantity is *cost per validated binder at a stated affinity and novelty*.

!!! example "Worked Research Example 36.2: Designing an enzyme for a reaction nature has not catalyzed. No known answer"
    **Situation.** A group wants an enzyme that catalyzes a new-to-nature reaction (a C–H functionalization with a synthetic substrate). No natural enzyme is known; a crude catalytic motif (a metal site and a base) can be specified from quantum-chemical models of the transition state.

    **Question.** How would you set up the campaign, and what would count as progress?

    **Reasoning.**

    1. **Decompose the specification.** (a) Theozyme: the geometry of catalytic groups around the transition state (from DFT); (b) scaffolds that can hold the theozyme with a substrate-accessible pocket; (c) the sequence that folds into the scaffold.
    2. **Generators.** Backbone generation conditioned on the theozyme motif (RFdiffusion-family) → sequences (LigandMPNN) → filters: self-consistency, pocket geometry, ligand-pose retention in co-folding, predicted stability; and, for diversity, several generators.
    3. **Arms.** (A) the pipeline above, top by filter; (B) the same with a random selection among those passing; (C) existing enzymes redesigned for the reaction (the known baseline: directed evolution starting points); (D) negative controls (scrambled active-site residues). Same assay for all.
    4. **Test cascade with attrition.** Express (soluble fraction), fold (circular dichroism or thermal shift), bind substrate (an ITC or fluorescence assay), catalyze (product detection by LC-MS above background). Report attrition at each step with intervals: with 200 designs, a 1% final success is 2 designs, and its confidence interval is wide.
    5. **Learn.** Use the results to train a supervised model on assay outcomes (Chapter 46's active learning); designs that bind but do not turn over tell you about the theozyme; designs that do not fold tell you about the filter. Directed evolution on the best designs (typically needed to reach useful rates) provides the second stage.
    6. **Decision rules.** Pre-register that a design counts as active if the product exceeds the background by 5 standard deviations in three replicates and requires the catalytic residue (a mutant control abolishes activity).
    7. **What counts as progress.** Not the existence of one active design (which could be a promiscuous activity of a natural-like fold) but *a mechanistic signature*: loss of activity on mutating the designed catalytic residues, and a structure showing the designed active site geometry.

    **What is not known.** The achievable activity of a first-round design for a new-to-nature reaction (published de novo enzymes typically have low catalytic efficiencies that directed evolution raises by orders of magnitude), and whether any of the current generators produce preorganized active sites without evolution. The cascade's per-step attrition rates measured in this campaign are the field's missing data.

---

## 36.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: running a design campaign without fooling yourself"
    1. **Choose $p_0$ deliberately** (a family-aware model or language model) and report what the pipeline produces *without* the surrogate (the prior alone) as the baseline.
    2. **Estimate the predicted-versus-measured gap as a function of distance** from the training data in a pilot round, and set the trust region from it.
    3. **Hold out an independent surrogate** (a different architecture trained on different data) as a second filter and measure the disagreement of the two filters on selected designs: large disagreement means exploitation.
    4. **Enforce diversity** (cluster the candidates; choose from each cluster) and track the effective number of distinct designs.
    5. **Pre-register the success criterion** and report every tested design.
    6. **Treat expression and aggregation failures as part of the model's error**, not as noise.
    7. **Re-run the experiments of §36.3–36.4** with an assay-noise level matching yours and find the $\beta$ at which gap exceeds half the gain.

    **What it teaches.** Design is optimization against an imperfect proxy; the proxy's error structure, not the generator's creativity, sets the success rate.

    **An open question to carry forward.** The phase transition of §36.4 appeared at a particular $\beta$ because the surrogate's error grew with distance faster than its predicted gain. Could one estimate the *error-versus-distance curve* of a structure-based filter from the existing PDB by hold-out-by-similarity experiments (Chapter 45), and use it to set, before any experiment, a *trust-region radius* in sequence-and-structure space at which the filter's calibration begins to fail?

---

## 36.10 Connections

- **Backward:** diffusion and guidance (Chapter 15); equivariant networks (Chapter 16); KL-regularized alignment and reward overoptimization (Chapter 17); proteins and epistasis (Chapter 23); protein language models (Chapter 34); structure prediction and co-folding memorization (Chapter 35); winner's curse and adaptive reuse (Chapter 43); active learning and experimental design (Chapter 46); judge drift (Chapter 54).
- **Forward:** molecular design and drug discovery (Chapter 37); open problems in proteins and molecules (Chapter 52); biosecurity and publication (Chapter 59).

!!! takeaways "Key takeaways"
    1. Design is optimization of a surrogate under a trust region: the KL-regularized optimum is $p^\star\propto p_0e^{\hat f/\beta}$, and $\beta$ is the price of leaving the data.
    2. **The optimizer's curse**: predicted gain overstates true gain by about $1/R^2$; the overstatement was 0.8, 2.1 and 2.9 fitness units for surrogates with held-out $R^2$ of 0.70, 0.52 and 0.36, and grows with selection intensity.
    3. **A generator that ignores epistasis is a poor prior**: samples from a site-independent model had true fitness 6.8, half that of the natural sequences that produced it (13.2); designs from a good surrogate reached 22.8.
    4. **Model class beats data in some regimes**: an additive-only surrogate gave 18.4 where the correct pairwise class gave 22.8 with the same data.
    5. **Diversity collapses** at high optimization power (unique designs fell from 100% to 17–26% at near-argmax); select with a diversity criterion.
    6. **A trust-region phase transition**: with local data, designs inside the trust region were safe and unambitious; at the smallest $\beta$ the surrogate-optimized design left it and was 5.4 units worse than the starting reference while the surrogate predicted 22.0 for what was truly 18.3; an ensemble lower bound did not help because its members share the blind spot.
    7. Reported design success (BindCraft 10–100% across targets; Chai-2 about 16% of designs and 50% of targets; AlphaProteo 9–88%) is real for well-behaved targets; read every claim for denominator, selection, target difficulty, novelty, and controls. Screening against misuse must itself be evaluated adversarially.

---

## Further reading

- Watson, J. L. et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature* 620, 1089–1100. Dauparas, J. et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science* 378, 49–56. Dauparas, J. et al. (2025). Atomic context-conditioned protein sequence design using LigandMPNN. *Nat. Methods* 22, 717–723.
- Pacesa, M. et al. (2025). One-shot design of functional protein binders with BindCraft. *Nature* 646, 483–492. Zambaldi, V. et al. (2024). De novo design of high-affinity protein binders with AlphaProteo. *arXiv:2409.08022.* Chai Discovery (2025). Zero-shot antibody design in a 24-well plate. *bioRxiv* (Chai-2). Bennett, N. R. et al. (2025). Atomically accurate de novo design of antibodies with RFdiffusion. *Nature* (RFantibody).
- Hayes, T. et al. (2025). Simulating 500 million years of evolution with a language model. *Science* 387, 850–858. Madani, A. et al. (2023). Large language models generate functional protein sequences across diverse families. *Nat. Biotechnol.* 41, 1099–1106. King, S. H. et al. (2026). Generative design of novel bacteriophages with genome language models. *Science.* Jiang, K. et al. (2024). Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science.*
- Gao, L., Schulman, J. & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML.* Rafailov, R. et al. (2023). Direct preference optimization. *NeurIPS.* Brookes, D. H., Park, H. & Listgarten, J. (2019). Conditioning by adaptive sampling for robust design. *ICML.* Fannjiang, C. & Listgarten, J. (2020). Autofocused oracles for model-based design. *NeurIPS.* Trabucco, B. et al. (2022). Design-Bench: benchmarks for data-driven offline model-based optimization. *ICML.*
- Wittmann, B. J. et al. (2025). Strengthening nucleic acid biosecurity screening against generative protein design tools. *Science.* Baker, D. & Church, G. (2024). Protein design meets biosecurity. *Science* 383, 349. Bloomfield, D. et al. (2024). AI and biosecurity: the need for governance. *Science* 385, 831–833.
