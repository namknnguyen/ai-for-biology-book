# Chapter 46. Experimental Design, Active Learning, and Closed-Loop Discovery

!!! abstract "Chapter at a glance"
    **Motivation.** In AI for biology, data are not a fixed resource handed to the modeler; they are *chosen*: which perturbations to run, how many cells per condition, which variants to put in an assay, which compounds to synthesize next. Chapter 44 showed that observational data cannot identify causal effects, Chapter 45 that new domains need target labels, Chapter 31 that edit effects need decorrelated variation. This chapter turns those observations into a design discipline: the theory (information matrices and expected information gain), four quantitative experiments (coverage, depth versus breadth, designs that identify collinear effects, and closed-loop optimization), and the practical statistics of running experiments that models can learn from.
    **Prerequisites.** Chapters 4, 5, 25, 31, 44, 45, 56.
    **You will be able to:** (1) state D- and A-optimality, derive the leverage-based greedy rule, and explain why it is near-optimal; (2) choose perturbations for coverage rather than for volume and quantify the benefit for rare cases; (3) allocate a fixed cell budget between breadth and depth; (4) design edits that identify effects which are collinear in natural data; (5) run an active-learning or Bayesian-optimization loop and recognize model exploitation; (6) apply blocking, randomization, controls, and power analysis to biological screens.

---

## 46.1 Experiments as the interface between models and biology

A model-building project has three experimental needs.

* **Experiments to learn**: *training data* chosen to make the target quantity identifiable and the model accurate where it will be used.
* **Experiments to test**: *evaluation data* that are novel relative to training and cheap to label at scale (Chapters 43, 45).
* **Experiments to decide**: *discovery*, where the model proposes and the experiment confirms (design cycles; Chapters 36, 37, 54).

The cost structure differs sharply across domains: a sequence-variant MPRA gives $10^4$–$10^6$ measurements per experiment; a genome-scale Perturb-seq a few million cells but only one context and time point; a deep mutational scan $10^5$–$10^6$ variants of one protein; a small-molecule design cycle perhaps tens of compounds per week; a clinical trial a few hundred participants per year. *The right design depends on which resource is binding*: reagents, cells, sequencing, time, or ethical approval.

**The loop.** Most modern workflows are **design–build–test–learn** (DBTL) cycles: a model proposes a set of experiments; they are built and measured; the model is updated; repeat. The theory of this chapter says how to pick the batch.

---

## 46.2 Optimal design in the linear model

Consider $y=X\beta+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2I)$. The estimator $\hat\beta$ has covariance $\sigma^2(X^\top X)^{-1}$, so the **information matrix** $M=X^\top X/\sigma^2$ summarizes what the design can learn. Optimal design chooses the rows of $X$ (the experiments) from a candidate set to make $M^{-1}$ small in some sense.

| Criterion | Minimizes | Interpretation |
|---|---|---|
| **A-optimal** | $\mathrm{tr}\,M^{-1}$ | mean variance of the coefficient estimates |
| **D-optimal** | $-\log\det M$ | volume of the confidence ellipsoid |
| **E-optimal** | $\lambda_{\max}(M^{-1})$ | the worst direction |
| **G-optimal** | $\max_x x^\top M^{-1}x$ | worst-case prediction variance over the candidate set |

(By the Kiefer–Wolfowitz equivalence theorem, D- and G-optimality coincide for continuous designs.)

**The greedy rule and its guarantee.** Adding an experiment $x$ to a design with information matrix $M$ changes the log-determinant by
$$
\log\det(M+xx^\top/\sigma^2)-\log\det M=\log\big(1+x^\top M^{-1}x/\sigma^2\big),
$$
by the matrix determinant lemma. So the greedy D-optimal step is to add the candidate with the largest **leverage** $x^\top M^{-1}x$: the one the current design predicts worst. Since $\log\det$ is monotone and *submodular* in the set of experiments, the greedy design is within a factor $(1-1/e)$ of the optimum (Nemhauser et al., 1978; Krause et al., 2008) [[E]]. With a Gaussian prior $\beta\sim\mathcal N(0,\Sigma_0)$, the expected information gain of Chapter 56 is exactly $\tfrac12\log\det(I+\Sigma_0^{1/2}X^\top X\Sigma_0^{1/2}/\sigma^2)$, so **D-optimal design is expected-information-gain design for a linear Gaussian model**, and leverage-based selection is uncertainty sampling.

**What the theory ignores**: nonlinearity (use local linearizations or Bayesian designs), model misspecification (leverage is wrong if the linear model is), the possibility that features are themselves uncertain (embeddings), and batch effects (§46.7). It nevertheless gives the right qualitative guidance: *measure where the current model is most uncertain, and cover the space.*

---

## 46.3 Which perturbations to measure

`code/ch46_design.py`, experiment 1. Perturbations (genes) are described by a 12-dimensional feature vector (prior-knowledge embedding: pathway, expression, network position) with **clustered and imbalanced** structure: six clusters of sizes 45%, 25%, 15%, 8%, 5%, and 2%. Each perturbation's measured response over 500 genes is $z_p^\top\Theta$ plus noise. We choose $k$ perturbations to measure, fit a ridge model, and predict the response of the held-out rest (mean of 20 random worlds):

| Budget $k$ | Method | Held-out $R^2$ (all) | Held-out $R^2$ on the **rarest cluster** (2%) |
|---|---|---|---|
| 12 | Random | 0.823 | **0.618** |
| 12 | D-optimal (greedy) | 0.894 | **0.849** |
| 12 | Diversity (farthest point) | 0.883 | 0.839 |
| 24 | Random | 0.890 | 0.686 |
| 24 | D-optimal | 0.923 | 0.898 |
| 24 | Diversity | 0.917 | 0.887 |
| 48 | Random | 0.924 | 0.843 |
| 48 | D-optimal | 0.933 | 0.909 |
| 48 | Diversity | 0.931 | 0.905 |
| 96 | Random | 0.935 | 0.902 |
| 96 | D-optimal | 0.937 | 0.906 |
| 96 | Diversity | 0.937 | 0.917 |

Three conclusions. **(i) At small budgets, design beats random**: at $k=12$ the D-optimal design gains 7 points of overall $R^2$ and **23 points on the rarest cluster** (0.849 vs 0.618), because random sampling draws mostly from the large clusters and extrapolates to the rare one. **(ii) The advantage vanishes as the budget grows** (at $k=96$ all methods are within 0.5 points overall), since random sampling eventually covers every cluster. *Design matters most when experiments are scarce, which is exactly the situation of expensive contexts (primary cells, in-vivo, clinical).* **(iii) Overall $R^2$ hides the failure**: the rarest cluster, the one most likely to be a *novel* perturbation at deployment, is where random sampling fails. Evaluate by stratum (Chapter 43).

Farthest-point (diversity) selection, which needs no model, performs nearly as well as D-optimal and is more robust to model misspecification: a sensible default is to **combine model-based uncertainty with diversity**.

---

## 46.4 Breadth versus depth: cells per perturbation

A fixed budget of 200,000 cells can profile few perturbations deeply or many shallowly (experiment 2; features $d=12$, per-cell noise SD 1.0, so the mean of $n$ cells has noise SD $1/\sqrt n$):

| Perturbations $P$ | Cells each $n$ | RMSE predicting 2,000 held-out perturbations |
|---|---|---|
| 20 | 10,000 | 0.180 |
| 50 | 4,000 | 0.039 |
| 200 | 1,000 | 0.011 |
| 1,000 | 200 | **0.008** |
| 4,000 | 50 | 0.008 |
| 10,000 | 20 | 0.008 |

When the quantity to learn is the **map** from perturbation features to response (a low-dimensional function, here $d=12$), *breadth wins until the feature space is covered* ($P\gtrsim100$) and then the error is flat: the precision of the learned map depends on the total number of cells, not on how they are divided, as long as every perturbation is represented. In practice, genome-scale Perturb-seq screens use about 100–200 cells per perturbation (Chapter 25). **The conclusion reverses if the goal is to characterize each perturbation individually**, where the error of a perturbation's mean is $1/\sqrt n$ regardless of $P$ (0.22 at 20 cells, 0.07 at 200, 0.01 at 10,000) and breadth is irrelevant; it also reverses when effects are idiosyncratic (no shared structure to borrow), when per-perturbation sparsity is high and detectability requires depth (Chapter 25: with 100 cells only large effects on well-expressed genes are detected), and when the map is high-dimensional. *Ask what the data will be used for before choosing a design.*

**Pooled and compressed designs.** Instead of one perturbation per cell, measure *combinations* and deconvolve under a sparsity assumption (compressed sensing): composite perturbation screens measure many perturbations per sample and recover individual effects with sparse regression, reducing cost by an order of magnitude when effects are sparse (Cleary et al., 2021; Yao et al., 2024, "compressed Perturb-seq") [[S]]. **Combinatorial designs** (all pairs in a gene set: Norman et al., 2019) measure interactions but scale as $\binom{n}{2}$; *factorial* or *fractional-factorial* designs recover main effects and low-order interactions with far fewer conditions.

---

## 46.5 Designing for identifiability

Chapter 31 showed that when two motifs are perfectly collinear in reference sequences, observational data identify only their sum. Experiment 3 simulates the measurements one could make instead (true effects $+1.0$ for A and $-0.8$ for B; noise SD 0.3; ordinary least squares; mean absolute error over 300 repetitions):

| Design | $n=20$ | $n=100$ | $n=400$ |
|---|---|---|---|
| Observational reference loci (A and B always together) | 0.90 / 0.90 | 0.90 / 0.90 | 0.90 / 0.90 |
| Random mutagenesis (each motif disrupted with probability 0.15) | 0.24 / 0.25 | 0.07 / 0.07 | 0.03 / 0.04 |
| **Targeted single-motif deletions** (half A, half B) | **0.07 / 0.08** | 0.03 / 0.03 | **0.02 / 0.02** |
| Targeted $2\times2$ factorial (A, B, both, neither) | 0.09 / 0.10 | 0.04 / 0.04 | 0.02 / 0.02 |

(Entries: mean absolute error in the estimate of A / of B.) **The observational error does not decrease with $n$** (0.90 at 20, 100, and 400; the estimate is the minimum-norm split of the identified sum, Chapter 31). **Random mutagenesis decorrelates the motifs by chance** and works at moderate $n$ (error 0.07 at 100), but **targeted single-motif deletions achieve the same accuracy with about a quarter of the measurements** (0.07 at $n=20$ versus 0.07 at $n=100$): the design *chooses* the variation that is missing. In general:

* **Orthogonal (or balanced) designs** make the columns of $X$ uncorrelated, so the effects are estimated independently with minimal variance (Fisher's principle; factorial designs).
* **Randomization** breaks the association between the treatment and unmeasured variables (Chapter 44); **blocking** removes nuisance variation (batch) by comparing within blocks; **instruments** supply variation uncorrelated with confounders.
* **Controls** (non-targeting guides, scrambled sequences, positive controls of known effect, spike-ins) calibrate the assay and detect drift.

---

## 46.6 Sequential design: active learning and Bayesian optimization

When experiments are run in rounds, the model can choose the next batch using what it has learned.

* **Active learning** selects points to improve the *model*: uncertainty sampling (largest predictive variance), query-by-committee (disagreement among an ensemble), expected error reduction, and diversity for batches.
* **Bayesian optimization** selects points to *find the optimum*: a surrogate model with uncertainty (Gaussian process; or an ensemble) and an acquisition function that trades exploitation against exploration: **expected improvement**, **upper confidence bound** $\mu(x)+\kappa\sigma(x)$, or **Thompson sampling** (draw a function from the posterior and optimize it). Batches use diversity penalties or independent Thompson draws (Shahriari et al., 2016).
* In protein engineering, **machine-learning-guided directed evolution** and Gaussian-process-based navigation of fitness landscapes have been used to find improved variants with far fewer measurements than random screening (Romero & Arnold, 2013; Yang et al., 2019) [[S]].

**Experiment 4.** A rugged pairwise-epistatic landscape over 12-position, 4-letter sequences (global saturating readout), a pool of 100,000 candidates, 40 random starting measurements, and 8 rounds of 20 new measurements chosen by (a) random selection, (b) greedy ranking by a ridge surrogate on single-position and pair features, (c) bootstrap Thompson sampling (each batch member chosen from a bootstrap-refit surrogate); 20 runs each:

| Strategy | Best fitness found after rounds 0 / 2 / 4 / 8 (fraction of the best in the pool) | Probability of reaching the top 0.1% of the pool (20 runs) |
|---|---|---|
| Random selection | 0.977 / 0.985 / 0.991 / 0.999 | 0.20 |
| Greedy ranking by the ridge surrogate | 0.977 / 0.999 / 1.000 / 1.000 | 0.95 |
| Bootstrap Thompson sampling | 0.977 / 0.999 / 1.000 / 1.000 | 1.00 |

**Reading Experiment 4.**

1. **The fitness fraction hides the difference.** Because the readout saturates, even random selection reaches 0.999 of the pool's best fitness after the 8 rounds (200 measurements in total, including the 40 starting points). The discriminating metric is the probability of reaching the **top 0.1%** of the pool: 0.20 for random selection (as it should be: 200 random evaluations succeed with probability $1-0.999^{200}=0.18$), **0.95** for greedy ranking by the surrogate, and **1.00** for bootstrap Thompson sampling.
2. **A surrogate-guided search is five times as likely to find a top-0.1% sequence with the same 200 measurements.** The advantage appears already after two rounds (0.999 against 0.985).
3. **Thompson sampling is not distinguishable from greedy ranking here.** With 20 runs per strategy the binomial standard error of a probability near 0.95 is about 0.05, so 0.95 against 1.00 is within noise; the practical difference between the strategies appears when the surrogate is less accurate or batches must be diverse (Thompson draws are diverse by construction).
4. **Why it works so well, and why it may not transfer.** The surrogate's feature set (single positions plus 300 of the possible pair terms) contains most of the structure of the true landscape (pairwise epistasis); the pool is a fixed set of 100,000 random sequences, so the search is a *retrieval* within a known set rather than an exploration of an unbounded space; and the landscape's global nonlinearity is monotone. Chapter 36 shows what happens when the surrogate is misspecified (an additive surrogate wasted four fitness units), when data are local (a trust-region phase transition at small $\beta$), and when diversity collapses (unique designs fell to 17–26% at near-argmax selection).

!!! lens "Research lens: exploration, exploitation, and model exploitation"
    **Assumption it makes explicit:** the surrogate is accurate where the acquisition function sends you. **Failure modes:** (i) *model exploitation*: optimizing a learned predictor finds inputs where the predictor is wrong (Chapter 36); trust regions, KL penalties, ensembles, and pessimism help. (ii) *Batch collapse*: a greedy batch contains near-duplicates; use diversity. (iii) *Distribution shift between rounds* (Chapter 45): early data come from a different region than late data. (iv) *Noise*: expected improvement assumes a noise model; a screening assay with 30% noise misleads greedy selection.

---

## 46.7 Practical statistical design

* **Replication and power.** Decide the *unit of replication* (Chapter 4) and the effect size you need to detect; use the sample-size formula of Chapter 59; use pilot data to estimate variance; include *biological* (donors, independent cultures) not only technical replicates.
* **Randomization and blocking.** Randomize plate positions and processing order; **balance** conditions across batches (every batch contains every condition) so that batch effects are not confounded with treatment (Chapter 25's multiplexing: hashing places conditions in the same sequencing run). Treat batch as a random effect.
* **Controls.** Negative (non-targeting guides, empty vector, vehicle), positive (known effect), and process controls (spike-ins, a reference cell line in every batch: the *bridge sample* of Chapter 30).
* **Dose, time, and context.** Decide in advance the time points and doses; an effect measured at one time and context may not transport (Chapter 44). *Spend part of the budget on context variation if the model must transfer.*
* **Pre-registration of the analysis** (Chapter 59) and *a locked test set* of experiments run after the model is frozen (a prospective test; Chapter 43).
* **Failure planning.** Plan for dropouts (failed guides, low-quality cells): oversample by the expected failure rate.

---

## 46.8 The experiments, verbatim

```python
--8<-- "code/ch46_design.py"
```

```text
== 1. Learning a response map from k measured perturbations (features d = 12, clustered and imbalanced; 600 candidate perturbations, 500 genes) ==
budget k   method                       held-out R2 (all)   held-out R2 on the rarest cluster (2% of perturbations)
    12     random                          0.823                 0.618
    12     D-optimal (greedy)              0.894                 0.849
    12     diversity (farthest point)      0.883                 0.839
    24     random                          0.890                 0.686
    24     D-optimal (greedy)              0.923                 0.898
    24     diversity (farthest point)      0.917                 0.887
    48     random                          0.924                 0.843
    48     D-optimal (greedy)              0.933                 0.909
    48     diversity (farthest point)      0.931                 0.905
    96     random                          0.935                 0.902
    96     D-optimal (greedy)              0.937                 0.906
    96     diversity (farthest point)      0.937                 0.917

== 2. A fixed budget of 200,000 cells: many perturbations with few cells, or few with many? (d = 12 features, per-gene noise SD 1.0 per cell) ==
perturbations P   cells each n   error in predicting 2,000 held-out perturbations (RMSE of response, mean over genes)
          20         10000         0.180
          50          4000         0.039
         200          1000         0.011
        1000           200         0.008
        4000            50         0.008
       10000            20         0.008

== 3. Identifying two motif effects (true +1.0 and -0.8) whose occurrences are perfectly collinear in reference sequences ==
experiment design (n measurements)                         n     mean |error| in effect A    in effect B    (noise SD 0.3)
observational reference loci                                 20          0.896               0.904
observational reference loci                                100          0.899               0.901
observational reference loci                                400          0.900               0.900
random mutagenesis (each motif disrupted with prob 0.15)     20          0.242               0.245
random mutagenesis (each motif disrupted with prob 0.15)    100          0.070               0.071
random mutagenesis (each motif disrupted with prob 0.15)    400          0.033               0.035
targeted single-motif deletions (half A, half B)             20          0.073               0.078
targeted single-motif deletions (half A, half B)            100          0.034               0.034
targeted single-motif deletions (half A, half B)            400          0.017               0.017
targeted: A, B, both, neither (balanced factorial)           20          0.090               0.095
targeted: A, B, both, neither (balanced factorial)          100          0.039               0.037
targeted: A, B, both, neither (balanced factorial)          400          0.021               0.020

== 4. Design-build-test-learn: 12-position, 4-letter sequences, pairwise-epistatic landscape; 40 random starts, 8 rounds of batches of 20 ==
strategy                 best fitness found after rounds 0, 2, 4, 8 (as a fraction of the best in a 100,000 pool)      P(reach top 0.1% of the pool)  [200 random evaluations would succeed with probability 0.18]
random                   0.977  0.985  0.991  0.999                                                  0.20
greedy (ridge)           0.977  0.999  1.000  1.000                                                  0.95
bootstrap Thompson       0.977  0.999  1.000  1.000                                                  1.00
```

---

## 46.9 Worked research examples

!!! example "Worked Research Example 46.1: Designing a perturbation atlas to train a virtual-cell model"
    **Situation.** A consortium has funding for 5 million cells and wants to train a model that predicts transcriptional responses to unseen genetic perturbations in a primary cell type. There is no established answer for the best design.

    **Question.** What would you do?

    **Reasoning (the Expert Chain, compactly).**

    1. *Goal and metric.* Predict the response of held-out perturbations in the *same* context (C2) and, if possible, in a second context (transport). Metric: ceiling-normalized score on perturbations with real effects (Chapter 25).
    2. *Breadth versus depth.* The map is learned from many perturbations; but per-perturbation power (Chapter 25) requires about 100–200 cells to detect large effects, and the reliability analysis shows that perturbations with few affected genes need *more* cells to be usable as a benchmark. A reasonable allocation: about 12,000 expressed genes with 3 guides each at roughly 100 cells per guide (about 3.6 million cells), about 0.5 million cells to *re-measure* 500 perturbations at 1,000 cells (to calibrate reliability and give a high-quality test set), and the remainder for combinations, doses, and a second context (below).
    3. *Coverage design.* Choose the perturbation set to cover the *feature space* of a prior-knowledge embedding (pathway, complex, expression, network position) with farthest-point selection among genes expressed in the cell type; include *all* genes of key complexes so that held-out perturbations have mates.
    4. *Identifiability design.* Include **combinations** for a chosen subset (pairs within complexes and across pathways) to test additivity and to give identified interaction effects; include **dose series** for a few genes (titrated knockdown) for dose–response; include **non-targeting controls** at about 5–10% of cells distributed across batches.
    5. *Batch design.* Balance perturbations across batches and lanes with hashing; include a **bridge sample** (a reference cell population in every batch).
    6. *Context and time.* Allocate about 10% of the budget to a second context (another donor, another cell type) for the *same* subset of perturbations, to estimate transportability (Chapter 45), and two time points for a subset.
    7. *Evaluation plan.* Lock a *prospective test set* of perturbations (held out by pathway, not only by gene) and a second context; pre-register metrics and baselines (control mean, mean shift, additive; Chapter 39).
    8. *Kill criteria.* If a pilot (500,000 cells) shows that fewer than 10% of perturbations have reliability above the pre-specified ceiling, change the allocation toward depth and toward perturbations with stronger effects (essential genes and regulators).

    **What no one knows.** The best allocation among breadth, depth, combinations, doses, and contexts; it depends on how transportable effects are (Chapter 45) and on how the model will be used. Chapter 56's expected-information-gain framework suggests running a pilot to estimate the quantities that determine the optimum (reliability distribution; cross-context correlation) *before* committing the budget.

!!! example "Worked Research Example 46.2: Choosing variants for a validation MPRA"
    **Situation.** You have 120 GWAS credible sets (Chapter 26) with a total of 1,900 variants and a sequence-model score for each. You can put 8,000 sequences in an MPRA (variants plus controls).

    **Question.** Which sequences, and what for?

    **Reasoning.**

    1. *Purpose.* To discriminate causal from tag variants within credible sets and to calibrate the model (C3–C4).
    2. *Include all variants in each credible set* (about 1,900 × 2 alleles = 3,800 sequences, in multiple replicates and positions/barcodes) so that within-set contrasts are measured; this is a within-locus design (A8) that avoids LD confounding.
    3. *Controls.* Positive controls (known strong regulatory variants), negative controls (scrambled sequences, GC-matched), *neutral* variants from the same loci with low PIP.
    4. *Identifiability.* Where the model's predicted effect is collinear with the lead variant's (variants in LD that disrupt the same motif), include *single-nucleotide saturating edits* of the motif so the effect is attributed to the motif rather than a neighbor.
    5. *Evaluation.* Pre-specify: for each credible set, the model's top-ranked variant versus a baseline's (conservation/distance), and the measured top variant; metric: the fraction of sets in which the model's top pick is the measured top; with the number of sets (120) the resolution is about ±9 points (binomial), so *state it*.
    6. *Context.* The MPRA is in one cell type; choose the cell type on the strength of the eQTL/accessibility evidence for the loci, and state the limit.

    **Expert analysis.** The design is the experiment that *the information gain says is missing* (Chapter 56): variation within credible sets, decorrelated from LD by construction.

---

## 46.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: a design review"
    1. **What will the data be used for?** Map learning, per-item characterization, validation, or discovery? The optimal design differs (§46.4).
    2. **What is unidentified in natural data?** List correlated features and confounders; design the variation that separates them (§46.5).
    3. **Coverage.** Where could the model be asked to extrapolate? Include those regions (rare clusters, other contexts).
    4. **Batch and randomization.** Is any nuisance variable confounded with the treatment? Balance and randomize; include controls and a bridge sample.
    5. **Power and resolution.** Effect size, unit of replication, effective $n$ (Chapter 59).
    6. **Sequential plan.** If rounds are possible, which acquisition rule? What is the trust region against model exploitation?
    7. **Pre-specification.** Metrics, baselines, a locked prospective test set, kill criteria.
    8. **Cost per bit.** Estimate the expected information gain per unit cost for the top three designs (Chapter 56) and pick by that, not by convention.

    **What it teaches.** Design is where most of the leverage in AI for biology lies: the right 500 experiments can be worth more than 5 million routine ones, and the wrong 5 million cannot be rescued by a better model.

    **An open question to carry forward.** Optimal design theory assumes you know the model class; a foundation model is a *flexible* class whose uncertainty is poorly characterized. How should one design experiments to improve a foundation model whose posterior is not available: with ensembles, with the model's own embeddings (diversity in representation space), with conformal uncertainty? Test on a simulation in which the ground truth is a nonlinear function of perturbation features and compare uncertainty-sampling variants by held-out error per measurement.

---

## 46.11 Connections

- **Backward:** information and entropy (Chapter 5); perturbation measurement and power (Chapter 25); counterfactual identifiability and the composite-element toy (Chapter 31); causal identification (Chapter 44); shift and target labels (Chapter 45); expected information gain (Chapter 56).
- **Forward:** protein and molecular design cycles (Chapters 36, 37); virtual-cell data design (Chapters 39, 51); AI-driven discovery and self-driving laboratories (Chapter 54); capstone design (Chapter 59).

!!! takeaways "Key takeaways"
    1. Data are **chosen**: design determines what is identifiable, how accurate the model is where it matters, and what the test can show.
    2. For linear models D-optimal design maximizes $\log\det(X^\top X)$; the greedy leverage rule is within $(1-1/e)$ of optimal because $\log\det$ is submodular, and it equals expected-information-gain design.
    3. **Design beats random at small budgets**: at $k=12$ D-optimal selection raised held-out $R^2$ on the rarest cluster from 0.618 to 0.849; the advantage vanishes by $k=96$; overall $R^2$ hides rare-case failure.
    4. **Breadth versus depth** depends on the goal: to learn a low-dimensional response map, many perturbations at modest depth (error 0.008 for 1,000–10,000 perturbations versus 0.18 for 20); to characterize each perturbation, depth ($1/\sqrt n$).
    5. **Collinear effects need designed variation**: observational error stayed at 0.90 for any $n$; random mutagenesis reached 0.07 at $n=100$; targeted single-motif deletions reached the same at $n=20$.
    6. **Randomize, block, balance, and control** (non-targeting guides, bridge samples); decide the replication unit and power in advance.
    7. **Closed-loop designs** (active learning, Bayesian optimization) find optima with far fewer measurements than random screening (with 200 measurements, the probability of reaching the top 0.1% of a 100,000-sequence pool was 0.95 for greedy surrogate ranking and 1.00 for bootstrap Thompson sampling, against 0.20 for random selection), but fail by model exploitation, batch collapse, and shift unless trust regions and diversity are used (Chapter 36).
    8. Choose designs by **expected information gain per cost**, estimated from a pilot.

---

## Further reading

- Fisher, R. A. (1935). *The Design of Experiments.* Oliver & Boyd. Kiefer, J. & Wolfowitz, J. (1960). The equivalence of two extremum problems. *Can. J. Math.* 12, 363–366. Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: a review. *Stat. Sci.* 10, 273–304. Nemhauser, G. L., Wolsey, L. A. & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions. *Math. Program.* 14, 265–294. Krause, A., Singh, A. & Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes. *J. Mach. Learn. Res.* 9, 235–284.
- Settles, B. (2009). Active learning literature survey. *University of Wisconsin–Madison Computer Sciences Technical Report 1648.* Shahriari, B., Swersky, K., Wang, Z., Adams, R. P. & de Freitas, N. (2016). Taking the human out of the loop: a review of Bayesian optimization. *Proc. IEEE* 104, 148–175.
- Romero, P. A., Krause, A. & Arnold, F. H. (2013). Navigating the protein fitness landscape with Gaussian processes. *PNAS* 110, E193–E201. Yang, K. K., Wu, Z. & Arnold, F. H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nat. Methods* 16, 687–694. Hie, B. L. & Yang, K. K. (2022). Adaptive machine learning for protein engineering. *Curr. Opin. Struct. Biol.* 72, 145–152.
- Norman, T. M. et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* 365, 786–793. Cleary, B. et al. (2021). Compressed sensing for highly efficient imaging transcriptomics. *Nat. Biotechnol.* 39, 936–942. Yao, D. et al. (2024). Scalable genetic screening for regulatory circuits using compressed Perturb-seq. *Nat. Biotechnol.* 42, 1282–1295. Roohani, Y., Huang, K. & Leskovec, J. (2024). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nat. Biotechnol.* 42, 927–935.
