# Chapter 44. Causal Inference for Biology

!!! abstract "Chapter at a glance"
    **Motivation.** The question that biology and medicine care about is nearly always causal: *what happens if we knock this gene down, edit this variant, give this drug, change this sequence?* Models trained on observational data learn associations, and association is not causation: the Four Gaps' **inference gap** (G-I). This chapter gives the minimal formal toolkit (structural causal models, the do-operator, potential outcomes, adjustment, instruments), shows concretely how hidden cell state and batch defeat observational estimates, and uses a simulation to quantify what interventions (CRISPR knockdowns, perturbation screens, edits) buy. It closes with what this means for models trained on atlases and for the "virtual cell".
    **Prerequisites.** Chapters 4, 25, 26, 31, 43.
    **You will be able to:** (1) distinguish association, intervention, and counterfactual; (2) write a structural causal model and apply the do-operator, backdoor adjustment, and instrumental variables; (3) derive the bias of an observational regression under a hidden confounder and show that it does not shrink with sample size; (4) list the confounders and colliders that arise in biological data; (5) explain what perturbation data identify, how incomplete efficiency changes the estimand, and what causal discovery from observations can and cannot do; (6) design an analysis in which an intervention, an instrument, or a randomization identifies the quantity of interest.

---

## 44.1 Three rungs: seeing, doing, imagining

Pearl's **ladder of causation** distinguishes three kinds of questions.

1. **Association** (seeing): $P(Y\mid X=x)$. *Genes $A$ and $B$ are co-expressed.* Answerable from observational data.
2. **Intervention** (doing): $P(Y\mid\mathrm{do}(X=x))$. *If we knock down $A$, what happens to $B$?* Requires an experiment or assumptions that make the intervention identifiable from observations.
3. **Counterfactual** (imagining): $P(Y_{x}\mid X=x',Y=y')$. *This cell has high $B$ and wild-type $A$: what would $B$ have been had $A$ been knocked down?* Requires a model of the mechanism.

The Claim Ladder of Chapter 1 maps onto these: C1–C2 are associational (prediction), C4 (design) is interventional, and the "virtual cell" aspiration (Chapter 39) is a counterfactual one. **A model can be superb at rung 1 and uninformative at rung 2**, which is the content of Chapter 31's toy (accurate predictions, wrong edit effects).

---

## 44.2 The formalism

### 44.2.1 Structural causal models and the do-operator

A **structural causal model** (SCM) specifies each variable as a function of its direct causes and independent noise: $X_j:=f_j(\mathrm{pa}(X_j),\varepsilon_j)$. Its graph is a DAG (directed acyclic graph) over variables. An **intervention** $\mathrm{do}(X_i=c)$ *replaces the equation for $X_i$* by $X_i:=c$ and leaves the rest unchanged, so it **cuts the arrows into $X_i$**. Observing $X_i=c$ conditions the distribution; doing it surgically changes it. They differ whenever $X_i$ shares causes with the outcome.

**Linear example.** $x=Bx+\varepsilon$ gives $x=(I-B)^{-1}\varepsilon$. The **total effect** of a unit intervention on $x_i$ on $x_j$ is $[(I-B)^{-1}]_{ji}$ (sum over directed paths of products of edge weights), and the **direct effect** is $B_{ji}$. We use this model in §44.4.

### 44.2.2 Potential outcomes

For a binary treatment $T$ and outcome $Y$, each unit has potential outcomes $Y(1),Y(0)$; the individual effect $Y(1)-Y(0)$ is never observed (the fundamental problem). The **average treatment effect** $\mathrm{ATE}=\mathbb E[Y(1)-Y(0)]$ is identified if treatment is as-if random given covariates: **ignorability** $\{Y(0),Y(1)\}\perp T\mid Z$ and **positivity** $0<P(T=1\mid Z)<1$. Randomization ensures ignorability by design. In biology, **perturbation assignment by random guide RNA** is such a randomization (§44.5).

### 44.2.3 Confounding, adjustment, and the backdoor criterion

A **confounder** $Z$ causes both $X$ and $Y$, opening a "backdoor" path $X\leftarrow Z\rightarrow Y$. If a set $Z$ blocks all backdoor paths between $X$ and $Y$ and contains no descendant of $X$, then

$$
P\big(y\mid\mathrm{do}(x)\big)=\sum_zP(y\mid x,z)\,P(z)\qquad(\text{adjustment formula}).
$$

*Proof sketch.* In the intervened model, $z$ keeps its marginal $P(z)$ (the intervention does not affect its causes) and $y$ keeps its conditional $P(y\mid x,z)$ (the mechanism for $y$ is unchanged). Marginalizing over $z$ gives the formula. The requirement that $Z$ be **measured** is the practical difficulty: *adjustment cannot remove a confounder that is hidden*.

**Colliders.** Conditioning on a common effect $X\rightarrow C\leftarrow Y$ *creates* an association between $X$ and $Y$ (collider bias or selection bias). Examples in biology: selecting cells by a marker that both the perturbation and the outcome affect; restricting analysis to cells that survive the perturbation; case–control ascertainment (Chapter 26); selecting variants by significance in the same data.

### 44.2.4 Instrumental variables

When $Z$ is an **instrument** (affects $X$; independent of the confounders; affects $Y$ only through $X$), the effect of $X$ on $Y$ is identified even with hidden confounding. In the linear model $Y=\theta X+U+\epsilon_y$, $X=\gamma Z+U+\epsilon_x$, $\mathrm{Cov}(Z,Y)=\theta\,\mathrm{Cov}(Z,X)$, so $\theta=\mathrm{Cov}(Z,Y)/\mathrm{Cov}(Z,X)$ (the Wald ratio of Chapter 26). **Mendelian randomization** uses genetic variants as instruments; **CRISPR screens** use the *randomly assigned guide* as an instrument for gene expression (since guides differ in efficiency, §44.5).

### 44.2.5 Identification versus estimation

A causal quantity is **identified** if it is a function of the observed-data distribution under the assumptions. *Identification is a property of assumptions, not of sample size*: with hidden confounding, infinite observational data do not identify the effect; estimation error can be reduced by more data, *bias* by design.

---

## 44.3 Why observational biological data are confounded

| Source | Mechanism | Typical consequence |
|---|---|---|
| **Cell state / cell cycle / cell type** | A hidden state moves many genes together | Co-expression everywhere: hundreds of correlated gene pairs without direct regulation |
| **Batch, donor, technology** | Experimental variables affect many measurements | Spurious associations (Chapters 25, 30) |
| **Library size and compositionality** | Normalization couples genes (Chapter 25) | Negative or positive correlations by construction |
| **Ancestry and relatedness** | Population structure affects genotype and phenotype (Chapter 26) | Spurious variant–trait associations |
| **Tissue composition** | A bulk sample is a mixture of cell types | Gene–disease associations driven by composition |
| **Reverse causation** | The outcome influences the exposure | Biomarkers that change because of disease |
| **Selection (survival, ascertainment)** | Conditioning on a collider | Associations that exist only in the selected sample |
| **Measurement** | The assay depends on the exposure | Differences in detection rather than biology |

**A one-line derivation.** Let $x_i=\lambda_iu+e_i$ and $x_j=\beta x_i+\lambda_ju+e_j$ with hidden $u$ ($\mathrm{Var}\,u=1$). Then $\mathrm{Cov}(x_i,x_j)=\beta\,\mathrm{Var}(x_i)+\lambda_i\lambda_j$ and the regression slope of $x_j$ on $x_i$ is

$$
\hat\beta_\text{OLS}=\beta+\frac{\lambda_i\lambda_j}{\mathrm{Var}(x_i)}.
$$

The bias $\lambda_i\lambda_j/\mathrm{Var}(x_i)$ **does not depend on $n$**. With $\beta=0$ (no effect) and loadings of the same sign, the "effect" is positive and, for large $n$, arbitrarily significant.

---

## 44.4 A simulation: what do observations, and what do interventions, identify?

`code/ch44_causal.py` simulates a **linear SCM** with 40 genes. A sparse random DAG gives each gene 0–2 parents (edge weights $\pm0.4$–$0.9$); a **hidden confounder** $u$ (a cell state) affects all genes with loadings $\lambda_g\sim\mathcal N(0,\sigma_\lambda^2)$, noise SD 0.5. We generate 5,000 observational cells, and for half of the genes a **knockdown experiment** ($\mathrm{do}(x_i=-2)$ in 100 cells; equations of the intervened gene are cut). The task: rank ordered pairs $(i,j)$ by evidence that knocking down $i$ changes $j$ ($|{\rm total\ effect}|>0.15$); AUROC is the performance (mean of 8 random networks).

| Scenario | $\lvert r\rvert$ | Regression slope $j\!\sim\!i$ | Partial correlation | Slope, perturbed $i$ only | **Interventional** (perturbed $i$) | Orientation correct (observational) |
|---|---|---|---|---|---|---|
| No hidden confounder | 0.934 | 0.963 | 0.704 | 0.968 | **1.000** | 0.906 |
| Hidden confounder (SD 0.7) | 0.631 | 0.680 | 0.720 | 0.705 | **0.997** | 0.709 |
| Strong confounder (SD 1.5) | 0.541 | 0.583 | 0.720 | 0.617 | **0.975** | 0.615 |

Reading the table.

1. **Without a hidden confounder, observational data work well** (AUROC 0.93–0.97 for correlation and regression slope): in a linear DAG with measured variables, associations track effects. Interventions are still perfect (1.000).
2. **A hidden confounder collapses observational recovery** to 0.54–0.68 (strong: near chance) and degrades orientation from 0.91 to 0.62. **Interventions are almost unaffected** (0.975–0.997): the knockdown breaks the confounder's path into the perturbed gene, so the response of $j$ is a causal effect.
3. **Partial correlation** (graphical-model estimates of *direct* adjacency, undirected) scores about 0.70–0.72 against *total-effect* labels in every scenario: it answers a different question (direct adjacency) and cannot orient edges; its stability across confounding is not a success.
4. **More observational cells do not help.** In the strong-confounder setting, the AUROC of correlation, slope, and partial correlation at 500, 5,000, and 50,000 cells is 0.578, 0.582, 0.582 (correlation); 0.608, 0.609, 0.609 (slope): *a plateau set by confounding bias, not by sampling error.*
5. **Interventional power depends on cells per knockdown** (Chapter 25): with a strong confounder the interventional AUROC is 0.841 at 10 cells, 0.934 at 30, 0.975 at 100, 0.989 at 300, versus 0.617 for the observational slope on the same pairs.

```python
--8<-- "code/ch44_causal.py"
```

```text
== AUROC for finding pairs (i, j) such that knocking down gene i changes gene j (|total effect| > 0.15); 40 genes; mean of 8 random networks ==
scenario                                                     correlation   slope j~i   partial corr   slope (perturbed i only)   interventional (perturbed i only)   orientation correct
no hidden confounder, 5,000 cells                               0.934        0.963       0.704            0.968                      1.000                       0.906
hidden confounder (loading SD 0.7), 5,000 cells                 0.631        0.680       0.720            0.705                      0.997                       0.709
strong confounder (SD 1.5), 5,000 cells                         0.541        0.583       0.720            0.617                      0.975                       0.615

Interventional AUROC versus cells per knock-down (strong confounder, half of the genes perturbed):
cells per perturbation   interventional AUROC   observational slope AUROC on the same pairs
          10                 0.841                   0.617
          30                 0.934                   0.617
         100                 0.975                   0.617
         300                 0.989                   0.617

Observational sample size does not repair confounding (strong confounder):
observational cells   correlation   slope j~i   partial corr
         500             0.578        0.608       0.688
        5000             0.582        0.609       0.705
       50000             0.582        0.609       0.707
```

**What the simulation does not show.** It is linear, acyclic, with a single confounder and perfect knockdown; real regulation has feedback, nonlinearity, many confounders, off-target effects, and incomplete knockdown. The qualitative conclusions (confounding bias is independent of $n$; interventions identify what observations cannot; orientation requires intervention) should survive, but the *magnitudes* will not.

!!! rhyme "Structural rhyme: the confounded regression ↔ the collinear feature ↔ the unidentified edit"
    The bias $\lambda_i\lambda_j/\mathrm{Var}(x_i)$ here, the variance inflation factor $1/(1-\rho^2)$ of Chapter 31, and the weak-instrument bias of Chapter 26 are the same object: *two quantities that move together cannot be separated by observing them move together*. The remedy is the same in each case: **independent variation**, produced by an experiment, an instrument, or a natural randomization.

---

## 44.5 Interventions in biology

**Perturbation experiments.** CRISPR knockout, CRISPRi, and CRISPRa; Perturb-seq (Chapter 25); chemical perturbation (Tahoe-100M and sci-Plex); base and prime editing at specific variants; MPRA (random assignment of sequence variants to reporters). They are **interventions** on the system, and by randomizing (e.g., guide assignment to cells at random) they produce the ignorability that observational data lack.

**Efficiency and the instrumental view.** Not every cell that receives a guide has the intended knockdown (Chapter 25's effective fraction $q$). The *intention-to-treat* (ITT) effect, comparing all guide-carrying cells with controls, is $q$ times the effect among cells that were actually perturbed (the effect among "compliers"). The **IV estimate** $\hat\theta=\mathrm{ITT}_Y/\mathrm{ITT}_X$ (effect of guide on the outcome divided by effect of guide on target expression) estimates the *per-unit-knockdown* effect, **provided the guide affects the outcome only through the target** (exclusion restriction, violated by off-target effects, by effects of Cas9 binding on neighboring genes, or by guide-specific toxicity). This is the same Wald logic as Mendelian randomization (Chapter 26) with the guide as instrument and *measured target expression* as exposure. Cells with higher knockdown are *not* randomly assigned (dependent on cell state), so regressing the outcome on the measured knockdown level without the instrument reintroduces confounding.

**Natural experiments.** Genetic variation (MR), regulatory polymorphisms, copy-number variation, disease mutations, dietary and environmental shocks; also *sequence mutagenesis across species*, in the sense of evolution as an experiment (with phylogenetic confounding; Chapters 21, 42).

**Randomization within observational designs.** Randomizing wells or plates across conditions; hashing and multiplexing to place conditions in the same batch (Chapter 25); balanced designs that decorrelate batch and perturbation.

**What interventions do not give for free.** (i) *External validity*: an effect measured in one cell line at one time point may not **transport** to another context (the effect may depend on the cell state: *effect modification*); (ii) *interference and compensation*: knocking down one gene triggers compensation by paralogs or feedback, so the measured effect differs from the immediate causal effect; (iii) *dose and time*: knockdown level and measurement time determine what is seen (direct versus downstream effects); (iv) *combinations*: the effect of two perturbations is not determined by the single effects without further assumptions (Chapter 39).

---

## 44.6 Causal discovery from data

**Observational causal discovery** estimates the DAG from data under assumptions. **Constraint-based** methods (PC, FCI) use conditional independences; **score-based** methods (GES) optimize a penalized likelihood; **functional** methods (LiNGAM, additive noise models) exploit non-Gaussianity or nonlinearity to orient edges. Fundamental limits: (i) a DAG is identified only up to its **Markov equivalence class** from conditional independences (e.g., $A\to B$ and $B\to A$ are equivalent); (ii) *faithfulness* (no accidental cancellations) and *causal sufficiency* (no hidden confounders) are assumed, or one uses FCI-type methods that allow latent variables at the cost of weaker output; (iii) high dimensionality, nonlinearity, feedback, and noise degrade finite-sample performance. **Interventional data** break equivalence classes: a perturbation of $A$ orients the edges incident to $A$ (and, with several interventions, more), with specialized methods (GIES).

**In practice with single-cell perturbation data.** CausalBench (Chevalley et al.) assembled large-scale single-cell perturbation datasets (over 200,000 interventional samples) to benchmark network inference; a notable finding was that **methods that use interventional information did not outperform methods that use only observational data on real data**, contrary to synthetic benchmarks, and that poor scalability of many methods limits their performance [[S]]. Plausible reasons (hypotheses, [[P]]): the real regulatory network is not a sparse DAG; low reliability of single-cell measurements (Chapter 25); incomplete knockdown and compensation; a gold standard that is itself incomplete; and algorithms that do not exploit the interventions well. The lesson is not that interventions do not help (the simulation shows they do when the model is correct), but that **the gap between an idealized causal model and a cell is large**, and benchmarks on real interventional data are essential.

---

## 44.7 Counterfactual prediction, transportability, and causal representation learning

**Predicting the effect of an unseen perturbation** is an extrapolation in the space of interventions. It can be identified only with assumptions that link the unseen to the seen: *additivity* of effects (the additive baseline of Chapter 39), a *shared mechanism* (perturbations of genes in one complex have similar effects), *sparse mechanism shift* (each perturbation changes few causal mechanisms; the basis of causal representation learning for single cells, Lopez et al., 2022) [[S]], or *prior-knowledge graphs*. Without such assumptions, the effect of an unseen perturbation is unconstrained by the data (Chapter 31's non-identification, in perturbation space).

**Transportability.** An effect estimated in context $A$ applies in context $B$ only if the differences between the contexts do not modify the effect, or the modifiers are measured and adjusted for (Pearl & Bareinboim, 2014). In biology, effect modification by cell state, genotype, ancestry, and environment is the norm: *an intervention effect is a function of context*. Foundation models trained across contexts could in principle represent this dependence; whether they do is an empirical question (Chapters 38, 39, 45).

**Counterfactual (cell-level) prediction**, as in "what would this cell look like under the perturbation", requires modeling the cell-specific noise and the mechanism; methods such as CPA and scGen assume a latent disentanglement of the cell's basal state from the perturbation effect, which is **an assumption**, not an identified quantity from unpaired cells.

---

## 44.8 Implications for foundation models

1. **Pretraining on atlases is observational.** The representations encode association, including all the hidden confounders of §44.3. A model that predicts well on held-out cells from the same distribution has not been shown to know any intervention effect.
2. **Interventional data are the missing ingredient** for the causal claims (C4) that matter: perturbation atlases (Chapter 25) and edit-level data (Chapter 31). Their *design* determines what is identifiable (Chapter 46).
3. **Evaluation must be interventional.** Test on held-out *interventions* (new genes, new combinations, new contexts) with ceiling-normalized metrics (Chapters 25, 43), and compare with baselines that are causally naive (mean, additive).
4. **Zero-shot or in-context "reasoning" claims** about regulation from language models trained on text and sequences should be treated as hypotheses to be tested with experiments (Chapter 54).
5. **Causal assumptions should be stated** in every paper that interprets a model's output as a mechanism (Chapter 48's interpretability: a "circuit" is a causal claim about the model, not about the cell).

---

## 44.9 Worked research examples

!!! example "Worked Research Example 44.1: Master regulators from a large atlas"
    **Situation.** A group trains a foundation model on 20 million cells and ranks transcription factors by the effect of in-silico "knockout" (setting the TF's expression to zero in the input and measuring the change in predicted expression of other genes). They report the top 20 as "master regulators" of a differentiation program and validate by overlap with known regulators.

    **Question.** What does the ranking identify?

    **Reasoning.**

    1. *What is the in-silico knockout?* An intervention on the model's *input*, not on the cell: it asks the model to complete a profile in which that gene is zero. The model has only seen the *conditional* distributions of cells; it answers "what do cells with low TF look like", an **association** ($P(Y\mid X=0)$), not $P(Y\mid\mathrm{do}(X=0))$. Hidden cell state (a TF that marks the state the program belongs to) will drive the answer.
    2. *Why the overlap with known regulators is weak evidence.* Known regulators are *markers* of the states they define; any associative method will rank them highly. The relevant comparison is with a trivial baseline: rank TFs by correlation with the program score, or by variance. If it overlaps equally, the foundation model added nothing.
    3. *What would identify causal regulators?* Perturbation data: CRISPRi of the candidate TFs in the relevant cell type, with the readout of the program (Chapter 25). Use the model to **choose** which TFs to perturb (prioritization), then evaluate it against the experimental outcomes in the held-out set, with enrichment of strong effects in the model's top 20 versus the baseline's top 20 (C4).
    4. *If experiments are not possible.* Use natural or quasi-experiments (genetic variants affecting the TF's expression: MR; Chapter 26), time-course or lineage data (temporal precedence), and sensitivity analyses of unmeasured confounding.

    **Expert analysis.** The statement "the model identifies regulators" has two readings (associational and causal); the paper's evidence supports only the first. The best use of the model is as a **prioritization tool** whose output is *tested*.

!!! example "Worked Research Example 44.2: Incomplete knockdown and what a guide-level instrument identifies"
    **Situation.** In a CRISPRi screen with 150 cells per guide and three guides per gene, knockdown efficiency varies between guides (40–95%) and across cells. A lab ranks guides by the effect on a target program and reports the relation: stronger knockdown gives a larger program effect (dose–response).

    **Question.** What is identified?

    **Reasoning.**

    1. *Intention-to-treat per guide.* The mean program shift for cells with guide $g$ versus controls estimates $\mathrm{ITT}_g=q_g\Delta_g$ (effective fraction times effect), which is causal for "assigning guide $g$" (randomization), whatever the mechanism.
    2. *Dose–response across guides.* Using **guide as the instrument** for target expression: the per-guide ratio $\mathrm{ITT}_{Y,g}/\mathrm{ITT}_{X,g}$ is a Wald estimate for each guide; if guides act only through the target, the estimates are consistent across guides (a falsification check). Heterogeneity across guides indicates off-target effects or non-linearity.
    3. *Within-guide dose–response across cells is confounded.* The cells with the strongest knockdown within a guide may also be in a different cell-cycle state; regressing on measured knockdown within a guide mixes the causal effect with this selection.
    4. *Design improvements.* Tilings of guides with graded efficiency; titrated dCas9-KRAB expression; rescue experiments (re-expression) as a control; separate measurement of target expression in the same cells.

    **Expert analysis.** The problem is the **same** as Mendelian randomization with weak instruments (Chapter 26) and as Chapter 25's power analysis (dilution by $q$); it is solved by the same tools (instrument-based ratios, falsification across instruments).

---

## 44.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: a causal audit of a modeling claim"
    1. **State the rung** of the claim: association, intervention, or counterfactual.
    2. **Draw the DAG** for the data-generating process, including *hidden* cell state, batch, donor, and selection. Mark which variables were measured and adjusted for.
    3. **Identification.** Is the target quantity identified from the data under stated assumptions (backdoor set, instrument, randomization, front-door)? If not, what assumption would identify it, and is it testable?
    4. **Colliders.** Does the analysis condition on a post-treatment variable, a selected subset, or a significance filter?
    5. **Falsification checks.** Negative controls (outcomes that cannot be affected), placebo interventions, multiple instruments agreeing, effects that should vanish under permutation.
    6. **Sensitivity.** How strong would an unmeasured confounder have to be to explain the result (E-values; simulation as in §44.4)?
    7. **Transport.** In which contexts is the claim intended to hold, and what evidence is there that the effect does not depend on context?
    8. **Remedy.** Which intervention or natural experiment would convert the association into an intervention estimate, and what is its cost (Chapter 46)?

    **What it teaches.** The causal audit is the same exercise as the baseline ladder (Chapter 29), applied to meaning rather than accuracy.

    **An open question to carry forward.** Could a foundation model pretrained on observational atlases *plus* a modest number of perturbations produce *identified* counterfactual predictions, if the perturbations are chosen to span the confounding directions? Formalize "span the confounding directions" using the identifiability argument of Chapter 31 and the simulation of §44.4 (how many perturbed genes and cells per perturbation are needed as a function of the confounder's rank and strength?), and test whether the optimal design differs from random perturbation selection.

---

## 44.11 Connections

- **Backward:** confounding and pseudoreplication (Chapters 4, 25); instruments and Mendelian randomization (Chapter 26); counterfactual identifiability in sequence models (Chapter 31); the evaluation of perturbation predictors (Chapters 25, 39, 43).
- **Forward:** distribution shift and transportability (Chapter 45); experimental design to maximize identifiability (Chapter 46); interpretability as a causal analysis of the model (Chapter 48); open problems in cells and virtual cells (Chapter 51); AI scientists proposing and testing hypotheses (Chapter 54).

!!! takeaways "Key takeaways"
    1. Association, intervention, and counterfactual are different questions; **accuracy at the first does not imply the second**.
    2. The **do-operator** cuts arrows into the intervened variable; adjustment (backdoor) requires *measured* confounders; **instruments** identify effects despite hidden confounding.
    3. A hidden confounder biases a regression slope by $\lambda_i\lambda_j/\mathrm{Var}(x_i)$, **independent of $n$**; more observational data does not repair it (AUROC plateau 0.58 from 500 to 50,000 cells).
    4. In simulation with a strong hidden confounder, observational AUROC for causal pairs was 0.54–0.62 while knockdown experiments gave 0.975 (100 cells per knockdown), 0.84 at 10 cells, 0.99 at 300; orientation from observations was only 0.62.
    5. **Perturbation guides are instruments**; incomplete knockdown dilutes the ITT effect and requires IV logic; off-target effects violate exclusion.
    6. **Causal discovery from observations** is limited to Markov equivalence classes and assumes sufficiency and faithfulness; on real single-cell perturbation data, interventional methods did **not** outperform observational ones in CausalBench, so real-data benchmarks are essential.
    7. **Counterfactual and unseen-perturbation prediction** require assumptions (additivity, shared mechanism, sparse mechanism shift); effects are context-dependent, so transportability must be argued.
    8. Models pretrained on atlases encode **association**; their in-silico knockouts are conditional queries, useful for **prioritizing** experiments, not for stating mechanism.

---

## Further reading

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press. Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If.* Chapman & Hall/CRC. Peters, J., Janzing, D. & Schölkopf, B. (2017). *Elements of Causal Inference.* MIT Press. Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *J. Educ. Psychol.* 66, 688–701.
- Angrist, J. D., Imbens, G. W. & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *J. Am. Stat. Assoc.* 91, 444–455. Pearl, J. & Bareinboim, E. (2014). External validity: from do-calculus to transportability across populations. *Stat. Sci.* 29, 579–595. VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Ann. Intern. Med.* 167, 268–274.
- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search.* MIT Press. Shimizu, S. et al. (2006). A linear non-Gaussian acyclic model for causal discovery. *J. Mach. Learn. Res.* 7, 2003–2030. Hauser, A. & Bühlmann, P. (2012). Characterization and greedy learning of interventional Markov equivalence classes of directed acyclic graphs. *J. Mach. Learn. Res.* 13, 2409–2464.
- Chevalley, M., Roohani, Y., Mehrjou, A., Leskovec, J. & Schwab, P. (2025). A large-scale benchmark for network inference from single-cell perturbation data. *Commun. Biol.* (2025), s42003-025-07764-y. Lopez, R. et al. (2022). Learning causal representations of single cells via sparse mechanism shift modeling. *arXiv:2211.03553.* Lotfollahi, M., Wolf, F. A. & Theis, F. J. (2019). scGen predicts single-cell perturbation responses. *Nat. Methods* 16, 715–721. Dixit, A. et al. (2016). Perturb-Seq. *Cell* 167, 1853–1866. Replogle, J. M. et al. (2022). Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575.
