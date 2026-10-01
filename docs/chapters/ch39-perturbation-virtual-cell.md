# Chapter 39. Perturbation Response and the Virtual Cell

!!! abstract "Chapter at a glance"
    **Motivation.** The most ambitious target in AI for biology is a *virtual cell*: a model that predicts how a cell's state changes when a gene is knocked down, a drug is added, or two genes are perturbed together, in a cell type for which it has seen few or no perturbations. If it worked, it would turn the most expensive step in biology (an experiment per hypothesis) into a computation. Since 2023 a dozen foundation-model-scale systems have claimed progress; in 2025 a careful comparison found that none of five foundation models and two other deep models beat simple additive or mean baselines for double-perturbation prediction, and a community competition showed both real advances and the difficulty of choosing metrics that reward them. This chapter formalizes the prediction problem as interventions on a causal system (Chapter 44), builds a controlled simulation in which the generating network is known and the baselines can be placed exactly, and shows what each baseline, metric, and inductive bias does: a raw-expression correlation above 0.97 for a model that predicts *no effect at all*; a knowledge-graph model whose accuracy scales with the recall of its graph; a generic embedding regression that is worse than predicting nothing; and an additive model that is close to optimal because interactions are small relative to measurement noise. It closes with the data resources, the 2025–26 evidence, and the open problems.
    **Prerequisites.** Chapters 8, 16, 25, 30, 31, 38, 43–46.
    **You will be able to:** (1) pose perturbation prediction as an intervention problem and name the generalization tasks (unseen perturbation, unseen context, combination, dose, time); (2) choose and interpret baselines and metrics, including noise ceilings and systematic variation; (3) explain why a learned map from gene embeddings to expression changes needs an inductive bias that ties embedding coordinates to outputs; (4) judge whether an interaction is predictable from the data design; (5) read the evidence on foundation models and the Virtual Cell Challenge with grades; (6) design a prospective test of a virtual-cell model for target discovery.

---

## 39.0 What "virtual cell" means, and a ladder of claims

"Virtual cell" is used for four different ambitions, and most confusion is about which one is meant (Bunne et al., *Cell* 2024, give the program):

| Rung | Claim | Test |
|---|---|---|
| **V1: a cell-state map** | A pretrained model embeds cells so that types and states are separable | Label transfer, clustering (Chapter 38) |
| **V2: perturbation-response prediction** | Predict the expression change after an *unseen* perturbation, context, or combination | Held-out perturbations against baselines with ceilings (this chapter) |
| **V3: simulation of dynamics** | Predict trajectories (time courses, doses, differentiation) | Held-out time points; interventions mid-trajectory (Chapter 19) |
| **V4: a mechanistic cell** | A model whose structure corresponds to the cell's mechanisms and generalizes to conditions it was not fit to | Prospective interventions; causal tests (Chapters 44, 48) |

Everything published as of October 2026 sits at V1–V2, with early V3 work. The scientific value of V2 depends on *what is unseen*, which gives the generalization tasks of this chapter: (T1) an unseen perturbation in a seen cell type; (T2) a seen perturbation in an unseen cell type or context; (T3) an unseen combination of seen perturbations; (T4) an unseen dose or time; (T5) all of these together. Almost all of the evidence in the literature is on T1 and T3 within one or two cell lines.

!!! lens "Research lens: three views of the same data"
    *As a regression problem*: learn $f:(\text{context},\text{perturbation})\to\Delta x$ with $\Delta x$ the change in mean expression; *as a distribution problem*: learn the conditional distribution $p(x\mid c,\pi)$ over cells, since perturbation changes composition and variance as well as the mean; *as an intervention problem*: the perturbation $\pi$ is a $\text{do}$ operation in a structural causal model of the gene network, and generalization to a new perturbation is *extrapolation in the interventional distribution* that is identifiable only under structural assumptions (Chapter 44). The first treats the problem as pattern matching, the second as a conditional generative model (Chapters 14, 15), the third as causal inference. They give different baselines and different failure modes; a model that works in one view can fail in the others.

---

## 39.1 Biology for modeling: perturbations and what is measured

!!! bio "Biology for modeling: the perturbation experiment"
    **What is it?** A *perturbation* is an intervention on a component: CRISPR knockout (Cas9), CRISPR interference (CRISPRi; dCas9-KRAB represses a gene without cutting; knockdown typically 70–95% and variable by guide), activation (CRISPRa), base editing, a small molecule or cytokine, or a combination. **How is it measured?** *Perturb-seq* and its relatives (Dixit et al. 2016; Adamson et al. 2016; CROP-seq, Datlinger et al. 2017) deliver guide RNAs in a pooled library, read the guide identity and the transcriptome of each single cell, and thereby label each cell with its perturbation; *optical pooled screens* read in situ phenotypes; proteomic and chromatin readouts exist at lower throughput. **How is it generated?** The effect is a *cascade*: the targeted gene's product falls; direct targets change within hours; secondary and tertiary effects accumulate for days; cell-cycle, stress, and fitness responses follow, and the *timing of the readout* selects which stage is observed. **What information does it contain?** The causal effect of the perturbation on the other genes, in the contexts sampled, entangled with *guide efficacy*, *incomplete knockdown*, *selection* (cells with fitness-reducing perturbations drop out), and *confounding by cell state* at the time of perturbation. **What variation exists?** Between cell types (the same perturbation has different effects in different contexts: Chapter 45's concept shift), between donors, across doses. **What can ML not observe?** Perturbations that were not in the library; effects below noise; the cells that died; the unperturbed counterfactual of the *same* cell (we compare populations).

**Data resources (as of October 2026).** Genome-scale Perturb-seq in K562 and RPE1 cells (Replogle et al., *Cell* 2022; more than 2.5 million cells; CRISPRi); combinatorial CRISPRa perturbations (Norman et al., *Science* 2019; 100 single and 131 double perturbations, the standard data set for double-perturbation models); X-Atlas/Orion (Xaira Therapeutics, 2025; about 8 million cells with genome-wide perturbation in HCT116 and HEK293T cells); Tahoe-100M (Tahoe Therapeutics, 2025; more than 100 million cells exposed to more than a thousand small-molecule conditions across 50 cancer cell lines); and harmonized collections (scPerturb, Peidli et al. 2024). The scale of observational atlases (hundreds of millions of cells) far exceeds interventional data in most contexts, which is why *what a model can learn from observation about intervention* is the central question.

---

## 39.2 The prediction problem, and what determines whether it is solvable

Write the (noise-free) steady-state expression of $N$ genes as the solution of a nonlinear system $x=\phi(Wx+b)$ with network $W$ and basal drive $b$. A *clamp* of gene $g$ (a knockdown) replaces the $g$-th equation by $x_g=c$. Then:

- **First-order (direct) effects.** To first order the change of gene $j$ is $\Delta x_j\approx \phi'_j\,W_{jg}\,\Delta x_g$: it depends on the *edge* $W_{jg}$ and the local gain $\phi'_j$. A model that knows the (signed) edges out of $g$ can predict direct effects.
- **Cascades.** Higher-order terms propagate through paths of length $\ge2$: $\Delta x\approx(\,I-D W)^{-1}D\,W_{\cdot g}\Delta x_g$ with $D=\text{diag}(\phi')$ in the linear regime. Predicting these requires the *network*, not only the target's neighbors.
- **Non-additivity (interactions).** For a pair $(g_1,g_2)$, $\Delta x^{(g_1,g_2)}-\Delta x^{(g_1)}-\Delta x^{(g_2)}\ne0$ arises from saturation (both act on the same downstream gene beyond its dynamic range), from one gene acting through the other, and from the cell's compensating (buffering) responses. Interactions are *rare and structured*: concentrated in pairs that share targets or sit in the same pathway.
- **Identifiability.** The effect of an unseen perturbation on $N$ genes cannot be determined from training perturbations of other genes *unless something ties the two together*: gene function, network structure, or shared pathways (the embedding or knowledge graph of the model). A model that treats gene identity as an arbitrary label has no basis for extrapolating to a new gene.

!!! rhyme "Structural rhyme: unseen perturbation ↔ new protein family in Chapter 34 ↔ unseen cell type in Chapter 31"
    In each, generalization to a new *entity* requires a representation in which similar entities have similar effects; the benefit of pretraining is bounded by what the new entity *shares* with the old ones (Chapter 34's experiment), and no amount of data on other entities helps when nothing is shared.

---

## 39.3 Model families, with their assumptions

| Family | Representatives | Idea | Information used | Typical failure |
|---|---|---|---|---|
| **No-change / mean-shift / additive baselines** | — | Predict control; the average effect of training perturbations; or the sum of single effects | Training perturbations only | Cannot predict a perturbation's specific targets (mean shift); cannot predict interactions (additive) |
| **Latent-space arithmetic** | scGen (Lotfollahi et al. 2019), CPA (Lotfollahi et al. 2023) | Perturbations are vectors in an autoencoder's latent space, with covariates | Seen perturbations and contexts | Unseen perturbations have no vector; no interactions |
| **Knowledge-graph GNNs** | GEARS (Roohani et al., *Nat. Biotechnol.* 2024) | Propagate gene-embedding information over a gene-gene graph (co-expression, GO) to predict a new gene's effects | A graph; training perturbations | Depends on graph quality; graphs are curated for well-studied genes |
| **Single-cell foundation models, fine-tuned or with in-silico perturbation** | Geneformer (Theodoris et al., *Nature* 2023), scGPT (Cui et al., *Nat. Methods* 2024), scFoundation (Hao et al. 2024) | Pretrain on atlases; perturb by changing gene tokens or fine-tune a head | Atlas-scale observational data | Observation does not identify intervention (Chapter 44); zero-shot embeddings lose to simple baselines (Chapter 38) |
| **Generative distribution models** | Optimal transport (CellOT, Bunne et al. 2023), diffusion and flow-matching models, State Transition (Arc, 2025) | Model the distribution shift as transport or a conditional generator, predict heterogeneity | Perturbed and control populations | Distributional metrics are noisy; weak identification for unseen contexts |
| **Mechanistic and dynamical models** | SCMs, ODE/SDE with RNA velocity, SERGIO-like simulators | A structured causal model fit to interventional data | Interventions; structure priors | Scale; unidentifiable structure without enough interventions |
| **Large cross-context models** | State (Adduri et al., bioRxiv 2025), large perturbation models | Train on very large pooled perturbation data from many contexts and shared embeddings | Interventions across contexts | Generalization beyond the training contexts is the open question |

---

## 39.4 A controlled experiment: baselines, metrics, and inductive bias

The simulated world uses the steady-state model of §39.2 with $N=100$ genes, a sparse signed network (8% density, spectral radius 0.9), saturating $\phi=\tanh$, and a strong knockdown clamp. Each perturbation is measured as the mean over 400 cells with per-gene noise SD $1/\sqrt{400}=0.05$. Models are trained on 70 single perturbations and tested on 30 *unseen* singles and on 150 doubles of seen genes (10 independent networks). *Knowledge* enters through gene embeddings: each true edge is visible with probability $\rho$ (the "recall" of a knowledge graph or a pretrained embedding), and false edges are added at 1%. Four kinds of prediction are compared: no change; the mean effect of training perturbations; a **generic embedding regression** (ridge from the 200-dimensional embedding of a gene to its 100-dimensional response, the architecture of a standard prediction head); and a **graph-propagation** model whose structure is tied to the visible network (first- and second-hop propagation with two scalars fit on the training perturbations). The *noise ceiling* is the agreement between two independent measurements of the same perturbation, and the *first-order oracle* uses the true edges for direct effects only.

```python
--8<-- "code/ch39_perturbation.py"
```

```text
== Perturbation response prediction in a simulated network: 100 genes, 400 cells per perturbation (measurement SD 0.05 per gene), 10 networks; 70 training singles; 30 UNSEEN singles ==

unseen single perturbations: correlation of predicted and observed CHANGE across genes (delta-correlation); RMSE as a fraction of the no-change baseline's RMSE (1.0 = no better than predicting 'no effect'); and the correlation on RAW expression
method                       delta-correlation   RMSE / baseline RMSE   raw-expression correlation
noise ceiling                     0.828                0.532                   0.995
mean shift                        0.058                1.063                   0.970
oracle first-order                0.713                0.760                   0.984
generic embedding regression, rho=0.0       -0.006                1.018                   0.970
generic embedding regression, rho=1.0        0.123                1.631                   0.947
graph propagation, rho=0.3        0.285                0.964                   0.973
graph propagation, rho=0.6        0.434                0.918                   0.976
graph propagation, rho=1.0        0.581                0.849                   0.981

delta-correlation normalised by the noise ceiling: mean shift 0.07, oracle first-order 0.86, generic embedding regression, rho=0.0 -0.01, generic embedding regression, rho=1.0 0.15, graph propagation, rho=0.3 0.34, graph propagation, rho=0.6 0.52, graph propagation, rho=1.0 0.70

double perturbations of seen genes (150 pairs): RMSE of the no-change baseline 0.218; RMSE of the additive prediction from measured singles 0.093; additive prediction from noise-free singles 0.060
RMS size of the true interaction (non-additivity) 0.027; measurement noise per gene 0.050  -> the interaction is smaller than the noise
interaction model (ridge on products of gene embeddings, 100 training doubles): correlation between predicted and true interaction vectors on 50 held-out doubles
   embedding recall rho = 0.0:  -0.001
   embedding recall rho = 0.6:  0.037
   embedding recall rho = 1.0:  0.067
```

**Reading the results.**

1. **The raw-expression correlation is a trap.** Every method, including "mean shift" and a model that predicts *no effect*, has a correlation with raw expression of 0.97 or higher, because the baseline expression of a gene (its level in control cells) dominates the variance across genes. The *delta-correlation* (between predicted and observed *changes*) separates them: 0.06 for mean shift; 0.83 for the noise ceiling. **Always evaluate the change from control**, never the expression level, or the evaluation measures the cell's identity rather than the perturbation's effect.
2. **Use ceilings and baselines together.** The noise ceiling is 0.83 (not 1.0), and the first-order oracle (the true direct edges, no cascades) reaches 0.71, i.e., 86% of the ceiling. A reported delta-correlation of 0.4 means little until compared with these.
3. **Inductive bias, not capacity, carries the prediction.** The graph-propagation model's delta-correlation rises with the *recall of its graph*: 0.29 ($\rho=0.3$), 0.43 (0.6), 0.58 (1.0), i.e., 0.34, 0.52 and 0.70 of the ceiling. The *generic embedding regression*, which has the same information at $\rho=1$ (the true edges are in its input), reaches only 0.12 and has an RMSE 1.63 times *larger* than predicting no effect: with 70 training perturbations it cannot learn that coordinate $j$ of the embedding corresponds to output gene $j$. The structure that makes an embedding useful (that the $j$th entry of a gene's edge vector is the effect on gene $j$) must be built into the architecture or supplied by enough data. This is why graph-structured and gene-token-structured models can work where a generic head does not, and why a *foundation model's gene embeddings help only if the downstream head respects gene identity*.
4. **Additive prediction is near-optimal for doubles in this world.** For 150 doubles of seen genes the RMSE of the no-change baseline is 0.22 and of the additive prediction from *measured* singles is 0.093; the additive prediction from noise-free singles reaches 0.060, close to the measurement noise (0.05). The true interaction has an RMS of only 0.027 per gene, smaller than the noise. The expected additive error from noise alone (two noisy singles and a noisy double: $\sqrt3\times0.05=0.087$) nearly equals the observed 0.093: **no model can improve on the additive baseline here, because the interactions are below the noise.** A model that beat the additive baseline on this metric would be overfitting noise.
5. **Interactions are not learnable from this design.** A ridge model on products of gene embeddings, trained on 100 doubles, reaches a correlation with the true interaction vectors of 0.07 at $\rho=1$ (0.04 at 0.6, 0.00 at 0). With interactions that are sparse and small, the training set for an interaction model must be enriched for *interacting* pairs (Chapter 46: choose double perturbations to measure by the information they carry about interactions, e.g., pairs of genes that share neighbors) rather than random pairs.

!!! lens "Research lens: assumptions of the experiment"
    A known steady-state network, a single cell type, noise-free knowledge graphs apart from recall and 1% false edges (real graphs are biased toward well-studied genes and are *unsigned*), measurement noise that is Gaussian and independent across genes (real data have shared technical and cell-state structure), and interactions that arise only from saturation. Real interactions include buffering, synthetic lethality, and epistasis between pathways that this world lacks. The qualitative results (metrics and baselines decide what you conclude; inductive bias ties embeddings to outputs; additive is near-optimal when interactions are below noise) generalize; the numbers do not.

---

## 39.5 Evaluation: the pitfalls, in a checklist

- **Baselines**: no-change, mean-of-training-perturbations (captures *systematic variation* between perturbed and control cells that is shared across perturbations, such as a general stress response), additive for combinations, and a *similar-perturbation* lookup (the nearest training perturbation in a functional embedding) for unseen ones. Viñas Torné et al. (2025) show that much of the apparent skill of published models comes from systematic variation.
- **Metrics**: delta-correlation per perturbation; error on the *top differentially expressed genes* (where signal is) and on all genes (where noise is); direction-of-change accuracy; distributional distances (energy distance, MMD) for heterogeneous effects; rank of the correct perturbation among candidates (retrieval); and *ceiling-normalized* versions of each.
- **Splits**: split by *perturbation* (and by pathway or complex, since perturbing two members of a complex gives near-identical effects, a form of leakage), by *context*, and by *combination*; report per-class performance; never split cells at random (Chapters 7 and 43).
- **Power and noise**: most perturbations have small or no effect on most genes; per-perturbation correlations are dominated by a minority of strong perturbations; report distributions, not just means.
- **Technical replicates**: guide-level and batch-level replicates define the ceiling; unsupervised *effect-size* filters (a minimum number of differentially expressed genes) select the perturbations on which any method is testable.

---

## 39.6 The evidence as of October 2026

!!! paper "Paper dissection: 'Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines' (Ahlmann-Eltze, Huber & Anders; *Nature Methods* 22, 2025)"
    **Problem.** Do foundation models and other deep models predict the effect of *combinations* of perturbations better than trivial baselines?
    **Insight.** Compare against baselines that *cannot* learn interactions or gene-specific knowledge: the additive prediction (sum of single effects) and a mean/no-change prediction, on the standard double-perturbation data (Norman et al. 2019) and genome-scale single-perturbation data.
    **Evaluation.** Five foundation models (including scGPT and scFoundation) and two other deep-learning models (such as GEARS and CPA), predicting held-out double perturbations and unseen single perturbations; error on expression changes; analysis of the ability to find genuine genetic interactions.
    **Result (as reported).** None of the deep models outperformed the simple baselines for double perturbations, and for unseen single perturbations a linear model with embeddings extracted from the data performed as well as the deep models; genetic interactions predicted by the deep models were not better than a trivial guess.
    **Why.** As in §39.4: when interactions are small relative to noise, additive is near-optimal; and unseen-perturbation prediction depends on an inductive bias the pretrained embeddings did not supply.
    **Limitations.** A few data sets, a particular set of metrics; a follow-up preprint (October 2025) argues that with well-calibrated metrics the deep models do beat uninformative baselines, and that the apparent parity is partly a consequence of metrics that reward the mean effect. The *disagreement is about the metric*, which supports §39.5 [[S]] for "no consistent advantage on standard metrics in 2025", [[H]] for "no advantage under calibrated metrics".

!!! paper "Paper dissection: The Virtual Cell Challenge (Arc Institute, 2025) and State (Adduri et al., bioRxiv June 2025)"
    **Problem.** Create an open, blind benchmark for predicting perturbation responses in a new cell type with a small amount of context-specific data.
    **Design.** Arc released a perturbation data set in H1 human embryonic stem cells and ran a competition with a held-out set and a blind final evaluation and two grand prizes of $100,000. Reported participation: more than 5,000 registrants from 114 countries and more than 1,200 teams submitting predictions, with more than 300 in the final evaluation [[S]].
    **Results (as reported).** The winning entries combined deep learning with classical statistical features; a generative flow-matching model from Altos Labs won the "generalist" prize; organizers and participants noted that models did not consistently beat naive baselines on all metrics, and that metric choice strongly affected rankings.
    **State (Arc's model).** Two modules: *State Embedding*, a cell-set encoder pretrained on observational cells (reported 167 million cells), and *State Transition*, a model that predicts perturbation effects on *sets* of cells in a context, trained on more than 100 million perturbed cells across 70 contexts (as reported).
    **How to read it.** The competition design (blind test, a common metric suite, prospective data) is as important as any result: it creates exactly the benchmark that was missing, and it exposes how sensitive rankings are to metrics. The claims for State are developer-reported on benchmarks of its own choosing [[P]] until independent replications on held-out contexts exist.

| Evidence | Finding | Grade |
|---|---|---|
| Ahlmann-Eltze et al. 2025 | Deep models did not beat additive/mean baselines on standard double-perturbation metrics | [[S]] |
| Follow-up (2025 preprint) | Under calibrated metrics deep models beat uninformative baselines | [[P]] |
| Virtual Cell Challenge 2025 | Hybrids of deep learning and classical features win; no model consistently beats naive baselines on all metrics | [[S]] |
| Zero-shot single-cell FMs (Kedzierska et al. 2025; Chapter 38) | Embeddings lose to simple baselines (HVG, scVI, Harmony) on several tasks | [[S]] |
| Transfer across cell types | Some transfer for core cellular machinery; context-specific regulators need target data | [[P]] |
| Scaling of perturbation data (X-Atlas/Orion, Tahoe-100M) | Larger perturbation data sets now exist; scaling laws for perturbation prediction are not established | [[H]] |

!!! openproblem "Open problem: when does observation predict intervention?"
    A virtual cell trained on observational atlases must predict interventions, which observation does not identify without assumptions (Chapter 44). **Diagnostic questions:** Which classes of perturbations are predictable from observation alone (for example, gene knockdowns whose effect is a *shared program* that varies naturally across cells, like cell-cycle or stress)? Could one estimate the *interventional identifiability* of a perturbation from its natural variation (is there a natural experiment, such as eQTLs or stochastic expression noise, that varies the target independently of confounders)? What is the *minimum interventional data* per context (in number of perturbations and cells) that turns an atlas-pretrained model into a reliable predictor, and is that number a function of the context's similarity to the training contexts (a scaling law for transfer in the sense of Chapter 47)?

---

## 39.7 Worked research examples

!!! example "Worked Research Example 39.1: Auditing a claim of double-perturbation prediction"
    **Situation.** A paper reports that its new model predicts expression after double CRISPRa perturbations with Pearson $r=0.93$ against the observed profile, "outperforming all baselines", and proposes it for choosing combination therapies.

    **Question.** What would you check, and what would the claim need to be to deserve the word "outperforming"?

    **Reasoning (Expert Chain).**

    1. **L2 What is the metric?** If $r$ is computed on raw expression across genes, a no-change prediction would also give about 0.97 (§39.4, point 1). Ask for the delta-correlation (change from control), the top-DE-gene error, and the error normalized by the noise ceiling from replicate guides.
    2. **L3 Which baselines?** Additive from measured singles (the baseline of Ahlmann-Eltze et al.), mean shift, and a nearest-neighbor perturbation lookup. A model that does not beat additive on delta-metrics is not predicting interactions.
    3. **L4 Where is the split?** If doubles of *seen* genes are in the test set but the genes' singles are in training, additive is a strong baseline; if the test doubles include unseen genes, additive cannot be computed, and the relevant baseline is mean/nearest-neighbor. Check for *complex leakage* (two members of one complex across the split).
    4. **L5 Are there interactions to predict?** Compute the fraction of test pairs whose measured interaction exceeds the noise (from replicates); if it is 5%, the aggregate metric cannot distinguish models on interactions, and the claim of interaction prediction needs a *targeted* test: precision and recall of the top-ranked interacting pairs against the experimentally defined set.
    5. **L10 Experiments to request.** A prospective test: freeze the model, choose 100 new pairs by a pre-specified rule (including pairs with the highest predicted interaction and a random sample), measure them, and report enrichment of true interactions in the top predictions against the additive-plus-noise null.
    6. **L11 Interpretation.** If it passes, claim "predicts the expression of combinations with accuracy $x$ of the noise ceiling and ranks interactions with enrichment $y$"; for combination *therapy*, a further step is needed: mapping expression to a clinically relevant phenotype and testing it in a disease model.

    **Expert analysis.** $r=0.93$ is the least informative number in the abstract. The decision-relevant quantities are the delta-metrics against baselines and the prospective interaction enrichment.

!!! example "Worked Research Example 39.2: A prospective test of a virtual cell for target discovery, with no known answer"
    **Situation.** A group proposes to use a virtual cell model to nominate genes whose knockdown shifts a disease-relevant transcriptional program in primary macrophages, saving a genome-scale screen that costs about two million dollars. No one knows the model's accuracy on primary macrophages.

    **Question.** How would you evaluate the model's usefulness before committing, and what would you do with the answer?

    **Reasoning.**

    1. **Define usefulness.** The decision is a *ranking* of 20,000 genes into a shortlist of 200; success is the fraction of true hits in the shortlist compared with random (about 1–2%) and with simple baselines (a knowledge-graph neighbor lookup of known pathway members; the top genes by expression in the program).
    2. **Measure a pilot with a design built to test the model.** Perturb 400 genes (CRISPRi in primary macrophages from 3 donors; at least 50 cells per guide; replicates at donor level): 200 chosen by the model's rank (top 100 and 100 sampled from the middle to estimate the full ranking's calibration), 100 chosen by the baseline, 100 random. Pre-specify the hit definition (the program's score moving by at least 2 standard deviations of control, with FDR control across genes).
    3. **Analysis.** Report the precision of each arm with binomial intervals; the area under the ROC for the model's ranks over the 400 measured genes; and the calibration of predicted effect sizes against measured ones. Normalize by the noise ceiling (replicate agreement across donors).
    4. **Power.** With a hit rate of 5% in the random arm and a hoped-for 25% in the model's top 100, the expected difference is detectable with 100 per arm (about 95% power at $\alpha=0.05$); a 12% hit rate would not be, which should be stated before the experiment.
    5. **Decision rule.** If the model's top-100 precision is at least twice the best baseline's with a lower confidence bound above it, use the model to prioritize a screen of 3,000 genes (a 7-fold saving); if not, use the pilot data to *train* a context-specific model (Chapter 46) and treat the exercise as having bought a data set rather than a prediction.
    6. **Interpretation.** A positive result is a statement about *this* context and program (C2); a claim of general utility needs several contexts (C3). A negative result is informative about the transfer problem of §39.2 and should be reported with its denominators.

    **What is not known.** The accuracy of any current model in primary macrophages; the pilot is the experiment that measures it, and it costs about 5% of the screen it would justify.

---

## 39.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: from a perturbation data set to an honest model comparison"
    1. **State the task** (T1–T5 of §39.0) and make the split match it, by perturbation, pathway/complex, context, or combination.
    2. **Always include** no-change, mean-of-training-perturbations, additive (for combinations), and nearest-neighbor baselines.
    3. **Compute delta-metrics** and ceilings from technical replicates; report ceiling-normalized scores.
    4. **Filter by effect size** (perturbations with detectable effect) and report performance by effect-size stratum.
    5. **Test interactions separately**: how many exceed noise? Rank-based precision for top-predicted interactions.
    6. **Check inductive bias** by giving the model a *shuffled* gene embedding or knowledge graph; if performance is unchanged, the knowledge is not used.
    7. **Re-run the experiment of §39.4** at several noise levels and interaction strengths and find where each method stops being distinguishable from the baseline.
    8. **Design the next data set** by information (Chapter 46), not by what is convenient to collect.

    **What it teaches.** Most of the apparent progress and most of the apparent failure in this field is a property of metrics and baselines; deciding them *before* looking at results is the practice.

    **An open question to carry forward.** The experiment shows accuracy tied to the recall of the graph. Real knowledge graphs have *non-random* recall: well-studied genes have rich neighborhoods and understudied genes have none, which are precisely the genes on which a discovery tool is needed. Could one *quantify the bias of predictive accuracy against understudied genes* by measuring performance as a function of the number of publications or annotations of the target gene, and design perturbation experiments to fill the gap (a data collection strategy that targets the *dark genome*)?

---

## 39.9 Connections

- **Backward:** latent-variable models and identifiability (Chapter 8); geometric deep learning and graphs (Chapter 16); the measurement model of perturbation screens (Chapter 25); batch and composition effects (Chapter 30); counterfactuals in sequence models (Chapter 31); single-cell foundation models (Chapter 38); benchmarks, causal inference, and shift (Chapters 43–45); experimental design (Chapter 46).
- **Forward:** multimodal and multi-scale models (Chapter 40); open problems of cells (Chapter 51); mechanistic interpretability of cell models (Chapter 48); AI scientists proposing perturbations (Chapter 54).

!!! takeaways "Key takeaways"
    1. Perturbation prediction is an *intervention* problem; observational pretraining does not identify interventions without structure (Chapter 44), and generalization to an unseen perturbation needs a representation that ties the perturbation to its effects.
    2. **Metrics decide conclusions**: in a controlled simulation, raw-expression correlation was at least 0.97 for every method, including "no effect"; the delta-correlation spanned 0.06 (mean shift) to 0.83 (noise ceiling). Evaluate changes from control, and normalize by a ceiling.
    3. **Inductive bias carries the prediction**: a graph-propagation model's accuracy rose with graph recall (0.29, 0.43, 0.58 of delta-correlation at $\rho=0.3,0.6,1.0$), whereas a generic embedding regression with the *same* information reached 0.12 and an error 1.63 times worse than predicting nothing.
    4. **Additive is near-optimal when interactions are below noise**: the additive double-perturbation error (0.093) matched the noise-limited bound (0.087); a model that beats it on aggregate metrics is probably fitting noise, and interactions need *targeted* measurement.
    5. In 2025, five foundation models and two other deep models did not beat additive or mean baselines on standard double-perturbation metrics (Ahlmann-Eltze et al.); follow-up work argues calibrated metrics reverse part of that, so the disagreement is about metrics [[S]]/[[H]].
    6. The Virtual Cell Challenge (2025) produced the first blind, community benchmark: more than 1,200 submitting teams, winners that combined deep learning with classical statistical features, and no model consistently beating naive baselines on all metrics.
    7. Judge a virtual-cell claim by its split (what is unseen?), its baselines, its ceiling, its metric on *changes*, and a prospective test whose denominator is reported.

---

## Further reading

- Dixit, A. et al. (2016). Perturb-seq: dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens. *Cell* 167, 1853–1866. Adamson, B. et al. (2016). A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response. *Cell* 167, 1867–1882. Replogle, J. M. et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575. Norman, T. M. et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* 365, 786–793.
- Lotfollahi, M., Wolf, F. A. & Theis, F. J. (2019). scGen predicts single-cell perturbation responses. *Nat. Methods* 16, 715–721. Lotfollahi, M. et al. (2023). Predicting cellular responses to complex perturbations in high-throughput screens. *Mol. Syst. Biol.* 19, e11517 (CPA). Roohani, Y., Huang, K. & Leskovec, J. (2024). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nat. Biotechnol.* 42, 927–935. Bunne, C. et al. (2023). Learning single-cell perturbation responses using neural optimal transport. *Nat. Methods* 20, 1759–1768.
- Cui, H. et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nat. Methods* 21, 1470–1480. Theodoris, C. V. et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. Hao, M. et al. (2024). Large-scale foundation model on single-cell transcriptomics. *Nat. Methods* 21, 1481–1491.
- Ahlmann-Eltze, C., Huber, W. & Anders, S. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat. Methods* 22, 1657–1661. Viñas Torné, R. et al. (2025). Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation (preprint). Wenteler, A. et al. (2024). PertEval-scFM: benchmarking single-cell foundation models for perturbation effect prediction (preprint). Kedzierska, K. Z. et al. (2025). Zero-shot evaluation reveals limitations of single-cell foundation models. *Genome Biol.*
- Bunne, C. et al. (2024). How to build the virtual cell with artificial intelligence: priorities and opportunities. *Cell* 187, 7045–7063. Adduri, A. K. et al. (2025). Predicting cellular responses to perturbation across diverse contexts with State (preprint). Peidli, S. et al. (2024). scPerturb: harmonized single-cell perturbation data. *Nat. Methods* 21, 531–540. Chevalley, M. et al. (2025). A large-scale benchmark for network inference from single-cell perturbation data. *Commun. Biol.* (2025), s42003-025-07764-y.
