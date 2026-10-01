# Chapter 58. Reasoning Without Answers: Five Open-Ended Case Studies

!!! abstract "Chapter at a glance"
    **Motivation.** The rest of this book gives you tools; this chapter practices their use on questions whose answers nobody knows. Each case is worked in full along the twelve links of the Expert Chain (L1–L12): the problem and its measurement, the existing approaches and what they assume, why they might work and how they fail, the bottleneck, the open question, ranked hypotheses, candidate solutions, experiments with predicted outcomes, interpretation tables, and new directions. The cases are chosen to cover the field: why zero-shot cell foundation models lose to PCA (Chapter 38), why genomic deep learning fails across individuals (Chapters 31, 32), whether scale buys design success (Chapter 36), how to evaluate an "AI-discovered" drug (Chapter 37), and whether a connectome predicts behavior (Chapter 53). They use the quantitative results of earlier chapters as pilot data. None has a known answer; each ends with a decision rule fixed before the experiment, a statement of what would change your mind, and the strongest claim the evidence could support.
    **Prerequisites.** Chapters 1, 43, 45, 46, 55, 56 (and the chapter each case cites).
    **You will be able to:** (1) carry a vague open question through the Expert Chain to a ranked set of hypotheses and a pre-registered experiment; (2) compute and use the expected information gain of candidate experiments; (3) write a decision rule with explicit stopping and success criteria; (4) distinguish what a pilot can establish from what only the full experiment can; (5) state, for each result, the highest rung of the claim ladder it supports.

---

## 58.0 How to read this chapter

Each case has the same skeleton, so that you can reuse it.

| Link | What you write | Guard against |
|---|---|---|
| L1 Problem | The question as a measurable quantity | A "why" without an outcome measure |
| L2 Existing approaches | The main families, with evidence grades | Treating one paper as the field |
| L3 Assumptions | What each approach takes for granted | Unstated invariances |
| L4 Why it might work | The mechanism that would make it succeed | Magical thinking about scale |
| L5 Failure modes | Specific, testable ways it fails | Vague "generalization gap" |
| L6 Bottleneck | The binding constraint (data, objective, identifiability, evaluation) | Optimizing a non-bottleneck |
| L7 Open question | The sharpest unanswered question | A question that is a bundle |
| L8 Hypotheses | 3 to 5 mutually distinguishable explanations, with priors | Hypotheses that predict the same data |
| L9 Candidate solutions | What you would build if each hypothesis were true | Building before diagnosing |
| L10 Experiments | Designs that separate the hypotheses, with predicted outcomes | Experiments whose outcome cannot change the decision |
| L11 Interpretation | A table mapping each possible outcome to a conclusion | Post-hoc interpretation |
| L12 New directions | What the answer opens | A conclusion that stops |

**Priors and information gain.** In each case the hypotheses $H_1,\dots,H_K$ carry subjective prior probabilities that you write down, and the *expected information gain* of an experiment with possible outcomes $o$ is $\mathrm{EIG}=H(\mathbf p)-\sum_o P(o)\,H(\mathbf p\mid o)$ (Chapter 56). The numbers in the cases are illustrative priors, not facts; the point is the *procedure* of choosing the experiment that separates the hypotheses most per unit cost.

!!! lens "Research lens: what makes a case 'open'"
    A question is open when (i) the measurement exists, (ii) several explanations survive the available data, and (iii) an experiment could separate them. If (iii) fails, the question is not yet scientific; if (i) fails, the first project is the measurement.

---

## 58.1 Case 1: Why do zero-shot single-cell foundation models lose to PCA?

!!! example "Worked Research Example 58.1: Single-cell foundation models and the simple baseline"
    **Situation.** Zero-shot embeddings from several single-cell foundation models give worse cell-type separation and batch mixing than PCA on highly variable genes, scVI, or Harmony on several benchmarks (Kedzierska et al. 2025; Chapter 38). In a miniature simulation (Chapter 38), a small rank-encoding transformer pretrained on 19,800 cells by masked-gene prediction did not match PCA either (label transfer 0.49 against 0.94–0.96), but the same model with a contrastive depth-invariance term added reached 0.94: the objective, not the data size (0.51 to 0.49 for ten times more cells), made the difference, in one seed of a favorable simulation. Teams propose bigger models and more data. You are asked what to do.

    **L1 Problem.** *Measure*: label-transfer accuracy, novel-type detection, batch mixing of a zero-shot embedding relative to the best simple baseline, on held-out studies. *Question*: which factor limits the foundation model: (i) its pretraining data and scale, (ii) its objective, (iii) its input encoding, (iv) how it is evaluated?

    **L2 Existing approaches.** Rank-based encodings (Geneformer-style), expression-value tokens (scGPT, scFoundation), cell-set encoders (State Embedding), VAEs (scVI) as baselines [[S]] for the failure of zero-shot embeddings on several tasks, [[P]] for any one explanation.

    **L3 Assumptions.** Pretraining on many cells teaches a gene-gene regulatory structure that makes cell embeddings more useful than the data's own principal components; masked-gene prediction is a good objective; tokenization by rank or value preserves biology; benchmark tasks measure what matters.

    **L4 Why it might work.** Atlases contain the variation across tissues and conditions, which a single data set lacks; a model might learn gene programs and transfer them to a new study with batch shifts.

    **L5 Failure modes.** (a) *The objective is easy*: masked-gene prediction can be solved from highly expressed housekeeping and co-expression structure without learning cell identity. (b) *Tokenization loses information*: ranking the top 2,048 genes discards expression magnitudes and low-expressed markers. (c) *Batch and depth shortcuts*: the embedding encodes sequencing depth or study. (d) *Evaluation is insensitive*: if PCA already saturates the benchmark, no model can exceed it. (e) *Scale is too small or data too redundant*: effective diversity of an atlas is far below its cell count (Chapter 42's $N_\text{eff}$).

    **L6 Bottleneck.** Distinguishing (a)–(e) requires controlled comparisons that vary one factor at a time; published benchmarks vary architecture, data, and objective together.

    **L7 Open question.** *At fixed architecture and compute, does pretraining on more diverse cells improve zero-shot embeddings, and does a better objective change the answer?*

    **L8 Hypotheses (priors).**

    | | Hypothesis | Prior |
    |---|---|---|
    | H1 | Scale and diversity are insufficient but would suffice (the learning curve is simply early) | 0.15 |
    | H2 | The masked-gene objective is mismatched: it does not reward cell-identity features | 0.30 |
    | H3 | Tokenization (rank or truncation) discards the information that separates types | 0.20 |
    | H4 | Benchmarks are saturated by simple baselines; the gap cannot be measured | 0.25 |
    | H5 | Pretraining acquires shortcuts (depth, batch) that harm transfer | 0.10 |

    **L9 Candidate solutions by hypothesis.** H1: more cells, more studies. H2: contrastive or cell-level objectives (invariance to depth and batch), supervised pretraining on type labels. H3: continuous-value tokenization, full-gene input. H4: harder benchmarks (rare types, small effect sizes, perturbation-response tasks). H5: adversarial depth/batch removal; augmentations.

    **L10 Experiments.** *Factorial pretraining study* with a fixed small transformer (so that results are cheap and the comparisons controlled):

    - Factor A, data: 2k, 20k, 200k cells; diverse (many studies) versus redundant (few studies, same cell count).
    - Factor B, objective: masked-gene prediction; masked-gene plus a contrastive term (two augmentations of a cell, with different depths and dropout, should embed together); supervised cell-type pretraining (as an upper bound on what labels can add).
    - Factor C, tokenization: rank top-96 (as in Chapter 38); value-binned tokens of the top 512 genes; all genes via a linear embedding.
    - Evaluation: label transfer to a held-out study, novel-type detection, within-type study separation, and *a harder task* (rare-subtype separation with 1% abundance), each normalized by PCA on 300 HVGs and by a noise ceiling (replicate agreement).
    - Predicted outcomes: if H1, accuracy increases log-linearly with data and the diverse arm beats the redundant one at equal cells; if H2, the contrastive term improves transfer at every data size more than a 10× increase in data does; if H3, value tokens improve it; if H4, PCA reaches the ceiling and no condition differs; if H5, adding adversarial depth removal improves it.

    **L11 Interpretation table.**

    | Outcome | Conclusion | Rung |
    |---|---|---|
    | Diverse data scale gains, objective and tokens do not | H1: scale; fund more data | C2 in this setting |
    | Contrastive objective gives a larger gain than a 10× data increase | H2: change the objective | C2 |
    | Tokenization change dominates | H3 | C2 |
    | All conditions at PCA's level with the ceiling reached | H4: the benchmark cannot discriminate; build harder tasks | C1 |
    | Gains from depth-adversarial training | H5 | C2 |
    | Mixed: no single factor, interactions | Report the interactions; no single-factor claim | C1 |

    **Pilot evidence.** The simulation above updates the prior toward H2 (objective) before any real data are used, but it is a single seed in a world designed with a nuisance (depth) that the augmentation targets; the factorial study below is what would test H2 on real atlases.

    **Expected information gain.** The five-way prior has entropy 2.2 bits. A single yes/no experiment that tests H2 with 80% reliability earns only about 0.24 bits; the factorial design, whose outcomes can separate all five, can earn up to 2.2 bits (realistically 1–1.5 bits, because the predictions of neighboring hypotheses overlap) for weeks of work, whereas "train a 10× bigger model and see" earns little, since most hypotheses predict that it will not help. Cheap controlled experiments beat expensive uncontrolled ones.

    **L12 New directions.** If H2 wins, a family of cell-level pretraining objectives (contrastive, invariance to technical variation, multi-study alignment) becomes the research agenda; if H4 wins, benchmark design (rare states, perturbation response, ceilings) becomes the bottleneck of the field.

    **Decision rule (pre-registered).** Adopt the factor whose gain, relative to the PCA baseline normalized by the ceiling, exceeds 0.1 on the harder rare-subtype task, replicated in three data seeds with non-overlapping 95% intervals. If no factor does, conclude H4 or a combination, and do not scale. **What would change your mind:** a replicated scale effect at 200k cells (H1), or a perturbation-response task in which the pretrained embeddings beat PCA at equal labels.

---

## 58.2 Case 2: Why do genomic deep-learning models fail to predict expression differences between individuals?

!!! example "Worked Research Example 58.2: Across-individual prediction"
    **Situation.** Sequence-to-function models explain between-gene expression variance but have near-zero across-individual correlations per gene, sometimes with the wrong sign for a variant's effect (Huang et al. 2023; Sasse et al. 2023; Chapter 31). Candidate explanations multiply.

    **L1 Problem.** *Measure*: for each gene, the Spearman correlation between predicted and observed expression across individuals (and the sign accuracy for the strongest cis-eQTL), against a ceiling set by the heritability of expression of that gene.

    **L2 Existing approaches.** Enformer, Borzoi, AlphaGenome-type supervised models; per-gene linear regression on nearby variants; fine-mapped-variant priors (Chapters 31, 41).

    **L3 Assumptions.** A model that predicts expression from reference sequence across genes will also predict the effect of small sequence differences within a gene; training on reference-genome tracks provides variation-response information.

    **L4 Why it might work.** If the model learned regulatory logic (motif strengths, spacing), small edits should change its predictions in the right direction.

    **L5 Failure modes.** (a) *Training never varies sequence within a gene*: the objective rewards across-gene discrimination; the within-gene gradient is not constrained (Chapter 31's identifiability). (b) *Eroded signal*: cis-eQTL effects are small relative to noise; heritable fraction is low for most genes (ceiling). (c) *Distal context ignored* (Karollus et al. 2023). (d) *Label noise and LD*: eQTL labels are tags, not causal variants (Chapter 26). (e) *Cell-type mismatch*: bulk tissue versus the model's tracks.

    **L6 Bottleneck.** Evaluation power: per-gene correlations over a few hundred individuals have a standard error near 0.05–0.07 and heritability caps the achievable value; many genes cannot show anything.

    **L7 Open question.** *Which of (a)–(e) accounts for the gap between across-gene accuracy and across-individual accuracy, and what training data would close it?*

    **L8 Hypotheses (priors).**

    | | Hypothesis | Prior |
    |---|---|---|
    | H1 | Within-gene variation is not in the training signal (objective gap) | 0.35 |
    | H2 | The effects are below the ceiling for most genes (measurement gap) | 0.25 |
    | H3 | Distal elements are not used (architecture/context) | 0.20 |
    | H4 | LD-tag labels hide the causal variants (inference gap) | 0.15 |
    | H5 | Cell-type mismatch | 0.05 |

    **L9 Candidate solutions.** H1: train with *paired* data (individual genomes plus expression, or MPRA saturation mutagenesis) and a within-gene loss; H2: restrict evaluation to genes with high cis-heritability and report ceiling-normalized scores; H3: long-context models with distal perturbation data (CRISPRi screens); H4: fine-map first, evaluate on causal variants; H5: matched cell types.

    **L10 Experiments.**

    1. *Ceiling-normalized re-evaluation*: restrict to the top-quintile cis-heritable genes and report the correlation divided by $\sqrt{h^2_\text{cis}}$ (the best achievable); predicts: if H2, ceiling-normalized correlations of the models are at least 0.5 for the highly heritable genes and approach those of the per-gene regression; if H1, they stay near zero even there.
    2. *MPRA test (counterfactual)*: for 2,000 regulatory elements with saturation mutagenesis in the matched cell type, compare the model's predicted effect of every single-nucleotide change with the measured effect (Chapter 31). If the correlation is high here but low across individuals, H1/H2 are separated from the model's within-sequence logic; if it is also low, H1 (the model has not learned within-element logic) is strongly supported.
    3. *Fine-tuning ablation*: fine-tune the model with a within-gene objective on a training split of individuals (hold out genes and individuals); compare to the supervised per-gene regression: predicts that H1 gives a large gain when the model has the paired training signal.
    4. *Distal perturbation test*: use CRISPRi enhancer-gene links (known distal effects) to test whether the model's predictions respond to deletion of the distal element (H3).
    5. *Fine-mapped variants*: evaluate sign accuracy only on variants with posterior inclusion probability above 0.9 (H4).

    **L11 Interpretation.**

    | Result | Conclusion |
    |---|---|
    | Normalized correlation high for heritable genes; MPRA correlation high | H2: measurement-limited; the model is better than it looks |
    | MPRA correlation high, fine-tuning gives no gain, across-individual low | Labels/LD (H4) or cell type (H5) |
    | MPRA correlation low; fine-tuning gives a large gain | H1: objective gap; paired training data close it |
    | Model ignores distal deletion; CRISPRi-linked enhancers matter | H3 |
    | Several effects | Report each contribution as a fraction of the gap, with intervals |

    **L12 New directions.** A benchmark of *within-locus* prediction with ceilings, composed of MPRA saturation mutagenesis, CRISPRi screens, and paired-genome expression, becomes the standard for regulatory models (Chapter 50).

    **Decision rule.** If experiment 1 explains at least half of the gap, publish the corrected evaluation and a ceiling-normalized leaderboard; otherwise run experiments 2–3 before any model change. **What would change your mind:** a long-context model that, without paired data, shows a within-gene correlation at the ceiling on matched cell types.

---

## 58.3 Case 3: Does scale buy design success?

!!! example "Worked Research Example 58.3: Scaling and the hit rate of designed binders"
    **Situation.** Reported hit rates of de novo binder design have risen from a few percent (2020) to tens of percent on favorable targets (Chapter 36). Companies argue that larger models and more data will push hit rates toward 100% for any target; skeptics argue that target difficulty, not model scale, dominates. There is no accepted evidence.

    **L1 Problem.** *Measure*: per-target experimental hit rate (fraction of tested designs with $K_D$ below a fixed threshold), as a function of model scale, training data, and target difficulty.

    **L2 Existing approaches.** RFdiffusion-family, BindCraft, AlphaProteo, Chai-2; hallucination and generative models; selection by structure-predictor confidence (§36.1).

    **L3 Assumptions.** Hit rate is determined by how well the model captures interface physics and by how well its filters predict binding.

    **L4 Why it might work.** Larger co-folding and diffusion models that have seen more complexes interpolate better in the space of interfaces; better filters reject more non-binders.

    **L5 Failure modes.** Target difficulty (polar, flat, glycosylated surfaces), memorization (§35.5): hit rates reflect similarity of targets to the training complexes; filter exploitation (§36.4); selection bias in reported targets.

    **L6 Bottleneck.** There is no *controlled* measurement of scale at fixed targets and protocol: companies compare their own systems to old baselines on their own target sets.

    **L7 Open question.** *At fixed targets and assay, how does hit rate depend on model scale and training data, and does the dependence vanish for targets dissimilar from training complexes?*

    **L8 Hypotheses.**

    | | Hypothesis | Prior |
    |---|---|---|
    | H1 | Hit rate improves smoothly with scale for all targets | 0.15 |
    | H2 | Improvement is concentrated in targets similar to training complexes (memorization/interpolation) | 0.40 |
    | H3 | Gains come from better filters and selection protocols, not generators | 0.25 |
    | H4 | Target difficulty (epitope properties) explains most of the variance; scale matters little | 0.20 |

    **L9 Candidate solutions.** H1: scale; H2: data diversity and physics-informed training; H3: invest in filters and in active learning; H4: target-specific strategies.

    **L10 Experiments.** A *scaling campaign* with a public protocol: 40 targets stratified by (a) similarity of the target-binder interface motif to the PDB (training analogs: high versus none) and (b) epitope hydrophobicity/flatness; 3 model scales of one open architecture (e.g., an open co-folding/diffusion family trained at 3 sizes and 2 data volumes); for each target and model, 24 designs chosen by a *fixed* filter, plus 24 chosen at random from the model's samples (to separate generator from filter); expression and single-concentration binding screen followed by $K_D$ for hits; one lab, one assay, blinded to the model. Cost: 40 × 3 × 48 ≈ 5,800 designs, within the reach of a pooled-oligo, yeast-display pipeline.

    - Predicted pattern if H2: slope of hit rate versus scale is positive for targets with analogs and flat for those without; if H3: the random-selection arm shows a flat slope while the filter-selected arm rises; if H4: target random effects explain most of the variance and the scale slope is near zero; if H1: parallel positive slopes.
    - Analysis: a mixed-effects logistic model, $\mathrm{logit}\,p=\beta_0+\beta_1\log(\text{scale})+\beta_2\,\text{analog}+\beta_3\,\log(\text{scale})\times\text{analog}+\beta_4\,\text{filter}+u_\text{target}$.

    **L11 Interpretation.**

    | Result | Conclusion |
    |---|---|
    | $\beta_3>0$ large; no slope without analogs | H2: scale helps interpolation, not extrapolation |
    | $\beta_1>0$ for the random arm | H1: generators improve |
    | $\beta_1\approx0$ random, $>0$ filtered | H3: filters |
    | Variance dominated by $u_\text{target}$ | H4 |
    | $\beta_1$ positive for both groups | scale helps even without analogs (a genuinely surprising result; check for hidden analogs) |

    **L12 New directions.** A public *design leaderboard* of per-target hit rates stratified by novelty; calibrated predictors of target difficulty; targeted data collection (interfaces without analogs) to break the memorization limit.

    **Decision rule.** Pre-specify that the scaling claim is accepted if $\beta_1$ for non-analog targets has a lower 95% bound above 0 and the corresponding hit-rate gain per doubling exceeds 3 percentage points. **What would change your mind:** a model that matches the large model's hit rate with 10× less data on non-analog targets (evidence for objective rather than scale).

---

## 58.4 Case 4: Is an AI-discovered drug evidence for AI?

!!! example "Worked Research Example 58.4: The evidential value of one AI-discovered clinical success"
    **Situation.** An AI-discovered small molecule (rentosertib, from Insilico Medicine; target and molecule both nominated by AI) reported a randomized placebo-controlled Phase IIa in idiopathic pulmonary fibrosis (71 patients, 22 sites in China, 12 weeks): mean forced vital capacity change of +98.4 mL at the highest dose against −20.3 mL for placebo (as reported), and a Phase III trial in China was announced in July 2026. An analysis of AI-discovered drugs through 2023 found Phase I success of 80–90% (21 completed trials) and Phase II success of about 40% (10 trials), the latter in line with historical averages (Jayatunga et al. 2024). A commentator concludes that "AI has been validated"; another that "nothing has changed".

    **L1 Problem.** *Measure*: the probability that a drug candidate nominated through an AI-enabled process reaches approval, compared with a matched non-AI process, and the cost and time to each stage. *Question*: what evidence would show that AI changes this probability?

    **L2 Existing approaches.** Retrospective success-rate analyses by developer-classified status (Jayatunga et al.); single-program case reports; company disclosures.

    **L3 Assumptions.** "AI-discovered" is a classification of a program's origin; success rates for classes of drugs are comparable across periods and targets.

    **L4 Why AI might help.** Faster target nomination and molecule optimization (30 months from target to Phase I in one report), better properties (Phase I success is largely safety and pharmacokinetics, which better-optimized molecules satisfy).

    **L5 Failure modes.** (a) *Definition and selection*: companies label programs; successes are announced and failures quietly dropped (survivorship); (b) *small samples*: 10 Phase II trials give a 95% interval for a 40% success rate of roughly 12% to 74%; (c) *target novelty*: a novel target has lower prior success; (d) *surrogate endpoints*: a 12-week FVC change in 71 patients is an early signal of efficacy, not approval; (e) *regional and population differences*; (f) *confounding by period*: modern trial design and biomarker selection improve success for all drugs.

    **L6 Bottleneck.** There is no counterfactual: the same program would not have been run without AI, so the comparison group is historical and heterogeneous.

    **L7 Open question.** *Does AI-enabled discovery increase the probability of Phase II and Phase III success per program, and reduce time and cost, after accounting for target novelty and period?*

    **L8 Hypotheses.**

    | | Hypothesis | Prior |
    |---|---|---|
    | H1 | AI raises Phase I success (better properties) and leaves efficacy attrition unchanged | 0.50 |
    | H2 | AI raises success at all stages | 0.10 |
    | H3 | AI mainly reduces time and cost, not probabilities | 0.30 |
    | H4 | No effect beyond selection and period | 0.10 |

    **L9 Candidate solutions.** Evidence: (i) a registry of *all* AI-labeled programs at the time of initiation (not afterwards), with pre-specified definitions; (ii) matched comparators (same target class, indication, sponsor type, year); (iii) program-level cost and time accounting; (iv) for individual programs, mechanism-confirming trial designs.

    **L10 Experiments.** *Observational*: a pre-registered cohort analysis of programs initiated 2018–2022 and followed to 2027, with success by stage modeled by a Bayesian hierarchical model including target novelty, indication, and year. Predictions: H1 gives a Phase I odds ratio above 2 with Phase II odds ratio near 1. *Interventional (within a sponsor)*: randomly assign which of several candidate targets of a portfolio are pursued by an AI-enabled versus a conventional discovery route (a portfolio-level randomized study, ethically unproblematic because both routes go through the same preclinical gates); compare the proportion reaching Phase II with a pre-specified endpoint. *Single program*: for the rentosertib mechanism (TNIK inhibition), the evidence that the target is causal comes from the Phase III results and from independent target-validation studies, not from the AI process.

    **L11 Interpretation.**

    | Outcome | Conclusion |
    |---|---|
    | Phase I odds ratio $>2$; Phase II/III not different | H1: AI helps molecular quality, not target biology |
    | Improvement at all stages after adjustment | H2 |
    | No difference in probabilities, lower cost and time | H3 (still economically important) |
    | No differences after adjustment | H4 |
    | One Phase III success | An existence proof of a successful program (C2), not an estimate of the effect of AI |

    **L12 New directions.** Reporting standards for AI-discovery claims (what the AI contributed: target, molecule, trial design); prospective registries; target-validation experiments that separate target biology from molecule quality.

    **Decision rule.** Treat "AI improves clinical success" as supported only if a pre-registered, adjusted comparison shows a Phase II odds ratio whose 95% interval excludes 1. **What would change your mind:** a portfolio-randomized comparison.

    **Strongest claim today.** A drug nominated by AI has shown early efficacy in a small randomized trial (C2 for that molecule and indication); the evidence for AI's effect on clinical success rates is [[P]] at Phase I, [[H]] beyond.

---

## 58.5 Case 5: Does a connectome predict behavior?

!!! example "Worked Research Example 58.5: From a wiring diagram to a behaving fly"
    **Situation.** The adult fly connectome (139,255 neurons, over 50 million synapses) and a whole-brain integrate-and-fire model built from it (Shiu et al. 2024) predicted sensorimotor circuit responses without fitting. A project proposes to *predict the behavior of a fly* from the connectome and a handful of recordings (Chapter 53).

    **L1 Problem.** *Measure*: given a stimulus (a taste, a mechanical touch, a visual looming stimulus), predict the behavioral response (a proboscis extension, a grooming bout, an escape) and the activity of identified neurons; score against held-out flies and held-out perturbations (silencing a cell type).

    **L2 Existing approaches.** Counts-only whole-brain LIF models; connectome-constrained models with trained parameters (Lappalainen et al. 2024); data-driven population models without connectomes; digital twins (Chapter 53).

    **L3 Assumptions.** Synapse counts approximate synaptic strengths up to a global scale; neurotransmitter identity (predicted from images) sets the sign; neuromodulation and intrinsic properties contribute little at the timescale of interest; one connectome represents all flies.

    **L4 Why it might work.** Wiring is highly stereotyped across individuals for many cell types, and the first-order structure of sensorimotor circuits may dominate (as in the successful feeding and grooming predictions).

    **L5 Failure modes.** Strength heterogeneity (our simulation: counts-only predicted held-out responses at $r\approx0.92$–$0.97$ but silencing effects at only 0.54–0.60 for log-normal heterogeneity $\sigma=1$); neuromodulation and state dependence (hunger, arousal); individual variation; incomplete recordings; plasticity.

    **L6 Bottleneck.** Parameters that the connectome does not supply (strengths, intrinsic parameters, neuromodulation), and the *perturbation tests* that identify them.

    **L7 Open question.** *Which behaviors can be predicted from the connectome and a small number of functional measurements, and how many measurements per cell-type pair does it take?*

    **L8 Hypotheses.**

    | | Hypothesis | Prior |
    |---|---|---|
    | H1 | Counts and signs suffice for first-order sensorimotor circuits | 0.25 |
    | H2 | The connectome as a mask plus functional data (tens of recordings) suffices | 0.40 |
    | H3 | State-dependent neuromodulation is needed for most behaviors | 0.25 |
    | H4 | Individual variation limits any model from one connectome | 0.10 |

    **L9 Candidate solutions.** H1: counts-only LIF; H2: mask-and-fit with data (§53.5's model B); H3: add neuromodulator-state variables measured by imaging; H4: per-individual connectome + function in the same fly (the MICrONS design, for the fly).

    **L10 Experiments.**

    1. *Ablation-first benchmark*: for 20 cell types with genetic driver lines, silence each (optogenetic or genetic) and record the behavioral change and downstream neural responses; test the model's predictions of the sign and rank of effects for held-out cell types (the causal test of §53.5).
    2. *Data-efficiency curve*: fit model B with 10, 30, 100, 300 stimulus conditions and plot ablation-prediction accuracy versus the number of conditions; predicted (from the simulation of §53.5): the ablation accuracy of the mask-and-fit model is already high with fewer stimulus conditions than neurons, whereas an unconstrained model needs 10 to 20 times more.
    3. *State manipulation*: repeat with flies fed and starved; measure whether a state variable fit on one state predicts the other (H3).
    4. *Individual variation*: record two flies' responses to the same stimuli and compare the between-fly variability to the model–fly discrepancy (H4): if the model's error is within the between-fly variability, no model of one connectome can do better.

    **L11 Interpretation.**

    | Result | Conclusion |
    |---|---|
    | Counts-only matches ablation within between-fly variability | H1 |
    | Mask-and-fit matches; counts-only does not | H2: the connectome plus a modest amount of function |
    | Model fits fed state, fails starved | H3 |
    | Between-fly variability as large as model error | H4: the benchmark ceiling is reached |

    **L12 New directions.** A registered, sealed *perturbation benchmark in the fly* (silencing and activation experiments with predictions submitted before measurement) analogous to CASP; neuromodulatory connectomes.

    **Decision rule.** Claim "the connectome predicts behavior X" only if predictions of the *sign of the silencing effect* for held-out cell types reach at least 80% accuracy with a binomial lower bound above chance, after normalizing by between-fly variability. **What would change your mind:** a model that predicts silencing effects at chance despite excellent response prediction (the pattern of model A in the simulation).

---

## 58.6 A method for making your own cases

1. **Start from a number in a paper that bothers you** (a gap, a failure, an unexplained ceiling) and write it as a measurable $y$.
2. **List every explanation you can think of**, then merge those that predict the same data; keep at most five.
3. **Write priors** (they can be wrong; they are the record of what you believed).
4. **For each cheap experiment**, tabulate possible outcomes, the posterior each implies, and the expected information gain; run the one with the best gain per cost.
5. **Write the interpretation table before the experiment**, and the decision rule with a stopping criterion.
6. **After the experiment, update the priors and write what changed your mind.**
7. **Archive** the case with its code and data: the habit of this chapter is the habit of a research notebook (Chapter 59).

!!! notebook "Researcher's Notebook: three cases to work yourself"
    1. *Why do protein language models' zero-shot scores correlate better with some deep mutational scans than others?* Use §34.4 and the toy of §34.2.
    2. *Does an atlas-pretrained model predict perturbations in a cell type absent from the atlas?* Use §39.4 and the context-transfer problem.
    3. *Do genomic language models learn anything about regulatory grammar beyond $k$-mer statistics?* Use §32.2 and §48.4.

    **What it teaches.** Open questions become tractable when they are decomposed into hypotheses that predict different experimental outcomes. The skill is the decomposition.

    **An open question to carry forward.** Could the Expert Chain itself be evaluated? For a set of historical open problems (for example, before CASP14, "can deep learning predict structure?"), would the procedure of this chapter have ranked the eventually successful approach among the top hypotheses, and what is the base rate at which a well-formed chain's top-ranked hypothesis proves correct? A retrospective study of this kind would calibrate the priors that this chapter asks you to write.

---

## 58.7 Connections

- **Backward:** the Expert Chain (Chapter 1); benchmarks and baselines (Chapter 43); shift (Chapter 45); experimental design (Chapter 46); idea generation (Chapter 55); evaluating ideas and information gain (Chapter 56); and Chapters 31, 32, 36–38, 53 for the content of the cases.
- **Forward:** from idea to publication (Chapter 59).

!!! takeaways "Key takeaways"
    1. An open question becomes a research project when it is written as a measurable outcome, decomposed into hypotheses that predict different data, and attached to an experiment whose outcomes map to conclusions in a table fixed in advance.
    2. **Case 1** (cell foundation models vs PCA): a factorial pretraining study at small scale (data diversity × objective × tokenization, evaluated on a hard rare-subtype task normalized by PCA and a ceiling) separates five explanations at about 1–2 bits of information for weeks of work.
    3. **Case 2** (across-individual prediction): ceiling-normalized evaluation, MPRA saturation mutagenesis, and a within-gene fine-tuning ablation separate the measurement, objective, context, and label explanations.
    4. **Case 3** (scale and design): a mixed-effects analysis of per-target hit rates stratified by training analogs, with generator and filter separated by a random-selection arm.
    5. **Case 4** (AI-discovered drugs): a single success is an existence proof (C2); evidence on the effect of AI on success probabilities requires pre-registered cohorts with adjustment, and ideally portfolio-level randomization; Phase I success of 80–90% and Phase II of about 40% (n=10) are compatible with an effect on molecular quality only.
    6. **Case 5** (connectome to behavior): predictions must be tested on perturbations (the sign of silencing effects for held-out cell types) with between-animal variability as the ceiling.
    7. Every case ends with a decision rule, a statement of what would change your mind, and the strongest claim currently supported.

---

## Further reading

- Platt, J. R. (1964). Strong inference. *Science* 146, 347–353. Chamberlin, T. C. (1890/1965). The method of multiple working hypotheses. *Science* 148, 754–759. Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Ann. Math. Stat.* 27, 986–1005. Kass, R. E. & Raftery, A. E. (1995). Bayes factors. *J. Am. Stat. Assoc.* 90, 773–795.
- Kedzierska, K. Z., Crawford, L., Amini, A. P. & Lu, A. X. (2025). Zero-shot evaluation reveals limitations of single-cell foundation models. *Genome Biol.* Huang, A. C. et al. (2023). *Nat. Genet.* 55, 2056–2059. Sasse, A. et al. (2023). *Nat. Genet.* 55, 2060–2064. Karollus, A., Mauermeier, T. & Gagneur, J. (2023). *Genome Biol.* 24, 56.
- Jayatunga, M. K. P., Ayers, M., Bruens, L., Jayanth, D. & Meier, C. (2024). How successful are AI-discovered drugs in clinical trials? A first analysis and emerging lessons. *Drug Discov. Today* 29, 104009. Xu, Z. et al. (2025). A generative AI-discovered TNIK inhibitor for idiopathic pulmonary fibrosis: a randomized phase 2a trial. *Nat. Med.* (2025) (rentosertib, phase 2a).
- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634. Shiu, P. K. et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634, 210–219. Lappalainen, J. K. et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140.
