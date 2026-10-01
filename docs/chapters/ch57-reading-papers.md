# Chapter 57. Reading Papers Like a Researcher

!!! abstract "Chapter at a glance"
    **Motivation.** A paper is a compressed argument that was written to persuade. In AI for biology, a plausible-looking figure can sit on top of a leaky split, a weak baseline, a saturated benchmark, or an interpretation that the data cannot bear. This chapter gives a *protocol*: three passes of increasing depth, a claim–evidence table, a catalog of red flags with the quick test for each, and a fixed dissection template. It then applies the protocol to two real papers and shows how reading feeds idea generation (Chapter 55).
    **Prerequisites.** Chapters 1, 24, 25, 26, 27, 28, 29, 31 (the failure modes catalogued here were each demonstrated in those chapters), 55–56.
    **You will be able to:** (1) classify a paper's headline claim on the Claim Ladder and find the evidence that supports each rung; (2) reconstruct the objective, the information flow, and the evaluation split of a paper from its methods; (3) run the red-flag tests quickly; (4) dissect a paper with the fixed template; (5) extract assumptions and candidate attacks from any paper; (6) maintain a reading log that turns reading into a research asset.

---

## 57.1 The goal of reading

You read to answer four questions, in order: **(1) What is claimed? (2) What would have to be true for the claim to hold? (3) Does the evidence show it? (4) What does the paper make possible or leave open?** The first two take minutes; the third takes hours; the fourth is where research ideas come from.

Reading is also *costly*: a graduate student reading ten papers a week for a year reads about 500, yet the number of papers worth a three-hour dissection is perhaps 30–50 a year. The protocol below allocates effort accordingly: *every* paper gets a ten-minute first pass; *a few* get a full second pass; *a handful* get a third (reproduction).

---

## 57.2 The three-pass protocol

**Pass 1 (10–15 minutes): the claim.** Read the title, abstract, introduction's last paragraph, the figures (look at axes, error bars, what is compared to what), the conclusion. Write:

* the *headline claim* in one sentence;
* its **Claim-Ladder rung** (C0–C4; Chapter 1): is it a fit to training data, a held-out prediction, a prediction under shift, a mechanism claim, or an intervention claim?
* the **dataset(s)** and the **comparison** (to what baseline, on what metric);
* your *prior*: before looking at the evidence, how likely is the claim?

Decide: stop, continue to pass 2, or put aside.

**Pass 2 (45–90 minutes): the argument.** Read methods, results, and supplementary information, reconstructing four things.

1. **The objective and the information flow** (Chapter 1's measurement view): what is the loss, what does the model see as input, what does it *not* see, what is the target? Draw the diagram.
2. **The split.** How were train/validation/test chosen? What is the nearest-neighbor relationship between test and train (homology, scaffold, donor, chromosome)? Which choices were made *after* looking at test results?
3. **The baselines and the ceiling.** What simple models were compared? Is there a noise ceiling or a replicate-based reliability (Chapters 1, 25)? Are the baselines tuned as carefully as the proposed method?
4. **The statistics.** What is the unit of replication, how many are there, what are the error bars, and how were multiple comparisons handled?

Fill the **claim–evidence table** (§57.3). Flag red flags (§57.4).

**Pass 3 (2–6 hours, rarely): reproduce.** Run the released code on the released data; re-derive a key equation; implement the central idea on a toy problem with a known answer (as in Chapters 24–31); change one assumption and see whether the claim survives. The aim is to *own* the result: to know where it breaks.

!!! lens "Research lens: what each pass produces"
    Pass 1 produces a claim and a rung. Pass 2 produces a table of evidence and a list of flags. Pass 3 produces a *new fact*: a reproduction, a counterexample, a boundary of validity, or a derived expression. Only pass 3 yields assets that did not exist before you read the paper.

---

## 57.3 The claim–evidence table

| Claim (as stated) | Rung | Evidence offered | What it would take to be true | Comparison & ceiling | Unit & replicates | Gap |
|---|---|---|---|---|---|---|
| "Model X predicts perturbation responses" | C2? | Held-out perturbations in one cell line | Held-out perturbations are not near-duplicates of training ones; reliability above ceiling | Mean/additive baseline? Reliability (Chapter 25)? | Perturbation (how many with real effects?) | Whether it beats the additive baseline; effect-size strata |
| "The model learned regulatory grammar" | C3 | Attribution maps and motifs | Interventions on the model *and* the biology agree | Motif-ablation; PWM/gkm-SVM baseline (Chapter 29) | Loci | Causality of attributions |
| "Zero-shot variant effects" | C2 | Correlation with a DMS or ClinVar labels | No train/test homology; labels not used in pretraining | Conservation, site-independent baselines | Proteins / genes (not variants) | Family-level generalization |
| "Our model designs binders" | C4 | Wet-lab hit rates | Selection before testing; matched controls; novelty | Hit rate of baseline methods at same cost | Targets (how many?) | Prospective, blind, across targets |

Fill one row per claim (usually three to six per paper). *Most papers' disappointments live in the last column.*

---

## 57.4 A red-flag catalog

Each flag has a **quick test**. They are ordered by how often they occurred in the chapters of this book.

**Data and splits**

1. *Random split of correlated units.* Near-duplicates (analog series; homologs; cells of one donor; overlapping windows) place copies of every test item in training (Chapters 24, 25, 43). **Test:** report performance as a function of similarity to the nearest training item; require a cluster or family split. *In BACE-1, random split $r=0.85$, cluster split $r=0.62$.*
2. *Leakage through the labeling pipeline.* Labels derived from alignment, annotation, or clustering of the *same* data (Chapters 27, 28). **Test:** ask how a label was produced and whether the model's input was involved.
3. *Homology across the split in either orientation.* Repeats, paralogs, reverse-complement copies (Chapter 28). **Test:** search the test set against the training set at the level of 25-mers and their reverse complements. *An RC-augmented high-order model reached 1.09 bits on a repeat copy versus 2.02 on unique DNA.*
4. *Ancestry/batch/donor confounding* (Chapters 25, 26, 45). **Test:** can the inputs predict batch or ancestry, and does the phenotype differ by it?
5. *Test-set reuse*: hyperparameters, early stopping, or model selection on the test data. **Test:** check whether a validation set exists and whether its use is described.

**Baselines and ceilings**

6. *Weak or untuned baseline.* A PWM against a CNN; an untuned random forest (Chapter 29). **Test:** the baseline ladder; are the baselines given the same tuning budget?
7. *Benchmark near its noise ceiling* (Chapters 1, 24, 25). **Test:** compute $1-\sigma^2/\mathrm{Var}(y)$; if the improvements lie within the gap to the ceiling, they are not resolvable.
8. *No null model for the background.* PWM hit counts, motif enrichment, E-values (Chapters 27, 29). **Test:** was the background matched for composition and order?
9. *Trivial baselines omitted*: mean prediction, size-only, control state, additive effects (Chapters 24, 25). **Test:** do they appear in the main table?

**Metrics and statistics**

10. *Metric dominated by noise or by null items.* All-gene correlation for sparse perturbations (Chapter 25). **Test:** reliability or ceiling-normalized score.
11. *Wrong unit of replication.* Cells as replicates (false-positive rate 0.70 versus 0.03 at donor level; Chapter 25); variants rather than genes or proteins. **Test:** find $n$ for each error bar.
12. *Averaging over heterogeneous strata.* A mean hides that a method works only on easy items. **Test:** stratified results.
13. *AUROC at balanced classes for a rare-event problem.* **Test:** AUPRC at the realistic prevalence.
14. *Multiple comparisons.* Many methods, datasets, thresholds, with the best reported. **Test:** is there a pre-specified primary analysis?
15. *Selected examples.* A qualitative figure shows the best case. **Test:** are examples chosen before results, or are random examples shown?

**Interpretation**

16. *Attribution as mechanism.* Saliency/attention maps offered as evidence of causal grammar (Chapters 18, 48). **Test:** is there an intervention (ablation, in-silico edit, experiment) that agrees? Redundant motifs give zero single-site attribution (Chapter 18).
17. *Predictive accuracy used to claim causal effect* (Chapter 31). **Test:** is the claimed quantity a counterfactual? Is it identified by the data?
18. *"Zero-shot" with pretraining overlap.* The evaluation family was in the pretraining set. **Test:** similarity of test families to pretraining data.
19. *Scale as explanation*: gains attributed to scale without ablations. **Test:** are there controls at matched data, compute, and tuning?
20. *Biological generalization from one context*: one cell line, one ancestry, one species (Chapters 25, 26, 45). **Test:** is the claim scoped to the context?

**Reproducibility and framing**

21. *Code or data unavailable*, or only partially. **Test:** can the main table be regenerated?
22. *Seeds and variance*: a single run. **Test:** standard deviation over seeds *and* data resamples.
23. *Compute-unfair comparison.* A large model against a small baseline at equal "training data" but not equal compute. **Test:** compute-matched.
24. *Moving the goalposts*: claims in the abstract larger than the evidence in the results; titles that name the rung above the evidence. **Test:** compare abstract verbs ("predicts", "discovers", "designs") with the rung of the experiments.
25. *Prospective vs retrospective.* "Designed" molecules chosen after seeing assay results. **Test:** timing and selection protocol.

**The three-minute filter.** For a paper with a performance claim, check red flags 1, 6, 7, 11, and 24; if two or more are present, downgrade the claim by a rung until the authors address them.

---

## 57.5 Reading different kinds of papers

**Methods papers.** The question is *what problem the method solves that existing methods do not*. Read the ablations first (what was removed, and what happened), then the comparison table. A methods paper with no ablation is a demonstration.

**Benchmark papers.** The questions are *what the benchmark measures, how it can be gamed, and what ceiling it has*. Check task definition, split construction, label provenance, ceiling, baseline strength. Benchmarks that expose the *failure* of simple methods and the *ceiling* of strong ones are the most useful.

**Biology-discovery papers using ML.** Separate the *machine-learning claim* (this model predicts well) from the *biological claim* (this mechanism exists). The second needs experiments, not predictions. Ask: were the experiments chosen *before* or *after* seeing the model's output, and were negative controls included?

**Preprints.** They are fast and unreviewed; read them as *claims in progress*. Check the version history and the comments; look for revisions in response to review. Preprints from reputable groups have also contained errors that were corrected in the journal version, which is a reason to check the published version.

**Reviews and perspectives.** Use them as *maps*: for the citation structure, vocabulary, and the authors' view of open problems, but verify specific numerical claims in the primary source.

**Model cards and technical reports (industry).** They report the training data, compute, evaluation, and intended use, often without the details needed to reproduce. Treat *capabilities as claims about a product* and weigh the benchmarks against the baselines that are *not* reported.

**LLM-assisted reading.** Language models can summarize and answer questions about papers, but they also fabricate numbers, citations, and details. Use them to *navigate* (find the figure, locate the methods) and *always verify against the text*. Never cite a number you have not seen in the paper.

---

## 57.6 The fixed dissection template

For a paper worth a second pass, write the following (this is the template used throughout the book):

1. **Problem.** What is being predicted or discovered, and why it matters.
2. **Key insight.** The idea that makes the approach work (in a sentence).
3. **Architecture / method.** Components and what each does; tensor shapes if a neural network.
4. **Objective.** The loss, the data distribution, the gap to the target (Chapter 17).
5. **Data.** Sources, size, curation, the measurement process (Chapter 25), the split.
6. **Training.** Compute, optimization, regularization, augmentation.
7. **Evaluation.** Metrics, baselines, ceiling, unit of replication.
8. **Results.** What was shown, with numbers.
9. **Why it worked.** The authors' explanation, and *your* explanation (they may differ).
10. **Assumptions.** Stated and unstated.
11. **Limitations.** Stated and the ones you found.
12. **What followed.** Later work that extended, replicated, or refuted it.
13. **Unresolved.** The open questions and which Attack they suggest (Chapter 55).

!!! paper "Paper dissection: Ahlmann-Eltze, Huber & Anders, *Nature Methods* (2025), \"Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines\""
    **Problem.** Several deep-learning and foundation-model approaches claim to predict the transcriptional effect of unseen genetic perturbations (including combinations); the community needs to know whether they outperform simple models.
    **Key insight.** Define *deliberately simple baselines*: predict the unperturbed control profile (no change), and for double perturbations an **additive model** that sums the two single-perturbation effects; compare. Compare also linear models that use learned embeddings.
    **Method (as reported).** Benchmark of five single-cell foundation models (including scGPT and scFoundation) and two other deep models (including the graph-based GEARS) against the baselines on single- and double-perturbation data; performance measured as the $L_2$ distance between the mean predicted and observed expression profiles. [[S]]
    **Data.** Public Perturb-seq datasets of single and double perturbations (e.g., the Norman et al. combinatorial CRISPR-activation screen and genome-scale single-gene screens).
    **Evaluation.** Held-out perturbations; the additive and no-change baselines; additional analyses (linear model with pretrained embeddings; distinguishing genetic-interaction types).
    **Results.** For double perturbations, **none of the deep models beat the additive baseline**; for unseen single perturbations, none beat a simple linear model with embeddings. [[S]]
    **Why it worked (as a critique).** The $L_2$ distance over mean expression is dominated by large, shared effects (the *mean shift across perturbations* and the additive component), which a simple model captures; the genetic interactions that deep models are meant to capture are a small part of the signal and are hard to measure (Chapter 25's reliability analysis). A follow-up preprint argues that with calibrated metrics deep models do beat uninformative baselines [[S]] for the existence of the argument; the question of which metric is appropriate is [[P]].
    **Assumptions.** That the metric reflects the intended capability; that the baselines are *meaningful* (no change, additive) and not too weak; that the benchmark datasets are representative.
    **Limitations.** A particular set of data and models; benchmark ceilings and effect-size strata not fully resolved; the study cannot say what a *better* model should be.
    **What followed.** Re-examinations of metrics, new benchmarks that emphasize calibrated metrics and distributional measures, and the Arc Virtual Cell Challenge (2025), whose winning submissions combined deep learning with classical statistical features (Chapter 39).
    **Unresolved.** What the right evaluation is (A5), whether any current objective is aligned with causal prediction (A3), and whether interventional data at larger scale changes the conclusion (A4).

!!! paper "Paper dissection: Huang et al., *Nature Genetics* (2023), \"Personal transcriptome variation is poorly explained by current genomic deep learning models\""
    **Problem.** Sequence-to-expression models (Enformer, Basenji2, ExPecto, Xpresso) are used to score regulatory variants. Do they explain differences in gene expression *between people*?
    **Key insight.** Evaluate the models on **paired whole-genome and RNA-seq data from individuals** (421 people in the Geuvadis cohort), asking for *within-gene, across-individual* correlation and for the *sign* of eQTL effects, which are the quantities relevant to variant interpretation; compare with per-gene regularized regression on nearby variants (PrediXcan-type).
    **Data and evaluation.** Personal genomes input to the models; predicted versus observed expression across individuals for each gene; sign agreement for fine-mapped variants.
    **Results.** Per-gene cross-individual correlations were distributed around zero for all four models, though with tails of genes that are strongly positively or negatively correlated; the models often predicted the wrong direction of effect; linear regression on nearby variants explained substantially more cross-individual variation. [[S]]
    **Why it worked (as a diagnostic).** The models were trained to predict *reference-genome* tracks across genes, a between-gene objective; variants within a gene move predictions by small amounts that the training objective never constrained (Chapter 31's identifiability argument is a candidate mechanism, [[P]]).
    **Limitations.** One cohort and tissue (lymphoblastoid cell lines), European and African ancestry individuals; models fine-tuned later may perform differently.
    **What followed.** Fine-tuning on personal-genome data, haplotype-aware modeling, new benchmarks (Chapter 31), and larger-context models (Borzoi, AlphaGenome), whose claims should be evaluated with the *same* within-gene tests.
    **Unresolved.** The causes (objective, data, architecture, biology) and the remedy; this is the starting point of Chapter 55's Worked Example 55.1.

---

## 57.7 Using a paper to generate ideas

After the dissection, run a short **attack pass**:

1. **Assumptions** (A1): copy §57.6 item 10 and mark each as *known true*, *approximate*, or *convenient*.
2. **Representation and data** (A2, A4): what does the input omit; what *variation* is missing in the data for the claimed capability?
3. **Objective** (A3): the loss is aligned with the claim to what degree? Which shortcut would solve it?
4. **Evaluation** (A5): which red flags apply; what is the adversarial test?
5. **Structure** (A7, A8): what mathematical structure is this, and where else does it appear?
6. **Biology** (A9, A10): what constraint is unused; what unexplained pattern in the *errors* does the paper show or hide?

Write the top three ideas (If/then/because) and the cheapest test for each. Papers that yield no ideas after this pass were either mature, or read too fast.

---

## 57.8 Worked research examples

!!! example "Worked Research Example 57.1: \"Our attention maps reveal the regulatory logic of the genome\""
    **Situation.** A paper trains a long-context sequence model on chromatin and expression data and shows attention maps with strong, structured signals between a promoter and distal elements. It concludes that the model "discovered enhancer–promoter interactions", compares overlap with Hi-C loops and with ABC-model predictions, and highlights several loci.

    **Reading.**

    1. *Pass 1: claim and rung.* "Discovered interactions" is a C3 claim: the model's internal computation reflects mechanism. Evidence needed: interventions on the model *and* on the biology that agree.
    2. *Pass 2: reconstruct.* (a) The objective is track prediction; attention is an *internal* quantity with no loss on it. (b) Overlap with Hi-C loops: what is the null? Loops cluster near strong promoters and CTCF sites; a distance-matched random pairing may overlap similarly. (c) The *ceiling*: ABC itself has precision limits (Chapter 22). (d) The highlighted loci: selected after the fact?
    3. *Red flags:* 16 (attribution as mechanism), 8 (no null model for overlap), 15 (selected examples), 24 (title vs rung).
    4. *Quick tests.* **Shuffle the context**: replace distal sequence by a dinucleotide-shuffled sequence and compare the model's prediction change to the claimed element's attention (Karollus et al. found that such models derive most signal from the promoter-proximal region; Chapter 31). **Distance-matched null** for loop overlap. **In-silico deletion** of the highlighted element vs random equal-sized elements. **Compare with ABC** at equal recall.
    5. *Experiments that would raise the rung.* CRISPRi or deletion of a random sample of predicted elements (not chosen post hoc) with expression readout: a prospective C3–C4 test.

    **Expert analysis.** A map can look biological because biology and the *data-generating process* share structure (distance, accessibility). The claim becomes strong only when an intervention on the model's output agrees with an intervention in the cell. This reading would downgrade the claim to "attention correlates with known interactions" (C2) until those tests are done; it also yields idea 3 of Chapter 55's sheet.

!!! example "Worked Research Example 57.2: A leaderboard result for protein fitness prediction"
    **Situation.** A paper reports a new protein language model with Spearman 0.51 averaged across a benchmark of deep mutational scans, versus 0.47 for the previous best, and concludes that "scale improves fitness prediction".

    **Reading.**

    1. *Pass 1.* Claim: a 0.04 improvement attributed to scale: C1 or C2 depending on the splits.
    2. *Pass 2.* (a) **Ceiling**: DMS assays have replicate correlation of roughly 0.8–0.95 (Chapter 23), so 0.51 is far from the ceiling for a single average; the improvement is not saturated. (b) **Unit of replication**: the benchmark's assays (proteins), not variants: 0.04 on how many proteins, with what dispersion? (c) **Averaging**: stratify by MSA depth, protein family, assay type (stability vs binding vs growth: the *link function* of Chapter 23). (d) **Preference confound**: language-model likelihoods reflect sequence typicality and species bias (Gordon et al., 2025): are gains concentrated on proteins whose wild type is well represented? (e) **Compute and data matched?** Parameters, training data, and recipe: did *scale* change, or the data and the objective as well? (f) **Overlap**: do the DMS proteins or close homologs appear in pretraining data?
    3. *Quick tests.* A paired bootstrap *over assays*; the improvement as a function of $N_\text{eff}$ and of assay type; compare with a Potts/site-independent baseline at the same coverage (Chapter 29); a small-model control at matched data.
    4. *What the answer implies.* If the gain is concentrated at high $N_\text{eff}$ and for stability assays, it supports "better use of evolutionary information", not "scale in general"; if it holds for low-$N_\text{eff}$ families and binding assays, it supports a more general claim.

    **Expert analysis.** The abstract's causal verb ("scale improves") requires a *controlled comparison* that holds everything else fixed; most leaderboard papers cannot provide it. The reading converts a leaderboard line into a *testable decomposition*.

---

## 57.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: the reading log"
    **Setting.** Reading is a habit; the log makes it cumulative.

    **For every paper that reaches pass 2, record (one page):**

    1. Citation, date, venue/version, code/data link.
    2. Headline claim and rung.
    3. Objective and information-flow sketch (3 lines).
    4. Split, baselines, ceiling, unit of replication.
    5. Red flags found (by number from §57.4).
    6. Assumptions (three most important).
    7. Two attacks (Chapter 55) and the cheapest test for each.
    8. One question for the authors.

    **A weekly practice.** One paper per week at pass 2; one per month at pass 3 (reproduce a figure or run the toy). A shared reading group improves the log: have one person defend the paper and one prosecute it.

    **What it teaches.** After about twenty entries, patterns appear (the same three flags in most papers), and your pass-1 filter improves. The log is also the raw material for related-work sections and for the prior-art searches of Chapter 56.

    **An open question to carry forward.** Could one build a *quantitative* reading tool: given a paper's reported numbers and split description, estimate the probability that the headline claim survives a stricter evaluation? What features of papers (benchmark age, baseline count, ceiling reporting, code availability) predict failures to replicate in AI for biology? Collect twenty replication attempts and test the question, taking care with selection bias (the papers that get replicated are not random).

---

## 57.10 Connections

- **Backward:** the Four Gaps and Claim Ladder (Chapter 1); leakage and ceilings (Chapters 24–29); the tokenization and identifiability examples (Chapters 28, 31); idea generation and triage (Chapters 55–56).
- **Forward:** reasoning without known answers (Chapter 58); from idea to publication (Chapter 59); evaluation and benchmarks (Chapter 43).

!!! takeaways "Key takeaways"
    1. Read in **three passes**: claim and rung (10 minutes), argument (an hour: objective, split, baselines, statistics), reproduction (hours; rare).
    2. Keep a **claim–evidence table**; most weaknesses are in the *gap* column: the unit of replication, the baseline, the ceiling, and the rung.
    3. A **red-flag catalog** (25 items) with quick tests catches the common failures: near-duplicate splits, leakage through labels, weak baselines, saturated benchmarks, noise-dominated metrics, wrong unit of replication, attribution as mechanism, and titles that outrun evidence.
    4. The **three-minute filter** (flags 1, 6, 7, 11, 24) downgrades a claim by a rung when two or more apply.
    5. Different paper types need different questions: methods (ablations), benchmarks (ceiling and gaming), discovery (experiments), preprints (versions), reviews (maps), model cards (missing baselines).
    6. Use the **fixed dissection template** and finish with an **attack pass** to convert reading into ideas.
    7. Language models help navigate papers but fabricate details; verify every number.
    8. Keep a **reading log**; patterns across entries sharpen your filters.

---

## Further reading

- Keshav, S. (2007). How to read a paper. *ACM SIGCOMM Computer Communication Review* 37, 83–84. Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Med.* 2, e124. Munafò, M. R. et al. (2017). A manifesto for reproducible science. *Nat. Hum. Behav.* 1, 0021.
- Ahlmann-Eltze, C., Huber, W. & Anders, S. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat. Methods* 22, 1657–1661. Huang, A. C. et al. (2023). Personal transcriptome variation is poorly explained by current genomic deep learning models. *Nat. Genet.* 55, 2056–2059. Kedzierska, K. Z. et al. (2025). Zero-shot evaluation reveals limitations of single-cell foundation models. *Genome Biol.* 26, 101.
- Lones, M. A. (2024). How to avoid machine learning pitfalls: a guide for academic researchers. *Patterns* 5, 101046. Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4, 100804. Whalen, S., Schreiber, J., Noble, W. S. & Pollard, K. S. (2022). Navigating the pitfalls of applying machine learning in genomics. *Nat. Rev. Genet.* 23, 169–181.
