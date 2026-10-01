# Chapter 43. Benchmarks, Leakage, and the Statistics of Evaluation

!!! abstract "Chapter at a glance"
    **Motivation.** Part VII described what models claim; this chapter is about how those claims are *measured*. A benchmark is an instrument. Like any instrument it has a validity (does it measure the capability we care about?), a reliability (how noisy is it?), and a resolution (what differences can it detect?). Biology adds special difficulties: data are strongly structured by evolution, so ordinary random splits leak; labels come from pipelines (Chapter 27); and the interesting generalizations (new families, new cell types, new ancestries, new interventions) are *shifts* that most benchmarks do not test. We organize the failure modes into a taxonomy, quantify the statistical ones by simulation (winner's curse, adaptive reuse, metric choice, resolution), and give design rules for benchmarks and blind challenges.
    **Prerequisites.** Chapters 1, 4, 7, 24–29, 31, 32; Chapter 59's power calculation.
    **You will be able to:** (1) choose a split that matches a claim and say what each split type tests and leaks; (2) classify a leakage route and write the test that detects it; (3) explain why a leaderboard's top ranks are unreliable and compute the resolution of a benchmark; (4) choose metrics for rare-event problems; (5) design a benchmark and a blind challenge; (6) read published benchmarks critically.

---

## 43.1 A benchmark as a measuring instrument

A benchmark pairs a **dataset**, a **task definition**, a **split**, a **metric**, and a **protocol** (what may be used, how many submissions). It is a proxy for a capability. Four properties matter.

* **Validity.** Does a high score require the capability? A benchmark for "regulatory grammar" that can be solved by GC content has low validity (Chapter 29). Validity is argued with *adversarial baselines*: the simple models that should fail if the benchmark is valid (size-only, composition, nearest neighbor, copy rule; Chapters 24, 28).
* **Reliability.** Replicate or split-half agreement of the measured scores; the **noise ceiling** (Chapters 1, 25).
* **Resolution.** The smallest model difference the benchmark can resolve, given the effective number of test units (§43.5, Chapter 59's power calculation).
* **Utility and durability.** Does improving the score improve a decision, and how fast does the benchmark saturate or become contaminated by pretraining data?

The **Claim Ladder** (Chapter 1) gives the link to claims: random-split benchmarks support C1; structured, shift-aware benchmarks support C2; prospective, blind, intervention-based challenges support C4. *The same model can pass C1 and fail C2*; reporting only C1 and writing the abstract at C2 is the commonest overstatement in the field.

---

## 43.2 Splits: what each tests and what each leaks

The split should reproduce, in the test set, **the shift you want the model to survive**.

| Domain | Split | Tests generalization to | Leaks if not careful |
|---|---|---|---|
| Any | Random | Same distribution | Near-duplicates, homologs, correlated units |
| Genomic sequence | By chromosome | Other regions of the same genome | Repeats, paralogs, inverted repeats (Chapter 28) |
| | By sequence-identity cluster (e.g., MMseqs2 or CD-HIT at 30–50%) | Unseen families | Remote homology below the threshold; domains shared between families |
| | By species or clade | New organisms | Horizontal transfer; conserved elements |
| Proteins | By family / superfamily / fold (Pfam, CATH, SCOP) | New families/folds | Domain-level sharing between folds |
| | By time (deposition date) | Prospective-like | Slow drift; re-deposits of old proteins |
| Molecules | Scaffold | New ring systems | Analogs with different scaffold strings (Chapter 24: scaffold split $r=0.83$ vs cluster $r=0.62$) |
| | Cluster (Butina, Tanimoto) | Chemically novel molecules | Assay and series effects |
| | By target / by time | New targets / prospective | Cross-target transfer (shared ligands) |
| Cells | By donor | New people | Donor effects (Chapter 25) |
| | By batch / study / technology | New labs | Composition differences (Chapter 30) |
| | By perturbation / cell line / context | New interventions / contexts | Perturbation similarity (same complex, same pathway) |
| Genetics | By cohort / ancestry (as a continuum) | New populations | Relatives; shared environments (Chapter 26) |
| Models trained on pretrained models | Remove pretraining overlap | True zero-shot | The pretraining corpus contains the test examples |

Two rules. **(1) Identity thresholds are conventions, not guarantees**: 30% identity does not remove structural homology; for protein-ligand or structure benchmarks, similarity should be measured on *structure and ligand* as well as sequence (Graber et al., 2025; Škrinjar et al., 2025) [[S]]. **(2) Stratify, do not only filter**: reporting performance *as a function of similarity to the training set* (Chapter 24: RMSE 0.58 at nearest-neighbor Tanimoto 0.80–0.95, 1.00 below 0.50) shows how the claim degrades with novelty and is more informative than one number at one threshold.

---

## 43.3 A taxonomy of leakage

| # | Leakage | Mechanism | Detection | Example in this book |
|---|---|---|---|---|
| L1 | **Duplicates and near-duplicates** | Test items have copies in training | Similarity search (exact and approximate) | Analog series, Chapter 24 |
| L2 | **Homology** | Evolutionary relatives across the split, including reverse complements | $k$-mer hashing in both orientations; profile search | 1.09 vs 2.02 bits on an RC repeat copy, Chapter 28 |
| L3 | **Label-pipeline leakage** | Labels generated from the same data as the inputs (alignment, annotation, clustering) | Trace each label's provenance | Mappability-dependent tracks, Chapter 27 |
| L4 | **Feature leakage** | An input encodes the target (future information, label-derived features, a covariate that proxies the outcome) | Ablate features; check timestamps | Overlapping $k$-mer masking solved by copying, Chapter 28 |
| L5 | **Batch/donor leakage** | Test cells share donor, batch, or sample with training | Predict batch/donor from embeddings | Cell-level tests, Chapter 25 |
| L6 | **Temporal leakage** | Training uses information from after the test event | Time split | Retrospective drug or structure benchmarks |
| L7 | **Selection leakage** | Hyperparameters, early stopping, or thresholds chosen on the test set | Pre-registration; separate validation | §43.5.2 |
| L8 | **Pretraining contamination** | The test items (or close relatives) occurred in a foundation model's pretraining corpus | Overlap search against the corpus; hold-out of post-cutoff data | Zero-shot claims (Chapters 32, 34, 38) |
| L9 | **Benchmark-design leakage** | Decoys or negatives differ from positives in a trivial way | Train a model on the ligand/decoy alone | DUD-E decoy bias; PDBbind–CASF overlap (Chapter 24) |
| L10 | **Normalization leakage** | Statistics (means, scales, size factors) computed on all data including test | Pipeline audit | Standardization before splitting |

**A unit test for leakage.** For any benchmark, train a *trivially weak* model (a copy rule, a nearest neighbor on a distinct feature, a size-only regressor, a model of the label from the *non-causal* covariates). If it scores well, the benchmark is not measuring the intended capability; if a strong model's advantage over that baseline is small, the benchmark cannot show progress.

---

## 43.4 Metrics

**Correlations (Pearson, Spearman).** Appropriate for continuous outputs with a ceiling (replicate correlation). Averages over proteins, genes, or perturbations should be *per-unit*, with the unit-level distribution reported, and the ceiling per unit (Chapter 25's reliability: 0.0 to 0.96 across perturbations). A single pooled correlation can be dominated by between-unit variance.

**Classification metrics and prevalence.** AUROC does not depend on class balance; precision-based metrics do. For a fixed classifier with score distributions $\mathcal N(1.5,1)$ for positives and $\mathcal N(0,1)$ for negatives (`code/ch43_benchmarks.py`):

| Positive rate | AUROC | AUPRC | Precision at 50% recall | True hits in the top 100 | Hits by chance |
|---|---|---|---|---|---|
| 0.5 | 0.855 | 0.853 | 0.882 | 100 | 50 |
| 0.05 | 0.856 | 0.338 | 0.283 | 89 | 5 |
| 0.005 | 0.859 | 0.065 | 0.037 | 29 | 0.5 |
| 0.0005 | 0.830 | 0.009 | 0.003 | 1 | 0.05 |

The AUROC stays at 0.83–0.86 while the *useful* quantities collapse: at a prevalence of $5\times10^{-4}$ (typical of genome-scale variant or enhancer problems) the top 100 predictions contain about one true positive, and precision at half recall is 0.3%. *A benchmark with balanced positives and negatives cannot predict deployment performance in a genome-wide scan.* Report AUPRC or precision/recall at the **deployment prevalence**, or construct the benchmark with realistic negatives.

**Ranking and enrichment.** When the decision is to choose $k$ items to test (Chapter 46), report enrichment of true hits in the top $k$ and the number of experiments saved. **Calibration** (expected calibration error; conformal coverage; Chapter 18) matters when predictions are used as probabilities. **Ceiling-normalized scores** $(\text{score}-\text{baseline})/(\text{ceiling}-\text{baseline})$ express progress on a common scale.

---

## 43.5 The statistics of comparing models

### 43.5.1 Winner's curse and the reliability of ranks

Suppose 100 models have true accuracies spread with SD 0.01 around 0.80 and are evaluated on independent test sets of $n$ items. The model ranked first on test set A has an inflated score; on fresh data it falls back toward its true value, and the ranking itself is hardly reproducible:

| Test items $n$ | SE of one accuracy | Accuracy of the #1 model on test A | Same model on fresh test B | Its true accuracy | Rank correlation A vs B | P(#1 on A in top 10 on B) |
|---|---|---|---|---|---|---|
| 200 | 0.0283 | 0.871 | 0.808 | 0.809 | 0.11 | 0.18 |
| 1,000 | 0.0126 | 0.840 | 0.817 | 0.816 | 0.37 | 0.42 |
| 5,000 | 0.0057 | 0.828 | 0.822 | 0.822 | 0.74 | 0.83 |
| 50,000 | 0.0018 | 0.825 | 0.825 | 0.825 | 0.96 | 1.00 |

With 1,000 test items the winner's reported accuracy exceeds its true value by 2.4 points and its rank on a second test set is nearly unrelated to its first rank (Spearman 0.37). *If neighboring leaderboard entries differ by less than about twice the standard error, their order is mostly noise.* Most biological benchmarks have effective $n$ of tens to hundreds of units (proteins, families, loci, perturbations), not thousands of items, so **the resolution is far worse than the item count suggests** (Chapter 59: 40 proteins with $s=0.15$ resolve about 0.07 in Spearman).

### 43.5.2 Adaptive reuse of a test set

A researcher who tries $K$ variants and reports the best on the same test set will find "improvement" even when none exists. With a baseline of 0.800 accuracy, true improvement zero, and $n=2{,}000$:

| Variants tried $K$ | Reported improvement (mean) | On a fresh test set | P(reported exceeds 0.009, about 1 SE of a paired difference) |
|---|---|---|---|
| 1 | $+0.0004$ | $-0.0001$ | 0.26 |
| 5 | $+0.0104$ | $+0.0001$ | 0.55 |
| 20 | $+0.0170$ | $+0.0003$ | 0.80 |
| 100 | $+0.0223$ | $0.0000$ | 0.93 |
| 1,000 | $+0.0285$ | $+0.0007$ | 0.99 |

Twenty variants produce a reported gain of 1.7 points that has exactly zero true basis; a hundred variants make it look "significant" 93% of the time. The same mechanism operates *across a community* (many labs reusing a public test set) and *within a lab* (hyperparameter and architecture search). Remedies: a **validation set** for all choices and a **test set touched once**; a **private test set** held by organizers with limited submissions; **adjusted inference** (the reusable holdout: Dwork et al., 2015); **fresh data** (prospective test); and, when none of these are available, reporting the number of variants tried. Empirically, large image benchmarks have shown less overfitting to their test sets than this simulation suggests (Roelofs et al., 2019; Recht et al., 2019), which reflects large $n$; the small, structured test sets of biology are the regime in which the problem is real [[P]].

### 43.5.3 Resolution

To detect a difference $d$ in accuracy between two paired models with probability 0.8 at two-sided $\alpha=0.05$ requires about $2p(1-p)(1-\rho)\,(z_{\alpha/2}+z_\beta)^2/d^2$ independent items, with $\rho$ the error correlation between models (0.5 below):

| Accuracy near | $d=0.05$ | $d=0.02$ | $d=0.01$ |
|---|---|---|---|
| 0.60 | 754 | 4,710 | 18,838 |
| 0.80 | 503 | 3,140 | 12,559 |
| 0.90 | 283 | 1,766 | 7,064 |
| 0.95 | 150 | 933 | 3,729 |

Resolving a 1-point difference near 0.8 needs about 12,600 *independent* items; a benchmark of 200 proteins resolves differences of about 8–10 points (and *effective* independence is lower when items cluster in families). Benchmarks whose top models differ by less than the resolution are **saturated**, regardless of how far the scores are from 100%, and should be refreshed (harder items, new data) or retired.

!!! lens "Research lens: what a leaderboard position is evidence of"
    A rank on a public leaderboard is evidence of (i) performance on a *particular, finite sample* of a particular distribution, (ii) *selection* by many tries, and (iii) *similarity to the benchmark's biases*. It is weak evidence of general capability unless the differences exceed the resolution, the split matches the claim, and a prospective test agrees. Report intervals; compare by paired bootstrap over *units*; and treat differences below the resolution as ties.

---

## 43.6 Blind, prospective challenges

The most credible evaluation in the field is the **blind challenge**: organizers withhold the test data until predictions are submitted, and the data are *generated after the submission deadline* or released only then, so that neither leakage nor test-set reuse is possible.

* **CASP** (Critical Assessment of protein Structure Prediction), running since 1994, predicts structures of proteins whose structures are about to be solved; AlphaFold 2's CASP14 result (median GDT_TS about 92 across targets) was accepted as a solution of the single-chain monomer problem because the test was blind [[E]]. CASP also shows what remains hard: RNA, ligands, complexes, and conformational ensembles (the CASP15 assessment of RNA 3D structure found that deep-learning methods were significantly worse than the top-ranked classical and human-expert groups, Chapter 33) [[S]].
* **CAGI** (Critical Assessment of Genome Interpretation) and **DREAM** challenges for variant interpretation and regulatory prediction; **CAFA** for protein function.
* **Arc Virtual Cell Challenge (2025):** more than 5,000 registrants from 114 countries, over 1,200 teams submitting predictions and over 300 reaching the final stage for predicting transcriptional responses to perturbations in a held-out cell context, with winners that combined deep learning with classical statistical features and models that did not consistently beat naive baselines on all metrics (Chapter 39) [[S]].

**Designing a blind challenge.** (1) Fix the task, metric(s), baselines, and ceilings *before* data release; (2) generate the test data after the submission deadline, or hold it privately; (3) limit submissions per team; (4) ensure the test set is *novel* relative to all plausible training data (time or family split with a documented overlap analysis); (5) publish the full results including baselines and failure cases; (6) *evaluate the metric's validity* in advance (could a trivial baseline win?); (7) plan for lifecycle: successive rounds with harder tasks.

---

## 43.7 Case studies

**Well designed (or well corrected).**

* *CASP*: blind, time-split, many targets, independent assessors, multiple metrics.
* *ProteinGym* (Notin et al., 2023): a large collection of deep mutational scans (217 assays and about 2.5 million variants in the current version) with family-level aggregation, several metrics, and standardized splits, which has been used to reveal that language-model likelihoods reflect sequence preference and species bias (Gordon et al., 2025) [[S]].
* *PoseBusters* (Buttenschoen et al., 2024): tests not only RMSD of docked poses but also **physical validity** (bond lengths, clashes, stereochemistry), exposing predictions that looked right and were physically implausible [[E]].
* *Runs N' Poses* (Škrinjar et al., 2025): 2,600 protein–ligand complexes stratified by similarity to training data, showing that co-folding accuracy declines substantially on dissimilar complexes [[S]].

**Corrected or criticized.**

* *PDBbind–CASF leakage* (Graber et al., 2025) and *DUD-E decoy bias* (Chen et al., 2019): affinity and virtual-screening benchmarks solvable without protein–ligand interaction learning [[S]].
* *DNA language-model benchmarks* (Tang & Koo, 2025; DART-Eval): pretrained models often not better than one-hot supervised baselines on regulatory tasks [[S]].
* *Perturbation benchmarks* (Ahlmann-Eltze et al., 2025): baselines that were missing from the original comparisons; metrics dominated by shared effects [[S]].
* *Random-split molecular benchmarks* (Chapter 24): near-duplicate analogs and saturation near the noise ceiling.

The pattern: **the corrections came from adding a baseline, stratifying by similarity, or adding a validity check, not from a bigger model**.

---

## 43.8 A design checklist for a new benchmark

1. **Construct and decision.** What capability, and which decision does it inform? Write the claim at a rung of the Claim Ladder.
2. **Provenance.** Data sources, measurement process, label pipeline (L3), consent and licensing.
3. **Ceiling.** Replicate-based reliability per item and per unit; the maximum attainable score.
4. **Splits at the claim's level**, with similarity measured by sequence, structure, scaffold, donor, or ancestry as appropriate; stratified reporting by similarity.
5. **Baseline ladder**: trivial, simple, strong classical (Chapter 29), and *adversarial* baselines (copy rule, composition, nearest neighbor, ligand-only).
6. **Metrics**: a primary metric and the pre-specified secondary; per-unit statistics; deployment-prevalence precision; calibration.
7. **Statistics**: unit of replication and effective $n$; paired bootstrap over units; resolution table; multiple-comparison rule.
8. **Contamination control**: private hold-out; post-cutoff data; documented overlap analysis with known pretraining corpora; canary entries.
9. **Protocol**: limited submissions; a validation set; one-shot test; disclosure of the number of variants tried.
10. **Documentation and maintenance**: datasheet (Appendix G), versioning, errata, and a planned refresh when saturated.

---

## 43.9 The experiments, verbatim

```python
--8<-- "code/ch43_benchmarks.py"
```

```text
== 1. 100 models whose true accuracies differ by SD 0.01 around 0.80; two independent test sets of n items ==
test items n   SE of one accuracy   mean accuracy of the model ranked #1 on test A   its accuracy on fresh test B   true accuracy of that model   Spearman(rank A, rank B)   P(#1 on A is in top 10 on B)
      200         0.0283                0.8714                                     0.8076                         0.8093                      0.11                       0.18
     1000         0.0126                0.8395                                     0.8165                         0.8158                      0.37                       0.42
     5000         0.0057                0.8284                                     0.8220                         0.8219                      0.74                       0.83
    50000         0.0018                0.8254                                     0.8247                         0.8247                      0.96                       1.00

== 2. A researcher tries K variants (true improvement 0 over a baseline with accuracy 0.800) and reports the best on the SAME test set (n = 2,000) ==
variants tried K   reported improvement over baseline (mean)   improvement on a fresh test set (mean)   P(reported improvement > 1 SE of a paired difference)
           1            +0.0004                                    -0.0001                                   0.26
           5            +0.0104                                    +0.0001                                   0.55
          20            +0.0170                                    +0.0003                                   0.80
         100            +0.0223                                    -0.0000                                   0.93
        1000            +0.0285                                    +0.0007                                   0.99

== 3. One classifier (scores N(1.5,1) for positives, N(0,1) for negatives), varying the positive rate ==
positive rate   AUROC    AUPRC    precision at 50% recall    expected true hits among the top 100   (hits if chance)
     0.5000   0.855    0.853        0.882                         100                           50.00
     0.0500   0.856    0.338        0.283                          89                            5.00
     0.0050   0.859    0.065        0.037                          29                            0.50
     0.0005   0.830    0.009        0.003                           1                            0.05

== 4. Test items needed to detect an accuracy difference (two-sided alpha 0.05, power 0.8; paired models with error correlation 0.5) ==
accuracy near   difference   items needed
    0.60      0.05 ->    754    0.02 ->   4710    0.01 ->   18838
    0.80      0.05 ->    503    0.02 ->   3140    0.01 ->   12559
    0.90      0.05 ->    283    0.02 ->   1766    0.01 ->    7064
    0.95      0.05 ->    150    0.02 ->    933    0.01 ->    3729
```

---

## 43.10 Worked research examples

!!! example "Worked Research Example 43.1: A leaderboard where the top five differ by 0.3 points"
    **Situation.** A public benchmark of 200 proteins ranks genomic or protein models by mean Spearman correlation; the top five lie within 0.003 of each other. A new model enters at rank 1 with +0.002. The authors claim state of the art.

    **Question.** What does the ranking support?

    **Reasoning.**

    1. *Resolution.* With 200 proteins and a per-protein paired-difference SD of, say, 0.08, the standard error of the mean paired difference is $0.08/\sqrt{200}=0.0057$: a difference of 0.002 is 0.35 SE. From Table 43.5.1, rank correlations between independent evaluations are about 0.1–0.4 at this scale.
    2. *Selection.* How many variants did the authors try? Per §43.5.2, 20 variants produce a "gain" of the size of the observed one with no true basis.
    3. *Effective $n$.* If the 200 proteins cluster into 60 families, the effective $n$ is closer to 60, and the SE is 0.010.
    4. *What would be evidence?* A paired bootstrap over *families* with a confidence interval; a *held-out* confirmation on new assays; a *prospective* test; a *compute- and data-matched* comparison; and an *a priori* hypothesis about *where* the model should help (stratify: low $N_\text{eff}$, binding assays).
    5. *Conclusion.* The ranking is a tie. The paper may still contribute (a cheaper model, a new capability, an ablation), but the claim "state of the art" is not supported.

    **Expert analysis.** The appropriate response is not to dismiss the model but to *state the resolution*: "differences below ~0.02 are not resolved by this benchmark."

!!! example "Worked Research Example 43.2: Designing a benchmark for zero-shot noncoding variant effect prediction"
    **Situation.** A consortium wants a benchmark to compare genomic language models and sequence-to-function models for the effect of noncoding variants. There is no agreed answer; the design choices determine which models win.

    **Reasoning (the checklist).**

    1. *Construct and decision.* Which decision: prioritizing noncoding variants for clinical interpretation, or fine-mapping GWAS loci? These differ in prevalence (pathogenic variants are $\sim10^{-3}$ of rare noncoding variants), in what counts as a label, and in the acceptable false-positive rate.
    2. *Labels.* Candidates: saturation mutagenesis and MPRA (direct, in specific cell lines); fine-mapped eQTL and GWAS variants (LD-confounded; Chapter 26); ClinVar (annotation-driven, skewed to coding/splice and to what is tested clinically: label leakage L3); allele-specific expression and binding (within-individual controls). *Each label type has a different noise ceiling and different shortcuts*; combine several and report each.
    3. *Splits.* By locus and by LD block (not by variant), with homology removal in both orientations (Chapter 28); by cell type held out where the model allows; by ancestry for population-frequency–dependent claims.
    4. *Baselines.* Conservation, distance to TSS, GC, a supervised one-hot CNN trained on the same labels (Chapter 29), PrediXcan-style per-gene regression (Chapter 31). The benchmark should require beating these *per stratum*.
    5. *Metrics.* Within-locus ranking (Spearman) for MPRA; AUPRC at realistic prevalence for pathogenic classification; sign accuracy conditional on predicted magnitude for eQTLs; calibration.
    6. *Contamination.* Many variants and loci appear in public resources included in pretraining; define a post-cutoff, privately held evaluation set or generate new MPRA data after model release.
    7. *Validity checks.* A model trained on distance to TSS alone should do poorly; a model that sees the label-generating annotation should be disqualified.
    8. *Resolution.* Compute the number of independent loci needed to resolve the differences of interest (§43.5.3) and refuse to publish a ranking below it.

    **What no one knows.** Whether any such benchmark reflects clinical utility; whether the best benchmark is *prospective* (a consortium that designs variants, tests them, and scores predictions made before the tests: C4). Chapter 46 treats the design of the experiments that would generate this data.

---

## 43.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: auditing a benchmark in a day"
    1. **Write the claim** and its rung. Does the split reproduce the shift implied by the claim?
    2. **Run the leakage unit test**: a copy rule, nearest neighbor, composition-only, and (for pairs) ligand-only or protein-only baselines. Record their scores next to the headline model.
    3. **Measure similarity to training** for every test item and plot performance against it.
    4. **Compute the ceiling** from replicates, and the **resolution** (items or effective units needed for 1 and 2 points).
    5. **Check prevalence** and recompute metrics at deployment prevalence.
    6. **Count the tries**: how many models, hyperparameters, and checkpoints were compared on this test set? Estimate the expected inflation (Table 43.5.2).
    7. **Check contamination**: does the test set appear in the training corpora of any pretrained model compared?
    8. **Decide**: tie, win, or unresolved? State the decision rule before looking at the final numbers.

    **What it teaches.** Most of the value of a benchmark is in its *negative controls*; a day spent on them changes what you believe about a leaderboard more than a month spent on the model.

    **An open question to carry forward.** Pretrained foundation models are trained on essentially all public biological data, so any public benchmark is potentially contaminated. Is there a design in which *contamination is measurable*, for example by including held-out "canary" sequences and variants that were deliberately generated after the training cutoff, and by reporting performance as a function of the model's estimated exposure to near-duplicates? How would you estimate exposure for a closed model?

---

## 43.12 Connections

- **Backward:** the noise ceiling and leakage (Chapter 1); statistical learning and splits (Chapter 7); molecular benchmark case study (Chapter 24); pseudoreplication and per-unit reliability (Chapter 25); LD and ancestry (Chapter 26); mappability and label provenance (Chapter 27); homology and RC leakage (Chapter 28); the baseline ladder (Chapter 29); evaluation of sequence-to-function models (Chapter 31).
- **Forward:** causal inference (Chapter 44); distribution shift (Chapter 45); experimental design for benchmark data (Chapter 46); open problems that need new benchmarks (Chapters 50–54); reading benchmark papers (Chapter 57); release and reporting practice (Chapter 59).

!!! takeaways "Key takeaways"
    1. A benchmark is an instrument with **validity, reliability, resolution, and durability**; the split must reproduce the shift implied by the claim; random splits support C1 only.
    2. **Leakage has ten routes** (duplicates, homology, label pipelines, features, batch/donor, time, selection, pretraining contamination, design, normalization); detect with trivial baselines and similarity-stratified reporting.
    3. With 100 near-equal models and 1,000 test items, the #1 model's reported accuracy exceeds its true value by about 2.4 points and its rank correlates only 0.37 with its rank on a fresh test set; **differences below about twice the standard error are ties**.
    4. **Adaptive reuse** inflates results: 20 variants tried gave +1.7 points of pure selection effect; 100 variants gave a "significant" gain 93% of the time.
    5. **AUROC hides rare-event failure**: at a $5\times10^{-4}$ positive rate AUROC stayed 0.83 while the top 100 contained one true positive.
    6. **Resolution**: resolving a 1-point difference near 0.8 accuracy needs about 12,600 independent items; effective units, not items, are what count.
    7. **Blind, prospective challenges** (CASP, CAGI, DREAM, the Virtual Cell Challenge) are the gold standard because neither leakage nor reuse is possible.
    8. The corrections that improved benchmarks (PoseBusters validity checks, Runs N' Poses stratification, CleanSplit, additive baselines) were **baselines, strata, and validity checks**, not larger models.

---

## Further reading

- Critical Assessment of Structure Prediction: Moult, J. et al. (1995). A large-scale experiment to assess protein structure prediction methods. *Proteins* 23, ii–iv. Kryshtafovych, A., Schwede, T., Topf, M., Fidelis, K. & Moult, J. (2021). Critical assessment of methods of protein structure prediction (CASP)—Round XIV. *Proteins* 89, 1607–1617.
- Dwork, C. et al. (2015). The reusable holdout: preserving validity in adaptive data analysis. *Science* 349, 636–638. Recht, B., Roelofs, R., Schmidt, L. & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*. Roelofs, R. et al. (2019). A meta-analysis of overfitting in machine learning. *NeurIPS*.
- Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nat. Biotechnol.* 35, 1026–1028. Li, W. & Godzik, A. (2006). Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. *Bioinformatics* 22, 1658–1659.
- Notin, P. et al. (2023). ProteinGym: large-scale benchmarks for protein fitness prediction and design. *NeurIPS Datasets & Benchmarks*. Buttenschoen, M., Morris, G. M. & Deane, C. M. (2024). PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. *Chem. Sci.* 15, 3130–3139. Graber, D. et al. (2025). Resolving data bias improves generalization in binding affinity prediction. *Nat. Mach. Intell.* 7. Chen, L. et al. (2019). Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening. *PLoS ONE* 14, e0220113. Tang, Z. & Koo, P. K. (2025). Evaluating the representational power of pre-trained DNA language models for regulatory genomics. *Genome Biol.* 26. Marin, F. I. et al. (2024). BEND: benchmarking DNA language models on biologically meaningful tasks. *ICLR*.
- Gebru, T. et al. (2021). Datasheets for datasets. *Commun. ACM* 64, 86–92. Raji, I. D. et al. (2021). AI and the everything in the whole wide world benchmark. *NeurIPS Datasets & Benchmarks*. Koch, B., Denton, E., Hanna, A. & Foster, J. G. (2021). Reduced, reused and recycled: the life of a dataset in machine learning research. *NeurIPS Datasets & Benchmarks*.
