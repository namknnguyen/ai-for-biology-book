# Chapter 31. Sequence-to-Function Models

!!! abstract "Chapter at a glance"
    **Motivation.** The most direct ambition of regulatory genomics is a function from DNA sequence to molecular activity: given the sequence of a locus, predict its chromatin accessibility, transcription-factor binding, histone marks, transcription, splicing, and 3-D contacts in each cell type; and then, by comparing the prediction for a reference and a variant sequence, read off the effect of the variant. A decade of models (DeepSEA, Basset, Basenji, Enformer, Borzoi, AlphaGenome) has pushed this program from kilobases to megabases and from a handful of assays to thousands of tracks. This chapter formalizes the task, traces what each generation relaxed, and then examines the central tension: **models trained to predict reference-genome tracks across the genome are being used to predict the effect of single-nucleotide edits within a locus**, which is a different, counterfactual quantity that the training data may not identify.
    **Prerequisites.** Chapters 10–12, 16, 18, 22, 25–29.
    **You will be able to:** (1) specify the input, output, loss, and tensor shapes of a sequence-to-function model; (2) describe the lineage of models by the assumption each one relaxed; (3) distinguish three levels of evaluation (across positions, across genes, across individuals/variants) and say which the literature supports; (4) explain, with an identifiability argument and a controlled experiment, why predictive accuracy does not imply correct variant effects; (5) design evaluations and data (single-edit experiments) that test and repair this; (6) dissect Enformer and AlphaGenome with the fixed template.

---

## 31.1 The task

**Input.** A genomic window $x\in\{0,1\}^{4\times L}$ (one-hot; Chapter 28), with $L$ from $10^3$ to $10^6$ bases. **Output.** A matrix $\hat y\in\mathbb R^{(L_\text{out}/b)\times T}$ of **tracks**: $T$ experimental readouts (DNase/ATAC accessibility, ChIP-seq for histone marks and TFs, CAGE, RNA-seq coverage, splice-site usage, contact maps) at resolution $b$ (128 bp for Enformer, 32 bp for Borzoi, 1 bp for AlphaGenome's finest outputs), each measured in a specific cell type or tissue. The **cell type is an index of the output**, not an input.

**Loss.** For count-like coverage tracks, a Poisson (or negative-binomial) negative log-likelihood per bin and track, $\sum_{t,j}\big(\hat\mu_{tj}-y_{tj}\log\hat\mu_{tj}\big)$; for profile shapes, a multinomial likelihood plus a separate loss on total counts (BPNet); for contact maps, a mean-squared error on log-transformed, distance-normalized maps.

**Tensor shapes (Enformer-style).** $x$: $(B,4,196{,}608)$ $\to$ convolutional stem and pooling $\to$ $(B,C,1{,}536)$ (one position per 128 bp) $\to$ $L'=11$ transformer blocks with multi-head attention over the 1,536 positions (cost $O(L'^2d)$ with $L'=1{,}536$, tractable only because convolutions downsample 128-fold) $\to$ heads $\to$ $(B,896,T)$ for the central 114,688 bp (896 bins), with $T=5{,}313$ human tracks [[S]]. The central window is the output; the flanks are context that supplies information to the center.

**Why this is a plausible mapping.** The molecular phenotype at a locus in a given cell type is shaped by the DNA sequence (TF motifs and their syntax; Chapter 22) *and* by the cell's TF repertoire and chromatin state, which are not in the sequence. The models learn the mapping $x\mapsto\mathbb E[y\mid x,\text{cell type}]$, **marginalizing** the cell-type-specific factors into the output heads: *sequence determines a potential; the cell type selects which potential is realized*. This is the source of both their success (the sequence potential is learnable) and their limit (they cannot extrapolate to cell types with no training track).

---

## 31.2 The lineage: what each generation relaxed

| Year | Model | Input | Outputs | Key architectural step | Assumption relaxed |
|---|---|---|---|---|---|
| 2015 | **DeepBind** (Alipanahi et al.) | tens to hundreds of bp | binding of one TF/RBP | single convolution + pooling | learned motifs replace hand-built PWMs |
| 2015 | **DeepSEA** (Zhou & Troyanskaya) | 1 kb | 919 chromatin features | three convolutional layers; multi-task | motif combinations; variant scoring by in-silico mutagenesis |
| 2016 | **Basset** (Kelley et al.) | 600 bp | 164 DNase tracks | deeper CNN | quantitative accessibility across cell types |
| 2018 | **Basenji** (Kelley et al.) | 131 kb | thousands of tracks at 128 bp | **dilated** convolutions; coverage as Poisson target | long-range context; whole-locus coverage |
| 2018 | **ExPecto** (Zhou et al.) | 40 kb around TSS | expression (via a second model) | spatial transformation of CNN outputs | expression prediction from chromatin predictions |
| 2021 | **BPNet** (Avsec et al.) | 1 kb | base-resolution ChIP-nexus profiles | dilated residual CNN; profile + count heads | motif syntax from base-resolution data |
| 2021 | **Enformer** (Avsec et al.) | 196,608 bp | 5,313 human and mouse tracks at 128 bp | **transformer** on top of convolutions | receptive field from about 20 kb to about 100 kb effective |
| 2025 | **Borzoi** (Linder et al.) | 524 kb | RNA-seq coverage at fine resolution, many tracks | U-Net-style convolutions with attention | transcription, splicing, polyadenylation from one model |
| 2026 | **AlphaGenome** (Avsec et al.) | 1 Mb | 5,930 human tracks, 11 output types, up to 1-bp resolution | U-Net with transformer; multimodal | contacts, splicing, expression, accessibility, TF binding jointly |

(Dates: Enformer, *Nat. Methods* 18:1196–1203, 2021; Borzoi, *Nat. Genet.* 57:949–961, 2025; AlphaGenome, *Nature* 649, published online 28 January 2026 [[S]].) Each step relaxed one assumption of the previous generation, in the manner of Chapter 55's Scale and Representation attacks: more context (A6), more modalities (A2/A4), finer resolution (A2). The loss has stayed the same in spirit: **match reference-genome tracks across the genome, with held-out chromosomes as test**. This is the thread to keep in view.

!!! lens "Research lens: what the objective rewards"
    **Assumption it makes explicit:** the function from reference sequence to track is learned from *one genome* (plus mouse), each locus appearing once. **Information used:** between-locus variation in sequence and in track. **Information ignored:** within-locus variation among people (variants), because individuals' genomes are not in the training data. **Failure mode it predicts:** the model is accurate where the reference differs across loci, and uncertain about the effect of the *small edit* that distinguishes two alleles at one locus.

---

## 31.3 Three levels of evaluation

The word "accuracy" in this literature refers to at least three different tasks.

1. **Across positions within a cell type** (genome-wide track prediction on held-out chromosomes). Strong: Enformer's mean correlation between predicted and observed CAGE at transcription start sites in held-out genes increased from about 0.81 (Basenji2) to 0.85 [[S]]. *Caveats:* held-out chromosomes share repeats and paralogs with training chromosomes (Chapter 28), and tracks inherit mappability limits (Chapter 27).
2. **Across genes** (expression of different genes in one cell type from the reference sequence). The same: models explain much of the between-gene variance (reference-prediction Spearman correlations with observed expression of about 0.5–0.6 for Enformer, Basenji2, and ExPecto in one benchmark; Huang et al., 2023) [[S]].
3. **Across individuals and variants** (within a gene, how expression differs between people with different alleles; the sign and size of a variant's effect). This is what *variant interpretation* requires, and the evidence is **much weaker**: evaluated on paired genomes and RNA-seq of 421 people, four models (Enformer, Basenji2, ExPecto, Xpresso) had per-gene across-individual correlations distributed around zero (with tails), often predicted the wrong direction of the effect of cis-regulatory variants, and were outperformed by per-gene regularized regression on nearby variants (Huang et al., 2023) [[S]]. A companion analysis reached similar conclusions (Sasse et al., 2023) [[S]]. Independently, a study of Enformer found that most of its predictive signal came from promoter-proximal sequence, with sequence beyond about 30 kb contributing little to explained variance, despite distal elements known to have large effects (Karollus et al., 2023) [[S]].

The newest models report improvements on variant-effect benchmarks (Borzoi on eQTL and splicing scores; AlphaGenome reporting that it matches or exceeds the strongest external models in 25 of 26 variant-effect evaluations and outperforms specialized models in 22 of 24 track-prediction benchmarks) [[S]], but **these are the authors' evaluations**, and the within-gene, across-individual test of level 3, with LD-stratified and fine-mapped analysis (Chapter 26), is the independent test of whether the problem has been solved [[H]].

---

## 31.4 Why accuracy does not imply correct variant effects

### 31.4.1 An identifiability argument

Let $f_\theta$ be a model trained on pairs $(x_g,y_g)$, $g=1,\dots,G$, where $x_g$ are reference sequences at different loci (so one example per locus). Suppose the true function is additive in the features of the sequence, $y=\sum_k\beta_k\phi_k(x)+\varepsilon$, with $\phi_k(x)\in\{0,1\}$ indicating the presence of motif $k$. The effect of destroying motif $k$ at a locus is $-\beta_k$: a *counterfactual*, the change in $y$ when only $\phi_k$ changes. From the data, $\beta$ is estimated by regression, and **$\beta_k$ is identified only if $\phi_k$ varies independently of the other features**. For two features $a,b$ with correlation $\rho_{ab}$, the least-squares variance of $\hat\beta_A$ is
$$
\Var(\hat\beta_A)=\frac{\sigma^2}{n\,\Var(a)\,(1-\rho_{ab}^2)},
$$
inflated by the **variance inflation factor** $1/(1-\rho_{ab}^2)$: 2.8 at $\rho_{ab}=0.8$, 25 at $0.98$, and infinite at $\rho_{ab}=1$, where only the sum $\beta_A+\beta_B$ is identified. A flexible model with an implicit bias (minimum-norm, early stopping, weight decay) resolves the ambiguity *arbitrarily*, by splitting the identified sum according to its inductive bias: a regression with $a\equiv b$ and $\beta_A+\beta_B=0.2$ returns $\hat\beta_A=\hat\beta_B=0.1$ under the minimum-norm solution, predicting that deleting the repressor $B$ *lowers* expression when it truly raises it.

**In genomics the features are correlated:** motifs of cooperating TFs co-occur in composite elements; promoter and enhancer strengths co-vary; GC content, repeat class, and chromatin context correlate with many motifs; LD ties variants together (Chapter 26). The model is *accurate on the training distribution* because it learns the identified combination, and *unconstrained along the directions the data never separate*, which are exactly the directions that a single-nucleotide edit (destroying one half of a composite element) moves along. This is **a failure of identification in the data, not of the architecture or optimizer**: a larger model fits the same combination.

### 31.4.2 The experiment

`code/ch31_counterfactual.py` builds a toy locus (200 bp) with three motifs: an activator A (effect $+1.0$), a repressor B ($-0.8$), and an independent activator C ($+0.5$); noise SD 0.3. A and B form a **composite element**: with co-occurrence rate $c$, B accompanies A (adjacent) and appears without A with probability $1-c$, so that $\mathrm{corr}(a,b)=2c-1$. A small CNN (two convolutional layers, global max and mean pooling) is trained on 6,000 reference-like loci; we then evaluate (i) i.i.d. test error, and (ii) the **counterfactual**: for loci carrying both A and B, destroy A alone, or B alone, and compare the change in the model's prediction with the truth ($-1.0$ for deleting A, $+0.8$ for deleting B).

| Training loci | Co-occurrence $c$ | corr(A,B) | I.i.d. test RMSE ($R^2$) | Sign correct: delete A | delete B | Mean predicted effect: A (true $-1.0$) | B (true $+0.8$) |
|---|---|---|---|---|---|---|---|
| 6,000 | 0.50 | 0.00 | 0.338 (0.795) | 1.00 | 1.00 | $-1.04$ | $+0.78$ |
| 6,000 | 0.90 | 0.80 | 0.367 (0.450) | 1.00 | 1.00 | $-0.94$ | $+0.72$ |
| 6,000 | 0.99 | 0.98 | 0.377 (0.166) | 0.99 | **0.53** | $-0.22$ | $+0.01$ |
| 6,000 | 1.00 | 1.00 | 0.360 (0.189) | 0.71 | **0.03** | $-0.05$ | $-0.19$ |
| 40,000 | 0.99 | 0.98 | 0.314 (0.417) | 1.00 | 1.00 | $-0.53$ | $+0.35$ |
| 40,000 | 1.00 | 1.00 | 0.316 (0.381) | 1.00 | **0.00** | $-0.08$ | $-0.16$ |
| 6,000 + 100 single-edit | 1.00 | 1.00 | 0.362 (0.195) | 0.98 | 0.67 | $-0.22$ | $+0.04$ |
| 6,000 + 400 single-edit | 1.00 | 1.00 | 0.352 (0.223) | **1.00** | **1.00** | $-0.93$ | $+0.70$ |


Reading the table.

1. **I.i.d. accuracy hides the problem.** The test RMSE is 0.34–0.38 for every co-occurrence level, against a noise floor of 0.30; the apparent fall in $R^2$ (0.795 to 0.166) is because the total variance of $y$ shrinks as B cancels A (the composite $A{+}B$ has net effect only $+0.2$), not because errors grow. A reviewer reading RMSE would see no problem.
2. **Counterfactual accuracy collapses at high co-occurrence.** At $c=0.5$ (independent) the model recovers both effects ($-1.04$ and $+0.78$ against $-1.0$ and $+0.8$); at $c=0.9$ ($\rho_{ab}=0.8$) still ($-0.94$, $+0.72$); at $c=0.99$ ($\rho_{ab}=0.98$) the predicted effect of deleting A shrinks to $-0.22$ and of B to $+0.01$ (sign correct for B in only 53% of loci); at $c=1$ the effect of deleting B has the **wrong sign** in 97% of loci ($-0.19$ against $+0.8$). *The model has learned the sum, and the split is arbitrary.* The transition is sharp: between $c=0.9$ and $0.99$, the number of loci with A alone or B alone falls from about 300 each to about 30.
3. **More observational data helps only as it adds decorrelated examples.** Raising the training set from 6,000 to 40,000 loci at $c=0.99$ partly restores the effects ($-0.53$ and $+0.35$, signs correct in 100% of loci) because the number of loci carrying A without B, or B without A, rises from about 30 to about 200; the effects remain attenuated. At $c=1$ there are no such loci and more data does not help: deleting B is predicted at $-0.16$, with the correct sign in 0% of loci.
4. **A small number of single-edit experiments repairs it.** Adding single-edit experiments (half deleting A, half deleting B, in loci carrying both; the equivalent of a small MPRA) repairs the model at $c=1$: 100 edits partially ($-0.22$, $+0.04$; sign 0.98 and 0.67), 400 edits (6% of the training-set size) almost completely ($-0.93$, $+0.70$; signs 1.00 and 1.00), with no change in the i.i.d. RMSE (0.35–0.36). 400 targeted interventions did what 34,000 additional observational loci could not. This is the **A4 (data) attack** in its cleanest form: the missing information is *interventional variation along the edit direction*, not more of the same.

```python
--8<-- "code/ch31_counterfactual.py"
```

```text
true effects of destroying a motif: A -> -1.0, B -> +0.8 (destroying the repressor B raises expression); noise SD 0.3 (the RMSE floor).

training loci   co-occurrence c   corr(A,B)   i.i.d. test RMSE (R2)   sign correct: A deletion   B deletion   mean predicted effect: A (true -1.0)   B (true +0.8)
     6000           0.50           0.00        0.338 (0.795)            1.00                   1.00                  -1.04                           +0.78
     6000           0.90           0.80        0.367 (0.450)            1.00                   1.00                  -0.94                           +0.72
     6000           0.99           0.98        0.377 (0.166)            0.99                   0.53                  -0.22                           +0.01
     6000           1.00           1.00        0.360 (0.189)            0.71                   0.03                  -0.05                           -0.19
    40000           0.99           0.98        0.314 (0.417)            1.00                   1.00                  -0.53                           +0.35
    40000           1.00           1.00        0.316 (0.381)            1.00                   0.00                  -0.08                           -0.16
 6000 + 100 MPRA    1.00            1.00        0.362 (0.195)            0.98                   0.67                  -0.22                           +0.04
 6000 + 400 MPRA    1.00            1.00        0.352 (0.223)            1.00                   1.00                  -0.93                           +0.70
```

!!! rhyme "Structural rhyme: collinearity ↔ confounding ↔ unidentified counterfactuals"
    The variance inflation factor of a regression, confounding in causal inference (Chapter 44), the instrument-strength problem in Mendelian randomization (Chapter 26), and the unidentified edit effect here are the same phenomenon: **an effect cannot be separated from a correlate that it never varies independently of**. Training sequence models on observational genomes is regression with enormously many correlated features; the tools that fix it elsewhere (experiments, instruments, interventions) are the tools that fix it here.

**What the toy does and does not show.** It shows a *mechanism* by which accurate reference-trained models can be wrong about edits, with a clean threshold, and that interventional data remove it. It does *not* show that this is the explanation for the failures in Huang et al. (2023); other mechanisms (the architecture's limited use of distal context, the noise in eQTL labels, LD tags, reference-trained objectives that never see variation within a gene) may contribute, and Chapter 55's attack sheet and Chapter 56's information-gain example turn these into a ranked set of experiments [[P]].

---

## 31.5 Paper dissections

!!! paper "Paper dissection: Enformer (Avsec et al., *Nature Methods*, 2021)"
    **Problem.** Predict gene expression and chromatin tracks from DNA sequence while using distal regulatory elements (enhancers can lie 10–100 kb or more from a promoter), which convolutional models with receptive fields of about 20 kb could not.
    **Key insight.** Place **transformer** layers on top of a convolutional trunk so that every position can attend to every other position within the window regardless of distance: the effective receptive field increases about five-fold compared with dilated convolutions (to about 100 kb).
    **Architecture.** A convolutional stem with pooling (down to 128-bp bins, 1,536 positions for a 196,608-bp input), 11 transformer layers with relative positional encodings (to retain distance information), and species-specific heads predicting thousands of tracks. [[S]]
    **Objective.** Poisson loss over coverage tracks (human and mouse jointly), with training/validation/test split by chromosome regions.
    **Data.** Human (5,313 tracks) and mouse (1,643 tracks) CAGE, DNase/ATAC, histone ChIP-seq, TF ChIP-seq from ENCODE, Roadmap, FANTOM5. [[S]]
    **Evaluation.** Track-level correlations on held-out regions; correlation of predicted and observed CAGE at TSS across genes (0.85 vs 0.81 for Basenji2); eQTL effect-direction and fine-mapped eQTL analyses; MPRA and saturation-mutagenesis comparisons; attention analyses.
    **Why it worked.** Longer-range information (distal enhancers and promoter context) improved held-out-gene expression prediction, and attention gave a flexible way to use it; multi-task training across thousands of tracks regularizes the sequence representation.
    **Assumptions.** Reference-genome training data represent regulatory logic; tracks measured in bulk cell populations; held-out chromosomes are independent of training.
    **Limitations (documented later).** Predictions for variant effects across individuals are weak (Huang et al., 2023; Sasse et al., 2023); signal from beyond about 30 kb contributes little to explained variance (Karollus et al., 2023); cell types are output indices; coverage tracks inherit mappability limits [[S]].
    **What followed.** Borzoi (longer context, RNA-seq coverage); AlphaGenome (1 Mb, bp resolution, many modalities); fine-tuning on personal-genome data; new benchmark suites.
    **Unresolved.** Whether long context is *used* causally, and whether counterfactual effects are identified (§31.4).

!!! paper "Paper dissection: AlphaGenome (Avsec et al., *Nature*, published online 28 January 2026)"
    **Problem.** Interpret non-coding variants by predicting their effect on many regulatory modalities, at high resolution, over long context, with one model.
    **Key insight.** Combine a **U-Net-style** convolutional encoder–decoder (for base-pair resolution outputs) with **transformer** layers over the compressed representation (for megabase context), and train on many modalities and both human and mouse genomes, so that one model supplies expression, accessibility, histone marks, TF binding, splice-site and junction usage, and 3-D contacts.
    **Architecture and data (as reported).** Input: 1 Mb of DNA sequence and species identity; output: 5,930 human or 1,128 mouse tracks across 11 output types at resolutions from 1 bp to coarser bins; training data from ENCODE, GTEx, 4D Nucleome, and FANTOM5 [[S]].
    **Evaluation (as reported).** The authors report that the model outperformed the strongest external models in 22 of 24 benchmarks for sequence-based prediction of regulatory activity, and matched or exceeded them in 25 of 26 variant-effect evaluations; the model is available through an API [[S]].
    **Why it might work.** Longer context and multimodal training provide the information that single-modality, shorter-context models lack: for instance, predicting splicing and expression jointly constrains the representation, and base-pair resolution lets the model use motif-level detail.
    **Assumptions.** As Enformer: reference-genome training; cell types as outputs; tracks as proxies of function.
    **Limitations (to be tested independently).** (a) Whether the *within-gene, across-individual* test (§31.3 level 3) improves; (b) whether distal context is causally used (shuffle-context and enhancer-deletion tests; Karollus et al., 2023); (c) counterfactual identifiability (§31.4); (d) benchmark selection and overlap with training data; (e) generalization to unseen cell types and species.
    **What followed.** Early external assessments and perspectives on generalizable and interpretable regulatory AI; as of this writing the independent evaluations lag behind the model release [[X]] for any claim beyond the authors' benchmarks.
    **Unresolved.** The central questions of this chapter: are variant effects *identified*, and does the added context and modality convert predictive accuracy into counterfactual accuracy?

---

## 31.6 Worked research examples

!!! example "Worked Research Example 31.1: \"Our model predicts the direction of eQTL effects with 71% accuracy\""
    **Situation.** A new sequence-to-function model is evaluated on 5,000 fine-mapped eQTL variants (posterior inclusion probability above 0.9). It predicts the sign of the effect correctly for 71% of variants, versus 64% for a previous model. The authors conclude that it "captures regulatory grammar."

    **Question.** What does 71% mean, and what would you ask?

    **Reasoning.**

    1. *Base rate and the null.* Sign accuracy of 50% is chance, but *simple baselines* may already score above it: distance to the TSS, the sign of the nearest enhancer annotation, or conservation. A per-gene linear model (PrediXcan-like) has a known level. Compare to those.
    2. *Selection of the test set.* High-PIP variants are *enriched near TSSs and in strong motifs*, which are the easiest. Stratify by distance to TSS, by predicted-effect magnitude (accuracy usually rises with predicted magnitude: report the curve, not one number), and by LD (a tag variant in perfect LD with the causal one could show the correct sign for the wrong reason: Chapter 26).
    3. *Tissue and context.* The eQTL tissue must match a model output; a mismatched tissue lowers accuracy for reasons unrelated to sequence.
    4. *Identifiability* (§31.4): are the variants in the test set in composite elements? If so, accuracy there is the signal of identified effects; for unidentified, the model's sign is determined by its inductive bias and is not evidence for or against grammar.
    5. *Statistics.* The unit of replication is the **locus** (variants in one credible set are not independent); use a bootstrap over loci. With 5,000 variants at 71% the binomial standard error is 0.6%, but across loci it is larger.
    6. *Decisive experiment.* Prospective MPRA or base-editing of variants in the *uncertain* tertile, chosen before seeing outcomes; compare model-chosen versus baseline-chosen sets at equal cost (C4, Chapter 1).

    **Expert analysis.** The headline number is a mixture of easy cases and baselines. The question that matters for clinical use is *accuracy conditional on a high predicted magnitude and a novel locus*, which a single figure hides.

!!! example "Worked Research Example 31.2: Extending the context to a megabase did not improve expression prediction"
    **Situation.** You extend a model's context from 200 kb to 1 Mb and find no improvement in held-out-gene expression correlation.

    **Question.** Three explanations, and how to discriminate them.

    **Reasoning (the Expert Chain).**

    1. *Hypotheses.* **H1:** distal effects are weak or context-specific (biology); **H2:** the training data do not identify them (data/identifiability, as §31.4); **H3:** the architecture does not use the extra context (optimization or capacity).
    2. *Cheapest discriminators.* A *context-masking test* (replace the distal 80% of the window with shuffled sequence at inference and see whether predictions change): if predictions barely change, the model is not using distal context (H3 or learned irrelevance); in Chapter 56's computation this has the highest information per cost. Then *enhancer-deletion* at known strong enhancers (from CRISPRi screens or ABC predictions) and comparing the model's response with measured effects; and an *MPRA of enhancer–promoter pairs* to check whether distal effects are strong in reporter assays.
    3. *Interpretation by cases.* If masking changes the prediction a lot but deletions are not predicted right, the model uses context but has the wrong effect: **H2** (counterfactual not identified). If masking does nothing: **H3** or **H1**. If CRISPRi shows weak distal effects: **H1**.
    4. *Remedy per hypothesis.* H2: add interventional data (A4) or a within-gene objective (A3). H3: change the architecture, or auxiliary losses requiring distal context (A2, A3). H1: the context scale-up was the wrong attack (A6 without a limiting resource).

    **Expert analysis.** An unproductive scale-up is informative only if the three hypotheses are separated. The chapter's toy shows that H2 can produce *exactly* this null result on an architecture that has the capacity, so a larger window should not be taken as evidence of a better model until the counterfactual tests are done.

---

## 31.7 Researcher's Notebook

!!! notebook "Researcher's Notebook: evaluating a sequence-to-function model"
    **Setting.** You will rely on a published model to score variants.

    1. **Three levels.** Report across-position, across-gene, and **across-individual, within-gene** performance. If only the first two are reported, treat variant effect claims as untested.
    2. **Ceilings.** For each track, the replicate correlation (Chapter 25's reliability); for eQTL labels, the fine-mapping resolution (Chapter 26).
    3. **Baselines.** Conservation; distance to TSS; PrediXcan-type per-gene regression; a model trained on one-hot input with the same data and tuning (Chapter 29).
    4. **Stratification.** By distance to TSS, LD with the lead variant, PIP, predicted magnitude, mappability and repeat class (Chapter 27), and cell type.
    5. **Counterfactual test.** In a locus with a known composite element, do single edits give effects consistent with experiments? Add the toy of §31.4 to your unit tests.
    6. **Context use.** Mask or shuffle the distal context and measure the change; compare to known distal effects.
    7. **Leakage.** Homology between training and test regions in both orientations (Chapter 28).
    8. **Cell-type limit.** State that the model cannot extrapolate to cell types with no training track; if you need a new cell type, use a model conditioned on that cell type's accessibility or expression.

    **What it teaches.** The best single test of such a model is a *prospective* experiment on variants chosen before the result, with a baseline-chosen set as control.

    **An open question to carry forward.** Could a **within-locus objective** repair the identifiability problem *without new experiments*, for example by training on population data (haplotypes and allele-specific expression: each heterozygote gives a within-individual, within-cell, within-gene contrast with the genetic background held fixed)? Formalize the identifying variation, estimate how many heterozygote observations per locus are needed to separate two motifs in a composite element at a given co-occurrence rate (using the VIF argument of §31.4), and test it in the toy with simulated haplotypes.

---

## 31.8 Connections

- **Backward:** convolutions and syntax (Chapter 10); long context and attention (Chapters 11, 12); attribution and in-silico mutagenesis (Chapter 18); thermodynamic regulation (Chapter 22); mappability and tracks (Chapter 27); tokenization and RC symmetry (Chapter 28); classical baselines (Chapter 29); fine-mapping, LD, and eQTL labels (Chapter 26); idea generation and triage (Chapters 55–56).
- **Forward:** genomic language models as an alternative (Chapter 32); RNA, splicing, and translation models (Chapter 33); genotype-to-phenotype models (Chapter 41); benchmarks and leakage (Chapter 43); causal inference and identification (Chapter 44); distribution shift across ancestry and cell type (Chapter 45); experimental design for variant effects (Chapter 46); mechanistic interpretability (Chapter 48); open problems for genomes (Chapter 50).

!!! takeaways "Key takeaways"
    1. A sequence-to-function model maps a long one-hot window to hundreds or thousands of cell-type-specific tracks; the **cell type is an output index**, so the model cannot extrapolate to unseen cell types.
    2. The lineage from DeepSEA to AlphaGenome relaxed context (1 kb to 1 Mb), resolution (128 bp to 1 bp), and modality count; the **objective has not changed**: match reference-genome tracks with held-out chromosomes.
    3. Evaluation has three levels; across positions and genes the evidence is strong, across individuals (within-gene) it is weak: per-gene cross-individual correlations were centered near zero for four models in 421 individuals, and linear regression on nearby variants did better.
    4. **Single-edit effects are counterfactuals identified only if the feature varies independently of its correlates**; the variance inflation factor $1/(1-\rho^2)$ diverges with collinearity.
    5. In a controlled toy, i.i.d. RMSE was constant (0.34–0.38) while counterfactual accuracy collapsed with composite-element co-occurrence ($c=0.9$: effects recovered; $c=0.99$: deletion of B predicted $+0.01$ versus $+0.8$; $c=1$: wrong sign in 97% of loci).
    6. **Single-edit (MPRA-like) experiments repair it**: 400 edits (6% of the training set size) restored the effects at $c=1$ ($-0.93$ and $+0.70$), where 40,000 additional observational loci did not.
    7. Larger context, more modalities, and bigger models do not by themselves repair non-identification; the A4 (interventional data) and A3 (within-locus objective) attacks do.
    8. **Independent, prospective, within-gene tests** (not the authors' own benchmark) are the evidence needed for the claims that new models make.

---

## Further reading

- Alipanahi, B., Delong, A., Weirauch, M. T. & Frey, B. J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nat. Biotechnol.* 33, 831–838. Zhou, J. & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning-based sequence model. *Nat. Methods* 12, 931–934. Kelley, D. R., Snoek, J. & Rinn, J. L. (2016). Basset. *Genome Res.* 26, 990–999. Kelley, D. R. et al. (2018). Sequential regulatory activity prediction across chromosomes with convolutional neural networks. *Genome Res.* 28, 739–750. Zhou, J. et al. (2018). Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk. *Nat. Genet.* 50, 1171–1179.
- Avsec, Ž. et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nat. Genet.* 53, 354–366. Avsec, Ž. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nat. Methods* 18, 1196–1203. Linder, J., Srivastava, D., Yuan, H., Agarwal, V. & Kelley, D. R. (2025). Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. *Nat. Genet.* 57, 949–961. Avsec, Ž. et al. (2026). Advancing regulatory variant effect prediction with AlphaGenome. *Nature* 649.
- Karollus, A., Mauermeier, T. & Gagneur, J. (2023). Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers. *Genome Biol.* 24, 56. Huang, A. C. et al. (2023). Personal transcriptome variation is poorly explained by current genomic deep learning models. *Nat. Genet.* 55, 2056–2059. Sasse, A. et al. (2023). Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings. *Nat. Genet.* 55, 2060–2064.
- Chen, K. M., Wong, A. K., Troyanskaya, O. G. & Zhou, J. (2022). A sequence-based global map of regulatory activity for deciphering human genetics. *Nat. Genet.* 54, 940–949. Fudenberg, G. et al. (2020). Predicting 3D genome folding from DNA sequence with Akita. *Nat. Methods* 17, 1111–1117. Zhou, J. (2022). Sequence-based modeling of three-dimensional genome architecture from kilobase to chromosome scale (Orca). *Nat. Genet.* 54, 725–734.
