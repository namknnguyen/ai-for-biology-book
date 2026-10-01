# Chapter 30. Single-Cell Methods Before Foundation Models

!!! abstract "Chapter at a glance"
    **Motivation.** The single-cell foundation models of Chapter 38 enter a field with a mature toolkit: normalization, dimension reduction, graphs and clustering, probabilistic latent-variable models, batch integration, trajectory inference, annotation, and regulatory-network inference. Each makes an assumption about what the data are and what variation is "biology". A foundation model that claims to improve on them must be compared with them, and must avoid their failure modes. This chapter derives the main methods, tests the most consequential ones on simulated data with known cell types, a known trajectory, and batch effects, and states which conclusions are robust and which are artifacts of method choice.
    **Prerequisites.** Chapters 2, 8, 13, 14, 25.
    **You will be able to:** (1) describe the standard workflow and the assumption each step encodes; (2) derive the negative-binomial variational autoencoder (scVI-type) and say what its latent variable and its "denoised" output mean; (3) formulate batch integration as an identifiability problem and explain why composition differences make *batch mixing* a misleading objective; (4) state what trajectory inference and RNA velocity assume; (5) name the failure modes of annotation, imputation, and regulatory-network inference; (6) design controls that detect overcorrection and false signals.

---

## 30.1 The standard workflow, and what each step assumes

| Step | Typical implementation | What it assumes |
|---|---|---|
| Quality control | Filter by UMI count, genes detected, mitochondrial fraction; remove doublets and ambient RNA (Chapter 25) | Low-quality droplets differ systematically from cells; mitochondrial-high cells are dying (not always: cardiomyocytes) |
| Normalization | Divide by size factor, take $\log(1+x)$ | Total counts reflect technical capture, not biology; most genes do not change (Chapter 25) |
| Feature selection | Highly variable genes (2,000–5,000) | Variance above the mean–variance trend is biological |
| Dimension reduction | PCA (10–50 components) | Biological variation is low-dimensional and well described by linear directions of high variance |
| Graph and clustering | $k$-nearest-neighbor graph; Leiden/Louvain community detection | Cell types are connected clusters in expression space; the resolution parameter is a choice |
| Visualization | UMAP or t-SNE | None that is metric: these are *pictures*, not measurements |
| Marker genes, annotation | Differential expression; reference mapping | Types are separable by few genes; the reference contains the types present |
| Trajectories | Pseudotime on a graph; RNA velocity | Cells lie on a continuum sampled densely; for velocity, a kinetic model (§30.5) |
| Batch integration | Regression, MNN, Harmony, scVI | Batch effects are simpler than biology; cell types are shared (§30.4) |

**PCA** is optimal for Gaussian noise (Chapter 2); counts are not Gaussian, but the log transform approximately stabilizes the variance. A systematic comparison found that a shifted logarithm of size-factor-normalized counts performs about as well as more elaborate variance-stabilizing alternatives (Ahlmann-Eltze & Huber, 2023) [[S]]. **Leiden** clustering maximizes modularity-like objectives with a resolution parameter; *there is no unique number of clusters*, so "the number of cell types" is a function of that parameter and of the graph. **UMAP and t-SNE** embed the neighbor graph in two dimensions and do not preserve global distances, densities, or cluster sizes; they are useful for visual checks, but statements about distances or continuity in the embedding are not statements about the data (Kobak & Berens, 2019; Chari & Pachter, 2023) [[S]].

---

## 30.2 A negative-binomial variational autoencoder

Chapter 14 derived the VAE; Chapter 25 derived the count model. Their combination, **scVI** (Lopez et al., 2018) [[E]], is the dominant probabilistic model for single-cell counts.

**Generative model.** For cell $n$ with batch label $s_n$ and observed library size $\ell_n=\sum_gy_{ng}$:
$$
z_n\sim\mathcal N(0,I_d),\qquad \rho_n=\mathrm{softmax}\big(f_\theta(z_n,s_n)\big)\in\Delta^{G-1},\qquad y_{ng}\mid z_n\sim\mathrm{NB}\big(\mu=\ell_n\rho_{ng},\ \text{inverse dispersion }\vartheta_g\big),
$$
with $f_\theta:\mathbb R^{d+S}\to\mathbb R^G$ a neural network, $d$ the latent dimension (typically 10–30), and $\vartheta_g$ learned per gene. The softmax encodes the compositional nature of the data (Chapter 25); multiplying by the observed library size handles capture efficiency. (The original model treats the library size as latent with a learned prior; fixing it to the observed value is a common simplification.) The NB log-likelihood per gene is
$$
\log\mathrm{NB}(y\mid\mu,\vartheta)=\log\Gamma(y+\vartheta)-\log\Gamma(\vartheta)-\log y!+\vartheta\log\frac{\vartheta}{\vartheta+\mu}+y\log\frac{\mu}{\vartheta+\mu}.
$$

**Inference.** An encoder $q_\phi(z\mid y)=\mathcal N\big(\mu_\phi(\log(1+y)),\mathrm{diag}\,\sigma^2_\phi\big)$ and the evidence lower bound (Chapter 8)
$$
\mathcal L(\theta,\phi)=\mathbb E_{q_\phi(z\mid y)}\big[\log p_\theta(y\mid z,s)\big]-\mathrm{KL}\big(q_\phi(z\mid y)\,\Vert\,\mathcal N(0,I)\big),
$$
optimized by minibatch SGD with the reparameterization trick $z=\mu_\phi+\sigma_\phi\odot\varepsilon$. **Tensor shapes:** for a minibatch of $B$ cells, $y\in\mathbb N^{B\times G}$; encoder $B\times G\to B\times h\to B\times d$ (two heads, mean and log-variance); decoder $B\times(d+S)\to B\times h\to B\times G$ (logits); $\mu=\ell\cdot\mathrm{softmax}(\text{logits})$ is $B\times G$. The cost per minibatch is $O\big(Bh(G+d)\big)$ with $h\sim128$–$512$: dominated by the first and last layers, which is why $G$ is restricted to a few thousand highly variable genes.

**What the latent variable and the output mean.** The posterior mean $\mathbb E[z\mid y]$ is a *representation* of the cell with the count noise removed by construction; $\mathbb E[\rho\mid y]$ is a **denoised** expression estimate. Both are *model outputs*, not measurements: the denoising borrows strength across genes through the shared latent code, so it also *introduces dependence between genes* (§30.7). Differential expression can be done by comparing $\rho$ under two conditions, with Bayes factors derived from posterior samples; this inherits the replicate issue of Chapter 25 (cells are not independent). Extensions: scANVI (semi-supervised with labels), totalVI (RNA plus protein), and multi-omic models (Chapter 40).

---

## 30.3 Batch integration is an identifiability problem

Suppose cells from several experiments are to be combined. The data are $y_n=\mathcal M\big(\text{state}_n,\ \text{batch}_n\big)$: biology and batch both change the counts. We want a representation of state that does not depend on batch. *Without assumptions this is not possible*: any difference between two batches could be a difference of composition (biology) or a batch effect, and nothing in the data labels which.

**Assumptions that make it solvable** (each method uses some):

1. **The batch effect is simple**: low-rank, additive in log-expression, gene-wise multiplicative (a per-gene scale for each batch), constant across cell types.
2. **Cell types are shared** across batches (so that matching nearest neighbors across batches identifies the same type).
3. **The composition differences are modest** (so that centering or matching distributions removes batch, not biology).

When (3) fails, *removing everything that differs between batches removes biology*: a cell type present in only one batch is, to a method that regards all batch-specific differences as nuisance, indistinguishable from a batch effect.

**Methods.**

* *Regression* (limma) and *empirical-Bayes location–scale* (ComBat) remove a per-gene batch mean (and scale), assuming equal composition.
* **Mutual nearest neighbors** (MNN; Haghverdi et al., 2018) and **anchors** (Seurat; Stuart et al., 2019) use pairs of cells that are each other's nearest neighbors across batches to estimate local correction vectors; they assume shared types.
* **Harmony** (Korsunsky et al., 2019) iteratively clusters cells in PCA space with a penalty for batch diversity within clusters and applies a cluster-specific linear correction.
* **scVI / scANVI** model batch as a covariate of the decoder (§30.2); **Scanorama** performs a panorama stitching of overlapping batches.
* In a large benchmark of 16 methods and 68 method–preprocessing combinations on 85 batches across 13 integration tasks, scANVI, scVI, Scanorama, and others were among the best performers, with rankings depending on the task and on how *batch removal* and *biological conservation* were weighted (Luecken et al., 2022) [[S]].

The benchmark's two-sided scoring (batch removal *and* biological conservation) is the right frame: **batch mixing alone is not the goal**.

---

## 30.4 A controlled test: mixing versus preservation

`code/ch30_single_cell_methods.py` simulates two batches of 2,500 cells and 1,000 genes from the measurement model of Chapter 25: four cell types ($\sim30\%$ differential genes per type, effect SD 0.9 in log scale), a continuous **trajectory** within type 2 (100 genes that change with a latent time $t\in[0,1]$), gene-wise multiplicative **batch effects** (log-SD 0.35), NB noise (dispersion 0.3), and library size lognormal with batch 1 sequenced 1.6 times deeper. Crucially, **composition differs**: batch 0 contains types $\{0,1,2\}$ and batch 1 types $\{1,2,3\}$ (types 1 and 2 are shared; types 0 and 3 are batch-specific). We compare five embeddings, with mean (SD) over three independent simulated datasets: PCA on log-normalized counts; PCA after per-batch gene centering; an NB-VAE without batch information; an NB-VAE with batch as a **free decoder covariate** (input concatenated to $z$, as in scVI); and an NB-VAE with an **additive per-gene batch bias** in log-space (matching the true form of the batch effect).

Metrics: **own-batch neighbors** (fraction of the 30 nearest neighbors of a shared-type cell that come from its own batch: 0.5 means perfect mixing, 1.0 none); **label transfer** (a 15-NN classifier trained on batch 0 labels, tested on batch 1's shared types); **ARI** of $k$-means with $k=4$ against the true types; **trajectory $|\rho|$** (Spearman correlation between the first principal component of the type-2 cells' embedding and the true time $t$); and **batch-specific-type purity** (fraction of the neighbors of cells of types 0 and 3, which exist in one batch only, that are of the same type).

**3,000 UMI per cell (mean over the batches):**

| Method | Own-batch neighbors | Label transfer | ARI | Trajectory $\lvert\rho\rvert$ | Batch-specific purity |
|---|---|---|---|---|---|
| PCA, log-normalized | 1.00 (0.00) | 1.00 | 1.00 | 0.03 (0.03) | 1.00 |
| PCA after per-batch centering | 1.00 (0.00) | 1.00 | 1.00 | 0.08 (0.08) | 1.00 |
| NB-VAE, no batch covariate | 1.00 (0.00) | 1.00 | 1.00 | 0.30 (0.12) | 1.00 |
| NB-VAE, **free** batch decoder covariate | **0.61** (0.01) | 1.00 | **0.70** (0.01) | **0.95** (0.00) | **0.71** (0.11) |
| NB-VAE, **additive** per-gene batch bias | 0.99 (0.00) | 1.00 | **1.00** | 0.87 (0.05) | **1.00** |

**400 UMI per cell:**

| Method | Own-batch neighbors | Label transfer | ARI | Trajectory $\lvert\rho\rvert$ | Batch-specific purity |
|---|---|---|---|---|---|
| PCA, log-normalized | 1.00 (0.00) | 1.00 | 1.00 | 0.02 (0.02) | 1.00 |
| PCA after per-batch centering | 0.92 (0.03) | 1.00 | 1.00 | 0.05 (0.04) | 0.99 |
| NB-VAE, no batch covariate | 1.00 (0.00) | 1.00 | 1.00 | 0.18 (0.22) | 1.00 |
| NB-VAE, **free** batch decoder covariate | **0.53** (0.01) | 1.00 | **0.73** (0.05) | 0.67 (0.17) | **0.57** (0.05) |
| NB-VAE, **additive** per-gene batch bias | **0.53** (0.01) | 1.00 | **1.00** | 0.72 (0.07) | **1.00** |

Six observations.

1. **Cell-type labels are not the hard part here.** Label transfer is 1.00 for every method, because type differences (30% of genes at SD 0.9) exceed batch effects (SD 0.35); *a benchmark that scores only cell-type recovery cannot discriminate the methods*. The informative metrics are the other three.
2. **PCA misses the trajectory** (|ρ| 0.02–0.08): within type 2, the largest axes of variation are the batch difference, not the latent time. Fine-scale continuous structure is invisible when the leading components are dominated by nuisance variation.
3. **Modeling batch explicitly recovers the trajectory**: $|\rho|=0.95$ (free) and 0.87 (additive) at 3,000 UMI; 0.67 and 0.72 at 400 UMI. Lower depth degrades the recovered continuum, as the reliability analysis of Chapter 25 predicts.
4. **The free decoder covariate mixes batches and destroys biology.** Own-batch neighbors fall to 0.61 (0.53 at 400 UMI): excellent "integration" by the usual metric, but **ARI drops to 0.70–0.73** and the purity of the two batch-specific types falls to 0.71 and 0.57: cells of type 0 and type 3 are merged into other types. A decoder that may change *any* output as a function of batch can explain "type 0 exists only in batch 0" as a batch effect. **This is overcorrection, driven by composition differences**, and it is rewarded by a mixing metric.
5. **The additive per-gene bias preserves biology**: ARI 1.00 and purity 1.00 in both regimes, while recovering the trajectory. It encodes the *form* of the batch effect (per-gene multiplicative), which in our simulation is true. At 3,000 UMI the embedding is not batch-mixed (0.99) because nothing forces the encoder to discard batch information from $z$ (the decoder simply ignores it), whereas at 400 UMI, where counts are noisier, the embedding mixes (0.53). *Batch mixing in the latent is neither necessary nor sufficient for correct integration.*
6. **Per-batch centering fails when composition differs** (own-batch neighbors 1.00 and 0.92): subtracting each batch's mean subtracts the composition difference, not only the technical effect.

**What this does and does not show.** The additive model is correct *by construction* (it matches the simulator), so its success is an existence proof that a well-specified batch model *can* integrate without overcorrecting, not evidence that real batch effects are additive (they are not: capture, ambient RNA, and protocol effects interact with cell type). The deeper lesson is the **trade-off**: flexible batch models fit any difference, rigid ones may be wrong. **The form of the batch model is a hypothesis (A1/A9 in Chapter 55), and its identifiability depends on assumptions about composition**; metrics must reward preserving *batch-specific* biology (purity of types present in one batch only) and not only mixing. Real atlases contain such types, as when a tissue region is sampled in only one study.

```python
--8<-- "code/ch30_single_cell_methods.py"
```

```text

== mean UMI per cell about 3000 (batch 1 sequenced 1.6x deeper); 5000 cells, 1000 genes; batch 0 lacks cell type 3, batch 1 lacks cell type 0; mean (sd) over 3 simulated datasets ==
method                                     own-batch neighbours   label transfer     ARI (4 types)     trajectory |rho|   batch-specific-type purity
PCA, log-normalized, uncorrected            1.00 (0.00)          1.00 (0.00)       1.00 (0.00)       0.03 (0.03)         1.00 (0.00)
PCA after per-batch gene centering          1.00 (0.00)          1.00 (0.00)       1.00 (0.00)       0.08 (0.08)         1.00 (0.00)
NB-VAE without batch covariate              1.00 (0.00)          1.00 (0.00)       1.00 (0.00)       0.30 (0.12)         1.00 (0.00)
NB-VAE, batch as free decoder covariate     0.61 (0.01)          1.00 (0.00)       0.70 (0.01)       0.95 (0.00)         0.71 (0.11)
NB-VAE, additive per-gene batch bias        0.99 (0.00)          1.00 (0.00)       1.00 (0.00)       0.87 (0.05)         1.00 (0.00)
(own-batch neighbour fraction: 0.50 = perfect mixing of the shared cell types, 1.00 = no mixing)

== mean UMI per cell about 400 (batch 1 sequenced 1.6x deeper); 5000 cells, 1000 genes; batch 0 lacks cell type 3, batch 1 lacks cell type 0; mean (sd) over 3 simulated datasets ==
method                                     own-batch neighbours   label transfer     ARI (4 types)     trajectory |rho|   batch-specific-type purity
PCA, log-normalized, uncorrected            1.00 (0.00)          1.00 (0.00)       1.00 (0.00)       0.02 (0.02)         1.00 (0.00)
PCA after per-batch gene centering          0.92 (0.03)          1.00 (0.00)       1.00 (0.00)       0.05 (0.04)         0.99 (0.01)
NB-VAE without batch covariate              1.00 (0.00)          1.00 (0.00)       1.00 (0.00)       0.18 (0.22)         1.00 (0.00)
NB-VAE, batch as free decoder covariate     0.53 (0.01)          1.00 (0.00)       0.73 (0.05)       0.67 (0.17)         0.57 (0.05)
NB-VAE, additive per-gene batch bias        0.53 (0.01)          1.00 (0.00)       1.00 (0.00)       0.72 (0.07)         1.00 (0.00)
(own-batch neighbour fraction: 0.50 = perfect mixing of the shared cell types, 1.00 = no mixing)
```

---

## 30.5 Trajectories and RNA velocity

**Pseudotime.** Methods such as Monocle (Trapnell et al., 2014), diffusion pseudotime (Haghverdi et al., 2016), and PAGA (Wolf et al., 2019) order cells along a path or graph on the neighborhood structure and call the ordering a *pseudotime*. The assumptions: the population contains cells from *all stages* of a continuous process; the observed manifold is one-dimensional (or a tree) with a root you choose; and distance in the embedding reflects progress. A benchmark of 45 trajectory-inference methods found that the best method depends on the topology of the trajectory, that performance varies across datasets, and that scalability and usability also matter (Saelens et al., 2019) [[S]]. **Important caveats:** (i) pseudotime is *not time*: a population with a stationary distribution has no preferred direction; (ii) **doublets** and **batch** produce false intermediates (Chapter 25; in our simulation, about one in five "transitional" cells was a doublet); (iii) as shown in §30.4, **the leading components can be dominated by nuisance variation**, so trajectory recovery should be tested with controls.

**RNA velocity** (La Manno et al., 2018; Bergen et al., 2020) uses the ratio of unspliced to spliced reads to infer, per gene, whether expression is rising or falling, assuming a kinetic model $\dot u=\alpha-\beta u$, $\dot s=\beta u-\gamma s$ with constant splicing ($\beta$) and degradation ($\gamma$) rates and a transcription rate $\alpha$ that switches between states. The velocity is $\dot s=\beta u-\gamma s$. The fitted $\gamma$ is often estimated from the steady-state extremes of a gene's phase plane; assumptions of gene-wise constant rates, simple switching, and adequate coverage of both steady states are *frequently violated*. Reviews have documented inconsistencies and sensitivities (Bergen et al., 2021; Gorin et al., 2022) [[S]], and velocity arrows on an embedding often reflect the embedding and the smoothing rather than the biology. Treat velocity as a hypothesis generator and validate with independent measurements (metabolic labeling, lineage tracing; Chapter 25).

**Optimal transport** formulations (Waddington-OT; Schiebinger et al., 2019) couple time-course populations by minimizing transport cost, which is the structural rhyme with Chapter 15's flow matching: learn a map between marginal distributions without paired cells, with the assumptions that dynamics are smooth and that proliferation and death are modeled.

---

## 30.6 Annotation and reference mapping

**Annotation** assigns labels to clusters or cells: manually from marker genes, or by *reference mapping* (Azimuth; Hao et al., 2021), *classification* (CellTypist; Domínguez Conde et al., 2022), or *transfer learning* (scArches; Lotfollahi et al., 2022). Pitfalls: **annotation is a model output**, not a measurement, so benchmarks that use public annotations as truth measure agreement with a pipeline (Chapter 25). **Reference completeness**: cells whose type is absent from the reference are assigned to the nearest wrong type with high confidence (the "missing type" error of Chapter 25's deconvolution, in a different guise); good tools report *unassigned/low-confidence* cells and hierarchical labels. **Circularity**: if clusters are defined by expression and then tested for differential expression between clusters, the p-values are meaningless (selective inference): a gene that defined the cluster will appear differential. **Ontology granularity** varies between studies (a "T cell" in one atlas is "CD4 naive T" in another), and label noise between annotators is high; the ceiling for annotation benchmarks is annotator agreement, rarely reported.

---

## 30.7 Imputation, denoising, and false signals

Denoising methods (scVI decoder means, MAGIC, other imputations) replace noisy counts by smoothed estimates that borrow strength across cells and genes. They can improve visualization and clustering, but **they create dependence that is not in the data**: two genes that have no relationship will be correlated in the smoothed values if both are smoothed with the same neighbors or decoded from the same latent code (Andrews & Hemberg, 2018) [[S]]. Consequences: gene–gene correlation networks, co-expression modules, and regulatory-network inference computed on denoised values double-count the shared structure.

**Test** (`code/ch30_false_correlation.py`): within a single homogeneous population (cells of one type in one batch), genes outside the trajectory program are independent given library size.

```python
--8<-- "code/ch30_false_correlation.py"
```

```text
759 cells of one type in one batch; 476 expressed genes with no association to the true trajectory; 4000 random gene pairs
log-normalized counts : mean |r| = 0.029; fraction of pairs with |r| > 0.2 = 0.000
NB-VAE decoder means  : mean |r| = 0.513; fraction of pairs with |r| > 0.2 = 0.810
(sampling noise alone gives |r| about 0.029 for independent genes with 759 cells)
```

Among 759 cells of a single type in one batch, we took the 476 expressed genes that have no association with the true trajectory (they are independent given library size) and 4,000 random pairs of them. In **log-normalized counts** the mean absolute Spearman correlation is 0.029, equal to what sampling noise alone gives for 759 cells ($0.8/\sqrt{759}=0.029$), and **none** of the pairs has $|r|>0.2$. In the **VAE decoder means** the mean absolute correlation is **0.51** and **81% of the pairs exceed 0.2**. *Every gene is made correlated with every other through the shared latent code* (and, through the softmax, through the compositional constraint). A co-expression network built on these values would connect the whole genome. (Some of this dependence is a real compositional effect and some is noise absorbed into the latent code by the encoder, which sees the counts; the two cannot be separated from the denoised values alone.)

**Gene regulatory network inference** (SCENIC; Aibar et al., 2017) combines co-expression with motif enrichment to define regulons; benchmarks on synthetic and curated networks found that methods perform modestly (Pratapa et al., 2020, BEELINE) [[S]]: co-expression is not regulation, a single snapshot does not give the direction of influence, and without perturbation data causal structure is not identified (Chapters 25, 44).

---

## 30.8 Worked research examples

!!! example "Worked Research Example 30.1: An integrated atlas reveals a new cell state \"conserved across studies\""
    **Situation.** An atlas integrates six studies with a deep generative model and finds a cluster, absent from the original annotations, that appears in all six after integration, expresses an unusual combination of markers, and is enriched in disease. The authors name it a novel disease-associated state.

    **Question.** What are the measurement and method explanations, and what would establish the state?

    **Reasoning.**

    1. *Which method artifacts produce a cluster present in every batch after integration?* (a) **Overcorrection**: batch-specific cell types merged by a flexible batch model (§30.4: purity of batch-specific types fell to 0.57–0.71); a cluster formed by merging pieces of different types would "appear in all studies" and co-express markers of several types. (b) **Doublets** (Chapter 25: 19% of apparent intermediates) shared across studies because *every* droplet study has doublets. (c) **Ambient RNA** with a common source tissue. (d) **Annotation circularity** if the new cluster's markers were used to define it.
    2. *Diagnostics.* **Examine each study before integration:** is the state present as a distinct group in each *unintegrated* study, with the same markers? **Leave-one-study-out integration:** does the cluster persist when any study is withheld? **Composition-aware purity:** compute the batch-specific-type purity for known batch-specific types; a method that merges them cannot be trusted for new states. **Doublet scores** within the cluster; UMI distribution; co-expression of mutually exclusive markers.
    3. *Disease association.* Donors are the replicates (Chapter 25): with 6 studies and unequal numbers of donors per condition, the association must be tested at the donor level, with study as a random effect. Composition: is the abundance difference due to differences in dissociation or sorting between studies?
    4. *Decisive experiments.* Orthogonal validation of the markers by imaging or flow cytometry in new samples; sorting the putative state and profiling it in isolation; a perturbation or lineage experiment that places it in a developmental order.

    **Expert analysis.** The claim "conserved across studies" is *less* informative than it sounds, because the integration method was designed to make studies agree. Integration is a tool for *finding* candidate states; the evidence is in the unintegrated data and orthogonal assays.

!!! example "Worked Research Example 30.2: Gene regulation inferred from a denoised latent model"
    **Situation.** A group fits a VAE to a perturbation-free atlas, uses the decoder to impute expression, and builds a co-expression network from the imputed values. They report that transcription factor T is "co-regulated" with 120 genes and propose a regulon.

    **Question.** What does the network measure?

    **Reasoning.**

    1. *Where does the dependence come from?* The imputed values are functions of the latent code $z$; two genes affected by the same latent direction are correlated, whatever their relationship in the cell (§30.7). A latent axis that tracks cell type will connect all markers of that type.
    2. *Controls.* (a) **Null genes**: pairs of genes known to be independent given cell type (e.g., from different chromosomes with no known interaction) in the same data; the distribution of their correlations in raw versus imputed values is the "false-positive rate". (b) **Within-type analysis**: compute correlations within a single cell type or state, removing the type structure. (c) **Raw counts with an appropriate model** (e.g., partial correlation, NB models with covariates).
    3. *What would raise the claim?* A *perturbation*: knock down T and test whether the 120 genes respond (Chapter 25: Perturb-seq), with the power and reliability caveats of that chapter. The regulon claim is a C3–C4 claim; co-expression is at most a hypothesis.

    **Expert analysis.** Denoising is appropriate for visualization and sometimes for clustering; it is *not* a preprocessing step that is neutral for dependence-based inference. Use the unsmoothed data (with a proper count model) for tests about relationships.

---

## 30.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: auditing an integration or denoising step"
    **Setting.** Your analysis uses a batch-corrected or denoised representation.

    1. **Pre-integration check.** For each batch alone, cluster and annotate; list the types per batch (the *composition table*). Which types are batch-specific?
    2. **Two-sided metrics.** Compute batch mixing *and* preservation: purity of batch-specific types; conservation of within-batch cluster structure; trajectory/gradient recovery if relevant. A method must not lose batch-specific types.
    3. **Method as hypothesis.** Write what the batch model assumes (additive, multiplicative, low-rank, conditional on type). Is it plausible for your technology difference?
    4. **Leave-one-batch-out and composition perturbation.** Remove a type from one batch (or one batch) and see whether the integrated structure of the rest changes.
    5. **Null controls.** Genes known to be independent; random batch labels (a method that "finds" batch structure in random labels is overfitting); shuffled cell labels.
    6. **Downstream use.** If you will test differential expression, use donors as replicates on the *original counts* (Chapter 25); if you will infer networks, use raw counts or interventions.

    **What it teaches.** *Integration is a modeling choice with consequences for what can be discovered.* Methods that are best on average benchmarks can be wrong for *your* composition.

    **An open question to carry forward.** Could batch effects be identified by design instead of by assumption: a **bridge sample** (the same reference cells profiled in every batch), spike-in cell lines, or genetically multiplexed donors? How many bridge cells per batch are needed to identify a free-form batch effect, and does a foundation model trained on many batches *with* such controls learn batch-invariant biology better than one trained on the integrated data? Frame it with the identifiability argument of §30.3 and the experimental-design tools of Chapter 46.

---

## 30.10 Connections

- **Backward:** the NB measurement model and doublets/donors (Chapter 25); the VAE and ELBO (Chapters 8, 14); representation and invariance (Chapters 13, 16); PCA (Chapter 2).
- **Forward:** single-cell foundation models and their evaluation against these baselines (Chapter 38); perturbation prediction (Chapter 39); multimodal integration (Chapter 40); benchmark design (Chapter 43); causal inference from perturbations (Chapter 44); open problems in cell modeling (Chapter 51).

!!! takeaways "Key takeaways"
    1. The standard workflow encodes assumptions at every step; **UMAP/t-SNE are pictures**, and the number of clusters is a function of a resolution parameter.
    2. The **NB-VAE (scVI-type)** is a latent-variable model with a softmax-composition decoder and NB likelihood; its latent and denoised outputs are model outputs, not measurements.
    3. **Batch integration is an identifiability problem**: it requires assumptions about the form of the batch effect and about composition.
    4. In simulation with batch-specific cell types, a **free batch decoder covariate mixed batches (own-batch neighbors 0.61) but destroyed biology** (ARI 0.70; batch-specific-type purity 0.57–0.71), while an **additive per-gene batch bias** preserved it (ARI 1.00, purity 1.00) and recovered the trajectory (|ρ| 0.87 at 3,000 UMI).
    5. **PCA missed the trajectory** (|ρ| ≤ 0.08) because nuisance variation dominated the leading axes; per-batch centering failed when composition differed.
    6. **Batch mixing is a misleading objective**; report preservation (especially of batch-specific types) and test with leave-one-batch-out and null controls.
    7. **Pseudotime is not time; RNA velocity assumes constant kinetic rates**; both need orthogonal validation.
    8. **Annotation is a model output; denoising induces correlations; co-expression is not regulation**: use raw counts and perturbations for relational claims.

---

## Further reading

- Luecken, M. D. & Theis, F. J. (2019). Current best practices in single-cell RNA-seq analysis: a tutorial. *Mol. Syst. Biol.* 15, e8746. Heumos, L. et al. (2023). Best practices for single-cell analysis across modalities. *Nat. Rev. Genet.* 24, 550–572. Wolf, F. A., Angerer, P. & Theis, F. J. (2018). SCANPY. *Genome Biol.* 19, 15. Traag, V. A., Waltman, L. & van Eck, N. J. (2019). From Louvain to Leiden. *Sci. Rep.* 9, 5233.
- Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nat. Methods* 15, 1053–1058. Xu, C. et al. (2021). Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models. *Mol. Syst. Biol.* 17, e9620. Gayoso, A. et al. (2021). Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nat. Methods* 18, 272–282.
- Haghverdi, L., Lun, A. T. L., Morgan, M. D. & Marioni, J. C. (2018). Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors. *Nat. Biotechnol.* 36, 421–427. Stuart, T. et al. (2019). Comprehensive integration of single-cell data. *Cell* 177, 1888–1902. Korsunsky, I. et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat. Methods* 16, 1289–1296. Luecken, M. D. et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nat. Methods* 19, 41–50.
- Trapnell, C. et al. (2014). The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells. *Nat. Biotechnol.* 32, 381–386. Haghverdi, L., Büttner, M., Wolf, F. A., Buettner, F. & Theis, F. J. (2016). Diffusion pseudotime robustly reconstructs lineage branching. *Nat. Methods* 13, 845–848. Saelens, W., Cannoodt, R., Todorov, H. & Saeys, Y. (2019). A comparison of single-cell trajectory inference methods. *Nat. Biotechnol.* 37, 547–554. La Manno, G. et al. (2018). RNA velocity of single cells. *Nature* 560, 494–498. Bergen, V., Lange, M., Peidli, S., Wolf, F. A. & Theis, F. J. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nat. Biotechnol.* 38, 1408–1414. Bergen, V., Soldatov, R. A., Kharchenko, P. V. & Theis, F. J. (2021). RNA velocity: current challenges and future perspectives. *Mol. Syst. Biol.* 17, e10282. Schiebinger, G. et al. (2019). Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell* 176, 928–943.
- Hao, Y. et al. (2021). Integrated analysis of multimodal single-cell data. *Cell* 184, 3573–3587. Domínguez Conde, C. et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science* 376, eabl5197. Lotfollahi, M. et al. (2022). Mapping single-cell data to reference atlases by transfer learning. *Nat. Biotechnol.* 40, 121–130. Aibar, S. et al. (2017). SCENIC: single-cell regulatory network inference and clustering. *Nat. Methods* 14, 1083–1086. Pratapa, A., Jalihal, A. P., Law, J. N., Bharadwaj, A. & Murali, T. M. (2020). Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. *Nat. Methods* 17, 147–154. Andrews, T. S. & Hemberg, M. (2018). False signals induced by single-cell imputation. *F1000Research* 7, 1740. Kobak, D. & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. *Nat. Commun.* 10, 5416. Chari, T. & Pachter, L. (2023). The specious art of single-cell genomics. *PLoS Comput. Biol.* 19, e1011288.
