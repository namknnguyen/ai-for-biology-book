# Chapter 25. Single-Cell, Spatial, and Perturbation Measurement

!!! abstract "Chapter at a glance"
    **Motivation.** The cell-state foundation models of Chapters 38–39 are trained on count matrices, and what they can learn is bounded by how those matrices were generated. This chapter derives the **measurement model** of single-cell RNA sequencing, then the measurement models of spatial and perturbation data, and tests each with simulation. The goal is that you can look at a count matrix and say *which of its patterns are biology and which are the instrument*.
    **Prerequisites.** Chapters 4, 5, 8, 19 (the NB as Gamma–Poisson and as bursting), 22.
    **You will be able to:** (1) derive the single-cell count model (binomial capture of a bursty Gamma–Poisson process) and explain why "dropout" is mostly sampling; (2) compute the per-cell noise ceiling as a function of expression level; (3) derive and demonstrate the compositional artifact of library-size normalization; (4) explain how doublets, ambient RNA, and donor effects generate false "states" and false discoveries, with quantitative checks; (5) describe spatial technologies as trade-offs among resolution, gene coverage, and depth, and the spot-as-mixture model; (6) explain how Perturb-seq data are generated, how many cells a perturbation needs, and why the *reliability* of a perturbation's effect vector bounds any prediction metric.

---

## 25.1 The seven questions for a single-cell measurement

!!! bio "Biology for modeling: a single-cell measurement"
    **What is it?** A destructive snapshot of one cell's molecular content (usually mRNA, sometimes chromatin accessibility, surface proteins, or several at once), obtained after dissociating tissue into cells or nuclei, optionally with its position in tissue (spatial) or with a known genetic or chemical perturbation (perturbation screens).
    **Information it contains.** A small random sample (typically 5–20% in droplet methods) of the molecules in the cell at the instant of lysis, as a vector of integers over genes.
    **How is it generated?** Biology (transcription bursts, cell state, cell size, cell cycle) filtered by a sequence of physical steps: dissociation, encapsulation, lysis, reverse transcription with barcodes and unique molecular identifiers (UMIs), amplification, sequencing, and alignment.
    **How is it measured?** Droplet microfluidics (Drop-seq, inDrop, 10x Chromium), plates (Smart-seq), or split-pool combinatorial indexing (sci-RNA-seq, SPLiT-seq). Spatial: sequencing-based capture arrays and in-situ imaging.
    **Computational representation.** A sparse matrix $X\in\mathbb N^{n_\text{cells}\times n_\text{genes}}$, with metadata (donor, batch, annotations); an embedding; a graph of neighbors; a point cloud in space.
    **What varies.** Cell type and state (the signal), and also capture efficiency, cell size, ambient RNA, doublets, batch, donor, dissociation stress, and sequencing depth (the instrument).
    **What can ML learn?** Co-expression structure, cell-type and state manifolds, gene programs, and (with perturbations) associations between interventions and expression.
    **What can ML not observe?** Time (each cell is destroyed), most of the molecules in each cell, proteins and their modifications (in RNA-only data), spatial context (after dissociation), and the *counterfactual* (what this same cell would have looked like under a different perturbation).

---

## 25.2 Single-cell RNA-seq: from tissue to count matrix

### 25.2.1 The pipeline, and what each step costs

Tissue is dissociated (enzymatically or mechanically) to single cells or isolated as nuclei; each cell is captured with a barcoded bead in a droplet (or in a well, or by combinatorial barcoding in fixed cells); polyadenylated mRNA is reverse-transcribed with a cell barcode and a **UMI** (a random sequence marking each captured molecule, so PCR duplicates can be collapsed); cDNA is amplified and sequenced; reads are aligned and collapsed by (barcode, UMI, gene) into the count matrix. Representative landmarks: Drop-seq and inDrop (Macosko et al., 2015; Klein et al., 2015), commercial droplet systems (Zheng et al., 2017), combinatorial indexing (Cao et al., 2017; Rosenberg et al., 2018). Each step introduces a specific bias:

* **Dissociation** over-represents cells that survive and detach easily, induces stress-response genes (a signature that can masquerade as a biological state), and loses fragile cell types; **single-nucleus** protocols avoid some of this but sample nuclear RNA (unspliced transcripts are over-represented, cytoplasmic and mitochondrial content is lost).
* **Capture.** Only a fraction of mRNA molecules in a cell is converted to a barcoded cDNA that is sequenced. For droplet 3′ methods this *capture efficiency* is on the order of 5–20% per molecule [[S]], and it varies from cell to cell.
* **Amplification and sequencing** introduce length and GC biases; UMIs remove duplicate counting but not capture sampling.
* **Annotation** (assigning cell types by clustering and marker genes) adds human and algorithmic judgment: *labels in public atlases are model outputs*, not measurements.

### 25.2.2 The measurement model: binomial thinning of a bursty process

Let $m_{cg}$ be the (unobserved) number of mRNA molecules of gene $g$ in cell $c$. Chapter 19 showed that bursty transcription gives $m_{cg}\sim\mathrm{NB}$ with mean $\mu_g$ and dispersion $\phi_g$ (variance $\mu_g+\phi_g\mu_g^2$), equivalently Gamma–Poisson. Capture selects each molecule independently with probability $p_c$:
$$
y_{cg}\mid m_{cg}\sim\mathrm{Binomial}(m_{cg},p_c).
$$

!!! math "Thinning preserves the negative binomial"
    For $m\sim\mathrm{NB}(\mu,\phi)$ and $y\mid m\sim\mathrm{Bin}(m,p)$: $\mathbb E[y]=p\mu$, and by the law of total variance
    $$
    \mathrm{Var}(y)=\mathbb E[\mathrm{Var}(y\mid m)]+\mathrm{Var}(\mathbb E[y\mid m])=p(1-p)\mu+p^2(\mu+\phi\mu^2)=p\mu+\phi(p\mu)^2 .
    $$
    So $y\sim\mathrm{NB}(p\mu,\phi)$ with the *same* dispersion $\phi$ (the full Gamma–Poisson argument gives the distribution, not only the moments). Capture rescales the mean and leaves the *relative* overdispersion unchanged. Hence the probability of a zero is
    $$
    P(y=0)=(1+\phi\,p\mu)^{-1/\phi}\ \xrightarrow{\phi\to0}\ e^{-p\mu}.
    $$

**There is no need for an extra "dropout" or zero-inflation component in UMI data to explain the zeros.** In simulation (`code/ch25_measurement.py`; 5,000 genes, 2,000 cells, $p=0.1$, $\phi=0.2$, mean 13 molecules per gene per cell) the empirical variance divided by $\mu_y+\phi\mu_y^2$ is $1.000$ for genes with mean above 1 UMI, and the observed zero fraction equals the NB prediction in every expression bin (for example 0.914 vs 0.914 at mean 0.02–0.2 UMI; 0.249 vs 0.249 at 1–3 UMI), with no excess. The Poisson prediction is too low at higher means (0.204 vs 0.249 at 1–3 UMI; 0.015 vs 0.041 at 3–10) because it ignores biological overdispersion. This reproduces, as an algebraic fact about the model, the empirical conclusion of Svensson (2020) that droplet UMI data are consistent with NB sampling without zero inflation [[S]]; it does not prove that no real data have additional zero excess (full-length plate protocols and ambient RNA contamination can), and the simulation is a *model check*, not a data analysis.

!!! rhyme "Structural rhyme: capture ↔ dropout in neural networks ↔ compressed sensing ↔ undersampling"
    Binomial thinning is a lossy random subsampling of a vector of counts. It has the same algebraic role as **dropout** in neural networks and as *undersampling in compressed sensing*: the observed object is a random projection of the latent one. A model of the data must either learn the latent (as scVI does with a decoder + NB likelihood; Chapter 30) or become invariant to the sampling (rank-based or binarized inputs; Chapter 38).

### 25.2.3 The per-cell noise ceiling

For a gene with true mean molecule count $\mu$ and capture $p$, the biological cell-to-cell signal in the expected count $p\,m$ has variance $p^2\phi\mu^2$, and the total variance of $y$ is $p\mu+\phi p^2\mu^2$. Let $\lambda=p\mu$ be the mean UMI count. The fraction of variance in a single cell's observed count that reflects the cell's true molecule number, i.e. the **per-cell reliability** (and the maximum $R^2$ any model could achieve against a single-cell count), is
$$
\rho(\lambda)=\frac{\phi\lambda^2}{\lambda+\phi\lambda^2}=\frac{\phi\lambda}{1+\phi\lambda}.
$$
At $\phi=0.2$: $\lambda=0.1\Rightarrow\rho=0.02$; $\lambda=1\Rightarrow0.17$; $\lambda=10\Rightarrow0.67$. In simulation, averaging over genes in each bin gives:

| Mean UMI per cell | Genes | Observed zeros | NB-predicted zeros | Poisson-predicted zeros | Per-cell reliability $\rho$ |
|---|---|---|---|---|---|
| 0.02–0.2 | 1,857 | 0.914 | 0.914 | 0.913 | 0.018 |
| 0.2–0.5 | 945 | 0.731 | 0.731 | 0.723 | 0.061 |
| 0.5–1 | 649 | 0.518 | 0.518 | 0.495 | 0.124 |
| 1–3 | 730 | 0.249 | 0.249 | 0.204 | 0.252 |
| 3–10 | 361 | 0.041 | 0.041 | 0.015 | 0.497 |
| >10 | 116 | 0.001 | 0.001 | 0.000 | 0.772 |

*Consequence.* **Most genes in most cells are measured with reliability below 0.25**; 56% of genes in this simulation have mean below 0.5 UMI. Per-cell, per-gene predictions (what a "perfect" generative model of single-cell counts would be scored on) are limited by a noise ceiling that is a property of the instrument, and the model's likelihood cannot fall below the entropy of the sampling noise. (This is the within-population ceiling: variation *between* cell types is much larger relative to noise and is what makes clustering feasible.) It also explains the success of aggregation: averaging $n$ cells of one type raises the reliability of a mean from $\rho$ to $n\rho/(1+(n-1)\rho)$, so that a pseudobulk of 100 cells has reliability above 0.95 for a gene with $\rho=0.17$ (the Spearman–Brown formula, Chapter 4).

### 25.2.4 Normalization and the compositional trap

Because $p_c$ (and total RNA content) vary between cells, raw counts are not comparable across cells; the standard remedy divides each cell's counts by a **size factor** (typically the total UMI count, "counts per 10,000"), then takes $\log(1+x)$. Shifted-logarithm transforms are simple and perform comparably to more elaborate variance-stabilizing alternatives in a published comparison (Ahlmann-Eltze & Huber, 2023) [[S]]. But library-size normalization converts counts into **shares** that sum to a constant, so an increase in *some* genes lowers the shares of all others.

!!! math "The compositional artifact"
    Let a set of genes carrying a fraction $f$ of the molecules be multiplied by $\rho$ between two conditions, with all other genes unchanged in absolute number. The new total is $1+f(\rho-1)$ (relative to the old), so every unchanged gene's *share* is multiplied by $1/(1+f(\rho-1))$ and its apparent log$_2$ fold change is
    $$
    \log_2\mathrm{FC}_\text{apparent}=-\log_2\big(1+f(\rho-1)\big).
    $$
    For $f=0.3$, $\rho=3$: $-\log_2(1.6)=-0.68$.

In simulation, 52 genes carrying 30% of the molecules were tripled in condition B; the other 2,234 well-detected genes were unchanged. CPM normalization gave a median apparent log$_2$FC of **−0.68** for the unchanged genes (91% of them below −0.5, i.e. "down-regulated"), exactly as derived; a **median-ratio** normalization (the DESeq2-style assumption that most genes are unchanged) gave +0.00. *No normalization is assumption-free*: median-ratio fails when more than half of the genes change; spike-ins or cell counting are needed to recover *absolute* change. This is the molecular version of the "closure" problem for compositional data, and it matters for foundation models because **a model trained on normalized shares inherits the compositional coupling** and may represent "gene A is up" as "all other genes are down".

### 25.2.5 Sampling: how many cells do you need?

To see at least $k$ cells of a type with frequency $f$ with probability $0.95$, we need $n$ such that $P(\mathrm{Bin}(n,f)\ge k)\ge0.95$. For $k=20$: $f=5\%$ needs about 565 cells; $1\%$ needs 2,853; $0.1\%$ needs 27,924; $0.01\%$ needs 280,228. *A rare cell type present at 1 in 10,000 requires a quarter of a million cells just to see 20 of them*, before any statistical power for comparison between conditions. This is why atlases are scaled to $10^6$–$10^8$ cells (Chapter 38), and why rare-state discoveries need enrichment or targeted capture.

### 25.2.6 Doublets, ambient RNA, and false states

Droplet loading is Poisson: with mean $\lambda$ cells per droplet, the fraction of occupied droplets holding two or more cells is $(1-e^{-\lambda}-\lambda e^{-\lambda})/(1-e^{-\lambda})\approx\lambda/2$ for small $\lambda$. (Commercial systems report multiplet rates on the order of 0.4–0.9% per 1,000 recovered cells, rising linearly with loading [[S]].) A heterotypic doublet (cells of two types in one droplet) has the **sum** of two profiles; after normalization it lies between the two types. **Ambient RNA** (free mRNA from lysed cells captured in every droplet) adds a background profile to every cell and creates apparent expression of marker genes in the wrong cell types. Empty-droplet calling (Lun et al., 2019), ambient removal (Fleming et al., 2023), and doublet detection (Wolock et al., 2019; McGinnis et al., 2019) are standard preprocessing.

We simulated a continuum from cell type A to B, with 4,000 cells distributed so that most sit near the two ends (547 truly intermediate cells), plus 240 doublets (6%) formed from an A-end and a B-end cell, with realistic variation in cell size:

| Quantity | Result |
|---|---|
| Cells that look intermediate in the expression embedding | 612 |
| of which, doublets | **19%** |
| Fraction of all doublets landing in the intermediate region | **48%** |
| Total-UMI ratio, doublets / singlets | 2.14× |
| AUROC for doublet detection from total UMI alone | 0.934 |
| AUROC of a synthetic-doublet neighbor score (Scrublet-style), overall / within the intermediate region | 0.967 / 0.947 |

*Roughly one in five "transitional cells" in this setting is a doublet.* A trajectory-inference method or a foundation model trained on such data learns a continuum that is partly an artifact of the encapsulation process. The remedy is quantitative (detect doublets by simulation; check that intermediates have a *singlet* UMI distribution and co-express markers coherently) and, crucially, *orthogonal*: validate intermediate states by imaging (single-molecule FISH), lineage tracing, or hashing (cell-hashing multiplexes samples, so heterotypic doublets between differently hashed samples are identified directly).

### 25.2.7 Donors, not cells, are the replicates

Cells from one donor share a genome, environment, processing batch, and sampling noise; they are not independent samples of "the population". Treating them as independent replicates in a case–control comparison is **pseudoreplication** (Chapter 4). We simulated 6 donors (3 case, 3 control), 400 cells each, 1,000 genes, with a donor-level random effect (SD 0.35 on the log scale) on every gene and 50 truly differential genes ($\log_2\mathrm{FC}=0.6$):

| Analysis | False-positive rate at $p<0.05$ (950 null genes) | Power on 50 true DE genes |
|---|---|---|
| Cells as replicates (Welch $t$, 1,200 vs 1,200 cells) | **0.696** | 0.860 |
| Donors as replicates (pseudobulk, 3 vs 3) | 0.031 | 0.160 |

At $p<0.001$, cell-level testing still calls 51% of null genes, versus none at the donor level. Squair et al. (2021) documented the same problem on real data (single-cell methods that ignore donor structure produced spuriously many differential genes, and pseudobulk or mixed-model methods agreed better with matched bulk data) [[S]]. The price of correctness is power: with three donors per arm the effective sample size is 6, no matter how many cells are sequenced. **Increasing the number of cells per donor does not repair a design with few donors; only more donors does.** This should be considered in the design of every single-cell atlas and every claim of a disease-associated cell state.

---

## 25.3 Beyond RNA: other modalities and what none of them see

| Modality | What is measured | Typical sparsity / noise | Cannot see |
|---|---|---|---|
| scRNA-seq / snRNA-seq | Steady-state mRNA counts (3′ or 5′ ends; full-length for plate methods) | 90–95% zeros; ρ below 0.25 for most genes | Protein, modification, localization; time |
| scATAC-seq (Buenrostro et al., 2015) | Accessible chromatin from transposase insertions | At most two insertions per diploid site per cell: effectively binary and ultra-sparse | Which TF is bound; whether the region acts |
| CITE-seq / ADT (Stoeckius et al., 2017) | Surface proteins via oligo-tagged antibodies | Higher counts per feature; ambient and non-specific binding | Intracellular proteins; antibody-limited panel |
| Multiome (RNA + ATAC), spatial multi-omics | Two layers from the same cell | Each layer sparser than alone | Causal order of layers |
| scBS-seq and related | DNA methylation | Very sparse, binary per CpG | Dynamics |
| Lineage barcoding (CRISPR scars, Cas9 tracing) | Clonal relationships | Barcode saturation, dropout | Fate decisions not recorded in lineage |
| Metabolic labeling (e.g., 4sU, scNT-seq-style) | Newly synthesized vs old RNA | Costly, perturbs cells | Cells with slow turnover |

**All of these are snapshots.** Because measurement kills the cell, time can only be inferred: from *unspliced vs spliced* RNA (RNA velocity; La Manno et al., 2018), from metabolic labeling, from lineage barcodes, or from population-level distributions across a time series (Chapter 19, which treated the cell-state manifold as a *stationary* snapshot of a dynamical system). A model trained on snapshots to predict dynamics must assume a link (for example, that observed states are samples from a quasi-stationary distribution or a known drift) that the data cannot confirm; *the gap G-I (inference/causal) is built in*. Differences between modalities also matter for **multimodal foundation models** (Chapter 40): the joint distribution requires *paired* measurements, which are an order of magnitude scarcer than unpaired ones.

---

## 25.4 Spatial transcriptomics

Tissue structure matters because cell identity is partly *defined* by neighbors (signals, contacts). Spatial technologies trade three axes against each other: **spatial resolution**, **gene coverage**, and **molecular sensitivity (depth per cell)**.

* **Sequencing-based capture.** Tissue sections are placed on arrays of barcoded capture oligonucleotides; mRNA diffuses onto the array and is sequenced with its position (Ståhl et al., 2016). Whole-transcriptome coverage; resolution set by the array: 55-µm spots with ~100-µm spacing in the original 10x Visium design (each spot contains from one to tens of cells) [[E]], bead arrays near cellular scale (Slide-seq; Rodriques et al., 2019), and sub-cellular lattices (Stereo-seq, Chen et al., 2022; Visium HD with 2-µm bins) [[S]]. Sensitivity per bin falls as bins shrink.
* **In-situ imaging.** Single-molecule FISH and its multiplexed descendants (MERFISH, Chen et al., 2015; seqFISH) and in-situ sequencing platforms (commercial systems such as Xenium and CosMx) image individual transcripts with sub-cellular resolution for panels of hundreds to a few thousand pre-chosen genes [[S]]; reads are assigned to cells by **segmentation**, whose errors (merged or split cells, transcripts assigned to the wrong neighbor) become the dominant noise source; the panel is *chosen*, so unexpected genes are invisible.
* **Sections are 2-D samples of a 3-D tissue** (typical thickness 5–20 µm): a cell cut through its edge contributes a partial profile, and neighboring-cell RNA contaminates it.

### 25.4.1 A spot is a mixture: deconvolution and its failure modes

For a capture spot containing $n_k$ cells of type $k$ with RNA content $c_k$ and gene-fraction signature $S_{\cdot k}$ (a column of a $G\times K$ matrix with each column summing to 1), the expected expression of the spot is
$$
\mathbb E[y_{s}]=D_s\,\frac{\sum_kn_{k}c_{k}\,S_{\cdot k}}{\sum_kn_{k}c_{k}}\ \odot\ b,
$$
with $D_s$ the depth and $b$ a vector of platform-specific gene efficiencies. *Deconvolution* (cell2location, Kleshchevnikov et al., 2022; RCTD, Cable et al., 2022; Tangram, Biancalani et al., 2021) estimates the mixture weights from a single-cell **reference** by (constrained) regression; the simplest version is non-negative least squares. Three failure modes follow directly from this equation and were tested in simulation (5 cell types, 300 genes, 3–14 cells per spot, true mixtures from a Dirichlet):

| UMI per spot | Matched reference | Reference from another platform ($b$, log-SD 0.5) | Reference *missing* one cell type |
|---|---|---|---|
| 200 | 0.023 | 0.024 | 0.093 |
| 1,000 | 0.010 | 0.018 | 0.084 |
| 5,000 | **0.004** | 0.017 | 0.088 |

(Mean absolute error of the estimated RNA-fraction per cell type.) (i) **Depth helps only until a floor**: with a matched reference the error falls 6-fold from 200 to 5,000 UMI per spot, but a platform mismatch sets a floor of about 0.017, and a *missing* cell type sets an error of about 0.09 that depth cannot reduce, because its RNA is reassigned to the most similar available types. Rather than failing visibly, the estimator returns a confident-looking mixture. (ii) **RNA fraction is not cell fraction**: the regression returns the share of *RNA*; converting to the share of *cells* requires knowing $c_k$ (here 0.6–2.5 relative RNA content), and reading RNA fractions as cell fractions gives a persistent mean absolute error of about 0.06 at every depth. (iii) **The reference must contain what the tissue contains.** *A reference atlas from healthy tissue applied to diseased tissue with a disease-specific cell state will silently misallocate that state.*

!!! lens "Research lens: the measurement view applied to spatial data"
    Ask of any spatial dataset: *what is the unit of observation* (a spot, a segmented cell, a pixel), *what fraction of each unit's molecules is observed*, *which genes are in the panel*, and *how was the unit defined*, because cell-type calls, neighborhood statistics, and ligand–receptor analyses all inherit these choices. The same tissue measured on two platforms can yield different cell-type proportions for purely technical reasons.

---

## 25.5 Perturbation measurement: Perturb-seq and its relatives

Observational data show correlations between genes; a **perturbation** screen intervenes and measures the response, which is what causal questions require (Chapter 44) and what a "virtual cell" must predict (Chapter 39).

**Perturb-seq** (Dixit et al., 2016; Adamson et al., 2016; related CRISP-seq, Jaitin et al., 2016) combines pooled CRISPR perturbations with single-cell RNA-seq: each cell receives one (or a few) guide RNAs; the guide identity is read out together with the transcriptome (via an expressed barcode or direct capture of the guide RNA); cells are then grouped by guide, and the expression of cells carrying a given guide is compared with control cells carrying non-targeting guides. Variants use CRISPR **interference** (CRISPRi; repression without DNA breaks), **activation** (CRISPRa), or knockout; combinations of perturbations are encoded by multiple guides per cell (Norman et al., 2019). Chemical perturbations are profiled with multiplexed designs (sci-Plex; Srivatsan et al., 2020) and, at larger scale, the **Tahoe-100M** collection (over 100 million cells from 50 cancer cell lines exposed to on the order of a thousand small molecules at several doses; see the dataset documentation for the exact design) [[S]]. Genome-scale genetic perturbation datasets: **Replogle et al. (2022)** profiled more than 2.5 million cells in K562 (CRISPRi against all expressed genes) and in RPE1 (essential genes), and the Xaira **X-Atlas/Orion** dataset (2025) profiled about 8 million cells with genome-wide perturbations in HCT116 and HEK293T cells [[S]].

!!! paper "Paper dissection: Replogle et al., *Cell* (2022), \"Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq\""
    **Problem.** Pooled CRISPR screens that read out growth or a single marker give low-dimensional phenotypes; understanding what a gene *does* requires a high-dimensional phenotype, ideally for every gene in the genome.
    **Key insight.** Scale Perturb-seq to the whole genome with CRISPRi, using direct capture of guide RNAs to improve guide assignment, so that the transcriptome of cells perturbed for each gene becomes a "signature" that can be compared among genes.
    **Data.** More than 2.5 million cells; K562 (CRISPRi against all expressed genes) and RPE1 (essential genes); median target knockdown of about 85% (K562) and 92% (RPE1) at the reported time points; about 87% of essential-gene perturbations produced a significant transcriptional response, a far higher fraction than non-essential genes [[S]].
    **Analysis.** Clustering of perturbation signatures into functional modules (complexes and pathways), guilt-by-association assignment of function to poorly characterized genes, and comparison of transcriptional with growth phenotypes.
    **Why it matters.** At publication it was the largest single-cell genetic perturbation screen, and it underlies benchmarks for perturbation-response models (Chapter 39).
    **Assumptions.** CRISPRi in one cell line (K562: a transformed blood cancer line; RPE1: immortalized epithelial) reveals regulatory relationships that generalize; the transcriptome at a single day-6-to-7 time point reflects the perturbation's function; guide assignment and knockdown are accurate.
    **Limitations.** Single cell lines in culture (not physiological context); one time point; knockdown efficiency heterogeneity; most perturbations produce small effects relative to noise (§25.5.2); transcriptome-only readout; non-essential perturbations less well detected.
    **What followed.** Larger and multi-cell-line perturbation atlases, chemical-perturbation scale-ups (Tahoe-100M), perturbation-prediction challenges (the Arc Virtual Cell Challenge, 2025) and foundation models (State; Chapter 39).

### 25.5.1 What a perturbation measurement is

A perturbation dataset is a set of cell populations indexed by $(\text{perturbation},\text{context})$, each measured as $n$ cells. The **effect** of a perturbation $a$ is a shift in the distribution of expression; the usual estimand is the mean shift $\Delta_a=\mathbb E[x\mid a]-\mathbb E[x\mid \text{control}]$ (a vector over genes). It is estimated by the difference of sample means. Properties of the process that determine what $\hat\Delta_a$ means:

* **Effective perturbation fraction $q$.** Not every cell with a guide is effectively perturbed: guides differ in efficiency and cells vary in the degree of knockdown. If a fraction $q$ of the cells carry the full effect $\Delta$, the observed mean shift is $q\Delta$ (a dilution) and the sample size needed to detect it grows as $1/q^2$.
* **Time point.** A single time point after a perturbation shows a mix of the direct (primary) and downstream (secondary) responses, with adaptation and selection (cells with strong growth defects drop out; survivors are not a random sample).
* **Context.** The effect is specific to the cell line, culture condition, and state.

### 25.5.2 How many cells does a perturbation need?

Simulating one gene with baseline mean 2 UMI per cell and $\phi=0.3$ (a Welch $t$-test on log1p counts, $\alpha=10^{-3}$ to allow for multiple testing), the power to detect the shift as a function of cells per perturbation is:

| Cells per perturbation | 25 | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|---|
| log$_2$FC 0.25 (all cells effective) | 0.00 | 0.01 | 0.03 | 0.09 | 0.37 | 0.81 |
| log$_2$FC 0.50 (all cells effective) | 0.02 | 0.09 | 0.27 | 0.70 | 1.00 | 1.00 |
| log$_2$FC 1.00 (all cells effective) | 0.26 | 0.72 | 0.99 | 1.00 | 1.00 | 1.00 |
| log$_2$FC 0.50, only 60% of cells effective | 0.01 | 0.02 | 0.05 | 0.14 | 0.64 | 0.97 |

With 100 cells per perturbation (typical for a genome-scale screen), a single gene with a 40% change (log$_2$FC 0.5) is detected one time in four, and one with a 19% change (0.25) almost never; *only large effects on well-expressed genes are individually detectable*. A dilution to $q=0.6$ costs a $1/q^2\approx2.8$-fold increase in cells (in the table, 2.5-fold): power of 0.70 at 200 cells becomes 0.14, and 0.64 needs 500.

### 25.5.3 The reliability of a perturbation's effect vector

Consider evaluating a model that predicts the expression shift $\Delta_a$ of a held-out perturbation, scored by the Pearson correlation between predicted and observed shifts across genes. **No prediction can correlate with the observed shift better than the observed shift correlates with itself in a replicate.** We estimate this *ceiling* by split-half reliability: split the cells of a perturbation into two halves (each with its own control cells), estimate $\hat\Delta$ in each, and compute $r$ between them; the Spearman–Brown step gives the reliability of the full-sample estimate $\rho=2r/(1+r)$ and the ceiling for the correlation of any predictor with it, $r_\max=\sqrt\rho$. For 2,000 genes, with true effects on $s$ genes (log$_2$ effects drawn from $N(0,0.5^2)$; means of 8 repeats):

| True DE genes | Cells per half | $r$ (all genes) | $r$ (true DE genes only; oracle) | Reliability $\rho$ | Ceiling $r_\max$ |
|---|---|---|---|---|---|
| 0 | 100 | 0.009 | n/a | ≈0 | ≈0 |
| 0 | 1,000 | 0.012 | n/a | ≈0 | ≈0 |
| 5 | 100 | 0.008 | 0.914 | ≈0 | ≈0 |
| 5 | 1,000 | 0.081 | 0.975 | 0.149 | 0.386 |
| 50 | 100 | 0.045 | 0.675 | 0.086 | 0.293 |
| 50 | 1,000 | 0.347 | 0.955 | 0.516 | 0.718 |
| 500 | 100 | 0.358 | 0.690 | 0.528 | 0.726 |
| 500 | 1,000 | 0.846 | 0.955 | 0.917 | 0.958 |

(Values of $r$ below about 0.02 are indistinguishable from zero given sampling error across 2,000 genes.) The conclusions are sharp. **(i)** For a perturbation that truly alters five genes, the vector of observed shifts over all 2,000 genes has reliability near zero at 100 cells and 0.15 at 1,000: *the all-gene correlation metric measures noise*, and the best conceivable model scores below 0.4. **(ii)** The same estimates restricted to the truly affected genes are highly reliable (0.9+), so the information is in the *sparse set of affected genes*, which a good metric must focus on, as long as the set is chosen without peeking at the test data (otherwise, selection bias). **(iii)** The ceiling depends on both the perturbation's effect size and the number of cells, so *averaging a metric over perturbations mixes perturbations with ceiling 0 and ceiling 0.96*. **(iv)** For perturbations with no real effect (most non-essential genes in many screens), the correct prediction is "the control state", and a model that predicts the control mean scores as well as any other. This is the measurement-theoretic root of the observations of Chapter 39 that simple baselines are hard to beat on standard perturbation benchmarks: a high proportion of the benchmark's variance is not signal.

!!! rhyme "Structural rhyme: split-half reliability ↔ noise ceiling ↔ inter-annotator agreement"
    Split-half reliability is the *same object* as the replicate-correlation noise ceiling in DMS (Chapter 23), the measurement-noise ceiling of potency data (Chapter 24), and inter-annotator agreement in NLP. Every benchmark for a biological prediction task should come with its ceiling; a model "at 0.35" may be at 100% of the attainable score (50 DE genes at 100 cells: ceiling 0.29) or at 36% (500 DE genes at 1,000 cells: ceiling 0.96).

---

## 25.6 The experiments, verbatim

```python
--8<-- "code/ch25_measurement.py"
```

```text
== 1. Binomial thinning of a Gamma-Poisson gives a Gamma-Poisson with the SAME dispersion ==
empirical variance/(mean + phi*mean^2) over genes with mean>1: 1.000  (theory: 1)
mean UMI/cell      genes   observed zero frac   NB prediction   Poisson prediction   per-cell reliability (true signal var / total var)
[ 0.02,  0.2)    1857        0.914              0.914           0.913                0.018
[ 0.20,  0.5)     945        0.731              0.731           0.723                0.061
[ 0.50,  1.0)     649        0.518              0.518           0.495                0.124
[ 1.00,  3.0)     730        0.249              0.249           0.204                0.252
[ 3.00, 10.0)     361        0.041              0.041           0.015                0.497
[10.00, 99.0)     116        0.001              0.001           0.000                0.772

== 2. A few genes go up; library-size normalization makes everything else go down ==
the 52 most abundant genes carry 30% of molecules and are tripled in condition B; the other 2234 well-detected genes are unchanged
apparent log2 fold-change of the unchanged genes: CPM normalization median -0.68 (true 0.00); median-ratio normalization +0.00
fraction of unchanged genes with apparent log2FC < -0.5: CPM 0.91; median-ratio 0.00

== 3. Cells needed to observe at least 20 cells of a type with frequency f, with probability 0.95 ==
  f = 0.05   : about       565 cells
  f = 0.01   : about     2,853 cells
  f = 0.001  : about    27,924 cells
  f = 0.0001 : about   280,228 cells

== 4. Doublets masquerade as an intermediate cell state ==
4240 droplets, 240 (6%) are A+B doublets; true intermediate cells (0.35<t<0.65): 547
cells that look intermediate in the expression embedding (position 0.35-0.65 on the A-B axis): 612; of these 19% are doublets; 48% of all doublets land there
total UMI per droplet, doublets / singlets: 2.14x;  AUROC for doublet detection from total UMI alone = 0.934
synthetic-doublet neighbor score: AUROC = 0.967 over all droplets; within the 'intermediate' region only, AUROC = 0.947
Poisson loading with mean 0.05 cells/droplet: doublet fraction among occupied droplets = 0.025 (~lambda/2 = 0.025)

== 5. Pseudoreplication in single-cell differential expression: 3 vs 3 donors, 400 cells each ==
  cells as replicates (Welch t on 1,200 vs 1,200 cells)  : false-positive rate at p<0.05 on null genes = 0.696; power on DE genes = 0.860
  donors as replicates (pseudobulk, 3 vs 3)              : false-positive rate at p<0.05 on null genes = 0.031; power on DE genes = 0.160
  (null genes called at p<0.001 with cells as replicates: 0.51; with donors: 0.000)

== 6. Spatial spots as mixtures: NNLS deconvolution, and what breaks it ==
                                       MAE RNA-fraction   MAE cell-fraction   (RNA-fraction estimate read as cell fraction)
  UMI/spot =   200, matched reference                : 0.023              0.026              0.062
  UMI/spot =   200, reference from another platform  : 0.024              0.028              0.060
  UMI/spot =   200, reference missing a cell type    : 0.093              0.094              0.123
  UMI/spot =  1000, matched reference                : 0.010              0.012              0.059
  UMI/spot =  1000, reference from another platform  : 0.018              0.020              0.070
  UMI/spot =  1000, reference missing a cell type    : 0.084              0.081              0.112
  UMI/spot =  5000, matched reference                : 0.004              0.005              0.057
  UMI/spot =  5000, reference from another platform  : 0.017              0.017              0.062
  UMI/spot =  5000, reference missing a cell type    : 0.088              0.085              0.120

== 7. Perturb-seq: how many cells per perturbation? (one gene, baseline mean 2 UMI, phi = 0.3, Welch t-test on log1p, alpha = 1e-3) ==
cells per perturbation:         25     50    100    200    500   1000
log2FC 0.25, all cells effective:  0.00   0.01   0.03   0.09   0.37   0.81
log2FC 0.50, all cells effective:  0.02   0.09   0.27   0.70   1.00   1.00
log2FC 1.00, all cells effective:  0.26   0.72   0.99   1.00   1.00   1.00
log2FC 0.50, 60% of cells effective:  0.01   0.02   0.05   0.14   0.64   0.97

split-half reliability of the observed mean-shift vector (2,000 genes; log2 shift estimates from independent halves)
true DE genes   cells/half   r(all genes)   r(true DE genes only)   reliability 2r/(1+r) of the full-sample estimate -> ceiling r_max = sqrt(reliability)   [means of 8 repeats]
       0           100         0.009            n/a                  0.018 -> 0.133
       0          1000         0.012            n/a                  0.024 -> 0.155
       5           100         0.008          0.914                  0.015 -> 0.123
       5          1000         0.081          0.975                  0.149 -> 0.386
      50           100         0.045          0.675                  0.086 -> 0.293
      50          1000         0.347          0.955                  0.516 -> 0.718
     500           100         0.358          0.690                  0.528 -> 0.726
     500          1000         0.846          0.955                  0.917 -> 0.958
```

---

## 25.7 Worked research examples

!!! example "Worked Research Example 25.1: A new transitional state in a differentiation atlas"
    **Situation.** An atlas of 120,000 cells from 4 donors reveals a small cluster of cells expressing both a progenitor program and a mature-lineage program. Trajectory analysis places it between the two as a transitional state, and a clinical association is proposed: the cluster is more abundant in 2 of the 4 donors, who have a disease.

    **Question.** What are the measurement explanations, and which analyses would distinguish a real transitional state?

    **Reasoning.**

    1. *Which instrument artifacts produce a cluster co-expressing two programs?* Heterotypic doublets (§25.2.6: in our simulation, 19% of apparent intermediates were doublets, and 48% of all doublets landed in the intermediate region), ambient RNA from abundant cell types, and annotation circularity (programs defined from the same data).
    2. *What distinguishes them?* Doublets have **higher total UMI** (here 2.1×; AUROC 0.93 from UMI alone), higher gene counts, co-expression of *mutually exclusive* markers (genes that are never co-expressed in singlets), and neighbors that include simulated doublets (AUROC 0.97). Real transitional cells have a *singlet-like* UMI distribution and *graded*, not additive, expression, with intermediate expression of both programs that makes biological sense (a gradient of regulators).
    3. *What about the donor association?* With 2 vs 2 donors, the "abundance difference" is a comparison of two donor means, with donor effects (sampling day, dissociation) fully confounded: cells are not replicates (§25.2.7). A cell-level test would produce a false-positive rate of order 70% in our simulation.
    4. *Decisive experiments.* (a) Run doublet detection on the raw matrix per sample and repeat the analysis after removal; (b) test the abundance difference with donor as the unit (pseudobulk, a mixed model such as a beta-binomial for proportions with donor random effects), with more donors; (c) validate the transitional state *orthogonally*: multiplexed FISH showing cells with intermediate expression and one nucleus, lineage tracing, or RNA velocity with metabolic labeling; (d) make a falsifiable prediction (e.g., the cluster should be depleted when a transition regulator is perturbed).

    **Expert analysis.** The transitional state is a hypothesis with three independent artifact explanations; each costs hours to test. The general lesson: *in count-matrix data, "novelty" is the first thing to explain away*. Expert practice treats a new cell state as a null-hypothesis problem: first show that the state is not the instrument, then not the donor, then show that it is stable across methods, and only then name it. Weak links in the Expert Chain (L5 failure modes, L10 experiments) are where this analysis spends its effort.

!!! example "Worked Research Example 25.2: Is a perturbation-response model any good?"
    **Situation.** A group trains a model to predict, for held-out gene knockdowns, the post-perturbation mean expression profile in K562 cells. They report an average Pearson correlation of 0.31 between predicted and observed shifts across 2,000 genes (and 0.28 for the best baseline). They conclude that the model captures a third of the variance.

    **Question.** How would you interpret 0.31, and what would you ask for?

    **Reasoning.**

    1. *What is the ceiling?* Without a replicate-based estimate the number is uninterpretable. From §25.5.3, a perturbation with 50 affected genes at 100 cells per half has a ceiling near 0.29 (and $r\approx0.05$ between halves on all genes), so 0.31 would be *at or above* the ceiling and would indicate a leak or an artifact (e.g., the predicted vector uses the shared control cells of the test perturbation, or the test perturbations are correlated with training perturbations through shared mean shifts such as the generic stress response). For a strong perturbation at 1,000 cells, the ceiling is 0.96 and 0.31 would be poor.
    2. *What is the effective sample?* The number of *perturbations* with real effects, not cells. If 80% of the perturbations are null, the average correlation is dominated by noise for 80% of the data.
    3. *Which baselines?* (a) Predict the control mean (zero shift); (b) predict the **mean shift across all training perturbations** (captures the generic response shared by most perturbations); (c) the additive model for combinatorial perturbations; (d) nearest-neighbor perturbation by prior-knowledge similarity (e.g., same complex, GO term, or STRING neighbor). A model that cannot beat (b) has learned only the average perturbation.
    4. *Which metrics?* Evaluate on the **differentially expressed genes of each perturbation**, chosen on held-out cells independent of the evaluation (so the choice is not leaked), report the *ceiling-normalized* score $r/r_\max$ (with a confidence interval), and separate perturbations by effect size. Evaluate also *ranking of perturbations* (does the model identify which perturbations have large effects?) and *direction of change* on affected genes.
    5. *What decision does this serve?* If the model will be used to prioritize perturbations for experiment (Chapter 46), the right metric is the *enrichment of true strong hits* among its top predictions, not the average correlation.

    **Expert analysis.** The benchmark's information content is set by its replicate reliability and the effective number of perturbations with real effects, not by the number of cells (a "million-cell" dataset can contain a few hundred informative perturbations). Concluding "a third of the variance" from an uncalibrated correlation conflates three things: biology, noise, and shared structure. This example previews the evaluation discussion of Chapters 39 and 43.

---

## 25.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: reading a single-cell methods section as a measurement model"
    **Setting.** You are about to train a model on a published dataset, or to rely on a claim from one.

    **Questions to answer from the methods, in order.**

    1. **What is the unit of replication?** Number of donors (or mice, or independent cultures), not cells. If it is below ~5 per condition, claims about condition effects are weak.
    2. **What was sampled?** Cells vs nuclei; enrichment; dissociation protocol; fraction of cells recovered; cell types known to be lost.
    3. **What chemistry and depth?** UMI per cell, genes per cell; capture efficiency; where in the transcript reads fall (3′/5′/full-length).
    4. **What was removed?** Empty droplets, doublets, ambient RNA, low-quality cells, mitochondrial-high cells (which can be *real* biology, e.g. cardiomyocytes).
    5. **How was it normalized?** Shifted logarithm of size-factor-normalized counts, scaling, regression of covariates. Does the normalization assume that most genes are unchanged?
    6. **How were batches handled?** Was batch correction fitted using the labels you are about to predict? Could a model learn the donor instead of the condition?
    7. **How were labels made?** From clustering on the same data, from reference mapping, from marker genes? Is an annotation a model output?
    8. **For perturbation data: what is the effective perturbation fraction, the number of cells per perturbation, the time point, the control cell definition, and the replicate reliability?**
    9. **For spatial data: what is the unit (spot, segmented cell), the panel, and the segmentation method?**
    10. **What is not in the data?** Time, protein, context, intervention.

    **What it teaches.** The Expert Chain begins with the measurement: it determines which questions the data can answer (Chapter 1, the Four Gaps: G-M).

    **An open question to carry forward.** The count model of §25.2.2 says that single-cell data are samples from a *latent* count distribution. If a foundation model's job is to learn the latent distribution (the cell's true molecule counts and state) rather than the observed counts, what is the right objective and the right evaluation, given that the latent is never observed? Design a simulation, as in this chapter, where the latent is known and compare candidate objectives (NB likelihood, masked reconstruction, contrastive learning) by how well they recover it; Chapter 38 revisits this.

---

## 25.9 Connections

- **Backward:** the NB as bursting and the snapshot–dynamics gap (Chapter 19); pseudoreplication and Spearman–Brown (Chapter 4); latent-variable models and VAEs (Chapters 8, 14); noise ceilings (Chapters 1, 23, 24).
- **Forward:** statistical genetics and population-scale studies (Chapter 26); single-cell methods before foundation models (Chapter 30); single-cell foundation models (Chapter 38); perturbation prediction and the virtual cell (Chapter 39); multimodal integration (Chapter 40); benchmark design (Chapter 43); causal inference from interventional data (Chapter 44); experimental design (Chapter 46); open problems in cell modeling (Chapter 51).

!!! takeaways "Key takeaways"
    1. UMI counts are **binomially thinned Gamma–Poisson** samples: thinning preserves the NB and its dispersion, so zeros are explained by low expression without a separate "dropout" process; simulation confirms the zero fractions to three decimals.
    2. The **per-cell reliability** of a gene is $\rho=\phi\lambda/(1+\phi\lambda)$: 0.02 at 0.1 UMI, 0.25 at about 1–3 UMI, above 0.75 only for genes with >10 UMI; aggregating cells raises reliability by Spearman–Brown.
    3. **Library-size normalization is compositional**: tripling genes carrying 30% of molecules makes the other genes appear down by $\log_2 1.6=0.68$; median-ratio normalization removes this only if most genes are unchanged.
    4. Rare cell types need very large samples: 20 cells at frequency $10^{-4}$ requires about $2.8\times10^5$ cells.
    5. **Doublets create fake intermediate states** (19% of apparent intermediates in our simulation); **donors, not cells, are the replicates** (cell-level testing gave a 70% false-positive rate vs 3% at donor level).
    6. A spatial spot is a **mixture**; deconvolution error has a floor from reference mismatch and cannot be reduced by depth when the reference lacks a cell type; RNA fraction is not cell fraction.
    7. Perturb-seq intervenes at scale (over 2.5 million cells in Replogle et al., 2022; Tahoe-100M for chemicals; X-Atlas/Orion), but only large effects on well-expressed genes are individually detectable at ~100 cells per perturbation, and the effective perturbation fraction dilutes effects.
    8. The **reliability of a perturbation's effect vector** bounds any metric: a perturbation affecting 5 of 2,000 genes has reliability ≈0.15 even at 1,000 cells, so all-gene correlation measures noise; report ceiling-normalized scores and null-aware baselines.

---

## Further reading

- Macosko, E. Z. et al. (2015). Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets. *Cell* 161, 1202–1214. Klein, A. M. et al. (2015). Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells. *Cell* 161, 1187–1201. Zheng, G. X. Y. et al. (2017). Massively parallel digital transcriptional profiling of single cells. *Nat. Commun.* 8, 14049. Cao, J. et al. (2017). Comprehensive single-cell transcriptional profiling of a multicellular organism. *Science* 357, 661–667. Rosenberg, A. B. et al. (2018). Single-cell profiling of the developing mouse brain and spinal cord with split-pool barcoding. *Science* 360, 176–182.
- Svensson, V. (2020). Droplet scRNA-seq is not zero-inflated. *Nat. Biotechnol.* 38, 147–150. Ahlmann-Eltze, C. & Huber, W. (2023). Comparison of transformations for single-cell RNA-seq data. *Nat. Methods* 20, 665–672. Hafemeister, C. & Satija, R. (2019). Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression. *Genome Biol.* 20, 296. Lopez, R. et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nat. Methods* 15, 1053–1058.
- Lun, A. T. L. et al. (2019). EmptyDrops: distinguishing cells from empty droplets in droplet-based single-cell RNA sequencing data. *Genome Biol.* 20, 63. Fleming, S. J. et al. (2023). Unsupervised removal of systematic background noise from droplet-based single-cell experiments using CellBender. *Nat. Methods* 20, 1323–1335. Wolock, S. L., Lopez, R. & Klein, A. M. (2019). Scrublet. *Cell Syst.* 8, 281–291. McGinnis, C. S., Murrow, L. M. & Gartner, Z. J. (2019). DoubletFinder. *Cell Syst.* 8, 329–337.
- Squair, J. W. et al. (2021). Confronting false discoveries in single-cell differential expression. *Nat. Commun.* 12, 5692. La Manno, G. et al. (2018). RNA velocity of single cells. *Nature* 560, 494–498. Buenrostro, J. D. et al. (2015). Single-cell chromatin accessibility reveals principles of regulatory variation. *Nature* 523, 486–490. Stoeckius, M. et al. (2017). Simultaneous epitope and transcriptome measurement in single cells. *Nat. Methods* 14, 865–868.
- Ståhl, P. L. et al. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science* 353, 78–82. Rodriques, S. G. et al. (2019). Slide-seq. *Science* 363, 1463–1467. Chen, K. H. et al. (2015). Spatially resolved, highly multiplexed RNA profiling in single cells (MERFISH). *Science* 348, aaa6090. Chen, A. et al. (2022). Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. *Cell* 185, 1777–1792. Kleshchevnikov, V. et al. (2022). Cell2location. *Nat. Biotechnol.* 40, 661–671. Cable, D. M. et al. (2022). Robust decomposition of cell type mixtures in spatial transcriptomics (RCTD). *Nat. Biotechnol.* 40, 517–526. Biancalani, T. et al. (2021). Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram. *Nat. Methods* 18, 1352–1362.
- Dixit, A. et al. (2016). Perturb-Seq. *Cell* 167, 1853–1866. Adamson, B. et al. (2016). A multiplexed single-cell CRISPR screening platform. *Cell* 167, 1867–1882. Jaitin, D. A. et al. (2016). Dissecting immune circuits by linking CRISPR-pooled screens with single-cell RNA-seq. *Cell* 167, 1883–1896. Norman, T. M. et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* 365, 786–793. Replogle, J. M. et al. (2022). Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575. Srivatsan, S. R. et al. (2020). Massively multiplex chemical transcriptomics at single-cell resolution. *Science* 367, 45–51.
