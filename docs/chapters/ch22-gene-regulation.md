# Chapter 22. Gene Regulation: From Sequence to Expression

!!! abstract "Chapter at a glance"
    **Motivation.** The regulatory code, the mapping from DNA sequence (plus cellular context) to *when, where, and how much* a gene is expressed, is the central target of genomic AI (Chapters 31–33, 41). This chapter explains the machinery, derives the physical model that connects sequence to occupancy to expression, catalogs the assays and what each sees, and states precisely what a sequence-only model can and cannot capture.
    **Prerequisites.** Chapters 4, 5 (motif information), 10, 19–21.
    **You will be able to:** (1) describe promoters, enhancers, transcription factors, chromatin, and 3-D genome organization and what each contributes; (2) derive the thermodynamic model of TF binding and see PWMs as energy matrices; (3) quantify the specificity problem and the role of cooperativity; (4) explain how splicing, 3′ processing, and translation add regulatory layers beyond transcription; (5) list the assays used to measure regulation and their noise, resolution, and biases; (6) formulate the **cis/trans decomposition** and use it to state what any sequence model must take as input.

---

## 22.1 The problem and the seven questions

!!! bio "Biology for modeling: the regulatory code"
    **What is it?** The set of rules by which DNA sequence and cellular context together determine where, when, and how much each gene is transcribed (and how its RNA is processed and translated).
    **Information it contains.** *Cis* information: the sequence of promoters, enhancers, silencers, insulators, and RNA-processing signals near a gene. *Trans* information: the concentrations and activities of transcription factors, chromatin regulators, and RNA-binding proteins in the cell, which are products of *other* genes.
    **How is it generated?** Evolution shapes cis-regulatory sequences; development and signaling set trans state; both interact at the chromatin and RNA level through biophysical binding and enzymatic reactions.
    **How is it measured?** Assays of RNA (RNA-seq, CAGE), chromatin (ATAC-seq, ChIP-seq, Hi-C), reporter activity (MPRA, STARR-seq), and perturbation (CRISPRi/a screens, base editing, Perturb-seq) (§22.6).
    **Computational representation.** Sequence windows (one-hot or tokenized), tracks over genomic bins, cell-type-specific output heads, cell-by-gene matrices, graphs of enhancer–promoter links.
    **What varies.** Between genes (promoter strength spans $>10^4$-fold), cell types (hundreds), individuals (cis variants shift expression by a few percent to tens of percent), and conditions (signaling, stress).
    **What can ML learn?** Sequence determinants of binding, accessibility, and expression within the training cell types; motif syntax; effects of variants on local molecular readouts.
    **What can ML not observe?** Trans state not provided as input; causal effects without perturbation data; regulation in contexts absent from training; spatial organization beyond the context window; single-molecule dynamics.

---

## 22.2 The machinery

### 22.2.1 Promoters, enhancers, and other cis-regulatory elements

- **Promoter.** The region around the transcription start site (TSS), roughly $-1$ kb to $+0.5$ kb, where RNA polymerase II and general transcription factors assemble. Features: core elements (TATA box, initiator, downstream elements) in a minority; **CpG islands** (~1 kb stretches of high CpG density, typically unmethylated) at roughly 70% of human promoters. Promoters initiate transcription at a *distribution* of nearby start sites (sharp or broad), measured by CAGE.
- **Enhancers.** Short (100–1000 bp) elements that increase transcription of target genes, acting at a distance (typically tens of kb, often up to several hundred kb; occasionally > 1 Mb) and in a cell-type-specific manner. They are marked by accessible chromatin and histone marks H3K27ac and H3K4me1. Hundreds of thousands to over a million candidate regulatory elements have been annotated across human cell types (ENCODE registry of candidate cis-regulatory elements); most are *candidates* with no demonstrated target.
- **Silencers and insulators.** Elements that repress transcription or block enhancer–promoter communication (insulators are often bound by **CTCF**).
- **Super-enhancers and locus control regions.** Clusters of enhancers with high activity; their status as distinct entities versus extended arrays of ordinary enhancers is debated.

### 22.2.2 Transcription factors and motifs

Roughly **1,600 human proteins** are putative sequence-specific transcription factors (TFs; Lambert et al., 2018), grouped into families by DNA-binding domain (zinc finger, homeodomain, bHLH, bZIP, nuclear receptor, ETS, ...). Each TF recognizes short, degenerate motifs (6–20 bp), summarized by a position weight matrix (Chapters 2, 5, 29). Motif databases (JASPAR, HOCOMOCO) contain motifs for several hundred TFs; for many TFs the motif is unknown or ambiguous. TFs bind **cooperatively** (with each other and with cofactors), and **pioneer factors** (e.g., FOXA, GATA, OCT4/SOX2) can bind nucleosomal DNA and open chromatin for others.

### 22.2.3 Chromatin

DNA is wrapped around histone octamers in **nucleosomes** (147 bp of DNA per nucleosome; ~$10^7$ nucleosomes per haploid human genome). Chromatin state strongly influences accessibility:

| Mark / state | Typical meaning | Assay |
|---|---|---|
| Open (nucleosome-depleted) DNA | Regulatory element accessible to TFs | ATAC-seq, DNase-seq |
| H3K4me3 | Active promoters | ChIP-seq, CUT&Tag |
| H3K27ac | Active enhancers and promoters | ChIP-seq |
| H3K4me1 | Enhancers (poised or active) | ChIP-seq |
| H3K36me3 | Transcribed gene bodies | ChIP-seq |
| H3K27me3 | Polycomb-repressed regions | ChIP-seq |
| H3K9me3 | Constitutive heterochromatin | ChIP-seq |
| DNA methylation (5mC at CpG) | ~70–80% of CpGs methylated in somatic cells; CpG islands usually unmethylated; promoter methylation usually represses | Bisulfite sequencing, nanopore |

Histone marks and methylation are largely **consequences and maintainers** of regulatory state as well as causes; the causal direction (does the mark drive expression or follow it?) is mixed and context-dependent. This matters for ML: *chromatin features predict expression strongly, but using them as input to predict expression is not "explaining" it*; and *predicting chromatin from sequence* (the task of DeepSEA/Enformer) is a different problem from predicting expression.

### 22.2.4 3-D genome organization

The genome is folded. **Loop extrusion** by cohesin, halted by convergently oriented CTCF sites, forms loops and **topologically associating domains (TADs)** (median ~0.5–1 Mb) within which enhancers and promoters preferentially contact (Dixon et al., 2012; Rao et al., 2014; Fudenberg et al., 2016). At larger scales, active (A) and inactive (B) **compartments** segregate. Hi-C and Micro-C measure contacts. TAD boundaries restrict enhancer–gene interactions, and disrupting a boundary can cause disease by rewiring enhancers; but *most enhancer–promoter pairs show weak, transient contacts that are difficult to detect*, and the quantitative relationship between contact frequency and regulatory output is the subject of active research (§22.8).

---

## 22.3 From sequence to occupancy: the thermodynamic model

### 22.3.1 Derivation

Consider a single TF at concentration $c$ and a DNA site $s$ with binding free energy $E(s)$ relative to a reference (the consensus), in units of $k_BT$. In equilibrium, the bound and unbound states have Boltzmann weights $c\,e^{-\Delta G_s/k_BT}$ and 1, with $\Delta G_s=-k_BT\ln(c_0/K_d)$... Writing $K_d(s)=K_0\,e^{E(s)}$ (a higher energy penalty means weaker binding), the **occupancy** (probability the site is bound) is the two-state partition-function result

$$
\theta(s)=\frac{c/K_d(s)}{1+c/K_d(s)}=\frac{c}{c+K_0e^{E(s)}}=\sigma\big(\ln c-\ln K_0-E(s)\big),
$$

a **logistic (sigmoid) function** of the log-concentration and the site energy. Chapters 3–4 already told us *logistic regression is a log-linear model*; here it is derived from physics.

### 22.3.2 PWMs are energy matrices

If the binding energy is approximately **additive across positions**, $E(s)=\sum_j\epsilon_j(s_j)$ with $\epsilon_j(a)$ the energy penalty for base $a$ at position $j$ (zero for the preferred base), then $-E(s)$ is a position-weight-matrix score (Chapter 5, §5.6.1; Berg & von Hippel, 1987): $-\epsilon_j(a)=\log(p_{j,a}/p_{j,a^*})$. **A PWM is the energy function of an additive binding model**, and *a convolutional filter followed by a sigmoid is the same model learned end-to-end* (Chapter 10; DeepBind). Real binding deviates from additivity (DNA shape, dinucleotide interactions, cooperativity within the DNA-binding domain), which is one reason deeper models outperform PWMs.

### 22.3.3 The specificity problem, quantified

Chapter 5 showed that a 10–20-bit motif cannot be unique in a 3-Gb genome. The thermodynamic model quantifies the consequence. Take a 10-bp motif with an additive mismatch penalty of 2.5 $k_BT$ per mismatched base (a typical order of magnitude), a TF at the concentration where the consensus site is 50% occupied, and random genomes of increasing size. The *summed* occupancy of all non-consensus positions is $\sum_i\theta_i$, whose expectation per position is $\big(\tfrac14+\tfrac34e^{-\epsilon}\big)^L$ for large $E$.

| Genome (positions) | Occupancy of the consensus site | Summed occupancy elsewhere | Fraction of bound TF at the consensus site |
|---|---|---|---|
| $10^4$ | 0.50 | 0.13 | **0.790** |
| $10^6$ (bacterial scale) | 0.50 | 8.27 | **0.057** |
| $10^8$ | 0.50 | 799.65 | **0.001** |

The analytic expectation for $G=10^6$ is $8.6$ (observed $8.27$). So in a megabase genome, **~94% of bound TF molecules sit at non-consensus sites**, and in a 100-Mb genome, over 99.9%. Cells overcome this through (i) *much stronger discrimination* than 2.5 $k_BT$ per base can supply (some TFs achieve more through cooperative and compositional recognition), (ii) **chromatin**: most of the genome is occupied by nucleosomes or compacted, so the *accessible* search space is a small fraction, (iii) **cooperativity** between TFs (§22.3.4), (iv) **concentrations** below saturation, and (v) functional redundancy (many weak sites collectively matter, and not every binding event is functional). **Modeling lesson: motif presence is a weak predictor of binding, and binding a weak predictor of function**; context (chromatin, cooperating factors, 3-D contact) supplies the missing bits (Chapters 5, 10, 31).

### 22.3.4 Cooperativity and regulatory logic

Consider a promoter regulated by two sites. Let $a=[A]/K_A$ and $b=[B]/K_B$ be the normalized concentrations, and let $\omega\ge1$ be a *cooperativity factor* (the extra weight when both are bound, due to protein–protein interaction or nucleosome-mediated effects). The **statistical-weights** (Shea–Ackers) model assigns each configuration a weight and normalizes by the **partition function** $Z$:

| Configuration | Weight |
|---|---|
| Neither bound | $1$ |
| A only | $a$ |
| B only | $b$ |
| Both | $\omega\,a\,b$ |

$Z=1+a+b+\omega ab$. If transcription requires both bound (an **AND** gate), $P(\text{active})=\omega ab/Z$; if either suffices (**OR**), $P(\text{active})=(a+b+\omega ab)/Z$. For the same TF at two sites ($a=b=x$), the response to TF concentration has an *apparent Hill coefficient* (ultrasensitivity) that increases with cooperativity from $n_H=1.19$ at $\omega=1$ (independent sites) to $1.50$, $1.79$, $1.93$, and $1.99$ at $\omega=10,10^2,10^3,10^5$, approaching the maximum of 2 (the number of sites). **Cooperativity creates sharp, switch-like responses and AND-like logic** (the code output is below). With $\omega=100$ and $a=b=0.1$ (each TF at 10% of its $K_d$), activity is already 45% because binding of one TF recruits the other. AND and OR gates have identical sharpness by a duality ($x\to1/x$) but differ in where the threshold lies.

!!! rhyme "Structural rhyme: statistical weights ↔ softmax ↔ Potts/Ising models ↔ attention"
    The probability of a configuration $=$ weight$/Z$ is a **softmax** over configurations; the weights $e^{-E}$ with pairwise interaction terms ($\omega$) are an **Ising/Potts model** over TF occupancy (Chapter 29), the same structure that models residue coevolution. A neural network that predicts expression from sequence is learning a flexible approximation to this partition function over regulatory configurations. The interaction terms $\omega_{ij}$ correspond to *pairwise couplings between motif occurrences*; attention layers (Chapter 12) can represent them as content-dependent pairwise interactions between positions. This is a concrete reason to expect attention-like architectures to help with *motif syntax*, and a baseline: **a fitted statistical-weights model with a few TFs is a competitive, interpretable baseline for promoter and enhancer activity** (Bintu et al., 2005; Segal & Widom, 2009).

**Nucleosome competition.** Nucleosomes compete with TFs for DNA. A TF that displaces a nucleosome makes the neighboring sites accessible to other TFs even without direct protein–protein interaction: **indirect (nucleosome-mediated) cooperativity** (Polach & Widom, 1995; Mirny, 2010). In the statistical-weights framework this is an additional configuration class. This is one mechanism by which *motif spacing and density* produce non-additive effects ("billboard" vs. "enhanceosome" syntax; §22.7).

```python
--8<-- "code/ch22_thermodynamic_model.py"
```

Output:

```text
site with  0.0 kT penalty: Kd =       1.0; occupancy at c = Kd(consensus) = 5.00e-01
site with  2.5 kT penalty: Kd =      12.2; occupancy at c = Kd(consensus) = 7.59e-02
site with  5.0 kT penalty: Kd =     148.4; occupancy at c = Kd(consensus) = 6.69e-03
site with 10.0 kT penalty: Kd =   22026.5; occupancy at c = Kd(consensus) = 4.54e-05

specificity problem (10-bp motif, 2.5 kT per mismatch, TF concentration c = Kd of the consensus site):
  genome of    1e+04 positions: occupancy of the consensus site = 0.50; summed occupancy of all other positions =      0.13; fraction of bound TF at the consensus site = 0.790
  genome of    1e+06 positions: occupancy of the consensus site = 0.50; summed occupancy of all other positions =      8.27; fraction of bound TF at the consensus site = 0.057
  genome of    1e+08 positions: occupancy of the consensus site = 0.50; summed occupancy of all other positions =    799.65; fraction of bound TF at the consensus site = 0.001
  analytic: expected summed background occupancy ~ G * (1/4 + 3/4 e^-eps)^L = 8.6 for G = 10^6

same TF at two promoter sites, AND logic (a = b = x): apparent Hill coefficient of expression vs TF concentration:
  cooperativity omega =       1: apparent Hill coefficient n_H = 1.19
  cooperativity omega =      10: apparent Hill coefficient n_H = 1.50
  cooperativity omega =     100: apparent Hill coefficient n_H = 1.79
  cooperativity omega =    1000: apparent Hill coefficient n_H = 1.93
  cooperativity omega =  100000: apparent Hill coefficient n_H = 1.99
expression P(active) under AND logic with omega = 100  (rows: [A]/K_A = 0.1, 1, 10; columns: [B]/K_B = 0.1, 1, 10):
    0.455  0.826  0.900
    0.826  0.971  0.988
    0.900  0.988  0.998
```

---

## 22.4 From occupancy to transcription

Occupancy sets the probability that a promoter is *licensed* to fire; transcription then occurs in **bursts** (Chapter 19). Evidence from single-cell and live-imaging studies suggests that **enhancers mainly modulate burst frequency** (how often the promoter fires) rather than burst size, whereas promoter sequence has relatively more influence on burst size [[S]] (Larsson et al., 2019; Bartman et al., 2016). In a thermodynamic picture, enhancer occupancy raises the probability of the productive state; in the NB language of Chapter 19, it changes $k_b$ (and therefore the dispersion parameter $r=k_b/\gamma$) more than $b$. The observable mean expression is then $\mu=rb=(k_b/\gamma)b$, the product of an enhancer-driven term and a promoter-driven term, which supports the **multiplicative** interactions between enhancers and promoters observed in many reporter experiments and exploited by the ABC model (§22.8).

---

## 22.5 RNA processing and translation: more layers of regulation

Transcription is only the first step; the mature RNA and its translation are regulated by sequence features that a model of transcription alone does not see.

- **Splicing.** Introns are removed by the spliceosome, guided by the 5′ splice site (consensus ~`MAG|GURAGU`), the branch point, the polypyrimidine tract, and the 3′ splice site (`AG`), plus exonic and intronic **splicing enhancers and silencers** bound by SR proteins and hnRNPs. > 90% of multi-exon human genes undergo alternative splicing; an appreciable minority (order 10%) of disease-causing point mutations act by disrupting splicing, including *cryptic* splice sites deep within introns. Splicing prediction from sequence is among the most successful applications of deep learning (SpliceAI; Chapter 33).
- **3′ end processing.** Cleavage and polyadenylation near the signal `AAUAAA`; alternative polyadenylation changes 3′ UTR length and thus miRNA/RNA-binding-protein sites.
- **RNA stability and decay.** mRNA half-lives range from minutes to days (median several hours in mammalian cells); regulated by AU-rich elements, miRNA sites, m6A modification, codon optimality, and **nonsense-mediated decay**, which degrades transcripts with premature stop codons.
- **Translation.** Initiation at the start codon depends on the **Kozak** context and **upstream open reading frames (uORFs)** in the 5′ UTR; elongation speed depends on codon usage and tRNA availability; ribosome profiling measures ribosome positions genome-wide.
- **Protein degradation** and **post-translational modification** further decouple protein activity from mRNA level.

Across genes, mRNA abundance explains only a *moderate* fraction (order 40–50%) of protein abundance variance in many datasets, with the remainder reflecting translation and degradation. *A sequence model that predicts mRNA level from the promoter and gene body should not be expected to predict protein or phenotype without the UTR and coding-sequence features that govern post-transcriptional control* (Chapters 33, 41).

---

## 22.6 How regulation is measured: the assays

| Assay | What it measures | Resolution | Key biases / limitations | Typical reproducibility |
|---|---|---|---|---|
| **RNA-seq (bulk)** | RNA abundance (mixed cells) | gene/isoform | Composition averaging; length/GC bias; depth | Replicate correlation $>0.95$ for log expression of expressed genes |
| **scRNA-seq** | RNA counts per cell | gene | Capture efficiency; dropout; batch | Per-gene per-cell very noisy (Chapter 5, 25) |
| **CAGE / TSS-seq** | Transcription start sites and promoter activity | single-nucleotide TSS | Capping bias; low coverage for weak promoters | High at strong promoters |
| **ChIP-seq** | TF binding / histone marks in vivo | ~100–200 bp | Antibody quality; crosslinking; open-chromatin bias ("hyper-ChIPable" regions); peak-calling thresholds | Moderate; peak overlap often 60–90% between replicates |
| **CUT&RUN / CUT&Tag** | Binding/marks with lower background | ~bp–100 bp | Enzyme access; accessibility bias | Higher signal-to-noise than ChIP |
| **ATAC-seq / DNase-seq** | Chromatin accessibility | ~bp–100 bp | Cut/insertion sequence bias; GC bias | High at strong peaks |
| **Hi-C / Micro-C** | 3-D contacts | kb to nucleosome | Resolution vs. depth trade-off; single-cell data very sparse | Depends on depth |
| **MPRA (massively parallel reporter assay)** | Activity of $10^4$–$10^6$ designed sequences (typically 100–300 bp) in an *episomal* or integrated reporter | per-sequence | Out of genomic context; length limit; plasmid vs. integrated differences; barcode effects | Replicate $r\approx0.8$–$0.95$ |
| **STARR-seq** | Enhancer activity of genomic fragments | ~500 bp | Reporter context; plasmid episomal chromatin | Moderate–high |
| **CRISPRi/a screens** (with FlowFISH, scRNA-seq readouts) | *Causal* effect of repressing/activating an element on a gene in its native context | ~sgRNA window (hundreds of bp) | Incomplete knockdown; guide efficiency; indirect effects; power for small effects | Replicate-dependent; power limits |
| **Base/prime editing, saturation mutagenesis** | Effect of specific nucleotide changes | single-nucleotide | Editing efficiency, bystander edits | Variable |
| **Ribosome profiling** | Translating ribosome positions | codon | Artifacts of drugs/pausing; coverage | Moderate |
| **Mass spectrometry proteomics** | Protein abundance | protein/peptide | Dynamic range; incomplete coverage | Moderate |

**Which assay measures which arrow?** *Correlational* assays (ChIP, ATAC, RNA-seq) observe states; *reporter* assays (MPRA) observe cis activity out of context; *perturbation* assays (CRISPRi/a, editing) observe **causal effects in native context** at lower throughput. An ML model trained on one class of assay learns *that assay's* relationship, and the gap between them is the reason claims about regulation require orthogonal validation (Chapters 44, 46).

---

## 22.7 The regulatory code as a modeling target

### 22.7.1 The cis/trans decomposition

For gene $g$ in cell type (or state) $t$, write expression as

$$
y_{g,t}=F\big(\mathbf{s}_g;\ \boldsymbol\tau_t\big)+\varepsilon_{g,t},
$$

where $\mathbf{s}_g$ is the **cis** sequence (promoter, enhancers, gene body, UTRs) and $\boldsymbol\tau_t$ is the **trans** state (the concentrations and activities of TFs and cofactors, chromatin remodelers, RNA-binding proteins). The same cis sequence gives different outputs in different cells because $\boldsymbol\tau_t$ differs; the same trans state gives different outputs for different genes because $\mathbf{s}_g$ differs. This is the formal version of "one genome, hundreds of cell types."

**Consequences for models.**

1. *A model that takes only $\mathbf{s}_g$ must learn separate output functions per cell type* (multi-task heads, as in DeepSEA/Enformer), i.e., it *memorizes* $F(\cdot;\boldsymbol\tau_t)$ for the $t$ seen in training. It cannot generalize to a new cell type $t'$ unless $\boldsymbol\tau_{t'}$ is represented (via TF expression as input, via cell-type embeddings informed by other data).
2. *The decomposition identifies the variance sources:* between-gene variance (cis: large), between-cell-type variance (trans: large), between-individual variance for the same gene (small cis effects from variants), and gene × cell-type interaction (the regulatory logic: where models earn their keep).
3. *Interventions on trans variables* (knocking out a TF) require $\boldsymbol\tau$ as an input and a model of how $\boldsymbol\tau$ changes (Chapter 39); interventions on cis variables (editing an enhancer) require only $\mathbf{s}_g$ (Chapter 31).

### 22.7.2 Syntax: billboard or enhanceosome?

Two idealized models of enhancer organization: the **enhanceosome** (a rigid arrangement of motifs with precise spacing; e.g., the interferon-β enhancer) and the **billboard** (a flexible collection of motifs whose positions matter little; additive/independent contributions). Genomic data suggest a continuum, with most enhancers showing *soft* syntax (preferred spacings and combinations with weak constraint; Avsec et al., 2021; Chapter 10) and a few rigid cases. For models: architectures that allow **flexible, content-dependent interactions between motifs** with some positional sensitivity (convolution plus attention) capture both ends of the continuum.

### 22.7.3 Evolutionary turnover and redundancy

Enhancer *function* is often conserved across species even when enhancer *sequence* is not (**turnover** by gain and loss of TF sites), so conservation misses many functional elements. Conversely, **redundancy** (shadow enhancers; Hong et al., 2008) means that deleting one element often has little effect, hiding function in single-perturbation assays (Chapter 10, Worked Example 10.2). Both patterns reduce the correlation between *conservation*, *single-perturbation effect*, and *model attribution*, and each is a source of disagreement among methods (Chapters 18, 41).

---

## 22.8 Worked research examples

!!! example "Worked Research Example 22.1: A model trained on MPRA data predicts native-genome enhancer activity poorly"
    **Situation.** A convolutional model trained on lentiviral MPRA data (200-bp fragments) predicts held-out MPRA sequences with $r=0.85$ (close to the replicate ceiling), but correlates only $r=0.3$ with the effect of CRISPRi on the same elements in their native genomic contexts.

    **Question.** What explains the gap? What would you do?

    **Reasoning.**

    1. *What does each assay measure?* The MPRA reports the activity of an isolated 200-bp sequence driving a reporter from a minimal promoter, in a plasmid or integrated at a random site. CRISPRi reports the effect of silencing the *native* element on the *native* gene, which depends on chromatin context, element length, distance to the promoter, promoter identity, redundancy, and contact.
    2. *The measurement gap (G-M).* MPRA omits distal context, chromatin, native promoter, and neighboring elements. A perfect MPRA model captures *intrinsic sequence activity*, which is one factor of several.
    3. *Alternative explanations.* (H1) *Context gap*: native effect = intrinsic activity × contact × promoter responsiveness. (H2) *Redundancy*: native deletions are buffered. (H3) *Detection*: CRISPRi power for small effects is low, so measured native effects are noisy (a low ceiling); the correlation of 0.3 might be near the ceiling. (H4) *Distribution shift*: native elements are longer and located in regions with different GC content/chromatin than MPRA libraries.
    4. *Experiments.* (a) Compute the CRISPRi replicate ceiling; (b) fit a multiplicative model $\text{effect}\approx\text{MPRA activity}\times\text{contact}(d)\times\text{promoter factor}$ and see how much variance each term explains (the structure of the ABC model); (c) test the MPRA model's *relative* ranking of elements *within* a locus; (d) run MPRA in a *genomically integrated, native-context* format (CRISPR knock-in) for a subset; (e) double-perturbation for suspected redundancy.
    5. *Predictions.* Under H1, adding contact and promoter terms raises the correlation substantially; under H3 the gap closes once ceilings are accounted for; under H2, double perturbations reveal effects.

    **Expert analysis.** The result is not that the MPRA model is wrong but that it answers a narrower question. *Matching the assay to the claim* is the first rule: use MPRA-trained models for statements about intrinsic sequence activity and perturbation data for native regulatory effects (Chapters 31, 46).

!!! example "Worked Research Example 22.2: Is a deep model needed to link enhancers to genes? The strength of simple baselines"
    **Situation.** You want to predict which genes each enhancer regulates. Candidate approaches: (i) nearest gene; (ii) distance-weighted (closer is more likely); (iii) the **Activity-by-Contact (ABC)** model; (iv) a deep model on sequence.

    **Reasoning.**

    1. *ABC from first principles.* The effect of an enhancer on a promoter is approximately (enhancer activity) × (probability of enhancer–promoter contact), normalized across all enhancers acting on that gene (Fulco et al., 2019). Activity is estimated from accessibility and H3K27ac; contact from Hi-C or a power-law decay in genomic distance ($\propto d^{-1}$ roughly). This is a *mechanistic, two-parameter* model.
    2. *Baselines matter.* A substantial fraction of validated enhancers (reported as about half or fewer in CRISPRi screens) regulate a gene other than the nearest one, so "nearest gene" is a poor but non-trivial baseline; ABC predicts CRISPRi-validated links with much higher precision at a given recall in K562 data. [[S]]
    3. *What could a deep model add?* Sequence determinants of activity beyond DNase/H3K27ac signal, cell-type extrapolation, and *variant-level* effects. What it must beat on gene-level linking is ABC with the *same* epigenomic inputs.
    4. *What would convince you?* A comparison against ABC and distance baselines on held-out CRISPRi data from a *different cell type and lab*, with ceilings, including enhancers that regulate non-nearest genes.
    5. *Why does a simple model work?* It encodes a physical law (activity × contact). **Inductive biases built from biophysics** can match or beat flexible models in regimes with limited perturbation data (Chapter 7; attack A9).

    **Expert analysis.** Reporting a deep model without ABC and distance baselines on enhancer–gene linking is incomplete. The comparison also clarifies *what information* each method uses: ABC uses measured chromatin activity and contact; a sequence-only model uses neither, and its apparent success may reflect that sequence predicts chromatin.

---

## 22.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: separating cis and trans before modeling"
    **Problem.** A model predicts expression of each gene in 100 cell types from 200 kb of reference sequence and performs well on held-out *genes* but poorly on held-out *cell types*.

    **Decompose with $y_{g,t}=F(\mathbf{s}_g;\boldsymbol\tau_t)$.**

    1. *Held-out genes, seen cell types.* Tests generalization over $\mathbf{s}_g$ with $\boldsymbol\tau_t$ fixed: the model memorized each cell type's regulatory logic. High performance is expected.
    2. *Held-out cell types, seen genes.* Tests generalization over $\boldsymbol\tau_t$. A model with cell-type-specific output heads has *no way* to extrapolate: the new cell type's head is untrained. The failure is structural, not a matter of data volume.
    3. *Possible fixes (each an attack).* Provide $\boldsymbol\tau_t$ as input: TF expression, chromatin accessibility of the TF genes, or a cell-type embedding from single-cell data (A2, A4); train on many more cell types so that the model can interpolate (A6); make the output head a function of a *learned cell-state vector* (A8); incorporate a mechanistic factorization $F=\sum_k\text{(TF}_k\text{ activity in }t)\times\text{(TF}_k\text{ motif effect in }\mathbf{s}_g)$ (A9: a bilinear model whose learned cell-type factor is the TF activity).
    4. *Experiment.* Compare (i) per-cell-type heads, (ii) a cell-state embedding fed by TF expression, (iii) the bilinear factorization, on held-out cell types of increasing evolutionary/developmental distance from training. *Predict* that (i) fails, (ii)–(iii) degrade smoothly with distance.
    5. *Interpretation.* If (iii) recovers biologically sensible TF activities, the model has learned a *mechanistic* trans factorization (rung C3 of the Claim Ladder); if not, it is interpolation.

    **Distinguishing limitation types.** A cell-type-generalization failure in a multi-head model is an **architectural/representation** limitation (G-O/G-G), fixable by changing inputs and factorization; it is not evidence that "the regulatory code cannot be learned."

---

## 22.10 Connections

- **Backward:** motif information and the specificity problem (Chapter 5); the NB and bursting (Chapters 4, 19); CNN motif detectors (Chapter 10); PWMs and Potts models (Chapters 13, 29); coevolution and conservation (Chapter 21).
- **Forward:** classical regulatory models and gkm-SVM (Chapter 29); sequence-to-function deep models (Chapter 31); genomic language models (Chapter 32); splicing and translation models (Chapter 33); perturbation prediction and trans state (Chapter 39); regulatory variant interpretation (Chapter 41); open problems in regulation (Chapter 50); experimental design for MPRA and CRISPR screens (Chapter 46).

!!! takeaways "Key takeaways"
    1. Regulation = **cis sequence × trans state**; expression is $F(\mathbf{s}_g;\boldsymbol\tau_t)$. A sequence-only model has no way to generalize to new cell types without a trans representation.
    2. The **thermodynamic model** gives $\theta=\sigma(\ln c-\ln K_0-E(s))$; **PWMs are energy matrices**; CNN filter + sigmoid is the same model learned from data.
    3. **Specificity problem:** in a $10^6$-position genome, ~94% of bound TF sits at non-consensus sites (consensus fraction 0.057); chromatin accessibility, cooperativity, and context supply the missing bits.
    4. **Cooperativity** yields ultrasensitivity (Hill coefficient 1.19 → 1.99 as $\omega$ grows) and AND-like logic; the statistical-weights model is a softmax/Potts model over regulatory configurations.
    5. Layers beyond transcription (splicing, 3′ processing, translation, degradation) decouple mRNA from protein; mRNA explains only a moderate fraction of protein variance.
    6. **Assays differ in what they identify**: correlational (ChIP/ATAC/RNA-seq), reporter out of context (MPRA/STARR-seq), causal in native context (CRISPRi/a, editing). Match the assay to the claim.
    7. Simple **mechanistic baselines** (ABC, statistical-weights models) can match deep models in data-limited regimes; always include them.

---

## Further reading

- Ptashne, M. & Gann, A. (2002). *Genes and Signals*. Cold Spring Harbor Laboratory Press. Phillips, R., Kondev, J., Theriot, J. & Garcia, H. (2012). *Physical Biology of the Cell* (2nd ed.). Garland Science. (Thermodynamic models of regulation.)
- Berg, O. G. & von Hippel, P. H. (1987). Selection of DNA binding sites by regulatory proteins. *J. Mol. Biol.* 193, 723–750. Bintu, L. et al. (2005). Transcriptional regulation by the numbers: models. *Current Opinion in Genetics & Development* 15, 116–124. Segal, E. & Widom, J. (2009). From DNA sequence to transcriptional behaviour: a quantitative approach. *Nature Reviews Genetics* 10, 443–456. Polach, K. J. & Widom, J. (1995). Mechanism of protein access to specific DNA sequences in chromatin. *J. Mol. Biol.* 254, 130–149. Mirny, L. A. (2010). Nucleosome-mediated cooperativity between transcription factors. *PNAS* 107, 22534–22539.
- Lambert, S. A. et al. (2018). The human transcription factors. *Cell* 172, 650–665.
- Dixon, J. R. et al. (2012). Topological domains in mammalian genomes identified by analysis of chromatin interactions. *Nature* 485, 376–380. Rao, S. S. P. et al. (2014). A 3D map of the human genome at kilobase resolution. *Cell* 159, 1665–1680. Fudenberg, G. et al. (2016). Formation of chromosomal domains by loop extrusion. *Cell Reports* 15, 2038–2049.
- Fulco, C. P. et al. (2019). Activity-by-contact model of enhancer–promoter regulation from thousands of CRISPR perturbations. *Nature Genetics* 51, 1664–1669. Gasperini, M. et al. (2019). A genome-wide framework for mapping gene regulation via cellular genetic screens. *Cell* 176, 377–390.
- Larsson, A. J. M. et al. (2019). Genomic encoding of transcriptional burst kinetics. *Nature* 565, 251–254. Bartman, C. R., Hsu, S. C., Hsiung, C. C.-S., Raj, A. & Blobel, G. A. (2016). Enhancer regulation of transcriptional bursting parameters revealed by forced chromatin looping. *Molecular Cell* 62, 237–247.
- Melnikov, A. et al. (2012). Systematic dissection and optimization of inducible enhancers in human cells using a massively parallel reporter assay. *Nature Biotechnology* 30, 271–277. Arnold, C. D. et al. (2013). Genome-wide quantitative enhancer activity maps identified by STARR-seq. *Science* 339, 1074–1077. Hong, J.-W., Hendrix, D. A. & Levine, M. S. (2008). Shadow enhancers as a source of evolutionary novelty. *Science* 321, 1314.
- Jaganathan, K. et al. (2019). Predicting splicing from primary sequence with deep learning. *Cell* 176, 535–548.
- Avsec, Ž. et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nature Genetics* 53, 354–366.
- Schwanhäusser, B. et al. (2011). Global quantification of mammalian gene expression control. *Nature* 473, 337–342.
