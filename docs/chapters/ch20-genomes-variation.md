# Chapter 20. Genomes, Mutation, and Variation

!!! abstract "Chapter at a glance"
    **Motivation.** The genome is the raw material for every sequence model in Part VII. To use it well, a modeler must know what is *in* a genome, how mutation generates variation, which statistical regularities in sequence reflect mutation (not function), and what the reference genome leaves out.
    **Prerequisites.** Chapters 5 (mutation–selection equilibrium and the mutation-bias term), 19.
    **You will be able to:** (1) describe the composition and organization of the human genome and compare genome sizes across life; (2) distinguish classes of variation (SNVs, indels, structural variants, repeat expansions) and quote mutation-rate scales; (3) explain why mutation is strongly context-dependent (CpG hypermutability) and how that confounds frequency- and likelihood-based variant scores, with the simulation; (4) explain reference-genome bias and population structure of variation; (5) describe how genomes are sequenced and what each technology cannot see; (6) apply the seven-question ladder to a genome and to a variant.

---

## 20.1 What is a genome?

!!! bio "Biology for modeling: a genome"
    **What is it?** The complete DNA sequence of an organism (nuclear genome; plus organellar genomes such as the 16.6-kb human mitochondrial genome).
    **Information it contains.** Instructions for making all RNAs and proteins, the *cis*-regulatory sequence that determines when and where they are made, replication and chromosome-maintenance elements, and a large amount of sequence with no known function (repeats, degraded transposable elements, evolutionary history).
    **How is it generated?** By billions of years of mutation, recombination, drift, and selection (Chapter 21); each individual's genome is a recombined mosaic of its parents' genomes plus ~60–70 new mutations per generation in humans.
    **How is it measured?** DNA sequencing (short-read, long-read), assembly and variant calling; the output is a string over $\{A,C,G,T\}$ plus annotations (§20.6).
    **Computational representation.** A string; for variation, a reference plus a list of variants (VCF), or a genome graph; $k$-mer count vectors; tokenized sequence for language models.
    **What varies.** Between individuals (~0.1% of positions differ), between species (percent to tens of percent), along the genome (GC content, gene density, repeat content), and between cells (somatic mutation).
    **What can ML learn?** Sequence composition, repeat structure, conservation-linked constraint, coding structure, motif-level regularities; the *probability* of a sequence under evolutionary processes.
    **What can ML not observe?** Which cell type uses which sequence, how DNA is packaged and folded, which variants are causal in a given context, anything requiring perturbation or phenotype.

### 20.1.1 The human genome by the numbers

| Feature | Approximate value | Note |
|---|---|---|
| Size | 3.1 Gb (haploid); two copies per somatic cell | GRCh38 reference; T2T-CHM13 gave the first gapless assembly (Nurk et al., 2022) |
| Protein-coding genes | ~20,000 | GENCODE; many more transcripts ($>10^5$ with isoforms) |
| Protein-coding exons | ~1–1.5% of the genome | ~30 Mb |
| Repeat-derived sequence | ~50% | LINEs (~20%), Alu and other SINEs (~11%), LTR elements (~8%), DNA transposons (~3%), plus tandem repeats and satellites |
| Sequence under purifying selection | ~5–10% | Estimates range 3–15%; of this, only a minority is coding |
| GC content | ~41% (genome average) | Varies along the genome in large "isochores" (~100 kb–Mb) |
| CpG dinucleotide frequency | ~1/4 of the value expected from base composition | Depleted by methylation-driven mutation (§20.3.2) |
| Nuclear DNA length per cell | ~2 m of DNA in a ~10-µm nucleus | Folded hierarchically (Chapter 22) |

**Consequence for modeling.** Roughly 98% of the genome is non-coding. Most disease-associated variants from genome-wide association studies lie outside genes (Chapter 26). A model that predicts function from sequence must therefore be a model of *regulation* (Chapters 22, 31), not of coding sequence alone. At the same time, *most of the sequence is not under selection*: it is background that a sequence model must nevertheless predict (Chapter 3, Worked Example 3.1; Chapter 5).

### 20.1.2 Genome size across life

| Organism | Genome size | Genes (approx.) |
|---|---|---|
| *Escherichia coli* | 4.6 Mb | 4,300 |
| *Saccharomyces cerevisiae* (yeast) | 12 Mb | 6,000 |
| *Caenorhabditis elegans* | 100 Mb | 19,000 |
| *Drosophila melanogaster* | ~140 Mb | 14,000 |
| *Homo sapiens* | 3.1 Gb | 20,000 |
| Marbled lungfish; axolotl | ~40 Gb; ~32 Gb | ~$2\times10^4$–$5\times10^4$ |
| *Paris japonica* (a flowering plant) | ~150 Gb | — |

Genome size is only loosely correlated with organismal complexity or gene number (the "C-value paradox"): differences are dominated by repeat content and polyploidy. **Gene density and the proportion of coding sequence therefore differ by orders of magnitude between prokaryotic and mammalian genomes**: bacterial genomes are ~85–90% coding; this partly explains why language models trained on prokaryotic genomes (Evo; Chapter 32) find more transferable functional signal per nucleotide than models trained on the human genome.

---

## 20.2 Genes, transcripts, and annotation

A **gene** is a region transcribed into RNA (protein-coding or non-coding) with its regulatory context. In eukaryotes, genes consist of **exons** (retained in mature RNA) separated by **introns** (removed by splicing); a human protein-coding gene has on average ~9–10 exons and a long intron structure so that *the gene footprint (~25–50 kb, with some > 1 Mb) is far larger than its coding sequence (~1.5 kb)*. **Alternative splicing** and alternative promoters produce multiple **isoforms** (≥ 90% of multi-exon human genes are alternatively spliced). The functional output of a gene is thus not a single sequence but a *family of transcripts* with tissue-specific proportions.

**Annotation** (Ensembl/GENCODE, RefSeq) assigns gene models by combining transcript evidence, conservation, and computational prediction. It is *incomplete and evolving* (new genes and isoforms are added each release), *biased* toward well-studied tissues, and *versioned* (use consistent versions; Chapter 6). Any model that takes annotated features as input (transcription start sites, exon boundaries) inherits these biases.

---

## 20.3 Mutation

### 20.3.1 Types of variation

| Class | Size | Typical per-genome count (vs. reference) | Examples |
|---|---|---|---|
| Single-nucleotide variants (SNVs) | 1 bp | ~4–5 million | Transitions (A↔G, C↔T) are ~2× as frequent as transversions |
| Short insertions/deletions (indels) | 1–50 bp | ~0.5–1 million | Frameshifts in coding regions |
| Structural variants (SVs) | ≥ 50 bp | ~$10^4$ (long-read studies) | Deletions, duplications, inversions, insertions (including mobile elements), translocations |
| Copy-number variants (CNVs) | kb–Mb | tens–hundreds | Gene dosage changes |
| Repeat expansions | variable | few | Huntington's (CAG)$_n$; fragile X (CGG)$_n$ |
| Aneuploidy | whole chromosomes | rare germline; common in cancer | Down syndrome; tumor genomes |

Most of the **number** of variants are SNVs; most of the **base pairs** affected are in SVs and repeats. Short-read technology sees SNVs and small indels well and SVs and repeats poorly (§20.6), so *what we know about variation is biased toward the variant types that are easy to measure*.

### 20.3.2 Mutation rates and context dependence

The human germline mutation rate is about $1.2\times10^{-8}$ per base pair per generation, i.e. **~60–70 new single-nucleotide mutations per child**, rising with paternal age (Kong et al., 2012). Somatic mutation rates are higher and tissue- and process-dependent; typical cancer genomes carry ~1–10 mutations per megabase, with hypermutated tumors above 10 (e.g., mismatch-repair deficient).

Mutation is **not uniform**. Three regularities every sequence modeler must know:

1. **Transitions outnumber transversions.** The transition/transversion ratio (Ts/Tv) is ~2 genome-wide, though a random substitution would give 0.5 (there are twice as many transversion outcomes as transitions).
2. **CpG hypermutability.** Cytosine in a CpG dinucleotide is frequently methylated (5-methylcytosine). Spontaneous deamination converts 5mC to thymine, an error that is poorly repaired, so **CpG→TpG transitions occur ~10× faster than other transitions**. This single mechanism explains the *depletion* of CpG in the genome (the CpG dinucleotide is ~4× rarer than base composition predicts) and is the dominant source of recurrent mutation in humans.
3. **Local sequence context and replication/repair dependence.** Rates depend on the flanking bases (the trinucleotide context), replication timing, transcription (strand asymmetry), and exposures (UV, tobacco, alkylating agents). Cancer genomics decomposes somatic mutations into **mutational signatures** (COSMIC SBS signatures; Alexandrov et al., 2013): SBS1 is the clock-like CpG signature, SBS7 UV, SBS4 tobacco.

**Why this matters for ML.** In Chapter 5 we derived that, at mutation–selection equilibrium, the log-likelihood ratio of a variant satisfies
$$
\log\pi(x')-\log\pi(x)=2(N-1)\,\Delta\log f+\underbrace{\log\frac{\mu(x\to x')}{\mu(x'\to x)}}_{\text{mutation bias}},
$$
so a sequence model trained on a corpus of related genomes, or on an alignment, will assign **much higher likelihood to frequently mutating changes even if they are neutral**. The **simulation** below makes this concrete.

### 20.3.3 A simulation: mutation bias versus selection

We simulate 800 independent lineages diverging from a common ancestral sequence of 6,000 sites, with mutation rates that include Ts/Tv = 2 and a 10-fold CpG→TpG boost, and with graded purifying selection (a fraction 0.3–0.9 of mutations purged) at 20% of sites. We then compute the frequency-based score $\log\big((f_\text{alt}+10^{-3})/(f_\text{ref}+10^{-3})\big)$ for every possible substitution, the same quantity a simple independent-site model of the alignment would assign.

```python
--8<-- "code/ch20_mutation_bias.py"
```

Output:

```text
mean log-likelihood-ratio score  log(f_alt / f_ref)  for NEUTRAL substitutions, by mutation class:
  CpG transition : mean score  -1.49  (mean substitution frequency 0.226)
  transition     : mean score  -3.87  (mean substitution frequency 0.021)
  transversion   : mean score  -5.17  (mean substitution frequency 0.006)
  constrained sites (all classes): mean score  -5.50

AUROC for separating constrained from neutral substitutions:
  raw score, all substitution classes pooled          : 0.777
  score corrected for mutation rate, pooled           : 0.894
  raw score within CpG transition  only                  : 1.000  (n = 768)
  raw score within transition      only                  : 0.964  (n = 5232)
  raw score within transversion    only                  : 0.875  (n = 12000)
```

**Reading the result.**

- The score *among neutral substitutions* spans from $-1.49$ (CpG transitions) to $-5.17$ (transversions): a range of 3.7 nats caused entirely by **mutation rate**, with no selection involved.
- The score at constrained sites averages $-5.50$: only 0.33 nats below neutral transversions. **The selection signal is smaller than the mutation-class spread**, so *pooling classes* gives AUROC 0.777, well below what *within-class* comparison achieves (CpG: 1.000; transitions: 0.964; transversions: 0.875).
- **Correcting for the mutation model** (subtracting $\log\mu$, i.e., the second term of the Sella–Hirsh relation) raises the pooled AUROC to **0.894**.

**Implications.** (i) Any evaluation of a likelihood-based variant-effect score should *stratify by mutation type* (CpG vs. non-CpG; transitions vs. transversions) or *calibrate against a mutation-rate model*. (ii) Constraint metrics from population databases (observed/expected loss-of-function variants; Karczewski et al., 2020) use a **context-dependent mutation model** to define the "expected" count for exactly this reason. (iii) A large language model trained on real genomes learns the mutation spectrum as part of the sequence distribution: its $\Delta\log p$ conflates mutability and fitness unless the effect is controlled. [[E]] for the mathematics; [[S]] for the empirical impact on real genomic language models (Chapter 32).

---

## 20.4 Variation across individuals and populations

A random human is **heterozygous at ~3 million positions** (nucleotide diversity $\pi\approx1/1000$) and carries ~4–5 million differences from the reference. Several population-level facts shape any model trained or tested on human variation:

1. **Most variants are rare.** Under neutrality the expected number of variants at allele count $i$ in a sample of size $n$ is proportional to $1/i$ (Chapter 21), so the site-frequency spectrum is dominated by singletons. In large samples (gnomAD, with ~800,000 individuals) the majority of variants are seen once or twice. **Rare variants carry most of the (non-reference) novelty and are the variants that clinical interpretation must handle.**
2. **Variation is structured by ancestry.** Genetic diversity is highest in African populations and decreases with distance from Africa (serial founder effects). Allele frequencies, LD patterns, and the sets of variants differ across populations. Datasets and references are *skewed toward European ancestry* (§20.5; Chapter 26).
3. **Hardy–Weinberg equilibrium (HWE).** Under random mating with no selection, migration, or drift, a biallelic locus with allele frequencies $p,q=1-p$ has genotype frequencies $p^2$, $2pq$, $q^2$. Departures indicate genotyping error, population structure, or selection; HWE tests are a standard data-quality filter.
4. **Selection shapes the spectrum.** Variants that damage important genes are removed; the *depletion* of loss-of-function variants relative to expectation (pLI/LOEUF constraint scores) marks genes intolerant to heterozygous loss. Constraint is itself a signal that sequence models can learn (Chapters 21, 41).

!!! note "Phased haplotypes"
    An individual carries two *haplotypes* (one per chromosome copy). Two variants in the same gene may be on the same haplotype (*cis*) or on different ones (*trans*), with different functional consequences (a compound heterozygote with one damaging variant on each copy has no functional copy; two on the same copy leave the other intact). **VCF genotypes without phasing lose this information**; sequence models of personal genomes need phased haplotype sequences (Chapter 41).

---

## 20.5 Reference genomes and their biases

The **reference genome** (GRCh38; T2T-CHM13) is a single *linear* sequence used as a coordinate system. It is a mosaic of a small number of individuals (GRCh38 is ~70% from a single donor), and therefore:

- **Reference bias**: reads carrying non-reference alleles align less well, so non-reference variants are under-called, especially in divergent regions.
- **Ancestry bias**: the reference reflects a limited set of ancestries; sequences present in other populations may be absent.
- **Representation of variation**: a linear reference cannot represent large structural variation or highly polymorphic regions (HLA, KIR) well; pangenome *graphs* can. The Human Pangenome Reference Consortium released a first draft in 2023 based on 47 diverse individuals (94 haplotypes; Liao et al., 2023), with expanded releases following.
- **Training-data implications**: a sequence model trained on the reference sees *one* haplotype per locus; its learned distribution is the *distribution of sequences across loci in that haplotype*, not the *population distribution of alleles at a locus*. Estimating the effect of a variant by in silico mutation of the reference assumes that the variant acts on the reference background, ignoring linked variants on the individual's actual haplotype (Chapter 41).

---

## 20.6 How genomes are measured

| Technology | Read length | Accuracy | Strengths | Blind spots |
|---|---|---|---|---|
| **Illumina (short-read)** | 100–300 bp | ~99.9% per base | Cheap, high-throughput, accurate SNVs | Repeats, SVs, phasing, GC-extreme regions |
| **PacBio HiFi** | 10–25 kb | >99.9% (consensus) | Accurate long reads; phasing; SVs | Higher cost per base |
| **Oxford Nanopore** | 10 kb–Mb | ~99%+ (improving) | Ultra-long reads; base modifications; direct RNA | Homopolymer indels; systematic error modes |
| **Hi-C / optical mapping** | N/A | — | Scaffolding; large-scale structure | Base-level detail |

Typical human whole-genome sequencing uses ~30× coverage (each base covered by 30 reads on average). The pipeline: reads → alignment to a reference (Chapter 27) → duplicate marking and recalibration → variant calling (GATK; **DeepVariant**, a convolutional network on pileup images, was an early demonstration of deep learning in genomics; Poplin et al., 2018) → filtering → annotation. **Errors are not uniform**: false positives and false negatives cluster in repeats, low-complexity regions, segmental duplications, and regions of low mappability. A model that uses variant calls as ground truth learns from a *censored* picture of variation.

**Complete assemblies** (T2T-CHM13; Nurk et al., 2022) resolved centromeric satellites, segmental duplications, and short arms of acrocentric chromosomes, adding ~200 Mb that earlier references missed. Those regions are repetitive and largely absent from training sets built on older references, so *models trained on GRCh38 have never seen a large class of sequence*.

---

## 20.7 Genomes as the corpus for sequence models

Sequence models in Part VII are trained on genomes from public archives (NCBI/ENA GenBank and RefSeq, Ensembl, the Genome Taxonomy Database, metagenomic collections). The scale of such corpora is now $10^{12}$–$10^{13}$ nucleotides (Evo 2: over 100,000 species, ~9 trillion nucleotides; Chapter 32). Four facts about the corpus matter:

1. **It samples life unevenly.** Model organisms, pathogens, and cultivable microbes dominate; most of microbial diversity is uncultured; eukaryotic coverage is improving (Earth BioGenome, Vertebrate Genomes, Zoonomia).
2. **It is phylogenetically structured.** Genomes of related species and strains are near-copies (Chapter 21); effective sample size is the number of independent lineages, not nucleotides. Deduplication and taxonomic balancing are modeling decisions.
3. **It contains mutation spectrum and selection together.** A model's learned distribution mixes mutational biases (§20.3.2) and functional constraint; separating them requires evolutionary models or controlled evaluation.
4. **It is annotation-poor.** Function is known for a tiny fraction of sequences. Evaluation requires external labels (assays, annotations) with their own biases.

---

## 20.8 Worked research examples

!!! example "Worked Research Example 20.1: A genomic language model's variant score correlates with conservation but fails on CpG sites"
    **Situation.** A DNA language model's $\Delta\log p$ for single-nucleotide variants correlates $\rho=0.4$ with a cross-species conservation score overall, but when the authors stratify by mutation type, performance on CpG transitions is near chance and on non-CpG transversions is strong.

    **Question.** Is this a model failure? What does it imply for evaluation?

    **Reasoning.**

    1. *What does the simulation predict?* A likelihood-based score reflects both selection and mutation rate (§20.3.3). For CpG transitions the *mutation rate* is so high that they are frequent even at moderately constrained sites; the score reflects mostly the rate and the difference between constrained and neutral CpG transitions is smaller *relative to noise* after saturation.
    2. *Alternative explanations.* (H1) The model has learned the mutation spectrum correctly (not a failure; a different quantity). (H2) Conservation scores themselves are saturated or poorly estimated at CpG sites (multiple hits: a site can mutate several times on a lineage, saturating divergence-based scores). (H3) Methylation-dependent CpG status is context- and tissue-specific and not in the sequence.
    3. *Experiments.* Stratify by CpG status and mutation class; compare with a baseline that uses a *context-dependent mutation model* alone; evaluate the *mutation-rate-adjusted* score (Sella–Hirsh calibration); compare against functional readouts (MPRA, ClinVar) *within* mutation class.
    4. *Predictions.* If H1, the raw score tracks mutation-rate predictions (correlates with population allele frequency and with the mutation model); after adjustment, conservation correlations in CpG sites improve.

    **Expert analysis.** The stratified analysis converts an apparent failure into a measurement of *how much of the model's score is mutation bias*. The scientifically correct reporting is a table of performance by mutation class with a mutation-model baseline, and a calibrated score for clinical use (Chapters 4, 32, 41).

!!! example "Worked Research Example 20.2: Training on the reference genome and scoring a population's variants"
    **Situation.** A model trained on the reference genome is used to predict the effects of variants observed in individuals of a population poorly represented in the reference. Performance is lower than in the benchmark population.

    **Reasoning.**

    1. *What is the training distribution?* One haplotype per locus; sequence features specific to other ancestries (lineage-specific insertions, population-specific alleles at LD-linked sites) are absent.
    2. *Which variants are most affected?* (i) Variants in regions missing or divergent in the reference; (ii) variants whose effect depends on *linked* variants on the same haplotype (the reference haplotype differs); (iii) population-specific alleles that the model sees as "out of distribution."
    3. *Alternative explanations.* (H1) Reference bias in training; (H2) differences in LD and allele frequency that affect the *benchmark labels* (e.g., eQTL effect sizes estimated from tagging variants); (H3) different noise ceilings in the benchmark across populations (smaller samples → noisier labels).
    4. *Experiments.* Evaluate on *haplotype-resolved* personal genome sequences; compare on regions with and without reference gaps; compute the noise ceiling per population; test with variants fine-mapped to be causal in each population.
    5. *Predictions.* If H1 dominates, performance improves when the model is fine-tuned on haplotypes from the target ancestry; if H2/H3, benchmark labels, not the model, explain the gap.

    **Expert analysis.** *Generalization across ancestry is a generalization across data-generating processes at the level of labels as well as inputs.* Fairness in genomic AI starts with measuring each ancestry's ceiling and with representative training and reference data (Chapters 26, 41, 45).

---

## 20.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: separating mutation from selection in a sequence model"
    **Setting.** You trained a language model on aligned genomes and want to claim it captures *functional constraint*.

    **Decompose.**

    1. **Name the confounders.** Mutation rate (CpG, trinucleotide context, replication timing), alignment/annotation artifacts (low-mappability regions look "variable"), repeat content, GC content.
    2. **State the null.** The model's score is explained by a *mutation model alone* (context-dependent substitution rates fitted to neutral sites such as ancestral repeats).
    3. **Test.** Fit the null model on neutral sequence; compute the residual of your model's score after regressing out the null model's predicted rate; evaluate the *residual* against independent functional data (saturation mutagenesis, MPRA, ClinVar benign/pathogenic).
    4. **Alternatives.** If the residual still predicts function, the model contains constraint information beyond mutation rate. If not, it has learned the mutation spectrum.
    5. **What would change your mind?** A residual that predicts functional effects *within* mutation class and in regions where conservation scores are uninformative would be positive evidence; a model whose raw score is dominated by CpG status is a mutability predictor.

    **Distinguishing limitation types.** Mutation-rate confounding is a *known, modelable* nuisance (control it); lack of function-relevant information is a *representation/objective* limit (change the objective or data). The notebook converts a vague claim ("learns constraint") into a testable decomposition.

---

## 20.10 Connections

- **Backward:** mutation bias term of the Sella–Hirsh relation (Chapter 5, §5.6.4); substitution rates as Markov chains (Chapter 8); Chapter 6's coordinate and VCF conventions; the measurement view (Chapter 1).
- **Forward:** population genetics (drift, selection, coalescent, LD; Chapter 21); regulation of expression (Chapter 22); sequence alignment and variant calling (Chapter 27); tokenization and genome representations (Chapter 28); genomic language models and their evaluation (Chapter 32); genotype-to-phenotype models and personal genomes (Chapter 41).

!!! takeaways "Key takeaways"
    1. The human genome is ~3.1 Gb, ~98% non-coding, ~50% repeat-derived, ~5–10% under constraint; genome size does not track complexity.
    2. Variation spans SNVs, indels, SVs, CNVs, and repeat expansions; the **germline mutation rate is $\sim1.2\times10^{-8}$/bp/generation** (~60–70 new SNVs per child); most variants are rare.
    3. **Mutation is context-dependent**: Ts/Tv ≈ 2; **CpG→TpG transitions are ~10× faster**, explaining CpG depletion.
    4. **A frequency- or likelihood-based variant score conflates mutation rate and selection**: neutral substitutions spanned a 3.7-nat range in the simulation, larger than the 0.33-nat selection signal; pooled AUROC 0.777, mutation-corrected 0.894, within-class up to 1.000.
    5. The **reference genome** is a single, ancestry-skewed linear sequence; genotypes without phasing lose the cis/trans information.
    6. Sequencing technologies have systematic **blind spots** (repeats, SVs, phasing for short reads), so variant catalogues are censored.
    7. Genome corpora sample life unevenly and phylogenetically; **effective sample size is the number of independent lineages**, not nucleotides.

---

## Further reading

- Lander, E. S. et al. (2001). Initial sequencing and analysis of the human genome. *Nature* 409, 860–921. Nurk, S. et al. (2022). The complete sequence of a human genome. *Science* 376, 44–53. Liao, W.-W. et al. (2023). A draft human pangenome reference. *Nature* 617, 312–324.
- 1000 Genomes Project Consortium (2015). A global reference for human genetic variation. *Nature* 526, 68–74. Karczewski, K. J. et al. (2020). The mutational constraint spectrum quantified from variation in 141,456 humans. *Nature* 581, 434–443. Chen, S. et al. (2024). A genomic mutational constraint map using variation in 76,156 human genomes. *Nature* 625, 92–100.
- Kong, A. et al. (2012). Rate of de novo mutations and the importance of father's age to disease risk. *Nature* 488, 471–475. Alexandrov, L. B. et al. (2013). Signatures of mutational processes in human cancer. *Nature* 500, 415–421.
- Rands, C. M., Meader, S., Ponting, C. P. & Lunter, G. (2014). 8.2% of the human genome is constrained. *PLoS Genetics* 10, e1004525. Lindblad-Toh, K. et al. (2011). A high-resolution map of human evolutionary constraint using 29 mammals. *Nature* 478, 476–482.
- Poplin, R. et al. (2018). A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* 36, 983–987.
- Sella, G. & Hirsh, A. E. (2005). The application of statistical physics to evolutionary biology. *PNAS* 102, 9541–9546. (Mutation–selection equilibrium; Chapter 5.)
- Zoonomia Consortium (2020). A comparative genomics multitool for scientific discovery and conservation. *Nature* 587, 240–245. Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361.
