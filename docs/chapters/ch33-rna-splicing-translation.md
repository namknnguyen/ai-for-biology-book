# Chapter 33. RNA, Splicing, and Translation

!!! abstract "Chapter at a glance"
    **Motivation.** Between DNA and protein lies a series of RNA processing steps, each with a sequence code that models can learn: splicing decides which parts of a transcript are kept; untranslated regions and codons set translation and stability; RNA folds into structures whose accuracy is the single hardest prediction task in molecular biology after protein dynamics. This chapter treats the RNA layer as a set of well-defined prediction problems, explains why splicing has been the most successful case for deep learning (and why even there the within-gene variant question of Chapter 31 recurs), and tests the classical thermodynamic baseline for structure on real tRNAs.
    **Prerequisites.** Chapters 5, 6, 10, 22, 28, 29, 31, 32.
    **You will be able to:** (1) list the sequence signals of splicing, 3′ end processing, translation, and decay and what a model must do to decode them; (2) explain information accounting for splice signals (why local motifs are insufficient and context is needed); (3) describe SpliceAI and its successors, and the evidence for and limits of splice-variant prediction; (4) describe MPRA-trained models of UTR translation and RNA stability; (5) derive the dynamic-programming structure of RNA folding and test it on real tRNAs; (6) assess claims of RNA language models with family-level generalization in mind.

---

## 33.1 The seven questions for an RNA

!!! bio "Biology for modeling: a transcript"
    **What is it?** A single-stranded RNA copy of a gene: precursor mRNA (pre-mRNA) with exons and introns; mature mRNA (a 5′ cap, a 5′ untranslated region, a coding sequence, a 3′ UTR, a poly(A) tail); or a non-coding RNA (tRNA, rRNA, snRNA, lncRNA, miRNA) whose function depends on structure.
    **Information it contains.** The primary sequence (including signals for processing, localization, translation, decay), its secondary and tertiary structure (stems, loops, pseudoknots), modifications (m$^6$A, pseudouridine), and its isoform composition.
    **How is it generated?** Transcription, co-transcriptional capping and splicing, 3′ cleavage and polyadenylation, nuclear export, translation by ribosomes, decay by exonucleases and miRNA pathways.
    **How is it measured?** RNA-seq (short or long reads; isoform quantification, Chapter 27), CAGE (5′ ends), ribosome profiling and polysome profiling (translation), SHAPE/DMS chemical probing (structure), MPRA of UTRs and splicing minigenes (sequence variants), mass spectrometry (proteins).
    **Computational representation.** Sequence; secondary structure as a base-pair matrix or dot-bracket string; a graph; embeddings from RNA language models.
    **What varies.** Isoform usage across tissues, cell states, and individuals; structure across conditions (temperature, ions, binding proteins); translation across cells.
    **What can ML learn?** Splice sites and the effect of variants on them; ribosome load and half-life from UTR and CDS sequence (in the contexts where MPRAs were done); coding structure; secondary structure for families it has seen.
    **What can ML not observe?** Trans-acting factor concentrations (RBPs, miRNAs, spliceosome components) that vary by cell; structure as an ensemble in vivo; co-transcriptional kinetics; the cell's translational capacity.

---

## 33.2 Signals and the information problem

**Splicing.** Introns are removed by the spliceosome, which recognizes: the **5′ splice site** (donor; consensus around `MAG|GURAGU`), the **branch point** (an A, 18–40 nt upstream of the 3′ splice site), the **polypyrimidine tract**, and the **3′ splice site** (acceptor; `YAG|`), plus **exonic and intronic splicing enhancers and silencers** bound by SR proteins and hnRNPs. Mammalian exons are short (median around 120–150 nt) and introns long (median several kb), so the spliceosome recognizes **exons** (exon definition) as units. About 95% of human multi-exon genes undergo alternative splicing [[E]].

**Why local motifs are not enough: information accounting.** Suppose a donor signal has information content $I$ bits (as in Chapter 29, $I=\sum_j\mathrm{KL}(P_j\Vert B)$). In random sequence, the expected number of chance matches at a threshold is about $N/2^{I}$ per $N$ bases; for $I=8$ bits that is about 4 per kilobase on one strand. A human gene has tens of kilobases of intron, so thousands of sequences resemble a donor but only a few hundred are used. **The discrimination comes from context**: the surrounding exonic and intronic sequence, the presence of a matching acceptor at an exon-like distance, enhancer and silencer content, RNA structure, and the cell's splicing factors. Any model of splicing must integrate information over hundreds to thousands of nucleotides, which is why local motif models plateau and why context length matters (Chapters 10–11).

**3′ end processing** (the polyadenylation signal `AAUAAA` and downstream elements), **nuclear export and localization** (zipcodes in 3′ UTRs), **translation initiation** (a cap-proximal scanning ribosome selects the first start codon in a good Kozak context, `gccRccAUGG`; upstream open reading frames reduce translation), **codon usage** (tRNA availability and elongation speed), and **decay** (AU-rich elements, miRNA sites, m$^6$A) are further sequence-encoded signals. Most are learnable from MPRA data, because libraries of random or designed sequence variants provide labeled examples *in a specific cell type* (Chapter 25's remark: the cell type enters through the measurement, not the model input).

---

## 33.3 Splicing: SpliceAI and the deep-learning paradigm

**SpliceAI** (Jaganathan et al., *Cell*, 2019) predicts, for each nucleotide of a pre-mRNA, the probability of being a splice donor, an acceptor, or neither, from a window of 10 kb of sequence (5 kb either side) by a **32-layer deep residual network of dilated convolutions** (Chapter 10). Training used the GENCODE-annotated splice junctions (about 130,000 donor–acceptor pairs, split by chromosome; paralog handling included). [[S]]

**Variant effect.** Compare predictions for reference and alternative sequences, $\Delta=\max_{\text{window}}\lvert p_\text{alt}-p_\text{ref}\rvert$ over donor/acceptor gain and loss within a window (e.g., ±50 bp). The paper reported that cryptic splice variants explain a notable fraction of pathogenic de novo mutations in neurodevelopmental disorders (about 9–11%) and that about 75% of predicted cryptic splice variants validated in RNA-seq data from patients [[S]]. The model transfers to variants deep in introns, far from the canonical splice sites, because its context spans 10 kb: the first major demonstration that a learned sequence model *extends clinical variant interpretation beyond the exome*.

**Why splicing worked better than regulation.** (i) The labels are *annotated junctions*, a clean, nearly noise-free, sequence-determined target (contrast with coverage tracks, Chapter 27); (ii) the signals are short and strong, so the model's job is *context integration*, not integration of cell-specific factors; (iii) a variant effect on a splice site is often *large* (loss of a donor), so the counterfactual is large relative to model error. These differences explain why Chapter 31's identifiability problem is milder here, though not absent.

**Limits and successors.** (a) *Tissue specificity*: SpliceAI is not tissue-specific; **Pangolin** (Zeng & Li, 2022) predicts tissue-specific splice-site usage and extends across species [[S]]; **long-read isoform** data and models that predict exon inclusion (percent spliced in, PSI) in each tissue extend the task. (b) *Quantitative effects*: predicting *how much* a variant changes inclusion is harder than predicting that it creates a site. (c) *Calibration and thresholds*: a score $\Delta>0.2$ is used as a clinical filter; its positive predictive value depends on prevalence (Chapter 43: at low prevalence even a good classifier yields mostly false positives). (d) *Benchmark*: recent genome language models report zero-shot splice-variant performance on curated sets such as SpliceVarDB (Evo 2: highest zero-shot performance for exonic and intronic splice variants among models compared, by the authors' evaluation) [[S]], to be compared with SpliceAI-type supervised baselines under the same leakage controls (Chapters 28, 43).

!!! paper "Paper dissection: SpliceAI (Jaganathan et al., *Cell* 176:535–548, 2019)"
    **Problem.** Non-coding genetic variants that disrupt splicing cause disease but are hard to identify, because splicing depends on long-range context and signals deep in introns.
    **Key insight.** Train a very deep, dilated convolutional network to predict splice sites *directly from pre-mRNA sequence over a 10-kb window*, so that context is learned rather than hand-engineered; then score variants as differences in predicted splice probabilities.
    **Architecture.** A 32-layer residual network with dilated convolutions; input one-hot (4 channels × 10,000 + flanks), output three-class probabilities per position (acceptor, donor, neither). Ensemble of models.
    **Objective.** Cross-entropy on annotated splice sites (GENCODE), per position; chromosome-based train/test split with paralog-aware filtering.
    **Data.** Reference genome and GENCODE annotations; evaluation in RNA-seq (GTEx) and in patient cohorts (de novo mutations in neurodevelopmental disorders).
    **Results (as reported).** High top-$k$ accuracy for annotated junctions; cryptic splice variants predicted and validated in RNA-seq (about 75%); enrichment of predicted cryptic splice mutations in individuals with neurodevelopmental disorders relative to controls (explaining about 9–11% of pathogenic de novo variants in these cohorts). [[S]]
    **Why it worked.** Large context and depth; clean labels; a strong prior that splice signals are local plus context; evaluation on held-out genes and on orthogonal RNA-seq validation.
    **Assumptions.** Annotated junctions are the truth; splice-site choice is determined by local sequence; the effect of a variant on splicing equals the change in the predicted site strength.
    **Limitations.** No tissue specificity; no quantitative inclusion; dependence on annotation; some predicted effects not seen in the tissue examined.
    **What followed.** Pangolin and other tissue-specific models; integration into clinical variant pipelines; long-read–based isoform predictors; evaluation against genome language models.
    **Unresolved.** Quantitative, cell-type-specific, and combinatorial (multi-variant) prediction; the role of RNA structure and of trans-acting factors; calibrated clinical thresholds.

---

## 33.4 Translation, UTRs, and stability

**Translation efficiency from the 5′ UTR.** Sample et al. (2019) made a library of random 5′ UTR variants, measured the mean ribosome load of each by polysome profiling (hundreds of thousands of sequences), and trained a convolutional network (**Optimus 5-Prime**) to predict it; the model generalized to held-out random UTRs and to designed sequences, which the authors used to design UTRs with targeted ribosome loads [[S]]. The same design logic gives a template for MPRA-trained models: **a library of random or varied sequences in a defined context gives independently varying features and therefore identified effects** (Chapter 31's remedy, built into the experiment). The price is *context specificity*: the model describes translation of this reporter in these cells and may not transfer to a different CDS or cell type (Chapter 45). Language-model extensions (UTR-LM; Chu et al., 2024) add pretraining on natural UTRs [[S]].

**Stability.** **Saluki** (Agarwal & Kelley, 2022) predicts mRNA half-life from the full transcript sequence including splice-site and codon information and *the 3′ UTR*, using a hybrid convolutional/recurrent network trained on compiled half-life measurements across many cell types [[S]]. Half-life is a function of codon optimality, UTR elements (miRNA sites, AU-rich elements), and m$^6$A, which a model can learn from sequence only to the extent that their effects are context-independent.

**Codon usage and design.** Synonymous codons are not equivalent: they differ in tRNA availability, translation speed, co-translational folding, and mRNA structure. In our chloroplast data, the codon-position-specific statistics alone were worth 0.05 bits per base of held-out prediction, twice the information of all local context (Chapter 28). **mRNA design** for vaccines and therapeutics optimizes codons and structure jointly: **LinearDesign** (Zhang et al., 2023) casts the search over all synonymous sequences encoding a protein as a lattice-parsing problem (a dynamic program similar in spirit to Chapter 6's HMM algorithms) that finds sequences of maximal stability (minimum free energy) while honoring codon usage [[S]]. This is **A8 (reformulation) and A9 (biological constraint)** in the language of Chapter 55: the genetic code is a hard constraint that turns an astronomically large design problem into a tractable one.

---

## 33.5 RNA secondary structure: thermodynamics as the classical baseline

RNA folds by base pairing: Watson–Crick (A–U, G–C) and wobble (G–U) pairs forming stems, with loops between them. Under the **nearest-neighbor thermodynamic model** (Turner rules), the free energy of a secondary structure is the sum of contributions of its loops (hairpin, bulge, internal, multibranch) and the **stacking** of adjacent pairs, with parameters measured on small oligonucleotides (Turner & Mathews, 2010). The **minimum free energy (MFE)** structure is found by dynamic programming.

**The recursion (Nussinov, Zuker).** Let $W(i,j)$ be the optimal energy of the subsequence $i..j$ and $V(i,j)$ the optimal energy given that $i$ and $j$ pair. For the simplest (Nussinov) model, which maximizes the number of base pairs,

$$
W(i,j)=\max\Big\{W(i{+}1,j),\ W(i,j{-}1),\ W(i{+}1,j{-}1)+\delta(i,j),\ \max_{i<k<j}\big[W(i,k)+W(k{+}1,j)\big]\Big\},
$$

with $\delta(i,j)=1$ if bases $i$ and $j$ can pair. Zuker's algorithm replaces the pair count by loop energies (hairpin, stack, bulge, interior, multi-loop terms), and runs in $O(N^3)$ time and $O(N^2)$ memory for length $N$ (excluding pseudoknots, which make the problem NP-hard in general). **McCaskill's** partition function algorithm computes $Z=\sum_Se^{-E(S)/RT}$ and the **base-pair probabilities** $P(i,j)$ in $O(N^3)$, and thus an *ensemble* instead of a single structure (the Boltzmann weights of Chapter 22 once more).

**An experiment on real tRNAs** (`code/ch33_rna_structure.py`; ViennaRNA implementation of the Zuker algorithm). The chloroplast genome of *Arabidopsis* (Chapter 28) has 28 single-exon tRNA genes of 60–95 nt (mean 75 nt, mean GC 0.52). All tRNAs share a universal **cloverleaf**: a 7-bp acceptor stem enclosing three hairpins (D-arm, anticodon arm, T-arm).

| Measure on the 28 MFE structures | Result |
|---|---|
| Acceptor stem recovered (≥6 of the 7 terminal base pairs) | **25/28 = 0.89** |
| Full cloverleaf topology (acceptor stem enclosing exactly three hairpins) | **9/28 = 0.32** |
| Mean MFE | −25.0 kcal/mol (−0.33 per nucleotide) |

The thermodynamic model finds the dominant stem in 89% of tRNAs but the full four-stem topology in only 32%: **the MFE structure is not the native structure**, as is well known: a native fold may be a *suboptimal* member of the ensemble, stabilized by tertiary interactions (the D- and T-loop contact that defines the L-shape; Mg$^{2+}$) and by modifications that the nearest-neighbor model does not include. A second question, *can folding stability detect structured RNAs?*, was posed as the classic task of RNA gene finding. We scored the 28 tRNAs against 1,500 75-nt windows from sequence outside tRNA/rRNA genes (coding and non-coding):

| Detector | AUROC vs all windows | AUROC vs a GC-range-matched subset (435 windows) |
|---|---|---|
| GC content | 0.960 | 0.864 |
| MFE per nucleotide | 0.994 | 0.989 |
| MFE z-score vs 60 Markov (dinucleotide-preserving) shuffles | 0.930 | 0.940 |

**Composition does most of the work.** GC content alone separates tRNAs from random windows with AUROC 0.96 (the genome is AT-rich, 34% GC in windows, and tRNAs 52% GC, so G–C-rich sequence folds more stably regardless of structure), and the raw MFE per nucleotide (0.994) is only slightly better. The **z-score** against composition-matched shuffles (which removes the composition effect) is the evidence of *structure beyond composition*: AUROC 0.93–0.94. This is exactly the lesson of Chapter 29's PWM background: **a score's meaning depends on the null model**, and a detector that is "99% accurate" may be a GC detector.

```python
--8<-- "code/ch33_rna_structure.py"
```

```text
28 tRNA genes with a single exon and length 60-95 nt (mean length 75.1, mean GC 0.522)

== 1. MFE structures of the 28 tRNAs ==
acceptor stem recovered (>=6 of the 7 terminal base pairs): 25/28 = 0.89
full cloverleaf topology (acceptor stem enclosing exactly three hairpins): 9/28 = 0.32
mean MFE -25.0 kcal/mol (-0.332 per nucleotide)

== 2. Detecting tRNAs among 1500 75-nt windows outside tRNA/rRNA genes (coding and non-coding) (GC of windows 0.342; tRNAs 0.522); GC-matched subset: 435 windows ==
detector                               AUROC vs all windows    AUROC vs GC-matched windows
GC content                                 0.960                    0.864
MFE per nucleotide (more negative)         0.994                    0.989
MFE z-score vs Markov shuffles             0.930                    0.940
```

**Deep learning for RNA structure.** Learned secondary-structure predictors trained on known structures (SPOT-RNA, MXfold2, and others) achieve high scores on test sets drawn from the training families, but **do not generalize to new RNA families**: when test families are held out, their performance falls below that of thermodynamic methods (Szikszai et al., 2022) [[S]]. This is the RNA analogue of the random-versus-family split of Chapters 24 and 43. For **3D structure**, the CASP15 assessment (the first to include RNA) found that the top-ranked groups did not use deep learning and that deep-learning methods were significantly worse than the leading classical and expert pipelines (Das et al., 2023) [[S]]; newer methods (trRosettaRNA, RhoFold+, AlphaFold 3's RNA modeling) improve on specific benchmarks, and RNA structure remains limited by **data scarcity** (RNA-containing structures are a small fraction of the PDB) and by **conformational heterogeneity**. This is a domain in which *the classical baseline is still the strongest and a generalization-aware benchmark is the main contribution*.

---

## 33.6 RNA language models

Pretrained RNA language models (RNA-FM; RiNALMo, a 650-million-parameter model trained on tens of millions of non-coding RNA sequences) use masked-token pretraining as in Chapter 34 and are fine-tuned for structure, family classification, and function. RiNALMo reports that a large pretrained model can generalize to secondary-structure prediction on *unseen families* better than prior learned models [[S]]; whether it beats thermodynamic and comparative-sequence-analysis (covariation) baselines on families with no homologs is an active question [[H]]. Evaluate such claims by (i) family-level splits at a stated identity threshold (e.g., Rfam clans) with sequence *and* structure similarity checked, (ii) the thermodynamic baseline and a covariation baseline (Potts models; Chapter 29), and (iii) the usual ceiling (inter-annotator agreement of structure annotations is imperfect, and experimental probing data are noisy).

---

## 33.7 Worked research examples

!!! example "Worked Research Example 33.1: A model flags a deep intronic variant as splice-altering"
    **Situation.** A patient with a rare neurological disorder has a heterozygous variant 1.8 kb inside an intron of a candidate gene. SpliceAI-type scoring gives $\Delta=0.62$ for a cryptic donor gain; no other candidate variants were found.

    **Question.** How strongly should this change the diagnosis, and what should be done?

    **Reasoning.**

    1. *Prior and base rate.* In a gene with a plausible phenotype, the prior that a given rare deep-intronic variant is pathogenic is low. With a score threshold whose positive predictive value among all rare intronic variants is, say, 10–30% [[P]] (depends strongly on the threshold and the dataset), a score of 0.62 raises the probability to the upper end of that range but does not establish it (Chapter 43: prevalence).
    2. *Evidence beyond the score.* (a) Does the predicted cryptic exon preserve the reading frame or introduce a premature termination codon (nonsense-mediated decay)? (b) Is the gene expressed in the relevant tissue (so that RNA from blood or fibroblasts, or patient-derived cells, can be tested)? (c) Conservation of the region and the variant's population frequency (a variant seen in population controls is unlikely to be a rare-disease cause).
    3. *Decisive experiment.* RNA-seq from the patient and a control (aberrant splicing with allele-specific reads), or a **minigene assay** with and without the variant, or targeted RT-PCR. A positive result moves the variant to "likely pathogenic" under clinical guidelines (functional evidence), a negative one lowers it.
    4. *What the model should be asked next.* *Quantitative* inclusion changes (PSI) and tissue-specific effects (Pangolin-type predictions), because even correct cryptic-site creation can be minor if the canonical site dominates.

    **Expert analysis.** The model is a **prioritization and hypothesis-generating tool** whose output is converted into a clinical decision by an *assay*. In terms of the Claim Ladder (Chapter 1), the model gives C2 (held-out prediction), the assay supplies C4 for this variant.

!!! example "Worked Research Example 33.2: A language model of mRNAs claims to design high-expression 5′ UTRs"
    **Situation.** A group fine-tunes an RNA language model on a published MPRA of 5′ UTR translation in one cell line and reports that sequences it generates have predicted ribosome loads above the library maximum, claiming "design of superior UTRs for mRNA therapeutics".

    **Question.** What should be tested before accepting the claim?

    **Reasoning.**

    1. *Predicted versus measured.* A model optimized against its own predictor will find *adversarial* sequences outside the training distribution where the predictor is wrong (model exploitation; Chapter 36). The proper test is **experimental**: synthesize the designs and measure.
    2. *Distribution shift* (Chapter 45): the MPRA used one reporter CDS and one cell line; the therapeutic mRNA has a different CDS and is delivered to other cells. UTR effects are partly *context dependent* (interaction with the CDS start region, cell-specific factors).
    3. *Controls.* Designs from a simple baseline (the best library sequences; random mutations of the best; an optimizer on a PWM/CNN model) at equal budget; negative controls; the measured ceiling (replicate correlation of the assay).
    4. *Generalization.* Test the designs with a *different* CDS and in at least one other cell type, with the readout of interest (protein output, not only ribosome load: stability, immunogenicity, innate-immune sensing also matter).
    5. *Interpretation.* If designs outperform baselines across contexts, that is C4 for the UTR design problem; if only in the original reporter, it is a statement about the reporter.

    **Expert analysis.** Design claims need *prospective, cross-context, baseline-controlled* experiments. The key variable is the *gap between the model's predictions and the experiments for out-of-distribution designs*, the quantity that the information-gain framework of Chapter 56 would prioritize measuring.

---

## 33.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: evaluating an RNA model"
    1. **Which layer?** Splicing, 3′ end, translation, stability, structure, localization: each has different labels, ceilings, and shortcuts.
    2. **Labels and context.** Annotation-derived (clean, but circular), RNA-seq-derived (isoform quantification uncertainty; Chapter 27), MPRA-derived (one reporter, one cell type), probing-derived (noisy). State the context of the measurement.
    3. **Baselines.** MaxEnt/PWM splice scoring (Chapter 29), codon-usage models, thermodynamic folding with a shuffle null, covariation (DCA) for structure.
    4. **Splits.** Held-out genes with paralog removal (splicing); held-out families by Rfam clan (structure); held-out reporters/cell types (translation).
    5. **Variant effect.** Within-gene, ref-versus-alt scoring; calibrate thresholds at deployment prevalence (Chapter 43); include quantitative effect where available.
    6. **Nulls.** Composition-matched shuffles for structure detection (our tRNA example: GC alone gave AUROC 0.96).
    7. **Experiments.** Which minimal assay (minigene, RNA-seq, MPRA) tests the claim, and does it avoid the model's own training distribution?

    **What it teaches.** The RNA layer has a *clean* regime (splicing, with annotated labels and large effects) and a *hard* regime (structure, with scarce data and family shift); the strongest results come where labels are clean and effects are large, the weakest where they are not.

    **An open question to carry forward.** Splicing models predict *where spliceosomes assemble*; what determines *which* isoform a cell makes is the cell's splicing-factor state. Could a model that takes as input both sequence and a low-dimensional representation of the cell's RBP expression (from single-cell data) predict isoform choice for unseen cell types, and what data (long-read single-cell isoform atlases) and identifiability conditions (varying RBP levels independently of the cell type) would be needed? Frame it with Chapter 31's cell-type-as-output limitation and Chapter 44's identifiability.

---

## 33.9 Connections

- **Backward:** PWMs and MaxEnt models, HMMs (Chapter 29); convolutions and dilations (Chapter 10); thermodynamic models and Boltzmann weights (Chapter 22); sequence-to-function and variant effect (Chapter 31); genomic LMs (Chapter 32); tokenization and the codon frame (Chapter 28); leakage and family splits (Chapter 43).
- **Forward:** protein language models and codon-aware modeling (Chapter 34); structure prediction, including RNA (Chapter 35); design with generative models (Chapter 36); genotype-to-phenotype with splicing (Chapter 41); open problems for genomes (Chapter 50).

!!! takeaways "Key takeaways"
    1. RNA processing is a pipeline of sequence-encoded steps (splicing, 3′ end, export, translation, decay); each is a prediction problem with different labels, context dependence, and ceilings.
    2. **Local splice motifs have about 8 bits** (a few false sites per kb on one strand), so discrimination requires **context** over hundreds to thousands of nucleotides: the reason a 10-kb-context deep network succeeded.
    3. **SpliceAI** (32 dilated residual layers, 10-kb context) predicts splice sites from sequence, extends variant interpretation to deep intronic variants, and explains about 9–11% of pathogenic de novo variants in neurodevelopmental cohorts (as reported); thresholds need prevalence-aware calibration and assays for confirmation.
    4. **Splicing works better than regulation** because labels are clean and effects are large; tissue specificity and quantitative inclusion remain open.
    5. **MPRA-trained models** (Optimus 5-Prime; Saluki; UTR-LM) identify effects because the libraries decorrelate features, but are context-specific; **design** needs prospective, cross-context validation.
    6. **RNA folding** is a Zuker/McCaskill dynamic program; on 28 real tRNAs the MFE structure recovered the acceptor stem in 25 (0.89) and the full cloverleaf in only 9 (0.32): the MFE structure is not the native structure.
    7. **Detection of structured RNA is dominated by composition**: GC alone gave AUROC 0.96 and MFE per nucleotide 0.994; the shuffle z-score (0.93–0.94) isolates structure beyond composition.
    8. **Learned RNA structure models do not generalize across families** (Szikszai et al., 2022), and in CASP15 classical/expert pipelines beat deep learning for RNA 3D structure; RNA language models claim improved family-level generalization, to be tested against thermodynamic and covariation baselines.

---

## Further reading

- Jaganathan, K. et al. (2019). Predicting splicing from primary sequence with deep learning. *Cell* 176, 535–548. Zeng, T. & Li, Y. I. (2022). Predicting RNA splicing from DNA sequence using Pangolin. *Genome Biol.* 23, 103. Yeo, G. & Burge, C. B. (2004). Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals. *J. Comput. Biol.* 11, 377–394. Wang, E. T. et al. (2008). Alternative isoform regulation in human tissue transcriptomes. *Nature* 456, 470–476. Pan, Q., Shai, O., Lee, L. J., Frey, B. J. & Blencowe, B. J. (2008). Deep surveying of alternative splicing complexity in the human transcriptome by high-throughput sequencing. *Nat. Genet.* 40, 1413–1415.
- Sample, P. J. et al. (2019). Human 5′ UTR design and variant effect prediction from a massively parallel translation assay. *Nat. Biotechnol.* 37, 803–809. Agarwal, V. & Kelley, D. R. (2022). The genetic and biochemical determinants of mRNA degradation rates in mammals. *Genome Biol.* 23, 245. Chu, Y. et al. (2024). A 5′ UTR language model for decoding untranslated regions of mRNA and function predictions. *Nat. Mach. Intell.* 6, 449–460. Zhang, H. et al. (2023). Algorithm for optimized mRNA design improves stability and immunogenicity. *Nature* 621, 396–403.
- Zuker, M. & Stiegler, P. (1981). Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information. *Nucleic Acids Res.* 9, 133–148. Nussinov, R. et al. (1978). Algorithms for loop matchings. *SIAM J. Appl. Math.* 35, 68–82. McCaskill, J. S. (1990). The equilibrium partition function and base pair binding probabilities for RNA secondary structure. *Biopolymers* 29, 1105–1119. Turner, D. H. & Mathews, D. H. (2010). NNDB: the nearest neighbor parameter database for predicting stability of nucleic acid secondary structure. *Nucleic Acids Res.* 38, D280–D282. Lorenz, R. et al. (2011). ViennaRNA Package 2.0. *Algorithms Mol. Biol.* 6, 26.
- Szikszai, M., Wise, M., Datta, A., Ward, M. & Mathews, D. H. (2022). Deep learning models for RNA secondary structure prediction (probably) do not generalize across families. *Bioinformatics* 38, 3892–3899. Das, R. et al. (2023). Assessment of three-dimensional RNA structure prediction in CASP15. *Proteins* 91, 1747–1770. Shen, T. et al. (2024). Accurate RNA 3D structure prediction using a language model-based deep learning approach (RhoFold+). *Nat. Methods* 21, 2287–2298. Penić, R. J. et al. (2025). RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks. *Nat. Commun.*
