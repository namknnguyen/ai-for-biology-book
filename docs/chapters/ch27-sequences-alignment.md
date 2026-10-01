# Chapter 27. Sequences, Alignment, and the Classical Genomics Pipeline

!!! abstract "Chapter at a glance"
    **Motivation.** Almost every dataset used to train a biological model, whether peaks, variants, expression levels, or multiple sequence alignments, is *the output of a pipeline* that aligns reads or sequences to a reference. The pipeline's statistics, its failure regions, and its assumptions determine what the labels mean. This chapter teaches the classical machinery (log-odds scoring, significance theory, indexing, mapping, variant calling) at the level needed to reason about *label noise and blind spots*, and tests its main claims in simulation.
    **Prerequisites.** Chapters 4, 5, 6 (dynamic programming and HMMs), 8 (EM), 20.
    **You will be able to:** (1) derive substitution scores as log-likelihood ratios and compute the information per aligned position; (2) state and verify Karlin–Altschul statistics, convert scores to E-values, and say how long an alignment must be to be significant; (3) derive FM-index backward search and explain why mapping is $O(m)$ independent of genome size; (4) explain mappability, repeats, and why short reads cannot resolve recent repeats; (5) derive genotype likelihoods, and explain which errors depth fixes and which it does not; (6) formulate transcript quantification as a mixture model solved by EM; (7) say what the classical pipeline's blind spots imply for the training data of sequence models.

---

## 27.1 What the data are: reads, references, and alignment

A sequencing experiment returns **reads**: strings over $\{A,C,G,T\}$ with per-base quality scores (Phred $Q=-10\log_{10}P(\text{error})$). The main platforms differ in what the reads are good for:

* **Short-read (Illumina)**: 50–300 bases, per-base accuracy above 99.9% for most bases, cheap per base; blind to long repeats and structural variants.
* **Long-read**: PacBio HiFi reads of 10–25 kb with accuracy above 99.9% (Wenger et al., 2019); Oxford Nanopore reads from kilobases to over a megabase with accuracy that has improved with each chemistry and basecaller [[S]]. They span repeats and phase variants.

A **reference genome** is the coordinate system to which reads are aligned. The long-standing human reference (GRCh38) was incomplete in repeat-rich regions; the **T2T-CHM13** assembly (Nurk et al., 2022) was the first complete human genome sequence, adding about 200 Mb (about 8%) of new sequence, including centromeric satellite arrays and segmental duplications [[E]]. The **human pangenome reference** (Liao et al., 2023) represents the diversity of 47 individuals as a graph rather than a single linear sequence [[E]]. **A single linear reference induces reference bias**: reads carrying non-reference alleles align slightly worse and are more often misplaced or discarded.

The **classical pipeline** is: reads → quality control → alignment to the reference → (a) signal tracks and peaks (ChIP-seq, ATAC-seq), (b) variant calls, (c) expression quantification. Every downstream "label" in Parts VI and VII comes from one of these. Each step is statistical, and each has a *region of the genome where it fails silently*.

---

## 27.2 Scoring alignments: scores are log-likelihood ratios

An alignment aligns two sequences by matching residues, allowing substitutions and gaps. For the best alignment under an additive score, dynamic programming (Chapter 6) solves it in $O(mn)$ time: for local alignment (Smith–Waterman),

$$
H_{ij}=\max\{0,\ H_{i-1,j-1}+s(a_i,b_j),\ H_{i-1,j}-g,\ H_{i,j-1}-g\},
$$

and the alignment score is $\max_{ij}H_{ij}$ (affine gap penalties add a state, giving the three-matrix Gotoh algorithm). What are the numbers $s(a,b)$? They are not arbitrary:

!!! math "Substitution scores as log-odds"
    Let $p_a$ be the background frequency of residue $a$, and $q_{ab}$ the frequency with which $a$ and $b$ are aligned in *truly homologous* sequences. A homology hypothesis is tested by the likelihood ratio $q_{ab}/(p_ap_b)$ of a pair; the log of the product over aligned positions is the sum of

    $$
    s(a,b)=\frac1\lambda\ln\frac{q_{ab}}{p_ap_b},\qquad\text{equivalently}\qquad q_{ab}=p_ap_b\,e^{\lambda s(a,b)}.
    $$

    Because the $q_{ab}$ sum to one, $\lambda$ is the unique positive solution of $\sum_{a,b}p_ap_b\,e^{\lambda s(a,b)}=1$ (*Karlin & Altschul, 1990*). The expected score per aligned pair is *negative* under the background ($-\tfrac1\lambda\mathrm{KL}(pp\Vert q)$) and positive under homology: $\mathbb E_q[s]=H/\lambda$, with $H=\mathrm{KL}(q\Vert pp)$ the **relative entropy**, the information in nats that one aligned pair carries for homology versus chance (Chapter 5).

The widely used **BLOSUM** matrices (Henikoff & Henikoff, 1992) and earlier PAM matrices (Dayhoff et al., 1978) were estimated by exactly this reasoning from blocks of aligned protein families (BLOSUM62: clustering sequences at 62% identity). Using BLOSUM62 with Robinson–Robinson background frequencies we find (`code/ch27_alignment.py`): expected score per pair under the background $-0.938$; $\lambda=0.3172$ (the value in NCBI's BLAST statistics for BLOSUM62 is about 0.318); relative entropy $H=0.386$ nats, i.e. **0.56 bits per aligned pair**. *Each aligned residue pair carries about half a bit of evidence for homology*: a significant alignment needs many aligned positions, which is why short motifs cannot be detected by sequence alignment alone, and why a *profile* (a position-specific score matrix or HMM, $q$ estimated per column of an alignment) detects remote homologs that single-sequence BLOSUM-based searches miss.

---

## 27.3 Significance: Karlin–Altschul statistics and E-values

Given a scoring system, how high must a score be to be unlikely among unrelated sequences? For *ungapped* local alignment of random sequences of lengths $m,n$, local alignments are rare excursions of a random walk with negative drift, and the number of distinct alignments with score $\ge S$ is approximately Poisson with mean

$$
\boxed{\ E=K\,m\,n\,e^{-\lambda S}\ }
$$

(Karlin & Altschul, 1990; the **E-value**). Two consequences follow. **(i)** The maximum score $S_\max$ is **Gumbel-distributed**: $P(S_\max\ge S)=1-\exp(-Kmn\,e^{-\lambda S})$ with location $\ln(Kmn)/\lambda$ and scale $1/\lambda$, so the standard deviation is $\pi/(\lambda\sqrt6)$. **(ii)** A *bit score* $S'=(\lambda S-\ln K)/\ln2$ puts all scoring systems on a common scale, with $E=mn\,2^{-S'}$; searching a database of $N$ sequences multiplies $E$ by $N$ (a Bonferroni-type correction for the number of comparisons). The length required for significance follows from setting $E\approx1$: the alignment must contain roughly $\ln(Kmn)/H$ positions: for a 300-residue query against a $10^8$-residue database, $\ln(10^{10})/0.386\approx60$ well-conserved positions.

**Experiment.** We test the theory by brute force: $4{,}000$ random sequence pairs (length 250, background frequencies of amino acids), maximum ungapped local score via the diagonal DP.

| Check | Result |
|---|---|
| Standard deviation of $S_\max$, empirical | 4.10 |
| Gumbel theory $\pi/(\lambda\sqrt6)$ with the analytic $\lambda$ | **4.04** |
| Effective $K$ (fitted from the mean; includes finite-length edge effects) | 0.103 (the asymptotic value for BLOSUM62 in NCBI's tables is about 0.13) |

The standard deviation check involves no fitted parameters: it verifies $\lambda$. To test the calibration of $E$ we then ran a fresh set of $20$ queries against a database of $5{,}000$ random sequences ($100{,}000$ independent comparisons, none used to fit $K$):

| Score threshold $S$ | Predicted E-value per query | Observed false hits per query | 95% Poisson interval |
|---|---|---|---|
| 36 | 352.0 | 361.1 | (352.8, 369.4) |
| 40 | 99.0 | 100.8 | (96.4, 105.2) |
| 44 | 27.8 | 28.2 | (25.9, 30.5) |
| 48 | 7.82 | 7.55 | (6.35, 8.75) |
| 52 | 2.20 | 2.30 | (1.64, 2.97) |

*The E-value is calibrated across two orders of magnitude.* (Predictions are within 3% of observations at every threshold; the prediction of 352 sits just below its 95% interval, a difference consistent with the uncertainty in $K$ (fitted from only 4,000 samples) and the Gumbel approximation for integer scores.) The theory does **not** extend analytically to *gapped* alignments, where $\lambda$ and $K$ are estimated empirically by simulation (Altschul et al., 1997), nor to *real* databases, whose sequences are not random: they contain low-complexity regions and homologous families, which break the independence assumptions. **The E-value is the probability model for the null hypothesis "unrelated random sequences"**; when real unrelated sequences are non-random (compositional bias, repeats), the reported E-values are over-optimistic, which is why BLAST applies composition-based statistics and masks low-complexity regions.

!!! rhyme "Structural rhyme: the E-value ↔ multiple testing ↔ the noise ceiling"
    An E-value is a family-wise error rate for a database search, the same construction as the genome-wide significance threshold of Chapter 26 ($0.05/10^6$ tests). It also tells you the *evidence budget*: with half a bit per position, a search over a large database *requires* tens of conserved positions. Protein language models (Chapter 34) and structure-based search (Foldseek; van Kempen et al., 2024) detect more distant relationships by using richer representations whose per-position information about homology is higher than a substitution matrix's 0.56 bits [[S]]; but the *statistical question* (what is the null, and what is the multiplicity) is unchanged and is often ignored in benchmark comparisons.

---

## 27.4 Indexing: how to search $3\times10^9$ bases in $O(m)$

Aligning every read to the genome by dynamic programming is impossible ($10^9$ reads $\times$ $3\times10^9$ bases). Two ideas make it feasible.

**Seed-and-extend.** BLAST (Altschul et al., 1990) finds short exact or near-exact **word hits** (seeds) using an index, and only then performs DP around those seeds. Statistics: if two sequences share a true alignment with identity $\pi$ over length $L$, the probability of at least one exact seed of length $k$ is high when $\pi^k$ is not small; a smaller $k$ gives more sensitivity and more spurious hits. Modern long-read mappers use *minimizers* (minimap2; Li, 2018) to subsample seeds.

**The Burrows–Wheeler transform and FM-index.** For a text $T$ ending with a unique sentinel \$, sort all suffixes (the **suffix array** $SA$) and let the **BWT** be the character preceding each suffix: $\mathrm{BWT}[i]=T[SA[i]-1]$. The suffixes starting with a pattern $P$ form a contiguous interval $[lo,hi)$ of $SA$. The **LF-mapping** (the $i$-th occurrence of character $c$ in the last column corresponds to the $i$-th occurrence of $c$ in the first column) gives **backward search**: process $P$ right to left; if the current interval for the suffix $P[j+1..m]$ is $[lo,hi)$, then the interval for $cP[j+1..m]$ is

$$
lo'=C[c]+\mathrm{Occ}(c,lo),\qquad hi'=C[c]+\mathrm{Occ}(c,hi),
$$

where $C[c]$ is the number of text symbols smaller than $c$ and $\mathrm{Occ}(c,i)$ the number of $c$ in $\mathrm{BWT}[0..i)$. The pattern occurs $hi-lo$ times, found in **$m$ rank operations, independent of the genome size** (Ferragina & Manzini, 2000; Burrows & Wheeler, 1994). With rank/select structures and a sampled suffix array, the index of the human genome fits in a few gigabytes (BWA, Bowtie; Li & Durbin, 2009; Langmead et al., 2009). Allowing mismatches means branching in the backward search (cost grows with the number of allowed errors) or using seeds plus extension.

**Experiment.** We built the suffix array (by prefix doubling), the BWT, and the Occ table for a random 200,000-base genome and compared backward-search counts with brute-force counting for 300 random patterns of length 6–13: **0 mismatches**, as it must be (the index is an exact data structure). This is mundane but important: *classical algorithms are exact where learned models are approximate*; the speed-up comes from a data-structure insight, not from approximation.

---

## 27.5 Mapping: repeats, mappability, and blind spots

Even with an exact index, a read from a repeated sequence matches many places. The mapper reports a best location together with a **mapping quality** (MAPQ$=-10\log_{10}P(\text{location is wrong})$); reads with ambiguous best locations get low MAPQ and are typically discarded or split fractionally. About 45% of the human genome is derived from transposable elements (Lander et al., 2001), and recent copies are very similar to each other.

**Experiment.** A 300-kb random genome in which a 300-bp element appears in 300 copies, each diverged from the consensus by a random 0.5–10% (a spectrum of repeat ages); reads sampled from unique regions or fully inside repeats, with 1% sequencing error, mapped by 20-mer seed voting. A read counts as *unambiguous* if its best location has at least twice the seed support of the runner-up.

| Read length | Region | Mapped correctly & unambiguously | Ambiguous | Wrong or unmapped |
|---|---|---|---|---|
| 50 | unique | 0.99 | 0.00 | 0.01 |
| 50 | repeat | **0.33** | 0.65 | 0.02 |
| 100 | unique | 1.00 | 0.00 | 0.00 |
| 100 | repeat | **0.43** | 0.57 | 0.00 |
| 250 | unique | 1.00 | 0.00 | 0.00 |
| 250 | repeat | **0.56** | 0.44 | 0.00 |

Reads from unique sequence map essentially perfectly at any length; reads from repeats are *ambiguous* for 44–65% of reads, improving only slowly with length, because the young copies are nearly identical over the length of a read. (Long reads of several kilobases that span a repeat and its unique flank resolve these.) Three consequences for machine learning.

1. **Missing is not zero.** A region with no confidently mapped reads has *undefined*, not zero, signal. Models trained on ChIP-seq, ATAC-seq, or RNA-seq tracks (Chapters 31–32) typically mask low-mappability regions or treat them as zeros; either choice biases what the model learns, and it means *transposable-element–derived regulatory sequences are systematically under-represented in training and evaluation*, even though they carry many regulatory elements.
2. **Reference bias and ancestry.** Reads from non-reference alleles align worse, which distorts allele-specific measurements and makes variant effect benchmarks derived from reference-aligned data ancestry-dependent (Chapters 41, 45).
3. **The reference defines the model's world.** Genomic language models trained on a linear reference (Chapter 32) inherit its gaps, errors, and unresolved regions; T2T-CHM13 added hundreds of megabases that earlier training sets did not contain.

---

## 27.6 Variant calling: genotype likelihoods and what depth cannot fix

At a site covered by $d$ reads, $k$ of which show the alternative allele, a **genotype likelihood** under independent errors (per-base error rate $e$) is binomial. For the three diploid genotypes with alternative-allele fraction $\theta_g\in\{e,\tfrac12,1-e\}$,

$$
P(k\mid g)=\binom dk\theta_g^k(1-\theta_g)^{d-k},\qquad
P(g\mid\text{reads})\propto P(k\mid g)\,\pi_g,
$$

where $\pi_g$ is the Hardy–Weinberg prior with alternative allele frequency $p$: $\pi=((1-p)^2,\ 2p(1-p),\ p^2)$ (Li, 2011). The genotype with the highest posterior is called. Per-read base quality $Q$ replaces $e$ read by read, and mapping quality discounts reads whose placement is uncertain. This is the model behind the SAMtools/GATK family; **DeepVariant** (Poplin et al., 2018) replaces the hand-built model by a convolutional network on images of the read pileup around the candidate, learning the error modes from truth sets (it won the 2016 PrecisionFDA Truth Challenge for SNP accuracy), an early success of deep learning in the *classical* genomics pipeline.

**Experiment.** Simulated sites (alternative allele frequency 0.15; Poisson depth; per-base error 0.01) called with the model above, which assumes independent errors at the nominal rate. Rows vary the *truth*: independent errors; 20% reference bias (alt-carrying reads are 20% less likely to be mapped); and site-specific error rates (log-normal, log-SD 1.0, as when particular sequence contexts or paralogous mismapping produce *correlated* errors).

| Depth | Het. sensitivity (independent / ref. bias / site-specific) | Hom-ref sites miscalled as variant, per $10^6$ sites (independent / ref. bias / site-specific) |
|---|---|---|
| 2× | 0.627 / 0.586 / 0.627 | 17,869 / 17,995 / 29,076 |
| 5× | 0.873 / 0.849 / 0.877 | 22,277 / 21,626 / 37,804 |
| 10× | 0.963 / 0.946 / 0.966 | 5,720 / 4,672 / 16,369 |
| 30× | 1.000 / 0.999 / 1.000 | **69 / 23 / 7,770** |

Four lessons. **(i) Sensitivity saturates with depth** (0.63 at 2×, 0.96 at 10×, 1.0 at 30× for heterozygous sites), because a heterozygote needs both alleles sampled. **(ii) False positives from independent errors fall steeply with depth** (17,869 → 69 per million at 30×): a single erroneous alt read is no longer enough. **(iii) Depth does not fix correlated errors**: with site-specific error rates, the false-positive rate at 30× is 7,770 per million, 110 times the independent-error value. Real correlated errors arise from sequence context, strand-specific artifacts, and *paralogous reads that map to the wrong copy*, which all produce consistent alternative alleles. A caller that assumes independence is miscalibrated exactly where depth is highest; this is why callers filter by strand bias, mapping quality, and population-level error models, and why learned callers that see the *pattern* of the pileup help. **(iv) Reference bias** lowers heterozygote sensitivity slightly (0.946 vs 0.963 at 10×), shifts the allele balance of true heterozygotes (from 0.5 to 0.5·0.8/(0.5·0.8+0.5) ≈ 0.44), and is far more consequential for allele-specific measurements (allele-specific expression or binding) than for calling. At low depth, the model's *prior* dominates: with 2× coverage a single alt read gives a posterior favoring a heterozygous call, which is correct when errors are rare and the prior reflects variant density, but inflates false positives when it does not.

**Imputation.** Cheap arrays genotype $\sim10^6$ variants; the rest are filled in using a haplotype-copying hidden Markov model (Li & Stephens, 2003; Chapter 6's forward algorithm over a reference panel of haplotypes). Imputation accuracy depends on the match between the panel and the target ancestry, again a source of ancestry-dependent label noise (Chapter 26).

---

## 27.7 Quantification as a mixture model: EM on equivalence classes

Gene and transcript expression is estimated by *counting reads assigned to transcripts*, but a read may be compatible with several isoforms. Model it as a mixture. Let transcript $t$ have effective length $\ell_t$ and relative abundance $\theta_t$ ($\sum_t\theta_t=1$); a read $r$ is generated by picking a transcript with probability proportional to $\theta_t\ell_t$ (longer transcripts yield more fragments), then a position uniformly, so $P(r\mid t)=\mathbb 1[r\sim t]/\ell_t$. The likelihood is

$$
L(\theta)=\prod_r\sum_t\alpha_t\,\frac{\mathbb 1[r\sim t]}{\ell_t},\qquad\alpha_t=\frac{\theta_t\ell_t}{\sum_u\theta_u\ell_u}.
$$

EM (Chapter 8) alternates the **E-step** $z_{rt}=\alpha_t\mathbb 1[r\sim t]/\sum_u\alpha_u\mathbb 1[r\sim u]$ (fractional assignment of each read among compatible transcripts) and the **M-step** $\alpha_t\leftarrow\frac1R\sum_rz_{rt}$, then $\theta_t\propto\alpha_t/\ell_t$. Reads with the same compatibility set (an **equivalence class**) are processed together, so the algorithm runs on counts per class: this is the core of the pseudo-alignment methods kallisto and Salmon (Bray et al., 2016; Patro et al., 2017) [[E]]. Two lessons: *expression levels from RNA-seq are model-based estimates*, with higher uncertainty for isoforms that share most of their sequence; and the same mixture-plus-EM formulation reappears in spatial deconvolution (Chapter 25) and in many probabilistic single-cell models (Chapter 30).

---

## 27.8 Why the pipeline matters for AI

1. **Labels are pipeline outputs.** Peak calls, variant calls, and expression values carry the mappability, reference bias, and calling errors above. Models are scored against these labels, so the *noise and blind spots of the pipeline enter evaluation*: a model cannot be penalized for being right where the pipeline was wrong, nor rewarded for predicting what the pipeline cannot see.
2. **Alignments are the input to protein models.** Multiple sequence alignments built by homology search (Chapters 23 and 35) feed structure predictors and coevolution models; their *depth* $N_\text{eff}$ and quality limit accuracy, and E-value thresholds determine which sequences are included.
3. **Exactness.** Classical algorithms (FM-index, DP) are exact and *interpretable*; where they fail (repeats, divergent homologs), the failure is predictable. Learned replacements should be compared on those failure regions specifically, not only on aggregate accuracy.
4. **The reference as a representation.** Whether the human genome is best represented as a string, a graph (pangenome), or a set of haplotypes is a representation question (Chapter 28), and the choice changes what a foundation model sees.

---

## 27.9 The experiments, verbatim

```python
--8<-- "code/ch27_alignment.py"
```

```text
== 1. Karlin-Altschul theory for ungapped local alignment, BLOSUM62 ==
expected score per aligned pair under the background model = -0.938 (must be negative); lambda = 0.3172; relative entropy H = 0.386 nats = 0.56 bits per aligned pair
4000 random pairs of length 250: mean max score 29.45; analytic lambda = 0.3172; effective K fitted from the mean = 0.103
standard deviation of the max score: empirical 4.10; Gumbel theory pi/(lambda sqrt 6) = 4.04  (this checks lambda without fitting it)

== 2. E-value calibration: 20 queries x a database of 5,000 random sequences (all length 250), 100,000 comparisons ==
score threshold S   predicted E-value per query (Ndb K m n e^{-lambda S})   observed false hits per query   (95% Poisson interval)
        36                 351.984                                     361.100                      (352.772, 369.428)
        40                  98.962                                     100.800                      (96.400, 105.200)
        44                  27.824                                      28.200                      (25.873, 30.527)
        48                   7.823                                       7.550                      (6.346, 8.754)
        52                   2.199                                       2.300                      (1.635, 2.965)

== 3. FM-index backward search ==
genome of 200,000 bases; suffix array + BWT + Occ table; 300 random patterns of 6-13 bases: mismatches between FM-index counts and brute force = 0
a pattern of length m is located in O(m) rank operations independent of genome size; memory here 4.0 MB for the full Occ table (production indexes sample it)

== 4. Repeats make short reads ambiguous (300 kb genome with a 300-bp element in 300 copies diverged by 0.5-10%) ==
read length   region     reads   mapped correctly & unambiguous (ratio>=2)   ambiguous   wrong or unmapped
      50      unique      400         0.99                                  0.00        0.01
      50      repeat      400         0.33                                  0.65        0.02
     100      unique      400         1.00                                  0.00        0.00
     100      repeat      400         0.43                                  0.57        0.00
     250      unique      400         1.00                                  0.00        0.00
     250      repeat      400         0.56                                  0.44        0.00

== 5. Genotype calling from a read pileup ==
depth   het sensitivity   hom-ref miscalled as variant (per Mb)   scenario
    2        0.627               17869                  independent errors, e = 0.01
    2        0.586               17995                  reference bias 20%
    2        0.627               29076                  site-specific error rates (log-SD 1.0)
    5        0.873               22277                  independent errors, e = 0.01
    5        0.849               21626                  reference bias 20%
    5        0.877               37804                  site-specific error rates (log-SD 1.0)
   10        0.963                5720                  independent errors, e = 0.01
   10        0.946                4672                  reference bias 20%
   10        0.966               16369                  site-specific error rates (log-SD 1.0)
   30        1.000                  69                  independent errors, e = 0.01
   30        0.999                  23                  reference bias 20%
   30        1.000                7770                  site-specific error rates (log-SD 1.0)
```

---

## 27.10 Worked research examples

!!! example "Worked Research Example 27.1: A language model \"finds remote homologs better than BLAST\""
    **Situation.** A paper reports that embeddings from a protein language model, compared by cosine similarity, retrieve more same-superfamily proteins than BLAST on a benchmark, measured as the fraction of true homologs ranked above the first false positive.

    **Question.** What must be checked before accepting this as a statement about remote-homology detection?

    **Reasoning.**

    1. *What is the null model?* BLAST reports E-values with a known statistical interpretation (§27.3). A cosine similarity has no null distribution unless one is built (e.g., from shuffled or unrelated sequences); comparing "rank above the first false positive" removes the question of *calibration* but also of *thresholds*: one cannot choose a cutoff for a new query.
    2. *What is the correct comparison?* Both should be compared at the *same false-positive rate per query* on a database with the same size and composition, using coverage versus errors per query (the standard for homology search benchmarks), and compared with the *right* classical baseline: profile-based methods (jackhmmer, HHblits), not single-sequence BLAST, because profile methods are the state of the art for remote homology.
    3. *What is a true positive?* "Same superfamily" in SCOP or CATH is itself defined partly using sequence methods and structure; remote pairs defined by structure may share no detectable sequence signal, and a language model trained on large sequence databases may have seen the family and memorized family-level features. Hold out families, not sequences (Chapter 43).
    4. *Cost and failure modes.* Embedding search is fast and not interpretable (no alignment); classical search gives an alignment, from which one can build an MSA. The failure regions differ: language-model similarity is expected to degrade for low-complexity or compositionally biased sequences (where BLAST masks) and for orphan proteins.
    5. *The decisive experiment.* Per-query coverage at 1 error per query across identity bins (below 20%, 20–30%, 30–40%) with family-level splits and profile baselines; an *ablation* that shuffles residues while preserving composition.

    **Expert analysis.** The classical machinery gives the benchmark a language: null models, multiplicity, calibrated thresholds. A learned method does not need to adopt it, but its claims have to be *translatable* into it. The Expert Chain link L9 (candidate solution) is a similarity function; L10 (experiments) has to hold the *error rate* fixed.

!!! example "Worked Research Example 27.2: A regulatory sequence model is \"noisy\" in repeat-rich loci"
    **Situation.** A sequence-to-function model (Chapter 31) is trained on tracks of ChIP-seq and RNA-seq coverage. Its correlation with the observed signal is low in a set of transposable-element–rich loci; the authors attribute this to "noisy biology" and exclude the loci.

    **Question.** How would you decide whether the model or the labels are at fault?

    **Reasoning.**

    1. *What do the labels mean there?* With 100-bp reads, 57% of reads inside our simulated repeats were ambiguous (§27.5): the measured coverage is the result of the mapper's *policy* for multi-mapped reads (discard, random, or fractional assignment), not a property of the cell.
    2. *Test.* Stratify the loci by *mappability* (the fraction of $k$-mers unique in the genome). If the correlation is high at high mappability and falls with mappability in a way a *replicate-to-replicate* comparison also falls, the labels are the bottleneck and the model is at its ceiling (Chapter 1). Compute the ceiling from the replicate correlation per stratum.
    3. *Policy sensitivity.* Re-derive the labels under different multi-mapping policies; if the model's score changes by more than the replicate noise, the policy is part of the benchmark.
    4. *What is the model trained to do there?* If those loci were masked or set to zero during training, the model has *no signal* there; if they were included with zero-biased coverage, the model has learned that repeats are silent and may mispredict functional TE-derived enhancers.
    5. *Remedy.* Long-read data (spans repeats), unique-mappability masks reported alongside every score, and *explicit evaluation* of repeat classes rather than exclusion.

    **Expert analysis.** The exclusion silently narrows the claim to "regulatory activity in uniquely mappable sequence", about half to two thirds of the genome. A model that is evaluated only where the pipeline works cannot reveal its behavior where regulatory biology may be richest (recently expanded repeat families). The lesson is general: **every benchmark has a domain defined by its pipeline; state it.**

---

## 27.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: a provenance card for every label"
    **Setting.** You are about to train or evaluate a model against labels derived from sequencing.

    **For each label type, write down:**

    1. **Raw data.** Platform, read length, depth, replicates, quality filters.
    2. **Reference and annotation.** Which build (GRCh38, T2T-CHM13), which gene annotation and version; which alt contigs and decoys.
    3. **Aligner and parameters.** Seed length, mismatch tolerance, multi-mapping policy, MAPQ filter.
    4. **Mappability mask.** What fraction of the genome (and of each repeat class) is excluded, and was the model told?
    5. **Caller or quantifier.** Model (independent-error vs learned), filters (strand bias, depth), population error model.
    6. **Known blind spots.** Repeats, segmental duplications, high-GC regions, structural variants, indels in homopolymers, the MHC.
    7. **Ancestry assumptions.** Reference bias; which panel was used for imputation.
    8. **Noise ceiling.** Replicate agreement *by stratum* (mappability, GC, depth).
    9. **Policy sensitivity.** Do the labels change meaningfully under reasonable alternative pipeline choices?
    10. **Domain statement.** In one sentence: "this benchmark measures performance in regions where ...".

    **What it teaches.** This list takes less than an hour per dataset and routinely changes the claim in the abstract.

    **An open question to carry forward.** The FM-index and DP are exact; learned sequence models are approximate but can use *context* that exact algorithms ignore. Could a learned model of read placement (given the whole genome's sequence, including repeat copy-number structure and population variation) assign ambiguous reads *better than uniform random* within a repeat family, and how would you measure that without ground-truth locations? (Hint: spike-in synthetic reads, long-read-resolved truth, or simulated reads from T2T.)

---

## 27.12 Connections

- **Backward:** dynamic programming and HMMs (Chapter 6); relative entropy (Chapter 5); EM (Chapter 8); multiple testing (Chapters 4, 26); sequence variation (Chapter 20).
- **Forward:** representing genomes as strings, graphs, and tokens (Chapter 28); position-specific scoring, profile and Potts models (Chapter 29); sequence-to-function training labels (Chapter 31); genomic language models and reference bias (Chapter 32); MSAs for protein models (Chapters 34–35); benchmark leakage and family splits (Chapter 43); ancestry and distribution shift (Chapter 45).

!!! takeaways "Key takeaways"
    1. A pipeline of alignment, calling, and quantification produces every label in genomics. Its statistics and blind spots become the labels' noise and domain.
    2. Substitution scores are **log-likelihood ratios**; BLOSUM62 carries about **0.56 bits** per aligned pair, so significance needs tens of conserved positions.
    3. **Karlin–Altschul**: $E=Kmn\,e^{-\lambda S}$, with Gumbel-distributed maximum scores; verified: score SD 4.10 vs 4.04, and E-values calibrated from 2 to 350 expected false hits per query.
    4. **BWT/FM-index backward search** counts a pattern in $O(m)$ steps independent of genome size; it is exact (0 mismatches against brute force).
    5. **Repeats break mapping**: 44–65% of reads from our simulated repeat family were ambiguous at 50–250 bp, versus about 100% unambiguous in unique sequence; low mappability is *missing*, not zero, signal.
    6. **Genotype likelihood** calling saturates in depth for sensitivity (0.63 → 1.0 from 2× to 30×) and removes independent-error false positives (17,869 → 69 per $10^6$ sites), but **correlated errors persist at any depth** (7,770 per $10^6$ at 30×).
    7. Quantification is a **mixture model fitted by EM** on read equivalence classes.
    8. A benchmark's *domain* is defined by its pipeline: state it, and evaluate in the blind spots, not only outside them.

---

## Further reading

- Karlin, S. & Altschul, S. F. (1990). Methods for assessing the statistical significance of molecular sequence features by using general scoring schemes. *PNAS* 87, 2264–2268. Altschul, S. F. et al. (1990). Basic local alignment search tool. *J. Mol. Biol.* 215, 403–410. Altschul, S. F. et al. (1997). Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. *Nucleic Acids Res.* 25, 3389–3402. Henikoff, S. & Henikoff, J. G. (1992). Amino acid substitution matrices from protein blocks. *PNAS* 89, 10915–10919.
- Krogh, A. et al. (1994). Hidden Markov models in computational biology. *J. Mol. Biol.* 235, 1501–1531. Eddy, S. R. (1998). Profile hidden Markov models. *Bioinformatics* 14, 755–763. Remmert, M. et al. (2012). HHblits: lightning-fast iterative protein sequence searching by HMM–HMM alignment. *Nat. Methods* 9, 173–175. van Kempen, M. et al. (2024). Fast and accurate protein structure search with Foldseek. *Nat. Biotechnol.* 42, 243–246.
- Burrows, M. & Wheeler, D. J. (1994). A block-sorting lossless data compression algorithm. DEC SRC Research Report 124. Ferragina, P. & Manzini, G. (2000). Opportunistic data structures with applications. *FOCS*. Li, H. & Durbin, R. (2009). Fast and accurate short read alignment with Burrows–Wheeler transform. *Bioinformatics* 25, 1754–1760. Langmead, B. et al. (2009). Ultrafast and memory-efficient alignment of short DNA sequences to the human genome. *Genome Biol.* 10, R25. Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics* 34, 3094–3100.
- Lander, E. S. et al. (2001). Initial sequencing and analysis of the human genome. *Nature* 409, 860–921. Nurk, S. et al. (2022). The complete sequence of a human genome. *Science* 376, 44–53. Liao, W.-W. et al. (2023). A draft human pangenome reference. *Nature* 617, 312–324. Wenger, A. M. et al. (2019). Accurate circular consensus long-read sequencing improves variant detection and assembly of a human genome. *Nat. Biotechnol.* 37, 1155–1162.
- Li, H. (2011). A statistical framework for SNP calling, mutation discovery, association mapping and population genetical parameter estimation from sequencing data. *Bioinformatics* 27, 2987–2993. Poplin, R. et al. (2018). A universal SNP and small-indel variant caller using deep neural networks. *Nat. Biotechnol.* 36, 983–987. Li, N. & Stephens, M. (2003). Modeling linkage disequilibrium and identifying recombination hotspots using single-nucleotide polymorphism data. *Genetics* 165, 2213–2233.
- Bray, N. L., Pimentel, H., Melsted, P. & Pachter, L. (2016). Near-optimal probabilistic RNA-seq quantification. *Nat. Biotechnol.* 34, 525–527. Patro, R., Duggal, G., Love, M. I., Irizarry, R. A. & Kingsford, C. (2017). Salmon provides fast and bias-aware quantification of transcript expression. *Nat. Methods* 14, 417–419.
