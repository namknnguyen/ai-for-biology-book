# Chapter 28. Representing Biological Data: Alphabets, Tokens, Symmetries

!!! abstract "Chapter at a glance"
    **Motivation.** Every model of this book begins with a decision that is usually made in one line of code: how to turn a biological object into a tensor. For DNA that decision includes the alphabet, the strand, the unit of tokenization, and how to represent a *variant*; each choice sets what the model can and cannot learn, introduces symmetries or breaks them, and sometimes makes a benchmark trivially solvable. This chapter works through those decisions with a real genome (the 154-kb *Arabidopsis thaliana* chloroplast), measuring the consequences instead of asserting them.
    **Prerequisites.** Chapters 5, 10, 13 (representation, sufficiency), 16 (symmetry), 20, 27.
    **You will be able to:** (1) estimate the information available in DNA with held-out entropy and explain why gains from modeling are small in local context; (2) detect and explain homology leakage through reverse-complement repeats; (3) compare nucleotide, $k$-mer, BPE, and codon tokenizations on compression, shift sensitivity, variant locality, codon alignment, and leakage; (4) state the symmetries of biological objects (reverse complement, permutation, rotation) and how to impose or test them; (5) represent variants and haplotypes for a sequence model; (6) audit a representation by asking what it discards.

---

## 28.1 Representation is a modeling decision

Chapter 13 established the principle: a representation $s(x)$ is **sufficient** for a task when $I(s(x);y)=I(x;y)$, and every deterministic transformation can only lose information. Chapter 16 added that a representation should respect the symmetries of the problem. Biology adds a third consideration: the objects are *evolved*, so they carry units of selection (codons, motifs, domains, genes) whose boundaries are not marked in the raw sequence. A representation therefore has to answer four questions:

1. **What is the object?** A sequence, a molecule, a structure, a cell, a population; with what context?
2. **What are its symmetries?** Which transformations leave the biology unchanged (reverse complement of a double-stranded site; relabeling of cells or atoms) and which change it (reverse complement of a gene; mirror image of a protein)?
3. **What information does it keep and discard?** For example, one-hot DNA keeps all nucleotide identities and discards methylation, chromatin state, and the diploid phase.
4. **What does it make easy to learn, and what does it make *trivial*?** A tokenization can turn a pretraining task into a lookup (§28.4.2).

---

## 28.2 DNA: alphabet, strand, and the reverse-complement symmetry

DNA is a string over $\{A,C,G,T\}$ (with IUPAC ambiguity codes and the unknown base $N$). The default input to sequence models is **one-hot encoding**, $x\in\{0,1\}^{4\times L}$, which is lossless for the nucleotide sequence, linear in the model's first layer, and natural for convolutions (Chapter 10). The decisions that matter are about *strand* and *context*.

**Strand and reverse complement.** DNA is double-stranded; the two strands carry the same information, written in opposite directions, with $A\!\leftrightarrow\!T$ and $C\!\leftrightarrow\!G$ pairing. The reverse complement $\mathrm{RC}(x)$ of a sequence read on one strand is the sequence read on the other. For *binding of a double-stranded DNA-binding protein* the two are equivalent; for *transcription* they are not: a gene is read from one specific strand and $\mathrm{RC}(\text{gene})$ is not a gene. A model of a double-stranded property, $f(\mathrm{RC}(x))=f(x)$, should be **RC-invariant**; for per-position, strand-specific outputs the right property is **RC-equivariance**, $f(\mathrm{RC}(x))=\mathrm{flip}(f(x))$ with the outputs reversed in position and the strand labels swapped. Three ways to impose it: (i) **augmentation** (train on both orientations); (ii) **test-time averaging** (average $f(x)$ and the flipped $f(\mathrm{RC}(x))$); (iii) **architectural equivariance** with weight sharing across the group $\{e,\mathrm{RC}\}$ (as in RC-equivariant convolutions and the Caduceus architecture; Schiff et al., 2024) [[S]].

**Chargaff's second parity rule** is the statistical reason this is natural: in a single strand of a genome, the frequency of a word is close to that of its reverse complement (Rudner et al., 1968; Mitchell & Bridge, 2006) [[E]]. On the chloroplast genome (both strands mixed, since genes lie on both), the correlation between the count vector of $k$-mers and that of their reverse complements is **0.995, 0.989, and 0.972** for $k=2,4,6$. But restricted to the *concatenated sense strands of the genes*, the correlation falls to **0.961, 0.906, and 0.819**, and the mean absolute log$_2$ ratio between a word and its reverse complement rises from $0.046,0.097,0.261$ to $0.128,0.309,0.667$. *RC symmetry is a property of genomes averaged over both strands, and it is broken by genes*: codon usage, the genetic code, start and stop signals, and strand-specific mutational processes. A model that is forced to be RC-invariant cannot represent *which strand a gene is on*, and one that is not constrained must learn the symmetry from data (which RC augmentation provides cheaply). The right choice depends on the task (§28.6).

**Variants and haplotypes.** A variant is not a token but an *edit*. Common representations: (a) the **reference/alternative window pair** (the same genomic window with the reference and the alternative allele; the model scores both; the effect is a difference of outputs, as in DeepSEA, Enformer, and AlphaGenome); (b) a **sequence with the variant substituted**, enabling the model to see multi-variant haplotypes, which matters because variants in LD act together (Chapter 26); (c) **features of the variant** (allele frequency, conservation) combined with the sequence model's score. For *indels*, coordinates shift downstream, and positional outputs must be re-aligned. For **personal genomes**, the diploid state means two haplotypes, which one-hot encoding of the reference cannot represent; phased personal-genome input is rare in current models, yet is exactly what *genotype-to-phenotype* prediction (Chapter 41) requires. A **graph genome** (Chapter 27) represents population variation structurally and is a natural target for graph neural networks (Chapter 16), but few sequence foundation models consume graphs.

**Strand, context, and position.** The window length (kb to Mb), whether to supply coordinates or position encodings, and whether to include **both** flanks (bidirectional) or only the left context (autoregressive) are modeling choices that define the task (Chapters 31–32). Absolute position matters (a TSS is at a particular place), which breaks the translation equivariance that convolutions assume; relative position encodings (RoPE, Chapter 12) and long-range structure (Chapter 11) address the rest.

---

## 28.3 How much is there to learn? Entropy, leakage, and the codon frame

Before building a model of DNA, measure what a *simple* model can extract. For a stationary source the best achievable cross-entropy (bits per base) is the entropy rate $\bar H\le2$. We estimate it on held-out DNA with **order-$k$ Markov models** (smoothed counts of the next base given the previous $k$ bases), the simplest nontrivial predictors (`code/ch28_representation.py`).

**The genome has a lot of structure and little compressible local context.** The chloroplast genome has GC content 0.363 (so order-0 entropy is already below 2: 1.939 bits). On a held-out *unique* region, context adds little: order-2 gives 1.917 and order-4 1.914, a gain of only **0.025 bits per base (1.3%)** over base composition; higher orders *overfit* (order 8: 2.10 bits, worse than the 2-bit uniform baseline) because a 130-kb training sequence cannot populate $4^8$ contexts. The qualitative conclusion, **that local-context models of non-repetitive DNA are close to incompressible, with perplexity near 3.8 out of a maximum of 4**, is consistent with the reported compressibility of larger genomes (typically about 1.9 bits per base for the human genome, depending on the method) [[S]], but our numbers come from a single small genome, so treat the exact value as illustrative. For a language model of DNA, **this is the baseline to beat**: most improvements in log-likelihood come from *long-range, repeated, or functional* structure, not from local statistics.

**Leakage through reverse-complement repeats.** Many genomes contain duplicated sequence, including *inverted* repeats: copies in the reverse-complement orientation. By hashing all 25-mers and searching for reverse-complement matches (no alignment needed), we found that **34% of 25-mers in this genome have a reverse-complement partner elsewhere**, and located two blocks (positions 84,170–110,434 and 128,214–154,478), *each exactly 26,264 bp*, the known large inverted repeat of this genome (Sato et al., 1999). Now evaluate a Markov model *trained on the rest of the genome* on one copy of the repeat:

| Order $k$ | Unique test region: forward only | Unique region: + RC augmentation | Repeat copy: forward only | Repeat copy: + RC augmentation |
|---|---|---|---|---|
| 0 | 1.939 | 1.939 | 1.999 | 1.999 |
| 2 | 1.917 | 1.917 | 1.976 | 1.975 |
| 4 | 1.914 | 1.914 | 1.973 | 1.964 |
| 6 | 1.968 | 1.957 | 2.043 | **1.912** |
| 8 | 2.104 | 2.145 | 2.130 | **1.507** |
| 10 | 2.024 | 2.049 | 2.025 | **1.094** |

On the unique region the model is no better than uniform at high order; on the repeat copy, a model with reverse-complement augmentation and high capacity reaches **1.09 bits** (versus 2.02 for the unique region and 2.03 without augmentation): *it has memorized the other copy*. Two lessons. **(i) Held-out regions share sequence with training regions** whenever the genome contains segmental duplications, transposable-element families, or paralogs; a "held-out chromosome" evaluation of a genomic language model is leaky unless these are removed (Chapter 43). **(ii) RC augmentation, which is the right symmetry, also enlarges the leak**: a forward-only model could not exploit an inverted repeat, and an RC-augmented one can. Evaluation must therefore be homology-aware *in both orientations*.

**The codon frame is worth as much as the context.** In coding sequence, the position within a codon strongly shapes base composition (here, GC at the third codon position is only 0.274 versus 0.363 overall). Training order-$k$ models on 41 genes and testing on 18 held-out genes:

| Order $k$ | Frame-agnostic | Frame-aware (3 position-specific tables) | Gain (bits/base) |
|---|---|---|---|
| 0 | 1.961 | 1.936 | 0.024 |
| 1 | 1.948 | 1.912 | 0.036 |
| 2 | 1.943 | 1.893 | **0.050** |
| 3 | 1.941 | 1.897 | 0.044 |
| 4 | 1.947 | 1.918 | 0.029 |

Knowing the frame is worth up to 0.05 bits per base, about **twice the gain from all local context (0.025 bits)**. A DNA language model that is to model coding regions *must discover the frame* (position mod 3 within a gene) from the sequence, and tokenizers that cut codons arbitrarily make this harder (§28.4.3). Evolved structure appears in the representation as much as in the data.

---

## 28.4 Tokenization

A **tokenizer** maps a string to a sequence of discrete units; the model's vocabulary, context length (in tokens), and inductive bias follow. The options in use:

| Scheme | Tokens | Used by (examples) | Context for fixed token budget |
|---|---|---|---|
| Single nucleotide | $A,C,G,T,N$ | HyenaDNA, Evo (byte-level), GPN | 1 base/token |
| Overlapping $k$-mers (stride 1) | all $k$-mers | DNABERT (Ji et al., 2021) | 1 base/token (but each token is $k$ bases) |
| Non-overlapping $k$-mers | consecutive $k$-mers | Nucleotide Transformer ($k=6$; Dalla-Torre et al.) | $k$ bases/token |
| Byte-pair encoding (BPE) | learned merges | DNABERT-2 (Zhou et al., 2024) | $\approx4$–5 bases/token |
| Codon / amino acid | codons within coding regions | codon language models (e.g., CaLM; Outeiral & Deane, 2024) | 3 bases/token |

The properties that matter are measured below on the chloroplast genome: BPE tokenizers were trained on the first 75% of the genome and evaluated on the held-out 25%.

### 28.4.1 Compression, shift sensitivity, and variant locality

| Tokenizer | Bases/token (held-out) | Token boundaries retained after a 1-base shift | Downstream tokens identical after a 1-base insertion |
|---|---|---|---|
| Single nucleotide | 1.00 | 1.000 | 1.000 |
| Non-overlapping 3-mer | 3.00 | **0.000** | **0.000** |
| Non-overlapping 6-mer | 6.00 | **0.000** | **0.000** |
| BPE, vocabulary 260 | 3.51 | 1.000 | 1.000 |
| BPE, vocabulary 1,024 | 4.20 | 1.000 | 1.000 |
| BPE, vocabulary 4,096 | 4.74 | 1.000 | 1.000 |

and for a single-base substitution (400 random sites; windows of 2,000 bases):

| Tokenizer | Mean tokens changed | Fraction of substitutions that change the **number** of tokens |
|---|---|---|
| Single nucleotide | 1.00 | 0.000 |
| Non-overlapping 6-mer | 1.00 | 0.000 |
| Overlapping 6-mer (stride 1) | 6.00 | 0.000 |
| BPE, vocabulary 4,096 | 1.77 | **0.237** |

*Fixed-stride tokenizers are exquisitely sensitive to indels*: a one-base insertion shifts the frame of every downstream token so that **none** of them is identical afterwards; a model that has learned 6-mer tokens in one frame sees a completely different token sequence for the same biology. BPE is *content-defined*: boundaries are determined by local sequence, so they resynchronize immediately after a shift and an insertion perturbs only nearby tokens, a real advantage, which comes at a price for variant effect scoring: **24% of single-base substitutions change the number of tokens**, which breaks the alignment between the reference and alternative token sequences and makes per-position comparisons (effect = difference of per-position outputs) ill-defined. Single-nucleotide tokens have neither problem but cost a factor of 3–5 in context length. Using non-overlapping $k$-mers, a substitution changes exactly one token, but its *meaning* depends on the frame in which the token was cut.

### 28.4.2 The masked-token leak of overlapping $k$-mers

DNABERT-style models use *overlapping* $k$-mers: token $i$ covers bases $i,\dots,i+k-1$. Masking one token leaves its neighbors visible, and tokens $i-1$ and $i+1$ already contain all $k$ of its bases: base $i$ is the second base of token $i-1$, and bases $i+1,\dots,i+k-1$ are the first $k-1$ bases of token $i+1$. A trivial rule that copies those bases recovers the masked 6-mer **exactly in 20,000 of 20,000 cases** (accuracy 1.000), with no learning. The fix is to mask *contiguous* spans of $k$ tokens ($2k-1$ bases), which removes the shortcut; on held-out DNA a greedy order-4 Markov model then achieves 0.320 accuracy per hidden base, against 0.310 for always guessing the most common base (chance 0.250). *The masked-LM objective was a lookup table until the masking was changed*; a reported "98% masked-token accuracy" is not evidence of biological learning unless the masking prevents this leak.

### 28.4.3 Do token boundaries respect biology?

Within 26 forward-strand genes, we asked what fraction of token boundaries fall on codon boundaries (chance: one third):

* Non-overlapping 3-mers or 6-mers laid from the start of the genome: **0.224**. This is *all-or-nothing per gene*: 8 of the 26 genes happen to begin in the tokenizer's frame (fraction 1.0) and the other 18 are out of frame (fraction 0), so the token carries amino-acid meaning for about a third of genes and cuts every codon of the rest.
* BPE (vocabulary 4,096): **0.404**, modestly above chance because BPE merges frequent substrings (some of which are codon-structured), but the boundaries are still not aligned.

Genes occur at arbitrary offsets and strands; **no fixed tokenization of the genome is in frame for all genes**. Remedies are *frame-aware tokenization* given annotation (codon tokens in coding sequence), *augmentation* over all three frames and both strands, or models with sufficient capacity to discover the frame (which §28.3 suggests is valuable, 0.05 bits per base). The deeper point: **a tokenizer encodes an assumption about the units of biology**, and the units differ across scales: codons (3 bases), motifs (6–20), exons (about 150), genes (thousands), TADs (hundreds of kb).

---

## 28.5 Beyond DNA: proteins, structures, molecules, cells

| Object | Standard representations | Symmetries | What a representation can discard |
|---|---|---|---|
| Protein sequence | 20-letter one-hot or learned embeddings; MSAs (Chapter 27); language-model embeddings (Chapter 34) | none (sequence has direction); permutation of homologs in an MSA | Structure, dynamics, modifications, cellular context |
| Protein structure | Residue graph (distance cutoff), frames and torsions, atom point clouds (Chapter 16), voxel grids | Rotation and translation ($SE(3)$); chirality is *not* a symmetry | Conformational ensembles, solvent, ligands |
| Small molecule | SMILES, graph, fingerprint, 3-D conformers (Chapter 24) | Atom permutation; rotation/translation for 3-D | Stereochemistry (if 2-D), protonation, conformer ensemble |
| RNA | Sequence, secondary-structure graph, 3-D | As proteins; base-pairing structure | Co-transcriptional folding, modifications |
| Single cell | Count vector over genes; normalized log values; rank-ordered gene tokens; embeddings (Chapters 25, 38) | Permutation of genes; (non-)permutation of cells in a set | Absolute abundance (compositional), protein, time |
| Population / patient | Genotype matrix, summary statistics, graphs of relatedness | Permutation of individuals, allele coding flips | Phasing, environment, rare variants |
| Knowledge | Graphs (genes, pathways, drugs, diseases), ontologies, text | Node permutation | Context, evidence strength, contradiction |

Common to all: **the symmetry should be built in or tested, and the information discarded should be stated.** A gene-expression vector has no order, so a transformer over genes needs a rule for ordering or must be permutation-invariant (the *tokenization* of a cell as a sentence of genes ranked by expression, as in Geneformer, injects an order, and Chapter 38 asks what that ordering means). In structure, SE(3)-equivariant networks (Chapter 16) avoid learning rotations; with sequence input, relative positions and RC symmetry play the same role.

---

## 28.6 A symmetry audit

!!! lens "Research lens: the symmetry audit"
    For any representation, write the group $G$ of transformations of the *input* that should leave the *target* unchanged (invariance) or transform it predictably (equivariance), and check each element empirically:

    | Transformation | Should the target be unchanged? | Test |
    |---|---|---|
    | Reverse complement of a TF binding site window | Yes (binding to dsDNA) | Average $|f(x)-f(\mathrm{RC}(x))|$ over test windows |
    | Reverse complement of a gene | **No** (strand is part of the identity) | Model must distinguish: $f(x)\ne f(\mathrm{RC}(x))$ for transcribed sequence |
    | Shift of the window by a few bases | Usually yes for motif presence; **no** for position-specific signals (TSS) | Predictions vs shift |
    | One-base insertion upstream | Yes for distant biology; **no** for in-frame coding sequence | Compare predictions |
    | Permutation of gene order in a cell | Yes | Shuffle genes, compare embeddings |
    | Rotation of a protein structure | Yes | Equivariance error |
    | Mirror image of a protein | **No** | Chirality (Chapter 16) |
    | Change of batch label | Yes for cell type | Embedding batch mixing (Chapter 13) |

    A representation that passes the invariances *and fails the non-invariances* has the right symmetry; most published models test only the first column.

---

## 28.7 The experiments, verbatim

```python
--8<-- "code/ch28_representation.py"
```

```text
genome: 154,478 bp, GC = 0.363, features: 85 CDS, 37 tRNA, 7 rRNA

== 1. Finding the large inverted repeat with 25-mer hashing (no alignment) ==
34.0% of 25-mers have a reverse-complement copy elsewhere; large inverted-repeat blocks (start, end, length): (84,170, 110,434, 26,264); (128,214, 154,478, 26,264)

== 2. Held-out cross-entropy (bits per base) of order-k Markov models; the maximum is 2.0 ==
inverted repeat copies at (84170, 110434) and (128214, 154478)
order k   unique test region               | test = one copy of the inverted repeat
          forward only   + RC augmentation | forward only   + RC augmentation
    0       1.939          1.939          |   1.999          1.999
    2       1.917          1.917          |   1.976          1.975
    4       1.914          1.914          |   1.973          1.964
    6       1.968          1.957          |   2.043          1.912
    8       2.104          2.145          |   2.130          1.507
   10       2.024          2.049          |   2.025          1.094

== 3. Knowing the codon frame (held-out genes; coding sequences on the sense strand) ==
59 complete CDS (>=300 bp); 41 training genes (56,373 bp), 18 held-out genes (18,585 bp); GC3 = 0.274
order k   frame-agnostic   frame-aware (3 position-specific tables)   gain (bits/base)
    0        1.961           1.936                                  0.024
    1        1.948           1.912                                  0.036
    2        1.943           1.893                                  0.050
    3        1.941           1.897                                  0.044
    4        1.947           1.918                                  0.029

== 4. Tokenization ==
tokenizer              bases/token on held-out DNA   mean token length   boundaries retained after a 1-base shift   tokens identical after a 1-base insertion (downstream)
single nucleotide            1.00                        1.00               1.000                                   1.000
non-overlapping 3-mer        3.00                        3.00               0.000                                   0.000
non-overlapping 6-mer        6.00                        6.00               0.000                                   0.000
BPE, vocabulary 260          3.51                        3.51               1.000                                   1.000
BPE, vocabulary 1024         4.20                        4.20               1.000                                   1.000
BPE, vocabulary 4096         4.74                        4.74               1.000                                   1.000

effect of a single-base substitution (window of 2,000 bases around the site): tokens that change, and whether the token count changes
tokenizer              mean tokens changed (ref side)   fraction of SNPs that change the number of tokens
single nucleotide                 1.00                              0.000
non-overlapping 6-mer             1.00                              0.000
overlapping 6-mer (stride 1)      6.00                              0.000
BPE, vocabulary 4096              1.77                              0.237

fraction of token boundaries inside forward-strand genes that fall on codon boundaries (chance = 1/3), 26 genes:
  non-overlapping 3-mers from the genome start: 0.224  (all-or-nothing per gene: 8 of 26 genes happen to start in the tokenizer's frame)
  non-overlapping 6-mers from the genome start: 0.224
  BPE (vocabulary 4096), tokenizing the whole genome: 0.404

overlapping 6-mer MLM: mask ONE token; a rule that copies from the two neighboring tokens recovers the masked token exactly in 1.000 of 20000 cases
mask k = 6 contiguous tokens (hides 11 bases): greedy order-4 Markov accuracy per hidden base = 0.320; most-common-base baseline = 0.310; chance = 0.250

== 5. Reverse-complement symmetry: Chargaff's second rule, and where it breaks ==
k   whole genome, one strand: corr(count(w), count(rc(w)))   concatenated sense strands of genes only   mean |log2 ratio| genome / genes
2     0.9952                                        0.9609                               0.046 /  0.128
4     0.9892                                        0.9059                               0.097 /  0.309
6     0.9720                                        0.8185                               0.261 /  0.667
```

---

## 28.8 Worked research examples

!!! example "Worked Research Example 28.1: A genomic language model reports near-perfect masked-token accuracy"
    **Situation.** A paper pretrains a BERT-style model on overlapping 6-mers of the human genome and reports 95% masked-token accuracy on held-out chromosomes, and claims the model "learned the grammar of the genome."

    **Question.** What would you check before accepting that claim?

    **Reasoning.**

    1. *Does the objective have a shortcut?* With overlapping $k$-mers and single-token masking, the masked token is exactly recoverable from its neighbors (§28.4.2: 20,000/20,000). If the paper masks single tokens, 95% accuracy is *below* what a copy rule achieves, and is evidence that the model has not even learned to copy. If it masks contiguous spans, the shortcut is gone and the accuracy must be compared to a *baseline*: order-$k$ Markov predictor, most-common-base (0.31 in our genome), and composition-only. In human DNA, repeats make high-order baselines strong.
    2. *Is the held-out set homologous to the training set?* Held-out chromosomes share segmental duplications, transposon families, and paralogs with training chromosomes. In our genome, a model with RC augmentation dropped from 2.02 to 1.09 bits on a repeat copy: leakage can masquerade as learning. Remove regions with sequence similarity above a threshold *in both orientations*.
    3. *What does the accuracy decompose into?* Stratify by region class (repeat, unique, coding, regulatory), by repeat age, and by distance to nearest training homolog. If all accuracy is in repeats, the model has learned the repeat library.
    4. *What is the downstream test?* Zero-shot variant-effect prediction or regulatory-element prediction against *held-out experimental labels* (Chapters 31–32) with strong supervised baselines trained on one-hot input.

    **Expert analysis.** The checks take a day and routinely reduce a headline to a sentence. They also illustrate the general rule for foundation-model papers: *the pretraining metric is evidence about the pretraining task, not about biology.* An internally consistent critique uses A5 (evaluation attack) and A3 (objective attack).

!!! example "Worked Research Example 28.2: Scoring a SNP with a BPE-tokenized model"
    **Situation.** You want to use a pretrained DNA language model with a BPE tokenizer (vocabulary 4,096) to score variants by the change in per-position log-likelihood between the reference and alternative sequences.

    **Question.** What goes wrong, and how do you design the score?

    **Reasoning.**

    1. *The coordinate problem.* In our experiment 23.7% of single-base substitutions change the number of tokens in a 2,000-base window, and the average substitution changes 1.77 tokens. The reference and alternative token sequences are therefore not position-aligned, so "difference of per-token log-likelihoods at the variant position" is not defined for about a quarter of variants.
    2. *A tokenization-independent score.* Compute the **total sequence log-likelihood** $\log p(\text{window}_\text{alt})-\log p(\text{window}_\text{ref})$ over the *whole* window (both tokenized independently); this is well defined, but it also mixes the effect of the variant with changes in tokenization length (more tokens means more terms), so *normalize per base* or use the log-likelihood of the *bases* given the model, marginalizing over tokenizations if the model supports it.
    3. *Control for tokenization artifacts.* Compare against a *null* variant set (random substitutions matched for local sequence composition) and include the **change in token count** as a covariate. If the score correlates with token-count change, it measures the tokenizer.
    4. *Alternative.* Use a single-nucleotide model (no alignment problem; costs context length) or a model that produces per-base outputs.
    5. *Calibration.* Test on variants with experimentally measured effects (MPRA, saturation mutagenesis; Chapter 31), and compare against a simple supervised baseline.

    **Expert analysis.** The tokenizer is part of the *measurement instrument* for variant effects. The same logic applies to indel scoring and multi-variant haplotypes, where fixed-stride tokenizers fail entirely (§28.4.1).

---

## 28.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: five decisions before you encode a biological object"
    **Setting.** You are about to feed a new biological object to a model.

    1. **Unit.** What is the unit of selection or function, and does your tokenization align with it, cut across it, or hide it? Write down the typical length and the offset relative to the biology.
    2. **Symmetry.** List the transformations that should leave the target unchanged and those that must change it. For each, decide: build in, augment, average at test time, or ignore (and measure the violation).
    3. **Information.** Write what the representation discards. If the target depends on it (chirality, protonation, modifications, phasing), the ceiling is below the noise ceiling (Chapter 1).
    4. **Shortcuts.** Does the pretraining or labeling procedure admit a lookup (overlapping tokens, duplicated sequences, shared motifs, label leakage across the split)? Design the *adversarial baseline*: copy rule, nearest-neighbor, Markov predictor.
    5. **Edits.** How will variants, perturbations, and indels be represented, and is the effect of an edit well defined under your tokenization?

    **What it teaches.** These five decisions are inexpensive to document and expensive to change after training. Treating the representation as a hypothesis, with the null baselines above, is the A2 attack (representation) done carefully.

    **An open question to carry forward.** Could tokenization be *learned*, from bytes to units, jointly with the model, so that boundaries land on biological units without annotation? Hierarchical byte-level architectures that pool bytes into learned "patches" exist in language modeling. Design a test on this chapter's data: train a model with learned pooling on the chloroplast genome and ask whether pooled boundaries concentrate on codon boundaries inside genes (against the 1/3 chance rate) and whether they are stable to a one-base shift and one-base insertion.

---

## 28.10 Connections

- **Backward:** sufficiency and invariance (Chapters 13, 16); entropy and information (Chapter 5); convolutions and motif grammar (Chapter 10); alignment and repeats (Chapter 27); genome structure (Chapter 20).
- **Forward:** classical models whose representations are explicit (Chapter 29); sequence-to-function models, which use one-hot input and long context (Chapter 31); genomic language models and their tokenizers (Chapter 32); protein language models (Chapter 34); single-cell tokenization (Chapter 38); leakage-free benchmarks (Chapter 43).

!!! takeaways "Key takeaways"
    1. A representation fixes what a model can learn, what is trivial, and what is symmetric; it is a hypothesis to be tested against null baselines.
    2. **DNA is nearly incompressible in local context**: on a held-out unique region, order-4 Markov gives 1.914 bits per base against 1.939 for composition alone; high-order models overfit small genomes.
    3. **Reverse-complement symmetry** holds for genomes averaged over strands (corr 0.995 at $k=2$) but is **broken in genes** (0.961), so it must be imposed for double-stranded properties and not for transcribed sequence.
    4. **Inverted repeats leak**: 34% of 25-mers had a reverse-complement partner; with RC augmentation a high-order model reached 1.09 bits on a repeat copy versus 2.02 on unique DNA. Held-out regions must be homology-filtered in both orientations.
    5. **The codon frame is worth about twice the information of all local context** (0.050 vs 0.025 bits per base); fixed-stride tokenizers are in frame for only about a third of genes (8 of 26).
    6. **Tokenization trade-offs**: fixed $k$-mers are fragile to indels (0.000 of downstream tokens survive a one-base insertion) but give local variant effects; BPE is shift-robust but **24% of SNPs change the token count**; single nucleotides are clean but expensive.
    7. **Overlapping $k$-mer masked-token prediction is a lookup**: copying from neighbors solves it exactly (20,000/20,000).
    8. Perform a **symmetry audit** (what should be invariant, and what must not be) and write down what the representation discards.

---

## Further reading

- Ji, Y., Zhou, Z., Liu, H. & Davuluri, R. V. (2021). DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. *Bioinformatics* 37, 2112–2120. Zhou, Z. et al. (2024). DNABERT-2: efficient foundation model and benchmark for multi-species genome. *ICLR*. Dalla-Torre, H. et al. (2025). Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* 22, 287–297. Nguyen, E. et al. (2023). HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. *NeurIPS*. Outeiral, C. & Deane, C. M. (2024). Codon language embeddings provide strong signals for use in protein engineering. *Nat. Mach. Intell.* 6, 170–179.
- Schiff, Y. et al. (2024). Caduceus: bi-directional equivariant long-range DNA sequence modeling. *ICML*. Sennrich, R., Haddow, B. & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL*.
- Rudner, R., Karkas, J. D. & Chargaff, E. (1968). Separation of *B. subtilis* DNA into complementary strands. III. Direct analysis. *PNAS* 60, 921–922. Mitchell, D. & Bridge, R. (2006). A test of Chargaff's second rule. *Biochem. Biophys. Res. Commun.* 340, 90–94.
- Sato, S. et al. (1999). Complete structure of the chloroplast genome of *Arabidopsis thaliana*. *DNA Res.* 6, 283–290.
- Benegas, G., Batra, S. S. & Song, Y. S. (2023). DNA language models are powerful predictors of genome-wide variant effects. *PNAS* 120, e2311219120.
