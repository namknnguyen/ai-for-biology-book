# Chapter 6. Computational Thinking and Research Implementation

!!! abstract "Chapter at a glance"
    **Motivation.** Research ideas are tested through code. Subtle implementation errors (off-by-one coordinates, leaked splits, silent broadcasting, nondeterminism) can create results that are more impressive than the idea deserves or hide results that are real. This chapter teaches the algorithmic thinking and the engineering discipline that make experiments trustworthy.
    **Prerequisites.** Chapters 2–5; basic Python.
    **You will be able to:** (1) derive and implement dynamic-programming algorithms for sequences (alignment, HMM forward/Viterbi) and explain them as semiring computations; (2) estimate time, memory, and compute for a training run before launching it; (3) handle genomic coordinates and file formats without off-by-one errors; (4) write the unit tests that catch the majority of model bugs (overfit-one-batch, shape asserts, symmetry tests, shuffled-label controls); (5) organize an experiment so that it is reproducible and its splits cannot silently leak.

---

## 6.1 Computational thinking for biological data

A research programmer's first questions about any task are:

1. **What is the size of the problem?** ($L=10^6$ bases? $N=10^7$ cells? $10^9$ sequences?) Which algorithms scale, which don't?
2. **What is the structure?** Is there *optimal substructure* (dynamic programming)? *Locality* (convolutions)? *Sparsity* (sparse matrices)? *Symmetry* (reverse-complement, permutation of cells)?
3. **What is the invariant I can test?** (a reference allele must match the genome; probabilities must sum to one; a symmetrized model must be exactly symmetric.)

The chapter has three parts: algorithms (§6.2–6.3), systems and formats (§6.4), and the engineering of trustworthy experiments (§6.5–6.7).

---

## 6.2 Dynamic programming: the algorithm behind classical sequence analysis

### 6.2.1 Needleman–Wunsch alignment, derived

**Problem.** Given sequences $a_{1:n}$ and $b_{1:m}$, find the best *global alignment*: insert gaps so that residues line up, maximizing a score $\sum$(match/mismatch scores) $+$ (gap penalties). The number of possible alignments is exponential in $n+m$ (there are $\binom{n+m}{n}$ monotone paths even ignoring gap placement subtleties), so naive enumeration is hopeless.

**Optimal substructure.** Let $S_{i,j}$ be the best score for aligning prefixes $a_{1:i}$ and $b_{1:j}$. The last column of an optimal alignment of these prefixes is one of three types: $a_i$ aligned to $b_j$; $a_i$ aligned to a gap; $b_j$ aligned to a gap. Removing the last column leaves an optimal alignment of a shorter pair of prefixes (otherwise we could improve the whole alignment). So

$$
S_{i,j}=\max\begin{cases}
S_{i-1,j-1}+s(a_i,b_j)&\text{(match/mismatch)}\\
S_{i-1,j}+g&\text{(gap in }b)\\
S_{i,j-1}+g&\text{(gap in }a)
\end{cases},
\qquad S_{i,0}=ig,\quad S_{0,j}=jg .
$$

Filling the $(n+1)\times(m+1)$ table takes $O(nm)$ time and memory; a *traceback* from $(n,m)$ recovers the alignment. **Smith–Waterman** local alignment adds a fourth option $0$ to the $\max$ (allowing the alignment to restart) and takes the maximum over the whole table. **Affine gap penalties** (a cost $g_o+k\,g_e$ for a gap of length $k$) need three tables (Gotoh, 1982) and are the biologically standard choice because indels of several bases are about as likely as single-base ones.

The verified implementation (`code/ch06_dynamic_programming.py`) reproduces brute-force enumeration on 30 random pairs and aligns GATTACA/GCATGCA with score 5.

### 6.2.2 HMMs: the forward algorithm and Viterbi

A hidden Markov model has hidden states $z_t\in\{1,\dots,K\}$ following a Markov chain with transition matrix $T_{jk}=P(z_t=k\mid z_{t-1}=j)$ and initial distribution $\pi$, and observations $o_t$ emitted with $E_k(o)=P(o\mid z_t=k)$. Typical biological hidden states: *coding vs. non-coding*, *CpG island vs. background*, *domain boundaries* in a protein family (profile HMMs; Chapter 27).

**The likelihood requires summing over $K^L$ state paths.** Dynamic programming does it in $O(LK^2)$. Define the *forward variable* $\alpha_t(k)=P(o_{1:t},z_t=k)$. Then

$$
\alpha_1(k)=\pi_kE_k(o_1),\qquad
\alpha_t(k)=E_k(o_t)\sum_j\alpha_{t-1}(j)\,T_{jk},\qquad
P(o_{1:L})=\sum_k\alpha_L(k).
$$

*Derivation of the recursion.* $\alpha_t(k)=\sum_jP(o_{1:t},z_{t-1}=j,z_t=k)=\sum_jP(o_{1:t-1},z_{t-1}=j)\,P(z_t=k\mid z_{t-1}=j)\,P(o_t\mid z_t=k)$, using the Markov property and conditional independence of emissions. $\square$ In practice work in log space: $\log\alpha_t(k)=\log E_k(o_t)+\mathrm{logsumexp}_j[\log\alpha_{t-1}(j)+\log T_{jk}]$, where $\mathrm{logsumexp}(v)=m+\log\sum_je^{v_j-m}$ with $m=\max_jv_j$ to avoid underflow.

**Viterbi** replaces the sum by a max, $\delta_t(k)=E_k(o_t)\max_j\delta_{t-1}(j)T_{jk}$, and records back-pointers to recover the single most probable state path. Run on a 30-base sequence with a GC-rich stretch, the two-state CpG-island HMM in the code marks exactly the GC-rich region (`........IIIIIIIIIIIII.........`), and its forward log-likelihood on a 14-base prefix equals brute force over $2^{14}$ paths to six decimals ($-18.338810$).

!!! rhyme "Structural rhyme: alignment ↔ HMM inference ↔ attention ↔ message passing, all one dynamic program in different semirings"
    Replace *sum* and *product* in the forward recursion by *max* and *plus* (in log space) and you get Viterbi; Needleman–Wunsch is the same computation on a two-sequence lattice. Using $(\mathrm{logsumexp},+)$ instead of $(\max,+)$ on the alignment lattice gives the *partition function over all alignments* (a "pair HMM"; Durbin et al., 1998), whose gradient yields posterior alignment probabilities. These are all instances of the *same algorithm* in different **semirings** (Goodman, 1999). The same viewpoint explains why alignment-like operations have been made differentiable (soft-DTW; learned alignment; CTC loss in speech) and why a recurrent model's hidden state can be viewed as an approximate forward message. The practical meaning: when you need an alignment as part of a learned model, a *soft* (sum-product) version gives gradients, a *hard* (max-product) version gives decisions.

### 6.2.3 Complexity cheatsheet for genomics

| Task | Naive | Standard method | Notes |
|---|---|---|---|
| Count all $k$-mers in a length-$L$ sequence | $O(Lk)$ | Rolling hash, $O(L)$ | A 4^k-sized table is feasible for $k\le12$ |
| Find exact pattern (length $m$) in a genome ($G$) | $O(Gm)$ | Suffix array / FM-index (Burrows–Wheeler) $O(m)$ | The basis of fast read aligners |
| Pairwise alignment | exponential | DP $O(nm)$ | Heuristics (BLAST, minimap2) for large $G$ |
| Multiple alignment of $N$ sequences | $O(L^N)$ | Progressive/iterative heuristics | NP-hard in general |
| Interval overlap among $n$ genomic features | $O(n^2)$ | Interval trees / sweep $O(n\log n)$ | `bedtools`-style operations |
| PCA on $N\times G$ | $O(NG\min(N,G))$ | Randomized SVD $O(NGk)$ | Chapter 2 |
| Self-attention over $L$ tokens | $O(L^2d)$ | Local, sparse, linear, or SSM alternatives | Chapter 12 |

---

## 6.3 Genomic data: formats, coordinates, and the bugs they cause

Most data-handling mistakes in genomic machine learning are *coordinate* mistakes.

| Format | Content | Coordinates |
|---|---|---|
| FASTA / FASTQ | Sequences; reads with per-base quality | — |
| SAM/BAM/CRAM | Read alignments to a reference | 1-based (SAM text), 0-based in binary BAM |
| **VCF/BCF** | Variants: `CHROM POS REF ALT` | **1-based**; `REF` must equal the reference base(s) at `POS` |
| **BED** | Genomic intervals: `chrom start end` | **0-based, half-open** $[\text{start},\text{end})$ |
| **GFF/GTF** | Annotations (genes, exons) | **1-based, closed** $[\text{start},\text{end}]$ |
| bigWig / bedGraph | Signal tracks (coverage, ChIP) | bedGraph 0-based half-open |
| HDF5 / Zarr / AnnData (`.h5ad`) | Matrices (cells × genes), with annotations | Index-based |
| PDB / mmCIF | Atomic structures | Residue numbering varies; chain-specific |
| SMILES / SDF | Small molecules | — |

**The off-by-one rule.** A BED interval `chr1 100 200` covers 100 bases whose 1-based positions are $101,\dots,200$. The same region in GTF is `chr1 101 200`. Mixing the two silently shifts labels by one base. For 1 kb windows this is invisible; for base-resolution tasks (splice sites, transcription start sites, variant effects) it destroys the signal or, worse, produces a model that learns the *shifted* signal.

**Reference builds.** Human data exist on GRCh37/hg19, GRCh38/hg38, and the complete T2T-CHM13 assembly; coordinates differ between them. Mixing builds (e.g., variants from hg19, sequences from hg38) generates silently wrong inputs. Use `liftOver` for coordinate conversion, and record the build in every dataset's metadata.

**Strand and orientation.** DNA is double-stranded; a feature on the "−" strand is read from the reverse complement. A one-hot encoder that ignores strand scrambles motif orientation.

**The single most valuable sanity check** for any variant dataset: for every variant, verify that `REF` equals the reference genome base at `POS`. A mismatch rate above a fraction of a percent indicates a build mismatch, a coordinate error, or a normalization problem (left-alignment of indels). Similarly, for transcript/gene tasks, verify that annotated coding sequences start with `ATG` and end with a stop codon.

**Memory layout.** A one-hot encoding stores 4 floats per base; for 3 Gb that is 48 GB in float32, 12 GB in uint8. Store sequences as 2-bit or uint8 arrays and expand to one-hot *on the fly* in the data loader. Sparse matrices (CSR/CSC) are standard for cell × gene counts, which are 90–99% zeros.

---

## 6.4 The PyTorch research workflow

### 6.4.1 Tensors, autograd, modules

A `torch.Tensor` is an $n$-dimensional array with a device (CPU/GPU) and dtype; operations on tensors with `requires_grad=True` build a computation graph that `loss.backward()` traverses in reverse (Chapter 9). A model is an `nn.Module` whose parameters are leaf tensors; an optimizer updates them. The training loop is always

```text
for batch in loader:
    logits = model(batch.x)            # forward: builds the graph
    loss   = criterion(logits, batch.y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                     # backward: gradients for all parameters
    clip_grad_norm_(model.parameters(), max_norm)   # optional
    optimizer.step()
    scheduler.step()
```

### 6.4.2 A complete, tested skeleton

The following script (also in `code/ch06_training_skeleton.py`) trains a tiny convolutional model to detect a planted motif on either strand. It is deliberately small, and it contains the tests a careful researcher writes *before* trusting any model.

```python
--8<-- "code/ch06_training_skeleton.py"
```

Output (seed 0):

```text
shapes: (2000, 4, 100) (2000,)  params: 545
[test 1] overfit 32 examples: final loss = 0.0008  (should be near 0)
[test 2] held-out accuracy = 0.986
[test 3] max |f(x) - f(rc(x))|: plain = 11.9796, symmetrised = 0.00e+00
[test 4] shuffled-label control: held-out accuracy = 0.526  (should be ~0.5)
```

What each test establishes:

- **Test 1 (overfit one batch).** If a model cannot memorize 32 examples, the problem is a bug (wrong loss, no gradient flow, bad learning rate), not biology.
- **Test 2 (held-out accuracy).** The model has found the motif (98.6%). The 545-parameter model has 16 filters of width 8: the architecture's inductive bias (a motif detector applied everywhere, then a global max) *matches the data-generating process* (Chapter 10).
- **Test 3 (reverse-complement symmetry).** The plain model's outputs differ between a sequence and its reverse complement by up to 12 logits, although they describe the *same double-stranded molecule*. Wrapping the network in the symmetrizer $f_{\text{sym}}(x)=\tfrac12[f(x)+f(\mathrm{rc}(x))]$ makes it exactly invariant (error $0$). Without this, any strand-specific bias in training data would be silently baked in. **Invariances you know should hold are free unit tests.**
- **Test 4 (shuffled-label control).** With randomly permuted training labels, held-out accuracy is 52.6%, consistent with chance. If this had been high, the split or data pipeline would be leaking label information. *Every* biological ML pipeline should be run once with shuffled labels.

!!! tip "The five tests to run before any scientific claim"
    (1) overfit one batch; (2) shape assertions at module boundaries; (3) invariance/equivariance tests (reverse complement, permutation of cells, batch size independence); (4) shuffled-label control; (5) simple-baseline comparison (mean predictor, linear model on one-hot or $k$-mers, random-initialized model with the same readout).

### 6.4.3 Budgeting memory and compute before you launch

**Parameters and optimizer state.** In mixed-precision training with AdamW, each parameter typically costs: 2 bytes (bf16 weights) + 2 bytes (bf16 gradients) + 4 bytes (fp32 master weights) + 8 bytes (two fp32 Adam moments) $\approx$ **16 bytes per parameter** (12 if no separate low-precision copy is kept). A 1B-parameter model needs about 16 GB for states; a 40B-parameter model about 640 GB, which must be *sharded* across devices (ZeRO/FSDP) before a single activation is stored.

**Activations.** Memory for stored activations scales with batch × sequence length × width × depth; activation checkpointing trades roughly one extra forward pass (≈ +33% compute) for large memory savings. For attention, naive memory is $O(BHL^2)$; FlashAttention (Chapter 12) avoids materializing the $L\times L$ score matrix.

**Training FLOPs.** For a dense model with $N$ parameters trained on $D$ tokens, a forward pass costs about $2N$ FLOPs per token (one multiply and one add per weight), and the backward pass about twice that, so

$$
C\approx6\,N\,D\ \text{FLOPs}
$$

(ignoring the $L^2$ attention term and embedding costs). [[E]] as a standard approximation (Kaplan et al., 2020; Hoffmann et al., 2022). To convert to time: divide by (number of accelerators) × (peak throughput) × (model-FLOPs utilization, typically 0.3–0.5).

!!! example "Worked Research Example 6.1: Budget a genomic foundation-model run before you start"
    **Situation.** You want to train a 1-billion-parameter DNA language model on $10^{12}$ nucleotides (one token per nucleotide). A collaborator asks whether it can be done on 64 GPUs in a month, and whether the same budget could instead train a 40B-parameter model on $9\times10^{12}$ tokens, the scale reported for Evo 2.

    **Reasoning.**

    1. *Compute for the 1B model.* $C\approx6\times10^9\times10^{12}=6\times10^{21}$ FLOPs.
    2. *Throughput.* A current-generation datacenter GPU has a peak dense bf16 throughput of order $10^{15}$ FLOP/s; at 40% utilization that is $4\times10^{14}$ FLOP/s. So one GPU does $6\times10^{21}/4\times10^{14}=1.5\times10^7$ seconds, i.e., about 4,200 GPU-hours. On 64 GPUs: 65 hours, roughly three days. The month is ample.
    3. *Compute for a 40B model on $9\times10^{12}$ tokens.* $C\approx6\times4\times10^{10}\times9\times10^{12}=2.2\times10^{24}$ FLOPs: 360 times more. At $4\times10^{14}$ FLOP/s per GPU: $5.4\times10^9$ GPU-seconds $\approx1.5\times10^6$ GPU-hours, which on 64 GPUs would take about 2.7 years. It requires thousands of GPUs for months, which is why such models are built by consortia with major compute partners. (The approximation ignores the architecture-specific cost of long-context operators; consult the authors' reported numbers for actual figures.)
    4. *Memory.* The 40B model's optimizer state alone is $\approx640$ GB; 1B needs only $\approx16$ GB.
    5. *What can you learn from the cheap run?* A small run is not just a toy: if you train a *family* of sizes (10M, 30M, 100M, 300M, 1B) on matched data you can fit a scaling law (Chapter 17) and predict whether the larger model is worth building. A cheap run can also *falsify* an idea: if the representation from a 100M model gives no benefit on your downstream task over a one-hot baseline, a 100× larger run is unlikely to rescue it *unless* the failure is plausibly a scale effect, which you can then test by looking at the trend across sizes.

    **Expert analysis.** Always do the arithmetic first: it prevents requests that are off by orders of magnitude and it frames *what experiment the compute can buy*. The scientifically interesting answer is usually "a smaller family of models trained carefully answers the question for 1% of the cost."

---

## 6.5 Reproducibility and research hygiene

A result is reproducible if someone else (including yourself in six months) can regenerate it from code, data, and configuration. In practice, five habits cover most of the benefit.

1. **Version everything**: code (git), data (checksums or a data-versioning tool), configuration (one file per run, saved with the outputs), and environment (lockfile or container).
2. **Seed and record randomness.** Set seeds for the language, numerical libraries, and data-loader workers; log them. Know that GPU kernels may be non-deterministic unless explicitly configured, and that bitwise reproducibility is often unattainable; *statistical* reproducibility (same conclusion across seeds) is what you need. Run $\ge3$ seeds for any comparison where the claimed difference is small.
3. **Treat the data split as an artifact.** Save the split as a *manifest* (a file listing which examples belong to which partition and *why*: chromosome, homology cluster, donor, time). Never regenerate splits by re-running a script with a new random state. Check **overlap between train and test** explicitly: genomic intervals with `bedtools intersect`, protein sequences by clustering (MMseqs2, CD-HIT), molecules by scaffold, cells by donor.
4. **Never touch the test set while developing.** Tune on a validation set drawn the same way as the test set, and evaluate on the test set once. When the test set has been used many times, assume it has been overfit (Chapter 43).
5. **Log everything needed to interpret a curve**: loss per component, gradient norm, learning rate, data-order seed, wall-clock, git hash.

**Layered evaluation.** Build the evaluation harness *first*: it should run any model (including trivial ones) through the identical pipeline and emit the same metrics with confidence intervals. When comparing to a baseline in a paper, re-run the baseline in *your* harness; numbers copied from other papers rarely correspond to identical splits, preprocessing, or metric definitions.

---

## 6.6 Worked research example: a bug that looks like a discovery

!!! example "Worked Research Example 6.2: The model that predicts splice sites suspiciously well"
    **Situation.** A convolutional model is trained to classify whether the center of a 400-bp window is a splice donor site. The data pipeline builds windows from a GTF annotation (1-based, closed) and a genome FASTA (0-based indexing in the code). Held-out AUROC is 0.999. Looking at the model's top-activated positions, you find the window consistently contains `GT` *one base to the left of center* in positives.

    **Question.** Is the model superb or is something wrong?

    **Reasoning.**

    1. *Which facts are known a priori?* Splice donors in introns begin with `GT` (the canonical dinucleotide) in ~99% of cases. A model centered exactly on a donor should see `GT` at the two positions *after* the exon–intron boundary.
    2. *What does a `GT` one base left of center imply?* The window center is mis-registered by one base relative to the annotated boundary: a coordinate convention error between GTF (1-based inclusive) and Python slicing (0-based half-open).
    3. *Why is AUROC 0.999 anyway?* The model easily learns the consistent *shifted* pattern: whatever the registration, positives share a fixed offset pattern. The metric has no way to know the pattern is mis-registered. The error does **not** hurt this evaluation but does hurt anything that relies on exact base positions: variant effect predictions at a specific nucleotide, in silico mutagenesis maps, attribution.
    4. *What tests catch it?* (a) A **landmark assertion**: for all positives, assert that bases at positions $c,c+1$ (with $c$ the intended donor start) equal `GT`; the assertion fails for most. (b) Check the reference allele of known splice-disrupting variants. (c) Plot the mean nucleotide composition around labeled sites; the sequence logo should have the known consensus at the expected index.
    5. *What would the consequence have been?* The "discovery" would have propagated: attribution maps would be one base offset; predictions for variants at the canonical `GT` would appear to have no effect (since the model "reads" the wrong index), producing an incorrect conclusion that the model learned a weaker dependency than it did.

    **Expert analysis.** *A bug that preserves the metric is more dangerous than a bug that destroys it.* The mitigation is not carefulness in general but **invariant assertions at the data boundary** (known consensus sequences, known reference alleles, known start/stop codons) executed on every dataset build. The same pattern applies to protein numbering (chain offsets in PDB files versus UniProt numbering), to cell indexing after filtering (row/column misalignment between a count matrix and its metadata), and to gene-ID versions (Ensembl IDs with version suffixes).

---

## 6.7 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"It works too well\""
    **Observation.** A new model reaches AUROC 0.99 on a task where the field's best is 0.82.

    **Prior.** In my experience, results this far above the state of the art are more often *leaks, shortcuts, or bugs* than breakthroughs. This is not cynicism; it is a calibrated prior (Chapter 4: low prior odds, so even a "significant" result yields a modest posterior).

    **The checklist, in order of cost:**

    1. **Shuffled-label control.** Chance performance expected. (Minutes.)
    2. **Trivial-feature baselines.** Can GC content, sequence length, or a single metadata field (batch, cell line, source) predict the label? (Minutes.)
    3. **Remove the input you think the model uses and see if performance survives.** Mask the motif, mask the gene, or zero the embedding; a surviving 0.97 means it's using something else.
    4. **Duplicate and homology audit** between train and test (hash exact duplicates; cluster by sequence identity; check genomic overlap).
    5. **Split by the unit of generalization you claim** (chromosome, family, donor, lab) and re-evaluate. The gap *is* the shortcut component (Chapter 1).
    6. **Inspect the most confident errors and the most confident correct predictions.** Look at actual sequences.
    7. **Replicate in an independent dataset** produced by a different lab or technology.

    **Distinguishing bug from discovery.** Discoveries *survive* all of 1–7 and *shrink gracefully* (a plausible, smaller improvement under the hardest split); bugs and shortcuts collapse at one of them. Note which step collapsed it: that identifies *what the model was actually using*, which is itself informative (often a publishable observation about the benchmark).

---

## 6.8 Connections

- **Backward:** the cost model for matrix products (Chapter 2) gives $6ND$; the gradient check (Chapter 3) is a unit test; shuffled labels and baselines operationalize Chapter 4's priors; compression bits per base (Chapter 5) are the likelihood your sequence model reports.
- **Forward:** alignment and profile HMMs reappear in Chapter 27 and underlie MSAs used by AlphaFold-style models (Chapter 35). Message-passing on chains generalizes to graphical models (Chapter 8). Reverse-complement equivariance returns in Chapters 10 and 32 as an architectural property. Leakage and split manifests are formalized in Chapter 43. The $6ND$ budget returns in the scaling chapters (17, 47).

!!! takeaways "Key takeaways"
    1. **Dynamic programming** exploits optimal substructure: NW alignment in $O(nm)$; HMM forward/Viterbi in $O(LK^2)$. They are the *same algorithm in different semirings* ($\max,+$ vs. $\mathrm{logsumexp},+$).
    2. Coordinates are the leading source of silent genomic bugs: **BED is 0-based half-open; VCF and GTF are 1-based.** Always verify `REF` against the reference and landmarks (start codons, splice dinucleotides).
    3. Training compute is $\approx6ND$ FLOPs; optimizer state is $\approx16$ bytes per parameter. **Do the arithmetic first**; cheap model families can answer many questions.
    4. The five tests: overfit-one-batch, shape asserts, invariance tests, shuffled-label control, simple baselines. **Known invariances (reverse complement) are free unit tests.**
    5. Treat **splits as saved artifacts** with explicit overlap checks; do not touch the test set while developing.
    6. A bug that preserves the metric is more dangerous than one that destroys it. When a result is too good, work down the checklist from cheap to expensive.

---

## Further reading

- Needleman, S. B. & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *J. Mol. Biol.* 48, 443–453. Smith, T. F. & Waterman, M. S. (1981). Identification of common molecular subsequences. *J. Mol. Biol.* 147, 195–197. Gotoh, O. (1982). An improved algorithm for matching biological sequences. *J. Mol. Biol.* 162, 705–708.
- Durbin, R., Eddy, S., Krogh, A. & Mitchison, G. (1998). *Biological Sequence Analysis*. Cambridge University Press. Alignment, HMMs, and pair HMMs.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77, 257–286.
- Goodman, J. (1999). Semiring parsing. *Computational Linguistics* 25, 573–605.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. arXiv:2001.08361. Hoffmann, J. et al. (2022). Training compute-optimal large language models. arXiv:2203.15556.
- Rajbhandari, S. et al. (2020). ZeRO: Memory optimizations toward training trillion-parameter models. *SC20*.
- Wilson, G. et al. (2014). Best practices for scientific computing. *PLoS Biology* 12, e1001745. Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLoS Computational Biology* 13, e1005510.
- Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4, 100804.
- Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*.
