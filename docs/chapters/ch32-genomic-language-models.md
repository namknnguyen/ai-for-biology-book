# Chapter 32. Genomic Language Models: What Does Next-Base Prediction Learn?

!!! abstract "Chapter at a glance"
    **Motivation.** If a language model trained to predict the next word learns grammar, facts, and reasoning, a model trained to predict the next nucleotide on trillions of bases might learn gene structure, regulatory grammar, and the effects of mutations, without a single label. Since 2021 (DNABERT, Nucleotide Transformer, GPN, HyenaDNA, Caduceus, Evo, Evo 2) this has been tested at growing scale. The record is genuinely mixed: strong results for coding variants and for evolutionary-aware models, weak or no advantage over simple supervised baselines on most regulatory tasks, and striking generative demonstrations whose evidence is still thin. This chapter makes the question precise (what does a likelihood measure in a genome?), supplies the architectures and the objective, runs a controlled small-scale experiment on a real genome that separates *local statistics* from *memorization* from *function*, dissects Evo 2, and organizes the evidence by task so that you can tell which claims to trust.
    **Prerequisites.** Chapters 5, 11, 12, 13, 17, 20, 28, 29, 31, 43.
    **You will be able to:** (1) state what a genomic LM's log-likelihood and likelihood ratio mean and what confounds them (composition, repeats, mutation bias); (2) compare masked and causal objectives, tokenizations, and long-context architectures for DNA; (3) separate memorization from generalization with a repeat-copy control; (4) read zero-shot variant-effect results critically; (5) summarize the evidence for and against DNA LMs in regulatory genomics; (6) design a fair comparison between a pretrained DNA model and a supervised baseline.

---

## 32.0 What a likelihood means in a genome

A genomic language model assigns a probability $p_\theta(x_{1:L})$ to a DNA sequence. Written autoregressively,

$$
\log p_\theta(x_{1:L})=\sum_{t=1}^{L}\log p_\theta(x_t\mid x_{<t}),\qquad \text{cross-entropy per base}=-\tfrac1L\log_2 p_\theta(x_{1:L})\ \text{bits}.
$$

Two properties of DNA make this unlike natural language (Chapters 5 and 28):

1. **Most of a genome is close to incompressible by local context.** Non-repetitive DNA has an entropy of about 1.9 bits per base under good local models (the maximum is 2), so a model's room for improvement over a simple Markov chain is a few hundredths of a bit; the signal for function lives in that margin.
2. **A likelihood confounds several things**: base composition (GC content, CpG depletion), repeats (a model that has seen a sequence assigns it high probability), mutational biases (Chapter 20), conservation by selection (what we want), and the idiosyncrasies of the reference assembly. A variant's likelihood *ratio* $\log p(x^\text{alt})-\log p(x^\text{ref})$ inherits every one.

!!! lens "Research lens: what information does a DNA LM use and ignore?"
    *Uses*: the training sequences, hence whatever statistical structure is *predictable from left context* (local composition, codon periodicity, repeats, motif co-occurrence). *Ignores*: cell type, developmental time, the environment, population variation (frequencies of alleles in a population), the other strand unless built in (reverse-complement symmetry, Chapter 28), and anything that varies across cells but not across the reference sequence. The most important omission is the last: *regulatory function is a property of sequence in a cellular context*, and a sequence-only likelihood does not know the context (Chapter 31).

---

## 32.1 Objectives, tokens, and architectures

**Objectives.** *Masked* language modeling (DNABERT, Nucleotide Transformer, GPN) predicts masked bases from both sides, which suits variant effect prediction (a bidirectional context around the variant). *Causal* modeling (HyenaDNA, Evo, Evo 2) predicts left to right, which supports generation and likelihoods of whole sequences. Bidirectional causal training for DNA with reverse-complement symmetry is implemented in Caduceus, which uses a bidirectional Mamba block and RC-equivariance by construction.

**Tokenization** (Chapter 28). $k$-mer tokens (DNABERT), learned byte-pair encodings, and single-nucleotide tokens (HyenaDNA, Evo, Evo 2, Caduceus). Single-nucleotide tokens preserve resolution for variants and avoid frame-dependent segmentation, at the cost of sequence length: a megabase of context is $10^6$ tokens, and attention cost $O(L^2)$ rules out dense attention (Chapter 12); hence the use of *sub-quadratic* operators.

**Long-context architectures.** HyenaDNA (Nguyen et al., NeurIPS 2023) replaces attention with long implicit convolutions (Chapter 11) and reached single-nucleotide context lengths of 1 million tokens with far fewer parameters than contemporary transformers on the Nucleotide Transformer benchmarks (about 1.6 million vs 2.5 billion), reporting up to 160-fold faster training than a FlashAttention transformer at the longest lengths. **Evo** (Nguyen et al., *Science* 2024) used StripedHyena (alternating attention and data-controlled convolutions), 7 billion parameters, a 131,072-token context, and about 300 billion prokaryotic nucleotides. **Evo 2** (§32.4) generalizes the recipe to 40 billion parameters and more than 9 trillion nucleotides.

**Complexity and memory.** A convolution over $L$ tokens by FFT costs $O(L\log L)$ time; attention costs $O(L^2)$ time and (with FlashAttention) $O(L)$ memory. At $L=10^6$ the difference is $10^6$-fold in the number of pairwise interactions, which is why a megabase context is in practice only available to sub-quadratic or hybrid models.

```python
# causal DNA LM: shapes.  B batch, T length (<= context), d width, V = 4 bases (+1 start token)
def forward(x):                                  # x: (B, T) integer bases, start token prepended and last base dropped
    h = emb(x) + pos(arange(T))                  # (B, T, d)
    for blk in blocks: h = blk(h, causal_mask)   # self-attention (B, H, T, T) or a long convolution
    return out(ln(h))                            # (B, T, 4): logits for the NEXT base at every position
loss = cross_entropy(forward(x_in).reshape(-1, 4), x.reshape(-1))     # mean over B*T, converted to bits by / ln 2
```

---

## 32.2 A controlled experiment on a real genome

The question of this section is deliberately modest: at a scale we can run, does a transformer trained on DNA learn anything beyond the local $k$-mer statistics that a Markov chain already has, and does it assign higher likelihood to sequences it has memorized? The data are the Arabidopsis chloroplast genome (NC_000932; 154,478 bp; the genome used in Chapters 28, 29, 33). The *training* set is the first 60 kb and the sequence between the two inverted repeats (104 kb in total). Two test sets are held out: a **unique region** (60,000–84,170) and the **reverse-complement copy of a repeat that is in the training set** (IRa, 128,214–154,478, the reverse complement of IRb, which is in training; Chapter 28). A Markov chain of high order can memorize the repeat; a model that has *generalized* should do about as well on the unique region as on a chain of the right order, and *memorization* appears as a large gap between the two columns.

The model is a causal transformer (3 layers, width 64, 4 heads, context 128, 158,852 parameters, single-nucleotide tokens) trained for 2,500 steps with reverse-complement augmentation; the baselines are Markov chains of order 2, 4, 6, 8 (also trained on the reverse complement of the training data). For the variant experiment, genes completely inside the held-out unique region supply every single-nucleotide variant of the coding sequence, labelled synonymous, missense, or stop-gain by the genetic code (bacterial table 11); each variant is scored by the *drop in log-likelihood* of a 97-base window that contains it (the Markov chain and the transformer score the same window).

```python
--8<-- "code/ch32_dna_lm.py"
```

```text
training bases: 104,044; test unique region 24,170 bp; test repeat copy 26,264 bp (reverse complement of a training region)

== 1. Held-out cross-entropy (bits per base) ==
model                                   unique test region    repeat copy (RC of a training region)
order-2 Markov, RC augmentation             1.897                 1.973
order-4 Markov, RC augmentation             1.887                 1.961
order-6 Markov, RC augmentation             1.921                 1.900
order-8 Markov, RC augmentation             2.072                 1.465

transformer: 158,852 parameters, context 128, single-nucleotide tokens
trained 2500 steps in 1289 s; final training loss 1.910 bits/base
transformer LM, RC augmentation                1.883                 1.940

== 2. Zero-shot variant scoring: log-likelihood change of a 96-base window, genes inside the held-out unique region ==
26,460 single-nucleotide variants in held-out genes: synonymous 5,938, missense 19,236, stop-gain 1,286
scoring a stratified subsample of 6,286 variants
AUROC for ranking a variant class above synonymous variants by the drop in log-likelihood (0.5 = no signal)
class vs synonymous        order-4 Markov (no frame)    transformer LM
missense                      0.460                     0.476
stop-gain                     0.423                     0.397
(reference: a rule that knows the genetic code separates stop-gain from synonymous with AUROC 1.000 by construction)
```

**Reading Part 1 (likelihoods).**

1. **Local statistics are nearly everything.** On the unique region the order-4 chain reaches 1.887 bits per base and the transformer 1.883. The difference (0.004 bits) is smaller than the spread among Markov orders 2 to 6 (0.034) and we did not estimate its sampling error: treat the two as indistinguishable. At this scale the transformer *has not learned anything a 4-mer chain does not know*. (The final training loss, 1.910, is a batch average over all 128 positions of a window including the first few that have almost no left context, whereas the held-out numbers score every base with at least 64 bases of context; the two are not comparable.)
2. **Memorization is large and invisible to the unique-region number.** On the reverse-complement copy of a training region the order-8 chain achieves 1.465 bits: it has stored 8-mers and their reverse complements and effectively *looks up* the repeat, while on the unique region the same chain is the worst model (2.072: overfitting, since 65,536 contexts are estimated from 208,000 events). Higher order is a *memorization dial*. The transformer shows almost no memorization (1.940 on the copy), which says only that at 2,500 steps and 159k parameters it has too little capacity or training to store a 26-kb sequence.
3. **Two numbers, not one.** A single held-out perplexity averaged over a genome mixes repeats and unique DNA in proportions that depend on the genome; every genomic LM comparison should report a *repeat-controlled* number (as here) or filter test sequences by similarity to the training set (Chapter 43; Chapter 45).

**Reading Part 2 (variant scoring).** If the likelihood drop measured selective constraint, then missense and especially stop-gain variants would cause a larger drop than synonymous ones, and the AUROC would exceed 0.5. It is **below 0.5**: 0.460 (Markov) and 0.476 (transformer) for missense, and 0.423 and 0.397 for stop-gain. The follow-up script tests the obvious explanation: that the likelihood is dominated by *base composition*. The chloroplast genome is 64.5% A+T, and stop codons are created by changes *toward* A and T.

```python
--8<-- "code/ch32b_composition_check.py"
```

```text
training base composition: A 0.322, C 0.180, G 0.175, T 0.323  (A+T = 0.645)

class         variants   alt is A/T   change toward A/T (G/C -> A/T)   change away from A/T (A/T -> G/C)   mean order-0 drop in log2-likelihood
synonymous       5938      0.386             0.192                             0.560                          0.316
missense        19236      0.455             0.253                             0.393                          0.119
stop-gain        1286      0.817             0.422                             0.151                         -0.231

AUROC of the order-0 (composition-only) drop in log-likelihood for ranking a class above synonymous variants:
  missense   0.415
  stop-gain  0.308
```

Composition explains it. 82% of stop-gain variants have an A or T alternative allele (against 39% of synonymous variants); a model that knows nothing but base frequencies already assigns smaller likelihood drops, or even increases, to stop-gain variants (mean drop $-0.23$ bits against $+0.32$ for synonymous), and ranks them *below* synonymous variants with an AUROC of 0.31. The order-4 chain (0.42) and the transformer (0.40) do better than composition alone (0.31), so they have learned something beyond base frequencies, but not enough to overcome it.

This is the same trap as Chapter 20 (a frequency-based score confounding mutation bias with selection) in a new coat, and it is why *zero-shot likelihood ratios must be calibrated against a composition-only null* and why the field's successful variant-effect models are *evolutionary* (they train on many species so that the likelihood reflects what is *conserved*, not what is *common*; §32.3–32.4).

!!! lens "Research lens: assumptions of this experiment"
    A tiny model (the learning curve is far from saturated), a single small genome with 85 genes, one species (so no evolutionary information beyond the genome), left-context-only scoring, the drop in a 97-base window as the score, and a protein-coding variant task in which *selection is mostly on the amino acid*, which a nucleotide model that does not know the genetic code cannot easily represent. Each assumption can change the sign of the finding. Nothing here says that a 40-billion-parameter model trained on a hundred thousand species behaves the same. It says what a *controlled test* of "does the likelihood measure function?" looks like, and what a negative control must include: a composition-only null.

---

## 32.3 Zero-shot variant effects: when does the likelihood ratio work?

The zero-shot score of a variant is a log-likelihood ratio (LLR),

$$
\text{LLR}(v)=\log p_\theta(x^{\text{alt}})-\log p_\theta(x^{\text{ref}}).
$$

For a model trained on *many sequences from related organisms* it approximates a log-odds of the alternative allele being tolerated *given the evolutionary context*. Formally, if the training distribution is a mixture of selected families, $p_\theta(x)\approx\sum_f\pi_f\,p_f(x)$ with $p_f$ a Potts-like energy-based model (Chapter 29), then a strongly constrained position has low probability for alternatives under every $p_f$ and a large negative LLR. The approximation breaks when (i) the model has memorized sequences instead of learning constraints, (ii) the context is too short to include the relevant covarying site, (iii) the position is neutral in the training species and constrained in the species of interest, or (iv) base composition or mutational biases dominate (§32.2).

**What makes the LLR meaningful in practice.** (a) *Evolutionary breadth*: multi-species alignments (GPN-MSA, Benegas et al., *Nature Biotechnology* 2025, which feeds an alignment of 100 vertebrate genomes to a masked-language-model objective and reports outperforming CADD and the Nucleotide Transformer on several deleteriousness benchmarks while training in a few hours) or training on tens of thousands of species (Evo 2). (b) *Evaluation designed against confounds*: matched allele frequency, matched composition, held-out genes, and comparison with an explicit conservation baseline (phyloP, phastCons). A zero-shot LLR that does not beat a conservation score computed from the same species is not adding what the language model is claimed to add.

!!! rhyme "Structural rhyme: DNA-LM log-likelihood ratio ↔ the protein-LM score of Chapter 34 ↔ a Potts-model $\Delta E$ (Chapter 29)"
    For a protein family the LLR of a mutation under a masked LM approximates the Potts energy difference $\Delta E$ of the family (Chapter 34's experiment makes the correspondence explicit in a toy). For DNA the training distribution is not one family but a whole genome of heterogeneous elements, so the "family" is implicit and the statistical power per position is lower; coding sequences, which have family-like structure (orthologs across species), are the part of the genome where the correspondence is best.

---

## 32.4 The frontier: Evo 2

!!! paper "Paper dissection: Evo 2 (Brixi et al., *Nature* 2026; bioRxiv February 2025)"
    **Problem.** A single model of DNA across all domains of life that can *score* variants and *generate* sequences, at the scale of a bacterial genome or a mammalian locus.
    **Insight.** Treat the genome as a language and scale: a 40-billion-parameter (and a 7B) model trained at single-nucleotide resolution on a curated atlas with more than 9 trillion nucleotides from over 100,000 species, with a context window of 1 million bases, and with the whole stack (data, code, weights) openly released.
    **Architecture.** *StripedHyena 2*, a convolutional multi-hybrid combining three operator types: short explicit convolutions, medium regularized convolutions, and long implicit (Hyena) convolutions, interleaved with attention. Pretraining at a context of 8,192 followed by a *midtraining* phase extending to 1 million.
    **Objective.** Autoregressive next-nucleotide prediction (cross-entropy); no labels.
    **Data.** A curated genomic atlas across bacteria, archaea, eukaryotes, and bacteriophages; the authors excluded the genomes of eukaryotic viruses from training as a biosecurity measure, and report that the model performs poorly on such sequences [[S]].
    **Evaluation.** (i) Zero-shot likelihood-based variant-effect prediction: coding and non-coding human variants (including BRCA1 and BRCA2 variants, ClinVar), and deep mutational scans of proteins and non-coding RNAs; (ii) *mechanistic interpretability*: sparse autoencoders (Chapter 18, Chapter 48) trained on internal activations find features that align with exon–intron boundaries, transcription-factor motifs, protein secondary structure, and prophage regions, learned without annotations; (iii) *generation*: sequences at the scale of mitochondrial and bacterial genomes and yeast chromosomes that are scored as plausible by annotation and structure-prediction tools; (iv) *guided design*: inference-time search against predictors of chromatin accessibility to generate sequences with a prescribed accessibility pattern.
    **Results (as reported).** Zero-shot performance on coding variants is strong and, on BRCA1 non-coding SNVs, higher than the other models compared; a supervised classifier trained on Evo 2's embeddings reaches AUROC of about 0.94–0.95 on held-out BRCA1 SNVs [[S]].
    **Why it worked.** Scale; evolutionary breadth (many species make the likelihood reflect conservation rather than composition); long context for non-local dependencies; the use of embeddings as features for supervised heads.
    **Assumptions.** Reference sequences alone carry enough information about constraint; the evaluation sets (ClinVar, deep mutational scans) are not leaked into pretraining sequences in a way that explains the result (the human genome is, of course, in the training data, and the evaluation involves variants *of* that genome).
    **Limitations.** Zero-shot scores for non-coding variants remain modest in absolute terms and below the best supervised sequence-to-function models for tissue-specific regulatory effects (Chapter 31; AlphaGenome, Avsec et al., *Nature* 2026, takes the supervised route with base-pair resolution predictions up to 1 Mb for expression, splicing, accessibility, binding, and contacts); the likelihood is not a measure of function in a cellular context; training on reference sequence ignores population variation; the generated sequences are plausible by computational criteria, which are not experimental ones; biosecurity.
    **What followed.** Evo-designed bacteriophage genomes (King et al., *Science* 2026), SAE-based interpretation, and a range of downstream embeddings.
    **Unresolved.** Whether further scaling improves *regulatory* variant prediction, which depends on cell-type context the model does not see; and whether the SAE features are causally used by the model (Chapter 48).

---

## 32.5 The evidence ledger: what DNA LMs do and do not deliver

| Task | Evidence for pretrained DNA LMs | Evidence against / caveats | Grade |
|---|---|---|---|
| Coding variant pathogenicity (zero-shot) | Evo 2, GPN-MSA, and earlier protein-LM-like scores compete with specialized predictors | Largely recovers conservation; gains over conservation scores are modest | [[S]] |
| Non-coding variant pathogenicity | GPN-MSA beats CADD on several benchmarks; Evo 2 leads on BRCA1 non-coding | Benchmarks are small and noisy; matched controls often absent | [[P]] |
| Regulatory activity (cell-type-specific) | Fine-tuned or probed LMs improve over one-hot linear models | **Highly tuned supervised models trained from scratch on one-hot sequence were competitive with or better than pretrained DNA LMs across the datasets tested** (Tang and Koo, *Genome Biology* 2025); **DART-Eval** (Patel et al., NeurIPS 2024 Datasets & Benchmarks) found that annotation-agnostic DNA LMs showed inconsistent performance and no compelling gains for most regulatory tasks, with simpler models often better, especially for counterfactual variant prediction | [[S]] for "no consistent advantage" |
| Representation of biological features (motifs, exons) | SAE features in Evo 2; probes in several models | Probing power depends on the probe (Chapter 18); correlation of features with annotations is not causal use | [[P]] |
| Generation of functional genomes | Evo phage genomes: about 300 synthesized, 16 viable (King et al., *Science* 2026) | A 5% hit rate on a small, well-studied genome (phiX174 template); the viable ones may sit close to natural phages | [[S]] for existence, [[H]] for generality |
| Long-range regulatory effects | Context length up to 1 Mb | Gains from context beyond tens of kb are not consistently shown; sequence-to-function models (Chapter 31) show the same difficulty with distal enhancers | [[H]] |

**Why the regulatory evidence is weak (hypotheses, graded [[P]]).** (i) *Information density*: the next-base objective rewards modeling the 95% of the genome that is repeats, composition, and neutral sequence, and the regulatory grammar is a small part of the loss. (ii) *Context*: regulatory function depends on the cell type and the chromatin state, which a sequence-only model cannot know; a supervised model sees labels from the context of interest. (iii) *Evolutionary signal for non-coding DNA is weak*: regulatory elements turn over rapidly, so multi-species training provides less constraint than for coding genes. (iv) *Benchmark design*: many tasks give a baseline of one-hot supervised models heavy tuning (as in Tang and Koo) and the LM a light probe, or the reverse. Chapter 43's guidance for fair comparisons applies: tune both, report the budget, test on held-out loci or cell types.

!!! openproblem "Open problem: an objective that rewards regulatory function"
    Next-nucleotide prediction spends most of its capacity on what is easy to predict. **Diagnostic questions:** Could one *reweight* the loss toward positions that are conserved across species, or include a contrastive objective between orthologous regulatory sequences with different cell-type usage? Could pretraining include *population variation* (alleles and frequencies; Chapter 21) as data, so that the model learns the distribution of tolerated variation rather than a single reference? What is the smallest cell-type-labelled data set that would turn a DNA LM into a regulatory predictor competitive with supervised models, and does the answer depend on pretraining scale (a *scaling law for transfer*, Chapter 47)?

---

## 32.6 Generative design and biosecurity

The generative demonstrations raise a question that the likelihood does not answer: *is a generated sequence functional?* King et al. (*Science*, 6 August 2026; first posted September 2025) fine-tuned Evo models on bacteriophage genomes related to phiX174, generated candidate genomes, synthesized about 300, and found 16 that produced viable phage with a range of fitness, some with novel sequence content, with an experimental pipeline that is the real result (a computational filter, synthesis, and a functional assay). A 5% success rate is high for de novo genome design and low for a design pipeline in a field with a functional assay that takes a day; the relevant comparison is to *mutating natural phage genomes* with the same assay, which defines the baseline for novelty and fitness.

*Dual use.* A model that writes viable viral genomes is the kind of tool whose misuse has high consequence. Mitigations in the open literature include excluding the training sequences of human pathogens (Evo 2's exclusion of eukaryotic viruses and the reported degradation in performance on them, §32.4), restricting capabilities by data, red-teaming generated sequences against pathogen databases, and sequence-synthesis screening (Chapter 59). Evidence from protein design that screening can be defeated by generative tools (Wittmann et al., *Science* 2025, on AI-reformulated toxins evading biosecurity screening software, with patches deployed afterward) shows that a mitigation must itself be evaluated adversarially.

---

## 32.7 Worked research examples

!!! example "Worked Research Example 32.1: Is a zero-shot genomic-LM score evidence of regulatory understanding?"
    **Situation.** A paper reports that a new DNA LM's zero-shot LLR separates pathogenic non-coding ClinVar variants from benign ones with AUROC 0.82, "demonstrating that the model has learned regulatory grammar".

    **Question.** What controls would decide whether the claim holds?

    **Reasoning (Expert Chain).**

    1. **L1–L3 Problem and assumptions.** The claim equates *discrimination of pathogenic from benign non-coding variants* with *understanding of regulatory grammar*. ClinVar non-coding pathogenic variants are heavily enriched for splice-region and promoter-proximal variants near coding exons, and for variants in a few genes, and benign ones come from a different ascertainment process (population studies).
    2. **L4–L5 Failure modes.** (a) *Distance to exon*: a feature that predicts pathogenicity without any regulatory model. (b) *Composition and CpG content* (§32.2). (c) *Gene-level confounding*: variants in constrained genes. (d) *Leakage*: the benign variants are common in gnomAD, and models trained on the reference genome have seen the reference allele; the label is correlated with allele frequency, which correlates with mutability.
    3. **L6 Bottleneck.** The benchmark's *power to discriminate* mechanisms. An AUROC is not a mechanism.
    4. **L10 Experiments.** (a) Compare with *simple baselines computed from the same data*: distance to the nearest exon, phyloP, CADD without the LM feature, a composition-only LLR; require the LM to add value *conditional on* them (a nested model comparison, §43). (b) *Stratify* by variant class (splice-region, promoter, deep intronic, UTR) and report each. (c) *Match* benign and pathogenic variants for distance to exon, allele frequency, and trinucleotide context. (d) Test on *experimental* data with a known mechanism, such as saturation-mutagenesis MPRAs of a promoter, where ground truth does not depend on clinical ascertainment.
    5. **L11 Interpretation.** If the LM adds nothing beyond distance-to-exon and conservation, then the claim is falsified for this evidence. If it adds in splice-region variants only, the model has learned something about splicing signals (valuable and specific), not about regulatory grammar generally.

    **Expert analysis.** The AUROC is a summary of the *confounds* as much as of the model. The decisive evidence is a *mechanistically controlled* experiment, which is what Chapter 31's MPRA remedy provides.

!!! example "Worked Research Example 32.2: Using a DNA LM to design a regulatory element, with no known answer"
    **Situation.** You want a 200-bp enhancer that drives expression in cell type A but not in cell type B. A DNA LM can generate candidate sequences conditioned on a prompt (for example the flanking sequence of a known enhancer) and score them; a supervised sequence-to-function model (Chapter 31) can predict activity in both cell types. Nothing is known about whether such sequences exist.

    **Question.** How would you generate, select, and validate designs, and how would you know whether the LM helped?

    **Reasoning.**

    1. **Set up arms.** (a) LM-generated and filtered by the supervised predictor; (b) random mutagenesis of known enhancers filtered by the same predictor; (c) predictor-guided optimization alone (gradient ascent or evolutionary search against the predictor); (d) known natural enhancers as positive controls and shuffled sequences as negatives. The *same* predictor filters the arms, so the comparison isolates the *generator*.
    2. **Beware optimization against the surrogate** (Chapter 36): predictor-guided search finds sequences the predictor loves and the cell does not. Hold out an *independent* predictor (a different architecture, different training data) as a second opinion, and report the gap between the filter predictor's score and the independent predictor's score as a function of search depth.
    3. **Assay.** An MPRA in both cell types with at least 50 designs per arm, 10 barcodes each, and the noise ceiling from replicate barcodes (Chapter 25); pre-specify success as activity above the 90th percentile of natural enhancers in A and below the median in B.
    4. **Analysis.** Compare arms by the *fraction meeting the success criterion* with a binomial interval, and by sequence diversity (edit distance to the nearest natural enhancer), since success by copying a natural enhancer is not design.
    5. **Decision.** If the LM arm exceeds the random-mutagenesis arm by a pre-specified margin at equal predictor score, the LM supplies a useful prior over functional sequence; if not, the supervised predictor is doing the work.

    **What is not known.** Whether cell-type-selective enhancers of arbitrary specificity exist at this length, and whether a language model's prior contributes beyond that of the predictor; the answers will vary by cell-type pair and are what the field's design experiments are for.

---

## 32.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: evaluating a genomic LM claim"
    1. **Check for a composition-only null** (order-0 or order-2 Markov) in every zero-shot variant analysis.
    2. **Report repeat-controlled likelihoods** (unique versus repeat-copy test sets) and the identity-to-training distribution of the test set.
    3. **Insist on baselines at equal tuning budget**: a supervised one-hot CNN trained from scratch, a conservation score, and a probe on the LM at several layers.
    4. **Stratify by variant class** and by distance to the nearest exon; match controls on allele frequency and trinucleotide context.
    5. **Use experiments with known mechanism** (saturation mutagenesis, MPRAs) in addition to clinical labels.
    6. **Ask what the model never saw**: cell type, population variation, the other strand (unless built in), species not in training.
    7. **Reproduce at small scale**: run `code/ch32_dna_lm.py` on another genome (a bacterium, a mitochondrion) and see what changes with GC content and repeat structure.

    **What it teaches.** A likelihood is a number about a *distribution of sequences*; to say it measures function you must remove every other thing it measures.

    **An open question to carry forward.** If a language model's variant score mostly recovers conservation, then its value is in *positions where it disagrees with conservation scores computed from the same alignments*. Could one find a set of such disagreements (positions where Evo 2's LLR is large and phyloP is small, or vice versa), test a sample by saturation mutagenesis, and estimate which of the two is right more often, and in which genomic context? That would measure the *marginal information* of the language model over alignment-based conservation, which is the quantity the zero-shot literature has not yet isolated.

---

## 32.9 Connections

- **Backward:** entropy and compression (Chapter 5); long-range sequence operators (Chapter 11); representation learning (Chapter 13); scaling (Chapter 17); interpretability with SAEs (Chapter 18); mutation bias versus selection (Chapter 20); DNA representations (Chapter 28); Potts models and coevolution (Chapter 29); sequence-to-function (Chapter 31); benchmark design (Chapter 43).
- **Forward:** protein LMs and the same LLR logic (Chapter 34); design and surrogate exploitation (Chapter 36); interpretability of biological models (Chapter 48); open problems in genomes (Chapter 50).

!!! takeaways "Key takeaways"
    1. A genomic LM's likelihood measures *predictability under the training distribution*: composition, repeats, and mutational bias as well as constraint; a likelihood ratio inherits all of them.
    2. In a real-genome experiment, a 159k-parameter transformer matched a 4th-order Markov chain on unique DNA (1.883 vs 1.887 bits per base) and showed almost no memorization of a repeat copy (1.940), while an 8th-order chain *memorized* it (1.465) and failed on unique DNA (2.072): report repeat-controlled likelihoods.
    3. Zero-shot scoring in the same experiment *failed*: AUROC for stop-gain versus synonymous variants was 0.40 (transformer) and 0.42 (Markov), because composition alone gives 0.31 (82% of stop-gain variants move toward A/T). Calibrate against a composition-only null.
    4. What makes the likelihood ratio informative is **evolutionary breadth** (GPN-MSA, Evo 2) and evaluation designed against confounds.
    5. Evo 2 (40B parameters, more than 9 trillion nucleotides, 1 Mb context, open) predicts coding and some non-coding variant effects zero-shot, with interpretable internal features, and is a major step [[S]]; its zero-shot advantage for tissue-specific regulatory effects is not established.
    6. For regulatory genomics, tuned supervised one-hot models are competitive with or better than pretrained DNA LMs (Tang and Koo 2025), and DART-Eval finds no compelling gains for most tasks [[S]].
    7. Generative genome design is real but its success rates are modest (16 viable of about 300 phage genomes) and need baselines; biosecurity mitigations need adversarial evaluation.

---

## Further reading

- Ji, Y., Zhou, Z., Liu, H. & Davuluri, R. V. (2021). DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. *Bioinformatics* 37, 2112–2120. Dalla-Torre, H. et al. (2025). Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* 22, 287–297. Benegas, G., Batra, S. S. & Song, Y. S. (2023). DNA language models are powerful predictors of genome-wide variant effects. *PNAS* 120, e2311219120 (GPN).
- Benegas, G., Albors, C., Aw, A. J., Ye, C. & Song, Y. S. (2025). A DNA language model based on multispecies alignment predicts the effects of genome-wide variants. *Nat. Biotechnol.* (GPN-MSA). Nguyen, E. et al. (2023). HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. *NeurIPS.* Schiff, Y. et al. (2024). Caduceus: bi-directional equivariant long-range DNA sequence modeling. *ICML.*
- Nguyen, E. et al. (2024). Sequence modeling and design from molecular to genome scale with Evo. *Science* 386, eado9336. Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361. King, S. H. et al. (2026). Generative design of novel bacteriophages with genome language models. *Science* (6 August 2026).
- Tang, Z., Somia, N., Yu, Y. & Koo, P. K. (2025). Evaluating the representational power of pre-trained DNA language models for regulatory genomics. *Genome Biol.* (2025; first posted as a 2024 bioRxiv preprint). Patel, A. et al. (2024). DART-Eval: a comprehensive DNA language model evaluation benchmark on regulatory DNA. *NeurIPS Datasets and Benchmarks.* Avsec, Ž. et al. (2026). AlphaGenome. *Nature* (28 January 2026; preprint title: "AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model").
- Wittmann, B. J. et al. (2025). Strengthening nucleic acid biosecurity screening against generative protein design tools. *Science* (2025).
- Pollard, K. S., Hubisz, M. J., Rosenbloom, K. R. & Siepel, A. (2010). Detection of nonneutral substitution rates on mammalian phylogenies (phyloP). *Genome Res.* 20, 110–121. Rentzsch, P., Witten, D., Cooper, G. M., Shendure, J. & Kircher, M. (2019). CADD: predicting the deleteriousness of variants throughout the human genome. *Nucleic Acids Res.* 47, D886–D894.
