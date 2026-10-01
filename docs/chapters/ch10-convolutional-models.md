# Chapter 10. Convolutional Networks and Sequence Motifs

!!! abstract "Chapter at a glance"
    **Motivation.** Convolutional neural networks (CNNs) were the first deep models to succeed in regulatory genomics, and they remain the backbone of most sequence-to-function models (Chapter 31), including the early layers of Enformer, Borzoi, and AlphaGenome. Understanding *why* convolution fits DNA, what *grammar* a given CNN can and cannot express, and how far it can "see" is prerequisite to evaluating every such model.
    **Prerequisites.** Chapters 2, 3, 9.
    **You will be able to:** (1) define convolution/cross-correlation and compute output shapes, parameter counts, and receptive fields; (2) derive the gradients of a convolutional layer and prove translation equivariance; (3) explain how first-layer filters relate to PWMs and what determines whether filters learn whole motifs; (4) show by experiment which *syntax* (motif spacing) a given architecture can express; (5) build reverse-complement-equivariant layers; (6) reason about receptive fields against genomic distance scales.

---

## 10.1 The biological problem convolution solves

Transcription factors recognize short DNA words (6–20 bp) *wherever they occur*. A regulatory sequence is a long string in which such words appear at unknown positions, in either orientation, and in combinations with characteristic spacing. A model for this should:

1. **detect a pattern independent of where it occurs** (translation equivariance),
2. **be local at first** (a motif is short) and **compose** patterns into larger ones (grammar),
3. **share parameters** across positions, because training data per position are scarce but there are millions of positions,
4. **respect strand symmetry** (a motif on either strand is the same factor binding).

A **convolutional layer** has exactly properties 1–3 built in. Property 4 can be added by weight tying (§10.6). In 2015 three papers, **DeepBind** (Alipanahi et al.), **DeepSEA** (Zhou & Troyanskaya), and **Basset** (Kelley et al., 2016), showed that CNNs on raw one-hot DNA outperformed hand-engineered $k$-mer and motif methods for predicting TF binding and chromatin accessibility. [[E]]

---

## 10.2 The convolutional layer

### 10.2.1 Definition and shapes

Deep-learning "convolution" is, strictly, **cross-correlation**. For an input $\mathbf{x}\in\R^{C_\text{in}\times L}$ (channels × positions; for one-hot DNA, $C_\text{in}=4$) and a filter bank $\mathbf{w}\in\R^{C_\text{out}\times C_\text{in}\times k}$ (kernel width $k$), the output at channel $c'$ and position $t$ is

$$
y_{c',t}=b_{c'}+\sum_{c=1}^{C_\text{in}}\sum_{j=0}^{k-1}w_{c',c,j}\;x_{c,\,t+j},\qquad t=0,\dots,L-k.
$$

Each output channel is a *motif detector*: the dot product of a $C_\text{in}\times k$ window with a $C_\text{in}\times k$ template. A batch has shape `(B, C_in, L)` → `(B, C_out, L_out)`. With stride $s$, padding $p$, and dilation $d$,

$$
L_\text{out}=\Big\lfloor\frac{L+2p-d(k-1)-1}{s}+1\Big\rfloor .
$$

**Parameter count:** $C_\text{out}(C_\text{in}k+1)$, *independent of $L$*. A fully connected layer mapping $C_\text{in}L$ inputs to $C_\text{out}L_\text{out}$ outputs would need $C_\text{in}L\,C_\text{out}L_\text{out}$ weights, which is roughly $L^2$ times more. For a 1-kb sequence, 32 filters of width 9 use $32\cdot(4\cdot9+1)=1{,}184$ parameters instead of about $4{,}000\times32{,}000\approx1.3\times10^8$.

**Compute:** $2\,C_\text{out}\,C_\text{in}\,k\,L_\text{out}$ FLOPs per example, linear in $L$: *this linearity is the reason CNNs scale to megabase sequences and attention (quadratic) does not* (Chapter 12).

### 10.2.2 Equivariance and locality

Let $(S_\tau\mathbf{x})_{c,t}=x_{c,t-\tau}$ denote a shift by $\tau$. Away from the boundaries,
$$
\mathrm{conv}(S_\tau\mathbf{x})=S_\tau\,\mathrm{conv}(\mathbf{x}),
$$
since the same filter is applied at every position. *Proof.* $y'_{c',t}=b+\sum_{c,j}w_{c',c,j}x_{c,t+j-\tau}=y_{c',t-\tau}$. $\square$ Convolution is **translation-equivariant** (shifting the input shifts the output); *invariance* (output unchanged by a shift) is obtained by a subsequent pooling over positions. **Locality**: each output depends only on $k$ consecutive inputs.

### 10.2.3 Backpropagation through a convolution

Let $\delta_{c',t}=\partial\mathcal{L}/\partial y_{c',t}$. Then (the chain rule, summing over every output a weight or input touches):

$$
\frac{\partial\mathcal{L}}{\partial w_{c',c,j}}=\sum_t\delta_{c',t}\,x_{c,t+j},\qquad
\frac{\partial\mathcal{L}}{\partial b_{c'}}=\sum_t\delta_{c',t},\qquad
\frac{\partial\mathcal{L}}{\partial x_{c,s}}=\sum_{c'}\sum_{j}\delta_{c',s-j}\,w_{c',c,j}.
$$

The filter gradient is a *cross-correlation of the error signal with the input*: every position contributes evidence about the filter ("weight sharing means the filter is trained by all positions"). The input gradient is the error convolved with the filter (equivalently, a *transposed convolution*). This is Chapter 9's $\boldsymbol\delta\,\mathbf{a}^\top$ and $\mathbf{W}^\top\boldsymbol\delta$, with the weight matrix replaced by a structured (Toeplitz) matrix.

### 10.2.4 Pooling, stride, and multi-resolution

**Max pooling** (take the max over windows of size $m$) gives local *invariance* to small shifts and reduces length by $m$; **global max pooling** gives full positional invariance (a "bag of motifs": did this detector fire anywhere?). **Average pooling** counts occurrences. **Strides** and pooling reduce resolution; sequence-to-function models predict at coarse bins (128 bp for Basenji and Enformer; 1 bp for BPNet and AlphaGenome) and need up-sampling for base-resolution outputs (U-Net-style skip connections; Chapter 31).

---

## 10.3 Receptive fields: how far can the model see?

The **receptive field** of a unit is the set of input positions that can influence it. For a stack of layers with kernel $k_\ell$ and dilation $d_\ell$ (stride 1), the receptive field of the final layer is
$$
r=1+\sum_\ell(k_\ell-1)\,d_\ell .
$$
Pooling or striding by factor $s_\ell$ multiplies the "jump" between adjacent units, so later layers' kernels cover more input: with pooling by 2 after each of $n$ layers of kernel 3, the receptive field roughly doubles per layer. **Dilated convolutions** insert gaps ($d>1$) and, with dilations $1,2,4,\dots,2^{n-1}$ and $k=3$, give $r=2^{n+1}-1$: exponential growth with depth at constant parameters per layer. A 10-layer stack reaches 2,047 positions; Basenji's dilated residual tower reaches into the tens of kilobases.

**Theoretical versus effective receptive field.** The set of positions that *can* influence a unit is larger than the set that *effectively* does. The influence of input position $s$ on a unit is a sum over paths, which for stacked convolutions with positive weights resembles a repeated convolution: its profile tends to a Gaussian whose width grows only like $\sqrt{\text{depth}}$ (Luo et al., 2016) [[S]]. So a model with a theoretical receptive field of 100 kb may have an effective one much smaller. Residual connections and attention layers change this picture.

**Receptive field versus biology.** Regulatory interactions span scales:

| Scale | Typical distance | Example |
|---|---|---|
| Motif | 6–20 bp | TF binding site |
| Motif syntax | 5–50 bp | Cooperative pairs; nucleosome-scale periodicity (~10 bp helical, ~150–200 bp nucleosome repeat) |
| Promoter | 0.1–2 kb | Core promoter plus proximal elements |
| Enhancer–gene | 10–100+ kb (often tens of kb; some > 1 Mb) | Distal enhancers; contact via looping |
| Topologically associating domain | ~0.1–1 Mb (median several hundred kb) | Insulated regulatory neighborhoods |
| Chromosome compartment | Mb | A/B compartments |

A model with a 1-kb receptive field *cannot* use a distal enhancer 50 kb away, no matter how much data it has. This is the origin of the long-range push: 131 kb (Basenji), 196 kb (Enformer), 524 kb (Borzoi), and 1 Mb (AlphaGenome) input windows (Chapter 31).

---

## 10.4 First-layer filters, PWMs, and whether filters learn whole motifs

A first-layer filter $\mathbf{w}\in\R^{4\times k}$ is a log-odds-like template. For a PWM with probabilities $p_{j,a}$ against a background $q_a$, the log-odds score of window $\mathbf{x}$ is $\sum_j\sum_a x_{j,a}\log(p_{j,a}/q_a)$: *exactly* a convolution with $w_{a,j}=\log(p_{j,a}/q_a)$ followed by a threshold. A convolutional filter is therefore a *generalized PWM*, and it can be trained end-to-end rather than estimated from aligned sites (Chapter 29).

**But filters need not learn whole motifs.** Koo & Eddy (2019) showed that whether first-layer filters look like *complete* motifs or *fragments* depends on architecture: shallow networks with large first-layer filters and aggressive max pooling learn whole motifs (they must, since the later layers cannot reassemble parts); deeper networks with small filters or little pooling distribute motifs across filters and *assemble* them in later layers, so first-layer filters look like partial or noisy motifs even when the network has learned the right thing. [[S]] Consequence: **visually inspecting first-layer filters is not a reliable way to compare what models have learned**; use attribution on predictions (Chapter 18) or motif-discovery on attributions (TF-MoDISco).

In our experiment (below), the two-layer network's first-layer filters reach normalized correlation $0.82$ with each planted motif, imperfect partly because 32 filters of width 9 share the load.

---

## 10.5 What *grammar* can an architecture express? An experiment

Regulatory "grammar" includes *syntax*: spacing and order of motifs. Consider a synthetic grammar: a sequence is positive iff motif A (`TGACTCA`) is followed by motif B (`CACGTG`) at a gap of 8–12 bp. Negatives are: A and B present but ≥30 bp apart, A alone, B alone, or neither. We train two CNNs on 8,000 sequences (length 120):

- **M1**: conv(4→32, $k=9$) → ReLU → **global max pool** → linear. Receptive field 9 bp.
- **M2**: conv(4→32, $k=9$) → ReLU → conv(32→32, $k=25$) → ReLU → global max pool → linear. Receptive field 33 bp.

```python
--8<-- "code/ch10_cnn_grammar.py"
```

Output:

```text
receptive fields: M1 = 9 bp, M2 = 33 bp; a positive spans 7+gap+6 = 21-25 bp, a 'far' negative spans >= 43 bp
M1 (1 conv layer)    AUROC all test = 0.819; AUROC positives vs 'far' negatives only = 0.502; params = 1217
M2 (2 conv layers)   AUROC all test = 0.999; AUROC positives vs 'far' negatives only = 0.999; params = 26849
first-layer filter vs planted motif correlation: A = 0.82, B = 0.82
RC-tied conv layer: max |f(rc(x)) - T(f(x))| = 2.4e-07  (equivariant)
```

**Reading the result.** M1 reaches AUROC 0.82 overall because it can detect *whether* each motif is present, and this separates positives from "only A", "only B", and "neither". But it scores **0.502 (chance)** when asked to distinguish positives from "far" negatives, because every difference between those two classes is in the *spacing*, which M1 *cannot represent*. Formally, M1's output is $g\big(\max_th_1(x_t),\dots,\max_th_F(x_t)\big)$, a function of the per-filter maxima only; any two sequences with the same maxima at different positions receive identical outputs. M1 is a **bag-of-motifs model**. M2's second layer sees, at each position, the *joint* pattern of first-layer activations within a 33-bp window, so "A then B within a short gap" is a single feature; its AUROC is 0.999 on both evaluations.

!!! lens "Research lens: architecture as a hypothesis about grammar"
    **Assumes (M1):** the regulatory output is a function of *which* motifs are present, not of their arrangement. **Assumes (M2):** relevant combinations occur within the second layer's receptive field. **Information ignored:** M1 ignores all positional information; M2 ignores arrangements beyond 33 bp. **Failure modes:** a benchmark on which motif presence suffices (many MPRAs, many ChIP-seq peak classification tasks) *cannot reveal* whether a model has learned syntax; both M1 and M2 would score well on "positive vs. neither". **Design consequence:** to claim a model has learned *syntax*, the evaluation must contain examples where *only* the syntax differs: the "far negatives" of this experiment.

!!! example "Worked Research Example 10.1: A paper claims its CNN \"has learned regulatory grammar\" because it classifies enhancers vs. random sequence with AUROC 0.95"
    **Situation.** An enhancer-classification CNN is trained on 50,000 putative enhancers versus GC-matched random genomic sequences. The authors show motif logos from first-layer filters and conclude that the model "learned the grammar of enhancers."

    **Question.** What does the evidence support? What experiment would test the claim?

    **Reasoning.**

    1. *What is the grammar claim?* That the model uses *arrangement* (spacing, order, orientation, co-occurrence) of motifs, beyond motif presence.
    2. *Does the benchmark test that?* No: from §10.5, a bag-of-motifs model separates enhancers from random sequence on presence alone. Matching GC content removes one shortcut (composition) but not motif presence.
    3. *Is the first-layer logo evidence of grammar?* No. It shows that motif detectors exist (and, per Koo & Eddy, may be fragments). It says nothing about how detections are combined.
    4. *What alternative explanations produce AUROC 0.95?* (i) Bag-of-motifs; (ii) residual composition biases (CpG content, repeat content, nucleosome-positioning signals); (iii) leakage among related enhancers (paralogous or overlapping elements across splits).
    5. *Discriminating experiments.* (a) *Synthetic syntax tests:* insert a pair of motifs into a neutral background at varied spacing/orientation and measure the model's response curve; a model with syntax shows structured dependence on spacing (e.g., periodicities, cooperativity), a bag-of-motifs model a spacing-independent response. (b) *Shuffle-preserving controls:* shuffle motif order or positions while preserving motif content: does the score drop? (c) *Compare to a bag-of-motifs baseline* (e.g., a linear model on motif counts, or M1): the *excess* over this baseline is the evidence for syntax. (d) *Prospective:* test predicted syntax with a reporter assay (MPRA) on designed constructs (Chapter 46).
    6. *Predictions.* If the model has learned syntax, (a) shows spacing dependence that is *reproduced in the wet lab*; if not, the response is flat and (c) shows no excess.

    **Expert analysis.** The claim is at best rung C1 of the Claim Ladder (in-distribution prediction) and the grammar interpretation is [[X]]. This is the CNN version of a general pattern: *a capability claim requires a benchmark in which only that capability differs.*

---

## 10.6 Reverse-complement symmetry

DNA is double-stranded; a binding site on the reverse strand reads as the reverse complement. Define $\mathrm{rc}(\mathbf{x})$ by reversing positions and swapping $A\leftrightarrow T$, $C\leftrightarrow G$. For a conv layer to be **RC-equivariant**, with $f(\mathrm{rc}(\mathbf{x}))=\mathcal{T}f(\mathbf{x})$ for a fixed channel/position permutation $\mathcal{T}$, it suffices to **tie filters in pairs**: for every filter $\mathbf{w}$ include $\mathrm{rc}(\mathbf{w})$ as a second filter. Then applying $\mathrm{rc}(\mathbf{w})$ to $\mathbf{x}$ is the same as applying $\mathbf{w}$ to $\mathrm{rc}(\mathbf{x})$, up to reversal of positions. The code verifies it: the maximum deviation is $2.4\times10^{-7}$ (float32 precision). Three options exist, in increasing order of strictness:

1. **Data augmentation:** randomly feed reverse complements; the model *learns* approximate symmetry.
2. **Test-time averaging / symmetrization:** $\tfrac12[f(\mathbf{x})+f(\mathrm{rc}(\mathbf{x}))]$ (Chapter 6): exact invariance for the output.
3. **Equivariant architectures:** *reverse-complement parameter sharing* (Shrikumar et al., 2017) as above; **Caduceus** (Schiff et al., 2024) builds RC-equivariance into a bidirectional state-space model (Chapter 32).

When *strand-specific* function matters (transcription direction, strand-specific RNA features), invariance is wrong: the model must distinguish strand. The symmetry to impose depends on the *biology of the target*: another instance of attack A9 (biological constraint as inductive bias).

---

## 10.7 A short history of convolutional models of the genome

!!! paper "Paper dissection: DeepSEA (Zhou & Troyanskaya, *Nature Methods*, 2015)"
    **Problem.** Predict the effect of non-coding sequence variants. Most disease-associated variants lie outside genes, and the regulatory code was not understood.

    **Key insight.** Train a model to predict, from 1,000 bp of sequence, a large panel of chromatin features measured by ENCODE and Roadmap Epigenomics, then use the model to score *how a single-nucleotide change alters those predictions* (a variant-effect score by in silico mutagenesis).

    **Architecture.** Three convolutional layers (320, 480, 960 filters) with pooling and dropout, followed by a fully connected layer and sigmoid outputs: a deeper stack than DeepBind's single layer, allowing combinations of motifs.

    **Objective.** Multi-task binary cross-entropy over **919 chromatin features** (DNase-I hypersensitivity in 125 cell types, 690 TF-binding profiles for 160 TFs, and 104 histone-mark profiles), with the label for each 200-bp bin from peak calls.

    **Data.** ENCODE and Roadmap Epigenomics ChIP-seq and DNase-seq; held-out chromosomes for validation and testing.

    **Evaluation.** AUC per feature on held-out chromosomes (the authors reported high median AUCs for TF binding and DNase hypersensitivity, roughly 0.96 and 0.92, and lower for histone marks, about 0.86); variant prioritization on eQTLs, GWAS SNPs, and ClinVar variants, using the predicted chromatin effects as features for a classifier.

    **Why it worked.** Multi-task training shares features across 919 tasks (a form of data augmentation across assays); convolutions encode motif structure; the output space (chromatin) has abundant, noisy-but-informative labels; held-out chromosome evaluation is a reasonable (if imperfect) generalization test.

    **Assumptions.** Reference sequence determines chromatin features (cell-type context is represented only by the choice of output head); local sequence (1 kb) suffices; peak-calling labels are accurate; variants act by altering the same features that predict chromatin (not, for example, by changing RNA or protein function).

    **Limitations.** 1-kb context (no distal effects); labels are binary peaks (no quantitative signal); trained on a single reference genome so it never sees allelic variation; variant-effect "validation" is mostly retrospective enrichment.

    **What followed.** Basset and Basenji (quantitative signals; 131 kb context with dilated convolutions); ExPecto (predicting expression with spatial transformation of DeepSEA predictions); BPNet (base-resolution profiles and syntax discovery); Enformer (attention for 196-kb context); Borzoi (524 kb, RNA-seq coverage); AlphaGenome (1 Mb, multimodal, base-pair resolution). Each step relaxed an assumption above.

    **What remains unresolved.** Whether such models capture *causal*, *allele-specific* cis-regulatory effects in personal genomes (Chapters 7, 31, 41, 50).

!!! paper "Paper dissection: BPNet (Avsec et al., *Nature Genetics*, 2021)"
    **Problem.** *Interpret* what a sequence-to-function model has learned about TF binding syntax.
    **Key insight.** Predict **base-resolution ChIP-nexus profiles** (a high-resolution TF-footprinting assay) instead of peak-level labels, so that the shape of the signal around each motif reveals binding; then interpret the model by attribution and motif mining (DeepLIFT + TF-MoDISco).
    **Architecture.** A 1-kb input; a stack of dilated residual convolutions; separate heads for the profile shape (multinomial) and total counts (Poisson/regression).
    **Result.** Models of Oct4, Sox2, Nanog, and Klf4 in mouse embryonic stem cells learned the motifs and *syntax rules*: cooperative spacing between Oct4–Sox2 and periodic (~10.5 bp) Nanog preferences. Selected predicted rules were tested with CRISPR-edited and reporter experiments. [[S]]
    **Why it matters.** It showed that a CNN can discover *testable* syntax hypotheses, and that the *form of the label* (base-resolution profile) matters: richer supervision enabled richer interpretation, an instance of the data attack (A4).
    **Limitations.** One cell type and four factors; the model's rules are hypotheses until validated; attribution methods have known failure modes (Chapter 18).

---

## 10.8 Beyond one-dimensional sequences

- **2D CNNs** on contact maps (Hi-C), on pairwise residue-feature maps (the early contact-prediction networks, RaptorX and AlphaFold 1; Chapter 35), and on images (Cell Painting, histology).
- **3D CNNs** on voxelized density maps (cryo-EM map interpretation, docking-pocket detection).
- **Hybrid** CNN–transformer models: the CNN "stem" reduces length and extracts local features, a transformer models long-range interactions among the reduced tokens (Enformer; Chapter 31). This division of labor respects the compute table of Chapter 2: convolution is linear in length, attention quadratic.

---

## 10.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: a receptive-field audit for a regulatory-variant model"
    **Setting.** You are evaluating a published CNN for predicting expression from sequence. Before reading any result, do an audit.

    1. **Compute the receptive field** from the architecture (kernels, dilations, pooling). *Also estimate the effective receptive field* by measuring the gradient of one output unit with respect to all input positions in a trained model; plot it.
    2. **Overlay biology.** What fraction of the gene's known enhancers (from CRISPRi screens, Hi-C, or eQTLs) lies *inside* the effective receptive field? If the strongest distal regulator is 80 kb away and the model sees ±10 kb, the model *cannot* represent it.
    3. **Identify what the architecture cannot express.** Global max pooling → no syntax; no cell-type input → no cell-type specificity beyond head selection; reference training → no allelic variation; a fixed bin size → no sub-bin effects.
    4. **Predict failure modes.** Variants in distal enhancers should be mis-scored (near zero effect); variants affecting mutation-sensitive motifs should be scored well. *State your predictions before looking at results.*
    5. **Design a test.** From fine-mapped eQTLs or MPRA or CRISPRi data, bin causal variants by distance to the transcription start site and compute performance per bin. If performance decays beyond the effective receptive field, the audit is confirmed. If performance remains high at long distance, either the effective field is larger than estimated or the model exploits shortcuts correlated with distance (e.g., gene density).

    **Distinguishing limitation types.** A *receptive-field limitation* is *architectural*: more data won't fix it. A *context-specificity limitation* is an *input* limitation. A *syntax limitation* is an *inductive-bias* limitation. Each calls for a different intervention (longer context; cell-type conditioning; a second layer).

---

## 10.10 Worked research example on the inference gap

!!! example "Worked Research Example 10.2: In silico mutagenesis says a motif is crucial; the CRISPR deletion has no effect"
    **Situation.** A CNN predicts expression of gene $G$ in a cell line. In silico mutagenesis (ISM; mutate each base and see the change in predicted expression) identifies a 12-bp motif 5 kb upstream as the most important element, with a predicted 3-fold effect. Deleting that element by CRISPR in the same cell line changes expression by less than 10%.

    **Question.** Why might the model and the experiment disagree? What should you do?

    **Reasoning.**

    1. *What does ISM measure?* A *local derivative of the model*: how the output changes under a point mutation in the *current context*. It reflects what the model has learned, not necessarily what the cell does.
    2. *Candidate explanations.* (H1) **Redundancy:** a shadow enhancer or a second motif compensates in the cell; the model, trained on observational data in which motif strength correlates with expression, attributes the whole effect to this motif. (H2) **Confounding:** the motif covaries with the *true* driver (e.g., an adjacent element or CpG-island structure), and the model uses it as a proxy (G-I). (H3) **Context mismatch:** the cell line used for the experiment lacks a chromatin or trans-factor state that the model's head assumes. (H4) **Saturation/nonlinearity:** ISM's local effect does not reflect the larger perturbation of deleting the element (the model's response is nonlinear). (H5) **Technical:** the deletion removed only the motif but left flanking sequence that contains the true effect, or expression was measured at a different time point.
    3. *What experiments distinguish them?* (a) Test the model on *combinatorial* in silico perturbations (delete motif plus candidate compensating elements); if the model's predicted effect vanishes when the second element is present, H1 is a model-consistent explanation. (b) Delete or mutate the motif in *several* cell types; correlate effect with model-predicted effect. (c) Use a tiled CRISPRi/CRISPRa screen across the locus (a "saturation mutagenesis" of the region) and compare to attribution maps. (d) Check whether the motif's predicted effect is explained by a *correlate*: reproduce with a model trained without the motif's flank or with decorrelated training sequences.
    4. *Predictions.* H1: the double-perturbation CRISPR (motif + shadow element) shows the large effect. H2: perturbing the true driver, not the motif, changes expression. H4: a smaller, graded perturbation (point mutations) reproduces a smaller effect than ISM predicts.

    **Expert analysis.** This is the *inference gap* (G-I): attributions summarize an *associative* model, and association does not license a causal reading of a single perturbation. Chapter 18 (attribution methods), Chapter 44 (causal identification), and Chapter 46 (designing the perturbation experiment) develop each step. The point to take away now: **model-derived hypotheses are cheap and wet-lab tests are the arbiter**; an expert reads a disagreement not as "the model failed" but as a *hypothesis-discriminating* result.

---

## 10.11 Connections

- **Backward:** convolution is a structured linear layer (Chapter 2); its gradients follow Chapter 9's backprop; global max pooling is a $\max$-semiring operation (Chapter 6); PWMs are Chapter 5's information-theoretic motifs.
- **Forward:** long-range alternatives to CNNs: state-space models and long convolutions (Chapter 11), attention (Chapter 12), hybrids (Chapter 31); interpretation of filters and attributions (Chapters 18, 48); sequence-to-function models in full (Chapter 31); classical motif models (Chapter 29); RC-equivariance in genomic language models (Chapter 32).

!!! takeaways "Key takeaways"
    1. A convolutional layer is a **bank of motif detectors applied at every position**: translation-equivariant, local, parameter-sharing, cost linear in length.
    2. Parameters: $C_\text{out}(C_\text{in}k+1)$; receptive field: $1+\sum(k_\ell-1)d_\ell$; filter gradients are cross-correlations of the error with the input.
    3. First-layer filters are generalized PWMs, **but** deeper or less-pooled networks learn *partial* motifs; do not judge learning from filter logos.
    4. **Architecture is a hypothesis about grammar:** global max pooling after one layer gives a bag-of-motifs model that scores 0.502 on spacing-only discrimination, while two layers score 0.999. A grammar claim needs a benchmark where *only* the grammar differs.
    5. **Reverse-complement symmetry** can be built in by filter pairing (verified to $2\times10^{-7}$), but only if the target is strand-invariant.
    6. Receptive fields must be compared with biological distance scales: a 1-kb model cannot use an enhancer 50 kb away, whatever the data.
    7. Attribution on an associative model (ISM) is a hypothesis about causality, not evidence of it.

---

## Further reading

- Alipanahi, B., Delong, A., Weirauch, M. T. & Frey, B. J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology* 33, 831–838.
- Zhou, J. & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning–based sequence model. *Nature Methods* 12, 931–934.
- Kelley, D. R., Snoek, J. & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. *Genome Research* 26, 990–999. Kelley, D. R. et al. (2018). Sequential regulatory activity prediction across chromosomes with convolutional neural networks. *Genome Research* 28, 739–750.
- Avsec, Ž. et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nature Genetics* 53, 354–366.
- Koo, P. K. & Eddy, S. R. (2019). Representation learning of genomic sequence motifs with convolutional neural networks. *PLoS Computational Biology* 15, e1007560.
- Luo, W., Li, Y., Urtasun, R. & Zemel, R. (2016). Understanding the effective receptive field in deep convolutional neural networks. *NeurIPS*.
- Shrikumar, A., Greenside, P. & Kundaje, A. (2017). Reverse-complement parameter sharing improves deep learning models for genomics. *bioRxiv*. Zhou, H., Shrikumar, A. & Kundaje, A. (2022). Towards a better understanding of reverse-complement equivariance for deep learning models in genomics. *MLCB*.
- Schiff, Y. et al. (2024). Caduceus: bi-directional equivariant long-range DNA sequence modeling. *ICML*.
- Eraslan, G., Avsec, Ž., Gagneur, J. & Theis, F. J. (2019). Deep learning: new computational modelling techniques for genomics. *Nature Reviews Genetics* 20, 389–403.
- Dumoulin, V. & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv:1603.07285.
