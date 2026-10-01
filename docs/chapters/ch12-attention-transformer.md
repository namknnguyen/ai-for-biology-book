# Chapter 12. Attention and the Transformer

!!! abstract "Chapter at a glance"
    **Motivation.** The Transformer is the substrate of protein language models, genomic language models, single-cell foundation models, AlphaFold's Evoformer, and the Pairformer/diffusion modules of AlphaFold 3. Reading these papers *means* reading attention. This chapter derives attention from first principles, with every tensor shape and cost, and then asks what attention represents for biological sequences.
    **Prerequisites.** Chapters 2, 3, 9; Chapters 10–11 for comparison.
    **You will be able to:** (1) derive scaled dot-product attention and explain the $1/\sqrt{d_k}$ factor; (2) write multi-head attention with tensor shapes and compute its parameters and FLOPs; (3) explain why positional information must be added and derive rotary embeddings; (4) derive the online-softmax trick behind FlashAttention; (5) relate attention to kernel regression, Hopfield networks, and Potts models; (6) critique claims that attention maps "reveal" biology.


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M2](m02-functions-exponentials-logs.md) (softmax and log-sum-exp) and [M5](m05-linear-algebra-1.md) (matrix products and shapes).

---

## 12.1 Why attention, for biology

Three biological facts motivate attention:

1. **Pairwise dependence between distant positions.** In a protein, residues far apart in sequence contact in space and coevolve; in a regulatory region, an enhancer motif influences a distant promoter. A model that can directly relate *any pair of positions* matches these structures.
2. **Set-valued and unordered data.** A cell's expressed genes, a family of homologous sequences, an ensemble of atoms are better described as sets or graphs than as ordered lists. Attention is natively set-based.
3. **Variable relevance.** Which residues matter for predicting position $i$ depends on the *content* (which residues they are). Attention *routes information by content*.

CNNs relate positions only within a receptive field and with fixed weights; RNNs relate them through a bottleneck; SSMs through fixed or selectively modulated dynamics. Attention computes all $L^2$ pairwise interactions *directly*, at a cost.

---

## 12.2 Scaled dot-product attention

### 12.2.1 Attention as soft dictionary lookup

A dictionary lookup takes a **query**, compares it to **keys**, and returns the **value** of the matching entry. *Soft* lookup compares the query to *all* keys and returns a weighted average of the values, with weights reflecting similarity. Given a query $\mathbf{q}\in\R^{d_k}$, keys $\mathbf{k}_j\in\R^{d_k}$, and values $\mathbf{v}_j\in\R^{d_v}$ for $j=1,\dots,L$:

$$
\alpha_j=\frac{\exp\big(\mathbf{q}^\top\mathbf{k}_j/\sqrt{d_k}\big)}{\sum_{j'}\exp\big(\mathbf{q}^\top\mathbf{k}_{j'}/\sqrt{d_k}\big)},\qquad
\text{output}=\sum_{j=1}^L\alpha_j\,\mathbf{v}_j .
$$

The weights are non-negative and sum to 1 ($\boldsymbol\alpha=\softmax(\text{scores})$), so the output is a **convex combination of the values**. In matrix form with $\mathbf{Q}\in\R^{L_q\times d_k}$, $\mathbf{K}\in\R^{L\times d_k}$, $\mathbf{V}\in\R^{L\times d_v}$:

$$
\boxed{\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})=\softmax\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V}}\qquad\text{(softmax over each row).}
$$

In **self-attention** all three come from the same sequence $\mathbf{X}\in\R^{L\times d}$ by learned linear maps: $\mathbf{Q}=\mathbf{X}\mathbf{W}_Q$, $\mathbf{K}=\mathbf{X}\mathbf{W}_K$, $\mathbf{V}=\mathbf{X}\mathbf{W}_V$. In **cross-attention** the queries come from one sequence and the keys/values from another (conditioning).

**Why separate queries, keys, and values?** The query–key *similarity* decides *where to look*; the value carries *what to retrieve*. Decoupling them lets the model look up by one criterion (e.g., "is this residue a hydrophobic core position?") and retrieve different information.

### 12.2.2 Why divide by $\sqrt{d_k}$

If the entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with zero mean and unit variance, then $\mathbf{q}^\top\mathbf{k}=\sum_{i=1}^{d_k}q_ik_i$ has mean $0$ and variance $d_k$ (a sum of $d_k$ independent terms, each with variance $\E[q_i^2]\E[k_i^2]=1$). Without scaling, the logit scale grows like $\sqrt{d_k}$, driving the softmax into saturation (one weight near 1, the rest near 0), where its gradient vanishes (Chapter 3: $\partial p/\partial z=p(1-p)\to0$). Dividing by $\sqrt{d_k}$ restores unit variance. The code confirms: for $d_k=16,64,256$ the variance of $\mathbf{q}^\top\mathbf{k}$ is 16.0, 63.9, 252.0, and after scaling 1.00, 1.00, 0.98. For contrast, logits with standard deviation 8 give a softmax whose maximum weight is 0.878: nearly one-hot, with vanishing gradients.

### 12.2.3 Masks

*Causal (autoregressive) mask:* set scores for $j>i$ to $-\infty$ so position $i$ attends only to earlier positions (GPT-style generation; the "next-token" objective of Chapter 13). *Padding mask:* ignore padded positions. *Bidirectional* (no mask): masked-language-model encoders (BERT, ESM). Masks are added to the scores *before* the softmax.

### 12.2.4 Attention as kernel regression

The attention output for a query is

$$
\sum_j\frac{\kappa(\mathbf{q},\mathbf{k}_j)}{\sum_{j'}\kappa(\mathbf{q},\mathbf{k}_{j'})}\mathbf{v}_j,\qquad\kappa(\mathbf{q},\mathbf{k})=e^{\mathbf{q}^\top\mathbf{k}/\sqrt{d_k}},
$$

which is the **Nadaraya–Watson kernel regression** estimator with an exponential-dot-product kernel: *a locally weighted average of values*. This classical interpretation (Chapter 7's kernel methods) explains attention's strengths and weaknesses: it is a *non-parametric memory lookup*, great at retrieval, with no built-in notion of order or arithmetic.

---

## 12.3 Multi-head attention

A single attention map can express only one "relation" per position. **Multi-head attention** runs $H$ attentions in parallel on lower-dimensional projections:

$$
\mathrm{head}_h=\mathrm{Attn}(\mathbf{X}\mathbf{W}_Q^{(h)},\mathbf{X}\mathbf{W}_K^{(h)},\mathbf{X}\mathbf{W}_V^{(h)}),\qquad
\mathrm{MHA}(\mathbf{X})=[\mathrm{head}_1;\dots;\mathrm{head}_H]\,\mathbf{W}_O ,
$$

with $\mathbf{W}_Q^{(h)},\mathbf{W}_K^{(h)},\mathbf{W}_V^{(h)}\in\R^{d\times d_h}$, $d_h=d/H$, $\mathbf{W}_O\in\R^{d\times d}$. In practice all heads are computed by *one* linear projection to width $d$ followed by a reshape:

| Step | Operation | Shape |
|---|---|---|
| Input | $\mathbf{X}$ | $(B,L,d)$ |
| Project | $\mathbf{Q}=\mathbf{X}\mathbf{W}_Q$ (same for $\mathbf{K},\mathbf{V}$) | $(B,L,d)$ |
| Split heads | reshape $(B,L,H,d_h)\to$ transpose | $(B,H,L,d_h)$ |
| Scores | $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}$ | $(B,H,L,L)$ |
| Weights | softmax over last axis | $(B,H,L,L)$ |
| Mix values | weights $\times\mathbf{V}$ | $(B,H,L,d_h)$ |
| Merge heads | transpose, reshape | $(B,L,d)$ |
| Output | $\cdot\,\mathbf{W}_O$ | $(B,L,d)$ |

The number of parameters is $4d^2$ **independent of $H$** (the heads share the budget). Per-head rank: each head's score matrix has rank $\le d_h$ before the softmax (Chapter 2), so splitting into heads trades per-head expressivity for diversity.

**Variants for efficient inference.** *Multi-query attention* (MQA) shares one key/value projection across all heads; *grouped-query attention* (GQA) shares across groups of heads. They shrink the key–value cache (below) by $H\times$ or $H/G\times$ with small quality loss.

The code implements this exactly and verifies it against `torch.nn.functional.scaled_dot_product_attention` (max difference $1.2\times10^{-7}$).

---

## 12.4 Positional information

### 12.4.1 Attention is permutation-equivariant

Permute the input rows by $\mathbf{P}$. Then $\mathbf{Q}\to\mathbf{P}\mathbf{Q}$, $\mathbf{K}\to\mathbf{P}\mathbf{K}$, $\mathbf{V}\to\mathbf{P}\mathbf{V}$, the score matrix becomes $\mathbf{P}\mathbf{S}\mathbf{P}^\top$, the row-wise softmax commutes with the permutation of columns, and the output is $\mathbf{P}\,\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})$. **Without positional information, self-attention treats its input as a set**: shuffling the sequence just shuffles the output (verified: $8.9\times10^{-8}$). This is exactly what we want for sets (a cell's genes) and exactly what we do *not* want for sequences, where order carries the information (the reading frame, motif spacing). So positions must be injected.

### 12.4.2 Encodings

- **Sinusoidal** (Vaswani et al., 2017): add $\mathrm{PE}_{p,2i}=\sin(p/10000^{2i/d})$, $\mathrm{PE}_{p,2i+1}=\cos(p/10000^{2i/d})$ to the input. Different dimensions oscillate at different wavelengths; the encoding of $p+\Delta$ is a fixed linear function of the encoding of $p$ (a rotation), so relative offsets are *representable*.
- **Learned absolute** embeddings: a free vector per position (cannot extrapolate beyond the training length).
- **Relative position biases** (T5; ALiBi, Press et al., 2022): add a learned or fixed function of $j-i$ directly to the attention scores. Enformer uses relative positional features built from exponential, gamma, and central-mask basis functions (Chapter 31).
- **Rotary position embedding (RoPE)** (Su et al., 2021): rotate query and key vectors by a position-dependent angle.

**Derivation of RoPE.** Take a 2-D slice of a query/key and let $\mathbf{R}(\phi)$ be the rotation by angle $\phi$. Define $\tilde{\mathbf{q}}_m=\mathbf{R}(m\theta)\mathbf{q}_m$ and $\tilde{\mathbf{k}}_n=\mathbf{R}(n\theta)\mathbf{k}_n$. Since rotations are orthogonal and compose by adding angles ($\mathbf{R}(a)^\top\mathbf{R}(b)=\mathbf{R}(b-a)$),

$$
\tilde{\mathbf{q}}_m^\top\tilde{\mathbf{k}}_n=\mathbf{q}_m^\top\mathbf{R}(m\theta)^\top\mathbf{R}(n\theta)\mathbf{k}_n=\mathbf{q}_m^\top\mathbf{R}\big((n-m)\theta\big)\mathbf{k}_n .
$$

The score depends on **absolute positions only through their difference $n-m$**: a purely relative encoding implemented without extra parameters. A $d_h$-dimensional head is split into $d_h/2$ planes with frequencies $\theta_i=10000^{-2i/d_h}$, covering many wavelengths. The code confirms $\mathbf{q}_m\cdot\mathbf{k}_{m+7}$ is identical for $m=0,5,100,1000$ (4.4367 each, up to float32 rounding). RoPE is standard in modern protein and DNA transformers; extending context beyond the training length is done by rescaling the angles (position interpolation, NTK/YaRN scaling).

!!! lens "Research lens: positional encoding in biology"
    **Absolute vs. relative.** Should a DNA model know the absolute genomic coordinate? *No*: absolute coordinates identify chromosomes and loci, enabling memorization (a shortcut) and preventing transfer. *Relative* position (distance between two tokens; distance to the transcription start site) is the biologically meaningful variable. **Set-like biology.** For single-cell models where tokens are genes (Chapter 38), gene *identity* embeddings replace positions; some models impose an order by expression rank (Geneformer), which makes the "sequence" an artifact of the representation. **Periodicity.** DNA has ~10.5 bp helical periodicity and ~150–200 bp nucleosome spacing; sinusoidal/rotary encodings can represent periodicities, but whether models *use* them is an empirical question (Chapter 18).

---

## 12.5 The Transformer block, parameters, and FLOPs

A (pre-LN) **Transformer block** applies, with $\mathbf{x}\in\R^{L\times d}$:

$$
\mathbf{x}\leftarrow\mathbf{x}+\mathrm{MHA}(\mathrm{LN}(\mathbf{x})),\qquad
\mathbf{x}\leftarrow\mathbf{x}+\mathrm{MLP}(\mathrm{LN}(\mathbf{x})),
$$

with $\mathrm{MLP}(\mathbf{x})=\mathbf{W}_2\,\mathrm{GELU}(\mathbf{W}_1\mathbf{x})$, $\mathbf{W}_1\in\R^{4d\times d}$, $\mathbf{W}_2\in\R^{d\times4d}$ (modern LLMs use a gated SwiGLU with a different multiplier). The two sub-layers are residual (Chapter 9). Stack $N_\text{layers}$ blocks between an embedding layer and an output head.

**Parameters per layer:** attention $4d^2$ + MLP $8d^2$ = $\mathbf{12d^2}$ (plus $O(d)$ biases and norms). The code counts 789,760 for $d=256$ versus $12d^2=786{,}432$, and 12,596,224 for $d=1024$ versus 12,582,912. A model with $N_\text{layers}=32$ and $d=4096$ has $\approx12\times32\times4096^2\approx6.4\times10^9$ parameters (plus embeddings).

**Forward FLOPs per token per layer:** the projections and MLP perform $2\times12d^2=24d^2$ FLOPs (2 FLOPs per parameter); the score and value mixing perform $2\times2\,Ld=4Ld$ ($\mathbf{Q}\mathbf{K}^\top$ and weights$\times\mathbf{V}$, each $2Ld$ per token). So

$$
\text{FLOPs per token per layer}\approx24d^2+4Ld .
$$

Attention's $L$-dependent term dominates when $L>6d$. At $d=1024$, that is $L>6{,}144$. At the context lengths of genomic models ($L\sim10^5$–$10^6$), the attention term dominates by orders of magnitude (the table in Chapter 11 §11.6).

**KV cache.** During autoregressive generation one stores the keys and values of all previous tokens: $2\cdot N_\text{layers}\cdot L\cdot d$ numbers (times bytes per number). For $N_\text{layers}=32$, $d=4096$, $L=10^5$ in 16-bit: $2\times32\times10^5\times4096\times2\ \text{bytes}\approx52$ GB per sequence: the reason inference at long contexts is memory-bound and why MQA/GQA and fixed-state alternatives (Chapter 11) exist.

---

## 12.6 Efficient attention

### 12.6.1 FlashAttention: exact attention without materializing the $L\times L$ matrix

Standard attention writes the $B\times H\times L\times L$ score matrix to GPU memory (HBM), which is slow relative to arithmetic. **FlashAttention** (Dao et al., 2022) computes *exact* attention in tiles that fit in fast on-chip SRAM, never storing the full matrix, so memory is $O(L)$ and wall-clock time drops substantially even though FLOPs are unchanged.

The key is the **online softmax**. For a query row with scores $s_1,\dots,s_L$ processed in blocks, maintain a running maximum $m$, a running normalizer $\ell=\sum_je^{s_j-m}$, and a running output $\mathbf{o}=\ell^{-1}\sum_je^{s_j-m}\mathbf{v}_j$. When a new block with scores $\mathbf{s}'$ and values $\mathbf{V}'$ arrives:

$$
\begin{aligned}
m^\text{new}&=\max\big(m,\max\mathbf{s}'\big),\qquad
\ell^\text{new}=e^{m-m^\text{new}}\,\ell+\textstyle\sum_je^{s'_j-m^\text{new}},\\
\mathbf{o}^\text{new}&=\frac{e^{m-m^\text{new}}\,\ell\,\mathbf{o}+\sum_je^{s'_j-m^\text{new}}\mathbf{v}'_j}{\ell^\text{new}} .
\end{aligned}
$$

*Why it is exact.* The softmax is shift-invariant ($e^{s-c}/\sum e^{s-c}$ does not depend on $c$), so we may use any running maximum for numerical stability, and rescaling the old accumulators by $e^{m-m^\text{new}}$ re-expresses them relative to the new maximum. The code implements this with blocks of 16 keys and matches full attention to $2.4\times10^{-7}$. FlashAttention-2/3 add better parallelism and hardware-specific kernels. [[E]]

**Take-away for genomics.** FlashAttention moves the practical wall from *memory* to *FLOPs*: $L=10^5$ becomes feasible but $L=10^6$ at $\sim10^{16}$ FLOPs/layer remains expensive (Chapter 11).

### 12.6.2 Reducing the FLOPs

- **Sparse/local attention** (sliding windows plus a few global tokens: Longformer, BigBird): $O(Lw)$ with window $w$. Natural for genomics (local context matters most), but loses arbitrary long-range pairs.
- **Linear attention** replaces $\exp(\mathbf{q}^\top\mathbf{k})$ by a *feature-map* kernel $\phi(\mathbf{q})^\top\phi(\mathbf{k})$. Then

$$
\sum_j\phi(\mathbf{q})^\top\phi(\mathbf{k}_j)\mathbf{v}_j=\phi(\mathbf{q})^\top\Big(\sum_j\phi(\mathbf{k}_j)\mathbf{v}_j^\top\Big)
$$

by associativity of matrix products: the bracketed $d_k\times d_v$ matrix is computed once, giving $O(Ld_kd_v)$, *linear in $L$*. The code verifies agreement with the quadratic computation to $1.6\times10^{-7}$ for $\phi=\mathrm{elu}+1$. In the causal case the bracket becomes a running sum, **an RNN with a matrix-valued state**: the bridge to SSMs (Chapter 11, Mamba-2's duality). The cost is that linear kernels cannot reproduce the sharp, selective softmax, and recall quality suffers.
- **Low-rank/approximate** (Linformer, Performer): approximate the attention matrix by projections or random features.
- **Hybrid** (Chapter 11): few attention layers among cheap operators.

---

## 12.7 Axial and structured attention for biological tensors

Biological inputs are often *two-dimensional*: a multiple sequence alignment (MSA) is a matrix of $N$ sequences × $L$ positions; a structure model carries a pair representation of $L\times L$ residue pairs. Full attention over all $NL$ tokens costs $(NL)^2$ pairs. **Axial attention** factorizes: *row attention* (each sequence attends across its own positions): $N$ sequences × $L^2$ pairs $=NL^2$; *column attention* (each alignment column attends across sequences): $L$ positions × $N^2$ pairs $=LN^2$. Total $NL^2+LN^2\ll(NL)^2$. The **MSA Transformer** (Rao et al., 2021) uses this, and **AlphaFold 2's Evoformer** alternates row attention (with a bias from the pair representation), column attention, and triangular updates on the pair representation (Chapter 35). So "attention" in structural biology typically means *structured* attention matched to the biology of homology and geometry.

---

## 12.8 Three rhymes: attention as kernel regression, Hopfield memory, and Potts coupling

!!! rhyme "Structural rhyme: attention ↔ modern Hopfield network ↔ Potts model ↔ kernel regression"
    **Hopfield.** A *modern Hopfield network* with stored patterns $\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_L]$ retrieves a pattern from a query state $\boldsymbol\xi$ by the update $\boldsymbol\xi^\text{new}=\mathbf{X}\,\softmax(\beta\mathbf{X}^\top\boldsymbol\xi)$, **which is attention** with keys = values = stored patterns and $\beta=1/\sqrt{d_k}$ (Ramsauer et al., 2021). Attention is therefore an *associative memory* with exponential storage capacity. [[E]]
    **Potts.** A Potts model of a protein family has $p(x)\propto\exp\big(\sum_ih_i(x_i)+\sum_{i<j}J_{ij}(x_i,x_j)\big)$ (Chapter 29): *pairwise couplings between positions*. Fitting it to an alignment (direct-coupling analysis) predicts residue–residue contacts from coevolution. An attention weight $\alpha_{ij}$ is a *content-dependent* coupling from position $j$ to position $i$ in a particular sequence, whereas a Potts coupling $J_{ij}$ is a fixed number shared across the family. A transformer trained on sequences *can* learn pairwise statistics similar to Potts couplings, and attention maps of protein language models do contain contact information (§12.9).
    **Kernel regression.** As above, attention is Nadaraya–Watson smoothing in value space.
    **What the rhyme buys.** (i) A reason to expect attention heads to align with pairwise statistics of sequence data; (ii) a baseline: any claim that attention "understands structure" must beat a Potts/DCA model, which has *no* neural network at all; (iii) a hint about failure: associative memory retrieves what it stored (training-set-like patterns), and generalizes via the structure of the kernel.

---

## 12.9 Attention in protein and genomic models: what the maps do and don't show

**Protein language models.** Rao et al. (2020) showed that a *linear classifier on attention maps* of a protein language model (ESM-1b) trained *without any structural supervision* predicts residue–residue contacts, trained on only a small number of known structures [[S]], and Vig et al. (2021) found attention heads specializing for contacts and binding sites. This is evidence that the masked-language objective drives the model to represent *pairwise co-variation*, because predicting a masked residue is easier if you attend to the residues that co-vary with it. It is *not* evidence that the network has an internal 3D model.

!!! example "Worked Research Example 12.1: \"Attention maps of our protein language model reveal residue contacts, so it has learned protein structure\""
    **Situation.** A paper shows that, in a protein language model, several attention heads have high weights at residue pairs that are in contact in the experimental structure, and concludes that the model "has learned the physics of protein folding."

    **Question.** What does this evidence support, and what would you need to test the stronger claim?

    **Reasoning.**

    1. *What is the weakest reading of the observation?* Attention weights correlate with contacts. Since contacts co-evolve, and the masked LM objective rewards using co-varying positions (§12.8), the correlation is *expected* if the model learned pairwise statistics of the training families, with no reference to physics.
    2. *Alternative explanations.* (H1) Pairwise coevolution statistics learned from the corpus (Potts-like); (H2) *local-sequence* proximity: many "contacts" are short-range (|i−j|<6) and trivially predictable; (H3) *memorization* of homologs in the training data, with structure inferred by similarity to a family member; (H4) *the probe is supervised*: a classifier trained on a few structures can extract contact information from *any* rich-enough feature set, including random networks.
    3. *Controls.* (a) Restrict to *long-range* contacts ($|i-j|>24$); (b) use a **randomly initialized** model with the same probe; (c) compare with an explicit Potts/DCA model on the same family; (d) evaluate on **orphan proteins** and de novo designed proteins with no homologs in the training data; (e) test whether predicted-contact quality tracks the *depth of the family's MSA in the training set*.
    4. *Predictions.* Under H1/H3, performance scales with family size and collapses on orphans; under physics-like understanding, performance on orphans would stay appreciably high.
    5. *Evidence from the literature.* Contact accuracy from language models correlates with family size and is much weaker on orphan sequences; structure predictors built on language models (ESMFold) do worse than MSA-based AlphaFold 2 on low-homology targets (Chapter 35). [[S]]

    **Expert analysis.** The strongest defensible claim is that the language model has learned *statistical* structure sufficient to *infer* contacts for proteins in well-populated families. Whether it has learned *physical* constraints that transfer to proteins without evolutionary precedent is an open question [[H]] tested with orphans, de novo designs, and mutational effects (Chapters 34–35, 52).

**DNA and cell models.** Attention in sequence-to-function models (Enformer, Chapter 31) relates *binned positions* up to ~200 kb apart; attention maps have been examined for enhancer–promoter interactions. Attention in single-cell models relates *genes* (Chapter 38). The interpretation issue is the same.

**Attention is not explanation.** An attention weight says how much a position's *value vector* contributes to an output at one layer, not how much it *affects the prediction*: values can have small norm, later layers transform and mix the outputs, and different attention patterns can yield the same prediction (Jain & Wallace, 2019; Serrano & Smith, 2019). Better tools: ablating heads or positions, gradient-based attributions, *attention rollout* across layers, and mechanistic circuit analysis (Chapters 18, 48). [[S]]

---

## 12.10 Worked research example on tokenization and positions

!!! example "Worked Research Example 12.2: Choosing tokens and positions for a DNA transformer"
    **Situation.** You must choose how to turn 100 kb of DNA into transformer inputs for variant-effect prediction. Options: (a) 1-nt tokens; (b) non-overlapping 6-mers; (c) byte-pair-encoded tokens (average ~4–5 nt per token); (d) a convolutional stem producing one token per 128 bp.

    **Reasoning.**

    1. *Cost.* Attention cost scales as $L^2$. With 100,000 nt: (a) $10^{10}$ pairs per head per layer; (b) $L=16{,}667$: $2.8\times10^8$; (c) $L\approx22{,}000$: $5\times10^8$; (d) $L=781$: $6\times10^5$.
    2. *Resolution.* A single-nucleotide variant changes (a) one token; (b) one token that *changes its identity from among 4,096*; (c) possibly *re-tokenizes* the neighborhood (BPE merges depend on context, so a single SNP can change several tokens); (d) one position in the stem's input, whose effect is carried through convolutional features.
    3. *Biological alignment.* Codons and reading frames are 3-nt units; motifs are variable-length; BPE tokens are learned from frequency, so frequent repeats get single tokens while rare regulatory words are fragmented. 6-mer tokens have no reading-frame alignment unless the stride is chosen to match.
    4. *Failure modes by choice.* (a) infeasible for global attention at 100 kb without efficient operators; (b)/(c) mask-prediction objectives can leak or become trivial if tokens overlap or if BPE makes masked tokens guessable from neighbors; (c) variant scoring is confounded by re-tokenization; (d) sub-bin sensitivity depends entirely on the stem.
    5. *Experiments that decide.* (i) A *tokenization-invariance test*: score the same variant at several shifts of the window relative to token boundaries (a robust model's score should not depend on where the boundary falls); (ii) a *single-nucleotide sensitivity test* using known splice-disrupting variants; (iii) *compute-matched* comparisons across tokenizations on a downstream functional task.

    **Expert analysis.** Tokenization is a **representation choice** (attack A2): it decides what information is cheap to express, what costs quadratic compute, and what invariances the model must learn. The best choice depends on the *resolution of the claim*: single-nucleotide effects need single-nucleotide input resolution, whereas genome-scale organization can tolerate coarse tokens. Chapter 28 treats biological tokenization in depth; Chapter 32 evaluates its consequences.

---

## 12.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: reading an attention heatmap critically"
    **Observation.** A figure shows an attention head that puts weight on the start codon from every position in a transcript model, and the caption says "the model has discovered the start codon."

    **Decompose.**

    1. *What does the heatmap actually show?* One head, one layer, averaged over some set of examples. Check whether it is *typical* (cherry-picking is easy among $N_\text{layers}\times H$ heads).
    2. *Is the pattern positional?* Many heads simply attend to the first token, the previous token, or a delimiter ("attention sink"; Xiao et al., 2024), unrelated to semantics. Compare to heads attending to the first position in a *shuffled* sequence.
    3. *Is it causal for the output?* **Ablate** the head (zero its output, or replace it by the mean over examples) and measure the change in the prediction. A head can look interpretable yet be redundant, or look boring yet be essential.
    4. *Is it specific?* If you move the start codon, does the attention follow it? If you mutate ATG to ATC, does the pattern vanish?
    5. *What alternative mechanisms exist?* The information may be carried by MLP layers or by the residual stream; the head may be a relay.

    **Possible outcomes and meaning.** Ablation changes outputs and attention tracks the codon under edits: strong evidence for a start-codon circuit (rung C3 of the Claim Ladder, within the model). Attention looks interpretable but ablation does nothing: a *decoy*. Attention is generic: no claim.

    **Distinguishing what the model computes from what a picture suggests.** Heatmaps generate hypotheses; interventions test them. This habit is the seed of mechanistic interpretability (Chapter 48).

---

## 12.12 Connections

- **Backward:** attention is built from Chapter 2's matrix products, softmax (Chapter 3), and residual/LayerNorm (Chapter 9); FlashAttention's online softmax is Chapter 3's log-sum-exp stabilization; linear attention shows the SSM connection (Chapter 11); the $1/\sqrt{d_k}$ argument is Chapter 4's variance calculation.
- **Forward:** masked and causal objectives (Chapter 13); protein language models (Chapter 34), the Evoformer/Pairformer (Chapter 35), diffusion transformers (Chapters 15, 36); sequence-to-function transformers (Chapter 31); gene-token transformers (Chapter 38); interpretation of heads and circuits (Chapters 18, 48); scaling laws for transformers (Chapter 17).

!!! takeaways "Key takeaways"
    1. **Attention** $=\softmax(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$: soft dictionary lookup, a kernel-weighted average of values. The $1/\sqrt{d_k}$ restores unit logit variance and prevents softmax saturation.
    2. **Multi-head attention:** reshape to $(B,H,L,d_h)$; parameters $4d^2$ regardless of $H$; scores cost $2BHL^2d_h$ FLOPs and $BHL^2$ memory.
    3. Self-attention is **permutation-equivariant**; positions must be injected. **RoPE** makes scores depend on relative offsets only.
    4. A transformer layer has $\approx12d^2$ parameters and $\approx24d^2+4Ld$ forward FLOPs per token; attention dominates when $L>6d$. The KV cache dominates inference memory.
    5. **FlashAttention** (online softmax) gives exact attention in $O(L)$ memory; **linear attention** (associativity) gives $O(L)$ time with a recurrent view; **axial attention** factorizes MSA attention into row and column passes.
    6. Attention ↔ Hopfield memory ↔ kernel regression ↔ Potts-like pairwise couplings: *a method must beat Potts/DCA to claim more than coevolution.*
    7. **Attention weights are hypotheses, not explanations**: validate with ablations and interventions.

---

## Further reading

- Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*. Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
- Su, J. et al. (2021). RoFormer: enhanced transformer with rotary position embedding. arXiv:2104.09864. Press, O., Smith, N. A. & Lewis, M. (2022). Train short, test long: attention with linear biases enables input length extrapolation. *ICLR*.
- Shazeer, N. (2019). Fast transformer decoding: one write-head is all you need. arXiv:1911.02150. Ainslie, J. et al. (2023). GQA: training generalized multi-query transformer models. *EMNLP*.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. & Ré, C. (2022). FlashAttention: fast and memory-efficient exact attention with IO-awareness. *NeurIPS*. Dao, T. (2024). FlashAttention-2. *ICLR*.
- Katharopoulos, A., Vyas, A., Pappas, N. & Fleuret, F. (2020). Transformers are RNNs: fast autoregressive transformers with linear attention. *ICML*. Choromanski, K. et al. (2021). Rethinking attention with Performers. *ICLR*. Beltagy, I., Peters, M. E. & Cohan, A. (2020). Longformer. arXiv:2004.05150.
- Ramsauer, H. et al. (2021). Hopfield networks is all you need. *ICLR*.
- Rao, R., Meier, J., Sercu, T., Ovchinnikov, S. & Rives, A. (2020). Transformer protein language models are unsupervised structure learners. *bioRxiv* (ICLR 2021). Rao, R. et al. (2021). MSA Transformer. *ICML*. Vig, J. et al. (2021). BERTology meets biology: interpreting attention in protein language models. *ICLR*.
- Avsec, Ž. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods* 18, 1196–1203. (Enformer.)
- Jain, S. & Wallace, B. C. (2019). Attention is not explanation. *NAACL*. Serrano, S. & Smith, N. A. (2019). Is attention interpretable? *ACL*. Xiao, G. et al. (2024). Efficient streaming language models with attention sinks. *ICLR*.
- Elhage, N. et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*.
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589.
