# Chapter 11. Recurrent, State-Space, and Long-Convolution Sequence Models

!!! abstract "Chapter at a glance"
    **Motivation.** Genomes are long: a gene with its regulators spans $10^5$–$10^6$ bases, a chromosome $10^8$. Attention costs $O(L^2)$ (Chapter 12), so models aimed at megabase contexts need alternatives. This chapter develops the lineage from recurrent networks to **state-space models (SSMs)** and **implicit long convolutions (Hyena)**, the architectures behind HyenaDNA, Caduceus, Evo, and Evo 2.
    **Prerequisites.** Chapters 2, 3, 9, 10.
    **You will be able to:** (1) derive backpropagation through time and explain vanishing/exploding gradients and the LSTM fix; (2) derive the discretization of a linear state-space model and prove its recurrent and convolutional forms are equivalent; (3) explain how selective (input-dependent) SSMs are trained in parallel by an associative scan; (4) describe implicit long convolutions; (5) compare architectures along cost, memory, and what *computation* each can perform; (6) reason about what "long context" does and does not buy for genomic modeling.

---

## 11.1 The long-context problem

A sequence model must map $x_{1:L}$ to outputs that depend on far-apart positions. Three design axes matter:

1. **Cost in $L$.** Training and inference FLOPs and memory as the context grows.
2. **Parallelism.** Can all positions be processed at once on a GPU (fast training), or must they be processed in order?
3. **What can be computed.** Which functions of the sequence can the architecture represent and learn: local motifs, counts, copying a distant token, content-based retrieval?

Chapter 10's CNNs are linear in $L$ and fully parallel but have a bounded receptive field. Recurrent networks have unbounded theoretical context but are sequential and hard to train. Attention has global context and parallelism but quadratic cost. SSMs and long convolutions aim for *global context, near-linear cost, and parallel training*.

---

## 11.2 Recurrent neural networks and backpropagation through time

### 11.2.1 The recurrence

A vanilla RNN carries a hidden state: $\mathbf{h}_t=\tanh(\mathbf{W}_h\mathbf{h}_{t-1}+\mathbf{W}_x\mathbf{x}_t+\mathbf{b})$, with outputs $\hat{\mathbf{y}}_t=\mathbf{W}_o\mathbf{h}_t$. The same weights are applied at every step; a length-$L$ sequence is processed by $L$ sequential steps, and the state $\mathbf{h}_t$ is a fixed-size summary of $\mathbf{x}_{1:t}$: an *information bottleneck* (Chapter 5).

### 11.2.2 BPTT and the product of Jacobians

Unrolling the recurrence gives a deep network with *shared* weights across $L$ layers (one per time step); backpropagation through time (BPTT) is Chapter 9's backprop on the unrolled graph, with gradients *summed* over time for the shared weights. For loss $\mathcal{L}=\sum_t\ell_t$,

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{h}_t}=\frac{\partial\ell_t}{\partial\mathbf{h}_t}+\Big(\frac{\partial\mathbf{h}_{t+1}}{\partial\mathbf{h}_t}\Big)^{\!\top}\frac{\partial\mathcal{L}}{\partial\mathbf{h}_{t+1}},\qquad
\frac{\partial\mathbf{h}_{t+1}}{\partial\mathbf{h}_t}=\mathrm{diag}\big(1-\mathbf{h}_{t+1}^{\odot2}\big)\,\mathbf{W}_h .
$$

The influence of a distant step $t$ on step $T$ is the product $\partial\mathbf{h}_T/\partial\mathbf{h}_t=\prod_{k=t+1}^T\mathrm{diag}(1-\mathbf{h}_k^{\odot2})\mathbf{W}_h$. Since $|\tanh'|\le1$, its norm is bounded by $\|\mathbf{W}_h\|_2^{\,T-t}$: **if the largest singular value of $\mathbf{W}_h$ is below 1, gradients vanish exponentially with the time lag**, so the network cannot learn dependencies across long lags. Conversely, a large spectral radius makes gradients explode, though saturating $\tanh$ units limit the growth.

The simulation (64 hidden units) shows the gradient path norm $\|\partial\mathbf{h}_T/\partial\mathbf{h}_0\|$:

| Spectral radius $\rho$ of $\mathbf{W}_h$ | $T=10$ | $T=50$ | $T=200$ |
|---|---|---|---|
| 0.8 | $6.6\times10^{-2}$ | $4.7\times10^{-8}$ | $1.1\times10^{-31}$ |
| 1.0 | $3.8\times10^{-1}$ | $3.3\times10^{-4}$ | $5.2\times10^{-17}$ |
| 1.5 | 3.5 | 4.6 | 3.3 |
| 3.0 | $1.2\times10^{2}$ | $4.6\times10^{6}$ | $2.9\times10^{24}$ |

Even at $\rho=1$ the gradient vanishes (the $\tanh$ derivative below 1 contributes a contraction), while $\rho=1.5$ is near the *edge of stability* where saturation roughly offsets expansion, and $\rho=3$ explodes. Learnable long memory requires $\mathbf{W}_h$ to sit on a narrow ridge. This is the same *gain per step to the power of the number of steps* phenomenon as Chapter 9's depth analysis, with time playing the role of depth.

### 11.2.3 LSTMs and the additive memory path

The **LSTM** (Hochreiter & Schmidhuber, 1997) adds a **cell state** $\mathbf{c}_t$ updated *additively*, controlled by gates:
$$
\begin{aligned}
\mathbf{f}_t&=\sigma(\mathbf{W}_f\mathbf{x}_t+\mathbf{U}_f\mathbf{h}_{t-1}+\mathbf{b}_f),\ \ \mathbf{i}_t=\sigma(\cdots),\ \ \mathbf{o}_t=\sigma(\cdots),\\
\tilde{\mathbf{c}}_t&=\tanh(\mathbf{W}_c\mathbf{x}_t+\mathbf{U}_c\mathbf{h}_{t-1}+\mathbf{b}_c),\\
\mathbf{c}_t&=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,\qquad
\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t).
\end{aligned}
$$
The direct path $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}=\mathrm{diag}(\mathbf{f}_t)$ has *no weight matrix and no saturating nonlinearity*: if the forget gate is near 1, the error signal flows backward unchanged (the *constant error carousel*). The model *learns when to remember* ($f\approx1$) and when to overwrite. The GRU is a simplified variant. LSTMs powered pre-transformer sequence modeling, including protein representations (UniRep, Alley et al., 2019; SeqVec/ELMo, Heinzinger et al., 2019) and early secondary-structure predictors.

**Why RNNs lost to attention and SSMs for large-scale modeling.** (i) *Sequential training*: step $t$ cannot be computed before step $t-1$, leaving GPUs underused. (ii) *State bottleneck*: all history must pass through a fixed-size vector. (iii) *Optimization difficulty at long lags* despite gating. Nevertheless, the RNN's $O(1)$ per-token memory at inference remains attractive for megabase sequences, a property that SSMs recover with parallel training.

---

## 11.3 Linear state-space models (S4)

### 11.3.1 From continuous dynamics to a discrete recurrence

A **linear state-space model** describes a continuous input signal $x(t)$ via a latent state:
$$
\dot{\mathbf{h}}(t)=\mathbf{A}\,\mathbf{h}(t)+\mathbf{B}\,x(t),\qquad y(t)=\mathbf{C}\,\mathbf{h}(t)\ (+\,D\,x(t)).
$$
With state $\mathbf{h}\in\R^N$, $\mathbf{A}\in\R^{N\times N}$, $\mathbf{B},\mathbf{C}^\top\in\R^{N}$. The solution is $\mathbf{h}(t)=e^{\mathbf{A}t}\mathbf{h}(0)+\int_0^te^{\mathbf{A}(t-s)}\mathbf{B}x(s)\,ds$.

**Discretization (zero-order hold).** Sample at step $\Delta$ and hold $x$ constant over each interval. Then $\mathbf{h}_{k+1}=\bar{\mathbf{A}}\mathbf{h}_k+\bar{\mathbf{B}}x_k$ with
$$
\bar{\mathbf{A}}=e^{\Delta\mathbf{A}},\qquad\bar{\mathbf{B}}=(\Delta\mathbf{A})^{-1}\big(e^{\Delta\mathbf{A}}-\mathbf{I}\big)\Delta\mathbf{B}=\mathbf{A}^{-1}\big(e^{\Delta\mathbf{A}}-\mathbf{I}\big)\mathbf{B}.
$$
*Derivation.* Over $[k\Delta,(k+1)\Delta)$ with $x\equiv x_k$: $\mathbf{h}_{k+1}=e^{\mathbf{A}\Delta}\mathbf{h}_k+\int_0^\Delta e^{\mathbf{A}(\Delta-s)}\mathbf{B}\,ds\,x_k$, and $\int_0^\Delta e^{\mathbf{A}(\Delta-s)}ds=\mathbf{A}^{-1}(e^{\mathbf{A}\Delta}-\mathbf{I})$. $\square$ The parameter $\Delta$ (a learned timescale) controls how much of the past is retained: small $\Delta$ means long memory.

### 11.3.2 Two equivalent computations

Unrolling the recurrence from $\mathbf{h}_{-1}=0$ gives $\mathbf{h}_t=\sum_{j=0}^{t}\bar{\mathbf{A}}^{\,j}\bar{\mathbf{B}}\,x_{t-j}$, and hence

$$
y_t=\sum_{j=0}^{t}\bar K_j\,x_{t-j},\qquad\bar K_j=\mathbf{C}\bar{\mathbf{A}}^{\,j}\bar{\mathbf{B}} .
$$

**The same model is a recurrence and a convolution** with a (generally very long) kernel $\bar K$.

- *Recurrent mode:* $O(1)$ memory and $O(N)$–$O(N^2)$ compute per new token. Ideal for inference/generation of long sequences.
- *Convolutional mode:* compute $\bar K$ once, then convolve the whole sequence with it using the FFT in $O(L\log L)$, fully parallel. Ideal for training.

The code constructs a diagonal stable $\mathbf{A}$ with $N=16$, discretizes it, and runs both modes on a length-512 sequence: they agree to $9\times10^{-16}$. The kernel $\bar K_j$ is a **sum of decaying exponentials** (for diagonal $\mathbf{A}$, $\bar K_j=\sum_nC_n\bar a_n^{\,j}\bar B_n$): in the simulation $|\bar K|$ decays from 0.22 at $j=0$ to $1.8\times10^{-2}$ at 100 and $4.7\times10^{-5}$ at 500. The model's memory is determined by the eigenvalues of $\mathbf{A}$: modes with small $|\mathrm{Re}(\lambda)|$ remember for $\sim1/(\Delta|\mathrm{Re}\lambda|)$ steps. **Long-range memory requires slow modes, and a good initialization of $\mathbf{A}$ spreads eigenvalues to cover many timescales.**

### 11.3.3 HiPPO and S4

A random $\mathbf{A}$ forgets quickly. **HiPPO** (Gu et al., 2020) derives a specific $\mathbf{A}$ such that the state $\mathbf{h}(t)$ holds the coefficients of the best polynomial (Legendre) approximation of the *entire input history* up to time $t$. **S4** (Gu, Goel & Ré, 2022) parameterizes $\mathbf{A}$ as a normal matrix plus low-rank correction, enabling the kernel $\bar K$ to be computed in $\tilde O(N+L)$ and solving the Long Range Arena tasks involving dependencies over thousands of steps that RNNs and early transformers failed. [[E]] as results; S4 is a *linear time-invariant* (LTI) system.

**Limitation of LTI systems.** The kernel $\bar K$ is *independent of the input content*. The layer cannot decide, based on what it reads, to remember one token and ignore another. Tasks that require content-based selection, such as **associative recall** ("if you saw key $k$ earlier, output the value that followed it") and copying, are hard for LTI SSMs. This matters for biology: *recognizing that a motif seen 50 kb ago matches the one here* is content-based retrieval.

---

## 11.4 Selective state spaces (Mamba) and the associative scan

**Mamba** (Gu & Dao, 2023) makes the discretization step and the projections *functions of the input*: $\Delta_t,\mathbf{B}_t,\mathbf{C}_t=f(x_t)$. The recurrence is then
$$
\mathbf{h}_t=\bar{\mathbf{A}}_t\mathbf{h}_{t-1}+\bar{\mathbf{B}}_tx_t,\qquad y_t=\mathbf{C}_t\mathbf{h}_t,
$$
with $\bar{\mathbf{A}}_t=\exp(\Delta_t\mathbf{A})$ varying with the content of $x_t$. A large $\Delta_t$ resets the state toward the current input (forget the past, focus on this token); a small $\Delta_t$ retains the past and ignores the current input. This *selection* restores content-based memory. The price is that the system is *time-varying*: there is no single kernel $\bar K$, so the FFT convolution trick no longer applies.

**Parallel scan.** The recurrence $h_t=a_th_{t-1}+b_t$ (scalar for clarity) is a composition of *affine maps* $h\mapsto a_th+b_t$. Composition of affine maps is **associative**:
$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\ a_2b_1+b_2),
$$
and associativity allows computing all prefixes in $O(\log L)$ parallel rounds with $O(L)$ total work (a *parallel prefix scan*; Blelloch, 1990). The code implements a Hillis–Steele scan, using 10 rounds for $L=1000$, and agrees with the sequential recurrence to $1.3\times10^{-15}$. Mamba additionally fuses the scan into a single GPU kernel that keeps the expanded state in fast on-chip memory (hardware-aware implementation), which is what makes it fast in practice.

**Relation to attention.** Dao & Gu (2024, "Mamba-2") showed that SSMs of a restricted form are equivalent to a masked *linear attention* with a semiseparable mask ("state-space duality"), tying together the two families. [[S]] This anticipates Chapter 12: an SSM is an attention-like mixing with a *structured* (low-rank-in-time) mixing matrix.

---

## 11.5 Implicit long convolutions: Hyena

A convolutional layer with a kernel as long as the sequence has $L$ parameters per channel, impractical to learn directly and prone to overfitting. **Hyena** (Poli et al., 2023) defines the kernel **implicitly**: a small neural network maps *position* $t$ to the kernel value $h(t)$, multiplied by an exponentially decaying window, so the kernel has few parameters yet can span the whole sequence and (unlike a sum of exponentials) can represent oscillatory or multi-scale shapes. A Hyena operator interleaves such long convolutions (computed by FFT in $O(L\log L)$) with **multiplicative gating** by learned projections of the input, $z^{n+1}_t=g^n_t\cdot(h^n*z^n)_t$: the gates make the layer *data-controlled* (input-dependent), approximating the content-based mixing of attention at sub-quadratic cost.

**In genomics:**

- **HyenaDNA** (Nguyen et al., 2023): a Hyena-based language model on the human reference genome with **single-nucleotide tokens and context up to 1 million nucleotides**, reported to train up to 160× faster than a transformer with FlashAttention at that scale, and competitive on benchmarks with far fewer parameters (1.6M vs. 2.5B in one comparison on the Nucleotide Transformer tasks). [[S]]
- **Caduceus** (Schiff et al., 2024): bidirectional Mamba-style SSM blocks with reverse-complement equivariance built in (Chapter 10, §10.6), designed for DNA.
- **Evo** (Nguyen et al., *Science*, 2024): **StripedHyena**, a hybrid of data-controlled convolutional (Hyena) layers and attention layers, 7 billion parameters, **131,072-nucleotide context**, trained on ~300 billion nucleotides of prokaryotic genomes (OpenGenome) with byte-level single-nucleotide tokenization. [[S]]
- **Evo 2** (Brixi et al., *Nature*, 2026): **StripedHyena 2**, a convolutional *multi-hybrid* mixing *short explicit*, *medium regularized*, and *long implicit* Hyena operators with attention blocks; 40 billion parameters; trained on >9 trillion nucleotides with a **1-million-token context**, via pretraining at 8,192 tokens followed by progressive context extension ("midtraining"). [[S]]

(Chapter 32 treats what these models learn and how well; this chapter explains *how they can be run at that length*.)

---

## 11.6 Comparing architectures

| | CNN (Ch. 10) | RNN / LSTM | Attention (Ch. 12) | S4 (LTI SSM) | Mamba (selective) | Hyena / StripedHyena |
|---|---|---|---|---|---|---|
| Training cost in $L$ | $O(L)$ | $O(L)$ sequential | $O(L^2)$ | $O(L\log L)$ | $O(L)$ (scan) | $O(L\log L)$ |
| Training parallel over $L$ | Yes | **No** | Yes | Yes (FFT) | Yes (scan) | Yes (FFT) |
| Inference per token | $O(k)$ window | $O(1)$ | $O(L)$ (KV cache grows) | $O(1)$ | $O(1)$ | $O(1)$–$O(L)$ depending on mode |
| State / memory at inference | window | fixed vector | grows with $L$ | fixed | fixed | fixed or growing |
| Context length | bounded (receptive field) | unbounded in principle; short in practice | full within window | long, fixed memory profile | long | long |
| Content-based selection | No | Yes (gates) | **Yes (core)** | **No** | Yes | Partial (gating) |
| Associative recall / copying | Poor | Moderate | **Strong** | Poor | Good | Moderate; hybrids with attention close the gap |
| Inductive bias | locality, equivariance | recency, order | none (needs position encoding) | smooth, exponentially-decaying memory | selective decay | multiscale, smooth kernels |

The empirical lesson from language modeling is that *hybrids* (a few attention layers among SSM/convolution layers) tend to capture most benefits: attention supplies precise retrieval, cheap operators supply bulk mixing. Evo/Evo 2 follow this design. [[S]] for language modeling; [[P]] for the specifics in genomics.

```python
--8<-- "code/ch11_sequence_models.py"
```

Output (abridged; see the full script for the RNN table above):

```text
SSM: max |recurrent - convolutional| = 8.9e-16   (kernel decays: K[0]=-0.221, K[100]=-1.8e-02, K[500]=-4.7e-05)
parallel scan vs sequential recurrence: max difference = 1.3e-15  (10 rounds instead of 1000 sequential steps)

cost per layer of mixing information across L positions (d = 4096; illustrative orders of magnitude):
           L    attention FLOPs 4 L^2 d   attn score memory (fp16)    long-conv FLOPs ~ 10 d L log2 L    SSM scan FLOPs ~ 10 d N L
       8,192                   1.10e+12                     0.1 GB                           4.36e+09                     5.37e+09
     131,072                   2.81e+14                    34.4 GB                           9.13e+10                     8.59e+10
   1,000,000                   1.64e+16                  2000.0 GB                           8.16e+11                     6.55e+11
```

At $L=10^6$, naive attention needs $\approx1.6\times10^{16}$ FLOPs per layer (about 20× the convolutional cost) and a 2-TB score matrix per head-batch; FlashAttention removes the memory but not the FLOPs. *These numbers are the reason for the whole chapter.* (The FLOP expressions are order-of-magnitude, ignoring constants and the cost of the projections.)

---

## 11.7 What does "long context" buy in genomics?

Longer context is not automatically better. It helps only if the target depends on distant information that the model can *use*. Two very different sources of long-range information coexist in genomes:

1. **Regulatory long-range dependence**: enhancers tens to hundreds of kb from their target genes, domain structure, insulators. This is what we usually want.
2. **Repetitive and duplicated sequence**: the genome contains thousands of copies of repeat families (e.g., Alu, LINE-1), tandem repeats, and segmental duplications. Given a long context that contains an earlier copy of a repeat, a sequence model can *copy* it: next-token prediction improves with no regulatory understanding at all.

!!! example "Worked Research Example 11.1: Perplexity improves as the context grows from 8 kb to 1 Mb. Is the model using long-range regulation?"
    **Situation.** A genomic language model reports that held-out per-nucleotide perplexity decreases monotonically as context length increases from 8 kb to 128 kb to 1 Mb, and the authors say the model "leverages long-range genomic structure."

    **Question.** What does the curve show, and what alternative explanations exist?

    **Reasoning.**

    1. *What does perplexity measure?* The average predictive quality over *all* positions (Chapter 5). The mean gain from longer context can come from any subset of positions.
    2. *Where is the information in long context?* (H1) Distal regulatory elements informing nearby regulatory sequence (rare, subtle). (H2) **Repeat copying:** a longer window is more likely to contain an earlier copy of the repeat family member being predicted, and tandem repeats and segmental duplications span tens to hundreds of kb. (H3) **Isochore/compositional state:** GC content and gene density vary at 100 kb–Mb scales and shift the base distribution; a model that estimates the local state from a longer window predicts better. (H4) **Gene structure:** predicting exon/intron boundaries from a distant coding frame. (H5) **Data leakage:** validation windows overlapping or duplicating training windows (segmental duplications).
    3. *Predictions that separate them.* Per-position loss as a function of *distance to the nearest earlier similar sequence in the context*: under H2 the gain concentrates where an earlier near-copy exists; under H1 it concentrates in regulatory regions and depends on the *location* of the distal element rather than on sequence similarity. Repeat-masked evaluation: under H2 the long-context gain shrinks substantially when repeats are masked. Scrambled-context control: replace distal context (beyond 8 kb) with a shuffled sequence preserving composition; under H3 a composition-only context recovers most of the gain.
    4. *Experiments.* (a) Plot the loss improvement by annotation class (repeat family, coding, promoter, enhancer, intergenic unique); (b) shuffled-distal-context control; (c) compute the gain for positions with no near-duplicates in the preceding window (via k-mer sketch/minimizers); (d) insert/remove a known distal enhancer from the context and test whether predicted regulatory-track values at the promoter (if the model is conditional on function; Chapter 31) change with its presence and distance; (e) check train–validation homology.
    5. *What would change your mind?* Gains concentrated in *unique, non-repetitive* regulatory annotations that depend on *where* the distal element is (not just its sequence) would support H1. Gains explained by repeats and composition would support H2/H3, in which case "long context" is a *real* improvement in sequence modeling but not evidence of regulatory understanding.

    **Expert analysis.** The perplexity curve is *consistent with* H1 but is *predicted by* H2 and H3 at least as strongly. The study design principle is **decompose the aggregate by the mechanism you want to claim**. For regulatory claims, the right evaluations are *functional*: does the long context improve prediction of *chromatin and expression tracks* at the promoter, or *variant effects*, relative to a short-context control, particularly for variants in distal enhancers? (Chapters 31, 32, 50.)

!!! example "Worked Research Example 11.2: Choosing an architecture for a 500-kb locus at single-nucleotide resolution is an argument"
    **Situation.** You want a model that, given 500 kb of DNA around a gene, predicts expression and chromatin tracks in 100 cell types, then scores variants by in silico mutagenesis. You have training data for ~$10^5$ loci (randomly sampled genomic windows) and about 1,000 GPU-days.

    **The argument you should be able to write.**

    1. *Requirement from biology:* distal enhancers up to ~100–200 kb from the TSS ⇒ effective context ≥ 200 kb on each side. *Requirement from the task:* base-pair sensitivity for variants ⇒ per-nucleotide *input* resolution; output resolution can be coarser (128 bp bins) for expression/chromatin.
    2. *Cost:* with $d=1024$ and a 500-kb input at 1-bp tokens, full attention is infeasible ($L^2=2.5\times10^{11}$ scores per head). Standard fix: **CNN stem with pooling** reduces $L$ by 128× to ~4,000 tokens, then attention (Enformer-style; Chapter 31) is affordable ($L^2\approx1.6\times10^7$). A SSM/long-conv backbone is an alternative that avoids the pooling bottleneck.
    3. *Inductive bias:* the CNN stem supplies motif detectors with RC symmetry (§10.6); attention supplies pairwise enhancer–promoter interactions; the SSM supplies a smooth, long memory.
    4. *Data:* $10^5$ loci is small for a long-context model with $10^8$+ parameters. Bias is valuable where data are scarce (Chapter 7, §7.6): *convolutional stems and RC symmetry are cheap bias*; a fully content-based long-range module has more freedom to overfit. Evaluate by held-out chromosomes.
    5. *Risk register:* (i) the effective context of an SSM may be shorter than its nominal one; measure it by distance-dependence of attributions; (ii) attention over pooled tokens loses sub-bin resolution that variant scoring needs (mitigate with a U-Net-style decoder); (iii) the training set may not contain enough distal-regulation examples to learn it at all (a data limitation, not an architecture limitation: test by synthetic spike-in enhancers at known distances).
    6. *Pre-registered test:* distance-stratified performance on CRISPRi-validated enhancer–gene pairs.

    **Expert analysis.** No architecture is "best"; each is a bet on which assumptions are true of the data. The research skill is to write the bets down and design the experiment that tests the riskiest one.

---

## 11.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: what can a fixed-size state remember?"
    **Setup.** A selective SSM with state dimension $N=16$ per channel and width $d=4096$ holds $Nd=65{,}536$ numbers per layer, about 1 Mbit at 16-bit precision. A 1-Mb DNA window contains 2 Mbit of raw information (2 bits per base) and, by Chapter 5, perhaps 1.6–1.9 Mbit after compression.

    **Observation.** The state cannot store the sequence. It must store a *compressed summary* sufficient for predicting the future. That is a *learned compression objective*: the model keeps what the next-token loss rewards.

    **Questions an expert asks.**

    1. *What is rewarded?* Next-token loss rewards remembering things that improve predictions: repeat copies (which pay off enormously per bit of memory), local composition, gene-structure context. Regulatory interactions pay off *rarely*; a loss-optimal state may not allocate capacity to them.
    2. *What would a bottleneck imply?* If memory is scarce, the model prioritizes by loss reduction per bit. Distal enhancer information might be systematically *under-represented* in a state shaped by next-token prediction, even if the architecture could in principle store it. That is an **objective-induced** limitation, not an architectural one (G-O).
    3. *How to test.* Probe the state at a promoter for the *presence of a distal enhancer* (a linear probe on the hidden state at position $t$ for a spike-in element inserted at distance $\Delta$); the probe accuracy versus $\Delta$ and versus state size $N$ gives the *effective memory curve* for regulatory information. Compare to a copy task (repeat at distance $\Delta$). The *gap between the copy curve and the enhancer curve* measures how much the training objective favors one over the other.
    4. *What intervention follows?* If the enhancer curve is much worse: change the objective (add functional supervision, Chapter 31), reweight the loss toward regulatory regions, or add retrieval/attention modules.

    **What this teaches.** Architectural capacity and *what the objective makes the model use* are different things. Probing a model's memory with *controlled spike-ins* is a general technique for separating the two.

---

## 11.9 Connections

- **Backward:** BPTT is Chapter 9's chain rule on an unrolled graph; vanishing gradients are the product-of-Jacobians analysis; the SSM kernel is Chapter 2's matrix powers and eigenvalues; the recurrence–convolution equivalence is Chapter 10's convolution in disguise; associative scan is Chapter 6's semiring/dynamic-programming structure.
- **Forward:** attention as the content-based alternative (Chapter 12); self-supervised objectives for these backbones (Chapter 13); genomic LMs built on them (Chapter 32); sequence-to-function models mixing convolutions with attention (Chapter 31); scaling of context length and compute (Chapters 17, 47).

!!! takeaways "Key takeaways"
    1. RNN gradients are products of Jacobians over time: $\|\partial\mathbf{h}_T/\partial\mathbf{h}_t\|\lesssim\|\mathbf{W}_h\|^{T-t}$; the LSTM's additive cell path (gradient $\mathrm{diag}(\mathbf{f}_t)$) fixes it, but RNNs are sequential and bottlenecked.
    2. A **linear SSM** discretizes as $\bar{\mathbf{A}}=e^{\Delta\mathbf{A}}$, $\bar{\mathbf{B}}=\mathbf{A}^{-1}(e^{\Delta\mathbf{A}}-\mathbf{I})\mathbf{B}$; it equals a **convolution** with $\bar K_j=\mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}$ (recurrent mode for inference, FFT mode for training; identical to $10^{-15}$).
    3. LTI SSMs cannot select by content; **Mamba** makes parameters input-dependent and trains with an **associative scan** (composition of affine maps is associative, $O(\log L)$ parallel depth).
    4. **Hyena** uses implicitly parameterized long convolutions with multiplicative gating; **StripedHyena** (Evo) and **StripedHyena 2** (Evo 2) hybridize them with attention to reach 131 kb and 1 Mb contexts.
    5. Attention at $L=10^6$ costs $\sim10^{16}$ FLOPs/layer and TBs of scores; sub-quadratic operators cost $\sim10^{12}$.
    6. Long context helps *only if information is there and the objective makes the model use it*; perplexity gains can come from repeat copying and composition, not regulation. Decompose aggregate gains by mechanism.

---

## Further reading

- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9, 1735–1780. Gers, F. A., Schmidhuber, J. & Cummins, F. (2000). Learning to forget: continual prediction with LSTM. *Neural Computation* 12, 2451–2471.
- Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
- Gu, A., Dao, T., Ermon, S., Rudra, A. & Ré, C. (2020). HiPPO: recurrent memory with optimal polynomial projections. *NeurIPS*. Gu, A., Goel, K. & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR*.
- Gu, A. & Dao, T. (2023). Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752. Dao, T. & Gu, A. (2024). Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. *ICML*.
- Blelloch, G. E. (1990). Prefix sums and their applications. Technical Report CMU-CS-90-190.
- Poli, M. et al. (2023). Hyena hierarchy: towards larger convolutional language models. *ICML*.
- Nguyen, E. et al. (2023). HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. *NeurIPS*.
- Schiff, Y. et al. (2024). Caduceus: bi-directional equivariant long-range DNA sequence modeling. *ICML*.
- Nguyen, E. et al. (2024). Sequence modeling and design from molecular to genome scale with Evo. *Science* 386, eado9336.
- Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361. Ku, J. et al. (2025). Systems and algorithms for convolutional multi-hybrid language models at scale. arXiv:2503.01868.
- Tay, Y. et al. (2021). Long Range Arena: a benchmark for efficient transformers. *ICLR*.
- Arora, S. et al. (2023). Zoology: measuring and improving recall in efficient language models. arXiv:2312.04927.
- Alley, E. C., Khimulya, G., Biswas, S., AlQuraishi, M. & Church, G. M. (2019). Unified rational protein engineering with sequence-based deep representation learning. *Nature Methods* 16, 1315–1322.
