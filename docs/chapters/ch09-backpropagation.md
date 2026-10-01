# Chapter 9. Neural Networks and Backpropagation

!!! abstract "Chapter at a glance"
    **Motivation.** Every model in Parts VII–IX (genomic language models, AlphaFold-style networks, single-cell foundation models, diffusion models) is trained by backpropagation. If you can derive it, implement it, and diagnose its failure modes, you can read any architecture paper as a statement about *what gradients flow where*.
    **Prerequisites.** Chapters 2–3 (matrix calculus, chain rule, gradient descent).
    **You will be able to:** (1) define a feedforward network and state what universal approximation does and does not say; (2) **derive backpropagation** for an MLP with matrix shapes, and for an arbitrary computational graph; (3) implement reverse-mode autodiff from scratch and verify it against finite differences and PyTorch; (4) explain vanishing/exploding gradients and derive He initialization; (5) explain normalization layers and residual connections as *gradient-flow engineering*; (6) diagnose a failing training run by forming and testing hypotheses.


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M3](m03-single-variable-calculus.md) (the chain rule) and [M4](m04-multivariable-calculus-optimization.md) (Jacobians) and [M10](m10-python-numerical-computing.md) (autograd and floating point).

---

## 9.1 From linear models to networks

A **linear model** computes $\hat y=\mathbf{w}^\top\mathbf{x}+b$. To represent non-linear functions, compose linear maps with element-wise nonlinearities. A **feedforward network (multilayer perceptron, MLP)** with $L$ layers computes, for input $\mathbf{a}^{(0)}=\mathbf{x}$,

$$
\mathbf{z}^{(\ell)}=\mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)},\qquad
\mathbf{a}^{(\ell)}=\sigma\big(\mathbf{z}^{(\ell)}\big),\qquad \ell=1,\dots,L,
$$

with $\mathbf{W}^{(\ell)}\in\R^{d_\ell\times d_{\ell-1}}$, bias $\mathbf{b}^{(\ell)}\in\R^{d_\ell}$, and (usually) no nonlinearity on the final layer: the output $\mathbf{z}^{(L)}$ are *logits* or a regression value. The *pre-activation* $\mathbf{z}$ and *activation* $\mathbf{a}$ are distinguished because both appear in the gradient.

**Activation functions.**

| Name | $\sigma(z)$ | $\sigma'(z)$ | Notes |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)\le\frac14$ | Saturates; derivative $\le1/4$ causes vanishing gradients |
| Tanh | $\tanh z$ | $1-\tanh^2z\le1$ | Zero-centered; saturates |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | Non-saturating for $z>0$; can "die" (always 0) |
| GELU | $z\,\Phi(z)$ | smooth | Standard in transformers |
| SiLU/Swish | $z\,\sigma(z)$ | smooth | Used in many modern models; gated variants (SwiGLU) in LLMs |

**What does a hidden unit compute?** Unit $j$ in layer $\ell$ is a *soft detector*: $a_j=\sigma(\mathbf{w}_j^\top\mathbf{a}^{(\ell-1)}+b_j)$ responds strongly when the previous layer's pattern aligns with its weight vector. In the first layer of a genomics model this is literally a PWM scan (Chapter 2, §2.2.3; Chapter 10). Deeper units detect combinations of earlier detections.

### 9.1.1 What universal approximation says (and doesn't)

**Theorem (Cybenko 1989; Hornik 1991).** A feedforward network with one hidden layer and a non-polynomial activation can approximate any continuous function on a compact domain to arbitrary accuracy, if given enough hidden units. [[E]]

It says a good network *exists*. It does not say (i) how many units are needed (can be exponential in the input dimension), (ii) that gradient descent will find it, or (iii) that it will generalize from finite data. Depth buys *efficiency*: there are functions that deep networks represent with polynomially many units but shallow ones need exponentially many (Telgarsky, 2016). Moreover, standard networks can fit *random labels* perfectly (Zhang et al., 2017): capacity alone does not explain why they generalize on real data (Chapter 7). Take away: **representability is cheap; learnability and generalization are the scientific questions.**

---

## 9.2 Backpropagation, derived

**Goal.** Given a scalar loss $\mathcal{L}$ computed from the network output, compute $\partial\mathcal{L}/\partial\mathbf{W}^{(\ell)}$ and $\partial\mathcal{L}/\partial\mathbf{b}^{(\ell)}$ for every layer, efficiently.

### 9.2.1 The single-example derivation

Define the **error signal** (or *delta*) at layer $\ell$:

$$
\boldsymbol\delta^{(\ell)}\;:=\;\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(\ell)}}\in\R^{d_\ell}.
$$

All parameter gradients are simple functions of the deltas, so the task reduces to computing the deltas.

**Step 1: output layer.** For the common cases, $\boldsymbol\delta^{(L)}$ is known in closed form. For softmax cross-entropy (Chapter 3, §3.3.1), $\boldsymbol\delta^{(L)}=\mathbf{p}-\mathbf{e}_y$. For squared error with a linear output, $\boldsymbol\delta^{(L)}=\hat{\mathbf{y}}-\mathbf{y}$.

**Step 2: propagate backward.** The only way $\mathbf{z}^{(\ell)}$ influences the loss is through $\mathbf{z}^{(\ell+1)}$. Applying the chain rule component-wise,

$$
\delta^{(\ell)}_j=\sum_k\frac{\partial\mathcal{L}}{\partial z^{(\ell+1)}_k}\cdot\frac{\partial z^{(\ell+1)}_k}{\partial z^{(\ell)}_j}
=\sum_k\delta^{(\ell+1)}_k\cdot W^{(\ell+1)}_{kj}\,\sigma'\big(z^{(\ell)}_j\big),
$$

because $z^{(\ell+1)}_k=\sum_jW^{(\ell+1)}_{kj}\,\sigma(z^{(\ell)}_j)+b^{(\ell+1)}_k$. In matrix form,

$$
\boxed{\boldsymbol\delta^{(\ell)}=\Big(\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\Big)\odot\sigma'\big(\mathbf{z}^{(\ell)}\big).}
$$

This is the **backward recursion**: multiply the error signal by the *transposed* weights (spreading blame to the units that contributed) and by the local derivative of the nonlinearity ($\odot$ is elementwise).

**Step 3: parameter gradients.** Since $z^{(\ell)}_j=\sum_kW^{(\ell)}_{jk}a^{(\ell-1)}_k+b^{(\ell)}_j$,

$$
\frac{\partial\mathcal{L}}{\partial W^{(\ell)}_{jk}}=\delta^{(\ell)}_j\,a^{(\ell-1)}_k,\qquad
\frac{\partial\mathcal{L}}{\partial b^{(\ell)}_j}=\delta^{(\ell)}_j,
$$

i.e.,

$$
\boxed{\nabla_{\mathbf{W}^{(\ell)}}\mathcal{L}=\boldsymbol\delta^{(\ell)}\,\mathbf{a}^{(\ell-1)\top},\qquad\nabla_{\mathbf{b}^{(\ell)}}\mathcal{L}=\boldsymbol\delta^{(\ell)}.}
$$

A weight's gradient is *(blame at its output unit) × (activity at its input unit)*: a Hebbian-looking outer product.

### 9.2.2 The batched version, with shapes

With a batch of $B$ examples stacked as rows, $\mathbf{A}^{(\ell-1)}\in\R^{B\times d_{\ell-1}}$:

| Quantity | Formula | Shape |
|---|---|---|
| Pre-activation | $\mathbf{Z}^{(\ell)}=\mathbf{A}^{(\ell-1)}\mathbf{W}^{(\ell)\top}+\mathbf{1}\mathbf{b}^{(\ell)\top}$ | $B\times d_\ell$ |
| Activation | $\mathbf{A}^{(\ell)}=\sigma(\mathbf{Z}^{(\ell)})$ | $B\times d_\ell$ |
| Delta (output, mean loss) | $\boldsymbol\Delta^{(L)}=(\mathbf{P}-\mathbf{Y})/B$ | $B\times d_L$ |
| Delta (recursion) | $\boldsymbol\Delta^{(\ell)}=(\boldsymbol\Delta^{(\ell+1)}\mathbf{W}^{(\ell+1)})\odot\sigma'(\mathbf{Z}^{(\ell)})$ | $B\times d_\ell$ |
| Weight gradient | $\nabla_{\mathbf{W}^{(\ell)}}\mathcal{L}=\boldsymbol\Delta^{(\ell)\top}\mathbf{A}^{(\ell-1)}$ | $d_\ell\times d_{\ell-1}$ |
| Bias gradient | $\nabla_{\mathbf{b}^{(\ell)}}\mathcal{L}=\boldsymbol\Delta^{(\ell)\top}\mathbf{1}$ | $d_\ell$ |

Note the sum over the batch is *inside* the matrix product $\boldsymbol\Delta^\top\mathbf{A}$.

### 9.2.3 Cost: why training is $\approx3\times$ the forward pass

The forward pass performs one matrix multiplication per layer ($2Bd_\ell d_{\ell-1}$ FLOPs). The backward pass performs *two* per layer: $\boldsymbol\Delta\mathbf{W}$ (to propagate the delta) and $\boldsymbol\Delta^\top\mathbf{A}$ (the weight gradient), each the same cost as the forward multiplication. So training costs $\approx3\times$ the forward pass, which is the "$6ND$" rule of Chapter 6: $2ND$ forward, $4ND$ backward. Memory: the backward pass needs the stored activations $\mathbf{A}^{(\ell-1)}$ and masks $\sigma'(\mathbf{Z}^{(\ell)})$ from the forward pass; **activation memory grows with depth × batch × width**, motivating activation checkpointing (recompute instead of store).

!!! math "Derivation check: why transposes?"
    Think of the Jacobian of layer $\ell+1$ with respect to layer $\ell$'s activation: $\partial\mathbf{z}^{(\ell+1)}/\partial\mathbf{a}^{(\ell)}=\mathbf{W}^{(\ell+1)}$ (shape $d_{\ell+1}\times d_\ell$). The chain rule for a *scalar* loss multiplies a *row* gradient on the left: $(\partial\mathcal{L}/\partial\mathbf{a}^{(\ell)})^\top=(\partial\mathcal{L}/\partial\mathbf{z}^{(\ell+1)})^\top\mathbf{W}^{(\ell+1)}$. Transposing gives the column-vector form $\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}$. **Backprop is a vector–Jacobian product** (Chapter 3, §3.2.4): the gradient is carried as a vector (never as a matrix), which is why a *single* backward pass gives the gradient with respect to *all* parameters.

---

## 9.3 Reverse-mode automatic differentiation on general graphs

Backprop for MLPs is a special case of **reverse-mode automatic differentiation** on a **computational graph**: nodes are tensors, edges are operations. The algorithm:

1. **Forward pass.** Evaluate each node in topological order, recording for every operation what it needs for its local derivative (inputs, outputs, masks).
2. **Backward pass.** Initialize $\bar{\mathcal{L}}=\partial\mathcal{L}/\partial\mathcal{L}=1$. Visit nodes in *reverse* topological order. For each node $v$ with children $c$ (nodes that consume $v$), accumulate

$$
\bar v\;=\;\sum_{c}\bar c\;\frac{\partial c}{\partial v}\qquad(\text{the multivariate chain rule: sum over all paths}).
$$

Each operation implements one function: given the upstream gradient $\bar c$, return the gradients for its inputs.

**Fan-out requires accumulation.** If a value is used twice (a residual connection, a shared weight, a repeated token), its gradient is the *sum* of contributions from each use. Forgetting to accumulate (overwriting instead of adding) is a classic bug.

**Forward-mode versus reverse-mode.** For $f:\R^n\to\R^m$, forward-mode computes a Jacobian–vector product $\mathbf{J}\mathbf{v}$ in one pass (so a full Jacobian costs $n$ passes); reverse-mode computes a vector–Jacobian product $\mathbf{u}^\top\mathbf{J}$ in one pass (full Jacobian: $m$ passes). Deep learning has $n=$ millions–trillions of parameters and $m=1$ (a scalar loss), so **reverse-mode gives all $n$ partial derivatives in one backward pass**, at a constant-factor overhead over the forward pass (the "cheap gradient principle").

### 9.3.1 Implementation from scratch

The following 60-line scalar engine (the `Value` class) implements exactly this algorithm. Each operation records its parents and a closure that adds its contribution to the parents' gradients; `backward()` performs a topological sort and a reverse sweep. We verify it against central finite differences, use it to train a small MLP (2–8–8–1, 105 parameters) on a two-spiral problem, and confirm that a manual, batched, matrix-form backprop for a ReLU/softmax-cross-entropy MLP matches PyTorch autograd to $10^{-16}$.

??? example "Full code: `code/ch09_autograd.py` (click to expand)"
    ```python
    --8<-- "code/ch09_autograd.py"
    ```

Output:

```text
A. autograd: [-1.238432, -0.91255, 0.837255]  finite differences: [-1.238432, -0.91255, 0.837255]
   scalar-autograd MLP (2-8-8-1, 105 params): final loss 0.143, train accuracy 0.97
B. manual backprop vs torch autograd: loss 1.642553 vs 1.642553; max gradient difference = 5.6e-17
C. RMS activation at layer 50 and RMS gradient at layer 1 (width 256, ReLU):
   N(0,1)                   activation RMS =  1.269e+53   gradient RMS =  4.499e+53
   N(0,1/n) (Xavier-like)   activation RMS =  7.895e-08   gradient RMS =  2.800e-07
   N(0,2/n) (He)            activation RMS =  2.649e+00   gradient RMS =  9.394e+00
```

**The three checks you should perform on any new layer or loss**: (i) finite-difference gradient check on tiny inputs; (ii) agreement with a trusted framework; (iii) overfit a single batch (Chapter 6).

---

## 9.4 Vanishing and exploding gradients, and initialization

### 9.4.1 The product of Jacobians

The gradient reaching an early layer is a *product* of per-layer Jacobians:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{a}^{(\ell)}}=\Big(\prod_{k=\ell+1}^{L}\mathbf{W}^{(k)\top}\,\mathrm{diag}\big(\sigma'(\mathbf{z}^{(k-1)})\big)\Big)\,\frac{\partial\mathcal{L}}{\partial\mathbf{a}^{(L)}} .
$$

If the typical factor has gain $g<1$ in each layer, the gradient shrinks like $g^{L-\ell}$ (**vanishing**); if $g>1$ it grows like $g^{L-\ell}$ (**exploding**). With sigmoid ($\sigma'\le1/4$) vanishing is nearly guaranteed in deep networks; this was the main obstacle to training deep networks before ReLU, careful initialization, normalization, and residual connections. Recurrent networks multiply the *same* matrix many times, making the problem acute (Chapter 11).

### 9.4.2 Variance-preserving initialization (He, Xavier)

Choose initial weights so that signal and gradient scales neither shrink nor grow across layers. Take $\mathbf{W}^{(\ell)}$ with i.i.d. zero-mean entries of variance $\sigma_w^2$, inputs $a_k^{(\ell-1)}$ with second moment $\E[a^2]$ and independent of weights. Then

$$
\Var\big(z^{(\ell)}_j\big)=\sum_{k=1}^{n_\text{in}}\Var(W_{jk})\,\E[a_k^2]=n_\text{in}\,\sigma_w^2\,\E[a^2].
$$

For ReLU, $a=\max(0,z)$ with $z$ symmetric about zero, so $\E[a^2]=\tfrac12\E[z^2]=\tfrac12\Var(z^{(\ell-1)})$. Hence

$$
\Var(z^{(\ell)})=\frac{n_\text{in}\sigma_w^2}{2}\,\Var(z^{(\ell-1)}),
$$

and the variance is preserved across layers if **$\sigma_w^2=2/n_\text{in}$** (**He initialization**, He et al., 2015). The analogous argument for the backward pass (preserving gradient variance) gives $\sigma_w^2=2/n_\text{out}$; **Xavier/Glorot** initialization, $\sigma_w^2=2/(n_\text{in}+n_\text{out})$ (for tanh-like activations with $\E[a^2]\approx\Var(z)$ giving $1/n_\text{in}$ per direction), compromises between the two.

**Numerical confirmation (Part C of the code).** Width 256, 50 ReLU layers:

- $\sigma_w^2=1$: per-layer gain $\sqrt{256/2}=11.3$, so after 50 layers activations and gradients are $\sim11.3^{50}\approx10^{52.7}$ (observed: $1.3\times10^{53}$ and $4.5\times10^{53}$): **explosion**.
- $\sigma_w^2=1/n$: per-layer gain $\sqrt{1/2}=0.707$, so $0.707^{50}\approx3\times10^{-8}$ (observed: $7.9\times10^{-8}$ and $2.8\times10^{-7}$): **vanishing**, because a factor of 2 is lost to the zeroed half of ReLU units at each layer.
- $\sigma_w^2=2/n$ (He): activation RMS 2.6 and gradient RMS 9.4 after 50 layers: **stable** (the residual drift is a finite-size/correlation effect).

!!! tip "A rule worth memorizing"
    *A mis-scaled initialization is exponential in depth.* A 10% error per layer in gain is a factor of $1.1^{100}\approx10^4$ in a 100-layer network. This is why initialization details (scaled residual branches, zero-init of output projections) appear in the appendix of every large-model paper.

---

## 9.5 Normalization and residual connections: engineering gradient flow

### 9.5.1 Normalization layers

**LayerNorm** (Ba et al., 2016) normalizes each example's feature vector $\mathbf{x}\in\R^d$ across features:

$$
\mathrm{LN}(\mathbf{x})=\boldsymbol\gamma\odot\frac{\mathbf{x}-\mu(\mathbf{x})}{\sqrt{\sigma^2(\mathbf{x})+\epsilon}}+\boldsymbol\beta,\qquad
\mu=\tfrac1d\sum_ix_i,\quad\sigma^2=\tfrac1d\sum_i(x_i-\mu)^2 .
$$

**RMSNorm** drops the mean subtraction: $\mathbf{x}/\sqrt{\tfrac1d\sum_ix_i^2+\epsilon}\odot\boldsymbol\gamma$. **BatchNorm** (Ioffe & Szegedy, 2015) normalizes each feature across the *batch*.

**Why they help.** (i) They re-center and rescale activations at every layer, so the scale of the pre-activations does not drift through depth (§9.4); (ii) $\mathrm{LN}(c\,\mathbf{x})=\mathrm{LN}(\mathbf{x})$ for $c>0$: the output is *invariant to the scale of the weights before it*, so the gradient is orthogonal to the weight vector and the effective learning rate adapts as weight norms grow (Arora et al., 2019); (iii) they improve the conditioning of the loss (Chapter 3).

**BatchNorm and biology: a pitfall.** BatchNorm uses *batch statistics* during training and *running averages* at inference. This makes a prediction for one example depend on *which other examples are in the batch*. In biological data, batches often *are* experimental batches. A model with BatchNorm trained on mixed-donor or mixed-batch mini-batches can encode batch information in its normalization statistics, and at test time on a different composition (a single new donor, a different assay), the running statistics are mismatched. LayerNorm or group-level normalization avoids per-batch coupling and is the default in transformers for this reason (Worked Example 9.2).

### 9.5.2 Residual connections

A **residual block** computes $\mathbf{y}=\mathbf{x}+F(\mathbf{x})$ (He et al., 2016). Its Jacobian is

$$
\frac{\partial\mathbf{y}}{\partial\mathbf{x}}=\mathbf{I}+\frac{\partial F}{\partial\mathbf{x}},
$$

so the gradient to $\mathbf{x}$ contains the *identity* term: the upstream gradient flows through unchanged *in addition to* whatever $F$ contributes. Across $L$ blocks, $\prod_\ell(\mathbf{I}+\mathbf{J}_\ell)$ expands into a sum over $2^L$ paths, including the all-identity path, so gradients reach the earliest layer undiminished. Veit et al. (2016) interpret residual networks as ensembles of shallow paths. This is why *every* modern deep architecture (ResNets, transformers, AlphaFold's Evoformer, diffusion U-Nets) is built from residual blocks. In **pre-LN transformers**, $\mathbf{y}=\mathbf{x}+F(\mathrm{LN}(\mathbf{x}))$ keeps the identity path completely clean and is markedly more stable at depth than post-LN, which is why most large models use it.

**Biological intuition for residual stacks.** A residual stream is a "communication channel" onto which each layer *writes small updates*; later layers read what earlier layers wrote. Mechanistic interpretability (Chapters 18, 48) exploits this additive structure: the final representation is a sum of layer contributions, which can be attributed.

---

## 9.6 Regularization and the practice of training networks

- **Weight decay** (Gaussian prior, Chapter 4) with AdamW; typical $10^{-2}$–$10^{-1}$.
- **Dropout.** Randomly zero each unit with probability $1-p$ during training; scale surviving units by $1/p$ ("inverted dropout") so expectations match at test time. Interpretations: noise injection, approximate averaging over an ensemble of $2^n$ subnetworks, and, for linear models, an adaptive L2 penalty (Wager et al., 2013).
- **Early stopping** (an implicit regularizer: Chapter 3, §3.8).
- **Data augmentation** encodes known invariances: reverse-complement, random shifts, in silico mutation noise, subsampling reads (cell-level count downsampling).
- **Optimization practice:** AdamW, warmup, cosine decay, gradient clipping, mixed precision (Chapter 3, §3.5).

---

## 9.7 Biological interpretation: what the network is and is not

1. **A network is a differentiable program**: a composition of simple differentiable operations with learned constants. Its "units" are *features of the data*, not neurons of a cell; interpretation is an empirical question (Chapters 18, 48).
2. **Biological neural networks versus artificial ones.** The name suggests an analogy that is thin. Real neurons spike, have dendritic computation, local learning rules, and no obvious *weight transport* (backprop requires the backward pass to use the transpose of the forward weights). Whether and how the brain approximates gradient-based credit assignment is an active research area (Lillicrap et al., 2020; Whittington & Bogacz, 2019), relevant to the NeuroAI thread in Chapter 53.
3. **Gene regulatory "networks" are not neural networks.** Treating a gene-regulatory graph as a neural network with nodes = genes can be a useful *parameterization* (Chapter 16) but is not a claim about the biological mechanism.

---

## 9.8 Worked research examples

!!! example "Worked Research Example 9.1: A deep convolutional model on 100-kb sequences does not train: gradients are $10^{-9}$ in the first layers"
    **Situation.** You build a 40-layer dilated CNN to predict chromatin tracks from 100 kb of DNA. After a day of training the loss has plateaued near its initial value. A hook on gradient norms shows per-layer gradient RMS spanning $10^{-2}$ at the top layers to $10^{-9}$ at the bottom.

    **Question.** What are the candidate explanations and experiments?

    **Reasoning.**

    1. *What does the gradient profile say?* Exponential decay with depth: a signature of a gain $g<1$ per layer ($g^{40}\approx10^{-7}$ implies $g\approx0.67$). That is exactly Part C of the code for sigmoid/ReLU with too-small initialization.
    2. *Candidate causes (ranked by prior probability).* (H1) **Initialization**: weights drawn with variance $1/n$ for a ReLU network (loses a factor $\sqrt2$ per layer). (H2) **Missing residual connections** or normalization. (H3) **Saturating activations** (tanh/sigmoid) with large pre-activations. (H4) **Dead ReLUs** from a too-large learning rate. (H5) A **bug** (detached graph, in-place operation). (H6) The target is *genuinely unpredictable* from the input (low ceiling; Chapter 1), so the plateau is real.
    3. *Experiments, in order of cost.* (a) Overfit one batch (separates bugs and optimization from signal-to-noise); (b) print per-layer activation RMS at initialization: a gain $\ne1$ identifies H1/H2; (c) fraction of exactly-zero activations per layer (H4); (d) re-initialize with He and add residual connections plus LayerNorm: if the gradient profile flattens and loss moves, H1/H2 are confirmed; (e) train a *shallower* control on the same data (4 layers): if it learns a nontrivial signal quickly, depth/initialization is at fault and H6 is unlikely.
    4. *Predictions.* Under H1/H2, fixing initialization and adding residuals changes the gradient profile *before* any change in loss; under H6, even a perfectly conditioned network plateaus at a loss equal to the noise floor; under H5, one batch cannot be overfit.

    **Expert analysis.** A gradient profile is a *diagnostic instrument*, and the exponential form of the decay points to a multiplicative cause (gain per layer), which fits initialization and architecture rather than data. The key habit: *measure the quantity the hypothesis predicts (per-layer gains) before changing anything.* In practice most "deep model won't train" reports in genomics trace to H1/H2 or to H5.

!!! example "Worked Research Example 9.2: The same sample gets different predictions depending on what else is in the batch"
    **Situation.** A model that predicts disease status from single-cell profiles uses BatchNorm. At evaluation, predictions for one patient's cells change by 10 percentage points depending on whether they are evaluated alone, together with other patients' cells, or together with healthy controls.

    **Reasoning.**

    1. *Mechanism.* In eval mode BatchNorm uses running statistics from training, but implementations or workflows that put it in train mode, or that recompute statistics on the evaluation batch, make outputs depend on batch composition. Even in eval mode, the running averages encode the *training* batch composition (a mixture of donors/batches).
    2. *Why biology makes it worse.* Training mini-batches are drawn from a mixture of donors and batches; the BatchNorm statistics are partly a *batch/donor signature*. The network may learn to *use deviations from the batch mean* as a feature: such "relative" features transfer poorly to a new donor whose cells are all alike in their technical signature.
    3. *Experiment.* Evaluate the same cells in (i) eval mode, (ii) train-mode statistics, (iii) with different companions; measure prediction variance. Replace BatchNorm with LayerNorm or GroupNorm; compare.
    4. *Interpretation.* If variance disappears with LayerNorm, the batch coupling was a source of leakage-like information. If performance on held-out donors *improves*, the BatchNorm model had been exploiting batch-composition signals.

    **Expert analysis.** This is an instance of the *generalization gap* (G-G) arising from architecture rather than data: a component that couples examples at inference violates the independence assumption of the evaluation. Normalization choice is a modeling decision with biological consequences.

---

## 9.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: debugging a training run as a sequence of experiments"
    Treat each symptom as an *observation*, generate hypotheses, and pick the cheapest discriminating experiment.

    | Symptom | Likely hypotheses | Cheapest discriminating experiment |
    |---|---|---|
    | Loss is NaN/inf after a few steps | Learning rate too high; log of zero; overflow in fp16; divide-by-zero in normalization | Print the first batch where it occurs; lower LR by 10×; use bf16; add $\epsilon$ |
    | Loss flat at $\log K$ (chance) | Not learning: wrong labels/loss; dead network; LR too low; bug | Overfit one batch; check label alignment; check gradient norms |
    | Loss decreases, then spikes | Bad batch; Adam $\beta_2$ too low; LR too high | Log the batch IDs at spikes; lower $\beta_2$ sensitivity; clip gradients |
    | Train loss ≪ validation loss | Overfitting; leakage in train; distribution shift | Learning curve vs. $n$; regularize; check split overlap |
    | Both losses plateau high | Under-capacity; noise floor; objective mismatch | Scale model; compute ceiling; per-class loss (Chapter 3) |
    | Gradient norm → 0 early | Vanishing (init/saturation); dead ReLUs | Per-layer gradients; activation statistics |
    | Gradient norm grows without bound | Exploding gain; no clipping; unstable architecture | Clip; check init; add normalization |
    | Validation metric noisy epoch to epoch | Small validation set; high LR; non-determinism | Bootstrap CI; average checkpoints; more seeds |

    **Distinguishing fundamental from implementation problems.** *Implementation* problems respond to one-line changes (init, LR, normalization) and show up in diagnostics at initialization. *Fundamental* problems (noise floors, missing information, objective mismatch) persist across all reasonable settings and are predicted by information arguments (Chapters 1, 5). The notebook's real lesson: **never change two things at once**.

---

## 9.10 Connections

- **Backward:** Backprop is Chapter 3's chain rule realized as a vector–Jacobian product; the softmax-CE delta is Chapter 3's $\mathbf{p}-\mathbf{y}$; He initialization is Chapter 2's variance/gain reasoning; the $6ND$ cost is Chapter 6's.
- **Forward:** CNNs (Chapter 10) are MLPs with weight sharing and locality; RNNs (Chapter 11) multiply the same Jacobian repeatedly (vanishing gradient in time); transformers (Chapter 12) are residual blocks of attention and MLPs with LayerNorm; interpretability uses gradients as attributions (Chapter 18); diffusion uses backprop only through a simple regression loss (Chapter 15).

!!! takeaways "Key takeaways"
    1. A network is composed linear maps and nonlinearities; universal approximation is an *existence* statement, not a learning guarantee.
    2. **Backprop:** $\boldsymbol\delta^{(\ell)}=(\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)})\odot\sigma'(\mathbf{z}^{(\ell)})$ and $\nabla_{\mathbf{W}^{(\ell)}}\mathcal{L}=\boldsymbol\delta^{(\ell)}\mathbf{a}^{(\ell-1)\top}$. It is a **vector–Jacobian product**; one backward pass yields all gradients at about 2× the forward cost.
    3. General reverse-mode autodiff: forward in topological order, backward in reverse, **accumulating at fan-out**.
    4. Gradients are **products of Jacobians**; gain per layer $\ne1$ is exponential in depth. **He init** ($\sigma_w^2=2/n_\text{in}$) preserves variance for ReLU.
    5. **LayerNorm** and **residual connections** engineer well-behaved scales and gradient highways; BatchNorm couples examples and can import batch composition into predictions.
    6. Debug training with *hypotheses and cheap discriminating experiments*; separate implementation problems from fundamental ones.

---

## Further reading

- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature* 323, 533–536.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A. & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *JMLR* 18, 1–43.
- Karpathy, A. *micrograd* and "The spelled-out intro to neural networks and backpropagation" (educational implementation).
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, chapters 6, 8. MIT Press.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Math. Control Signals Systems* 2, 303–314. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks* 4, 251–257. Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Zhang, C., Bengio, S., Hardt, M., Recht, B. & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR*.
- Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*. He, K., Zhang, X., Ren, S. & Sun, J. (2015). Delving deep into rectifiers. *ICCV*.
- Ioffe, S. & Szegedy, C. (2015). Batch normalization. *ICML*. Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). Layer normalization. arXiv:1607.06450. Zhang, B. & Sennrich, R. (2019). Root mean square layer normalization. *NeurIPS*.
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*. Veit, A., Wilber, M. & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. *NeurIPS*. Xiong, R. et al. (2020). On layer normalization in the transformer architecture. *ICML*.
- Arora, S., Li, Z. & Lyu, K. (2019). Theoretical analysis of auto rate-tuning by batch normalization. *ICLR*.
- Wager, S., Wang, S. & Liang, P. (2013). Dropout training as adaptive regularization. *NeurIPS*.
- Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J. & Hinton, G. (2020). Backpropagation and the brain. *Nature Reviews Neuroscience* 21, 335–346. Whittington, J. C. R. & Bogacz, R. (2019). Theories of error back-propagation in the brain. *Trends in Cognitive Sciences* 23, 235–250.
