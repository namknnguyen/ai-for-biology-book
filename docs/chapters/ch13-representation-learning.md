# Chapter 13. Representation Learning and Self-Supervision

!!! abstract "Chapter at a glance"
    **Motivation.** Labeled biological data are scarce and expensive; unlabeled sequences, structures, and expression profiles are abundant. Self-supervised learning turns unlabeled data into *representations*, which every foundation model in Part VII is built to produce. The central lesson of this chapter, and a thread through the whole book, is that **the pretraining objective decides what the representation keeps and what it forgets.**
    **Prerequisites.** Chapters 2, 5, 8, 9, 12.
    **You will be able to:** (1) define what makes a representation good using information-theoretic language; (2) derive autoencoders, autoregressive and masked language modeling, and their relationships; (3) **derive the InfoNCE bound** and explain why its estimates saturate at $\log K$; (4) explain non-contrastive methods (BYOL, VICReg) and predictive-latent methods (JEPA); (5) design positive pairs as statements about nuisance variation, including how *evolution* supplies natural augmentations; (6) evaluate representations with probes, and know their pitfalls.

---

## 13.1 What is a representation, and what makes one good?

A **representation** is a map $\phi_\theta:x\mapsto\mathbf{h}$ from raw data to a vector (or a set of vectors, one per token). A representation is not good or bad in itself: it is good *for a purpose*. Useful desiderata, each expressible in the language of Chapter 5:

| Property | Informal | Formal statement |
|---|---|---|
| **Sufficiency** | Retains what the downstream task needs | $\MI(\mathbf{h};y)\approx\MI(x;y)$ |
| **Invariance** | Discards nuisance factors $n$ (batch, depth, background) | $\MI(\mathbf{h};n)\approx0$ |
| **Minimality** | Retains nothing else | small $\MI(\mathbf{h};x)$ given sufficiency |
| **Accessibility** | Task information is easily decodable (e.g., linearly) | high *usable* information (Xu et al., 2020) |
| **Smoothness** | Nearby inputs map to nearby representations | Lipschitz-like |
| **Factorization** | Distinct factors occupy distinct directions | (not identifiable in general; Chapter 8) |

Two consequences. First, *"good representation" is meaningless without naming the task and the nuisance.* Second, by the data-processing inequality, $\MI(\mathbf{h};y)\le\MI(x;y)$: **a representation can only lose information about any $y$; its virtue is that it makes the remaining information accessible and the nuisance removable.**

**The pretraining dilemma.** Pretraining optimizes an objective $\mathcal{L}_\text{pre}$ on unlabeled data, without knowing the downstream $y$. The optimal representation for $\mathcal{L}_\text{pre}$ keeps what $\mathcal{L}_\text{pre}$ rewards. Whether that overlaps with the downstream need is **the objective gap** (G-O, Chapter 1). The experiment in §13.5 makes this concrete.

---

## 13.2 Autoencoders

### 13.2.1 Definition and the linear case

An **autoencoder** has an encoder $\mathbf{h}=f_\phi(x)$ and decoder $\hat x=g_\theta(\mathbf{h})$ trained to minimize reconstruction error $\ell(x,\hat x)$, with a **bottleneck** ($\dim\mathbf{h}<\dim x$) or other regularizer. For squared error with linear $f,g$, the optimal solution spans the **top-$k$ principal subspace** (Chapter 2: Eckart–Young); the nonlinear autoencoder generalizes PCA to curved manifolds. Reconstruction loss for categorical data (one-hot DNA) is cross-entropy; for counts, a negative-binomial likelihood.

**Why a bottleneck is needed.** An overcomplete autoencoder can copy its input (the identity), learning nothing. Regularizers: low dimension; *sparsity* of $\mathbf{h}$; *denoising* (reconstruct clean $x$ from corrupted $\tilde x$; Vincent et al., 2008, which implicitly learns the score $\nabla\log p(x)$, §15); *contractive* penalties; *variational* regularization (the VAE's KL term; Chapter 14).

### 13.2.2 What a reconstruction objective rewards

Reconstruction loss is a sum over input dimensions. The encoder allocates its limited capacity **in proportion to loss reduction per code dimension**: it encodes whatever *explains the most variance (or likelihood)*, whether or not that is what we care about (Chapter 2's PCA lesson: variance is not importance). A strong nuisance factor (GC bias, sequencing depth, batch) that affects every input dimension is *cheap to encode and reduces reconstruction loss a lot*, so reconstruction objectives preferentially represent it. The experiment below shows this directly.

---

## 13.3 Predictive self-supervision for sequences

Sequence data admit a natural supervisory signal: *predict part of the data from the rest.*

### 13.3.1 Autoregressive language modeling

By the chain rule of probability, $p_\theta(x_{1:L})=\prod_{t=1}^Lp_\theta(x_t\mid x_{<t})$. Train by maximizing the log-likelihood (minimizing cross-entropy, Chapter 5), using a **causal** model (causal mask in attention; unidirectional recurrence). This is **exact maximum likelihood on the joint distribution**: it minimizes $\KL{p_\text{data}}{p_\theta}$ (Chapter 5). Properties:

- It supports *generation*: sample $x_1\sim p(x_1)$, $x_2\sim p(x_2\mid x_1)$, and so on.
- It gives the *sequence likelihood* $\log p_\theta(x)$, usable for scoring (e.g., variant effects: $\log p_\theta(x')-\log p_\theta(x)$, Chapter 5, §5.6.4).
- Each position sees only its *left* context, so single-position embeddings are less informative about the right context. Order matters: reading DNA 5′→3′ is a convention; the opposite strand is the reverse complement (Chapter 10).
- Examples: ProGen (proteins), ProtGPT2, Evo and Evo 2 (DNA; Chapter 32), GPN-style variants.

### 13.3.2 Masked language modeling

Mask a random subset $M$ of positions (typically 15%) and predict the masked tokens from the *unmasked* context in both directions:

$$
\mathcal{L}_\text{MLM}(\theta)=-\E_{x}\ \E_{M}\Big[\sum_{i\in M}\log p_\theta\big(x_i\mid x_{\setminus M}\big)\Big].
$$

The encoder is *bidirectional* (no causal mask). In a strict sense this is **not** maximum likelihood of a joint distribution; it is a *pseudo-likelihood* (Besag, 1975) objective: a sum of conditional log-probabilities. Its population optimum is that $p_\theta(x_i\mid x_{\setminus M})$ equals the true conditional $p_\text{data}(x_i\mid x_{\setminus M})$ for every mask. The sequence of full conditionals $\{p(x_i\mid x_{\setminus i})\}_i$ determines the joint distribution under mild conditions (the Hammersley–Clifford theory behind Markov random fields), so *the MLM optimum encodes the same statistical information as the joint*.

**The link to Potts models.** If the data are generated by a Potts model $p(x)\propto\exp\big(\sum_ih_i(x_i)+\sum_{i<j}J_{ij}(x_i,x_j)\big)$ (Chapter 29), the full conditional is

$$
p(x_i\mid x_{\setminus i})\propto\exp\Big(h_i(x_i)+\sum_{j\ne i}J_{ij}(x_i,x_j)\Big),
$$

a softmax over the 20 amino acids with logits that are sums of *pairwise couplings to the other positions*. Fitting couplings by maximizing the pseudo-likelihood of exactly this conditional is **pseudo-likelihood direct-coupling analysis** (plmDCA; Ekeberg et al., 2013). A masked protein language model is therefore a *nonlinear, much more expressive* generalization of plmDCA: a single network predicts any masked residue from the others, trained on millions of unrelated families. [[E]] (as the mathematical correspondence; the *extent* to which transformers realize Potts-like computation is [[S]]).

**Use for scoring variants.** Because the MLM defines conditionals, a variant at position $i$ from $a$ to $b$ can be scored by the *masked marginal* $\log p_\theta(x_i=b\mid x_{\setminus i})-\log p_\theta(x_i=a\mid x_{\setminus i})$ (Meier et al., 2021, ESM-1v), or by the *pseudo-log-likelihood* of the full mutated sequence. These are consistent proxies for $\Delta\log p$ under the Sella–Hirsh reading of Chapter 5, subject to the caveats there.

**Biological MLM-like objectives.** *Masked MSA positions* (AlphaFold 2 auxiliary loss), *masked gene expression* (scBERT, Geneformer, scGPT; Chapter 38), *masked structure tokens* (ESM3), *masked nucleotides* (DNABERT, Nucleotide Transformer; Chapter 32).

**Pitfalls of MLM design.** (i) *Information leakage through overlapping tokens*: with overlapping $k$-mer tokens, a masked $k$-mer is partly revealed by its neighbors; DNABERT-style models needed contiguous masks to prevent trivial prediction. (ii) *The mask ratio and strategy change what is learned*: random single-token masking rewards local statistics; contiguous span masking rewards longer-range structure. (iii) *Train–test mismatch*: the $\texttt{[MASK]}$ token never appears at test time (BERT's 80/10/10 mixture mitigates).

### 13.3.3 Other pretext tasks

*Denoising/span corruption* (T5); *permutation language modeling* (XLNet); *next-sentence-style* tasks (rarely helpful); *discrete diffusion* models (Chapter 15) that learn to denoise over a corruption process and unify masked and autoregressive modeling. In biology: *predict structure from sequence* (a "tokenized structure" auxiliary as in ESM3), *predict conservation* from sequence, *predict one assay from another*.

!!! lens "Research lens: autoregressive vs. masked objectives for biology"
    **Autoregressive:** exact joint likelihood; generation; scoring; but unidirectional features. **Masked:** bidirectional context for every position (better embeddings for classification/structure); pseudo-likelihood only; generation requires iterative procedures. **Information used:** both use only co-occurrence within the sequence. **Information ignored:** anything not statistically linked to the sequence itself (cell type, environment, assay), the same missing-conditioning problem as in Chapter 1's Worked Example 1.2. **Failure modes:** both are dominated by the *most predictable* parts of the data (repeats, low-complexity regions), and both can be *trivially* solved by memorization of near-duplicates in the training set.

---

## 13.4 Contrastive learning

### 13.4.1 The idea

Instead of predicting the input, learn a representation in which **two views of the same underlying object are close, and views of different objects are far**. We need (i) a notion of *positive pair* $(x,x^+)$ (two views of the same thing), (ii) *negatives* $x^-_k$ (views of other things), and (iii) a loss that pulls positives together and pushes negatives apart.

### 13.4.2 InfoNCE and its derivation

Let $s(x,y)=\langle\phi(x),\psi(y)\rangle/\tau$ be a similarity score (with temperature $\tau$). For one positive pair $(x,y^+)$ and $K$ negatives $y_1^-,\dots,y_K^-$ (drawn from the marginal), the **InfoNCE** loss (Oord et al., 2018) is

$$
\mathcal{L}_\text{NCE}=-\E\Big[\log\frac{e^{s(x,y^+)}}{e^{s(x,y^+)}+\sum_{k=1}^Ke^{s(x,y_k^-)}}\Big],
$$

a $(K+1)$-way softmax cross-entropy asking the model to *identify the true partner* among candidates.

!!! math "Derivation: InfoNCE lower-bounds mutual information"
    Suppose $(x,y^+)\sim p(x,y)$ and $y^-_k\sim p(y)$ independently. The Bayes-optimal critic for the $(K+1)$-way identification problem assigns the true partner probability proportional to the density ratio $r(x,y)=p(y\mid x)/p(y)$; equivalently, the optimal score satisfies $e^{s^\star(x,y)}\propto r(x,y)$. Plugging it in,

    $$
    \mathcal{L}^\star_\text{NCE}=\E\log\Big[1+\frac{p(y^+)}{p(y^+\mid x)}\sum_{k=1}^K\frac{p(y_k^-\mid x)}{p(y_k^-)}\Big].
    $$

    Replacing the sum over negatives by its expectation, which equals $K\cdot\E_{y\sim p(y)}\big[p(y\mid x)/p(y)\big]=K$ (the density ratio averages to one under the marginal), gives

    $$
    \mathcal{L}^\star_\text{NCE}\approx\E\log\Big[1+K\frac{p(y^+)}{p(y^+\mid x)}\Big]\ \ge\ \E\log\Big[K\frac{p(y^+)}{p(y^+\mid x)}\Big]=\log K-\MI(X;Y).
    $$

    Rearranged: $\boxed{\MI(X;Y)\ge\log K-\mathcal{L}_\text{NCE}}$ (with $K+1$ in place of $K$ for the exact count; Oord et al., 2018; the replacement of the sum by its mean is made rigorous in Poole et al., 2019).

**Reading the bound.** (i) Minimizing InfoNCE maximizes a *lower bound on the mutual information* between the two views. (ii) The bound **cannot exceed $\log K$**: with $K=255$ negatives the loss floors at 0 and the certified information at $\log256=5.55$ nats. Contrastive estimates of high mutual information require many negatives, which is why large batches or memory banks (MoCo) matter. (iii) The *temperature* $\tau$ controls how sharply the loss focuses on hard negatives. (iv) The representation is optimized to preserve **exactly the information shared between the two views**: *what the views share is signal; what they do not share is discarded.*

**Alignment and uniformity.** Wang & Isola (2020) showed that, on the unit hypersphere, the contrastive loss asymptotically optimizes (i) *alignment*: positive pairs have nearby embeddings, and (ii) *uniformity*: embeddings spread uniformly on the sphere (maximizing retained information). *Dimensional collapse* (embeddings occupying a low-dimensional subspace) is a failure of uniformity.

### 13.4.3 Positive pairs are statements about nuisance

The choice of positives *is* the modeling assumption. Any variation *between* the two views is declared nuisance and is pushed out of the representation; any information *shared* is preserved. In vision, "views" are crops, color jitters, and rotations. In biology:

| Positive pair | Declares as nuisance | Preserves | Risk |
|---|---|---|---|
| Two orthologous sequences from related species | Neutral divergence | Conserved (functional) content | Discards lineage-specific function; phylogenetic bias |
| Same cell, two modalities (RNA and ATAC) | Modality-specific technical effects | Cell state shared across modalities | Misses modality-specific biology |
| Same cell with random gene dropout | Dropout/depth noise | Expression pattern | Does not remove batch or donor effects |
| Same cell type across donors | Donor variation | Cell-type identity | Removes real inter-individual biology (e.g., eQTL effects) |
| Sequence and its predicted structure | Representation format | Fold-level information | Depends on the predictor's biases |
| A protein and its text description | Language vs. sequence | Function-level concepts | Annotation bias and noise |
| Wild-type and a random mutant | A few residues | Overall fold | Washes out the single-mutation effects you may need |

!!! rhyme "Structural rhyme: contrastive learning ↔ natural selection ↔ conservation"
    Evolution is a giant *contrastive* process: lineages diverge, neutral sites change freely, and **functional sites are conserved**. A pair of orthologous sequences is therefore a *naturally augmented positive pair*, in which selection decides which positions are "content" (conserved) and which are "nuisance" (free to change). Contrastive learning on orthologs, like MSA-based and language-model objectives, uses evolution to *define* the invariances. The same logic explains why *phylogenetic structure in the corpus is a double-edged sword* (Chapters 21, 34): it supplies the signal (conservation) and the confound (shared ancestry).

---

## 13.5 An experiment: the objective decides what is kept

We build synthetic "loci." Each carries a conserved signal $s\in\{0,1,2,3\}$ (one of four 6-mer motif variants at fixed positions) embedded in a background with a **locus-specific GC bias** (nuisance, drawn uniformly from 0.1 to 0.9). Two orthologs of a locus share $s$ and have independently drawn backgrounds. We train $k$-dimensional encoders on **unlabeled** one-hot sequences using (a) reconstruction (autoencoder) and (b) InfoNCE with orthologs as positives, then fit **linear probes** for the signal $s$ (4-way accuracy; chance 0.25) and for the nuisance (GC content; $R^2$).

```python
--8<-- "code/ch13_representation.py"
```

Output:

```text
raw one-hot input, linear probe for the conserved signal s: accuracy = 1.000 (chance 0.25)
  k |   autoencoder: signal acc / GC R^2 |   contrastive: signal acc / GC R^2
  1 |            0.196 /  0.517       |            0.351 /  0.081
  2 |            1.000 /  0.909       |            0.995 /  0.001
  4 |            1.000 /  0.911       |            0.960 /  0.009
 16 |            1.000 /  0.917       |            1.000 /  0.042
```

**Reading the table.**

- The raw input contains the signal perfectly (probe accuracy 1.0). Any loss is the representation's fault, by the data-processing inequality.
- **Tight bottleneck ($k=1$):** the autoencoder spends its single dimension on **GC content** ($R^2=0.52$) and **loses the signal** (accuracy 0.196, at chance level 0.25: it is *below* chance by finite-sample noise). Reconstruction loss reduction per dimension is largest for the global nuisance. The contrastive encoder retains partial signal information (0.351: a 1-d normalized embedding is only a sign, so it can separate about two groups) and almost no GC ($R^2=0.08$).
- **Adequate capacity ($k\ge2$):** both retain the signal. But the **autoencoder also retains the nuisance** ($R^2\approx0.91$), because reconstruction rewards it, while the **contrastive representation discards it** ($R^2\le0.04$), because the positives differ in GC, so GC cannot help identify the partner.

**What this demonstrates.** (1) *Reconstruction objectives allocate scarce capacity by loss reduction*, which favors high-variance nuisance. (2) *Contrastive objectives retain what the views share and forget what differs*: invariance is created by the choice of positives. (3) Neither is "better": an autoencoder's retention of nuisance is a *feature* if the nuisance matters downstream (GC may be the relevant variable for another task), and a bug if it does not. (4) The experiment is the miniature of the foundation-model question (Chapters 17, 32, 38): *what does the pretraining objective make the representation know?*

!!! lens "Research lens: the forgetting audit"
    For any self-supervised representation, list: **(i)** *Retained by construction*: what the objective rewards (check by probing). **(ii)** *Forgotten by construction*: what the positive pairs vary, what the corruption destroys, what the loss under-weights. **(iii)** *Nuisance retained by accident*: variance sources that are cheap to encode (depth, batch, GC, ancestry). **(iv)** *Needed downstream*: map each downstream task to an item in (i)–(iii). A foundation model is promising for a task *only if* the needed information is in (i), and is *dangerous* for a task if it is in (ii).

---

## 13.6 Non-contrastive and latent-predictive methods

Contrastive learning needs negatives and large batches. Alternatives avoid negatives yet avoid *collapse* (the trivial solution where all inputs map to the same vector):

- **BYOL / SimSiam** (Grill et al., 2020; Chen & He, 2021): a student network with a *predictor head* is trained to match a *stop-gradient* target (a slowly updated copy, or the same network); asymmetry prevents collapse.
- **Barlow Twins / VICReg** (Zbontar et al., 2021; Bardes et al., 2022): explicit regularizers. VICReg penalizes (i) low *variance* of each embedding dimension (preventing collapse), (ii) *covariance* between dimensions (decorrelating them, preventing redundancy), and (iii) *invariance*: the distance between positives.
- **Joint-embedding predictive architectures (JEPA)** (LeCun, 2022; Assran et al., 2023): predict the *latent representation* of a masked or future part from the visible part, rather than predicting the raw data. The loss lives in representation space.

**Why latent prediction matters for biology.** Raw biological measurements contain a large irreducible noise component: Poisson counting noise in single-cell data, measurement noise in assays, neutral variation in sequences (§5.6.2). A reconstruction loss in *data space* forces the model to spend capacity on that unpredictable noise. A loss in *latent space* can ignore noise that no representation could predict. This is a principled version of the *objective attack* (A3) and the *reformulation attack* (A8) (Chapter 55): *predict the denoised state, not the noisy observation.* [[H]] for biological models; the method is established in vision, and its adaptation to single-cell and sequence data is an open research direction (Chapters 38, 51).

---

## 13.7 Identifiability of learned representations

Chapter 8 showed that unsupervised latent axes are not identifiable in general. Self-supervision can restore identifiability *under assumptions*. **Nonlinear ICA with auxiliary variables** (Hyvärinen et al., 2019) and **contrastive learning with positive pairs generated from known conditional dependence** (Zimmermann et al., 2021) prove that, if the data are generated by an injective function of independent latents and positive pairs are related by a known latent-noise process, then a contrastive encoder recovers the latents *up to a simple (e.g., linear or permutation) ambiguity*. [[E]] as theorems; the assumptions (injective generator, correct pairing process) are unlikely to hold exactly in biology, so *treat recovered axes as hypotheses*. The practical message is that **how the views are generated determines what, if anything, is identifiable.**

---

## 13.8 Evaluating representations

**Evaluation protocols.**

| Protocol | What it measures | Caveats |
|---|---|---|
| **Linear probe** | Linearly accessible information about $y$ | A weak probe underestimates; a strong probe may do the work itself |
| **$k$-NN probe** | Local geometry | Sensitive to the metric and to near-duplicates (leakage) |
| **Few-shot fine-tuning** | Transfer with limited labels | Confounded by fine-tuning budget and hyperparameters |
| **Zero-shot scoring** | Whether the pretraining objective directly encodes the target (e.g., $\Delta\log p$) | Needs a theory connecting objective to target (Chapter 5, §5.6.4) |
| **Intrinsic geometry** | Effective rank, alignment/uniformity, spectrum decay | Not tied to any task |

**Pitfalls.** (i) **Probe complexity:** a very expressive probe can learn the task from near-raw input, so *report the same probe on raw features and on a random encoder*. Hewitt & Liang (2019) introduced **control tasks** (probe a random labeling of the same data) and *selectivity* (task accuracy minus control accuracy) to separate representation quality from probe capacity. (ii) **Dataset shortcuts:** if the probe task is solvable from batch or ancestry, the representation may succeed for the wrong reason. (iii) **Leakage between pretraining and probe data**: if the probe's test sequences are homologs of pretraining sequences, "transfer" may be memorization.

The table in §13.5 applies these principles: the raw input is the reference, and the autoencoder at $k=1$ is *worse than the raw input* by exactly the amount the objective forgot.

---

## 13.9 Worked research examples

!!! example "Worked Research Example 13.1: A single-cell autoencoder's first latent dimension is sequencing depth"
    **Situation.** You train an autoencoder on log-normalized single-cell counts. The leading latent axis correlates 0.9 with the number of detected genes per cell, and downstream clustering is poor for rare populations.

    **Question.** Why, and what should change?

    **Reasoning.**

    1. *What does the loss reward?* Reconstruction of expression across all genes. Sequencing depth ("library size") multiplies *every* gene's expected count; it is the single largest source of coherent variance across the matrix (Chapter 2: the PCA lesson).
    2. *What does the information view say?* Encoding depth in one latent dimension reduces reconstruction loss for all 20,000 genes simultaneously: *maximal loss reduction per code dimension*. A cell-type axis helps only a subset of genes.
    3. *Alternative explanations.* (H1) Insufficient normalization (log-normalization does not fully remove depth effects, especially for low counts; Chapter 30). (H2) The likelihood is mismatched (Gaussian on log counts vs. counts with Poisson/NB variance depending on depth; Chapter 4). (H3) Depth is genuinely correlated with biology (larger cells have more RNA). (H4) Batch confounds depth.
    4. *Experiments.* (a) Regress depth out and test whether the first axis disappears; (b) train with a **count likelihood and an explicit library-size covariate** (scVI-style; the library-size term absorbs depth so the latent need not encode it); (c) check whether biological variables (cell size, cell cycle) predict depth within a cell type; (d) look at whether depth differs by batch.
    5. *Predictions.* Under H1/H2, adding an explicit depth term and an NB likelihood removes the axis and improves rare-population resolution. Under H3, some depth–biology association persists after modeling (e.g., within cell type).

    **Expert analysis.** The depth axis is an example of *nuisance retained because it is cheap to reconstruct*. The fix is to change the **objective's likelihood and inputs** so that depth is explained by a dedicated variable, or to use an **invariance-inducing** objective (e.g., augment by subsampling counts so depth is a nuisance between views). Both are instances of the *objective attack* (A3).

!!! example "Worked Research Example 13.2: Designing positives for a single-cell contrastive model"
    **Situation.** You want embeddings of cells that capture *cell state* for downstream perturbation response prediction. Candidate positive pairs: (P1) the same cell with two random count-downsamplings; (P2) cells of the same annotated type; (P3) the same cell measured with RNA and ATAC (multiome); (P4) the same cell type from different donors.

    **Reasoning.**

    1. *Each is a different nuisance declaration.* P1 removes depth/dropout. P2 removes within-type variation (and trusts the labels, circular if the labels came from the same data). P3 removes modality-specific technical effects and keeps *shared* state. P4 removes donor and batch effects, but also real donor differences.
    2. *What does the downstream task need?* Perturbation response depends on *within-type state* (cell cycle, signaling activity) and on *context* (donor genotype can modify response). P2 would *erase* the within-type state that matters; P4 would erase donor context.
    3. *Which are safest?* P1 (low risk) and P3 (the shared latent state between assays is exactly "cell state" in the measurement view of Chapter 1).
    4. *Test.* A **forgetting audit** (§13.5): probe the embeddings for cell cycle phase, interferon response, donor ID, depth, batch; check which survive.
    5. *What would change your mind?* If probes show cell-cycle phase is gone under P2, the representation is unsuitable for perturbation prediction regardless of benchmark scores on cell-type annotation.

    **Expert analysis.** The best positives are *the ones that vary what you can defend as nuisance and keep what you need.* This is why multimodal alignment (P3) is a recurring theme (Chapter 40), and why embedding quality on *cell-type annotation* (which P2 optimizes) can be anti-correlated with quality on *state-dependent* tasks.

---

## 13.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: from a failed transfer to an objective hypothesis"
    **Observation.** A masked-language-model protein encoder gives excellent features for fold classification but is unhelpful for predicting thermostability.

    **Step 1: What did the objective reward?** Recovering masked residues from context rewards *evolutionary conservation* and *pairwise co-variation* (§13.3.2). Fold identity is strongly encoded in family-level statistics.

    **Step 2: What did it ignore?** Thermostability is a property of the *environment of the organism*: thermophile proteins differ from mesophile homologs by systematic residue-composition shifts (more charged residues, fewer thermolabile ones). In a pooled corpus, these shifts are small relative to family variation, and the loss gives little reason to encode them, *unless the species or growth temperature is given*.

    **Step 3: Hypotheses.** (H1) The information is present but not linearly accessible (a non-linear probe works). (H2) The information is absent because the objective never needed it. (H3) The thermostability benchmark is confounded by family (fold-level labels).

    **Step 4: Experiments.** Compare linear vs. non-linear probes; check *within-family* prediction (families with both thermophile and mesophile members; this removes the family confound); condition the model on organism growth temperature as a token (does the LM *loss* improve?). If the loss improves when conditioning on temperature, the model's *distribution* depends on temperature but the *representation without it* cannot know it.

    **Step 5: Candidate objectives.** *Conditional* language modeling with environment tokens; *contrastive* pairs across homologs from different temperature classes (making the *difference* the signal rather than the nuisance); *multi-task* supervised fine-tuning.

    **Distinguishing limitation types.** H1 is a *readout* limitation; H2 an *objective* limitation (G-O); H3 an *evaluation* limitation. Experiments distinguish them cheaply. The lesson: *transfer failures are information about the objective.*

---

## 13.11 Connections

- **Backward:** PCA/linear autoencoders (Chapter 2); KL, MI, and the data-processing inequality (Chapter 5); ELBO and amortized inference (Chapter 8); attention encoders (Chapter 12); Potts models (preview of Chapter 29).
- **Forward:** VAEs and diffusion as generative self-supervision (Chapters 14–15); foundation-model objectives and scaling (Chapter 17); probing, sparse autoencoders, and mechanistic interpretability (Chapters 18, 48); genomic/protein/cell LMs and their objectives (Chapters 32, 34, 38); multimodal contrastive alignment (Chapter 40); the objective attack in idea generation (Chapter 55).

!!! takeaways "Key takeaways"
    1. A representation is good *for a task and a nuisance*; by the data-processing inequality it can only lose information, so what matters is **which information is kept and how accessible it is**.
    2. **Autoencoders** allocate capacity by reconstruction-loss reduction, which favors high-variance nuisance (depth, GC, batch). Linear AE = PCA.
    3. **Autoregressive** LMs maximize exact joint likelihood; **masked** LMs maximize pseudo-likelihood of full conditionals, a nonlinear generalization of plmDCA.
    4. **InfoNCE** lower-bounds mutual information, $\MI\ge\log K-\mathcal{L}$, so it saturates at $\log K$ and needs many negatives; it keeps what views share and forgets what differs.
    5. **Positive pairs encode nuisance assumptions**; orthologs are *natural augmentations* supplied by evolution.
    6. **Non-contrastive/JEPA** methods predict latents rather than raw data, avoiding the modeling of irreducible noise: a promising but largely unexplored direction in biology.
    7. Evaluate with probes **and controls** (raw features, random encoders, control tasks); run a **forgetting audit**.
    8. In the experiment, a $k=1$ autoencoder lost the signal (accuracy 0.196) while keeping GC ($R^2=0.52$); at $k\ge2$ it kept both signal and nuisance (GC $R^2\approx0.91$), whereas the contrastive encoder kept the signal and dropped the nuisance ($R^2\le0.04$).

---

## Further reading

- Bengio, Y., Courville, A. & Vincent, P. (2013). Representation learning: a review and new perspectives. *IEEE TPAMI* 35, 1798–1828.
- Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. *ICML*.
- Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). BERT. *NAACL*. Radford, A. et al. (2018, 2019). GPT and GPT-2 technical reports.
- Besag, J. (1975). Statistical analysis of non-lattice data. *The Statistician* 24, 179–195. Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M. & Aurell, E. (2013). Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models. *Physical Review E* 87, 012707.
- Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *NeurIPS*. Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118, e2016239118.
- Oord, A. van den, Li, Y. & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748. Poole, B., Ozair, S., van den Oord, A., Alemi, A. & Tucker, G. (2019). On variational bounds of mutual information. *ICML*.
- Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*. He, K. et al. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR*. Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML*.
- Wang, T. & Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. *ICML*.
- Grill, J.-B. et al. (2020). Bootstrap your own latent. *NeurIPS*. Chen, X. & He, K. (2021). Exploring simple Siamese representation learning. *CVPR*. Bardes, A., Ponce, J. & LeCun, Y. (2022). VICReg. *ICLR*. LeCun, Y. (2022). A path towards autonomous machine intelligence. Assran, M. et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR*.
- Hyvärinen, A., Sasaki, H. & Turner, R. (2019). Nonlinear ICA using auxiliary variables and generalized contrastive learning. *AISTATS*. Zimmermann, R. S., Sharma, Y., Schneider, S., Bethge, M. & Brendel, W. (2021). Contrastive learning inverts the data generating process. *ICML*.
- Hewitt, J. & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP*. Xu, Y. et al. (2020). A theory of usable information under computational constraints. *ICLR*.
- Ji, Y., Zhou, Z., Liu, H. & Davuluri, R. V. (2021). DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. *Bioinformatics* 37, 2112–2120.
