# Chapter 2. Linear Algebra as the Language of Representation

!!! abstract "Chapter at a glance"
    **Motivation.** Every object in this book (a DNA sequence, a cell, a protein, a genotype matrix, an attention layer) becomes a vector, matrix, or tensor before a model touches it. Linear algebra is the language in which *representations* are defined, compared, compressed, and analyzed.
    **Prerequisites.** Basic familiarity with vectors and matrices. This chapter is compact: it teaches only what later chapters use.
    **You will be able to:** (1) read tensor shapes fluently and compute the cost of a matrix operation; (2) derive PCA/SVD and the Eckart–Young theorem; (3) explain what "low-rank structure" means for cells, genotypes, and attention; (4) recognize when a leading principal component is a batch effect or population structure rather than the biology you care about; (5) use high-dimensional geometry (near-orthogonality, concentration) to reason about embeddings.

---

## 2.1 Why linear algebra comes first

Three facts organize the chapter.

1. **Biological data are arrays.** A one-hot DNA sequence of length $L$ is an $L \times 4$ array. A single-cell experiment is an $N \times G$ cells-by-genes matrix. A genotype dataset is an $N \times M$ individuals-by-variants matrix. A batch of protein sequences embedded by a model is a $B \times L \times d$ tensor.
2. **Learning is mostly linear maps and nonlinearities composed.** A neural network layer is $\mathbf{h}' = \sigma(\mathbf{W}\mathbf{h} + \mathbf{b})$. An attention layer is three linear maps and a softmax. Understanding what linear maps *can* and *cannot* do (their rank, their singular values, their invariances) tells you what the network can and cannot represent.
3. **Compression is the first and most reliable tool of biological data analysis.** PCA (a truncated SVD) is applied before almost every genomics or single-cell analysis, and its failures (confusing batch with biology, ancestry with disease) are among the most important lessons in the field.

---

## 2.2 Vectors, matrices, tensors, and their biological meaning

### 2.2.1 Notation, shapes, and the dot product

A **vector** $\mathbf{x} \in \R^d$ is an ordered list of $d$ numbers. The **dot product** $\mathbf{a}^\top \mathbf{b} = \sum_i a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\angle(\mathbf{a},\mathbf{b})$ measures alignment. Two uses recur constantly:

- *Similarity.* Two embeddings with a large dot product (or cosine) are "similar" under the representation.
- *Scoring against a template.* The score of a sequence window against a motif is a dot product between the one-hot window and a weight matrix (§2.2.3).

A **matrix** $\mathbf{A} \in \R^{m \times n}$ represents a linear map $\R^n \to \R^m$. Matrix–vector multiplication $\mathbf{y} = \mathbf{A}\mathbf{x}$ takes dot products of $\mathbf{x}$ with each *row* of $\mathbf{A}$; equivalently, $\mathbf{y}$ is a linear combination of the *columns* of $\mathbf{A}$ weighted by the entries of $\mathbf{x}$. Both views are useful:

$$
\mathbf{y} = \mathbf{A}\mathbf{x}
\quad\Longleftrightarrow\quad
y_i = \mathbf{a}_{i,:}^\top \mathbf{x}
\quad\Longleftrightarrow\quad
\mathbf{y} = \sum_{j} x_j\, \mathbf{a}_{:,j}.
$$

Matrix–matrix multiplication $\mathbf{C} = \mathbf{A}\mathbf{B}$, with $\mathbf{A}\in\R^{m\times n}$, $\mathbf{B}\in\R^{n\times p}$, gives $\mathbf{C}\in\R^{m\times p}$ with $C_{ik} = \sum_j A_{ij}B_{jk}$. The cost is $mnp$ multiply–adds, i.e., about $2mnp$ floating-point operations (FLOPs). **This single formula is the basis for nearly every compute estimate in the book.**

A **tensor** is simply an array with more than two axes. We always write the shape: $\mathbf{X} \in \R^{B\times L\times d}$. In code, the batched matrix product `torch.matmul(A, B)` with shapes `(B, m, n)` and `(B, n, p)` returns `(B, m, p)`.

!!! tip "Shape discipline"
    A large fraction of research-code bugs are shape bugs that happen to broadcast silently. Habit: annotate every tensor with its shape in a comment, name axes (`B, L, d`), and assert shapes at module boundaries. `einsum` (below) makes axis semantics explicit.

### 2.2.2 `einsum`: the notation that makes shapes explicit

Einstein summation writes a tensor contraction by naming the indices and summing over repeated ones that do not appear in the output:

$$
C_{ik} = \sum_j A_{ij} B_{jk}\quad\leftrightarrow\quad \texttt{einsum("ij,jk->ik", A, B)}.
$$

Attention scores for one head are `einsum("bqd,bkd->bqk", Q, K)`: for each batch element, the dot product between every query vector and every key vector. A multi-head version is `einsum("bhqd,bhkd->bhqk", Q, K)`. Writing the contraction this way makes it obvious which axis is summed (the feature axis `d`), which axes are kept (query and key positions), and what the output costs: $O(B H L^2 d_h)$ FLOPs. Chapter 12 builds on exactly this.

### 2.2.3 Biological objects as arrays

!!! bio "Biology for modeling: sequences, motifs, and matrices"
    **What is it?** A DNA sequence $x_{1:L}$ over $\{A,C,G,T\}$.
    **Information it contains.** The order of nucleotides; everything else (chromatin state, which cell type, which transcription factors are present) is *absent* from the sequence itself.
    **How generated and measured.** Sequenced by short or long reads, assembled and aligned to a reference (Chapters 20, 27). Errors, coverage biases, and reference biases enter before any model sees it.
    **Computational representation.** One-hot encoding $\mathbf{X} \in \{0,1\}^{L\times 4}$ with $X_{i,a} = 1$ iff $x_i = a$. Embedding the four letters as $d$-dimensional vectors turns the sequence into $\mathbf{H}\in\R^{L\times d}$.
    **What varies.** Individuals differ at about 0.1% of positions (Chapter 20); species differ more; sequence composition varies along the genome.
    **What ML can learn.** Motifs, composition, periodicities, long-range co-occurrence patterns.
    **What ML cannot observe from the sequence alone.** The cellular context that determines which sequence features are *used*.

**A motif score is a dot product.** A *position weight matrix* (PWM) $\mathbf{W}\in\R^{w\times 4}$ for a transcription factor of width $w$ scores the window starting at position $i$ as

$$
s_i = \sum_{j=1}^{w}\sum_{a\in\mathcal{A}} W_{j,a}\, X_{i+j-1,\,a} \;=\; \langle \mathbf{W},\, \mathbf{X}_{i:i+w}\rangle_F,
$$

the Frobenius inner product of the PWM with the one-hot window. Sliding the window across the sequence is a *cross-correlation*; a convolutional layer with $C$ filters of width $w$ computes $C$ such motif scores at every position. This is why convolutional networks were the natural first deep model for genomics (Chapter 10), and it already tells us the key *inductive bias*: the same template is applied everywhere (translation equivariance).

**A single-cell dataset is a matrix.** $\mathbf{X}\in\R_{\ge0}^{N\times G}$ with $X_{ng}$ the count of transcripts of gene $g$ detected in cell $n$. Rows are cells (observations), columns are genes (features). This matrix is *sparse* (most entries are zero), *noisy* (counts are sampled), and *non-negative*. Standard analysis normalizes it (Chapter 30) and applies PCA.

**A genotype matrix** $\mathbf{G}\in\{0,1,2\}^{N\times M}$ counts the number of copies of the alternate allele that individual $n$ carries at variant $m$. Statistical genetics (Chapter 26) is largely linear algebra on $\mathbf{G}$ and a phenotype vector $\mathbf{y}\in\R^N$.

---

## 2.3 Linear maps: rank, null space, and what a layer cannot do

A matrix $\mathbf{A}\in\R^{m\times n}$ has a **column space** (the set of all outputs $\mathbf{A}\mathbf{x}$), a **null space** (inputs mapped to zero), and a **rank** $r = \dim(\text{column space}) \le \min(m,n)$. Rank-nullity says $\dim(\text{null space}) = n - r$.

Why do these matter for models?

- A linear layer $\mathbf{W}\in\R^{d_{\text{out}}\times d_{\text{in}}}$ **discards** the $d_{\text{in}} - r$ input directions in its null space. Information in those directions cannot influence the output. If $d_{\text{out}} < d_{\text{in}}$ this is forced. Bottlenecks in networks are deliberate rank constraints.
- A product of matrices has rank at most the minimum rank of its factors: $\rank(\mathbf{A}\mathbf{B}) \le \min(\rank\mathbf{A}, \rank\mathbf{B})$. In attention, the score matrix $\mathbf{Q}\mathbf{K}^\top$ with $\mathbf{Q},\mathbf{K}\in\R^{L\times d_h}$ has rank at most $d_h$, which is typically much smaller than $L$ for genomic sequences with $L$ in the thousands. Whatever pairwise structure a single head expresses is at most rank $d_h$ *before* the softmax (the softmax is a nonlinearity that can raise the effective rank, which is one reason it matters).
- Fitting a linear model $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ with $N < G$ samples cannot identify $\mathbf{w}$ uniquely because $\mathbf{X}$ has a non-trivial null space; regularization (Chapter 7) selects among the infinitely many solutions.

---

## 2.4 Eigendecomposition and the SVD

### 2.4.1 Eigenvalues for symmetric matrices

For a symmetric matrix $\mathbf{S}\in\R^{n\times n}$, there exist an orthonormal basis of eigenvectors $\mathbf{q}_1,\dots,\mathbf{q}_n$ and real eigenvalues $\lambda_1\ge\dots\ge\lambda_n$ with $\mathbf{S}\mathbf{q}_i=\lambda_i\mathbf{q}_i$, so $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ (the spectral theorem). The eigenvalues are the stretch factors along orthogonal axes. A symmetric matrix is *positive semi-definite* (PSD) iff all $\lambda_i\ge0$; covariance matrices and Gram matrices are always PSD.

The **Rayleigh quotient** characterizes the extreme eigenvalue:

$$
\lambda_1 = \max_{\|\mathbf{v}\|=1} \mathbf{v}^\top\mathbf{S}\mathbf{v},
$$

attained at $\mathbf{v}=\mathbf{q}_1$. *Proof sketch.* Write $\mathbf{v}=\sum_i c_i\mathbf{q}_i$ with $\sum c_i^2=1$. Then $\mathbf{v}^\top\mathbf{S}\mathbf{v}=\sum_i\lambda_i c_i^2\le\lambda_1\sum c_i^2=\lambda_1$, with equality iff all weight is on eigenvectors with eigenvalue $\lambda_1$. $\square$

### 2.4.2 The singular value decomposition

Every real matrix $\mathbf{A}\in\R^{m\times n}$ (not just symmetric ones) has a **singular value decomposition**

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,
\qquad \sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0,
$$

with orthonormal left singular vectors $\mathbf{u}_i\in\R^m$, orthonormal right singular vectors $\mathbf{v}_i\in\R^n$, and $r=\rank\mathbf{A}$. The SVD says: *every linear map is a rotation, a stretch along orthogonal axes, and another rotation.*

**Relation to eigendecomposition.** $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$ and $\mathbf{A}\mathbf{A}^\top=\mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$. So the right singular vectors are eigenvectors of $\mathbf{A}^\top\mathbf{A}$ and $\sigma_i^2$ are its eigenvalues. This is how one proves existence: take the eigendecomposition of the PSD matrix $\mathbf{A}^\top\mathbf{A}$, set $\sigma_i=\sqrt{\lambda_i}$, $\mathbf{v}_i$ the eigenvectors, and define $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$. One checks that the $\mathbf{u}_i$ are orthonormal: $\mathbf{u}_i^\top\mathbf{u}_j=\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j/(\sigma_i\sigma_j)=\sigma_j^2\delta_{ij}/(\sigma_i\sigma_j)=\delta_{ij}$.

### 2.4.3 Eckart–Young: truncated SVD is the best low-rank approximation

!!! math "Theorem (Eckart–Young–Mirsky) and proof sketch"
    Among all matrices $\mathbf{B}$ of rank at most $k$, the minimizer of $\|\mathbf{A}-\mathbf{B}\|_F$ is the truncated SVD $\mathbf{A}_k=\sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$, and

    $$
    \min_{\rank\mathbf{B}\le k}\|\mathbf{A}-\mathbf{B}\|_F^2=\|\mathbf{A}-\mathbf{A}_k\|_F^2=\sum_{i>k}\sigma_i^2 .
    $$

    *Sketch (spectral-norm version).* $\|\mathbf{A}-\mathbf{A}_k\|_2=\sigma_{k+1}$. For any rank-$k$ matrix $\mathbf{B}$, its null space has dimension at least $n-k$, so it intersects the $(k+1)$-dimensional span of $\mathbf{v}_1,\dots,\mathbf{v}_{k+1}$ non-trivially; take a unit vector $\mathbf{z}$ in that intersection. Then $\|(\mathbf{A}-\mathbf{B})\mathbf{z}\|=\|\mathbf{A}\mathbf{z}\|\ge\sigma_{k+1}$ because $\mathbf{z}$ lies in the span of the top $k+1$ right singular vectors. Hence $\|\mathbf{A}-\mathbf{B}\|_2\ge\sigma_{k+1}$. The Frobenius case follows from a similar argument applied to each tail singular value (e.g., via Weyl's inequalities). $\square$

Eckart–Young is the reason for the *rank-and-reconstruction-error* view of everything that follows: the squared error of the best rank-$k$ summary of a matrix is the sum of the squared singular values you throw away. The code below verifies the identity numerically.

---

## 2.5 PCA, derived

Let $\mathbf{X}\in\R^{N\times D}$ hold $N$ observations (cells, individuals) of $D$ features, with columns centered (mean zero). The sample covariance is $\mathbf{S}=\frac{1}{N-1}\mathbf{X}^\top\mathbf{X}$.

**Goal.** Find a unit direction $\mathbf{v}$ such that the projections $\mathbf{X}\mathbf{v}\in\R^N$ have maximum variance.

**Derivation.** The variance of the projection is $\frac{1}{N-1}\|\mathbf{X}\mathbf{v}\|^2=\mathbf{v}^\top\mathbf{S}\mathbf{v}$. By the Rayleigh-quotient result, the maximizer over unit vectors is the top eigenvector $\mathbf{q}_1$ of $\mathbf{S}$, and the variance achieved is $\lambda_1$. Repeating under the constraint of orthogonality to previous directions yields $\mathbf{q}_2,\mathbf{q}_3,\dots$ with variances $\lambda_2,\lambda_3,\dots$ The *principal components* (scores) are the columns of $\mathbf{X}\mathbf{Q}_k$.

**Equivalence with SVD.** With the SVD $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$: $\mathbf{S}=\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top/(N-1)$, so the principal directions are the right singular vectors, the eigenvalues are $\lambda_i=\sigma_i^2/(N-1)$, and the PC scores are $\mathbf{U}\boldsymbol{\Sigma}$. In practice one computes the SVD of the centered data (never forms $\mathbf{S}$).

**Equivalence with reconstruction.** By Eckart–Young, projecting onto the top $k$ PCs gives the best rank-$k$ reconstruction in squared error. So PCA is simultaneously (i) maximum-variance projection, (ii) minimum-reconstruction-error projection, and (iii) a *linear autoencoder* with tied weights (which is the key to Chapter 13: a nonlinear autoencoder generalizes it).

**Probabilistic view.** PCA is the maximum-likelihood solution of a Gaussian latent-variable model $\mathbf{x}=\mathbf{W}\mathbf{z}+\boldsymbol{\epsilon}$, $\mathbf{z}\sim\Normal(0,\mathbf{I}_k)$, $\boldsymbol{\epsilon}\sim\Normal(0,\sigma^2\mathbf{I})$ as $\sigma\to0$ (probabilistic PCA; Chapter 8). The same model with a diagonal noise covariance is *factor analysis*. Single-cell models such as scVI (Chapter 30) are nonlinear, count-valued relatives.

!!! rhyme "Structural rhyme: PCA ↔ factor analysis ↔ linear autoencoder ↔ latent semantic analysis ↔ GWAS ancestry correction"
    These are the same low-rank factorization applied to different matrices: cells×genes, individuals×SNPs, documents×words, sequences×positions (as in the principal components of a protein family's one-hot alignment, which recovers sectors of coevolving residues). What differs is *what the rows share* and *what the leading components mean*. Whenever you see "top-$k$ components," ask: *what is the shared latent structure, and is it the one I care about?*

---

## 2.6 What do the leading components mean? Two cautionary tales

PCA finds directions of **maximum variance**, not directions of **biological interest**. Any source of large, structured variation in the matrix will dominate the top components. Two examples recur throughout genomics, and the code in this chapter reproduces both.

```python
--8<-- "code/ch02_linear_algebra.py"
```

Output (seed 1):

```text
rank 3: ||A-A_k||_F^2 =  1440.483   sum of discarded sigma^2 =  1440.483
rank 5: ||A-A_k||_F^2 =    19.609   sum of discarded sigma^2 =    19.609
rank 8: ||A-A_k||_F^2 =    15.185   sum of discarded sigma^2 =    15.185

PCA genotypes: N=200, M=4949.  top eigenvalues: [11.05  1.39  1.37  1.37], noise edge ~ 1.44
PC1 means: pop0 =  -16.51, pop1 =   16.51, within-pop sd = 1.00
PC1: |corr with cell type| = 0.08, |corr with batch| = 1.00
PC2: |corr with cell type| = 0.99, |corr with batch| = 0.02
PC3: |corr with cell type| = 0.00, |corr with batch| = 0.00
```

**Cautionary tale 1: population structure.** Individuals in a genotype matrix are related through shared ancestry. Two populations that diverged recently differ in allele frequencies by an amount measured by $F_{ST}$ (Chapter 21). The first principal component of the standardized genotype matrix separates the two populations (PC1 means $\mp 16.5$ with within-population standard deviation 1.0). The noise eigenvalues of a standardized $N\times M$ matrix of pure noise concentrate below the **Marchenko–Pastur edge** $(1+\sqrt{N/M})^2\approx1.44$; only the leading eigenvalue (11.05) rises far above it, so *one* component is real. This is the standard rationale for using the top few PCs as covariates in GWAS to avoid spurious associations driven by ancestry (Chapter 26), and for counting significant PCs by comparing to the random-matrix edge.

**Cautionary tale 2: batch effects.** In the simulated expression matrix, 60% of genes shift between two technical batches, while only 10% differ between two cell types. PC1 has correlation 1.00 with *batch* and 0.08 with cell type; the biology of interest only appears in PC2. A pipeline that simply "takes the first PC as the main axis of cell variation" would have been wrong. Real datasets are harder: batch and biology are often *confounded* (all samples of one condition were processed together), in which case no linear algebra can separate them; this is a design problem, not a modeling problem (Chapters 25, 44, 45).

!!! lens "Research lens: PCA"
    **Assumes:** structure of interest is *linear* and carries *large variance*; features are comparably scaled. **Uses:** second-order statistics only. **Ignores:** nonlinear manifold structure, rare cell types (low variance), higher-order dependencies, labels. **Fails when:** batch or ancestry dominates variance; low-abundance populations carry the signal; features are on different scales (so scaling choices decide the answer); the data are counts and the variance depends on the mean.

---

## 2.7 High-dimensional geometry: why embeddings work and where they strain

Modern representations live in $\R^d$ with $d$ from 64 to 8,192. A few facts about high dimensions explain behavior that otherwise seems mysterious.

### 2.7.1 Near-orthogonality of random vectors

Let $\mathbf{u},\mathbf{v}$ be independent random unit vectors in $\R^d$ (for instance, drawn from an isotropic Gaussian and normalized). Then $\E[\mathbf{u}^\top\mathbf{v}]=0$ and $\Var(\mathbf{u}^\top\mathbf{v})=1/d$. Therefore the cosine between random directions is of order $1/\sqrt d$, and in $d=512$, two random directions have $|\cos|\approx0.044$.

**Consequence.** A $d$-dimensional space can hold *many more than $d$* directions that are pairwise nearly orthogonal. By the Johnson–Lindenstrauss lemma, $n$ points can be mapped into $O(\varepsilon^{-2}\log n)$ dimensions preserving all pairwise distances up to $1\pm\varepsilon$. This is why a 1,280-dimensional embedding can represent hundreds of thousands of distinct sequences, and, in interpretability (Chapters 18, 48), why networks seem to represent *more features than dimensions* by *superposition*: assigning nearly-orthogonal (rather than exactly orthogonal) directions to many sparsely active features.

### 2.7.2 Concentration of distances

For $\mathbf{x},\mathbf{y}$ i.i.d. with independent coordinates, the squared distance $\|\mathbf{x}-\mathbf{y}\|^2$ concentrates around its mean with relative fluctuations of order $1/\sqrt d$. In high dimension, "nearest" and "farthest" neighbors have nearly the same distance unless the data have low *intrinsic* dimension. This is why nearest-neighbor retrieval in raw gene space (20,000 genes) is poor, why PCA before clustering is not just a speed trick but a *denoising* step, and why a good representation is one in which meaningful neighbors are well separated *relative to this concentration*.

### 2.7.3 Intrinsic dimension

Data in $\R^D$ may lie near a $k$-dimensional manifold with $k\ll D$ (cells vary along a handful of continuous axes: cell cycle, differentiation, signaling states; proteins of a family vary in a low-dimensional space of functional sub-types). The decay of the singular-value spectrum is a crude estimate. Nonlinear methods (UMAP, autoencoders, diffusion maps) estimate manifolds but, as Chapter 13 and Chapter 30 explain, they distort distances in ways that matter for biological interpretation.

---

## 2.8 Least squares, the pseudo-inverse, and ridge: the linear model as geometry

Given $\mathbf{X}\in\R^{N\times D}$ and targets $\mathbf{y}\in\R^N$, least squares minimizes $\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^2$. Setting the gradient to zero gives the normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w}=\mathbf{X}^\top\mathbf{y}$. If $\mathbf{X}^\top\mathbf{X}$ is invertible, $\hat{\mathbf{w}}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$. Geometrically, $\mathbf{X}\hat{\mathbf{w}}$ is the *orthogonal projection* of $\mathbf{y}$ onto the column space of $\mathbf{X}$.

With the SVD $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, the minimum-norm solution is $\hat{\mathbf{w}}=\mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top\mathbf{y}=\sum_i\frac{\mathbf{u}_i^\top\mathbf{y}}{\sigma_i}\mathbf{v}_i$: the **pseudo-inverse**. Notice $1/\sigma_i$: directions with *small singular values* are amplified, so noise in $\mathbf{y}$ along those directions blows up. This is the *ill-conditioning* problem (condition number $\kappa=\sigma_1/\sigma_r$). **Ridge regression** adds $\lambda\|\mathbf{w}\|^2$ and replaces $1/\sigma_i$ by $\sigma_i/(\sigma_i^2+\lambda)$, a *shrinkage filter* that damps the small-$\sigma$ directions:

$$
\hat{\mathbf{w}}_{\text{ridge}}=\sum_i\frac{\sigma_i}{\sigma_i^2+\lambda}\,(\mathbf{u}_i^\top\mathbf{y})\,\mathbf{v}_i .
$$

This formula is worth remembering: it explains why ridge works when features are correlated (small $\sigma_i$ ↔ correlated features), why PCA regression is a hard-thresholded version of ridge, and (Chapter 7) how a regularization strength implements a *prior* on which directions matter. In genomics it is the basis of many linear predictors, including polygenic scores (Chapter 26).

---

## 2.9 Computational cost: what scales and what doesn't

| Operation | Shapes | FLOPs (approx.) | Memory | Where it appears |
|---|---|---|---|---|
| Matrix–vector | $(m\times n)(n)$ | $2mn$ | $mn$ | Single token through a layer |
| Matrix–matrix | $(m\times n)(n\times p)$ | $2mnp$ | $mn+np+mp$ | Everything |
| Full SVD of $m\times n$ ($m\ge n$) | | $O(mn^2)$ | $O(mn)$ | PCA of a cells×genes matrix |
| Truncated SVD, rank $k$ | | $O(mnk)$ | $O(mk+nk)$ | PCA on large matrices (randomized/Lanczos) |
| Attention scores | $(B,H,L,d_h)\cdot(B,H,d_h,L)$ | $2BHL^2d_h$ | $BHL^2$ | Transformers; the long-genome bottleneck |
| Linear layer on a sequence | $(B,L,d)\cdot(d,d')$ | $2BLdd'$ | | MLP blocks |

Two takeaways: (1) PCA is cheap for $10^5$ cells × $10^4$ genes with randomized algorithms but infeasible as a full SVD at $10^8$ cells without streaming; (2) attention cost grows *quadratically* in sequence length $L$ while linear layers grow linearly. At $L=10^6$ (a megabase of nucleotides) full attention would require $10^{12}$ score entries per head per layer: impossible. This single comparison motivates Chapter 11's convolution/state-space alternatives and sparse or hybrid attention (Chapter 12).

---

## 2.10 Worked research examples

!!! example "Worked Research Example 2.1: Is the top principal component biology?"
    **Situation.** You compute PCA on a single-cell expression matrix from 20 donors processed in 4 batches. PC1 explains 18% of variance and cleanly separates the cells into two groups. A colleague says "PC1 is the healthy/diseased axis."

    **Question.** How do you decide what PC1 is?

    **Reasoning.**

    1. *What does PCA optimize?* Variance, not biological relevance. So PC1 is the largest *coherent* source of variation.
    2. *What coherent sources exist besides disease?* Batch/sequencing depth; donor identity; cell-type composition; cell cycle; dissociation stress; sex; ambient RNA.
    3. *What information would discriminate?* Annotate PC scores with each covariate; compute the fraction of PC1 variance explained by each (e.g., $R^2$ of PC1 on batch, donor, depth); check whether the top loading genes are known markers of disease, of stress, or housekeeping genes of technical origin.
    4. *Are disease and batch confounded?* Tabulate disease status by batch. If all diseased samples were in batch 1, **no analysis** can separate them; the claim must be downgraded to [[X]].
    5. *What experiment would settle it?* Re-run with balanced batches; include technical replicates across batches; use a negative-control batch (same donor in two batches) to estimate the batch-only effect on PC1.
    6. *What would you predict under each hypothesis?* If PC1 is disease: it correlates with disease within every batch, and the sign is preserved after batch correction. If PC1 is batch: it vanishes after regressing out batch, and the donors replicated across batches show a batch-only displacement.

    **Expert analysis.** The deepest point is not about PCA: it is that **variance is not a measure of importance**, and that **confounding is a property of the study design**, not of the algorithm. This is the first of many cases in the book where the right response to a surprising representation is an *experiment on the data-generating process*, not a better algorithm.

!!! example "Worked Research Example 2.2: Why are single-cell embeddings poor at rare cell types?"
    **Situation.** A 30-dimensional PCA embedding of a tissue atlas separates the major lineages beautifully but merges two rare immune subtypes that together make up 0.3% of cells.

    **Reasoning.** The rare subtypes differ from each other on genes that vary little *across the whole dataset*, so they contribute little to total variance; the top PCs are dominated by the large populations. By Eckart–Young, the best rank-30 reconstruction in squared error spends its capacity where the mass is. Possible fixes (each attacking a different assumption): (i) subset to the lineage and recompute PCA (changes the population on which variance is measured); (ii) choose features by *discriminative* rather than variance criteria; (iii) use a loss that up-weights rare populations; (iv) use a nonlinear model that can allocate a latent dimension to a rare mode.

    **Lesson.** Reconstruction-based objectives implicitly weight the data by frequency. This is a statement about the *objective* (G-O), and it reappears whenever a self-supervised model's loss is dominated by common patterns (Chapters 13, 32, 38).

---

## 2.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The embedding is 1,280-dimensional, but how many dimensions does it use?\""
    **The observation.** A protein language model produces 1,280-dimensional residue embeddings. You compute the singular values of a matrix of 100,000 such embeddings and find that 95% of the variance lies in 40 dimensions.

    **Tempting conclusion.** "The model uses only 40 dimensions; the rest is waste, so we can compress it."

    **Decompose.** *Variance* is not *information*. A direction with small variance may still be read by a downstream layer that is sensitive to it (a probe can amplify small-variance directions; in networks with LayerNorm or large downstream weights, low-variance directions can carry decisive information). Anisotropy is also common: language-model embeddings often have a few high-variance "rogue" dimensions that dominate PCA without being semantically informative.

    **Hidden assumptions.** (i) Linearity: nonlinear structure may need many linear directions. (ii) The sample (these 100,000 residues) represents the use-case distribution. (iii) Task relevance is proportional to variance.

    **Discriminating experiments.** Project onto the top-$k$ PCs and measure the *downstream task metric* as $k$ varies (not reconstruction error); then do the reverse: *remove* the top-$k$ PCs and measure what remains. If performance is retained when the top PCs are removed, low-variance directions carry task information.

    **Possible outcomes and what each means.** (a) Metric saturates at $k\approx40$ → compression is safe for *this* task. (b) Metric keeps improving to $k=400$ → task information is in low-variance directions. (c) Removing the top 40 barely hurts → the high-variance directions are a nuisance (anisotropy).

    **What this teaches.** *Intrinsic dimension of the representation* and *dimension needed for a task* are different quantities. A claim about either needs the matching experiment.

---

## 2.12 Connections

- **Backward:** none; this is the foundation.
- **Forward:** gradients of matrix functions (Chapter 3); covariance, Gaussians, and probabilistic PCA (Chapters 4, 8); the SVD as the root of *low-rank* (LoRA, Chapter 17) and of *attention as a low-rank-plus-softmax pairwise model* (Chapter 12); population structure correction and LD (Chapters 21, 26); embeddings and probing (Chapters 13, 18); single-cell PCA pipelines (Chapters 25, 30); the coupling matrices of Potts models (Chapter 29).

!!! takeaways "Key takeaways"
    1. Biological data are arrays; always annotate **shapes**, and use **einsum** to make axis semantics explicit. A matrix product costs about $2mnp$ FLOPs.
    2. A linear map's **rank** and **null space** say what it can represent and what it discards; attention scores per head have rank $\le d_h$ before the softmax.
    3. The **SVD** is the master decomposition. **Eckart–Young:** the best rank-$k$ approximation error is $\sum_{i>k}\sigma_i^2$.
    4. **PCA = maximum variance = minimum reconstruction error = linear autoencoder.** It finds *variance*, not *biological meaning*: batch effects and ancestry routinely dominate the top components.
    5. High-dimensional geometry: random vectors are nearly orthogonal ($\cos\sim1/\sqrt d$); distances concentrate; $d$-dimensional spaces can hold many more than $d$ nearly orthogonal features (superposition).
    6. Ridge regression filters singular directions by $\sigma_i/(\sigma_i^2+\lambda)$; ill-conditioning is amplification of small-$\sigma$ directions.

---

## Further reading

- Strang, G. *Linear Algebra and Its Applications*; and *Introduction to Linear Algebra*. A gentle, geometric treatment.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM. The best treatment of the SVD and conditioning.
- Eckart, C. & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika* 1, 211–218.
- Patterson, N., Price, A. L. & Reich, D. (2006). Population structure and eigenanalysis. *PLoS Genetics* 2, e190. Random-matrix theory for counting significant PCs in genotype data.
- Price, A. L. et al. (2006). Principal components analysis corrects for stratification in genome-wide association studies. *Nature Genetics* 38, 904–909.
- Johnson, W. B. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemporary Mathematics* 26.
- Elhage, N. et al. (2022). Toy models of superposition. *Transformer Circuits Thread*. [Chapters 18, 48]
- Leek, J. T. et al. (2010). Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics* 11, 733–739.
- Tipping, M. E. & Bishop, C. M. (1999). Probabilistic principal component analysis. *J. R. Stat. Soc. B* 61, 611–622.
