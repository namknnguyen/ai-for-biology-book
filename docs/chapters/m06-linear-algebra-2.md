# Chapter M6. Linear Algebra II: Spaces, Eigenvalues, and the SVD

!!! abstract "Chapter at a glance"
    **Motivation.** Chapter M5 treated matrices as tables, maps and systems. This chapter explains their *structure*: which outputs a matrix can reach (its column space), which inputs it erases (its null space), which directions it merely stretches (eigenvectors), and how any matrix decomposes into rotate–stretch–rotate (the singular value decomposition). These ideas are the content of principal-component analysis, of evolutionary substitution models, of Markov models of cell-state transitions, of covariance matrices and of every statement of the form "this data has low-dimensional structure".

    **Prerequisites.** Chapter M5. The idea of a linear map and of rank from the previous chapter.

    **You will be able to:** (1) define span, linear independence, basis and dimension and use them to count free parameters; (2) state the four fundamental subspaces and the rank–nullity theorem; (3) interpret least squares as orthogonal projection; (4) compute eigenvalues and eigenvectors of small matrices and use diagonalization to take matrix powers and matrix exponentials; (5) use eigenvalues of a Markov transition matrix to find stationary distributions and mixing times; (6) state the spectral theorem and recognize covariance matrices as symmetric positive semi-definite; (7) state the SVD, the Eckart–Young theorem, and derive PCA; (8) separate "variance explained" from "dimension of the signal".

---

## M6.1 Vector spaces, span, independence, basis, dimension

A **vector space** is a collection of vectors closed under addition and scalar multiplication (any linear combination of members is a member). $\R^d$ is the standard example. A **subspace** is a vector space sitting inside a larger one: a line or a plane through the origin in $\R^3$. Subspaces never exclude the origin; a line not through the origin is not a subspace.

- The **span** of vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ is the set of all their linear combinations: everything reachable from them.
- Vectors are **linearly independent** if none is a linear combination of the others; equivalently, $c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k=\mathbf{0}$ only when all $c_i=0$. Dependent vectors contain redundancy.
- A **basis** of a space is an independent set that spans it. Every basis of a given space has the same number of vectors, the **dimension**. The standard basis of $\R^d$ is the $d$ coordinate vectors, and dimension $d$ means "$d$ free numbers describe a point".

The word *dimension* is the book's tool for counting **degrees of freedom**: how many numbers are needed to specify an object. A cell-by-gene matrix of 20,000 genes is a point in 20,000 dimensions, but if cells lie near a 10-dimensional subspace then 10 numbers per cell (plus the subspace) describe most of what matters. Whether data are *effectively* low-dimensional is the central empirical question behind embeddings and "manifold" claims (Chapters 13, 25, 38).

---

## M6.2 Rank and the four fundamental subspaces

For $\mathbf{A}\in\R^{m\times n}$ four subspaces describe everything the matrix does:

| Subspace | Lives in | Definition | Dimension |
|---|---|---|---|
| **Column space** $\mathcal{C}(\mathbf{A})$ | $\R^m$ | all outputs $\mathbf{A}\mathbf{x}$ (span of the columns) | $r$ |
| **Row space** $\mathcal{C}(\mathbf{A}^\top)$ | $\R^n$ | span of the rows | $r$ |
| **Null space** $\mathcal{N}(\mathbf{A})$ | $\R^n$ | all $\mathbf{x}$ with $\mathbf{A}\mathbf{x}=\mathbf{0}$ (inputs erased) | $n-r$ |
| **Left null space** $\mathcal{N}(\mathbf{A}^\top)$ | $\R^m$ | all $\mathbf{y}$ with $\mathbf{A}^\top\mathbf{y}=\mathbf{0}$ | $m-r$ |

where $r=\operatorname{rank}\mathbf{A}$ is the number of independent columns, which equals the number of independent rows. The **rank–nullity theorem** is the dimension count $\operatorname{rank}+\dim\mathcal{N}(\mathbf{A})=n$: *the dimension of what survives plus the dimension of what is erased equals the input dimension*. The script builds a $6\times8$ matrix as a product $(6\times3)(3\times8)$; it has rank 3, a 5-dimensional null space ($8-3$) and a 3-dimensional left null space ($6-3$).

Two orthogonality facts complete the picture: the null space is *orthogonal* to the row space (an erased input has zero dot product with every row), and the left null space is orthogonal to the column space. So $\R^n$ splits into row space $\oplus$ null space, and $\R^m$ into column space $\oplus$ left null space.

!!! bio "Biology for modeling: counting what is identifiable"
    **What is it?** Whenever a model is linear in its parameters, $\mathbf{y}=\mathbf{A}\boldsymbol\theta$, the null space of $\mathbf{A}$ is exactly the set of parameter changes that leave the data unchanged.

    **Information it contains / discards.** Only the row space of $\mathbf{A}$ is determined by the data. Parameters differing by a null-space vector are indistinguishable.

    **Examples.** In the metabolic example of Chapter M5, the null space of $\mathbf{S}$ is the set of steady-state flux patterns. In a regression with more variants than samples, the null space has dimension at least $p-n$ (Worked example M5.1). In a deep network with a bottleneck, the null space of a weight matrix is the set of input directions the layer ignores.

    **What would change the interpretation.** Real measurements have noise: directions with tiny singular values are *nearly* in the null space, and noise decides whether the data determine them. Rank is exact; effective rank (§M6.6) is what matters in practice.

The rank can be computed by Gaussian elimination, but numerically it is read from the singular values (§M6.6): rank is the number of singular values distinguishable from zero.

---

## M6.3 Orthogonality, projection, and least squares

Two vectors are **orthogonal** if $\mathbf{a}^\top\mathbf{b}=0$. An **orthonormal basis** $\mathbf{q}_1,\dots,\mathbf{q}_k$ has $\mathbf{q}_i^\top\mathbf{q}_j=\delta_{ij}$ (Chapter M1's Kronecker delta). Orthonormal bases make everything easy: the coordinates of $\mathbf{v}$ are just dot products $c_i=\mathbf{q}_i^\top\mathbf{v}$, and lengths are preserved. The **QR decomposition** $\mathbf{X}=\mathbf{Q}\mathbf{R}$ (from the Gram–Schmidt procedure: orthogonalize the columns one after another) produces an orthonormal basis $\mathbf{Q}$ of the column space; the script checks $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$.

**Projection.** To approximate a vector $\mathbf{y}$ by something in the column space of $\mathbf{X}$, take the *closest* point $\hat{\mathbf{y}}=\mathbf{X}\hat{\mathbf{w}}$. The error $\mathbf{y}-\hat{\mathbf{y}}$ must be **orthogonal to every column** (otherwise sliding along a column would shrink the error):

$$
\mathbf{X}^{\top}(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}})=\mathbf{0}
\quad\Longrightarrow\quad
\hat{\mathbf{w}}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y},
\qquad
\hat{\mathbf{y}}=\mathbf{P}\mathbf{y},\ \ \mathbf{P}=\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top .
$$

These are the *normal equations* of Chapter M4, now with a picture: **least squares is orthogonal projection**. The matrix $\mathbf{P}$ (the "hat matrix") satisfies $\mathbf{P}^2=\mathbf{P}$ (projecting twice changes nothing) and $\operatorname{tr}\mathbf{P}=\operatorname{rank}\mathbf{X}$, the number of fitted parameters. In the script, with 30 observations and 3 features, $\mathbf{X}^\top$(residual) is zero to ten decimals, $\mathbf{P}\mathbf{P}=\mathbf{P}$, $\operatorname{tr}\mathbf{P}=3.000$, and Pythagoras holds: $\|\mathbf{y}\|^2=187.232=\|\mathbf{P}\mathbf{y}\|^2+\|\mathbf{y}-\mathbf{P}\mathbf{y}\|^2$.

That last identity is the geometry behind **$R^2$**: the fraction of squared length explained by the fit, $\|\mathbf{P}\mathbf{y}\|^2/\|\mathbf{y}\|^2$ (for centered data). It also explains **overfitting**: with as many features as observations, $\operatorname{tr}\mathbf{P}=n$, so $\mathbf{P}=\mathbf{I}$ and the "fit" is perfect and meaningless (Chapter 7).

---

## M6.4 Eigenvalues and eigenvectors

Most vectors change direction when a matrix acts on them. An **eigenvector** $\mathbf{v}\ne\mathbf{0}$ of a square matrix $\mathbf{A}$ does not: it is only stretched,

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v},
$$

and the stretch factor $\lambda$ is its **eigenvalue**. Eigenvalues are the roots of $\det(\mathbf{A}-\lambda\mathbf{I})=0$; two quick checks: their sum equals the trace of $\mathbf{A}$ and their product equals its determinant. For $\mathbf{A}=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ (script) the eigenvalues are $1$ and $3$ (sum $4$ = trace, product $3$ = determinant) with eigenvectors $(1,-1)/\sqrt2$ and $(1,1)/\sqrt2$: along the diagonal the matrix stretches by 3, along the anti-diagonal by 1.

**Diagonalization.** If $\mathbf{A}$ has a full set of independent eigenvectors collected as the columns of $\mathbf{V}$, then $\mathbf{A}=\mathbf{V}\boldsymbol\Lambda\mathbf{V}^{-1}$ with $\boldsymbol\Lambda=\operatorname{diag}(\lambda_i)$. In the eigenbasis the map is just coordinatewise scaling, so powers and functions of $\mathbf{A}$ are easy:

$$
\mathbf{A}^{k}=\mathbf{V}\boldsymbol\Lambda^{k}\mathbf{V}^{-1},\qquad
e^{\mathbf{A}t}=\mathbf{V}\,\operatorname{diag}(e^{\lambda_it})\,\mathbf{V}^{-1}.
$$

In the script, $\mathbf{A}^{10}$ computed from eigenvalues $1^{10}$ and $3^{10}$ matches repeated multiplication exactly ($29525$ on the diagonal, $29524$ off it). **Long-run behaviour is governed by the largest eigenvalue in magnitude**: the component along the dominant eigenvector grows (or decays) fastest and eventually dominates.

### M6.4.1 Evolutionary substitution models: eigenvalues of a rate matrix

!!! bio "Biology for modeling: the Jukes–Cantor model"
    **What is it?** DNA sites mutate over evolutionary time. In the simplest model (Jukes–Cantor, 1969) every base changes to each of the three others at the same rate $\alpha$. The **rate matrix** $\mathbf{Q}$ has off-diagonal entries $\alpha$ and diagonal entries $-3\alpha$ so that rows sum to 0; the probability matrix over a time $t$ is the **matrix exponential** $\mathbf{P}(t)=e^{\mathbf{Q}t}$, where $P_{ij}(t)$ is the probability that a site that starts as base $i$ is base $j$ after time $t$.

    **Using eigenvalues.** $\mathbf{Q}$ is symmetric with eigenvalues $0$ (once) and $-4\alpha$ (three times), the script reports $(0,-0.4,-0.4,-0.4)$ for $\alpha=0.1$. Hence $e^{\mathbf{Q}t}$ has eigenvalues $1$ and $e^{-4\alpha t}$, giving the classical formula

    $$
    P(\text{same base after time }t)=\tfrac14+\tfrac34e^{-4\alpha t}.
    $$

    **Check (script).** $t=1$: exact matrix exponential $0.75274$, formula $0.75274$; $t=5$: $0.35150$ in both; $t=20$: $0.25025$. Over long times every row tends to the **stationary distribution** $(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$: the sequences forget their ancestors and the information in them decays at the rate $4\alpha$ set by the nonzero eigenvalue.

    **Modelling consequence.** This decay of identity with divergence time is why sequence similarity can detect homology only up to a point (Chapter 27), why ancestral reconstruction is harder for deep ancestors (Chapter 42), and why the *observed* fraction of differing sites underestimates evolutionary distance at large divergence: sites that changed twice are counted once ($d=-\tfrac34\ln\!\big(1-\tfrac43p\big)$ corrects for it).

    **What would change the interpretation.** Real mutation rates differ between bases (transitions exceed transversions), between sites, and in context (CpG sites mutate much faster; Chapter 20). More realistic rate matrices have more complicated eigenstructure, but the eigen-decomposition method is the same.

### M6.4.2 Markov chains: state transitions and mixing times

A **Markov chain** with transition matrix $\mathbf{T}$ ($T_{ij}$ = probability of moving from state $i$ to state $j$; each *row* sums to 1) evolves a distribution row vector by $\mathbf{p}_{n+1}=\mathbf{p}_n\mathbf{T}$, so $\mathbf{p}_n=\mathbf{p}_0\mathbf{T}^n$. Eigenvalues of $\mathbf{T}$ have magnitude at most 1, and 1 is always one of them. A distribution $\boldsymbol\pi$ with $\boldsymbol\pi\mathbf{T}=\boldsymbol\pi$ is **stationary**: the left eigenvector for eigenvalue 1. For a chain in which every state can reach every other, it is unique and the chain converges to it, at the rate of the **second-largest eigenvalue magnitude** $|\lambda_2|$: the distance to $\boldsymbol\pi$ shrinks like $|\lambda_2|^{n}$.

In the script, a toy chain over the cell states *quiescent, cycling, differentiated* has eigenvalues $(1,0.8225,0.5775)$ and stationary distribution $(0.600,0.267,0.133)$. Starting from quiescent, the distance to stationarity is $0.227$ at step 5, $0.0113$ at step 20 and $4.6\times10^{-6}$ at step 60, tracking $|\lambda_2|^n=0.0201$ and $8.1\times10^{-6}$ (the match is to within a constant). A chain with an *absorbing* state (death, terminal differentiation) instead converges to the distribution concentrated on it, with eigenvalue 1 repeated.

The same mathematics governs: the *stationary* behaviour of Markov chain Monte Carlo (Chapter M7), the equilibrium of a population in a fitness landscape (Chapter 21), the long-time behaviour of random walks on gene-regulatory graphs and the pseudotime/transition-matrix methods in single-cell analysis (Chapter 30). Caution: a stationary distribution is an equilibrium statement; a developing tissue is *not* at equilibrium, so applying stationary-distribution reasoning to a snapshot of a developing system is an assumption that needs defending.

---

## M6.5 Symmetric matrices, quadratic forms, and positive definiteness

Symmetric matrices ($\mathbf{A}=\mathbf{A}^\top$) are the best-behaved, and the most common: covariance matrices, Hessians, Gram matrices and graph Laplacians are all symmetric.

!!! math "The spectral theorem"
    A real symmetric matrix $\mathbf{A}$ has **real** eigenvalues and an **orthonormal** set of eigenvectors. Thus $\mathbf{A}=\mathbf{V}\boldsymbol\Lambda\mathbf{V}^\top$ with $\mathbf{V}^\top\mathbf{V}=\mathbf{I}$: a rotation, a coordinate-wise stretch, and the inverse rotation. Equivalently,

    $$
    \mathbf{A}=\sum_{i}\lambda_i\,\mathbf{v}_i\mathbf{v}_i^\top,
    $$

    a sum of rank-one pieces along orthogonal directions.

The **quadratic form** $\mathbf{v}^\top\mathbf{A}\mathbf{v}$ is a number built from a vector; with $\mathbf{v}=\sum c_i\mathbf{v}_i$ it equals $\sum_i\lambda_ic_i^2$. A symmetric matrix is **positive definite** if $\mathbf{v}^\top\mathbf{A}\mathbf{v}>0$ for all $\mathbf{v}\ne\mathbf{0}$ (all eigenvalues $>0$) and **positive semi-definite** if $\ge0$. Positive definite Hessians mean minima (Chapter M4).

**Covariance matrices.** For data vectors $\mathbf{z}_n\in\R^d$ with mean $\bar{\mathbf{z}}$, the covariance matrix $\mathbf{C}=\frac{1}{N-1}\sum_n(\mathbf{z}_n-\bar{\mathbf{z}})(\mathbf{z}_n-\bar{\mathbf{z}})^\top$ is symmetric, and for any direction $\mathbf{v}$

$$
\mathbf{v}^\top\mathbf{C}\mathbf{v}=\operatorname{Var}(\mathbf{v}^\top\mathbf{z})\ \ge0 .
$$

So $\mathbf{C}$ is positive semi-definite *because variances are nonnegative*, and its quadratic form is "the variance of the data projected onto direction $\mathbf{v}$". In the script, for four correlated variables the eigenvalues are $(1.11,3.44,5.06,8.68)$ (all positive), the eigenvectors are orthonormal, $\mathbf{C}=\mathbf{V}\boldsymbol\Lambda\mathbf{V}^\top$ reconstructs, and for a random direction $\mathbf{v}^\top\mathbf{C}\mathbf{v}=28.090$ equals the sample variance of $\mathbf{Z}\mathbf{v}$ exactly. The **largest eigenvalue is the maximum variance over all unit directions**, attained along the corresponding eigenvector: this is the first principal component.

---

## M6.6 The singular value decomposition and PCA

Eigen-decomposition needs a square matrix. The **singular value decomposition (SVD)** works for any $\mathbf{A}\in\R^{m\times n}$:

$$
\mathbf{A}=\mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top=\sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,
$$

where $\mathbf{U}\in\R^{m\times m}$ and $\mathbf{V}\in\R^{n\times n}$ are orthogonal, $\boldsymbol\Sigma$ is "diagonal" with the **singular values** $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$ and $r$ is the rank. Geometrically, *every* linear map is: rotate (by $\mathbf{V}^\top$), stretch each axis by $\sigma_i$, rotate (by $\mathbf{U}$). The sum form says every matrix is a sum of $r$ rank-one pieces, ordered by importance $\sigma_i$.

How it relates to eigenvalues: $\mathbf{A}^\top\mathbf{A}=\mathbf{V}\boldsymbol\Sigma^2\mathbf{V}^\top$, so the right singular vectors $\mathbf{v}_i$ are eigenvectors of $\mathbf{A}^\top\mathbf{A}$ with eigenvalues $\sigma_i^2$ (and $\mathbf{U}$ likewise for $\mathbf{A}\mathbf{A}^\top$).

!!! math "Eckart–Young: the best low-rank approximation"
    Keep only the top $k$ terms: $\mathbf{A}_k=\sum_{i\le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. Among *all* matrices of rank at most $k$, $\mathbf{A}_k$ is the closest to $\mathbf{A}$, in the Frobenius norm (the root of the sum of squared entries), with error

    $$
    \|\mathbf{A}-\mathbf{A}_k\|_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}.
    $$

    (Chapter 2 gives the proof sketch.) *Compression is possible exactly to the extent that the singular values decay.*

**PCA is the SVD of centered data.** Let $\mathbf{X}\in\R^{N\times G}$ have its column means subtracted. Its covariance is $\mathbf{X}^\top\mathbf{X}/(N-1)=\mathbf{V}\boldsymbol\Sigma^2\mathbf{V}^\top/(N-1)$: the right singular vectors $\mathbf{v}_i$ are the **principal axes** (directions of maximal variance), the variance along axis $i$ is $\sigma_i^2/(N-1)$, and the **scores** (coordinates of each cell along axes) are the columns of $\mathbf{U}\boldsymbol\Sigma$. The "fraction of variance explained" by the first $k$ components is $\sum_{i\le k}\sigma_i^2/\sum_i\sigma_i^2$.

!!! bio "Biology for modeling: PCA of a cells × genes matrix"
    **What is it?** A synthetic experiment in the script: 300 cells of three types, 60 genes; each type has its own mean expression profile, plus independent noise.

    **What the SVD finds.** The singular values are $129.0,\,113.2,\,23.7,\,23.6,\,23.2,\dots$: two large values and then a flat floor near 23. *Three* types with centered data span a **two**-dimensional subspace (three points define a plane, and subtracting the overall mean uses up one degree of freedom). The floor of nearly equal singular values is the noise: independent noise of unit variance in a $300\times60$ matrix has singular values of about $\sqrt{N}+\sqrt{G}\approx25$ at the top edge (a random-matrix result, Marchenko–Pastur). The first two components explain $35.8\%+27.5\%=63.3\%$ of the variance; the remaining $36.7\%$ is noise spread over 58 dimensions.

    **Reading the scores.** Cell-type centroids in the PC1–PC2 plane are $(-10.1,2.4)$, $(7.4,6.4)$ and $(2.7,-8.8)$ against a within-type spread of about 1.0: the three types are well separated.

    **Truncation error.** The rank-2 approximation has relative error $0.606$: *60% of the matrix's norm is unexplained* even though the signal is perfectly captured, because most of the matrix is noise. Even the rank-20 approximation still leaves $0.438$. The Eckart–Young prediction $\sqrt{\sum_{i>k}\sigma_i^2}/\|\mathbf{X}\|$ agrees at every $k$.

    **What would change the interpretation.** If a batch effect were as large as the biological difference between cell types it would add a component with a large singular value, and PCA would offer no way to tell which is which: PCA finds *variance*, not meaning (Chapter 2).

!!! warning "Percent of variance explained is not the dimension of the signal"
    In the experiment above, two components explain 63% of the variance, and 20 components explain 81%, but the signal is exactly two-dimensional. A noisy matrix keeps "explaining a little more" with each added component. The **gap** in the singular-value spectrum, and a comparison with the noise floor expected under a null (random matrix theory or a permutation test), tell you how many components are signal.

```python
--8<-- "code/m06_linalg2.py"
```

Output (seed 0):

```text
A is 6 x 8 but built as (6x3)(3x8): rank 3; column space dim 3, row space dim 3, null space dim 5 (= 8 - rank), left null space dim 3 (= 6 - rank)
three vectors in R^4 where the second is twice the first: rank = 2 -> linearly dependent

least squares: w = [ 0.841 -0.952  2.095]; X^T residual = [ 0.  0. -0.] (orthogonal);  P is idempotent (P P = P): True;  trace(P) = 3.000 = number of columns
Pythagoras: |y|^2 = 187.232 = |Py|^2 + |y - Py|^2 = 187.232
QR: Q^T Q = I: True ; same fitted values from Q Q^T y: True

A = [[2,1],[1,2]]: eigenvalues [1. 3.] (trace 4 = sum, det 3 = product), eigenvectors (columns)
[[-0.7071  0.7071]
 [ 0.7071  0.7071]]
A^10 via eigen: [[29525.0, 29524.0], [29524.0, 29525.0]]  direct: [[29525.0, 29524.0], [29524.0, 29525.0]]

Jukes-Cantor rate matrix eigenvalues: [-0.  -0.4 -0.4 -0.4]  (0 and -4*alpha = -0.4)
  t =   1.0: P(same base) = 0.75274  (formula 1/4 + 3/4 e^(-4 alpha t) = 0.75274);  rows sum to 1.0
  t =   5.0: P(same base) = 0.35150  (formula 1/4 + 3/4 e^(-4 alpha t) = 0.35150);  rows sum to 1.0
  t =  20.0: P(same base) = 0.25025  (formula 1/4 + 3/4 e^(-4 alpha t) = 0.25025);  rows sum to 1.0
  as t grows every row of P(t) tends to the stationary distribution (1/4,1/4,1/4,1/4): [0.25 0.25 0.25 0.25]

recurrent Markov chain (states quiescent, cycling, differentiated): eigenvalues [1.     0.8225 0.5775]; stationary distribution [0.6    0.2667 0.1333] (pi T = pi: True)
  after  1 steps: [0.9 0.1 0. ]  distance to stationary 6.00e-01   (|lambda_2|^n = 8.22e-01)
  after  5 steps: [0.7136 0.2334 0.053 ]  distance to stationary 2.27e-01   (|lambda_2|^n = 3.76e-01)
  after 20 steps: [0.6057 0.2656 0.1287]  distance to stationary 1.13e-02   (|lambda_2|^n = 2.01e-02)
  after 60 steps: [0.6    0.2667 0.1333]  distance to stationary 4.55e-06   (|lambda_2|^n = 8.08e-06)
absorbing chain T (state 3 never leaves): the long-run distribution from state 1 is [0. 0. 1.]

covariance matrix of 4 variables: eigenvalues [1.112 3.437 5.055 8.677] (all >= 0), eigenvectors orthonormal: True; reconstruct C = V diag(lam) V^T: True
quadratic form v^T C v = 28.090 = variance of the projection Z v: 28.090 (>0, as positive definiteness requires)

SVD of a 300x60 centered matrix (3 cell types + noise): top singular values [129.  113.2  23.7  23.6  23.2], then [22.8 22.5 22.3]...
  variance fraction of PC1..PC4: [0.358 0.275 0.012 0.012] -> two components for three types (centered: types span a 2D subspace)
  rank- 1 approximation: relative error 0.801;   Eckart-Young prediction sqrt(sum of dropped s^2)/|X| = 0.801
  rank- 2 approximation: relative error 0.606;   Eckart-Young prediction sqrt(sum of dropped s^2)/|X| = 0.606
  rank- 5 approximation: relative error 0.576;   Eckart-Young prediction sqrt(sum of dropped s^2)/|X| = 0.576
  rank-20 approximation: relative error 0.438;   Eckart-Young prediction sqrt(sum of dropped s^2)/|X| = 0.438
  PC1/PC2 centroids of the three types:
[[-10.1   2.4]
 [  7.4   6.4]
 [  2.7  -8.8]]   typical within-type spread 1.0 -> clusters are well separated
```

---

## M6.7 Worked examples

!!! example "Worked example M6.1: How many parameters can the data determine?"
    **Situation.** A linear model $\mathbf{y}=\mathbf{X}\mathbf{w}$ uses 30 genes as features for 20 samples. The $20\times30$ design matrix is generated from 12 underlying independent signals (the genes are noisy mixtures of 12 pathways).

    **Question.** How many dimensions of $\mathbf{w}$ are constrained by the data?

    **Reasoning.** $\operatorname{rank}\mathbf{X}\le\min(20,30,12)=12$. Only the projection of $\mathbf{w}$ onto the 12-dimensional row space affects predictions; the remaining $30-12=18$ dimensions form the null space and are *unconstrained by the data*. Parameters differ only by null-space vectors are observationally equivalent. So 12 combinations of the 30 coefficients are determined; there is no way to say which *individual* genes matter without extra assumptions. If the 12 signals are measured with noise, the weakest signal directions have small singular values and are effectively in the null space too; the effective rank can be below 12.

    **Lesson.** Rank, not the number of columns, is the number of questions the data can answer. The Expert Chain's "information used / information ignored" link (Chapter 1, L4) is a statement about row space and null space.

!!! example "Worked example M6.2: How long until a cell population reaches equilibrium?"
    **Situation.** A time-lapse experiment suggests the toy chain of §M6.4.2 for transitions per hour among quiescent, cycling and differentiated states. You want to know how long to culture cells so that the state proportions reach 99% of the way to their long-run mixture.

    **Reasoning.** The distance to equilibrium shrinks like $|\lambda_2|^{n}$ with $|\lambda_2|=0.8225$ (up to a constant). Requiring $0.8225^{n}\le0.01$ gives $n\ge\ln0.01/\ln0.8225=23.5$, so about 24 hours if a step is one hour. Direct computation agrees: starting from the quiescent state the total-variation-type (L1) distance to the stationary mixture is $0.80$ at the start and $0.0052$ after 24 steps (0.65% of its starting value). The *stationary mixture* is $(0.60,0.27,0.13)$ regardless of the start.

**Interpretation.** The slow eigenvector (the one with $\lambda_2=0.82$) is the *rate-limiting process*: here, the exchange between quiescent and cycling states. Making the other transitions faster would not shorten the time much. The measurable lesson is that **mixing time is set by the second eigenvalue, not by the typical transition rate**.

    **Caveat.** This assumes a time-homogeneous Markov chain: that transition probabilities do not depend on how long the cell has been in a state (no memory) and do not change over the experiment. Real cell-state durations are often not exponentially distributed, which a Markov model cannot represent; a test is whether the dwell-time distribution matches the geometric distribution the chain predicts.

---

## M6.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"Two components explain 63% of the variance, so the data are essentially two-dimensional\""
    **The observation.** PCA of a single-cell dataset shows PC1 and PC2 explain 36% and 27% of the variance and a clean scatter plot separates three groups.

    **Tempting conclusion.** "The biology is two-dimensional; the other 58 dimensions are noise or redundancy."

    **Decompose.** In this chapter's synthetic example the conclusion happens to be *right*, but the argument is not valid. (i) Variance is not information: a direction with small variance can carry decisive biology, such as a rare cell type, whose contribution to the covariance is tiny because few cells belong to it (Chapter 2). (ii) The singular-value spectrum must be compared with a noise floor; a flat tail at 23 against signals at 129 and 113 is clear evidence here, while a gradually decaying spectrum shows no gap to read. (iii) PCA is linear: a curved one-dimensional structure (a developmental trajectory) spreads across several components, so a 1-D process can *look* like 4-D. (iv) The number of components needed to reproduce a *downstream task's performance* is a different quantity from the number of components that explain variance.

    **Hidden assumptions.** (i) Gene scaling and transformation were appropriate (a few very highly expressed genes can dominate variance on raw counts). (ii) Technical factors (batch, depth) are not among the large components. (iii) Cells are exchangeable samples from the population of interest.

    **Discriminating experiments.** Plot the singular-value spectrum on a log scale against a permutation null (shuffle each gene independently across cells); colour the leading components by batch and depth; compare downstream accuracy using $k=2,5,20$ components; check whether the component structure persists in a held-out set of cells.

    **What this teaches.** "Dimension" has at least three meanings in this book (variance, signal, task), and a statement about one is not a statement about another.

---

## M6.9 Connections

- **Forward:** Chapter 2 builds the research version: tensor shapes, `einsum`, Eckart–Young with a proof sketch, PCA failure cases, high-dimensional geometry, cost tables. PCA/SVD underlie the embeddings of Chapters 13 and 38, the population-structure corrections of Chapters 21 and 26, and low-rank adaptation (Chapter 17). Eigenvalues of transition matrices return in Markov chain Monte Carlo (M7) and RNA-velocity and trajectory methods (Chapter 30). Covariance and positive definiteness reappear in Gaussian models (Chapter M7, 8). Rate-matrix eigendecompositions return in phylogenetics (Chapters 21, 42).
- **Backward:** vectors, matrices, systems, conditioning (M5); Hessians and quadratic forms (M4).

!!! takeaways "Key takeaways"
    1. A **basis** is an independent spanning set; **dimension** counts free parameters. The **rank** of $\mathbf{A}$ is the dimension of what it can output; rank + nullity = number of inputs.
    2. The **four subspaces**: column and row space (what the matrix sees), null and left null space (what it erases). Parameters that differ by a null-space vector are indistinguishable from the data.
    3. **Least squares is orthogonal projection:** the residual is orthogonal to every column. The hat matrix is idempotent with trace equal to the number of fitted parameters; $\|\mathbf{y}\|^2=\|\mathbf{P}\mathbf{y}\|^2+\|\mathbf{y}-\mathbf{P}\mathbf{y}\|^2$.
    4. **Eigenvectors** are directions a matrix only stretches; $\mathbf{A}^k$ and $e^{\mathbf{A}t}$ act on eigenvalues. The sum of eigenvalues is the trace, the product is the determinant.
    5. For the Jukes–Cantor model, eigenvalues $0,-4\alpha$ give $P(\text{same})=\tfrac14+\tfrac34e^{-4\alpha t}$. Markov chains converge to the stationary distribution (eigenvalue 1) at the rate $|\lambda_2|^n$.
    6. **Symmetric matrices** have orthonormal eigenvectors (spectral theorem). Covariance matrices are positive semi-definite because $\mathbf{v}^\top\mathbf{C}\mathbf{v}=\operatorname{Var}(\mathbf{v}^\top\mathbf{z})\ge0$; the top eigenvector is the maximum-variance direction.
    7. The **SVD** $\mathbf{A}=\sum\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ exists for every matrix; the truncated SVD is the best low-rank approximation (Eckart–Young). **PCA is the SVD of centered data.**
    8. Percent variance explained is not the dimension of the signal. Compare the singular-value spectrum with a noise floor.

---

## Further reading

- Strang, G. *Linear Algebra and Its Applications*; and his MIT 18.06 lectures. The "four fundamental subspaces" picture is his.
- Trefethen, L. N. & Bau, D. *Numerical Linear Algebra*. SIAM. The SVD and conditioning, with clear geometry.
- Jukes, T. H. & Cantor, C. R. (1969). Evolution of protein molecules. In *Mammalian Protein Metabolism*, Academic Press. The substitution model.
- Gavish, M. & Donoho, D. (2014). The optimal hard threshold for singular values is $4/\sqrt3$. *IEEE Trans. Information Theory* 60, 5040–5053. How many components are signal.
- Levin, D. A. & Peres, Y. *Markov Chains and Mixing Times*. AMS. Mixing times and eigenvalues, rigorously.
