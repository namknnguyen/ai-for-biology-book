# Chapter M5. Linear Algebra I: Vectors, Matrices, and Linear Systems

!!! abstract "Chapter at a glance"
    **Motivation.** A cell measured by a sequencing experiment is a list of 20,000 numbers; a DNA sequence is a table of 0s and 1s; a metabolic network is a table of integers saying which reaction makes or consumes which molecule. Linear algebra is the mathematics of such tables and of the *linear* operations on them. It is the language in which a neural-network layer, a principal-component analysis, a regression and a flux-balance model are all written. This chapter builds the first half: vectors, matrices, how they multiply, how to solve a system of linear equations, and what matrices do geometrically. Chapter M6 finishes with spaces, eigenvalues and the SVD.

    **Prerequisites.** Chapters M1 and M2 (sums and indices, functions). High-school geometry (distance, angle).

    **You will be able to:** (1) represent biological data as vectors and matrices and state shapes; (2) compute dot products, norms, angles and cosine similarity and say what each measures; (3) multiply matrices in three equivalent ways and keep track of shapes; (4) solve a linear system by Gaussian elimination and say when a solution is unique, absent, or non-unique; (5) interpret a matrix as a geometric transformation and a determinant as a volume scale factor; (6) recognize a singular or ill-conditioned matrix and what it means for estimation; (7) read a stoichiometric matrix and find the steady-state fluxes as its null space.

---

## M5.1 Vectors

A **vector** in $\R^{d}$ is an ordered list of $d$ real numbers $\mathbf{x}=(x_1,\dots,x_d)$. Two ways to think about it, and you will switch between them constantly:

- **As data.** A cell's expression of $d$ genes; a one-hot encoded base $(0,1,0,0)$; the six numbers describing a molecule's properties.
- **As geometry.** An arrow from the origin to a point in $d$-dimensional space, with a length and a direction. Adding vectors is placing arrows head to tail; multiplying by a number stretches the arrow.

**Linear combination.** Given vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ and numbers $c_1,\dots,c_k$, the vector $c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k$ is a **linear combination**. Almost every model in this book builds outputs as linear combinations of features.

### M5.1.1 The dot product, length, and angle

The **dot product** of $\mathbf{a},\mathbf{b}\in\R^{d}$ is $\mathbf{a}^{\top}\mathbf{b}=\sum_{i=1}^{d}a_ib_i$. The **length** (Euclidean norm) of $\mathbf{a}$ is $\|\mathbf{a}\|=\sqrt{\mathbf{a}^\top\mathbf{a}}$ (Pythagoras in $d$ dimensions), and the **angle** $\theta$ between two vectors satisfies

$$
\mathbf{a}^{\top}\mathbf{b}=\|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta .
$$

So the dot product measures *alignment*: positive when the vectors point roughly the same way, zero when they are **orthogonal** (perpendicular), negative when opposed. The **cosine similarity** is $\cos\theta=\mathbf{a}^\top\mathbf{b}/(\|\mathbf{a}\|\|\mathbf{b}\|)$, which ignores length and compares only direction. Example from the script: $\mathbf{a}=(3,4,0)$, $\mathbf{b}=(4,3,5)$ give $\mathbf{a}^\top\mathbf{b}=24$, $\|\mathbf{a}\|=5$, $\|\mathbf{b}\|=7.07$, cosine $0.679$, angle $47.2^\circ$.

!!! bio "Biology for modeling: cosine similarity and depth normalization"
    **What is it?** Comparing two cells (or two samples) as expression vectors. Euclidean distance $\|\mathbf{a}-\mathbf{b}\|$ is sensitive to *overall scale*; cosine similarity is not.

    **Information it contains / discards.** Cosine keeps the *pattern* of expression and discards the total. This is useful when total counts differ for technical reasons (sequencing depth), and harmful when total counts carry biological signal (cell size, RNA content).

    **Example.** In the script, a profile of 2,000 gene values and a noisy copy scaled five-fold have cosine similarity $0.997$ yet Euclidean distance $445$: by cosine they are nearly the same; by distance they are far apart.

    **What would change the interpretation.** Deciding which to use is a *modelling choice about which differences count as biology*. This is the "measurement gap" of Chapter 1 in miniature; it recurs in embedding comparisons (Chapters 13, 38).

Other norms exist: the $L_1$ norm $\sum_i|x_i|$ (total absolute amount; favours sparse solutions) and the $L_\infty$ norm $\max_i|x_i|$. When a book says "the norm", it means $L_2$ unless noted.

---

## M5.2 Matrices

A **matrix** $\mathbf{A}\in\R^{m\times n}$ is a rectangular table with $m$ rows and $n$ columns; $A_{ij}$ is the entry in row $i$, column $j$. It has two jobs:

1. **A table of data.** $\mathbf{X}\in\R^{N\times G}$: $N$ cells (rows) by $G$ genes (columns).
2. **A linear map.** $\mathbf{A}$ takes $\mathbf{x}\in\R^{n}$ to $\mathbf{A}\mathbf{x}\in\R^{m}$, where entry $i$ of the output is the dot product of row $i$ with $\mathbf{x}$. "Linear" means $\mathbf{A}(\mathbf{x}+\mathbf{y})=\mathbf{A}\mathbf{x}+\mathbf{A}\mathbf{y}$ and $\mathbf{A}(c\mathbf{x})=c\mathbf{A}\mathbf{x}$: the map respects addition and scaling. Equivalently, $\mathbf{A}\mathbf{x}$ is a linear combination of the *columns* of $\mathbf{A}$ with weights $x_j$.

Special matrices to know: the **identity** $\mathbf{I}$ (ones on the diagonal; $\mathbf{I}\mathbf{x}=\mathbf{x}$), **diagonal** matrices (scale each coordinate separately), **symmetric** matrices ($\mathbf{A}=\mathbf{A}^\top$, where the **transpose** swaps rows and columns), **permutation** matrices (reorder coordinates) and **orthogonal** matrices ($\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$: rotations and reflections, which preserve lengths and angles).

### M5.2.1 Matrix multiplication

For $\mathbf{A}\in\R^{m\times n}$ and $\mathbf{B}\in\R^{n\times p}$, the product $\mathbf{C}=\mathbf{A}\mathbf{B}\in\R^{m\times p}$ has entries $C_{ik}=\sum_{j=1}^{n}A_{ij}B_{jk}$. The inner dimensions must match; they are summed away. Keeping shapes straight is half the battle: $(m\times n)(n\times p)\to(m\times p)$. Three equivalent views, each useful in a different situation (the script verifies they agree):

1. **Entries are dot products.** $C_{ik}$ = row $i$ of $\mathbf{A}$ dotted with column $k$ of $\mathbf{B}$.
2. **Columns are transformed columns.** Column $k$ of $\mathbf{C}$ is $\mathbf{A}$ applied to column $k$ of $\mathbf{B}$.
3. **Sum of outer products.** $\mathbf{C}=\sum_{j=1}^{n}\mathbf{a}_{:,j}\mathbf{b}_{j,:}$: each term is a rank-one $m\times p$ matrix. This view underlies the SVD and low-rank structure (Chapter M6).

Important properties: $(\mathbf{AB})\mathbf{C}=\mathbf{A}(\mathbf{BC})$ (associative), $\mathbf{A}(\mathbf{B}+\mathbf{C})=\mathbf{AB}+\mathbf{AC}$, $(\mathbf{AB})^\top=\mathbf{B}^\top\mathbf{A}^\top$, but in general $\mathbf{AB}\ne\mathbf{BA}$: **order matters**. (A rotation followed by a shear is a different map from a shear followed by a rotation.) The cost of multiplying is about $2mnp$ arithmetic operations, and a modern accelerator (GPU) is built to do exactly this operation fast, which is why almost every deep-learning computation is phrased as matrix multiplication (Chapters 2 and 12).

!!! bio "Biology for modeling: matrices you will meet"
    **One-hot sequence.** A DNA sequence of length $L$ becomes a matrix $\mathbf{X}\in\{0,1\}^{L\times4}$ with exactly one 1 per row. A motif scanner with weights $\mathbf{W}\in\R^{w\times4}$ computes at each position the sum of elementwise products of $\mathbf{W}$ with a window of $\mathbf{X}$ (a convolution, Chapter 10).

    **Expression matrix.** $\mathbf{X}\in\R^{N\times G}$, rows cells, columns genes. Multiplying by a $G\times k$ matrix of gene weights gives $k$ scores per cell: a linear *embedding*.

    **Transition and substitution matrices.** The $4\times4$ matrix of nucleotide substitution probabilities (Chapter 20); an amino-acid scoring matrix such as BLOSUM ($20\times20$, Chapter 27).

    **Adjacency and stoichiometry.** A graph's adjacency matrix (M9); a metabolic network's stoichiometric matrix (§M5.5).

---

## M5.3 Linear systems and Gaussian elimination

A **linear system** is a set of equations in unknowns $x_1,\dots,x_n$, each linear in the unknowns:

$$
\begin{aligned}
2x_1+x_2-x_3&=8\\
-3x_1-x_2+2x_3&=-11\\
-2x_1+x_2+2x_3&=-3
\end{aligned}
\qquad\Longleftrightarrow\qquad
\mathbf{M}\mathbf{x}=\mathbf{b}.
$$

**Gaussian elimination** solves it by the method you learned in school, done systematically: use the first equation to eliminate $x_1$ from the others, then the second to eliminate $x_2$, and so on, producing a triangular system that is solved by back-substitution. In practice you choose the largest available number as the pivot ("partial pivoting") to limit round-off error. The script implements it in 14 lines and recovers the exact solution $\mathbf{x}=(2,3,-1)$, agreeing with NumPy's solver to machine precision on random systems of size 50, 100 and 200.

**Cost.** Elimination needs about $\tfrac23n^3$ arithmetic operations: the script counts $89{,}525$ for $n=50$ ($1.07\times\tfrac23n^3$), $691{,}550$ for $n=100$, and $5{,}433{,}100$ for $n=200$. Doubling $n$ multiplies the cost by 8. Solving with a $20{,}000\times20{,}000$ matrix (as in a gene-by-gene system) is feasible, but with $10^{6}$ unknowns it is not unless the matrix is sparse or structured.

### M5.3.1 When is a solution unique?

Think of each equation as one constraint. Three outcomes are possible for $\mathbf{M}\mathbf{x}=\mathbf{b}$ with $\mathbf{M}\in\R^{n\times n}$:

1. **Exactly one solution** if $\mathbf{M}$ is **invertible** (also called nonsingular): there is a matrix $\mathbf{M}^{-1}$ with $\mathbf{M}^{-1}\mathbf{M}=\mathbf{M}\mathbf{M}^{-1}=\mathbf{I}$, and $\mathbf{x}=\mathbf{M}^{-1}\mathbf{b}$. (In code, solve the system directly; do not form the inverse.)
2. **No solution** if the equations contradict each other.
3. **Infinitely many solutions** if the equations are redundant: some are combinations of others, leaving free parameters.

Cases 2 and 3 happen when $\mathbf{M}$ is **singular**: its rows (or columns) are linearly dependent. For a *rectangular* system with more unknowns than equations (the usual case in genomics: more variants than samples) there are always infinitely many solutions or none, and extra assumptions are needed to choose one (regularization, sparsity). That is the heart of the "$p\gg n$" problem of Chapters 7 and 26.

---

## M5.4 Matrices as geometry, and the determinant

A $2\times2$ or $3\times3$ matrix can be *seen*: it moves the plane (or space). The script applies four maps to the unit square and measures the area of the image:

| Map | Matrix | Effect | $\det$ | Area scale |
|---|---|---|---|---|
| Rotation by $30^\circ$ | $\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$ | turns without distortion | $1$ | $1$ |
| Scaling | $\mathrm{diag}(2,1.5)$ | stretches axes independently | $3$ | $3$ |
| Shear | $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ | slides rows sideways | $1$ | $1$ |
| Singular | $\begin{pmatrix}1&2\\2&4\end{pmatrix}$ | collapses the plane onto a line | $0$ | $0$ |

The **determinant** $\det\mathbf{A}$ is the signed factor by which the map scales volume (area in 2D). Its absolute value is the volume scale; its sign says whether orientation is preserved. $\det\mathbf{A}=0$ means the map flattens space into fewer dimensions, so information is lost and **no inverse exists**: the script shows the singular matrix sending both $(2,-1)$ and the origin to $(0,0)$, so the two cannot be distinguished afterwards. Rotations and reflections are the **orthogonal** maps: $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$ and they preserve lengths ($\|\mathbf{R}\mathbf{x}\|=\|\mathbf{x}\|$, checked for $(3,4)\to5$) and angles.

Connections to the rest of the book: the Jacobian determinant is how densities change under a smooth change of variables (Chapter M4); an invertible map that keeps information is a prerequisite for normalizing flows (Chapter 14); a *non-invertible* feature map (for example, projecting onto fewer dimensions) is how every embedding loses information.

!!! warning "Determinants are theory, not computation"
    Do not compute determinants to decide whether a system is solvable on a computer: the value is extremely sensitive to scaling and floating-point round-off. Use the rank or the condition number (§M5.6) instead.

---

## M5.5 A worked biological linear system: steady states of a metabolic network

!!! bio "Biology for modeling: the stoichiometric matrix"
    **What is it?** Take a tiny metabolic network with three metabolites $A,B,C$ and five reactions: R1 imports $A$; R2 converts $A\to B$; R3 converts $A\to C$; R4 converts $B\to C$; R5 exports $C$. The **stoichiometric matrix** $\mathbf{S}\in\R^{3\times5}$ has a row per metabolite and a column per reaction; the entry $S_{ij}$ is the net number of molecules of metabolite $i$ produced ($+$) or consumed ($-$) by reaction $j$:

    $$
    \mathbf{S}=\begin{pmatrix}
    1&-1&-1&0&0\\
    0&1&0&-1&0\\
    0&0&1&1&-1
    \end{pmatrix}
    \begin{array}{l}A\\B\\C\end{array}.
    $$

    If $\mathbf{v}\in\R^{5}$ is the vector of **reaction fluxes** (molecules per hour through each reaction), the rate of change of the metabolite concentrations is $d\mathbf{c}/dt=\mathbf{S}\mathbf{v}$. This is a *linear map from fluxes to concentration changes*.

    **Steady state.** At steady state, concentrations do not change: $\mathbf{S}\mathbf{v}=\mathbf{0}$. This is a *homogeneous* linear system, and its solutions form the **null space** of $\mathbf{S}$ (the set of inputs mapped to zero).

    **What the script finds.** $\mathbf{S}$ has rank 3 and the null space has dimension $5-3=2$: of five fluxes only two can be chosen freely. Every steady-state flux is $\mathbf{v}=v_1(1,0,1,0,1)+v_2(0,1,-1,1,0)$: the first pattern is "all material goes $A\to C$ directly", the second is a *cycle shift* "reroute some material via $B$". For $v_1=10$ and $v_2=4$ the flux is $(10,4,6,4,10)$ and $\mathbf{S}\mathbf{v}=\mathbf{0}$ (script); a flux with the export reduced to 5 violates mass balance and leaves $\mathbf{S}\mathbf{v}=(0,0,5)$: metabolite $C$ would pile up at 5 units per hour.

    **Modelling consequence.** The steady-state space is a *linear subspace*, and flux-balance analysis (FBA) picks the point in it (subject to bounds on fluxes) that maximizes an objective such as biomass production, using linear programming. Genome-scale models have thousands of reactions and a null space of high dimension; the geometry is the same. Their limitation is also visible: the model *ignores kinetics, regulation and concentrations*, so it says what is *possible* at steady state and not what the cell *does*.

**Why this matters beyond metabolism.** "The set of solutions to a homogeneous linear system is a subspace whose dimension is the number of unknowns minus the rank" is a theorem you will use again: in identifiability (which parameters can the data determine?), in the null space of an attention matrix (Chapter 2), and in determining how many directions of a representation are unconstrained (Chapter 13).

---

## M5.6 Conditioning: when the answer is fragile

A system can be solvable on paper and numerically hopeless. The **condition number** $\kappa(\mathbf{M})$ measures how much a relative error in $\mathbf{b}$ (or in $\mathbf{M}$) can be amplified into the relative error of $\mathbf{x}$: roughly, you lose $\log_{10}\kappa$ digits of accuracy. For a symmetric positive definite matrix $\kappa=\lambda_{\max}/\lambda_{\min}$ (the ratio of the largest to smallest eigenvalue; Chapter M6), the same number that controlled gradient descent in Chapter M4. A matrix with $\kappa=\infty$ is singular.

The classic example is the **Hilbert matrix** $H_{ij}=1/(i+j-1)$, whose columns are nearly parallel. In the script, we solve $\mathbf{H}\mathbf{x}=\mathbf{H}\mathbf{1}$ (so the true answer is all ones):

| $n$ | $\kappa$ | max error in $\mathbf{x}$ | digits lost |
|---|---|---|---|
| 4 | $1.6\times10^{4}$ | $5\times10^{-14}$ | 4 |
| 8 | $1.5\times10^{10}$ | $7\times10^{-8}$ | 10 |
| 12 | $1.6\times10^{16}$ | $0.28$ | 16 (all of them) |

At $n=12$ the condition number reaches the reciprocal of machine precision and the computed "solution" is garbage: the answer is wrong by 28% although the code ran without any warning. **Nearly collinear features** (highly correlated gene expression, SNPs in LD) give biology's version of the Hilbert matrix. A model can fit the training data to near-perfect accuracy while its coefficients are meaningless. The remedies, all revisited later, are regularization, feature reduction, and a better parametrization.

```python
--8<-- "code/m05_linalg1.py"
```

Output (seed 0):

```text
a.b = 24, |a| = 5, |b| = 7.071, cosine = 0.6788, angle = 47.2 degrees
cosine similarity of a profile and its 5x-scaled noisy copy: 0.9968 (Euclidean distance between them is large: 445)

shapes (3x4)(4x2) -> (3x2); three views agree: True
AB == BA for 2x2 matrices? False (matrix multiplication does not commute)

solve Mx = b:  Gaussian elimination [ 2.  3. -1.]  numpy [ 2.  3. -1.]  (exact solution is [2, 3, -1])
  n =  50: operations     89,525 = 1.07 x (2/3)n^3;  max |x - numpy| = 2.1e-17
  n = 100: operations    691,550 = 1.04 x (2/3)n^3;  max |x - numpy| = 4.9e-17
  n = 200: operations  5,433,100 = 1.02 x (2/3)n^3;  max |x - numpy| = 3.3e-17

map            det      area of image of unit square   invertible?
  rotation 30deg  1.000    1.000                       True
  scale (2,1.5)   3.000    3.000                       True
  shear           1.000    1.000                       True
  singular        0.000    0.000                       False
  rotation is orthogonal: R^T R = I: True ; it preserves lengths: True
  singular map sends (2,-1) to [0. 0.] -> a whole line collapses to a point, so no inverse exists

stoichiometric matrix S (3 metabolites x 5 reactions): rank 3, null-space dimension 2
flux vector v = (10, 4, 6, 4, 10):  S v = [0. 0. 0.] -> steady state: every metabolite is produced as fast as it is consumed
every steady-state flux has the form v = v1*(1,0,1,0,1) + v2*(0,1,-1,1,0): True
a flux that violates mass balance: S v = [0. 0. 5.] (metabolite C accumulates at 5 per hour)

Hilbert matrix H_ij = 1/(i+j-1): condition number and error of solving H x = H x_true (x_true = ones)
  n =  4: cond = 1.6e+04   max error in x = 5.4e-14   (rule of thumb: digits lost ~ log10(cond) = 4)
  n =  8: cond = 1.5e+10   max error in x = 7.2e-08   (rule of thumb: digits lost ~ log10(cond) = 10)
  n = 12: cond = 1.6e+16   max error in x = 2.8e-01   (rule of thumb: digits lost ~ log10(cond) = 16)
```

---

## M5.7 Worked examples

!!! example "Worked example M5.1: A matrix with more unknowns than equations"
    **Situation.** You want a linear model predicting a phenotype from $p=500$ genetic variants measured in $N=100$ people: $\mathbf{X}\mathbf{w}=\mathbf{y}$ with $\mathbf{X}\in\R^{100\times500}$.

    **Question.** How many exact solutions $\mathbf{w}$ exist, and what does that imply for interpretation?

    **Reasoning.** At most 100 of the columns of $\mathbf{X}$ can be linearly independent, so the rank is at most 100. The null space of $\mathbf{X}$ has dimension at least $500-100=400$. If $\mathbf{w}_0$ is one solution, then $\mathbf{w}_0+\mathbf{z}$ is also a solution for *every* $\mathbf{z}$ in the null space. So there are infinitely many perfect fits ("interpolating" solutions).

    **Consequences.** (1) A perfect training fit is *automatic* and says nothing about predictive ability. (2) The data cannot determine $\mathbf{w}$: a 400-dimensional family of coefficient vectors gives identical predictions on these 100 people. (3) To pick one, a method needs an extra principle: the smallest-norm solution (pseudo-inverse; Chapter 2), a sparse solution (lasso; Chapter 7), or a prior (Bayesian methods; Chapter 8). The *choice of principle*, not the data, determines which variants appear to matter.

    **Lesson.** Counting equations against unknowns is the quickest check on whether a coefficient is *identifiable*. Whenever a result depends on which of many equally good fits an algorithm picks, report the principle that picked it.

!!! example "Worked example M5.2: Do two measurements carry the same information?"
    **Situation.** Two assays $a$ and $b$ are run on the same four samples and produce vectors $\mathbf{a}=(1,2,3,4)$ and $\mathbf{b}=(3,6,9,12)$. A collaborator proposes using both as separate features in a regression.

    **Reasoning.** $\mathbf{b}=3\mathbf{a}$: the vectors are parallel (cosine $=1$), so the matrix $[\mathbf{a}\ \mathbf{b}]$ has rank 1 rather than 2 and $\mathbf{X}^\top\mathbf{X}$ is singular. Any coefficients $(w_a,w_b)$ with the same value of $w_a+3w_b$ give identical predictions: the regression has a one-dimensional null space, and the individual coefficients cannot be estimated. If $\mathbf{b}=3\mathbf{a}+\text{small noise}$ the system is *technically* solvable but has an enormous condition number, and the estimated coefficients will be huge and opposite in sign.

    **What to do.** Drop one feature, average them, or regularize. And ask whether the two assays are really independent measurements of different things or the same thing in different units.

---

## M5.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"We normalized every cell to unit length, so the data are comparable\""
    **The observation.** To remove depth effects, each cell's expression vector is divided by its Euclidean norm. Distances between normalized cells are then monotone in cosine similarity.

    **Tempting conclusion.** "Technical depth differences are gone, so remaining differences are biological."

    **Decompose.** Normalizing to unit length forces every cell onto the unit sphere. *Information about total expression is discarded*, including biological information: a large cell with more RNA and a small cell with less RNA are mapped to the same point if their profiles are proportional. Moreover, the normalization introduces dependence between genes: if one gene's share rises, the others' shares must fall. This compositional constraint produces spurious negative correlations between genes even when the underlying molecules are independent.

    **Hidden assumptions.** (i) Total RNA content is a nuisance and not a signal. (ii) Depth acts as a single multiplicative factor common to all genes (real depth effects are gene-dependent: short and high-GC genes behave differently). (iii) The few genes dominating the length of the vector (highly expressed genes) are not themselves the biology of interest.

    **Discriminating experiments.** Use spike-in controls (known RNA added at fixed amounts) to separate depth from content; compare conclusions under several normalizations; check whether the first principal component of the normalized data still correlates with depth.

    **What this teaches.** Every normalization is a *projection* (a many-to-one linear or nonlinear map, in the sense of Chapter M1) and discards what it maps together. Ask which dimension of the data you have just thrown away.

---

## M5.9 Connections

- **Forward:** Chapter M6 completes the structure: subspaces, rank, orthogonality, eigenvalues and the SVD. Chapter 2 uses all of this for PCA, low-rank structure and attention; Chapter 3 returns to conditioning; Chapter 9 treats a layer as an affine map and backprop as Jacobian products; Chapter 12 builds attention from three matrix multiplications; Chapters 25 and 30 apply normalization and embedding to single-cell data; Chapter 26 treats linear mixed models and LD as covariance matrices; Chapter 19 uses stoichiometric matrices for networks.
- **Backward:** sums and indices (M1); gradients and Jacobians as matrices (M4).

!!! takeaways "Key takeaways"
    1. A **vector** is data and an arrow; a **matrix** is a table and a **linear map**. $\mathbf{A}\mathbf{x}$ is a linear combination of the columns of $\mathbf{A}$.
    2. The **dot product** measures alignment: $\mathbf{a}^\top\mathbf{b}=\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$. Cosine similarity ignores scale; whether that is right is a modelling decision.
    3. Matrix multiplication $(m\times n)(n\times p)\to(m\times p)$ has three views (dot products, transformed columns, sum of outer products), costs about $2mnp$ operations, is associative but **not commutative**.
    4. **Gaussian elimination** costs $\approx\tfrac23n^3$ operations. A square system has a unique solution iff the matrix is invertible; wide systems ($p>n$) have infinitely many solutions or none.
    5. A matrix acts geometrically; $|\det|$ is the volume scale; $\det=0$ means information is lost and there is no inverse. Orthogonal matrices preserve lengths and angles.
    6. The **null space** of the stoichiometric matrix is the set of steady-state flux patterns: $\dim=\#\text{unknowns}-\text{rank}$.
    7. The **condition number** says how many digits a solve will lose. Nearly collinear features give biology's Hilbert matrix: a good fit with meaningless coefficients.
    8. Every normalization projects data and discards something; name what.

---

## Further reading

- Strang, G. *Introduction to Linear Algebra*. Wellesley–Cambridge Press. The standard gentle introduction; his MIT OpenCourseWare lectures are free.
- Axler, S. *Linear Algebra Done Right*. Springer. A proof-oriented second pass.
- Boyd, S. & Vandenberghe, L. *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares* (free online). Applications-first and very clear on shapes and least squares.
- Trefethen, L. N. & Bau, D. *Numerical Linear Algebra*. SIAM. Conditioning and elimination in depth.
- Orth, J. D., Thiele, I. & Palsson, B. Ø. (2010). What is flux balance analysis? *Nature Biotechnology* 28, 245–248. A short, clear account of the stoichiometric null-space view.
