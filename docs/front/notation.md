# Notation and Conventions

Consistency matters more than elegance. The book fixes one notation and uses it everywhere. When a symbol must be overloaded, the chapter says so explicitly.

## Typography

| Object | Convention | Example |
|---|---|---|
| Scalars | italic lowercase or uppercase | $a$, $L$, $\lambda$ |
| Vectors | bold lowercase, column vectors | $\mathbf{x} \in \R^d$ |
| Matrices | bold uppercase | $\mathbf{W} \in \R^{m \times n}$ |
| Tensors | bold uppercase with shape stated | $\mathbf{X} \in \R^{B \times L \times d}$ |
| Sets, spaces | calligraphic | $\mathcal{X}$, $\mathcal{D}$ |
| Random variables | uppercase; realizations lowercase | $X$, $x$ |
| Distributions | $p$, $q$ with subscripts when needed | $p_\theta(x)$, $p_{\text{data}}(x)$ |
| Expectation, variance | $\E$, $\Var$, $\Cov$ | $\E_{x \sim p}[f(x)]$ |
| Indicator | $\mathbb{1}[\cdot]$ | $\mathbb{1}[y=1]$ |
| Transpose | $\mathbf{A}^\top$ | |

## Core machine-learning symbols

| Symbol | Meaning |
|---|---|
| $\theta$ | Model parameters |
| $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ | Dataset ($n$ examples) |
| $\mathcal{L}(\theta)$ | Loss as a function of parameters; $\ell(\hat{y}, y)$ is per-example loss |
| $f_\theta$ | A predictive function; $\hat{y} = f_\theta(x)$ |
| $\phi_\theta(x) = \mathbf{h}$ | Representation (embedding) of $x$ |
| $p_\theta(x)$ | Model distribution |
| $\nabla_\theta \mathcal{L}$ | Gradient; $\mathbf{J}_f$ Jacobian of $f$ |
| $\eta$ | Learning rate |
| $\KL{p}{q}$ | Kullback–Leibler divergence $\E_p[\log p/q]$ |
| $\Ent(X)$, $\MI(X;Y)$ | Entropy, mutual information (in nats unless stated) |
| $\softmax(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$ | Softmax |

## Tensor-shape conventions

Whenever code is shown, shapes appear in comments.

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length (nucleotides, residues, tokens, or genes, as stated) |
| $V$ | Vocabulary size |
| $d$ | Model width (embedding dimension) |
| $H$ | Number of attention heads; $d_h = d/H$ |
| $N$ | Number of cells, individuals, or sequences in a dataset (stated per chapter) |
| $G$ | Number of genes |
| $M$ | Number of genetic variants (SNPs) |
| $K$ | Number of clusters, latent dimensions, or components (stated) |
| $T$ | Number of diffusion/time steps |

A tensor shape is written `(B, L, d)` in code and $B \times L \times d$ in text.

## Biological symbols (fixed across the book)

| Symbol | Meaning |
|---|---|
| $\mathcal{A}$ | Alphabet: $\{A,C,G,T\}$ for DNA, $\{A,C,G,U\}$ for RNA, 20 amino acids for protein (plus special tokens) |
| $x_{1:L}$ | A biological sequence of length $L$ |
| $\mathbf{s}$ | The full *latent biological state* (of a cell, tissue, or organism) |
| $g$ | Genotype (the DNA sequence, including variants) |
| $e$ | Environment |
| $c$ | A *condition* or perturbation (a CRISPR guide, a drug and dose, a cytokine) |
| $t$ | Cell type or cell state label |
| $y$ | A phenotype or target quantity |
| $\mathcal{M}_k$ | Measurement operator for assay $k$: $x^{(k)} = \mathcal{M}_k(\mathbf{s}, c; \xi)$ |
| $\xi$ | Technical noise and batch variables of a measurement |
| $\mathbf{X} \in \R_{\ge 0}^{N \times G}$ | Cell-by-gene count matrix |
| $\mathbf{G} \in \{0,1,2\}^{N \times M}$ | Genotype matrix (allele counts) |
| $\mathbf{C} \in \R^{L \times L}$ | Contact or coupling map |
| $N_e$ | Effective population size |
| $s$ (population genetics) | Selection coefficient (context disambiguates from latent state $\mathbf{s}$; the chapter says which) |
| $\mu$, $r$ | Mutation rate, recombination rate |
| $\Delta\log p(x \to x')$ | Log-likelihood ratio of a variant, $\log p_\theta(x') - \log p_\theta(x)$ |

## The Expert Chain: link labels

| Label | Link |
|---|---|
| **L1** | Problem |
| **L2** | Existing approaches |
| **L3** | Assumptions |
| **L4** | Why they might work |
| **L5** | Failure modes |
| **L6** | Bottlenecks |
| **L7** | Open questions |
| **L8** | Hypotheses |
| **L9** | Candidate solutions |
| **L10** | Experiments |
| **L11** | Interpretation |
| **L12** | New research directions |

## The Four Gaps: labels

| Label | Gap |
|---|---|
| **G-M** | Measurement gap |
| **G-O** | Objective gap |
| **G-I** | Inference (causal) gap |
| **G-G** | Generalization gap |

## The Ten Attacks: labels

| Label | Attack |
|---|---|
| **A1** | Assumption |
| **A2** | Representation |
| **A3** | Objective |
| **A4** | Data |
| **A5** | Evaluation |
| **A6** | Scale |
| **A7** | Cross-domain transfer |
| **A8** | Problem reformulation |
| **A9** | Biological constraint |
| **A10** | Biological discovery |

## Units, logarithms, and indices

- $\log$ is the natural logarithm unless the base is stated. Information is in **nats** by default; bits are used when discussing information content of sequences ($\log_2$).
- Indices: $i, j$ over examples or positions; $k$ over assays or components; $\ell$ over layers; $t$ over time steps or cell types (disambiguated by context).
- Sequences are 1-indexed in text and 0-indexed in code.
- Genomic coordinates are given for a stated reference assembly (for example GRCh38/hg38); intervals are half-open in code.

## Evidence badges

[[E]] Established &nbsp; [[S]] Strong evidence &nbsp; [[P]] Plausible &nbsp; [[H]] Open hypothesis &nbsp; [[X]] Speculative
