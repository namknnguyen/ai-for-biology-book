# Chapter 29. Classical Models: What Simple Models Already Know

!!! abstract "Chapter at a glance"
    **Motivation.** Before 2015, almost everything computational biology could do was done with a short list of models: position weight matrices, hidden Markov models, maximum-entropy (Potts) models, mixed models, and physics-based energy functions. They are still the right *baselines*, still the right *tools* when data are scarce or mechanism is known, and the source of ideas that deep models reimplement (attention is a Potts model with learned features; a convolution is a bank of PWMs). This chapter derives each, tests it on real or controlled data, and says plainly what it can and cannot do, so that when a foundation model reports a gain you know which classical model it should be compared with.
    **Prerequisites.** Chapters 5, 6 (HMMs), 7, 8, 10, 22, 23, 27, 28.
    **You will be able to:** (1) derive the PWM as a log-likelihood ratio and as an additive energy model, and show that a PWM threshold is meaningless without a background model; (2) build a hidden Markov gene finder and measure what the HMM structure adds to a local classifier; (3) derive the Potts/maximum-entropy model and mean-field and pseudolikelihood inference, and test coevolution methods against ground truth including the effect of phylogenetic redundancy; (4) explain knowledge-based energy functions and free-energy methods as log-odds models; (5) construct a "baseline ladder" for any task in Part VII.

---

## 29.1 Why classical models still matter

Classical models are:

* **Baselines**: a deep model's claim is a *difference* from the strongest simple model. Chapter 26's mixed model, Chapter 24's descriptor regression, and the models of this chapter are those baselines.
* **Interpretable by construction**: the parameters are biological quantities (binding energies, transition probabilities, couplings).
* **Data-efficient**: a PWM from 100 binding sites is estimable; a CNN with $10^6$ parameters needs more.
* **Ancestors**: many neural architectures are classical models with learned features, and understanding the correspondence tells you what a deep model must be *adding* (§29.7).

The unifying statement is that each classical model is a **probabilistic model with an explicit independence or sparsity assumption**: PWM (independent positions), HMM (Markov structure with latent segments), Potts (pairwise interactions only), energy functions (additive pairwise terms). The assumption is both the model's strength (few parameters, closed-form estimates) and its limit.

---

## 29.2 Position weight matrices

### 29.2.1 Derivation

Let $\{x^{(n)}\}$ be $N$ aligned binding sites of width $w$ over $\{A,C,G,T\}$. The **independent-sites model** assumes position $j$ has its own base distribution $P_{j}(b)$ and positions are independent, so $P(x)=\prod_jP_j(x_j)$. The maximum-likelihood estimate is the column frequency $\hat P_j(b)=n_{jb}/N$, regularized with pseudocounts $\alpha$: $\hat P_j(b)=(n_{jb}+\alpha)/(N+4\alpha)$ (a Dirichlet prior; Chapter 4). Against a background distribution $B(b)$ (iid, or Markov), the log-likelihood ratio of a candidate site is

$$
S(x)=\sum_{j=1}^w\log_2\frac{\hat P_j(x_j)}{B(x_j)},
$$

the **PWM score** (also *position-specific scoring matrix*, PSSM). It is the log-odds that $x$ was drawn from the binding-site model rather than the background (a naive Bayes classifier with one feature per position). The expected score under the motif model is $\sum_j\mathrm{KL}(\hat P_j\Vert B)$, the **information content** of the motif in bits (Chapter 5; Schneider et al., 1986): CTCF (JASPAR matrix MA0139.1, 19 positions) has 17.1 bits, the TBP-like matrix MA0108.2 has 9.9 bits.

!!! rhyme "Structural rhyme: the PWM ↔ an additive binding-energy model ↔ a one-layer convolution"
    Under a thermodynamic model (Chapter 22), $P_j(b)\propto e^{-\epsilon_j(b)/RT}$ with $\epsilon_j(b)$ the energy contribution of base $b$ at position $j$, so $-RT\ln$ of the PWM *is* an additive energy matrix (Berg & von Hippel, 1987), and the occupancy of a site is the sigmoid of the score (the same sigmoid as Chapters 23–24). A convolutional filter followed by a nonlinearity (Chapter 10) is exactly a *learned* PWM scan; first-layer filters of trained genomic CNNs are visualized as PWMs for this reason. **A PWM is the one-layer, one-motif, linear-in-features model; everything beyond is interaction or syntax.**

### 29.2.2 The threshold has no meaning without a background

A "hit" is a window whose score exceeds a threshold. How many hits should we expect by chance? The score distribution under a background iid distribution $B$ can be computed *exactly* by dynamic programming (convolving the per-position score distributions), giving a p-value $p(t)=P_B(S\ge t)$. The conventional choice is a *uniform* background ($B=1/4$), but real genomes are not uniform.

**Experiment** (`code/ch29_classical_models.py`): two real JASPAR matrices (JASPAR 2024) scanned across both strands of the 154-kb *Arabidopsis* chloroplast genome (308,920 windows; GC content 0.363). Thresholds were chosen so that the p-value under a *uniform* background is $10^{-3}$, $10^{-4}$, $10^{-5}$; the table compares expected hits under three null models with the observed count:

| Motif | Uniform $p$ | Threshold (bits) | Expected hits: uniform | Expected: genome composition | Expected: order-2 Markov (simulated) | Observed |
|---|---|---|---|---|---|---|
| CTCF (GC-rich) | $10^{-3}$ | 2.33 | 308.2 | 98.4 | 115.4 | 121 |
| CTCF | $10^{-4}$ | 8.36 | 30.8 | 7.3 | 8.3 | 7 |
| CTCF | $10^{-5}$ | 13.14 | 3.1 | 0.6 | 1.2 | 0 |
| TBP-like (AT-rich) | $10^{-3}$ | 6.87 | 307.6 | 742.3 | 798.4 | 713 |
| TBP-like | $10^{-4}$ | 10.69 | 30.8 | 66.1 | 72.3 | 60 |
| TBP-like | $10^{-5}$ | 13.20 | 3.1 | 4.2 | 4.9 | 4 |

A uniform-background threshold **overestimates** hits of the GC-rich motif by 2.5–5× in an AT-rich genome, and **underestimates** hits of the AT-rich motif by 1.4–2.4×; matching composition and then order-2 context brings the expectation close to the observation (115 vs 121 for CTCF at the loosest threshold, 798 vs 713 for the TBP-like matrix; the stricter thresholds have only a handful of hits, so those comparisons are noisy). The scores themselves also change: log-odds depend on $B$. *A reported "number of motif occurrences" is a statement about the motif and the background together.* The same issue pervades enrichment tests: a motif is "enriched" in peaks relative to a background that must match composition, repeat content, and CpG depletion (Chapters 22, 31).

**The specificity paradox again.** At a stringent uniform-background p-value of $10^{-4}$, a 19-bp motif matches about $0.2$ sites per kb on both strands, or of order $10^5$–$10^6$ matches in a $3\times10^9$ bp genome, whereas a TF typically occupies far fewer sites in one cell type [[S]]. Most strong motif matches are unbound; a PWM captures *binding energy of a site in isolation*, not accessibility, cooperativity, or competition (Chapter 22). Predicting *occupancy* needs the features that determine the cellular context.

### 29.2.3 Beyond independent positions: dinucleotides, gapped $k$-mers, and spacing

Adjacent-position dependencies (DNA shape, stacking) are captured by dinucleotide weight matrices or Markov models; composite elements need *two* motifs and a distance. **Gapped $k$-mer SVMs** (gkm-SVM; Ghandi et al., 2014) take features to be counts of all subsequences of length $\ell$ with $k$ informative positions (the others wild-card), so that degenerate and spaced motifs are represented without alignment, with kernel $K(x,y)=\langle\phi(x),\phi(y)\rangle$ evaluated efficiently by a trie. Multiple spaced motifs, flexible spacing, and non-additive combinations ("syntax") were the motivation for the CNNs of Chapter 10, where the AUROC gap between a single-motif PWM scan (0.502) and a CNN (0.999) on spacing-dependent data was demonstrated. **In practice, gkm-SVM is the strongest classical baseline for sequence classification tasks and should be reported alongside any CNN**; the gap above it is the evidence for deeper syntax.

---

## 29.3 Hidden Markov models for annotation: a gene finder

Chapter 6 derived HMMs and the Viterbi algorithm. Genome annotation is a natural application: the latent variable is the *functional class* of each base (non-coding, coding on forward strand in codon position 0/1/2, coding on reverse strand), and the emissions are sequence statistics specific to each class (Krogh et al., 1994; Burge & Karlin, 1997).

**Model.** States $\{N,F_0,F_1,F_2,R_0,R_1,R_2\}$, where $F_k$ is the $k$-th codon position of a forward-strand gene and $R_k$ the $k$-th position (along the plus strand) of a reverse-strand gene. Emissions are order-2 Markov: $P(x_t\mid x_{t-1},x_{t-2},s_t)$ estimated by counting in annotated training data (so the genetic code, codon usage, and the coding periodicity are encoded in the *state-specific* tables). Transitions: the coding states cycle deterministically $F_0\to F_1\to F_2\to F_0$ and may exit to $N$ only after $F_2$ (so genes have whole codons), with exit probability set by $1/(\text{mean codons per gene})$, and entry from $N$ with probability set by the mean intergenic length. Decode with Viterbi: $\hat s_{1:T}=\arg\max_sP(s,x)$.

**Experiment.** The first 128,214 bases of the chloroplast genome (excluding the second copy of the inverted repeat, an exact duplicate, to avoid leakage), 6-fold block cross-validation (train on 5 blocks of 21 kb, decode the sixth):

| Method | Coding precision | Coding recall | F1 | Accuracy |
|---|---|---|---|---|
| Predict everything coding (53% coding) | 0.530 | 1.000 | 0.693 | 0.530 |
| Per-base argmax of emission probabilities (no context) | 0.530 | 0.998 | 0.692 | 0.530 |
| Window log-likelihood-ratio classifier (121 bases, best of six frames; no HMM) | 0.735 | 0.855 | 0.790 | 0.759 |
| **HMM, Viterbi path** | **0.866** | 0.825 | **0.845** | **0.840** |

Of the bases that are truly coding, the Viterbi path assigns the *exact* strand and codon phase to 0.792. Three lessons. **(i)** *Local base statistics alone are useless* (the per-base argmax learns nothing beyond the base rate): the signal is in the **periodicity** and in the consistency of the segment. **(ii)** A **window classifier** captures much of the signal by pooling evidence across 121 bases (F1 0.790); the HMM adds another 0.055 F1 by *enforcing structural consistency* (genes have a frame that persists for hundreds of bases, whole codons, and lengths), the same benefit that a CRF layer or constrained decoding gives a neural sequence tagger. **(iii)** This is a *small, easy, gene-dense* genome (53% coding, no introns except a few, no alternative splicing, no pseudogenes), and the model has no stop-codon, start-codon, or splice-site signals; real eukaryotic gene finders (GENSCAN, AUGUSTUS) add those and reach high accuracy only with species-specific training. Modern deep gene finders couple a neural network emission model with an HMM decoder, keeping the structural prior, and are a template for hybrid models in this book (Chapters 31–33). The caveat is also for evaluation: the folds are contiguous blocks, so a gene spanning a boundary is split between train and test, a small leak, and the sample (a few dozen genes) is far too small for tight confidence intervals.

---

## 29.4 Coevolution: the Potts model and direct-coupling analysis

A protein family's sequences are not independent draws: positions that contact each other co-vary under selection to preserve structure (Chapter 23). The task is to infer *which positions are coupled* from an MSA (Chapter 27).

### 29.4.1 Maximum entropy and the Potts model

Take $B$ aligned sequences $s^{(b)}\in\{1,\dots,q\}^L$ and compute single-site frequencies $f_i(a)$ and pair frequencies $f_{ij}(a,b)$. Among all distributions that reproduce these marginals, the one with maximum entropy (Jaynes, 1957) has the Gibbs form

$$
P(s)=\frac1Z\exp\Big(\sum_ih_i(s_i)+\sum_{i<j}J_{ij}(s_i,s_j)\Big),
$$

the **Potts model** (the $q=2$ case is the Ising model). The fields $h_i$ capture conservation, the couplings $J_{ij}$ the *direct* interactions. The central point: **correlation is not coupling**: positions $i$ and $k$ can be correlated because both couple to $j$ (a chain $i$–$j$–$k$), and the inverse problem of finding $J$ from the marginals, that is, *removing indirect correlations*, is what distinguishes DCA from simply computing mutual information (Weigt et al., 2009; Morcos et al., 2011).

**Inference.** Exact maximum-likelihood needs $Z$ (intractable: $q^L$ terms). Two classical approximations: **mean-field DCA** ($J=-C^{-1}$, the inverse of the covariance matrix with pseudocount regularization; a single matrix inversion; Morcos et al., 2011) and **pseudolikelihood** maximization (plmDCA; Ekeberg et al., 2013), which maximizes $\sum_b\sum_r\log P(s^{(b)}_r\mid s^{(b)}_{\setminus r})$, a set of $L$ multinomial logistic regressions on the one-hot encoding of the other positions, with an $\ell_2$ penalty. The pair score is the Frobenius norm $\lVert J_{ij}\rVert_F$ in a zero-sum gauge, corrected by the average-product correction (APC; Dunn et al., 2008), which removes the background caused by phylogeny and conservation. **Sequence reweighting**: sequences with at least 80% identity to others are down-weighted by one over the cluster size, yielding $N_\text{eff}=\sum_b w_b$, the effective number of independent sequences (Chapter 23).

!!! rhyme "Structural rhyme: Potts ↔ attention ↔ masked-language modeling"
    The pseudolikelihood is *masked language modeling with a one-layer, pairwise model*: predict the residue at position $r$ from all others by $\mathrm{softmax}\big(h_r+\sum_jJ_{rj}[s_j]\big)$. This is a factored form of one attention layer (Chapter 12) with position-specific rather than content-dependent interaction weights; **single attention layers with a Potts-like parameterization recover DCA-quality contacts** (Bhattacharya et al., 2022) [[S]], and trained protein language models (Chapter 34) can be seen as replacing the pairwise-only, family-specific, linear-in-features model by a deep, shared, nonlinear one. A language model's contact predictions are therefore benchmarked against a *Potts model fitted to the same MSA*.

### 29.4.2 A controlled test

A toy family: $L=30$ positions and $q=4$ states with *known* couplings on 27 randomly chosen pairs (of 351 candidate pairs with $|i-j|\ge4$; fields $h_i\sim\mathcal N(0,0.3^2)$, couplings $J_{ij}(a,b)\sim\mathcal N(0,1)$), sampled from the Gibbs distribution (250 sweeps per chain). We score all pairs by (a) mutual information with APC, (b) mean-field DCA, (c) pseudolikelihood DCA, and report the **precision of the top-$K$ pairs** ($K=27$, so a random ranking gives 0.077).

| Sample | $N_\text{eff}$ | MI + APC | Mean-field DCA | Pseudolikelihood DCA |
|---|---|---|---|---|
| Independent, $N=100$ | 100 | 0.70 | 0.59 | 0.74 |
| Independent, $N=500$ | 500 | 0.85 | 0.89 | 0.96 |
| Independent, $N=2{,}000$ | 1,986 | 0.89 | 0.96 | **1.00** |
| Independent, $N=8{,}000$ | 8,000 | 0.85 | 0.96 | **1.00** |
| **Tree**, depth 10, 1 mutation/branch, $N=1{,}024$ (no weighting) | 1,024 → **6.1** | 0.15 | 0.07 | 0.22 |
| Tree, same, 80%-identity reweighting | 6.1 | 0.19 | 0.04 | 0.19 |
| **Tree**, depth 10, 3 mutations/branch, $N=1{,}024$ (no weighting) | 1,024 → 126.7 | 0.44 | 0.48 | 0.59 |
| Tree, same, 80%-identity reweighting | 126.7 | 0.48 | 0.44 | 0.63 |

(Tree rows: sequences evolved along a binary tree under the same Potts landscape, so that leaves share ancestry.) Four lessons. **(i)** **Direct coupling beats raw correlation:** pseudolikelihood DCA reaches perfect precision at $N\ge2{,}000$ while MI+APC plateaus at 0.85–0.89 because indirect correlations survive APC. **(ii)** **The sample size that matters is $N_\text{eff}$, and it can be tiny**: a family of 1,024 sequences with a shallow tree has $N_\text{eff}=6$ and accuracy near chance (0.04–0.22), *worse* than 100 independent sequences (0.74). **(iii)** **Reweighting corrects the count, not the information**: with $N_\text{eff}=6$ there are simply no independent observations; with $N_\text{eff}=127$ reweighting gives a modest gain for the best methods (0.59 → 0.63). **(iv)** Mean-field DCA is poor at small $N$ (0.59 at $N=100$) because it inverts a noisy covariance matrix, and pseudolikelihood is better regularized. In real families, $N_\text{eff}/L$ of a few tens is typically needed for reliable contact prediction [[S]], and many proteins (orphans, fast-evolving families) have much less: *this is the regime in which deep protein models trade on knowledge shared across families* (Chapters 34–35).

**Potts models as fitness models.** Given $(h,J)$, the energy $E(s)=-\sum_ih_i(s_i)-\sum_{i<j}J_{ij}(s_i,s_j)$ is a statistical fitness landscape: the effect of a mutation $s\to s'$ is $\Delta E=E(s')-E(s)$, and the correlation of $-\Delta E$ with measured effects (deep mutational scans; Chapter 23) is the benchmark for **zero-shot** variant-effect predictors (EVmutation; Hopf et al., 2017; Figliuzzi et al., 2016). Independent-site models are the weaker baseline and Potts models the stronger one; both are routinely beaten by large protein language models only where the language model has seen evolutionary information from related families.

---

## 29.5 Energy functions: physics and statistics

**Force fields and free energies** (molecular dynamics; free-energy perturbation, Chapter 24) compute interaction energies from first-principles-inspired functional forms; **Rosetta** and **FoldX** combine physics-based and statistical terms (van der Waals, electrostatics, solvation, hydrogen bonds, torsion preferences) into an empirical energy used for design and for $\Delta\Delta G$ prediction (Chapter 36). **Knowledge-based potentials** estimate energies from the frequencies of structural features in the Protein Data Bank by *inverse Boltzmann*: $E(r)=-kT\ln\big(p_\text{obs}(r)/p_\text{ref}(r)\big)$, where $p_\text{ref}$ is the distribution in a reference state (Sippl, 1990). *This is a PWM in structure space*: a log-odds ratio against a reference, and shares its vulnerability: the result depends on the choice of reference state (§29.2.2), and the database frequencies are not Boltzmann-distributed (structures are biased by crystallization and by evolution), so they are not true free energies.

Where the classical energy functions are strong: pose ranking, mutational effects on stability within a family ($\Delta\Delta G$ prediction correlation around 0.5–0.6 for popular methods on standard sets [[S]]), and *design* in physically constrained settings (Rosetta designed proteins that were experimentally validated for two decades). Where they are weak: absolute binding affinity, conformational change, anything that depends on the unmodeled ensemble. Deep structure predictors (Chapter 35) and generative design (Chapter 36) replaced the *search* and the *fold-level* energy, but physics-based refinement remains a standard final step.

---

## 29.6 The baseline ladder

!!! lens "Research lens: the baseline ladder"
    For each task in Part VII, name the classical baselines in increasing strength. A deep model's claim is the gap above the *top* rung, at equal data and compute:

    | Task | Rung 1 (trivial) | Rung 2 | Rung 3 (strong classical) |
    |---|---|---|---|
    | TF binding / chromatin from sequence | Base composition, nearest-neighbor | PWM scan | gkm-SVM, logistic regression on $k$-mers |
    | Gene annotation | Everything non-coding | Window classifier | HMM (frame-aware) |
    | Protein contacts | Sequence separation | MI + APC | Potts / pseudolikelihood DCA |
    | Variant effect (protein) | Conservation, BLOSUM | Independent-site model | Potts (EVmutation), supervised on one-hot |
    | Genotype → phenotype | Covariates | C+T PGS | LMM / ridge (Chapter 26) |
    | Molecular property | Size, logP | Descriptor regression | Random forest on fingerprints (Chapter 24) |
    | Perturbation response | Control mean | Mean over training perturbations / additive combination | Regression on prior-knowledge features |
    | Expression normalization / mixture | Total counts | NB regression | Hierarchical / empirical Bayes |

    A model can be beaten by Rung 1 on a benchmark whose variance is mostly noise (Chapters 25, 39), and *the baseline ladder is the cheapest honest test of whether a task needs deep learning*.

---

## 29.7 What deep models add, and where classical models win

What a neural model adds is (i) **nonlinearity and interactions** between features beyond pairwise (syntax beyond PWMs and Potts), (ii) **learned features** instead of hand-built ones (shared across tasks and families), and (iii) **scale**: it can use data beyond the family or locus. What it gives up is explicit assumptions, interpretability, data efficiency, and (often) calibration. Classical models win when **data are scarce** (a new TF with 50 sites; an orphan family), **mechanism is known** (energy models), **extrapolation must be controlled** (a linear model extrapolates predictably), or **interpretation is required** (couplings, weights). Deep models win when the signal is a **high-order function of long contexts** and data are abundant (regulatory grammar at megabase scale, structure from sequence across families). The best current systems often combine them: a deep emission model with an HMM decoder, a neural network with a mixed-model correction for relatedness, a language model with Potts-style fine-tuning on a family's MSA.

---

## 29.8 The experiments, verbatim

```python
--8<-- "code/ch29_classical_models.py"
```

```text
== A. A PWM threshold means nothing without a background model (JASPAR matrices scanned over the 154-kb chloroplast genome, both strands) ==

CTCF (GC-rich, 19 bp), matrix MA0139.1: width 19, information content 17.1 bits; genome GC = 0.363; 308,920 windows scanned
threshold chosen so that the p-value under a UNIFORM background is   expected hits (uniform)   expected hits (composition)   expected hits (order-2 Markov sim)   observed
   p_uniform =   1e-03 (score >=  2.33 bits)                        308.2                     98.4                        115.4                       121
   p_uniform =   1e-04 (score >=  8.36 bits)                         30.8                      7.3                          8.3                         7
   p_uniform =   1e-05 (score >= 13.14 bits)                          3.1                      0.6                          1.2                         0

TBP (AT-rich TATA-box-like), matrix MA0108.2: width 15, information content 9.9 bits; genome GC = 0.363; 308,928 windows scanned
threshold chosen so that the p-value under a UNIFORM background is   expected hits (uniform)   expected hits (composition)   expected hits (order-2 Markov sim)   observed
   p_uniform =   1e-03 (score >=  6.87 bits)                        307.6                    742.3                        798.4                       713
   p_uniform =   1e-04 (score >= 10.69 bits)                         30.8                     66.1                         72.3                        60
   p_uniform =   1e-05 (score >= 13.20 bits)                          3.1                      4.2                          4.9                         4

== B. A seven-state HMM gene finder, trained on annotated genes, 6-fold block cross-validation ==
annotated: 0.530 of the 128,214 bases used are coding (0.165 forward, 0.366 reverse)
method                               coding precision   coding recall   F1      accuracy
predict everything coding                  0.530            1.000       0.693   0.530
per-base argmax of emissions (no HMM)      0.530            0.998       0.692   0.530
window LLR classifier, 121 bases (no HMM)  0.735            0.855       0.790   0.759
HMM, Viterbi path                          0.866            0.825       0.845   0.840
of the truly coding bases, the Viterbi path has the exact strand and codon phase for 0.792

== C. Coevolution: contacts from sequence families generated by a known Potts model (L = 30 positions, q = 4 states) ==
27 true coupled pairs among 351 candidate pairs (|i-j| >= 4)
sequences (effective number after 80%-identity weighting)       precision of the top-K pairs:   MI+APC   mean-field DCA   pseudolikelihood DCA
independent samples, N =   100 (N_eff =   100.0)                                       0.70        0.59              0.74
independent samples, N =   500 (N_eff =   500.0)                                       0.85        0.89              0.96
independent samples, N =  2000 (N_eff =  1986.0)                                       0.89        0.96              1.00
independent samples, N =  8000 (N_eff =  8000.0)                                       0.85        0.96              1.00
tree, depth 10, 1 mutation(s) per branch, N = 1024: no weighting (N_eff = 1024)   0.15        0.07              0.22
      with 80%-identity reweighting (N_eff =     6.1)                               0.19        0.04              0.19
tree, depth 10, 3 mutation(s) per branch, N = 1024: no weighting (N_eff = 1024)   0.44        0.48              0.59
      with 80%-identity reweighting (N_eff =   126.7)                               0.48        0.44              0.63
```

---

## 29.9 Worked research examples

!!! example "Worked Research Example 29.1: A CNN outperforms a PWM at predicting TF binding peaks"
    **Situation.** A paper trains a CNN on ChIP-seq peaks for a TF and reports AUROC 0.93 versus 0.74 for the JASPAR PWM. It concludes that the network "learned regulatory grammar beyond the motif."

    **Question.** What does the 0.19 gap measure, and how would you decompose it?

    **Reasoning.**

    1. *Is the PWM baseline properly built?* From §29.2.2: a score threshold or a negative-set background that does not match composition and repeat content changes counts by 2–5× and the PWM's AUROC along with it. Reconstruct the PWM from the *training* peaks (not a library matrix from a different cell type), score both strands, use the *maximum* over windows (not the first), and match the negative set for GC, repeats, and accessibility.
    2. *Is it the strongest classical baseline?* Add a gkm-SVM or logistic regression on $k$-mer counts (Rung 3). A large fraction of the gap between PWM and CNN is typically closed by features a PWM does not have (flanking composition, degenerate variants, multiple motifs); the *residual* gap above gkm-SVM is the evidence for syntax.
    3. *What generates the gap?* Run motif-ablation tests on the CNN (replace the motif instance by a shuffled sequence: if the score collapses, the CNN uses the motif; if the score barely changes, it uses something else). Insert two copies of a motif at varying spacing (synthetic grammar test; Chapter 10) and compare the response against a PWM-sum baseline.
    4. *Are the negatives informative?* If peaks are in accessible chromatin and negatives are random genomic windows, the CNN may be predicting *accessibility* (GC content, CpG islands), a covariate unrelated to this TF (Chapter 22). Use accessible, GC-matched negatives.
    5. *Calibration and decision.* Report AUPRC at the realistic positive rate (a genome-wide scan has positives of order $10^{-3}$, not 50%), since AUROC at balanced classes overstates utility.

    **Expert analysis.** The gap is the sum of: (a) baseline weakness, (b) uninformative negatives, (c) features that are not syntax, (d) true syntax. Only (d) supports the claim, and it is typically the smallest. The decomposition is done by the *ladder*, and the labels A5 (evaluation) and A2 (representation) apply.

!!! example "Worked Research Example 29.2: A protein language model replaces coevolution analysis"
    **Situation.** A group reports that a protein language model predicts contacts with precision of 0.6 for top-$L$ pairs without an MSA, comparable to DCA on deep MSAs. They propose abandoning MSAs.

    **Question.** What would show that the model is doing something other than memorizing families, and when should MSAs still be used?

    **Reasoning.**

    1. *The right comparison.* A Potts model fitted to the family's MSA (pseudolikelihood DCA with APC) at the *same* $N_\text{eff}$, and at the language model's training-set neighborhood for that family. If the family was heavily represented in training, the model has implicitly seen the MSA.
    2. *Stratify by $N_\text{eff}$.* In our simulation the Potts model's precision went from 0.74 ($N_\text{eff}=100$) to 1.00 ($N_\text{eff}\ge2{,}000$) and collapsed to 0.04–0.22 at $N_\text{eff}=6$ despite 1,024 sequences (§29.4.2). A language model that is *flat* in $N_\text{eff}$ is using shared knowledge across families; one that tracks $N_\text{eff}$ is repackaging coevolution.
    3. *Hold out families, not sequences.* Evaluate on families with no close homolog in the pretraining data (a similarity-clustered split; Chapter 43), including orphans, and on *designed* or *de novo* sequences where no evolutionary information exists.
    4. *Mechanism.* Test whether the model's attention heads or its representations recover contacts *beyond* what independent-site statistics plus secondary structure would give, by ablating the specific residues (replace a coupled pair by random residues that preserve the single-site statistics and observe the logits change at the partner).
    5. *When to keep the MSA.* For deep families, an MSA-based model (with its explicit evolutionary information) still adds precision (Chapter 35); for shallow families the language model is the only option, and its uncertainty matters.

    **Expert analysis.** The Potts model plays the role of a *null* in which all information is coevolutionary and pairwise. A language model that beats it must be using something else (higher-order structure, cross-family knowledge, secondary structure prior), and the research question is *what*, which an $N_\text{eff}$-stratified comparison starts to answer. (A9: biological discovery attack: the model's errors as evidence about what is not captured by pairwise coevolution.)

---

## 29.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: implement the baseline in an afternoon"
    **Setting.** You are about to spend a month on a deep model for a biological prediction task.

    1. **Write the baseline ladder** (§29.6) for the task before training anything.
    2. **Implement Rung 3 in one afternoon**: a regularized linear model or gkm-SVM or HMM or Potts, with the *same* train/validation/test split as the deep model. If it takes more than an afternoon, the benchmark is harder than assumed.
    3. **Report the ceiling** (Chapter 1) and the baseline's score side by side. If the baseline is within noise of the ceiling, the task is saturated.
    4. **State the assumption each baseline makes** (independence, pairwise, linearity) and design the *one experiment* that violates it (planted grammar, higher-order coupling, nonadditive effect). Show the baseline failing there, the deep model succeeding or not.
    5. **Check calibration and background**: p-values, thresholds, negatives, and composition.

    **What it teaches.** Most of the credibility of a deep-learning-in-biology paper is in its baselines. This list is also a research method: *where a classical model fails by design is where a deep model has something to add*.

    **An open question to carry forward.** Potts models treat interactions as pairwise and the family as exchangeable draws from one Gibbs distribution, whereas real families have phylogenetic structure, *epistasis of higher order*, and context dependence. How would you design a family-level simulation (Chapter 23's global epistasis plus this chapter's tree) in which pairwise Potts models *provably* fail and a transformer succeeds, and what would the failing mode of the Potts model look like in the ranking of contacts?

---

## 29.11 Connections

- **Backward:** HMMs and Viterbi (Chapter 6); additive thermodynamic models (Chapter 22); global epistasis and DMS (Chapter 23); the noise ceiling and baselines (Chapters 1, 7); alignments and MSAs (Chapter 27); tokenization and representation (Chapter 28).
- **Forward:** sequence-to-function models (Chapter 31); genomic language models and gene annotation (Chapters 32–33); protein language models (Chapter 34) and structure prediction (Chapter 35), where Potts models are the baseline; protein design and energy functions (Chapter 36); benchmark design (Chapter 43).

!!! takeaways "Key takeaways"
    1. A **PWM** is a log-likelihood ratio of an independent-sites model against a background, and an additive energy matrix; its threshold has no meaning without a background model: uniform-background expectations were off by 2.5–5× in the real chloroplast genome, while an order-2 Markov background tracked the observed counts (within about 5% for CTCF and 12% for the TBP-like matrix at the loosest threshold).
    2. **Hidden Markov models** add structural consistency: on the chloroplast genome a frame-aware HMM reached F1 0.845 versus 0.790 for the best window classifier and 0.69 for a local per-base classifier, with exact strand and phase for 79% of coding bases.
    3. **Potts models** separate direct coupling from indirect correlation: pseudolikelihood DCA reached precision 1.00 at $N_\text{eff}\ge2{,}000$ versus 0.85–0.89 for MI+APC.
    4. **$N_\text{eff}$, not $N$, governs coevolution inference**: 1,024 phylogenetically related sequences ($N_\text{eff}=6$) gave near-chance contacts (0.04–0.22), worse than 100 independent ones (0.74).
    5. **Knowledge-based potentials are PWMs in structure space** (inverse Boltzmann against a reference state) and inherit their background problem.
    6. Deep models add **nonlinearity, learned shared features, and scale**; classical models win when data are scarce, mechanisms are known, or interpretation is required.
    7. **The baseline ladder** (trivial → strong classical) is the cheapest honest test of whether a deep model is needed; implement Rung 3 before training.

---

## Further reading

- Stormo, G. D. (2000). DNA binding sites: representation and discovery. *Bioinformatics* 16, 16–23. Schneider, T. D., Stormo, G. D., Gold, L. & Ehrenfeucht, A. (1986). Information content of binding sites on nucleotide sequences. *J. Mol. Biol.* 188, 415–431. Berg, O. G. & von Hippel, P. H. (1987). Selection of DNA binding sites by regulatory proteins. *J. Mol. Biol.* 193, 723–750. Ghandi, M., Lee, D., Mohammad-Noori, M. & Beer, M. A. (2014). Enhanced regulatory sequence prediction using gapped k-mer features. *PLoS Comput. Biol.* 10, e1003711. Castro-Mondragon, J. A. et al. (2022). JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles. *Nucleic Acids Res.* 50, D165–D173.
- Krogh, A., Mian, I. S. & Haussler, D. (1994). A hidden Markov model that finds genes in *E. coli* DNA. *Nucleic Acids Res.* 22, 4768–4778. Burge, C. & Karlin, S. (1997). Prediction of complete gene structures in human genomic DNA. *J. Mol. Biol.* 268, 78–94. Stanke, M. & Waack, S. (2003). Gene prediction with a hidden Markov model and a new intron submodel. *Bioinformatics* 19, ii215–ii225.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Phys. Rev.* 106, 620–630. Weigt, M., White, R. A., Szurmant, H., Hoch, J. A. & Hwa, T. (2009). Identification of direct residue contacts in protein–protein interaction by message passing. *PNAS* 106, 67–72. Morcos, F. et al. (2011). Direct-coupling analysis of residue coevolution captures native contacts across many protein families. *PNAS* 108, E1293–E1301. Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M. & Aurell, E. (2013). Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models. *Phys. Rev. E* 87, 012707. Dunn, S. D., Wahl, L. M. & Gloor, G. B. (2008). Mutual information without the influence of phylogeny or entropy dramatically improves residue contact prediction. *Bioinformatics* 24, 333–340.
- Hopf, T. A. et al. (2017). Mutation effects predicted from sequence co-variation. *Nat. Biotechnol.* 35, 128–135. Figliuzzi, M., Jacquier, H., Schug, A., Tenaillon, O. & Weigt, M. (2016). Coevolutionary landscape inference and the context-dependence of mutations in beta-lactamase TEM-1. *Mol. Biol. Evol.* 33, 268–280. Bhattacharya, N. et al. (2022). Single layers of attention suffice to predict protein contacts (bioRxiv 2020; published later); see also Rao, R. et al. (2021). MSA Transformer. *ICML*.
- Sippl, M. J. (1990). Calculation of conformational ensembles from potentials of mean force. *J. Mol. Biol.* 213, 859–883. Alford, R. F. et al. (2017). The Rosetta all-atom energy function for macromolecular modeling and design. *J. Chem. Theory Comput.* 13, 3031–3048. Schymkowitz, J. et al. (2005). The FoldX web server. *Nucleic Acids Res.* 33, W382–W388.
