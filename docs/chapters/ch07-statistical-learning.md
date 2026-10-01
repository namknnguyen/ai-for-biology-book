# Chapter 7. Statistical Learning: Generalization, Inductive Bias, and Classical Models

!!! abstract "Chapter at a glance"
    **Motivation.** Everything in this book is an instance of supervised or self-supervised learning from finite data. Whether a model that fits the training set will work on new data, and *what "new" means in biology*, is the question statistical learning theory organizes. Classical models (linear, kernel, tree-based) remain the baselines that deep models must beat.
    **Prerequisites.** Chapters 2–5.
    **You will be able to:** (1) state the learning problem and decompose error into approximation, estimation, and optimization; (2) derive the bias–variance decomposition and a finite-class generalization bound; (3) explain double descent and why the interpolation threshold is where least squares explodes; (4) choose among ridge, LASSO, kernels, and boosting by reasoning about genetic/regulatory architecture; (5) run cross-validation without leaking; (6) pick metrics (AUROC vs. AUPRC, between- vs. within-group correlation) that match the decision.

---

## 7.1 The learning problem

**Setup.** There is an unknown joint distribution $p(x,y)$ over inputs $x\in\mathcal{X}$ and targets $y\in\mathcal{Y}$. We observe a training set $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$ and choose a function $f$ from a **hypothesis class** $\mathcal{F}$ (linear functions, trees, networks of a given architecture) to minimize a **loss** $\ell(f(x),y)$.

- **True risk:** $R(f)=\E_{(x,y)\sim p}[\ell(f(x),y)]$, the quantity we care about.
- **Empirical risk:** $\hat R(f)=\frac1n\sum_i\ell(f(x_i),y_i)$, the quantity we can compute.
- **Empirical risk minimization (ERM):** $\hat f=\arg\min_{f\in\mathcal{F}}\hat R(f)$, possibly with regularization.

**The Bayes-optimal predictor.** For squared loss it is the conditional mean $f^\star(x)=\E[y\mid x]$; for 0–1 loss it is the most probable class $\arg\max_cp(c\mid x)$. Its risk $R^\star$ (the **Bayes error**) is the *irreducible* error: no learner can beat it. In biology the Bayes error is often large because the input does not determine the target (noise, missing context): this is Chapter 1's noise ceiling and measurement gap in learning-theory language.

**Error decomposition.** Let $f_{\mathcal{F}}=\arg\min_{f\in\mathcal{F}}R(f)$ be the best function in the class. Then

$$
R(\hat f)-R^\star=\underbrace{R(f_\mathcal{F})-R^\star}_{\text{approximation error}}+\underbrace{R(\hat f)-R(f_\mathcal{F})}_{\text{estimation error}}.
$$

Approximation error says whether the class contains a good function at all (an under-expressive model has high approximation error). Estimation error says whether we can *find* it from $n$ finite samples (a rich class with little data has high estimation error). In practice a third term, **optimization error** (we do not exactly minimize $\hat R$), plays a role (Chapter 3). Large models reduce approximation error and increase estimation error: the classical trade-off, which Chapter 7's later sections complicate.

---

## 7.2 Bias–variance, derived

Consider squared loss and a *random* training set $\mathcal{D}$, so the fitted $\hat f_\mathcal{D}$ is itself random. Fix a test input $x$ and let $y=f^\star(x)+\varepsilon$, $\E\varepsilon=0$, $\Var\varepsilon=\sigma^2$, independent of $\mathcal{D}$. Write $\bar f(x)=\E_\mathcal{D}[\hat f_\mathcal{D}(x)]$. Then

$$
\E_{\mathcal{D},\varepsilon}\big[(y-\hat f_\mathcal{D}(x))^2\big]
=\underbrace{\sigma^2}_{\text{noise}}+\underbrace{\big(f^\star(x)-\bar f(x)\big)^2}_{\text{bias}^2}+\underbrace{\E_\mathcal{D}\big[(\hat f_\mathcal{D}(x)-\bar f(x))^2\big]}_{\text{variance}}.
$$

*Proof.* Add and subtract $f^\star(x)$ and $\bar f(x)$: $y-\hat f=\varepsilon+(f^\star-\bar f)+(\bar f-\hat f)$. Square and take expectations; the cross terms vanish because $\varepsilon$ is independent of everything and mean-zero, and $\E_\mathcal{D}[\bar f-\hat f]=0$. $\square$

**Interpretation.** *Bias* is systematic error from the class being too restrictive or too regularized (underfitting). *Variance* is sensitivity to the particular training sample (overfitting). *Noise* is irreducible. Increasing model flexibility lowers bias and raises variance; regularization does the reverse.

**Biological reading.** With $n=300$ individuals and $p=10^6$ SNPs, an unregularized linear model has near-zero bias but enormous variance. A model that predicts the population mean has zero variance but large bias. Polygenic scoring methods sit between (§7.7).

---

## 7.3 A finite-class generalization bound

How large can $R(\hat f)-\hat R(\hat f)$ be? For a *finite* class $\mathcal{F}$ and a loss bounded in $[0,1]$, Hoeffding's inequality says that for any *fixed* $f$, $P\big(R(f)-\hat R(f)>\epsilon\big)\le e^{-2n\epsilon^2}$. ERM chooses $\hat f$ *after* seeing the data, so we need the bound to hold for all $f\in\mathcal{F}$ simultaneously. The union bound gives $P(\exists f:R(f)-\hat R(f)>\epsilon)\le|\mathcal{F}|e^{-2n\epsilon^2}$. Setting the right side to $\delta$ and solving for $\epsilon$: with probability at least $1-\delta$,

$$
R(f)\le\hat R(f)+\sqrt{\frac{\log|\mathcal{F}|+\log(1/\delta)}{2n}}\qquad\text{for all }f\in\mathcal{F}.
$$

The gap shrinks as $1/\sqrt n$ and grows with the *logarithm of the class size* (the number of bits needed to specify a hypothesis). For infinite classes, $\log|\mathcal{F}|$ is replaced by measures of capacity (VC dimension, Rademacher complexity). Two honest caveats: such bounds are usually **vacuous for modern deep networks** (the capacity is huge, yet generalization occurs), and they assume **i.i.d. data**, which biology routinely violates (§7.9). What the bound *does* teach is a robust principle: **selecting the best of many candidates (many hyperparameters, many architectures, many random seeds, many papers' worth of tuning on the same benchmark) inflates the apparent performance by roughly $\sqrt{\log(\#\text{candidates})/n}$.** This is the winner's curse of Chapter 4 in learning-theoretic form, and it governs how much you can trust a benchmark that has been tuned against for years (Chapter 43).

---

## 7.4 Cross-validation: what it estimates and how it leaks

**$k$-fold cross-validation** partitions the data into $k$ folds; for each fold train on the other $k-1$ and test on the held-out one; average. It estimates the performance of the *training procedure* on new data **drawn from the same distribution as the folds**. Three rules prevent most errors.

1. **Everything that uses labels goes inside the loop.** Feature selection, hyperparameter tuning, normalization fit on labels, class-balancing: all must be done using the *training folds only*. The simulation illustrates the standard failure: on pure noise ($n=60$ samples, $p=4000$ features, labels independent of features), selecting the 20 features most correlated with the labels *using all the data* and then cross-validating gives **0.92 ± 0.05** accuracy (truth 0.50). Selecting inside each training fold gives **0.49 ± 0.08**. The first pipeline reports a confident, completely spurious discovery. This is the commonest way small-$n$, large-$p$ genomics papers manufacture signal.
2. **Folds must match the unit of generalization.** If you want to generalize to new donors, split by donor; to new chromosomes, by chromosome; to new protein families, by cluster. Random splits of correlated data estimate *interpolation* performance (Chapter 1).
3. **Nested cross-validation** for model selection: an outer loop estimates performance; an inner loop (within each outer training set) selects hyperparameters. Reporting the best inner-CV score as the performance estimate is optimistic by the winner's-curse gap.

---

## 7.5 Double descent: overparameterization and the interpolation threshold

Classical theory predicts a U-shaped test-error curve in model complexity. Modern practice (interpolating neural networks) shows test error that *keeps falling* past the point where the model fits the training set exactly. The resolution, a **double descent** curve, is easy to see in least squares.

Fit minimum-norm least squares (Chapter 3, §3.8) on $n=40$ training points, using the first $p$ of $D=200$ Gaussian features, with a dense true signal and noise standard deviation 0.5. The simulation gives test MSE:

| $p$ | 5 | 10 | 20 | 30 | 38 | **40** | 42 | 50 | 80 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test MSE | 1.39 | 1.62 | 2.30 | 4.25 | 20.6 | **153.5** | 18.6 | 5.5 | 1.93 | 1.39 | 1.11 |

Error rises as $p\to n$, explodes at the **interpolation threshold** $p=n$, then *decreases again* as $p$ grows, ending *lower than the best classical choice* ($p=5$: 1.39) at $p=200$ (1.11).

**Why the peak?** At $p=n$ the design matrix $\mathbf{X}\in\R^{n\times n}$ is square; its smallest singular value is close to zero with high probability. The minimum-norm solution is $\hat{\mathbf{w}}=\sum_i\frac{\mathbf{u}_i^\top\mathbf{y}}{\sigma_i}\mathbf{v}_i$ (Chapter 2, §2.8): dividing by tiny $\sigma_i$ amplifies the noise component of $\mathbf{y}$ without bound. For $p>n$, the smallest nonzero singular values grow again (like $\sqrt p-\sqrt n$), noise amplification falls, and the *minimum-norm* choice among interpolating solutions acts as an implicit regularizer. For $p<n$ the variance grows as $p/(n-p-1)$, diverging as $p\to n$. [[E]] (Belkin et al., 2019; Hastie et al., 2022; Nakkiran et al., 2021 for deep networks).

**Lessons for biology.** (i) The worst place to be is **$p\approx n$**: numbers of features comparable to numbers of samples. (ii) *More features can help* even when $p\gg n$, provided the learner has a good implicit bias (min norm, early stopping, regularization). (iii) Comparing a deep model with a linear baseline *requires tuning the baseline's regularization* with the same care: an unregularized linear model near $p\approx n$ is a straw man.

```python
--8<-- "code/ch07_generalization.py"
```

Output (seed 0):

```text
double descent (min-norm least squares): test MSE vs number of features p, n = 40
  p=5:1.39  p=10:1.62  p=20:2.30  p=30:4.25  p=38:20.60  p=40:153.49  p=42:18.61  p=50:5.53  p=80:1.93  p=120:1.39  p=200:1.11

R^2 on held-out data (p=1000, heritability 0.5 so the ceiling is 0.5), mean of 4 simulations:
  n =  300  sparse (20 causal features)      LASSO =  0.31   ridge =  0.06
  n =  300  dense (all 1000 small effects)   LASSO = -0.01   ridge =  0.05
  n = 1500  sparse (20 causal features)      LASSO =  0.46   ridge =  0.24
  n = 1500  dense (all 1000 small effects)   LASSO =  0.18   ridge =  0.26

selection leakage on pure noise (n=60, p=4000, 40 datasets): CV accuracy with selection on all data = 0.92 +/- 0.05; selection inside folds = 0.49 +/- 0.08 (truth: 0.50)
prevalence  0.500: AUROC = 0.899 (rank formula 0.899), AUPRC = 0.898 (random-guess AUPRC = 0.500)
prevalence  0.010: AUROC = 0.899 (rank formula 0.899), AUPRC = 0.200 (random-guess AUPRC = 0.010)
prevalence  0.001: AUROC = 0.897 (rank formula 0.897), AUPRC = 0.045 (random-guess AUPRC = 0.001)

between/within: Pearson over all (gene, individual) points = 0.995; mean within-gene Pearson across individuals = 0.000
```

---

## 7.6 Regularization and inductive bias

**Regularization** restricts or biases the solution: L2 (ridge) shrinks all coefficients; L1 (LASSO) sets many exactly to zero; early stopping, dropout, data augmentation, and weight sharing are other forms. Statistically each corresponds to a prior (Chapter 4); algorithmically to a path through parameter space (Chapter 3).

**Why L1 gives sparsity.** The L1 ball $\{\|\mathbf{w}\|_1\le t\}$ has corners on the coordinate axes, and the level sets of a quadratic loss typically first touch a *corner*, where several coordinates are exactly zero; the L2 ball is smooth and touches off-axis. In one dimension the L1 solution is *soft thresholding*: $\hat w=\mathrm{sign}(z)\max(|z|-\lambda,0)$, which zeros small estimates and shrinks large ones by $\lambda$.

**Inductive bias** is the set of assumptions that lets a learner prefer one consistent hypothesis over another. The **no-free-lunch theorem** (Wolpert, 1996) says that averaged over *all* possible target functions, no learner beats any other: *generalization is only possible because real problems are structured and the learner's bias matches that structure.* Every architecture in this book is an inductive bias, and the central design question in biological ML is *which structure of biology to build in* (Chapter 55, attack A9):

| Architecture | Built-in assumption | Matches biology when… | Violated when… |
|---|---|---|---|
| Linear/ridge | Additive, dense effects | Polygenic traits | Epistasis; threshold effects |
| LASSO | Few relevant features | Oligogenic traits; a few causal cis-variants | Highly polygenic traits |
| CNN (Ch. 10) | Local motifs; translation equivariance | Motif-based regulation | Long-range, context-dependent regulation |
| Transformer (Ch. 12) | Any pairwise interaction; permutation-agnostic without position | Coevolution, long-range contacts | Very long sequences (cost), data-poor regimes |
| GNN (Ch. 16) | Interactions follow a graph | Molecules, interaction networks | Graph unknown or wrong |
| State-space/long conv (Ch. 11) | Smooth, translation-invariant long-range dependence | Megabase-scale regulation | Content-dependent routing |
| Trees/boosting | Piecewise-constant, feature interactions by splits | Tabular, heterogeneous features | Smooth, high-dimensional signals |

---

## 7.7 Classical models you must know, and what each is good for

**Linear and logistic regression.** The baseline for everything. Under the Gaussian noise model the MLE is least squares; the estimator is unbiased with variance $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$, which explodes with collinearity. Add L2 or L1 penalties for $p\gg n$. Polygenic risk scores are linear predictors $\hat y=\sum_m\hat\beta_mG_m$ (Chapter 26).

**Genetic architecture decides which regularizer wins.** In the simulation ($p=1000$ features, heritability 0.5, so the best possible $R^2$ is 0.5):

- *Sparse architecture* (20 causal features): LASSO attains $R^2=0.31$ at $n=300$ and $0.46$ at $n=1500$ (close to the ceiling), while ridge is poor (0.06; 0.24) because it spreads weight over irrelevant features.
- *Dense architecture* (all 1,000 features contribute): at $n=300$ *nothing* works (LASSO $-0.01$, ridge $0.05$): there simply isn't enough data to estimate 1,000 small effects. At $n=1500$ ridge (0.26) beats LASSO (0.18), because LASSO's sparsity assumption is false.

The qualitative message: **the best regularizer is the one that matches the architecture, and highly polygenic architectures need huge $n$** (the reason polygenic scores need biobank-scale cohorts; Chapter 26).

**$k$-nearest neighbors.** Predict by averaging the $k$ closest training examples. No training, no assumptions beyond a meaningful metric. A formidable baseline *when the metric is good* (e.g., sequence-similarity-based function transfer by BLAST is nearest-neighbor learning in disguise), and the reason homology leakage is so damaging: kNN-like models exploit near-duplicates.

**Kernel methods and SVMs.** A kernel $K(x,x')=\langle\phi(x),\phi(x')\rangle$ computes an inner product in a (possibly infinite-dimensional) feature space without constructing it. *Kernel ridge regression* has the closed form $\hat f(x)=\sum_i\alpha_iK(x_i,x)$ with $\boldsymbol\alpha=(\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}$, where $\mathbf{K}_{ij}=K(x_i,x_j)$. *Derivation:* the primal ridge solution $\hat{\mathbf{w}}=(\boldsymbol\Phi^\top\boldsymbol\Phi+\lambda\mathbf{I})^{-1}\boldsymbol\Phi^\top\mathbf{y}$ equals $\boldsymbol\Phi^\top(\boldsymbol\Phi\boldsymbol\Phi^\top+\lambda\mathbf{I})^{-1}\mathbf{y}$ (push-through identity), so predictions are $\hat f(x)=\phi(x)^\top\boldsymbol\Phi^\top\boldsymbol\alpha=\sum_i\alpha_iK(x_i,x)$. Cost is $O(n^3)$. For sequences, the **$k$-spectrum kernel** counts shared $k$-mers; **gapped $k$-mer SVMs** (gkm-SVM; Ghandi et al., 2014) were state-of-the-art sequence-to-regulatory-activity models before CNNs and remain strong baselines (Chapter 29). Kernels can be viewed as the infinite-width limit of neural networks (Jacot et al., 2018), which connects "classical" and "deep" views.

**Gaussian processes (GPs).** A GP places a distribution over functions with covariance $K$; the posterior gives both a prediction *and* calibrated uncertainty: $\mu(x_*)=\mathbf{k}_*^\top(\mathbf{K}+\sigma^2\mathbf{I})^{-1}\mathbf{y}$, $\Var(x_*)=K(x_*,x_*)-\mathbf{k}_*^\top(\mathbf{K}+\sigma^2\mathbf{I})^{-1}\mathbf{k}_*$. GPs are the engine of Bayesian optimization for experimental design (Chapter 46) and a standard model of protein fitness landscapes.

**Trees, random forests, gradient boosting.** A decision tree partitions input space by axis-aligned splits. Random forests average many decorrelated trees (reducing variance). **Gradient boosting** builds an additive model $f_M(x)=\sum_m\eta\,h_m(x)$ by fitting each new tree $h_m$ to the *negative gradient of the loss with respect to the current predictions* (functional gradient descent; for squared loss the negative gradient is the residual). On heterogeneous tabular data (clinical variables, QSAR descriptors, variant annotations) gradient-boosted trees (XGBoost, LightGBM) are frequently the strongest model, and **should be a default baseline in any biological prediction paper that uses tabular features** (Grinsztajn et al., 2022).

**Clustering.** *$k$-means* alternates assigning points to the nearest centroid and recomputing centroids (a hard-assignment EM; Chapter 8). *Hierarchical clustering* builds a dendrogram. *Graph-based community detection* (Louvain, Leiden) on a $k$-nearest-neighbor graph is the standard in single-cell analysis (Chapter 30). All encode the assumption that "similar points belong together," and all return clusters whether or not true clusters exist: *clustering is a representation of the data, not a discovery of cell types* unless validated.

---

## 7.8 Evaluation metrics: match the metric to the decision

**Regression.** *MSE/RMSE* (scale-dependent, penalizes outliers); $R^2$ (fraction of variance explained, relative to the mean predictor); *Pearson $r$* (linear association, scale-free) and *Spearman $\rho$* (rank association, robust to monotone transforms). Pearson/Spearman ignore calibration: a prediction $10\times$ too large can have $r=1$.

**Classification.** *Accuracy* is misleading under imbalance. **AUROC** equals the probability that a randomly chosen positive is scored above a randomly chosen negative (the Mann–Whitney statistic): $\mathrm{AUROC}=\frac{\sum_{i\in\text{pos}}\mathrm{rank}_i-n_+(n_++1)/2}{n_+n_-}$, verified numerically in the code. **AUPRC** (area under precision–recall) depends on prevalence: the random-guess AUPRC *equals the prevalence*.

The simulation holds the score separation fixed so that AUROC $\approx0.90$ at all prevalences, but AUPRC falls from 0.898 (prevalence 50%) to 0.200 (1%) to 0.045 (0.1%), compared with random-guess levels of 0.5, 0.01, 0.001. **Variant pathogenicity prediction, hit identification in screens, and enhancer calling are all extremely imbalanced.** Report the metric tied to the decision: if the user will examine the top 100 of 100,000 candidates, precision at 100 (or enrichment) matters; AUROC may look excellent while the top of the list is poor.

**Calibration.** A model is *calibrated* if events predicted with probability $q$ occur with frequency $q$. Reliability diagrams and the expected calibration error assess it; temperature scaling or isotonic regression repair it. Calibration matters whenever predictions feed decisions or Bayesian combination (Chapter 4's variant classification).

**Between-group versus within-group correlation: the most important metric trap in genomics.** Let $y_{gi}$ be the expression of gene $g$ in individual $i$. Decompose the variance:
$$
\Var(y)=\underbrace{\Var_g(\bar y_g)}_{\text{between genes}}+\underbrace{\E_g[\Var_i(y_{gi}\mid g)]}_{\text{within genes, across individuals}} .
$$
Between-gene variation (housekeeping genes vs. tissue-specific genes; 10,000-fold range) is typically *orders of magnitude* larger than between-individual variation for the same gene. Take a predictor that knows each gene's mean *perfectly* and knows nothing about individuals. In the simulation (between-gene SD 2.0, within-gene SD 0.2) its Pearson correlation computed over all (gene, individual) pairs is **0.995**, but its **mean within-gene correlation across individuals is 0.000**. A model that predicts expression "with $r=0.8$" across genes may have $r\approx0$ for the question that matters for personal genomics and variant effects: *how does this person's expression of this gene differ from another person's?* This is not hypothetical: evaluations in 2023 found that state-of-the-art sequence-to-expression models explained little of between-individual variation in gene expression and often predicted the wrong *direction* of cis-genetic effects, despite high between-gene performance (Huang et al., 2023; Sasse et al., 2023; Chapter 31). The statistic must be matched to the *unit of the question*.

---

## 7.9 When data are not i.i.d.

Generalization bounds, cross-validation, and the bias–variance decomposition assume that test data are drawn from the *same distribution* as training data and that examples are independent. Biological data violate both:

- **Dependence:** phylogeny (related species and sequences), population structure (related individuals), linkage (nearby genomic positions), batch (samples processed together), spatial autocorrelation.
- **Shift:** the deployment distribution differs from training: a new cell type, a new species, a new assay, a new chemistry.

Under **covariate shift** ($p_\text{train}(x)\ne p_\text{test}(x)$ but $p(y\mid x)$ unchanged), importance weighting $w(x)=p_\text{test}(x)/p_\text{train}(x)$ gives an unbiased risk estimate, but requires overlap in support and has high variance. Under **concept shift** ($p(y\mid x)$ changes), no reweighting suffices. Chapter 45 develops these; for now, internalize the rule from Chapter 1: **the test distribution defines the claim.**

---

## 7.10 Worked research examples

!!! example "Worked Research Example 7.1: A deep network beats ridge regression by 2 points of $R^2$ on expression-to-phenotype prediction"
    **Situation.** A paper predicts a clinical phenotype from 20,000-gene expression profiles of 800 patients. A five-layer neural network obtains held-out $R^2=0.31$; the ridge baseline obtains 0.29. The authors conclude that nonlinear interactions matter.

    **Question.** What would you check before accepting that conclusion?

    **Reasoning.**

    1. *Regime.* $p\gg n$ ($20{,}000$ vs. $800$). From §7.5 and §7.7, the winning model depends on implicit bias and tuning, not just class richness.
    2. *Fair baselines.* Was ridge's $\lambda$ tuned by nested CV with the same budget as the network's hyperparameters? Were LASSO, elastic net, and gradient boosting tried? Was the network ensemble compared to a ridge ensemble? Hyperparameter-search budget alone can produce a 2-point gap (winner's curse; §7.3).
    3. *Uncertainty.* With $n=800$ test-set variability, the standard error of $R^2$ is of order 0.02–0.03. Variance across training seeds and across CV folds must be reported. A 2-point difference is within noise.
    4. *Leakage.* Was feature selection (e.g., top-variance genes) done inside CV? Were patients from the same family/site/batch kept in the same fold?
    5. *Nonlinearity test.* If interactions matter, a model with explicit pairwise terms (or a tree ensemble) should also beat ridge. Does the gap shrink as $n$ is reduced or does it grow? A learning curve (performance vs. $n$) that diverges in favor of the network suggests real nonlinear signal; parallel curves suggest a tuning artifact.
    6. *What would change your mind?* A consistent gain across seeds, folds, and an independent cohort; a gain that survives a tuned linear + boosted-tree baseline; and an ablation showing the gain disappears when the network is made linear.

    **Expert analysis.** The gain is *plausibly* noise and tuning; the conclusion "nonlinear interactions matter" is [[X]] on this evidence. The ceiling is likely low (replicate variation in both expression and phenotype), and with $n=800$ and $p=20{,}000$ the achievable $R^2$ for dense architectures is intrinsically limited (cf. the dense $n=300$ row of the simulation, where nothing beats 0.05).

!!! example "Worked Research Example 7.2: \"Our sequence model predicts gene expression with $r=0.82$\""
    **Situation.** A model predicts the expression of each gene from 200 kb of reference sequence, evaluated by Pearson correlation across all genes in a held-out chromosome. A later study uses the same model to predict expression in 400 individuals whose genomes differ at common variants and finds the model correlates with the *variation between individuals* for almost no genes.

    **Question.** Are these results contradictory?

    **Reasoning.**

    1. *What variance does the first evaluation measure?* Between-gene variance. Gene identity (promoter strength, housekeeping vs. tissue-specific, GC content, gene length) explains most of it.
    2. *What variance does the second evaluation require?* Within-gene variance across individuals, driven by tens of cis-regulatory variants with small effects and by trans, environmental and technical factors.
    3. *Information structure.* The reference-sequence model's inputs are *almost identical* across individuals for the same gene (they differ at ~0.1% of positions), so its outputs vary little *by construction*; unless the model is sensitive in the right *direction* to those variants, within-gene predictions will be noise-dominated.
    4. *Quantitatively.* As in the simulation, a predictor that gets gene means perfectly gives total $r\approx0.995$ and within-gene $r\approx0$.
    5. *Alternative explanations for the second result.* Low heritability of expression in the cohort (ceiling), trans effects, tissue mismatch, cohort-specific batch effects, or a real model deficiency in cis-variant effect modeling. A heritability-aware ceiling (the cis-heritability of each gene from eQTL studies) separates these.
    6. *Experiments.* Report within-gene correlation and sign concordance of predicted vs. observed cis-eQTL effects against a *fine-mapped* set of causal variants; compare to a baseline that uses only the lead eQTL variant; stratify by gene cis-heritability.

    **Expert analysis.** Not contradictory: they measure different quantities (the first, a between-gene ability that is valuable for annotating regulatory sequence; the second, a within-gene ability required for personal genomics and variant interpretation). The correct claim for the first study is "ranks genes by baseline expression," not "predicts expression." The lesson recurs in Chapters 31, 41, and 50: **match the statistic to the scientific question, and report the variance decomposition so readers can see which component the model captures.**

---

## 7.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: auditing a model's inductive bias against the biology"
    **Task.** You are asked to propose a model for predicting the effect of non-coding variants on gene expression in a given cell type. Before choosing an architecture, write the *bias audit*.

    1. **What structure does the biology have?** Motifs (local); motif syntax (spacing and orientation, medium range); enhancer–promoter looping (long range, up to $10^5$–$10^6$ bp); cell-type-specific trans factors (global, not in sequence); strand symmetry; additivity of weak effects, but saturation and cooperativity (nonlinear).
    2. **What does the architecture assume?** (Table in §7.6.) CNN: locality and equivariance ✔, long-range ✗ (unless dilations/pooling); attention: any pair ✔, quadratic cost, no built-in equivariance; ...
    3. **What does the data support learning?** With $10^3$–$10^4$ cell-type tracks and $10^8$ positions, thousands of free parameters in attention are supportable, but variant effects (in-distribution rare) are a distribution shift from reference training. Inductive biases substitute for data where data are scarce: *bias is most valuable where data are most limited*.
    4. **What would you test to see whether the inductive bias is right?** Ablate each component (Chapter 18); evaluate on held-out cell types; test reverse-complement symmetry; probe whether predictions respect known biology (motif insertion, spacing).
    5. **Where would adding the "right" bias change the result most?** Typically at the extremes of the data distribution (rare cell types, rare variants, long-range interactions).

    **What this teaches.** An architecture paper is an *argument* that a particular inductive bias matches a particular structure. Your job as a reader is to restate that argument, find its weakest premise, and design the experiment that tests it.

---

## 7.12 Connections

- **Backward:** The pseudo-inverse and singular values (Chapter 2) explain the interpolation peak; min-norm GD (Chapter 3) explains the right side of double descent; priors (Chapter 4) are the Bayesian view of regularization; the winner's curse (Chapter 4) is the finite-class bound's practical face.
- **Forward:** The probabilistic view of models with latent variables (Chapter 8). Deep networks as flexible hypothesis classes (Chapters 9–12). Kernel baselines for regulatory sequence (Chapter 29), mixed models for genetics (Chapter 26), benchmark design and leakage (Chapter 43), shift (Chapter 45).

!!! takeaways "Key takeaways"
    1. Risk = approximation + estimation (+ optimization); the Bayes error is irreducible, and in biology it is often large.
    2. Bias–variance: $\text{MSE}=\sigma^2+\text{bias}^2+\text{variance}$. Selection among many candidates inflates apparent performance by $\sim\sqrt{\log(\#)/n}$.
    3. **Everything that uses labels (feature selection, tuning) must be inside the cross-validation loop**: selection leakage turned pure noise into 92% accuracy.
    4. **Double descent:** error explodes at $p=n$ (tiny singular values), then falls; the minimum-norm solution is an implicit regularizer. Never compare against an untuned linear baseline near $p\approx n$.
    5. **The best regularizer matches the architecture** (sparse → LASSO, dense → ridge); dense architectures need large $n$.
    6. AUROC is prevalence-independent; **AUPRC is not** (random-guess level = prevalence). Choose the metric tied to the decision.
    7. **Between-group vs. within-group correlation:** a model can have $r=0.995$ overall and $r=0$ within groups. Match the statistic to the scientific question.
    8. Every architecture is an inductive bias; the research question is whether it matches the biology.

---

## Further reading

- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. (Chapter 7: the wrong and right way to do cross-validation.)
- Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS* 116, 15849–15854. Hastie, T., Montanari, A., Rosset, S. & Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. *Annals of Statistics* 50, 949–986. Nakkiran, P. et al. (2021). Deep double descent. *J. Stat. Mech.* 2021, 124003.
- Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. *Neural Computation* 8, 1341–1390.
- Ghandi, M., Lee, D., Mohammad-Noori, M. & Beer, M. A. (2014). Enhanced regulatory sequence prediction using gapped k-mer features. *PLoS Computational Biology* 10, e1003711.
- Jacot, A., Gabriel, F. & Hongler, C. (2018). Neural tangent kernel. *NeurIPS*.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS Datasets and Benchmarks*.
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Huang, C. et al. (2023). Personal transcriptome variation is poorly explained by current genomic deep learning models. *Nature Genetics* 55, 2056–2059. Sasse, A. et al. (2023). Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings. *Nature Genetics* 55, 2060–2064.
- Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10, e0118432.
