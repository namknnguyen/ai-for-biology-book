# Chapter 4. Probability, Statistics, and Bayesian Reasoning

!!! abstract "Chapter at a glance"
    **Motivation.** Data are noisy samples from a process; models are probability distributions; scientific claims are inferences under uncertainty. Almost every recurring failure in AI-for-biology papers (inflated significance, leaky evaluation, overconfident predictions, "state-of-the-art" gains within noise) is a statistical failure.
    **Prerequisites.** Chapters 2–3.
    **You will be able to:** (1) derive common loss functions as negative log-likelihoods; (2) derive the negative binomial from a Gamma–Poisson mixture and explain why it models sequencing counts; (3) perform Bayesian updates with conjugate priors and read ridge/LASSO as MAP estimation; (4) explain multiple testing, FDR, and the winner's curse; (5) diagnose pseudoreplication and compute an effective sample size; (6) use likelihood ratios to reason about how much a result should change your belief.

---

## 4.1 Why this chapter matters more than its length suggests

Chapter 1 introduced the *unit of independence* and the *noise ceiling*. This chapter supplies the mathematics for both. It also supplies the language ("likelihood", "posterior", "prior", "evidence") in which Chapter 56 will phrase the evaluation of research ideas. If you retain only one thing: **probability is the language in which a model states what it believes about data; statistics is the discipline of deciding, from finite noisy data, how much to believe it.**

---

## 4.2 Probability essentials

### 4.2.1 Definitions we use constantly

A **random variable** $X$ has a distribution $p(x)$ (mass or density). Joint $p(x,y)$, marginal $p(x)=\sum_yp(x,y)$ (or $\int$), conditional $p(y\mid x)=p(x,y)/p(x)$. **Bayes' rule**:

$$
p(\theta\mid\mathcal{D})=\frac{p(\mathcal{D}\mid\theta)\,p(\theta)}{p(\mathcal{D})},\qquad p(\mathcal{D})=\int p(\mathcal{D}\mid\theta)p(\theta)\,d\theta.
$$

$X$ and $Y$ are **independent** iff $p(x,y)=p(x)p(y)$; they are **conditionally independent given $Z$** ($X\indep Y\mid Z$) iff $p(x,y\mid z)=p(x\mid z)p(y\mid z)$. Conditional independence is the engine of graphical models (Chapter 8) and causal inference (Chapter 44): "the expression of two genes is correlated, but independent given the transcription factor that drives both."

**Expectation and variance.** $\E[f(X)]=\sum_xf(x)p(x)$; $\Var(X)=\E[(X-\E X)^2]$; $\Cov(X,Y)=\E[(X-\E X)(Y-\E Y)]$. Two identities carry a lot of weight:

$$
\E[Y]=\E\big[\E[Y\mid X]\big],\qquad
\Var(Y)=\E\big[\Var(Y\mid X)\big]+\Var\big(\E[Y\mid X]\big).
$$

The second (law of total variance) was used in Chapter 1 to show what a sequence-only model cannot capture. *Proof of the second.* $\Var(Y)=\E[Y^2]-(\E Y)^2$. Condition: $\E[Y^2]=\E[\Var(Y\mid X)+(\E[Y\mid X])^2]$, so $\Var(Y)=\E[\Var(Y\mid X)]+\big(\E[(\E[Y\mid X])^2]-(\E[\E[Y\mid X]])^2\big)=\E[\Var(Y\mid X)]+\Var(\E[Y\mid X])$. $\square$

### 4.2.2 Distributions for biological data

| Distribution | Models | Mean / variance | Appears in |
|---|---|---|---|
| Bernoulli / Binomial | Presence/absence; number of hits among $n$ trials | $np$ / $np(1-p)$ | Binding site occupied; allele counts; design hit rates |
| Categorical / Multinomial | One of $K$ classes (nucleotide, amino acid, cell type) | | Next-token prediction; read counts at a locus |
| Poisson | Rare independent events per unit | $\lambda$ / $\lambda$ | Sequencing reads per position; mutations per Mb |
| **Negative binomial (NB)** | Overdispersed counts | $\mu$ / $\mu+\mu^2/r$ | RNA-seq and UMI counts (Chapters 25, 30) |
| Gaussian | Sums of many small effects; measurement error | $\mu$ / $\sigma^2$ | Log-expression; quantitative traits; noise models |
| Beta | Probability of a probability | | Allele frequency; prior for rates |
| Dirichlet | Distribution over simplices | | Cell-type proportions; topic models |
| Gamma | Positive rates | $k\theta$ / $k\theta^2$ | Rate heterogeneity (phylogenetics); NB construction |

### 4.2.3 The negative binomial as a Gamma–Poisson mixture

Why do sequencing counts have variance exceeding their mean? Because the *underlying rate* varies between samples (cells, replicates) beyond Poisson sampling. Model it as a hierarchy:

$$
\lambda\sim\mathrm{Gamma}(\text{shape}=r,\ \text{scale}=\mu/r),\qquad Y\mid\lambda\sim\mathrm{Poisson}(\lambda).
$$

**Moments.** $\E[\lambda]=r\cdot\mu/r=\mu$ and $\Var(\lambda)=r(\mu/r)^2=\mu^2/r$. By the laws of total expectation and variance,

$$
\E[Y]=\E[\lambda]=\mu,\qquad
\Var(Y)=\E[\lambda]+\Var(\lambda)=\mu+\frac{\mu^2}{r}.
$$

**Marginal distribution.** Integrating out $\lambda$,
$$
p(y)=\int_0^\infty\frac{\lambda^ye^{-\lambda}}{y!}\cdot\frac{\lambda^{r-1}e^{-\lambda r/\mu}}{\Gamma(r)(\mu/r)^r}\,d\lambda
=\frac{\Gamma(y+r)}{y!\,\Gamma(r)}\Big(\frac{r}{r+\mu}\Big)^{r}\Big(\frac{\mu}{r+\mu}\Big)^{y},
$$
the negative binomial. The code at the end of the chapter verifies the variance formula numerically (observed 17.53 vs predicted 17.50 for $\mu=5$, $r=2$; Poisson alone would give 5).

**Interpretation.** The parameter $1/r$ (the *dispersion*) measures how much the *true* rate varies from sample to sample. As $r\to\infty$, NB $\to$ Poisson. For bulk RNA-seq, biological replicates have genuine variation in true expression, so NB is standard (edgeR, DESeq2). For UMI-based single-cell data, much of the apparent overdispersion for a gene with a *single* cell type is small, but heterogeneity between cells (cell size, cell state, ambient RNA) reintroduces it. A frequently repeated claim that droplet scRNA-seq data are "zero-inflated" is largely contradicted by the observation that NB with appropriate means already predicts the observed zeros (Svensson, 2020) [[S]]; the *zero-inflated* variants in older methods mostly fit unmodeled biological heterogeneity. This matters for foundation-model design: whether the likelihood of a model is Gaussian on log-transformed counts, NB, or something else changes what it treats as "noise" (Chapters 30, 38).

!!! bio "Biology for modeling: a UMI count"
    **What is it?** The number of distinct mRNA molecules for gene $g$ captured and sequenced from cell $n$.
    **Information it contains.** An *estimate* of transcript abundance, scaled by capture efficiency, plus sampling noise.
    **How generated and measured.** Cell lysis, reverse transcription with a barcode and a unique molecular identifier, amplification, sequencing, demultiplexing. Each step has a stochastic efficiency.
    **Computational representation.** An integer in a sparse $N\times G$ matrix.
    **What varies.** Capture efficiency per cell (library size), ambient RNA, doublets, cell size, true expression, technical batch.
    **What ML can learn.** Co-variation across genes (modules, cell types, states).
    **What ML cannot see.** Absolute abundance; protein levels; molecules not captured (the 80–95% that are lost); the same cell at another time.

---

## 4.3 Maximum likelihood: every loss function is a negative log-likelihood

Given i.i.d. data $\mathcal{D}=\{(x_i,y_i)\}$ and a model $p_\theta(y\mid x)$, the **likelihood** is $\mathcal{L}(\theta)=\prod_ip_\theta(y_i\mid x_i)$, and the **maximum-likelihood estimate** is $\hat\theta=\arg\max_\theta\sum_i\log p_\theta(y_i\mid x_i)$, equivalently minimizing the **negative log-likelihood (NLL)**.

| Assumed noise model $p_\theta(y\mid x)$ | NLL per example (up to constants) | Familiar name |
|---|---|---|
| $\Normal(y;f_\theta(x),\sigma^2)$ | $\frac1{2\sigma^2}(y-f_\theta(x))^2$ | Mean squared error |
| Bernoulli($\sigma(f_\theta(x))$) | $-y\log\hat p-(1-y)\log(1-\hat p)$ | Binary cross-entropy |
| Categorical($\softmax(f_\theta(x))$) | $-\log\hat p_y$ | Cross-entropy (masked/next-token LM) |
| Poisson($e^{f_\theta(x)}$) | $e^{f}-yf$ | Poisson regression (read counts) |
| NB($\mu=e^{f_\theta(x)}$, $r$) | $-\log\mathrm{NB}(y;\mu,r)$ | Single-cell expression likelihood |
| Laplace($f_\theta(x)$, $b$) | $\frac1b\lvert y-f_\theta(x)\rvert$ | Mean absolute error |

**Consequence 1: your loss is your noise model.** MSE assumes Gaussian homoscedastic errors; applying it to counts or to heavy-tailed measurements silently assumes something false. A model trained with MSE on log-transformed counts treats a gene with 1 read and one with 10,000 reads as equally noisy in log space. That may or may not be reasonable. State it, test it.

**Consequence 2: the likelihood sets what the model considers surprising.** A language model trained with cross-entropy on masked tokens assigns high likelihood to what it has seen often; *its loss on a variant is its judgment of how surprising that variant is under the training distribution*. Whether surprise tracks functional consequence (Chapters 32, 34) is an empirical question about the data-generating process, not a mathematical fact.

**Fisher information and the limits of estimation.** The **Fisher information** $I(\theta)=\E[(\partial_\theta\log p_\theta(X))^2]=-\E[\partial^2_\theta\log p_\theta(X)]$ measures how sharply the data constrain $\theta$. For $n$ i.i.d. observations the **Cramér–Rao bound** states that any unbiased estimator has $\Var(\hat\theta)\ge1/(nI(\theta))$, and the MLE asymptotically attains it with $\sqrt n(\hat\theta-\theta)\to\Normal(0,I(\theta)^{-1})$. Practical implications: (i) uncertainty shrinks as $1/\sqrt n$ *where $n$ is the effective sample size* (§4.8); (ii) parameters that barely change the likelihood (small Fisher information) are poorly determined, which is the statistical face of the ill-conditioning in Chapter 3.

---

## 4.4 Bayesian inference

### 4.4.1 The mechanics

Treat the parameter as a random variable with a **prior** $p(\theta)$. After seeing data, the **posterior** is $p(\theta\mid\mathcal{D})\propto p(\mathcal{D}\mid\theta)p(\theta)$. Predictions integrate over the posterior: $p(y^\ast\mid x^\ast,\mathcal{D})=\int p(y^\ast\mid x^\ast,\theta)p(\theta\mid\mathcal{D})d\theta$.

**Conjugate example (Beta–Binomial).** Suppose a protein-design pipeline yields $k$ binders among $n$ designs tested, with unknown hit rate $\theta$. With a $\mathrm{Beta}(\alpha,\beta)$ prior and a Binomial likelihood, the posterior is $\mathrm{Beta}(\alpha+k,\ \beta+n-k)$. *Proof.* The posterior is proportional to $\theta^{k}(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1}$, which is the kernel of $\mathrm{Beta}(\alpha+k,\beta+n-k)$. $\square$ With a flat $\mathrm{Beta}(1,1)$ prior, 3 hits in 20 designs gives posterior mean $4/22\approx0.18$ and a 90% credible interval of roughly 0.07–0.33: very wide. *This is the quantitative content of "we tested 20 designs".* Reports of hit rates without such intervals overstate precision.

**Gamma–Poisson** is also conjugate: Gamma prior on a rate, Poisson counts, Gamma posterior. This conjugacy is what makes the NB marginal likelihood closed-form and the basis of scalable Bayesian count models.

### 4.4.2 MAP estimation: regularization is a prior

The **maximum a posteriori** estimate maximizes $\log p(\mathcal{D}\mid\theta)+\log p(\theta)$. With a Gaussian prior $\theta\sim\Normal(0,\tau^2\mathbf{I})$, $-\log p(\theta)=\frac{1}{2\tau^2}\|\theta\|^2+\text{const}$, so MAP = **ridge** (L2) regularization with strength $\lambda=\sigma^2/\tau^2$ for a Gaussian likelihood with noise variance $\sigma^2$. With a Laplace prior $p(\theta_j)\propto e^{-|\theta_j|/b}$, MAP = **LASSO** (L1), which prefers sparse solutions. *Weight decay is a Gaussian prior; the strength you choose is a statement about how large you believe parameters can be.* In genomics, ridge corresponds to the infinitesimal model of many tiny effects; LASSO and spike-and-slab priors to a sparse architecture where few variants matter (Chapter 26).

### 4.4.3 Bayes factors and the weight of evidence

To compare hypotheses $H_1,H_0$ given data $D$, write the posterior odds as prior odds times the **Bayes factor**:

$$
\frac{P(H_1\mid D)}{P(H_0\mid D)}=\underbrace{\frac{P(H_1)}{P(H_0)}}_{\text{prior odds}}\times\underbrace{\frac{P(D\mid H_1)}{P(D\mid H_0)}}_{\text{likelihood ratio (Bayes factor)}} .
$$

Taking logs, **evidence is additive**: $\log\text{posterior odds}=\log\text{prior odds}+\sum_k\log\mathrm{LR}_k$ for conditionally independent pieces of evidence. The log likelihood ratio is the *weight of evidence* (in bits if $\log_2$). This is how a good scientist reasons informally: *"how much more likely is this result if my hypothesis is true than if it is false?"* The answer, not the p-value, is what should move your belief.

**A biological instance with published calibration.** The ACMG/AMP guidelines for classifying human genetic variants as pathogenic or benign were recast as a Bayesian framework by Tavtigian et al. (2018): with a prior probability of pathogenicity of 0.10, the "very strong" level of evidence corresponds to an odds-of-pathogenicity of 350:1; "strong", "moderate", and "supporting" correspond to $350^{1/2}\approx18.7$, $350^{1/4}\approx4.33$, and $350^{1/8}\approx2.08$. Evidence from independent sources multiplies: starting from prior odds $0.10/0.90=0.111$, one moderate and two supporting items give odds $0.111\times4.33\times2.08\times2.08\approx2.08$, i.e., posterior probability $0.68$. [[E]] for the published framework; the *calibration* of individual evidence types (including computational predictors) is still debated. Notice the Bayesian structure is the same as the Expert Chain's L8–L11: enumerate hypotheses, assign likelihoods to evidence, and update.

!!! rhyme "Structural rhyme: scientific hypothesis testing ↔ Bayesian updating ↔ clinical variant classification"
    All three are the same computation: prior odds × a product of likelihood ratios. What differs is whether the likelihood ratios are *calibrated* and whether the evidence items are *independent* (when they are not, multiplying them double-counts). Computational predictors used as evidence for variant pathogenicity are a case in point: several predictors trained on overlapping data are not independent, and their joint evidence is less than the product (Chapter 41).

---

## 4.5 Frequentist inference: tests, errors, and what a p-value is

A **null hypothesis** $H_0$ (no effect) specifies a distribution for a test statistic $T$. The **p-value** is $P(T\ge t_{\text{obs}}\mid H_0)$: the probability, *if the null were true*, of seeing a statistic at least as extreme. It is **not** $P(H_0\mid\text{data})$, not the probability the result is a fluke, and not a measure of effect size. A test with significance level $\alpha$ has Type I error rate (false positive rate under the null) $\alpha$; its **power** $1-\beta$ is the probability of rejecting the null when a specified alternative is true. Power depends on effect size, noise, **and the effective sample size** (§4.8).

**Confidence intervals** are random intervals that cover the true value with probability $1-\alpha$ over repetitions of the experiment; they report both effect size and its precision and should be preferred to a bare p-value.

### 4.5.1 Multiple testing

If you test $m$ hypotheses each at level $\alpha$ and all nulls are true, you expect $m\alpha$ false positives. With $m=20{,}000$ genes and $\alpha=0.05$, that is 1,000; with $10^6$ independent genetic variants in GWAS, 50,000. Two standard corrections:

- **Bonferroni:** reject if $p_i\le\alpha/m$. It controls the *family-wise error rate* (FWER: probability of any false positive) because $P(\bigcup_i\{p_i\le\alpha/m\})\le m\cdot\alpha/m=\alpha$ (union bound). It is conservative. The conventional genome-wide significance $p<5\times10^{-8}$ is Bonferroni for about $10^6$ effectively independent common-variant tests.
- **Benjamini–Hochberg (BH):** sort p-values $p_{(1)}\le\dots\le p_{(m)}$; find the largest $k$ with $p_{(k)}\le qk/m$; reject the $k$ smallest. Under independence (or positive dependence) of the tests this controls the **false discovery rate** $\mathrm{FDR}=\E[\text{FDP}]$, where FDP is the fraction of rejections that are false, at level $\pi_0q\le q$ ($\pi_0$ the fraction of true nulls). FDR control is usually the right criterion for discovery screens, where a few false positives are tolerable as long as their *proportion* is controlled.

In the simulation (20,000 genes, 10% truly different, effect of 3 standard errors): raw $p<0.05$ rejects 2,584 genes of which 877 are false (FDP 0.34); Bonferroni rejects only 95 (power 0.05, FDP 0); BH at $q=0.05$ rejects 1,023 with FDP 0.045 and power 0.49. *The same data, three different scientific conclusions.* Notice also that "3,000 significant genes" without a stated procedure is almost uninterpretable.

### 4.5.2 The winner's curse and regression to the mean

When you select the best of many noisy measurements, the winner's measured value is biased upward. Formally, suppose true effects $\theta_i\sim\Normal(0,\tau^2)$ and measurements $\hat\theta_i\mid\theta_i\sim\Normal(\theta_i,s^2)$. The posterior mean of $\theta_i$ given $\hat\theta_i$ is

$$
\E[\theta_i\mid\hat\theta_i]=\frac{\tau^2}{\tau^2+s^2}\,\hat\theta_i,
$$

a **shrinkage** toward the prior mean by the factor $\tau^2/(\tau^2+s^2)<1$. The larger the noise relative to the true spread, the more the measured value should be shrunk. Selecting the top 10 of 1,000 candidates with $\tau=s=1$ in the simulation gives mean *measured* effect 3.76 but mean *true* effect 1.65; the shrunken estimate ($\times0.5$) is 1.88, far closer.

**This is the central statistical fact behind every generate-and-filter pipeline** (protein design, molecule generation, variant prioritization): the *predicted score of the top-ranked candidates systematically overstates their true quality*, and the overstatement grows with the number of candidates and with the noise. Chapter 36 revisits it for design pipelines; Chapter 26 for discovery GWAS effect sizes.

### 4.5.3 Empirical Bayes: borrowing strength

The shrinkage formula needs $\tau^2$. If you have thousands of parallel measurements (genes, variants, perturbations), you can *estimate* $\tau^2$ from the data (the variance of the $\hat\theta_i$ minus $s^2$) and then shrink each gene's estimate. This is **empirical Bayes**, the basis of limma/DESeq2's moderation of per-gene variances and dispersions and of many "joint model across genes" methods. It is the statistical ancestor of *pretraining*: a model trained on many tasks supplies a *prior* that stabilizes inference on each new one. Whenever data per unit are scarce and units are many (rare cell types, single guides, individual variants), empirical Bayes ideas apply.

---

## 4.6 Resampling methods

- **Bootstrap.** Resample the data with replacement, recompute the statistic, repeat; the distribution of recomputed values approximates the sampling distribution. Essential for confidence intervals on arbitrary metrics (AUROC, Spearman, top-$k$ enrichment).
- **Permutation test.** Under $H_0$ that labels are exchangeable, shuffling labels generates the null distribution of the statistic exactly (up to Monte Carlo error). The shuffling must respect the **unit of exchangeability**: permute donor labels, not cell labels, when cells from the same donor are dependent.
- **Block resampling.** For dependent data (genomic positions, time series, spatially correlated tissue), resample contiguous *blocks* long enough to contain the dependence (e.g., a chromosome arm or an LD block) so the bootstrap respects it.

---

## 4.7 Dependence: pseudoreplication and effective sample size

Suppose each observation $x_i$ has variance $\sigma^2$ and any two observations from the same group have correlation $\rho$. For $n$ observations in one group,

$$
\Var(\bar x)=\frac1{n^2}\Big(n\sigma^2+n(n-1)\rho\sigma^2\Big)=\frac{\sigma^2}{n}\big(1+(n-1)\rho\big).
$$

Define the **effective sample size** $n_{\text{eff}}=n/\big(1+(n-1)\rho\big)$: the number of *independent* observations that would give the same precision. As $n\to\infty$ with $\rho>0$, $n_{\text{eff}}\to1/\rho$: **adding more correlated observations stops helping.** If observations are clustered (cells within donors), the real sample size is closer to the number of clusters.

**Pseudoreplication** is analyzing correlated observations as if independent. In the simulation, five donors per condition contribute 200 cells each, there is **no true effect**, and donors differ (intra-donor correlation $\rho=0.5$). A t-test treating all 1,000 cells per condition as independent rejects at $p<0.05$ in **86%** of simulated experiments; a t-test on the five donor-level means ("pseudobulk") rejects 4%, matching the nominal 5%. The effective number of independent observations per condition is about 10 (not 1,000). This is not an obscure technicality: it is the dominant error in single-cell differential-expression analysis, and was shown to inflate false discoveries by orders of magnitude (Squair et al., 2021; Zimmerman et al., 2021). Mixed-effects models with a donor random effect are the alternative to pseudobulk; both respect the unit of independence.

**Genomic analogues.** Positions along a genome are correlated through linkage disequilibrium; sequences within a protein family are correlated through phylogeny; samples within a study are correlated through batch. Each reduces $n_{\text{eff}}$.

```python
--8<-- "code/ch04_statistics.py"
```

Output (seed 0):

```text
Gamma-Poisson: mean = 5.012 (mu = 5.0), var = 17.528 (mu + mu^2/r = 17.500; Poisson alone would give 5.0)

pseudoreplication (5 donors x 200 cells per condition, NO true effect): false positive rate, cell-level test = 0.86; donor-level (pseudobulk) test = 0.04
  intra-donor correlation rho = 0.50;  effective n per condition ~ 10.0 (nominal 1000)

multiple testing (20,000 genes, 2,000 truly different):
  raw p < 0.05           rejections =  2584, false discoveries =   877, FDP = 0.339, power = 0.85
  Bonferroni (FWER 0.05) rejections =    95, false discoveries =     0, FDP = 0.000, power = 0.05
  Benjamini-Hochberg q=.05 rejections =  1023, false discoveries =    46, FDP = 0.045, power = 0.49

Bayesian variant classification (prior P(pathogenic) = 0.10): odds of pathogenicity per evidence level
  supporting = 2.08, moderate = 4.33, strong = 18.71, very strong = 350.00
  one moderate + two supporting -> posterior odds = 2.08, P(pathogenic) = 0.68
  prior odds that a tested hypothesis is true =  1.00: PPV of a significant result = 0.94
  prior odds that a tested hypothesis is true =  0.10: PPV of a significant result = 0.62
  prior odds that a tested hypothesis is true =  0.01: PPV of a significant result = 0.14

winner's curse: top 10 of 1000: mean measured = 3.76, mean TRUE = 1.65, shrunken estimate (x0.50) = 1.88
```

---

## 4.8 Simpson's paradox and why confounding is a design problem

A treatment can look better in every subgroup and worse overall. The classic numbers (Charig et al., 1986), relabeled here as two assay batches (easy/hard):

| | Batch 1 (easy) | Batch 2 (hard) | Overall |
|---|---|---|---|
| Method A | 81/87 (93%) | 192/263 (73%) | 273/350 (78%) |
| Method B | 234/270 (87%) | 55/80 (69%) | 289/350 (83%) |

A wins within each batch, yet B wins overall, because A was used mostly on the hard batch. The aggregate comparison is *confounded* by batch. No statistical method applied to the pooled table alone can know which comparison is right; the answer depends on the causal structure (is batch a confounder that you should condition on, or a mediator you should not?). This is the first appearance of an argument developed fully in Chapter 44: **statistical association needs a causal model to be interpreted.**

---

## 4.9 Worked research examples

!!! example "Worked Research Example 4.1: \"3,000 of 20,000 genes are differentially expressed\""
    **Situation.** A paper reports that 3,000 of 20,000 genes are significantly differentially expressed (adjusted $p<0.05$) between treated and control cells, and interprets the 3,000 as "the treatment response program."

    **Question.** What would you ask before interpreting that number?

    **Reasoning.**

    1. *Which multiple-testing procedure, and what is the target?* BH at $q=0.05$ implies ~150 of the 3,000 are expected to be false; with unadjusted p-values at 0.05 one would expect 1,000 false positives *even if there were no effect*.
    2. *What is the unit of independence?* If the test is across cells from a handful of donors, the nominal p-values are invalid (§4.7); recompute with donor-level or mixed-model tests.
    3. *What are the effect sizes?* A gene can be "significant" with a $1.02$-fold change when $n$ is large. Significance counts conflate power with importance. Report effect-size distributions and set a biological threshold.
    4. *What is the power structure?* Highly expressed genes have more counts and thus more power; the 3,000 may be biased toward highly expressed genes, an artifact of *detection*, not of biology.
    5. *What alternative explanations exist for a very large response?* Ambient RNA; cell-composition shifts (treated cells contain more of a cell type; the "response" is composition); stress from dissociation; batch confounded with condition.
    6. *What experiment or analysis discriminates?* Compare within cell type; use batch-balanced design with technical replicates; check that the response replicates in an independent donor cohort; compute effect-size concordance across donors.

    **Expert analysis.** A large count of "significant genes" is evidence that the data are *powered*, not that the biology is *large*. The scientifically meaningful quantities are effect sizes with uncertainty, consistency across independent units, and the fraction of response explained by composition change versus within-cell-type change. This habit of replacing a count by an *effect-size distribution with calibrated uncertainty* transfers directly to evaluating model predictions (Chapter 43).

!!! example "Worked Research Example 4.2: \"Our model beats the baseline by 1.5 AUROC points\""
    **Situation.** A foundation model scores AUROC 0.865 versus 0.850 for a baseline on a variant-effect benchmark of 20,000 variants drawn from 400 genes. The paper calls this "a significant improvement."

    **Question.** How would you assess this claim statistically?

    **Reasoning.**

    1. *What is the unit of independence?* Variants in the same gene share sequence context and often share labels (pathogenic variants cluster in functional domains); effective $n$ is closer to the number of genes (400) than variants (20,000).
    2. *What is the null distribution of the difference?* Resample at the gene level (cluster bootstrap) and compute the paired AUROC difference for each resample. Also run several training seeds for both models: seed-to-seed standard deviations of 0.5–1 AUROC points are common, which can swallow a 1.5-point gap.
    3. *What does the paired comparison show?* A paired design (same variants, same resamples for both models) greatly reduces variance because the two models' errors are correlated; report the CI of the *difference*.
    4. *What prior should you hold?* Of the many method papers each year that report small gains on a popular benchmark, a substantial fraction do not replicate or are not robust to split changes. With prior odds 1:10 that a given small improvement reflects a real, robust advance, power 0.8 and $\alpha=0.05$, the posterior probability that a "significant" result is real is $0.8\times0.1/(0.8\times0.1+0.05)\approx0.62$ (Ioannidis-style positive predictive value; the code shows 0.94, 0.62, and 0.14 for prior odds 1:1, 1:10, 1:100). Fewer than two-thirds, *even assuming no bias*; more if there is analytical flexibility or benchmark overfitting.
    5. *What would change your mind?* Gains that persist under a harder split (new gene families), across independent benchmarks, with ablations that remove the proposed component and lose the gain.

    **Expert analysis.** Notice that the answer is not "the paper is wrong" but a *quantified skepticism*: the improvement is plausible but not established until it survives cluster-level uncertainty, seed variance, and a shifted test set. The same logic scales to larger improvements: the burden of proof is set by effect size relative to the *noise at the right unit*, not by the number of test items.

---

## 4.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: how much should this result move me? (likelihood-ratio bookkeeping)"
    **Claim to evaluate.** "Model X has learned the regulatory grammar of mammalian enhancers."

    **Step 1: State the alternative.** The natural alternative $H_0$: "Model X exploits composition (GC content, repeats) and local neighborhood shortcuts, not grammar."

    **Step 2: List evidence items and assign rough likelihood ratios *before looking*.** For each, ask: how probable is this observation under $H_1$ versus under $H_0$?

    | Evidence | $P(\cdot\mid H_1)$ | $P(\cdot\mid H_0)$ | LR | Comment |
    |---|---|---|---|---|
    | High held-out AUROC with random split | 0.95 | 0.90 | ≈1.1 | Both hypotheses predict it; nearly uninformative |
    | Performance holds under chromosome split | 0.85 | 0.40 | ≈2 | Moderately informative |
    | Beats GC-content baseline by a wide margin | 0.90 | 0.35 | ≈2.6 | Rules out a simple shortcut |
    | In silico motif insertion changes predictions in the *expected* direction and *dose-dependently* | 0.75 | 0.25 | 3 | Needs control motifs |
    | Predicted motif-pair spacing preferences confirmed by an MPRA designed *after* the model | 0.60 | 0.05 | 12 | Prospective, orthogonal, specific |

    **Step 3: Combine, being honest about dependence.** If all five were independent, the product is about $1.1\times2\times2.6\times3\times12\approx205$. But the first three are correlated (all rely on the same benchmark), so credit them as one item of LR≈3–4 and the total LR is about $3.5\times3\times12\approx126$. With a skeptical prior of 1:20 the posterior odds are about 6:1: *probably, not certainly.* **The prospective MPRA supplies about half of the total weight of evidence** (3.6 of roughly 7.0 bits), from a single experiment, while the three benchmark-based items together supply about 1.8 bits and the motif-insertion test about 1.6.

    **What this teaches.** (i) Decide the likelihoods before the experiment, so you cannot rationalize afterward. (ii) Identify which experiment has the highest LR and *do that one*. (iii) A result both hypotheses predict (random-split AUROC) is nearly worthless as evidence however impressive the number looks.

---

## 4.11 Connections

- **Backward:** Chapter 3's gradients $\mathbf{p}-\mathbf{y}$ are NLL gradients; ridge's $\sigma_i/(\sigma_i^2+\lambda)$ filter is MAP under a Gaussian prior; Chapter 1's noise ceiling and law of total variance come from §4.2.
- **Forward:** Entropy and KL divergence (Chapter 5) are expectations of log-likelihood ratios. The ELBO and latent-variable models (Chapters 8, 14) are Bayesian inference made tractable. Statistical genetics (Chapter 26) is mixed-model and multiple-testing machinery applied at $10^6$–$10^7$ tests. Experimental design (Chapter 46) maximizes expected information gain, i.e., the expected Bayes factor. Idea evaluation (Chapter 56) uses the bookkeeping of §4.10.

!!! takeaways "Key takeaways"
    1. A loss function *is* a noise model: MSE ↔ Gaussian, cross-entropy ↔ categorical, Poisson/NB ↔ counts. Choose it knowingly.
    2. **Negative binomial** = Gamma–Poisson mixture with $\Var=\mu+\mu^2/r$; it is the default for overdispersed sequencing counts.
    3. **Regularization is a prior** (ridge = Gaussian, LASSO = Laplace). **Posterior odds = prior odds × likelihood ratio**; evidence is additive in log space.
    4. A p-value is $P(\text{data this extreme}\mid H_0)$, not $P(H_0\mid\text{data})$. **Multiple testing:** report the procedure; control FDR for discovery, FWER when any false positive is costly.
    5. **Selecting the best of many noisy candidates overstates it** (winner's curse); shrink by $\tau^2/(\tau^2+s^2)$. This governs every generate-and-filter pipeline.
    6. **Effective sample size** is $n/(1+(n-1)\rho)$; clustered data have $n_{\text{eff}}\approx$ number of clusters. **Pseudoreplication** inflated the false-positive rate from 5% to 86% in our simulation.
    7. Association needs a causal model for interpretation (Simpson's paradox). Statistics tells you how precisely, not what it means.

---

## Further reading

- Casella, G. & Berger, R. L. (2002). *Statistical Inference*. The standard graduate text.
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Efron, B. & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge University Press. Empirical Bayes, the bootstrap, and large-scale testing.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B* 57, 289–300.
- Storey, J. D. & Tibshirani, R. (2003). Statistical significance for genomewide studies. *PNAS* 100, 9440–9445.
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine* 2, e124.
- Tavtigian, S. V. et al. (2018). Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework. *Genetics in Medicine* 20, 1054–1060. Richards, S. et al. (2015). *Genetics in Medicine* 17, 405–424 (the guidelines).
- Svensson, V. (2020). Droplet scRNA-seq is not zero-inflated. *Nature Biotechnology* 38, 147–150.
- Squair, J. W. et al. (2021). Confronting false discoveries in single-cell differential expression. *Nature Communications* 12, 5692. Zimmerman, K. D. et al. (2021). A practical solution to pseudoreplication bias in single-cell studies. *Nature Communications* 12, 738.
- Love, M. I., Huber, W. & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology* 15, 550.
- Charig, C. R. et al. (1986). Comparison of treatment of renal calculi by open surgery, percutaneous nephrolithotomy, and extracorporeal shockwave lithotripsy. *BMJ* 292, 879–882.
