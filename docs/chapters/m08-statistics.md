# Chapter M8. Statistics from the Ground Up: Estimation, Uncertainty, and Testing

!!! abstract "Chapter at a glance"
    **Motivation.** Probability (Chapter M7) runs from a known model to the data it would produce. Statistics runs the other way: from the data you have, to what you can say about the process that generated them, and *how sure you can be*. Every experimental claim in biology is a statistical claim, and nearly every method in this book (training a model, comparing two models, calling a differentially expressed gene, finding a GWAS hit) is a statistical procedure. This chapter teaches the logic: estimators and their errors, maximum likelihood, confidence intervals, hypothesis tests, p-values, the multiple-testing problem that dominates genomics, regression, and the confounding traps that cause most false discoveries.

    **Prerequisites.** Chapter M7 (random variables, distributions, CLT). Chapter M3 (derivatives, for likelihood). Chapter M6 helps for regression.

    **You will be able to:** (1) distinguish a parameter from an estimator and compute bias, variance and standard error; (2) derive a maximum-likelihood estimate and connect it to loss minimization; (3) interpret a confidence interval correctly and compute its coverage by simulation; (4) state exactly what a p-value is and is not; (5) compute power and choose a sample size; (6) control the false discovery rate with Benjamini–Hochberg; (7) use the bootstrap and permutation tests; (8) explain regression to the mean, the winner's curse and Simpson's paradox; (9) update a belief with Bayes' rule using a conjugate prior.

---

## M8.1 From data to claims about a population

We observe a **sample** of $n$ values $x_1,\dots,x_n$ drawn from some process, the **population** (or data-generating distribution), that we cannot see directly. A **parameter** $\theta$ is a fixed unknown property of the population: its mean, a rate, a regression slope. An **estimator** $\hat\theta=T(x_1,\dots,x_n)$ is a *rule* that turns a sample into a guess for $\theta$; its value on a particular sample is an **estimate**.

The key conceptual step: **an estimator is a random variable**. If we repeated the experiment, we would get a different sample and a different estimate. The distribution of $\hat\theta$ across repeated experiments is its **sampling distribution**, and its properties describe the quality of the rule:

- **Bias:** $\operatorname{bias}(\hat\theta)=\mathbb{E}[\hat\theta]-\theta$ (systematic error).
- **Variance:** $\operatorname{Var}(\hat\theta)$ (how much it fluctuates). Its square root is the **standard error**. For the sample mean, $\mathrm{SE}=\sigma/\sqrt n$ (Chapter M7).
- **Mean squared error:** $\operatorname{MSE}=\operatorname{bias}^2+\operatorname{Var}$. Accepting a little bias to cut variance can lower MSE, which is the entire logic of regularization (Chapter 7).

An estimator is **consistent** if it converges to $\theta$ as $n\to\infty$. A simple example of bias: estimating a variance. The MLE divides the sum of squared deviations by $n$; but deviations are measured from the *sample* mean, which is closer to the data than the true mean, so the sum is too small. In the script, with $n=5$ and true variance 4, the divide-by-$n$ estimator averages $3.207$ (theory $\tfrac{n-1}{n}\times4=3.20$), while dividing by $n-1$ gives $4.009$, unbiased. This "degrees of freedom" correction accounts for one parameter (the mean) having been estimated from the same data. It matters most for small $n$, which is the usual situation for experiments with a few replicates.

---

## M8.2 Maximum likelihood

The **likelihood** $L(\theta)=P(\text{data}\mid\theta)$, viewed as a function of $\theta$ for the *fixed observed data*, says how plausible each parameter value makes the data. The **maximum-likelihood estimate (MLE)** is the $\theta$ that maximizes it. For independent observations, $L=\prod_ip(x_i\mid\theta)$, and one maximizes the **log-likelihood** $\ell(\theta)=\sum_i\ln p(x_i\mid\theta)$ (the log turns the product into a sum, Chapter M2, and has the same maximizer).

Three examples, each a short calculus exercise (Chapter M3):

| Model | Log-likelihood | MLE |
|---|---|---|
| Bernoulli($p$), $k$ successes in $n$ | $k\ln p+(n-k)\ln(1-p)$ | $\hat p=k/n$ |
| Poisson($\lambda$), counts $k_i$ | $\ln\lambda\sum k_i-n\lambda+\text{const}$ | $\hat\lambda=\bar k$ |
| Gaussian($\mu,\sigma^2$) | $-\tfrac n2\ln\sigma^2-\frac{1}{2\sigma^2}\sum(x_i-\mu)^2+\text{const}$ | $\hat\mu=\bar x$, $\hat\sigma^2=\tfrac1n\sum(x_i-\bar x)^2$ |

**Loss functions are negative log-likelihoods.** For a Gaussian with fixed variance, maximizing the likelihood in $\mu$ is the same as *minimizing the sum of squared errors*; that is why least squares is the "natural" loss for Gaussian noise. For a Bernoulli (binary) outcome, the negative log-likelihood is the **cross-entropy** loss used to train every classifier. For Poisson counts, the Poisson negative log-likelihood is the loss. Training a model "to minimize cross-entropy" and "to maximize the likelihood of the data under the model" are the same statement (Chapters 5 and 9). Choosing a loss is therefore choosing a noise model, and a mismatched noise model (Gaussian loss on over-dispersed counts) is a modelling error, not a technicality (Chapter M7's negative binomial).

**Properties.** Under regularity conditions the MLE is consistent, asymptotically unbiased and asymptotically the most precise estimator possible; its standard error is governed by the curvature of the log-likelihood at the maximum (the **Fisher information**, $-\ell''$): *a sharply peaked likelihood means the data pin the parameter down*. The second derivative of Chapter M3 reappears as a measure of uncertainty. In a model with many parameters the Hessian matrix (Chapter M4) plays that role, and a near-singular Hessian means unidentifiable parameters (Chapter M6).

---

## M8.3 Confidence intervals

A point estimate is not enough; we want a range. A **95% confidence interval** is computed by a procedure that, *over repeated experiments*, contains the true parameter in 95% of them. For a mean with unknown variance the standard recipe is $\bar x\pm t_{n-1,0.975}\cdot s/\sqrt n$, using the **Student $t$ distribution** with $n-1$ degrees of freedom (which is wider than the Gaussian to allow for the estimated $s$).

!!! warning "What a confidence interval is not"
    A computed interval such as $[9.2,\,11.5]$ does **not** have a 95% probability of containing the true value: the true value is a fixed number and either is or is not inside. The 95% describes the *procedure's* long-run success rate. (The Bayesian credible interval of §M8.8 does give a probability statement about the parameter, but it needs a prior.) Whenever you quote "95% CI", be clear about which meaning you intend.

**Coverage is checkable by simulation.** In the script, 20,000 experiments sample $n$ values from $\mathcal{N}(10,3^2)$ and form two intervals: one using the Gaussian critical value 1.96 (the "z-interval") and one using the $t$ critical value:

| $n$ | z-interval coverage | t-interval coverage |
|---|---|---|
| 5 | 0.881 | 0.952 |
| 10 | 0.920 | 0.951 |
| 30 | 0.937 | 0.948 |
| 100 | 0.949 | 0.952 |

A nominal 95% interval that covers only 88% of the time with 5 replicates is a typical way small experiments overstate their certainty. Whenever you can, *check the coverage of an interval procedure by simulation*, especially for non-Gaussian data or complicated estimators.

**The bootstrap.** When no formula for the sampling distribution is available, *use the data themselves* as the population: resample $n$ values from the sample with replacement, compute the statistic, and repeat thousands of times; the spread of the resampled statistics approximates the sampling distribution. For 10 skewed values (script) the bootstrap 95% interval for the *median*, a statistic with no simple formula, is $[2.85,\,7.25]$ around the sample median of $4.00$. The bootstrap is simple, general and widely used for confidence intervals on arbitrary statistics (a correlation, a model's AUROC), but it assumes the sample is representative and the observations are independent, and it can fail for extreme statistics and very small $n$.

---

## M8.4 Hypothesis tests and p-values

A **hypothesis test** asks whether the data are surprising under a default assumption, the **null hypothesis** $H_0$ (typically "no effect": two groups have the same mean). You choose a **test statistic** (for example, the difference of group means divided by its standard error), work out its distribution *assuming $H_0$ is true*, and compute the **p-value**:

$$
p=P\big(\text{a statistic at least as extreme as the one observed}\ \big|\ H_0\text{ is true}\big).
$$

!!! warning "What a p-value is not"
    The p-value is **not** the probability that the null hypothesis is true, **not** the probability that the result is a fluke, and **not** a measure of effect size. It is $P(\text{data}\mid H_0)$, not $P(H_0\mid\text{data})$: the conditional-probability confusion from Chapter M7 once more. A tiny p-value with a trivial effect (huge $n$) and a large p-value with a large effect (tiny $n$) are both common. Report the **effect size and its interval** along with the p-value.

**Properties.** When $H_0$ is true and the test is valid, p-values are **uniformly distributed** on $[0,1]$: any threshold $\alpha$ is crossed with probability exactly $\alpha$. In the script, 10,000 genes with no true difference (10 vs 10 samples) give a flat p-value histogram, $[1025,1022,1019,964,1046,979,970,995,944,1036]$ over ten bins, and $500$ genes with $p<0.05$, exactly the expected $5\%$. *Testing many null hypotheses at $\alpha=0.05$ guarantees about 5% false positives*, which is the origin of the multiple-testing problem below.

**Errors and power.** Rejecting a true $H_0$ is a **type I error** (false positive; rate $\alpha$). Failing to reject a false $H_0$ is a **type II error** (false negative; rate $\beta$). **Power** $=1-\beta$ is the probability of detecting a real effect of a given size. Power increases with sample size, with effect size and with $\alpha$. Script: with a true effect of 0.5 standard deviations and $\alpha=0.05$, a two-sample $t$-test has power $0.181$ at $n=10$ per group, $0.335$ at $20$, $0.805$ at $64$ and $0.979$ at $128$. The rule of thumb $n\approx16/d^2$ per group for 80% power at standardized effect $d$ gives $64$ for $d=0.5$, matching the simulation. **Under-powered studies not only miss real effects; the effects they do find are exaggerated** (the winner's curse, §M8.6). Always compute the power of a planned experiment *before* running it.

**Permutation tests.** To test $H_0$ without distributional assumptions, shuffle the group labels many times, recompute the statistic each time, and ask how often the shuffled statistic is as extreme as the observed one. For two groups of five (script), with observed difference of means $1.42$, the permutation p-value is $0.0213$ (50,000 shuffles) and the $t$-test gives $0.0105$: same conclusion, different arithmetic. The permutation test is valid whenever labels are *exchangeable under $H_0$*, and it generalizes to complicated statistics (Chapter 54 uses it to control a search over many analyses).

---

## M8.5 Multiple testing

A genomics experiment tests thousands of hypotheses at once. At $\alpha=0.05$ with 10,000 null genes you expect 500 false positives; if only 500 genes truly differ, the false positives can *outnumber* the true ones. Two standard ways to control the error:

- **Family-wise error rate (FWER)**, the probability of *any* false positive. The **Bonferroni** correction tests each hypothesis at $\alpha/m$ ($m$ tests). It is simple and conservative: power is lost quickly as $m$ grows.
- **False discovery rate (FDR)**, the expected *proportion* of false positives among the discoveries. The **Benjamini–Hochberg (BH)** procedure: sort the p-values $p_{(1)}\le\dots\le p_{(m)}$; find the largest $k$ with $p_{(k)}\le\frac{k}{m}q$; declare the $k$ smallest significant. It controls $\mathrm{FDR}\le q$ under independence and some forms of positive dependence.

The script simulates 10,000 genes of which 500 are truly differential (effect 1.5 standard deviations, 10 vs 10 samples):

| Method | Discoveries | True | False | False discovery proportion | Power |
|---|---|---|---|---|---|
| $p<0.05$, uncorrected | 964 | 452 | 512 | 0.531 | 0.904 |
| Bonferroni | 9 | 9 | 0 | 0.000 | 0.018 |
| Benjamini–Hochberg, $q=0.05$ | 175 | 165 | 10 | 0.057 | 0.330 |

Without correction, more than half of the "hits" are false. Bonferroni is nearly empty. BH finds $175$ genes with a realized false-discovery proportion of $5.7\%$, close to the $5\%$ promised. The lesson is not that one method is best: *FWER is for claims where a single false positive is costly; FDR is for exploratory screens where a modest share of false leads is acceptable and follow-up is cheap.*

!!! bio "Biology for modeling: where multiple testing bites"
    **What is it?** Differential expression across 20,000 genes; GWAS testing $10^6$ variants (the genome-wide significance threshold $5\times10^{-8}$ is a Bonferroni correction for about a million independent tests); scanning thousands of model checkpoints or hyperparameters and reporting the best.

    **The hidden version.** If you try several analysis pipelines, normalization choices or subgroups and report the one that gives $p<0.05$, you have performed multiple tests without counting them ("researcher degrees of freedom" or the *garden of forking paths*). The same applies to machine-learning benchmarks: selecting the best of many models on one test set inflates the reported performance (Chapters 43 and 54).

    **What would change the interpretation.** Tests are rarely independent: neighbouring variants (LD) and co-expressed genes are strongly correlated, so the *effective* number of tests is smaller than the nominal one and Bonferroni over-corrects; FDR procedures that allow arbitrary dependence exist.

---

## M8.6 Regression, regression to the mean, and the winner's curse

**Linear regression** models a response as $y=\mathbf{x}^\top\mathbf{w}+\varepsilon$ with noise $\varepsilon$ of mean 0 and variance $\sigma^2$. Chapter M4 derived the least-squares fit $\hat{\mathbf{w}}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ and Chapter M6 showed it as a projection. Its standard errors come from $\operatorname{Cov}(\hat{\mathbf{w}})=\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$: **correlated features inflate the variance of the coefficients** (the near-singular $\mathbf{X}^\top\mathbf{X}$ of Chapter M5). The **coefficient of determination** $R^2=1-\sum(y_i-\hat y_i)^2/\sum(y_i-\bar y)^2$ is the fraction of variance explained; it never decreases when a feature is added, so a high $R^2$ on training data is no evidence of predictive value (Chapter 7).

**Regression to the mean and the winner's curse.** Suppose each measurement is the true effect plus independent noise of the same size. If you select the *extreme* values, many are extreme partly by luck, and a re-measurement will be less extreme. In the script, true effects are $\mathcal{N}(0,1)$ and measurement noise is $\mathcal{N}(0,1)$. The top 1% of a first screen has mean measured value $3.80$, but re-measured it is $1.88$, and the mean *true* effect is $1.93$: **the first measurement overstates the true effect by a factor of two**. The best linear predictor of the true effect from a single measurement shrinks it by the factor $\operatorname{Var}(\text{signal})/\operatorname{Var}(\text{signal}+\text{noise})=0.50$. This is the **winner's curse**: the top hits of any noisy screen are biased upward; GWAS effect sizes, drug-screen hits and "state-of-the-art" benchmark numbers all follow the pattern. The remedies are to replicate on independent data, to shrink estimates (empirical Bayes), and to evaluate on data not used to select.

---

## M8.7 Confounding and Simpson's paradox

A **confounder** is a variable that influences both the exposure (or treatment) and the outcome, producing an association that is not causal. **Simpson's paradox** is the extreme form: a trend in each subgroup *reverses* when the groups are combined.

The classic data (adapted from the study of kidney-stone treatments, Charig et al., 1986) are in the script, with the treatments relabelled "treated" and "untreated" and the two strata called "batches":

| | Batch 1 (easy cases) | Batch 2 (hard cases) | Overall |
|---|---|---|---|
| treated | $81/87=0.93$ | $192/263=0.73$ | $273/350=0.78$ |
| untreated | $234/270=0.87$ | $55/80=0.69$ | $289/350=0.83$ |

The treatment does better in each batch (0.93 vs 0.87; 0.73 vs 0.69) but worse overall (0.78 vs 0.83), because the treated group contains mostly hard cases: **case difficulty is a confounder** that influences both who gets treated and the outcome. Neither the aggregate nor the stratified numbers are "the truth" by themselves: *which is right depends on the causal structure* (is the batch a confounder to adjust for, or a consequence of the treatment?). Chapter 44 develops that reasoning with causal graphs; for now the rule is: **before you adjust for a variable or leave it out, say what causes what.**

Biological versions abound: tissue composition confounds differential expression between disease and control (more immune cells in disease samples); ancestry confounds genotype–phenotype association; sequencing batch confounds condition when samples were not randomized across batches (Worked Research Example 2.1).

---

## M8.8 A taste of Bayesian inference

The Bayesian approach (Chapter M7's Bayes' rule) treats the parameter as uncertain and updates a **prior** $p(\theta)$ with data to a **posterior**:

$$
p(\theta\mid\text{data})\ \propto\ p(\text{data}\mid\theta)\,p(\theta).
$$

For a probability $\theta$ with a **Beta**$(a,b)$ prior and $k$ successes in $n$ trials, the posterior is exactly **Beta**$(a+k,\,b+n-k)$: the prior acts like $a$ earlier successes and $b$ earlier failures (a **conjugate** prior). With a uniform prior ($a=b=1$) and $7$ successes in $10$, the posterior is Beta$(8,4)$ with mean $0.667$ and a central 95% **credible interval** $[0.390,\,0.891]$, a range that *does* carry the direct meaning "given the model and prior, the parameter lies here with probability 95%". A skeptical Beta$(2,2)$ prior shrinks it to Beta$(9,5)$ with mean $0.643$. With 100 trials and 70 successes the prior hardly matters: the posterior Beta$(71,31)$ has mean $0.696$ and interval $[0.604,\,0.781]$. The frequentist (Wald) interval for the original $7/10$ is $[0.42,\,0.98]$, wider on the upper side, and for $3$ successes out of $3$ the MLE is exactly $1$, while the Beta$(4,1)$ posterior says $0.80$ with interval $[0.40,\,0.99]$: a sensible hedge against extreme conclusions from tiny samples.

The Bayesian view is not primarily a different calculus; it is a way to **state assumptions** (the prior) openly and to **propagate uncertainty** through a pipeline. It underlies shrinkage estimators for thousands of genes at once (empirical Bayes, Chapter 4), regularization as a prior (Chapter 7), and variational inference (Chapter 8).

```python
--8<-- "code/m08_statistics.py"
```

Output (seed 0):

```text
variance estimators, n = 5, true variance 4.0: mean of MLE (divide by n) = 3.207 (theory 3.20);  mean of unbiased (divide by n-1) = 4.009

coverage of 95% confidence intervals for a mean (true mean 10, sd 3), 20000 repeated experiments:
  n =   5: z-interval covers 0.881,  t-interval covers 0.952
  n =  10: z-interval covers 0.920,  t-interval covers 0.951
  n =  30: z-interval covers 0.937,  t-interval covers 0.948
  n = 100: z-interval covers 0.949,  t-interval covers 0.952

10000 genes, no true differences, 10 vs 10 samples: p-value histogram (10 bins) = [1025, 1022, 1019, 964, 1046, 979, 970, 995, 944, 1036]  (flat = uniform); p < 0.05 for 500 genes (expected 500)

multiple testing, 10000 genes, 500 truly differential:
  p < 0.05 (uncorrected)         discoveries   964   true  452   false  512   false discovery proportion 0.531   power 0.904
  Bonferroni (p < 0.05/10000)    discoveries     9   true    9   false    0   false discovery proportion 0.000   power 0.018
  Benjamini-Hochberg, FDR 5%     discoveries   175   true  165   false   10   false discovery proportion 0.057   power 0.330

bootstrap 95% interval for the median of 10 skewed values: [2.85, 7.25]  (sample median 4.00)
permutation test, difference of means 1.42: p = 0.0213;  t-test p = 0.0105

winner's curse: top 1% of a noisy screen: mean first measurement 3.80, mean re-measurement 1.88, mean true effect 1.93
  the best linear prediction of the true effect from one measurement shrinks it by the factor 0.50 (= signal variance / total variance = 1/2)
  treated    success rates by batch: 0.93, 0.73   overall 0.78
  untreated  success rates by batch: 0.87, 0.69   overall 0.83
  -> treated does better in each batch, but worse overall, because most treated cases are in the hard batch (confounding)

power of a two-sample t-test, true effect 0.5 standard deviations, alpha = 0.05 (20000 simulated experiments each):
  n =  10 per group: power 0.181
  n =  20 per group: power 0.335
  n =  64 per group: power 0.805
  n = 128 per group: power 0.979
  rule of thumb: n per group = 16 / d^2 for 80% power at alpha = 0.05 -> d = 0.5 gives 64
```

---

## M8.9 Worked examples

!!! example "Worked example M8.1: How many replicates do I need?"
    **Situation.** You plan to compare expression of a marker gene between two conditions. Pilot data suggest a within-group standard deviation of 0.4 (in log$_2$ units) and you care about detecting a difference of 0.4 or larger (a 32% change).

    **Reasoning.** The standardized effect is $d=0.4/0.4=1$. The rule of thumb $n\approx16/d^2$ gives $16$ per group for 80% power at $\alpha=0.05$ (an exact noncentral-$t$ calculation gives $17$). If instead the difference of interest is $0.2$ (a 15% change), $d=0.5$ and $n=64$: *halving the effect quadruples the sample*. If you will test 1,000 genes and control the FDR at 5%, the per-gene threshold is far stricter than $0.05$: between about $5\times10^{-5}$ (a Bonferroni-sized threshold, when almost nothing is called) and $5\times10^{-3}$ (when 100 genes are called). At $\alpha=5\times10^{-4}$ and $d=1$ the exact calculation gives $41$ per group, more than double the unadjusted $17$.

**Sources of failure.** The pilot standard deviation is itself an estimate from few samples (and is a biased guess if the pilot samples were chosen for being clean). If the true standard deviation is 0.6 (rather than 0.4), then $d=0.67$ and the required $n$ (at $\alpha=0.05$) rises from 17 to 37. Budget for the optimistic-pilot error.

    **Lesson.** Sample-size calculations are as strong as their assumptions about effect and noise. Use conservative inputs, and state what you assumed.

!!! example "Worked example M8.2: Reading a volcano plot"
    **Situation.** A differential-expression figure plots each gene's log$_2$ fold change (x) against $-\log_{10}$ p-value (y). Many genes with $|\log_2\mathrm{FC}|<0.3$ have $p<10^{-6}$, and a handful with $|\log_2\mathrm{FC}|>2$ have $p\approx0.01$.

    **Reasoning.** The first group shows *precisely estimated small effects*: with thousands of cells the standard error is tiny, so even a 20% change is "significant". The second group shows *large but imprecisely estimated effects* (few cells in the group, or high variance). Neither the p-value nor the fold change alone says what matters: the p-value mixes effect size and precision. A defensible reading reports the **effect size with an interval**, applies a multiple-testing correction across the whole plot, and asks whether tiny effects are *biologically meaningful* (a 20% change in a housekeeping gene may be irrelevant) and whether the large noisy ones *replicate*.

    **Lesson.** Statistical significance is a statement about evidence against a null; biological importance is a statement about effect size in context. They are different axes and must be reported on both.

---

## M8.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"We tested three normalizations and two clusterings, and one gave $p=0.03$\""
    **The observation.** A differential-expression result is non-significant under the default pipeline. After trying three normalizations, two clustering resolutions and two outlier-removal rules, one combination gives $p=0.03$ for the gene of interest. It is reported as the main analysis.

    **Tempting conclusion.** "The effect is real; we found the right way to analyse it."

    **Decompose.** Twelve analyses ($3\times2\times2$) were available, and the p-values among them are correlated but not identical. If the null is true for this gene, the probability that *at least one* of 12 independent looks has $p<0.05$ is $1-0.95^{12}=0.46$; with positive correlation it is lower, but still several times $0.05$. The reported $p=0.03$ is therefore not what it claims. This is a **forking-paths** problem and it is the same arithmetic as the 10,000-gene screen: many chances, one reported success. Biology compounds it because analysts *understandably* stop once an interpretable result appears.

    **Hidden assumptions.** (i) The choice among analyses was made without looking at the outcome (it was not). (ii) The 12 analyses represent all the reasonable ones (they do not: the unreported choices of filtering thresholds, cell inclusion criteria, and reference genome version are also forks).

    **Discriminating experiments.** Pre-specify the analysis before seeing the result; or correct over the whole family of analyses (a permutation test of the *maximum* statistic across pipelines; Chapter 54); or hold out a validation dataset analysed once with the chosen pipeline.

    **What this teaches.** The p-value belongs to the *whole procedure*, including the decisions made along the way, not just to the final calculation. An analysis that depends on the data to choose its path needs a correction, or an untouched test set.

---

## M8.11 Connections

- **Forward:** Chapter 4 is the research version of this material (estimation theory, Fisher information, hierarchical and empirical-Bayes models, multiple testing in genomics). Chapter 7 uses bias–variance to explain regularization and double descent. Chapter 26 applies regression and multiple testing to GWAS and builds mixed models. Chapter 43 covers evaluation metrics, confidence intervals on AUROC and the statistics of benchmark comparison; Chapter 44 resolves Simpson's paradox with causal graphs; Chapter 45 treats confounding and shift; Chapter 54 studies forking paths in AI-driven research.
- **Backward:** probability, distributions, CLT, Monte Carlo (M7); derivatives for the MLE (M3); least squares (M4, M6).

!!! takeaways "Key takeaways"
    1. An **estimator** is a random variable with a **sampling distribution**; judge it by bias, variance (standard error $\sigma/\sqrt n$) and MSE. Dividing by $n-1$ removes the bias of the variance estimator.
    2. **Maximum likelihood** maximizes $P(\text{data}\mid\theta)$; its curvature gives uncertainty. Squared error, cross-entropy and Poisson loss are negative log-likelihoods: choosing a loss is choosing a noise model.
    3. A **95% confidence interval** is a procedure that covers the truth 95% of the time over repeated experiments, not a probability for this interval. Nominal intervals can under-cover for small $n$ (88% at $n=5$ with the $z$ value). The **bootstrap** gives intervals for any statistic.
    4. A **p-value** is $P(\text{data this extreme}\mid H_0)$, uniform under the null; it is not $P(H_0)$ and not an effect size. **Power** needs planning: $n\approx16/d^2$ per group for 80% power.
    5. Testing 10,000 null genes at 0.05 yields about 500 false positives. **Bonferroni** controls any false positive (very conservative); **Benjamini–Hochberg** controls the false discovery rate (here 5.7% realized at a 5% target, power 0.33).
    6. Selected extremes are biased upward: the **winner's curse** halved the top hits' apparent effects on re-measurement ($3.80\to1.88$ vs true $1.93$).
    7. **Simpson's paradox**: a confounder can reverse a trend; whether to adjust depends on the causal structure, not on statistics alone.
    8. **Bayesian updating** with a conjugate prior (Beta–binomial) gives a posterior and a credible interval, and keeps extreme conclusions from tiny samples in check. The p-value belongs to the whole analysis path, including unreported forks.

---

## Further reading

- Wasserman, L. *All of Statistics*. Springer. A compact, rigorous tour of the whole field.
- Efron, B. & Hastie, T. *Computer Age Statistical Inference*. Cambridge University Press (free online). Bootstrap, FDR and the modern view.
- Gelman, A. & Hill, J. *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. Regression and uncertainty with a practical focus.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. Royal Statistical Society B* 57, 289–300.
- Wasserstein, R. L. & Lazar, N. A. (2016). The ASA's statement on p-values: context, process, and purpose. *The American Statistician* 70, 129–133.
- Charig, C. R. et al. (1986). Comparison of treatments of renal calculi by open surgery, percutaneous nephrolithotomy, and extracorporeal shockwave lithotripsy. *BMJ* 292, 879–882. The source of the classic Simpson's paradox data.
- Gelman, A. & Loken, E. (2014). The statistical crisis in science. *American Scientist* 102, 460–465. The garden of forking paths.
