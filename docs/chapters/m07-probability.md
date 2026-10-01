# Chapter M7. Probability from the Ground Up

!!! abstract "Chapter at a glance"
    **Motivation.** Biological measurements are noisy, biological processes are stochastic, and biological models are uncertain about almost everything. Probability is the mathematics of reasoning under uncertainty, and it is the language of every method in the book: a classifier outputs a probability; a generative model *is* a probability distribution; a likelihood says how probable the data are under a model; a p-value says how surprising they are under a null. This chapter builds probability from its definitions, with the distributions you will actually meet in biology and the two theorems (the law of large numbers and the central limit theorem) that justify averaging.

    **Prerequisites.** Chapters M1 (counting, sums, functions), M2 (exponentials, logs) and M3 (integrals).

    **You will be able to:** (1) set up a sample space and compute probabilities of events; (2) compute conditional probabilities and apply Bayes' rule, including the base-rate effect; (3) distinguish independence from zero correlation; (4) compute and interpret expectation, variance and covariance; (5) recognize and use the Bernoulli, binomial, categorical, Poisson, exponential, Gaussian, beta, gamma and negative-binomial distributions in a biological context; (6) explain the law of large numbers and the central limit theorem and state their conditions; (7) estimate integrals and expectations by Monte Carlo with a known error rate.

---

## M7.1 Probability as a calculus of uncertainty

A **random experiment** has an outcome that is not determined in advance: the base at a position of a randomly chosen read, the number of molecules of a transcript in a cell, whether a patient responds. The **sample space** $\Omega$ is the set of all possible outcomes, and an **event** is a subset $E\subseteq\Omega$ (a statement about the outcome, such as "the read has a G at this position"). A **probability** assigns a number $P(E)$ to every event so that three rules (the **axioms**) hold:

1. $P(E)\ge0$.
2. $P(\Omega)=1$ (something happens).
3. If events $E_1,E_2,\dots$ are *disjoint* (cannot occur together) then $P(E_1\cup E_2\cup\cdots)=\sum_iP(E_i)$.

Everything else follows, including $P(\text{not }E)=1-P(E)$ and the **inclusion–exclusion** rule $P(E\cup F)=P(E)+P(F)-P(E\cap F)$ (Chapter M1's rule for counting, with probabilities in place of sizes). When the $N$ outcomes are equally likely, $P(E)=|E|/N$: probability is counting, and the counting techniques of Chapter M1 are what you use.

**Two readings of the word "probability".** In the **frequentist** view, $P(E)$ is the long-run fraction of repeated experiments in which $E$ occurs. In the **Bayesian** view, $P(E)$ is a degree of belief that is updated by evidence. The mathematics is identical; what differs is which questions are considered meaningful ("what is the probability that this gene is a driver?" is natural for a Bayesian and awkward for a strict frequentist). Both views appear in this book, and you should be able to translate between them.

---

## M7.2 Conditional probability, independence, and Bayes' rule

The **conditional probability** of $E$ given $F$ is the probability of $E$ in the world where $F$ is known to have happened:

$$
P(E\mid F)=\frac{P(E\cap F)}{P(F)}\qquad(P(F)>0).
$$

Rearranged, it is the **chain rule**: $P(E\cap F)=P(E\mid F)P(F)$. Two events are **independent** if knowing one tells you nothing about the other: $P(E\mid F)=P(E)$, equivalently $P(E\cap F)=P(E)P(F)$.

**Law of total probability.** If $F_1,\dots,F_k$ are disjoint events covering $\Omega$ (a *partition*), then

$$
P(E)=\sum_{j}P(E\mid F_j)\,P(F_j).
$$

You compute the probability of something by splitting the world into cases.

!!! math "Bayes' rule"
    Combine the chain rule in both orders, $P(E\mid F)P(F)=P(F\mid E)P(E)$, to get

    $$
    P(F\mid E)=\frac{P(E\mid F)\,P(F)}{P(E)}=\frac{P(E\mid F)\,P(F)}{\sum_jP(E\mid F_j)P(F_j)}.
    $$

    In inference language: $F$ is a hypothesis and $E$ is evidence. $P(F)$ is the **prior**, $P(E\mid F)$ the **likelihood**, $P(F\mid E)$ the **posterior**. *Evidence reweights prior beliefs by how well each hypothesis predicts the evidence.* Chapter M1 warned that $P\Rightarrow Q$ and its converse differ; Bayes' rule is the quantitative bridge between "if disease then positive" and "if positive then disease".

!!! example "Worked example M7.1: The base-rate effect in a screening test"
    **Situation.** A test for a genetic condition has sensitivity $P(+\mid D)=95\%$ and specificity $P(-\mid\text{not }D)=95\%$. A person tests positive. What is $P(D\mid+)$?

    **Reasoning.** By Bayes' rule and the law of total probability,

    $$
    P(D\mid+)=\frac{0.95\,\pi}{0.95\,\pi+0.05\,(1-\pi)},
    $$

    where $\pi$ is the prevalence. The answer depends on $\pi$, which most people forget: for $\pi=50\%$ it is $0.950$; for $\pi=10\%$, $0.679$; for $\pi=1\%$, $0.161$; for $\pi=0.1\%$, $0.019$ (script). At 1% prevalence, in a million people there are 10,000 true cases and 990,000 healthy people; $95\%$ of the cases give $9{,}500$ true positives, and $5\%$ of the healthy give $49{,}500$ false positives. Of $59{,}000$ positives, only $16\%$ are real. The simulation reproduces it: $58{,}773$ positives, $9{,}477$ true, fraction $0.161$.

    **Lesson.** A test that is "95% accurate" can be right only 16% of the time when it is positive, because errors in the large healthy group outnumber correct calls in the small diseased group. The base rate is part of the answer. This is the same arithmetic as variant classification (a rare causal variant among many benign ones), as a genome-wide screen with few true hits, and as a model whose top predictions are "enriched" but mostly wrong (Chapters 26 and 43).

---

## M7.3 Random variables, expectation, variance, covariance

A **random variable** $X$ is a number determined by the outcome of a random experiment: the read count of a gene, the expression value of a cell, the fitness of a mutant. It is **discrete** if it takes countable values, described by a **probability mass function** $p(x)=P(X=x)$; and **continuous** if described by a **density** $f(x)$ such that $P(a\le X\le b)=\int_a^bf(x)\,dx$ (the integral of Chapter M3). Both satisfy "total probability 1": $\sum_xp(x)=1$ or $\int f=1$. The **cumulative distribution function** (CDF) is $F(x)=P(X\le x)$.

**Expectation** (the mean) is the probability-weighted average of the values:

$$
\mathbb{E}[X]=\sum_xx\,p(x)\quad\text{or}\quad\int x f(x)\,dx,\qquad
\mathbb{E}[g(X)]=\sum_xg(x)p(x)\ \text{or}\ \int g(x)f(x)\,dx .
$$

Two crucial properties. **Linearity:** $\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$, *always*, with no independence assumption. **Not multiplicative for functions:** in general $\mathbb{E}[g(X)]\ne g(\mathbb{E}[X])$ (Jensen's inequality, Chapter M3, for convex $g$ the left side is larger).

**Variance** measures spread: $\operatorname{Var}(X)=\mathbb{E}\big[(X-\mathbb{E}X)^2\big]=\mathbb{E}[X^2]-(\mathbb{E}X)^2$, and the **standard deviation** is $\sigma=\sqrt{\operatorname{Var}X}$, in the same units as $X$. Basic rules: $\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$, and for two variables

$$
\operatorname{Var}(X+Y)=\operatorname{Var}X+\operatorname{Var}Y+2\operatorname{Cov}(X,Y).
$$

The **covariance** is $\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$ and the **correlation** is the unit-free version $\rho=\operatorname{Cov}(X,Y)/(\sigma_X\sigma_Y)\in[-1,1]$. For independent variables the covariance is 0 and variances add. In the script, with $Y=0.6X+0.8\varepsilon$ the variance of the sum is $3.177$, equal to $\operatorname{Var}X+\operatorname{Var}Y+2\operatorname{Cov}$, whereas a (wrong) independence assumption would give $1.989$: ignoring positive covariance makes a sum look more precise than it is. (This is exactly why averaging many *correlated* replicates, such as cells from one donor, helps far less than averaging independent ones.)

!!! warning "Uncorrelated does not mean independent"
    Correlation measures only *linear* association. Let $X\sim\mathcal{N}(0,1)$ and $Y=X^2$: $Y$ is completely determined by $X$, yet their correlation is $-0.004$ in the script (the true value is 0). The mean of $Y$ is $0.997$ but the mean of $Y$ among cases with $|X|>2$ is $5.76$: knowing $X$ changes what to expect from $Y$, so they are dependent. A correlation of zero between two measurements says nothing about whether one *regulates* the other if the relationship is non-monotonic (an activator at low dose, a repressor at high dose).

---

## M7.4 The distributions you will meet

A **distribution** is a specific recipe for the probabilities of a random variable. Learn the following by their *generating story*, not by their formulas: the story tells you when to use each.

| Distribution | Story | Parameters | Mean | Variance | Where it appears |
|---|---|---|---|---|---|
| **Bernoulli**($p$) | one yes/no trial | $p$ | $p$ | $p(1-p)$ | is a variant present? is a read a mutation? |
| **Binomial**($n,p$) | number of successes in $n$ independent trials | $n,p$ | $np$ | $np(1-p)$ | reads supporting a variant; mutated sites in a gene |
| **Categorical / multinomial** | one (or $n$) draws from $K$ categories | $p_1..p_K$ | | | a base, an amino acid, a token; the softmax output (Chapter 12) |
| **Poisson**($\lambda$) | count of rare independent events in a fixed window | $\lambda$ | $\lambda$ | $\lambda$ | reads per window; mutations per genome; mRNA counts |
| **Geometric / exponential**($\lambda$) | waiting time until the first event | rate $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | time to next mutation; mRNA lifetimes |
| **Uniform**($a,b$) | all values in an interval equally likely | $a,b$ | $(a+b)/2$ | $(b-a)^2/12$ | random positions; priors |
| **Gaussian** $\mathcal{N}(\mu,\sigma^2)$ | sum of many small independent effects | $\mu,\sigma$ | $\mu$ | $\sigma^2$ | measurement noise; polygenic traits; latent variables |
| **Beta**($a,b$) | an unknown probability in $(0,1)$ | $a,b$ | $a/(a+b)$ | | allele frequencies; prior for a rate |
| **Gamma**($k,\theta$) | positive quantity; waiting time for $k$ events | $k,\theta$ | $k\theta$ | $k\theta^2$ | variation of rates between cells or genes |
| **Negative binomial** | Poisson whose rate varies (Gamma-mixed Poisson) | $\mu$, dispersion | $\mu$ | $\mu+\mu^2/r$ | single-cell and RNA-seq counts |

**Binomial and its Poisson limit.** If $n$ is large and $p$ is small with $np=\lambda$ fixed, $\text{Binomial}(n,p)\to\text{Poisson}(\lambda)$. For $n=1000$, $p=0.003$ the largest difference between the two probability mass functions over $k=0,\dots,7$ is $3.4\times10^{-4}$ (script). That is why rare-event counts, such as the number of mutations in a gene or reads in a window, are modelled as Poisson. The Poisson has the defining property **mean = variance**, the **Fano factor** (variance/mean) equals 1.

!!! bio "Biology for modeling: counts are over-dispersed, and why it matters"
    **What is it?** In sequencing data the count for a gene in a cell is the outcome of random molecule capture. If every cell had exactly the same underlying rate $\lambda$, the counts would be Poisson. But cells differ in the *true* rate (cell size, state, transcriptional bursting), so the effective rate $\Lambda$ varies from cell to cell.

    **The model.** Draw $\Lambda\sim\text{Gamma}$ with mean $\mu$ and shape $r$, then the count $\sim\text{Poisson}(\Lambda)$. Marginally this is the **negative binomial**, with $\mathbb{E}=\mu$ and $\operatorname{Var}=\mu+\mu^2/r$. The script uses $\mu=5$, $r=2$: the simulated mean is $5.01$ and variance $17.45$ (Fano factor $3.48$; theory $1+\mu/r=3.50$), far above the Poisson value of 1.

    **Modelling consequence.** Methods that assume Poisson counts understate the variability, so differences between conditions look more significant than they are, producing false positives. Single-cell and bulk RNA-seq methods therefore use negative-binomial models (Chapters 25, 30). Over-dispersion is *information*: the dispersion parameter measures biological variability beyond sampling noise.

    **What would change the interpretation.** Extra zeros beyond the negative binomial prediction (*zero inflation*) have often been attributed to "dropout", but in droplet protocols they are largely explained by the negative binomial itself (Chapter 25); whether a zero-inflation term is needed is an empirical question for each protocol.

**Gaussian.** The **normal** distribution has density $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ and is symmetric around $\mu$. The **68–95–99.7 rule**: about 68%, 95% and 99.7% of values lie within 1, 2 and 3 standard deviations (script: $0.6828$, $0.9542$, $0.9973$). Its prominence comes from the central limit theorem (§M7.6). A random vector $\mathbf{x}\in\R^{d}$ is **multivariate Gaussian**, $\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$, with mean vector $\boldsymbol\mu$ and symmetric positive semi-definite *covariance matrix* $\boldsymbol\Sigma$ (Chapter M6); its marginals and conditionals are again Gaussian. The eigenvectors of $\boldsymbol\Sigma$ are the principal axes of its ellipsoidal contours, which is why PCA is "diagonalize the covariance".

**Exponential and memorylessness.** A waiting time with constant event rate $\lambda$ is exponential, with the unusual **memoryless** property $P(T>s+t\mid T>s)=P(T>t)$: given that the event has not happened by time $s$, the remaining wait has the same distribution as at the start. In the script, with mean 4, $P(T>2)=0.6068$ and $P(T>5\mid T>3)=0.6054$ (theory $0.6065$). A molecule that decays at a constant rate has this property, so its remaining lifetime does not depend on its age; real systems with ageing or multi-step processes do not.

---

## M7.5 Joint, marginal, and conditional distributions

For two random variables the **joint** distribution gives the probability of each pair. The **marginal** distribution of one is obtained by summing (or integrating) the other out; the **conditional** distribution fixes one.

!!! example "Worked example M7.2: Two linked loci"
    **Situation.** At two nearby positions on a chromosome, alleles $A/a$ and $B/b$ have joint frequencies $P(AB)=0.40$, $P(Ab)=0.10$, $P(aB)=0.05$, $P(ab)=0.45$ (script's table).

    **Reasoning.** Marginals: $P(A)=0.40+0.10=0.50$, $P(B)=0.40+0.05=0.45$. Conditionals: $P(B\mid A)=0.40/0.50=0.80$, $P(B\mid a)=0.05/0.50=0.10$. If the loci were independent, $P(AB)$ would be $0.50\times0.45=0.225$; the observed $0.40$ is far higher. The difference $D=P(AB)-P(A)P(B)=0.175$ is the **linkage disequilibrium** coefficient (Chapters 21 and 26): alleles at nearby sites are inherited together more often than chance, so knowing the allele at one tells you about the other.

    **Consequence.** In a GWAS, a variant that is itself harmless can show association with a trait because it is correlated with the truly causal variant (Chapter 26). The conditional-probability structure of the genome is why *association is not causation* in genetics.

The chain rule for distributions, $p(x,y)=p(y\mid x)p(x)$, extends to any number of variables, $p(x_1,\dots,x_n)=\prod_ip(x_i\mid x_1,\dots,x_{i-1})$, which is how an *autoregressive* language model factorizes the probability of a sequence (Chapters 12, 32): each base is predicted given the previous ones.

---

## M7.6 The law of large numbers and the central limit theorem

Let $X_1,\dots,X_n$ be independent draws from a distribution with mean $\mu$ and variance $\sigma^2$, and $\bar X_n=\frac1n\sum X_i$ their average.

!!! math "Two theorems about averages"
    **Law of large numbers (LLN).** $\bar X_n\to\mu$ as $n\to\infty$. Averages converge to the true mean.

    **Central limit theorem (CLT).** The *fluctuation* of the average, scaled by $\sqrt n$, is approximately Gaussian regardless of the shape of the original distribution:

    $$
    \frac{\bar X_n-\mu}{\sigma/\sqrt n}\ \approx\ \mathcal{N}(0,1)\qquad(n\text{ large}).
    $$

    Hence the **standard error** of the mean is $\sigma/\sqrt n$: to halve the error you need four times the data.

    **Conditions.** Independent (or weakly dependent) draws with finite variance. Fat-tailed distributions, strongly correlated samples, and tiny $n$ can violate them.

The script starts from a very skewed distribution, the exponential with mean 1 and skewness $+2$. For $n=1,5,30,200$: the mean of the sample means is $1.008,\,1.000,\,1.001,\,1.000$ (LLN); the standard deviation of the sample mean is $1.004,\,0.448,\,0.184,\,0.071$, matching $1/\sqrt n=1,\,0.447,\,0.183,\,0.071$; the skewness of the sample mean shrinks from $+2.01$ to $+0.89,\,+0.37,\,+0.14$ (the distribution becomes symmetric), and $P(\text{standardized mean}<1.96)$ moves from $0.948$ toward the Gaussian $0.975$ ($0.956,\,0.965,\,0.970$): convergence is real but *slow* for a skewed parent; even at $n=200$ it is 0.5 percentage points short.

!!! bio "Biology for modeling: what 'n' means"
    **What is it?** The CLT and the $1/\sqrt n$ rate apply to $n$ *independent* observations. Ten thousand cells from one donor are not ten thousand independent samples of "a person"; they are correlated through the donor (shared genotype, environment, batch).

    **Modelling consequence.** The effective $n$ for a donor-level claim is closer to the number of donors than the number of cells. Treating cells as independent replicates is the classic **pseudoreplication** error, and it produces extremely small p-values for differences that are not reproducible across donors (Chapters M8, 25, 43). The standard error is $\sigma/\sqrt{n_{\text{eff}}}$ with the *effective* sample size.

    **What would change the interpretation.** If between-donor variability is small compared with within-donor variability, cells carry more independent information and $n_{\text{eff}}$ is larger. The quantity that governs this is the intra-class correlation.

---

## M7.7 Monte Carlo: expectations by sampling

Many expectations $\mathbb{E}[g(X)]=\int g(x)f(x)\,dx$ have no closed form, especially in high dimensions. The LLN gives a recipe: draw $X_1,\dots,X_n\sim f$ and average $g(X_i)$. The CLT says the error is of order $\sigma_g/\sqrt n$ **whatever the dimension**, which is why Monte Carlo is the default tool for high-dimensional integrals. (Grid-based numerical integration, Chapter M3, needs exponentially many points as dimension grows.)

The classic illustration estimates $\pi$: sample random points in the unit square and count the fraction inside the quarter circle; four times the fraction estimates $\pi$. In the script the typical absolute error is $0.143$ at $n=100$, $0.0140$ at $n=10^4$ and $0.0012$ at $n=10^6$, following the predicted $1.31/\sqrt n$ ($0.131$, $0.0131$, $0.0013$): a hundred-fold increase in $n$ buys a ten-fold improvement. To gain one more decimal digit costs a hundred times more samples; that is the price of the method.

Practical points that matter in later chapters:

- **Random seeds.** A pseudorandom generator is a deterministic function of its seed. Fix the seed to make an experiment reproducible, and *vary* it to measure how much the result depends on chance (Chapter M10).
- **Markov chain Monte Carlo (MCMC).** When you cannot sample directly from $f$, construct a Markov chain (Chapter M6) whose stationary distribution is $f$ and run it; samples are correlated, so the effective $n$ is smaller than the number of steps. Used in Bayesian phylogenetics and in sampling protein sequences from a Potts model (Chapters 34, 36, 42).
- **Variance reduction.** Importance sampling and control variates reduce $\sigma_g$ at fixed $n$; the same trick of subtracting a baseline appears in the policy-gradient estimators used to fine-tune generative models with a reward (Chapter 17).

```python
--8<-- "code/m07_probability.py"
```

Output (seed 0):

```text
screening test, sensitivity 95%, specificity 95%: P(disease | positive) by prevalence
  prevalence  50.0%: P(disease | positive) = 0.950
  prevalence  10.0%: P(disease | positive) = 0.679
  prevalence   1.0%: P(disease | positive) = 0.161
  prevalence   0.1%: P(disease | positive) = 0.019
simulation of 1,000,000 people at 1% prevalence: P(disease | positive) = 0.161;  58,773 positives of which 9,477 are true

Y = X^2 with X ~ N(0,1): correlation(X, Y) = -0.0036 (uncorrelated), but E[Y] = 0.997 while E[Y | |X| > 2] = 5.761 (strongly dependent)

Binomial(n=1000, p=0.003) vs Poisson(3), k = 0..7:
  binomial [0.0496 0.1491 0.2242 0.2244 0.1683 0.1009 0.0503 0.0215]
  poisson  [0.0498 0.1494 0.224  0.224  0.168  0.1008 0.0504 0.0216]   max |difference| = 3.4e-04
Poisson counts: mean = variance (Fano factor 1). A gamma-mixed Poisson (negative binomial): mean 5.01, variance 17.45, Fano 3.48  (theory 1 + mu/r = 3.50)
Gaussian: fraction within 1, 2, 3 standard deviations = 0.6828, 0.9542, 0.9973  (theory 0.6827, 0.9545, 0.9973)
exponential waiting times are memoryless: P(T > 2.0) = 0.6068,  P(T > 5.0 | T > 3.0) = 0.6054  (theory 0.6065)

sample mean of n exponential(1) draws: LLN (the mean approaches 1) and CLT (standardized mean approaches N(0,1))
      n   mean of means   std of means   1/sqrt(n)   skewness   P(standardized mean < 1.96)  (normal: 0.975)
      1   1.0075        1.0035         1.0000      +2.007     0.9482
      5   0.9998        0.4476         0.4472      +0.887     0.9559
     30   1.0010        0.1838         0.1826      +0.367     0.9648
    200   1.0001        0.0708         0.0707      +0.140     0.9704

Monte Carlo estimate of pi (fraction of random points in the unit square falling inside the quarter circle, times 4):
  n =       100: typical |error| = 0.1432   (theory: 1.31/sqrt(n) = 0.1310)
  n =    10,000: typical |error| = 0.0140   (theory: 1.31/sqrt(n) = 0.0131)
  n = 1,000,000: typical |error| = 0.0012   (theory: 1.31/sqrt(n) = 0.0013)

joint table of two linked loci (rows A/a, columns B/b):
[[0.4  0.1 ]
 [0.05 0.45]]
  marginals: P(A) = 0.50, P(B) = 0.45;  P(B | A) = 0.800, P(B | a) = 0.100
  independence would give P(A,B) = 0.225; observed 0.400 -> covariance of the indicators 0.175 (linkage disequilibrium D)

Var(X+Y) = 3.177; Var X + Var Y + 2 Cov = 3.177;  if independent it would be 1.989
```

---

## M7.8 Worked examples

!!! example "Worked example M7.3: How much sequencing coverage is enough? (Lander–Waterman)"
    **Situation.** You sequence a genome of length $G$ with reads of length $\ell$ whose start positions are random. The *average coverage* is $c=N\ell/G$ for $N$ reads.

    **Question.** What fraction of the genome is covered by no read?

    **Reasoning.** Look at one base. Each of the $N$ reads covers it with probability $p\approx\ell/G$ (the read must start in the $\ell$ positions that include it). The number of reads covering the base is Binomial($N,p$), which for large $N$ and small $p$ is Poisson with mean $Np=c$. So

    $$
    P(\text{base uncovered})=P(\text{Poisson}(c)=0)=e^{-c}.
    $$

    At $c=5$: $e^{-5}=0.0067$, so about 0.67% of a $3.1\times10^{9}$-base genome, roughly $2\times10^{7}$ bases, are never read. At $c=10$: $4.5\times10^{-5}$. At $c=30$: $9\times10^{-14}$, effectively zero. The fraction of the genome covered *at least $k$ times* is $1-\sum_{j<k}e^{-c}c^j/j!$; to call heterozygous variants reliably one needs at least $\sim8$–10 reads at the site, which explains why 30x is a standard depth for whole-genome sequencing.

    **What breaks.** Real coverage is *over-dispersed*, not Poisson: GC-rich and repetitive regions are under-sampled because amplification and sequencing introduce length and GC biases (Chapter 25), so some regions have near-zero coverage even at high average depth. Poisson sets the *best case*; the over-dispersion (negative binomial, §M7.4) explains why a measured 30x average still leaves gaps.

!!! example "Worked example M7.4: Are two replicates independent, and does it matter?"
    **Situation.** An experiment measures a gene's expression in 3 donors × 1,000 cells. A t-test on all 3,000 cells comparing treated and untreated finds $p<10^{-30}$.

    **Reasoning.** The test assumes 3,000 independent observations. But cells from one donor share many factors, so they are positively correlated. With intra-class correlation $\rho$ (the fraction of total variance due to donor), the effective sample size for a group mean with $m$ cells in each of $d$ donors is $n_{\text{eff}}=\dfrac{md}{1+(m-1)\rho}$. With $m=1000$, $d=3$ and a modest $\rho=0.1$: $n_{\text{eff}}=3000/(1+99.9)=29.7$, roughly the *number of donors times $\sim10$*, a hundred times fewer than 3,000. For large $\rho$ it approaches the number of donors $d=3$. The standard error is therefore about $\sqrt{100}=10$ times larger than the naive one, and the p-value is correspondingly many orders of magnitude weaker.

    **Lesson.** The CLT's $\sqrt n$ is about the number of independent units. Count the units at the level at which the hypothesis is stated: a hypothesis about people is tested on people.

---

## M7.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The classifier is 99% accurate\""
    **The observation.** A model predicts which genomic windows are enhancers. On held-out data, accuracy is 99%.

    **Tempting conclusion.** "The model has learned enhancer grammar."

    **Decompose.** Suppose 1% of windows are enhancers. A model that predicts "not an enhancer" for *every* window achieves 99% accuracy and has learned nothing. With the numbers of Worked example M7.1 in mind, accuracy blends two very different quantities and is dominated by the majority class. The informative quantities are conditional probabilities: **sensitivity (recall)** $P(\text{predicted}+\mid\text{enhancer})$, **specificity** $P(\text{predicted}-\mid\text{not enhancer})$, and **precision** $P(\text{enhancer}\mid\text{predicted}+)$, which is Bayes' rule and depends on the base rate. If the test set was balanced for convenience but deployment has 1% prevalence, precision in the field will be far lower than in the paper.

    **Hidden assumptions.** (i) The test windows' class mix matches the use case. (ii) Windows are independent (neighbouring windows overlap, so the effective $n$ is smaller than the number of windows; §M7.6). (iii) Labels are accurate: "not an enhancer" often means "not yet tested".

    **Discriminating experiments.** Report precision–recall (not only ROC); compute the metric at the deployment base rate; compare with a trivial baseline (always-negative, GC-content, distance to a gene); test on a different chromosome and cell type.

    **What this teaches.** A probability statement is always conditional on something. Ask "conditional on what?" every time you read a performance number (Chapter 43 develops this fully).

---

## M7.10 Connections

- **Forward:** Chapter M8 turns probability into statistical inference; Chapter 4 is the research version (estimators, priors, hierarchical models, multiple testing); Chapter 5 defines entropy as the expected surprise of a distribution; Chapter 8 treats latent-variable models and variational inference; Chapters 14–15 treat generative models, which are learned distributions; Chapter 21 builds population genetics from binomial sampling (genetic drift); Chapters 25 and 30 use the negative binomial for counts; Chapter 43 uses conditional probabilities for evaluation metrics.
- **Backward:** counting, sets, the birthday problem (M1); logarithms and exponentials (M2); integrals (M3); eigenvalues of transition matrices and covariance matrices (M6).

!!! takeaways "Key takeaways"
    1. Probability assigns numbers to events subject to three axioms; with equally likely outcomes it is counting. **Conditional probability** is $P(E\mid F)=P(E\cap F)/P(F)$; **independence** means $P(E\cap F)=P(E)P(F)$.
    2. **Bayes' rule** converts $P(\text{evidence}\mid\text{hypothesis})$ into $P(\text{hypothesis}\mid\text{evidence})$ using the prior. A 95%-sensitive, 95%-specific test is right only 16% of the time when positive at 1% prevalence.
    3. **Expectation is linear** (always); variance of a sum includes the covariance term; **uncorrelated does not mean independent** ($Y=X^2$).
    4. Know the distributions by their stories: binomial (successes), **Poisson** (rare counts, mean = variance), exponential (memoryless waiting), Gaussian (sums of small effects), beta (an unknown probability), negative binomial (Gamma-mixed Poisson: over-dispersed counts).
    5. The **LLN** says averages converge to the mean; the **CLT** says their error is Gaussian with standard error $\sigma/\sqrt n$. Convergence is slow for skewed distributions, and $n$ means *independent* units: cells within a donor are not.
    6. **Monte Carlo** estimates expectations with error $\sim1/\sqrt n$ regardless of dimension; the typical error for $\pi$ is $0.0012$ at $n=10^6$.
    7. Joint, marginal and conditional distributions are tied by $p(x,y)=p(y\mid x)p(x)$; autoregressive models are this chain rule applied repeatedly. Linkage disequilibrium is a statement about the joint distribution of alleles.
    8. Always ask "conditional on what?" when reading a probability, an accuracy, or a p-value.

---

## Further reading

- Blitzstein, J. & Hwang, J. *Introduction to Probability*. CRC Press (free online with Harvard Stat 110 lectures). The best modern introductory text, with stories for every distribution.
- Grinstead, C. M. & Snell, J. L. *Introduction to Probability* (free online). A gentle, example-driven alternative.
- Jaynes, E. T. *Probability Theory: The Logic of Science*. Cambridge University Press. The Bayesian view as an extension of logic.
- Lander, E. S. & Waterman, M. S. (1988). Genomic mapping by fingerprinting random clones: a mathematical analysis. *Genomics* 2, 231–239. The coverage calculation.
- Hafemeister, C. & Satija, R. (2019). Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression. *Genome Biology* 20, 296.
