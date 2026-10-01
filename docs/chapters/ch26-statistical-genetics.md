# Chapter 26. Statistical Genetics: From Association to Causation

!!! abstract "Chapter at a glance"
    **Motivation.** Human genetic variation is the largest natural experiment available for connecting DNA to phenotype, and it is the bridge between *sequence models* (Part VII) and *disease*. Genome-wide association studies (GWAS), polygenic scores, fine-mapping, and Mendelian randomization are the statistical machinery that turns genotype–phenotype data into claims, and each has a characteristic way of failing. This chapter derives the main estimators and then tests each failure with simulation, so that when a deep-learning paper says "our model prioritizes causal variants" or "our model predicts disease from genotype," you know what the ceiling and the confounders are.
    **Prerequisites.** Chapters 3, 4, 7, 20, 21.
    **You will be able to:** (1) state the additive model, define heritability variants (twin, SNP, liability-scale) and what $h^2$ does and does not mean; (2) derive how LD blurs causal signal into association signal, $z\approx\sqrt n R\beta$, and use it to explain why lead SNPs are not causal variants; (3) derive LD score regression and show why its intercept, not $\lambda_\text{GC}$, diagnoses confounding; (4) show that a linear mixed model is ridge regression; (5) compute Bayes-factor fine-mapping posterior inclusion probabilities and credible sets and know their limits; (6) derive the expected accuracy of a polygenic score and explain why portability across populations fails through tagging; (7) derive Mendelian randomization estimators and the effects of weak instruments and pleiotropy.

---

## 26.1 The seven questions for a genetic association

!!! bio "Biology for modeling: a genetic association"
    **What is it?** A statistical relationship, across individuals, between the genotype at a variant and a phenotype (height, blood pressure, a disease, a molecular trait such as expression).
    **Information it contains.** Meiosis shuffles parental alleles at conception, so, *conditional on ancestry and family structure*, a person's genotype is allocated independently of most environmental exposures: a natural randomized assignment that is the root of the causal claims in this chapter. Association also reflects **linkage disequilibrium** (alleles inherited together), so the associated variant is generally not the causal one.
    **How is it generated?** Mutation, recombination, drift, and selection (Chapter 21) shape which variants exist and how they are correlated; the phenotype arises from many such variants plus environment.
    **How is it measured?** Genotyping arrays (with statistical imputation to millions of variants), exome and whole-genome sequencing; phenotypes from biobanks (electronic health records, surveys, measurements) and cohort studies. Large studies now reach hundreds of thousands to millions of individuals; a meta-analysis of height in about 5.4 million people (Yengo et al., 2022) identified roughly 12,000 independent genome-wide significant variants and appeared to saturate the common-variant contribution to height heritability [[S]].
    **Computational representation.** A genotype matrix $X\in\{0,1,2\}^{n\times M}$ (allele dosages) or summary statistics $(\hat\beta_j,\mathrm{SE}_j,N_j)$ with an LD matrix $R\in\mathbb R^{M\times M}$.
    **What varies.** Ancestry, relatedness, environment, age and sex, and the phenotype definition and ascertainment.
    **What can ML learn?** Predictors of phenotype from genotype (polygenic scores), enrichment of associations in functional annotations, priors for which variants are causal.
    **What can ML not observe?** Causal variants hidden by LD, effects in the people absent from the study, gene–environment interaction, and anything about variants too rare to be seen.

---

## 26.2 The additive model and what heritability means

Take $n$ individuals and standardize the phenotype ($\mathbb E y=0,\ \mathrm{Var}\,y=1$) and each genotype column to mean 0 and variance 1. The **additive model** is
$$
y=X\beta+\varepsilon,\qquad \varepsilon_i\sim\mathcal N(0,\sigma_e^2),
$$
with $\beta\in\mathbb R^M$ the vector of per-standardized-allele effects. The variance explained by the genotypes, the **(SNP) heritability**, is
$$
h^2=\frac{\mathrm{Var}(X\beta)}{\mathrm{Var}(y)}=\beta^\top R\,\beta,\qquad R=\tfrac1n X^\top X\ (\text{the LD matrix}),\qquad \sigma_e^2=1-h^2 .
$$

!!! math "Which heritability?"
    *Broad-sense* $H^2=V_G/V_P$ includes all genetic variance (additive, dominance, epistatic); *narrow-sense* $h^2=V_A/V_P$ only the additive part, which is what parents transmit and what selection acts on; *SNP heritability* $h^2_\text{SNP}$ is the variance captured by a given set of genotyped or imputed variants (a lower bound on $h^2$). Twin studies estimate $h^2$ by **Falconer's formula**: if identical twins share all additive variance and fraternal twins half, and if shared environment and non-additive effects are negligible, then $r_\text{MZ}-r_\text{DZ}=h^2/2$, so $h^2=2(r_\text{MZ}-r_\text{DZ})$. For a binary trait with prevalence $K$, the observed-scale heritability of a 0/1 outcome is converted to a **liability-scale** heritability by $h^2_\ell=h^2_\text{obs}\,K(1-K)/\varphi(\Phi^{-1}(K))^2$ (Dempster–Lerner; $\varphi,\Phi$ the standard normal density and CDF; ascertained case–control samples need a further correction).

Three cautions that a machine-learning reader must internalize. (i) **$h^2$ is a property of a population in an environment**, not of an individual or a trait in the abstract: it changes when environmental variance changes. (ii) **It says nothing about determinism or modifiability**: height is highly heritable and also rose by many centimeters within a century with nutrition. (iii) **It is an upper bound for the predictive $R^2$ of any additive genotype-only predictor**, a noise ceiling in the sense of Chapter 1: no model that uses only additive effects of common variants can explain more than $h^2_\text{SNP}$, whatever its architecture. (Models can exceed the additive ceiling only by capturing non-additive effects or by using the *environmental* information that correlates with genotype, including ancestry.)

**Single-variant testing.** For each variant the GWAS fits the marginal regression $\hat\beta_j=x_j^\top y/n$ with $\mathrm{SE}\approx1/\sqrt n$, so that $z_j=\sqrt n\,\hat\beta_j$ and under the null $z_j\sim\mathcal N(0,1)$, $\chi^2_j=z_j^2\sim\chi^2_1$. About $10^6$ independent common-variant tests in humans motivate the **genome-wide significance threshold** $0.05/10^6=5\times10^{-8}$ (Bonferroni; Risch & Merikangas, 1996; Pe'er et al., 2008). Power for a variant explaining a fraction $v$ of variance is governed by the non-centrality $n\,v$: a variant explaining 0.1% of phenotypic variance has $z\approx\sqrt{0.001n}$, which reaches $z=5.45$ ($p=5\times10^{-8}$) at $n\approx30{,}000$. *This is why modern GWAS need hundreds of thousands of samples: most effects are individually tiny.*

---

## 26.3 Linkage disequilibrium: association is a blurred picture of causation

Suppose the true model is $y=X\beta+\varepsilon$ and we test variant $j$ alone. Then
$$
\hat\beta_j=\tfrac1n x_j^\top y=\sum_k R_{jk}\beta_k+\tfrac1n x_j^\top\varepsilon,
$$
the marginal effect of $j$ is the **$R$-weighted sum of all effects in LD with it**. In vector form, with $z=\sqrt n\hat\beta/\sqrt{1-\hat\beta^2}\approx\sqrt n\hat\beta$ and the residual noise having covariance $R$,
$$
\boxed{\;z\sim\mathcal N\big(\sqrt n\,R\beta,\ R\big)\;}
$$
This is the central equation of statistical genetics: *the observed association map is the causal-effect map convolved with the LD matrix*. Four consequences.

1. **The lead SNP is a tag.** The most significant variant at a locus has the largest $|(R\beta)_j|$, not necessarily $\beta_j\ne0$. If a causal variant has an LD partner with $r=0.97$, the partner will be significant too, and by noise the partner is the lead variant in a substantial fraction of samples (§26.5).
2. **Deconvolution is ill-posed.** Recovering $\beta$ from $z$ means inverting $R$, and when variants are in near-perfect LD, $R$ has near-zero eigenvalues: the same ill-conditioning as in Chapter 2 and Chapter 3. Regularization (a prior on $\beta$: sparsity for fine-mapping, Gaussian for polygenic prediction) is what makes the problem solvable.
3. **LD is population-specific**, because it is a product of demographic history and recombination (Chapter 21). The same causal $\beta$ gives different $z=\sqrt nR\beta$ in different populations; this underlies the portability results of §26.6.
4. **A machine-learning model of association inherits this.** A neural network trained to predict phenotype from genotype learns *whichever* of the correlated variants is most useful; saliency over variants highlights tags as readily as causal sites (Chapter 18).

**Simulating LD.** In `code/ch26_statgen.py` genotypes are generated by thresholding a latent AR(1) Gaussian chain along each haplotype, in blocks of SNPs, with strength $\rho$ between adjacent variants (so the LD correlation is approximately $\rho^{|j-k|}$ in the latent scale; thresholding at allele frequencies attenuates the binary correlations). This is a toy for LD structure, not a population-genetic simulator: it has no recombination hotspots, no population history, and no rare-variant structure.

---

## 26.4 Confounding, LD score regression, and mixed models

### 26.4.1 Population stratification

If ancestry affects both allele frequency and the phenotype (through environment, culture, or genetic variants elsewhere in the genome), then variants whose frequency differs between ancestries become associated with the phenotype *without any causal role*. Let $a_i$ be an ancestry coordinate. Then $\mathrm{Cov}(x_{ij},y_i)=\mathrm{Cov}\big(\mathbb E[x_{ij}\mid a_i],\mathbb E[y_i\mid a_i]\big)$ is nonzero if both conditional means vary with $a$, and the resulting $\chi^2$ inflation grows with $n$: $\mathbb E\chi^2_j\approx1+n\,(\text{freq-slope}_j\times\text{phenotype-slope})^2$. Standard remedies are to include the leading **principal components** of the genotype matrix as covariates (Price et al., 2006) or to use a mixed model (§26.4.3).

### 26.4.2 LD score regression

Inflated test statistics ($\lambda_\text{GC}=\mathrm{median}(\chi^2)/0.455>1$) were once read as evidence of confounding. But polygenicity also inflates them: with thousands of small real effects spread over a correlated genome, *every* variant tags some causal variant. LD score regression (Bulik-Sullivan et al., 2015) separates the two. Suppose each variant has an independent small effect with $\beta_k\sim\mathcal N(0,h^2/M)$. From §26.3, $z_j=\sqrt n\sum_kR_{jk}\beta_k+\text{noise}$, so
$$
\mathbb E[\chi^2_j]=1+n\sum_kR_{jk}^2\frac{h^2}{M}=1+\frac{n\,h^2}{M}\,\ell_j,\qquad \ell_j=\sum_kR_{jk}^2\ \ (\text{the LD score of variant }j).
$$
Adding a confounding term that does not depend on LD gives
$$
\boxed{\ \mathbb E[\chi^2_j]=1+n\,a+\frac{n\,h^2}{M}\ell_j\ }
$$
**Slope $=nh^2/M$ is polygenic signal; intercept $=1+na$ is confounding.** Variants in high-LD regions tag more causal variants and have larger $\chi^2$; stratification inflates all variants equally regardless of LD. Estimated from summary statistics alone, a regression of $\chi^2_j$ on $\ell_j$ separates the two (with block jackknife standard errors in practice; we show point estimates).

**Experiment** (6,000 SNPs in 300 LD blocks of varying strength, $n=8{,}000$, $h^2=0.40$ with 10% of SNPs causal; two ancestry groups with $F_\text{ST}=0.03$ and a 0.3-SD environmental difference):

| Analysis | Mean $\chi^2$ | $\lambda_\text{GC}$ | LDSC intercept | $h^2$ from slope (true 0.40) |
|---|---|---|---|---|
| Homogeneous population | 2.09 | 1.60 | **0.99** | 0.366 |
| Two ancestry groups, uncorrected | 3.68 | 3.40 | **2.30** | 0.468 |
| Two ancestry groups, leading PC as covariate | 2.15 | 1.67 | **1.16** | 0.336 |

Even without any confounding, $\lambda_\text{GC}=1.60$: *genomic inflation alone cannot be read as confounding in a polygenic trait*. The intercept is 0.99 (no confounding) in the homogeneous sample, 2.30 under stratification, and 1.16 after a single PC correction (we did not investigate whether the small residual excess reflects imperfect adjustment by one PC or estimation noise). The slope-derived $h^2$ is biased up by confounding (0.47) and close to truth otherwise.

### 26.4.3 A linear mixed model is ridge regression

Relatedness and fine-scale structure are handled by the **linear mixed model** (LMM): $y=g+\varepsilon$ with a random genetic effect $g\sim\mathcal N(0,\sigma_g^2K)$ and $K=XX^\top/M$ the genetic relatedness matrix, $\varepsilon\sim\mathcal N(0,\sigma_e^2I)$ (the model behind GCTA, GEMMA, BOLT-LMM, REGENIE; Yang et al., 2011). The best linear unbiased predictor of the genetic value is
$$
\hat g=\sigma_g^2K\,(\sigma_g^2K+\sigma_e^2I)^{-1}y .
$$
Since $\sigma_g^2K+\sigma_e^2I=\tfrac{\sigma_g^2}{M}\big(XX^\top+\lambda I\big)$ with $\lambda=\sigma_e^2M/\sigma_g^2$,
$$
\hat g=\tfrac{\sigma_g^2}{M}XX^\top\tfrac{M}{\sigma_g^2}(XX^\top+\lambda I)^{-1}y=X\underbrace{X^\top(XX^\top+\lambda I)^{-1}y}_{\hat\beta_\text{ridge}} .
$$
*The LMM's genetic value is the fitted value of ridge regression with penalty $\lambda=\sigma_e^2M/\sigma_g^2$*, equivalently Gaussian-prior MAP estimation of $\beta\sim\mathcal N(0,\sigma_g^2/M\,I)$ (Chapter 7) and a Gaussian process with a linear kernel (Chapter 7). Numerically (400 individuals, 1,500 SNPs) the maximum absolute difference between the two fitted vectors is $5.9\times10^{-7}$ (float32 precision). The consequence for ML: *the standard genetic-statistics workhorse is a linear model with an $\ell_2$ prior*, and any proposed deep model must be compared with it (a strong baseline that is also well calibrated for relatedness).

---

## 26.5 Fine-mapping: from a locus to a variant

Fine-mapping asks, *given association signal at a locus, which variants are causal?* The simplest model assumes **one** causal variant per locus. For each candidate $j$, let $\hat\beta_j\sim\mathcal N(\beta_j,V)$ with $V=1/n$ and place the prior $\beta_j\sim\mathcal N(0,W)$ under "$j$ is causal". The marginal likelihood under that hypothesis is $\mathcal N(\hat\beta_j;0,V+W)$ against $\mathcal N(\hat\beta_j;0,V)$ under the null, giving the **Wakefield approximate Bayes factor**
$$
\mathrm{BF}_j=\sqrt{\frac{V}{V+W}}\ \exp\!\Big(\frac{z_j^2}{2}\cdot\frac{W}{V+W}\Big).
$$
With prior inclusion probabilities $\pi_j$, the **posterior inclusion probability** (PIP) under the single-causal-variant model is $\mathrm{PIP}_j=\pi_j\mathrm{BF}_j/\sum_k\pi_k\mathrm{BF}_k$, and the **95% credible set** is the smallest set of variants whose PIPs sum to at least 0.95. Because $\mathrm{BF}$ is monotone in $|z|$, *with a uniform prior, the PIP ranking is the $|z|$ ranking*; what the Bayesian computation adds is **calibrated uncertainty**: how much probability mass is shared among LD partners. Multi-causal extensions are SuSiE (the "sum of single effects" model fitted by iterative Bayesian stepwise selection; Wang et al., 2020), FINEMAP (Benner et al., 2016), and others; functionally informed priors $\pi_j$ (PolyFun; Weissbrod et al., 2020) place more mass on variants in relevant annotations. **The prior $\pi_j$ is exactly where sequence-to-function models (Chapters 31 and 32) can enter statistical genetics**: a model's predicted regulatory effect of each variant is an informative prior for causality.

**Experiment** (100 SNPs in an AR(1) LD structure, one causal variant of standardized effect $\beta=0.03$, prior SD 0.05, 400 replicate loci per row; summary statistics sampled from $z\sim\mathcal N(\sqrt nR\beta,R)$):

| $\rho$ (adjacent LD) | $n$ | $z$ at causal | Lead SNP is the causal variant | Median 95% credible-set size | Coverage of the causal variant | Mean PIP of the causal variant |
|---|---|---|---|---|---|---|
| 0.70 | 20,000 | 4.2 | 0.86 | 3 | 0.99 | 0.70 |
| 0.70 | 100,000 | 9.5 | 1.00 | 1 | 1.00 | 1.00 |
| 0.90 | 20,000 | 4.2 | 0.57 | 6 | 0.98 | 0.40 |
| 0.90 | 100,000 | 9.5 | 0.96 | 1 | 1.00 | 0.94 |
| 0.97 | 20,000 | 4.2 | **0.28** | **15** | 0.97 | 0.16 |
| 0.97 | 100,000 | 9.5 | 0.73 | 3 | 0.98 | 0.63 |

*In tight LD the top-ranked SNP is the causal one in only 28% of loci at $z\approx4.2$, and even at $n=100{,}000$ in 73%*; the credible set (median 15 variants) honestly reports the ambiguity, and its coverage is near the nominal 95% because the single-causal-variant model is correct in the simulation. Adding sample size narrows the set only down to the limit set by LD; breaking ties requires *different information*: multi-ancestry data with different LD (trans-ancestry fine-mapping), functional annotation, or experiments. **Calibration is conditional on the model:** with two causal variants in a locus, or with LD estimated from a mismatched reference panel, a single-effect credible set is no longer valid, and published fine-mapping results often rely on reference-panel LD.

---

## 26.6 Polygenic scores and why they port badly

A **polygenic score** (PGS, PRS) is a weighted sum of allele dosages, $\mathrm{PGS}_i=\sum_jw_jx_{ij}$, with weights $w_j$ from a training GWAS. Methods differ in how they set $w$: clumping and thresholding (keep the top variant per LD block above a $p$ threshold; C+T), LD-aware Bayesian shrinkage (LDpred, PRS-CS), or ridge/LMM (BLUP), and, recently, deep-learning and annotation-informed variants.

### 26.6.1 Expected accuracy

Take $M$ independent standardized variants with $\beta_j\sim\mathcal N(0,h^2/M)$ and OLS estimates $\hat\beta_j=\beta_j+e_j$, $\mathrm{Var}(e_j)\approx1/N$ for a training set of size $N$. The BLUP shrinks each estimate by $s=\frac{h^2/M}{h^2/M+1/N}=\frac{Nh^2}{Nh^2+M}$. For $\mathrm{PGS}=\sum_js\hat\beta_jx_j$ one finds $\mathrm{Cov}(\mathrm{PGS},y)=s\,h^2$ and $\mathrm{Var}(\mathrm{PGS})=s^2(h^2+M/N)=s\,h^2$. The squared correlation is therefore
$$
\boxed{\ R^2=h^2\cdot\frac{Nh^2}{Nh^2+M}\ }
$$
(Daetwyler et al., 2008; Dudbridge, 2013), the infinitesimal-model accuracy: it approaches $h^2$ as $N\to\infty$ and is small when $M\gg Nh^2$. For $h^2=0.5$, $M=2{,}000$, and $N=20{,}000$ it equals $0.5\times\frac{10{,}000}{12{,}000}=0.417$. Real traits have *far* fewer effective independent loci than the $\sim10^7$ variants, and effect sizes are heavy-tailed, so effective $M$ is smaller and sparse-prior methods beat the Gaussian one when architecture is sparse; in our simulation (100 causal variants among 2,000, all genotyped) a ridge score achieved $R^2=0.460$ in the discovery population, higher than 0.417 because the architecture is sparser than the infinitesimal model.

### 26.6.2 Portability: it is tagging, not biology

A score trained in one population is often much less accurate in another. We simulate two populations with **identical causal effects**, but with adjacent-variant LD of 0.8 (population 1) and 0.4 (population 2) and different allele frequencies, training on 20,000 individuals of population 1:

| Variants available to the score | Method | $R^2$ in population 1 (test) | $R^2$ in population 2 | Ratio |
|---|---|---|---|---|
| Causal variants **are** genotyped | C+T (33 loci) | 0.351 | 0.324 | 0.92 |
| Causal variants **are** genotyped | Ridge, all variants | 0.460 | 0.428 | 0.93 |
| Causal variants **not** genotyped (tags only) | C+T (33 loci) | 0.114 | 0.020 | **0.17** |
| Causal variants **not** genotyped (tags only) | Ridge, all variants | 0.172 | 0.030 | **0.17** |

*The biology is identical in the two populations in this simulation, yet the score loses 83% of its accuracy.* The cause is **tagging**: when the causal variant is not in the predictor set, the score uses its LD partners, and a partner captures $r^2_{jk}$ of the causal variant's contribution. With $r^2=0.64$ in population 1 and $0.16$ in population 2 for the adjacent variant, the single-tag ratio is $0.16/0.64=0.25$, the right order of magnitude for the observed 0.17 (multiple tags and frequency differences change the detail). When causal variants are genotyped, the loss is only 7–8%, due to the allele-frequency difference.

In real data, the same mechanism acts together with effect-size heterogeneity (gene–environment interaction), differences in allele frequencies and environment, and discovery samples dominated by European ancestry. Early analyses found markedly lower accuracy in non-European ancestries (Martin et al., 2019; Duncan et al., 2019: for example, median effect sizes in African-ancestry samples around 40% of those in matched European-ancestry samples), and Ding et al. (2023) showed that accuracy declines *continuously* with genetic distance from the training population within every labeled group, so that discrete ancestry labels are a poor description of the problem [[S]]. **Methods that estimate causal variants (fine-mapped, functionally informed) should port better, and multi-ancestry training uses the *differences* in LD to localize causal variants**; this is a statement about information, not just fairness. It is also a **G-G generalization gap** (Chapter 1) of a clean, quantifiable type, and it is a warning for any foundation-model-based predictor of phenotype from genotype (Chapter 41).

---

## 26.7 Causation: Mendelian randomization

Association is not causation. Because alleles are allocated at conception independently of later environment (*conditional on ancestry*), a genetic variant associated with an exposure can serve as an **instrumental variable** for asking whether the exposure causes an outcome. Let $G$ be a variant, $X$ an exposure (a biomarker), $Y$ an outcome, and $U$ unmeasured confounders. Three assumptions: **(1) relevance:** $G$ affects $X$; **(2) independence:** $G$ is independent of $U$; **(3) exclusion restriction:** $G$ affects $Y$ only through $X$. With $Y=\theta X+U+\epsilon$,
$$
\mathrm{Cov}(G,Y)=\theta\,\mathrm{Cov}(G,X)\ \Rightarrow\ \theta=\frac{\beta_{Y,j}}{\beta_{X,j}}\quad(\text{the Wald ratio for variant }j).
$$
With many variants, the **inverse-variance-weighted (IVW)** estimator combines the ratios: $\hat\theta_\text{IVW}=\sum_jw_j\hat\beta_{X,j}\hat\beta_{Y,j}\big/\sum_jw_j\hat\beta_{X,j}^2$, $w_j=1/\mathrm{SE}_{Y,j}^2$ (a weighted regression of $\hat\beta_Y$ on $\hat\beta_X$ through the origin). **MR-Egger** adds an intercept (after orienting alleles so $\hat\beta_X>0$): the intercept estimates the average *directional* pleiotropic effect, and the slope remains consistent if the instrument strength is independent of the pleiotropic effect (the InSIDE assumption).

**Experiment** (30 instruments, two non-overlapping samples of 200,000; confounder $U$ affecting both $X$ and $Y$; true causal effect $\theta=0.20$):

| Scenario | OLS $Y\sim X$ | IVW | MR-Egger slope | Egger intercept | Mean instrument $F$ |
|---|---|---|---|---|---|
| Strong instruments, no pleiotropy | 0.514 | **0.199** | 0.201 | −0.0002 | 1,290 |
| Weak instruments, no pleiotropy | 0.589 | 0.127 | 0.221 | −0.0010 | 9.9 |
| Strong instruments, directional pleiotropy (0.005 per SNP) | 0.519 | 0.229 | **0.189** | **0.0056** | 1,298 |

The observational regression is *biased upward by confounding* (0.51 vs 0.20); IVW recovers the effect (0.199). With **weak instruments** ($F\approx10$) the two-sample estimate is biased *toward zero* (0.127), whereas in a one-sample design with overlapping samples weak-instrument bias is toward the confounded observational estimate; **directional pleiotropy** biases IVW (0.229), while MR-Egger recovers the slope (0.189) and its intercept detects the pleiotropy (0.0056 vs the true 0.005). The Egger estimate is also noisier, and in practice all of these failures coexist.

*Limits.* Pleiotropy is common (most variants affect many traits); instruments are lifelong small perturbations, not the effect of a drug given in adulthood; population stratification, assortative mating, and indirect (dynastic) genetic effects violate independence (within-family designs reduce but do not eliminate them); winner's curse in the discovery sample biases instrument effect sizes; and the exposure must be well defined (the expression of a gene in which tissue, at which age?). **Genetic evidence remains the best large-scale predictor of drug-target success we have**: drug mechanisms with genetic support are about 2.6 times more likely to succeed in the clinic than those without (Minikel et al., *Nature* 2024) [[S]], which ties this chapter to the target-hypothesis problem of Chapter 24 (the dominant cause of clinical failure).

---

## 26.8 Missing heritability and rare variants

Twin and family studies give narrow-sense heritabilities of 0.5–0.8 for many traits, while early array-based SNP heritabilities were smaller (the "missing heritability" problem). Whole-genome sequencing narrowed the gap: in 25,465 unrelated Europeans, WGS-based estimates were 0.68 for height and 0.30 for BMI (Wainschtein et al., 2022), with rare variants in low-LD regions contributing substantially [[S]]. A much larger analysis of whole-genome sequences from 347,630 UK Biobank participants (40 million variants, 34 traits; *Nature*, published online November 2025) estimated that WGS captures on average about 88% of pedigree-based narrow-sense heritability, about 20% from rare variants (MAF < 1%) and 68% from common variants [[S]]. For ML, rare variants are where **variant-effect predictors** (Chapters 31–34, 41) are most needed, because rare variants cannot be assessed by association statistics at feasible sample sizes; they must be assessed by *prior knowledge* of constraint and function, aggregated in **burden tests** across the variants of a gene. This is the regime in which sequence models are most likely to supply information that association statistics cannot.

---

## 26.9 The experiments, verbatim

```python
--8<-- "code/ch26_statgen.py"
```

```text
== 1. Stratification inflates GWAS statistics; LD score regression separates it from polygenicity ==
homogeneous population (polygenic, h2 = 0.40): mean chi2  2.09, lambda_GC 1.60, LDSC intercept  0.99, h2 from slope 0.366
two ancestry groups, uncorrected          : mean chi2  3.68, lambda_GC 3.40, LDSC intercept  2.30, h2 from slope 0.468
two ancestry groups, leading PC as covariate: mean chi2  2.15, lambda_GC 1.67, LDSC intercept  1.16, h2 from slope 0.336
(true h2 = 0.4; LDSC intercept near 1 means no confounding; polygenicity inflates lambda_GC through the slope, confounding through the intercept)

== 2. A linear mixed model is ridge regression ==
max |X beta_ridge - g_BLUP| = 5.89e-07  (same estimator; n = 400, SNPs = 1500)

== 3. Fine-mapping one causal variant among 100 SNPs in LD (single-effect model, Wakefield approximate Bayes factors) ==
rho(adjacent)  N        effect (z at causal)   lead SNP = causal   median 95% credible set   coverage   mean PIP(causal)   median rank by |z|
  0.70          20000     4.2                 0.86                     3                   0.99       0.70                 1
  0.70         100000     9.5                 1.00                     1                   1.00       1.00                 1
  0.90          20000     4.2                 0.57                     6                   0.98       0.40                 1
  0.90         100000     9.5                 0.96                     1                   1.00       0.94                 1
  0.97          20000     4.2                 0.28                    15                   0.97       0.16                 3
  0.97         100000     9.5                 0.73                     3                   0.98       0.63                 1

== 4. Polygenic score portability: same causal effects, different LD ==
GWAS sample size 20000; M = 2000 SNPs (100 causal), h2 = 0.5; population 1 has adjacent-SNP LD 0.8, population 2 has 0.4 (r^2 0.64 vs 0.16)
predictors available                  method                     R2 in population 1 (test)   R2 in population 2   ratio
causal SNPs genotyped (oracle)        clump & threshold (33 loci)    0.351                      0.324             0.92
causal SNPs genotyped (oracle)        ridge on all available SNPs    0.460                      0.428             0.93
causal SNPs NOT genotyped (tags only) clump & threshold (33 loci)    0.114                      0.020             0.17
causal SNPs NOT genotyped (tags only) ridge on all available SNPs    0.172                      0.030             0.17

== 5. Mendelian randomization: confounded OLS vs instrumental variables (true causal effect theta = 0.20) ==
scenario                                      OLS     IVW     MR-Egger slope   Egger intercept   mean instrument F
strong instruments, no pleiotropy            0.514   0.199   0.201           -0.0002          1289.7
weak instruments, no pleiotropy              0.589   0.127   0.221           -0.0010             9.9
strong, directional pleiotropy (0.005 per SNP)   0.519   0.229   0.189            0.0056          1297.9
```

---

## 26.10 Worked research examples

!!! example "Worked Research Example 26.1: The sequence model says the lead variant disrupts a motif"
    **Situation.** A GWAS locus for a cardiac trait has a lead SNP, rs-X, with $z=5.6$ ($p\approx2\times10^{-8}$). A sequence-to-function model predicts that rs-X reduces the binding of a transcription factor expressed in heart and lowers expression of a nearby gene. The authors conclude that rs-X is the causal variant and the mechanism.

    **Question.** What is the probability that this reasoning identified the causal variant, and what experiments would decide?

    **Reasoning.**

    1. *Which variants are candidates?* All variants in LD with rs-X. From §26.5: at $z\approx4$–6 and strong LD, the lead SNP is the causal variant in perhaps a third to a half of loci, and the 95% credible set has several to tens of members. The model's prediction for rs-X is *one* of many predictions for the candidates.
    2. *What is the evidence about the others?* Run the model on every credible-set variant. If rs-X is the only candidate with a large predicted effect in a relevant cell type, the model has *discriminating* information; if three candidates have comparable predictions, the model has not resolved the locus. A model with a false-positive rate of 10% per variant would flag one or two of 15 candidates by chance.
    3. *Use it as a prior, not a verdict.* Combine in a Bayesian way: $\pi_j\propto\exp(\text{predicted effect score}_j)$ and recompute PIPs (§26.5). If the model's score is well calibrated for causal regulatory variants (benchmarked on held-out fine-mapped or MPRA-validated variants: Chapter 43), the posterior is meaningfully sharper; if not, the prior should be flat. *Check the calibration on independent causal variants, not on the GWAS hits the model was tuned with* (circularity).
    4. *Does the molecular mechanism connect to the trait?* Colocalize with an expression QTL of the candidate gene in the right tissue (the eQTL and the GWAS signal should share the same causal variant), and test directionality.
    5. *Decisive experiments.* A massively parallel reporter assay (MPRA) on all credible-set variants (the fastest discriminating test), or CRISPR base editing or prime editing of the endogenous locus in a relevant cell type with a readout of expression, then a phenotype-relevant cellular assay (Chapter 25's perturbation measurement).

    **Expert analysis.** The error is to treat a model's high score at the lead SNP as confirmation when the lead SNP is only one of a credible set, and to *select* the explanation after seeing both the association and the model's output. A more informative design is *prospective*: pre-register the model's ranking of credible-set variants, then test all. The reasoning is an application of the Four Gaps: measurement (the GWAS signal is LD-blurred), inference (association is not causation), objective (the model predicts molecular effects, not disease).

!!! example "Worked Research Example 26.2: A polygenic score trained on a biobank is deployed in a new population"
    **Situation.** A health system plans to use a PGS for coronary artery disease built from a large European-ancestry biobank in a patient population with mixed and non-European ancestry.

    **Question.** Predict how performance will differ, and design the validation.

    **Reasoning.**

    1. *The expected loss.* In our simulation, a score trained on tag variants lost 83% of its $R^2$ in a population with weaker LD around the causal variants, even with identical causal effects, and only about 7% when causal variants were directly typed (§26.6.2). Real losses include effect heterogeneity and environment on top.
    2. *The unit of analysis.* Accuracy varies continuously with genetic distance (Ding et al., 2023), so measure it as a function of each individual's genetic similarity to the training population, not by self-reported group.
    3. *Controls.* Compare against baselines that do not use the genotype at all (age, sex, family history, clinical risk scores), and test *incremental* value ($\Delta R^2$ or $\Delta$AUROC over clinical covariates), calibration in each stratum (a score can rank well and be miscalibrated by a constant shift between groups), and net clinical benefit at the decision threshold.
    4. *Improving portability.* (a) Fine-map and use causal-variant-weighted scores; (b) multi-ancestry discovery and joint modeling using differing LD; (c) functional priors from sequence models; (d) local-ancestry-aware weights in admixed individuals; (e) re-estimate weights in the target population if possible.
    5. *Hazards.* The training and evaluation sets must be separated by *relatedness* (kinship) and *ancestry confounding*: a score can partly predict through ancestry-correlated environment, which inflates apparent accuracy in structured test sets.

    **Expert analysis.** Portability is a *measurement* problem before it is a modeling problem: without a validation cohort of matching ancestry composition, it is not possible to say how the score will behave. The ethical dimension is concrete: a score that is accurate for some groups and not others, deployed without that knowledge, systematically misallocates clinical resources (Martin et al., 2019).

---

## 26.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: sanity checks for any genotype-to-phenotype ML claim"
    **Setting.** A paper reports that a neural network predicts a trait (or classifies cases) from genotype or from a sequence-model embedding, and outperforms a polygenic score.

    **Checks, in order.**

    1. **What is the ceiling?** $h^2_\text{SNP}$ for the trait and the infinitesimal limit $h^2\cdot Nh^2/(Nh^2+M)$ for the sample size. A reported $R^2$ above the SNP heritability must be explained (non-additive effects? ancestry? leakage?).
    2. **Is ancestry a hidden predictor?** Can the model's inputs predict ancestry or recruitment site, and does the phenotype differ by those? Include PCs and compare with and without.
    3. **Is relatedness controlled?** Related individuals across train and test inflate accuracy; use kinship-aware splits.
    4. **How were variants selected?** If variants were chosen using the *test* individuals' association statistics, the estimate has "winner's curse" leakage. Selection and weights must be fitted in the training set only.
    5. **Is LD respected in the split?** If the model uses annotation of variants or windows of sequence, ensure that train and test regions are separated beyond LD (e.g., held-out chromosomes for sequence models, held-out loci for fine-mapping benchmarks).
    6. **What are the baselines?** LMM/ridge, C+T, LDpred-type methods, with the same variants. A deep model must beat these *at the same sample size*.
    7. **What is the portability?** Evaluate across ancestry (continuous genetic distance) and across cohorts.
    8. **Is the claim causal?** For "the model identified causal variants", demand experimental validation or at least colocalization and fine-mapping posterior probabilities.
    9. **How were labels derived?** Electronic-health-record phenotypes carry misclassification and healthcare-utilization biases that differ by ancestry.

    **What it teaches.** Genotype–phenotype ML is the clearest case in this book where a noise ceiling (heritability), a nuisance structure (LD, ancestry, relatedness), and a causal question (MR) all interact. Mastering it is a template for evaluating any biological ML claim.

    **An open question to carry forward.** The additive LMM baseline is strong and cheap. Under what assumptions about the *architecture* of trait genetics (sparsity, non-additivity, context-specific effects across cell types) would a foundation model that reads the DNA sequence *around* each variant beat it, and how large a discovery sample would you need to know? Frame this with the sufficiency bound of Chapter 17 and design a simulation in which the true effect depends on a nonlinear function of the local sequence (for instance, a motif-disruption model as in Chapter 22) and compare an LMM with a sequence-based model, varying $N$.

---

## 26.12 Connections

- **Backward:** the noise ceiling and heritability as a bound (Chapters 1, 7); ridge as a Gaussian prior and the Gaussian-process view (Chapter 7); ill-conditioning (Chapters 2, 3); confounding and pseudoreplication (Chapter 4); LD and recombination (Chapter 21); variant effects (Chapter 20); Bayes factors (Chapters 4, 8).
- **Forward:** sequence-to-function models as variant-effect priors (Chapter 31); protein and RNA variant effects (Chapters 33, 34); genotype-to-phenotype modeling with foundation models (Chapter 41); benchmark leakage across LD (Chapter 43); causal inference and instrumental variables (Chapter 44); distribution shift across ancestry (Chapter 45); open problems for genomes (Chapter 50).

!!! takeaways "Key takeaways"
    1. In the additive model $h^2=\beta^\top R\beta$ is the variance explained by genotype; it is a population-level noise ceiling for any additive genotype-only predictor, not a measure of determinism.
    2. **LD blurs causation into association**: $z\sim\mathcal N(\sqrt nR\beta,R)$. Lead SNPs are tags; recovering $\beta$ is an ill-conditioned deconvolution.
    3. **LD score regression**: $\mathbb E\chi^2_j=1+na+nh^2\ell_j/M$; slope is polygenicity, intercept is confounding. In simulation the intercept was 0.99 without and 2.30 with stratification, while $\lambda_\text{GC}$ was 1.60 in a confound-free polygenic trait.
    4. **A linear mixed model is ridge regression**, with $\lambda=\sigma_e^2M/\sigma_g^2$: the standard genetics baseline is an $\ell_2$-penalized linear model.
    5. **Fine-mapping** turns $z$-scores into posterior inclusion probabilities; in tight LD ($\rho=0.97$, $z\approx4$) the lead SNP is causal in 28% of loci and the median 95% credible set has 15 variants; calibration is conditional on the model assumptions.
    6. **PGS accuracy** $R^2\approx h^2\,Nh^2/(Nh^2+M)$; portability across populations fails by *tagging* (83% loss with identical causal effects) rather than because biology differs; causal-variant-based scores port better.
    7. **Mendelian randomization** estimates $\theta$ by the Wald ratio and IVW; weak instruments (two-sample: bias toward zero) and directional pleiotropy bias it; MR-Egger corrects the latter at the cost of variance. Genetic support is associated with a 2.6-fold higher probability of clinical success.
    8. **Sequence models can enter as priors** for causal variants and as predictors of rare-variant effects, where association statistics are underpowered.

---

## Further reading

- Visscher, P. M. et al. (2017). 10 years of GWAS discovery: biology, function, and translation. *Am. J. Hum. Genet.* 101, 5–22. Yengo, L. et al. (2022). A saturated map of common genetic variants associated with human height. *Nature* 610, 704–712. Risch, N. & Merikangas, K. (1996). The future of genetic studies of complex human diseases. *Science* 273, 1516–1517. Pe'er, I. et al. (2008). Estimating and improving the power of genome-wide association studies. *Genet. Epidemiol.* 32, 381–385.
- Yang, J. et al. (2010). Common SNPs explain a large proportion of the heritability for human height. *Nat. Genet.* 42, 565–569. Yang, J., Lee, S. H., Goddard, M. E. & Visscher, P. M. (2011). GCTA: a tool for genome-wide complex trait analysis. *Am. J. Hum. Genet.* 88, 76–82. Loh, P.-R. et al. (2015). Efficient Bayesian mixed-model analysis increases association power in large cohorts (BOLT-LMM). *Nat. Genet.* 47, 284–290. Mbatchou, J. et al. (2021). Computationally efficient whole-genome regression for quantitative and binary traits (REGENIE). *Nat. Genet.* 53, 1097–1103.
- Price, A. L. et al. (2006). Principal components analysis corrects for stratification in genome-wide association studies. *Nat. Genet.* 38, 904–909. Bulik-Sullivan, B. K. et al. (2015). LD score regression distinguishes confounding from polygenicity in genome-wide association studies. *Nat. Genet.* 47, 291–295.
- Wakefield, J. (2009). Bayes factors for genome-wide association studies. *Genet. Epidemiol.* 33, 79–86. Benner, C. et al. (2016). FINEMAP. *Bioinformatics* 32, 1493–1501. Wang, G., Sarkar, A., Carbonetto, P. & Stephens, M. (2020). A simple new approach to variable selection in regression, with application to genetic fine mapping (SuSiE). *J. R. Stat. Soc. B* 82, 1273–1300. Weissbrod, O. et al. (2020). Functionally informed fine-mapping and polygenic localization of complex trait heritability. *Nat. Genet.* 52, 1355–1363.
- Daetwyler, H. D., Villanueva, B. & Woolliams, J. A. (2008). Accuracy of predicting the genetic risk of disease using a genome-wide approach. *PLoS ONE* 3, e3395. Dudbridge, F. (2013). Power and predictive accuracy of polygenic risk scores. *PLoS Genet.* 9, e1003348. Vilhjálmsson, B. J. et al. (2015). Modeling linkage disequilibrium increases accuracy of polygenic risk scores (LDpred). *Am. J. Hum. Genet.* 97, 576–592. Martin, A. R. et al. (2019). Clinical use of current polygenic risk scores may exacerbate health disparities. *Nat. Genet.* 51, 584–591. Duncan, L. et al. (2019). Analysis of polygenic risk score usage and performance in diverse human populations. *Nat. Commun.* 10, 3328. Ding, Y. et al. (2023). Polygenic scoring accuracy varies across the genetic ancestry continuum. *Nature* 618, 774–781.
- Davey Smith, G. & Ebrahim, S. (2003). 'Mendelian randomization': can genetic epidemiology contribute to understanding environmental determinants of disease? *Int. J. Epidemiol.* 32, 1–22. Bowden, J., Davey Smith, G. & Burgess, S. (2015). Mendelian randomization with invalid instruments: effect estimation and bias detection through Egger regression. *Int. J. Epidemiol.* 44, 512–525. Burgess, S. et al. (2017). A review of instrumental variable estimators for Mendelian randomization. *Stat. Methods Med. Res.* 26, 2333–2355. Minikel, E. V., Painter, J. L., Dong, C. C. & Nelson, M. R. (2024). Refining the impact of genetic evidence on clinical success. *Nature* 629, 624–629.
- Wainschtein, P. et al. (2022). Assessing the contribution of rare variants to complex trait heritability from whole-genome sequence data. *Nat. Genet.* 54, 263–273. "Estimation and mapping of the missing heritability of human phenotypes" (2025/2026). *Nature* 649, 1219–1227.
