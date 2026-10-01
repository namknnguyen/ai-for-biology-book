# Chapter 5. Information Theory: Entropy, Compression, and Evolution as a Channel

!!! abstract "Chapter at a glance"
    **Motivation.** "What does this representation retain?", "how much does the measurement tell us?", and "what does a language model's likelihood mean?" are all questions about *information*. Information theory turns each into a quantity that can be bounded, estimated, and compared.
    **Prerequisites.** Chapter 4 (likelihood, expectation).
    **You will be able to:** (1) define entropy, cross-entropy, KL divergence, and mutual information and prove their key properties; (2) show that maximum likelihood minimizes a KL divergence and explain forward versus reverse KL; (3) prove the data-processing inequality and state what it forbids; (4) explain why a language model is a compressor and what "bits per base" measures; (5) analyze motifs, measurements, and selection as information channels; (6) derive the relationship between a sequence model's log-likelihood ratio and evolutionary fitness, with its assumptions.


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M2](m02-functions-exponentials-logs.md) (logarithms) and [M7](m07-probability.md) (expectation and distributions).

---

## 5.1 Why information theory belongs in a biology-ML book

Claude Shannon's 1948 theory was built for telegraph lines, but its objects are exactly the objects of our field.

- A **genome** is a message written with a four-letter alphabet; **mutation** is noise; **selection** is a filter that keeps messages that work.
- A **sequencing assay** is a noisy channel from latent molecular state to counts.
- A **representation learner** is a (lossy) code: it chooses what to keep.
- A **language model** assigns a probability to a sequence, which is, by Shannon's source-coding theorem, the same thing as assigning it a code length.

Three questions recur in this book and are answered with the quantities of this chapter:

1. *How much does $X$ tell us about $Y$?* → mutual information.
2. *How well does model $q$ describe reality $p$?* → cross-entropy and KL divergence.
3. *What is the best any procedure could possibly do?* → bounds such as the data-processing inequality.

---

## 5.2 Entropy and surprise

For a discrete random variable $X$ with distribution $p$, the **surprise** (information content) of outcome $x$ is $-\log p(x)$: rare outcomes are more surprising. The **entropy** is the expected surprise:

$$
\Ent(X)=-\sum_xp(x)\log p(x)=\E_{x\sim p}\big[-\log p(x)\big].
$$

With $\log_2$ the unit is **bits**; with $\ln$, **nats** ($1$ nat $=1.4427$ bits). Entropy is zero for a deterministic variable and maximal, $\log|\mathcal{X}|$, for a uniform one. For DNA, the maximum is $\log_24=2$ bits per base.

**Operational meaning (source coding).** If you must encode i.i.d. symbols from $p$ as binary strings, the best achievable expected length per symbol is within one bit of $\Ent(p)$ (and approaches it for long blocks). A code that assigns length $\ell(x)=-\log_2q(x)$ to outcome $x$ is optimal for a source distributed as $q$. So *entropy is the minimum average number of bits per symbol needed to represent the source.*

**Conditional entropy and the chain rule.** $\Ent(Y\mid X)=\E_x[\Ent(Y\mid X=x)]$ and $\Ent(X,Y)=\Ent(X)+\Ent(Y\mid X)$. For a sequence, $\Ent(x_{1:L})=\sum_{i}\Ent(x_i\mid x_{<i})$: the autoregressive factorization of a language model is the chain rule of entropy.

**Entropy rate of genomes.** A simple i.i.d. model gives $\approx1.9$–$2.0$ bits per base for human DNA, because base composition is nearly uniform within windows. Order-$k$ context models and specialized DNA compressors reduce this to roughly $1.6$–$1.9$ bits per base depending on the region and method; repeats compress well while unique sequence barely compresses at all. [[S]] The gap between 2 bits and the best achievable code length is, in this view, the *total information in the genome's statistical regularities*; most regularities (repeats, composition) are not the functional information we care about.

---

## 5.3 Cross-entropy and KL divergence

**Cross-entropy.** The expected code length when data come from $p$ but we use a code built for $q$:

$$
H(p,q)=-\sum_xp(x)\log q(x)=\Ent(p)+\KL{p}{q},\qquad \KL{p}{q}=\sum_xp(x)\log\frac{p(x)}{q(x)} .
$$

So the **KL divergence is the excess code length** (the inefficiency) from modeling $p$ with $q$.

### 5.3.1 Gibbs' inequality: $\KL{p}{q}\ge0$

*Proof.* $-\KL{p}{q}=\sum_xp(x)\log\frac{q(x)}{p(x)}\le\log\sum_xp(x)\frac{q(x)}{p(x)}=\log\sum_{x:p(x)>0}q(x)\le\log1=0$, using Jensen's inequality for the concave $\log$. Equality holds iff $q=p$ wherever $p>0$. $\square$

Hence $H(p,q)\ge\Ent(p)$ with equality iff $q=p$: **no model can compress data better than the true entropy, and the true distribution is the unique optimum of the cross-entropy loss.** This is why cross-entropy is a *proper* scoring rule: it is minimized by telling the truth.

### 5.3.2 Maximum likelihood is KL minimization

Given samples $x_1,\dots,x_n\sim p_{\text{data}}$, the average log-likelihood converges to $\E_{p_{\text{data}}}[\log p_\theta(x)]$. Then

$$
\arg\max_\theta\E_{p_\text{data}}[\log p_\theta(x)]=\arg\min_\theta\Big(\underbrace{-\E_{p_\text{data}}\log p_\text{data}(x)}_{\text{constant in }\theta}+\E_{p_\text{data}}\log\frac{p_\text{data}(x)}{p_\theta(x)}\Big)=\arg\min_\theta\KL{p_\text{data}}{p_\theta}.
$$

**Training a language model by cross-entropy is minimizing the forward KL divergence from the data distribution to the model**, equivalently minimizing the expected code length of the data under the model. *Perplexity* is $e^{H(p_\text{data},p_\theta)}$ (or $2^{\text{bits}}$): the effective number of equally likely choices per token. A DNA language model with cross-entropy of 1.9 bits per base has perplexity $2^{1.9}\approx3.7$, against 4 for a uniform model.

### 5.3.3 KL is asymmetric: mode covering versus mode seeking

$\KL{p}{q}\ne\KL{q}{p}$, and the difference shapes the behavior of generative models.

- **Forward KL $\KL{p}{q}$** (maximum likelihood) penalizes $q$ heavily wherever $p>0$ but $q\approx0$ ($\log(p/q)\to\infty$). So $q$ must **cover all modes** of $p$, even at the cost of placing mass in low-density regions between them.
- **Reverse KL $\KL{q}{p}$** (variational inference; some RL objectives) penalizes $q$ for placing mass where $p\approx0$ but does not penalize missing modes. So $q$ is **mode-seeking**: it locks onto one mode.

The code fits a single Gaussian to a two-component mixture: the forward-KL fit has mean 0 and standard deviation 3.08 (a broad blob between the modes), the reverse-KL fit has mean $-3.00$ and standard deviation 0.70 (it picks one mode exactly). *For biology, this is a design choice with consequences*: a model trained for coverage can generate sequences that are blends of different functional classes (non-functional "in-between" proteins), whereas a mode-seeking procedure produces high-quality but low-diversity designs. Chapters 14, 15, and 36 return to this.

```python
--8<-- "code/ch05_information.py"
```

Output (seed 0):

```text
bimodal p: forward KL(p||q) fit  -> mean  0.00, sd 3.08  (covers both modes, puts mass between them)
           reverse KL(q||p) fit  -> mean -3.00, sd 0.70  (locks onto one mode)

source entropy rate = 1.216 bits/base (uniform coding would need 2.000)
  order-0 model: held-out cross-entropy = 1.969 bits/base
  order-1 model: held-out cross-entropy = 1.885 bits/base
  order-2 model: held-out cross-entropy = 1.681 bits/base
  order-3 model: held-out cross-entropy = 1.214 bits/base
  order-4 model: held-out cross-entropy = 1.217 bits/base
  order-6 model: held-out cross-entropy = 1.244 bits/base
  order-8 model: held-out cross-entropy = 1.333 bits/base

motif information per position (bits): [1.37 1.15 0.   0.64 0.43 1.42]; total R_seq = 5.0 bits
  chance occurrences expected in a 3.1e+09-base genome (both strands) ~ 191,151,740; bits needed for a unique site ~ 31.5

data processing: I(X;Y) = 0.491 bits >= I(X;Z) = 0.171 bits
```

---

## 5.4 A language model is a compressor

The second block of output is a miniature of the central idea of Part VII. We generated a synthetic "genome" from a hidden order-3 Markov source whose entropy rate is 1.216 bits per base, trained order-$k$ context models on 120,000 bases, and measured held-out cross-entropy on 60,000 more.

- Order 0 (base composition only) achieves 1.969 bits, almost no better than uniform (2 bits): the regularities are in the *context*.
- Cross-entropy falls as the model's context grows until $k=3$, the true order, where it reaches $1.214$ bits per base, within estimation error of the source's entropy rate.
- Beyond the true order, held-out cross-entropy *rises* (1.217, 1.244, 1.333 for $k=4,6,8$): the model has more contexts than the data can support, so estimates are noisy. This is **overfitting seen as wasted code length**: $4^8=65{,}536$ contexts with 120,000 training bases means about two observations per context.

By arithmetic coding, any probabilistic model $p_\theta$ defines a lossless compressor whose output length for sequence $x$ is $\approx-\log_2p_\theta(x)$ bits. Therefore:

> **A better language model is a better compressor, and the amount of compression is the model's measure of how much structure it has captured.**

For real genomes this reframes pretraining objectives. A DNA language model with lower cross-entropy has *captured more statistical structure*, but it does not follow that the captured structure is *functional* structure. Most of the achievable compression in the human genome comes from repeats and composition. Chapters 3 and 32 return to this: the average loss is a frequency-weighted mixture of tasks.

!!! rhyme "Structural rhyme: compression ↔ prediction ↔ learning"
    Minimum description length (MDL; Rissanen) says the best model of data minimizes *model description length + data code length under the model*. Maximum-likelihood with regularization is MDL; Bayesian model comparison is MDL (the marginal likelihood penalizes complexity automatically, the "Occam factor"); and the held-out cross-entropy curve above is MDL's empirical face. **Prediction, compression, and generalization are one problem seen three ways.**

---

## 5.5 Mutual information and the data-processing inequality

The **mutual information** between $X$ and $Y$ is

$$
\MI(X;Y)=\KL{p(x,y)}{p(x)p(y)}=\Ent(X)-\Ent(X\mid Y)=\Ent(Y)-\Ent(Y\mid X)\ \ge0 .
$$

It measures how much knowing one variable reduces uncertainty about the other, in bits or nats: zero iff independent, and it captures **any** statistical dependence (nonlinear included), whereas correlation captures only linear dependence. Conditional mutual information $\MI(X;Y\mid Z)=\Ent(X\mid Z)-\Ent(X\mid Y,Z)$ obeys the **chain rule** $\MI(X;Y,Z)=\MI(X;Z)+\MI(X;Y\mid Z)$.

### 5.5.1 The data-processing inequality (DPI)

If $X\to Y\to Z$ is a Markov chain ($Z$ depends on $X$ only through $Y$, i.e., $X\indep Z\mid Y$), then

$$
\MI(X;Z)\le\MI(X;Y).
$$

*Proof.* Expand $\MI(X;Y,Z)$ in two ways using the chain rule: $\MI(X;Y,Z)=\MI(X;Y)+\MI(X;Z\mid Y)=\MI(X;Y)$, because $X\indep Z\mid Y$ makes the second term zero. Also $\MI(X;Y,Z)=\MI(X;Z)+\MI(X;Y\mid Z)\ge\MI(X;Z)$. Hence $\MI(X;Y)\ge\MI(X;Z)$. $\square$

**What it forbids.** *No processing of data can create information about the source.* In our notation, if $\mathbf{s}\to x\to h=\phi_\theta(x)$, then $\MI(\mathbf{s};h)\le\MI(\mathbf{s};x)$ (Chapter 1's measurement gap). The numerical check in the code draws a random chain and finds $\MI(X;Y)=0.491$ bits and $\MI(X;Z)=0.171$ bits.

**What it does not forbid.** Processing can make information *more accessible*: a nonlinear feature map can turn information that no linear probe could read into information a linear probe reads easily. Information that is *present* ($\MI$) is different from information that is *usable by a given class of readouts* (the "$\mathcal{V}$-information" of Xu et al., 2020). Pretrained representations help precisely when they make the relevant information usable with few labels.

### 5.5.2 Sufficiency, minimal sufficiency, and the information bottleneck

A representation $h=\phi(x)$ is **sufficient** for a target $y$ if $\MI(h;y)=\MI(x;y)$, i.e., it retains everything in $x$ relevant to $y$. It is **minimal** if it has the smallest $\MI(h;x)$ among sufficient representations (it discards everything irrelevant). The **information bottleneck** (Tishby et al., 1999) formalizes a trade-off: maximize $\MI(h;y)-\beta\MI(h;x)$. A *self-supervised* representation is sufficient for the pretraining target; Chapter 1's objective gap is the statement that sufficiency for the pretraining target does not imply sufficiency for the downstream target. We can now say it precisely: **if the downstream variable $y'$ is not a function of the pretraining target's sufficient statistic, then $\MI(h;y')$ may be less than $\MI(x;y')$, and no probe on $h$ can recover the difference.**

### 5.5.3 Estimating mutual information

MI is easy to define and hard to estimate from samples in high dimension. The practical estimators are variational *lower bounds* trained with neural networks. The most important for this book is the **InfoNCE** bound, derived in Chapter 13: $\MI(X;Y)\ge\log K-\mathcal{L}_{\text{InfoNCE}}$ where $K$ is the number of negatives; its limits ($\le \log K$) explain why contrastive methods use many negatives.

---

## 5.6 Biological applications of the information view

### 5.6.1 The information content of a motif

A position in a binding motif with nucleotide probabilities $p_{j,a}$ has information $2-\Ent_j$ bits ($\Ent_j=-\sum_ap_{j,a}\log_2p_{j,a}$): the reduction in uncertainty from the background 2 bits. Summed over positions, $R_{\text{seq}}=\sum_j(2-\Ent_j)$ is the *total information content* of the motif (Schneider, 1986), the quantity plotted as the height of **sequence logos**. In the example PWM the total is $R_{\text{seq}}=5.0$ bits: a site this specific occurs by chance once every $2^5=32$ bases, i.e., about 191 million times (counting both strands) in a 3.1-gigabase genome, whereas finding a *unique* site in the genome requires $\log_2(3.1\times10^9)\approx31.5$ bits.

*The punchline:* transcription factor motifs typically carry 10–20 bits, far less than 31.5. Most motif matches in the genome are therefore **not bound** in vivo. Specificity must come from *additional* information: cooperative binding of multiple factors, chromatin accessibility, nucleosome positioning, DNA shape, and genomic context (Chapter 22). A sequence model that sees only a short window is information-limited in a way no architecture change can fix; this is the information-theoretic form of the argument that regulation is a **context-dependent** function of sequence. [[E]] as arithmetic; [[S]] for the biological interpretation.

**Information and energy.** Under the Berg–von Hippel model, a factor's binding energy to a site is approximately additive across positions, so the log-odds weight matrix equals the energy matrix (in units of $k_BT$) up to constants; information content is the relative entropy between the bound-site distribution and the background. This is a *physical* reading of a PWM (Chapter 29).

### 5.6.2 The measurement as a channel: how many bits does a gene's count carry?

A noisy measurement is a channel from the true quantity $S$ to the observation $X$. For a Gaussian channel with signal variance $\sigma_s^2$ and noise variance $\sigma_n^2$, the mutual information is

$$
\MI(S;X)=\tfrac12\log_2\!\big(1+\sigma_s^2/\sigma_n^2\big)\ \text{bits}.
$$

For a count with mean $\mu$, the Poisson variance on the log scale is approximately $1/\mu$ (delta method). Suppose the cell-to-cell standard deviation of true log-expression is 1 ($\sigma_s^2=1$). Then a gene detected at $\mu=5$ counts has SNR $\approx5$ and carries $\approx\tfrac12\log_2 6\approx1.3$ bits per cell; a gene at $\mu=0.5$ has SNR $\approx0.5$ and carries $\approx0.3$ bits. [[P]] (a rough, Gaussian-approximate calculation; the delta-method noise estimate is poor at very low counts, where the true information is lower still.) **Most genes in a typical single-cell experiment are measured so noisily that each carries well under one bit**; the information resides in the *joint* pattern across thousands of genes, which is why cell-state structure is recoverable but individual-gene behavior in individual cells is not. This is why gene-level predictions evaluated cell by cell hit a low ceiling (Chapters 1, 38, 39).

### 5.6.3 Selection as a filter: how fast can genomes accumulate information?

Let a population have distribution $p(x)$ over genotypes and let fitness $w(x)\in[0,1]$ be the survival probability. After selection, $q(x)=p(x)w(x)/\bar w$ with $\bar w=\sum_xp(x)w(x)$. The information gained is

$$
\KL{q}{p}=\E_q\Big[\log\frac{w(x)}{\bar w}\Big]\le\log\frac{w_{\max}}{\bar w}\le\log\frac1{\bar w}.
$$

So one generation of selection can add at most $\log_2(1/\bar w)$ bits to the genome's description: *information gain is bounded by the (log) fraction of offspring that are removed*, the "cost of selection" (Kimura, 1961; Haldane's dilemma). [[E]] as mathematics. Biologically it explains why evolution can accumulate specific information about the environment only slowly, and why the genome is not a lookup table: the total functional information of the human genome, estimated from the fraction under purifying selection (roughly 5–10% by comparative-genomic estimates), is much smaller than its raw 6.2 gigabit capacity. [[S]] for the fraction, [[P]] for the interpretation as "information."

### 5.6.4 Why a sequence model's likelihood might track fitness

This is the central bridge between language modeling and evolution (Chapters 32, 34, 42), so we derive it with its assumptions visible.

!!! math "Derivation: stationary distribution of mutation–selection–drift and the log-likelihood ratio"
    Consider a population of effective size $N$ in the *weak-mutation* regime, in which a new mutation fixes or is lost before the next arises, so the population is nearly monomorphic and moves between genotypes. Let $\mu(x\to x')$ be the rate at which mutation proposes $x'$ from $x$, and $\rho(x\to x')$ the probability that the mutant fixes. The chain over genotypes has transition rate $\mu(x\to x')\rho(x\to x')$. At stationarity, detailed balance holds:

    $$
    \pi(x)\,\mu(x\to x')\,\rho(x\to x')=\pi(x')\,\mu(x'\to x)\,\rho(x'\to x).
    $$

    Let $s=\log f(x')-\log f(x)$ be the selection coefficient of the mutant relative to the resident. In one common convention (a haploid population of size $N$ under the diffusion approximation), the fixation probability of a new mutant is $\rho(s)=\dfrac{1-e^{-2s}}{1-e^{-2Ns}}$. The ratio of fixation probabilities for the forward and reverse substitutions is

    $$
    \frac{\rho(s)}{\rho(-s)}=\frac{(1-e^{-2s})(1-e^{2Ns})}{(1-e^{-2Ns})(1-e^{2s})}=e^{-2s}\cdot e^{2Ns}=e^{2(N-1)s}.
    $$

    Therefore

    $$
    \frac{\pi(x')}{\pi(x)}=\frac{\mu(x\to x')}{\mu(x'\to x)}\,e^{2(N-1)\,(\log f(x')-\log f(x))}.
    $$

    Taking logs, for a variant $x\to x'$,

    $$
    \underbrace{\log\pi(x')-\log\pi(x)}_{\text{equilibrium log-likelihood ratio}}=2(N-1)\,\Delta\log f+\underbrace{\log\frac{\mu(x\to x')}{\mu(x'\to x)}}_{\text{mutation bias}} .
    $$

    (Conventions differ by constants, e.g., $4N_e$ for diploid Wright–Fisher; Sella & Hirsh, 2005.)

**Reading the result.** If a language model $p_\theta$ perfectly learned the equilibrium distribution $\pi$, its log-likelihood ratio for a variant would equal a *scaled difference in log fitness* plus a *mutation-bias term*. This is the theoretical justification for **zero-shot variant-effect prediction by $\Delta\log p_\theta$** (Chapters 32, 34). It also lists exactly what has to hold, and each item is a potential failure mode:

1. **Equilibrium.** Sequences must be samples from a *stationary* distribution; recent adaptation, bottlenecks, or lineage-specific selection violate this.
2. **Independence across samples.** Extant sequences are related by common descent. A corpus of homologous sequences is not an i.i.d. sample from $\pi$; it is a *correlated* one (phylogenetic bias; Chapters 21, 34).
3. **The model has learned $\pi$**, including the right dependence between sites (epistasis), not just marginal frequencies.
4. **Mutation bias** is not negligible: transitions outnumber transversions; CpG sites mutate at higher rates. A raw $\Delta\log p$ conflates fitness and mutability; mutation-aware calibration helps.
5. **Fitness means ancestral fitness in natural environments.** A deep mutational scan measures a *laboratory proxy* (stability, binding, growth of a library); the correspondence to evolutionary fitness is only approximate and varies by protein and assay.
6. **Scale $N$.** The factor $2(N-1)$ means predicted effect sizes are in units that depend on population genetics, so only the *ranking* of variants, not the magnitude, is meaningful without calibration.

None of these is a flaw in the method; they are the *conditions under which a zero-shot claim is justified*, and failures of the claim can be traced to violations of a specific condition. This is the pattern of reasoning the Expert Chain asks for: assumptions (L3) → failure modes (L5) → hypotheses (L8).

---

## 5.7 Worked research examples

!!! example "Worked Research Example 5.1: A DNA language model improves from 1.95 to 1.90 bits per base. What happened?"
    **Situation.** A new DNA language model reports validation cross-entropy of 1.90 bits per base on human chromosomes, against 1.95 for a smaller predecessor, a 2.6% reduction. The authors describe it as "learning substantially more of the genome."

    **Question.** What does 0.05 bits per base buy, biologically?

    **Reasoning.**

    1. *What is the unit?* An average over all positions. Positions are not equally important: about half of the genome is derived from transposable elements, large fractions are low-complexity, while protein-coding sequence is ~1–2% and known regulatory elements a few percent more.
    2. *What upper bounds exist?* If the improvement came entirely from a region making up fraction $f$ of the genome, the gain at those positions is $0.05/f$ bits per base. If $f=0.5$ (repeats), that is 0.1 bit per base in repeat regions; if $f=0.02$ (coding), it would be 2.5 bits per base, impossible since the baseline is at most 2 bits: so *the improvement cannot be mainly from coding sequence*.
    3. *What alternative explanations exist?* (i) Better modeling of repeat families (memorization of high-copy elements). (ii) Better modeling of local composition (GC isochores, CpG depletion). (iii) Improved modeling of recently duplicated segments. (iv) Genuine regulatory or coding signals. (v) Train–test overlap: repeated elements and segmental duplications mean validation chromosomes contain near-copies of training sequence.
    4. *What experiments discriminate?* Report cross-entropy *by annotation class* (repeat family, coding, UTR, promoter, enhancer, intergenic unique); mask repeats (soft-masked genome) and rerun; compute loss on sequences with low homology to the training set; compare with a $k$-mer baseline within each class.
    5. *Predictions.* Under (i)–(iii), improvements concentrate in repeats and low-complexity regions and vanish on unique, low-homology regions. Under (iv), improvements appear in coding and conserved noncoding regions, strongest where cross-species conservation is high.

    **Expert analysis.** A 0.05-bit improvement is *unlabeled*: it cannot be interpreted without the by-class breakdown. This is a statement about the **objective** (G-O): aggregate likelihood weights tokens by frequency. The by-class table is a cheap diagnostic. In Chapter 32 you will see exactly such breakdowns used to argue about what Evo-style models learn; the reasoning here tells you which table to demand.

!!! example "Worked Research Example 5.2: Does $\Delta\log p$ predict the effect of a regulatory variant?"
    **Situation.** A genomic language model scores a noncoding variant by $\Delta\log p=\log p_\theta(x')-\log p_\theta(x)$. Authors report that it correlates with MPRA-measured expression effects at $\rho=0.15$ and with conservation at $\rho=0.4$. They conclude that the model "has learned regulatory constraint."

    **Question.** Evaluate the conclusion using the derivation of §5.6.4.

    **Reasoning.**

    1. *Which conditions of the derivation can be checked here?* (1) Equilibrium: noncoding regulatory sequence is under weaker, more lineage-specific selection and evolves faster than protein sequence, so the "stationary" assumption is weaker. (2) Independence: if the training corpus includes many related genomes, phylogenetic bias inflates the apparent conservation. (3) Mutation bias: CpG transitions have high mutation rate and low conservation constraint, producing a large $\Delta\log p$ signal unrelated to function. (5) The MPRA measures expression of a short fragment in one cell type, while selection acts on organismal fitness across contexts.
    2. *Alternative explanations for $\rho=0.4$ with conservation.* The model's likelihood already encodes conservation (it is trained on aligned-by-descent sequences), so correlating with conservation is *close to a tautology* and is not independent evidence of regulatory understanding.
    3. *Why only 0.15 with MPRA?* MPRA effects are context-specific and small; the MPRA ceiling (replicate correlation) may itself be low (0.5–0.7). The model captures *constraint*, not *cell-type-specific activity* (G-O, G-M).
    4. *Discriminating experiments.* (a) Stratify by mutation type (CpG vs non-CpG) and by distance to conserved elements. (b) Compare to a simple conservation score (phyloP) *and* to a supervised sequence-to-function model on the same variants; the language-model score should be evaluated by the *information it adds* beyond conservation (partial correlation / nested regression). (c) Evaluate on MPRA variants in elements that are *not conserved* but are active, where constraint-based scores should fail and a function-based model should succeed. (d) Use a held-out species or clade for training (does the score depend on the closeness of relatives in the corpus?).
    5. *What would change your mind?* If $\Delta\log p$ adds substantial predictive information over phyloP and GC/CpG covariates on non-conserved active elements, then the model is capturing more than conservation. If its predictive power vanishes after conditioning on conservation, it is a (learned, scalable) conservation score and should be described as such.

    **Expert analysis.** The central move is to ask *what the theory says the score should equal*, then enumerate where the data-generating process deviates. A theoretically motivated score has *named failure modes* that you can test one at a time; an atheoretical one does not. That is the practical value of §5.6.4.

---

## 5.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: is the information in the input at all?"
    **Problem.** A model predicts the expression change of gene $g$ after knocking out transcription factor $c$, from sequence embeddings of $g$'s promoter. It performs near the mean baseline. Is it a modeling failure or an information failure?

    **Step 1: Write the information structure.** Let $y$ be the response, $x$ the promoter sequence of $g$, $c$ the identity of the perturbed factor, $t$ the cell state. The response is determined jointly by $(x,c,t)$ and a lot of unobserved state. For the model to succeed you need $\MI(y;x,c,t)$ substantially above $\MI(y;c)$ or $\MI(y;x)$ alone.

    **Step 2: Estimate what each variable contributes.** Fit flexible models (gradient-boosted trees, small networks) with feature sets $\{c\}$, $\{x\}$, $\{c,t\}$, $\{x,c\}$, $\{x,c,t\}$ and compare held-out performance, using the *same* split and noise ceiling. The increase from adding a variable lower-bounds its conditional information.

    **Step 3: Interpret.** (a) $\{x\}$ alone near baseline, $\{c\}$ alone near baseline, $\{x,c\}$ much better: the information is in the **interaction** (does factor $c$'s motif occur in $x$?), a hint that the architecture must be able to represent that interaction. (b) $\{c,t\}$ explains most: the response is mostly a property of the cell state, not the sequence. (c) Nothing helps: the response is dominated by unmeasured variables; collect new data (A4) or redefine the problem (A8).

    **Step 4: Consider a bound.** If replicate perturbation effects correlate at 0.5, no feature set can exceed $R^2\approx0.5$ (Chapter 1).

    **What it teaches.** Before blaming a model, *measure how much each input could in principle tell you*. The same exercise applied to a published model (via ablations or by reproducing with simpler learners) often reveals that the headline architecture contributes little beyond one input feature.

---

## 5.9 Connections

- **Backward:** Chapter 4's likelihood is $-\log$ code length. The Gaussian-channel formula uses Chapter 2's variance/SNR reasoning. Ridge/MAP as MDL.
- **Forward:** The ELBO is $\log p(x)$ minus a KL (Chapter 8, 14). InfoNCE lower-bounds mutual information (Chapter 13). Forward/reverse KL drive the behavior of generative models (Chapters 14, 15, 36). The Sella–Hirsh relation underlies zero-shot variant scoring (Chapters 32, 34, 42). Population-genetic information (Chapter 21) and the DPI in the argument about foundation models and the measurement gap (Chapters 17, 45, 47).

!!! takeaways "Key takeaways"
    1. **Entropy** is expected surprise and minimal average code length; **cross-entropy** $=\Ent+\KL{p}{q}$; $\KL\ge0$ (Gibbs). **Maximum likelihood = minimizing forward KL** from data to model.
    2. **Forward KL covers modes; reverse KL seeks modes.** This matters for generative design.
    3. **A language model is a compressor.** Better compression means more captured *statistical* structure, not necessarily more *functional* structure; most genome compressibility is repeats and composition.
    4. **Mutual information** captures any dependence; the **data-processing inequality** forbids creating information by processing: $\MI(\mathbf{s};\phi(x))\le\MI(\mathbf{s};x)$. Processing can still change *accessibility* of information to a given readout.
    5. TF motifs carry ~10–20 bits against the ~31.5 bits needed to be unique in the human genome; **context carries the rest.**
    6. Per-gene counts in single cells typically carry **well under one bit**; information resides in the joint pattern.
    7. The equilibrium of mutation–selection–drift is Boltzmann-like in log fitness, so a perfect sequence model's $\Delta\log p\approx2(N-1)\Delta\log f+$ mutation bias, subject to the six conditions listed.

---

## Further reading

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal* 27, 379–423, 623–656.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. The standard text.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. Free online; unites coding and learning.
- Grünwald, P. D. (2007). *The Minimum Description Length Principle*. MIT Press.
- Schneider, T. D., Stormo, G. D., Gold, L. & Ehrenfeucht, A. (1986). Information content of binding sites on nucleotide sequences. *J. Mol. Biol.* 188, 415–431.
- Berg, O. G. & von Hippel, P. H. (1987). Selection of DNA binding sites by regulatory proteins. *J. Mol. Biol.* 193, 723–750.
- Kimura, M. (1961). Natural selection as the process of accumulating genetic information in adaptive evolution. *Genetical Research* 2, 127–140.
- Sella, G. & Hirsh, A. E. (2005). The application of statistical physics to evolutionary biology. *PNAS* 102, 9541–9546.
- Rands, C. M., Meader, S., Ponting, C. P. & Lunter, G. (2014). 8.2% of the human genome is constrained. *PLoS Genetics* 10, e1004525.
- Tishby, N., Pereira, F. C. & Bialek, W. (1999). The information bottleneck method. arXiv:physics/0004057.
- Xu, Y. et al. (2020). A theory of usable information under computational constraints. *ICLR*.
- Tang, Z. & Koo, P. K. and others (2024–2025) on DNA language model evaluation, cited in Chapter 32.
