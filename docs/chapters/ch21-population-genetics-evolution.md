# Chapter 21. Population Genetics and Evolutionary Models

!!! abstract "Chapter at a glance"
    **Motivation.** Evolution is the data-generating process behind every sequence in every genomic and protein corpus, and population genetics is its mathematics. To interpret what a sequence model learns (conservation, coevolution, likelihood as fitness) and to avoid its characteristic errors (phylogenetic non-independence, mutation-rate confounding, ancestry bias), you need drift, selection, the coalescent, linkage disequilibrium, and substitution models. This chapter derives them, with each result checked by simulation.
    **Prerequisites.** Chapters 3–5, 8, 20.
    **You will be able to:** (1) derive the Wright–Fisher variance of drift, heterozygosity decay, and neutral fixation probability; (2) state and use the fixation probability under selection and explain "nearly neutral"; (3) derive coalescent results (time to the most recent common ancestor, tree length, segregating sites, the neutral site-frequency spectrum); (4) derive linkage-disequilibrium decay; (5) derive the Jukes–Cantor model and use continuous-time Markov substitution models; (6) explain why phylogenetic structure inflates false positives and how to correct for it, and why this matters for sequence models.


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M7](m07-probability.md) (binomial sampling and Markov chains) and [M6](m06-linear-algebra-2.md) (rate matrices).

---

## 21.1 Why a machine-learning book needs population genetics

Three reasons, each of which returns repeatedly in Part VII.

1. **Sequences in a corpus are not independent draws.** They descend from common ancestors along a tree, so any estimate or benchmark that treats them as i.i.d. has the wrong effective sample size (Chapter 4's pseudoreplication; §21.8 below).
2. **Likelihood as fitness is a population-genetic claim.** Chapter 5 derived $\Delta\log\pi\approx2(N-1)\Delta\log f+$ mutation bias from fixation probabilities. This chapter supplies the model behind it and the conditions under which it holds.
3. **Genetic variation is the data of statistical genetics.** Allele frequencies, linkage disequilibrium, and population structure underlie GWAS, polygenic scores, and fine-mapping (Chapter 26).

---

## 21.2 Allele frequencies and Hardy–Weinberg

At a biallelic locus with allele frequencies $p$ and $q=1-p$ in a large, randomly mating population without selection, mutation, or migration, genotype frequencies are $p^2$ (AA), $2pq$ (Aa), $q^2$ (aa) after one generation of random mating, and stay there (Hardy–Weinberg equilibrium). The **expected heterozygosity** $H=2pq$ is the probability that two randomly drawn gene copies differ. Everything that follows describes how forces move $p$ over time.

---

## 21.3 Genetic drift: the Wright–Fisher model

### 21.3.1 The model

A population of $N$ diploid individuals has $2N$ gene copies. Each generation, the $2N$ copies of the next generation are drawn independently (with replacement) from the current generation. If the current frequency is $p$, the number of copies of allele A in the next generation is $\mathrm{Binomial}(2N,p)$, so

$$
\E[p'\mid p]=p,\qquad\Var(p'\mid p)=\frac{p(1-p)}{2N}.
$$

**Drift is the variance term.** There is no systematic push (the *expectation* is unchanged: a **martingale**), yet frequencies wander, with steps of size $\sqrt{p(1-p)/2N}$: small in large populations and large in small ones. The code verifies a one-generation variance of $0.00125$ for $N=100$, $p=0.5$ (theory $0.25/200=0.00125$).

### 21.3.2 Consequences

**(a) Heterozygosity decays.** Two gene copies in the new generation are copies of the *same* parental gene with probability $1/(2N)$ (identical by descent); otherwise they are independent draws. Therefore

$$
H_{t+1}=\Big(1-\frac1{2N}\Big)H_t\quad\Longrightarrow\quad H_t=H_0\Big(1-\frac1{2N}\Big)^t\approx H_0\,e^{-t/(2N)}.
$$

Simulated heterozygosity after 50 generations at $N=100$: $0.3896$ (theory $0.3892$). *Drift erodes variation at rate $1/(2N)$ per generation.*

**(b) Neutral fixation probability equals initial frequency.** Because $p_t$ is a martingale bounded in $[0,1]$ and eventually absorbed at 0 or 1 (loss or fixation), the optional stopping theorem gives $p_0=\E[p_\infty]=P(\text{fix})$. A new neutral mutation (one copy among $2N$) therefore fixes with probability $1/(2N)$. Simulation with $p_0=0.1$: fixation probability $0.1001$.

**(c) Time scales.** Conditional on fixation, the expected time for a neutral allele at initial frequency $p$ is $\bar t_1(p)=-4N\frac{1-p}{p}\ln(1-p)$ generations (Kimura & Ohta, 1969): $379$ for $N=100$, $p=0.1$ (simulated: $374$). *Fixation takes $O(N)$ generations*; for $p\to0$ this tends to $4N$ generations. This is why variants common in a population are typically *old*.

**(d) Mutation–drift balance.** With mutation rate $\mu$ per site per generation, equilibrium heterozygosity at a site is $\theta/(1+\theta)\approx\theta$ with $\theta=4N_e\mu$ (the "population mutation rate"). Human nucleotide diversity is $\pi\approx1\times10^{-3}$; with $\mu\approx1.2\times10^{-8}$ (pedigree rate) this gives $N_e\approx2\times10^4$, and with a phylogenetic rate $\mu\approx2.5\times10^{-8}$, $N_e\approx10^4$. **The effective population size $N_e$ is much smaller than the census size ($\sim10^{10}$)**: it reflects the *long-term harmonic mean* of population size through bottlenecks, and is the parameter that governs drift and the efficacy of selection (§21.4). The factor-of-two discrepancy between mutation-rate estimates is a known puzzle but does not affect the logic.

!!! rhyme "Structural rhyme: drift ↔ SGD noise ↔ finite-sample estimation noise"
    The Wright–Fisher step has the structure $\E[\Delta p]=\text{(selection)}$ plus noise with variance $p(1-p)/2N$, which is Chapter 3's *stochastic gradient descent* ($\E[\Delta w]=-\eta\nabla f$ plus noise of scale $\eta/B$). Population size $N$ plays the role of batch size: *large populations follow the fitness gradient faithfully; small populations let drift override weak selection*. The diffusion limit of the Wright–Fisher model (Kimura) is a Langevin equation whose stationary distribution is Boltzmann-like in log-fitness with inverse temperature $\propto N_e$, the same form as Chapter 5's mutation–selection equilibrium and Chapter 17's KL-regularized optimum. [[E]] (as mathematics)

---

## 21.4 Selection and the "nearly neutral" regime

### 21.4.1 Deterministic selection

Let allele A have relative fitness $1+s$ against 1 for allele a, in a haploid population of size $N$. In a large population, the ratio of frequencies grows geometrically: $\frac{p_{t+1}}{1-p_{t+1}}=(1+s)\frac{p_t}{1-p_t}$, so $p_t$ follows a logistic curve and $\Delta p=\frac{s\,p(1-p)}{1+sp}\approx sp(1-p)$. Selection moves frequencies at a rate $\propto s\,p(1-p)$, fastest at intermediate frequency.

### 21.4.2 Fixation probability of a new mutation

Combining drift and selection, the probability that a *new* mutation fixes is (Kimura, 1962; in the haploid convention used in Chapter 5)

$$
u(s)=\frac{1-e^{-2s}}{1-e^{-2Ns}}\ \approx\ \begin{cases}1/N&\text{for }|Ns|\ll1\ \text{(neutral)}\\2s&\text{for }Ns\gg1\ \text{(strongly beneficial)}\\\text{vanishingly small}&\text{for }Ns\ll-1.\end{cases}
$$

Simulations ($N=100$) confirm the formula: for $s=0$, $0.0100$ (theory $0.0100$); $s=+0.02$ ($Ns=+2$), $0.0397$ (theory $0.0399$, about $4\times$ neutral); $s=-0.02$ ($Ns=-2$), $0.0008$ (theory $0.0008$, about $0.08\times$ neutral).

**The "nearly neutral" threshold.** Selection can overcome drift only when $|s|\gtrsim1/N_e$. For humans with $N_e\approx10^4$, mutations with $|s|\lesssim10^{-4}$ behave as if neutral over most of human history; those with $|s|\gtrsim10^{-3}$ are efficiently selected. **Consequences for ML-based constraint inference:**

1. *Conservation and constraint detect selection that is strong relative to $1/N_e$.* Weakly deleterious variation (small $s$) accumulates and contributes to disease risk and to polygenic traits (Chapter 26), but may not look "conserved."
2. *Species differ in $N_e$.* In small-$N_e$ species (many large vertebrates), more mildly deleterious mutations fix; in large-$N_e$ microbes, selection is more efficient. A sequence model trained across the tree of life learns constraint *averaged over these regimes* (Chapter 32).
3. *The Sella–Hirsh relation scales with $N_e$* (Chapter 5): log-likelihood-ratio magnitudes depend on the effective population size of the lineages in the corpus, so absolute $\Delta\log p$ values are not comparable across proteins or species without calibration.

### 21.4.3 Other forces

*Background selection* (purging of deleterious alleles reduces diversity at linked neutral sites), *selective sweeps* (a beneficial allele's rise drags linked variation to fixation, creating a valley of diversity), *balancing selection* (maintains polymorphism, as in HLA and sickle-cell heterozygote advantage), *gene flow/migration*, and *non-random mating*. For ML purposes: **local genome diversity reflects both mutation rate and linked selection**, so sequence variability varies along the genome independently of local function (a confounder for any diversity-based score).

---

## 21.5 Population structure and $F_{ST}$

Two populations that split from a common ancestral population and drift independently diverge in allele frequency. A standard summary is $F_{ST}$, the proportion of total genetic variance attributable to differences between populations: $F_{ST}=\dfrac{\mathrm{Var}(p_i)}{\bar p(1-\bar p)}$ (variance of subpopulation frequencies, relative to the maximum possible for the pooled frequency). For two populations of size $N$ drifting for $t$ generations, heterozygosity within each decays as $e^{-t/2N}$ relative to the pooled expectation, so

$$
F_{ST}\approx1-e^{-t/(2N)}\approx\frac{t}{2N}.
$$

The simulation ($N=500$, $t=50$) gives $F_{ST}=0.0499$ against theory $0.0488$. Human populations have $F_{ST}\approx0.05$–$0.15$ between continental groups; **most human genetic variation (~85–90%) is within populations**, but the *structured* remainder is exactly what dominates the leading principal components (Chapter 2, §2.6).

**PCA and genealogy.** The principal components of a genotype matrix are closely related to average coalescence times between individuals (McVean, 2009): the leading components separate groups whose pairwise coalescence times are long. This is the evolutionary meaning of the "population structure" PC1 in Chapter 2's simulation (Balding–Nichols with $F_{ST}=0.05$).

---

## 21.6 The coalescent

Forward-in-time Wright–Fisher simulation is wasteful when we only want the genealogy of a sample. The **coalescent** (Kingman, 1982) runs time *backward*.

### 21.6.1 Waiting times

Pick two gene copies in the current generation. Their parents are the same copy with probability $1/(2N)$, so the time (in generations) back to their common ancestor is geometric with mean $2N$, approximately exponential with rate $1/(2N)$. In units of $2N$ generations, **the pairwise coalescence time is Exp(1)**. With $k$ lineages there are $\binom k2$ pairs, each coalescing at rate 1, so the waiting time $T_k$ until the next coalescence is Exp$\big(\binom k2\big)$ with mean $\E[T_k]=\dfrac2{k(k-1)}$.

### 21.6.2 Key quantities for a sample of $n$ sequences

- **Time to the most recent common ancestor (TMRCA):** $\E[T_\text{MRCA}]=\sum_{k=2}^n\frac2{k(k-1)}=2\Big(1-\frac1n\Big)$ (telescoping, since $\frac2{k(k-1)}=2(\frac1{k-1}-\frac1k)$), i.e. $4N(1-1/n)$ generations. *Even a huge sample has a TMRCA of at most $4N$ generations; the last coalescence (from 2 lineages to 1) accounts for half of it.*
- **Total tree length:** $\E[L]=\sum_{k=2}^nk\,\E[T_k]=\sum_{k=2}^n\frac2{k-1}=2\sum_{i=1}^{n-1}\frac1i=2a_n$.
- **Segregating sites.** Mutations arrive on the tree as a Poisson process of rate $\theta/2$ per unit time (with $\theta=4N\mu$ for the locus), so $\E[S]=\frac\theta2\E[L]=\theta\,a_n$. This gives **Watterson's estimator** $\hat\theta_W=S/a_n$.
- **The neutral site-frequency spectrum.** A mutation on a branch that subtends $i$ of the $n$ sampled sequences appears in $i$ copies. The expected number of such sites is

$$
\E[\xi_i]=\frac{\theta}{i},\qquad i=1,\dots,n-1.
$$

Rare variants are the most numerous; singletons are $\theta$, doubletons $\theta/2$, and so on.

The code simulates 20,000 coalescent trees with $n=10$, $\theta=4$: mean TMRCA $1.799$ (theory $1.800$), mean tree length $5.658$ (theory $5.658$), mean segregating sites $11.32$ (theory $11.32$), and the spectrum $\{3.99,2.03,1.31,0.82,0.44\}$ at $i=1,2,3,5,9$ against theory $\{4.00,2.00,1.33,0.80,0.44\}$.

### 21.6.3 What the coalescent gives us

- **A null model for variation.** Departures from $\theta/i$ indicate demography or selection. *Tajima's $D$* compares $\hat\theta_W$ with the average pairwise difference $\hat\theta_\pi$; *population expansion* produces an excess of rare variants (negative $D$), *bottlenecks* a deficit.
- **Demographic inference.** PSMC/MSMC and SMC++ recover population-size history from the distribution of coalescence times along a genome (a hidden Markov model over genealogies).
- **Ancestral recombination graphs (ARGs).** With recombination the genealogy changes along the genome; the full object, the ARG, is a latent variable model for the entire history of a sample. Scalable ARG inference (tsinfer, Relate, ARG-needle) is an active area where deep learning is beginning to be applied. [[H]]
- **A view of what "data size" means.** A sample of $n$ sequences has *a single tree* at each locus with $n-1$ coalescence events: additional samples add little deep history (the TMRCA converges to $4N$), they mostly add recent, rare variation. **Sequencing more individuals primarily adds rare variants, not independent deep evolutionary information.**

---

## 21.7 Linkage and linkage disequilibrium

**Linkage disequilibrium (LD)** is the non-random association of alleles at two loci. For alleles A (frequency $p_A$) and B ($p_B$) at nearby loci, $D=p_{AB}-p_Ap_B$, with the normalized measure $r^2=\dfrac{D^2}{p_A(1-p_A)p_B(1-p_B)}$, the squared correlation between the two loci's allele indicators.

**Decay with recombination.** Let $c$ be the recombination fraction between the loci. Each generation a fraction $c$ of haplotypes are recombinant, which breaks the association: $p_{AB}'=(1-c)\,p_{AB}+c\,p_Ap_B$. Subtracting $p_Ap_B$ (which is unchanged under random mating):

$$
D_t=(1-c)^t\,D_0 .
$$

For $c=0.01$, $D_0=0.1$, after 50 generations the code finds $D=0.06050$, matching $D_0(1-c)^{50}=0.06050$. LD decays at rate $c$ per generation, so **close loci stay correlated for long and distant loci decorrelate quickly**.

**Equilibrium LD.** With drift and recombination the expected squared correlation is approximately $\E[r^2]\approx\dfrac{1}{1+4N_ec}$ (Sved, 1971; for $4N_ec\gg1$), so LD extends over $\sim1/(4N_ec)$ and, since the human recombination rate is $\sim10^{-8}$ per bp per generation ($1\,\text{cM/Mb}$), LD blocks span tens of kilobases in non-African populations and are shorter in African populations (with larger long-term $N_e$ and more recombination history).

**Why LD matters to ML.**

1. **Tag SNPs and GWAS.** Any variant is correlated with its neighbors; an association at one variant implicates a *region* (Chapter 26).
2. **Genotype matrices are strongly low-rank locally** (Chapter 2): the correlation structure within an LD block means a handful of haplotypes describe thousands of variants. This is why imputation works (Li–Stephens, Chapter 8) and why fine-mapping is hard.
3. **Leakage.** Variants within an LD block are not independent: a train/test split on variants leaks (Chapters 1, 43); split by chromosome or LD block instead.
4. **Reference/ancestry dependence.** LD patterns differ by population, so tag-based predictors do not transfer (Chapter 26).

---

## 21.8 Molecular evolution: substitution models and phylogenies

### 21.8.1 Substitution as a continuous-time Markov chain

Model the evolution of a nucleotide at one site as a **continuous-time Markov chain** on $\{A,C,G,T\}$ with rate matrix $\mathbf{Q}$ (off-diagonal $Q_{ij}\ge0$ the rate of $i\to j$; rows sum to zero). The transition probabilities over time $t$ are the matrix exponential $\mathbf{P}(t)=e^{\mathbf{Q}t}$. A model is **time-reversible** if $\pi_iQ_{ij}=\pi_jQ_{ji}$ for the stationary distribution $\boldsymbol\pi$ (detailed balance), which makes the likelihood independent of the root position (Felsenstein's pulley principle).

**Jukes–Cantor (JC69).** All substitutions equally likely, total rate $\mu$ per site: $\mathbf{Q}=\frac\mu3(\mathbf{J}-4\mathbf{I})$ with $\mathbf{J}$ the all-ones matrix. Its eigenvalues are $0$ (eigenvector $\mathbf{1}$) and $-\frac{4\mu}3$ (multiplicity 3), so

$$
\mathbf{P}(t)=\tfrac14\mathbf{J}+\big(\mathbf{I}-\tfrac14\mathbf{J}\big)e^{-4\mu t/3},\qquad P_{ii}(t)=\tfrac14+\tfrac34e^{-4\mu t/3},\quad P_{ij}(t)=\tfrac14-\tfrac14e^{-4\mu t/3}.
$$

The observed fraction of differing sites between two sequences diverged for total time $2t$ (or branch length $t$ in a pairwise comparison) is $p=\frac34(1-e^{-4\mu t/3})$. Solving for the true distance gives the **Jukes–Cantor correction**

$$
d=\mu t=-\frac34\ln\Big(1-\frac43p\Big).
$$

It corrects for *multiple hits* (a site mutating more than once, which makes $p$ saturate at $3/4$): in the code, true distances $0.1,0.5,1.0,2.0$ give observed $p=0.094,0.365,0.552,0.698$, and the corrected $d$ recovers $0.100,0.500,1.000,2.000$ exactly. **Saturation is why raw sequence identity stops informing about distance beyond a certain divergence** (the twilight zone of homology detection at $\sim25$–$30\%$ protein identity; Chapter 27).

**Richer models.** *K80 / HKY* distinguish transitions and transversions and unequal base frequencies; *GTR* (general time-reversible) has six exchangeabilities plus base frequencies; *rate heterogeneity across sites* is modeled by a Gamma distribution (some sites evolve fast, many are near-invariant); *codon models* describe substitutions among 61 sense codons and define $\omega=d_N/d_S$, the ratio of nonsynonymous to synonymous substitution rates: $\omega<1$ purifying selection, $\omega\approx1$ neutral, $\omega>1$ positive selection (typically detected for only a few sites or lineages, since most of a protein is constrained).

### 21.8.2 Phylogenies and likelihoods

Given a tree with branch lengths, the likelihood of an alignment is the product over sites of the probability of the observed leaf states, computed by **Felsenstein's pruning algorithm**: dynamic programming from the leaves to the root, summing over unobserved ancestral states at internal nodes (Chapter 8's latent variables; Chapter 6's sum-product on a tree). Tree inference uses distance methods (neighbor joining), maximum likelihood (IQ-TREE, RAxML, FastTree) or Bayesian MCMC (MrBayes, BEAST). **Gene trees differ from species trees** because of incomplete lineage sorting, duplication, loss, and horizontal transfer.

### 21.8.3 Homology vocabulary

*Homologs* share ancestry; *orthologs* diverged by speciation (typically conserve function); *paralogs* diverged by duplication (often diverge in function); *xenologs* arise by horizontal transfer. Modeling assumes homologs are alignable; the choice of which homologs to include (orthologs vs. all) changes what a model learns (Chapters 27, 34).

---

## 21.9 Why evolution is the data-generating process of sequence models

Combine the above. A corpus of sequences is a *sample from the end points of a branching process of mutation, selection, and drift*. Five consequences:

**(1) Non-independence and effective sample size.** Sequences are correlated through shared ancestry. For traits (or sequence features) evolving by Brownian motion on a tree, tips have covariance equal to their shared path length. Felsenstein (1985) showed that $n$ tips supply only $n-1$ independent *contrasts*, weighted by branch length. In the simulation, two traits evolving **independently** on a 40-species coalescent tree (no true relationship) are declared "significantly correlated" at $\alpha=0.05$ in **61%** of trials by naive regression; phylogenetic GLS (whitening by the tree covariance) controls the rate at **5%**. This is **pseudoreplication by descent** (cf. Chapter 4's cells within donors: 86% false positives). *Any benchmark or probe that compares properties across homologous sequences without controlling for phylogeny is at risk of the same inflation.* Practical analogs in ML: **sequence reweighting** (down-weighting sequences by the number of neighbors within an identity threshold; $N_\text{eff}=\sum_i1/|\{j:\mathrm{id}_{ij}>\tau\}|$) in coevolution and EVE-style models; **clustered train/test splits** (Chapter 43).

**(2) Equilibrium and sampling assumptions.** The relation $\Delta\log p\approx2(N-1)\Delta\log f+$ mutation bias assumes the corpus samples a stationary equilibrium. Sequences are instead *correlated samples of lineages that may not have equilibrated* (recent adaptation; bottlenecks; lineage-specific selection) and are *not independent* (point 1). Violations produce systematic score errors that scale with the amount of shared ancestry.

**(3) Phylogenetic confounding in coevolution.** Shared ancestry produces *spurious residue correlations* in an alignment: two positions that differ between two clades appear coupled although they do not interact. Direct-coupling analysis addresses this through sequence reweighting and the average-product correction (Dunn et al., 2008; Chapter 29). A protein language model trained on a raw database *absorbs both signals*; whether it separates them is an open question (Chapters 34, 42).

**(4) Conservation is a *signal* and a *confound*.** Conservation indicates constraint, but a site can be conserved because (i) it is functionally constrained, (ii) the lineage is young, (iii) the mutation rate is low (recall Chapter 20), or (iv) alignments are poor. *Tests of a model against conservation conflate these.*

**(5) Neutral variation is noise relative to function, and signal about history.** Most substitutions between homologs are neutral; the *same* statistical structure (e.g., a codon-usage bias) can reflect history, mutation, or function. A model's "knowledge" of lineage structure (species identity, GC content) is a *shortcut* for many tasks (Chapter 45).

!!! lens "Research lens: a sequence corpus as a phylogenetic sample"
    **Assumes:** that sequences are exchangeable draws from an equilibrium distribution. **Information used:** statistics of co-occurrence across lineages. **Information ignored:** the tree. **Failure modes:** inflated significance and benchmark scores from related sequences across train/test (leakage by descent); clade-specific features mistaken for function; mutation-rate effects mistaken for constraint; ascertainment bias (well-sampled clades dominate). **Remedies:** reweighting, clade-aware splits, explicit tree-aware models (Chapter 42), mutation-rate calibration, and held-out-lineage evaluation.

```python
--8<-- "code/ch21_popgen.py"
```

Output:

```text
drift: one-generation Var(dp) = 0.00125 (theory p(1-p)/2N = 0.00125)
heterozygosity after 50 generations: simulated 0.3896, theory H0 (1-1/2N)^t = 0.3892
neutral allele starting at p0 = 0.1: fixation probability = 0.1001 (theory p0 = 0.1); mean time to fixation (given fixation) = 374 generations (Kimura-Ohta 379)
fixation probability of a new mutation, N = 100 (neutral expectation 1/N = 0.0100):
  s = +0.00 (N s = +0): simulated 0.0100, theory 0.0100
  s = +0.02 (N s = +2): simulated 0.0397, theory 0.0399
  s = -0.02 (N s = -2): simulated 0.0008, theory 0.0008

F_ST after 50 generations at N = 500: simulated 0.0499, theory 1 - exp(-t/2N) = 0.0488

coalescent, n = 10: mean T_MRCA = 1.799 (theory 1.800); mean tree length = 5.658 (theory 5.658); mean segregating sites = 11.32 (theory theta*a_n = 11.32)
  site-frequency spectrum, simulated vs theta/i : i=1: 3.99 vs 4.00, i=2: 2.03 vs 2.00, i=3: 1.31 vs 1.33, i=5: 0.82 vs 0.80, i=9: 0.44 vs 0.44

LD decay: D after 50 generations at c = 0.01: simulated 0.06050, theory D0 (1-c)^t = 0.06050
Jukes-Cantor: P_ii(0.5) from matrix exponential = 0.63506, formula = 0.63506
  true distance mu*t = 0.1: observed fraction different p = 0.094; JC-corrected distance = 0.100
  true distance mu*t = 0.5: observed fraction different p = 0.365; JC-corrected distance = 0.500
  true distance mu*t = 1.0: observed fraction different p = 0.552; JC-corrected distance = 1.000
  true distance mu*t = 2.0: observed fraction different p = 0.698; JC-corrected distance = 2.000

phylogenetic non-independence (40 species, NO true relationship between traits): false-positive rate at 0.05 = 0.61 for naive regression vs 0.05 for phylogenetic GLS
```

---

## 21.10 Worked research examples

!!! example "Worked Research Example 21.1: How many independent sequences does a protein family of 10,000 members contain?"
    **Situation.** You train a family-specific generative model (Chapter 14's DeepSequence/EVE-type VAE) on an MSA of 10,000 sequences and report held-out likelihood and variant-effect performance. A reviewer asks about the effective sample size.

    **Question.** How do you estimate it, and what changes if it is 200?

    **Reasoning.**

    1. *Why 10,000 is the wrong $n$.* Homologs descend from a tree; closely related sequences differ at a few positions. Their information about *what is tolerated* is nearly redundant, like cells from one donor (Chapter 4).
    2. *How to estimate.* (a) **Reweighting:** compute pairwise identity; weight each sequence by $1/|\{j:\mathrm{id}_{ij}\ge\tau\}|$ (e.g., $\tau=0.8$); $N_\text{eff}=\sum_iw_i$. (b) **Phylogenetic:** the number of *independent contrasts*, bounded by the number of tips but typically far fewer when the tree is unbalanced. (c) **Model-based:** compare held-out likelihood as sequences are subsampled by clade.
    3. *What if $N_\text{eff}\approx200$?* Models with thousands of parameters per position (a VAE) are in the $p\gg n$ regime (Chapter 7); estimated couplings have high variance; held-out likelihood random splits (random sequences held out) leak via near-duplicates; reported performance overstates generalization.
    4. *Experiments.* Evaluate with **clade-held-out splits** (cluster at 30–50% identity; hold out entire clusters); report $N_\text{eff}$ with every result; compare reweighted and unweighted training; plot performance against $N_\text{eff}$ across families.
    5. *Predictions.* Variant-effect performance should track $N_\text{eff}$ much more tightly than raw MSA size; clade-held-out likelihood is substantially worse than random-held-out.

    **Expert analysis.** Evolution constrains *how much information a family can contain*: the number of independent evolutionary experiments. The $\sqrt{n}$ scaling of estimation error (Chapter 4) applies to $N_\text{eff}$. This sets a *ceiling on what any family-specific model can know* and suggests why pretraining across families (protein language models) should help: it borrows strength across families (empirical Bayes, Chapter 4).

!!! example "Worked Research Example 21.2: Selection or drift? Interpreting a conserved but weakly constrained site"
    **Situation.** A site in a non-coding region has no substitutions across 100 mammals (looks highly conserved) but a deletion there has no detectable effect in a cell-line assay, while a model trained on conservation assigns it a high score.

    **Reasoning.**

    1. *What does conservation measure?* Whether the *lineage's selection coefficient* at the site was strong enough relative to drift ($|s|\gtrsim1/N_e$) over the evolutionary time spanned (§21.4). It does not measure effect size in a lab assay or in a particular tissue.
    2. *Alternative explanations.* (H1) Small but real fitness effects ($s\sim10^{-3}$) that are invisible in an assay (assays detect large effects in specific contexts). (H2) The function acts in a cell type, developmental stage, or environment not assayed (context). (H3) Low local mutation rate (a mutational coldspot) mimics conservation (Chapter 20). (H4) Alignment artifacts. (H5) Redundancy: another element compensates in the assay but not in nature.
    3. *Experiments.* Compare the site's mutability from a neutral-substitution model; test in additional cell types or in vivo; apply combinatorial perturbations (the element plus candidate redundant partners); use population data (rare-variant depletion at that site relative to expectation, which reflects recent human selection).
    4. *Predictions.* Under H3, the site's *observed/expected* variant count in humans is normal once mutation rate is accounted for; under H1/H2/H5, human constraint is detectable and the effect appears in the right context or in double perturbations.

    **Expert analysis.** An assay's null result and a model's high score are *both* consistent with real selection that operates on scales or contexts the assay cannot see. This is the central difficulty of regulatory-variant interpretation: *the readouts ML models are trained on (conservation, chromatin, expression in a cell line) are proxies for organismal fitness, and each proxy has its own blind spots* (Chapters 31, 41, 50).

---

## 21.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: auditing a benchmark for phylogenetic and population structure"
    **Setting.** You are given a benchmark that evaluates a sequence model by predicting a property across homologous sequences (or species, or individuals).

    **Audit steps.**

    1. **Identify the tree (or tree-like) structure** relating the test items: species tree, protein-family tree, population/pedigree structure, LD blocks.
    2. **Quantify non-independence**: compute pairwise identity or genetic relatedness among test items; estimate an effective sample size.
    3. **Check train–test relatedness**: for each test item, what is its nearest training item's identity or relatedness? Plot performance against it. If performance decays with distance, the headline score is mostly interpolation among relatives.
    4. **Choose the split for the claim**: leave-one-clade-out for "new lineage," leave-one-population-out for "new ancestry," leave-one-chromosome/LD-block-out for "new locus."
    5. **Check confounders aligned with the tree**: GC content, genome size, mutation spectrum, species-specific annotation depth.
    6. **State the null model**: a simple model that exploits only the tree (nearest-relative lookup, phylogenetic mean) and compare.

    **Distinguishing signal types.** Items far from the training set in the tree *and* well predicted indicate transferable function; items close in the tree and well predicted indicate memorization of relatives. The gap is the measurement.

---

## 21.12 Connections

- **Backward:** drift as noise (Chapter 3); pseudoreplication and effective sample size (Chapter 4); mutation–selection equilibrium and information (Chapter 5); Felsenstein pruning and Li–Stephens (Chapter 8); PCA and population structure (Chapter 2); mutation spectrum (Chapter 20).
- **Forward:** statistical genetics (LD, mixed models; Chapter 26); sequence alignment and homology (Chapter 27); direct-coupling analysis and phylogenetic correction (Chapter 29); genomic and protein language models and the evolutionary signal (Chapters 32, 34); tree-aware and ancestral models (Chapter 42); clade-aware benchmarks (Chapter 43); scaling with effective rather than nominal data (Chapter 47).

!!! takeaways "Key takeaways"
    1. **Wright–Fisher drift**: $\Var(\Delta p)=p(1-p)/2N$; heterozygosity decays as $(1-1/2N)^t$; a neutral allele fixes with probability equal to its frequency; $N_e\sim10^4$ for humans.
    2. **Selection vs. drift:** fixation probability $u(s)=(1-e^{-2s})/(1-e^{-2Ns})$; selection acts when $|s|\gtrsim1/N_e$; weakly deleterious variants are "invisible" to conservation.
    3. **Coalescent:** $\E[T_\text{MRCA}]=2(1-1/n)$, $\E[L]=2a_n$, $\E[S]=\theta a_n$, neutral SFS $\E[\xi_i]=\theta/i$ (verified); more samples add rare variants, not deep history.
    4. **LD decays** as $(1-c)^t$ and underlies tag SNPs, imputation, fine-mapping difficulty, and variant-level leakage.
    5. **Substitution models** are continuous-time Markov chains, $\mathbf{P}(t)=e^{\mathbf{Q}t}$; JC69 gives $d=-\tfrac34\ln(1-\tfrac43p)$; saturation limits sequence identity as a distance.
    6. **Phylogenetic non-independence** inflates false positives from 5% to 61% in a 40-species simulation; the number of independent evolutionary experiments is $N_\text{eff}\ll$ nominal sample size.
    7. For sequence models, **evolution is the data-generating process**: it supplies signal (conservation, coevolution) and confounds (shared ancestry, mutation rate, lineage structure).

---

## Further reading

- Hartl, D. L. & Clark, A. G. *Principles of Population Genetics* (4th ed.). Sinauer. Gillespie, J. H. (2004). *Population Genetics: A Concise Guide*. Johns Hopkins University Press.
- Kimura, M. (1962). On the probability of fixation of mutant genes in a population. *Genetics* 47, 713–719. Kimura, M. & Ohta, T. (1969). The average number of generations until fixation of a mutant gene in a finite population. *Genetics* 61, 763–771. Ohta, T. (1973). Slightly deleterious mutant substitutions in evolution. *Nature* 246, 96–98.
- Kingman, J. F. C. (1982). The coalescent. *Stochastic Processes and their Applications* 13, 235–248. Hudson, R. R. (1990). Gene genealogies and the coalescent process. *Oxford Surveys in Evolutionary Biology* 7, 1–44. Wakeley, J. (2008). *Coalescent Theory: An Introduction*. Roberts & Company.
- Li, H. & Durbin, R. (2011). Inference of human population history from individual whole-genome sequences. *Nature* 475, 493–496. (PSMC.) Zhang, B. et al. (2023). Biobank-scale inference of ancestral recombination graphs enables genealogical analysis of complex traits. *Nature Genetics* 55, 768–776. (ARG-needle.)
- McVean, G. (2009). A genealogical interpretation of principal components analysis. *PLoS Genetics* 5, e1000686. Sved, J. A. (1971). Linkage disequilibrium and homozygosity of chromosome segments in finite populations. *Theoretical Population Biology* 2, 125–141.
- Jukes, T. H. & Cantor, C. R. (1969). Evolution of protein molecules. In *Mammalian Protein Metabolism*. Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum likelihood approach. *J. Mol. Evol.* 17, 368–376. Felsenstein, J. (2004). *Inferring Phylogenies*. Sinauer. Yang, Z. (2014). *Molecular Evolution: A Statistical Approach*. Oxford University Press.
- Felsenstein, J. (1985). Phylogenies and the comparative method. *American Naturalist* 125, 1–15. Grafen, A. (1989). The phylogenetic regression. *Phil. Trans. R. Soc. B* 326, 119–157.
- Dunn, S. D., Wahl, L. M. & Gloor, G. B. (2008). Mutual information without the influence of phylogeny or entropy dramatically improves residue contact prediction. *Bioinformatics* 24, 333–340.
- Tajima, F. (1989). Statistical method for testing the neutral mutation hypothesis by DNA polymorphism. *Genetics* 123, 585–595.
