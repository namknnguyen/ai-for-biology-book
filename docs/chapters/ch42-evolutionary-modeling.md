# Chapter 42. Evolutionary Modeling: Trees, Ancestors, Landscapes, and Forecasts

!!! abstract "Chapter at a glance"
    **Motivation.** Every sequence in a biological database is a leaf of a tree. Evolution is the process that produced the data on which genomic and protein language models are trained, the source of the constraints that those models learn, and a scientific object in its own right: ancestors can be reconstructed and resurrected, selection can be detected, and in some systems (influenza, SARS-CoV-2) the next step of evolution can be forecast. This chapter covers the statistical machinery (substitution models, the pruning algorithm, rate heterogeneity), uses controlled simulations to test ancestral reconstruction and to quantify what shared ancestry does to sample sizes, summarizes the state of evolutionary forecasting with its evidence grades, and states the consequences for machine learning: non-independence of sequences, phylogenetic leakage in benchmarks, and the need for clade-aware evaluation. Two results are worth remembering. Reconstructing a root by maximum likelihood is accurate but *biased toward the consensus* (the reconstructed ancestor was more typical than the true one at every depth tested, by $+0.04$ at the shallowest depth and $+0.32$ at the deepest for the best model), and sampling from the posterior removes the bias only under the correct model. And 32 species related by a random coalescent tree carry the information of about 2.3 independent observations about a trait mean, so two independently evolving traits are "significantly correlated" in 56% of trees by ordinary regression.
    **Prerequisites.** Chapters 4, 5, 20, 21, 23, 29, 34, 43, 45.
    **You will be able to:** (1) write a continuous-time Markov substitution model and compute the likelihood of a tree by pruning; (2) explain what ancestral sequence reconstruction does and why maximum-likelihood ancestors are biased; (3) compute the effective sample size implied by a phylogeny and correct a comparative test; (4) say what is and is not known about forecasting evolution; (5) design clade-aware splits and evaluations for machine-learning models of sequences; (6) read evolutionary claims with their rung on the claim ladder.

---

## 42.0 Why a book on AI for biology has an evolution chapter

1. **Evolution made the data.** Natural sequences are not a random sample of possible sequences; they are the survivors of selection, sampled unevenly across a tree. Everything a protein or genomic language model learns (conservation, coevolution, motifs) is a statistic of that process (Chapters 29, 32, 34).
2. **Evolution is a source of labels.** Conservation across species is an unsupervised label for functional constraint (phyloP, GERP); the pattern of substitutions along a tree reveals selection (dN/dS), and human population variation (Chapter 21) reports the strength of purifying selection on every gene (constraint metrics).
3. **Evolution is a modeling target.** Phylogenetics is a statistical inference problem on a latent tree; ancestral reconstruction is a posterior over hidden states; fitness landscapes are models of selection; forecasting is out-of-sample prediction in a nonstationary system.
4. **Evolution breaks the independence assumption** that most of machine learning relies on (Chapters 4, 7, 43, 45). Related sequences are not independent samples, and ignoring that is the commonest hidden leak in biological benchmarks.

!!! lens "Research lens: what information does an evolutionary model use and ignore?"
    *Uses*: sequences at the leaves, a tree (inferred or given), and a substitution model that encodes assumptions about which changes are likely. *Ignores*: the function of the sequences (unless the model includes site-specific constraint), population-level processes (recombination, gene conversion, horizontal transfer, which violate the tree), and the environment. Every conclusion (an ancestor, a selection coefficient, a forecast) is conditional on the *tree and the model*; wrong models give confident wrong answers (§42.2).

---

## 42.1 The statistical machinery

**Substitution as a continuous-time Markov chain.** A site with $A$ states (4 nucleotides, 20 amino acids, 61 codons) changes state at rates given by a matrix $Q$ (off-diagonal $q_{ab}\ge0$, rows sum to zero). Transition probabilities over a branch of length $t$ (expected substitutions per site, or time times rate) are
$$
P(t)=e^{Qt},\qquad P_{ab}(t)=\Pr(\text{state }b\text{ at the end}\mid\text{state }a\text{ at the start}).
$$
A model is *time-reversible* if $\pi_aq_{ab}=\pi_bq_{ba}$ for the stationary distribution $\pi$, which allows the tree to be rooted anywhere for likelihood purposes. Examples: Jukes–Cantor ($q_{ab}=\mu$ for all $a\neq b$; equal frequencies); **F81** ($q_{ab}=\mu\pi_b$), for which
$$
P(t)=e^{-\mu t}I+(1-e^{-\mu t})\,\mathbf 1\pi^\top,
$$
a convex mixture of "no change" and "redraw from $\pi$" (the model of this chapter's simulations); HKY and GTR (nucleotide models with transition/transversion and exchangeability parameters); empirical amino-acid matrices (JTT, WAG, LG); codon models (Goldman–Yang; MG94) with a $\omega=d_N/d_S$ parameter, where $\omega<1$ indicates purifying selection, $\omega\approx1$ neutrality, $\omega>1$ positive selection.

**Rate heterogeneity.** Sites evolve at different rates; modeling the rate multiplier as Gamma-distributed with shape $\alpha$ (small $\alpha$ means strong heterogeneity) is standard. **Site-specific profiles** (the CAT model, profile mixtures) let each site have its own stationary distribution $\pi_i$, which matters because proteins have conserved positions that accept only a few residues; the simulations below use this structure.

!!! math "Derivation: the pruning algorithm"
    For a rooted tree with $N$ leaves and one site, define for each node $v$ and state $a$ the *partial likelihood* $L_v(a)=\Pr(\text{data below }v\mid\text{state at }v=a)$. At a leaf with observed state $x$, $L_v(a)=\mathbf 1[a=x]$. At an internal node with children $c_1,c_2$ and branch lengths $t_1,t_2$,
    $$
    L_v(a)=\prod_{k=1,2}\Big(\sum_{b}P_{ab}(t_k)\,L_{c_k}(b)\Big).
    $$
    At the root, the site likelihood is $\sum_a\pi_aL_{\text{root}}(a)$, and the likelihood of an alignment is the product over sites (or the sum of logs). Each node costs $O(A^2)$ per site, so the whole tree costs $O(NLA^2)$, linear in the number of leaves, instead of the $A^{N-1}$ sum over all internal assignments. The same recursion gives the **marginal posterior of the root state**, $\Pr(\text{root}=a\mid\text{data})\propto\pi_aL_{\text{root}}(a)$, and with a second (downward) pass the posterior at any internal node: the basis of ancestral reconstruction. $\square$

**Tree inference.** Searching tree space is NP-hard; heuristics (maximum parsimony; maximum likelihood with subtree-pruning-and-regrafting search in RAxML and IQ-TREE; Bayesian MCMC in MrBayes and BEAST) are standard. Deep-learning approaches (quartet classifiers, Zou et al., *Mol. Biol. Evol.* 2020; generative flow networks over trees) are active research but have not displaced likelihood methods on real data [[S]]. **Long-branch attraction** (inconsistency of parsimony when two long branches are unrelated) and **model misspecification** are the standard failure modes.

---

## 42.2 Ancestral sequence reconstruction (ASR) and its bias

**What it is.** Given extant sequences, a tree, and a model, ASR computes the posterior over the ancestral sequence at an internal node. The reconstructed sequence can be *resurrected* (synthesized and characterized), which turns an evolutionary hypothesis into an experiment: ancestral steroid receptors, ancestral enzymes with different substrate preferences or thermostability (Thornton and colleagues; Harms and Thornton 2013 is the review). The question for this book is statistical: how accurate is the reconstruction, and what are its systematic errors?

**Simulation.** 32 extant sequences of 300 sites evolve along a random coalescent tree with site-specific preferred distributions $\pi_i$ (Dirichlet with concentration 0.15, i.e., sharply peaked) and Gamma-distributed rates (shape 0.5), under the F81-type process above, with root-to-tip distances $T$ from 0.3 to 3.0 expected substitutions per site. The root is reconstructed by Fitch parsimony; by maximum likelihood (marginal posterior mode) under the Jukes–Cantor-type model, under global amino-acid frequencies, and under the *true* site-specific model (an oracle); by the majority-rule consensus of the extant sequences; and by *sampling* from the posterior. Alongside accuracy, a *typicality score* $s=\frac1L\sum_i\log\pi_i(a_i)$ measures how close a sequence is to the site-wise consensus (a stand-in for stability or fitness: higher means more consensus-like).

```python
--8<-- "code/ch42_ancestral.py"
```

```text
== Reconstructing the root of a 32-leaf tree (300 sites): accuracy, and the typicality score of the reconstruction compared with the TRUE root; mean of 30 trees per row ==
root-to-tip   method                                   accuracy   typicality s   bias in s (reconstruction - true)
     0.3      Fitch parsimony                                       0.885      -1.616         +0.020
     0.3      majority-rule consensus of the extant sequences       0.884      -1.604         +0.032
     0.3      ML, uniform frequencies (JC-type)                     0.905      -1.604         +0.032
     0.3      ML, uniform frequencies (JC-type), sampled from the posterior    0.889      -1.667         -0.031
     0.3      ML, global amino-acid frequencies                     0.906      -1.604         +0.032
     0.3      ML, global amino-acid frequencies, sampled from the posterior    0.885      -1.685         -0.049
     0.3      ML, true site-specific model (oracle)                 0.905      -1.592         +0.044
     0.3      ML, true site-specific model (oracle), sampled from the posterior    0.876      -1.632         +0.004
             (typicality of the TRUE root: -1.636)
     0.8      Fitch parsimony                                       0.778      -1.586         +0.047
     0.8      majority-rule consensus of the extant sequences       0.784      -1.575         +0.057
     0.8      ML, uniform frequencies (JC-type)                     0.809      -1.556         +0.076
     0.8      ML, uniform frequencies (JC-type), sampled from the posterior    0.762      -1.830         -0.198
     0.8      ML, global amino-acid frequencies                     0.810      -1.558         +0.075
     0.8      ML, global amino-acid frequencies, sampled from the posterior    0.761      -1.848         -0.216
     0.8      ML, true site-specific model (oracle)                 0.815      -1.504         +0.128
     0.8      ML, true site-specific model (oracle), sampled from the posterior    0.762      -1.636         -0.003
             (typicality of the TRUE root: -1.633)
     1.5      Fitch parsimony                                       0.699      -1.524         +0.090
     1.5      majority-rule consensus of the extant sequences       0.705      -1.509         +0.104
     1.5      ML, uniform frequencies (JC-type)                     0.731      -1.485         +0.129
     1.5      ML, uniform frequencies (JC-type), sampled from the posterior    0.615      -2.320         -0.706
     1.5      ML, global amino-acid frequencies                     0.732      -1.484         +0.130
     1.5      ML, global amino-acid frequencies, sampled from the posterior    0.614      -2.323         -0.709
     1.5      ML, true site-specific model (oracle)                 0.744      -1.394         +0.220
     1.5      ML, true site-specific model (oracle), sampled from the posterior    0.668      -1.611         +0.003
             (typicality of the TRUE root: -1.614)
     3.0      Fitch parsimony                                       0.611      -1.459         +0.155
     3.0      majority-rule consensus of the extant sequences       0.620      -1.434         +0.179
     3.0      ML, uniform frequencies (JC-type)                     0.646      -1.403         +0.211
     3.0      ML, uniform frequencies (JC-type), sampled from the posterior    0.339      -3.848         -2.234
     3.0      ML, global amino-acid frequencies                     0.647      -1.400         +0.213
     3.0      ML, global amino-acid frequencies, sampled from the posterior    0.338      -3.859         -2.246
     3.0      ML, true site-specific model (oracle)                 0.666      -1.295         +0.319
     3.0      ML, true site-specific model (oracle), sampled from the posterior    0.568      -1.620         -0.006
             (typicality of the TRUE root: -1.614)
```

**Reading Part 1 (reconstruction).**

1. **Accuracy declines with depth, and the model matters less than the data.** Oracle maximum-likelihood accuracy falls from 0.905 at $T=0.3$ to 0.815 (0.8), 0.744 (1.5), and 0.666 (3.0). The oracle's advantage over the Jukes–Cantor-type model is 0 to 2 percentage points at every depth (0.666 against 0.646 at $T=3$); parsimony (0.611) and majority-rule consensus (0.620) are within 2 to 4 points of maximum likelihood. A *better model buys little accuracy* in this regime; a *shallower node* buys a lot.
2. **Maximum-likelihood reconstructions are biased toward the consensus.** The typicality of the reconstructed root exceeds that of the true root at every depth and every method that returns the posterior mode: $+0.04$ (0.3) rising to $+0.32$ (3.0) for the oracle model, and $+0.03$ to $+0.21$ for the Jukes–Cantor-type model. The mode picks the most probable residue at each site; true ancestors sometimes carried the less probable one. The reconstructed ancestor is therefore *more typical, more consensus-like, and, under a typicality-stability correspondence, more stable* than the real one: the mechanism behind the long-noted finding that resurrected maximum-likelihood ancestors tend to appear unusually thermostable (Williams et al. 2006; Eick et al. 2017).
3. **Posterior sampling removes the bias only under the correct model.** Sampling each site from the oracle posterior gives a typicality bias of $+0.004$, $-0.003$, $+0.003$, $-0.006$ (zero within noise) at all depths, at the price of lower accuracy (0.568 at $T=3$). Sampling from the *wrong* model's posterior (Jukes–Cantor-type or global frequencies) is strongly *negatively* biased: at $T=3$ the sampled sequences have typicality $-3.85$ against $-1.61$ for the true root, because the model does not know which residues each site tolerates and fills in implausible ones. **The two fixes have opposite failure modes: the mode is too typical; wrong-model samples are too atypical.**
4. **What to do.** Report an *ensemble* of ancestors sampled from the posterior and characterize several, rather than one maximum-likelihood sequence; test the *robustness of the functional conclusion* to the uncertain sites (Eick et al. 2017); use site-specific models where possible; reconstruct at depths where accuracy supports the claim.

!!! rhyme "Structural rhyme: maximum-likelihood ancestor ↔ the consensus bias of any posterior-mode estimate ↔ shrinkage toward the prior mean (Chapter 4)"
    The posterior mode of each site is the most probable value; the true joint draw is, on average, less probable per site. The same happens in image restoration (denoised images look smoother than real ones), in denoised single-cell expression (Chapter 30: false correlations), and in a language model's greedy decoding (more typical text than natural text). Whenever a single "best" reconstruction is reported, ask what it systematically removes.

**Modern routes.** Protein language models and Potts models are *learned* site-specific, context-dependent models of the stationary distribution, so they can in principle supply better priors than empirical matrices; coevolution (epistasis) means that the state at one site constrains others, which matters for ancestors because compensating changes are correlated along branches (Chapters 23, 29). Whether a pLM-based reconstruction improves accuracy and bias on benchmarks with known ancestors (simulations with epistasis, or laboratory-evolved lineages with sequenced ancestors) is a live research question [[H]].

---

## 42.3 Shared ancestry: effective sample size and false correlations

Related species are not independent draws. If a trait evolves by Brownian motion on a tree, the covariance between two species is the length of the path they share from the root, $C_{ij}$, and the variance of the sample mean of $N$ species is $\mathbf 1^\top C\mathbf 1/N^2$, so the **effective number of independent observations** relative to independent draws with the same variance $C_{ii}=T$ is
$$
N_\text{eff}=\frac{N^2\,T}{\mathbf 1^\top C\,\mathbf 1}.
$$
For a star tree ($C=TI$) this is $N$; for a deep split into two clades it approaches 2. **Felsenstein (1985)** showed that comparative analyses that treat species as independent have inflated false-positive rates; the remedies are phylogenetic independent contrasts or **phylogenetic generalized least squares** (PGLS), which fits $y=X\beta+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2C)$ by generalized least squares (the same likelihood as a linear mixed model with a kinship matrix, Chapter 26, where the "relatedness" is the shared path length).

The second part of the script uses 32 species on 3,000 random coalescent trees, tests the correlation between two traits that evolved *independently*, and computes $N_\text{eff}$ for the mean.

```text
== 2. Shared ancestry makes 32 species far fewer than 32 independent observations ==
effective number of independent observations for estimating a trait mean, from 32 species on a random coalescent tree: mean 2.31 (median 2.26, 90% range 1.44-3.39)
two traits that evolved independently (no true association), across 32 species, 3000 random trees: nominal 5% test rejects in 0.559 of trees by ordinary least squares, in 0.058 with phylogenetic generalized least squares
```

**Reading Part 2.** The effective number of independent observations is about 2.3 (90% range 1.4 to 3.4) for 32 species on a coalescent tree, because the deepest split divides most species into two clades whose members share most of their history. Two traits with no association are declared correlated at the nominal 5% level in **56%** of trees by ordinary least squares, and in 5.8% with PGLS. Two consequences for machine learning on biological sequences: (i) a data set of $N$ sequences related by a tree has an *effective* size that can be a small fraction of $N$ (Chapter 29's $N_\text{eff}$ for coevolution; Chapter 34's family structure); (ii) *random train/test splits leak* because test sequences have close relatives in training, so reported accuracy measures how well the model interpolates within clades; the remedy is a **clade-held-out split** (the evolutionary analog of the cluster split of Chapter 24 and the group split of Chapter 43).

!!! lens "Research lens: assumptions of the simulations"
    A coalescent tree (all branches proportional to coalescent times), an F81-type model with independent sites (no epistasis), a single tree for all sites (no recombination), Gamma-distributed rates, a known tree (real analyses infer it), Brownian trait evolution. Real data violate each of these, in ways that usually make ASR *less* accurate and dependence *more* severe.

---

## 42.4 Fitness landscapes, selection, and forecasting

**Selection from sequences.** dN/dS and its relatives (site models, branch-site models, McDonald–Kreitman tests on within- versus between-species variation) detect selection when the signal is strong; they have low power for weak selection and are confounded by recombination, mutation bias, and biased gene conversion (Chapters 20, 21). Constraint metrics from human population data (the observed/expected loss-of-function ratio, LOEUF; *shet*, selection coefficients for heterozygous loss of function) estimate selection on genes directly and are among the most informative gene-level priors for variant interpretation (Chapter 41).

**Fitness landscapes.** A fitness landscape maps sequences to fitness; experimental data come from deep mutational scans (Chapter 23). The shape (additive, pairwise epistatic, globally epistatic through a nonlinearity) governs evolutionary predictability: with strong epistasis, the same mutation is beneficial on one background and deleterious on another, evolution depends on history, and forecasting from one background fails to transfer to another (Chapter 23's global epistasis experiment; Chapter 29's Potts models).

**Forecasting viral evolution.** [[S]] for *retrospective* and *short-horizon* claims, [[H]] for long-horizon prospective forecasts.

- *Influenza.* A fitness model of antigenic clade frequencies, combining antigenic drift and the accumulation of deleterious mutations, predicted the dominant clade of the next season better than chance (Łuksza and Lässig, *Nature* 2014); work on the shape of genealogical trees provides complementary predictors (Neher, Russell and Shraiman, *eLife* 2014).
- *SARS-CoV-2.* A hierarchical Bayesian regression on millions of genomes estimated the relative fitness of mutations (Obermeyer et al., *Science* 2022); deep mutational scans of the receptor-binding domain (Starr et al., *Cell* 2020) measured effects on ACE2 binding and antibody escape; protein-language-model-based scores (Hie et al., *Science* 2021) and EVEscape (Thadani et al., *Nature* 2023, combining a deep generative model of fitness with structural accessibility and chemical dissimilarity of mutations) predicted which mutations would arise and escape antibodies using only pre-pandemic data. The honest summary: *these methods enrich for mutations that later appeared*, evaluated retrospectively and on data the authors define; forecasting *which lineage will dominate* months ahead is dominated by epidemiology (transmission, immunity, recombination), not only by sequence-based fitness, and prospective performance with pre-registered horizons is rarely reported.

!!! openproblem "Open problem: prospective, pre-registered evolutionary forecasting"
    Most evaluations of forecasting are retrospective with data available to the authors. **Diagnostic questions:** Could one run a *community forecasting benchmark* with a fixed horizon (for example the identity of the dominant influenza H3N2 clade at the next vaccine-composition meeting, or the mutation frequencies in a defined SARS-CoV-2 spike region at 6 months) with forecasts registered before data arrive, and with baselines (persistence, the dominant lineage; a model using only recent growth rates) scored by proper scoring rules? Which part of forecast skill comes from sequence-based fitness and which from epidemiological dynamics, and can a joint model separate them?

---

## 42.5 What this means for machine-learning models of biological sequence

1. **Splits.** For any sequence benchmark, split by clade (e.g., cluster at 30% sequence identity for proteins; hold out an entire family, order, or species for genomes) and report the performance gap between random and clade-held-out splits as a measure of leakage (Chapter 43).
2. **Weights.** Reweight by cluster (the 80%-identity reweighting of DCA, Chapter 29; effective sample size) when estimating statistics, and report $N_\text{eff}$.
3. **Confounding by phylogeny.** A feature correlated with clade membership (GC content, taxon-specific motifs) can predict a label that is itself clade-structured; use the shortcut tests of Chapter 45 with clade as the nuisance variable.
4. **Training data are a tree-structured sample.** Language models trained on UniRef or genome corpora inherit the sampling biases of the tree of life (Chapter 34: over-representation of model organisms and pathogens); likelihoods are not comparable across taxa without calibration.
5. **Ancestral and evolutionary augmentation.** Ancestral sequences and phylogenetic sampling can augment data, but biased (consensus-like) ancestors (§42.2) are not representative of natural variation.

---

## 42.6 Worked research examples

!!! example "Worked Research Example 42.1: A resurrected ancestral enzyme is 18 °C more thermostable than its descendants"
    **Situation.** A paper reconstructs the maximum-likelihood ancestor of a family of enzymes at a deep node, characterizes it, and reports an apparent melting temperature 18 °C above that of modern enzymes. They conclude that the ancestor lived in a hot environment.

    **Question.** What would you test before accepting the paleoenvironmental inference?

    **Reasoning (Expert Chain).**

    1. **L1 Claim.** Ancestral enzyme stability implies ancestral habitat temperature.
    2. **L3–L5 Assumptions and failure modes.** (a) *Reconstruction bias*: the maximum-likelihood ancestor is biased toward consensus residues and therefore toward stability (§42.2); a bias of this type could produce part of the effect. (b) *Uncertainty*: posterior probabilities at ambiguous sites are not reflected in a single sequence. (c) *Model misspecification*: the substitution model and tree (including gene duplication and horizontal transfer events) may be wrong. (d) *Inference chain*: stability of the enzyme in vitro depends on the assay (pH, ions, ligand) and on the host's physiology; modern thermophiles have thermostable enzymes, but enzyme stability is only one determinant of growth temperature. (e) *Single tree*.
    3. **L10 Experiments.** (i) Sample 20 ancestors from the posterior (not only the mode) and characterize a subset including the *most atypical plausible* ones (the AltAll approach of Eick et al.: the second most probable state at each ambiguous site), to see whether the stability conclusion is robust. (ii) Reconstruct with several models and trees (site-specific profile model; alternative topologies) and report the range. (iii) *Simulation-based calibration*: simulate sequence evolution on the inferred tree under the fitted model with a known stability trait (using a stability-predicting model for the fitness), reconstruct, and measure the bias in stability of ML ancestors in that setting. (iv) Compare with *multiple deep ancestors* across the family: a trend of increasing stability with depth in many lineages versus one node. (v) Independent lines of evidence for temperature: isotopic geochemistry or the nucleotide composition of ancestral rRNA.
    4. **L11 Interpretation.** If the stability effect survives the ensemble and the model changes, it is a robust property of the ancestral *sequence family*; if the effect shrinks with posterior sampling, part was reconstruction bias. The habitat inference remains a separate, weaker claim (C1 for the enzyme's stability, not C3 for the environment).

    **Expert analysis.** The cheapest decisive step is to *characterize the ensemble*. A single maximum-likelihood ancestor reports the consensus; the claim is about the family.

!!! example "Worked Research Example 42.2: A protein-LM benchmark with random splits across homologous families. No known answer on the size of the leak"
    **Situation.** A group builds a benchmark of 50,000 labeled proteins from 400 families and trains models to predict a functional label. They report accuracy of 92% with random 80/20 splits and say it shows generalization. You suspect phylogenetic leakage but have no known answer for how much.

    **Question.** How would you measure the leak and the true generalization?

    **Reasoning.**

    1. **Define the target generalization** (to new members of a seen family; to a new family): the two have different splits.
    2. **Leak measurement.** Compute, for each test protein, its maximum sequence identity to the training set under the random split; plot accuracy versus identity bins. If accuracy is 98% at identity above 90% and 70% below 30%, the headline number is dominated by near-duplicates.
    3. **Alternative splits.** (a) Cluster at 30% identity (the clade-held-out split) and (b) hold out whole families. Report the accuracy under each, with the *fraction of the test set that has a training homolog above 30%* as an additional descriptor.
    4. **Estimate the effective sample size** with identity-based reweighting; compare the training curves under random versus clade-held-out splits (Chapter 47): a model whose clade-held-out learning curve is flat is memorizing families.
    5. **Baselines.** A nearest-neighbor (BLAST-like) classifier that copies the label of the most similar training protein; if it matches the model under the random split, the model has learned only similarity.
    6. **Decision.** Report the generalization as the accuracy under the split matching the deployment scenario, and the nearest-neighbor baseline gap.

    **What is not known.** The size of the leak for this data set and label; the diagnostic experiment is the answer. In published benchmarks (Chapter 43), gaps between random and homology-controlled splits of 10 to 40 percentage points are common.

---

## 42.7 Researcher's Notebook

!!! notebook "Researcher's Notebook: evolutionary hygiene for sequence models"
    1. **Draw or infer the tree** (or at least a cluster dendrogram) for your data before splitting; compute $N_\text{eff}$.
    2. **Use clade-held-out splits** and report the gap to random splits.
    3. **Reweight** sequences by redundancy when estimating family statistics.
    4. **Use PGLS or mixed models** when testing associations across species.
    5. **Report ensembles of ancestors**, not a single maximum-likelihood sequence, and test whether conclusions hold across the ensemble.
    6. **Calibrate any evolutionary inference by simulation** with a known truth under the same tree and model.
    7. **Re-run the simulations** with epistasis (a Potts model, Chapter 29) in place of site-independent profiles and see how accuracy and bias change.

    **What it teaches.** The tree is a hidden variable in every comparative analysis. Ignoring it silently reduces sample sizes and inflates confidence; modeling it explicitly is routine in evolutionary biology and rare in machine learning.

    **An open question to carry forward.** The simulations show that the maximum-likelihood root is more typical than the true root, and that wrong-model posterior sampling is less typical. For a language-model-based ancestral reconstruction, could one *calibrate the typicality of sampled ancestors* against held-out simulations or experimentally characterized ancestors, and use the calibrated temperature of the sampler (Chapter 36's $\beta$) to produce ensembles that are neither too consensus-like nor too noisy?

---

## 42.8 Connections

- **Backward:** pseudoreplication and mixed models (Chapters 4, 26); information and conservation (Chapter 5); mutation bias, drift, and selection (Chapters 20, 21); global epistasis and fitness landscapes (Chapter 23); Potts models and $N_\text{eff}$ (Chapter 29); protein LMs and database sampling (Chapter 34); benchmark leakage and splits (Chapters 43, 45).
- **Forward:** scaling and data diversity (Chapter 47); open problems in genomes and proteins (Chapters 50, 52); evolutionary forecasting as an AI-scientist target (Chapter 54).

!!! takeaways "Key takeaways"
    1. A phylogenetic likelihood is computed by pruning in $O(NLA^2)$; the same recursion gives the posterior of ancestral states.
    2. Ancestral reconstruction accuracy depends mainly on depth (0.905 to 0.666 for the oracle model from $T=0.3$ to 3.0), not on the substitution model (an oracle site-specific model added only 0–2 points over a Jukes–Cantor-type model).
    3. **Maximum-likelihood ancestors are biased toward the consensus** (typicality bias $+0.04$ to $+0.32$), so resurrected ancestors look unusually stable; sampling from the *correct* posterior is unbiased, whereas sampling from a wrong model gives atypical sequences ($-2.2$ at the deepest node). Report ensembles.
    4. **Thirty-two species on a coalescent tree carry about 2.3 independent observations**; two independently evolving traits were "significantly correlated" in 56% of trees by ordinary regression versus 5.8% with PGLS.
    5. Evolutionary forecasting enriches for later-observed mutations in retrospective tests (influenza clades; SARS-CoV-2 escape) [[S]]; prospective, pre-registered forecasting is rare [[H]].
    6. For machine learning, treat the tree as a hidden variable: clade-held-out splits, redundancy weighting, $N_\text{eff}$, and nearest-neighbor baselines; evaluate in the split that matches deployment.

---

## Further reading

- Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum likelihood approach. *J. Mol. Evol.* 17, 368–376. Felsenstein, J. (1985). Phylogenies and the comparative method. *Am. Nat.* 125, 1–15. Yang, Z. (2014). *Molecular Evolution: A Statistical Approach.* Oxford University Press. Minh, B. Q. et al. (2020). IQ-TREE 2: new models and efficient methods for phylogenetic inference in the genomic era. *Mol. Biol. Evol.* 37, 1530–1534. Suchard, M. A. et al. (2018). Bayesian phylogenetic and phylodynamic data integration using BEAST 1.10. *Virus Evol.* 4, vey016.
- Williams, P. D., Pollock, D. D., Blackburne, B. P. & Goldstein, R. A. (2006). Assessing the accuracy of ancestral protein reconstruction methods. *PLoS Comput. Biol.* 2, e69. Eick, G. N., Bridgham, J. T., Anderson, D. P., Harms, M. J. & Thornton, J. W. (2017). Robustness of reconstructed ancestral protein functions to statistical uncertainty. *Mol. Biol. Evol.* 34, 247–261. Harms, M. J. & Thornton, J. W. (2013). Evolutionary biochemistry: revealing the historical and physical causes of protein properties. *Nat. Rev. Genet.* 14, 559–571.
- Zou, Z., Zhang, H., Guan, Y. & Zhang, J. (2020). Deep residual neural networks resolve quartet molecular phylogenies. *Mol. Biol. Evol.* 37, 1495–1507. Grafen, A. (1989). The phylogenetic regression. *Phil. Trans. R. Soc. B* 326, 119–157. Martins, E. P. & Hansen, T. F. (1997). Phylogenies and the comparative method: a general approach to incorporating phylogenetic information into the analysis of interspecific data. *Am. Nat.* 149, 646–667.
- Łuksza, M. & Lässig, M. (2014). A predictive fitness model for influenza. *Nature* 507, 57–61. Neher, R. A., Russell, C. A. & Shraiman, B. I. (2014). Predicting evolution from the shape of genealogical trees. *eLife* 3, e03568. Obermeyer, F. et al. (2022). Analysis of 6.4 million SARS-CoV-2 genomes identifies mutations associated with fitness. *Science* 376, 1327–1332. Starr, T. N. et al. (2020). Deep mutational scanning of SARS-CoV-2 receptor binding domain reveals constraints on folding and ACE2 binding. *Cell* 182, 1295–1310. Hie, B., Zhong, E. D., Berger, B. & Bryson, B. (2021). Learning the language of viral evolution and escape. *Science* 371, 284–288. Thadani, N. N. et al. (2023). Learning from prepandemic data to forecast viral escape. *Nature* 622, 818–825.
