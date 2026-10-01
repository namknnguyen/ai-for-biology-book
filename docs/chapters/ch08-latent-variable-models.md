# Chapter 8. Latent-Variable Models and Probabilistic Inference

!!! abstract "Chapter at a glance"
    **Motivation.** The biology that matters (cell state, ancestry, regulatory program, ancestral sequence, the identity of the haplotype you inherited) is rarely observed. It is *latent*. Latent-variable models are the formal language for reasoning about hidden causes of observed data, and their inference machinery (EM, the ELBO, variational inference) underlies VAEs, diffusion models, and single-cell models later in the book.
    **Prerequisites.** Chapters 4–5 (likelihood, KL), Chapter 6 (HMM forward algorithm).
    **You will be able to:** (1) write a generative story with latent variables and read a plate diagram; (2) derive the ELBO and the EM algorithm and prove EM increases the likelihood; (3) derive the reparameterization gradient; (4) read conditional independence from a graphical model, including Gaussian precision matrices; (5) explain why latent axes are generally *not identifiable* and what that implies for biological interpretation; (6) recognize latent-variable structure in phylogenetics, haplotype imputation, and single-cell models.

---

## 8.1 Why latent variables?

Chapter 1 described a cell as a latent state $\mathbf{s}$ observed through assays. A *latent-variable model* makes that explicit: it posits hidden variables $z$ that generate observations $x$ through a probabilistic mechanism $p(x\mid z)$, with a prior $p(z)$:

$$
p(x)=\int p(x\mid z)\,p(z)\,dz\qquad\text{(or }\textstyle\sum_z\text{ for discrete }z).
$$

This one equation covers an astonishing range of methods:

| Model | Latent $z$ | Observed $x$ | Chapter |
|---|---|---|---|
| Gaussian mixture / $k$-means | Cluster identity (discrete) | Feature vector | 8 |
| Factor analysis / probabilistic PCA | Low-dimensional factors (continuous) | High-dimensional vector | 2, 8 |
| HMM | State path (discrete chain) | Sequence | 6, 27 |
| Topic model (LDA) | Mixture over "programs" | Counts | 8, 30 |
| scVI (single-cell VAE) | Cell state $z$ and library size | Counts | 30 |
| VAE | Learned continuous code | Image, sequence, expression | 14 |
| Diffusion model | Noised versions $x_1,\dots,x_T$ | Clean data $x_0$ | 15 |
| Phylogenetic model | Ancestral sequences, tree | Extant sequences | 21, 42 |
| Li–Stephens haplotype model | Which reference haplotype is copied | Observed genotypes | 8, 26 |

The computational difficulty is common to all: the posterior $p(z\mid x)$ and the marginal likelihood $p(x)$ require an integral over $z$ that is usually intractable. This chapter develops the standard approximations.

---

## 8.2 A concrete model: the Gaussian mixture

Suppose cells fall into $K$ discrete types. The generative story: draw a type $z\sim\mathrm{Categorical}(\boldsymbol\pi)$, then draw features $x\mid z=k\sim\Normal(\boldsymbol\mu_k,\boldsymbol\Sigma_k)$. The marginal likelihood is $p(x)=\sum_k\pi_k\Normal(x;\boldsymbol\mu_k,\boldsymbol\Sigma_k)$. If we *knew* $z_i$ for each point, maximum likelihood would be trivial (class-wise sample means and covariances). With $z$ latent, the log-likelihood $\sum_i\log\sum_k\pi_k\Normal(x_i;\ldots)$ has a sum inside the log and no closed-form maximizer.

**Posterior over the latent.** By Bayes' rule, the *responsibility* of component $k$ for point $x_i$ is

$$
r_{ik}=p(z_i=k\mid x_i)=\frac{\pi_k\Normal(x_i;\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_j\pi_j\Normal(x_i;\boldsymbol\mu_j,\boldsymbol\Sigma_j)}.
$$

---

## 8.3 The ELBO and the EM algorithm, derived

### 8.3.1 The evidence lower bound

For any distribution $q(z)$ over the latent variables,

$$
\log p(x\mid\theta)=\log\int p(x,z\mid\theta)\,dz=\log\int q(z)\frac{p(x,z\mid\theta)}{q(z)}dz\ \ge\ \int q(z)\log\frac{p(x,z\mid\theta)}{q(z)}dz\ =:\ \mathcal{L}(q,\theta),
$$

by Jensen's inequality ($\log$ is concave). $\mathcal{L}$ is the **evidence lower bound (ELBO)**. We can compute exactly how loose the bound is. Using $p(x,z\mid\theta)=p(z\mid x,\theta)p(x\mid\theta)$:

$$
\mathcal{L}(q,\theta)=\E_q\Big[\log\frac{p(z\mid x,\theta)\,p(x\mid\theta)}{q(z)}\Big]=\log p(x\mid\theta)-\KL{q(z)}{p(z\mid x,\theta)}.
$$

Rearranged, the **fundamental identity of variational inference**:

$$
\boxed{\log p(x\mid\theta)=\mathcal{L}(q,\theta)+\KL{q(z)}{p(z\mid x,\theta)}.}
$$

Since $\KL\ge0$ (Chapter 5), $\mathcal{L}\le\log p(x)$, with equality iff $q$ equals the true posterior. Maximizing the ELBO over $q$ therefore *minimizes the reverse KL to the posterior* while *raising a lower bound on the evidence*. The ELBO has two other useful forms:

$$
\mathcal{L}(q,\theta)=\underbrace{\E_q[\log p(x\mid z,\theta)]}_{\text{reconstruction / data fit}}-\underbrace{\KL{q(z)}{p(z\mid\theta)}}_{\text{complexity penalty}}
=\underbrace{\E_q[\log p(x,z\mid\theta)]}_{\text{expected complete-data log-lik.}}+\underbrace{\Ent(q)}_{\text{entropy}} .
$$

### 8.3.2 EM

**Expectation–maximization** alternates between maximizing $\mathcal{L}$ over $q$ and over $\theta$.

- **E-step.** Fix $\theta^{(t)}$; set $q^{(t+1)}(z)=p(z\mid x,\theta^{(t)})$. This makes the KL term zero, so the bound is tight: $\mathcal{L}(q^{(t+1)},\theta^{(t)})=\log p(x\mid\theta^{(t)})$.
- **M-step.** Fix $q^{(t+1)}$; set $\theta^{(t+1)}=\arg\max_\theta\E_{q^{(t+1)}}[\log p(x,z\mid\theta)]$ (the entropy term does not depend on $\theta$).

**EM never decreases the likelihood.** $\log p(x\mid\theta^{(t+1)})\ge\mathcal{L}(q^{(t+1)},\theta^{(t+1)})\ge\mathcal{L}(q^{(t+1)},\theta^{(t)})=\log p(x\mid\theta^{(t)})$: the first inequality is the ELBO bound, the second is the M-step's maximization, and the equality is the tightness from the E-step. $\square$ EM converges to a stationary point of the likelihood, which may be a local maximum or saddle: **initialization matters**, and one should run multiple restarts.

**GMM updates.** The E-step computes responsibilities $r_{ik}$ (above). The M-step maximizes $\sum_i\sum_kr_{ik}[\log\pi_k+\log\Normal(x_i;\boldsymbol\mu_k,\boldsymbol\Sigma_k)]$, giving, with $N_k=\sum_ir_{ik}$,

$$
\pi_k=\frac{N_k}{N},\qquad
\boldsymbol\mu_k=\frac1{N_k}\sum_ir_{ik}x_i,\qquad
\boldsymbol\Sigma_k=\frac1{N_k}\sum_ir_{ik}(x_i-\boldsymbol\mu_k)(x_i-\boldsymbol\mu_k)^\top .
$$

(Responsibility-weighted versions of the ordinary MLEs.) $k$-means is the limit of EM with $\boldsymbol\Sigma_k=\sigma^2\mathbf{I}$, $\sigma\to0$, where responsibilities become hard assignments.

The code (`code/ch08_latent_variables.py`) implements EM from scratch on 900 points from three Gaussians: the log-likelihood rises from $-4319$ after one iteration to $-3107.3$ and never decreases, matching scikit-learn's best-of-five fit ($-3107.3$) and recovering the three means $(0,0)$, $(2.1,3.4)$, $(4.0,0)$ to within 0.1.

!!! rhyme "Structural rhyme: ELBO ↔ variational free energy ↔ bits-back coding ↔ Bayesian evidence"
    Writing $-\mathcal{L}=\E_q[-\log p(x,z)]-\Ent(q)=\langle\text{energy}\rangle-\text{entropy}$ is exactly the **Helmholtz free energy** $F=U-TS$ with $T=1$. Minimizing the variational free energy over $q$ finds the Boltzmann-like distribution $q\propto e^{-E}$ that best trades energy against entropy; at the optimum $-\mathcal{L}=-\log p(x)$. The same quantity is a **code length**: the *bits-back* argument says a latent-variable model can encode $x$ in $-\mathcal{L}$ nats on average. It also appears as the **marginal likelihood** that automatically penalizes complexity. Protein folding landscapes (Chapter 23), Potts models (Chapter 29), score-based diffusion (Chapter 15), and variational inference all use the same free-energy structure: a distribution trading off energy and entropy.

---

## 8.4 Variational inference and the reparameterization trick

When the posterior $p(z\mid x)$ cannot be computed (neural-network likelihoods), we pick a tractable family $q_\phi(z\mid x)$, parameterized by $\phi$, and maximize the ELBO over $\phi$ and $\theta$.

**Mean-field** variational inference assumes $q(z)=\prod_jq_j(z_j)$, trading accuracy for tractability (it underestimates posterior variance and ignores correlations; recall that reverse KL is mode-seeking, Chapter 5). **Amortized inference** uses a neural network (an *encoder*) to output $q_\phi(z\mid x)$ for every $x$, so no per-datapoint optimization is needed: this is the VAE (Chapter 14).

### 8.4.1 Gradients through sampling

The ELBO contains an expectation $\E_{q_\phi(z)}[f(z)]$ whose gradient with respect to $\phi$ we need.

**Score-function estimator.** $\nabla_\phi\E_{q_\phi}[f(z)]=\E_{q_\phi}[f(z)\nabla_\phi\log q_\phi(z)]$ (using $\nabla q=q\nabla\log q$). It works for any $f$ (even discrete $z$) but has *high variance*.

**Reparameterization.** If a sample can be written as $z=g_\phi(\epsilon)$ with $\epsilon\sim p(\epsilon)$ independent of $\phi$ (for a Gaussian, $z=\mu+\sigma\epsilon$, $\epsilon\sim\Normal(0,1)$), then $\E_{q_\phi}[f(z)]=\E_{p(\epsilon)}[f(g_\phi(\epsilon))]$ and the gradient moves *inside* the expectation:

$$
\nabla_\phi\E_{q_\phi}[f(z)]=\E_{p(\epsilon)}\big[\nabla_zf(z)\big|_{z=g_\phi(\epsilon)}\,\nabla_\phi g_\phi(\epsilon)\big].
$$

For $f(z)=z^2$ and $q=\Normal(\mu,1)$ the true gradient with respect to $\mu$ is $2\mu$. Numerically ($\mu=1.5$), the reparameterized estimator has mean 2.998 and standard deviation 2.00; the score-function estimator has mean 2.985 and standard deviation **7.12**. Both are unbiased; reparameterization has a $3.6\times$ smaller standard deviation, i.e., $\approx13\times$ fewer samples for the same precision. This variance reduction is why VAEs train stably, and why *discrete* latent variables (sequences, graphs, cell-type labels) are harder: they cannot be reparameterized directly, and relaxations (Gumbel–softmax) or score-function estimators with baselines are needed (Chapter 14).

### 8.4.2 KL between Gaussians (the closed form used everywhere)

For $q=\Normal(m,s^2)$ and $p=\Normal(0,1)$:

$$
\KL{q}{p}=\E_q\Big[\log\frac{q(z)}{p(z)}\Big]=\E_q\Big[-\log s-\tfrac{(z-m)^2}{2s^2}+\tfrac{z^2}{2}\Big]=-\log s-\tfrac12+\tfrac12(m^2+s^2)=\tfrac12\big(m^2+s^2-1-\log s^2\big).
$$

(using $\E_q[(z-m)^2]=s^2$ and $\E_q[z^2]=m^2+s^2$). In $d$ dimensions with diagonal covariance, sum over coordinates. The Monte Carlo check in the code agrees to three decimals (0.4358 vs. 0.4365).

---

## 8.5 Graphical models: structure, conditional independence, and message passing

A **directed graphical model (Bayesian network)** factorizes a joint distribution along a DAG: $p(x_1,\dots,x_n)=\prod_ip(x_i\mid\mathrm{pa}(x_i))$. An **undirected model (Markov random field)** factorizes into positive potentials over cliques: $p(x)=\frac1Z\prod_c\psi_c(x_c)$. The graph encodes **conditional independence**: two variables are conditionally independent given a set $S$ iff $S$ separates them in the appropriate graph-theoretic sense (*d-separation* for directed graphs; Chapter 44 uses it for causal reasoning).

**The three elementary structures** (for variables $A,B,C$):

| Structure | Name | Independence |
|---|---|---|
| $A\to B\to C$ | Chain (mediation) | $A\indep C\mid B$ |
| $A\leftarrow B\to C$ | Fork (common cause) | $A\indep C\mid B$ |
| $A\to B\leftarrow C$ | Collider (common effect) | $A\indep C$ marginally, but **dependent given $B$** |

The collider case is the source of *selection bias*: conditioning on a common effect induces dependence between its causes (for instance, among patients selected into a study because of disease, risk factors become negatively correlated).

### 8.5.1 Gaussian graphical models: zeros of the precision matrix

For jointly Gaussian variables with covariance $\boldsymbol\Sigma$ and **precision matrix** $\boldsymbol\Omega=\boldsymbol\Sigma^{-1}$, the density is $p(x)\propto\exp(-\tfrac12x^\top\boldsymbol\Omega x)$. If $\Omega_{ij}=0$, the quadratic form contains no cross-term between $x_i$ and $x_j$, so *conditional on all other variables* the density factorizes in $x_i$ and $x_j$: they are conditionally independent. Conversely, the **partial correlation** is $\rho_{ij\cdot\text{rest}}=-\Omega_{ij}/\sqrt{\Omega_{ii}\Omega_{jj}}$. *Zeros in the precision matrix, not in the covariance matrix, encode direct interactions.*

In the code, $A\to B\to C$ with strong links gives $\mathrm{corr}(A,C)=0.769$ (they are strongly *correlated* through $B$), yet the partial correlation given $B$ is $0.000$ and the corresponding precision entry is $-0.001$. **Gene co-expression networks built from correlations connect everything that shares an upstream regulator; graphical-LASSO networks built from sparse precision matrices (Friedman et al., 2008) aim to retain only direct links.** Neither is causal without further assumptions (Chapter 44), but the distinction between *marginal* and *conditional* dependence is the first step.

### 8.5.2 Message passing and the forward algorithm

For tree-structured graphs, **belief propagation** (the sum-product algorithm) computes all marginals exactly by passing messages along edges, with cost linear in the number of edges. An HMM is a chain-structured graphical model; the forward–backward algorithm of Chapter 6 *is* belief propagation on a chain. On graphs with cycles, loopy BP and variational approximations are used. A *Potts model* (Chapter 29) is a pairwise MRF over residues; inferring its couplings from an alignment is the direct-coupling-analysis problem; the energy-based view of Chapter 14 generalizes this.

---

## 8.6 Identifiability: when are latent variables meaningful?

A latent-variable model is **identifiable** if different parameter values give different data distributions. Many popular latent models are *not*, and the consequences for biological interpretation are significant.

**Factor analysis.** The model $x=\mathbf{W}z+\boldsymbol\epsilon$, $z\sim\Normal(0,\mathbf{I}_k)$, $\boldsymbol\epsilon\sim\Normal(0,\boldsymbol\Psi)$ implies $\mathrm{Cov}(x)=\mathbf{W}\mathbf{W}^\top+\boldsymbol\Psi$. For any orthogonal matrix $\mathbf{R}$ ($\mathbf{R}\mathbf{R}^\top=\mathbf{I}$), the loadings $\mathbf{W}'=\mathbf{W}\mathbf{R}$ give $\mathbf{W}'\mathbf{W}'^\top=\mathbf{W}\mathbf{R}\mathbf{R}^\top\mathbf{W}^\top=\mathbf{W}\mathbf{W}^\top$: **an identical distribution**. The code confirms: the covariance matrices differ by $9\times10^{-16}$ while the loadings differ by up to $4.1$. *The individual latent axes are arbitrary*; only the *subspace* they span is determined.

**Mixtures** suffer *label switching* (permuting component indices leaves the likelihood unchanged). **VAEs** generalize rotational non-identifiability to nonlinear, non-Gaussian latent spaces: without additional assumptions (supervision, sparsity, auxiliary variables, specific inductive biases), there are infinitely many latent representations that produce identical distributions, and *unsupervised "disentanglement" is impossible in general* (Locatello et al., 2019). [[E]] as a theorem; the practical conclusion is that *the axis-aligned interpretation of a VAE's latent dimensions is a property of the optimizer and architecture, not a discovery about biology.*

**Implications.**

1. Compare latent spaces by the **subspaces they span** (principal angles, CCA, Procrustes alignment) or by **downstream-invariant quantities** (neighbor graphs, predicted distributions), never by axis-by-axis comparison across runs.
2. A latent dimension "corresponds to" a biological program only if it is *validated against something external*: known markers, perturbation response, orthogonal measurements.
3. Identifiability can be restored by structure: *conditional* priors given observed covariates, sparse loadings with known pathway annotations (as in expiMap and related interpretable models), or multi-view data in which each view shares the latent variable (Chapter 40).

---

## 8.7 Biological latent-variable models

### 8.7.1 Cell state: discrete and continuous

Single-cell RNA-seq data are well described by a latent-variable model with a continuous cell state $z_n\in\R^k$ and a nuisance library-size factor $\ell_n$:

$$
z_n\sim\Normal(0,\mathbf{I}),\qquad x_{ng}\mid z_n,\ell_n\sim\mathrm{NB}\big(\mu=\ell_n\,\rho_g(z_n),\ \theta_g\big),\qquad \rho(z)=\softmax(f_\phi(z)),
$$

where $f_\phi$ is a neural decoder and $\rho_g$ is the fraction of a cell's transcripts coming from gene $g$ (this is **scVI**; Lopez et al., 2018; derived in Chapter 30). The model separates biological state ($z$) from technical scaling ($\ell$), uses the count likelihood of Chapter 4, and is trained by maximizing the ELBO with an amortized encoder. Whether the true latent structure is *discrete* (cell types) or *continuous* (differentiation trajectories, signaling states) is an empirical and conceptual question; real tissue has both.

### 8.7.2 Haplotype copying: the Li–Stephens model

Chapter 6's HMM gets a beautiful genetic application. A new haplotype (the sequence of alleles along one chromosome) is modeled as an imperfect *mosaic* of haplotypes in a reference panel $H_1,\dots,H_K$: at each variant site, the **hidden state** is *which reference haplotype is being copied*. The state stays the same between adjacent sites with probability $1-\rho$ (no recombination), and switches to a uniformly chosen reference with probability $\rho$ (recombination, scaled by genetic distance). **Emission:** the observed allele equals the copied haplotype's allele with probability $1-\varepsilon$ (mutation or error with probability $\varepsilon$). Forward–backward gives the posterior over which haplotype is copied and hence **posterior allele probabilities at untyped sites**: *genotype imputation* (Li & Stephens, 2003). Genome-wide imputation tools (Beagle, IMPUTE, minimac) use variants of this model. The key points: (i) it is an HMM with $K$ states, so exact inference costs $O(LK^2)$ (or $O(LK)$ with the structure of the transition matrix); (ii) the latent state *has a physical meaning* (a shared ancestral segment); (iii) its identifiability depends on the reference panel's diversity.

### 8.7.3 Phylogenies

A phylogenetic model places the latent ancestral sequences at internal nodes of a tree; substitution along each branch is a continuous-time Markov chain (Chapter 21). The likelihood of the observed leaf sequences marginalizes over all ancestral states; **Felsenstein's pruning algorithm** does this by dynamic programming along the tree (sum-product message passing on a tree). *Ancestral sequence reconstruction* is posterior inference on the latent nodes (Chapter 42).

---

## 8.8 Model selection and the problem of "how many components?"

The number of latent components or clusters $K$ is not given by the data alone. Criteria include **BIC** ($=-2\log\hat L+d\log n$, approximating the marginal likelihood), held-out likelihood, stability under resampling, and Bayesian nonparametric priors (Dirichlet process). **Crucially, these answer "how many components best approximate the density," which is a different question from "how many discrete types exist."**

In the simulation, the data are a *single continuous spiral* (no discrete types at all). A Gaussian mixture is fit with $K\in\{1,2,3,5,8,12,16\}$; BIC falls steadily from 9961 ($K=1$) to **6876 at $K=12$** and only turns up at $K=16$ (6975). A practitioner using BIC would conclude that "the data contain 12 clusters." Mixtures tile a continuum with local Gaussians because that is the cheapest way to approximate its density.

!!! example "Worked Research Example 8.1: \"Our clustering resolves 12 cell states in this tissue\""
    **Situation.** A single-cell study applies graph-based clustering to 80,000 cells from a differentiating tissue, selects the resolution parameter that maximizes a silhouette score, and reports 12 discrete "states," giving each a marker-gene signature and a biological interpretation.

    **Question.** What does the number 12 mean? How would you test whether the states are real?

    **Reasoning.**

    1. *What does the method assume?* That the data consist of discrete groups. Algorithms such as Leiden and $k$-means return clusters for *any* data, including a spiral (§8.8).
    2. *What do the criteria measure?* Silhouette and BIC reward cluster compactness or density approximation. In a continuum, the optimal $K$ grows with sample size: more cells permit finer tiling.
    3. *Alternative hypotheses.* (H1) Discrete types. (H2) A continuous trajectory tiled by clusters. (H3) Discrete types plus continuous within-type variation (cell cycle, activation). (H4) Technical subdivision (batch, depth, doublets).
    4. *Predictions that discriminate.* Under H1, between-cluster gaps in density (low-density regions between clusters) and *stable* assignments under resampling and under different features; markers are bimodal/on–off. Under H2, cells are distributed continuously along a curve (a principal-curve or diffusion-pseudotime fit explains most variance); the number of clusters depends strongly on the resolution parameter; markers change gradually. Under H4, clusters correlate with batch or depth.
    5. *Experiments.* (a) Cluster stability: bootstrap cells and genes, compute the adjusted Rand index across runs and resolution settings; (b) examine the nearest-neighbor graph's connectivity between adjacent clusters (high boundary connectivity → continuum); (c) test for *discrete* markers: does a candidate marker show a bimodal expression distribution across cells? (d) check cluster–batch mutual information; (e) functional validation: sort cells by markers and test that they differ in function (perturbation response, lineage tracing, chromatin).
    6. *What would change your mind?* Bimodality of markers *plus* stable assignments *plus* functional differences supports discrete types. Gradual marker changes plus resolution-dependent $K$ supports a continuum, in which case a trajectory/latent-coordinate description is more faithful than discrete labels.

    **Expert analysis.** "12 clusters" is a property of the *analysis*, not necessarily of the tissue. The scientifically meaningful claims are (i) which axes of variation exist and how much they explain, and (ii) which discrete distinctions are supported by *independent* evidence. Cell-type atlases are best viewed as *coordinate systems with labels attached by consensus*; this will matter when evaluating cell-state foundation models and their annotation accuracy (Chapters 25, 38).

!!! example "Worked Research Example 8.2: Two training runs give different latent dimensions. Is the biology unstable?"
    **Situation.** You train a single-cell VAE twice with different seeds. Latent dimension 3 in run A correlates with interferon response; in run B the interferon signal is spread across dimensions 3, 7, and 9. A colleague concludes that "the model is unreliable."

    **Reasoning.**

    1. *What is identifiable?* From §8.6, only the *subspace* (and the induced neighborhood structure and decoded distribution) is determined; individual axes are arbitrary under rotation (and, for nonlinear VAEs, under far larger families of reparameterizations).
    2. *So what should be compared?* The distribution each run assigns to the data (held-out ELBO), the neighbor graphs they induce (fraction of shared $k$-nearest neighbors), and the *subspaces* (principal angles between the decoder Jacobians or between latent-space embeddings aligned by CCA).
    3. *Prediction under the identifiable-subspace view.* Held-out likelihood and neighbor-graph overlap similar across seeds; the interferon direction is recoverable *as a linear direction* in both latent spaces (e.g., by regressing the known response signature on $z$), even if not axis-aligned.
    4. *Prediction under genuine instability.* The two runs disagree on held-out likelihood or neighbor structure; the interferon direction cannot be found linearly in one run.
    5. *Remedy.* If you need stable, interpretable axes, add structure: supervise a subset of dimensions with known labels, use pathway-based sparse decoders, or define interpretable directions *post hoc* in a way that is invariant to rotations (linear probes for named signatures).

    **Expert analysis.** The colleague confused a non-identifiable *parameterization* with an unstable *model*. The correct move is to ask which *functionals of the model* are identifiable. A principle for the whole book: **interpret quantities that the likelihood pins down, and be skeptical of those it does not.**

---

## 8.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: is this latent variable real?"
    **Setting.** A foundation model's embedding has a direction that predicts "cell age." The authors call it "the aging axis of the cell."

    **Decompose.**

    1. *Descriptive or causal?* A latent direction that correlates with age *describes* variation in the data; it says nothing about whether intervening along it changes age-related biology (Chapter 44).
    2. *Identifiable?* The direction is defined by a regression on $z$; it is rotation-invariant in that sense, but depends on which cells and which covariates were used. Check that it is recovered under bootstrap, in a held-out donor set, and in another embedding.
    3. *Confounded?* Age correlates with batch (older donors collected at one site), sex, disease, and cell-type composition. Regress these out *and check whether the axis survives*.
    4. *Specific?* Does the axis predict age better than a standard transcriptomic clock built on a few hundred genes? If not, it has not added knowledge.
    5. *Perturbation test.* If the axis is a mechanism, perturbing the genes that load on it (CRISPR, drug) should move cells along it and change an independent aging readout; the response should be *in the predicted direction and dose-dependent*.

    **Possible outcomes.** (a) Survives 2–4 but fails 5: a good *biomarker*, not a *mechanism*. (b) Fails 3: a confounded proxy. (c) Passes all: a candidate mechanistic axis, a claim at rung C3–C4 of the Claim Ladder (Chapter 1).

    **What this teaches.** The word "axis" invites causal readings that the data have not earned. Always ask which rung of the ladder the evidence reaches.

---

## 8.10 Connections

- **Backward:** The ELBO is built from Chapter 5's KL; EM and conjugacy use Chapter 4's likelihood and Bayes; HMM forward–backward is Chapter 6's dynamic program on a chain; factor analysis is Chapter 2's low-rank structure with a noise model.
- **Forward:** The VAE (Chapter 14) is amortized variational inference with a neural decoder; diffusion (Chapter 15) is a hierarchical latent-variable model with a fixed encoder; scVI and its descendants (Chapter 30); Li–Stephens and phylogenetic models (Chapters 21, 26, 42); Potts/DCA (Chapter 29); d-separation and identification (Chapter 44); identifiability of representations (Chapters 13, 18, 45).

!!! takeaways "Key takeaways"
    1. A latent-variable model specifies $p(x)=\int p(x\mid z)p(z)dz$; the integral makes likelihood and posterior hard.
    2. $\log p(x)=\mathcal{L}(q,\theta)+\KL{q(z)}{p(z\mid x)}$: **the ELBO is a lower bound whose gap is the KL to the posterior.** EM alternates exact posterior (E) and expected-complete-likelihood maximization (M) and never decreases the likelihood.
    3. **Reparameterization** moves the gradient inside the expectation and cuts variance (here by $\approx3.6\times$ in standard deviation); discrete latents need other tools.
    4. **Conditional independence** is encoded in graph structure; in Gaussians, **zeros of the precision matrix** (not the covariance) mean no direct interaction. Colliders induce dependence by conditioning.
    5. Latent axes are generally **not identifiable** (factor-analysis rotations, label switching, VAE reparameterizations): interpret subspaces and identifiable functionals, not individual axes.
    6. Model-selection criteria choose **density approximations**, not the number of true types: a mixture finds 12 "clusters" in a spiral.
    7. Biological latent-variable structure is everywhere: cell state (scVI), haplotype copying (Li–Stephens), phylogeny (Felsenstein pruning).

---

## Further reading

- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat. Soc. B* 39, 1–38.
- Neal, R. M. & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In *Learning in Graphical Models*. (The ELBO view of EM.)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, chapters 9, 10. Springer.
- Blei, D. M., Kucukelbir, A. & McAuliffe, J. D. (2017). Variational inference: a review for statisticians. *J. Am. Stat. Assoc.* 112, 859–877.
- Kingma, D. P. & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*. (Reparameterization.)
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
- Friedman, J., Hastie, T. & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics* 9, 432–441.
- Locatello, F. et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML*.
- Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods* 15, 1053–1058.
- Li, N. & Stephens, M. (2003). Modeling linkage disequilibrium and identifying recombination hotspots using single-nucleotide polymorphism data. *Genetics* 165, 2213–2233.
- Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum likelihood approach. *J. Mol. Evol.* 17, 368–376.
- Hie, B., Zhong, E. D., Berger, B. & Bryson, B. (2021). Learning the language of viral evolution and escape. *Science* 371, 284–288. (A bridge to Chapter 34.)
