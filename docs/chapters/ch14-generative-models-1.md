# Chapter 14. Generative Models I: Likelihood Models, VAEs, Flows, GANs, and Energy-Based Models

!!! abstract "Chapter at a glance"
    **Motivation.** A generative model learns a distribution $p_\theta(x)$ over biological objects (sequences, structures, molecules, expression profiles) so that we can *sample* new ones, *score* existing ones, and *search* for ones with desired properties. Design (Chapter 36), drug discovery (Chapter 37), and in silico experiments (Chapter 39) all rest on generative modeling.
    **Prerequisites.** Chapters 4, 5, 8, 9, 13.
    **You will be able to:** (1) compare five generative families by what they guarantee (exact likelihood, coverage, speed, latent structure); (2) derive the VAE objective and diagnose posterior collapse; (3) derive the change-of-variables formula and coupling-flow log-determinant; (4) derive the GAN optimal discriminator and its Jensen–Shannon connection; (5) derive the energy-based-model gradient; (6) explain why *sampling from a model is not the same as finding its mode*, and why oracle-judged generation misleads.

---

## 14.1 What do we want from a generative model?

Four capabilities, which different families provide to different degrees:

1. **Sampling:** draw $x\sim p_\theta$. (Needed for generation and design.)
2. **Density evaluation:** compute $\log p_\theta(x)$ exactly. (Needed for scoring variants, anomaly detection, model comparison; Chapter 5.)
3. **Latent structure:** a low-dimensional code $z$ for interpolation, optimization, and interpretation (Chapter 8).
4. **Coverage:** capture *all* modes of the data distribution (diversity), not only the easy ones (Chapter 5: forward vs. reverse KL).

| Family | Training objective | Exact $\log p$? | Sample speed | Coverage | Latent $z$ | Main failure |
|---|---|---|---|---|---|---|
| **Autoregressive** | Max. likelihood | **Yes** | Slow (sequential) | Good | None explicit | Exposure bias; degenerate high-likelihood repeats |
| **VAE** | ELBO | Lower bound | Fast | Good, blurry/avg | **Yes** (amortized) | Posterior collapse; prior holes |
| **Normalizing flow** | Max. likelihood | **Yes** | Fast | Good | **Yes** (invertible) | Restricted architectures; discrete data awkward |
| **GAN** | Adversarial (JS/Wasserstein) | No | Fast | **Poor (mode collapse)** | Implicit | Instability; no likelihood |
| **EBM** | Max. likelihood via MCMC | Up to $Z$ | Slow (MCMC) | Good | None | Intractable $Z$; costly sampling |
| **Diffusion / flow matching** (Chapter 15) | Denoising / regression | Bound or ODE | Moderate | **Excellent** | Hierarchical | Many steps; discrete data subtle |

Autoregressive models were developed in Chapter 13; this chapter covers the rest of the first five rows, and §14.7 treats the evaluation problem common to all.

---

## 14.2 Sampling from autoregressive models, and the typical set

An autoregressive model draws $x_t\sim p_\theta(\cdot\mid x_{<t})$ step by step. Two knobs shape what you get:

- **Temperature** $T$: sample from $p_\theta^{1/T}$ (renormalized). Since $\log p^{1/T}=\frac1T\log p$, this is the Boltzmann distribution $e^{-E/T}$ with $E=-\log p$: $T\to0$ approaches the *mode* (greedy decoding), $T=1$ samples the model, $T>1$ flattens.
- **Truncation** (top-$k$, nucleus/top-$p$): restrict to the most probable tokens.

**Why "sample" ≠ "most likely."** For a long i.i.d. sequence with per-symbol entropy $H$, the asymptotic equipartition property says almost all the probability mass sits in the **typical set**, sequences whose per-symbol negative log-probability is within a small band around $H$. The most probable single sequence is *not* typical. In the simulation, sequences of 300 i.i.d. symbols with $P(1)=0.7$ have a per-symbol NLL of $0.611\pm0.022$ (the entropy is 0.611), while the single most probable sequence (all ones) has NLL $0.357$ but probability $3.4\times10^{-47}$: *the mode is more probable than any typical sequence yet is essentially never sampled.*

**Consequence for biological design.** A low-temperature or beam-search procedure drives sequences toward the *consensus/mode*: highly likely under the model, but potentially (i) low-diversity, (ii) degenerate (language models often assign high likelihood to long poly-A or other low-complexity repeats, which are "easy to predict" but biologically meaningless), and (iii) atypical of functional sequences. Conversely, sampling at $T=1$ reproduces the *statistics* of the training distribution, including its mutational noise. The right temperature is a *design hyperparameter* tied to the relation between $\log p$ and fitness (Chapter 5, §5.6.4): if fitness $\propto\log p$ over the relevant range, lowering $T$ shifts toward higher fitness at the cost of diversity and eventual off-distribution degeneracy; the optimum is empirical and must be tested prospectively (Chapter 36).

---

## 14.3 Variational autoencoders

### 14.3.1 Objective

The VAE (Kingma & Welling, 2014) is Chapter 8's amortized variational inference with neural networks: an **encoder** $q_\phi(z\mid x)$ approximating the posterior, a **decoder** $p_\theta(x\mid z)$, and a prior $p(z)=\Normal(0,\mathbf{I})$. Maximize the ELBO (Chapter 8, §8.3):

$$
\mathcal{L}(\theta,\phi;x)=\underbrace{\E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}-\underbrace{\KL{q_\phi(z\mid x)}{p(z)}}_{\text{regularizer}}\ \le\ \log p_\theta(x).
$$

With a diagonal Gaussian encoder $q_\phi(z\mid x)=\Normal(\boldsymbol\mu_\phi(x),\mathrm{diag}\,\boldsymbol\sigma_\phi^2(x))$, the KL term is closed-form (Chapter 8, §8.4.2): $\frac12\sum_i(\mu_i^2+\sigma_i^2-1-\log\sigma_i^2)$; the expectation is estimated with one reparameterized sample $z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$, $\boldsymbol\epsilon\sim\Normal(0,\mathbf{I})$. Decoder likelihoods: categorical for sequences, NB for counts, Gaussian for continuous measurements. **$\beta$-VAE** scales the KL term by $\beta$: $\beta>1$ promotes factorized, compressed latents; $\beta<1$ promotes reconstruction.

**Interpretation.** The reconstruction term is a *code length for the data given the code*; the KL term is a *code length for the code itself* (bits-back; Chapter 8's rhyme). The VAE balances the two: **it is a learned, lossy compressor with a probabilistic decoder.**

### 14.3.2 Posterior collapse

If the decoder is powerful enough to model $x$ without help from $z$ (a large autoregressive decoder), the optimum can set $q_\phi(z\mid x)=p(z)$ for all $x$: KL $=0$, the latent carries no information, and the model reduces to an unconditional decoder. This is **posterior collapse**. Remedies: weaken the decoder; **KL annealing** (start with $\beta\approx0$ and increase); **free bits** (do not penalize KL below a threshold per dimension); a larger reconstruction weight.

The experiment (8 latent dimensions) sweeps $\beta$ on held-out synthetic loci of Chapter 13 (signal $s$ plus GC-biased background):

| $\beta$ | $-\log p(x\mid z)$ (nats) | KL (nats) | ELBO (nats) | active dims | signal accuracy | GC $R^2$ |
|---|---|---|---|---|---|---|
| 0.1 | 65.8 | 15.3 | −81.1 | 8 | 1.000 | 0.911 |
| 1.0 | 69.2 | 3.4 | **−72.6** | 8 | 1.000 | 0.918 |
| 4.0 | 72.7 | 1.7 | −74.3 | 3 | 1.000 | 0.930 |
| 16.0 | 82.2 | 0.0 | −82.2 | **0** | 0.504 | 0.293 |

At $\beta=1$ (the true ELBO) the model attains the best bound; at $\beta=0.1$ it spends 15 nats on the code for 7 nats of reconstruction gain; at $\beta=4$ only three latent dimensions remain active while the signal is still perfectly decodable; at $\beta=16$ the posterior collapses (KL $=0.0$, no active dimensions), the reconstruction degrades to 82 nats, and the signal probe falls to 0.504 (not exactly 0.25 because the posterior means retain small residual variation). *Note again that the VAE with enough capacity encodes the GC nuisance* ($R^2\approx0.92$), as the plain autoencoder did in Chapter 13: the ELBO rewards reconstruction.

```python
--8<-- "code/ch14_generative.py"
```

Output:

```text
VAE (8 latent dims) KL-weight sweep on held-out data:
   beta   -log p(x|z) (nats)  KL (nats)  ELBO (nats)  active dims  signal acc  GC R^2
    0.1                 65.8       15.3        -81.1            8       1.000   0.911
    1.0                 69.2        3.4        -72.6            8       1.000   0.918
    4.0                 72.7        1.7        -74.3            3       1.000   0.930
   16.0                 82.2        0.0        -82.2            0       0.504   0.293

change of variables: p_X(1.7) via formula = 0.20385; scipy lognorm = 0.20385
affine coupling: log|det J| formula = -0.334261; autograd Jacobian = -0.334261; inverse error = 0.0e+00; Jacobian triangular: True
GAN: V(D*) = -1.16345; 2 JS - log 4 = -1.16345
EBM: max |analytic - numeric| log-likelihood gradient = 2.0e-10
typical set: per-symbol NLL of sampled sequences = 0.611 +/- 0.022 (entropy 0.611); the single most probable sequence (all 1s) has NLL 0.357, and occurs with probability 3.4e-47
```

### 14.3.3 VAEs in biology

- **DeepSequence** (Riesselman et al., 2018): a VAE trained on the multiple sequence alignment of a protein *family*, whose ELBO (or $\Delta$ELBO between mutant and wild type) predicts mutation effects; it captures higher-order epistasis beyond Potts models. **EVE** (Frazer et al., 2021) uses a Bayesian VAE per protein for clinical variant pathogenicity prediction. These are *family-specific* models: unlike a protein language model (Chapter 34), a new VAE is trained for each protein. [[E]] as methods; relative performance versus language models is benchmark-dependent (Chapter 34).
- **Chemical VAE** (Gómez-Bombarelli et al., 2018): a VAE on SMILES strings, with Bayesian optimization in latent space to optimize molecular properties.
- **scVI and relatives** (Lopez et al., 2018): a VAE with NB likelihood and library-size covariate for single-cell data (Chapter 30).
- **Perturbation VAEs** (scGen, CPA; Chapter 30, 39): latent arithmetic to model perturbation effects.

!!! lens "Research lens: latent-space optimization with a VAE"
    **Assumes:** (i) the decoder maps *every* region of latent space to a valid object ("no holes"); (ii) the property is a smooth function of $z$; (iii) the aggregate posterior $q(z)=\E_x q_\phi(z\mid x)$ matches the prior $p(z)$. **Information used:** the training distribution only. **Ignored:** anything outside it. **Failure modes:** *holes* where $q(z)\ll p(z)$ decode to invalid objects; *drift off the data manifold* (optimization exploits decoder errors); *property predictors unreliable away from data* (winner's curse and adversarial exploitation; Chapter 4). **Remedies:** restrict optimization to high-density latent regions; constrain with the predictor's uncertainty; validate with held-out experimental assays.

---

## 14.4 Normalizing flows

### 14.4.1 Change of variables

Let $Z\sim p_Z$ and $X=f(Z)$ with $f$ a differentiable invertible map. Then

$$
p_X(x)=p_Z\big(f^{-1}(x)\big)\,\Big|\det\frac{\partial f^{-1}}{\partial x}\Big|\;=\;p_Z(z)\,\Big|\det\frac{\partial f}{\partial z}\Big|^{-1},\qquad z=f^{-1}(x).
$$

*Intuition:* probability mass is conserved, so a region that $f$ stretches has its density diluted by the stretch factor $|\det\mathbf{J}|$. 1-D check: $Z\sim\Normal(0,1)$, $X=e^Z$; $p_X(x)=\varphi(\ln x)\cdot\frac1x$, which at $x=1.7$ equals 0.20385 and matches the lognormal density from SciPy.

A **normalizing flow** composes many simple invertible maps $f=f_K\circ\dots\circ f_1$; log-densities add, $\log p_X(x)=\log p_Z(z_0)-\sum_k\log|\det\mathbf{J}_{f_k}|$. Training maximizes the *exact* likelihood. The architectural challenge is making $f$ invertible with a cheap Jacobian determinant.

### 14.4.2 Coupling layers

An **affine coupling layer** (RealNVP; Dinh et al., 2017) splits $x=(x_1,x_2)$ and sets

$$
y_1=x_1,\qquad y_2=x_2\odot\exp\big(s(x_1)\big)+t(x_1).
$$

$s$ and $t$ are arbitrary neural networks of $x_1$, **never inverted**. The inverse is trivial, $x_2=(y_2-t(y_1))\odot\exp(-s(y_1))$, and the Jacobian is **block-triangular**, so $\log|\det\mathbf{J}|=\sum_is_i(x_1)$. The code verifies on a 4-D coupling layer: formula $-0.334261$, autograd Jacobian $-0.334261$, inverse error $0$, and the off-diagonal block is zero. Stacks of couplings with alternating splits and permutations give expressive flows.

**Biology-relevant flows.** *Boltzmann generators* (Noé et al., 2019) use flows to sample equilibrium conformations of molecules by training the flow to match the Boltzmann distribution $e^{-E(x)/k_BT}$ and reweighting, a route to equilibrium ensembles without long molecular dynamics (Chapter 52). *Flows on tori and $SE(3)$* parameterize protein backbone torsions or rigid frames (FoldFlow-type models; Chapter 36). Flows for *discrete* data (sequences) need *dequantization* or discrete flows, which have been less successful than autoregressive or diffusion models.

---

## 14.5 Generative adversarial networks

A **GAN** (Goodfellow et al., 2014) trains a generator $G:z\mapsto x$ and a discriminator $D(x)\in(0,1)$ in a minimax game:

$$
\min_G\max_D\ V(D,G)=\E_{x\sim p_\text{data}}[\log D(x)]+\E_{z\sim p(z)}[\log(1-D(G(z)))].
$$

**Optimal discriminator.** For fixed $G$ with generated density $p_g$, maximize pointwise: $\max_D\,p_\text{data}\log D+p_g\log(1-D)$ gives $D^\star(x)=\dfrac{p_\text{data}(x)}{p_\text{data}(x)+p_g(x)}$.

**Generator objective.** Substituting $D^\star$,

$$
V(D^\star,G)=\E_{p_\text{data}}\log\frac{p_\text{data}}{p_\text{data}+p_g}+\E_{p_g}\log\frac{p_g}{p_\text{data}+p_g}=2\,\mathrm{JS}(p_\text{data}\,\|\,p_g)-\log4,
$$

with $\mathrm{JS}(p\|q)=\tfrac12\KL{p}{m}+\tfrac12\KL{q}{m}$, $m=\tfrac12(p+q)$. A global minimum is reached iff $p_g=p_\text{data}$, with value $-\log4$. The code verifies numerically for $p=\Normal(0,1)$, $q=\Normal(1,1)$: $V(D^\star)=-1.16345=2\,\mathrm{JS}-\log4$.

**Failure modes.** *Mode collapse*: the generator covers a few modes the discriminator finds hard to reject (the JS divergence saturates when supports barely overlap, giving vanishing gradients; Arjovsky & Bottou, 2017). *Instability* from the two-player dynamics. **Wasserstein GANs** replace JS by the Earth Mover's distance with a Lipschitz-constrained critic, improving gradients. No likelihood is available. In biology, GANs generated functional enzyme variants (ProteinGAN; Repecka et al., 2021) and DNA with desired properties, but diffusion and autoregressive models have largely displaced them because of coverage and stability. [[S]]

---

## 14.6 Energy-based models

An **energy-based model** defines $p_\theta(x)=e^{-E_\theta(x)}/Z_\theta$ with $Z_\theta=\sum_xe^{-E_\theta(x)}$ (an integral for continuous $x$). Any function can serve as an energy, giving great flexibility, and *the normalizer $Z_\theta$ is intractable* in general. Examples: **Potts models** (Chapter 29: $E(x)=-\sum_ih_i(x_i)-\sum_{i<j}J_{ij}(x_i,x_j)$), **Boltzmann machines**, physical force fields, and Rosetta-style protein energy functions.

**The gradient of the log-likelihood.**

$$
\nabla_\theta\log p_\theta(x)=-\nabla_\theta E_\theta(x)-\nabla_\theta\log Z_\theta=-\nabla_\theta E_\theta(x)+\E_{x'\sim p_\theta}\big[\nabla_\theta E_\theta(x')\big],
$$

since $\nabla_\theta\log Z_\theta=\frac1{Z_\theta}\sum_x\nabla_\theta e^{-E_\theta(x)}=-\E_{p_\theta}[\nabla_\theta E_\theta]$. The first term is a "positive phase" (lower the energy of data), the second a "negative phase" (raise the energy of the model's own samples), and the expectation requires **MCMC sampling** from the current model (Langevin dynamics, Gibbs sampling) or approximations (contrastive divergence; Hinton, 2002). The code verifies the identity on an 8-state model by finite differences (error $2\times10^{-10}$).

**Why EBMs matter for this book.**

1. A Potts model of a protein family *is* an EBM; fitting it by maximum likelihood requires the negative phase (Boltzmann-machine learning), done in practice by pseudo-likelihood (Chapter 13) or by MCMC.
2. The **score** $\nabla_x\log p_\theta(x)=-\nabla_xE_\theta(x)$ *does not depend on $Z$*. Learning scores rather than densities sidesteps the normalizer and leads to diffusion models (Chapter 15).
3. Physical energy functions (force fields, Rosetta) are hand-designed EBMs, and the *Boltzmann distribution* at temperature $T$ links the statistical-mechanics and machine-learning views of sequence–structure–function (Chapters 23, 29, 35).

---

## 14.7 Evaluating generative models, and why it is hard

Evaluating a generator is harder than evaluating a predictor, because there is no single ground truth to compare against. Typical metrics and their pitfalls:

| Property | Metrics | Pitfall |
|---|---|---|
| **Fidelity** (samples look like data) | Held-out likelihood (if available); precision of sample manifold; discriminator accuracy; domain-specific validity (valid SMILES; foldable proteins) | Likelihood can be high for *degenerate* samples; precision ignores diversity |
| **Diversity / coverage** | Recall of the data manifold; pairwise distance among samples; cluster coverage | Diversity without fidelity (noise) scores well |
| **Novelty** | Distance to nearest training example | Memorization (a copy of training data is "high fidelity" but not new); novel ≠ functional |
| **Property optimization** | Predicted property under an *oracle* model | **Oracle exploitation** (below) |
| **Validity** | Chemical validity, secondary-structure plausibility, in-frame ORFs | Easily satisfied; weakly informative |
| **Ground truth** | Experimental hit rate; fitness; binding | Costly and limited (Chapter 46) |

**Oracle exploitation.** To avoid wet-lab cost, many papers evaluate generated designs with a *learned predictor* (a "surrogate" for fitness, binding, or expression). If the generator is optimized against the same predictor, or a correlated one, it finds **adversarial inputs** where the predictor is wrong but confident. Reported improvements then partly reflect *predictor error*, a winner's curse (Chapter 4) amplified by optimization. Safeguards: evaluate with an *independent held-out* oracle trained on different data; measure the predictor's uncertainty; penalize distance from the data; and above all **test a sample of designs experimentally** (the design–build–test loop of Chapter 36).

---

## 14.8 Worked research examples

!!! example "Worked Research Example 14.1: Generated sequences have a higher predicted fitness than natural ones"
    **Situation.** A model generates 10,000 enzyme sequences. A fitness predictor trained on deep-mutational-scanning data scores the top 100 generated sequences at 1.3 standard deviations above the best natural sequences. The authors conclude that the generator "designs improved enzymes."

    **Question.** What alternative explanations exist, and what would convince you?

    **Reasoning.**

    1. *What is the claim?* Generated sequences have higher *true* fitness. The evidence is *predicted* fitness.
    2. *Winner's curse (Chapter 4, §4.5.2).* The top 100 of 10,000 generated candidates, selected by a noisy predictor, have inflated predicted values: with predictor noise $s$ comparable to the spread $\tau$ of true effects, shrinkage by $\tau^2/(\tau^2+s^2)$ (e.g., 0.5) halves the apparent gain.
    3. *Oracle exploitation.* If the generator was conditioned on or filtered by the *same* predictor, the top sequences are those on which it errs most favorably.
    4. *Distribution shift.* The predictor was trained on near-natural variants; generated sequences may be far from the training manifold (many mutations, off-distribution combinations) where the predictor extrapolates unreliably (Chapter 45).
    5. *Alternative explanations.* The predictor rewards a *proxy* (e.g., stability, or sequence features correlating with the assay) that does not translate to catalysis; the DMS assay measures something other than the desired function.
    6. *Experiments that discriminate.* (a) Evaluate with an **independent oracle** trained on separate data or with a different architecture; (b) compute predictor uncertainty and restrict to in-domain designs; (c) *synthesize and assay a random sample* of generated sequences stratified by predicted score; (d) include natural controls and random-sampled-from-the-model controls to estimate how much *selection* (not generation) contributes; (e) check novelty and diversity.
    7. *Predictions.* If the improvement is real, assayed activity should increase *monotonically* with predicted score across the sample and exceed the natural controls; if oracle exploitation dominates, assayed activity will plateau or fall for the top-ranked designs (the characteristic "inverted U").

    **Expert analysis.** Predicted improvements are hypotheses. The scientifically strong statement is the *calibration curve between predicted and assayed score for designs*. A generator is useful to the extent that *enrichment for true function survives wet-lab testing*; this is why hit rates, not predicted scores, are the currency of Chapter 36.

!!! example "Worked Research Example 14.2: Latent-space optimization with a molecular VAE yields invalid and unsynthesizable molecules"
    **Situation.** You optimize a drug-likeness score by gradient ascent in the latent space of a SMILES VAE. After 50 steps, the decoded molecules have excellent predicted scores, but 60% are chemically invalid, and the valid ones are bizarre and unsynthesizable.

    **Reasoning.**

    1. *Where do latent holes come from?* The VAE regularizes $q_\phi(z\mid x)$ toward the prior, but the *aggregate posterior* $q(z)=\E_xq_\phi(z\mid x)$ typically differs from $p(z)$; regions with high prior density but little aggregate posterior mass ("prior holes") are never trained to decode to valid molecules.
    2. *Why does gradient ascent find them?* The property predictor is trained only on latent codes of *real* molecules. In holes it extrapolates unpredictably, and gradient ascent *seeks* regions of high predicted score regardless of validity: adversarial search against the predictor.
    3. *Alternative or compounding causes.* SMILES is a brittle representation (single-character errors give invalid strings); the decoder is autoregressive and sensitive to small latent changes; posterior collapse may have reduced the latent's information so the decoder relies on its own language model.
    4. *Experiments.* Measure validity as a function of distance from the nearest training latent code; compare latent optimization with *constrained* steps (trust region) or with optimization in a discrete space with a *graph-based* representation guaranteeing validity; add predictor uncertainty penalties.
    5. *Remedies.* Use a representation with built-in validity (graph, SELFIES), regularize the aggregate posterior (e.g., adversarial or MMD matching), use *Bayesian optimization with uncertainty-aware acquisition*, and perform *joint training* of the predictor on latent codes of optimized molecules (active learning; Chapter 46).

    **Expert analysis.** Optimization pressure finds the *weakest point of the model stack*: here, the decoder's behavior in untrained regions. Every design pipeline must be stress-tested by asking *which component is least reliable and what is the optimizer going to exploit?*

---

## 14.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: choosing a generative family for a biological design task"
    **Task.** Design 100 bp promoters that drive high expression in a given cell type, with a budget of ~$10^4$ MPRA measurements and $10^6$ unlabeled genomic promoters.

    **Decompose the requirements.**

    1. *Do I need a likelihood?* For *filtering* designs for "naturalness" and for comparing models: yes → autoregressive, flow, or VAE-ELBO.
    2. *Do I need conditional generation* ("high expression in cell type A, low in B")? → conditional model with the property as input (classifier-free guidance in diffusion; condition token in an LM).
    3. *Do I need coverage/diversity?* Likely: a diverse set reduces risk. → avoid GANs; consider autoregressive or diffusion.
    4. *Do I need sample efficiency with $10^4$ labels?* The labels are scarce: *pretrain on the $10^6$ unlabeled promoters* (generative prior), then *steer* using a predictor trained on the MPRA data (Chapter 36).
    5. *What will the optimizer exploit?* The MPRA-trained predictor will be unreliable beyond the training distribution. → measure predictor uncertainty; run **rounds**: generate, measure, retrain (active learning; Chapter 46).

    **Hidden assumptions.** (i) MPRA activity in reporter context predicts activity in the genome (G-M). (ii) Natural promoters represent the functional manifold. (iii) The predictor generalizes to model-generated sequences.

    **What would change my mind?** If a held-out MPRA round of generated sequences shows no enrichment over random draws from the *prior*, the guidance is not helping and the problem is the predictor, not the generator.

    **Distinguishing fundamental from implementation problems.** Collapse, holes, and degeneracy are implementation problems with known remedies. A predictor that cannot be extrapolated, and a noise floor in the MPRA, are *information limits*; only more or better data (A4) can change them.

---

## 14.10 Connections

- **Backward:** ELBO and reparameterization (Chapter 8); KL asymmetry (Chapter 5); autoregressive and masked objectives (Chapter 13); Potts as EBM (Chapter 13); winner's curse (Chapter 4).
- **Forward:** diffusion and flow matching (Chapter 15) resolve many of these trade-offs; protein, DNA, and molecule generators (Chapters 32, 36, 37); generative models for single-cell perturbation (Chapters 30, 39); Boltzmann generators and conformational ensembles (Chapter 52); evaluation of generated designs (Chapters 43, 46).

!!! takeaways "Key takeaways"
    1. Generative families trade **exact likelihood, sample speed, coverage, and latent structure**; no family wins on all.
    2. **Sampling at temperature 1 samples the typical set; the mode is atypical** (probability $3.4\times10^{-47}$ vs. NLL 0.357 vs. 0.611). Low-temperature design drifts to consensus and degenerate repeats.
    3. **VAE:** ELBO $=\E_q\log p(x\mid z)-\KL{q}{p(z)}$; posterior collapse at large KL weight ($\beta=16$ gave KL $=0$, signal accuracy 0.504). Capacity-sufficient VAEs encode nuisance too.
    4. **Flows:** $p_X(x)=p_Z(f^{-1}(x))|\det\partial f^{-1}/\partial x|$; coupling layers make the Jacobian triangular, $\log|\det\mathbf{J}|=\sum s$.
    5. **GAN:** $D^\star=p_\text{data}/(p_\text{data}+p_g)$ and $V(D^\star)=2\,\mathrm{JS}-\log4$; mode collapse and no likelihood.
    6. **EBM:** $\nabla_\theta\log p=-\nabla E(x)+\E_{p_\theta}\nabla E$; the **score** is $Z$-free, leading to diffusion.
    7. **Oracle exploitation** and the winner's curse make predicted-score evaluation of designs unreliable; the only trustworthy currency is experimental enrichment.

---

## Further reading

- Kingma, D. P. & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*. Rezende, D. J., Mohamed, S. & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML*. Higgins, I. et al. (2017). β-VAE. *ICLR*. Bowman, S. R. et al. (2016). Generating sentences from a continuous space. *CoNLL*. (Posterior collapse.)
- Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). Density estimation using Real NVP. *ICLR*. Papamakarios, G. et al. (2021). Normalizing flows for probabilistic modeling and inference. *JMLR* 22, 1–64.
- Goodfellow, I. et al. (2014). Generative adversarial nets. *NeurIPS*. Arjovsky, M. & Bottou, L. (2017). Towards principled methods for training generative adversarial networks. *ICLR*. Arjovsky, M., Chintala, S. & Bottou, L. (2017). Wasserstein GAN. *ICML*.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. & Huang, F. (2006). A tutorial on energy-based learning. Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation* 14, 1771–1800.
- Riesselman, A. J., Ingraham, J. B. & Marks, D. S. (2018). Deep generative models of genetic variation capture the effects of mutations. *Nature Methods* 15, 816–822. Frazer, J. et al. (2021). Disease variant prediction with deep generative models of evolutionary data. *Nature* 599, 91–95.
- Gómez-Bombarelli, R. et al. (2018). Automatic chemical design using a data-driven continuous representation of molecules. *ACS Central Science* 4, 268–276.
- Noé, F., Olsson, S., Köhler, J. & Wu, H. (2019). Boltzmann generators: sampling equilibrium states of many-body systems with deep learning. *Science* 365, eaaw1147.
- Repecka, D. et al. (2021). Expanding functional protein sequence spaces using generative adversarial networks. *Nature Machine Intelligence* 3, 324–333.
- Holtzman, A., Buys, J., Du, L., Forbes, M. & Choi, Y. (2020). The curious case of neural text degeneration. *ICLR*. (Nucleus sampling.)
- Brookes, D. H., Park, H. & Listgarten, J. (2019). Conditioning by adaptive sampling for robust design. *ICML*. Gao, W., Fu, T., Sun, J. & Coley, C. (2022). Sample efficiency matters: a benchmark for practical molecular optimization. *NeurIPS*.
