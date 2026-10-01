# Chapter 15. Generative Models II: Diffusion, Score Matching, and Flow Matching

!!! abstract "Chapter at a glance"
    **Motivation.** Diffusion and flow-matching models are the engine of modern biomolecular design and structure prediction: AlphaFold 3's structure module, RFdiffusion, Chroma, Boltz, Chai, DiffDock, EvoDiff, and several single-cell perturbation models all use them. They offer stable training, excellent coverage, and flexible conditioning, which are precisely the properties that Chapter 14 found missing in GANs and VAEs.
    **Prerequisites.** Chapters 3, 4, 5, 8, 14.
    **You will be able to:** (1) derive the DDPM forward process, its posterior, and the noise-prediction loss from the ELBO; (2) derive Tweedie's formula and the link between noise prediction and the score; (3) derive the probability-flow ODE and the continuous-time view; (4) derive flow matching and explain why regression onto a conditional velocity learns the marginal velocity; (5) derive classifier-free guidance and explain its failure when over-applied; (6) understand discrete (masked) diffusion and its relation to masked language modeling; (7) explain how equivariance and manifold structure enter diffusion on molecules.

---

## 15.1 The idea in one paragraph

Take data $x_0$. Gradually add Gaussian noise until nothing but noise remains. This **forward process** is fixed and simple. Now learn the **reverse process**: a neural network that, given a noisy sample and the noise level, removes a little noise. Starting from pure noise and applying the learned reverse process step by step generates a new sample. Training is *regression*: show the network a noised example and ask it to predict the noise (or the clean data, or a velocity). No adversary, no likelihood intractability, no MCMC; the difficulty has been moved from *modeling the whole distribution at once* to *denoising at every scale*.

---

## 15.2 Denoising diffusion probabilistic models (DDPM)

### 15.2.1 Forward process

Fix a variance schedule $\beta_1,\dots,\beta_T\in(0,1)$ and define

$$
q(x_t\mid x_{t-1})=\Normal\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t\mathbf{I}\big).
$$

Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Because a Gaussian passed through a linear-Gaussian step is Gaussian, the marginal at any step has a closed form:

$$
q(x_t\mid x_0)=\Normal\big(x_t;\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf{I}\big)\quad\Longleftrightarrow\quad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ \ \boldsymbol\epsilon\sim\Normal(0,\mathbf{I}).
$$

*Proof by induction.* If $x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}}\boldsymbol\epsilon_1$, then $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon_2=\sqrt{\bar\alpha_t}x_0+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\boldsymbol\epsilon_1+\sqrt{\beta_t}\boldsymbol\epsilon_2$; the two independent Gaussians sum to one with variance $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$. $\square$ The code verifies this by simulating 40 forward steps one at a time from $x_0=(1.3,-0.7)$: simulated mean $(1.177,-0.634)$ and variance $0.181$, against closed form $(1.176,-0.633)$ and $0.181$. This closed form is what makes training cheap: *any* noise level can be sampled in one step, with no simulation.

For large $T$ with a suitable schedule, $\bar\alpha_T\approx0$ and $q(x_T)\approx\Normal(0,\mathbf{I})$: the forward process destroys all information.

### 15.2.2 The true reverse posterior

The reverse step *given the clean data* is also Gaussian. By Bayes, $q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$, a product of two Gaussians in $x_{t-1}$:

- precision: $\dfrac{\alpha_t}{\beta_t}+\dfrac{1}{1-\bar\alpha_{t-1}}=\dfrac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}=\dfrac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$, using $\alpha_t-\alpha_t\bar\alpha_{t-1}+\beta_t=1-\bar\alpha_t$;
- hence variance $\tilde\beta_t=\dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ and mean

$$
\tilde{\boldsymbol\mu}_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t .
$$

The code checks the mean against brute-force Bayes on a fine 1-D grid: formula $1.69270$, numerical $1.69270$.

### 15.2.3 The ELBO and the noise-prediction loss

Treat $x_1,\dots,x_T$ as latent variables of a hierarchical VAE whose *encoder is the fixed forward process* and whose decoder is $p_\theta(x_{t-1}\mid x_t)=\Normal(\boldsymbol\mu_\theta(x_t,t),\sigma_t^2\mathbf{I})$ with $p(x_T)=\Normal(0,\mathbf{I})$. The ELBO (Chapter 8) decomposes into

$$
-\log p_\theta(x_0)\le\E_q\Big[\underbrace{\KL{q(x_T\mid x_0)}{p(x_T)}}_{L_T\ (\text{constant})}+\sum_{t=2}^T\underbrace{\KL{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}}_{L_{t-1}}\ \underbrace{-\log p_\theta(x_0\mid x_1)}_{L_0}\Big].
$$

Each $L_{t-1}$ is a KL between two Gaussians with the same variance (taking $\sigma_t^2=\tilde\beta_t$), hence $L_{t-1}=\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol\mu}_t-\boldsymbol\mu_\theta\|^2$. Substituting $x_0=(x_t-\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde{\boldsymbol\mu}_t$ gives

$$
\tilde{\boldsymbol\mu}_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon\Big),
$$

so we parameterize the learned mean the same way with a network $\boldsymbol\epsilon_\theta(x_t,t)$ predicting the noise. Then $L_{t-1}=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(x_t,t)\|^2$. Ho et al. (2020) found that dropping the weighting gives better samples, leading to the **simplified loss**

$$
\boxed{\mathcal{L}_\text{simple}(\theta)=\E_{t\sim\mathrm{U}\{1..T\},\,x_0,\,\boldsymbol\epsilon}\Big[\big\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,\ t\big)\big\|^2\Big].}
$$

**Training:** sample a data point, a random $t$, noise it in one shot, regress the noise. **Sampling:** $x_T\sim\Normal(0,\mathbf{I})$, then for $t=T,\dots,1$: $x_{t-1}=\frac1{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(x_t,t)\big)+\sigma_t\mathbf{z}$, $\mathbf{z}\sim\Normal(0,\mathbf{I})$. The weighting that the simplified loss drops corresponds to which noise levels the model emphasizes; modern practice (Karras et al., EDM) chooses noise-level distributions and weightings explicitly.

---

## 15.3 Scores, Tweedie, and why noise prediction learns the density

The **score** of a distribution is $\nabla_x\log p(x)$, the direction of steepest increase of the log-density (for an energy-based model, $-\nabla E$; Chapter 14). Score matching learns the score without the normalizer.

**Noise prediction is score estimation.** For the forward marginal, $\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}=-\frac{\boldsymbol\epsilon}{\sqrt{1-\bar\alpha_t}}$. The score of the *noisy marginal* $q(x_t)=\int q(x_t\mid x_0)q(x_0)dx_0$ is

$$
\nabla\log q(x_t)=\frac{\int\nabla q(x_t\mid x_0)\,q(x_0)\,dx_0}{q(x_t)}=\E\big[\nabla_{x_t}\log q(x_t\mid x_0)\ \big|\ x_t\big]=-\frac{\E[\boldsymbol\epsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}} .
$$

The minimizer of the squared-error noise-prediction loss is the *conditional mean* $\boldsymbol\epsilon^\star(x_t,t)=\E[\boldsymbol\epsilon\mid x_t]$ (Chapter 7: the Bayes predictor under squared loss). Therefore

$$
\boxed{\nabla_{x_t}\log q(x_t)=-\frac{\boldsymbol\epsilon^\star(x_t,t)}{\sqrt{1-\bar\alpha_t}}.}
$$

**Training a denoiser *is* learning the score of the noised data distribution at every noise level** (Vincent, 2011; Song & Ermon, 2019).

**Tweedie's formula.** Rearranging, the *posterior mean of the clean data* given the noisy observation is

$$
\E[x_0\mid x_t]=\frac{1}{\sqrt{\bar\alpha_t}}\Big(x_t+(1-\bar\alpha_t)\nabla\log q(x_t)\Big).
$$

(For Gaussian noise of variance $\sigma^2$ added to $x_0$: $\E[x_0\mid x_t]=x_t+\sigma^2\nabla\log p_\sigma(x_t)$.) *Check on a closed-form case*: $x_0\sim\Normal(0,s_0^2)$, $x_t=x_0+\sigma\epsilon$. The noisy marginal is $\Normal(0,s_0^2+\sigma^2)$ with score $-x_t/(s_0^2+\sigma^2)$, so $x_t+\sigma^2\cdot\text{score}=x_t\,s_0^2/(s_0^2+\sigma^2)$, which is the familiar Gaussian posterior mean. The code confirms both sides equal $1.79066$ for $s_0=1.5$, $\sigma=0.8$, $x_t=2.3$.

**Interpretation for biology.** The denoiser at noise level $\sigma$ is the *optimal estimator of the clean object from a corrupted version*, in the Bayes sense: it embeds *everything the data distribution knows* about how structures, sequences, or cells fit together at that scale. At high noise it must reconstruct global arrangement (fold topology; cell-type identity); at low noise it refines local geometry (bond lengths; fine expression structure). This **coarse-to-fine** decomposition by noise level is a natural fit for hierarchical structures such as proteins and is one reason diffusion works for structure.

---

## 15.4 The continuous-time view: SDEs and the probability-flow ODE

Let the forward process be an SDE $dx=f(x,t)\,dt+g(t)\,d\mathbf{w}$ (Song et al., 2021). The DDPM forward process is the discretization of the **variance-preserving SDE** $dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}\,d\mathbf{w}$. Two classical results:

1. **Reverse-time SDE** (Anderson, 1982): the same marginals $p_t$ are traced backward in time by

$$
dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]\,dt+g(t)\,d\bar{\mathbf{w}}.
$$

The *only* unknown is the score $\nabla_x\log p_t$, which the denoiser supplies (§15.3).

2. **Probability-flow ODE.** The marginal densities of the SDE obey the Fokker–Planck equation

$$
\partial_tp_t=-\nabla\!\cdot\!(fp_t)+\tfrac12g^2\Delta p_t .
$$

Since $\Delta p=\nabla\cdot(p\nabla\log p)$, this equals $\partial_tp_t=-\nabla\cdot\big[(f-\tfrac12g^2\nabla\log p_t)p_t\big]$, a *continuity equation* of the deterministic flow

$$
\frac{dx}{dt}=f(x,t)-\tfrac12g(t)^2\nabla_x\log p_t(x).
$$

So a **deterministic ODE** with the same marginals exists. DDIM (Song et al., 2021) is an Euler-type discretization of this ODE, allowing 10–50 steps instead of 1,000 and giving a *deterministic map* between noise and samples (hence an encoder, and exact likelihoods via the change-of-variables formula of Chapter 14, §14.4).

**Practical consequences.** Sampling is *ODE/SDE integration*: use better solvers (Heun, DPM-solver) for fewer steps; the stochastic sampler adds noise that corrects errors; the ODE sampler is deterministic and smooth. A trained model can be used for *likelihood evaluation* (exact via the ODE and the Hutchinson trace estimator, or via the ELBO).

---

## 15.5 Flow matching

### 15.5.1 Simulation-free training of a continuous flow

A **continuous normalizing flow** transports a simple distribution $p_0$ (noise) to the data distribution $p_1$ by an ODE $\dot x=v_\theta(x,t)$. Training CNFs by maximum likelihood requires simulating the ODE; **flow matching** (Lipman et al., 2023; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023) avoids this by regression.

Choose a **conditional path** from a noise sample $x_0\sim\Normal(0,\mathbf{I})$ to a data sample $x_1\sim p_\text{data}$ — the simplest is the straight line $x_t=(1-t)x_0+t\,x_1$, $t\in[0,1]$, whose conditional velocity is constant: $u_t(x_t\mid x_0,x_1)=x_1-x_0$. Train

$$
\mathcal{L}_\text{CFM}(\theta)=\E_{t,\,x_0,\,x_1}\Big[\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|^2\Big].
$$

**Why regressing onto a conditional target learns the marginal velocity.** The *marginal* path $p_t(x)=\E_{x_0,x_1}[\delta(x-x_t)]$ is generated by the marginal velocity field

$$
u_t(x)=\E\big[x_1-x_0\ \big|\ x_t=x\big].
$$

(*Sketch*: each conditional path satisfies a continuity equation with its conditional velocity; the continuity equation is linear in the density, so the mixture satisfies it with the posterior-averaged velocity.) The squared-error minimizer over $v_\theta$ is the conditional mean of the target given $(x_t,t)$, which is exactly $u_t(x)$ (Chapter 7: Bayes predictor). So **regressing on the easy conditional velocity trains the network to equal the hard marginal velocity**, with no ODE simulation during training. [[E]]

**Sampling:** integrate $\dot x=v_\theta(x,t)$ from $t=0$ ($x\sim\Normal(0,\mathbf{I})$) to $t=1$ with an ODE solver. In the code, 50 Euler steps from noise.

**Relation to diffusion.** Diffusion models correspond to choosing a *Gaussian path* $x_t=\alpha_tx_1+\sigma_tx_0$ with a particular schedule and a different (but linearly related) parameterization of the same regression target; flow matching with straight paths is the "rectified" special case. In practice: *simpler* (no variance schedule to tune), *straighter* trajectories (fewer solver steps), and *easy to extend to other spaces* (Riemannian manifolds, simplices) by redefining the path. [[S]]

### 15.5.2 Results on the toy problem

On four Gaussian modes ($\mathrm{sd}=0.3$), both models trained for 5,000 steps with a small MLP produce all four modes with near-uniform coverage:

```text
DDPM (200 steps), unconditional    mode coverage [0.25 0.25 0.26 0.24]  within-mode sd 0.349 (true 0.3)  mean dist to nearest mode 0.433
flow matching (50 Euler steps)     mode coverage [0.26 0.29 0.22 0.24]  within-mode sd 0.343 (true 0.3)  mean dist to nearest mode 0.390
```

Coverage of every mode with approximately correct proportions is the property that GANs struggle with (Chapter 14), and it arises here because the loss is a regression that must be correct at every sample, not an adversarial game. (Within-mode sd is slightly above the true 0.3: finite training and finite solver steps.)

---

## 15.6 Conditioning and guidance

### 15.6.1 Classifier guidance and classifier-free guidance

To sample $x\sim p(x\mid y)$ for a condition $y$ (a pocket, a property value, a cell type), we need the *conditional score*. By Bayes, $\nabla\log p(x\mid y)=\nabla\log p(x)+\nabla\log p(y\mid x)$.

- **Classifier guidance** (Dhariwal & Nichol, 2021) adds the gradient of a separately trained noise-aware classifier, scaled: $\nabla\log p(x)+(1+w)\nabla\log p(y\mid x)$, which targets the *sharpened* density $p(x)\,p(y\mid x)^{1+w}$.
- **Classifier-free guidance (CFG)** (Ho & Salimans, 2022) trains a single network on $y$ with random label dropout (15% in the code), so that it learns both $\epsilon_\theta(x_t,y)$ and the unconditional $\epsilon_\theta(x_t,\varnothing)$. Since $\nabla\log p(y\mid x)=\nabla\log p(x\mid y)-\nabla\log p(x)$, the guided noise estimate is

$$
\tilde{\boldsymbol\epsilon}=(1+w)\,\boldsymbol\epsilon_\theta(x_t,y)-w\,\boldsymbol\epsilon_\theta(x_t,\varnothing),
$$

which targets $p(x\mid y)^{1+w}\,p(x)^{-w}$: *the conditional distribution sharpened relative to the unconditional.* Larger $w$ increases conditional fidelity at the cost of diversity.

**Biological uses.** Conditioning on a **target pocket or binding partner** (RFdiffusion binder design), on a **partial structure** (motif scaffolding), on a **property** (expression level, solubility, binding affinity), on **perturbation identity and cell context** (virtual-cell generators; Chapter 39).

### 15.6.2 A cautionary experiment: over-guidance

The code samples from the class-conditional model targeting mode 0, $(+2,+2)$, using increasing guidance weight $w$:

| $w$ | fraction within 1.0 of target mode | sd around the mode |
|---|---|---|
| 0.0 | 1.000 | 0.272 |
| 1.0 | 0.999 | 0.261 |
| 3.0 | 0.996 | 0.267 |
| 6.0 | **0.583** | 0.280 |

Two lessons. (i) Here the conditional model is already near-perfect at $w=0$ (the classes are easy), so guidance **adds nothing**. In a harder, weakly conditioned problem it would sharpen. (ii) At $w=6$ **41.7% of samples leave the target mode**: guidance extrapolates *beyond* the data distribution ($\tilde{\boldsymbol\epsilon}$ is a linear extrapolation away from the unconditional prediction), so samples overshoot into regions the model never saw. This is a toy version of the failure of over-guided biomolecular samples: **high guidance weight produces designs that look extreme to the conditioning signal but are off the manifold of valid molecules** (non-foldable, strained geometry). Tuning $w$ requires *independent validation*, not just agreement with the conditioning model (Worked Example 15.1).

```python
--8<-- "code/ch15_diffusion.py"
```

Output:

```text
forward marginal after 40 steps: simulated mean [ 1.177 -0.634] var [0.181 0.181]; closed form mean [ 1.176 -0.633] var 0.181
posterior mean of x_(t-1): formula 1.69270, numerical Bayes 1.69270
Tweedie: posterior mean s0^2/(s0^2+sig^2)*xt = 1.79066; xt + sig^2*score = 1.79066
DDPM (200 steps), unconditional    mode coverage [0.25 0.25 0.26 0.24]  within-mode sd 0.349 (true 0.3)  mean dist to nearest mode 0.433
flow matching (50 Euler steps)     mode coverage [0.26 0.29 0.22 0.24]  within-mode sd 0.343 (true 0.3)  mean dist to nearest mode 0.390
classifier-free guidance, DDPM, target mode 0 = (+2,+2):
  w = 0.0: fraction within 1.0 of target mode = 1.000; sd around the mode = 0.272
  w = 1.0: fraction within 1.0 of target mode = 0.999; sd around the mode = 0.261
  w = 3.0: fraction within 1.0 of target mode = 0.996; sd around the mode = 0.267
  w = 6.0: fraction within 1.0 of target mode = 0.583; sd around the mode = 0.280
```

---

## 15.7 Diffusion for discrete data: sequences and masked diffusion

Gaussian noise does not apply to categorical tokens. **Discrete diffusion** replaces it with a corruption process on tokens: at each step, each token is resampled according to a transition matrix $\mathbf{Q}_t$ (D3PM; Austin et al., 2021). The most successful choice for text and biological sequences is the **absorbing-state (masking) process**: each token independently survives with probability $\alpha_t$ and otherwise becomes a special $[\mathrm{MASK}]$ token, with $\alpha_0=1$ and $\alpha_1=0$. The reverse model predicts the clean token at each masked position given the unmasked ones. The ELBO simplifies to a time-weighted **masked-language-modeling** loss:

$$
\mathcal{L}=\int_0^1\frac{-\alpha_t'}{1-\alpha_t}\ \E_{x_0,\,x_t}\Big[\sum_{i:\,x_t^i=[\mathrm{M}]}-\log p_\theta\big(x_0^i\mid x_t\big)\Big]\,dt
$$

(Sahoo et al., 2024; Shi et al., 2024).

**Relation to Chapter 13.** The BERT/ESM masked-LM loss with a *fixed* 15% mask rate is a single slice of this integral. Training with a *random* mask rate and the right weighting turns the masked LM into a **valid generative model whose ELBO bounds the likelihood**, sampled by starting from an all-masked sequence and iteratively unmasking. Thus **protein language models trained with variable-ratio masking are generative diffusion models** (e.g., EvoDiff, Alamdari et al., 2023; DPLM, Wang et al., 2024), and the sampling of a masked LM by iterative re-masking is a heuristic version of the reverse process. [[S]]

**Continuous relaxations.** *Dirichlet flow matching* and related methods place sequences on the probability simplex and define flows there (Stark et al., 2024, with applications to DNA enhancer design); *latent diffusion* diffuses in the continuous embedding space of a pretrained encoder. These are active research areas without a clear winner. [[H]]

---

## 15.8 Diffusion on structures: symmetry and manifolds

Molecules and proteins live in 3-D space, and the physics has symmetries: rotations and translations do not change a structure's identity. Two ways to handle this:

**Equivariant networks and invariant priors.** If the prior $p_T$ is rotation-invariant (an isotropic Gaussian) and the denoiser is **rotation-equivariant** ($s_\theta(\mathbf{R}x)=\mathbf{R}\,s_\theta(x)$), then the reverse-time dynamics commute with rotations and the generated distribution $p_0$ is rotation-invariant (Köhler et al., 2020; Xu et al., 2022). Translations are handled by working with centered coordinates (a Gaussian on the zero-center-of-mass subspace). Equivariant diffusion was developed for small molecules (Hoogeboom et al., 2022; Chapter 16).

**Data augmentation.** AlphaFold 3's diffusion module operates on raw atom coordinates and, rather than enforcing equivariance in the architecture, relies on random rotation and translation augmentation during training (Abramson et al., 2024). [[S]] This simplifies the network and uses the transformer's capacity to learn symmetry.

**Riemannian/Lie-group diffusion.** For protein backbones represented as sequences of rigid frames $(\mathbf{R}_i,\mathbf{t}_i)\in SE(3)^N$, diffusion is defined on the manifold: Gaussian noise on translations; *isotropic Gaussian on $SO(3)$* (IGSO(3)) for rotations (FrameDiff, Yim et al., 2023; RFdiffusion, Watson et al., 2023, which fine-tunes the RoseTTAFold structure network as the denoiser). *Torsional diffusion* operates on bond torsions for small-molecule conformers (Jing et al., 2022). *Flow matching on $SE(3)$* (FoldFlow, FrameFlow) gives the same advantages with straighter paths.

**Catalogue of biological uses** (to be treated in depth in Chapters 35–37, 39):

| System | Space | Noising process | Notes |
|---|---|---|---|
| AlphaFold 3 (2024) | atom coordinates (all-atom, multi-molecule) | Gaussian; EDM-style noise levels | Diffusion head conditioned on a trunk (Pairformer) |
| RFdiffusion (2023), RFdiffusion All-Atom (2024), RFdiffusion3 (open-sourced Dec 2025) | backbone frames $\to$ all-atom | $SE(3)$ / Gaussian | Binder design, motif scaffolding; RFD3 designs against DNA, small molecules, enzymes |
| Chroma (Ingraham et al., 2023) | backbone + sequence | correlated Gaussian respecting chain statistics | Programmable conditioning |
| Boltz-1/2, Chai-1/2 | all-atom complexes | diffusion/flow | Cofolding and design (Chai-2: zero-shot antibody design with reported ~16% hit rates in a 2025 preprint) |
| DiffDock (Corso et al., 2023) | ligand pose on $\R^3\times SO(3)\times$ torsions | product-manifold diffusion | Blind docking |
| EvoDiff (2023) | amino-acid sequences | discrete (masking, OADM) | Sequence-space generation, conditional on MSA |
| Perturbation-response generators | expression vectors | flow matching in expression/latent space | e.g., a flow-matching winner of the Generalist prize in the 2025 Virtual Cell Challenge (Chapter 39) |

---

## 15.9 Worked research examples

!!! example "Worked Research Example 15.1: Choosing the guidance weight for a binder-design diffusion model"
    **Situation.** A diffusion model generates protein binders conditioned on a target surface. Increasing the guidance weight raises the in silico predicted binding score of the generated designs monotonically. The authors choose the weight that maximizes it.

    **Question.** Why is "maximize the predicted score" the wrong selection criterion, and what should replace it?

    **Reasoning.**

    1. *What does guidance do?* $\tilde{\boldsymbol\epsilon}$ extrapolates *away* from the unconditional prediction toward the conditional one (§15.6). As $w$ grows, samples are pushed to extreme values of whatever the conditioning signal represents, then beyond the data manifold (toy: 41.7% of samples leave the target mode at $w=6$).
    2. *What is the predicted binding score?* A learned model's output. At extreme $w$ the designs are outside the predictor's training distribution, so its scores are unreliable and tend to be *optimistic for exactly the designs where the guidance has exploited it* (Chapter 14, §14.7).
    3. *What other metrics trend with $w$?* *Diversity* decreases; *foldability/self-consistency* (does an independent structure predictor refold the designed sequence into the designed backbone?) rises then falls; *novelty* may fall (mode-seeking); *experimental hit rate* is expected to show an inverted U.
    4. *Experiments.* (a) Plot at least four quantities versus $w$: predicted binding, self-consistency, diversity, off-manifold diagnostics (e.g., steric clashes, torsion outliers); (b) use an **independent** predictor (different architecture and training data) as a held-out judge; (c) *synthesize and test* designs at three guidance levels (low, mid, high) with random draws, not top-ranked picks; (d) include unguided samples as a control for "just sampling many designs and filtering."
    5. *Prediction.* If the inverted-U hypothesis holds, wet-lab success peaks at an intermediate $w$, and filtering ($w=0$ plus a good ranker) may match or beat heavy guidance.

    **Expert analysis.** Guidance is a knob that converts *diversity into conditional fidelity until it breaks the manifold*. The right criterion is the *experimental success rate per design* (Chapter 36), with independent in silico filters as a proxy and the explicit recognition that *the conditioning model and the evaluation model must not be the same.* The toy shows the mechanism in two dimensions: guidance is not a free lunch.

!!! example "Worked Research Example 15.2: A diffusion model generates \"designable\" backbones. Is the generator good?"
    **Situation.** A backbone generator is evaluated by **self-consistency designability**: for each generated backbone, design sequences with an inverse-folding model (Chapter 36), predict their structures with a structure predictor, and call the backbone *designable* if the predicted structure is within 2 Å RMSD of the generated backbone. The generator is designable 85% of the time.

    **Question.** What does "85% designable" establish?

    **Reasoning.**

    1. *What does the metric test?* That *some* inverse-folding model and *some* structure predictor agree that the backbone is consistent with a sequence. It is a *self-consistency* test within a computational pipeline.
    2. *Circularity and shared biases.* Inverse-folding and structure-prediction models are trained on the same PDB; designs that look like PDB-like regularities pass. A backbone made of idealized helical bundles will be designable *and* uninteresting (low novelty, low diversity, trivial topology).
    3. *Alternative explanations for 85%.* (H1) The generator learned real protein-like geometry. (H2) It mostly produces easy-to-design topologies (helical bundles). (H3) The predictor is *overconfident*, refolding any helix-rich sequence into the intended backbone.
    4. *Experiments.* (a) Report designability **stratified by secondary-structure content and topology novelty** (distance to nearest PDB fold); (b) check **diversity** (pairwise TM-score clustering); (c) compare to **non-diffusion baselines** and to *random idealized backbones*; (d) for a subset, **express and characterize** (monodispersity, circular dichroism, thermal stability, structure by crystallography/cryo-EM).
    5. *Predictions.* Under H2 the designability drops sharply for $\beta$-rich and novel topologies; under H3 experimental success rate is much lower than 85%.

    **Expert analysis.** Computational designability is a *necessary-condition filter*, not a measure of success. The only generation-quality statement that survives is **experimental success rate on novel, diverse, functional designs**. The reader's habit: *for any generative evaluation, ask which components of the pipeline are shared between generation and evaluation.*

---

## 15.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: making the forward process biological"
    **Observation.** Standard diffusion corrupts data with isotropic Gaussian noise (continuous) or uniform masking (discrete). These are *mathematically convenient*, not biologically motivated.

    **The idea (cross-domain transfer and biological constraint; attacks A7 and A9).** *The forward process is a modeling choice, and biology supplies its own natural noising processes.*

    | Data | Natural corruption | What the denoiser then learns | Open question |
    |---|---|---|---|
    | Protein sequence | **Evolutionary substitution process** (BLOSUM/PAM-like Markov chain) rather than uniform masking | To undo *plausible* mutations: generation as "reverse evolution" | Does the learned reverse process act as a fitness-aware sampler? |
    | Protein structure | **Thermal fluctuations** (Boltzmann noise) | Distribution of conformational ensembles | Can the noise level be tied to temperature (physical meaning)? |
    | Single-cell expression | **Count sampling noise** (Poisson/NB downsampling), dropout | Underlying expression state, denoised | Is the noising process matched to the measurement model (Chapter 4)? |
    | Cell state over time | **Developmental/perturbation trajectories** | Dynamics (Schrödinger-bridge or flow matching between observed distributions) | Can one learn flows between *unpaired* snapshots? |
    | DNA | **Mutation spectrum** (transitions > transversions, CpG decay) | Mutation-aware sequence prior | Does it improve variant-effect calibration? |

    **Hidden assumptions to test.** That the chosen process leaves the *signal* intact at low noise and destroys *only nuisance* at high noise (compare with the forgetting audit of Chapter 13); that the model can invert it with the available data.

    **A discriminating experiment.** Train two discrete diffusion models on the same protein-family data, one with uniform masking and one with an evolution-like substitution kernel. Compare (i) held-out ELBO, (ii) zero-shot mutation-effect correlation (DMS), (iii) diversity and fitness of conditional samples. If the substitution-kernel model improves DMS correlation at equal ELBO, the corruption process injected useful biology. If not, the choice is cosmetic. [[X]] (a candidate research direction, not an established result).

---

## 15.11 Connections

- **Backward:** the DDPM ELBO is Chapter 8's variational bound with a fixed encoder; the posterior calculation and Tweedie rely on Chapter 4's Gaussian algebra; the score is Chapter 14's EBM score; the probability-flow ODE is the change-of-variables flow of Chapter 14, continuous in time; masked diffusion extends Chapter 13's masked LM.
- **Forward:** equivariant networks for 3-D data (Chapter 16); AlphaFold 3 and its open reproductions (Chapter 35); protein and biomolecular design (Chapter 36); molecular generation and docking (Chapter 37); perturbation-response generators (Chapter 39); evaluation pitfalls (Chapters 14, 43, 46); the objective and cross-domain attacks (Chapter 55).

!!! takeaways "Key takeaways"
    1. **Forward process** $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ has an exact one-step form; the **reverse posterior** $q(x_{t-1}\mid x_t,x_0)$ is Gaussian with mean $\tilde{\boldsymbol\mu}_t$ and variance $\tilde\beta_t$.
    2. The **DDPM ELBO** is a sum of Gaussian KLs; with the noise parameterization it reduces to the **simplified loss** $\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\|^2$.
    3. **Noise prediction = score estimation** ($\nabla\log q(x_t)=-\boldsymbol\epsilon^\star/\sqrt{1-\bar\alpha_t}$); **Tweedie:** $\E[x_0\mid x_t]=(x_t+(1-\bar\alpha_t)\nabla\log q)/\sqrt{\bar\alpha_t}$.
    4. **Reverse SDE** needs only the score; the **probability-flow ODE** $\dot x=f-\tfrac12g^2\nabla\log p_t$ has the same marginals (Fokker–Planck) and enables fast, deterministic sampling and exact likelihoods.
    5. **Flow matching** regresses onto the conditional velocity $x_1-x_0$ and learns the marginal velocity; simple, simulation-free, and easy to extend to manifolds.
    6. **Guidance** $\tilde{\boldsymbol\epsilon}=(1+w)\boldsymbol\epsilon(y)-w\boldsymbol\epsilon(\varnothing)$ samples $p(x\mid y)^{1+w}p(x)^{-w}$: fidelity up, diversity down, and *off-manifold at large $w$* (58% on target at $w=6$ in the toy).
    7. **Masked diffusion** with a random mask rate is a generative masked LM; **equivariance** (or augmentation) and **manifold structure** handle 3-D symmetry.
    8. Evaluation of generated designs needs **independent** judges and wet-lab success rates; *which components do generation and evaluation share?*

---

## Further reading

- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *ICML*. Ho, J., Jain, A. & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation* 23, 1661–1674. Song, Y. & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*. Efron, B. (2011). Tweedie's formula and selection bias. *J. Am. Stat. Assoc.* 106, 1602–1614.
- Song, Y. et al. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*. Song, J., Meng, C. & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*. Anderson, B. D. O. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications* 12, 313–326. Karras, T., Aittala, M., Aila, T. & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS*.
- Lipman, Y. et al. (2023). Flow matching for generative modeling. *ICLR*. Liu, X., Gong, C. & Liu, Q. (2023). Flow straight and fast: learning to generate and transfer data with rectified flow. *ICLR*. Albergo, M. S. & Vanden-Eijnden, E. (2023). Building normalizing flows with stochastic interpolants. *ICLR*.
- Dhariwal, P. & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. *NeurIPS*. Ho, J. & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS Workshop*.
- Austin, J., Johnson, D. D., Ho, J., Tarlow, D. & van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. *NeurIPS*. Sahoo, S. S. et al. (2024). Simple and effective masked diffusion language models. *NeurIPS*. Shi, J. et al. (2024). Simplified and generalized masked diffusion for discrete data. *NeurIPS*.
- Hoogeboom, E., Satorras, V. G., Vignac, C. & Welling, M. (2022). Equivariant diffusion for molecule generation in 3D. *ICML*. Köhler, J., Klein, L. & Noé, F. (2020). Equivariant flows. *ICML*. Xu, M. et al. (2022). GeoDiff. *ICLR*. Jing, B., Corso, G., Chang, J., Barzilay, R. & Jaakkola, T. (2022). Torsional diffusion for molecular conformer generation. *NeurIPS*.
- Yim, J. et al. (2023). SE(3) diffusion model with application to protein backbone generation. *ICML*. Watson, J. L. et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature* 620, 1089–1100. Ingraham, J. B. et al. (2023). Illuminating protein space with a programmable generative model. *Nature* 623, 1070–1078. Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500. Corso, G., Stärk, H., Jing, B., Barzilay, R. & Jaakkola, T. (2023). DiffDock. *ICLR*.
- Alamdari, S. et al. (2023). Protein generation with evolutionary diffusion: sequence is all you need. *bioRxiv*. Stark, H. et al. (2024). Dirichlet flow matching with applications to DNA sequence design. *ICML*.
