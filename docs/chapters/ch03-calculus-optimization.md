# Chapter 3. Calculus and Optimization: How Models Learn

!!! abstract "Chapter at a glance"
    **Motivation.** Training a model means adjusting millions to trillions of parameters so that a loss decreases. Whether that works, how fast, and *what solution it finds* are questions of calculus and optimization. These determine many of the failures that later look like "the biology is hard."
    **Prerequisites.** Chapter 2; single-variable calculus.
    **You will be able to:** (1) compute gradients and Jacobians and apply the multivariate chain rule; (2) derive the softmax cross-entropy gradient $\mathbf{p}-\mathbf{y}$; (3) analyze gradient descent on a quadratic and explain the role of the condition number; (4) explain momentum, Adam, warmup, and the noise of SGD; (5) show that gradient descent on an underdetermined least-squares problem finds the minimum-norm solution (implicit regularization); (6) diagnose "my loss won't go down" versus "my loss can't go down."


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M3](m03-single-variable-calculus.md) (derivatives, the chain rule, Taylor series) and [M4](m04-multivariable-calculus-optimization.md) (gradients, Hessians, gradient descent).

---

## 3.1 Why optimization is a scientific question, not just an engineering one

A trained model is not "the best model in the hypothesis class." It is the model that a *particular optimizer* reached from a *particular initialization* using a *particular data order* in a *finite* budget. Two training runs that reach the same training loss can generalize very differently. When a biological foundation model underperforms a simple baseline, one of the explanations that must be ruled out is that it was *under-optimized*, that it found a poor solution, or that the loss it optimizes is dominated by uninteresting parts of the data.

This chapter therefore teaches optimization as something you *reason about*, with three recurring questions:

1. **What direction does the gradient point, and what information does it carry?** (calculus)
2. **How far and how reliably can we move along it?** (conditioning, noise, step sizes)
3. **Among all the solutions that fit the data, which does the algorithm pick?** (implicit bias)

---

## 3.2 From derivatives to gradients and Jacobians

### 3.2.1 Definitions

For $f:\R\to\R$, the derivative $f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$ is the slope of the best linear approximation. For a scalar function of a vector, $f:\R^n\to\R$, the **gradient** collects partial derivatives:

$$
\nabla f(\mathbf{x})=\Big(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\Big)^\top\in\R^n .
$$

For a vector-valued function $\mathbf{f}:\R^n\to\R^m$, the **Jacobian** $\mathbf{J}\in\R^{m\times n}$ has entries $J_{ij}=\partial f_i/\partial x_j$, and gives the best linear approximation $\mathbf{f}(\mathbf{x}+\boldsymbol{\delta})\approx\mathbf{f}(\mathbf{x})+\mathbf{J}\boldsymbol{\delta}$. The **Hessian** of a scalar function is the matrix of second derivatives $H_{ij}=\partial^2f/\partial x_i\partial x_j$, symmetric under mild conditions.

### 3.2.2 The gradient is the direction of steepest ascent

The **directional derivative** of $f$ at $\mathbf{x}$ along a unit vector $\mathbf{u}$ is $D_\mathbf{u}f=\nabla f(\mathbf{x})^\top\mathbf{u}$. By Cauchy–Schwarz, $\nabla f^\top\mathbf{u}\le\|\nabla f\|$, with equality iff $\mathbf{u}=\nabla f/\|\nabla f\|$. So the gradient points in the direction of steepest increase and $-\nabla f$ in the direction of steepest decrease, *measured in the Euclidean metric*. This caveat matters: with a different notion of distance (for instance, distance between probability distributions), the steepest direction is different, giving the *natural gradient* (see §3.9).

### 3.2.3 Taylor expansion

To second order,

$$
f(\mathbf{x}+\boldsymbol{\delta})\approx f(\mathbf{x})+\nabla f(\mathbf{x})^\top\boldsymbol{\delta}+\tfrac12\boldsymbol{\delta}^\top\mathbf{H}(\mathbf{x})\boldsymbol{\delta}.
$$

Everything in this chapter is an exploitation of this expansion: the first-order term tells you which way to move; the second-order term tells you how far you can move before the first-order prediction fails.

### 3.2.4 The multivariate chain rule

If $\mathbf{y}=\mathbf{g}(\mathbf{x})$ and $\mathbf{z}=\mathbf{f}(\mathbf{y})$, then

$$
\mathbf{J}_{\mathbf{f}\circ\mathbf{g}}(\mathbf{x})=\mathbf{J}_\mathbf{f}(\mathbf{g}(\mathbf{x}))\,\mathbf{J}_\mathbf{g}(\mathbf{x}).
$$

For a scalar loss $\mathcal{L}=\ell(\mathbf{f}(\mathbf{g}(\mathbf{x})))$, the gradient is a *vector–Jacobian product*:

$$
\nabla_\mathbf{x}\mathcal{L}=\mathbf{J}_\mathbf{g}^\top\,\mathbf{J}_\mathbf{f}^\top\,\nabla_\mathbf{z}\ell .
$$

**This is backpropagation.** The gradient of a composition is obtained by starting at the output and repeatedly multiplying by transposed Jacobians. Chapter 9 turns this observation into the full algorithm. Note the *ordering* matters for cost: multiplying right-to-left (output first) carries a vector, costing $O(\text{size of Jacobians})$; multiplying left-to-right would carry a matrix. Reverse-mode automatic differentiation exploits exactly this asymmetry when the output is a scalar loss and inputs are millions of parameters.

---

## 3.3 Matrix calculus you actually need

A handful of identities cover most of machine learning. Here $\mathbf{a},\mathbf{w}\in\R^n$, $\mathbf{A}\in\R^{n\times n}$, $\mathbf{X}\in\R^{N\times n}$, $\mathbf{y}\in\R^N$.

| Function | Gradient w.r.t. $\mathbf{w}$ |
|---|---|
| $\mathbf{a}^\top\mathbf{w}$ | $\mathbf{a}$ |
| $\tfrac12\mathbf{w}^\top\mathbf{A}\mathbf{w}$ | $\tfrac12(\mathbf{A}+\mathbf{A}^\top)\mathbf{w}$ (equals $\mathbf{A}\mathbf{w}$ if symmetric) |
| $\tfrac12\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2$ | $\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})$ |
| $\tfrac\lambda2\lVert\mathbf{w}\rVert^2$ | $\lambda\mathbf{w}$ |

**Derivation of the least-squares gradient.** $\tfrac12\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2=\tfrac12\mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}-\mathbf{y}^\top\mathbf{X}\mathbf{w}+\tfrac12\mathbf{y}^\top\mathbf{y}$. Apply the first two identities with $\mathbf{A}=\mathbf{X}^\top\mathbf{X}$ (symmetric) and $\mathbf{a}=\mathbf{X}^\top\mathbf{y}$ to get $\mathbf{X}^\top\mathbf{X}\mathbf{w}-\mathbf{X}^\top\mathbf{y}$. $\square$

### 3.3.1 The softmax cross-entropy gradient

This is the most-used derivative in classification, language modeling, and masked-token pretraining. Let $\mathbf{z}\in\R^K$ be logits, $p_i=e^{z_i}/\sum_je^{z_j}$ the softmax, and $\mathcal{L}=-\log p_y$ the loss for the true class $y$.

First, the softmax Jacobian: $\dfrac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$. *Proof.* If $i=j$: $\partial_{z_i}\frac{e^{z_i}}{S}=\frac{e^{z_i}}{S}-\frac{e^{z_i}e^{z_i}}{S^2}=p_i(1-p_i)$. If $i\ne j$: $\partial_{z_j}\frac{e^{z_i}}{S}=-\frac{e^{z_i}e^{z_j}}{S^2}=-p_ip_j$. $\square$

Then

$$
\frac{\partial\mathcal{L}}{\partial z_j}=-\frac1{p_y}\frac{\partial p_y}{\partial z_j}=-\frac1{p_y}p_y(\delta_{yj}-p_j)=p_j-\delta_{yj}.
$$

In vector form, $\nabla_\mathbf{z}\mathcal{L}=\mathbf{p}-\mathbf{e}_y$: *the predicted distribution minus the one-hot truth.* The gradient is bounded in $[-1,1]$ per coordinate, vanishes when the model is confident and correct, and is largest when the model is confident and wrong. Two consequences that will matter later:

1. **Easy tokens contribute no gradient.** When $p_y\approx1$, the gradient is $\approx0$. In a genomic language model, positions the model already predicts well (repeats, low-complexity sequence) stop driving learning; positions where $p$ is inherently spread out (high-entropy) *always* contribute a nonzero gradient but one that cannot reduce the loss below its irreducible level (§3.10).
2. **The numerical gradient check works.** One can verify any implementation of this loss against central finite differences $\big(\mathcal{L}(z_i+\epsilon)-\mathcal{L}(z_i-\epsilon)\big)/2\epsilon$. The code at the end of the chapter does this and finds agreement to $10^{-10}$. **Always gradient-check new loss functions and layers.**

---

## 3.4 Gradient descent and the geometry of conditioning

### 3.4.1 The algorithm and a guarantee

Gradient descent (GD) updates $\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\nabla f(\mathbf{w}_t)$ with step size (learning rate) $\eta>0$.

Say $f$ is **$L$-smooth** if $\|\nabla f(\mathbf{u})-\nabla f(\mathbf{v})\|\le L\|\mathbf{u}-\mathbf{v}\|$ (the Hessian's eigenvalues are bounded by $L$ in magnitude). Then the quadratic upper bound $f(\mathbf{v})\le f(\mathbf{u})+\nabla f(\mathbf{u})^\top(\mathbf{v}-\mathbf{u})+\frac L2\|\mathbf{v}-\mathbf{u}\|^2$ holds. Substituting one GD step $\mathbf{v}=\mathbf{u}-\eta\nabla f(\mathbf{u})$:

$$
f(\mathbf{w}_{t+1})\le f(\mathbf{w}_t)-\eta\Big(1-\frac{L\eta}{2}\Big)\|\nabla f(\mathbf{w}_t)\|^2 .
$$

So for $\eta\le1/L$ the loss decreases by at least $\frac\eta2\|\nabla f\|^2$ at every step (the **descent lemma**). The learning rate is limited by the *largest curvature* $L$ in any direction.

### 3.4.2 The quadratic model and the condition number

Local behavior near a minimum is governed by the Hessian, so analyze $f(\mathbf{w})=\tfrac12\mathbf{w}^\top\mathbf{A}\mathbf{w}$ with $\mathbf{A}$ symmetric positive definite, eigenvalues $\mu=\lambda_{\min}\le\dots\le\lambda_{\max}=L$, minimizer $\mathbf{w}^\star=\mathbf{0}$. GD gives $\mathbf{w}_{t+1}=(\mathbf{I}-\eta\mathbf{A})\mathbf{w}_t$. In the eigenbasis of $\mathbf{A}$, the component along eigenvector $i$ evolves independently:

$$
c_i^{(t)}=(1-\eta\lambda_i)^t\,c_i^{(0)} .
$$

*Convergence in every direction* requires $|1-\eta\lambda_i|<1$ for all $i$, i.e., $\eta<2/L$. *Speed* is limited by the slowest direction: the contraction factor per step is $\rho(\eta)=\max_i|1-\eta\lambda_i|=\max\{|1-\eta\mu|,|1-\eta L|\}$. This is minimized when the two terms balance, $1-\eta\mu=\eta L-1$, giving

$$
\eta^\star=\frac{2}{L+\mu},\qquad \rho^\star=\frac{L-\mu}{L+\mu}=\frac{\kappa-1}{\kappa+1},\qquad\kappa=\frac L\mu .
$$

To reduce the error by a factor $\varepsilon$ requires $t\approx\ln(1/\varepsilon)/\ln(1/\rho^\star)\approx\frac\kappa2\ln\frac1\varepsilon$ iterations (for large $\kappa$). **The number of iterations scales with the condition number.**

**Intuition.** The loss surface is a long, narrow valley. A step size large enough to make progress along the flat floor of the valley (small curvature $\mu$) is too large for the steep walls (large curvature $L$), causing oscillation across the valley; a step size safe for the walls makes crawling progress along the floor.

**Biological reading.** Ill-conditioning is the optimization face of *correlated features*. In regression on SNPs in strong linkage disequilibrium, or on genes in a co-expressed module, the Hessian $\mathbf{X}^\top\mathbf{X}$ has tiny eigenvalues in directions that distinguish between correlated features. Those directions are exactly where the data carry little information (Chapter 2, §2.8), so optimization is slow *and* the solution is poorly determined there. The same fact shows up as the unidentifiability of individual effect sizes within an LD block (Chapter 26).

```python
--8<-- "code/ch03_optimization.py"
```

Output (seed 0):

```text
softmax-xent gradient check: max |analytic - numeric| = 9.87e-11

quadratic, kappa = 100: iterations to shrink ||w|| by 1e-6
  gradient descent (optimal step):    612   theory ~ ln(1/tol)/ln(1/rho) = 691  (rho = (k-1)/(k+1) = 0.980)
  heavy-ball momentum            :     86   (rate improves from ~kappa to ~sqrt(kappa))
  Adam (lr=0.1, no decay)        :    570   (may oscillate; shown for contrast)

underdetermined LS (n=30, p=200): train residual = 8.6e-17, ||w_GD - w_pinv|| = 1.1e-15, ||w_GD|| = 0.405
another interpolating solution: residual = 1.4e-13, ||w|| = 12.928  (larger norm; GD did not choose it)
```

The theoretical figure of 691 iterations is a worst-case bound that assumes the starting error lies entirely along the slowest eigendirection; a random start has less energy there, so the observed 612 is below it. Adam's count is shown only for contrast: it depends strongly on its learning rate and, without decay, oscillates around the minimum rather than converging linearly.

---

## 3.5 Momentum, Adam, and why deep-learning optimizers look the way they do

### 3.5.1 Heavy-ball momentum

Keep a running average of past gradients: $\mathbf{v}_{t+1}=\beta\mathbf{v}_t-\eta\nabla f(\mathbf{w}_t)$, $\mathbf{w}_{t+1}=\mathbf{w}_t+\mathbf{v}_{t+1}$. Oscillating components (across the valley walls) average out because they alternate in sign, while the consistent component along the valley floor accumulates. For quadratics with optimal $\eta,\beta$ the contraction factor improves from $(\kappa-1)/(\kappa+1)$ to $(\sqrt\kappa-1)/(\sqrt\kappa+1)$, i.e., iterations scale as $\sqrt\kappa$ instead of $\kappa$. In the simulation ($\kappa=100$) heavy-ball needs 86 iterations versus 612 for GD, close to the predicted $\sqrt{100}=10$-fold gain per unit of accuracy. [[E]] for quadratics.

### 3.5.2 Adam

Adam (Kingma & Ba, 2015) maintains exponential moving averages of the gradient ($\mathbf{m}$) and of its elementwise square ($\mathbf{v}$):

$$
\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t,\qquad
\mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^{\odot2},
$$

$$
\hat{\mathbf{m}}_t=\frac{\mathbf{m}_t}{1-\beta_1^t},\quad
\hat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^t},\qquad
\mathbf{w}_{t}=\mathbf{w}_{t-1}-\eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}.
$$

**Why the bias correction.** With $\mathbf{m}_0=\mathbf{0}$, if gradients had a stationary mean $\boldsymbol{\mu}_g$, then $\E[\mathbf{m}_t]=(1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}\boldsymbol{\mu}_g=(1-\beta_1^t)\boldsymbol{\mu}_g$. Dividing by $1-\beta_1^t$ removes the initialization bias. The same applies to $\mathbf{v}$.

**What Adam does.** Each coordinate's step is the (smoothed) gradient divided by the (smoothed) root-mean-square gradient: a *per-parameter adaptive learning rate*. A parameter with consistently large gradients takes smaller steps; a parameter that rarely receives gradient (an embedding row for a rare token, a rarely-used expert, a gene seen in few cells) gets a larger effective step. Since $\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$ is of order 1 when gradients are consistent, the typical per-step change is about $\eta$ regardless of gradient scale. Adam is therefore roughly *invariant to the scale of the loss* and *partly* to per-parameter scaling, which is why it is forgiving for transformers whose layers have very different gradient magnitudes.

**AdamW** decouples weight decay from the gradient: $\mathbf{w}\leftarrow\mathbf{w}-\eta\big(\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}}+\epsilon)+\lambda\mathbf{w}\big)$. In the plain L2-regularized version the decay term would be rescaled by the adaptive denominator; decoupling it makes the regularization strength independent of gradient history (Loshchilov & Hutter, 2019).

**Practical recipe for large models** [[S]]: AdamW with $\beta_1=0.9$, $\beta_2\in[0.95,0.999]$, weight decay $\sim0.01$–$0.1$; **learning-rate warmup** over the first $10^2$–$10^3$ steps (the second-moment estimate is unreliable early and curvature at initialization is high); cosine or linear decay to a small fraction of the peak; **gradient clipping** at global norm $\sim1$ to survive rare large-gradient batches; mixed-precision arithmetic (bf16) with float32 master weights.

!!! lens "Research lens: adaptive optimizers"
    **Assumes:** per-coordinate rescaling is a useful preconditioner; gradient noise is not too heavy-tailed. **Uses:** first and second moments of gradients. **Ignores:** correlations between coordinates (full-matrix curvature). **Fails when:** the loss is dominated by a few huge-gradient events (loss spikes); learning rates are too large near the end of training; generalization depends on the optimizer in ways the training loss hides.

---

## 3.6 Stochastic gradients: noise as a feature and a cost

The full loss is an average over data, $f(\mathbf{w})=\frac1N\sum_n\ell_n(\mathbf{w})$. Computing its gradient needs a pass over all $N$ examples. Instead sample a *mini-batch* $\mathcal{B}$ of size $B$ and use $\hat{\mathbf{g}}=\frac1B\sum_{n\in\mathcal{B}}\nabla\ell_n$. This is an **unbiased estimator**: $\E[\hat{\mathbf{g}}]=\nabla f$, with covariance $\mathbf{\Sigma}/B$ where $\mathbf{\Sigma}$ is the covariance of per-example gradients.

**Two regimes.** When the true gradient is large relative to noise ($\|\nabla f\|^2\gg\mathrm{tr}\,\mathbf{\Sigma}/B$), a larger batch just wastes compute (the *critical batch size* is where noise and signal balance). Near a minimum the true gradient vanishes while noise persists, so *the iterate diffuses in a cloud around the minimum* whose size scales as $\eta\,\mathbf{\Sigma}/B$. This is why decaying the learning rate at the end of training reduces the loss: it shrinks the cloud.

**The temperature picture.** A continuous-time approximation (heuristic, [[P]]) treats SGD as noisy gradient flow, $d\mathbf{w}=-\nabla f\,dt+\sqrt{\eta/B}\,\mathbf{\Sigma}^{1/2}d\mathbf{W}_t$, in which $\eta/B$ plays the role of a *temperature*. Higher temperature favors flatter, wider basins that the noise cannot escape from only if they are wide; lower temperature allows sharp minima. This is one line of explanation for why smaller batches or larger learning rates often generalize better, though it is not a complete account and has known counterexamples.

!!! rhyme "Structural rhyme: SGD noise ↔ genetic drift"
    In a finite population of effective size $N_e$, allele frequencies change by *selection* (a deterministic drift toward higher fitness) plus *genetic drift* (random sampling of gametes, with variance $\propto 1/N_e$). Evolution is a stochastic dynamic with exactly the structure of noisy gradient flow: deterministic force + noise whose magnitude is inverse in a population (batch) size. The stationary distribution of the diffusion approximation (Wright–Fisher; Chapter 21) is Boltzmann-like, with the selection coefficient multiplied by a constant times $N_e$ in the exponent (the constant depends on the fitness convention), mirroring the SGD stationary distribution $\propto e^{-c\,(B/\eta)f}$ that holds for isotropic noise of fixed scale. Practical consequences of the analogy: with small $N_e$ (small batch) slightly deleterious variants (sharp, bad minima) fix by chance; with large $N_e$, selection is efficient. The analogy is exact only for the idealized diffusions, but it makes a good mental model and appears again when we discuss evolutionary algorithms and directed evolution (Chapters 21, 42). A second, more precise connection: the *replicator equation* of population genetics is mirror-descent/multiplicative-weights on the simplex, i.e., a gradient flow on the log-fitness under the Fisher–Shahshahani metric ([[E]] mathematically, Harper 2009).

---

## 3.7 Non-convexity: what deep-network loss surfaces look like

For convex $f$ (e.g., least squares, logistic regression) every local minimum is global. Deep networks are non-convex, and yet GD routinely finds good solutions. What we know:

- **Saddle points, not bad local minima, dominate** the critical points of high-dimensional random functions: at a critical point with $n$ Hessian eigenvalues, the chance that all are positive (a local minimum) decays rapidly with $n$. [[S]] for deep networks (Dauphin et al., 2014).
- **Overparameterization smooths the landscape.** When parameters far exceed data, there is typically a connected, high-dimensional manifold of zero-loss solutions, and GD finds one (§3.8). [[S]]
- **Edge-of-stability.** In practice, full-batch GD often operates at a learning rate where the sharpness (largest Hessian eigenvalue) hovers near $2/\eta$: the loss is not monotone, but still decreases on average (Cohen et al., 2021). [[S]] This breaks the intuition behind the descent lemma ($\eta<2/L$) and shows that the dynamics *self-stabilize*.
- **Loss spikes** in large-scale training are common and typically attributable to optimizer-state or data-batch anomalies; remedies include lower learning rate, higher $\beta_2$, skipped batches, and careful initialization. [[S]] For genomic or single-cell data, unusual batches (extreme sequencing depth, degenerate sequences, long runs of N) are plausible triggers; inspect the batch that preceded a spike.

---

## 3.8 Implicit regularization: which solution does the algorithm choose?

When the number of parameters exceeds the number of constraints there are *infinitely many* zero-training-loss solutions. The optimizer picks one. That choice is a hidden regularizer, and it is a large part of why overparameterized models generalize.

!!! math "Derivation: gradient descent on underdetermined least squares finds the minimum-norm solution"
    Let $\mathbf{X}\in\R^{n\times p}$ with $p>n$ and full row rank. GD on $f(\mathbf{w})=\tfrac12\|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2$ from $\mathbf{w}_0=\mathbf{0}$ updates by $\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\mathbf{X}^\top(\mathbf{X}\mathbf{w}_t-\mathbf{y})$. Each update adds a vector in the row space of $\mathbf{X}$ (a linear combination of the rows $\mathbf{x}_n$). Starting from $\mathbf{0}$, every iterate therefore stays in $\mathrm{row}(\mathbf{X})$. If GD converges to an interpolating solution $\mathbf{w}_\infty$ ($\mathbf{X}\mathbf{w}_\infty=\mathbf{y}$), then $\mathbf{w}_\infty\in\mathrm{row}(\mathbf{X})$. The interpolating solutions form the affine set $\mathbf{w}_{\min}+\mathrm{null}(\mathbf{X})$, and $\mathrm{null}(\mathbf{X})\perp\mathrm{row}(\mathbf{X})$. Hence the *only* interpolating solution lying in the row space is the one with no null-space component: $\mathbf{w}_{\min}=\mathbf{X}^+\mathbf{y}$, the minimum-norm solution. $\square$

    The simulation confirms it: with $n=30$, $p=200$, $\|\mathbf{w}_{\text{GD}}-\mathbf{w}_{\text{pinv}}\|\approx10^{-15}$ and $\|\mathbf{w}_{\text{GD}}\|=0.405$, whereas another interpolating solution that adds a random null-space component has norm 12.9.

**Further known results.** On linearly separable data, GD on logistic regression converges in direction to the **maximum-margin** classifier (Soudry et al., 2018) [[E]]. In deep networks the implicit bias is richer and still being mapped (towards low-rank solutions in deep linear networks; sparse solutions with small initialization), but the principle stands: *the algorithm, initialization, and parameterization are part of the model.*

**Why this matters for biology.** In genomics $p\gg n$ is the norm: $10^6$ SNPs, $10^4$–$10^5$ individuals; $2\times10^4$ genes, a few hundred perturbations. The effective hypothesis class is not "all linear models" but "linear models that the training procedure favors." This is why simple regularized linear baselines are so hard to beat in $p\gg n$ biology (Chapter 7), and why claims that a deep model "beat" a baseline need to specify the *optimizer and regularization of both*.

---

## 3.9 Natural gradient and the metric you didn't choose

Steepest descent in Euclidean parameter space depends on how the model is *parameterized*. If you rescale parameters, "steepest" changes. For probabilistic models $p_\theta$ a more natural notion of distance is the KL divergence between distributions, whose local quadratic form is the **Fisher information matrix** $\mathbf{F}(\theta)=\E_{x\sim p_\theta}[\nabla\log p_\theta\,\nabla\log p_\theta^\top]$. The *natural gradient* update $\theta\leftarrow\theta-\eta\,\mathbf{F}^{-1}\nabla\mathcal{L}$ is invariant to reparameterization. Adam can be viewed as a diagonal, empirical approximation to this idea; K-FAC, Shampoo, and Muon use richer structure. You do not need these for most of the book, but you will meet the Fisher information again in statistical estimation (Chapter 4), in uncertainty (Chapter 18), and in the replicator-equation rhyme above.

---

## 3.10 Optimization diagnostics for research practice

A practical checklist, ordered from cheapest to most informative:

1. **Overfit a tiny dataset (one batch of 8–32 examples).** If the loss does not reach near zero, there is a bug (shape, gradient flow, learning rate, data pipeline), *not* a science problem.
2. **Gradient-check new components** on small inputs with finite differences.
3. **Sweep the learning rate on a log scale** ($10^{-5}$ to $10^{-1}$) for a few hundred steps. The best rate is typically just below the one where the loss becomes unstable.
4. **Monitor gradient norm, update-to-weight ratio** ($\|\Delta\mathbf{w}\|/\|\mathbf{w}\|\sim10^{-3}$ is a common healthy value), and per-layer activation statistics.
5. **Compare to the irreducible loss.** What is the entropy of the target distribution? If the model's loss is close to the Bayes-optimal loss, "the loss is high" may just mean "the task is noisy."
6. **Look at the loss per position/class/example**, not just the mean. A mean that plateaus can hide a subset that is still improving and a subset that cannot improve.
7. **Run multiple seeds** before concluding anything from a difference smaller than seed-to-seed spread.

!!! example "Worked Research Example 3.1: The loss is going down, but the biological metric is flat"
    **Situation.** You train a masked-language model on 1 Mb DNA windows from the human genome. Training and validation loss fall smoothly for days. You then probe the final-layer embeddings for enhancer-vs-non-enhancer classification and find essentially no improvement over a $k$-mer baseline, and the probe is not much better at checkpoint 1 than at checkpoint 100.

    **Question.** Is this an optimization failure or something else? How do you find out?

    **Reasoning.**

    1. *What does the loss measure?* Average per-nucleotide cross-entropy. In human DNA about half the genome is repeat-derived and large parts are low-complexity or highly compositional; most of the *achievable* loss reduction is in modeling these. Regulatory positions (a few percent of the genome) are a small fraction of tokens and have high intrinsic entropy.
    2. *What would the loss look like if the model had learned regulatory grammar?* A small additional decrease concentrated at a few percent of positions, invisible in the mean.
    3. *Candidate explanations.* (H-a) Under-trained: more steps will eventually help. (H-b) The loss is *dominated* by non-regulatory structure, so the representation is shaped by it (G-O). (H-c) The probe/readout cannot access regulatory features (probe too weak, or features distributed nonlinearly). (H-d) The benchmark labels are noisy or the enhancer set is biased, so the ceiling is low (G-M).
    4. *Discriminating experiments.* (i) Plot loss *per annotation class* (repeats, coding, promoters, enhancers) over training: if enhancer-position loss is flat while repeat loss falls, H-b is supported. (ii) Compare with a randomly initialized model with the same probe: if the gap is small, the pretraining objective contributed little (G-O). (iii) Fit a stronger probe (fine-tune end-to-end) to test H-c. (iv) Measure the replicate agreement of the labels to estimate the ceiling (H-d). (v) Extrapolate: does loss on enhancer positions follow a power law in compute, and what exponent? (Chapter 17.)
    5. *Predictions.* Under H-b, per-class loss curves diverge; reweighting the loss toward regulatory annotations (or masking repeats) improves the probe more than additional steps. Under H-a, per-class loss for regulatory positions is still dropping. Under H-c, fine-tuning closes the gap.

    **Expert analysis.** The mean validation loss is a *weighted average of tasks* with weights set by token frequency, not by biological importance. This is an **objective gap** (G-O) that no amount of optimizer tuning fixes. The right response is an *evaluation change* (per-class loss, matched probes, random-init controls), then, if confirmed, an *objective or data change* (reweighting, masking, curriculum, or supervised multi-task training; Chapters 31–32). It is a good example of why "is the optimizer working?" and "is the objective right?" must be answered by different experiments.

---

## 3.11 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"Undertrained, or impossible?\""
    **Observation.** A model for predicting the effect of a variant on expression plateaus at Spearman 0.3 on held-out data.

    **The two stories.** *Undertrained*: more compute, data, or tuning would raise it. *Impossible*: the task has an irreducible noise floor near 0.3 (or the necessary information is not in the input).

    **How an expert separates them.**

    1. **Estimate the ceiling.** Use technical and biological replicate correlation of the labels (Chapter 1). If replicate Spearman is 0.35, you are done.
    2. **Look at the learning curve.** Train on 10%, 25%, 50%, 100% of the data. If performance still rises roughly log-linearly with data, more data likely helps; if flat from 25%, data is not the limiting factor.
    3. **Look at capacity.** Scale parameters at fixed data. Flat → the model is not capacity-limited; rising → capacity or optimization is limiting.
    4. **Check the information.** Add a feature known to be causal (for example, the cell-type label or a measured chromatin state). A large jump says the *input* was missing information (G-M, G-O); no jump says the feature is already implicit or irrelevant.
    5. **Check the optimization.** Does a different optimizer, schedule, or initialization change the plateau? If many different setups plateau at the same value, the bound is not the optimizer.

    **Distinguishing fundamental limits from implementation problems.** Limits that persist across optimizers, capacities, and dataset sizes, and that match an independent estimate of the noise ceiling or an information bound, are *fundamental to the data/task*. Limits that move with hyperparameters are *implementation*. The skill is to design the cheapest experiment that moves one factor at a time.

---

## 3.12 Connections

- **Backward:** Chapter 2's SVD tells us the curvature spectrum is the singular-value spectrum of $\mathbf{X}$ for least squares; ill-conditioning is small singular values.
- **Forward:** Backpropagation (Chapter 9) is the chain rule as an algorithm. Probabilistic losses (Chapter 4) are negative log-likelihoods with gradients of the form "prediction minus truth." Generalization (Chapter 7) depends on the implicit bias shown here. Scaling laws (Chapter 17) describe how the *achieved* loss depends on compute, which is a statement about optimization at scale. The temperature/drift analogy returns in Chapters 21 and 42.

!!! takeaways "Key takeaways"
    1. The gradient of a composition is a product of transposed Jacobians; **backpropagation is the chain rule** organized to carry a vector (Chapter 9).
    2. For softmax cross-entropy, $\nabla_\mathbf{z}\mathcal{L}=\mathbf{p}-\mathbf{e}_y$. Easy examples contribute no gradient; always gradient-check new code.
    3. On a quadratic, gradient descent converges at rate $(\kappa-1)/(\kappa+1)$; iterations scale with the **condition number**, which in regression is the spread of singular values (correlated features → ill-conditioning). Momentum improves this to $\sqrt\kappa$.
    4. **Adam** is a per-parameter adaptive method with bias-corrected moment estimates; large-model recipes add warmup, decay, clipping, and decoupled weight decay.
    5. **SGD noise** acts like temperature, with $\eta/B$ playing the role that $1/N_e$ plays in population genetics.
    6. Among many zero-loss solutions, **the optimizer chooses**: GD from zero on underdetermined least squares gives the minimum-norm solution. The optimizer is part of the model.
    7. A falling loss does not mean the biological capability is improving: it is an average weighted by token frequency. Diagnose per-class and compare with the irreducible loss.

---

## Further reading

- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Free online.
- Nocedal, J. & Wright, S. (2006). *Numerical Optimization*. Springer.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Dauphin, Y. et al. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*.
- Cohen, J. M. et al. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR*.
- Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S. & Srebro, N. (2018). The implicit bias of gradient descent on separable data. *JMLR* 19, 1–57.
- Harper, M. (2009). The replicator equation as an inference dynamic. arXiv:0911.1763.
- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation* 10, 251–276.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, chapters 4, 8. MIT Press.
