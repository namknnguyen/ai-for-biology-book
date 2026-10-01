# Chapter M4. Multivariable Calculus and Optimization

!!! abstract "Chapter at a glance"
    **Motivation.** A biological model has many parameters, and a cell has thousands of measured variables. The calculus of one variable (Chapter M3) extends to many: the derivative becomes a *gradient* (for a scalar output) or a *Jacobian* (for a vector output), second derivatives become a *Hessian*, and the chain rule becomes a product of Jacobians. With these tools you can say what "training a model" means mathematically (follow the gradient downhill), why some problems train slowly (conditioning), and how constraints produce the most important distribution in statistical physics and machine learning (Boltzmann/softmax) in a few lines.

    **Prerequisites.** Chapter M3 (derivatives, chain rule, Taylor series). Comfort with vectors as lists of numbers; Chapter M5 deepens that, and a few matrix operations used here (a matrix times a vector, eigenvalues) are re-explained there.

    **You will be able to:** (1) compute partial derivatives, gradients and Jacobians, and state what each means; (2) apply the multivariable chain rule as a product of Jacobians and check it numerically; (3) classify critical points with the Hessian; (4) run gradient descent, choose a learning rate, and explain how the condition number controls speed; (5) derive least squares from a gradient; (6) solve a constrained optimization with a Lagrange multiplier and recognize the Boltzmann/softmax distribution; (7) interpret a multiple integral and change variables.

---

## M4.1 Functions of many variables, and partial derivatives

A function of several variables takes a vector in and gives a number out: $f:\R^{d}\to\R$. Examples: the loss of a model as a function of its $d$ parameters; the energy of a protein as a function of its atom coordinates; the expression of a gene as a function of the concentrations of its regulators.

The **partial derivative** $\partial f/\partial x_i$ is the ordinary derivative with respect to $x_i$ *holding all other variables fixed*. For $f(x,y)=x^{2}y+\sin y$:

$$
\frac{\partial f}{\partial x}=2xy,\qquad \frac{\partial f}{\partial y}=x^{2}+\cos y .
$$

At the point $(1,2)$: $\partial_xf=4$ and $\partial_yf=1+\cos2=0.584$ (script output: the analytic and numerical values agree to five digits). Each partial derivative answers a *one-knob* question: "if I turn only this knob, how fast does the output change?"

The **gradient** collects them in a vector:

$$
\nabla f(\mathbf{x}) \;=\; \Big(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_d}\Big)^{\!\top}\in\R^{d}.
$$

!!! math "Why the gradient points uphill"
    Near a point $\mathbf{x}$, by the multivariable version of the linear approximation, $f(\mathbf{x}+\mathbf{v})\approx f(\mathbf{x})+\nabla f(\mathbf{x})^{\top}\mathbf{v}$. For a *unit* step $\mathbf{v}$ the change is the dot product $\nabla f\cdot\mathbf{v}=\|\nabla f\|\cos\theta$, where $\theta$ is the angle between $\mathbf{v}$ and the gradient (Chapter M5). It is largest when $\theta=0$: **the gradient is the direction of steepest ascent, and its length is the steepest rate of increase.** It is zero when the direction is perpendicular to the gradient (moving along a contour line).

    **Check.** At $(1,2)$, $\|\nabla f\|=\sqrt{4^2+0.584^2}=4.0424$. The script measures the rate of change along the gradient direction: $4.0424$. Among 200 random directions the best rate is $4.0418$: none exceeds the gradient's.

The consequence is the basis of all of machine learning: to *decrease* a function, step opposite the gradient. That is **gradient descent** (§M4.4).

---

## M4.2 Jacobians and the chain rule

If the output is also a vector, $\mathbf{F}:\R^{n}\to\R^{m}$, its derivative is the **Jacobian matrix**, with $m$ rows (outputs) and $n$ columns (inputs):

$$
\mathbf{J}_{\mathbf{F}}(\mathbf{x})=\Big[\frac{\partial F_i}{\partial x_j}\Big]_{i=1..m,\;j=1..n}\in\R^{m\times n}.
$$

It is the best linear approximation: $\mathbf{F}(\mathbf{x}+\mathbf{v})\approx\mathbf{F}(\mathbf{x})+\mathbf{J}\mathbf{v}$. For $\mathbf{F}(x_0,x_1)=(x_0x_1,\;x_0+x_1^{2},\;\sin x_0)$ the Jacobian is the $3\times2$ matrix with rows $(x_1,\,x_0)$, $(1,\,2x_1)$, $(\cos x_0,\,0)$; the script confirms it against finite differences to $4\times10^{-10}$. A scalar function is the special case $m=1$, where the $1\times n$ Jacobian is the transposed gradient.

!!! math "The multivariable chain rule"
    If $\mathbf{G}:\R^{m}\to\R^{p}$ and $\mathbf{F}:\R^{n}\to\R^{m}$, then the Jacobian of the composition $\mathbf{G}\circ\mathbf{F}$ is the **matrix product of the Jacobians**:

    $$
    \mathbf{J}_{\mathbf{G}\circ\mathbf{F}}(\mathbf{x}) \;=\; \mathbf{J}_{\mathbf{G}}\big(\mathbf{F}(\mathbf{x})\big)\;\mathbf{J}_{\mathbf{F}}(\mathbf{x}),
    \qquad (p\times m)(m\times n)=(p\times n).
    $$

    *Reason:* locally each function is a linear map (its Jacobian), and the composition of linear maps is their matrix product. In the script, the Jacobian of $\mathbf{G}\circ\mathbf{F}$ computed as $\mathbf{J}_{\mathbf{G}}\mathbf{J}_{\mathbf{F}}$ matches the finite-difference Jacobian to $2\times10^{-10}$.

    For a scalar loss $L(\mathbf{w})=\ell(\mathbf{h}(\mathbf{w}))$ this reads $\nabla_{\mathbf{w}}L=\mathbf{J}_{\mathbf{h}}^{\top}\nabla_{\mathbf{h}}\ell$: *multiply the gradient at the output by the transposed Jacobian of each layer, going backwards*. That sentence is backpropagation (Chapter 9). Example: the one-example logistic loss $L(\mathbf{w})=-\ln\sigma(\mathbf{w}^\top\mathbf{x})$ has $\nabla_{\mathbf{w}}L=-(1-\sigma(\mathbf{w}^\top\mathbf{x}))\,\mathbf{x}$, derived by applying the chain rule through $\ln$, $\sigma$, and the dot product (script: analytic and numerical gradients agree to $3\times10^{-10}$). The factor $1-\sigma$ is the model's *error*: the update is large when the model is wrong and vanishes when it is confident and right.

---

## M4.3 Second derivatives: the Hessian, curvature and saddles

The matrix of second partial derivatives is the **Hessian**, $\mathbf{H}_{ij}=\partial^{2}f/\partial x_i\partial x_j$ (symmetric for smooth functions). The second-order Taylor expansion is

$$
f(\mathbf{x}+\mathbf{v})\;\approx\; f(\mathbf{x})+\nabla f^{\top}\mathbf{v}+\tfrac12\,\mathbf{v}^{\top}\mathbf{H}\,\mathbf{v}.
$$

At a **critical point** ($\nabla f=0$) the quadratic term decides the type. Using the eigenvalues of $\mathbf{H}$ (the amount of curvature along each principal direction; Chapter M6):

| Eigenvalues of $\mathbf{H}$ | Type | Picture |
|---|---|---|
| all positive | local **minimum** | a bowl |
| all negative | local **maximum** | a dome |
| mixed signs | **saddle point** | a horse saddle: up in some directions, down in others |
| some zero | degenerate / flat directions | a valley floor or a plateau |

The script shows $x^2-y^2$ has Hessian eigenvalues $-2,+2$ (a saddle at the origin) while $x^2+3y^2$ has $+2,+6$ (a minimum). In *high dimensions* saddle points vastly outnumber local minima: a critical point needs *every* one of $d$ curvatures to be positive to be a minimum, and in random landscapes that gets rare as $d$ grows. For deep-network losses, with millions of parameters, the practical obstacle to optimization is much more often saddles, flat regions and ill-conditioning (below) than poor local minima (Chapter 3).

A function is **convex** if its Hessian is positive semi-definite everywhere (all curvatures $\ge0$). For a convex function, *every* local minimum is global, and gradient descent finds it. Least squares and logistic regression are convex in their parameters; deep networks are not.

---

## M4.4 Gradient descent

To minimize $f$ start somewhere and repeatedly step downhill:

$$
\mathbf{w}_{t+1}\;=\;\mathbf{w}_t-\eta\,\nabla f(\mathbf{w}_t),
$$

with **learning rate** (step size) $\eta>0$. Why it works: for small $\eta$, $f(\mathbf{w}_{t+1})\approx f(\mathbf{w}_t)-\eta\|\nabla f\|^2$, a decrease. Everything interesting is about *how small* $\eta$ must be and *how fast* the decrease is.

**On a quadratic bowl.** Take $f(\mathbf{w})=\tfrac12\mathbf{w}^\top\mathbf{A}\mathbf{w}$ with $\mathbf{A}$ symmetric positive definite, eigenvalues $\lambda_{\min}\le\dots\le\lambda_{\max}$. Then $\nabla f=\mathbf{A}\mathbf{w}$ and the update is $\mathbf{w}_{t+1}=(\mathbf{I}-\eta\mathbf{A})\mathbf{w}_t$. In the eigenbasis of $\mathbf{A}$, each coordinate shrinks by the factor $1-\eta\lambda_i$ per step. Two consequences:

1. **Stability:** every factor must have size below 1, so $\eta<2/\lambda_{\max}$. In the script, with $\lambda_{\max}=10$ the threshold is $0.2$: $\eta=0.19$ converges ($\|\mathbf{w}\|=2.7\times10^{-5}$ after 100 steps), while $\eta=0.21$ *diverges* ($1.4\times10^{4}$).
2. **Speed:** the *flattest* direction ($\lambda_{\min}$) shrinks the slowest, by $1-\eta\lambda_{\min}$. With the best fixed step $\eta=2/(\lambda_{\min}+\lambda_{\max})$ the worst factor is $(\kappa-1)/(\kappa+1)$ where

$$
\kappa=\lambda_{\max}/\lambda_{\min}
$$

is the **condition number**. The number of steps to shrink the error by a factor $10^{-6}$ is about $\tfrac{\kappa}{2}\ln10^{6}$: 65 steps for $\kappa=10$, 656 for $\kappa=100$ and 5,811 for $\kappa=1000$ (script; theory 69, 691, 6,908). *Ten times worse conditioning costs about ten times more steps.*

!!! bio "Biology for modeling: correlated features make problems ill-conditioned"
    **What is it?** Fitting a linear model to genotype or expression features. The curvature matrix of the least-squares loss is proportional to the covariance of the features, $\mathbf{X}^\top\mathbf{X}$.

    **Information it contains / discards.** If two features are nearly identical (two SNPs in strong linkage disequilibrium; two genes in one co-expression module), $\mathbf{X}^\top\mathbf{X}$ has a tiny eigenvalue in the direction "one up, the other down": the data barely constrain that direction.

    **Modelling consequence.** $\kappa$ is huge. Gradient descent crawls along the flat direction; the exact solution is sensitive to noise there; and the fitted coefficients on correlated features are individually unreliable even though their sum is well determined. Regularization (ridge, Chapter 7) adds $\lambda$ to every eigenvalue, shrinking $\kappa$; preconditioning and adaptive methods such as Adam (Chapter 3) rescale directions to the same effect.

    **What would change the interpretation.** In nonconvex deep networks the Hessian changes along the trajectory, so "the" condition number is a moving target; empirical work measures sharpness (the top Hessian eigenvalue) during training.

**Stochastic gradient descent (SGD).** For a loss that is an average over $N$ examples, $L=\frac1N\sum_{n}\ell_n$, computing the full gradient costs $N$ evaluations. SGD uses the gradient of a random *mini-batch* instead: it is an unbiased but noisy estimate of the full gradient (Chapter M7 explains why averages of random samples are close to the truth). The noise lets the iterate wander and sometimes escape sharp minima; the learning rate must be decreased or the noise averaged for convergence. Nearly all deep learning uses SGD variants (Chapter 3).

---

## M4.5 Least squares from the gradient

Given an $N\times D$ data matrix $\mathbf{X}$ and responses $\mathbf{y}\in\R^{N}$, the **least-squares** loss is $L(\mathbf{w})=\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^{2}$. Its gradient is $\nabla L=-2\mathbf{X}^{\top}(\mathbf{y}-\mathbf{X}\mathbf{w})$ (chain rule: the Jacobian of the residual is $-\mathbf{X}$, and the gradient of a squared norm $\|\mathbf{r}\|^2$ is $2\mathbf{r}$). Setting it to zero gives the **normal equations**

$$
\mathbf{X}^{\top}\mathbf{X}\,\mathbf{w}=\mathbf{X}^{\top}\mathbf{y}.
$$

In the script, with $N=200$, $D=5$ and true coefficients $(1,-2,0,0.5,3)$, the solution is $(1.005,-1.996,-0.008,0.487,2.988)$; the gradient at the solution is zero to $2\times10^{-13}$, and 500 gradient-descent steps reach the same answer. The Hessian of $L$ is the constant matrix $2\mathbf{X}^\top\mathbf{X}$, so $L$ is a perfect quadratic bowl: the gradient-descent analysis of §M4.4 applies exactly, with $\kappa$ equal to the condition number of $\mathbf{X}^\top\mathbf{X}$. Chapter M5 and Chapter 2 interpret the same solution geometrically as a projection.

---

## M4.6 Constrained optimization and Lagrange multipliers

Often we minimize or maximize $f(\mathbf{x})$ *subject to a constraint* $g(\mathbf{x})=0$: a probability vector must sum to 1; a mean energy is fixed by an experiment. At a constrained optimum the level set of $f$ is tangent to the constraint surface, so $\nabla f$ is parallel to $\nabla g$: there is a number $\lambda$ (the **Lagrange multiplier**) with

$$
\nabla f(\mathbf{x}^\ast)=\lambda\,\nabla g(\mathbf{x}^\ast),\qquad g(\mathbf{x}^\ast)=0.
$$

Equivalently, find stationary points of the **Lagrangian** $\mathcal{L}(\mathbf{x},\lambda)=f(\mathbf{x})-\lambda g(\mathbf{x})$ with respect to *both* $\mathbf{x}$ and $\lambda$. The multiplier also has a meaning: it is the *sensitivity of the optimum to the constraint*: how much the best value of $f$ changes if the constraint level is relaxed by one unit (a "shadow price").

!!! math "Derivation: maximum entropy under an energy constraint gives the Boltzmann distribution"
    **Problem.** Among all probability distributions $p_1,\dots,p_K$ on states with energies $E_1,\dots,E_K$, find the one with the largest entropy $H(p)=-\sum_ip_i\ln p_i$ subject to $\sum_ip_i=1$ and a fixed mean energy $\sum_ip_iE_i=\bar E$.

    **Lagrangian.** $\mathcal{L}=-\sum_ip_i\ln p_i-\alpha\big(\sum_ip_i-1\big)-\beta\big(\sum_ip_iE_i-\bar E\big)$.

    **Stationarity.** $\partial\mathcal{L}/\partial p_i=-\ln p_i-1-\alpha-\beta E_i=0$, so $p_i=e^{-1-\alpha}e^{-\beta E_i}$. Normalization fixes the constant:

    $$
    p_i=\frac{e^{-\beta E_i}}{Z(\beta)},\qquad Z(\beta)=\sum_je^{-\beta E_j}.
    $$

    This is the **Boltzmann distribution**, with $\beta$ (the inverse temperature) the multiplier that enforces the mean energy; with $E_i=-z_i$ and $\beta=1$ it is the **softmax**. So the softmax is *not an arbitrary choice*: it is the distribution that commits to nothing beyond a stated average score.

    **Check (script).** Energies $0,1,2,3,4$ with target mean energy $1.2$: solving for $\beta$ gives $\beta=0.4313$ and $p=(0.396,0.257,0.167,0.109,0.071)$ with entropy $1.4434$ nats. Among 20,000 random distributions that satisfy both constraints, the best entropy found is $1.4432$: none beats the Lagrange solution (the maximum is approached but never exceeded).

!!! bio "Biology for modeling: where Boltzmann/softmax distributions appear"
    **What is it?** The same functional form appears whenever a system is described by scores or energies and nothing else is known: the occupancy of binding sites in thermodynamic models of gene regulation (Chapter 22); the Potts model of protein co-evolution (Chapters 29, 34); the softmax of attention (Chapter 12) and of classifiers.

    **Information it contains.** Only a mean score. All else is maximally uncertain.

    **What it discards, and the failure mode.** It assumes the only relevant constraint is the mean energy. If other statistics matter (pairwise correlations between sites), the maximum-entropy model must include them as extra constraints, each with its own multiplier. That is exactly how the *Potts* model arises: constrain the single-site and pairwise frequencies of a protein family, and the maximum-entropy distribution has pairwise couplings (Chapter 34).

---

## M4.7 Multiple integrals and change of variables

An integral over several variables sums the function over a region: $\iint f(x,y)\,dx\,dy$ is the volume under the surface $f$. For a smooth function you can integrate one variable at a time (**Fubini**: the order does not matter when the integrand is absolutely integrable). It is how a joint probability density over several variables is normalized and how a marginal is computed (Chapter M7).

**Change of variables.** If you substitute $\mathbf{x}=\boldsymbol\phi(\mathbf{u})$, the region distorts, and the volume element is scaled by the absolute value of the **Jacobian determinant**:

$$
\int f(\mathbf{x})\,d\mathbf{x}=\int f(\boldsymbol\phi(\mathbf{u}))\,\big|\det\mathbf{J}_{\boldsymbol\phi}(\mathbf{u})\big|\,d\mathbf{u}.
$$

In polar coordinates, $dx\,dy=r\,dr\,d\theta$. This explains the two-dimensional Gaussian integral: $\iint e^{-(x^2+y^2)/2}dxdy=\int_0^{2\pi}\!\int_0^\infty e^{-r^2/2}r\,dr\,d\theta=2\pi$, which is the standard proof that the normal density's constant is $1/\sqrt{2\pi}$ per dimension. In the script, summing the 2D standard Gaussian density on a grid gives $1.000000$, and the probability of falling in the unit disc is $0.3933$ (exact $1-e^{-1/2}=0.3935$).

The Jacobian determinant is the key to **normalizing flows** (Chapter 14): to transform a simple distribution into a complex one by an invertible map, the density changes by $|\det\mathbf{J}|^{-1}$, so the map must be built to make that determinant cheap.

```python
--8<-- "code/m04_multivariable.py"
```

Output (seed 0):

```text
f(x,y) = x^2 y + sin y at (1, 2): f = 2.9093
  analytic gradient [4.      0.58385]   numeric gradient [4.      0.58385]
  rate of change along the gradient direction: 4.0424  (= ||grad|| = 4.0424); best of 200 random directions: 4.0418

chain rule for L(w) = -log sigmoid(w.x): max |analytic - numeric| = 2.65e-10
Jacobian of F: shape (3, 2)  max |analytic - numeric| = 3.8e-10
chain rule J_(G o F) = J_G J_F: max difference 2.3e-10  (shapes (2x3)(3x2) -> 2x2)
  Hessian of x^2 - y^2: eigenvalues [-2.  2.] -> saddle at the origin
  Hessian of x^2 + 3y^2: eigenvalues [2. 6.] -> minimum at the origin

gradient descent on 1/2 w^T A w, d = 20, best fixed step 2/(lmin + lmax):
  kappa =     1:      1 steps   (theory ~ (kappa/2) ln(1e6) = 7)
  kappa =    10:     65 steps   (theory ~ (kappa/2) ln(1e6) = 69)
  kappa =   100:    656 steps   (theory ~ (kappa/2) ln(1e6) = 691)
  kappa =  1000:   5811 steps   (theory ~ (kappa/2) ln(1e6) = 6908)
  learning rate 0.05: ||w|| after 100 steps = 5.921e-03  (diverges above 2/lambda_max = 0.2)
  learning rate 0.19: ||w|| after 100 steps = 2.656e-05  (diverges above 2/lambda_max = 0.2)
  learning rate 0.21: ||w|| after 100 steps = 1.378e+04  (diverges above 2/lambda_max = 0.2)

normal-equation solution [ 1.005 -1.996 -0.008  0.487  2.988]  gradient at the solution: max |g| = 2.2e-13
gradient descent after 500 steps agrees: True

max-entropy with mean energy 1.2: Lagrange solution p = [0.3962 0.2574 0.1672 0.1086 0.0706], beta = 0.4313, entropy 1.4434 nats
  best entropy among 20000 random feasible distributions that satisfied the constraints: 1.4432  (never exceeds the Boltzmann solution)

2D Gaussian integrates to 1.000000;  P(x^2 + y^2 < 1) = 0.3933 (exact 1 - e^-0.5 = 0.3935)
```

---

## M4.8 Worked examples

!!! example "Worked example M4.1: Why does the loss barely move along one direction?"
    **Situation.** You fit a linear model predicting a phenotype from 1,000 SNPs. Gradient descent reduces the loss quickly for the first 50 steps, then crawls. Coefficients of neighbouring SNPs keep changing in opposite directions while their sum stays fixed.

    **Question.** What is happening and what are your options?

    **Reasoning.**

    1. *Name the geometry.* The loss is a quadratic bowl in the coefficients with Hessian $2\mathbf{X}^\top\mathbf{X}$. SNPs in strong LD are nearly collinear columns of $\mathbf{X}$, so $\mathbf{X}^\top\mathbf{X}$ has tiny eigenvalues along directions like "$+1$ on SNP $a$, $-1$ on SNP $b$".
    2. *Predict the dynamics.* Fast directions (large $\lambda$) converge quickly; slow directions (tiny $\lambda$) take $\sim1/(\eta\lambda)$ steps. The curve "fast then crawl" is the fingerprint of a large condition number.
    3. *Interpret the sign flipping.* The data constrain the *sum* $\beta_a+\beta_b$ (a large-eigenvalue direction) but not the *difference*. Individual coefficients are not identifiable from the data. This is a statement about the **information in the data**, not about the optimizer.
    4. *Options.* (a) Add ridge regularization (shrinks the difference direction toward 0 and caps $\kappa$). (b) Reduce the features (LD pruning, PCA). (c) Use a method that models LD explicitly (Chapter 26). (d) Switch to a second-order or preconditioned optimizer. Only (a), (b), (c) change the *answer* to a well-posed one; (d) only speeds up reaching an answer that is still ambiguous.

    **Lesson.** Slow optimization is sometimes a software problem and sometimes a *statistical* signal that the question is ill-posed.

!!! example "Worked example M4.2: A constraint, a multiplier, and a design question"
    **Situation.** You must split a fixed sequencing budget of $T=100$ million reads among $K=3$ samples to minimize the total estimation variance $\sum_{k}\sigma_k^2/n_k$, where $\sigma_k^2$ is the per-read noise level of sample $k$ and $n_k$ the reads allocated.

    **Reasoning.** Minimize $f=\sum_k\sigma_k^2/n_k$ subject to $\sum_kn_k=T$. Lagrangian $\mathcal{L}=\sum_k\sigma_k^2/n_k+\lambda(\sum_kn_k-T)$. Stationarity: $-\sigma_k^2/n_k^2+\lambda=0$, so $n_k=\sigma_k/\sqrt{\lambda}\propto\sigma_k$. Normalization gives $n_k=T\sigma_k/\sum_j\sigma_j$.

    **Interpretation.** Allocate reads *in proportion to the noise standard deviation* (not the variance, and not equally). With $\sigma=(1,2,3)$: $n=(16.7,33.3,50.0)$ million, total variance $=(\sum_k\sigma_k)^2/T=36/100=0.36$, versus $(1+4+9)\cdot3/100=0.42$ for equal allocation: 14% lower variance for free. The multiplier has a direct meaning: from the stationarity condition, $\lambda=\sigma_k^2/n_k^2=(\sum_j\sigma_j/T)^2=0.0036$. It equals $-\,df^{\ast}/dT$, where $f^{\ast}(T)=(\sum_k\sigma_k)^2/T$ is the optimal variance, so one more million reads lowers the best achievable variance by $0.0036$ (sign conventions for $\lambda$ differ between texts).

    **Caveat.** This ignores that real counts have sample-dependent mean–variance relationships and diminishing returns from more reads at a fixed library complexity; the principle (allocate resources by marginal value, set equal at the optimum) survives, the formula does not.

---

## M4.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"Training loss stopped decreasing, so we've reached a minimum\""
    **The observation.** Training loss on a model plateaus at 0.31 for 20 epochs; the gradient norm is small, $10^{-3}$.

    **Tempting conclusion.** "We are at a minimum; the model has converged. Further training is pointless."

    **Decompose.** A small gradient only says the point is *critical*, not a minimum (§M4.3). In a high-dimensional landscape the plateau may be (i) a saddle point or flat region from which noise or momentum would eventually escape; (ii) a long, narrow, curved valley (large $\kappa$) in which each step is tiny; (iii) a true local minimum; (iv) a learning rate so small, or a step-size schedule decayed so far, that progress is slow in absolute terms. These have different remedies, and *the gradient norm alone cannot tell them apart*.

    **Hidden assumptions.** (i) The gradient is computed on the full data (a mini-batch gradient is noisy and its norm includes noise). (ii) Convergence of the *training* loss means anything about the model's behaviour on new data (it does not; Chapter 7). (iii) Floating-point precision does not hide small progress.

    **Discriminating experiments.** Estimate the top Hessian eigenvalues (Hessian-vector products) and the smallest few; restart from the plateau with a larger step or with momentum; perturb the weights and see whether the loss returns to the same value; compute the validation loss, which is the quantity that matters.

    **What this teaches.** "Zero gradient" is a *local* statement about the first derivative. Curvature (the Hessian) and the global shape of the landscape are separate questions that need separate evidence.

---

## M4.10 Connections

- **Forward:** gradient descent, momentum, Adam and conditioning are developed in Chapter 3; the chain rule as matrix products is backpropagation (Chapter 9); the Hessian appears in Laplace approximations, sharpness and influence functions (Chapters 8, 18); Lagrange multipliers give maximum-entropy models, the Potts model, softmax and the Boltzmann distribution (Chapters 5, 22, 34); Jacobian determinants appear in normalizing flows (Chapter 14); constrained optimization returns in experimental design (Chapter 46).
- **Backward:** derivatives, the chain rule, Taylor series (M3); logarithms and softmax (M2).

!!! takeaways "Key takeaways"
    1. The **gradient** collects partial derivatives; it points in the direction of steepest ascent, with length equal to the steepest rate. To minimize, step against it.
    2. The **Jacobian** is the matrix of first derivatives of a vector function; the **chain rule** is the product of Jacobians. A loss gradient is $\mathbf{J}^\top\nabla_{\text{out}}$ layer by layer, which is backpropagation.
    3. The **Hessian** gives curvature. All eigenvalues positive: minimum; mixed signs: saddle. In high dimensions saddles dominate.
    4. Gradient descent on a quadratic is stable only for $\eta<2/\lambda_{\max}$ and takes $\sim(\kappa/2)\ln(1/\text{tol})$ steps, where $\kappa=\lambda_{\max}/\lambda_{\min}$. Correlated features give large $\kappa$ and an ill-posed problem, not only a slow one.
    5. Least squares: $\nabla L=-2\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})=0$ gives the normal equations.
    6. A **Lagrange multiplier** enforces a constraint and measures the optimum's sensitivity to it. Maximum entropy with a mean-energy constraint yields the Boltzmann/softmax distribution, so softmax is the least-committal choice, not an arbitrary one.
    7. Multiple integrals are done one variable at a time; changing variables scales the volume by $|\det\mathbf{J}|$ (polar: $r\,dr\,d\theta$).
    8. A small gradient means a critical point, not a minimum, and says nothing about validation performance.

---

## Further reading

- Stewart, J. *Multivariable Calculus*. Cengage; or Marsden, J. & Tromba, A. *Vector Calculus*. Standard treatments of gradients, Jacobians and multiple integrals.
- Boyd, S. & Vandenberghe, L. *Convex Optimization* (free online). Convexity, duality and Lagrange multipliers done thoroughly.
- Nocedal, J. & Wright, S. *Numerical Optimization*. Springer. Gradient descent, Newton and conditioning, in depth.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review* 106, 620–630. The maximum-entropy derivation of the Boltzmann distribution.
- Dauphin, Y. et al. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*.
