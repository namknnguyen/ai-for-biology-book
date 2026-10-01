# Chapter M3. Single-Variable Calculus: Rates, Accumulation, and Differential Equations

!!! abstract "Chapter at a glance"
    **Motivation.** Calculus is the mathematics of *change*. A derivative says how fast something changes; an integral adds up many small changes; a differential equation says how a system changes as a function of its current state, which is exactly what a gene-regulation model, a growth model or a pharmacokinetic model is. Training a neural network is "follow the derivative downhill" (Chapter 3). If you understand this chapter you understand the engine under almost every method later in the book.

    **Prerequisites.** Chapters M1 and M2: functions, exponentials, logarithms.

    **You will be able to:** (1) explain a derivative as a rate, as a slope, and as the best linear approximation; (2) differentiate sums, products, quotients and compositions, and the basic functions; (3) find and classify maxima and minima, and derive a maximum-likelihood estimate; (4) approximate a function by its Taylor polynomial and say where the approximation fails; (5) interpret an integral as accumulated change and use the Fundamental Theorem of Calculus; (6) solve the linear ODE of production and degradation, and say what sets a gene's response time; (7) integrate an ODE numerically and recognize when a step size is too large; (8) state Jensen's inequality and use it.

---

## M3.1 From average rate to instantaneous rate

If a culture grows from 1,000 to 1,400 cells in two hours, the **average rate** is $(1400-1000)/2 = 200$ cells per hour. But growth is not constant; the rate at an *instant* is what a differential model needs. We get it by shrinking the interval:

$$
f'(x) \;=\; \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

This is the **derivative**: the limit of average rates over shorter and shorter intervals. A **limit** $\lim_{h\to0} g(h) = L$ means that $g(h)$ can be made as close to $L$ as you like by taking $h$ close enough to 0 (but not equal to 0). The precise ($\epsilon$–$\delta$) definition is in the references; for this book the intuition suffices: *plug in smaller and smaller $h$ and see where the numbers settle*.

Three readings of $f'(x)$, all used constantly:

1. **Rate of change.** Units are (output units)/(input units): molecules per hour, fitness per mutation.
2. **Slope of the tangent line.** The line through $(x, f(x))$ with slope $f'(x)$ touches the curve and points the way it is heading.
3. **Best linear approximation.** For small $h$: $f(x+h)\approx f(x) + f'(x)\,h$. This is the idea that generalizes to many variables (Chapter M4): *near a point, every smooth function looks like a linear function*, and the derivative is the linear function.

A function is **differentiable** where the limit exists. $|x|$ is not differentiable at 0 (a corner), and ReLU shares the corner; in practice, software takes the derivative to be 0 or 1 there and nothing goes wrong, because the point is rarely hit exactly.

---

## M3.2 Rules for differentiating

You do not compute limits each time. A short list of derivatives and four rules gets you almost everywhere.

| Function $f(x)$ | $f'(x)$ | Note |
|---|---|---|
| constant $c$ | $0$ | |
| $x^{p}$ | $p\,x^{p-1}$ | $p$ any real number |
| $e^{x}$ | $e^{x}$ | the defining property of $e$ |
| $e^{kx}$ | $k e^{kx}$ | chain rule (below) |
| $\ln x$ | $1/x$ | $x>0$ |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ | angles in radians |
| $\sigma(x)=\dfrac{1}{1+e^{-x}}$ | $\sigma(x)\big(1-\sigma(x)\big)$ | logistic sigmoid |

!!! math "The four rules"
    1. **Linearity:** $(af + bg)' = af' + bg'$.
    2. **Product rule:** $(fg)' = f'g + fg'$. *Reason:* when both factors change slightly, the product changes by "first changes, second stays" plus "second changes, first stays"; the double-change term is second order and vanishes in the limit.
    3. **Quotient rule:** $(f/g)' = (f'g - fg')/g^{2}$.
    4. **Chain rule:** $\big(f(g(x))\big)' = f'(g(x))\cdot g'(x)$. *Reason:* the inner function stretches changes by $g'(x)$, then the outer function stretches them by $f'$; stretch factors multiply.

**The chain rule is the most important rule in this book.** A neural network is a composition $f_L\circ\cdots\circ f_1$, and backpropagation (Chapter 9) is nothing more than applying the chain rule systematically, from the output back to the input. In Leibniz notation, with $u = g(x)$ and $y = f(u)$: $\dfrac{dy}{dx}=\dfrac{dy}{du}\dfrac{du}{dx}$.

*Example.* Differentiate $y = \ln(1 + e^{x})$ (the "softplus", a smooth ReLU). Let $u = 1+e^x$; then $dy/du = 1/u$, $du/dx = e^x$, so $y' = e^x/(1+e^x) = \sigma(x)$. The derivative of the softplus is the sigmoid. Check at $x=0$: $\ln 2 = 0.693$, slope $= 0.5$.

*Example (using the sigmoid derivative).* $\sigma'(x)=\sigma(1-\sigma)$ has maximum $1/4$ at $x = 0$ and tends to 0 for large $|x|$. A network of many sigmoid layers multiplies many factors of at most $1/4$ in the chain rule, so the gradient shrinks exponentially with depth: the **vanishing-gradient problem** of Chapter 9 follows from this one line.

---

## M3.3 Derivatives on a computer

A derivative can be approximated numerically by a **finite difference**: the forward difference $\big(f(x+h) - f(x)\big)/h$, or the more accurate central difference $\big(f(x+h)-f(x-h)\big)/(2h)$. Two errors compete:

- **truncation error**, from using a finite $h$: it shrinks as $h\to0$ (like $h$ for forward, like $h^2$ for central);
- **round-off error**, from subtracting two nearly equal numbers (*catastrophic cancellation*): it grows like $\varepsilon/h$, where $\varepsilon\approx2\times10^{-16}$ is the machine precision of double-precision numbers.

The sum of the two is U-shaped in $h$. For $f=\exp$ at $x=1$ (script output below), the forward-difference error falls from $1.4\times10^{-1}$ at $h=10^{-1}$ to a minimum of $6.6\times10^{-9}$ at $h=10^{-8}$ and then *climbs back* to $4.3\times10^{-4}$ at $h = 10^{-12}$ and to 2.7, a 100% error, at $h=10^{-16}$ (where $x+h = x$ in floating point). The central difference reaches $5.9\times10^{-11}$ near $h = 10^{-5}$. This is the practical reason **automatic differentiation** (Chapter 9) is used in deep learning instead of finite differences, and why gradient checks use central differences with $h\approx10^{-5}$ to $10^{-6}$.

---

## M3.4 What derivatives tell you: optimization

If $f'(x)>0$ the function is increasing; if $f'(x)<0$ decreasing. At a **local maximum or minimum** of a smooth function on an open interval, $f'(x)=0$. Such points are **critical points**. The **second derivative** $f''$ (the derivative of $f'$) says how the slope itself changes: $f''(x)>0$ at a critical point means a local minimum (the curve bends upward like a bowl); $f''(x)<0$ a local maximum; $f''=0$ is inconclusive.

!!! math "Maximum likelihood by calculus (the first estimator you will derive)"
    A coin is flipped 10 times and shows 7 heads. If the probability of heads is $p$, the probability of that data is $L(p)=p^{7}(1-p)^{3}$ (up to a constant), the **likelihood**. Maximize the **log-likelihood** $\ell(p)=7\ln p + 3\ln(1-p)$, which has the same maximizer because $\ln$ is increasing (Chapter M2):

    $$
    \ell'(p) \;=\; \frac{7}{p} - \frac{3}{1-p} \;=\; 0 \quad\Longrightarrow\quad 7(1-p) = 3p \quad\Longrightarrow\quad \hat p = 0.7 .
    $$

    Second derivative: $\ell''(p) = -7/p^{2} - 3/(1-p)^{2} < 0$ everywhere, so this is a maximum (and the only one). The estimate $\hat p=k/n$, the observed fraction, is the maximum-likelihood estimate; the grid search in the script returns $0.700$. For $k$ successes in $n$ trials, the same calculation gives $\hat p = k/n$. Chapter M8 builds all of statistical estimation from this idea.

**Newton's method.** To find where $g(x)=0$, repeatedly replace $x$ by where the *tangent line* crosses zero: $x\leftarrow x - g(x)/g'(x)$. To find a minimum of $f$, apply it to $g=f'$. Near a root it converges extremely fast: the number of correct digits roughly *doubles* each step. In the script, for $x^3-2x-5=0$ starting from $2$, the residual falls $6\times10^{-2}\to2\times10^{-4}\to2\times10^{-9}\to9\times10^{-16}$, reaching machine precision in four steps. Gradient descent (Chapter M4) is slower but needs only first derivatives; Newton's method needs second derivatives (a Hessian matrix in many dimensions), which is too expensive for large models, so it appears in modern training only through approximations.

---

## M3.5 Taylor series: local polynomial approximations

If $f$ has derivatives at a point $a$, then near $a$

$$
f(x) \;\approx\; f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots .
$$

Truncating after the linear term gives the tangent-line approximation; after the quadratic term gives the **second-order** approximation used to analyse minima and Newton's method. Standard series around 0:

$$
e^{x} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots,
\qquad
\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \ (|x|<1),
\qquad
\sigma(x)=\frac12+\frac{x}{4}-\frac{x^3}{48}+\cdots .
$$

Two uses you have already met: **(i)** for small $x$, $\ln(1+x)\approx x$ and $e^{-x}\approx1-x$, which justified the birthday-problem approximation in Chapter M1; **(ii)** the linear term of the sigmoid is $x/4$, so *near zero a sigmoid unit behaves like a linear unit* with slope $1/4$.

**Taylor approximations are local.** The script's table for the sigmoid shows the cubic approximation improving the linear one near zero (error $2\times10^{-8}$ against $2\times10^{-5}$ at $x = 0.1$) and failing badly far away: at $x=4$ the exact value is $0.982$, the linear one is $1.5$ (impossible for a probability) and the cubic one is $0.167$, *worse* than the linear one. A polynomial fitted at one point says nothing reliable about distant points. This is a mathematical form of a lesson used repeatedly later: **a local model (a linearization, a quadratic loss approximation, an influence function) is trustworthy only near the point where it was built**, and "near" must be checked.

---

## M3.6 Integrals: accumulation and area

The **definite integral** $\int_a^b f(x)\,dx$ is the limit of sums of thin rectangles: split $[a,b]$ into $n$ pieces of width $\Delta x=(b-a)/n$, form $\sum_i f(x_i)\Delta x$, and let $n\to\infty$. It represents the *accumulated* quantity: area under the curve; total mRNA produced if $f$ is a production rate; total probability if $f$ is a density (Chapter M7).

The **Fundamental Theorem of Calculus** connects the two halves of the subject: if $F'(x) = f(x)$ ($F$ is an **antiderivative** of $f$), then

$$
\int_a^b f(x)\,dx \;=\; F(b) - F(a).
$$

Accumulation of a rate gives the net change: the integral of speed is distance, and the integral of production-minus-degradation rate is the net change in molecules. It is the continuous form of the telescoping sum of Chapter M1: $\sum(a_i-a_{i-1})=a_n-a_0$.

Useful integrals: $\int x^{p}dx = x^{p+1}/(p+1)$ ($p\neq-1$); $\int e^{kx}dx=e^{kx}/k$; $\int dx/x=\ln|x|$. Two techniques cover most of what you need: **substitution** (the chain rule backwards: $\int f(g(x))g'(x)dx=F(g(x))$) and **integration by parts** ($\int u\,dv=uv-\int v\,du$, the product rule backwards).

**Numerical integration.** When there is no antiderivative formula, add up rectangles by computer. The script shows the **left-endpoint** rule for $\int_0^1x^2dx=1/3$ has error $1.1\times10^{-1}$ at $n=4$ and $4.9\times10^{-4}$ at $n = 1024$ (error shrinking like $1/n$), while the **midpoint** rule has error $7.9\times10^{-8}$ at $n=1024$ (error shrinking like $1/n^2$): the same effort buys far more accuracy by sampling the interval's middle. Integrals over *probability distributions* are usually computed by sampling instead (Monte Carlo, Chapter M7).

**The Gaussian integral.** $\int_{-\infty}^{\infty}e^{-x^{2}}dx=\sqrt{\pi}$ has no elementary antiderivative, yet its value is exact (the script's numerical estimate $1.77245385$ matches $\sqrt{\pi}=1.77245385$). It is the reason the normal distribution's density $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ integrates to 1 (Chapter M7).

---

## M3.7 Differential equations: change as a function of state

A **differential equation** (ODE, ordinary differential equation) specifies the rate of change of a quantity in terms of the quantity: $dx/dt=F(x)$. Solving it means finding $x(t)$. ODEs are the basic language of dynamical models in biology.

**Exponential growth/decay.** $dx/dt=kx$ has solution $x(t)=x_0e^{kt}$ (check: differentiate it). This is why exponentials appeared in Chapter M2: a rate proportional to size *is* an exponential.

**Logistic growth.** If growth is limited, $dx/dt=rx(1-x/K)$: nearly exponential while $x\ll K$, then saturating at the carrying capacity $K$. The fixed points are $x=0$ (unstable) and $x=K$ (stable).

**Production and degradation: the core model of gene expression.**

$$
\frac{dm}{dt} \;=\; \underbrace{\alpha}_{\text{transcription}} \;-\; \underbrace{\delta\, m}_{\text{degradation}},
$$

where $m(t)$ is the number of mRNA molecules, $\alpha$ the production rate (molecules per hour) and $\delta$ the degradation rate (per hour). Degradation is proportional to the amount present because each molecule decays independently. Setting $dm/dt = 0$ gives the **steady state** $m^{*}=\alpha/\delta$. The full solution, with starting level $m_0$, is

$$
m(t) \;=\; \frac{\alpha}{\delta} + \Big(m_0 - \frac{\alpha}{\delta}\Big)e^{-\delta t}.
$$

*Check:* differentiate: $m' = -\delta(m_0-\alpha/\delta)e^{-\delta t}$, and $\alpha-\delta m=\alpha-\alpha-\delta(m_0-\alpha/\delta)e^{-\delta t}$, equal. ✓ Three facts follow, each biologically meaningful:

1. The steady-state level is the *ratio* of production to degradation. Two genes with the same steady-state level can differ ten-fold in each rate.
2. The **approach to steady state** is exponential with rate $\delta$, whatever $\alpha$ is. It has half-life $\ln2/\delta$.
3. Therefore a gene's **response time** is set by its mRNA (and protein) *degradation* rate, not by its transcription rate. To respond quickly, a transcript must be unstable; the price is a continuous turnover cost to hold a given level (which requires a large $\alpha$ for the same $m^*$).

!!! bio "Biology for modeling: why immediate-early genes have unstable mRNAs"
    **What is it?** The production–degradation model above, applied to a transcript such as *FOS* or *MYC* whose mRNA half-life is under an hour, versus a stable housekeeping transcript with a half-life of many hours.

    **Information it contains.** Steady state ($\alpha/\delta$) and speed ($\delta$) are separate quantities.

    **What it discards.** Transcriptional bursting (production is not a smooth rate but random bursts), regulation of $\delta$ itself (RNA-binding proteins, microRNAs), and the protein layer (translation and protein degradation add further delays).

    **Modelling consequence.** With $\alpha=20$ molecules/h and $\delta=0.25\ \text{h}^{-1}$ (half-life 2.8 h), $m^*=80$; the 90% response time to a step in transcription is $\ln10/\delta=9.2$ h. A transcript ten times less stable ($\delta=2.5$) reaches 90% in 0.9 h but needs $\alpha=200$ to hold $m^*=80$. RNA-velocity methods (Chapter 30) use precisely this model, relating spliced and unspliced counts to production and degradation, to infer where a cell is heading.

    **What would change the interpretation.** With bursty production the *distribution* of $m$ is wide (Poisson-like or worse, Chapter M7); the ODE describes only the mean.

**Numerical solution: Euler's method.** Most ODEs have no closed form. The simplest numerical scheme steps along the tangent: $x_{n+1}=x_n+\Delta t\,F(x_n)$ (Euler's method). In the script, for the mRNA model started at $m_0=0$, the exact value at 12 h is $76.017$, and Euler gives $78.75$ (error 2.7) for $\Delta t=2$, $76.7545$ (error 0.74) for $0.5$, $76.1661$ (0.15) for $0.1$ and $76.0320$ (0.015) for $0.01$: *the error is proportional to the step size* (first-order accuracy). Euler is also **unstable** for large steps: it requires $\delta\,\Delta t<2$. With $\Delta t=10$ ($\delta\Delta t = 2.5$) the computed values *diverge* ($-4533$ after 10 steps) although the true solution can never exceed 80. Always check that a simulation's result does not change when the step is halved; better solvers (Runge–Kutta methods) are used in practice and are available in `scipy.integrate`.

```python
--8<-- "code/m03_calculus.py"
```

Output:

```text
forward difference (f(x+h)-f(x))/h  vs  central difference (f(x+h)-f(x-h))/(2h), f = exp at x = 1
      h     forward error   central error
  1e-01      1.406e-01      4.533e-03
  1e-02      1.364e-02      4.530e-05
  1e-04      1.359e-04      4.531e-09
  1e-06      1.359e-06      1.635e-10
  1e-08      6.603e-09      6.603e-09
  1e-10      1.548e-06      6.727e-07
  1e-12      4.323e-04      2.103e-04
  1e-14      9.338e-03      9.338e-03
  1e-16      2.718e+00      2.718e+00
best forward: error 6.6e-09 at h = 1e-08;  best central: error 5.9e-11 at h = 1e-05

Taylor approximations of the sigmoid at 0:  sigma(x) vs 1/2 + x/4  vs  1/2 + x/4 - x^3/48
  x = 0.1: exact 0.52498   linear 0.52500 (err 2.1e-05)   cubic 0.52498 (err 2.1e-08)
  x = 0.5: exact 0.62246   linear 0.62500 (err 2.5e-03)   cubic 0.62240 (err 6.3e-05)
  x = 1.0: exact 0.73106   linear 0.75000 (err 1.9e-02)   cubic 0.72917 (err 1.9e-03)
  x = 2.0: exact 0.88080   linear 1.00000 (err 1.2e-01)   cubic 0.83333 (err 4.7e-02)
  x = 4.0: exact 0.98201   linear 1.50000 (err 5.2e-01)   cubic 0.16667 (err 8.2e-01)
e^x with 1+x+x^2/2+x^3/6 at x=1: 2.66667 vs 2.71828
log(1+x) ~ x for small x: x = 0.01 -> 0.00995 (and 1 - x ~ exp(-x): 0.99 vs 0.99005 )

Riemann sums for the integral of x^2 on [0,1] (exact 1/3):
  n =    4: left-endpoint 0.218750  (error 1.1e-01)   midpoint 0.328125  (error 5.2e-03)
  n =   16: left-endpoint 0.302734  (error 3.1e-02)   midpoint 0.333008  (error 3.3e-04)
  n =   64: left-endpoint 0.325562  (error 7.8e-03)   midpoint 0.333313  (error 2.0e-05)
  n = 1024: left-endpoint 0.332845  (error 4.9e-04)   midpoint 0.333333  (error 7.9e-08)
integral of exp(-x^2) over the real line: numeric 1.77245385, sqrt(pi) = 1.77245385

mRNA ODE: steady state alpha/delta = 80.0; half-life of approach = 2.77 h
  Euler dt =  2.00: m(12) =  78.7500   exact  76.0170   error 2.73e+00
  Euler dt =  0.50: m(12) =  76.7545   exact  76.0170   error 7.37e-01
  Euler dt =  0.10: m(12) =  76.1661   exact  76.0170   error 1.49e-01
  Euler dt =  0.01: m(12) =  76.0320   exact  76.0170   error 1.49e-02
  (halving dt roughly halves the error: Euler is first-order accurate; it is stable only if delta*dt < 2)
  Euler with dt = 10 (delta*dt = 2.5), after 10 steps: -4533.2 (the exact solution never exceeds 80: the scheme diverges)

Newton's method for x^3 - 2x - 5 = 0, starting at 2.0:
  step 1: x = 2.100000000000   |g(x)| = 6.1e-02
  step 2: x = 2.094568121104   |g(x)| = 1.9e-04
  step 3: x = 2.094551481698   |g(x)| = 1.7e-09
  step 4: x = 2.094551481542   |g(x)| = 8.9e-16
  step 5: x = 2.094551481542   |g(x)| = 8.9e-16

log-likelihood for 7 heads in 10 flips is maximized at p = 0.700  (calculus: d/dp = 7/p - 3/(1-p) = 0  =>  p = 0.7)
Jensen: mean(log(x)) = -0.271  <  log(mean(x)) = 0.469
```

---

## M3.8 Convexity and Jensen's inequality

A function is **convex** if the line segment between any two points on its graph lies above the graph ($f''\ge0$ for twice-differentiable $f$): a bowl. $x^2$ and $e^x$ are convex; $\ln x$ is **concave** ($f''=-1/x^2<0$), the cap shape. **Jensen's inequality** says that for a convex $f$ and a random quantity $X$,

$$
f\big(\mathbb{E}[X]\big) \;\le\; \mathbb{E}\big[f(X)\big],
$$

with the inequality reversed for concave $f$ (Chapter M7 defines $\mathbb{E}$, the average). The *function of the average is not the average of the function*. In the script, for $X$ drawn from a skewed gamma distribution, $\overline{\ln X}=-0.271$ but $\ln\overline{X}=+0.469$: the log of the mean exceeds the mean of the logs by 0.74. This single inequality explains: why averaging $\log(1+\text{counts})$ over cells is **not** the log of the average expression (Chapter M2's notebook), why the evidence lower bound (ELBO) of variational inference is a *lower* bound (Chapter 8), why KL divergence is nonnegative (Chapter 5), and why the expected fitness of a population can differ from the fitness at the average genotype.

---

## M3.9 Worked examples

!!! example "Worked example M3.1: Which rate should you change to make a gene respond faster?"
    **Situation.** A reporter gene has mRNA dynamics $dm/dt=\alpha-\delta m$ with $\alpha=20$ molecules/h and $\delta=0.25\ \text{h}^{-1}$. A collaborator proposes to double $\alpha$ (a stronger promoter) to make the gene "turn on faster" after induction.

    **Question.** Does it work? What would?

    **Reasoning.** After induction from $m_0=0$: $m(t)=m^*(1-e^{-\delta t})$ with $m^*=\alpha/\delta$. The time to reach a *fraction* $q$ of the new steady state is $t_q=-\ln(1-q)/\delta$, independent of $\alpha$. For $q=0.9$: $\ln10/0.25=9.2$ h before *and after* doubling $\alpha$. The stronger promoter raises the final level from 80 to 160, but the **fraction** reached at any time is unchanged. What speeds the approach is a larger $\delta$: with $\delta=0.5$, $t_{0.9}=4.6$ h (and the steady state falls to 40 unless $\alpha$ is also doubled).

    **What changes if the goal is a given *absolute* level?** To cross level 60 within 5 h: with $m^*=\alpha/\delta$, we need $m^*(1-e^{-5\delta})\ge60$. At $\delta=0.25$: $m^*(0.714)\ge60$, so $m^*\ge84$ and $\alpha\ge21$. A stronger promoter can win on the absolute-level criterion even though it does not change the relative speed.

    **Lesson.** Always ask which quantity the claim is about (fraction of steady state, or absolute level) and which parameter controls it. This is the Expert Chain's L4/L6 move: separate *mechanism* from *measurement*.

!!! example "Worked example M3.2: Deriving the maximum-likelihood estimate of a Poisson rate"
    **Situation.** Counts of a rare transcript in $n$ cells are $k_1,\dots,k_n$, modelled as independent Poisson random variables with mean $\lambda$: $P(k\mid\lambda)=\lambda^{k}e^{-\lambda}/k!$.

    **Reasoning.** The log-likelihood is $\ell(\lambda)=\sum_{i}\big(k_i\ln\lambda-\lambda-\ln k_i!\big)=\ln\lambda\sum_ik_i-n\lambda-\text{const}$. Differentiate: $\ell'(\lambda)=\frac{\sum_ik_i}{\lambda}-n=0\Rightarrow\hat\lambda=\frac1n\sum_ik_i=\bar k$. Second derivative $-\sum_ik_i/\lambda^2<0$: a maximum. The MLE of the Poisson rate is the sample mean.

    **Biological reading.** The sample mean is the right summary *if* the Poisson model is right. If counts are over-dispersed (variance larger than mean, as in real single-cell data because the rate itself varies between cells), the sample mean is still a sensible estimate of the average rate, but its uncertainty is larger than the Poisson formula claims (Chapters M7, 25, 30).

---

## M3.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The gradient check passes, so the backward pass is correct\""
    **The observation.** You implement the gradient of a custom loss and compare it with a finite-difference estimate. The relative error is $4\times10^{-5}$ with $h=10^{-2}$ and $3\times10^{-3}$ with $h=10^{-12}$.

    **Tempting conclusion.** "Close enough on the first, and the second is just noise: the implementation is fine."

    **Decompose.** The U-shaped error curve of §M3.3 means the *check itself* has a sweet spot. At $h=10^{-2}$ the forward-difference truncation error is $O(h)$, about $10^{-2}$ relative for a typical function, so $4\times10^{-5}$ with a central difference might be consistent with a correct gradient, or might hide a small mistake such as a missing factor close to 1. At $h=10^{-12}$ the error is dominated by cancellation noise, so a bug could hide inside. A trustworthy check scans $h$ over several decades, uses the central difference, and compares to the *expected* error shape: truncation error decreasing like $h^2$ until the cancellation floor near $10^{-6}$ to $10^{-8}$ takes over. A bug shows up as an error that does *not* decrease with $h$.

    **Hidden assumptions.** (i) The function is smooth at the test point (ReLU kinks and `max` operations break finite differences exactly at their kinks). (ii) The test point is generic: gradients that are exactly zero by symmetry at the origin test almost nothing. (iii) Double precision is used; the same check in half precision is meaningless.

    **Discriminating experiments.** Test at several random points; check each parameter block separately; check a case where the answer is known in closed form (the softmax cross-entropy gradient is $p-y$, as in Chapter 3).

    **What this teaches.** A passing numerical test is evidence only about the regime it probes. Knowing the *shape* of the numerical error lets you tell a floating-point artefact from a real bug.

---

## M3.11 Connections

- **Forward:** the chain rule becomes backpropagation (Chapter 9); the derivative becomes the gradient and Jacobian (M4); critical points and second derivatives become Hessians, saddle points and conditioning (M4, Chapter 3); the integral becomes expectation and normalization (M7); Taylor expansions underlie Laplace approximations and influence functions (Chapters 8, 18); ODEs reappear in diffusion and flow models (Chapter 15), RNA velocity (Chapter 30) and state-space models (Chapter 11); Jensen's inequality supports the ELBO (Chapter 8).
- **Backward:** functions, exponentials, logarithms (M2); sums and induction (M1).

!!! takeaways "Key takeaways"
    1. The **derivative** is the limit of average rates; it is also the slope of the tangent and the best linear approximation, $f(x+h)\approx f(x)+f'(x)h$.
    2. Know the table of derivatives, linearity, product, quotient, and above all the **chain rule**, which *is* backpropagation. $\sigma'=\sigma(1-\sigma)\le1/4$, which is the root of vanishing gradients.
    3. Numerical derivatives have a **U-shaped error** in the step size (truncation vs. round-off); central differences at $h\approx10^{-5}$ are the standard gradient check.
    4. Critical points ($f'=0$) with $f''>0$ are minima. Maximum likelihood is calculus: for a coin, $\hat p=k/n$; for a Poisson, $\hat\lambda=\bar k$. Newton's method roughly doubles the correct digits each step.
    5. **Taylor polynomials are local.** The sigmoid's cubic expansion is accurate at $x=0.1$ and worse than the linear one at $x=4$. Any linearization or quadratic model needs a check on how far it can be trusted.
    6. The **integral** accumulates a rate; the **Fundamental Theorem** says $\int_a^bf=F(b)-F(a)$. Midpoint rules converge like $1/n^2$; $\int e^{-x^2}dx=\sqrt\pi$.
    7. For $dm/dt=\alpha-\delta m$: steady state $\alpha/\delta$, approach rate $\delta$, so response *speed* is set by degradation and not by transcription. Euler's method has error $\propto\Delta t$ and is unstable when $\delta\Delta t>2$.
    8. **Jensen:** $f(\mathbb{E}X)\le\mathbb{E}f(X)$ for convex $f$. The average of a log is below the log of an average (here $-0.27$ vs $+0.47$).

---

## Further reading

- Stewart, J. *Calculus: Early Transcendentals*. Cengage. A standard first-year text with many worked examples.
- Strang, G. *Calculus* (free online, MIT OpenCourseWare). A conceptual first course emphasizing rates of change.
- Strogatz, S. H. *Nonlinear Dynamics and Chaos*. Westview. One-dimensional flows, stability and bifurcations, the right next step after §M3.7.
- Alon, U. *An Introduction to Systems Biology*. CRC Press. Production–degradation models and gene-circuit design.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice*. SIAM. For numerical differentiation and integration done properly.
