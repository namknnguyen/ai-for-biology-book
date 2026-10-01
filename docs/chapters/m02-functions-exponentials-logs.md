# Chapter M2. Functions, Exponentials, Logarithms, and Models

!!! abstract "Chapter at a glance"
    **Motivation.** A model is a function you believe describes some part of nature. Biology keeps returning to the same few function families: straight lines, power laws, exponentials, saturating curves, and oscillations. The logarithm is the tool that connects them, and it appears in every probabilistic model (as the log-likelihood), in every expression analysis (as the log fold change) and in every numerically stable implementation. This chapter makes you fluent in these families and shows how to recognize which one the data suggest.

    **Prerequisites.** Chapter M1 (functions, sums). High-school algebra and the idea of a graph.

    **You will be able to:** (1) fit and interpret linear, power-law, and exponential relationships by linearizing them with logarithms; (2) convert between growth rate, doubling time and half-life; (3) use logarithm rules and explain why log fold change and log-likelihood are the natural scales; (4) read a Hill (sigmoidal) curve in terms of threshold and steepness; (5) express a periodic signal by amplitude, period and phase; (6) see a neural-network layer as an affine map followed by a nonlinear function; (7) compute log-sum-exp without overflow.

---

## M2.1 Functions as models

In Chapter M1 a function was a rule from inputs to outputs. In science a function is a *hypothesis about a mechanism*: "the output is proportional to the input", "the output doubles every cycle", "the output saturates". Every such hypothesis has **parameters** (numbers fixed by the specific system) and a **form** (the shape that comes from the mechanism). Fitting a model means choosing the parameters; *choosing the form* is the scientific step, and it is the one that data alone cannot make for you.

The simplest forms:

- **Constant:** $f(x) = c$.
- **Linear (affine):** $f(x) = a x + b$. The slope $a$ is the change in output per unit input; $b$ is the output at $x=0$. Strictly, "linear" means $f(x + y) = f(x) + f(y)$ and $f(cx) = c f(x)$, which excludes $b \ne 0$; in everyday modelling "linear" often means affine, and Chapter M5 will be careful about the difference.
- **Polynomial:** $f(x) = a_0 + a_1 x + \cdots + a_m x^m$. Quadratics ($m=2$) describe the area of a growing colony and the leading behaviour of any smooth function near a minimum (Chapter M3).
- **Piecewise:** different formulas on different intervals, for example the **ReLU** $\max(0, x)$, which is zero for negative inputs and the identity for positive ones.

!!! tip "Slope is the unit conversion"
    A slope is always a ratio of units: expression change per hour, fitness change per mutation, signal per concentration. Writing the units next to every slope you compute prevents most interpretation errors, and tells you immediately whether two slopes can be compared.

---

## M2.2 Exponentials: growth, decay, and the number $e$

A quantity grows **exponentially** when its rate of increase is proportional to its current size. Each bacterium divides, so more bacteria produce more new bacteria. The function is

$$
N(t) \;=\; N_0\, b^{\,t}\qquad\text{or equivalently}\qquad N(t) = N_0\, e^{kt},
$$

where $b>1$ for growth and $0<b<1$ for decay, $N_0$ is the initial amount, and $k=\ln b$ is the continuous **rate**. The **number $e\approx 2.71828$** is the base for which the rate of increase equals the function itself; it is the base in which calculus is simplest (Chapter M3). Any base can be converted: $b^t = e^{t\ln b}$.

**Three equivalent descriptions** of the same process, each used in a different field:

| Description | Growth ($k>0$) | Decay ($k<0$) |
|---|---|---|
| Rate constant | $N = N_0 e^{kt}$ | $N = N_0 e^{-\lambda t}$ |
| Doubling / half-life | $t_{2} = \ln 2/k$ | $t_{1/2} = \ln 2/\lambda$ |
| Fold per step | $N_{c} = N_0 (1+E)^c$ per cycle $c$ | $N = N_0 \cdot 2^{-t/t_{1/2}}$ |

Memorize the middle row. $\ln 2\approx 0.693$, so $t_2\approx 0.69/k$ (the "rule of 70" in finance and in epidemiology).

!!! bio "Biology for modeling: PCR, mRNA half-lives, and what exponentials hide"
    **PCR.** Each cycle of the polymerase chain reaction ideally doubles the template, but real efficiency $E$ is below 1: $N_c = N_0(1+E)^c$. In the script below, with $E = 0.92$, a $10^6$-fold amplification needs $\ln 10^6/\ln 1.92 = 21.2$ cycles rather than $20$; the doubling takes 1.06 cycles. A small efficiency error is amplified *exponentially*: a 5% shortfall per cycle becomes a factor $(0.95)^{30}\approx 0.21$, a five-fold underestimate after 30 cycles. That is why quantitative PCR measures the *cycle at which a threshold is crossed* and why amplification bias is a core worry in sequencing library preparation.

    **mRNA decay.** A transcript with half-life $t_{1/2} = 4$ h decays with rate $\lambda = \ln 2/4 = 0.173\ \text{h}^{-1}$; after 12 h (three half-lives) $1/8$ remains. Half-lives of messenger RNAs span minutes to many hours, and they determine how quickly a gene's level can respond to a change in transcription (Chapter M3 derives this from a differential equation).

    **What exponentials hide.** Real growth is exponential only while nothing is limiting. Colonies, tumours and epidemics bend into logistic curves (§M2.5) as resources run out. Fitting an exponential to the early part of such data and extrapolating is a classic forecasting error.

**Geometric series connection.** If a process repeats in discrete steps, $N_c = N_0 r^{c}$ is a geometric sequence, and the sums in Chapter M1 apply: total signal accumulated over many steps is $N_0(1 - r^{C+1})/(1-r)$.

---

## M2.3 Logarithms

The **logarithm** is the inverse of the exponential. For a base $b>0$, $b\ne1$:

$$
y = \log_b x \quad\Longleftrightarrow\quad b^{\,y} = x .
$$

"The logarithm of $x$ is the power to which $b$ must be raised to get $x$." We write $\ln$ for the natural log (base $e$), $\log_{10}$ for base 10, and $\log_2$ for base 2. In this book $\log$ without a base means the natural log unless stated.

!!! math "The five logarithm rules (and why they hold)"
    For $x, y > 0$:

    1. $\log(xy) = \log x + \log y$ (multiplying means adding exponents: $b^{u}b^{v} = b^{u+v}$).
    2. $\log(x/y) = \log x - \log y$.
    3. $\log(x^{p}) = p\log x$.
    4. $\log_b x = \ln x / \ln b$ (change of base). So all logs are constant multiples of each other.
    5. $\log 1 = 0$, and $\log$ is **increasing**: $x<y \iff \log x<\log y$.

    **Why they matter.** Rule 1 turns a *product* into a *sum*. A sum is far easier to differentiate, to average, and to compute without underflow (§M2.7). Rule 3 turns a *power law* into a *line* (§M2.4).

**Why logarithmic scales appear in biology.** Many biological quantities are produced by *multiplicative* processes (each step scales the previous), spread over orders of magnitude, and perceived or measured on a ratio basis. Examples: pH ($=-\log_{10}[\text{H}^+]$), sound level (decibels), the fold change of a gene ($\log_2$ of the expression ratio), and Gibbs free energy, which is $\Delta G = -RT\ln K_{\text{eq}}$ (Chapter 22). Log scales make a ratio of 2 look the same whether the starting level is 10 or 10,000.

**Log fold change.** Raw ratios are asymmetric: a gene that doubles has ratio 2, a gene that halves has ratio 0.5, and the *average* of those two is 1.25 ("on average up"), which is wrong. In $\log_2$ they are $+1$ and $-1$ and average to 0 (script output). Up- and down-regulation are symmetric on the log scale, and equal biological effects get equal numerical size. This is why differential expression is analysed in $\log_2$ fold change.

**Logs of zero.** $\log 0$ is undefined (it tends to $-\infty$), and single-cell count data contain many zeros. The standard remedy is $\log(1 + x)$ (`log1p`) on counts, which maps 0 to 0 and behaves like $x$ for small $x$ and like $\log x$ for large $x$ (script: counts 0, 1, 4, 20, 150, 1200 map to 0, 0.69, 1.61, 3.04, 5.02, 7.09). Note that this is a *modelling choice*: the pseudo-count of 1 is arbitrary, and the log-transform changes what a "difference" means. Chapter 25 returns to how much this choice matters.

---

## M2.4 Power laws and log-log plots

A **power law** has the form $y = c\,x^{p}$. Taking logs: $\log y = \log c + p\log x$, a straight line in $(\log x, \log y)$ with slope $p$, the **exponent**. Power laws occur across biology, usually with a mechanism behind them:

- **Allometric scaling.** The metabolic rate of mammals scales roughly as (body mass)$^{3/4}$ (Kleiber's law). A 10,000-fold increase in mass raises metabolic rate by $10^{4\cdot 0.75}= 1000$-fold, not 10,000 (script). The exponent, not the prefactor, carries the biology: it says larger animals are *more efficient per unit mass*.
- **Degree distributions** of protein-interaction networks (a few hub proteins with very many partners).
- **Zipf-like frequency distributions** of $k$-mers and of expressed genes.

!!! warning "Be careful: many things are claimed to be power laws"
    A straight-looking segment on a log-log plot does not prove a power law: log-normal, stretched-exponential and other heavy-tailed distributions can look straight over one or two decades. Fit candidate models by maximum likelihood (Chapter M8) and compare, instead of reading a slope off a plot. For *regression* relationships like Kleiber's law, the exponent estimate is well defined and a log-log line is the standard method (script: fitted exponent 0.748 for a true 0.75 with 15% multiplicative noise). The noise in such data is multiplicative, so fitting in log space is also the *statistically* appropriate choice.

---

## M2.5 Saturating and sigmoidal functions

Biology saturates: receptors fill, enzymes reach $V_{\max}$, populations hit a carrying capacity. Three standard shapes:

1. **Michaelis–Menten / hyperbolic:** $f(x) = \dfrac{V x}{K + x}$. At $x = K$ the output is half of $V$.
2. **Hill function** (sigmoidal): $f(x) = \dfrac{x^{n}}{K^{n} + x^{n}}$. Here $K$ is the **half-saturation** level and $n$ the **Hill coefficient**, the steepness. For $n = 1$ it is Michaelis–Menten; for $n > 1$ the response is switch-like, which is how cooperative binding, such as multiple transcription-factor sites on one promoter, creates thresholds.
3. **Logistic (sigmoid):** $\sigma(x) = \dfrac{1}{1 + e^{-x}}$, which maps any real number to $(0,1)$. It is the Hill function in disguise: with $u = e^{x}$ the logistic becomes $u/(1+u)$, the $n=1$, $K=1$ Hill function of $u$, and a Hill function with general $n$ is a logistic function of $\ln x$ with slope $n$. It is *the* output nonlinearity for probabilities.

Steepness is easy to quantify: the ratio of input levels needed to go from 10% to 90% output is $81^{1/n}$. With $n = 1$ that is an **81-fold** range, $n = 2$ is 9-fold, and $n = 4$ is just 3-fold (script). A promoter that must switch on over a 3-fold change of regulator concentration needs cooperativity near $n=4$.

!!! bio "Biology for modeling: Hill functions and why neural networks use sigmoids"
    **What is it?** A promoter with $n$ cooperative binding sites is occupied with probability $f(x)$, and the transcription rate is proportional to occupancy. This is the standard "thermodynamic model" of gene regulation (Chapter 22).

    **What it discards.** Detailed binding configurations, nucleosome competition, and the stochastic nature of single binding events; it describes the *average* response of a population of promoters.

    **Modelling consequence.** A feedforward neural network layer applies an affine map and then a nonlinearity such as the sigmoid. Early neuron models were literally Hill-type switches. A regulatory network of genes is therefore a recurrent network of sigmoids (Chapter 19), and the mathematics of neural nets applies to gene circuits (Chapter 9).

---

## M2.6 Oscillations: sines, cosines, and phase

Cell cycles, circadian rhythms, calcium waves and heartbeats are periodic. A pure oscillation is

$$
y(t) \;=\; A\cos\!\Big(\frac{2\pi t}{T} - \varphi\Big) + c ,
$$

with **amplitude** $A$, **period** $T$ (so frequency $1/T$), **phase** $\varphi$ (when the peak occurs: the peak is at $t = \varphi T/2\pi$), and baseline $c$. By the angle-addition identity $\cos(\omega t - \varphi) = \cos\varphi\cos\omega t + \sin\varphi\sin\omega t$, any such oscillation with a *known* period is a **linear** combination of $\cos\omega t$ and $\sin\omega t$, so its amplitude and phase can be estimated by **linear regression** on those two columns. The amplitude is $\sqrt{a^2 + b^2}$ for coefficients $a, b$ and the phase is $\operatorname{atan2}(b, a)$. Scanning over candidate periods and choosing the best fit is the simplest form of a **periodogram** (Fourier analysis, Chapter 11).

In the script, a gene sampled every 2 h for 72 h with a true period of 24 h, amplitude 2 and noise standard deviation 0.3 is recovered at period $24.0$ h and amplitude $1.90$.

!!! tip "A preview: complex numbers make rotation simple"
    Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ (with $i^2 = -1$) packs a cosine and a sine into one exponential. Multiplying by $e^{i\theta}$ *rotates* a point in the plane by angle $\theta$. This is why oscillations, Fourier transforms, state-space sequence models (Chapter 11) and rotary position embeddings (Chapter 12) all use complex exponentials. You do not need complex numbers until then; just remember that rotation and exponentiation are connected.

---

## M2.7 Composition, inverses, and numerical safety

**Composition and layers.** A neural-network layer is the composition of an affine map $\mathbf{x}\mapsto\mathbf{W}\mathbf{x}+\mathbf{b}$ with a scalar nonlinearity (sigmoid, ReLU, ...) applied to each coordinate. A deep network repeats this. Without the nonlinearity, the composition of affine maps is *still affine*, so depth would add nothing; **nonlinearity is what makes composition expressive** (Chapter 9).

**Inverses.** A function that is increasing everywhere has an inverse: $\ln$ inverts $\exp$; the logit $\ln\frac{p}{1-p}$ inverts the sigmoid. The **logit** maps a probability in $(0,1)$ to the whole real line, which is why logistic regression models the *log-odds* linearly.

**The numerical-safety rule: use the log of a sum of exponentials carefully.** The softmax converts scores $z_1,\dots,z_K$ into probabilities $p_k = e^{z_k}/\sum_j e^{z_j}$, so computing it requires $\log\sum_j e^{z_j}$. With $z=(1000, 1001, 1002)$, $e^{1000}$ overflows to infinity in double precision, so the naive formula returns `inf`. The shift trick fixes it: with $m=\max_j z_j$,

$$
\log\sum_j e^{z_j} \;=\; m + \log\sum_j e^{\,z_j - m},
$$

which is exact (factor $e^m$ out of the sum) and safe because every exponent is $\le 0$. The script prints $1002.4076$ against `inf` for the naive version, with probabilities $(0.090, 0.245, 0.665)$. The same issue arises with long products: the product of 2,000 probabilities each between 0.2 and 0.9 underflows to exactly $0.0$, while the sum of their logs, $-1366.5$, is perfectly well defined. **Rule: compute with log-probabilities, and combine them with log-sum-exp.** Every probabilistic model in the later chapters follows this rule.

```python
--8<-- "code/m02_functions.py"
```

Output (seed 0):

```text
PCR: true efficiency 0.92; estimated from the log-line slope 0.920; cycles for a 1e6-fold increase: 21.2
doubling time with 100% efficiency: 1 cycle;  with 92%: 1.06 cycles
mRNA with half-life 4.0 h: decay rate k = 0.1733 /h; fraction left after 12 h = 0.125 (= 1/8)

power law: fitted exponent 0.748 (true 0.75), prefactor 3.55 (true 3.40)
a 10,000-fold increase in mass multiplies the rate by 1000 (not 10,000)

Hill function, K = 1:    x = [0.25 0.5  1.   2.   4.  ]
  n = 1:  f(x) = [0.2   0.333 0.5   0.667 0.8  ]   x for 10%->90%: 81.00-fold range
  n = 2:  f(x) = [0.059 0.2   0.5   0.8   0.941]   x for 10%->90%: 9.00-fold range
  n = 4:  f(x) = [0.004 0.059 0.5   0.941 0.996]   x for 10%->90%: 3.00-fold range

log2 fold change: 2x up = 1.0 | 2x down = -1.0 | 8x up = 3.0
raw ratios 2 and 0.5 average to 1.25 (wrongly >1); logs average to 0.0
log1p(counts) =  [0.   0.69 1.61 3.04 5.02 7.09]   (log(0) is undefined, so log1p is used for counts)

log(sum(exp(z))) for z = [1000, 1001, 1002]: naive = inf, shifted = 1002.407606
softmax probabilities: [0.09   0.2447 0.6652] sum = 1.0
product of 2000 probabilities = 0.0, but sum of logs = -1366.5

circadian fit: period 24.0 h (true 24), amplitude 1.90 (true 2.00), baseline 5.00 (true 5.00)
```

---

## M2.8 Worked examples

!!! example "Worked example M2.1: Is it exponential, a power law, or neither?"
    **Situation.** You measure the number of cells in a culture every 2 hours for 12 hours: 1,000; 1,400; 2,000; 2,800; 3,900; 5,500; 7,700.

    **Question.** Which model fits better: exponential ($N = N_0 e^{kt}$) or power law ($N = c\,t^{p}$)? How do you decide, and what is the doubling time?

    **Reasoning.**

    1. *Linearize each model.* Exponential: $\ln N$ is linear in $t$. Power law: $\ln N$ is linear in $\ln t$ (and $t=0$ is excluded, so a power law cannot even describe the first data point).
    2. *Check ratios.* Successive ratios are $1.40, 1.43, 1.40, 1.39, 1.41, 1.40$: nearly constant. A constant ratio per equal time step is the *signature* of exponential growth. For a power law the ratio between equally spaced steps would shrink.
    3. *Estimate the rate.* The ratio per 2 h is about 1.40, so $k = \ln 1.40/2 = 0.168\ \text{h}^{-1}$ and the doubling time is $\ln 2/k = 4.1$ h.
    4. *What would change the conclusion?* If the culture is running out of nutrients, later ratios fall below 1.40. Here they do not, over a $\sim7$-fold range, so the data are consistent with unlimited exponential growth *over this window*.

    **Lesson.** Look at ratios (for exponentials) and at log-log slopes (for power laws) before fitting. And always say over *what range* the model has been checked.

!!! example "Worked example M2.2: Reading a dose-response curve"
    **Situation.** A drug inhibits an enzyme. The response (fraction of activity inhibited) is $f(x) = x^{n}/(K^{n}+x^{n})$ with $K = 50$ nM. Experiment A gives $n=1$, experiment B gives $n=3$.

    **Question.** How do the two curves differ at 25 nM and 100 nM, and what does the difference suggest mechanistically?

    **Reasoning.** At 25 nM ($x/K = 0.5$): $n=1$ gives $0.5/1.5 = 0.33$; $n = 3$ gives $0.125/1.125 = 0.11$. At 100 nM ($x/K=2$): $n=1$ gives $0.67$; $n=3$ gives $8/9 = 0.89$. The half-effect point is the same (50 nM) but curve B is *switch-like*: ignorable below $K$, nearly complete above. A Hill coefficient greater than 1 suggests cooperative binding (several sites that help each other) or a downstream threshold mechanism; it does *not* by itself prove cooperativity, since multi-step signalling cascades also steepen responses.

    **What to do next.** Measure at several doses around $K$ to estimate $n$ with error bars; compare with a model with $n=1$ plus a threshold; look for mutations of the putative cooperating sites.

---

## M2.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The mean log fold change is 0.1, so the effect is negligible\""
    **The observation.** Across 10,000 cells, the average $\log_2$ fold change of a transcription factor's target gene after perturbation is only $+0.1$ (about 7% increase). The authors conclude the gene is barely affected.

    **Tempting conclusion.** "A 7% change is small, so the perturbation does not matter for this gene."

    **Decompose.** A *mean* over cells hides the *distribution*. Suppose 5% of cells upregulate the gene four-fold ($\log_2 = +2$) and 95% do not change. The mean is $0.05\times2 = +0.1$, exactly the reported value, yet those 5% of cells could be the biologically important sub-population, the ones that adopt a new state. Second, the **transformation** matters: averaging $\log(1+x)$ of counts is not the log of the average count, because log is a nonlinear function and averages do not commute with nonlinear functions (Jensen's inequality, Chapter M3). Third, a small average effect on a *high-expression* gene can mean large absolute changes in molecules.

    **Hidden assumptions.** (i) The effect is the same in every cell. (ii) The log transform with a pseudo-count of 1 is a faithful scale for these counts. (iii) Fold change is the effect size that matters (absolute change may matter more for low-abundance regulators).

    **Discriminating experiments.** Plot the full distribution of per-cell expression before and after; fit a two-component mixture; compare the fraction of responding cells and the response size within responders.

    **What this teaches.** The function you apply to data (a log, a mean, a ratio) is part of your model. Before interpreting a number, ask what it would look like on the raw scale, per cell, and under a different reasonable transformation.

---

## M2.10 Connections

- **Forward:** exponentials and logarithms return as probabilities and log-likelihoods (M7, M8), entropy (Chapter 5), softmax and cross-entropy (Chapters 3, 9, 12); the Hill function as the sigmoid (Chapters 9, 22); oscillations as Fourier and state-space models (Chapter 11); log-sum-exp as the numerically stable form of every partition function (Chapters 22, 29); power laws and scaling (Chapter 47).
- **Backward:** sums, functions, and counting (M1).

!!! takeaways "Key takeaways"
    1. A model has a **form** (from mechanism) and **parameters** (from data). The form is the scientific commitment.
    2. Exponential growth/decay: $N=N_0e^{kt}$, doubling time or half-life $=\ln2/|k|$. Small per-step errors compound exponentially (PCR efficiency 0.92 needs 21.2 cycles for $10^6$-fold).
    3. $\log$ turns products into sums, powers into multiples, and power laws into lines. Use $\log_2$ fold change for symmetry, $\log(1+x)$ for counts, and log-probabilities to avoid underflow.
    4. A power law $y=cx^p$ is a line on a log-log plot with slope $p$; but many heavy-tailed distributions mimic one, so compare models by likelihood.
    5. The Hill function $x^n/(K^n+x^n)$ has threshold $K$ and steepness $n$; the 10%→90% input range is $81^{1/n}$-fold.
    6. A periodic signal with known period is a linear combination of $\cos$ and $\sin$; amplitude is $\sqrt{a^2+b^2}$.
    7. A neural-network layer is an affine map followed by a nonlinearity; without the nonlinearity, depth adds nothing.
    8. Compute $\log\sum e^{z}$ as $m + \log\sum e^{z-m}$; this is how softmax and every partition function is computed safely.

---

## Further reading

- Strogatz, S. H. *Nonlinear Dynamics and Chaos* (chapters on one-dimensional flows). Exponentials and logistic growth from the dynamics view.
- Alon, U. *An Introduction to Systems Biology: Design Principles of Biological Circuits*. CRC Press. Hill functions, gene regulation and network motifs, with minimal prerequisites.
- Phillips, R., Kondev, J., Theriot, J. & Garcia, H. *Physical Biology of the Cell*. Garland Science. Scale, exponentials and statistical mechanics for cell biology.
- Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review* 51, 661–703. How to test a power-law claim properly.
- West, G. B., Brown, J. H. & Enquist, B. J. (1997). A general model for the origin of allometric scaling laws in biology. *Science* 276, 122–126.
