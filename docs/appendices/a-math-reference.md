# Appendix A. Mathematical Reference

A compact collection of the identities and results used repeatedly in the book, each with the chapter in which it is derived. Notation follows the notation page: sequences $x_{1:L}$, batch $B$, model width $d$, latent state $s$, parameters $\theta$.

!!! tip "If this appendix moves too fast"
    Every topic here is taught from the ground up in **Part 0**: calculus in Chapters [M3](../chapters/m03-single-variable-calculus.md) and [M4](../chapters/m04-multivariable-calculus-optimization.md), linear algebra in [M5](../chapters/m05-linear-algebra-1.md) and [M6](../chapters/m06-linear-algebra-2.md), probability and statistics in [M7](../chapters/m07-probability.md) and [M8](../chapters/m08-statistics.md), discrete mathematics in [M9](../chapters/m09-discrete-structures.md).

---

## A.1 Linear algebra (Chapter 2)

* **Spectral theorem.** A real symmetric $A$ has $A=Q\Lambda Q^\top$ with orthonormal $Q$. $\tr A=\sum\lambda_i$, $\det A=\prod\lambda_i$.
* **SVD.** $X=U\Sigma V^\top$; rank-$k$ truncation $X_k$ minimizes $\lVert X-M\rVert_F$ over rank-$k$ matrices (Eckart–Young); $\lVert X-X_k\rVert_F^2=\sum_{i>k}\sigma_i^2$.
* **PCA.** Principal components are eigenvectors of the covariance $C=\tfrac1nX^\top X$ (centered $X$); explained variance of component $i$ is $\lambda_i/\sum\lambda$.
* **Condition number.** $\kappa(A)=\sigma_{\max}/\sigma_{\min}$; gradient descent on a quadratic with Hessian $H$ converges at rate $\big(\tfrac{\kappa-1}{\kappa+1}\big)^t$.
* **Matrix inversion lemma.** $(A+UCV)^{-1}=A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$ (used in the dual form of ridge regression and the LMM, Chapter 26).
* **Ridge.** $\hat\beta=(X^\top X+\lambda I)^{-1}X^\top y=X^\top(XX^\top+\lambda I)^{-1}y$.

## A.2 Calculus and optimization (Chapter 3)

* **Gradient of a quadratic.** $\nabla\tfrac12x^\top Ax-b^\top x=Ax-b$.
* **Chain rule (matrix form).** For $y=f(Wx)$: $\partial L/\partial W=\delta\,x^\top$ with $\delta=\partial L/\partial(Wx)$ (Chapter 9).
* **Gradient descent.** $\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$; stable for $\eta<2/\lambda_{\max}(H)$.
* **Adam.** $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, $\theta_{t+1}=\theta_t-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ with bias corrections $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$.
* **Log-sum-exp and softmax.** $\mathrm{LSE}(z)=\log\sum_je^{z_j}$; $\mathrm{softmax}(z)_i=e^{z_i}/\sum_je^{z_j}=\partial\,\mathrm{LSE}/\partial z_i$; stable form $z-\max z$.
* **Softmax cross-entropy gradient.** $\partial L/\partial z=\mathrm{softmax}(z)-y$.

## A.3 Probability (Chapter 4)

* **Law of total variance.** $\Var(Y)=\E[\Var(Y\mid X)]+\Var(\E[Y\mid X])$.
* **Bayes.** $p(\theta\mid x)=p(x\mid\theta)p(\theta)/p(x)$.
* **Poisson.** $P(k)=e^{-\lambda}\lambda^k/k!$; mean and variance $\lambda$.
* **Gamma–Poisson = negative binomial.** $\lambda\sim\mathrm{Gamma}(r,\text{scale }\mu/r)$, $y\mid\lambda\sim\mathrm{Poisson}(\lambda)$ gives mean $\mu$, variance $\mu+\mu^2/r$ (dispersion $\phi=1/r$). $P(y=0)=(1+\phi\mu)^{-1/\phi}$. Binomial thinning with probability $p$ gives $\mathrm{NB}(p\mu,\phi)$ (Chapter 25).
* **Beta–binomial; Dirichlet–multinomial.** Conjugate pairs; posterior mean $(n_k+\alpha_k)/(n+\sum\alpha)$.
* **Gaussian conditioning.** For $(x,y)$ jointly Gaussian: $y\mid x\sim\mathcal N\big(\mu_y+\Sigma_{yx}\Sigma_{xx}^{-1}(x-\mu_x),\ \Sigma_{yy}-\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}\big)$.
* **Fisher information and Cramér–Rao.** $\Var(\hat\theta)\ge1/(nI(\theta))$ for unbiased estimators.
* **Spearman–Brown.** Reliability of the mean of $n$ replicates each with reliability $\rho$: $n\rho/(1+(n-1)\rho)$. Split-half: $\rho_\text{full}=2r/(1+r)$.
* **Noise ceiling.** If $y_\text{obs}=y_\text{true}+\varepsilon$, $\Var\varepsilon=\sigma^2$, the maximum $R^2$ of any predictor of $y_\text{obs}$ is $1-\sigma^2/\Var(y_\text{obs})$.

## A.4 Information theory (Chapter 5)

* **Entropy.** $H(X)=-\sum p\log p$; **KL divergence** $\KL{p}{q}=\sum p\log(p/q)\ge0$; **mutual information** $I(X;Y)=H(X)-H(X\mid Y)=\KL{p(x,y)}{p(x)p(y)}$.
* **Data-processing inequality.** If $X\to Y\to Z$ is a Markov chain, $I(X;Z)\le I(X;Y)$.
* **Fano.** $P_e\ge\big(H(X\mid Y)-1\big)/\log(|\mathcal X|-1)$.
* **Cross-entropy and compression.** Expected code length under model $q$ is $H(p)+\KL{p}{q}$ bits (nats); perplexity $=2^{\text{bits}}$.
* **Gaussian entropy.** $H=\tfrac12\log\big((2\pi e)^d\det\Sigma\big)$.
* **Sufficiency guarantee (Chapter 17).** For a representation $h=s(x)$ of $x$ used for $y$: $I(s(x);y)\le I(h;y)\le I(x;y)$ along any processing chain.

## A.5 Latent-variable models (Chapters 8, 14)

* **ELBO.** $\log p_\theta(x)\ge\mathbb E_{q(z\mid x)}[\log p_\theta(x\mid z)]-\KL{q(z\mid x)}{p(z)}$; gap $=\KL{q(z\mid x)}{p_\theta(z\mid x)}$.
* **EM.** E-step: $q^{(t)}(z)=p(z\mid x,\theta^{(t)})$; M-step: $\theta^{(t+1)}=\arg\max\,\mathbb E_{q^{(t)}}\log p(x,z\mid\theta)$.
* **Reparameterization.** $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$; Gaussian KL to $\mathcal N(0,I)$: $\tfrac12\sum(\mu^2+\sigma^2-1-\log\sigma^2)$.

## A.6 Neural-network building blocks (Chapters 9–12)

* **Convolution** (1-D, kernel $k$, $C_\text{in}\to C_\text{out}$): parameters $kC_\text{in}C_\text{out}$, cost $O(LkC_\text{in}C_\text{out})$; dilation $r$ expands the receptive field to $1+r(k-1)$ per layer; $n$ layers with dilations $2^i$ have receptive field $\sim(k-1)(2^n-1)+1$.
* **Attention.** $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k}\big)V$; cost $O(L^2d)$; the scale $1/\sqrt{d_k}$ keeps logits' variance near 1 when entries have unit variance.
* **He initialization.** $\Var(W_{ij})=2/n_\text{in}$ for ReLU layers preserves activation variance.
* **State-space model.** $\dot x=Ax+Bu$, $y=Cx$; zero-order-hold discretization $\bar A=e^{A\Delta}$, $\bar B=A^{-1}(e^{A\Delta}-I)B$; recurrence $x_t=\bar Ax_{t-1}+\bar Bu_t$ equals convolution with kernel $K_k=C\bar A^k\bar B$.
* **Rotary position embedding.** Rotates query/key pairs by angle proportional to position so that $\langle q_m,k_n\rangle$ depends on $m-n$.

## A.7 Generative models (Chapters 14–15)

* **DDPM forward process.** $q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\big)$, $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$; noise-prediction loss $\mathbb E\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2$.
* **Tweedie.** $\mathbb E[x_0\mid x_t]=\big(x_t+(1-\bar\alpha_t)\nabla\log p_t(x_t)\big)/\sqrt{\bar\alpha_t}$.
* **Flow matching.** Regress $v_\theta(x_t,t)$ on $u_t=x_1-x_0$ along $x_t=(1-t)x_0+tx_1$.
* **Classifier-free guidance.** $\tilde\varepsilon=(1+w)\varepsilon_\theta(x_t,c)-w\varepsilon_\theta(x_t,\varnothing)$.

## A.8 Representation learning and geometry (Chapters 13, 16)

* **InfoNCE.** $I(x;y)\ge\log N-\mathcal L_\text{InfoNCE}$ for $N-1$ negatives.
* **GCN layer.** $H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$.
* **Equivariance.** $f(g\cdot x)=g\cdot f(x)$; $E(n)$-equivariant message passing updates coordinates by $x_i\leftarrow x_i+\sum_j(x_i-x_j)\phi(\ldots)$ with invariant $\phi$.

## A.9 Scaling (Chapter 17)

* **Loss–compute relations.** $L(N,D)\approx E+A/N^\alpha+B/D^\beta$; compute $C\approx6ND$; compute-optimal $N^*\propto C^{a}$, $D^*\propto C^{b}$ with $a+b=1$.

## A.10 Population genetics (Chapters 20–21)

* **Hardy–Weinberg.** Genotype frequencies $p^2,2pq,q^2$.
* **Wright–Fisher drift.** Variance of allele-frequency change per generation $p(1-p)/(2N)$; heterozygosity decays as $(1-1/2N)^t$.
* **Fixation probability** of a new mutation with selection $s$ in a population of size $N$ (Kimura): $u=\dfrac{1-e^{-2s}}{1-e^{-4Ns}}\approx2s$ for $Ns\gg1$; neutral: $1/(2N)$.
* **Coalescent.** Time to the most recent common ancestor of two lineages is exponential with mean $2N$ generations ($k$ lineages: rate $\binom k2/(2N)$); expected number of segregating sites $S=\theta\sum_{i=1}^{n-1}1/i$ with $\theta=4N\mu$.
* **Mutation–selection equilibrium** (Sella–Hirsh): $\pi_i\propto g_i\,e^{S_i}$, where $g_i$ is the mutational (neutral) distribution over alleles and $S_i$ the scaled fitness (Chapter 21); the stationary distribution is the mutation distribution reweighted by fitness.
* **LD.** $D=p_{AB}-p_Ap_B$, $r^2=D^2/(p_A(1-p_A)p_B(1-p_B))$; decays as $(1-c)^t$ with recombination fraction $c$.

## A.11 Thermodynamics of binding and folding (Chapters 22–24)

* **Two-state equilibria.** Binding occupancy $\theta=\dfrac{[L]}{K_d+[L]}$; folded fraction $f=\dfrac1{1+e^{\Delta G/RT}}$ (with $\Delta G<0$ stable); $RT=0.593$ kcal/mol at 298 K.
* **Free energy and affinity.** $\Delta G^\circ=-RT\ln10\cdot pK_d=-1.37\,pK_d$ kcal/mol.
* **Cheng–Prusoff.** $IC_{50}=K_i(1+[S]/K_m)$ for a competitive inhibitor.
* **Kinetics.** $K_d=k_\text{off}/k_\text{on}$; residence time $1/k_\text{off}$.
* **Thermodynamic cycle.** $\Delta\Delta G_\text{bind}(A\to B)=\Delta G_\text{prot}(A\to B)-\Delta G_\text{solv}(A\to B)$.
* **Statistical weights.** $P(\text{state})=e^{-E/RT}/Z$ with $Z=\sum e^{-E/RT}$ (softmax over configurations).

## A.12 Alignment statistics (Chapter 27)

* **Scores as log-odds.** $s_{ab}=\lambda^{-1}\ln\dfrac{q_{ab}}{p_ap_b}$, with $\lambda$ the positive root of $\sum p_ap_be^{\lambda s_{ab}}=1$.
* **Karlin–Altschul.** $E=Kmn\,e^{-\lambda S}$; bit score $S'=(\lambda S-\ln K)/\ln2$; Gumbel maximum with scale $1/\lambda$ and standard deviation $\pi/(\lambda\sqrt6)$.
* **Backward search (FM-index).** $lo'=C[c]+\mathrm{Occ}(c,lo)$, $hi'=C[c]+\mathrm{Occ}(c,hi)$.
* **Genotype likelihood.** $P(k\mid g)=\binom dk\theta_g^k(1-\theta_g)^{d-k}$ with $\theta_g\in\{\epsilon,\tfrac12,1-\epsilon\}$.

## A.13 Statistical genetics (Chapter 26)

* **Heritability.** $h^2=\beta^\top R\beta$ (standardized); Falconer $h^2=2(r_\text{MZ}-r_\text{DZ})$; liability scale $h^2_\ell=h^2_\text{obs}K(1-K)/\varphi(\Phi^{-1}(K))^2$.
* **Association signal.** $z\sim\mathcal N(\sqrt nR\beta,R)$.
* **LD score regression.** $\E\chi^2_j=1+na+nh^2\ell_j/M$.
* **LMM = ridge.** $\hat g=\sigma_g^2K(\sigma_g^2K+\sigma_e^2I)^{-1}y=X\hat\beta_\text{ridge}$, $\lambda=\sigma_e^2M/\sigma_g^2$.
* **Wakefield ABF.** $\mathrm{BF}=\sqrt{\tfrac V{V+W}}\exp\big(\tfrac{z^2}2\tfrac W{V+W}\big)$; $\mathrm{PIP}_j\propto\pi_j\mathrm{BF}_j$.
* **PGS accuracy.** $R^2=h^2\cdot Nh^2/(Nh^2+M)$.
* **MR.** Wald ratio $\hat\beta_Y/\hat\beta_X$; IVW $=\sum w\hat\beta_X\hat\beta_Y/\sum w\hat\beta_X^2$.

## A.14 Single-cell measurement (Chapter 25)

* **Reliability of a single-cell count.** $\rho(\lambda)=\phi\lambda/(1+\phi\lambda)$ for mean UMI $\lambda$ and dispersion $\phi$.
* **Compositional artifact.** Apparent $\log_2$ fold change of unchanged genes: $-\log_2\big(1+f(\rho-1)\big)$ when genes carrying fraction $f$ are multiplied by $\rho$.
* **Doublet fraction.** $\big(1-e^{-\lambda}-\lambda e^{-\lambda}\big)/\big(1-e^{-\lambda}\big)\approx\lambda/2$.
* **Pseudoreplication.** Effective sample size equals the number of independent units (donors).

## A.15 Experimental design and ideas (Chapter 56)

* **EIG.** $I(H;O)=H(\pi)-\mathbb E_o[H(\pi(\cdot\mid o))]$.
* **EVPI.** $\E_\theta[\max_aU(a,\theta)]-\max_a\E_\theta[U(a,\theta)]$.
* **Sample size for paired comparisons.** $n\approx\big((z_{1-\alpha/2}+z_{1-\beta})s/\Delta\big)^2$.
