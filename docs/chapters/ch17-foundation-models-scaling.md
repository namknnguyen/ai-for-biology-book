# Chapter 17. Foundation Models: Scaling, Adaptation, and Alignment

!!! abstract "Chapter at a glance"
    **Motivation.** "Foundation model" is the organizing idea of Part VII: train one large model on broad unlabeled data and adapt it to many tasks. Whether this works for biology depends on three questions this chapter formalizes: *what does scaling buy*, *how is a pretrained model adapted*, and *what is guaranteed to transfer*. The chapter also develops the formal version of Chapter 1's **objective gap**.
    **Prerequisites.** Chapters 3–5, 6 (compute), 12, 13, 14.
    **You will be able to:** (1) state and fit a scaling law, including the role of the irreducible term, and derive compute-optimal allocation; (2) critique claims about "emergence" and about extrapolating scaling curves; (3) explain linear probing, fine-tuning, LoRA, prompting, and retrieval, and when each is appropriate; (4) derive the KL-regularized reward optimum underlying RLHF/DPO and see its connection to selection; (5) state the **sufficiency guarantee** and use it to predict which downstream tasks a pretrained biological model can support; (6) decide whether scaling is the right next experiment.

---

## 17.1 What is a foundation model?

Bommasani et al. (2021) define a **foundation model** as a model trained on broad data (generally with self-supervision at scale) that can be adapted to a wide range of downstream tasks. The definition has three components, each of which can fail in biology:

1. **Broad data.** Breadth of *what*: species, cell types, assays, conditions? A model trained on human reference DNA is broad across genomic loci but narrow across species, individuals, and cellular contexts.
2. **Self-supervision at scale.** The objective is unlabeled (Chapter 13); scale means parameters, tokens, and compute large enough that new behaviors appear.
3. **Adaptability.** One model, many tasks, with little task-specific data.

In biology there are four ways a pretrained model can be *used*, ordered by how much they ask of the pretraining:

| Use | What it demands of pretraining | Example |
|---|---|---|
| **Zero-shot scoring** | The pretraining distribution *is* the target quantity (e.g., $\Delta\log p\approx$ fitness) | Variant-effect prediction from likelihood ratios (Chapters 5, 32, 34) |
| **Feature extraction + probe** | Embeddings make task information linearly accessible | Cell-type annotation from single-cell embeddings |
| **Fine-tuning / adaptation** | The pretrained weights are a better *initialization* than random | Fine-tune a protein LM on 200 labeled mutants |
| **Generation / design** | The model's samples are valid and diverse objects | Genome and protein design (Chapters 32, 36) |

A genuine foundation model should succeed on **all four** across diverse tasks; several current biological models succeed on some. A practical diagnostic: *does it beat a simple, task-specific baseline with the same labeled data?* (§17.7 and Chapters 31–32, 38–39, which report that in several regimes it does not.)

---

## 17.2 Scaling laws

### 17.2.1 The empirical regularity

For autoregressive and masked language models, the held-out loss (cross-entropy; Chapter 5) follows smooth **power laws** in model size $N$ (parameters), dataset size $D$ (tokens), and compute $C$, over many orders of magnitude (Kaplan et al., 2020; Hoffmann et al., 2022). A convenient parametric form is

$$
\boxed{L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}}
$$

with fitted constants $A,B,\alpha,\beta>0$. Three terms, each with meaning:

- $E$, the **irreducible loss**: the entropy of the data-generating process (Chapter 5, §5.3). No model can go below it. For text it includes genuine unpredictability; **for genomes, $E$ is large** because most of a genome's sequence is not predictable from context (Chapter 5: the best compressors reach roughly 1.6–1.9 bits/base).
- $A/N^\alpha$: the penalty for *finite model capacity* (approximation error, Chapter 7).
- $B/D^\beta$: the penalty for *finite data* (estimation error, Chapter 7).

*Interpretation of the exponents.* $\alpha,\beta$ are small (~0.1–0.4): each 10× in $N$ reduces the reducible loss by a factor $10^{\alpha}\approx1.3$–$2.5$. Scaling is a *steady but expensive* source of improvement, and the improvements **only apply to the reducible part of the loss.**

### 17.2.2 Compute-optimal allocation, derived

Training compute is $C=6ND$ (Chapter 6). For a fixed budget $C$, which split of parameters and tokens minimizes the loss? Minimize $L(N,D)$ subject to $6ND=C$. Setting the gradient of the Lagrangian $L+\lambda(6ND-C)$ to zero:
$$
\frac{\partial L}{\partial N}=-\alpha AN^{-\alpha-1}+6\lambda D=0,\qquad
\frac{\partial L}{\partial D}=-\beta BD^{-\beta-1}+6\lambda N=0 .
$$
Multiply the first by $N$ and the second by $D$: $\alpha AN^{-\alpha}=6\lambda ND=\beta BD^{-\beta}$. Hence **at the optimum the two reducible-loss terms are in fixed ratio**, $\alpha AN^{-\alpha}=\beta BD^{-\beta}$, so $D\propto N^{\alpha/\beta}$. Combined with $C=6ND$:
$$
N_\text{opt}=G\Big(\frac{C}{6}\Big)^{\beta/(\alpha+\beta)},\qquad
D_\text{opt}=G^{-1}\Big(\frac{C}{6}\Big)^{\alpha/(\alpha+\beta)},\qquad
G=\Big(\frac{\alpha A}{\beta B}\Big)^{1/(\alpha+\beta)} .
$$
**Both parameters and tokens should grow with compute at comparable rates** (for $\alpha\approx\beta$, as $C^{0.5}$ each). The finding of Hoffmann et al. (2022) that contemporary large models were *under-trained* (too many parameters for their token count) follows from this. The code evaluates the closed form against numerical minimization using constants of the form fitted for text ($E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$; **illustrative only**, since subsequent replication attempts have questioned the original fit): the closed form and numerical optimum agree (e.g., $N=1.82\times10^9$ and $D=9.1\times10^{10}$ at $C=10^{21}$), with exponents $N\propto C^{0.45}$, $D\propto C^{0.55}$ and a **tokens-per-parameter ratio of 50–78** for these constants. The widely quoted "20 tokens per parameter" is a *fitted* quantity that depends on the constants and drifts with scale; for biological data the constants are unknown and must be measured (Chapter 47).

### 17.2.3 Extrapolating scaling curves: why the irreducible term matters

Scaling curves are used to *decide* whether to spend a large budget, so extrapolation errors are costly. The simulation generates loss values at six model sizes ($10^6$–$3\times10^8$) from a world with $E=1.2$, $A=8$, $\alpha=0.3$ and 1% noise, and fits two ways:

- **Free-$E$ fit:** recovers $E=1.20$ and $\alpha=0.279$ (truth 0.3); extrapolations to $10^9,10^{10},10^{12}$ parameters are $1.215,1.206,1.199$ against truth $1.216,1.208,1.202$.
- **Naive fit assuming $E=0$ (a straight line in log–log):** estimates $\alpha=0.014$ and predicts losses of $1.194,1.155,1.082$ at the same sizes, **dramatically over-optimistic** at large scale (1.082 vs. truth 1.202), because the loss is *flattening onto the entropy floor* and the naive line extrapolates a slope that no longer exists.
- A bootstrap of the free-$E$ fit at $N=10^{10}$ gives a 90% interval $[1.147,1.234]$ around the truth 1.208: **even the correct functional form leaves ±0.04 nats of uncertainty** after extrapolating 1.5 decades beyond the data.

*For biological data this is acute.* Because $E$ is large and unknown (most of a genome is hard to predict), a pretraining loss curve that "keeps going down" may be approaching a floor, and *the biologically relevant part of the loss may be a small, separate component whose own scaling is unmeasured* (Chapter 3, Worked Example 3.1). **Fit scaling laws per annotation class or per downstream metric, not only to the aggregate loss.**

```python
--8<-- "code/ch17_scaling.py"
```

Output:

```text
free-E fit : E = 1.20, alpha = 0.279 (truth E = 1.2, alpha = 0.3);  naive E=0 fit: alpha = 0.014
  extrapolate to N = 1e+09: truth 1.216   free-E fit 1.215   naive 1.194
  extrapolate to N = 1e+10: truth 1.208   free-E fit 1.206   naive 1.155
  extrapolate to N = 1e+12: truth 1.202   free-E fit 1.199   naive 1.082
  bootstrap 90% interval of the free-E prediction at N = 1e10: [1.147, 1.234] (truth 1.208)

compute-optimal allocation (numerical minimisation vs closed form N = G (C/6)^(beta/(alpha+beta))):
  C = 1e+21: N_opt = 1.82e+09 (closed form 1.82e+09), D_opt = 9.14e+10, tokens per parameter =  50.1
  C = 1e+22: N_opt = 5.16e+09 (closed form 5.16e+09), D_opt = 3.23e+11, tokens per parameter =  62.6
  C = 1e+23: N_opt = 1.46e+10 (closed form 1.46e+10), D_opt = 1.14e+12, tokens per parameter =  78.2
  exponents: N ~ C^0.45, D ~ C^0.55

LoRA r=8: function unchanged at init (B=0): True; rank of the update = 8; trainable parameters 8,192 vs full 262,144 (3.1%)
beta = 0.2: max |numerical optimum - pi0 exp(r/beta)/Z| = 3.4e-05; expected reward = 1.936 (prior 0.103); KL from prior = 1.800
beta = 1.0: max |numerical optimum - pi0 exp(r/beta)/Z| = 1.6e-06; expected reward = 1.148 (prior 0.103); KL from prior = 0.561
beta = 5.0: max |numerical optimum - pi0 exp(r/beta)/Z| = 2.4e-07; expected reward = 0.260 (prior 0.103); KL from prior = 0.016
```

### 17.2.4 Scaling in biology: what the evidence says

- **Protein language models.** Scaling ESM-2 from 8M to 15B parameters (Lin et al., 2023) improved perplexity and, in turn, the quality of structure predictions from sequence alone (ESMFold), with accuracy rising smoothly (roughly log-linearly) with scale in the ranges tested. [[S]] Whether *variant-effect prediction* improves monotonically with scale is less clear: the ProGen2 authors (Nijkamp et al., 2022) and the ProteinGym benchmark (Notin et al., 2023) observed that **larger models are not uniformly better at zero-shot fitness prediction**, and that performance can plateau or fall at the largest scales for some assays, a point we take up in Chapter 34. [[S]]
- **Genomic language models.** The Evo and Evo 2 papers report scaling analyses on DNA (Evo: hundreds of models up to 7B parameters comparing architectures; Evo 2: 7B and 40B models with perplexity improving with scale and context length) and find that the StripedHyena architecture scales better than a standard transformer at long context. [[S]] The downstream benefit of lower perplexity for *human regulatory prediction* is less settled (benchmarks in Chapter 32 find small or no advantages over supervised baselines on some tasks).
- **Single-cell models.** Scaling the number of cells and parameters has not yet shown clear, replicated improvements in zero-shot embedding quality or perturbation prediction over strong simple baselines (Chapters 38, 39). [[S]]

**Emergence.** Capabilities that appear "suddenly" with scale (Wei et al., 2022) are partly artifacts of discontinuous metrics: accuracy metrics (0/1) can jump while the underlying log-likelihood improves smoothly (Schaeffer et al., 2023). In biology the corresponding statement: *contact prediction from attention (Chapter 12) appears as scale grows* but is a thresholded view of smooth improvements in the pairwise statistics the model captures. Always ask whether the metric is smooth (log-likelihood, calibrated scores) or thresholded. [[P]]

### 17.2.5 Data scaling in biology: tokens are not independent samples

In text, more tokens are mostly new information. In biology, **data volume and data diversity decouple**:

- *Sequences are related by descent*: a database of $10^9$ protein sequences contains perhaps $10^7$ clusters at 30% identity and far fewer *independent evolutionary experiments* (Chapter 21). Duplicating near-identical sequences adds tokens but little information.
- *Repetition*: with a finite genome, additional epochs add little: in data-constrained settings of language modeling, repeating data for about four epochs is nearly as good as new data, with rapidly diminishing returns beyond (Muennighoff et al., 2023). [[S]] For genomes, sequencing more *species and individuals* adds more than repeating the human reference.
- *Sampling matters*: taxonomic and database biases (model organisms and pathogens are over-sequenced) shape what the model knows.

These facts suggest replacing "$D$ = tokens" by an *effective number of independent sequences or lineages* in biological scaling laws (Chapters 21, 47). [[H]]

---

## 17.3 Adapting a pretrained model

| Method | What is trained | Parameters trained | When it works best | Main risk |
|---|---|---|---|---|
| **Zero-shot** | Nothing | 0 | The pretraining objective *is* the target (e.g., likelihood = fitness) | Objective gap |
| **Linear probe** | A linear head on frozen features | $d\times K$ | Features already linearly encode the target; few labels | Underestimates fine-tuning gains; hides features not linearly accessible |
| **Full fine-tuning** | All weights | $N$ | Plenty of labeled data; target distribution near pretraining | Overfitting; forgetting; **feature distortion OOD** |
| **Parameter-efficient (LoRA, adapters)** | Small added modules | $\sim0.1$–$3\%$ of $N$ | Limited data and compute; many tasks sharing one base | Rank bottleneck; hyperparameter sensitivity |
| **In-context / prompting** | Nothing (conditioning in the input) | 0 | Models trained with prompt-like conditioning (retrieved homologs, tags) | Context length; prompt sensitivity |
| **Retrieval / MSA conditioning** | Nothing or a small retriever | small | Homologous information is informative (family-conditioned protein LMs, e.g., PoET) | Retrieval bias; leakage of test homologs |

### 17.3.1 LoRA: derivation and properties

**Low-rank adaptation** (Hu et al., 2022) freezes a pretrained weight $\mathbf{W}\in\R^{d_\text{out}\times d_\text{in}}$ and learns an additive update constrained to low rank:
$$
\mathbf{W}'=\mathbf{W}+\frac{\alpha}{r}\,\mathbf{B}\mathbf{A},\qquad\mathbf{B}\in\R^{d_\text{out}\times r},\ \mathbf{A}\in\R^{r\times d_\text{in}},\ r\ll\min(d_\text{in},d_\text{out}).
$$
$\mathbf{B}$ is initialized to zero (so the adapted model equals the pretrained one at initialization) and $\mathbf{A}$ randomly. The update has **rank at most $r$** and requires $r(d_\text{in}+d_\text{out})$ parameters instead of $d_\text{in}d_\text{out}$: for $d=512$, $r=8$ that is $8{,}192$ versus $262{,}144$ (3.1%). The code verifies the three properties: identical function at initialization, update rank exactly 8, and the parameter count. The method rests on the empirical observation that *task-specific weight changes have low intrinsic rank* (Aghajanyan et al., 2021). [[S]] In biology LoRA is the standard way to adapt large protein and DNA models to small assay datasets, and it permits storing many task-specific adapters per base model.

### 17.3.2 Fine-tuning can destroy what pretraining gave you

Kumar et al. (2022) showed that **full fine-tuning can distort pretrained features**, improving in-distribution accuracy while *hurting out-of-distribution* accuracy, because gradient updates to the head's random initialization propagate large changes into the feature extractor. Their remedy, **linear probing then fine-tuning (LP-FT)**, initializes the head by a linear probe before unfreezing. This matters in biology because *the whole point of a foundation model is out-of-distribution transfer* (a new cell line, a new species, a new assay): fine-tuning on one cell type may erode performance on others. [[S]] Evaluate adapted models on **held-out contexts**, not only on held-out examples from the fine-tuning distribution.

---

## 17.4 Alignment: steering a generative model with a reward

Pretrained generative models reproduce the training distribution; design wants *improved* samples. **Reinforcement learning from feedback** modifies a pretrained policy $\pi_0$ to increase an expected reward $r(x)$ (a learned preference model, a structure predictor's confidence, or a wet-lab assay) while staying close to $\pi_0$ so that samples remain valid:

$$
\max_\pi\ \E_{x\sim\pi}\big[r(x)\big]-\beta\,\KL{\pi}{\pi_0}.
$$

### 17.4.1 The optimum

!!! math "Derivation: the KL-regularized optimum is a Boltzmann reweighting of the prior"
    Define $\pi^\star(x)=\pi_0(x)\,e^{r(x)/\beta}/Z$ with $Z=\sum_x\pi_0(x)e^{r(x)/\beta}$. For any distribution $\pi$,
    $$
    \E_\pi[r]-\beta\KL{\pi}{\pi_0}=\E_\pi\Big[r-\beta\log\frac{\pi}{\pi_0}\Big]=\E_\pi\Big[\beta\log\frac{\pi_0e^{r/\beta}}{\pi}\Big]=\beta\log Z-\beta\KL{\pi}{\pi^\star}.
    $$
    Since $\KL{\pi}{\pi^\star}\ge0$ with equality iff $\pi=\pi^\star$ (Chapter 5), the objective is maximized by $\pi^\star$, and the maximum value is $\beta\log Z$. $\square$

The code confirms this on a six-outcome distribution: the numerically optimized policy matches $\pi_0e^{r/\beta}/Z$ to within $3\times10^{-5}$ for $\beta\in\{0.2,1,5\}$. As $\beta$ falls the policy moves farther from the prior (KL $=1.800$ at $\beta=0.2$ vs. $0.016$ at $\beta=5$) and its expected reward rises (1.936 vs. 0.260; prior 0.103). **$\beta$ is a temperature that trades reward against fidelity to the prior.**

!!! rhyme "Structural rhyme: KL-regularized RL ↔ natural selection ↔ Boltzmann sampling"
    The optimum $\pi^\star\propto\pi_0\,e^{r/\beta}$ has *exactly* the form of the mutation–selection equilibrium of Chapter 5 (§5.6.4): $\pi(x)\propto\pi_\text{mut}(x)\,f(x)^{2(N-1)}$, with the prior playing the mutation distribution, the exponentiated reward the fitness, and $\beta^{-1}\leftrightarrow2(N-1)$ the strength of selection (effective population size). It is also a Boltzmann distribution at temperature $\beta$ over energy $-r$. **Aligning a generative model with a reward is in-silico directed evolution**: generate from the prior, select by reward, with a strength set by $\beta$. This is why failure modes of directed evolution (e.g., exploiting an imperfect assay) reappear as reward hacking.

### 17.4.2 Practical algorithms and biological use

- **RLHF-style** pipelines fit a reward model from preference pairs (Bradley–Terry: $P(x_1\succ x_2)=\sigma(r(x_1)-r(x_2))$) and optimize the policy with PPO under the KL penalty.
- **Direct preference optimization (DPO)** (Rafailov et al., 2023) uses the optimum above to eliminate the explicit reward model: $\beta\log\pi_\theta(x)/\pi_0(x)$ *is* the implied reward, so preferences can be fit directly with a classification-like loss.
- **In biology:** aligning protein generative models to experimental measurements by DPO (e.g., on stability or activity data), steering structure-conditioned generators toward designs that an independent predictor scores as foldable, and *lab-in-the-loop* cycles where assays supply the reward (Chapter 46). [[S]] for the existence of such studies; [[H]] for how far they scale.

**Reward hacking and Goodhart.** The reward is a proxy. Optimizing hard against a *learned* reward (a structure predictor's confidence; a surrogate for binding) finds inputs where the proxy is wrong (Chapter 14, §14.7; Chapter 4, winner's curse). The KL term limits but does not eliminate this; independent evaluation and experimental ground truth do.

---

## 17.5 The objective gap, formalized

Chapter 1 asserted that "pretraining objective ≠ target capability". We can now say what *is* guaranteed.

Let $x$ be the input, $z$ the pretraining target (the masked token, the next token, the paired view), and $y$ any downstream variable. Define the **pretraining sufficient statistic**
$$
s_\text{pre}(x)\ :=\ \big(\text{minimal statistic with }p(z\mid x)=p(z\mid s_\text{pre}(x))\big),
$$
the information in $x$ that bears on the pretraining target. A model that minimizes the pretraining loss to its optimum must output predictions that are functions of $s_\text{pre}(x)$, and so its representation $h$ must retain $s_\text{pre}(x)$ (otherwise it could not produce $p(z\mid x)$). Therefore, by the data-processing inequality in one direction and the retention requirement in the other:

> **Sufficiency guarantee.** For any downstream $y$, a perfectly pretrained representation contains at least $\MI\big(s_\text{pre}(x);y\big)$ nats about $y$, and **everything beyond that is not guaranteed**: it is present only by the capacity and inductive bias of the network (accidental retention).

Conversely, the representation can contain *at most* $\MI(x;y)$ (Chapter 5). So
$$
\MI\big(s_\text{pre}(x);y\big)\ \le\ \MI(h;y)\ \le\ \MI(x;y).
$$
The **objective gap** for a task $y$ is the *width of this interval* in the worst case: $\MI(x;y)-\MI(s_\text{pre}(x);y)$. It is *zero* when the pretraining target already determines $y$ given $x$ (e.g., if $y$ is a function of the model's own predictive distribution: zero-shot scoring) and *large* when $y$ depends on variables or structure that the pretraining target neither uses nor depends on.

**A taxonomy of objective gaps for biological foundation models**, each a way $y$ can fall outside $s_\text{pre}(x)$:

| Gap type | Definition | Example | First encountered |
|---|---|---|---|
| **Conditioning gap** | $y$ depends on a variable absent from $x$ | Perturbation effect requires the perturbation identity and cell state; a sequence-only model sees neither | Ch. 1 (Ex. 1.2), 39 |
| **Distribution gap** | The pretraining distribution differs from the target's | Evolution-selected sequences vs. disease variants or engineered edits | Ch. 5, 32, 41 |
| **Frequency gap** | The loss is dominated by easy, frequent patterns; $s_\text{pre}$ over-represents them | Repeats and composition in DNA LMs; library size in expression autoencoders | Ch. 3, 13 |
| **Causal gap** | Pretraining is observational; the target is interventional | Predicting knockouts from unperturbed atlases | Ch. 1, 39, 44 |
| **Scale/timescale gap** | Selection acts over evolutionary time and organismal fitness; the assay measures a lab proxy | Likelihood vs. an enzyme's activity under assay conditions | Ch. 5, 34 |
| **Noise gap** | The pretraining target includes irreducible noise that $s_\text{pre}$ must encode | Count noise in expression reconstruction | Ch. 4, 13 |

!!! lens "Research lens: using the guarantee to forecast transfer"
    **Prediction rule.** *Estimate $\MI(s_\text{pre}(x);y)$ by asking: how well does the model's own predictive distribution (its zero-shot output) predict $y$?* A pretrained masked-language model's *per-position predictive distributions over residues* are $s_\text{pre}$; the correlation of $\Delta\log p$ with a measured fitness is a direct estimate of the guaranteed information. **Beyond-guarantee information** is revealed by *probes and fine-tuning* that exceed zero-shot performance: the gain from fine-tuning is an estimate of accidentally retained information, and *depends on architecture and optimization*, so it transfers less reliably across tasks and scales.

---

## 17.6 A taxonomy of biological foundation models by what they guarantee

This table previews Part VII, organizing models by pretraining target $z$ and what the sufficiency guarantee implies.

| Model class | Pretraining target $z$ | Guaranteed information $s_\text{pre}(x)$ | Not guaranteed | Chapter |
|---|---|---|---|---|
| Genomic LMs (Evo, Evo 2, Nucleotide Transformer, HyenaDNA) | Next/masked nucleotide | Conditional distribution of bases given context: composition, repeats, conservation, coding structure | Cell-type-specific regulation; causal effects; enhancer activity | 32 |
| Supervised sequence-to-function (Enformer, Borzoi, AlphaGenome) | Measured genomic tracks | Track-relevant sequence features per cell type | Variation absent from training (personal genomes), unmeasured modalities | 31 |
| Protein LMs (ESM, ProGen) | Masked/next residue | Conditional residue distributions: coevolution, family structure | Properties unrelated to evolutionary constraint; orphan proteins | 34 |
| Structure predictors (AlphaFold) | Experimental coordinates (supervised) + MSA masking | Sequence→structure map for natural-like proteins | Dynamics, effects of single mutations, conformational states | 35 |
| Single-cell FMs (Geneformer, scGPT, UCE, State) | Masked/ranked expression (some: perturbation prediction) | Gene–gene co-expression structure across cells | Causal gene regulation; perturbation responses (unless trained on perturbations) | 38, 39 |
| Multimodal (CLIP-like) | Cross-modality matching | Information shared between modalities | Modality-specific information | 40 |

---

## 17.7 Worked research examples

!!! example "Worked Research Example 17.1: Scaling a genomic language model by 100× lowers perplexity by 2%. Is the compute justified?"
    **Situation.** A group trains DNA language models at $10^8$, $10^9$, $10^{10}$ parameters on the same data. Held-out perplexity improves from 3.80 to 3.72 to 3.66 (about 2% in total). The next step would use 100× the compute. The aim is better variant-effect prediction at regulatory sites.

    **Question.** Should they scale?

    **Reasoning.**

    1. *What does the loss measure?* A frequency-weighted average over all positions. The share of positions that are regulatory is small (a few percent), and the average is dominated by repeats and composition.
    2. *What does the scaling curve say?* Fit $L=E+AN^{-\alpha}$ with a free $E$ (§17.2.3). With three points, the fit is underdetermined; the bootstrap interval after extrapolation is wide. Use more model sizes, and *fit the curve to the loss on regulatory annotation classes separately*.
    3. *Does the downstream metric track the loss?* The guarantee says only the *pretraining-relevant* information is guaranteed. Test: does zero-shot $\Delta\log p$ correlation with MPRA/eQTL effects improve with $N$? Plot *downstream metric vs. $N$* for $N=10^8,10^9,10^{10}$. If flat, additional scale is unlikely to help *this* task unless the improvement trend is rising.
    4. *Alternative levers.* (A3) change the objective: add supervised functional-genomics targets or reweight regulatory regions; (A4) add data: more diverse species or functional assays; (A2) add conditioning (cell type). Each can be tested at *small scale* (e.g., $10^8$ parameters) at 1% of the cost.
    5. *Prediction.* If the gap is an objective/conditioning gap, a small model with the right objective or data beats the 100×-larger generic model on the target task; if it is a capacity gap, the downstream metric climbs with $N$.
    6. *What would change your mind?* A downstream metric that rises along a clean log-linear trend with slope that predicts a useful value at the target scale, *and* a demonstration that small-scale objective changes do not capture the same gain.

    **Expert analysis.** The decision turns on *which term of the scaling law the task depends on*. Scaling reduces the reducible pretraining loss; whether that loss component is the one the task needs is an empirical question that cheap experiments answer. The default recommendation: **run the small-scale controls on objective, data, and conditioning before buying scale.**

!!! example "Worked Research Example 17.2: The largest protein language model ranks first on contact prediction and last on clinical variant-effect prediction"
    **Situation.** A 15B-parameter protein LM achieves the best long-range contact precision (via attention probes) but is outperformed by smaller models and by a family-specific generative model on a clinical variant benchmark.

    **Reasoning.**

    1. *What does each task need from $s_\text{pre}$?* Contact prediction needs pairwise co-variation structure of well-populated families: squarely inside the guaranteed information (§13.3.2, §12.8). Clinical variant effect needs the *relative* likelihood of single substitutions *in a specific protein*, whose calibration depends on local family depth, phylogenetic composition, and the match between evolutionary constraint and human disease.
    2. *Why might a larger model be worse?* (H1) *Over-fitting to the sequence database distribution* (popularity, phylogenetic composition) that does not reflect functional constraint: larger models memorize more of the dataset's biases (Chapter 34). (H2) *Calibration of $\Delta\log p$ across proteins*: larger models have sharper likelihoods; absolute scores are not comparable across proteins even when within-protein ranking is fine. (H3) *Benchmark properties*: many clinical variants lie in conserved, well-covered regions where even simple conservation scores perform well, so there is little headroom for larger models (ceiling and baselines; Chapter 43). (H4) *Training-data overlap*: test proteins and homologs in the pretraining set.
    3. *Experiments.* Report per-protein within-protein ranking (Spearman) separately from cross-protein pooled metrics; compare to a conservation baseline and an independent MSA-based model (EVE-like); stratify by MSA depth and by protein family size; test sensitivity to the *training-data composition* (retrain a small model with tax-balanced data).
    4. *Predictions.* Under H1, down-weighting over-represented clades improves variant scoring; under H2, within-protein ranking is competitive while pooled metrics are poor; under H3, a conservation baseline is within noise of the top model.

    **Expert analysis.** Scaling improves the *pretraining objective*; the clinical task relies on a *derived quantity* ($\Delta\log p$ calibrated across proteins) whose reliability depends on factors other than model size. The sufficiency guarantee tells us to *expect* non-monotonic transfer when the downstream quantity is not a direct function of $s_\text{pre}$.

---

## 17.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: a pre-registered checklist before scaling a biological foundation model"
    1. **Name the target capability** in one sentence and the **rung of the Claim Ladder** (Chapter 1) it requires.
    2. **State the sufficiency argument**: why is the capability inside the information guaranteed by the pretraining target? If it is not, name the gap type (§17.5).
    3. **Measure the zero-shot guarantee** at current scale (the model's own predictive distribution against the task).
    4. **Plot the downstream metric against scale** over at least three model sizes; add a **small-scale control** with a modified objective, conditioning, or data.
    5. **Fit a scaling law with a free irreducible term**; report extrapolation intervals (§17.2.3); fit per annotation class where applicable.
    6. **Audit the effective data**: independent lineages, deduplicated clusters, taxonomic balance (§17.2.5).
    7. **Decide in advance** which outcomes would lead you to scale, to change the objective, or to collect new data. *Write down what would change your mind.*

    **What this teaches.** Scaling is a *hypothesis* about which term limits performance. The cheapest experiments (changing objective, conditioning, or data at small scale) test whether the hypothesis is plausible before the expensive one is run. It transforms "should we scale?" from a matter of faith into an experiment.

---

## 17.9 Connections

- **Backward:** the $6ND$ cost (Chapters 6, 9); the entropy floor $E$ (Chapter 5); bias–variance (Chapter 7); self-supervised objectives (Chapter 13); generative models (Chapters 14–15); Chapter 1's objective gap; Sella–Hirsh selection (Chapter 5).
- **Forward:** genomic, protein, structure, single-cell, and multimodal foundation models (Chapters 31–42); scaling economics specific to biology (Chapter 47); benchmark design and baselines (Chapter 43); lab-in-the-loop alignment (Chapter 46); AI scientists as another form of foundation-model use (Chapter 54); the scale and objective attacks (Chapter 55).

!!! takeaways "Key takeaways"
    1. A **foundation model** is broad, self-supervised, adaptable. In biology test it by **zero-shot, probe, fine-tune, and generate**, and *always against a simple baseline*.
    2. **Scaling law:** $L=E+AN^{-\alpha}+BD^{-\beta}$. $E$ is the data's entropy; scaling only reduces the *reducible* part. Compute-optimal allocation has $\alpha AN^{-\alpha}=\beta BD^{-\beta}$ and $N,D\propto C^{\beta/(\alpha+\beta)},C^{\alpha/(\alpha+\beta)}$.
    3. **Extrapolation requires the right $E$**: assuming $E=0$ turned a 1.202 floor into a predicted 1.082; even the correct form had ±0.04 uncertainty 1.5 decades out.
    4. **Biological tokens are not independent samples**: effective data is lineages/clusters, not token counts.
    5. **Adaptation menu:** zero-shot, probe, fine-tune, LoRA ($r(d_\text{in}+d_\text{out})$ parameters; rank $\le r$), prompting, retrieval; **fine-tuning can distort features out-of-distribution** (use LP-FT; test on held-out contexts).
    6. **KL-regularized reward optimum** $\pi^\star\propto\pi_0e^{r/\beta}$: alignment is in-silico selection; $\beta$ is a temperature; reward hacking is Goodhart.
    7. **Sufficiency guarantee:** a pretrained representation provably contains $\MI(s_\text{pre}(x);y)$ about any $y$; everything else is accidental. The **objective gap** is the width of $[\MI(s_\text{pre};y),\MI(x;y)]$, with six gap types.
    8. **Before scaling**, run small-scale controls on objective, data, and conditioning.

---

## Further reading

- Bommasani, R. et al. (2021). On the opportunities and risks of foundation models. arXiv:2108.07258.
- Kaplan, J. et al. (2020). Scaling laws for neural language models. arXiv:2001.08361. Hoffmann, J. et al. (2022). Training compute-optimal large language models. arXiv:2203.15556. Besiroglu, T., Erdil, E., Barnett, M. & You, J. (2024). Chinchilla scaling: a replication attempt. arXiv:2404.10102. Muennighoff, N. et al. (2023). Scaling data-constrained language models. *NeurIPS*.
- Wei, J. et al. (2022). Emergent abilities of large language models. *TMLR*. Schaeffer, R., Miranda, B. & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *NeurIPS*.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130. Nijkamp, E., Ruffolo, J., Weinstein, E. N., Naik, N. & Madani, A. (2022). ProGen2: exploring the boundaries of protein language models. arXiv:2206.13517. Notin, P. et al. (2023). ProteinGym: large-scale benchmarks for protein fitness prediction and design. *NeurIPS Datasets & Benchmarks*.
- Nguyen, E. et al. (2024). Sequence modeling and design from molecular to genome scale with Evo. *Science* 386, eado9336. Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361.
- Hu, E. J. et al. (2022). LoRA: low-rank adaptation of large language models. *ICLR*. Aghajanyan, A., Gupta, S. & Zettlemoyer, L. (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *ACL*. Kumar, A., Raghunathan, A., Jones, R., Ma, T. & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. *ICLR*.
- Truong, T. & Bepler, T. (2023). PoET: a generative model of protein families as sequences-of-sequences. *NeurIPS*.
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*. Rafailov, R. et al. (2023). Direct preference optimization: your language model is secretly a reward model. *NeurIPS*. Widatalla, T., Rafailov, R. & Hie, B. (2024). Aligning protein generative models with experimental fitness via direct preference optimization. *bioRxiv*.
- Gao, L., Schulman, J. & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML*.
