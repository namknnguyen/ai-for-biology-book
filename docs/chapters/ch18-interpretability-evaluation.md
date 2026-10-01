# Chapter 18. Interpretability and Evaluation of Deep Models

!!! abstract "Chapter at a glance"
    **Motivation.** A deep model is a hypothesis machine: it produces predictions, but biologists want *reasons*. Attribution, probing, and sparse-feature methods convert network internals into candidate biological statements; calibration and uncertainty methods say when predictions can be trusted. The recurring lesson is that every such tool has *failure modes that can be tested with ground truth*, and a method that has not been tested on a problem with a known answer should be treated as a hypothesis generator, not as evidence.
    **Prerequisites.** Chapters 3, 4, 9, 10, 12, 13.
    **You will be able to:** (1) compute and compare attribution methods (ISM, gradient × input, integrated gradients, DeepLIFT, Shapley values) and derive the completeness property; (2) explain the one-hot gradient artifact for DNA and its correction; (3) test attribution methods against a planted ground truth and identify a redundancy failure; (4) compare representations with CKA and understand its invariances; (5) explain superposition and sparse autoencoders and their honest limits; (6) quantify uncertainty with calibration, ensembles, and conformal prediction, and know when coverage guarantees break.

---

## 18.1 Three different questions

"Interpretability" bundles questions with different logical status:

| Question | Example | What kind of evidence answers it |
|---|---|---|
| **What did the model compute for this input?** (local explanation) | Which nucleotides drove this prediction? | Attribution methods, validated against the model's own behavior (faithfulness) |
| **What does the model represent?** (global) | Does the embedding encode GC content, species, a TF motif? | Probes, representation comparison, sparse features |
| **Is the model right, and how sure is it?** (evaluation) | Is the prediction calibrated? Where does it fail? | Held-out evaluation, uncertainty quantification |

None of these answers the *biological* question "what does the cell do?" by itself. In the Claim Ladder (Chapter 1), attribution and probing reach rung C3 *within the model* only when the model-level claim is confirmed by interventions on the model, and rung C3–C4 *about biology* only with experimental confirmation (Chapters 44, 46, 48).

---

## 18.2 Attribution methods

Let $f(\mathbf{x})$ be a scalar model output (a logit, a predicted expression level). An **attribution** assigns each input feature $i$ a score $a_i$ meant to reflect its contribution to $f(\mathbf{x})$.

### 18.2.1 In silico mutagenesis (ISM)

Replace the base at position $i$ by each alternative and record the change: $a_i=\max_{b\ne x_i}|f(\mathbf{x}^{i\to b})-f(\mathbf{x})|$ (or the signed values for all three alternatives). It is **model-agnostic and directly interpretable** as a prediction of the effect of a point mutation (the basis of DeepSEA's variant scores; Chapter 10) but costs $3L$ forward passes per sequence and considers **only single mutations**.

### 18.2.2 Gradient-based methods

**Saliency** is $\partial f/\partial x_i$; **gradient × input** multiplies by the input value. Gradients are cheap (one backward pass) but are *local*: they describe sensitivity at $\mathbf{x}$, not contribution relative to a baseline, and they vanish where the network saturates.

**Integrated gradients** (Sundararajan et al., 2017) integrate the gradient along a straight path from a *baseline* $\mathbf{x}'$ (an uninformative reference: zeros, a uniform base distribution, a shuffled sequence) to the input:

$$
\mathrm{IG}_i(\mathbf{x})=(x_i-x'_i)\int_0^1\frac{\partial f\big(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}')\big)}{\partial x_i}\,d\alpha .
$$

**Completeness** (the key axiom): $\sum_i\mathrm{IG}_i=f(\mathbf{x})-f(\mathbf{x}')$. *Proof.* Let $F(\alpha)=f(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}'))$. By the fundamental theorem of calculus and the chain rule, $f(\mathbf{x})-f(\mathbf{x}')=\int_0^1F'(\alpha)d\alpha=\int_0^1\sum_i(x_i-x_i')\,\partial_if(\cdot)\,d\alpha=\sum_i\mathrm{IG}_i$. $\square$ The attributions *add up* to the difference between the prediction and the baseline prediction, which gradient × input does not guarantee. In practice the integral is approximated by a Riemann sum with 20–300 steps. **The result depends on the baseline**; a poor baseline produces meaningless attributions, so the baseline is part of the claim.

**DeepLIFT** (Shrikumar et al., 2017) propagates *differences from a reference activation* backward through the network using "multipliers" that avoid the saturation problem of gradients (its rescale rule satisfies completeness); it is widely used in genomics (as in BPNet; Chapter 10) with dinucleotide-shuffled references.

### 18.2.3 Shapley values

Treat features as players and the prediction as the "payout." The **Shapley value** of feature $i$ is the average marginal contribution over all orderings:

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\Big[v(S\cup\{i\})-v(S)\Big],
$$

where $v(S)$ is the model's value when only features $S$ are "present." It is the *unique* attribution satisfying efficiency ($\sum\phi_i=v(N)-v(\emptyset)$), symmetry, dummy, and additivity axioms (Shapley, 1953; Lundberg & Lee, 2017). It costs $2^n$ evaluations and requires choosing *how absent features are filled in* (marginal vs. conditional distributions), a choice that changes the answer; approximations (KernelSHAP, DeepSHAP) are used. For sequences, "removing" a nucleotide is not well-defined without a background distribution.

### 18.2.4 A genomics-specific pitfall: gradients on one-hot inputs

A one-hot DNA input lies on the vertices of a simplex ($\sum_ax_{i,a}=1$). The network is *trained only on these points* but is *defined* on all of $\R^{4\times L}$, so its gradient includes components in directions along which the training data never varied, in particular the direction $(1,1,1,1)$ at each position (which changes the sum of the channels). Two networks that agree on all one-hot inputs can differ arbitrarily in such off-simplex directions, yet **raw gradients reflect them**. Majdandžić et al. (2023) showed that this adds noise to gradient attributions of genomic networks and proposed a simple fix: **mean-center the gradient across the four nucleotides at each position**, removing the arbitrary common component. In the code, two models give identical outputs on one-hot inputs, but their raw gradients differ by $0.275$; after mean-centering, they differ by $3\times10^{-8}$. [[E]] (as a mathematical statement; [[S]] for its empirical impact on motif discovery)

### 18.2.5 Testing attribution methods with ground truth

Whether an attribution method is *trustworthy* is an empirical question that can be answered only when we know what the model does. We build a "model" whose logic is known: $f(\mathbf{x})=\sigma(\max_ts_t(\mathbf{x})-5)$, where $s_t$ is the match score of the motif `TAGCTG` at position $t$ (+1 per matching base, −1 per mismatch). The planted motif positions are the ground truth. Metric: the **fraction of total absolute attribution on the true motif positions** (a random attribution would give $6/40=0.15$ for one copy).

| Condition | ISM | Gradient × input (corrected) | Integrated gradients |
|---|---|---|---|
| One motif copy | **1.00** | **1.00** | **1.00** |
| **Two redundant copies** | **0.00** | 1.00 | 1.00 |

With one copy all three recover the motif. With **two redundant copies** (a "shadow enhancer" scenario), ISM assigns *exactly zero* importance to every base, because mutating one copy leaves the other intact, so the output does not change: **single-mutation scanning is blind to redundancy** (and, more generally, to any epistasis). Gradient × input and IG still attribute to the motif because they look at the local or path-integrated sensitivity, which includes both copies (here, the max-pool gradient is shared between tied copies). *This is a specific failure of one method, demonstrated by ground truth; it does not show that the gradient-based methods are generally reliable.*

**Other known failure modes.** (i) *Sanity-check failures*: some saliency methods (e.g., guided backpropagation) produce nearly the same maps for a trained and a randomly initialized network, i.e., they reflect input structure, not learned computation (Adebayo et al., 2018). (ii) *Baseline dependence*: IG with a zero baseline in one-hot encodings is not a valid "absence of information." (iii) *Saturation*: gradient × input can be near zero when the network is saturated even for important features. (iv) *Input shift sensitivity*: attributions can change under constant shifts of the input that do not change the prediction (Kindermans et al., 2019). (v) *Faithfulness is different from plausibility*: a map that *looks* like a motif is not thereby faithful to the network. **Evaluate attributions by intervening**: delete or insert the highlighted features and check that predictions change as the attribution says (deletion/insertion curves; ROAR, Hooker et al., 2019), and by *planted-ground-truth* tests like the above.

```python
--8<-- "code/ch18_interpretability.py"
```

Output:

```text
fraction of total |attribution| falling on the true motif positions (random-attribution baseline = 6/40 = 0.15 for one copy, 0.30 for two):
  1 motif copy(ies) (model output ~ 0.731): ISM (single mutations): 1.00; gradient x input (corrected): 1.00; integrated gradients: 1.00
  2 motif copy(ies) (model output ~ 0.731): ISM (single mutations): 0.00; gradient x input (corrected): 1.00; integrated gradients: 1.00

one-hot gradient artefact: same predictions (0e+00 apart) but raw gradients differ by 0.275; after mean-centring over the 4 channels they differ by 3.0e-08

CKA(X, XQ) with Q orthogonal = 1.000; CKA(X, 3 X) = 1.000; CKA(X, X M) with an invertible M = 0.691; CKA(X, independent) = 0.038

superposition and sparse autoencoders:
  12 features in 8 dims (1.5x compression), SAE width 12: recovered  9/12 true directions (cosine > 0.9); mean best cosine 0.93; active units/sample 0.9; recon MSE 0.0027
  16 features in 8 dims (2.0x compression), SAE width 32: recovered  7/16 true directions (cosine > 0.9); mean best cosine 0.88; active units/sample 2.4; recon MSE 0.0024
  24 features in 8 dims (3.0x compression), SAE width 24: recovered  6/24 true directions (cosine > 0.9); mean best cosine 0.83; active units/sample 2.4; recon MSE 0.0026
conformal 90% interval (half-width 0.73): empirical coverage on exchangeable test data = 0.897
conformal 90% interval (half-width 0.73): empirical coverage on covariate-shifted test data (x + 2.5) = 0.616
```

---

## 18.3 Probing and comparing representations

**Probes** were introduced in Chapter 13: a simple classifier on frozen features measures *accessible* information, with *control tasks* to separate representation quality from probe capacity. Beyond probes, we often want to ask whether *two representations* (two layers, two models, two training runs) contain similar information.

**Centered kernel alignment (CKA)** (Kornblith et al., 2019) compares two sets of representations $\mathbf{X}\in\R^{n\times p}$, $\mathbf{Y}\in\R^{n\times q}$ of the same $n$ inputs. For the linear kernel (columns centered),

$$
\mathrm{CKA}(\mathbf{X},\mathbf{Y})=\frac{\|\mathbf{Y}^\top\mathbf{X}\|_F^2}{\|\mathbf{X}^\top\mathbf{X}\|_F\,\|\mathbf{Y}^\top\mathbf{Y}\|_F}\in[0,1].
$$

It is **invariant to orthogonal transformations and isotropic scaling** of either representation (verified: $\mathrm{CKA}(\mathbf{X},\mathbf{X}\mathbf{Q})=1.000$ and $\mathrm{CKA}(\mathbf{X},3\mathbf{X})=1.000$) and therefore insensitive to the *rotation non-identifiability* of latent axes (Chapter 8). It is **not** invariant to general invertible linear maps ($\mathrm{CKA}(\mathbf{X},\mathbf{X}\mathbf{M})=0.691$ for a random invertible $\mathbf{M}$), and unrelated representations give a small value (0.038). Use CKA (or CCA) to ask "do these two models encode the *same subspace*?" rather than comparing axes (Worked Example 8.2). Caveat: CKA is dominated by high-variance directions, so two representations can score high while differing in low-variance but task-relevant directions.

---

## 18.4 Superposition and sparse autoencoders

### 18.4.1 Why single neurons are not interpretable

Chapter 2 (§2.7.1) showed that a $d$-dimensional space holds many more than $d$ *nearly orthogonal* directions. If the features a network needs to represent are **sparse** (rarely active together), it can store $m\gg d$ features in $d$ dimensions as nearly orthogonal directions, tolerating small interference, a regime called **superposition** (Elhage et al., 2022). Consequence: individual neurons respond to *mixtures* of unrelated features (**polysemanticity**), and *no basis of $d$ axes can isolate the features.*

### 18.4.2 Sparse autoencoders

A **sparse autoencoder (SAE)** is trained on a model's internal activations $\mathbf{h}\in\R^d$ to find an *overcomplete* dictionary of $m>d$ directions in which each activation is a *sparse* combination:

$$
\mathbf{f}=\mathrm{ReLU}\big(\mathbf{W}_e\mathbf{h}+\mathbf{b}_e\big),\qquad\hat{\mathbf{h}}=\mathbf{W}_d\mathbf{f},\qquad
\mathcal{L}=\|\mathbf{h}-\hat{\mathbf{h}}\|^2+\lambda\|\mathbf{f}\|_1,
$$

with unit-norm decoder columns so that the L1 penalty cannot be cheated by rescaling. The hope: each decoder column corresponds to one *monosemantic feature* (Bricken et al., 2023; Cunningham et al., 2023). This is dictionary learning (sparse coding), a technique with decades of history, applied to neural activations.

**A toy test with ground truth.** We generate sparse features, each active with probability 0.02–0.03 with random magnitude, embed them into $d=8$ dimensions through random unit-norm directions, and train SAEs on the resulting activations. We ask how many *true* directions appear (cosine > 0.9) among the learned decoder columns:

| True features in 8 dims | Compression | SAE width | True directions recovered | Mean best cosine |
|---|---|---|---|---|
| 12 | 1.5× | 12 | 9 / 12 | 0.93 |
| 16 | 2.0× | 32 | 7 / 16 | 0.88 |
| 24 | 3.0× | 24 | 6 / 24 | 0.83 |

**Honest reading.** Even in this ideal toy (exactly sparse, noise-free, known dictionary), recovery is *partial* and **degrades with compression**: SAEs may split a feature across several units, merge features, or settle on a different but equally sparse basis; the loss has no incentive to find the *true* one. In real models there is no ground-truth dictionary, features are not exactly sparse, and results depend on width, sparsity penalty, and random seed. **Treat SAE features as hypotheses to be validated** by (i) their *activation patterns* on held-out data, (ii) *causal interventions* (ablating or steering the feature changes outputs as predicted), and (iii) *correspondence to independent biological annotations*.

**Biological applications** (to be treated in Chapter 48): SAEs trained on protein language model activations have yielded features aligned with annotated motifs, domains, and binding sites (e.g., InterPLM, 2024–2025; Gujral et al., 2025), and mechanistic analysis of Evo 2 reports features corresponding to exon–intron boundaries, transcription-factor binding sites, protein secondary-structure elements, and prophage regions (Brixi et al., 2026). [[S]] They show that *models contain interpretable, annotation-aligned structure*; whether that structure is used *causally* by the model and whether it reveals *new* biology are open questions.

---

## 18.5 Evaluation under uncertainty: calibration, ensembles, conformal prediction

### 18.5.1 Calibration

A probabilistic classifier is **calibrated** if, among all examples assigned probability $q$, a fraction $q$ are positive. Deep networks trained with cross-entropy tend to be *over-confident* (Guo et al., 2017). The **expected calibration error** bins predictions and averages $|\text{accuracy}-\text{confidence}|$ per bin. *Temperature scaling* (replace $\softmax(\mathbf{z})$ by $\softmax(\mathbf{z}/T)$, fit $T$ on held-out data) fixes much of the miscalibration without changing the ranking. Calibration matters wherever probabilities are *combined* with other evidence (Chapter 4's variant classification) or used to *decide* (which designs to synthesize).

### 18.5.2 Epistemic and aleatoric uncertainty; ensembles

**Aleatoric** uncertainty is irreducible noise in the label (Chapter 1's noise ceiling); **epistemic** uncertainty is the model's ignorance, reducible with data. A single softmax cannot distinguish them. **Deep ensembles** (Lakshminarayanan et al., 2017) train several networks from different random seeds; disagreement among members estimates epistemic uncertainty. **Monte-Carlo dropout** and Bayesian approximations are cheaper but less reliable. Ensembles typically improve both accuracy and calibration and are a strong baseline for *out-of-distribution detection*; they are not guaranteed to flag all shifts (members trained on the same data often fail in the *same* way).

### 18.5.3 Conformal prediction: distribution-free coverage

**Split conformal prediction** converts any point predictor into a prediction *interval (or set)* with a finite-sample coverage guarantee. Hold out a calibration set of $n$ examples, compute **nonconformity scores** $s_i=|y_i-\hat f(x_i)|$, and let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil$-th smallest score. The prediction interval for a new $x$ is $\hat f(x)\pm\hat q$.

*Guarantee.* If calibration and test examples are **exchangeable**, the test score $s_{n+1}$ has a uniformly distributed rank among the $n+1$ scores (absent ties), so $P(s_{n+1}\le\hat q)=\lceil(n+1)(1-\alpha)\rceil/(n+1)\ge1-\alpha$. $\square$ The code uses a degree-7 polynomial fit to heteroscedastic data ($\alpha=0.1$): on exchangeable test data the empirical coverage is **0.897**, as promised.

*Failure under shift.* The guarantee **requires exchangeability**. When the test inputs are shifted (here, $x\to x+2.5$, partly outside the training range), coverage drops to **0.616**. The same is true in biology whenever the deployment distribution (a new cell type, species, chemotype, perturbation) differs from the calibration data (Chapter 45). Remedies: calibrate on data that match the deployment distribution (stratified or **group-conditional** calibration), **weighted conformal prediction** when the shift is known (Tibshirani et al., 2019), and *adaptive* methods.

**Why conformal belongs in biology.** Triage decisions (which of $10^4$ candidates to test; which variants to flag) are *selective prediction*: act on the confident subset and defer the rest. Conformal sets give a *guaranteed error budget* under the stated assumptions.

---

## 18.6 The logic of ablations and controls

An **ablation** removes or alters a component and measures the change in performance. It tests *necessity*, not *sufficiency*, and is only meaningful with controls:

1. **Retrain or retune after the ablation** where appropriate: removing a component and not re-optimizing penalizes the ablated model for being mismatched.
2. **Match capacity**: replacing a component by something with fewer parameters changes more than one variable.
3. **Multiple seeds** and confidence intervals: an ablation difference smaller than seed variance is noise.
4. **Random-initialization controls**: does pretraining matter, or is the architecture enough? (Chapter 13.)
5. **Shuffled-label and shuffled-input controls** (Chapter 6): does the evaluation detect nothing when there is nothing?
6. **Simple-baseline controls**: $k$-mers, linear models, nearest neighbors, mean predictors.
7. **Noise ceiling**: what is the best achievable score given measurement noise? (Chapter 1.)

**What each tool can claim:**

| Tool | Supports claims of the form | Does *not* support |
|---|---|---|
| Attribution (validated) | "The model's prediction depends on these inputs in this way" | "The cell depends on these bases" |
| Probe (with controls) | "This information is linearly accessible in the representation" | "The model *uses* this information" |
| Ablation (matched) | "This component is necessary for this behavior in this model" | "This component implements the behavior" (alternatives may exist) |
| SAE feature + intervention | "This direction mediates this behavior in the model" | "This is *the* feature" (non-unique) |
| Calibration/conformal | "Predictions are reliable at this rate under these assumptions" | "...under shift" |
| Experimental perturbation | "The biological system responds as predicted" | Mechanism beyond what was measured |

---

## 18.7 Worked research examples

!!! example "Worked Research Example 18.1: Two attribution methods disagree about which nucleotides matter for a regulatory prediction"
    **Situation.** For a sequence–expression model, ISM highlights a single 8-bp element 2 kb upstream of the promoter, while integrated gradients highlights the same element *and* a second, similar element 5 kb downstream. The authors say the two methods disagree and report whichever "looks like a known motif."

    **Question.** Which method is right? What should the authors do?

    **Reasoning.**

    1. *Both can be right about different things.* ISM answers "what happens if I mutate *one* base here?" IG answers "how does the prediction change along the path from a baseline to this sequence, apportioned among features?" The code shows an exact scenario for disagreement: **redundant elements**. With two functional copies, ISM assigns each copy zero importance (mutating one leaves the other), whereas path-based methods credit both.
    2. *What does the disagreement tell us?* Possibly that the model has learned a **redundant or OR-like logic** between the two elements. The disagreement is *information*, not noise.
    3. *Alternative explanations.* (H1) IG's baseline (e.g., a zero or uniform background) creates an artificial path that traverses regions the model has never seen. (H2) The second element is a proxy for something else (e.g., local GC content) via the network's shortcuts. (H3) Gradient noise from the one-hot artifact (§18.2.4) if raw gradients were used.
    4. *Experiments.* (a) **Double ISM**: mutate both elements jointly and compare to the sum of single effects (epistasis/redundancy test); (b) deletion/insertion curves for each element and both; (c) IG with several baselines (dinucleotide shuffles) to test baseline robustness, and use mean-centered gradients; (d) planted-synthetic tests of the *same model class* with known logic; (e) *wet-lab:* single and double deletions of the two elements.
    5. *Predictions.* Under redundancy: single deletions have small effects, the double deletion a large one, in both model and cell. Under a model-only shortcut (H2): the model's double-mutation prediction is large but the cell's double deletion is small.

    **Expert analysis.** The authors' selection of "whichever looks like a known motif" is a *plausibility filter*, not a faithfulness test. The correct deliverable is a **table of discordances with discriminating experiments** and a statement of which claim each method supports. A discordant attribution is a *hypothesis about the model's logic* that cheap in silico interventions test before any wet-lab work (Chapter 46).

!!! example "Worked Research Example 18.2: Can model uncertainty be used to triage variants?"
    **Situation.** A variant-effect model provides a point prediction and an ensemble variance. The group proposes to report predictions only for variants with low variance ("confident") and to defer the rest, claiming 95% accuracy on the confident subset.

    **Question.** What must hold for this to work, and how should it be evaluated?

    **Reasoning.**

    1. *What is the claim?* A *selective classification* claim: error rate on the retained subset is below some target at coverage (fraction retained) $c$. It must be reported together with $c$ (a model that retains 2% of variants can be 99% accurate).
    2. *Does ensemble variance track error?* Only if the ensemble's disagreement reflects genuine epistemic uncertainty. Members trained on the same data often share biases, so they can be *confidently wrong* on shifted inputs (§18.5.2).
    3. *Distribution shift.* The conformal experiment: coverage 0.897 on exchangeable data falls to 0.616 on shifted data. Variant triage is typically deployed on *different* genes, cell types, or populations than the calibration set, so the guarantee does not hold.
    4. *Experiments.* (a) Plot the **risk–coverage curve** and compare with a simple baseline (conservation score; variance of a few $k$-mer models); (b) evaluate on **held-out genes/families/ancestries** (shifted data), not random variants; (c) check calibration within *strata* (gene, conserved vs. non-conserved, coding vs. non-coding); (d) use *group-conditional* conformal calibration and report realized coverage per group.
    5. *Predictions.* If variance is informative, risk falls monotonically as coverage decreases on held-out genes; if it merely proxies conservation, the curve matches the conservation baseline's.

    **Expert analysis.** A confidence score is itself a model that must be evaluated, with its own baselines and its own distribution shift. The promised accuracy is a statement about a *specific population*; its transport to deployment is the generalization gap (G-G) again.

---

## 18.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: from an attribution map to a testable hypothesis"
    **Starting point.** An attribution map for a model of enhancer activity highlights a 10-bp stretch resembling a known TF motif.

    **A disciplined path.**

    1. **Faithfulness.** Do deletion/insertion curves confirm that the model's output depends on this stretch as the map implies? Use several methods and baselines; apply the one-hot correction.
    2. **Specificity.** Does the response follow the *motif* (mutations that preserve high-information positions have little effect; mutations at conserved positions have large effects) or merely the *region*?
    3. **Context dependence.** Move the motif to other backgrounds and positions (motif-insertion tests): does the effect depend on spacing, orientation, and neighboring motifs as a grammar would?
    4. **Model-level alternatives.** Is the motif a proxy for composition (GC), accessibility, or conservation that the model learned elsewhere? Compare with a baseline that has access only to those features.
    5. **Form the biological hypothesis.** "TF X binds this element and activates the gene in cell type Y." State **what result in the lab would refute it** (e.g., motif-disrupting point mutations in a reporter assay do not reduce activity; TF X knockdown does not change expression).
    6. **Design the cheapest discriminating experiment**: an MPRA with matched wild-type, motif-disrupted, and spacing-varied constructs (Chapter 46).

    **Possible outcomes and meaning.** (a) Model, in silico variants, and wet-lab agree: strong evidence the motif is functional and the model has learned it (rung C3–C4). (b) In silico yes, wet lab no: the model has learned a *shortcut or proxy*; this is itself a scientific finding about the model and the training data. (c) Faithfulness fails at step 1: the attribution method, not the model, was the problem.

    **What it teaches.** Interpretability yields *hypotheses*; the work of science is in steps 5–6.

---

## 18.9 Connections

- **Backward:** gradients and the chain rule (Chapter 3, 9); invariance and rotation non-identifiability (Chapter 8); superposition and near-orthogonality (Chapter 2); probing and forgetting (Chapter 13); attention is not explanation (Chapter 12); winner's curse and selection (Chapter 4).
- **Forward:** mechanistic interpretability and experimental validation of features (Chapter 48); benchmark and baseline design (Chapter 43); causal identification behind "attribution ≠ intervention" (Chapter 44); calibration under shift (Chapter 45); closed-loop use of uncertainty in experimental design (Chapter 46); evaluation of AI-generated hypotheses (Chapter 54).

!!! takeaways "Key takeaways"
    1. Separate **local explanation, global representation, and evaluation**; none reaches a biological claim without experimental intervention.
    2. **ISM** is model-agnostic but blind to redundancy and epistasis (assigned *zero* to two redundant motif copies); **IG** satisfies *completeness* ($\sum\mathrm{IG}_i=f(\mathbf{x})-f(\mathbf{x}')$) but depends on the baseline; **Shapley** values are axiomatic but costly and background-dependent.
    3. **One-hot gradients** contain an arbitrary common component; **mean-center across nucleotides** (raw difference 0.275 → $3\times10^{-8}$).
    4. **Test attribution methods on planted ground truth** and with deletion/insertion faithfulness; plausibility is not faithfulness.
    5. **CKA** compares representations invariantly to rotations and scaling (1.000), not to general linear maps (0.691).
    6. **SAEs** find sparse dictionaries in superposition, but recovery is partial and degrades with compression (9/12, 7/16, 6/24 true features); validate by intervention.
    7. **Conformal prediction** gives finite-sample coverage under exchangeability (0.897) and **fails under shift (0.616)**; report coverage per deployment group.
    8. Use **matched ablations, multiple seeds, and baselines**; each tool supports a specific, limited claim.

---

## Further reading

- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*. Shrikumar, A., Greenside, P. & Kundaje, A. (2017). Learning important features through propagating activation differences. *ICML*. Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. Shapley, L. S. (1953). A value for n-person games.
- Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS*. Kindermans, P.-J. et al. (2019). The (un)reliability of saliency methods. In *Explainable AI*. Hooker, S., Erhan, D., Kindermans, P.-J. & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS*.
- Majdandzic, A., Rajesh, C. & Koo, P. K. (2023). Correcting gradient-based interpretations of deep neural networks for genomics. *Genome Biology* 24, 109. Koo, P. K. & Ploenzke, M. (2021). Improving representations of genomic sequence motifs in convolutional networks with exponential activations. *Nature Machine Intelligence* 3, 258–266. Shrikumar, A. et al. (2020). Technical note on transcription factor motif discovery from importance scores (TF-MoDISco). arXiv:1811.00416.
- Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML*.
- Elhage, N. et al. (2022). Toy models of superposition. *Transformer Circuits Thread*. Bricken, T. et al. (2023). Towards monosemanticity: decomposing language models with dictionary learning. Cunningham, H., Ewart, A., Riggs, L., Huben, R. & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600. Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature* 381, 607–609.
- Simon, E. & Zou, J. (2024). InterPLM: discovering interpretable features in protein language models via sparse autoencoders. *bioRxiv*. Gujral, O. et al. (2025). Sparse autoencoders uncover biologically interpretable features in protein language model representations. *PNAS*. Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349–1361.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning* 16, 494–591. Tibshirani, R. J., Barber, R. F., Candès, E. & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
