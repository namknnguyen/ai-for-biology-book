# Chapter 45. Distribution Shift, Confounding, and Generalization

!!! abstract "Chapter at a glance"
    **Motivation.** A model is trained on one set of cells, people, species, labs, or chemistries and used on another. Almost every interesting biological application is a *shift*: to a new cell type, a new ancestry, a new family, a new assay, a new disease. This chapter classifies shifts by *what changes* (the inputs, the labels' prevalence, the mechanism, or a spurious feature), states what can be corrected without target labels and what cannot, and quantifies each in controlled simulations: the damage to accuracy, calibration, and conformal coverage; the value of importance weighting and prior correction; the number of target labels needed after a mechanism shift; and the collapse of a model that has learned a shortcut. The corollary is a decision procedure: diagnose the shift *before* choosing a remedy.
    **Prerequisites.** Chapters 7, 13, 18, 24–26, 30, 43, 44.
    **You will be able to:** (1) distinguish covariate, label, concept, and shortcut shift, and novelty/support shift; (2) say which are identifiable from unlabeled target data; (3) apply importance weighting and prior correction and know their limits (effective sample size, overlap); (4) predict how calibration and conformal coverage fail under shift; (5) estimate the labels needed to adapt to a mechanism shift; (6) design diagnostics and monitoring for deployment.

---

## 45.0 Where shift appears in this book

Shift is the common thread of earlier chapters: a polygenic score loses about 83% of its $R^2$ when LD around causal variants differs (Chapter 26); a spatial deconvolution reference from another platform sets an error floor (Chapter 25); batch-specific cell types are lost by an over-flexible integration model (Chapter 30); sequence-to-function models cannot extrapolate to new cell types (Chapter 31); molecular models degrade from $r=0.85$ to $0.62$ when the test set is chemically novel (Chapter 24). This chapter gives the language to treat these as instances of one problem.

---

## 45.1 A taxonomy of shift

Write the joint distribution of inputs $x$ and labels $y$ in the source (training) domain $S$ and target (deployment) domain $T$. What differs defines the type.

| Type | What changes | What is fixed | Biological examples |
|---|---|---|---|
| **Covariate shift** | $P(x)$ | $P(y\mid x)$ | New cell line with a different expression distribution but the same regulatory rules; new patient cohort with different age distribution; new chemical series |
| **Label (prior) shift** | $P(y)$ | $P(x\mid y)$ | Disease prevalence in a screening population vs a referral population; frequency of a cell type in a new tissue sample |
| **Concept shift** | $P(y\mid x)$ | (nothing) | A variant's effect depends on ancestry or environment; a perturbation's effect depends on cell state; an enzyme's activity depends on conditions; the label definition changes |
| **Spurious (shortcut) shift** | A feature that predicted $y$ in the source does not in the target | The causal features | Batch signature, scanner, lab, depth, ancestry correlated with phenotype |
| **Support/novelty shift** | The target contains regions of $x$ with no source support | | New protein family or fold, new scaffold, new species, a cell state absent from training |

Real shifts combine these. The distinction matters because **each type has a different remedy, and the remedies for one do not work for another**: reweighting does nothing about a shortcut; prior correction does nothing about a mechanism change; no unlabeled-data method repairs a concept shift.

!!! lens "Research lens: shifts and the causal graph"
    Chapter 44's DAG tells you which shifts a prediction survives. A predictor that uses *causes* of $y$ (stable mechanisms) transports across contexts that change $P(x)$ or $P(y)$ but not the mechanism; a predictor that uses *effects* of $y$ or *correlates through a confounder* does not (Peters et al., 2016; the idea behind invariant risk minimization). **Spurious features are, by definition, non-causal associations that differ across domains.** Whether a feature is causal is a question for experiments (Chapters 44, 46), which is why generalization and causality are the same research problem.

---

## 45.2 What can be corrected without target labels

**Covariate shift** is correctable *if the supports overlap*. The target risk is a reweighted source risk,

$$
\mathbb E_T[\ell(f(x),y)]=\mathbb E_S\Big[w(x)\,\ell(f(x),y)\Big],\qquad w(x)=\frac{p_T(x)}{p_S(x)},
$$

and the weights can be estimated from unlabeled data by a **domain classifier**: if $d(x)=P(\text{target}\mid x)$, then $w(x)=\frac{n_S}{n_T}\frac{d(x)}{1-d(x)}$ (Shimodaira, 2000; Sugiyama et al., 2007). Cost: the **effective sample size** $\mathrm{ESS}=(\sum w)^2/\sum w^2$ shrinks as the shift grows, increasing variance; where the target has no source support ($p_S=0,p_T>0$), no weighting helps (novelty).

**Label shift** is correctable from *unlabeled* target data: if $P(x\mid y)$ is fixed, the target posterior is $P_T(y\mid x)\propto P_S(y\mid x)\,P_T(y)/P_S(y)$, and the target prior $P_T(y)$ is estimated by EM on the target's unlabeled predictions (Saerens et al., 2002) or by black-box shift estimation (Lipton et al., 2018).

**Concept shift** is **not identifiable** without target labels (or strong assumptions such as sparse mechanism change or a known causal structure): from unlabeled data one cannot tell that $P(y\mid x)$ changed.

**Shortcuts** can be removed only by *knowing or discovering* them: ablating the feature, augmenting to break the correlation, collecting data in which it is decorrelated, or enforcing invariance across multiple domains with different shortcut strengths (which needs domain labels).

---

## 45.3 Four controlled experiments

`code/ch45_shift.py` uses a 10-dimensional Gaussian input and a logistic label with an interaction term (so that a linear model is *misspecified*, which is why covariate shift matters).

### 45.3.1 Covariate shift

The target has shifted means on three coordinates and larger variance on two; the label rule is the same.

| Model | Source acc / AUROC | Target acc | Target AUROC | Target ECE | 90% conformal coverage: source / target |
|---|---|---|---|---|---|
| Linear, source-trained | 0.721 / 0.809 | **0.617** | 0.749 | **0.242** | 0.907 / **0.795** |
| Boosted trees, source-trained | 0.752 / 0.833 | 0.816 | 0.894 | 0.074 | 0.906 / 0.930 |
| Linear, **importance-weighted** | | **0.817** | 0.884 | **0.014** | |
| Boosted trees, importance-weighted | | 0.827 | 0.901 | 0.065 | |
| Reference: right features, target labels | | 0.875 | 0.945 | | |

(Target prevalence is 0.67 against 0.50 in the source, so accuracies are not directly comparable across domains.) Four lessons. **(i) Covariate shift hurts a misspecified model**: the linear model falls from 0.72 to 0.62 and becomes badly calibrated (ECE 0.24), because the target emphasizes the region where the missing interaction matters most. **(ii) It need not hurt a flexible model** (trees: 0.82 on the target), which is a warning against assuming shift always degrades performance, and against extrapolating from one model's behavior to another. **(iii) Importance weighting repairs the misspecified model without target labels** (0.617 to 0.817; ECE 0.242 to 0.014) by fitting the region that matters for the target, at a cost: the effective sample size dropped from 6,000 to about 560. **(iv) Conformal coverage is guaranteed only under exchangeability**: coverage is 0.907 on held-out source data and 0.795 for the linear model on the shifted target, below the nominal 0.90; the sets for the tree model over-cover (0.930), so *uncertainty is miscalibrated in either direction under shift*. Weighted conformal prediction (Tibshirani et al., 2019) restores coverage under covariate shift if the weights are known.

### 45.3.2 Label shift

A classifier trained at 50% prevalence is applied at 5%.

| | Accuracy | AUROC | AUPRC | ECE | Precision at the 0.5 threshold |
|---|---|---|---|---|---|
| Source-trained (prior 0.50) | 0.728 | 0.800 | 0.230 | **0.321** | **0.119** |
| Prior-corrected (EM estimate 0.059; true 0.05) | **0.952** | 0.800 | 0.230 | **0.010** | **0.559** |

The ranking (AUROC, AUPRC) is *unchanged* by the correction, but **calibration and threshold-based decisions are repaired**: ECE falls from 0.321 to 0.010, and the precision of a positive call rises from 12% to 56% (the majority-class rule has accuracy 0.95, so accuracy is not informative here: Chapter 43's prevalence lesson). **This is the single most common shift in clinical and screening settings, and the cheapest to fix**, provided $P(x\mid y)$ is truly unchanged (it is not if, for instance, the disease is milder in the screened population: that is concept shift).

### 45.3.3 Concept shift and the value of target labels

In the target, 35% of the weights change sign or magnitude, and the interaction weakens. The source-trained model reaches target accuracy 0.690 with no target labels.

| Target labels | Target-only model | Pooled (source down-weighted) + target | Source model fine-tuned |
|---|---|---|---|
| 10 | 0.581 | **0.692** | 0.611 |
| 30 | 0.660 | **0.695** | 0.680 |
| 100 | 0.713 | 0.705 | **0.715** |
| 300 | 0.726 | 0.720 | 0.726 |
| 1,000 | **0.737** | 0.733 | 0.737 |

With *very few* target labels the source model (pooled or fine-tuned) is better than training on the target alone; around **100 target labels the target-only model catches up**, and thereafter source data add nothing in this problem (all converge to about 0.74, the limit set by label noise). The numbers are specific to the toy, but the pattern is general: the value of the source is a *prior* that is worth a fixed number of target examples, and **a concept shift is repaired by target data, not by cleverness**. In biology, this is why even a handful of labeled examples from the new cell type, population, or assay is the most valuable thing you can collect, and why *how* to select them (Chapter 46) matters.

### 45.3.4 A shortcut

In the source a feature $s$ (a batch signature) equals the label with probability 0.9; in the target it is independent of the label.

| Model | Source-like accuracy | Target accuracy |
|---|---|---|
| ERM with the shortcut (weight on $s$ 2.85 vs mean $\lvert$biological weight$\rvert$ 0.50) | **0.947** | **0.561** |
| Shortcut known and removed | 0.767 | **0.770** |
| Augmentation (shuffle the shortcut) | 0.879 | 0.727 |
| 300 target-like examples *added to* the 6,000 source examples | 0.942 | **0.586** |

*The model that is best on held-out source data is the worst on the target*: 0.947 versus 0.561. Removing the shortcut costs 18 points in-distribution and gains 21 on the target. Augmentation that breaks the correlation recovers much of it (0.727). Adding 300 target-like examples to 6,000 shortcut-correlated ones barely helps (0.586): the shortcut still dominates the fit unless the new examples are *weighted* or the model is constrained. **In-distribution validation cannot detect a shortcut**; only a shifted test set (a held-out batch, donor, site, ancestry) can (Chapter 43).

```python
--8<-- "code/ch45_shift.py"
```

```text
== 1. Covariate shift: P(x) changes, P(y|x) fixed; the true logit has an interaction x1*x2 that a linear model misses ==
source prevalence 0.50; target prevalence 0.67
linear model, source-trained                   source acc 0.721 AUROC 0.809 | target                                         acc 0.617  AUROC 0.749  AUPRC 0.854  ECE 0.242 
    90% conformal sets: coverage on source (exchangeable) 0.907; on shifted target 0.795
boosted trees, source-trained                  source acc 0.752 AUROC 0.833 | target                                         acc 0.816  AUROC 0.894  AUPRC 0.942  ECE 0.074 
    90% conformal sets: coverage on source (exchangeable) 0.906; on shifted target 0.930
importance weights: mean 0.64, max 20.0, effective sample size 561 of 6000
linear model, importance-weighted              acc 0.817  AUROC 0.884  AUPRC 0.937  ECE 0.014 
boosted trees, importance-weighted             acc 0.827  AUROC 0.901  AUPRC 0.949  ECE 0.065 
reference: right features, target labels       acc 0.875  AUROC 0.945

== 2. Label (prevalence) shift: P(y) changes from 0.50 to 0.05 with P(x|y) fixed ==
source-trained (prior 0.50)                    acc 0.728  AUROC 0.800  AUPRC 0.230  ECE 0.321 
EM prevalence estimate (true 0.05)             0.059
prior-corrected posteriors                     acc 0.952  AUROC 0.800  AUPRC 0.230  ECE 0.010
precision at the 0.5 threshold: uncorrected    0.119; corrected 0.559

== 3. Concept shift: the mechanism differs in the target (35% of weights change sign or size); accuracy vs number of target labels ==
source-trained model, no target labels: target acc 0.690
target labels   target-only   pooled source + target   source model fine-tuned (shrunk toward source)
      10           0.581           0.692                   0.611
      30           0.660           0.695                   0.680
     100           0.713           0.705                   0.715
     300           0.726           0.720                   0.726
    1000           0.737           0.733                   0.737

== 4. A shortcut feature (batch signature) correlates with the label in the source (r about 0.9) and not in the target ==
ERM with the shortcut: source-like test acc 0.947; target acc 0.561; weight on shortcut 2.85 vs mean |biological weight| 0.50
shortcut known and removed:           source-like acc 0.767; target acc 0.770
augmentation (shuffle the shortcut):  source-like acc 0.879; target acc 0.727
300 target-like examples added:       source-like acc 0.942; target acc 0.586
```

---

## 45.4 Biological shifts, in concrete form

* **Ancestry** (Chapter 26): a polygenic score trained on one ancestry loses accuracy continuously with genetic distance; the mechanism is a *covariate/LD shift* acting on tag variants (the toy lost 83% of $R^2$) plus *concept shift* through gene–environment and effect heterogeneity.
* **Cell type and cell line** (Chapters 31, 39): sequence-to-function models treat the cell type as an output index; a perturbation model trained in K562 meets a primary cell with different regulatory wiring (*concept shift*), different composition (label shift), and different technology (batch shortcuts). In the Arc Virtual Cell Challenge, evaluation in a held-out cell context was the point [[S]].
* **Species**: a model trained on human and mouse (Enformer) applied to a third species is a covariate shift with partial concept shift (regulatory logic is partly conserved); protein language models trained on all of UniRef generalize across species less well for those poorly represented (Chapter 34; species bias).
* **Assay and platform**: a reference built from droplet RNA-seq used to deconvolve spatial data (platform shift; Chapter 25: an error floor from platform mismatch); a classifier trained on one proteomics platform applied to another.
* **Laboratory and site**: the shortcut case. Histopathology and imaging models learn staining and scanner signatures; transcriptomic classifiers learn dissociation and chemistry signatures (Chapters 25, 30).
* **Time**: drug discovery models trained on past chemistry applied to new chemotypes (support shift, Chapter 24); pathogen sequence models trained before a new variant emerges.
* **Novelty (support shift)**: a protein from an unseen fold; a compound in an empty region of chemical space; a perturbation in a pathway absent from training. Importance weighting is undefined there; the correct response is *abstention* (an applicability domain; Chapter 24) and targeted data collection.

---

## 45.5 Uncertainty under shift

**Calibration** (the match between predicted probabilities and frequencies) degrades under shift: ECE 0.24 for the linear model under covariate shift; 0.32 under label shift. **Conformal prediction** gives marginal coverage only under exchangeability (observed: 0.795 instead of 0.90), but variants (weighted conformal prediction for covariate shift; adaptive procedures for label shift) restore it under *known* shift types. **Out-of-distribution detection** (distance to training data in an embedding, Mahalanobis distance, ensembles, density models) flags novelty, but biological embeddings (Chapter 13) can map novel inputs onto familiar regions (a model that has never seen a fold may still produce a confident embedding); report the *error as a function of distance* (Chapter 24: RMSE rose from 0.58 to 1.00 as nearest-neighbor similarity fell below 0.5) rather than a binary flag, and calibrate the flag against held-out novelty. The standard empirical finding in machine learning is that **predictive uncertainty is least trustworthy where it matters most, under shift** (Ovadia et al., 2019); biology gives no exemption.

---

## 45.6 Foundation models and shift

*Pretraining as a hedge.* Training on diverse data (many species, tissues, cell types, labs) increases the coverage of $P(x)$ and so reduces *covariate and support* shift; this is a principal argument for scale (Chapter 17) and for atlas-scale models (Chapter 38). *But diversity also supplies shortcuts*: with many sources, a model can learn source identity as a feature, and its apparent invariance may be memorization.

*Zero-shot versus fine-tuned.* A pretrained model applied zero-shot meets the full shift; **concept shifts require labels**, so fine-tuning with a small target set is the pragmatic remedy. Fine-tuning can *distort* pretrained features and hurt out-of-distribution accuracy relative to linear probing; the recommended practice is to try a linear probe first, then fine-tune with care (Kumar et al., 2022), or interpolate between pretrained and fine-tuned weights (Wortsman et al., 2022) [[S]]. *Evaluation*: pretraining data cover almost all public data, so a "held-out domain" must be defined relative to the pretraining corpus (Chapter 43's contamination).

---

## 45.7 A decision procedure

1. **Name the intended shift** (which of §45.1's types, in biological terms) *before* building the evaluation.
2. **Diagnose** with unlabeled data: a *domain classifier* (AUROC near 0.5 means no covariate shift; near 1 means severe, with an ESS warning); compare predicted prevalence (label shift); check marginal distributions of embeddings; look for a batch/site signature in the embeddings (shortcut).
3. **Test the mechanism** with a *small labeled target sample* (as few as 30–100 in the toy): is $P(y\mid x)$ the same? If the target-only model with 100 labels clearly beats the source model, the mechanism changed.
4. **Match the remedy**: covariate shift: importance weighting (check ESS), weighted conformal; label shift: prior correction; shortcut: ablate, augment, decorrelate, multi-domain invariance; concept shift: collect and use target labels; support shift: abstain and design new experiments (Chapter 46).
5. **Evaluate on a held-out domain**: the *only* test that detects shortcuts (Chapter 43).
6. **Monitor** after deployment: input drift, prevalence, calibration, performance by site.
7. **Report per-domain performance**, not only the average.

---

## 45.8 Worked research examples

!!! example "Worked Research Example 45.1: An annotation model moved to a new tissue and a disease dataset"
    **Situation.** A cell-type classifier is trained on a healthy blood atlas and applied to a tumor-infiltrating immune dataset from a different lab and chemistry. On a small labeled subset it has accuracy 0.78 against 0.96 in-distribution. The group asks whether to retrain or to correct.

    **Question.** Which shifts are present, and what should they do?

    **Reasoning.**

    1. *Diagnose by shift type.* **Covariate**: tumor-infiltrating cells have different expression (hypoxia, exhaustion). **Label**: composition differs from blood (exhausted T cells, macrophages abundant; some blood types absent), a prior shift and partly *support* shift (types absent from training). **Concept**: a "CD8 T cell" in tumor has a different program from the blood CD8 T cell, so $P(y\mid x)$ changes with state. **Shortcut**: lab and chemistry signatures (Chapter 30).
    2. *Diagnostics.* A domain classifier on embeddings (is the shift visible?); inspect the *composition table* of both datasets (Chapter 30); examine confusion by class: label shift predicts errors concentrated in classes whose prevalence changed; concept shift predicts errors in classes whose states differ; support shift predicts confident errors on cells of unseen types.
    3. *Remedies in order of cost.* (a) Prior correction on the unlabeled target (cheap). (b) Importance weighting or batch-aware training with the target's unlabeled data (domain-adaptive embeddings; with the Chapter 30 cautions on overcorrection). (c) **Label 100–300 target cells chosen to span the confusions** (§45.3.3) and fine-tune or pool. (d) For unseen types, enable an *unassigned* label with a calibrated rejection threshold.
    4. *Evaluation.* Hold out a *donor* or *study* from the target, not random cells; report per-class metrics with the reliability ceiling of annotation (annotator agreement; Chapter 30).

    **Expert analysis.** The drop from 0.96 to 0.78 is a mixture of four effects with different cures. The decision is *diagnose, then act*: spending a month on a bigger model would address none of them, whereas 200 labels and a rejection class may address most.

!!! example "Worked Research Example 45.2: A perturbation predictor trained in a cancer cell line is proposed for primary cells"
    **Situation.** A group has a perturbation-response model trained on genome-scale Perturb-seq in a leukemia cell line and wishes to predict responses in primary T cells, where only a handful of perturbations can be afforded. There is no established answer for how much transfers.

    **Question.** What would you test, and how would you decide?

    **Reasoning.**

    1. *Which parts are likely to transfer?* Perturbations with strong effects on **core cellular machinery** (essential genes; ribosomal, proteasomal, cell cycle) produce shared responses across lines; **context-specific regulators** (lineage TFs, signaling) do not. The hypothesis is a *partition* of perturbations into shared and context-dependent, which can be tested on any two cell lines with overlapping perturbations.
    2. *Estimate transport on available pairs.* Use two or more public cell-line datasets: train on one, test on another; stratify by perturbation class and by the reliability ceiling (Chapter 25). The *ratio of cross-context to within-context accuracy* by class estimates transportability.
    3. *Choose the few primary-cell experiments by information* (Chapter 46): perturbations that discriminate shared from context-specific response, and *anchor* perturbations with strong effects in both.
    4. *Adaptation.* Use the primary-cell experiments to fit a *context offset* (shared responses) or a low-rank cell-type-specific correction; evaluate with held-out perturbations and ceiling-normalized metrics.
    5. *Decision rule.* Pre-specify: transfer is accepted for a perturbation class if the ceiling-normalized cross-context score exceeds a threshold relative to the additive/mean baseline; otherwise the model is used only for ranking, with experiments for confirmation.

    **What is not known.** Whether context transfer is better achieved by scale (many cell types in pretraining), by structure (explicit cell-type-conditional mechanisms; Chapter 44), or by data; this is a central open problem of Chapter 51.

---

## 45.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: a shift audit"
    1. **Write the deployment shift** in the four-type language, with a biological example for each.
    2. **Run a domain classifier** on inputs (and embeddings) between your training set and the target; report AUROC and the effective sample size of importance weights.
    3. **Compare prevalence** (predicted vs expected) and examine class-conditional distributions.
    4. **Look for shortcuts**: can batch, donor, site, or depth be predicted from the embedding? Train a model with those only as a negative control.
    5. **Estimate the mechanism change**: fit a target-only model with 30 to 300 labels and compare with the source model.
    6. **Select the remedy** per §45.7 and re-evaluate on a held-out *domain*.
    7. **Calibration under shift**: ECE and conformal coverage on the target.
    8. **Write the scope** of validity: "validated for domains similar in … ; not validated for …".

    **What it teaches.** Most generalization failures are diagnosable with a day of work on unlabeled data and a small labeled sample, and each has a cheaper remedy than scale.

    **An open question to carry forward.** Pretrained biological foundation models see many domains but are usually evaluated on one or two. Could one construct a *shift atlas*: a standardized set of paired domains (species, ancestry, cell line, platform, time) with measured mechanism change, so that the transportability of any model can be reported as a profile (by shift type and magnitude)? What would the minimal set of paired experiments be to identify, for a given model family, which shift types it handles and which need labels?

---

## 45.10 Connections

- **Backward:** generalization and splits (Chapters 7, 43); representation and invariance (Chapters 13, 16); conformal prediction (Chapter 18); molecular applicability domains (Chapter 24); batch effects and integration (Chapters 25, 30); ancestry and portability (Chapter 26); cell-type extrapolation (Chapter 31); causal structure and transportability (Chapter 44).
- **Forward:** experimental design for adaptation (Chapter 46); scaling and diversity (Chapter 47); open problems of context and virtual cells (Chapters 50–53); AI scientists and out-of-distribution hypotheses (Chapter 54).

!!! takeaways "Key takeaways"
    1. Shifts differ by *what changes*: covariate, label, concept, shortcut, or support; the remedies differ and do not substitute for one another.
    2. **Covariate shift** is correctable by importance weighting if supports overlap (the linear model went from 0.617 to 0.817 target accuracy, ECE 0.242 to 0.014, at the cost of an effective sample size falling from 6,000 to about 560); flexible models may be unaffected.
    3. **Label shift** is correctable from unlabeled target data (EM estimate 0.059 for a true 0.05): calibration error fell from 0.321 to 0.010 and precision at 0.5 rose from 12% to 56%, while AUROC and AUPRC were unchanged.
    4. **Concept shift** requires target labels: in the toy, about 100 labels sufficed for a target-only model to match source-based adaptation; below that the source acted as a prior worth a few dozen examples.
    5. **Shortcuts** make the best in-distribution model the worst in the target (0.947 vs 0.561); ablation or decorrelating augmentation helps (0.770, 0.727); simply adding 300 target-like examples to a shortcut-dominated training set does not (0.586). Only a held-out domain test detects them.
    6. **Conformal coverage and calibration fail under shift** (coverage 0.795 vs nominal 0.90); report error as a function of novelty and abstain on support shift.
    7. Pretraining on diverse data reduces covariate and support shift but supplies shortcuts; zero-shot cannot repair concept shift; fine-tune carefully.
    8. **Diagnose first**: domain classifier, prevalence comparison, shortcut check, small labeled target sample; then match the remedy.

---

## Further reading

- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. & Lawrence, N. D. (Eds.) (2009). *Dataset Shift in Machine Learning.* MIT Press. Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *J. Stat. Plan. Inference* 90, 227–244. Sugiyama, M., Krauledat, M. & Müller, K.-R. (2007). Covariate shift adaptation by importance weighted cross validation. *J. Mach. Learn. Res.* 8, 985–1005.
- Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput.* 14, 21–41. Lipton, Z., Wang, Y.-X. & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML*. Tibshirani, R. J., Foygel Barber, R., Candès, E. & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Peters, J., Bühlmann, P. & Meinshausen, N. (2016). Causal inference by using invariant predictions: identification and confidence intervals. *J. R. Stat. Soc. B* 78, 947–1012. Arjovsky, M., Bottou, L., Gulrajani, I. & Lopez-Paz, D. (2019). Invariant risk minimization. *arXiv:1907.02893.* Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nat. Mach. Intell.* 2, 665–673.
- Koh, P. W. et al. (2021). WILDS: a benchmark of in-the-wild distribution shifts. *ICML.* Gulrajani, I. & Lopez-Paz, D. (2021). In search of lost domain generalization. *ICLR.* Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS.* Kumar, A., Raghunathan, A., Jones, R., Ma, T. & Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. *ICLR.* Wortsman, M. et al. (2022). Robust fine-tuning of zero-shot models. *CVPR.*
