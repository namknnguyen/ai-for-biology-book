# Chapter 48. Mechanistic Interpretability of Biological Models

!!! abstract "Chapter at a glance"
    **Motivation.** A biological model that predicts well is a hypothesis generator only if one can say *what it has learned*: which motifs, which interactions, which rules. Interpretability serves three different purposes in biology: **debugging** (finding shortcuts and failures before a model is deployed), **discovery** (extracting regulatory grammar, structural principles, or mechanisms from a model that has seen more data than any person), and **auditing** (checking what a model can do in domains with biosecurity consequences). The methods differ in what they claim: *attribution* says which inputs mattered, *probing* says what a representation encodes, *circuit analysis* says which computation was performed, and *causal abstraction* says that the model implements a particular algorithm. This chapter builds the toolkit on a model whose true circuit is known, so that every method can be graded against ground truth, and runs seven experiments: shortcut detection, filter-to-motif matching, ablation, probing against causal erasure, activation patching, and sparse autoencoders in two regimes (no superposition and strong superposition). The results include two cautions that matter for the literature: single-neuron ablation underestimates importance when a feature is distributed, and in this toy a sparse autoencoder did *not* produce more selective features than raw neurons. It then reviews what interpretability has found in biological models, grades it, and gives a validation protocol.
    **Prerequisites.** Chapters 10, 12, 13, 18, 31, 34, 38, 43, 45.
    **You will be able to:** (1) distinguish attribution, probing, patching, and causal abstraction and state what each can and cannot show; (2) explain superposition and what a sparse autoencoder assumes; (3) design a faithfulness test for an attribution; (4) interpret a probe with a control task and an erasure experiment; (5) validate a claimed circuit or feature by an intervention, in silico and in the lab; (6) write an interpretability claim with its rung on the claim ladder.

---

## 48.0 Why interpretability is a scientific tool in biology, not only an engineering one

Two features distinguish biology from the usual setting of interpretability research.

1. **Ground truth exists and can be generated.** Motifs have position weight matrices (Chapter 29) and binding sites can be mutated; epistatic grammar (spacing and orientation rules) can be tested by synthesizing sequences (Chapter 31's MPRA remedy); a structural contact can be broken by mutation (Chapter 23). An interpretation of a sequence model is therefore a *testable prediction about the world*, and the test is a wet-lab experiment.
2. **The goal is often discovery.** In language there is no hidden law that the interpretation reveals. In genomics a model trained on thousands of tracks has absorbed regularities that may not be annotated (new motifs, new syntax), and the point of interpreting it may be a new fact.

The consequence is a stricter standard: *an interpretation is a hypothesis that predicts the outcome of an intervention on the model and on the biological system.* Pretty visualizations are not evidence.

!!! lens "Research lens: the interpretation ladder"
    **I1 Attribution**: which input features changed the output (gradients, in silico mutagenesis). **I2 Representation**: what information a layer encodes (probes, clustering). **I3 Mechanism**: which components compute the output, and how (ablation, patching, circuits). **I4 Causal abstraction**: the model implements an *algorithm* described at a higher level, verified by interventions that map between the model's states and the algorithm's variables (Geiger et al.). **I5 Biological validation**: the interpretation predicts the result of an experiment on the real system. A claim at rung $k$ requires evidence at that rung; most published interpretations are I1–I2, and the biology-specific value is at I5.

---

## 48.1 The toolkit

**Attribution.** *Gradient × input*, *integrated gradients* (Sundararajan et al. 2017), and *DeepLIFT* (Shrikumar et al. 2017) assign scores to input positions; for one-hot DNA there are gauge issues (Chapter 18: gradients should be corrected for the constraint that exactly one letter is active per position). **In silico mutagenesis (ISM)** replaces each base by each alternative and records the change in prediction; it is a *counterfactual* and therefore faithful by construction for single substitutions, and it is the standard against which other attributions are checked. TF-MoDISco (Shrikumar et al. 2020) clusters high-attribution seqlets into motifs.

**Ablation.** Replace the activation of a unit (or a direction, or a layer) by a reference value (zero, the mean over data, or a resampled value) and measure the change in output. The choice of reference is a hypothesis (mean ablation asks "what if the unit carried no information about this input"; resampling asks "what if it carried the information of another input").

**Probing.** Train a (linear) classifier to predict a property from a layer's activations. *What a probe shows*: the property is linearly decodable. *What it does not show*: that the model *uses* it. Two controls are essential: a **control task** (a probe of equal capacity on a random labeling; Hewitt and Liang 2019) to separate representation from probe capacity, and a **causal test** (erase the probe direction and see whether the output changes).

**Activation patching (causal tracing).** Run the model on a *clean* input and on a *corrupted* input that differs in one factor; copy an activation from the clean run into the corrupted run; measure how much of the output difference is *recovered*. Components whose patching recovers the output carry the information the factor changes (Vig et al. 2020; Meng et al. 2022; Heimersheim and Nanda 2024 for practice and pitfalls). **Path patching** isolates the effect through specific connections.

**Causal abstraction and interchange interventions.** If the model implements a high-level algorithm with variables (e.g., "motif A present", "motif B present", "AND"), then swapping the model's internal representation of "A present" between two inputs should change the output as the algorithm predicts. Passing the interchange test for all variables supports I4 (Geiger et al. 2021, 2023).

**Sparse autoencoders and superposition.** If a model represents more features than it has dimensions, it can *superpose* them in near-orthogonal directions, exploiting the fact that features are rarely active together (Elhage et al. 2022). A sparse autoencoder (SAE) trains a dictionary to reconstruct activations from sparse combinations of learned directions,

$$
\hat h=D f(h),\quad f(h)=\mathrm{ReLU}(Eh+b),\qquad \mathcal L=\|h-\hat h\|_2^2+\lambda\|f(h)\|_1,
$$

with $f\in\mathbb R^{m}$ and $m>\dim h$, in the hope that each latent corresponds to one human-interpretable feature (Bricken et al. 2023; Templeton et al. 2024). The assumptions are strong: features are approximately *linear directions*, *sparsely active*, and recoverable by this optimization; none is guaranteed, and the objective has many near-equivalent solutions (feature splitting at larger $m$; absorption of general features by specific ones).

!!! math "Derivation: why superposition is possible for sparse features"
    Let $n$ features $x_i\in[0,1]$ be active independently with probability $s$ (sparsity $1-s$), and embed them in $d<n$ dimensions by $h=Wx$, reconstructing by $\hat x=\mathrm{ReLU}(W^\top h+b)$. For features with directions $w_i$ ($\|w_i\|=1$) the reconstruction of feature $i$ is $x_i+\sum_{j\ne i}(w_i\!\cdot\!w_j)x_j+b_i$. The interference term has expected magnitude $\propto s\sqrt{\sum_{j\ne i}(w_i\!\cdot\!w_j)^2}$, which is *small when $s$ is small*; a negative bias plus the ReLU removes the small positive interference. Hence for small $s$ the loss is lower when many features share dimensions at the price of a little interference than when most features are dropped, and a transition to superposition occurs as $s$ falls (Elhage et al. 2022). The implication for interpretation: **neurons need not align with features**, and the number of recoverable features can exceed the width. $\square$

---

## 48.2 A model with a known circuit: seven experiments

**Task.** A sequence of 50 bases is positive iff motif A *and* motif B are both present (six-letter motifs planted at random positions). Two further motifs are planted: **S**, a *shortcut* present in 95% of positives and 5% of negatives in the training distribution, and **D**, a *distractor* that co-occurs with A in training (95%) but is otherwise unrelated to the label. A small CNN (16 filters of width 6, ReLU, global max-pooling, one hidden layer) is trained for five epochs on this distribution. The true circuit is therefore: *detect A, detect B, AND*; the shortcut and the distractor are the confounds a real model might exploit.

```python
--8<-- "code/ch48_interp_circuit.py"
```

```text
== 1. Behavior: the model was trained where the shortcut S predicts the label 95% of the time and D co-occurs with A ==
accuracy on in-distribution test data: 0.985;  on shortcut-free data (S independent of the label): 0.946
among shortcut-free NEGATIVES, accuracy when S is present: 0.959, when S is absent: 0.969
among shortcut-free POSITIVES, accuracy when S is present: 0.993, when S is absent: 0.864

== 2-3. What does each of the 16 filters respond to, and what happens when it is ablated (its pooled activation fixed at its mean)? ==
filter   most associated motif (AUROC of its activation for motif presence)   accuracy shortcut-free after ablation   change
   12    B  (0.95)                                                          0.864                            -0.081
    9    A  (1.00)                                                          0.877                            -0.069
   15    B  (0.95)                                                          0.889                            -0.056
    1    A  (1.00)                                                          0.898                            -0.048
    8    A  (1.00)                                                          0.898                            -0.048
   11    D  (0.43)                                                          0.946                            +0.001
   14    A  (0.43)                                                          0.946                            +0.001
    3    S  (0.44)                                                          0.947                            +0.001
(the remaining 8 filters change accuracy by at most 0.029)

== 4. Linear probes decode every motif; only some are used ==
motif   probe AUROC (shortcut-free data)   accuracy after erasing the probe direction: shortcut-free data | training distribution
A       0.997                      0.496 (change -0.450)                         0.502 (change -0.483)
B       0.955                      0.831 (change -0.114)                         0.834 (change -0.151)
S       0.756                      0.950 (change +0.004)                         0.926 (change -0.059)
D       0.758                      0.889 (change -0.056)                         0.961 (change -0.024)

== 5. Activation patching: clean = positive sequence with A and B; corrupted = the same sequence with motif A replaced by random bases ==
mean logit difference (positive minus negative): clean 2.64, corrupted -6.77  (corrupting motif A flips 0.99 of the predictions)
patching one filter's pooled activation from the clean run into the corrupted run: fraction of the logit difference recovered
   filter 9 (matches A): recovered 0.37
   filter 1 (matches A): recovered 0.30
   filter 8 (matches A): recovered 0.30
   filter 14 (matches A): recovered 0.01
   all remaining filters: at most 0.01

== 6. A sparse autoencoder (64 latents, L1 penalty) on the 16 pooled activations ==
37 active latents, fraction of variance unexplained 0.001, mean L0 34.1
motif   best single NEURON AUROC (of 16)   best single SAE LATENT AUROC (of 37 active)
A       0.997                              0.996
B       0.946                              0.928
S       0.904                              0.875
D       0.825                              0.708

== 7. Superposition control: 24 independently planted 6-mers (each present with probability 0.04, about one per sequence), a model with only 8 filters trained to report which are present ==
the model reports the motifs with mean AUROC 0.769 through an 8-dimensional bottleneck (24 features in 8 dimensions)
best single NEURON per motif: mean AUROC 0.746 (motifs with a neuron above 0.9: 6 of 24)
sparse autoencoder with 96 latents, by L1 coefficient:  active latents   mean L0   unexplained variance   mean best-latent AUROC   motifs with a latent above 0.9
                                       l1 = 0.001           55        51.6             0.000                   0.738                   5 of 24
                                       l1 = 0.01            48        27.0             0.000                   0.741                   6 of 24
                                       l1 = 0.1             15         8.9             0.000                   0.709                   5 of 24
                                       l1 = 0.3             16         9.4             0.000                   0.730                   4 of 24
                                       l1 = 1               12         7.5             0.001                   0.709                   3 of 24
```

**Reading the experiments.**

1. **Behavior reveals the shortcut.** Accuracy is 0.985 in the training distribution and 0.946 when S is independent of the label. The cut is in positives lacking S: accuracy 0.864 against 0.993 for positives with S. The model *partly* uses S. Only a test on data where the confound is decorrelated reveals it (Chapter 45).
2. **Filter-to-motif matching works for the main features.** Filters with AUROC 0.95–1.00 for A or B identify the detectors; filters with AUROC near 0.5 or below respond to nothing in particular. The match is *associational*.
3. **Single-unit ablation understates importance because the features are distributed.** Mean-ablating any single A or B filter reduces shortcut-free accuracy by only 0.05–0.08, although motif A is essential: there are at least three A-detecting filters and two B-detecting ones. Redundant filters compensate (the "Hydra effect"; McGrath et al. 2023). A conclusion "no single unit is important" would be wrong.
4. **The causal unit is a direction, and probes plus erasure identify it.** A linear probe for A from the 16 pooled activations has AUROC 0.997, and *erasing the probe direction* (projecting it out) drops accuracy to 0.50, i.e., chance: the model's use of A is mediated by this one direction. For B the drop is 0.11–0.15 (the direction captures part of B's use). For S the probe is decodable (AUROC 0.76), and erasing it changes accuracy by $-0.059$ in the training distribution but $+0.004$ in shortcut-free data: **S is used where it is informative and harmless where it is not**; whether a feature is "used" depends on the distribution on which the question is asked. The distractor D is also decodable (0.76) and erasing its direction reduces accuracy by 0.056 on shortcut-free data: *D is used as a proxy for A*, because in training it co-occurred with A. **Decodability does not imply use, and use does not imply the intended feature**; only intervention distinguishes them.
5. **Activation patching localizes the information and exposes the distribution.** Corrupting motif A in positive sequences flips 99% of predictions (logit difference 2.64 clean, $-6.77$ corrupted). Patching one filter's pooled activation from the clean run recovers 37%, 30%, and 30% of the logit difference for the three A filters (97% in total) and at most 1% for any other filter. Each A filter carries a third of the information, which is exactly what ablation could not show.
6. **A sparse autoencoder did not beat the raw neurons here.** On the 16 pooled activations, the best single SAE latent had AUROC 0.996 for A, 0.928 for B, 0.875 for S, and 0.708 for D, against 0.997, 0.946, 0.904, and 0.825 for the best raw neuron: no gain. This is the expected outcome in a representation without superposition: max-pooled ReLU activations already have a privileged basis (the filters), so the neurons *are* the features.
7. **Under heavy superposition, the SAE still did not help in this toy.** With 24 planted motifs reported through an 8-filter bottleneck, the network reports the motifs with a mean AUROC of 0.77; the best single neuron per motif has mean AUROC 0.75 (6 of 24 motifs above 0.9), and SAEs with 96 latents over five sparsity penalties (from $10^{-3}$ to 1.0) reach 0.71–0.74 (3–6 of 24 above 0.9) while the reconstruction stays nearly perfect. SAEs are not magic: in this setting the dictionary found by this optimizer is not more aligned with the planted features than the neurons are, the result depends on sparsity penalty and architecture, and it would be unjustified to conclude either that SAEs fail in general or that they work. **A feature claim from an SAE needs an independent validation: ground truth where available, causal tests, and stability across training seeds.**

!!! lens "Research lens: assumptions of the experiments"
    A tiny CNN on synthetic sequences; one planted circuit; five epochs of training (the shortcut reliance is an early-training phenomenon); mean ablation as the reference; probes on pooled activations; an SAE with the standard $\ell_1$ objective and no architectural tricks (no top-$k$ gating, no decoder norm constraints, no resampling of dead latents), tuned only over the penalty. Real models are deeper, with learned positional and attention structure; the point is the *grading* of methods by ground truth, which real models rarely allow.

---

## 48.3 What has been found in biological models

| Model | Method | Finding | Grade |
|---|---|---|---|
| CNNs for TF binding and accessibility (DeepSEA, Basset, BPNet) | Filters, attribution, ISM, DeepLIFT/MoDISco | Learned filters match known motifs; BPNet's base-resolution profiles revealed motif syntax (spacing and cooperativity) *validated by experiments* (Avsec et al., *Nat. Genet.* 2021) | [[E]] for motif recovery; [[S]] for syntax |
| CNN filter interpretability | Architecture analysis | Whether a filter looks like a motif depends on architecture (receptive field, pooling, activation); exponential activations and shallower designs yield more interpretable filters (Koo and Eddy 2019; Koo and Ploenzke 2021) | [[S]] |
| Enformer-class models | Attribution, ISM, shuffling | Attributions highlight known enhancers and promoters, but ISM/shuffle controls show that distal contributions are weak (Chapter 31) | [[S]] |
| Protein language models (ESM-2) | Attention, probes, SAEs | Attention maps contain contacts; probes decode secondary structure; SAEs find features aligned with domains, binding sites, and families (Simon and Zou 2024/25; Adams et al., *PNAS* 2025); steering along some features changes generations | [[S]] for decodability; [[P]] for causal use |
| ESM-2 contact prediction | Analysis of mechanism | Contacts are predicted largely by recalling pairwise motifs seen in training (Zhang et al., *PNAS* 2024), not by a coupling analysis of the target family | [[S]] |
| Evo 2 | SAEs on internal activations | Latents aligned with exon–intron boundaries, transcription-factor motifs, protein secondary structure, and prophage regions, found without annotations (Brixi et al.) | [[P]]: alignment shown, causal use not tested |
| Single-cell foundation models | Attention, SAEs (2025 preprints) | Reports of gene-program features; recovery of known regulatory relationships is inconsistent across models | [[H]] |
| Structure networks (AF2 pair representation) | Probing, ablation | Pair representation encodes distances and contacts; the roles of triangle updates are partly ablation-verified | [[P]] |

---

## 48.4 Validating an interpretation: a protocol

1. **Planted-signal check.** Before interpreting a real model, plant known motifs/rules in synthetic data of the same format, train a model of the same class, and verify that your method recovers them (the experiments above). A method that fails here cannot be trusted there.
2. **Sanity checks for attribution.** Randomize model weights (Adebayo et al. 2018): attributions that do not change are not model-dependent. Compare with ISM on a sample of sequences.
3. **Stability.** Re-train with different seeds; features or circuits that do not replicate are artifacts of one run (a standard concern for SAE latents).
4. **Necessity and sufficiency.** *Necessity*: ablating or erasing the feature degrades the output; *sufficiency*: inserting the feature (patching it in, or planting the motif) produces the output. The experiments show both can be misleading in isolation (redundancy, proxies).
5. **Distribution awareness.** Test interpretations in the training distribution and in a distribution where confounds are decorrelated (§48.2, reading 4).
6. **Interchange test** for a claimed algorithm (I4).
7. **Biological validation (I5).** Convert the interpretation into a sequence-level prediction (a new motif, a spacing rule, a variant) and test it by MPRA or genome editing; report the fraction of predictions confirmed.

---

## 48.5 Worked research examples

!!! example "Worked Research Example 48.1: An SAE latent 'fires on beta hairpins'"
    **Situation.** A protein-LM SAE has a latent that activates strongly on positions annotated as beta hairpins in 38 of 40 inspected proteins. The authors propose that the model has a "beta-hairpin feature" and that boosting the latent steers generation toward hairpins.

    **Question.** What would you test, and what could the claim become?

    **Reasoning (Expert Chain).**

    1. **L1 Claim.** A feature in the SAE dictionary corresponds to a structural concept used by the model.
    2. **L3–L5 Assumptions and failure modes.** (a) *Selection*: the 40 proteins were inspected because the latent fired; the false-positive rate is unknown. (b) *Correlates*: hairpins co-occur with glycine/proline motifs and low-complexity patterns; the latent may track sequence composition. (c) *Splitting/absorption*: a more general "beta strand" latent may be absorbed in the hairpin one. (d) *Stability*: other seeds or dictionary sizes may split or lose it. (e) *Use*: firing is not use.
    3. **L10 Experiments.** (i) Compute precision and recall of the latent against an *automatic* hairpin annotation (DSSP) over a held-out set of thousands of proteins; report the AUROC and the false-positive structure by sequence context. (ii) Compare to a probe for hairpins from the raw activations (does the SAE latent add anything?). (iii) *Ablate* the latent's direction (resample ablation) and measure the change in downstream tasks: the model's predicted structure or the likelihood of known hairpin-rich proteins. (iv) *Steer*: add the latent's decoder direction at a position and check whether the generated sequence folds (by an independent predictor) into a hairpin with the correct register, compared with steering by a random direction of equal norm. (v) *Seed stability*: retrain the SAE with five seeds and see the fraction with a matching latent. (vi) For biology, take generated hairpin designs and test stability or fold in the lab.
    4. **L11 Interpretation.** If (i)–(iii) hold, the claim is I3: "this direction carries hairpin information used by the model's downstream computation"; (iv)–(v) support causal sufficiency and stability; (vi) is I5. "The model has the concept of a beta hairpin" requires the interchange test (I4) and is unlikely to be established by one latent.

    **Expert analysis.** The cheapest decisive test is (iii)/(iv) against a random-direction control; the commonest failure of such claims is the absence of any intervention.

!!! example "Worked Research Example 48.2: Extracting new regulatory grammar from a sequence model. No known answer"
    **Situation.** A group has a base-resolution model of chromatin accessibility in a cell type that has never been studied by motif-grammar methods. Attribution highlights recurring seqlets; some match known motifs, some do not. The group wants to claim *new motifs* and *new syntax rules* discovered by interpreting the model.

    **Question.** How do you turn the interpretation into a discovery that survives experiment?

    **Reasoning.**

    1. **Define the claims.** (a) *Novel motif*: a sequence pattern with high attribution that matches no entry in JASPAR/HOCOMOCO; (b) *syntax*: the effect of motif A depends on the distance or orientation of motif B.
    2. **In silico tests first (cheap).** (a) ISM and *motif-planting* in random sequence backgrounds: insert the candidate motif into hundreds of backgrounds; the predicted effect distribution tells if the effect is context-dependent. (b) For syntax, plant pairs at all distances and orientations; plot predicted cooperativity against distance, and compare to a *shuffled-weights* model for artifacts. (c) *Baselines*: do the patterns exist in a simple linear model on $k$-mers? A discovery that a $k$-mer regression also finds is not a deep-learning discovery.
    3. **Experiments.** An MPRA with (i) endogenous genomic instances of the motif with the motif intact versus disrupted; (ii) synthetic sequences from step 2 at graded spacing. Pre-specify the effect-size threshold. Include decoys, motifs the model did *not* highlight.
    4. **Biology of the claim.** Identify the factor: a candidate TF from the motif's similarity to known families, from expression in the cell type, and from a binding assay (a CUT&RUN or a protein-binding microarray). A motif with no identified binder is a regulatory element, not yet a factor.
    5. **Decision rule.** Count the claim as discovered if the disrupted-motif MPRA effect is in the predicted direction in at least 70% of instances with FDR below 5%, and the planted-syntax curve agrees with the model's curve with a rank correlation above 0.5.
    6. **Failure analysis.** A model-only result (the model says so) remains C1 (a property of the model). A confirmed MPRA result is C2 for the element; a general syntax rule across loci and cell types is C3.

    **What is not known.** What fraction of model-derived syntax rules replicate; the published successes (BPNet) are cases where the model was designed for interpretability and the data were base-resolution. A systematic account of replication rates across model classes is missing and would be a useful meta-scientific contribution.

---

## 48.6 Researcher's Notebook

!!! notebook "Researcher's Notebook: from 'the model uses X' to evidence"
    1. **Rung**: write which rung (I1–I5) your claim sits on.
    2. **Plant first**: recover planted motifs/rules with your method on synthetic data of the same architecture.
    3. **Intervene**: ablate, erase, or patch the claimed component, with a random-direction control of the same dimension.
    4. **Check redundancy**: ablate groups, not only single units; use subspace methods.
    5. **Check distributions**: evaluate the intervention where confounds are decorrelated.
    6. **Check stability**: multiple seeds, multiple dictionary sizes.
    7. **Validate in biology**: choose the cheapest experiment that could falsify the interpretation.
    8. **Reproduce** the experiments of §48.2 with a larger CNN, a transformer, or a protein LM head, and see which conclusions survive.

    **What it teaches.** Interpretability is an experimental science of a model; the model is the system, and the interventions are the experiments.

    **An open question to carry forward.** In the toy, erasing one probe direction removed all use of motif A, while single-unit ablation removed almost none. For deep biological models, is there a *sample-efficient way to find the causal subspace* (the smallest set of directions whose erasure removes a capability) without training probes for every hypothesis in advance, and does its dimension predict robustness of the capability to fine-tuning, a quantity that matters for both discovery and safety (a capability is harder to remove, or to induce, if distributed)?

---

## 48.7 Connections

- **Backward:** CNN motifs and grammar (Chapter 10); attention (Chapter 12); representation learning (Chapter 13); attribution with ground truth and SAEs (Chapter 18); sequence-to-function models (Chapter 31); protein LMs (Chapter 34); shortcuts and shift (Chapter 45).
- **Forward:** open problems in genomes, cells, and proteins (Chapters 50–52); AI scientists that use interpretability to propose hypotheses (Chapter 54); evaluating ideas with experiments (Chapter 56).

!!! takeaways "Key takeaways"
    1. An interpretation is a hypothesis with a rung: attribution (I1), representation (I2), mechanism (I3), causal abstraction (I4), biological validation (I5).
    2. **Grade methods on planted circuits**: in a CNN with a known AND circuit and two confounds, the shortcut was visible only on decorrelated data (accuracy 0.985 vs 0.946; 0.864 for positives lacking S).
    3. **Single-unit ablation understates importance** when a feature is distributed: each A or B filter's ablation cost 0.05–0.08 accuracy, while erasing the probe *direction* for A reduced accuracy to chance (0.50) and patching showed the three A filters each carry about a third of the signal (97% together).
    4. **Decodable is not used, and used is not intended**: S and D were both decodable (AUROC 0.76); S was used only where it was informative, and D was used as a proxy for A.
    5. In this toy a sparse autoencoder did not beat raw neurons in either regime (no superposition: 0.996 vs 0.997 for A; heavy superposition: mean best-latent AUROC 0.71–0.74 vs 0.75), so SAE features need ground truth, stability, and causal tests.
    6. Biological models have been interpreted successfully where the model, data, and validation were designed together (BPNet motif syntax confirmed by experiment); SAE features in protein and genomic LMs are aligned with annotations (domains, exon boundaries, motifs) but their causal use is mostly untested [[P]].
    7. A validation protocol: planted-signal check, randomization sanity check, seed stability, necessity and sufficiency, decorrelated distributions, interchange tests, and wet-lab validation.

---

## Further reading

- Elhage, N. et al. (2022). Toy models of superposition. *Transformer Circuits Thread.* Bricken, T. et al. (2023). Towards monosemanticity: decomposing language models with dictionary learning. *Transformer Circuits Thread.* Templeton, A. et al. (2024). Scaling monosemanticity. *Transformer Circuits Thread.* Olah, C. et al. (2020). Zoom in: an introduction to circuits. *Distill.*
- Vig, J. et al. (2020). Investigating gender bias in language models using causal mediation analysis. *NeurIPS.* Meng, K., Bau, D., Andonian, A. & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS.* Heimersheim, S. & Nanda, N. (2024). How to use and interpret activation patching. *arXiv:2404.15255.* McGrath, T. et al. (2023). The Hydra effect: emergent self-repair in language model computations. *arXiv:2307.15771.* Geiger, A. et al. (2021). Causal abstractions of neural networks. *NeurIPS.*
- Hewitt, J. & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP.* Belinkov, Y. (2022). Probing classifiers: promises, shortcomings, and advances. *Comput. Linguist.* 48, 207–219. Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS.* Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML.* Shrikumar, A., Greenside, P. & Kundaje, A. (2017). Learning important features through propagating activation differences. *ICML.*
- Avsec, Ž. et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nat. Genet.* 53, 354–366. Shrikumar, A. et al. (2018). Technical note on transcription factor motif discovery from importance scores (TF-MoDISco). *arXiv:1811.00416.* Koo, P. K. & Eddy, S. R. (2019). Representation learning of genomic sequence motifs with convolutional neural networks. *PLoS Comput. Biol.* 15, e1007560. Koo, P. K. & Ploenzke, M. (2021). Improving representations of genomic sequence motifs in convolutional networks with exponential activations. *Nat. Mach. Intell.* 3, 258–266.
- Simon, E. & Zou, J. (2024). InterPLM: discovering interpretable features in protein language models via sparse autoencoders. *bioRxiv.* Adams, E. et al. (2025). Sparse autoencoders uncover biologically interpretable features in protein language model representations. *PNAS* 122. Zhang, Z. et al. (2024). Protein language models learn evolutionary statistics of interacting sequence motifs. *PNAS* 121, e2406285121. Brixi, G. et al. (2026). Genome modelling and design across all domains of life with Evo 2. *Nature.*
