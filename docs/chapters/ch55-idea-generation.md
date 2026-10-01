# Chapter 55. Idea Generation: The Ten Attacks

!!! abstract "Chapter at a glance"
    **Motivation.** Reading papers and running experiments are learnable. The step that is rarely taught is the one before them: *where does a good idea come from?* This chapter treats idea generation as a procedure with ten repeatable moves, the **Ten Attacks** previewed in Chapter 1, each with a core question, a set of sub-moves, a way to test the idea cheaply, and its characteristic failure. The attacks are not recipes that guarantee novelty; they are a way of turning a *diagnosed failure* into a *list of candidate hypotheses* faster, more systematically, and with less self-deception than waiting for inspiration.
    **Prerequisites.** Chapter 1 (Expert Chain, Four Gaps, Claim Ladder); the examples draw on Chapters 5–54 and are cross-referenced where they occur.
    **You will be able to:** (1) diagnose a problem into Four-Gap terms and choose which attacks apply; (2) run each of the ten attacks as a structured procedure and produce at least three candidate ideas per relevant attack; (3) recognize the characteristic *anti-pattern* of each attack; (4) fill in an **Attack Sheet** and attach a cheapest decisive experiment to every idea; (5) decompose a historical breakthrough into the attacks that produced it; (6) combine attacks (the products are often better than the parts) and kill weak ideas early.

---

## 55.1 What counts as an idea, and where ideas come from

A research **idea** here means a *testable proposal that changes what we believe or what we can do*: a hypothesis about why a model fails; a modification of data, representation, objective, or architecture that should remove the failure; an evaluation that would discriminate between explanations; a biological mechanism that explains a pattern. It must come with an experiment that could prove it wrong. (§55.12 gives the minimal format.)

Ideas arise from four sources:

1. **Failures.** A model that fails reveals an assumption (the cell-type head cannot extrapolate; the effect of an edit is not identified; the benchmark is saturated). *Failures are the richest source, because they tell you where information is missing* (Chapter 1's Four Gaps).
2. **Mismatches in structure.** The mathematics of the problem matches a solved problem elsewhere (Potts models for coevolution came from statistical physics; diffusion models for backbones came from image generation).
3. **New measurements or capabilities.** A new assay, a larger dataset, a new model class makes an old question answerable (genome-scale Perturb-seq; long-context sequence models).
4. **Anomalies in biology.** A consistent residual between model and data is a pointer to unmodeled mechanism.

Most weak research proposals fail one of two ways. *Solution-first*: "let us apply technique X to dataset Y" with no diagnosed failure; the idea is a hammer looking for a nail. *Gap-first but vague*: "models do not generalize"; the diagnosis does not say *which* information is missing. The attacks fix both by connecting a **specific gap** to a **specific kind of move**.

**The four gaps and the ten attacks.** The map below is a heuristic for choosing attacks. A cross indicates that the attack commonly addresses that gap; most real ideas touch two.

| Attack | G-M (measurement) | G-O (objective) | G-I (inference/causal) | G-G (generalization) |
|---|:-:|:-:|:-:|:-:|
| A1 Assumption | ✓ | | ✓ | ✓ |
| A2 Representation | ✓ | | | ✓ |
| A3 Objective | | ✓ | ✓ | |
| A4 Data | ✓ | | ✓ | ✓ |
| A5 Evaluation | | ✓ | | ✓ |
| A6 Scale | | | | ✓ |
| A7 Cross-domain transfer | ✓ | ✓ | ✓ | ✓ |
| A8 Problem reformulation | ✓ | ✓ | ✓ | |
| A9 Biological constraint | | | ✓ | ✓ |
| A10 Biological discovery | ✓ | | ✓ | |

---

## 55.2 The procedure in one page

!!! lens "Research lens: the Attack Sheet"
    1. **State the problem** in one sentence with an *operational* success criterion (a metric on named data).
    2. **State the current best approach and its score**, including the strongest classical baseline (Chapter 29) and the noise ceiling (Chapter 1).
    3. **Diagnose the failure** with the Four Gaps. Which information is missing from the *data*, the *objective*, the *causal structure*, or the *test distribution*?
    4. **List the assumptions** of the pipeline from raw measurement to metric (at least ten; include the ones that feel "obvious").
    5. **Run the attacks that match the diagnosis**, aiming at three ideas per attack. Write each as: *If [change], then [measurable consequence], because [mechanism].*
    6. **Attach the cheapest decisive experiment** to each idea (simulation first, then a re-analysis of existing data, then new data). Include the result that would kill it.
    7. **Cross the attacks.** Combine pairs (A2×A9, A3×A4, A5×A10, A7×A9) and look for ideas that touch two gaps.
    8. **Triage** (Chapter 56): information gain per unit cost, novelty, feasibility, and downstream value.

The sections below teach steps 4–7 by attack. Each follows the same pattern: *core question → sub-moves → worked mini-case from this book → test → characteristic anti-pattern.*

---

## 55.3 A1: Assumption attack

**Core question.** *Which assumption does every method in this area make, and is it necessary?*

**Sub-moves.**

1. **Enumerate the pipeline** from the biological system to the reported number, and write every assumption, including *independence* (cells, samples, positions), *stationarity* (snapshot equals steady state), *additivity* (effects add), *completeness* (the reference is the genome), *ergodicity* (a population sample reveals dynamics), and *identifiability* (the data determine the quantity).
2. **Mark each assumption** as *known true*, *approximately true*, *convenient*, or *unexamined*.
3. **For each convenient or unexamined assumption**, ask the three questions: *What if it is false? How would I know? What is the cheapest model that drops it?*
4. **Look for assumptions shared across competing methods.** Differences between methods are usually where the community has looked; the shared assumption is where nobody has.

**Worked mini-cases from this book.**

* *Cells are independent replicates* (Chapter 25): dropping it changed false-positive rates from 70% to 3% and changed what a single-cell study can claim.
* *A protein family is a sample from one equilibrium distribution* (Chapter 29): in a phylogenetically structured family, $N_\text{eff}=6$ out of 1,024 sequences gave near-chance contacts. The assumption (exchangeability) can be replaced by an explicit tree model; whether that improves coevolution inference in real families is an open question [[H]].
* *The effect of an edit is identified by reference-genome data* (Chapter 31): when two motifs co-occur, the edit effects are not identified; the data cannot support the claim.
* *Additive effects and independent sites* (Chapters 23, 29): relaxing additivity through a latent trait plus a nonlinear link (global epistasis) explained apparent epistasis in 31–58% of pairs without any specific interaction.

**Testing an assumption attack.** Build a *minimal simulation* in which the assumption is true and a variant in which it is false, and show that the standard method's score *changes in the direction the attack predicts*. If the method is insensitive, the assumption is not load-bearing (also a finding).

**Anti-pattern.** *Assumption-listing without consequences*: a paper that says "we do not assume X" and gets the same result adds nothing. The attack is only productive when the relaxation changes a conclusion.

---

## 55.4 A2: Representation attack

**Core question.** *Does the input representation discard information the task needs, or force the model to rediscover structure it could be given?*

**Sub-moves.**

1. **State what the representation keeps and discards** (Chapter 28's "information" decision). For each discarded quantity, ask whether the target depends on it: chirality, protonation state, phase, haplotype, cell position, modification, time.
2. **Change the object**: sequence $\to$ graph $\to$ set $\to$ structure $\to$ ensemble; single genome $\to$ pangenome; cell $\to$ cell in neighborhood.
3. **Change the unit**: nucleotides, codons, motifs, domains; cells, programs, states.
4. **Add the missing coordinate**: position, context, time, species, cell type, donor.
5. **Impose or relax symmetry** (Chapter 16): equivariance, invariance, relative position, permutation.

**Worked mini-cases.**

* *AlphaFold 2's pair representation.* Rather than treating a protein as a sequence with an attention map to be predicted, the **Evoformer** maintains an explicit **pair representation** (a matrix of residue–residue features) alongside the MSA representation and updates it with triangle-style operations that encode geometric consistency (Jumper et al., 2021) [[E]]. The representation made distances *first-class objects*.
* *Variant representation* (Chapter 28): scoring a single-base substitution requires a tokenization in which the edit is local: single nucleotides or non-overlapping $k$-mers, not BPE (24% token-count change).
* *Cells in context* (Chapter 25): in spatial data a cell is more informative when represented with its neighbors; the unit of analysis changes from a cell to a niche.
* *Genomes as graphs.* Pangenome graphs (Chapter 27) represent variation without reference bias; a model that consumes graphs is *an open direction* because few sequence models do [[H]].

**Testing.** *Information-sufficiency test*: add the candidate missing variable to the input (or give it as an oracle) and measure the gain. If the oracle gain is small, the representation was not the bottleneck; if large, the idea is to produce that variable (by measurement or inference).

**Anti-pattern.** *Architecture change presented as representation change.* Replacing a CNN by a transformer on the same one-hot input is not a representation attack; replacing one-hot sequence by sequence plus accessibility, or sequence plus structure, is.

---

## 55.5 A3: Objective attack

**Core question.** *Does the loss reward the capability we want, or only something correlated with it?*

**Sub-moves.**

1. **Write the loss, the data distribution, and the quantity you actually care about**; compute (or bound) their relationship. The sufficiency guarantee of Chapter 17 and the objective-gap taxonomy there give six types of mismatch.
2. **Look for shortcuts**: features that reduce the loss without the intended capability (leakage in masked LM, Chapter 28; batch signal in reconstruction).
3. **Swap in a task-aligned objective**: predict the effect of an intervention rather than the control state; rank rather than regress; predict a distribution rather than a mean; contrast across conditions; optimize likelihood ratio between variants.
4. **Add a second objective** that penalizes the shortcut or rewards a symmetry (equivariance loss, invariance to batch, consistency across augmentations).
5. **Match the objective to the decision**: if the decision is to prioritize 50 perturbations from 10,000, the objective is a top-$k$ metric, not mean-squared error across all genes.

**Worked mini-cases.**

* *Masked language modeling for proteins and genomes* is an objective chosen because the labels are free; its alignment with function is tested by zero-shot correlation with DMS (Chapter 34). The *preference* of a language model for common sequences (species bias; Gordon et al., 2025) is an objective-gap effect: likelihood measures how typical a sequence is, and typicality is only partly fitness [[S]].
* *AlphaFold 2's FAPE loss* (frame-aligned point error) is invariant to global rotation and translation but sensitive to local frames and chirality, matching what a structure model should be penalized for (Jumper et al., 2021) [[E]].
* *Perturbation prediction* (Chapters 25, 39): reconstructing expression minimizes error mostly on the thousands of genes that do not change; an objective weighted toward the *affected* genes, or one that targets the *difference* from control, is aligned with the downstream use.

**Testing.** Build a *synthetic task where the shortcut is available and the capability is measurable*, and show that the current objective takes the shortcut while the new one does not. Then check the capability on real data with an evaluation that excludes the shortcut (A5).

**Anti-pattern.** *Adding auxiliary losses without a mechanism.* An extra loss term is an idea only if you can say which gap in the objective it closes and which measurement shows that it did.

---

## 55.6 A4: Data attack

**Core question.** *Could better, differently structured, or interventional data dissolve the problem?*

**Sub-moves.**

1. **Ask what variation the data contain along the direction you care about.** If the question is "what happens when I edit this site", the data must contain edits at that site (Chapter 31's toy: a handful of single-edit experiments restored identifiability).
2. **Replace observational with interventional data**: perturbations, saturation mutagenesis, CRISPR screens, MPRAs, time-resolved labeling.
3. **Increase the information per sample**: base-resolution profiles instead of peaks (BPNet), multiplexed readouts, paired modalities, spatial position, lineage.
4. **Change what is sampled**: more donors, not more cells (Chapter 25); more families, not more sequences (Chapter 34); more species (Chapter 42); negatives, not only positives.
5. **Synthesize or distill**: self-distillation on predicted structures (AlphaFold 2 trained on its own high-confidence predictions for roughly 350,000 diverse sequences in addition to PDB data) [[E]]; simulation-based pretraining.
6. **Design the dataset for identifiability** (Chapter 46): choose perturbations to maximize information about the quantity of interest.

**Worked mini-cases.**

* *Richer supervision* (BPNet; Chapter 10): predicting base-resolution footprints, rather than peak calls, made motif syntax visible.
* *Genome-scale interventions* (Chapter 25): Perturb-seq at scale supplies the *causal* data that observational atlases cannot.
* *Experimental design as the idea* (Chapter 46): for a given cost, a design that spans co-varying features is worth more than more of the same.

**Testing.** *Learning-curve and identifiability tests*: show that the quantity of interest is unidentified (or poorly identified) from existing data (simulation), and that a small amount of the proposed data fixes it (Chapter 31's toy; Chapter 45 for shift).

**Anti-pattern.** *"Collect more data"* without specifying *which variation* is missing. More observational data along the same correlated directions will not identify an effect.

---

## 55.7 A5: Evaluation attack

**Core question.** *Does the benchmark measure what everyone believes it measures?*

**Sub-moves.**

1. **Compute the noise ceiling** and the strongest simple baseline (Chapters 1, 29). If they are close to each other or to the model, the benchmark cannot discriminate.
2. **Hunt for leakage and shortcuts**: homology (Chapters 27, 28), batch and donor, ancestry (Chapter 26), duplicates, label-generating pipelines.
3. **Check the unit of replication** (Chapter 4).
4. **Stratify** by the dimension along which the claim should hold (similarity to training, effect size, mappability, depth).
5. **Match the metric to the decision** (enrichment of true hits, not mean correlation).
6. **Design the adversarial test**: the data where the shortcut and the capability make different predictions.

**Worked mini-cases.**

* *Baselines beating foundation models* (Chapters 1, 39): the strong additive baseline for double perturbations (Ahlmann-Eltze et al., 2025) is a canonical evaluation attack, followed by arguments about whether the metrics were well calibrated (the follow-up preprint claims that with calibrated metrics deep models do beat uninformative baselines) [[S]] for the first claim; the *interpretation* is [[P]].
* *Reliability of perturbation effect vectors* (Chapter 25): a perturbation affecting 5 of 2,000 genes has reliability near zero, so all-gene correlation measures noise.
* *Molecular benchmarks* (Chapter 24): random-split BACE results sat near their noise ceiling; the cluster split revealed generalization.

**Testing.** The benchmark attack is *self-testing*: apply the diagnostics (ceiling, baseline ladder, shuffled-label, leakage test) to the published results.

**Anti-pattern.** *Evaluation attack as complaint*: "the benchmark is flawed" is not an idea unless you propose and run a better one, and show that it changes a conclusion.

---

## 55.8 A6: Scale attack

**Core question.** *What changes with more data, parameters, context, modalities, or compute, and is the changing quantity the one that limits this problem?*

**Sub-moves.**

1. **Identify the limiting resource**: data, capacity, context, optimization, or the objective. Scaling laws (Chapter 17) say loss decreases as a power of each when the others do not bind; the *exponents* differ by domain and tell you the marginal value of each axis.
2. **Scale the right axis.** Context length (kb to Mb), *diversity* of the training data (species, cell types, donors) versus its *volume*, modality count, sequence depth per cell, number of perturbations.
3. **Search for emergence**: capabilities that appear only above a size (and verify that they are not artifacts of the metric).
4. **Search for saturation**: where further scale cannot help because the information is absent (the noise ceiling, an unidentified effect).
5. **Combine with cheaper alternatives**: distillation, sparse models, retrieval (Chapter 47).

**Worked mini-cases.**

* *Context* (Chapters 11, 31): moving from 131 kb (Basenji) to 196 kb (Enformer), 524 kb (Borzoi), and 1 Mb (AlphaGenome) is a scale attack on a documented bottleneck (distal regulatory elements); whether the long context is *used* is separately testable (Chapter 31) [[S]].
* *Diversity versus volume* (Chapters 34, 42): protein language models gain from sequences across many families; whether adding more of the same families helps is less clear.
* *Diminishing returns by construction* (Chapter 24): adding molecules to a benchmark already near its noise ceiling will not change rankings.

**Testing.** *Scaling curves* with at least four points per axis, with confidence intervals and the unscaled baseline; fit exponents and extrapolate to the target accuracy, then ask whether the extrapolation is believable given the ceiling.

**Anti-pattern.** *Scale-as-argument*: "a larger model should do better" without identifying why the current model is capacity-limited. The most common error is scaling a model whose bottleneck is the *data* or the *objective*.

---

## 55.9 A7: Cross-domain transfer

**Core question.** *Which idea from another field has the same mathematical structure as this problem?*

**Sub-moves.**

1. **Abstract the problem**: write it as one of a small set of *structures*: a sequence with long-range dependencies; a set; a graph; a field on a manifold; a latent-variable model; a dynamical system observed in snapshots; an inverse problem; a compositional (simplex) object; a mixture; a causal graph with latent confounders; a decision under uncertainty.
2. **Search for solved instances** in other fields: physics, economics, linguistics, vision, signal processing, control, genetics, ecology.
3. **Check what the transfer assumes** and whether biology satisfies it (A1 applied to the borrowed method).
4. **Transfer the *diagnostic*, not only the method**: a fluid-dynamics intuition about conditioning, an econometrics test for instruments, a linguistics test for compositionality.

**Worked mini-cases (structural rhymes in this book).**

| Biological problem | Same structure elsewhere | Transferred idea |
|---|---|---|
| Residue coevolution | Ising/Potts models in statistical physics | Direct-coupling analysis (Chapter 29) |
| Rate of a TF binding a site | Chemical equilibrium; logistic regression | Thermodynamic sequence models (Chapter 22) |
| Backbone generation | Image diffusion models | RFdiffusion-style design (Chapters 15, 36) |
| Long genomic context | Long-sequence language modeling | State-space and Hyena operators (Chapters 11, 32) |
| Cell-state trajectories from snapshots | Optimal transport; probability-flow ODEs | Cell fate maps; flow matching for perturbation (Chapters 15, 39) |
| LMM for relatedness | Animal breeding; Gaussian processes | Mixed models; REML (Chapter 26) |
| Instrument for causality | Econometrics | Mendelian randomization (Chapter 26) |
| Batch effects | Domain adaptation; causal representation | Conditional VAEs; invariance penalties (Chapters 13, 30) |

**Testing.** *Toy-first*: reproduce the transferred method on a synthetic problem with the same structure and a known answer before touching real data (as Chapter 29 did for DCA). Then ask *what biology violates* in the toy.

**Anti-pattern.** *Fashionable transfer*: importing a model because it is in the news. A good transfer starts with the *structure*, so the question is "which field has seen this structure?" and not "where can I use this method?".

---

## 55.10 A8: Problem reformulation

**Core question.** *Can the problem be restated so that a different class of methods applies, or so that the hard part becomes easy?*

**Sub-moves.** Restate the problem as each of: **prediction** (regression, classification), **generation** (sample from a conditional distribution), **inference** (posterior over a hidden variable), **retrieval** (find the nearest known case), **search/optimization** (find an object maximizing a property), **control** (choose interventions), **decision** (rank actions by expected value), **testing** (hypothesis test with a calibrated null), **explanation** (find a mechanism). Then ask which restatement makes the missing information *measurable*.

**Worked mini-cases.**

* *Variant effect as likelihood ratio*: instead of predicting a phenotype from a variant, ask how much a variant changes a model's sequence likelihood (zero-shot; Chapters 32, 34).
* *Fine-mapping as Bayesian variable selection* (Chapter 26): replacing "find the best SNP" by "compute a posterior over causal sets" changed the output from a point to a credible set.
* *Perturbation response as a conditional distribution* rather than a mean (Chapter 39): enables questions about cell-to-cell heterogeneity and distributional metrics.
* *Drug discovery as retrieval and generation* (Chapters 24, 37): ultra-large make-on-demand libraries turned design into search.
* *Protein folding as inverse problem on distance constraints*: the formulation that led from coevolution to structure (Chapters 29, 35).

**Testing.** The reformulation is testable by a *toy equivalence*: on a problem with a known answer, show that the new formulation recovers it with less data or fewer assumptions, and state precisely what it cannot do.

**Anti-pattern.** *Relabeling.* Calling "regression" "generation" without a changed objective or a changed evaluation changes nothing.

---

## 55.11 A9: Biological-constraint attack

**Core question.** *Can known biology be built into the model as an inductive bias or a hard constraint?*

**Sub-moves.**

1. **List the true constraints**: conservation laws and mass balance; physical symmetries (rotation, reflection, reverse complement); the genetic code and reading frame; thermodynamic relations (sigmoids from two-state equilibria); sparsity of regulatory networks; modularity of pathways; causality in time; monotonic dose–response; bounded noise from counting statistics.
2. **Decide the form**: *hard* (architecture enforces it), *soft* (a penalty), *data-level* (augmentation), or *post hoc* (calibration or projection).
3. **Check that the constraint is actually true** in the regime of use (reverse-complement symmetry breaks in genes: Chapter 28).
4. **Look for constraints that reduce the parameters** by orders of magnitude (the NB likelihood replaces zero inflation; a thermodynamic layer replaces a free sigmoid).

**Worked mini-cases.**

* *SE(3)-equivariant structure modules* (Chapter 16; IPA in AlphaFold 2): the model cannot be wrong about global orientation, so capacity is spent on relations.
* *Count likelihoods for single-cell data* (Chapters 25, 30): the thinned-NB measurement model is a biological–statistical constraint that explains the zeros.
* *Thermodynamic output layers* (Chapter 22): a binding-energy matrix feeding a sigmoid imposes saturation and is interpretable.
* *Global-epistasis link* (Chapter 23): a latent additive trait with a monotone nonlinear readout recovers $R^2=0.84$ where an additive model failed ($-19.8$).
* *Codon structure* (Chapter 28): the frame is worth twice as much as all local context; a frame-aware model or tokenization imposes biology (A2×A9).

**Testing.** *Ablation under shift*: the constraint matters most where data are scarce or the test distribution differs; compare constrained and unconstrained models on a structured split and in a small-data regime. (Chapter 24's descriptor model degraded 21% under scaffold shift versus 62% for the fingerprint model.)

**Anti-pattern.** *Constraint worship*: hard-coding a constraint that is only approximately true can cap accuracy at the constraint's error. Test the constraint, and where it fails, let the failure be a discovery (A10).

---

## 55.12 A10: Biological-discovery attack

**Core question.** *What unknown biological mechanism could explain a persistent modeling failure?*

This is the attack most often skipped, and the one that connects the machine-learning loop back to biology. A systematic residual between model and data is a *measurement of what the model's assumptions omit*; sometimes that is a data problem, sometimes it is biology.

**Sub-moves.**

1. **Collect the errors**: where is the model *consistently* wrong? Cluster by gene, region, cell state, protein family, or condition.
2. **Annotate the clusters with biology**: function, chromatin state, evolutionary rate, tissue, pathway, structural class.
3. **Rule out the instrument** (A5): mappability, label noise, leakage, depth (Chapter 27).
4. **Propose mechanisms** that would produce the observed pattern: a feedback loop, a context-dependent enhancer, a conformational switch, a post-translational modification, a missing cell state.
5. **Design the experiment** that would confirm the mechanism and *the model change* that would encode it.
6. **Invert the logic**: where a model succeeds *unexpectedly* (an unsupervised model that recovers exon boundaries or contacts), ask what that says about the information content of evolutionary data (Evo 2's reported learning of exon–intron boundaries and TF binding sites; Chapter 32) [[S]].

**Worked mini-cases.**

* *Distal enhancers ignored* (Chapter 31): models such as Enformer get most predictive signal from promoter-proximal sequence (Karollus et al., 2023) [[S]]. Possible mechanisms: true distal effects are smaller or more context-specific than assumed; training data do not identify them (the toy); the architecture does not use the context. These three have different experimental signatures: a CRISPRi enhancer screen, an MPRA design, and an in-silico test of attention. The *discovery* is whichever survives.
* *Phylogenetic structure of coevolution* (Chapter 29): the finding that near-chance contact prediction arises at low $N_\text{eff}$ despite thousands of sequences is a statement about *evolutionary sampling*, motivating new tools.
* *The cell-type head* (Chapter 31): the inability of a trained model to extrapolate to a new cell type reveals that what is learned is a lookup of cell-specific tracks; the biological question is which sequence features are cell-type-invariant (promoters) versus specific (enhancers).

**Testing.** The test is an *experiment on the biology*, not the model: a perturbation that distinguishes the mechanisms.

**Anti-pattern.** *Just-so stories*: a mechanism that explains the error post hoc without a distinguishing prediction.

---

## 55.13 Combining attacks

Single attacks yield incremental ideas; **products of attacks** tend to yield the larger ones because they address two gaps.

| Product | What it means | Example |
|---|---|---|
| A2 × A9 | Representation that embeds a constraint | Frame-aware codon tokens; $SE(3)$ frames; strand-aware inputs |
| A3 × A4 | Objective with data that makes it identifiable | A loss on edit effects *plus* single-edit (MPRA-like) data (Chapter 31) |
| A5 × A10 | An evaluation that doubles as a discovery tool | Residuals by mappability/family reveal unmodeled biology |
| A7 × A9 | Borrowed method with a biological constraint | Diffusion on $SE(3)$ frames; flow matching with mass conservation |
| A1 × A4 | Dropping an assumption by measuring what it hid | Time-resolved labeling to drop the stationarity assumption |
| A6 × A5 | Scaling with a trustworthy test | Scaling curves on family-split benchmarks, not random splits |
| A8 × A4 | Reformulation that creates a measurable target | Perturbation response as a distribution, supported by single-cell perturbation datasets |

A good practice is to write the $10\times10$ matrix of products for your problem and fill the cells for which a concrete experiment comes to mind.

---

## 55.14 Worked research examples

!!! example "Worked Research Example 55.1: An Attack Sheet for cross-individual regulatory variant prediction"
    **Problem (one sentence).** Predict, from DNA sequence, how a person's cis-regulatory variants change their gene expression relative to others; measured by Spearman correlation across individuals, per gene, in held-out genes, and sign accuracy on fine-mapped eQTLs.

    **Current best and its score.** Sequence-to-function models (Enformer, Borzoi, AlphaGenome-type; Chapter 31) trained on reference-genome tracks explain cross-gene variation well, but four such models evaluated on personal genomes of 421 individuals gave per-gene cross-individual correlations centered near zero and often the wrong direction of effect, while per-gene regularized regression on nearby variants (PrediXcan-type) explained more (Huang et al., 2023) [[S]]. Strongest classical baseline: LMM/ridge on cis variants (Chapter 26).

    **Four-Gap diagnosis.** G-O (the objective: cross-gene reference-genome tracks, not within-gene variation), G-I (variant effects are counterfactuals not identified by training on reference sequences, Chapter 31), G-M (LD-tagged eQTL labels, Chapter 26), G-G (ancestry, Chapter 45).

    **Assumptions (partial).** (1) A model fit to between-gene variation also gets within-gene variation right. (2) Variant effects are additive across variants. (3) Reference sequence is representative. (4) eQTL labels are causal. (5) Tissue/cell type matches. (6) Variants act through the same features that explain between-gene differences. (7) Expression is measured without confounding by batch/ancestry. (8) The test genes are not in the training distribution (homology). (9) Effects are context-independent. (10) The sign of an effect is learnable from local sequence.

    **Attacks and ideas.**

    | Attack | Idea (*if…, then…, because…*) |
    |---|---|
    | A1 | *If* the additive-across-variants assumption is dropped by scoring *haplotypes* rather than single variants, *then* cross-individual correlation for genes with multiple eQTLs improves, *because* common variants in LD act jointly. Test: haplotype-aware scores on Geuvadis-style data. |
    | A2 | *If* phased personal sequences with the variant combinations are the input (rather than reference plus one edit), *then* the model captures epistasis. |
    | A3 | *If* the loss includes a within-gene term (predict differences between alleles, contrastive across haplotypes) *then* the sensitivity to variants is trained directly, *because* the Jacobian along edit directions is otherwise unconstrained (Chapter 31). |
    | A4 | *If* the training set includes saturation-mutagenesis/MPRA effects or allele-specific expression, *then* sign accuracy rises, *because* the data identify counterfactual effects. The *cheapest decisive experiment*: fine-tune on an MPRA set and test on held-out fine-mapped eQTLs. |
    | A5 | *If* the evaluation is stratified by distance to TSS, LD with the lead variant, and fine-mapping PIP, *then* the apparent accuracy changes, *because* LD tags inflate sign accuracy. Kill: no stratum effect. |
    | A6 | *If* the context is extended to megabases, *then* distal-eQTL accuracy improves — only if the model actually uses the context (Karollus et al., 2023; test by in-silico enhancer deletion). |
    | A7 | *If* the problem is treated like *domain adaptation* (reference genome as source, personal genomes as target), *then* adaptation methods (e.g., fine-tuning on a few individuals) help, *because* the shift is structured (Chapter 45). |
    | A8 | *If* the task is reformulated as *predict the allelic imbalance* (ref/alt ratio in the same individual, same cells), *then* confounders are cancelled by design (the allele-specific expression readout is within-sample). |
    | A9 | *If* a thermodynamic or additive latent layer is used with a monotone link (global epistasis, Chapter 23) *then* extrapolation to multi-variant haplotypes improves. |
    | A10 | *If* the systematic errors (genes for which sign is wrong) cluster by enhancer-promoter distance or by chromatin state, *then* the mechanism is context-dependent regulation not captured by reference-trained models; experiment: CRISPRi of the clustered enhancers. |

    **Crosses.** A3×A4 (within-gene loss + allele-specific data), A5×A10 (stratified residuals as discovery), A8×A4 (allelic imbalance as the target).

    **Triage.** Highest information per cost: the A5 re-analysis (days; existing data), then A8 (use existing ASE data), then A4 fine-tuning. A6 is expensive and may fail; A10's experiment is expensive but would be a discovery.

    **What it teaches.** Ten attacks on one failure produce a ranked portfolio *in an afternoon*; most are cheap re-analyses.

!!! example "Worked Research Example 55.2: Decomposing AlphaFold 2 into attacks"
    **Situation.** AlphaFold 2 (Jumper et al., 2021; CASP14) is often described as "a better architecture". Reconstructing it as a sequence of attacks shows why it was more than that.

    **Reasoning.**

    1. *Problem and failure of prior work (L1–L6).* Coevolution-based methods (Chapter 29) predicted contacts, then separate modules built structures from contacts; errors compounded; the pipeline was not end-to-end.
    2. **A8 (reformulation):** predict the 3-D structure *directly and end-to-end* from the MSA and templates, with a differentiable path from sequence to coordinates.
    3. **A2 (representation):** the Evoformer keeps an MSA representation and a pair representation and exchanges information between them, with triangle updates enforcing geometric consistency of pairs.
    4. **A9 (biological/physical constraint):** the structure module represents each residue as a rigid *frame* and uses invariant point attention, equivariant to global rotations and translations; the loss (FAPE) is frame-aligned and respects chirality (Chapter 16).
    5. **A3 (objective):** auxiliary losses (distogram, masked MSA prediction) and a confidence head (pLDDT) trained to predict its own accuracy; the model *reports when to trust it*.
    6. **A4 (data):** self-distillation (training on confidently predicted structures for hundreds of thousands of sequences) enlarged the effective training set beyond the PDB.
    7. **A6 (scale):** a deep iterative architecture with recycling (repeated refinement), trained at substantial compute.
    8. **A5 (evaluation):** CASP's blind, time-split protocol made the claim a C2/C4 claim in the Claim Ladder, not a C1 claim (Chapter 43).

    **Expert analysis.** Seven of the ten attacks appear, in combination, and none alone would have produced the result. Equally informative are the attacks *not* used: A7 (transfer) from language models appears later (ESMFold, Chapter 34), and A10 (discovery) came *afterwards*, as errors were mapped onto disorder and dynamics (Chapters 35, 52).

---

## 55.15 Researcher's Notebook

!!! notebook "Researcher's Notebook: fill an Attack Sheet in two hours"
    **Setting.** Choose a problem you care about (from Part VII or IX) and a current model for it.

    1. **Hour 1: diagnose.** Write the problem sentence and the metric; find the noise ceiling and the strongest classical baseline; list at least ten assumptions; assign each of the Four Gaps a one-line diagnosis.
    2. **Hour 2: attack.** For each attack, produce *three* candidate ideas, each in the *If/then/because* format with the cheapest decisive experiment and the kill condition. Do not evaluate while generating; evaluation comes in Chapter 56.
    3. **Cross.** Fill five cells of the product table.
    4. **Count.** If fewer than half the attacks produced ideas you believe in, the diagnosis was too vague: return to step 1.

    **Habits that make the attacks work.**

    * Alternate *divergent* and *convergent* steps: generate without judging; then judge without generating.
    * Write the *failure* of each idea first: what would make it worthless?
    * Prefer ideas whose cheapest test is a simulation or re-analysis.
    * Keep a graveyard of killed ideas with the reason (A5-type and A1-type reasons recur).

    **What it teaches.** Creativity in research is mostly *disciplined breadth*. The procedure's value is not the ideas it produces but the ideas it prevents you from missing.

    **An open question to carry forward.** Which attacks have the highest historical hit rate in AI for biology? Build a dataset of 30 major methods papers from 2015–2026, label each with the attacks it used (as in Example 55.2), and ask which attacks (or products) correlate with lasting impact and which with retractions or failed replications. Be careful about survivorship bias (A5 applied to the question itself).

---

## 55.16 Connections

- **Backward:** the Four Gaps and Claim Ladder (Chapter 1); sufficiency and the objective-gap taxonomy (Chapter 17); structural rhymes (every chapter); the baseline ladder (Chapter 29); counterfactual identifiability (Chapter 31).
- **Forward:** evaluating ideas (Chapter 56); reading papers with the attacks in mind (Chapter 57); reasoning without known answers (Chapter 58); from idea to publication (Chapter 59).

!!! takeaways "Key takeaways"
    1. Ideas come from failures, structural mismatches, new measurements, and biological anomalies; the **Ten Attacks** convert a diagnosed failure into a list of candidate hypotheses.
    2. **A1 Assumption**: list every assumption, mark the unexamined ones, and build the minimal simulation in which dropping it changes a conclusion.
    3. **A2 Representation** and **A9 Biological constraint** supply information or structure; test with an oracle or with ablation under shift.
    4. **A3 Objective** and **A4 Data** make the target identifiable; counterfactual quantities need interventional variation (a handful of single-edit experiments restored identifiability in the Chapter 31 toy).
    5. **A5 Evaluation** is self-testing (ceiling, baseline, leakage, unit of replication) and often the cheapest, highest-value attack.
    6. **A6 Scale** is only an idea when the limiting resource is identified; **A7 Cross-domain** starts from *structure*, not from methods; **A8 Reformulation** must change the objective or the evaluation; **A10 Discovery** requires an experiment on the biology.
    7. **Products of attacks** (A2×A9, A3×A4, A5×A10) address two gaps and give larger ideas.
    8. Every idea carries an *If/then/because*, a cheapest decisive experiment, and a kill condition. A two-hour Attack Sheet usually yields a ranked portfolio dominated by cheap re-analyses.

---

## Further reading

- Platt, J. R. (1964). Strong inference. *Science* 146, 347–353. Popper, K. (1959). *The Logic of Scientific Discovery.* Hamming, R. (1986). You and your research (talk transcript). Kuhn, T. S. (1962). *The Structure of Scientific Revolutions.*
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589. Avsec, Ž. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nat. Methods* 18, 1196–1203. Gordon, C., Lu, A. X. & Abbeel, P. (2025). Protein language model fitness is a matter of preference. *ICLR*.
- Karollus, A., Mauermeier, T. & Gagneur, J. (2023). Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers. *Genome Biol.* 24, 56. Huang, A. C. et al. (2023). Personal transcriptome variation is poorly explained by current genomic deep learning models. *Nat. Genet.* 55, 2056–2059. Ahlmann-Eltze, C., Huber, W. & Anders, S. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat. Methods* 22, 1657–1661.
- Sutton, R. (2019). The bitter lesson (essay). Kaplan, J. et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*. Hoffmann, J. et al. (2022). Training compute-optimal large language models. *NeurIPS*.
