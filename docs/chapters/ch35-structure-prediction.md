# Chapter 35. Structure Prediction: From Coevolution to Co-folding

!!! abstract "Chapter at a glance"
    **Motivation.** Predicting a protein's three-dimensional structure from its sequence was the field's defining problem for fifty years, and its solution (AlphaFold2, 2020–21) is the most consequential demonstration so far that machine learning can answer a hard question in molecular biology. The second act (AlphaFold3 and open co-folding models, 2024–26) extends the problem to complexes with nucleic acids, ligands, and ions, where it meets drug discovery and a harder evaluation problem. This chapter builds the stack from its parts: what information in a structure is *recoverable* from coevolution; what the Evoformer, the structure module, and the diffusion module compute; why a loss that is invariant to rotations is not enough and what that does to chirality; how predicted confidence is calibrated; and where the field's own evidence says the models do not yet go (memorization of ligand poses, ensembles, dynamics). A real-protein experiment measures how much contact information a fold needs, and exposes a cliff: *precision*, not quantity, decides whether a set of contacts determines a fold.
    **Prerequisites.** Chapters 12, 16, 22, 29, 34, 43, 45.
    **You will be able to:** (1) state what a distogram, a contact map, FAPE, pLDDT and PAE are, and compute TM-score and lDDT; (2) explain why coevolution determines contacts and why contacts determine a fold only when they are accurate; (3) describe the Evoformer, invariant point attention, and the AF3 diffusion architecture, and the role of each representation; (4) say what is evidence and what is hypothesis in the memorization-versus-physics debate for co-folding; (5) list the open problems (ensembles, conformational switching, binding affinity, disorder) and the experiments that would address them; (6) design a defensible evaluation for a new structure model.

---

## 35.0 The problem and its history

**Anfinsen's postulate** (1960s): the sequence of a protein contains the information that determines its native structure. [[E]] for most single-domain globular proteins under physiological conditions, with caveats that this chapter treats as open problems: proteins with multiple stable states, disordered regions, and chaperone-assisted folding. **Levinthal's paradox** (a protein cannot find its structure by random search) tells us that folding is guided by a funnel-shaped energy landscape; it does not tell us how to *compute* the structure.

Three eras:

1. **Physics and fragments** (1970s–2000s). Energy functions plus conformational search (molecular dynamics, Monte-Carlo with fragment assembly as in Rosetta). Accuracy was good for small proteins and poor beyond.
2. **Coevolution** (2011–2018). The multiple sequence alignment (MSA) of a protein family contains the signature of structural contacts: pairs of residues that touch coevolve. Direct coupling analysis (Chapter 29) inferred contacts from the couplings of a Potts model; deep convolutional networks (RaptorX, trRosetta, AlphaFold1) turned contact and distance maps into structures (CASP13, 2018).
3. **End-to-end** (2020–). AlphaFold2 (Jumper et al., *Nature* 2021; CASP14, 2020) predicts atomic coordinates directly, with median accuracy on CASP14 targets near experimental resolution for many proteins. RoseTTAFold (Baek et al., *Science* 2021) followed independently. Language-model-based ESMFold (Lin et al., *Science* 2023) predicts structures from a *single* sequence, without an MSA, at much lower cost and lower accuracy. **AlphaFold3** (Abramson et al., *Nature* 2024) generalizes to biomolecular complexes with a diffusion module (§35.4).

!!! bio "Biology for modeling: what a structure is, and what the databases hold"
    **What is it?** A protein structure is a set of 3-D atomic coordinates, typically one *snapshot* from a crystal, a cryo-EM map, or an NMR ensemble. **What information does it contain?** Fold, active-site geometry, interfaces; it does not contain the *distribution* over conformations, the free-energy differences between states, or the kinetics. **How is it generated and measured?** X-ray crystallography requires crystals (biased toward rigid, soluble proteins; ligand and crystal-contact artifacts); cryo-EM needs particles (and tolerates larger, flexible assemblies, with local resolution varying); NMR gives distance restraints in solution for small proteins. **How is it represented?** Backbone frames (N, C$\alpha$, C) per residue and side-chain torsions; or all-atom coordinates. **What variation exists?** The PDB (more than 230,000 entries by 2025) is biased toward well-studied families, toward ligand-bound states of drug targets, toward proteins that crystallize; the AlphaFold Database (Varadi et al., 2022 and updates) provides predicted structures for over 200 million sequences. **What can ML not observe?** A model trained on PDB snapshots has learned the most *populated and most crystallizable* states, and has never been trained on a rate or a free energy.

---

## 35.1 What information determines a fold? An experiment on a real protein

Before dissecting a model, ask how much information a structure *needs*. Suppose a model predicts, for a protein of length $L$, a set of long-range contacts ($C_\alpha$–$C_\alpha$ distance below 8 Å, sequence separation at least 6) with a given *precision* (fraction of predicted contacts that are true). What accuracy of structure can be recovered from the contacts alone, plus perfect local geometry?

**Setup.** The protein is the C-terminal domain of the HIV-1 capsid protein (PDB 1A8O, chain A, 66 residues, 47 long-range contacts: 0.71 per residue). The reconstruction is *classical distance geometry*: (i) upper bounds from chain connectivity (3.8 Å between consecutive $C_\alpha$), local geometry ("secondary-structure restraints": the native $C_\alpha$ distances at separations 2, 3, 4, i.e. a perfect local-structure predictor), and the predicted contacts (8 Å); (ii) bound smoothing by shortest paths (Floyd–Warshall, triangle inequality); (iii) a metric-matrix embedding (classical MDS) of the smoothed distances to 3-D; (iv) gradient refinement of a restraint energy with a generic compactness prior (radius of gyration at most 1.2 × the value of a globular chain of this length, not tuned to this protein). Quality is measured by RMSD, TM-score ($\text{TM}=\frac1L\sum_i \frac{1}{1+(d_i/d_0)^2}$, $d_0=1.24\sqrt[3]{L-15}-1.8$ Å, after the superposition that maximizes it) and lDDT-C$\alpha$ (the fraction of pairwise distances within 0.5, 1, 2, 4 Å of the native ones, averaged: superposition-free and local).

```python
--8<-- "code/ch35_structure.py"
```

```text
66 residues; 47 long-range contacts (C-alpha distance < 8 A, separation >= 6); 0.71 per residue
native radius of gyration 11.2 A; compactness prior caps it at 13.0 A

secondary-structure   long-range contacts (precision)   far pairs   RMSD (A)   TM-score   lDDT-Calpha   (mean of 3 reconstructions)
                 no                            0 (1.0)           0      33.17      0.038      0.185
                yes                            0 (1.0)           0      16.29      0.073      0.366
                yes                           24 (1.0)           0       1.29      0.851      0.850
                yes                           47 (1.0)           0       1.14      0.892      0.866
                yes                           47 (0.7)           0       6.03      0.302      0.564
                yes                           47 (0.5)           0       7.84      0.237      0.493
                yes                           47 (0.3)           0       9.43      0.164      0.500
                yes                           47 (1.0)         100       1.14      0.892      0.866
                yes   full binary contact map (all pairs)           0       0.82      0.941      0.918

handedness: 8 independent reconstructions from the full distance information: 2 have the native handedness, 6 are mirror images (median RMSD after choosing the better hand 0.62 A)
```

**Reading the table.**

1. **No global information, no fold.** With local geometry alone (second row), the best the optimizer can do is a compact helical chain with a random arrangement: TM-score 0.07 (unrelated structures of this size typically score about 0.17), lDDT 0.37.
2. **A few accurate contacts determine the fold.** With 24 true long-range contacts (about one per three residues), the reconstruction has RMSD 1.3 Å and TM 0.85; with all 47, TM 0.89. The full binary contact map (every pair, contact and non-contact) gives TM 0.94 (RMSD 0.8 Å). This is the quantitative basis of the coevolution era: *accurate* contacts at the density of about $L/3$ determine the topology of small domains.
3. **Precision is a cliff, not a slope.** With 47 contacts of which 70% are true, TM drops to 0.30; at 50%, 0.24; at 30%, 0.16. Every false contact is a hard upper bound that tries to bring together residues that are far apart in the true fold, and the refinement then distorts the rest to satisfy it. **The number of contacts matters little; the precision of the set matters decisively.** The consequence for model building is that *calibrated confidence per contact* (so that the downstream solver can discard the low-confidence ones) is worth more than a higher recall of uncertain ones, and it explains why AF2's confidence heads, not only its accuracy, are central to its use.
4. **Negative information added nothing here.** One hundred "far pair" lower bounds (pairs more than 14 Å apart) left the result unchanged to three digits, because the contact set already pinned the fold and the far pairs were satisfied. In a regime with fewer contacts they would matter; here, a null result.
5. **Handedness is not determined by distances.** Distances are identical for a structure and its mirror image. Eight independent reconstructions from the *full* distance information had 2 native-handed and 6 mirror-image structures, which is what a coin flip does (the median RMSD to the better hand is 0.62 Å). A real model must break the symmetry by an objective or input that is *not* reflection-invariant: AF2's FAPE loss uses local frames and distinguishes chirality; AF3 dropped FAPE and residue frames for atom-level diffusion and relies on the training signal and on reference-conformer input features, and stereochemical errors in its outputs are one of the reasons predictions are checked for validity. This is the equivariance-versus-invariance point of Chapter 16 in its simplest form.
6. **Information versus inference.** An earlier version of this experiment used gradient descent from random starts and reached only TM 0.19 *even with all 47 true contacts and perfect local geometry*, a result that would have wrongly suggested that contacts are insufficient. The restraints were sufficient; the *optimizer* failed in a landscape with many tangled local minima. A distance-geometry initialization, which uses the global information (shortest-path-smoothed bounds) before refinement, fixed it. **When a model fails, separate "the information is not there" from "the inference does not find it"** (Chapter 1's information-used and information-ignored). The same distinction runs through every structure model: the Evoformer can be viewed as a learned, iterative bound-smoothing and embedding procedure for exactly this reason (§35.3).

!!! lens "Research lens: assumptions of this experiment"
    A *perfect* local-structure predictor, a protein with a compact helical fold (long-range contacts per residue near 0.7), restraints that are all-or-nothing upper bounds with errors drawn at random among non-contacts (real errors are *structured*: false contacts are usually near true ones, in the same helix packing), and three reconstructions per condition. Each assumption makes the problem easier than the real one; the qualitative findings (global information is necessary, accurate contacts suffice, precision beats quantity, handedness needs a non-invariant term) are the part to carry forward.

---

## 35.2 Metrics: how to say a structure is right

| Metric | Defined as | Properties and pitfalls |
|---|---|---|
| **RMSD** | root-mean-square deviation after optimal superposition | Dominated by the worst region; length-dependent; a single misplaced domain ruins it |
| **GDT_TS** | mean fraction of residues within 1, 2, 4, 8 Å after superposition | CASP's historical headline; less sensitive to outliers |
| **TM-score** | $\frac1L\sum 1/(1+(d_i/d_0)^2)$ with length-dependent $d_0$ | Scale-free (0.5 or more indicates the same fold, below about 0.2 unrelated); sensitive to global topology |
| **lDDT** | fraction of local interatomic distances preserved | No superposition; robust to domain motion; used for AF2's pLDDT |
| **DockQ** | composite of interface contacts, ligand and interface RMSD | Complexes |
| **Ligand pose RMSD < 2 Å** (plus *PoseBusters* checks) | success rate of ligand placement | Success of placement is not validity: steric clashes, wrong chirality, strained geometry must be checked |

Which is right depends on the question. A medicinal chemist wants the *pocket* to be right to 1 Å; a fold-recognition user wants TM; a geneticist ranking missense variants wants local reliability (pLDDT) at the variant site. A single number for a single protein hides the distribution across the proteome: report per-target distributions, stratified by similarity to training (Chapter 43).

---

## 35.3 AlphaFold2: the architecture as learned bound-smoothing

**Inputs.** The target sequence, an MSA of homologs (up to thousands of rows), and optional structural templates. **Two representations** are maintained throughout: an *MSA representation* $m\in\mathbb R^{N_\text{seq}\times L\times c_m}$ (per sequence and residue) and a *pair representation* $z\in\mathbb R^{L\times L\times c_z}$ (per residue pair).

**Evoformer** (48 blocks). Two kinds of update exchange information:

- *MSA stack*: row-wise gated self-attention *biased by the pair representation* (which residues of this sequence attend to which, informed by what is already believed about the pair), column-wise attention (the same position across sequences; the coevolutionary signal), transition layers.
- *Communication*: an *outer-product mean* writes MSA information into the pair representation (it computes, per pair of columns, the average over sequences of the outer product of their embeddings: a learned, high-dimensional analog of the covariance that DCA estimates, Chapter 29); the pair representation biases MSA attention in return.
- *Pair stack*: **triangle multiplicative updates** and **triangle attention**, updating $z_{ij}$ from $z_{ik}$ and $z_{kj}$. They enforce *consistency of distance-like quantities* under the triangle inequality. This is the learned analog of the shortest-path bound smoothing of §35.1, iterated and made probabilistic (an interpretation of the mechanism, [[P]], not a theorem about what the trained network does).

```python
# Triangle multiplicative update ("outgoing"): shapes in comments.   z: (L, L, c)
def triangle_out(z, Wa, Wb, Wg, Wo):
    a = sigmoid(z @ Wg_a) * (z @ Wa)              # (L, L, c) left edge  z_ik projected
    b = sigmoid(z @ Wg_b) * (z @ Wb)              # (L, L, c) right edge z_jk projected
    upd = einsum("ikc,jkc->ijc", a, b)            # sum over the third node k: O(L^3 c) time
    return z + sigmoid(z @ Wg) * (layer_norm(upd) @ Wo)
```

**Structure module** (8 blocks, weights shared). Starting from every residue at the origin in an identity frame ("black hole" initialization), it updates backbone *frames* $T_i=(R_i,t_i)\in SE(3)$ with **invariant point attention** (IPA): attention logits combine the usual query-key term, a pair-bias term from $z$, and a term from the *distances between 3-D points* (predicted in each residue's local frame and mapped to the global frame), which are invariant to a global rotation and translation. The output is a set of frames and side-chain torsion angles, thus all-atom coordinates.

**Loss.** The main loss is the **frame-aligned point error (FAPE)**: for every pair of a frame $T_i$ and a point $x_j$, compare the position of $x_j$ expressed in frame $i$ in the prediction and in the ground truth,
$$
\mathcal L_\text{FAPE}=\frac1{|F||X|}\sum_{i\in F}\sum_{j\in X}\min\!\big(d_\text{clamp},\ \|T_i^{-1}x_j-T_i^{\star-1}x_j^\star\|\big),
$$
which is invariant to global rigid motions but *not* to reflections: a mirrored structure has different local-frame coordinates, which is how AF2 acquires chirality (§35.1, point 5). Auxiliary losses include a **distogram** (a classification of pairwise distance bins from $z$), a masked-MSA loss (the BERT-style objective of Chapter 13, applied to the MSA), and confidence losses.

**Recycling.** The outputs (pair representation, a distance map from the predicted structure, the first MSA row) are fed back as inputs for 3 more passes, an iterative refinement that costs only compute.

**Confidence.** *pLDDT* is a per-residue prediction of the lDDT the model expects to achieve; *PAE* (predicted aligned error) is the expected position error of residue $j$ when the structure is aligned on residue $i$; it is the right quantity to read relative domain placement. Both are *trained* quantities, and calibration shifts with novelty (Chapter 45): confidence is trustworthy in the regime of the training data. Regions with low pLDDT often correspond to intrinsic disorder, so pLDDT doubles as a disorder predictor [[S]].

**Why it worked.** [[S]] (i) The MSA supplies evolutionary constraints that physics-based methods could not use at scale. (ii) The pair representation with triangle updates builds in the *geometry of distance matrices*. (iii) The SE(3)-aware structure module and FAPE give a differentiable, end-to-end objective on the target of interest. (iv) Large training data (PDB plus self-distillation on predicted structures of unlabeled sequences), recycling and ensembling. Ablations in the paper show each of the main components matters.

**Complexity.** Pair stack: $O(L^3)$ time (triangle updates) and $O(L^2)$ memory; MSA stack: $O(N_\text{seq}L^2)$ for column-wise and $O(N_\text{seq}^2 L)$ with row-attention variants, so long proteins and large MSAs are memory-limited (the origin of "cropping" at training and of the chunking in inference).

**Language-model alternatives.** ESMFold replaces the MSA with the internal representations of a 15-billion-parameter protein language model (Chapter 34) and a folding trunk; it predicts structure from one sequence, at low cost, and it is less accurate than MSA-based AF2, particularly for proteins with few homologs where the language model has less to recall; this places the relation between *language-model memory of families* and *explicit evolutionary information* at the center of the debate about what protein language models know.

---

## 35.4 AlphaFold3 and co-folding: diffusion on atoms

AF3 changes four things.

1. **Inputs are any biomolecule**: proteins, DNA, RNA, small molecules (given as chemical graphs through their atoms), ions, and modifications.
2. **The Evoformer is replaced by a lighter *Pairformer***: the MSA plays a much smaller role (an MSA module with fewer blocks), and most computation is in the pair and single representations.
3. **The structure module is replaced by a *diffusion module* acting directly on atom coordinates** (Chapter 15). Noisy coordinates are denoised by a transformer conditioned on the Pairformer's representations; there are no residue frames, no torsion parametrization, and no explicit stereochemical loss (chirality comes from the training signal and the reference-conformer inputs). Training uses noise levels in a multi-scale schedule so that the model learns both *large-scale* structure (high noise) and *local geometry* (low noise).
4. **Generation is stochastic**: multiple samples can be drawn per input, and ranked by a learned confidence (pTM/ipTM, pLDDT). This makes *ensemble-like behavior* possible in principle, but sample diversity is not calibrated as a physical distribution (§35.6).

The reported gains (Abramson et al. 2024): substantially higher accuracy than docking tools for protein-ligand complexes on the PoseBusters benchmark (even without a given pocket), improved protein-nucleic-acid and antibody-antigen prediction relative to earlier tools. Follow-on and open-source models, all in the AF3 family: **Chai-1**, **Boltz-1** and **Boltz-2** (the latter adds a binding-affinity module that its authors report approaches the accuracy of free-energy-perturbation calculations at about a thousand-fold lower cost; open source under the MIT license; 2025), **Protenix** (ByteDance), **RoseTTAFold All-Atom**, and in 2026 Isomorphic Labs' proprietary **IsoDDE** (technical report, 10 February 2026), which reports roughly twice AF3's accuracy on the hardest, training-dissimilar protein-ligand cases [[P]]: company-reported, on benchmarks the authors constructed, not independently replicated at the time of writing.

!!! paper "Paper dissection: AlphaFold3 (Abramson et al., *Nature* 630, 2024)"
    **Problem.** Predict the joint 3-D structure of complexes of proteins, nucleic acids, small molecules, ions, and modified residues from sequences and chemical identities.
    **Insight.** A single generative model on *atoms* with a diffusion head can replace the specialized, residue-frame-based structure module, and handles arbitrary ligands.
    **Architecture.** Input embedder with templates and an MSA module; a Pairformer trunk; a diffusion module (a token transformer of 24 blocks sandwiched between atom-level attention encoder and decoder) that denoises coordinates; a confidence head trained on denoised samples.
    **Objective.** A diffusion denoising loss (weighted MSE on coordinates with rigid-alignment for the system), plus a distogram loss and auxiliary stereochemical losses (bond-length, smooth LDDT), plus confidence losses.
    **Data.** The PDB up to a 2021 cutoff, plus distillation of predicted structures; training cutoffs and test splits use temporal and similarity filters.
    **Evaluation.** PoseBusters (ligand pose within 2 Å and physical validity), CASP15 RNA, antibody-antigen docking, protein-protein interface accuracy (DockQ), with comparisons to docking and earlier deep-learning baselines.
    **Why it worked.** The diffusion head converts a hard multimodal regression into learning a score function at each noise scale; atom-level generation obviates a bespoke parametrization for every chemistry; large and diverse structure data.
    **Assumptions.** Training distribution coverage of the binding modes and chemistries tested; the PDB's biases; static structures.
    **Limitations.** a tendency to hallucinate ordered-looking structure in disordered regions (the paper counters it with distillation from earlier-model predictions); stereochemical errors needing post-filtering; dependence on similarity to training (§35.5); no affinities, no dynamics, no free energies.
    **What followed.** Open reimplementations; affinity heads; antibody-specific co-folding; fine-tuning for design (Chapter 36); evaluation papers exposing memorization (§35.5).
    **Unresolved.** Whether the model has learned *interaction physics* or an *interpolation of known complexes* (§35.5).

---

## 35.5 Memorization or physics? What the evidence says

The scientific question: does a co-folding model predict a pose because it has learned the physics of binding, or because it recognizes a protein–ligand pattern from its training set? Two lines of evidence:

- **Adversarial perturbations** (Masters et al., first posted 2024; *Nature Communications* 2025). Mutating *all* binding-site residues of a protein to unfavourable residues (such as glycine, or to residues of opposite chemistry) should remove binding in any physics-based picture; the studied co-folding models (AF3 and RoseTTAFold All-Atom) often still placed the ligand in the original pocket, and produced overlapping, physically implausible poses in other adversarial setups. This is *evidence against pure physics* [[S]] and compatible with a strong role for memorized pocket geometry.
- **Similarity-stratified evaluation** (Škrinjar et al., "Runs N' Poses", bioRxiv 2025). On 2,600 high-resolution protein-ligand systems released after the training cutoffs of AF3, Protenix, Chai-1 and Boltz-1, success falls steeply as similarity to training-set complexes decreases, for all four models, and especially for ligands seen in only one pocket; promiscuous ligands such as cofactors do better. The authors' reading is that current co-folding largely recalls ligand poses from training [[S]].

These are in tension with high headline success rates on PoseBusters because that benchmark contains many systems with training-set analogs. The synthesis, graded: *co-folding models are very good at interpolating within the manifold of known binding modes and substantially worse outside it* [[S]]; whether this improves with scale, data diversity, or physics-informed training is [[H]]. IsoDDE's headline claim (more than doubling AF3's accuracy on the hardest dissimilar cases) addresses exactly this regime, and independent evaluation is the next step. Chapter 43's lesson applies: *report accuracy as a function of similarity to training*, not one number.

!!! rhyme "Structural rhyme: co-folding memorization ↔ the cluster-split gap in molecular ML (Chapter 24) ↔ shortcut shift (Chapter 45)"
    A random split of molecules gave $r=0.85$ and a scaffold split $r=0.62$; a co-folding benchmark with training analogs gives high success and one without gives low. The same structure appears in each: a high-capacity model on a dataset with strong local correlation scores well on *neighbors* of the training set. The remedy is a *stratified* report on similarity, and, for deployment, a calibrated novelty score.

!!! openproblem "Open problem: co-folding that provably depends on physics"
    **Diagnostic questions.** Would a model trained on structures *and* on energies or forces from quantum or molecular-mechanics calculations pass the adversarial binding-site tests without losing PoseBusters accuracy? Could a benchmark built from *prospective* crystallographic fragment screens (ligands and sites with no training analogs, measured after the model's release) measure the true out-of-distribution accuracy? What is the minimal set of physical tests (mutations that abolish a key interaction; ligands that cannot fit; charge reversal) that a model must pass before its poses are used in a design cycle?

---

## 35.6 Ensembles, dynamics, disorder: where one snapshot is the wrong answer

- **Conformational heterogeneity.** AF2 and AF3 output one (or a few) static structure(s). Proteins with two stable states (fold-switching proteins, kinases in active and inactive conformations, transporters) are predicted in one state or an uninterpretable mixture. Perturbing the input MSA (subsampling, Del Alamo et al.; clustering by sequence similarity, "AF-cluster", Wayment-Steele et al.) yields alternative conformations in some cases [[S]], without a calibrated weighting [[H]].
- **Generative ensemble emulators.** *BioEmu* (Lewis et al., *Science*, July 2025) is a diffusion model built on AlphaFold-derived embeddings, trained on the AlphaFold database, then on more than 200 milliseconds of aggregate molecular-dynamics simulations of thousands of proteins, and finally fine-tuned to more than 500,000 experimental protein-stability measurements; it samples thousands of independent structures per hour on one GPU and reproduces folding free energies to within about 1 kcal/mol on benchmarks, with 4–5 orders of magnitude speed-up for equilibrium distributions, according to its authors [[S]]. The training combines three kinds of supervision (structures, simulation, thermodynamics) because *no one of them contains the equilibrium distribution*. Whether such emulators are reliable for proteins unlike the MD training set is [[H]].
- **Intrinsically disordered regions.** Roughly a third of eukaryotic proteome residues lie in disordered regions; low pLDDT flags them. Predicting a single structure for them is the wrong question: the target is an ensemble and, for condensate-forming proteins, a phase behavior (Chapters 19, 42).
- **Kinetics and mechanism.** Folding pathways, allostery, and binding kinetics require time; no structure predictor outputs a rate, and the question of whether a learned model can produce the *committor* of a transition is open.

---

## 35.7 Evaluation: CASP, and what keeps benchmarks honest

CASP (Critical Assessment of protein Structure Prediction), a biennial blind competition since 1994, made the field's progress measurable by the simple device of testing on structures *not yet public*. It is the best-run benchmark in computational biology: targets are held out until predictions are submitted; assessors are independent; categories are separated. Its fragility is also instructive: NIH funding for CASP lapsed in 2025, and Google DeepMind provided a one-time gift to cover roughly a year of operations while permanent funding was sought; a 2026 round was planned. **A community benchmark is infrastructure**, and the fact that the field's flagship depended on a small grant is an argument for treating benchmark maintenance as a first-class funded activity.

CASP16 (2024) assessed, among other categories, protein-ligand and monomer structure prediction; the assessment articles (2025) report that AF3-based pipelines were prominent among the top groups, and identify ligands with unfamiliar binding modes, RNA, and conformational ensembles as the remaining weak points [[P]] (consult the category assessments for the numbers). Assessment of a method on *one year of targets* has a sampling error: the number of independent hard targets is small (dozens), and per-target differences between top groups are within the target-to-target noise (Chapter 43).

---

## 35.8 Worked research examples

!!! example "Worked Research Example 35.1: Is a predicted structure good enough to interpret a missense variant?"
    **Situation.** A clinical group wants to use predicted structures to explain why a rare missense variant in a gene of uncertain function is pathogenic. The AlphaFold2 model has pLDDT 92 at the residue, PAE 4 Å within the domain, and the variant changes a buried leucine to proline.

    **Question.** What can and cannot be concluded?

    **Reasoning.**

    1. **What the confidence supports.** The residue's *local* environment is probably right (pLDDT above 90 indicates high expected local accuracy): the burial and the helix context are credible [[S]].
    2. **What the structure does not provide.** The *effect* of the change on stability or function. A predicted wild-type structure is not a mutant structure: single-sequence perturbation of AF2 inputs is a weak predictor of stability change; a stability predictor trained on measurements (Chapter 34's fitness predictors, or physics-based $\Delta\Delta G$ tools) provides a separate, partly independent estimate; neither addresses effects on binding partners, regulation, or expression.
    3. **Triangulate with independent evidence** (Chapter 41): evolutionary conservation (language-model score, Chapter 34), population frequency (gnomAD constraint), clinical databases, and experimental *deep mutational scans* if they exist.
    4. **Calibrate on a matched benchmark.** For genes with known pathogenic and benign variants in similar structural contexts, estimate the likelihood ratios of the structural features, and report them with intervals (the ACMG/AMP framework's evidence-strength mapping; Chapter 41).
    5. **Claim.** "Structural and evolutionary evidence supports a destabilizing effect (PP3-level computational evidence)" at most; "pathogenic" requires clinical and functional evidence.

    **Expert analysis.** The structure answers *where the residue is* and *what it contacts*; the question asked is *what the substitution does*. The misreading in the literature is to treat structure-prediction accuracy as variant-effect accuracy.

!!! example "Worked Research Example 35.2: Prospectively testing whether a co-folding model has learned physics, with no known answer"
    **Situation.** A group proposes to use a co-folding model to prioritize fragment-sized ligands for a poorly characterized enzyme with no ligand-bound structures. There is no ground truth available, and the target resembles no protein in the training set.

    **Question.** How would you decide whether to trust the model's poses and ranks, and what would you measure?

    **Reasoning (Expert Chain).**

    1. **L1 Problem.** Pose and affinity prediction in a regime (no training analogs) where published benchmarks say models are least reliable (§35.5).
    2. **L3–L5 Assumptions and failure modes.** The model's confidence may be miscalibrated under novelty (Chapter 45); the fragments may be placed in a pocket the model "recalls" from a homolog rather than one the physics supports; fragments bind weakly, so ranking depends on fine energy differences (Chapter 24: noise ceilings of affinity measurements).
    3. **L7–L8 Hypotheses.** H1: poses are right and ranks informative; H2: poses are plausible but ranks uninformative; H3: both are wrong. A decisive set of experiments discriminates them.
    4. **L10 Experiments (ordered by cost).** (a) *In silico adversarial tests* on this target: mutate pocket residues to abolish the putative key interaction; if the ligand does not move, the pose is not physically grounded. (b) *Orthogonal computation*: docking with a physics-based scoring function and short MD of the best poses (stability of the pose), as a second opinion with *different* failure modes. (c) *Prospective crystallographic fragment screen* (hundreds of fragments soaked into crystals); compare the experimentally observed poses with predictions: this gives the pose success rate on this target with a binomial confidence interval and the rank correlation with measured binding (e.g., SPR or thermal shift). (d) *Mutational confirmation*: mutate the predicted key residues and test loss of binding.
    5. **L11 Interpretation.** Pre-specify the thresholds: with 100 fragments tested and a pose success of at least 40% (the lower bound of a 95% interval above 30%), accept the model for pose generation on this target class; require a rank correlation of at least 0.3 for prioritization. Otherwise use the model only to propose pockets.
    6. **L12 New directions.** The accumulated results for *this kind* of target (novel enzymes, no analog) form an out-of-distribution benchmark that the field lacks.

    **What is not known.** The accuracy to expect: current evidence suggests it is low for training-dissimilar systems, and company reports of improvement on hard dissimilar cases await independent prospective tests.

---

## 35.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: auditing a structure-prediction result"
    1. **Report the confidence** (pLDDT per residue, PAE between domains, ipTM for interfaces) alongside the structure, never the structure alone.
    2. **Stratify by similarity to training** (sequence identity and structural similarity of the closest PDB entry before the training cutoff, and for ligands the similarity of the pocket and the ligand).
    3. **Check physical validity** (clashes, bond geometry, chirality), e.g. with PoseBusters-style tools.
    4. **Run an adversarial test** (mutate the key interactions; swap the ligand for a decoy of similar size) and see whether the prediction responds.
    5. **Ask what the structure is for**; choose the metric (TM, lDDT, interface quality, pocket RMSD) that matches the use.
    6. **Sample**: look at several samples and their disagreement; if the model offers an ensemble, treat the spread as uncertainty only after checking its calibration on cases with known multiple states.
    7. **Reproduce the contact-precision experiment** for a domain of interest: take a predicted contact map, estimate its precision from the model's confidence, and reconstruct from the top contacts at several confidence thresholds. Where is the cliff?

    **What it teaches.** A structure prediction is a *claim with a confidence and a domain of validity*. The measurement, not the model, decides what to believe.

    **An open question to carry forward.** The precision cliff (§35.1) says that an inference engine consuming noisy pairwise information must weigh it by reliability. In a model with learned confidence, does the Pairformer implement something like *robust bound smoothing* that discards false contacts, or does it simply average? Could one *probe the pair representation* for contact reliability (Chapter 48) and show that its geometry is consistent with the triangle inequality only where the model is confident?

---

## 35.10 Connections

- **Backward:** invariance versus equivariance and chirality (Chapter 16); diffusion (Chapter 15); attention and the pair representation (Chapter 12); coevolution and Potts models (Chapter 29); protein language models (Chapter 34); benchmark stratification and winner's curse (Chapter 43); shift and calibration (Chapter 45).
- **Forward:** protein and biomolecular design built on structure models (Chapter 36); drug discovery and affinity (Chapter 37); ensembles and mechanism (Chapters 42, 48); open problems of proteins and molecules (Chapter 52).

!!! takeaways "Key takeaways"
    1. A fold is determined by a small set of *accurate* long-range contacts plus local geometry: on a real 66-residue domain, 24 true contacts gave TM 0.85, 47 gave 0.89 and the full binary map 0.94; with no global information TM was 0.07.
    2. **Precision beats recall**: 47 contacts at 70% precision gave TM 0.30 (50%: 0.24; 30%: 0.16). Calibrated confidence per prediction is what downstream geometry needs.
    3. Distances cannot fix handedness: in 8 reconstructions from the full distance information, 2 were native-handed and 6 were mirror images. Models need non-reflection-invariant terms (FAPE, chirality losses).
    4. **Information versus inference**: random-start optimization failed at TM 0.19 *with* all true contacts; a distance-geometry start succeeded. Diagnose whether a failure is a lack of information or of inference.
    5. AF2 combines an MSA and pair representation with triangle updates (learned bound smoothing), invariant point attention, FAPE, recycling, and trained confidence; AF3 swaps the structure module for atom-level diffusion and extends to complexes.
    6. **Co-folding results depend on similarity to the training set** [[S]]: adversarial binding-site mutations often leave poses unchanged, and accuracy falls with decreasing similarity to training analogs. Claims of improvement on dissimilar systems (IsoDDE, Boltz-2's affinity module) are company- or developer-reported and await independent prospective tests [[P]].
    7. A static structure is not an ensemble, a free energy, or a rate; ensemble emulators such as BioEmu add thermodynamics and dynamics supervision and are promising and not yet broadly validated [[H]].
    8. Benchmarks like CASP are infrastructure; fund and stratify them.

---

## Further reading

- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589. Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500. Baek, M. et al. (2021). Accurate prediction of protein structures and interactions using a three-track neural network. *Science* 373, 871–876. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130.
- Wohlwend, J. et al. (2024). Boltz-1: democratizing biomolecular interaction modeling. *bioRxiv*. Passaro, S. et al. (2025). Boltz-2: towards accurate and efficient binding affinity prediction. *bioRxiv.* Chai Discovery (2024). Chai-1: decoding the molecular interactions of life. *bioRxiv.* Isomorphic Labs (2026). IsoDDE technical report (10 February 2026).
- Masters, M. R., Mahmoud, A. H. & Lill, M. A. (2025). Investigating whether deep learning models for co-folding learn the physics of protein-ligand interactions. *Nat. Commun.* (first posted as a 2024 bioRxiv preprint). Škrinjar, P. et al. (2025). Have protein-ligand co-folding methods moved beyond memorisation? *bioRxiv* (Runs N' Poses). Buttenschoen, M., Morris, G. M. & Deane, C. M. (2024). PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. *Chem. Sci.* 15, 3130–3139.
- Lewis, S. et al. (2025). Scalable emulation of protein equilibrium ensembles with generative deep learning. *Science* 389 (6761). Del Alamo, D., Sala, D., Mchaourab, H. S. & Meiler, J. (2022). Sampling alternative conformational states of transporters and receptors with AlphaFold2. *eLife* 11, e75751. Wayment-Steele, H. K. et al. (2024). Predicting multiple conformations via sequence clustering and AlphaFold2. *Nature* 625, 832–839.
- Marks, D. S. et al. (2011). Protein 3D structure computed from evolutionary sequence variation. *PLoS ONE* 6, e28766. Morcos, F. et al. (2011). Direct-coupling analysis of residue coevolution captures native contacts across many protein families. *PNAS* 108, E1293–E1301. Vendruscolo, M., Kussell, E. & Domany, E. (1997). Recovery of protein structure from contact maps. *Fold. Des.* 2, 295–306. Havel, T. F. (1998). Distance geometry: theory, algorithms, and chemical applications. In *Encyclopedia of Computational Chemistry.*
- Zhang, Y. & Skolnick, J. (2004). Scoring function for automated assessment of protein structure template quality. *Proteins* 57, 702–710 (TM-score). Mariani, V., Biasini, M., Barbato, A. & Schwede, T. (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. *Bioinformatics* 29, 2722–2728. Kryshtafovych, A. et al. CASP15 and CASP16 assessment articles, *Proteins* (2023; 2025).
