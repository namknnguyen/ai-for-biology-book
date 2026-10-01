# Chapter 24. Small Molecules, Interactions, and the Chemistry of Drug Discovery

!!! abstract "Chapter at a glance"
    **Motivation.** Most approved drugs are small molecules, and the molecular side of AI for biology (docking, affinity prediction, molecular generation, ADMET models; Chapter 37) is built on a handful of facts about chemical representation, binding thermodynamics, and the *data that exist*. This chapter teaches those facts *for modeling*, and then uses two real public datasets to show how easily a molecular-ML result can overstate what it has learned.
    **Prerequisites.** Chapters 4, 5, 7, 13, 16, 22 (the thermodynamic-sigmoid habit).
    **You will be able to:** (1) represent a molecule as a string, a graph, a fingerprint, and a 3-D ensemble, and say what each representation discards; (2) derive the binding isotherm, the free energy–potency conversion ($1.37$ kcal/mol per log unit), the Cheng–Prusoff relation, and the thermodynamic cycle behind free-energy perturbation; (3) explain Morgan fingerprints as Weisfeiler–Lehman refinement and predict their collision and horizon behavior; (4) describe the drug-discovery funnel as an information-gathering process and locate where ML can and cannot help; (5) evaluate a molecular model with random, scaffold, and cluster splits, a noise ceiling, and an activity-cliff test; (6) list the standard biases of chemical bioactivity data.

---

## 24.1 The seven questions for a small molecule

!!! bio "Biology for modeling: a small molecule"
    **What is it?** A molecule with typically 10–60 heavy atoms (molecular weight below ~500–600 Da for oral drugs) that binds a biomolecule, usually a protein, and changes its activity: an enzyme inhibitor, a receptor agonist or antagonist, a modulator of a protein–protein interaction, or a "molecular glue".
    **Information it contains.** A covalent graph (atoms, bonds, charges), stereochemistry, and a *distribution over three-dimensional conformations and protonation or tautomeric states* in solution; function arises from how this ensemble complements a binding site.
    **How is it generated?** Synthesized by organic chemistry (or isolated as a natural product). Libraries are enumerated by combining building blocks with reliable reactions; the space of such products is astronomically larger than anything ever made.
    **How is it measured?** *Potency*: dose–response in biochemical or cell assays (IC$_{50}$, EC$_{50}$), biophysical binding (SPR, ITC) giving $K_d$ and kinetics. *Structure*: X-ray or cryo-EM of the complex. *Properties*: solubility, permeability, metabolic stability, toxicity; in vivo pharmacokinetics.
    **Computational representation.** SMILES strings; molecular graphs with atom/bond features; fingerprints; 3-D conformer ensembles and fields; protein–ligand complexes as graphs or point clouds.
    **What varies.** Across chemical space (scaffolds, substituents), stereochemistry, protonation state, assay conditions, target species and construct, and the cellular context.
    **What can ML learn?** Local structure–activity relationships (SAR) within a chemical series, property trends (lipophilicity, solubility), and which substructures tend to be tolerated at a site.
    **What can ML not observe?** Whether the *target hypothesis* is correct (the dominant cause of clinical failure), conformational energetics not present in the data, off-target and in vivo biology, and anything about chemistry far from the training distribution.

---

## 24.2 Representing a molecule

### 24.2.1 Graphs, strings, and what each discards

A molecule is, to a first approximation, a labeled graph $G=(V,E)$: vertices are heavy atoms (element, formal charge, aromaticity, implicit hydrogen count), edges are bonds (single, double, triple, aromatic). **SMILES** (Weininger, 1988) serializes a graph by a depth-first traversal: `CC(=O)Oc1ccccc1C(=O)O` is aspirin. Branches use parentheses, ring closures use digits, and chirality is marked with `@` and `@@`. Four facts matter for modeling.

1. **A SMILES string is not unique.** Any depth-first traversal from any starting atom is valid. Canonicalization algorithms pick one string per molecule, but randomized traversal is also legal, and *the same molecule has many strings*. In 2,000 random traversals of aspirin (13 heavy atoms) we obtained 300 distinct valid strings, and for caffeine (14 heavy atoms) 936 (a lower bound on the number of valid strings), all mapping back to a single canonical string. This is both a **data augmentation** (random SMILES improve generative models; Chapter 37) and a **tokenization hazard**: a sequence model trained on canonical strings sees one of hundreds of equivalent spellings, and nearby *strings* are not nearby *molecules* (a one-character change can turn a ring closure into a different molecule or an invalid string).
2. **The 2D graph discards stereochemistry unless told.** The two enantiomers of carvone, (*R*) and (*S*), have the same graph, the same formula, and identical standard descriptors; with chirality ignored their Morgan fingerprints have Tanimoto similarity exactly $1.00$ (§24.3), and even a chirality-aware fingerprint gives $0.93$. They are different molecules for a chiral receptor: (*R*)-(−)-carvone smells of spearmint and (*S*)-(+)-carvone of caraway [[E]]. In drug discovery, enantiomers can differ by orders of magnitude in potency and in toxicity. *Chirality is the molecular analogue of the chirality warning of Chapter 16*: a representation that is invariant to reflections cannot tell mirror images apart.
3. **Protonation and tautomer state are assigned, not measured.** The BACE-1 inhibitors we use below are written with amidine nitrogens protonated (`[NH2+]`); the charge state at assay pH depends on the local environment (a $pK_a$ shift in the active site can change it), and many datasets carry arbitrary or inconsistent conventions.
4. **A molecule is an ensemble.** Rotatable bonds, ring flips, and solvent produce a Boltzmann distribution over conformers; the *bound* conformation may be a high-energy member of the unbound ensemble (strain energy), and binding trades the cost of freezing flexible bonds against contact energy. A single 3-D conformer, as typically generated by a force-field embedding, is one sample, and different embedding runs return different conformers.

### 24.2.2 Chemical space is enormous and mostly unexplored

Bohacek, McMartin, and Guida (1996) estimated on the order of $10^{60}$ drug-like molecules of up to 30 heavy atoms (C, N, O, S) [[S]]. The GDB-17 enumeration (Ruddigkeit et al., 2012) lists $1.66\times10^{11}$ molecules of up to 17 heavy atoms; make-on-demand collections list molecules that can be ordered and typically delivered in weeks. Enamine's REAL Space, the largest searchable collection, was reported at about 76.9 billion molecules in mid-2025 and about 95 billion in an April 2026 update (vendor figures; each product is a *virtual* enumeration with a high but not perfect predicted synthesis success rate) [[E]]. So the enumerated, orderable region is about $10^{11}$ of an estimated $10^{60}$: *a fraction of $10^{-49}$*. A model that generalizes beyond known chemistry is therefore not an optional luxury; the interesting region of chemical space is, by construction, nearly all out of distribution.

---

## 24.3 Fingerprints: Morgan, Weisfeiler–Lehman, and message passing

The workhorse molecular featurization, for decades and still a strong baseline, is the **extended-connectivity fingerprint** (Morgan, 1965; Rogers & Hahn, 2010). It is the following algorithm.

!!! math "Morgan fingerprint as iterated hashing"
    Give every atom $i$ an initial integer identifier $I_i^{(0)}=h_0(\text{atomic invariants of }i)$ (element, degree, charge, hydrogen count, ring membership). For $r=1,\dots,R$ update

    $$
    I_i^{(r)}=h\Big(I_i^{(r-1)},\ \operatorname{sort}\big\{(b_{ij},\,I_j^{(r-1)}):\ j\in\mathcal N(i)\big\}\Big),
    $$

    where $b_{ij}$ is the bond type and $h$ is a hash function. The fingerprint is the *multiset* $\{I_i^{(r)}:\,i\in V,\ 0\le r\le R\}$; to obtain a fixed-length vector of dimension $B$ (usually $B=1024$ or $2048$) each identifier is **folded** by $I\bmod B$ and either set to 1 (bit vector) or counted (count vector). The cost is $O(R\,|E|)$ per molecule and the output is $\le(R+1)|V|$ nonzero entries.

**Rhyme with message passing.** This is *exactly* one round of message passing per radius step, with an injective, order-invariant aggregator (sort the neighbor multiset, then hash). It is the **1-dimensional Weisfeiler–Lehman (WL) color refinement** algorithm; Xu et al. (2019) proved that message-passing GNNs with injective aggregators are as discriminating as the 1-WL test, and no more. Consequently *a Morgan fingerprint with unlimited radius and no folding loses no information that a standard GNN can see*; what differs is that the fingerprint's functions are fixed hashes (no learned similarity between environments), while a GNN learns embeddings in which similar environments are nearby.

Three properties follow from the algorithm and can be checked in the data:

1. **Horizon.** Atom identifiers at radius $R$ encode the neighborhood within $R$ bonds. Two molecules that differ only by structure beyond that horizon (for example, the length of a polymethylene linker in a macrocycle) can have *identical* binary fingerprints. This is the receptive-field limit of a depth-$R$ GNN (Chapters 10 and 16).
2. **Folding collisions.** With $D$ distinct environments hashed uniformly into $B$ bins the expected load is $\lambda=D/B$ per bin, and the fraction of occupied bins that contain two or more environments is $(1-e^{-\lambda}-\lambda e^{-\lambda})/(1-e^{-\lambda})$. In our 1,513 molecules, 5,235 distinct environments ($R\le2$) fold into 2,048 bits, giving $\lambda=2.56$ and a predicted collision fraction of $0.78$; we measure $0.80$. *Most occupied bits are shared by several unrelated substructures*, so a bit is not a substructure, though models work well enough with collisions because the data's effective dimension is small.
3. **Stereochemistry is a flag.** By default the algorithm does not read chirality (§24.2.1).

**Tanimoto similarity.** For bit sets $A,B$: $T(A,B)=|A\cap B|/|A\cup B|$ (Jaccard); for counts, $\sum_k\min(a_k,b_k)/\sum_k\max(a_k,b_k)$. Both are positive-definite kernels (Ralaivola et al., 2005), so they can be used in SVMs and Gaussian processes. A value of $T\approx0.85$ (for Daylight-type fingerprints) was long used as a rule of thumb for "similar activity", but Martin et al. (2002) showed that even above that threshold the probability that a neighbor of an active compound is itself active is well below one [[S]]; as §24.7 shows, the relation between fingerprint similarity and activity is statistical and has a heavy tail.

---

## 24.4 Binding: thermodynamics, kinetics, and assays

### 24.4.1 The isotherm

For a protein $P$ binding a ligand $L$ in 1:1 stoichiometry, $P+L\rightleftharpoons PL$, mass action at equilibrium gives the dissociation constant

$$
K_d=\frac{[P][L]}{[PL]}.
$$

Let $\theta=[PL]/([P]+[PL])$ be the fraction of protein bound. In the usual regime where the ligand is in large excess over the protein, $[L]\approx[L]_\text{total}$ and

$$
\theta=\frac{[L]}{K_d+[L]}.
$$

This is a **sigmoid in $\log[L]$** with a fixed shape (Hill slope $=1$): $\theta$ rises from 10% to 90% over an 81-fold range of concentration, and the midpoint is $K_d$. It has the same form as TF occupancy (Chapter 22) and the fraction of folded protein (Chapter 23): *a sigmoid is what a single two-state thermodynamic equilibrium looks like*. The standard binding free energy is

$$
\Delta G^\circ=RT\ln\frac{K_d}{c^\circ}\quad(c^\circ=1\ \text{M}),\qquad
\Delta G^\circ=-RT\ln(10)\cdot pK_d=-1.37\ \text{kcal/mol}\times pK_d\ \ (\text{at }298\text{ K},\ RT=0.593).
$$

**One log unit of potency is 1.37 kcal/mol.** A 1 nM binder ($pK_d=9$) has $\Delta G^\circ\approx-12.3$ kcal/mol; a 10 µM binder ($pK_d=5$) has $-6.8$ kcal/mol. The 1,513 BACE-1 inhibitors we analyze below span $pIC_{50}$ 2.5–10.5, i.e. $-3.5$ to $-14.4$ kcal/mol (assuming $IC_{50}\approx K_d$; §24.4.2): *a factor of $10^8$ in potency is only 11 kcal/mol of free energy*, and a typical 10-fold optimization step is 1.4 kcal/mol, about the energy of a single good hydrogen bond *if nothing else changes*.

### 24.4.2 IC$_{50}$ is not $K_d$: Cheng–Prusoff

Most potency data are $IC_{50}$ values: the inhibitor concentration that halves an observed activity. For a **competitive** inhibitor of an enzyme with substrate $S$, the Michaelis–Menten rate is

$$
v=\frac{V_{\max}[S]}{K_m\,(1+[I]/K_i)+[S]}.
$$

Without inhibitor, $v_0=V_{\max}[S]/(K_m+[S])$. Setting $v=v_0/2$ and solving for $[I]$:

$$
K_m\big(1+[I]/K_i\big)+[S]=2(K_m+[S])\ \Rightarrow\ [I]_{50}=IC_{50}=K_i\Big(1+\frac{[S]}{K_m}\Big)\quad\text{(Cheng–Prusoff, 1973)}.
$$

So $IC_{50}$ depends on the *assay's substrate concentration*, and a pair of labs using different $[S]$ report different numbers for the same molecule. If $[S]\ll K_m$ then $IC_{50}\approx K_i$. For non-competitive or tight-binding inhibitors, and for cell-based assays (where permeability, efflux, and target abundance enter), the relation changes again. **A dataset of $pIC_{50}$ values pooled across papers is a mixture of different link functions from the latent $\Delta G$**, the same structure as the "assay link" of Chapter 23.

**Measurement noise.** Even after careful curation, the experimental uncertainty of individual published $K_i$ values in ChEMBL was estimated at a standard deviation of about 0.54 $pK_i$ units (mean absolute error 0.44), which limits the Pearson $R^2$ of any model against such a dataset to about 0.81 on large heterogeneous collections (Kramer et al., 2012) [[S]]; mixed-source $IC_{50}$ data are, if anything, noisier (Kalliokoski et al., 2013; Landrum & Riniker, 2024) [[S]]. We use $\sigma=0.54$ below as an illustration.

### 24.4.3 Kinetics: residence time

$K_d=k_\text{off}/k_\text{on}$. The association rate for a diffusion-limited encounter cannot exceed about $10^{8}$–$10^{9}\ \text{M}^{-1}\text{s}^{-1}$, and drug-like molecules typically have $k_\text{on}\sim10^{5}$–$10^{7}$. Hence potency is mostly set by **$k_\text{off}$**: a 1 nM binder with $k_\text{on}=10^6$ has $k_\text{off}=10^{-3}\,\text{s}^{-1}$, a residence time $1/k_\text{off}=1000$ s (half-life $\ln2/k_\text{off}\approx12$ minutes). *In vivo* the drug is not at equilibrium: concentrations fall as the drug is cleared, so the **residence time** can matter more than $K_d$ for duration of effect (Copeland et al., 2006) [[S]]. Datasets mostly report equilibrium quantities, so kinetics is a large piece of missing information.

### 24.4.4 Efficiency, size, and selectivity

**Ligand efficiency** $LE=-\Delta G^\circ/N_\text{heavy}$ (Hopkins et al., 2004) normalizes potency by size; 0.3 kcal/mol per heavy atom is a commonly cited target for good leads. In the BACE-1 set, mean $LE=0.27$ and the 95th percentile $0.37$. Potency increases with size (correlation of $pIC_{50}$ with heavy-atom count $=0.45$, so *size alone explains $R^2=0.20$* of the variance) because bigger molecules make more contacts, which will matter when we evaluate models (§24.7). **Selectivity** is a ratio of dissociation constants for target versus off-targets; ML models are usually trained on one target at a time and rarely see the off-target distribution that determines toxicity.

---

## 24.5 The discovery funnel as an information-gathering process

!!! lens "Research lens: the pipeline as a sequence of increasingly expensive, increasingly informative measurements"
    Every stage of drug discovery is a measurement with a cost per compound, a noise level, and a relation to the final clinical objective. ML's leverage is to *replace an expensive measurement with a cheaper prediction*; the question at each stage is whether the prediction's error is smaller than the stage's *own* assay noise and whether its **systematic** error correlates with the next stage's failure modes.

| Stage | What is measured | Scale (orders of magnitude) | Dominant uncertainty |
|---|---|---|---|
| Target identification and validation | Does modulating the target change the disease phenotype? | 1 target (genetics, perturbation; Chapters 26, 39, 41) | **Causal**: is the target hypothesis right? |
| Hit finding | Binding or activity of library compounds at a single concentration | HTS $10^5$–$10^6$ compounds; DNA-encoded libraries $10^8$–$10^{11}$; fragments $10^3$ (very weak, high LE); virtual docking $10^8$–$10^9$ | False positives (aggregators, assay interference), low hit rates |
| Hit-to-lead, lead optimization | Dose–response potency, selectivity, ADME (absorption, distribution, metabolism, excretion), toxicity | $10^2$–$10^4$ compounds synthesized and made | Multi-objective trade-offs; the *series* is local |
| Preclinical | PK/PD, safety in animals, efficacy models | $1$–$10$ candidates | Species translation |
| Clinical (phases I–III) | Safety, dose, efficacy in humans | 1 | Biology and patient heterogeneity |

*Hit finding technologies.* **High-throughput screening** tests physical compounds. **DNA-encoded libraries** (DEL; Brenner & Lerner, 1992) attach a DNA barcode to each molecule so that billions can be pooled, panned against an immobilized target, and read out by sequencing; the data are *enrichment counts* with substantial noise and dependence on the building-block synthesis chemistry. **Fragment-based discovery** finds tiny molecules (often below 250 Da) binding weakly (mM–µM) but with high ligand efficiency, then grows them (Erlanson et al., 2016) [[E]]. **Ultra-large docking** screens hundreds of millions to billions of make-on-demand molecules in silico and tests only the top few hundred (Lyu et al., 2019) [[E]]; its yield depends on the scoring function (§24.8). The 2024 BELKA challenge released about 133 million DEL molecules screened against three proteins (sEH, BRD4, and serum albumin; about 3.6 billion measurements) with a test set of *out-of-distribution* compounds, giving the field a large public DEL-trained benchmark [[E]].

*Drug-likeness.* **Lipinski's "rule of five"** (Lipinski et al., 1997) flags poor oral absorption likelihood when more than one of the following holds: $\text{MW}>500$, $\log P>5$, H-bond donors $>5$, acceptors $>10$. It is an empirical heuristic from compounds that reached clinical trials; many successful drugs (macrocycles, natural products, PROTACs) lie beyond it, and it says nothing about the target.

*Attrition.* Estimated probabilities of success from first-in-human trials to approval are around 10–14% across indications (Hay et al., 2014; Wong, Siah & Lo, 2019) [[S]]; analyses of phase II/III failures attribute roughly 40–50% to lack of clinical efficacy and about 30% to unmanageable toxicity, with a smaller share to poor pharmacokinetics (Sun et al., 2022) [[S]]. The consequence for an ML researcher is stark: **the dominant loss is not "wrong molecule" but "wrong target hypothesis" or "wrong patients"**. A better binder cannot repair a target that does not move the disease. This is why Chapters 39–41 (perturbation, virtual cell, genotype to phenotype) and 44–46 (causal inference, experiment design) are as relevant to drug discovery as the molecular models of Chapter 37.

!!! rhyme "Structural rhyme: the funnel ↔ cascade classifiers ↔ active learning"
    A discovery funnel is a **cascade**: cheap noisy filters first, expensive precise ones last. The theory of cascades (Chapter 5, information accounting; Chapter 46, experimental design) says that the *value of a cheap filter is its ability to remove candidates without removing the eventual winners*: a filter with recall below 50% on true winners is worse than no filter at all if the winners are rare. Evaluating a molecular ML model by *enrichment of known actives among decoys* measures the wrong thing; the right unit is *how many novel, confirmed winners per synthesized compound*, which requires prospective testing.

---

## 24.6 The data, and how they bias what models learn

The public data sources are **ChEMBL** (curated literature bioactivity, millions of compounds and tens of millions of activity records), **PubChem BioAssay** (screening deposits), **BindingDB**, and **PDBbind** (a curated set of protein–ligand complexes with measured affinity, on the order of $2\times10^4$ complexes). The model-relevant properties:

1. **Congeneric series.** Compounds come from medicinal-chemistry papers on one scaffold at a time, with dozens of close analogs differing in one or two substituents. Our BACE-1 set (1,513 compounds) contains 671 distinct Bemis–Murcko scaffolds and, at a Butina clustering threshold of Tanimoto $0.4$, only 106 clusters: about 14 compounds per cluster. A *random* split places near-duplicates of every test compound in the training set.
2. **Heterogeneous assays** (§24.4.2) and a noise floor of $\sim0.5$ log units.
3. **Missing negatives and censoring.** Inactives are rarely published; reported values such as "$>10\ \mu M$" are censored. Benchmarks that add *decoys* (assumed inactive) can introduce **bias**: in DUD-E, the property-matched decoys differ from the actives in ways that a model can exploit without seeing the protein (Chen et al., 2019) [[S]].
4. **Structure-based data leakage.** In PDBbind, many test complexes in the standard CASF benchmark have close protein and ligand analogs in the training set; when these are removed, the reported performance of several published deep-learning affinity models falls markedly (Graber et al., 2025; see the paper dissection below) [[S]].
5. **Distribution of what gets made.** Chemists synthesize what they expect to work; the dataset reflects *human medicinal chemistry priors*, not a random sample of chemical space. An ML model trained on it learns the priors.
6. **DEL noise and building-block bias** (§24.5).
7. **Unit and curation errors.** Transcription errors and undifferentiated stereoisomers are a known problem; the Kramer et al. analysis found "systematically detectable unit-transcription errors, undifferentiated stereoisomers and repeated citations of single measurements" that had to be removed before uncertainty could be estimated.

!!! paper "Paper dissection: Graber et al., *Nature Machine Intelligence* (2025), \"Resolving data bias improves generalization in binding affinity prediction\""
    **Problem.** Deep-learning scoring functions reported strong performance on the CASF benchmark when trained on PDBbind, but their performance on genuinely new complexes appeared much lower.
    **Key insight.** Many CASF test complexes are *near-duplicates* (by protein similarity, ligand similarity, and binding-conformation similarity) of PDBbind training complexes, so benchmark performance measured memorization of homologous complexes. They propose a structure-based filtering algorithm that removes leaking and redundant training examples, giving the **PDBbind CleanSplit**.
    **Evaluation.** They retrain published models on CleanSplit and evaluate on CASF; reported scores drop substantially (the paper states that the earlier performance was largely driven by leakage), and they present a graph neural network (GEMS) with language-model embeddings that retains competitive CASF performance when trained on CleanSplit. [[S]] (Headline conclusions as stated in the paper's abstract and summary; take the specific effect sizes from the paper.)
    **Why it matters.** It is the molecular instance of the leakage lesson of Chapter 1: a benchmark shared by many groups can embed the *same leak in every comparison*. Rankings of models then reflect their ability to exploit the leak.
    **Assumptions.** That similarity in protein, ligand, and pose, as measured by their algorithm, captures the relevant redundancy; that CASF is a reasonable proxy for the application (it is a small set of well-resolved crystal structures with known binding poses; a prospective virtual screen has neither).
    **Limitations.** Filtering reduces the training set, removes data that might be legitimately informative for related tasks, and cannot address *measurement* bias (e.g., affinity labels from heterogeneous assays) or the absence of true negatives.
    **What followed.** Growing use of time-split, target-split, and "no-leak" evaluation for binding affinity and docking benchmarks, and prospective challenges (Chapters 37, 43).

---

## 24.7 What do fingerprints and random forests learn? A controlled look at real data

We now use real data to answer four questions. **Setup** (`code/ch24_chemistry.py`): the BACE-1 inhibitor set (Subramanian et al., 2016, distributed in MoleculeNet; $n=1{,}513$; $pIC_{50}$ mean 6.52, SD 1.34) and the ESOL aqueous-solubility set (Delaney, 2004; $n=1{,}128$). Models are random forests (300 trees) on 2,048-dimensional Morgan count fingerprints ($R=2$), evaluated by 5-fold cross-validation.

### 24.7.1 Does similarity predict activity? (the similarity–property principle)

We computed all 1.1 million pairwise Tanimoto similarities and the absolute difference in $pIC_{50}$ for each pair:

| Tanimoto bin | Pairs | Mean $\lvert\Delta pIC_{50}\rvert$ | $P(\lvert\Delta\rvert\ge1)$ | $P(\lvert\Delta\rvert\ge2)$ |
|---|---|---|---|---|
| [0.0, 0.2) | 890,388 | 1.58 | 0.613 | 0.320 |
| [0.2, 0.3) | 162,343 | 1.43 | 0.590 | 0.277 |
| [0.3, 0.4) | 46,580 | 1.24 | 0.512 | 0.212 |
| [0.4, 0.5) | 17,800 | 1.01 | 0.406 | 0.137 |
| [0.5, 0.6) | 12,708 | 0.87 | 0.333 | 0.085 |
| [0.6, 0.7) | 8,609 | 0.81 | 0.310 | 0.069 |
| [0.7, 0.8) | 3,794 | 0.70 | 0.254 | 0.049 |
| [0.8, 1.0] | 1,606 | 0.62 | 0.199 | 0.044 |

The principle holds *on average* (mean difference falls from 1.58 to 0.62 log units) but the **tail does not vanish**: among the 5,400 pairs with Tanimoto $\ge0.7$, 257 differ by 100-fold or more and 1,282 by 10-fold or more. Even the most similar bin has a 20% chance of a 10-fold difference. Pairs of similar molecules with large potency differences are called **activity cliffs** (Maggiora, 2006; Stumpfe & Bajorath, 2012). Two sources: *real SAR discontinuities* (a methyl group that fills a pocket, a stereocenter, a charge that makes or breaks a salt bridge) and *measurement and curation noise*. With noise SD $0.54$ per measurement, a pair of equal-potency molecules differ by $\ge2$ log units with probability $2[1-\Phi(2/(0.54\sqrt2))]=0.9\%$, so noise explains a minority of the 4.4% observed in the most-similar bin, but an unknown fraction.

**Stereochemistry in the data.** 35 pairs have *identical* chirality-blind fingerprints (Tanimoto 1.0); their mean potency difference is 1.29 log units and 60% differ by 10-fold or more. A chirality-aware fingerprint separates 60% of them. Inspection of the first six unresolved pairs showed two causes: molecules differing only in a long macrocyclic linker length (beyond the radius-2 horizon: §24.3) and the same compound drawn with and without a specified stereocenter (a *labeling* inconsistency in the dataset). Both are failures of the *representation or data*, not of the learning algorithm.

### 24.7.2 How you split decides what you measure

| Split (5-fold) | Mean max-Tanimoto test→train | Pearson $r$ | RMSE ($pIC_{50}$) |
|---|---|---|---|
| random | 0.79 | 0.854 | 0.701 |
| scaffold (Bemis–Murcko groups) | 0.72 | 0.833 | 0.752 |
| cluster (Butina, Tanimoto $\ge0.4$ to centroid) | 0.55 | 0.620 | 1.056 |
| *predict the training mean* | | 0 | 1.342 |

Three lessons. **(i) A scaffold split is not a similarity split.** Splitting by exact Murcko scaffold left the nearest-training-neighbor similarity at $0.72$ (versus $0.79$) because analogs with slightly different ring systems have different scaffold strings but are still close, and the reported performance barely changed ($r=0.833$ vs $0.854$). Only the cluster split produced a real extrapolation test: $r=0.62$, RMSE $1.06$, near the level of predicting the mean (1.34). **(ii) Error is a smooth function of similarity to the training set.** On the random split, RMSE grows from 0.58 for test molecules whose nearest training neighbor has Tanimoto 0.80–0.95 to 0.84 at 0.50–0.65 and 1.00 below 0.50. (The bin 0.95–1.0 has RMSE $1.13$: near-duplicates with disagreeing labels, the stereochemistry and cliff cases above.) A model's applicability domain is a *distance*, and reporting a single number hides it. **(iii) The random split is close to the noise ceiling.**

### 24.7.3 The noise ceiling

If the best possible predictor knows the true $pIC_{50}$ exactly but the labels have independent noise of SD $\sigma$, its $R^2$ against the labels is $1-\sigma^2/\mathrm{Var}(y)$ (Chapter 1). With $\sigma=0.54$ and $\mathrm{SD}(y)=1.34$:

$$
R^2_\max=1-\frac{0.54^2}{1.34^2}=0.84,\qquad r_\max=0.92,\qquad \mathrm{RMSE}_\min=0.54.
$$

The random-split model has $r=0.854$ ($R^2=0.73$) and RMSE $0.70$. *If* the BACE-1 labels carry noise comparable to heterogeneous ChEMBL data, the random-split benchmark is within $\sim0.1$ of its ceiling and cannot distinguish a good model from a better one: the implied excess model error is $\sqrt{0.70^2-0.54^2}=0.45$ log units. The benchmark has saturated; the interesting evaluations are the cluster split and prospective tests. (Caveat: the noise level of this particular dataset is not measured; replicate data would give it.)

### 24.7.4 Activity cliffs are where the model stops being a model

We took the out-of-fold random-split predictions and examined the 544 pairs with Tanimoto $\ge0.7$ and true difference $|\Delta pIC_{50}|\ge1.5$. The model gets the *sign* of the difference right for 59% of them (chance is 50%), and its mean predicted difference is $0.54$ against a true difference of $2.08$. A random forest on fingerprints averages the labels of similar training molecules: it is a **smoother**, so by construction it regresses cliffs toward the local mean. The information needed to separate cliff pairs (the 3-D complementarity of the substituent with the pocket, protonation, stereochemistry) is **not in the 2-D fingerprint**, so no amount of capacity or data on the same representation resolves it; the missing information must be added (structure, free-energy calculations, a better assay).

### 24.7.5 Inductive bias: four physical descriptors versus 2,048 hashed substructures

On ESOL (aqueous solubility, $\log S$, SD $2.10$), we compared a linear model on four physical descriptors (Crippen $\log P$, molecular weight, rotatable bonds, aromatic-atom fraction), a random forest on the same four descriptors, and a random forest on the fingerprint:

| Split | Linear, 4 descriptors | RF, 4 descriptors | RF, fingerprint |
|---|---|---|---|
| random 5-fold | 1.016 | 0.739 | 0.905 |
| scaffold 5-fold | 1.051 | 0.892 | **1.463** |

(RMSE in $\log S$ units; single seed.) Under a random split the fingerprint model beats the linear four-descriptor model but is worse than the random forest on the same four descriptors. Under a scaffold split it *degrades by 62%* (0.905 to 1.463) and is worse than the plain linear model, while the descriptor model degrades by 21% (0.739 to 0.892). The reason is representational: a fingerprint is a lookup of substructure environments, so a molecule with new environments has zero features at those bits and the model reverts toward the mean; physically meaningful *continuous* coordinates (lipophilicity, size) are shared across scaffolds and carry the relation (hydrophobicity lowers solubility) to new chemistry. This is the representation lesson of Chapter 13 in miniature, and the reason that a model's "domain knowledge" (here a few lines of Crippen's atom contributions) can be worth more than thousands of learned or hashed features when the test set shifts.

### 24.7.6 The experiment, verbatim

```python
--8<-- "code/ch24_chemistry.py"
```

```text
== 1. One molecule, many strings; one graph, two molecules ==
aspirin: canonical = CC(=O)Oc1ccccc1C(=O)O;  distinct valid SMILES strings in 2000 random draws = 300;  all map back to 1 canonical string
caffeine (24 atoms with H, 14 heavy): distinct SMILES in 2000 draws = 936
carvone enantiomers: CIP labels [(4, 'R')] [(4, 'S')] | Tanimoto (Morgan r=2, chirality ignored) = 1.00; (chirality-aware) = 0.93
same molecular formula and the same 2D graph, identical standard descriptors: True True

== 2. BACE-1 inhibitors: n = 1513, pIC50 mean 6.52, sd 1.34, range [2.5, 10.5] ==
binding free energy  dG = -RT ln(10) pIC50:  1 pIC50 unit = 1.37 kcal/mol; dataset range -14.4 to -3.5 kcal/mol
heavy atoms: mean 34.1; ligand efficiency LE = -dG/N_heavy: mean 0.27, 95th pct 0.37 kcal/mol/atom
corr(pIC50, heavy-atom count) = 0.45  -> size alone explains R^2 = 0.20
distinct Morgan (r<=2) environments in 1513 molecules: 5235; folded to 2048 bits, mean bit occupancy 0.030; 0.80 of occupied bits are shared by >1 distinct environment

== 3. Similarity vs. property difference (all pairs) ==
Tanimoto bin   pairs      mean |dpIC50|   P(|dpIC50|>=1)   P(|dpIC50|>=2)
[0.0, 0.2)      890388       1.58         0.613            0.320
[0.2, 0.3)      162343       1.43         0.590            0.277
[0.3, 0.4)       46580       1.24         0.512            0.212
[0.4, 0.5)       17800       1.01         0.406            0.137
[0.5, 0.6)       12708       0.87         0.333            0.085
[0.6, 0.7)        8609       0.81         0.310            0.069
[0.7, 0.8)        3794       0.70         0.254            0.049
[0.8, 1.0]        1606       0.62         0.199            0.044
pairs with Tanimoto >= 0.7: 5400;  of these, 257 differ by >= 100-fold in IC50 (activity cliffs) and 1282 by >= 10-fold
pairs with identical chirality-blind fingerprints (Tanimoto = 1): 35; mean |dpIC50| = 1.29, 0.60 differ by >= 10-fold; a chirality-aware fingerprint separates 0.60 of them

== 4. Generalization depends on how the test set is chosen (RF on Morgan counts, 5-fold CV) ==
671 Bemis-Murcko scaffolds, 106 Butina clusters (distance cutoff 0.6)
random   : Pearson r = 0.854, RMSE = 0.701 pIC50, mean max-Tanimoto test->train = 0.79
scaffold : Pearson r = 0.833, RMSE = 0.752 pIC50, mean max-Tanimoto test->train = 0.72
cluster  : Pearson r = 0.620, RMSE = 1.056 pIC50, mean max-Tanimoto test->train = 0.55
baseline predicting the mean: RMSE = 1.342
random-split RF, error by similarity to the nearest training molecule:
   nearest-train Tanimoto in [0.00,0.50): n =   37, RMSE = 1.003
   nearest-train Tanimoto in [0.50,0.65): n =   98, RMSE = 0.841
   nearest-train Tanimoto in [0.65,0.80): n =  553, RMSE = 0.724
   nearest-train Tanimoto in [0.80,0.95): n =  751, RMSE = 0.580
   nearest-train Tanimoto in [0.95,1.00): n =   74, RMSE = 1.129

== 5. Can the (random-split) model rank the members of a cliff pair? ==
544 pairs with Tanimoto >= 0.7 and |dpIC50| >= 1.5; model gets the sign of the difference right for 0.59; mean |predicted difference| = 0.54 vs true 2.08

== 6. Noise ceiling ==
if a single reported pIC50 has noise SD 0.54, the best possible R^2 against such labels is 0.84 (r <= 0.92), RMSE >= 0.54

== 7. ESOL aqueous solubility, n = 1128: four physical descriptors vs. 2048-d fingerprint ==
random    RMSE (logS): linear on 4 descriptors 1.016 | RF on 4 descriptors 0.739 | RF on fingerprint 0.905   (sd of logS = 2.10)
scaffold  RMSE (logS): linear on 4 descriptors 1.051 | RF on 4 descriptors 0.892 | RF on fingerprint 1.463   (sd of logS = 2.10)
```

!!! lens "Research lens: what these results say about molecular ML"
    (1) *Always report a similarity-stratified error and a cluster-split result*, not just a random split. (2) *Compute the noise ceiling before comparing models.* (3) *Include a size-only baseline*: heavy-atom count alone gives $R^2=0.20$ for BACE-1 potency. (4) *Cliff pairs diagnose whether a model has learned 3-D SAR or only local smoothing.* (5) *A simple physical-descriptor baseline is a necessary control for any claimed representation learning.* Each is a cheap experiment that routinely changes the conclusion of a paper (A5, the evaluation attack, in the language of Chapter 55).

---

## 24.8 Structure-based methods: docking, scoring, and free-energy perturbation

**Docking** searches for the ligand's pose (position, orientation, conformation) in a protein binding site and ranks poses by a **scoring function** $S(\text{pose})$ that approximates the binding free energy. A rigorous decomposition is

$$
\Delta G_\text{bind}=\underbrace{\Delta E_\text{gas}}_{\text{contacts, H-bonds}}+\underbrace{\Delta G_\text{solv}(PL)-\Delta G_\text{solv}(P)-\Delta G_\text{solv}(L)}_{\text{desolvation}}-T\,\Delta S_\text{conf}-T\,\Delta S_\text{trans/rot}.
$$

Each term is large (tens of kcal/mol) and mostly cancels, and a **1.4 kcal/mol** error in the sum is an *order of magnitude* error in $K_d$. Fast scoring functions (empirical, knowledge-based, or learned) approximate this with few terms and a rigid receptor; they are reasonable at pose ranking but poor at *affinity ranking across chemically diverse ligands*, and their accuracy depends on protonation states, structural water, and receptor flexibility, all poorly handled. In practice docking is used as an *enrichment* tool (a modest improvement over random selection from a library), not a measurement.

**Free-energy perturbation (FEP)** computes *relative* binding free energies between two similar ligands $A$ and $B$ by alchemically transforming one into the other. Because $\Delta G$ is a state function, the thermodynamic cycle

$$
\begin{array}{ccc}
P+A & \xrightarrow{\ \Delta G_\text{bind}(A)\ } & PA\\
\Big\downarrow{\scriptstyle\Delta G_\text{solv}(A\to B)} & & \Big\downarrow{\scriptstyle\Delta G_\text{prot}(A\to B)}\\
P+B & \xrightarrow{\ \Delta G_\text{bind}(B)\ } & PB
\end{array}
$$

gives

$$
\Delta\Delta G_\text{bind}(A\to B)=\Delta G_\text{bind}(B)-\Delta G_\text{bind}(A)=\Delta G_\text{prot}(A\to B)-\Delta G_\text{solv}(A\to B).
$$

The two transformation free energies on the right are *computable* by molecular-dynamics sampling along a path of intermediate (non-physical) states (using, e.g., the Zwanzig or Bennett acceptance ratio estimators), and the hard-to-compute absolute terms cancel. With modern force fields and careful setup, prospective studies reported mean unsigned errors of about 1 kcal/mol for congeneric series (Wang et al., 2015) [[S]], roughly a 5-fold potency error: much better than docking, at a cost of GPU-hours per transformation, and subject to the quality of the force field, the starting structure, and sampling. FEP supplies precisely what the activity-cliff analysis found missing (3-D, physics) at orders of magnitude greater cost, which is why *combining* cheap ML triage with FEP on the shortlist is a standard industrial workflow, and why ML models that learn to approximate FEP-quality predictions (Chapter 37) are attractive.

---

## 24.9 Worked research examples

!!! example "Worked Research Example 24.1: A deep model of protein–ligand binding that may not look at the protein"
    **Situation.** A paper reports a structure-based deep network for binding-affinity prediction with $r=0.84$ on a standard benchmark, outperforming docking. The input is a protein–ligand complex graph. A colleague notes that heavy-atom count alone gives $R^2\approx0.2$ on one target and asks whether the model's performance is about proteins at all.

    **Question.** Design the experiments that separate "learned protein–ligand interactions" from "learned ligand priors, size, or leakage."

    **Reasoning.**

    1. *What can produce a high correlation without learning interactions?* (a) The test complexes have near-duplicates in training (§24.6, Graber et al.). (b) Affinity correlates with ligand size and properties across *the whole dataset*, so a ligand-only model ranks well on a diverse benchmark where proteins differ in typical affinity. (c) Protein identity alone predicts the typical affinity (a protein with many potent ligands in the data), so a protein-only model works. (d) Label noise or duplicates.
    2. *Controls that isolate each.* Ligand-only and protein-only ablations; a **size-only** baseline; a **shuffled-pairing control** (permute the protein among complexes of the same type and retrain: if performance holds, the model ignores interactions); a similarity-filtered or time-split test set; compare to a nearest-neighbor baseline by similarity (a "memorize the nearest training complex" model).
    3. *Predictions.* If the model learns interactions, performance should be *lost by shuffling* and degrade gracefully with decreasing similarity to training data; if it learns priors, ablations will retain most of the performance and the filtered test set will collapse.
    4. *The decisive evidence.* A prospective test on novel complexes (new protein families, new chemotypes) with a pre-registered metric, and a within-series ranking test (does it rank *analogs against the same protein* correctly, the task that matters in lead optimization).

    **Expert analysis.** The ligand-only ablation is the molecular version of the "the model might not use the input you care about" problem (Chapters 18, 43). A model that is excellent *between* targets and uninformative *within* a target series is useless for lead optimization, but would score well on a benchmark that mixes targets. The right metric is the one that matches the decision: within-series Spearman correlation per target, averaged across targets, with confidence intervals reflecting the number of *series*, not molecules (Chapter 4, pseudoreplication).

!!! example "Worked Research Example 24.2: A new molecular representation improves cross-validated $r$ from 0.85 to 0.88. Is that progress?"
    **Situation.** A group proposes a pretrained graph transformer for molecular property prediction. On BACE-1 it improves the Pearson $r$ over a random-forest/fingerprint baseline from 0.854 to 0.88 under random splits, with a bootstrap interval across molecules that excludes zero.

    **Question.** What would convince you that the representation is better rather than better at fitting this benchmark?

    **Reasoning.**

    1. *Compute the ceiling.* With noise SD $\sigma=0.54$, $r_\max\approx0.92$ and the baseline is already at 93% of it. A gain of 0.03 in $r$ corresponds to a change in RMSE of about 0.05 log units, a small fraction of the measurement noise. *The benchmark cannot resolve this difference scientifically*, whatever its p-value.
    2. *What is the unit of replication?* Molecules in the same series share SAR; the effective sample size is closer to the number of *clusters* (106) than molecules (1,513). A bootstrap by molecule understates uncertainty; bootstrap by cluster.
    3. *Where would a better representation help?* In the *out-of-distribution* regime: cluster splits, time splits, new targets, and cliff pairs. The baseline gets $r=0.62$ on the cluster split; a representation that learned something general would improve *there*.
    4. *Control for pretraining leakage.* If the pretraining corpus (e.g., ChEMBL or PubChem) contains the BACE-1 test compounds or their analogs, the representation has seen the neighborhood. Check for overlap by similarity.
    5. *Add the simple baselines.* Descriptor-based linear and RF models; a size-only model; and a *k*-nearest-neighbor model.

    **Expert analysis.** Claims of representation improvement should be evaluated on the *axis along which representations differ*: extrapolation and transfer. Random-split gains near the noise ceiling are ordinarily not evidence about representations. A concrete way to report: "Δ in cluster-split $r$ with cluster-bootstrap CI; Δ on cliff pairs; ceiling-normalized gain." This protocol is cheap to run and would change the interpretation of a large part of the published molecular-ML literature (A5 and A4, evaluation and data attacks).

---

## 24.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: before you trust (or write) a molecular ML result"
    **Setting.** You have a result: a model predicts potency, solubility, toxicity, or docking score better than a baseline.

    **Ten questions, in order.**

    1. **What is the label?** Which function of which latent variable ($K_d$, $K_i$, $IC_{50}$, % inhibition, cell viability, DEL enrichment) under which assay conditions? Are labels pooled across assays, and does the pooling depend on the substrate concentration (Cheng–Prusoff)?
    2. **What is the noise ceiling?** Is there replicate information? What is the best possible $R^2$ or AUROC? Is the benchmark already at it?
    3. **What is the unit of replication?** Compounds, series, scaffolds, or targets? Are the reported confidence intervals computed at that unit?
    4. **How was the split made?** Random, scaffold, cluster, time, target? What is the distribution of nearest-training-neighbor similarity of the test set? Report error versus that similarity.
    5. **What are the null models?** Size-only, descriptor-linear, kNN on fingerprints, ligand-only, protein-only, shuffled-protein.
    6. **What information does the representation lack?** Stereochemistry, protonation, conformation, the receptor. If the answer to "what separates these two molecules" is not in the input, no model on that input can use it.
    7. **Where are the decoys from?** Property-matched? Did the decoys leak the label?
    8. **Is the task the decision?** Within-series ranking, scaffold hopping, or enrichment from a library? Are these the same distribution as the benchmark?
    9. **Does it survive a prospective test?** A pre-specified list of compounds synthesized or purchased and assayed *blind*, with a stated metric.
    10. **What is the effect on the downstream decision?** Hit rate, number of cycles saved, cost, and whether the target hypothesis was the binding constraint.

    **What it teaches.** The questions are an instance of the Expert Chain (L3 assumptions, L5 failure modes, L10 experiments) applied to evaluation. Their answers frequently differ from the headline claim by more than any architectural improvement.

    **An open question to carry forward.** Could a model that observes *only 2-D chemical structure* ever be expected to predict potency across a cliff? Formalize "the information a representation retains about the label" with the sufficiency bound of Chapter 17 ($\mathrm{MI}(s(x);y)\le\mathrm{MI}(x;y)$) and design an experiment that measures the gap by adding a single 3-D feature (for example, a docked-pose contact count) to the fingerprint and measuring the change on cliff pairs only.

---

## 24.11 Connections

- **Backward:** the noise ceiling and leakage (Chapters 1, 7); Weisfeiler–Lehman, message passing, and chirality (Chapter 16); representation sufficiency (Chapters 13, 17); thermodynamic sigmoids (Chapters 22, 23); pseudoreplication (Chapter 4).
- **Forward:** classical molecular models and the physics baseline (Chapter 29); molecular ML and drug discovery, including generative molecular design (Chapter 37); perturbation-based target discovery (Chapter 39); benchmark construction and leakage (Chapter 43); experimental design and active learning for compound selection (Chapter 46); AI-driven discovery pipelines (Chapter 54).

!!! takeaways "Key takeaways"
    1. A molecule is a graph *plus* stereochemistry, protonation state, and a conformer ensemble; SMILES strings are non-unique (300 distinct strings for aspirin in 2,000 draws), and 2-D representations cannot distinguish enantiomers.
    2. **Binding is a two-state sigmoid**: $\theta=[L]/(K_d+[L])$; $\Delta G^\circ=-1.37\ \text{kcal/mol}\times pK_d$; $IC_{50}=K_i(1+[S]/K_m)$ for competitive inhibitors, so pooled $pIC_{50}$ data mix different link functions; reported noise is $\sim0.5$ log units.
    3. **Morgan fingerprints are Weisfeiler–Lehman color refinement** with a hash; they have a horizon of $R$ bonds, fold with heavy collisions (80% of occupied bits shared in our data), and are chirality-blind by default.
    4. The discovery funnel is a cascade of increasingly expensive measurements; the dominant clinical loss is the *target hypothesis* (efficacy), not the molecule, so molecular ML addresses the cheaper, earlier part.
    5. Public bioactivity data consist of **congeneric series** (14 compounds per cluster here), noisy heterogeneous assays, missing negatives, and benchmark leakage (DUD-E ligand bias; PDBbind–CASF overlap).
    6. **The split decides the result**: on BACE-1, random $r=0.85$, scaffold $0.83$, cluster $0.62$; error rises smoothly as similarity to training falls; the random-split benchmark is near its noise ceiling ($r_\max\approx0.92$).
    7. Fingerprint models smooth over neighbors: they get the sign of only 59% of activity cliffs and predict a quarter of the true difference. Missing 3-D and physics information must be added, not learned from 2-D.
    8. A four-descriptor physical baseline degraded 21% under scaffold shift while the fingerprint model degraded 62%: *inductive bias is worth most under shift*.
    9. Physics (FEP, $\sim1$ kcal/mol) and ML are complementary: ML triages cheaply, physics refines expensively.

---

## Further reading

- Weininger, D. (1988). SMILES, a chemical language and information system. *J. Chem. Inf. Comput. Sci.* 28, 31–36. Morgan, H. L. (1965). The generation of a unique machine description for chemical structures. *J. Chem. Doc.* 5, 107–113. Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.* 50, 742–754.
- Xu, K., Hu, W., Leskovec, J. & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*. Ralaivola, L., Swamidass, S. J., Saigo, H. & Baldi, P. (2005). Graph kernels for chemical informatics. *Neural Networks* 18, 1093–1110.
- Cheng, Y. & Prusoff, W. H. (1973). Relationship between the inhibition constant ($K_i$) and the concentration of inhibitor which causes 50 per cent inhibition ($I_{50}$) of an enzymatic reaction. *Biochem. Pharmacol.* 22, 3099–3108. Hopkins, A. L., Groom, C. R. & Alex, A. (2004). Ligand efficiency: a useful metric for lead selection. *Drug Discov. Today* 9, 430–431. Copeland, R. A., Pompliano, D. L. & Meek, T. D. (2006). Drug–target residence time and its implications for lead optimization. *Nat. Rev. Drug Discov.* 5, 730–739.
- Kramer, C., Kalliokoski, T., Gedeck, P. & Vulpetti, A. (2012). The experimental uncertainty of heterogeneous public $K_i$ data. *J. Med. Chem.* 55, 5165–5173. Kalliokoski, T., Kramer, C., Vulpetti, A. & Gedeck, P. (2013). Comparability of mixed IC$_{50}$ data: a statistical analysis. *PLoS ONE* 8, e61007. Landrum, G. A. & Riniker, S. (2024). Combining IC$_{50}$ or $K_i$ values from different sources is a source of significant noise. *J. Chem. Inf. Model.*
- Martin, Y. C., Kofron, J. L. & Traphagen, L. M. (2002). Do structurally similar molecules have similar biological activity? *J. Med. Chem.* 45, 4350–4358. Maggiora, G. M. (2006). On outliers and activity cliffs: why QSAR often disappoints. *J. Chem. Inf. Model.* 46, 1535. Stumpfe, D. & Bajorath, J. (2012). Exploring activity cliffs in medicinal chemistry. *J. Med. Chem.* 55, 2932–2942.
- Bohacek, R. S., McMartin, C. & Guida, W. C. (1996). The art and practice of structure-based drug design: a molecular modeling perspective. *Med. Res. Rev.* 16, 3–50. Ruddigkeit, L. et al. (2012). Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17. *J. Chem. Inf. Model.* 52, 2864–2875. Lyu, J. et al. (2019). Ultra-large library docking for discovering new chemotypes. *Nature* 566, 224–229. Brenner, S. & Lerner, R. A. (1992). Encoded combinatorial chemistry. *PNAS* 89, 5381–5383. Erlanson, D. A. et al. (2016). Twenty years on: the impact of fragments on drug discovery. *Nat. Rev. Drug Discov.* 15, 605–619. Lipinski, C. A. et al. (1997). *Adv. Drug Deliv. Rev.* 23, 3–25.
- Hay, M. et al. (2014). Clinical development success rates for investigational drugs. *Nat. Biotechnol.* 32, 40–51. Wong, C. H., Siah, K. W. & Lo, A. W. (2019). Estimation of clinical trial success rates and related parameters. *Biostatistics* 20, 273–286. Sun, D. et al. (2022). Why 90% of clinical drug development fails and how to improve it? *Acta Pharm. Sin. B* 12, 3049–3062.
- Chen, L. et al. (2019). Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening. *PLoS ONE* 14, e0220113. Graber, D. et al. (2025). Resolving data bias improves generalization in binding affinity prediction. *Nat. Mach. Intell.* 7. Wang, L. et al. (2015). Accurate and reliable prediction of relative ligand binding potency in prospective drug discovery by way of a modern free-energy calculation protocol and force field. *J. Am. Chem. Soc.* 137, 2695–2703.
- Delaney, J. S. (2004). ESOL. *J. Chem. Inf. Comput. Sci.* 44, 1000–1005. Subramanian, G., Ramsundar, B., Pande, V. & Denny, R. A. (2016). Computational modeling of β-secretase 1 (BACE-1) inhibitors using ligand based approaches. *J. Chem. Inf. Model.* 56, 1936–1949. Wu, Z. et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.* 9, 513–530. Blevins, A. et al. (2024). BELKA: the Big Encoded Library for Chemical Assessment (NeurIPS 2024 competition dataset).
