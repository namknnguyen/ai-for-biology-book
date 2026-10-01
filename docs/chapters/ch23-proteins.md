# Chapter 23. Proteins: Sequence, Structure, Function, and Evolution

!!! abstract "Chapter at a glance"
    **Motivation.** Proteins are the molecular machines of the cell and the main targets of drug discovery and design. Protein language models, structure predictors, and generative design methods (Chapters 34–37) all operate on a small number of data sources and rest on a few physical and evolutionary facts. This chapter teaches those facts *for modeling*: what a protein sequence encodes, how stability and function relate to sequence, which experiments produce the data, and why those data are biased.
    **Prerequisites.** Chapters 4, 5, 19, 21.
    **You will be able to:** (1) describe amino-acid chemistry, backbone geometry, and the hierarchy of structure; (2) explain folding as thermodynamics (Anfinsen, funnels, marginal stability) and derive the sigmoid relation between stability and fraction folded; (3) explain **global epistasis** and why additive models on the measured scale fail; (4) describe how structures and function are measured (X-ray, cryo-EM, NMR, deep mutational scanning) and their biases; (5) explain homology, MSAs, and coevolution as the evolutionary information in a sequence; (6) state what a model can learn from protein sequence data and what it cannot observe.

---

## 23.1 The seven questions for a protein

!!! bio "Biology for modeling: a protein"
    **What is it?** A polymer of amino acids (typically 100–1,000 residues; median ~350–400 in eukaryotes) that folds into a three-dimensional structure (or remains partly disordered) and carries out a function: catalysis, binding, transport, structure, signaling.
    **Information it contains.** A *sequence* over 20 amino acids that specifies, together with the cellular environment, an ensemble of conformations; function arises from structure plus dynamics. Evolutionary history is written in the sequence's relationship to its homologs.
    **How is it generated?** Translation of mRNA by the ribosome, followed by co-translational folding assisted by chaperones; sequences evolve by mutation under selection for stability, function, and expression.
    **How is it measured?** *Sequence*: from genomes and mass spectrometry. *Structure*: X-ray crystallography, cryo-EM, NMR. *Function and stability*: biochemical assays and **deep mutational scanning (DMS)** (§23.7).
    **Computational representation.** A string over 20 letters; an MSA (a matrix of aligned homologs); 3-D coordinates (atoms, residue frames, torsion angles), distance maps, graphs, surfaces; embeddings from language models.
    **What varies.** Between proteins (families, folds), within families (homologs, variants), between conformational states (ensembles), and between cellular contexts (modifications, partners).
    **What can ML learn?** Patterns of evolutionary constraint, coevolution, sequence–structure relationships for natural-like proteins, statistical regularities of function within families.
    **What can ML not observe?** Thermodynamic ensembles and kinetics (a static structure hides dynamics); behavior in the cell; function for families with no evolutionary data; the effect of conditions not sampled.

---

## 23.2 Amino acids and chemistry

Twenty standard amino acids share a backbone (N–C$\alpha$–C′) and differ in the **side chain** attached to C$\alpha$. For modeling, group them by properties:

| Class | Residues | Role |
|---|---|---|
| Hydrophobic (aliphatic, aromatic) | Ala, Val, Leu, Ile, Met, Phe, Trp, Tyr (Pro, Gly special) | Buried in the core; drive folding via the hydrophobic effect |
| Polar uncharged | Ser, Thr, Asn, Gln, Cys | Surface and active sites; H-bonding; Cys forms disulfides |
| Charged | Asp, Glu (−); Lys, Arg, His (+) | Surface; salt bridges; catalysis; binding |
| Special backbone | Gly (no side chain; flexible), Pro (ring; rigid) | Turns and loops |

**Backbone geometry.** The peptide bond is planar (torsion $\omega\approx180^\circ$ trans). The backbone conformation of each residue is described by two torsion angles, $\phi$ and $\psi$, whose allowed combinations occupy a few regions of the **Ramachandran plot** (helical, extended, left-handed); *steric exclusion alone removes most of $(\phi,\psi)$ space*. Side chains add torsions $\chi_1,\chi_2,\dots$ with a small number of preferred **rotamers**. A protein of $L$ residues therefore has $\sim2L$ backbone and $\sim2$–$3L$ side-chain torsional degrees of freedom: a space of dimension $\sim5L$ in which the folded structure is a very specific point.

**Post-translational modifications** (phosphorylation, glycosylation, ubiquitination, methylation, acetylation, ...) change protein properties after synthesis and are invisible in the genome-derived sequence. Amino-acid chirality: natural proteins are built from L-amino acids; a mirror-image protein is a *different* molecule (Chapter 16's chirality warning).

---

## 23.3 Folding as thermodynamics

### 23.3.1 Anfinsen, Levinthal, and funnels

**Anfinsen's thermodynamic hypothesis** (Nobel Prize 1972): for many small proteins, the native structure is the minimum of the free energy under physiological conditions, determined by the amino-acid sequence alone (shown by refolding denatured ribonuclease *in vitro*). This is the premise behind sequence→structure prediction. **Levinthal's paradox**: a protein cannot sample all conformations (even three states per residue gives $3^{100}\approx10^{47}$ for a 100-residue protein) yet folds in microseconds to seconds, so folding must be guided. The resolution is the **energy landscape funnel** (Onuchic et al., 1997): the landscape is biased toward the native state, with local roughness but a global slope. Real proteins also include exceptions: chaperone-assisted folding, kinetically trapped or metamorphic proteins (fold-switchers), and intrinsically disordered proteins.

### 23.3.2 Marginal stability and the sigmoid

The native state is stable by a *small margin*. Let $\Delta G$ be the folding free energy (negative = stable). In the two-state approximation the fraction of molecules in the folded state is

$$
f_\text{folded}=\frac{1}{1+e^{\Delta G/RT}},\qquad RT=0.593\ \text{kcal/mol at }298\,\text{K}.
$$

Typical globular proteins have $\Delta G\approx-5$ to $-10$ kcal/mol, which results from large opposing contributions (the favorable hydrophobic effect, packing, and hydrogen bonding; the unfavorable loss of conformational entropy). The sigmoid has consequences:

| $\Delta G$ (kcal/mol) | $-8$ | $-5$ | $-3$ | $-1$ | $0$ | $+1$ |
|---|---|---|---|---|---|---|
| Fraction folded | 1.0000 | 0.9998 | 0.9937 | 0.8437 | 0.5000 | 0.1563 |

*Saturation:* a protein with $\Delta G=-5$ is essentially fully folded; a mutation that destabilizes it by 2 kcal/mol (to $-3$) barely changes the fraction folded (0.9998 → 0.9937). *The measured phenotype is insensitive to stability in the stable regime and exquisitely sensitive near $\Delta G\approx0$.*

**Mutational effects on stability.** Random missense mutations have a mean $\Delta\Delta G\approx+1$ kcal/mol (destabilizing) with a long tail; a minority are stabilizing. Since most natural proteins have a stability reserve of several kcal/mol, they tolerate several mutations before unfolding (*threshold robustness*; Bloom et al., 2006), which is why neutral drift is possible and why *most single mutations have little effect on stability-dependent phenotypes while a few have a large effect*.

### 23.3.3 Global epistasis: derived and tested

Suppose each mutation $m$ shifts the latent stability by an *additive* amount $\Delta\Delta G_m$, so a genotype with mutations $\{m_1,\dots,m_k\}$ has $\Delta G=\Delta G_\text{wt}+\sum_j\Delta\Delta G_{m_j}$ (a good first approximation for distant sites). The *measured* phenotype is a nonlinear function of the latent value: $f=g(\Delta G)$, with $g$ the sigmoid above (or, for an enzyme with a stability requirement, a product of stability and intrinsic activity). Then **epistasis appears on the measured scale even though the latent model is perfectly additive**: for two mutations A and B,

$$
\epsilon_{AB}=f_{AB}-(f_A+f_B-f_\text{wt})\ne0\quad\text{generically, because }g\text{ is nonlinear.}
$$

This is **global epistasis** (Sailer & Harms, 2017; Otwinowski et al., 2018; Starr & Thornton, 2016): non-specific, nonlinearity-induced interactions, in contrast to *specific epistasis* arising from physical contacts between residues. Two consequences: (i) the *shape* of the genotype–phenotype map is partly a property of the *assay's response function*, and (ii) an additive model on the measured scale systematically fails for multi-mutants, while an additive latent model with a learned nonlinearity succeeds.

**Experiment** (`code/ch23_global_epistasis.py`). A 40-site, 20-amino-acid synthetic protein with additive $\Delta\Delta G$ (mean $\sim+0.96$ kcal/mol; 15% stabilizing). We train on all single mutants plus 3,000 random double mutants and test on held-out 3- to 6-fold mutants, comparing (a) an additive model on the measured scale (least squares) and (b) an additive latent stability model passed through a learned sigmoid.

| Wild-type $\Delta G$ | Pairs with $\lvert\epsilon\rvert>0.05$ | 3-fold | 4-fold | 5-fold | 6-fold |
|---|---|---|---|---|---|
| **−4.0** (stable) | 31% | add 0.75 / GE 0.89 | add 0.53 / GE 0.70 | add 0.34 / GE 0.54 | add 0.10 / GE 0.35 |
| **−1.5** (marginal) | **58%** | add 0.42 / **GE 0.97** | add −1.51 / **GE 0.94** | add −9.06 / **GE 0.89** | add −19.78 / **GE 0.84** |

(“add” = additive on the measured scale; “GE” = additive latent plus sigmoid; held-out $R^2$.) **For a marginally stable protein, the additive model on the measured scale collapses (negative $R^2$: worse than predicting the mean) while the global-epistasis model remains at 0.84–0.97**, because the training singles and doubles span the sigmoid's sloping region and identify the latent effects. **For a very stable protein, both degrade**: single and double mutants sit on the saturated plateau where the data barely reveal the latent effects (the sigmoid hides $\Delta\Delta G$), so extrapolation to many mutations is poorly constrained even for the correct model. *Identifiability depends on where the training data lie on the response function.*

```python
--8<-- "code/ch23_global_epistasis.py"
```

!!! rhyme "Structural rhyme: global epistasis ↔ generalized linear models ↔ neural network with one hidden nonlinearity ↔ the objective gap"
    "Additive latent + learned nonlinear link" is a **generalized linear model**. The best predictor of an *assay readout* often needs the latent variable (stability, binding energy) *and* the link (the assay's response). A protein language model's log-likelihood is a latent score; its relationship to an assay readout is another nonlinear link, which explains why *zero-shot correlations* (rank-based, link-invariant) are used for DMS benchmarks and why *supervised fine-tuning* adds the link (Chapter 34). The same structure underlies many biological measurements: **latent mechanism → saturating or thresholded readout**.

---

## 23.4 Structure: the hierarchy and its measurement

**Levels.** *Primary*: sequence. *Secondary*: local regular structures, **$\alpha$-helices** (3.6 residues per turn; $i\to i+4$ hydrogen bonds), **$\beta$-strands/sheets** (extended; hydrogen bonds between strands), turns and loops. *Tertiary*: the full 3-D arrangement of a chain (compact, usually with a hydrophobic core). *Quaternary*: assemblies of multiple chains. A **domain** is a compact, often independently folding unit (typically 50–250 residues); proteins are often multi-domain, linked by flexible linkers. Classification databases (CATH, SCOP) catalog on the order of $10^3$ distinct **folds** (counts depend on the classification): the number of folds is far smaller than the number of families, suggesting a limited repertoire of stable topologies.

**Intrinsic disorder.** A large fraction of eukaryotic protein sequence (roughly a third of residues; many proteins with long disordered regions) lacks a stable fold and functions as an *ensemble*: signaling hubs, linkers, regions in condensates. Predicted structure confidence from AlphaFold-type models correlates with order (low-confidence regions are enriched for disorder), but **a single predicted structure is not the right object for a disordered region** (Chapter 35).

**Dynamics and allostery.** Proteins sample an ensemble of conformations on timescales from ps to seconds. Function often requires conformational change (enzyme catalysis, transporters, kinases, GPCR activation). **Allostery**, regulation of one site by binding at another, couples distant residues. A model trained to predict a *single static structure* captures one state of an ensemble.

**How structures are measured, and what each cannot see.**

| Method | What it gives | Resolution | Limitations / biases |
|---|---|---|---|
| **X-ray crystallography** | Electron density of a crystal | 1–3 Å typical | Requires crystals; crystal packing; static average; flexible/disordered regions missing; **bias toward soluble, stable, crystallizable proteins** |
| **Cryo-EM (single particle)** | 3-D density from images of frozen particles | 2–4 Å for good cases | Particle size and heterogeneity; local resolution variation; flexible regions blurred; sample preparation artifacts |
| **NMR** | Distance restraints, dynamics in solution | atomic for small proteins | Size limit (tens of kDa); ensembles |
| **SAXS, HDX-MS, cross-linking MS, FRET** | Low-resolution shape, dynamics, contacts | coarse | Indirect |
| **Computational prediction** (AlphaFold, ESMFold) | Predicted coordinates | model-dependent | Trained on the biased PDB; single-state; accuracy varies (Chapter 35) |

The **Protein Data Bank (PDB)** contains on the order of $2\times10^5$ experimental structures, with the AlphaFold Database adding $>2\times10^8$ predicted models and the ESM Metagenomic Atlas $\sim6\times10^8$ predictions. **PDB biases:** (i) *taxonomic and functional*: well-studied, soluble, abundant, stable proteins; membrane proteins and large disordered proteins are under-represented; (ii) *redundancy*: thousands of near-identical entries (lysozyme, kinases, antibodies, proteases) so naive splits leak (Worked Example 23.2); (iii) *conformational*: one or a few states per protein, often with a bound ligand; (iv) *experimental conditions* (pH, cryo, crystallization additives) differ from the cell.

---

## 23.5 Evolutionary information: homologs, MSAs, and coevolution

A protein's **homologs** (proteins with common ancestry) are found by sequence search (BLAST, HMMER, MMseqs2; Chapter 27) and aligned into a **multiple sequence alignment (MSA)**, a matrix of $N$ sequences by $L$ positions. Three kinds of information live in an MSA:

1. **Conservation** (per-position variability): constrained positions (catalytic residues, the structural core) vary little (Chapter 5: position-specific information).
2. **Substitution patterns**: which amino acids replace which at each site.
3. **Coevolution** (pairwise dependence between positions): residues in contact tend to co-vary, because a destabilizing mutation at one position is compensated by a mutation at its contact partner. Statistical models (Potts/direct-coupling analysis; Chapter 29) separate *direct* couplings (contacts) from *indirect* correlations (transitive effects) and predicted residue contacts accurately for families with deep alignments, leading to the first structure predictions from evolutionary data alone (Marks et al., 2011; Morcos et al., 2011) and ultimately to AlphaFold's MSA-based Evoformer (Chapter 35).

**The evolutionary signal is limited by $N_\text{eff}$** (Chapter 21): a family with few effectively independent sequences ("orphans"; the "dark proteome") has little coevolutionary signal. Over $10^4$ families are catalogued (Pfam ~$2\times10^4$), a large fraction of metagenomic proteins have no annotated homologs, and *a model relying on MSAs will fail on proteins without them*, whereas single-sequence models (protein language models, Chapter 34) attempt to internalize the same statistics.

---

## 23.6 Function: binding, catalysis, and fitness

**Binding.** Described by the dissociation constant $K_d$ (concentration at which half of the binding sites are occupied), $\Delta G_\text{bind}=RT\ln K_d$ (Chapter 24). Antibody–antigen and drug–target interactions typically span $K_d\sim10^{-12}$–$10^{-6}$ M; protein–protein interactions span $\sim10^{-9}$–$10^{-3}$ M (weak transient to tight complexes).

**Catalysis.** For enzymes, $k_\text{cat}$ (turnover) and $K_M$ (substrate affinity) characterize activity; the **catalytic efficiency** $k_\text{cat}/K_M$ has a median near $10^5$ M$^{-1}$s$^{-1}$ across known enzymes (typical $k_\text{cat}\sim10$ s$^{-1}$), far below the diffusion limit ($10^8$–$10^9$): most enzymes are "moderately efficient" (Bar-Even et al., 2011).

**From molecular function to fitness.** Organismal fitness depends on molecular function through a *nonlinear, saturating* map (metabolic control theory): a 50% reduction in an enzyme's activity often has little effect on flux. The fitness landscape of a protein is therefore *plateau-and-cliff* shaped: neutral drift over plateaus, strong selection at cliffs. In the laboratory, a **DMS assay** defines its own map from sequence to a readout (growth rate under selection; FACS-based binding or surface expression; protease resistance as a stability proxy), which is *not* the natural fitness. *Assay choice is part of the definition of the target* (Chapter 5, §5.6.4 item 5).

!!! lens "Research lens: what a protein model assumes"
    **Assumes:** natural sequences are samples from an equilibrium of mutation–selection–drift (Chapter 5); evolutionary constraint reflects function and stability; structure and function are determined by sequence in the relevant environment. **Information used:** statistics of natural sequences (and structures, if trained on them). **Information ignored:** conformational ensembles, kinetics, cellular context, post-translational state, expression level and localization. **Failure modes:** orphan proteins; engineered or de novo proteins outside natural statistics; effects dominated by factors not constrained by evolution (e.g., in vitro stability for a protein with large stability margin); conformational changes; functions that evolve under selection not captured by the assay.

---

## 23.7 Deep mutational scanning

**DMS** (Fowler & Fields, 2014) measures the effect of thousands of mutations in one experiment. Protocol: (1) construct a library containing all single (and sometimes multiple) amino-acid substitutions of a protein; (2) express it in a system where function is coupled to **selection** (cell growth, phage or yeast display, FACS sorting by binding or surface expression, protease stability); (3) sequence the library before and after selection; (4) compute per-variant **scores** from enrichment of each variant, usually normalized so wild type is 1 and null/nonsense is 0.

| Aspect | Typical value / issue |
|---|---|
| Scale | $19L$ single substitutions ($\sim2\times10^3$–$10^4$ variants per protein); combinatorial libraries up to $10^5$–$10^8$ |
| Replicate reproducibility | Pearson $r\approx0.8$–$0.95$ between replicates for well-designed assays (noise ceiling) |
| Bias | Library coverage; codon/PCR bias; selection-pressure dependence; non-linear response (global epistasis, §23.3.3) |
| Distance from native function | Assays measure *a proxy* (display, growth), usually in a heterologous host |
| Repositories and benchmarks | MaveDB; ProteinGym (a curated benchmark of hundreds of DMS assays; Notin et al., 2023) |

DMS data are the main *experimental benchmark* for protein models (Chapters 34, 35, 42) and the main source of *supervised* labels for adaptation. Their limitations (§23.6) mean a model's performance on DMS data measures its agreement with *that assay's* map, which correlates only partially with fitness or clinical impact.

---

## 23.8 Worked research examples

!!! example "Worked Research Example 23.1: Additive models fail on multi-mutants: is the protein epistatic?"
    **Situation.** A group fits an additive model to single and double mutant DMS data for an enzyme and finds that it predicts triple and higher-order mutants poorly. They conclude that "extensive specific epistasis between residues" governs the landscape and propose a pairwise-interaction neural network.

    **Question.** How would you test whether the failure is specific epistasis or a nonlinear assay response?

    **Reasoning.**

    1. *What generates non-additivity?* (i) *Specific* epistasis (physical coupling, compensation, contacts); (ii) *global* epistasis from a nonlinear response of the readout to an additive latent trait (§23.3.3).
    2. *What does the simulation show?* A perfectly additive latent model with a sigmoid produces apparent epistasis in 31–58% of random pairs (mean $|\epsilon|>0.05$) and makes an additive-on-measured-scale model fail catastrophically on multi-mutants ($R^2=-19.8$ at 6-fold for a marginally stable protein), while the correct *latent additive + link* model recovers $R^2=0.84$. The failure alone does not distinguish (i) from (ii).
    3. *Discriminating experiments.* (a) Fit an additive latent model with a monotonic nonlinear link (global-epistasis model) and test whether the *residual* non-additivity is concentrated in specific pairs (contacts, coevolving residues); (b) check that apparent epistasis *depends on the background's position on the response curve* (it should change sign and magnitude as the wild-type stability changes); (c) measure a *second* assay for the same variants (e.g., stability and activity) whose latent traits differ: global epistasis predicts correlated nonlinearity across assays with different links; (d) test whether residual epistasis correlates with 3-D contact.
    4. *Predictions.* Under global epistasis only: after fitting a link, residuals are small and unstructured; under specific epistasis: residuals correlate with structural contacts and coevolution scores.

    **Expert analysis.** The pairwise-interaction network would fit the data whether the cause is global or specific, but would encode *assay-specific nonlinearity as pairwise interactions*, harming generalization to other assays and backgrounds. Fit the simplest mechanistically motivated model first (latent additive + link) and attribute only the *remaining* structure to specific interactions. This is the research habit of testing the *simplest alternative explanation first*.

!!! example "Worked Research Example 23.2: A structure model trained on random PDB splits \"generalizes\" to new proteins"
    **Situation.** A structure-prediction model is trained on 80% of PDB chains and tested on the held-out 20%, reaching a median C$\alpha$ RMSD of 1.2 Å. The authors claim generalization to unseen proteins.

    **Question.** What does the split test? What should replace it?

    **Reasoning.**

    1. *PDB redundancy.* Many entries are identical or near-identical proteins (mutants, ligand complexes, different species). A random split places *homologs* of test chains in training: the model can succeed by recalling a homolog.
    2. *What would a leak-free split be?* (i) **Sequence-identity clusters** (e.g., hold out clusters at <30% identity; Chapter 43); (ii) **fold/superfamily holdout** (CATH/SCOP) for "new fold" claims; (iii) **temporal splits** (train on structures deposited before a date, test on later ones; the CASP/CAMEO design); (iv) **orphans and de novo designed proteins** with no evolutionary precedent.
    3. *Alternative explanations for 1.2 Å.* (H1) The model learned physics-like sequence–structure rules. (H2) It memorized folds and aligned test sequences to similar training ones. (H3) Test chains share ligand/complex context with training chains.
    4. *Experiments.* Plot RMSD against the *maximum sequence identity to any training chain* and against MSA depth; evaluate on the strictest split and on de novo designs; test whether a **nearest-homolog-copy baseline** (template-based modeling) achieves similar accuracy.
    5. *Predictions.* Under H2, accuracy drops sharply as maximum training identity falls below ~30% and for proteins without homologs; a template baseline matches the model at high identity.

    **Expert analysis.** The headline number is a mixture of recall and generalization; the *dependence on distance to training* is the real result (Chapter 1's "gap between random and structured splits is a measurement"). CASP's prospective design made this unambiguous for AlphaFold 2 (Chapter 35); every new structure or design model should be reported with its identity-stratified curve.

---

## 23.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: what does this assay say about the target property?"
    **Setting.** You want a model of a protein property and must choose among data sources: DMS growth selection, display-based binding, thermal stability, and natural-sequence conservation. They disagree on which variants are "bad."

    **Decompose.**

    1. **Write the latent variables.** Stability $\Delta G$, binding energy $\Delta G_b$, catalytic rate, expression/folding yield, aggregation propensity, evolutionary constraint. *Each assay is a different function of these.*
    2. **Write the link for each assay.** Growth selection: saturating in the product of activity and abundance. Display binding: depends on expression level and $K_d$ with a dynamic range limit. Stability assays: sigmoid in $\Delta G$ (§23.3.2). Conservation: a function of selection over evolutionary time and *environment*.
    3. **Predict disagreements.** A destabilizing mutation in a very stable protein is invisible to growth but visible to a stability assay; a binding-interface mutation is neutral for stability but fatal for binding; a mutation in a region needed only in a rare environment looks conserved but is neutral in the assay.
    4. **Design the test.** Plot one assay against another per variant, colored by structural class (core, interface, surface); use a **two-assay latent model** (shared stability latent, assay-specific links) and test whether it explains both better than separate models.
    5. **Implication for modeling.** A foundation model's zero-shot likelihood may correlate with one assay and not another *by construction*; supervised heads should be trained against the assay that defines the application.

    **What it teaches.** "Fitness" is not one number; the data a protein model sees define *which of several latent properties* it learns to predict. Specifying the target property precisely is the first experiment (A5: evaluation attack; A3: objective).

---

## 23.10 Connections

- **Backward:** mutation–selection equilibrium and the Sella–Hirsh relation (Chapter 5); phylogenetic structure and $N_\text{eff}$ (Chapter 21); sigmoid/logistic links (Chapter 4); MSAs and alignment algorithms (Chapter 6); the leakage lesson (Chapters 1, 7).
- **Forward:** alignment and homology search (Chapter 27); direct-coupling analysis (Chapter 29); protein language models (Chapter 34); structure prediction (Chapter 35); generative design (Chapter 36); evolutionary models and fitness landscapes (Chapter 42); leakage-free benchmarks (Chapter 43); open problems in protein modeling (Chapter 52).

!!! takeaways "Key takeaways"
    1. Proteins are sequences over 20 residues whose folded structure (a point in a $\sim5L$-dimensional torsion space) and dynamics produce function; PTMs and chirality are invisible in sequence-only representations.
    2. **Folding is marginally stable**: $f_\text{folded}=1/(1+e^{\Delta G/RT})$; with $\Delta G\approx-5$ to $-10$ kcal/mol the phenotype saturates; random mutations have mean $\Delta\Delta G\approx+1$ kcal/mol.
    3. **Global epistasis**: an additive latent trait through a nonlinear readout creates apparent epistasis (31–58% of pairs) and breaks additive models on multi-mutants ($R^2$ −19.8 vs. 0.84 for latent + link); identifiability depends on where the data sit on the response curve.
    4. **Structure data are biased** (PDB redundancy, crystallizability, single conformations); random PDB splits leak by homology; evaluate by identity cluster, fold, and time.
    5. **Evolutionary information** in an MSA (conservation, substitution, coevolution) is limited by the effective number of independent sequences; orphans and dark proteins lack it.
    6. **DMS** gives dense experimental maps but of an *assay-defined* proxy for function, with its own link function and noise ceiling (replicate $r\approx0.8$–$0.95$).
    7. A model of protein sequence sees *evolutionary and structural statistics*; it cannot observe ensembles, kinetics, cellular context, or function in families without data.

---

## Further reading

- Anfinsen, C. B. (1973). Principles that govern the folding of protein chains. *Science* 181, 223–230. Onuchic, J. N., Luthey-Schulten, Z. & Wolynes, P. G. (1997). Theory of protein folding: the energy landscape perspective. *Annual Review of Physical Chemistry* 48, 545–600. Dill, K. A. & MacCallum, J. L. (2012). The protein-folding problem, 50 years on. *Science* 338, 1042–1046.
- Bloom, J. D., Labthavikul, S. T., Otey, C. R. & Arnold, F. H. (2006). Protein stability promotes evolvability. *PNAS* 103, 5869–5874. Tokuriki, N., Stricher, F., Schymkowitz, J., Serrano, L. & Tawfik, D. S. (2007). The stability effects of protein mutations appear to be universally distributed. *J. Mol. Biol.* 369, 1318–1332.
- Sailer, Z. R. & Harms, M. J. (2017). Detecting high-order epistasis in nonlinear genotype–phenotype maps. *Genetics* 205, 1079–1088. Otwinowski, J., McCandlish, D. M. & Plotkin, J. B. (2018). Inferring the shape of global epistasis. *PNAS* 115, E7550–E7558. Starr, T. N. & Thornton, J. W. (2016). Epistasis in protein evolution. *Protein Science* 25, 1204–1218.
- Fowler, D. M. & Fields, S. (2014). Deep mutational scanning: a new style of protein science. *Nature Methods* 11, 801–807. Notin, P. et al. (2023). ProteinGym. *NeurIPS Datasets & Benchmarks*.
- Marks, D. S. et al. (2011). Protein 3D structure computed from evolutionary sequence variation. *PLoS ONE* 6, e28766. Morcos, F. et al. (2011). Direct-coupling analysis of residue coevolution captures native contacts across many protein families. *PNAS* 108, E1293–E1301.
- Bar-Even, A. et al. (2011). The moderately efficient enzyme. *Biochemistry* 50, 4402–4410.
- Berman, H. M. et al. (2000). The Protein Data Bank. *Nucleic Acids Research* 28, 235–242. Varadi, M. et al. (2022). AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. *Nucleic Acids Research* 50, D439–D444.
- van der Lee, R. et al. (2014). Classification of intrinsically disordered regions and proteins. *Chemical Reviews* 114, 6589–6631.
