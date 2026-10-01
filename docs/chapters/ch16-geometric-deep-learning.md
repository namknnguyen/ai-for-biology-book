# Chapter 16. Geometric Deep Learning: Graphs, Symmetry, and Structure

!!! abstract "Chapter at a glance"
    **Motivation.** Molecules are graphs, proteins are chains of rigid bodies in 3-D space, cells are sets, interaction networks are graphs, and double-stranded DNA is a sequence with a mirror symmetry. Encoding the *symmetries* of these objects into the architecture is the most reliable way to reduce data requirements and prevent errors. This chapter gives the machinery behind graph networks, equivariant networks, and AlphaFold's structure module.
    **Prerequisites.** Chapters 2, 9, 10, 12, 15.
    **You will be able to:** (1) define invariance and equivariance and state why they help; (2) derive the GCN layer and prove permutation equivariance of message passing; (3) explain expressivity limits of GNNs (1-WL) and over-smoothing, with the numerical evidence; (4) construct and verify an E(n)-equivariant layer and explain why proteins need SE(3), not E(3); (5) explain why invariant point attention is invariant to global rigid motions; (6) recognize degree bias and scaffold leakage in graph benchmarks.


!!! note "If this chapter moves too fast"
    Part 0 teaches the prerequisites from scratch: [M9](m09-discrete-structures.md) (graphs and Laplacians) and [M6](m06-linear-algebra-2.md) (eigenvalues of symmetric matrices).

---

## 16.1 Symmetry as an inductive bias

A **symmetry** of a problem is a transformation of the input that does not change what we want to predict (or changes it in a predictable way). Let a group $G$ act on inputs by $x\mapsto g\cdot x$.

- A function $f$ is **invariant** if $f(g\cdot x)=f(x)$ for all $g\in G$ (a molecule's energy does not depend on how it is rotated).
- It is **equivariant** if $f(g\cdot x)=\rho(g)\,f(x)$ (rotating a molecule rotates its predicted forces; permuting atoms permutes per-atom predictions).

**Why symmetry helps (Chapter 7's bias–variance viewpoint).** A model that must *learn* a symmetry from data spends capacity and samples on it, and still fails off the training distribution. A model that has the symmetry *built in* gets it for free everywhere, effectively multiplying the dataset by the group size. The price is architectural constraints and sometimes compute. The blueprint of **geometric deep learning** (Bronstein et al., 2021) is to design layers equivariant to the relevant group and finish with an invariant readout.

| Biological object | Symmetry group | Standard architecture | Where it appears |
|---|---|---|---|
| Position along a genome | translations (approx.) | CNN (Chapter 10) | Motif scanning |
| Double-stranded DNA | $\mathbb{Z}_2$ (reverse complement) | Filter pairing; Caduceus | Chapters 10, 32 |
| Atoms in a molecule | permutations $S_n$ | GNN, DeepSets | Chapters 24, 37 |
| A cell's genes or a patient's cells (as sets) | $S_n$ | DeepSets, set/attention models | Chapters 38, 40 |
| Molecule or protein in space | $SE(3)$ (rotations, translations; **chirality preserved**) | IPA, frame networks, tensor-field nets | Chapters 35–36 |
| Rows of an MSA | $S_N$ (exchangeable *in principle*) | MSA Transformer's row attention with tied weights | Chapter 35 |
| Sequence label (cell type, species) | none (a classification symmetry is *assumed*) | — | — |

---

## 16.2 Graph neural networks

### 16.2.1 Message passing

A graph $G=(V,E)$ has node features $\mathbf{h}_i\in\R^d$ and optionally edge features $\mathbf{e}_{ij}$. A **message-passing neural network** (MPNN; Gilmer et al., 2017) updates each node by aggregating messages from its neighbors $\mathcal{N}(i)$:

$$
\mathbf{m}_i^{(\ell+1)}=\sum_{j\in\mathcal{N}(i)}\psi\big(\mathbf{h}_i^{(\ell)},\mathbf{h}_j^{(\ell)},\mathbf{e}_{ij}\big),\qquad
\mathbf{h}_i^{(\ell+1)}=\phi\big(\mathbf{h}_i^{(\ell)},\mathbf{m}_i^{(\ell+1)}\big),
$$

with a graph-level **readout** $\hat y=R(\{\mathbf{h}_i^{(L)}\})$ that is permutation-invariant (sum, mean, max, or attention pooling).

**Permutation equivariance.** If nodes are relabeled by a permutation matrix $\mathbf{P}$ (so $\mathbf{A}\to\mathbf{P}\mathbf{A}\mathbf{P}^\top$, $\mathbf{H}\to\mathbf{P}\mathbf{H}$), then $\mathbf{m}_i$ is a *sum over the same set of neighbors*, regardless of order; hence each layer maps $\mathbf{P}\mathbf{H}$ to $\mathbf{P}\mathbf{H}^{\text{new}}$. A sum readout is then invariant. The code verifies on a random 8-node graph and a 3-layer MPNN: equivariance error $6\times10^{-8}$, readout invariance error $1.2\times10^{-7}$ (float32 rounding).

### 16.2.2 The graph convolutional network, derived

Kipf & Welling (2017) obtained the GCN from spectral graph theory. The normalized graph Laplacian is $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ with eigendecomposition $\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$; a *spectral filter* acts as $g_\theta(\mathbf{L})\mathbf{x}=\mathbf{U}g_\theta(\boldsymbol\Lambda)\mathbf{U}^\top\mathbf{x}$ (the graph analogue of convolution as multiplication in the Fourier domain). Approximating $g_\theta$ by a first-order Chebyshev polynomial and tying parameters gives $\theta(\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2})\mathbf{x}$. Because the eigenvalues of this operator lie in $[0,2]$, repeated application is unstable; the **renormalization trick** replaces $\mathbf{A}$ by $\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}$ (self-loops) and $\mathbf{D}$ by $\tilde{\mathbf{D}}$, giving the layer

$$
\mathbf{H}^{(\ell+1)}=\sigma\Big(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\,\mathbf{H}^{(\ell)}\mathbf{W}^{(\ell)}\Big),
$$

a *normalized neighborhood average* followed by a shared linear map. Each node's new feature is a degree-normalized mean of itself and its neighbors, transformed. **Graph attention networks** (GAT; Veličković et al., 2018) replace the fixed normalized-adjacency weights by learned attention weights over neighbors.

!!! rhyme "Structural rhyme: Transformer ↔ graph attention on the complete graph"
    A transformer layer is a **GAT on the fully connected graph** whose edge weights are content-dependent softmax attention, with positional encodings playing the role of edge features. A GNN that restricts attention to a sparse graph is a transformer with a mask. This explains why (i) transformers are permutation-equivariant without positional information (Chapter 12), (ii) GNNs and transformers are often competitive on molecules, and (iii) the choice between them is a choice of *which pairs are allowed to interact* (a prior: chemical bonds, spatial neighbors, sequence neighbors).

### 16.2.3 What message passing cannot do: expressivity

The **Weisfeiler–Leman (1-WL) test** iteratively refines node "colors" by hashing a node's color with the multiset of its neighbors' colors; two graphs with different color histograms are non-isomorphic. Message-passing GNNs with sum aggregation are **at most as powerful as 1-WL**, and with injective aggregators exactly as powerful (Xu et al., 2019; Morris et al., 2019). [[E]]

*Consequence.* There are non-isomorphic graphs that **no** message-passing GNN can distinguish. The code uses the standard example: a hexagon ($C_6$) and two disjoint triangles ($2\times C_3$) are both 2-regular graphs on six nodes; with identical input features, **every** MPNN assigns them *exactly the same* graph embedding (difference $0.0$).

*Relevance to chemistry.* Message passing on the bond graph cannot count certain substructures (ring sizes) or distinguish certain isomers; a classic illustration is that decalin and bicyclopentyl have the same 1-WL colorings. Fixes: **higher-order GNNs** (costly), **substructure counts** or ring features as inputs, **positional/structural encodings** (Laplacian eigenvectors, random-walk statistics), and **3-D geometry** (distances and angles), which makes the graph a geometric object rather than a combinatorial one.

### 16.2.4 Over-smoothing

Repeated application of the propagation matrix $\hat{\mathbf{S}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ drives features toward its dominant eigenspace. For a connected graph, the top eigenvalue is 1 (eigenvector $\propto\tilde{\mathbf{D}}^{1/2}\mathbf{1}$) and all others satisfy $|\lambda_i|<1$, so $\hat{\mathbf{S}}^k\mathbf{H}\to$ (rank-one) at rate $|\lambda_2|^k$: **all node features become indistinguishable**. On a connected random geometric graph (200 nodes) with $\lambda_2=0.965$, the spread of node features across the graph falls from $0.997$ at $k=0$ to $0.071$, $0.042$, $0.027$, and $0.018$ at $k=10,20,30,40$.

*Consequences.* Deep GNNs lose node-level information (hence the surprise that "deeper is not better" for graphs, unlike CNNs); **over-squashing** (information from exponentially many neighbors compressed into a fixed-size vector) causes related long-range failures. Remedies: residual connections, normalization, skip connections to the input, attention, rewiring, and keeping depth small (2–6 layers). This is the graph analogue of Chapter 9's depth analysis.

---

## 16.3 Molecules as geometric objects

### 16.3.1 2-D (bond-graph) models

Nodes are atoms (features: element, formal charge, hybridization, aromaticity, number of hydrogens); edges are bonds (type, conjugation, ring membership). **D-MPNN** (Yang et al., 2019; Chemprop) passes messages along *directed bonds* to avoid immediate backtracking and is a strong baseline for molecular property prediction. These models cannot see 3-D conformation or stereochemistry unless explicitly provided (chirality tags are handled by input features).

### 16.3.2 3-D invariant models

For energies and binding, geometry matters. **SchNet** (Schütt et al., 2017) uses continuous-filter convolutions on interatomic *distances* (rotation- and translation-invariant by construction); **DimeNet** adds *angles*. Invariant models achieve symmetry by feeding only invariant quantities (distances, angles, torsions), at the cost of throwing away directional information that vector-valued predictions (forces, dipoles) need.

### 16.3.3 E(n)-equivariant networks (EGNN)

Satorras, Hoogeboom & Welling (2021) give a simple equivariant layer. With invariant node features $\mathbf{h}_i$ and coordinates $\mathbf{x}_i\in\R^3$:

$$
\mathbf{m}_{ij}=\phi_e\big(\mathbf{h}_i,\mathbf{h}_j,\|\mathbf{x}_i-\mathbf{x}_j\|^2\big),\qquad
\mathbf{x}_i'=\mathbf{x}_i+\frac1{n-1}\sum_{j\ne i}(\mathbf{x}_i-\mathbf{x}_j)\,\phi_x(\mathbf{m}_{ij}),\qquad
\mathbf{h}_i'=\phi_h\Big(\mathbf{h}_i,\sum_j\mathbf{m}_{ij}\Big).
$$

*Proof of equivariance.* Under $\mathbf{x}_i\mapsto\mathbf{R}\mathbf{x}_i+\mathbf{t}$ ($\mathbf{R}$ orthogonal), the squared distance $\|\mathbf{x}_i-\mathbf{x}_j\|^2$ is unchanged, so $\mathbf{m}_{ij}$ and $\mathbf{h}'_i$ are unchanged (invariant). The displacement $\mathbf{x}_i-\mathbf{x}_j\mapsto\mathbf{R}(\mathbf{x}_i-\mathbf{x}_j)$ while its scalar coefficient $\phi_x(\mathbf{m}_{ij})$ is invariant, so $\mathbf{x}'_i\mapsto\mathbf{R}\mathbf{x}'_i+\mathbf{t}$. $\square$ The code verifies on 7 points: coordinate equivariance error $4.8\times10^{-7}$, feature invariance $1.2\times10^{-7}$; a naive MLP applied to raw coordinates violates equivariance by $\|f(\mathbf{R}\mathbf{x})-\mathbf{R}f(\mathbf{x})\|=1.30$.

!!! warning "Chirality: E(3) is the wrong group for biomolecules"
    $E(n)$ includes **reflections**. The EGNN's invariant features are *identical* for a structure and its mirror image (code: difference $0.0$): **it cannot distinguish enantiomers.** Proteins are made of L-amino acids and are chiral; drugs' enantiomers can differ in activity and safety (the thalidomide story). Models for chiral molecules must be $SE(3)$-equivariant *but not reflection-equivariant*: use orientation-sensitive features (signed volumes, cross products, local frames) as in AlphaFold's frame-based modules. A model that is E(3)-equivariant and fed only distances will happily generate a protein and its mirror image as equally likely.

```python
--8<-- "code/ch16_geometric.py"
```

Output:

```text
permutation equivariance: max |f(PAP^T, Px) - P f(A, x)| = 6.0e-08; sum-readout invariance: 1.2e-07
1-WL limit: ||embedding(C6) - embedding(2 x C3)|| = 0.0e+00  (identical, though the graphs differ)
EGNN: coordinates equivariant: max |x'(Rx+t) - (Rx'+t)| = 4.8e-07; features invariant: 1.2e-07
naive MLP on raw coordinates: ||f(Rx) - R f(x)|| = 1.30  (not equivariant)
EGNN invariant features under a mirror image: max difference = 0.0e+00 (it cannot see chirality)
over-smoothing: feature spread across nodes after k = 0,10,20,30,40 propagations = [0.9968, 0.0707, 0.0416, 0.0267, 0.0181]  (second eigenvalue 0.965)
link prediction by degree alone: AUROC with uniformly random negatives = 0.704; with degree-matched negatives = 0.499
```

---

## 16.4 Proteins: frames, invariant point attention, and structure modules

### 16.4.1 Residues as rigid frames

A protein backbone can be described by one **rigid frame** per residue: the orthonormal frame $\mathbf{T}_i=(\mathbf{R}_i,\mathbf{t}_i)\in SE(3)$ built (by Gram–Schmidt) from the positions of N, C$\alpha$, and C atoms, with the origin at C$\alpha$. A structure is thus a point in $SE(3)^L$, and *global* rotation/translation acts on all frames simultaneously: $\mathbf{T}_i\mapsto\mathbf{g}\mathbf{T}_i$. Side-chain geometry is added by torsion angles within each residue's local frame.

### 16.4.2 Invariant point attention (IPA)

AlphaFold 2's structure module (Jumper et al., 2021) updates residue representations with **invariant point attention**. In addition to the usual scalar queries/keys/values, each residue emits *3-D points* $\vec{\mathbf{q}}_i^{\,p},\vec{\mathbf{k}}_i^{\,p},\vec{\mathbf{v}}_i^{\,p}\in\R^3$ **in its local frame**. Their global positions are $\mathbf{T}_i\circ\vec{\mathbf{q}}_i^{\,p}=\mathbf{R}_i\vec{\mathbf{q}}_i^{\,p}+\mathbf{t}_i$. The attention logit between residues $i$ and $j$ combines scalar similarity, a pair-representation bias, and a **geometric term**:

$$
a_{ij}\;\propto\;\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{c}}+b_{ij}-\frac{\gamma}{2}\sum_p\big\|\mathbf{T}_i\circ\vec{\mathbf{q}}_i^{\,p}-\mathbf{T}_j\circ\vec{\mathbf{k}}_j^{\,p}\big\|^2 .
$$

Output points are aggregated in the global frame and mapped back into residue $i$'s local frame: $\vec{\mathbf{o}}_i^{\,p}=\mathbf{T}_i^{-1}\circ\sum_ja_{ij}\,\mathbf{T}_j\circ\vec{\mathbf{v}}_j^{\,p}$.

*Why it is invariant.* If all frames undergo a global rigid motion $\mathbf{g}$, $\mathbf{T}_i\to\mathbf{g}\mathbf{T}_i$, then $\mathbf{T}_i\circ\vec{\mathbf{q}}\to\mathbf{g}(\mathbf{T}_i\circ\vec{\mathbf{q}})$; distances between transformed points are unchanged, so the logits are unchanged; and $\mathbf{T}_i^{-1}\circ\mathbf{g}^{-1}\mathbf{g}\,(\cdots)=\mathbf{T}_i^{-1}\circ(\cdots)$, so the output points in local frames are unchanged. $\square$ The structure module then predicts a *rigid update* to each frame, $\mathbf{T}_i\leftarrow\mathbf{T}_i\circ\Delta\mathbf{T}_i$, an $SE(3)$-equivariant operation by construction. The loss (**FAPE**, frame-aligned point error) measures atom positions *in each residue's local frame*, so it is invariant to global motion yet sensitive to *chirality* (a mirror image has large FAPE). Chapter 35 uses these pieces.

### 16.4.3 Proteins as graphs

For inverse folding (Chapter 36), a backbone is converted to a $k$-nearest-neighbor graph on C$\alpha$ atoms, with edge features built from *distances encoded by radial basis functions* and *relative orientations*, as in **ProteinMPNN** (Dauparas et al., 2022) and **geometric vector perceptrons** (Jing et al., 2021), which carry scalar *and* vector features and are $SE(3)$-equivariant. This is the "invariant features from geometry" approach.

---

## 16.5 Sets and the permutation symmetry in single-cell and patient data

A cell's expression profile is a set of (gene, value) pairs; a patient's single-cell sample is a set of cells. **DeepSets** (Zaheer et al., 2017) shows that a function on a set $X$ is permutation-invariant iff it can be written as

$$
f(X)=\rho\Big(\sum_{x\in X}\phi(x)\Big),
$$

for suitable $\phi,\rho$ (for sets of bounded size and a sufficiently high-dimensional $\phi$; Wagstaff et al., 2019, discuss the dimension requirement). [[E]] Set Transformers add attention among elements. This is the architecture of **multiple-instance learning** (a patient label from a bag of cells) and of any model whose input is a bag of genes or reads. *Caveat:* sum-pooling is blind to which cells *interact* (neighbors in tissue); spatial data call for graph models on a cell neighborhood graph (Chapter 40).

---

## 16.6 Biological graphs: networks, knowledge graphs, and degree bias

Biological graphs include protein–protein interaction (PPI) networks, gene regulatory networks, metabolic networks, drug–target–disease knowledge graphs (e.g., Hetionet, PrimeKG), and cell–cell neighborhood graphs. Three recurring problems:

1. **Incompleteness and noise.** Most true edges are unobserved; observed edges contain false positives (e.g., yeast two-hybrid artifacts).
2. **Ascertainment (study) bias and degree bias.** Well-studied genes and proteins have far more recorded interactions. A model can predict links from node *degree* alone.
3. **Evaluation by random negatives.** Sampling negative pairs uniformly at random (non-edges) is easy because most pairs involve low-degree nodes.

In a simulated preferential-attachment network (2,000 nodes, 10% of edges held out), a score based only on **node degree** achieves AUROC $0.704$ against *uniformly random* negatives but $0.499$ (chance) against **degree-matched** negatives. *The apparent signal was degree; no biology or learned structure was involved.* In real PPI and knowledge-graph benchmarks, simple degree heuristics are surprisingly strong, and many reported gains of graph embeddings shrink under degree-controlled evaluation. [[S]]

!!! lens "Research lens: graph learning in biology"
    **Assumes:** the graph's edges are informative about function and are *measured without bias*. **Information used:** topology (and node/edge attributes). **Information ignored:** edge direction/sign/context unless encoded; condition-specificity (a regulatory edge may exist only in some cells). **Failure modes:** degree and study bias; leakage when test edges are implied by training edges (transitivity); homophily assumed where none exists; over-smoothing and over-squashing with depth; reliance on noisy or circularly derived edges (e.g., edges inferred from the same expression data used for training).

---

## 16.7 Equivariance versus augmentation versus scale

Building in a symmetry guarantees it and improves data efficiency, but equivariant layers (especially higher-order tensor products) are expensive, constraining model width and depth. At large scale, **augmentation with a flexible architecture** can learn approximate symmetry from data. AlphaFold 3 chose augmentation for its diffusion module, whereas AlphaFold 2's structure module used frame-based invariance (Chapters 15, 35). [[S]] The trade-off follows Chapter 7's inductive-bias logic: **build in the symmetry when data are scarce and the symmetry is exact; learn it when data and compute are abundant or when the symmetry is only approximate** (e.g., local rotation symmetry is exact for isolated molecules but broken in a protein's anisotropic environment).

---

## 16.8 Worked research examples

!!! example "Worked Research Example 16.1: A GNN predicts aqueous solubility with $R^2=0.91$ on a random split, but only 0.45 on a scaffold split"
    **Situation.** A message-passing network is trained on 10,000 molecules with measured solubility. Random-split $R^2$ is 0.91; scaffold-split $R^2$ is 0.45.

    **Question.** What does the gap mean, and what does the "true" performance look like?

    **Reasoning.**

    1. *What differs between the splits?* A random split places close analogs (shared **Bemis–Murcko scaffolds**, series from one medicinal-chemistry campaign) in both train and test; a scaffold split places entire scaffold families on one side.
    2. *What would a flexible model do under a random split?* Memorize local structure–property relationships within each series: near-neighbor lookup (Chapter 7). Measurement conditions within a series are also correlated (same lab, same protocol), so lab effects leak.
    3. *Is expressivity relevant here?* The 1-WL limit (§16.2.3) concerns graph distinguishability; here the issue is **generalization across chemical series**, not expressivity.
    4. *What is the right claim?* Random-split performance estimates *interpolation among analogs* (useful for lead optimization within a series); scaffold/cluster/time-split performance estimates *generalization to novel chemotypes* (the task in hit discovery).
    5. *Experiments.* Report both, plus a **temporal split** (train on earlier compounds, test on later ones) that mimics prospective use; add **noise-ceiling** estimates from assay replicates; compare with a nearest-neighbor Tanimoto baseline and a gradient-boosted model on fingerprints; include **applicability-domain** diagnostics (similarity to nearest training molecule vs. error).
    6. *Prediction.* Error increases with decreasing nearest-neighbor similarity; the nearest-neighbor baseline matches the GNN under random splits and fails under scaffold splits just as the GNN does.

    **Expert analysis.** The gap is *itself the measurement* (Chapter 1): it estimates how much the model's skill depends on having seen analogs. A claim about drug discovery must be evaluated on the split that matches the deployment (Chapters 24, 37, 43).

!!! example "Worked Research Example 16.2: A graph embedding method \"predicts novel protein–protein interactions\" with AUROC 0.93"
    **Situation.** A GNN-based link predictor on a PPI network reports AUROC 0.93 against randomly sampled non-interacting pairs and proposes novel interactions for validation.

    **Reasoning.**

    1. *What is the baseline?* Degree. From the simulation: degree alone gives 0.70 against uniform negatives in a synthetic network; in real PPI networks, which are far more heavy-tailed and study-biased, degree-based scores are much stronger.
    2. *Why?* Positives concentrate on hubs (heavily studied proteins such as TP53 or ubiquitin ligases) because those proteins have been assayed more, not necessarily because they interact more in vivo.
    3. *Alternative explanations for 0.93.* (H1) The model learned biologically meaningful similarity (shared domains, co-expression, co-localization). (H2) Degree and study bias. (H3) Leakage: test edges are implied by training edges (transitive closure, protein complexes where every member interacts with every other; a held-out edge inside a clique is trivially predicted). (H4) Features derived from the *same* experiments as the test edges.
    4. *Experiments.* Evaluate with **degree-matched negatives** and compare against a degree-only score; use **cluster-aware splits** (hold out all interactions of a protein or of a complex); test **on edges from an independent assay** (different technology) and, for novel predictions, **prospective** validation (binary assays such as yeast two-hybrid or LUMIER on a random sample of top-ranked and random pairs).
    5. *Predictions.* Under H2, the 0.93 collapses toward the degree baseline under matched negatives; under H1, a gap over degree persists and prospective validation shows enrichment over degree-matched controls.

    **Expert analysis.** A link predictor is a claim about *where to look next in the lab*; its value is the **enrichment of true interactions over a degree-matched baseline in a prospective sample**, never the AUROC on random negatives.

---

## 16.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: the symmetry audit"
    **Procedure.** For any new modeling problem, list the transformations that *should not* change the answer and those that *should*, then check whether the model respects them.

    | Candidate symmetry | Should the answer be invariant/equivariant? | How to test | Common mistake |
    |---|---|---|---|
    | Reverse complement of DNA | Invariant for strand-agnostic targets; *not* for strand-specific ones | Feed rc(x); compare | Imposing it on transcription-direction-specific tasks |
    | Shift of the window by a few bp | Approximately invariant for 1-kb-scale labels; not for base-resolution labels | Shift and compare | Predicting 1-bp features with a model whose pooling blurs positions |
    | Permutation of cells in a sample | Invariant (for bag-level labels) | Shuffle cells | Pooling that depends on batch order |
    | Global rotation/translation of a molecule | Invariant (energy), equivariant (forces) | Rotate coordinates | Using absolute coordinates as features |
    | Reflection of a protein | **Not** a symmetry (chirality matters) | Mirror the structure; the score should change | E(3)-equivariant model fed distances only |
    | Renaming atoms/residues | Invariant | Permute indices | Position-indexed features in graph models |
    | Reordering MSA rows | Invariant in principle; but *rows are phylogenetically related* | Shuffle rows; also subsample | Assuming exchangeable rows are independent |
    | Species identity | Not a symmetry; *a confounder* | Test transfer across species | Treating species-specific codon bias as signal |

    **What the audit produces.** Each row is either a *constraint to build in*, an *augmentation to apply*, or a *test to run*. The audit also finds *false symmetries* (assumptions of invariance that biology violates).

    **Distinguishing fundamental from implementation problems.** A model that fails a symmetry test on a symmetry it should have is an *implementation* issue (fixable by weight tying, augmentation, or equivariant layers). A symmetry the biology *breaks* (chirality; strand-specific transcription) cannot be imposed without error: it is a *modeling-assumption* issue.

---

## 16.10 Connections

- **Backward:** convolution and equivariance (Chapter 10); attention as complete-graph message passing (Chapter 12); depth, normalization, residuals against over-smoothing (Chapter 9); equivariant denoisers for diffusion (Chapter 15); leakage in splits (Chapters 1, 7).
- **Forward:** AlphaFold 2/3 structure modules and Pairformer (Chapter 35); inverse folding and backbone generation (Chapter 36); molecular property prediction, docking, cofolding (Chapters 24, 37); cell graphs and spatial models (Chapters 25, 40); evaluating graph claims and degree bias (Chapter 43); symmetry as a source of research ideas (attack A9, Chapter 55).

!!! takeaways "Key takeaways"
    1. **Invariance** $f(gx)=f(x)$ and **equivariance** $f(gx)=\rho(g)f(x)$ encode symmetry; building them in multiplies effective data.
    2. **Message passing** with sum aggregation is permutation-equivariant (verified to $10^{-7}$); the **GCN** is a normalized neighborhood average derived from first-order spectral filters; **transformers are GATs on complete graphs**.
    3. Message-passing GNNs are **at most 1-WL expressive**: a hexagon and two triangles get identical embeddings. Over-smoothing drives node features together at rate $|\lambda_2|^k$ (0.997→0.018 in 40 steps).
    4. **EGNN** is $E(n)$-equivariant by using invariant distances and displacement vectors, but **cannot see chirality**; proteins and drugs need $SE(3)$ with orientation features.
    5. **Invariant point attention** uses point-to-point distances in the global frame, so it is invariant to global rigid motion; **FAPE** is frame-aligned and chirality-sensitive.
    6. **DeepSets**: permutation-invariant functions are $\rho(\sum\phi(x))$; the basis of bag-of-cells and bag-of-genes models.
    7. **Degree bias** gave AUROC 0.704 for degree alone against random negatives and 0.499 against degree-matched negatives; **scaffold splits** reveal the dependence on analogs.
    8. Run a **symmetry audit** before architecture selection; do not impose symmetries biology breaks.

---

## Further reading

- Bronstein, M. M., Bruna, J., Cohen, T. & Veličković, P. (2021). Geometric deep learning: grids, groups, graphs, geodesics, and gauges. arXiv:2104.13478.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *ICML*. Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. Veličković, P. et al. (2018). Graph attention networks. *ICLR*.
- Xu, K., Hu, W., Leskovec, J. & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*. Morris, C. et al. (2019). Weisfeiler and Leman go neural: higher-order graph neural networks. *AAAI*. Li, Q., Han, Z. & Wu, X.-M. (2018). Deeper insights into graph convolutional networks for semi-supervised learning. *AAAI*. Alon, U. & Yahav, E. (2021). On the bottleneck of graph neural networks and its practical implications. *ICLR*.
- Yang, K. et al. (2019). Analyzing learned molecular representations for property prediction. *J. Chem. Inf. Model.* 59, 3370–3388. Schütt, K. T. et al. (2017). SchNet. *NeurIPS*. Gasteiger, J., Groß, J. & Günnemann, S. (2020). Directional message passing for molecular graphs. *ICLR*.
- Satorras, V. G., Hoogeboom, E. & Welling, M. (2021). E(n) equivariant graph neural networks. *ICML*. Thomas, N. et al. (2018). Tensor field networks. arXiv:1802.08219. Geiger, M. & Smidt, T. (2022). e3nn. arXiv:2207.09453. Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L. & Dror, R. (2021). Learning from protein structure with geometric vector perceptrons. *ICLR*.
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589. Dauparas, J. et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science* 378, 49–56.
- Zaheer, M. et al. (2017). Deep sets. *NeurIPS*. Wagstaff, E. et al. (2019). On the limitations of representing functions on sets. *ICML*. Lee, J. et al. (2019). Set Transformer. *ICML*.
- Himmelstein, D. S. et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife* 6, e26726. (Hetionet.)
- Bemis, G. W. & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* 39, 2887–2893.
- Sieg, J., Flachsenberg, F. & Rarey, M. (2019). In need of bias control: evaluating chemical data for machine learning in structure-based virtual screening. *J. Chem. Inf. Model.* 59, 947–961.
