# Chapter 40. Multimodal and Multi-Scale Models

!!! abstract "Chapter at a glance"
    **Motivation.** A cell is measured through many operators (Chapter 25): RNA, chromatin accessibility, surface proteins, imaging, spatial position, and sequence; a patient is measured through genomes, images, laboratory values, and notes; a mechanism spans molecules, cells, tissues, and organisms. Multimodal models promise to combine these views into one representation, and multi-scale models to connect levels of organization. The central question of this chapter is *what a multimodal objective keeps and what it throws away*. The answer from a controlled simulation is sharp: when the modalities carry both shared and modality-private information, aligning them with a contrastive objective yields embeddings that keep only the shared part, discarding information that the concatenated raw features retain (cell-type accuracy 0.57–0.65 from aligned embeddings against 0.92 from the concatenated data); the number of *paired* observations decides how well alignment works (retrieval among 500 cells reached top-1 accuracy 0.06 even with 5,000 pairs); and unpaired alignment (matching cells across modalities with no pairs, by Gromov–Wasserstein transport) was near chance when each modality had private structure and weak (0.38 against a chance level of 0.17) even when geometry was fully shared, reaching 0.86 only with sixty anchor pairs and matched composition. The chapter then reviews the real systems (multiome and CITE-seq integration, pathology and vision–language models, text–cell models, spatial-histology models), the problem of bridging scales, and the evaluation pitfalls (shortcuts from site, stain, and batch), with evidence grades.
    **Prerequisites.** Chapters 8, 13, 25, 30, 38, 39, 43–46.
    **You will be able to:** (1) write a shared-plus-private latent model and say what each method estimates; (2) derive why a contrastive objective lower-bounds the *shared* information and ignores private information; (3) estimate the paired-data requirement for retrieval and prediction; (4) say when unpaired alignment is identifiable; (5) evaluate a multimodal model for shortcuts; (6) design a data collection plan that pairs the right modalities.

---

## 40.0 Why more than one view

**Measurement operators.** Each modality $k$ is a different operator $M_k$ applied to the cell's state $s$ (Chapter 25):
$$
x^{(k)}=M_k(s)+\varepsilon_k,\qquad k=1,\dots,K.
$$
RNA is a sampled, noisy readout of transcripts; accessibility reads the chromatin state; surface proteins read a different layer of the system; images read morphology; spatial positions read the neighborhood. No single $M_k$ is invertible; together they constrain $s$ more.

**Scales.** The scales of biology (molecule, complex, pathway, cell, tissue, organism, population) each have their own models; *multi-scale* modeling couples them: molecular simulations feed kinetic parameters to cell models; cell models feed tissue models; images and genomes feed clinical models.

!!! lens "Research lens: what information does a multimodal model use and ignore?"
    *Uses*: whatever the objective makes necessary. A *joint generative* model (a multimodal VAE) must explain *all* of each modality's variation, including private noise; a *contrastive* model must only identify which observation in one modality goes with which in another, so it keeps what *both* see; a *supervised* multimodal model keeps what predicts the label. *Ignores*: the same as each: the contrastive model ignores private information by construction, and the generative model spends capacity on noise.

---

## 40.1 Shared and private structure, and the information accounting

**A latent model.** Let a cell have a shared latent $z_0$ and private latents $z_1,z_2$:
$$
x^{(1)}=f_1(z_0,z_1)+\varepsilon_1,\qquad x^{(2)}=f_2(z_0,z_2)+\varepsilon_2.
$$
The mutual information between modalities is carried by $z_0$: $I(x^{(1)};x^{(2)})\le I(z_0;\,x^{(1)})$, with equality only if $z_0$ is recoverable from each modality alone. Everything else, the private parts $z_1,z_2$, may predict *other* targets (cell type, state, disease) but cannot help *cross-modal* prediction.

**Classical methods.** *CCA* finds linear projections $a^\top x^{(1)}$, $b^\top x^{(2)}$ of maximal correlation (it estimates the shared subspace if the relation is linear); *PLS* maximizes covariance; *MOFA/JIVE* decompose each modality into shared and individual factors; *multimodal VAEs* (totalVI for RNA and protein, MultiVI for RNA and chromatin; MVAE with product of experts; mixture-of-experts variants) put a joint latent over modalities; *GLUE* uses prior knowledge (a regulatory graph linking peaks to genes) to align unpaired modalities.

!!! math "Derivation: the contrastive (InfoNCE) objective bounds the shared information"
    A dual-encoder model embeds paired observations $(x,y)$ as $(u,v)=(f(x),g(y))$ and trains with the InfoNCE loss over a batch of $N$ pairs,
    $$
    \mathcal L_\text{NCE}=-\mathbb E\Big[\log\frac{e^{s(u_i,v_i)/\tau}}{\sum_{j=1}^{N}e^{s(u_i,v_j)/\tau}}\Big],
    $$
    with similarity $s$ (cosine) and temperature $\tau$. Then (van den Oord et al. 2018)
    $$
    I(u;v)\;\ge\;\log N-\mathcal L_\text{NCE},
    $$
    and since $u$ and $v$ are functions of $x$ and $y$ respectively, $I(u;v)\le I(x;y)$ (data processing, Chapter 5). The bound cannot exceed $\log N$ (a larger batch is needed to certify more shared information), and the optimum retains *only information common to the two views that helps discriminate pairs*. Private information that varies independently across the views contributes nothing to $I(x;y)$ and is therefore *not required* in $(u,v)$: the encoders are free to discard it, and the objective gives them a (weak) incentive to do so, because private variation is nuisance that makes matching harder. $\square$

**Consequence.** A CLIP-style alignment of RNA and chromatin will produce an embedding that supports cross-modal retrieval and the *shared* variation (cell type to the extent both modalities see it), and will discard information specific to one modality that may matter for the task at hand (a chromatin-only regulatory state; an RNA-only transient program). Whether that is a feature or a bug depends on the task.

---

## 40.2 A controlled simulation

**Design.** Cells carry a 6-dimensional latent state with 6 clusters (cell types). Modality $X$ (60 features, "RNA-like") sees dimensions 0–1 (private to $X$) and 4–5 (shared); modality $Y$ (40 features, "chromatin- or protein-like") sees dimensions 2–3 (private to $Y$, through a saturating nonlinearity) and 4–5 (shared); each has four nuisance dimensions of its own and measurement noise. Three questions are asked. (1) *How many paired cells does cross-modal retrieval need* with a linear method (CCA) and a contrastive dual encoder? (2) *What does an alignment objective keep*: cell-type classification accuracy from raw $X$, raw $Y$, both, and shared embeddings. (3) *Can unpaired datasets be aligned* by entropic Gromov–Wasserstein (GW) transport, which matches the within-modality distance structures, when the datasets have the same or different compositions of cell types, with and without a few anchor pairs?

```python
--8<-- "code/ch40_multimodal.py"
```

```text
== 1. Paired cells needed for cross-modal retrieval (500 held-out cells; chance top-1 = 0.002, top-5 = 0.010) ==
paired cells    CCA (6 components) top-1 / top-5     contrastive dual encoder top-1 / top-5
      50           nan /    nan                       0.010 /  0.022
     200         0.004 /  0.068                       0.014 /  0.056
    1000         0.024 /  0.118                       0.022 /  0.122
    5000         0.034 /  0.112                       0.058 /  0.226
(CCA is not fit with fewer than 100 pairs: the covariance estimates are rank-deficient)

== 2. What an alignment objective keeps: cell-type accuracy (6 types; chance 0.17) with a logistic classifier trained on 1,000 labeled cells ==
X alone (RNA-like, raw)                                    0.796
Y alone (raw)                                              0.708
X and Y concatenated (raw)                                 0.916
shared embedding from X (contrastive, 6-d)                 0.574
shared embedding from Y (contrastive, 6-d)                 0.564
shared embeddings of X and Y concatenated (contrastive)    0.650
CCA, X side (6-d)                                          0.618
CCA, X and Y sides concatenated                            0.810
reading: the cell types are separable using X-private, Y-private, and shared dimensions; shared embeddings keep only what both modalities see


== 3. Unpaired alignment by entropic Gromov-Wasserstein (400 cells per modality, no pairs) ==
matching accuracy = fraction of X cells matched to a Y cell of the same cell type (chance = 0.17 for equal composition; 0.26 for the different composition, where matching everything to the commonest type would give the share of that type)
modalities                       composition of the two datasets          GW, no anchors    with 12 anchor pairs   with 60 anchor pairs   (mean of 6 seeds)
modality-private structure       same (equal proportions in both)          0.227            0.242               0.282
modality-private structure       different (40% / 3% extremes, reversed)   0.077            0.137               0.168
fully shared geometry            same (equal proportions in both)          0.380            0.683               0.862
fully shared geometry            different (40% / 3% extremes, reversed)   0.199            0.433               0.524
```

**Reading Part 1 (paired data and retrieval).** The shared information is two of six latent dimensions, hidden under nuisance and noise, so retrieval is hard: among 500 held-out cells (chance top-1 0.002), the contrastive dual encoder reaches top-1 of 0.010 with 50 pairs, 0.014 with 200, 0.022 with 1,000 and **0.058 with 5,000**; CCA (which cannot be fit with fewer than about 100 pairs because the covariance matrices are rank-deficient) reaches 0.004, 0.024 and 0.034. Both methods keep improving with paired data and neither saturates at 5,000 pairs. The practical implication is that **paired cells are the scarce resource** and the retrieval accuracy ceiling is set by the shared information (here two latent dimensions) rather than by the encoder.

**Reading Part 2 (what is kept).** The cell types are separable from three sources: $X$-private, $Y$-private, and shared dimensions. Logistic classifiers trained on 1,000 labeled cells give: raw $X$ 0.80; raw $Y$ 0.71; **both concatenated 0.92**; but *aligned shared embeddings* only 0.57 (from $X$), 0.56 (from $Y$) and 0.65 (both concatenated); CCA gives 0.62 (from the $X$ side) and 0.81 (both sides). **Alignment into shared coordinates discards a quarter to a third of the classification accuracy that the raw data carry**, as the derivation predicts: the private dimensions that help distinguish types in each modality are not in the shared embedding, and combining the two shared embeddings does not recover them. When the downstream task is a cell-type label that depends on modality-private information, use the *concatenation* of modality-specific embeddings, not only the shared one (the practice of totalVI, MultiVI and multimodal VAEs with private latents).

**Reading Part 3 (unpaired alignment).**

Matching accuracy is the fraction of $X$ cells matched to a $Y$ cell of the same cell type (chance 0.17 for equal composition).

1. **With modality-private structure, unpaired alignment fails.** Each modality's distance geometry is dominated by dimensions the other does not see; Gromov–Wasserstein (GW) matching reaches 0.23 with equal compositions, close to chance, and 0.08 when the cell-type proportions differ (the proportions in $X$ reversed relative to $Y$: the method matches the commonest types to each other regardless of identity).
2. **Even with fully shared geometry (both modalities see all six latent dimensions), GW alone is weak.** With equal composition it reaches 0.38; with different composition 0.20. Clusters in a six-dimensional space with moderate noise have approximate symmetries (alternative matchings of nearly equal cost) that the within-modality distance structure cannot break.
3. **A few anchors help a great deal when geometry is shared.** Anchor pairs (cells known to be the same type in both modalities, for example from marker genes) fit a CCA on the PCA coordinates, and cells are matched by nearest neighbor in that space: 12 anchors (two per type) give 0.68 and 60 anchors (ten per type) 0.86 for equal composition, and 0.43 and 0.52 for different composition. With modality-private structure the anchors add little (0.24–0.28 at equal composition) because the CCA can only align the shared dimensions.
4. **Composition shift costs about half of the accuracy at every anchor level** (0.86 to 0.52, 0.68 to 0.43, 0.38 to 0.20), because the method assumes that the two datasets are samples of the same population. This is the same problem as batch integration with different cell-type compositions (Chapter 30).

**Implication.** Unpaired alignment is an identifiability problem: it requires (i) shared geometry, (ii) matched composition (or a model of the difference), and (iii) anchors or a prior (the regulatory graph of GLUE). When these fail, no amount of data or encoder capacity restores identifiability; *measure a small number of pairs*.

!!! lens "Research lens: assumptions of the simulation"
    Gaussian clusters, an additive structure with linear observation for $X$ and a saturating observation for $Y$, equal-sized shared and private parts, noise at moderate levels, small unpaired datasets (400 cells), and a GW method with a single regularization setting. Real modalities differ in sparsity (chromatin is nearly binary), in noise, and in how much is shared; the *qualitative* findings (contrastive alignment keeps shared information only; paired data dominate; unpaired alignment needs shared geometry and matched composition) are the ones to carry forward.

---

## 40.3 Real systems and their evidence

| System | Modalities | Idea | Evidence and caveats | Grade |
|---|---|---|---|---|
| totalVI, MultiVI, MOFA+, WNN | RNA + protein (CITE-seq), RNA + chromatin (multiome) | Joint latent model or weighted neighbor graph across modalities | Improve clustering and imputation over single modalities; benefits depend on how complementary the modalities are | [[S]] |
| GLUE (Cao & Gao 2022) | Unpaired omics | Graph-guided alignment with regulatory prior | Works when the prior graph is informative and compositions agree | [[P]] |
| Contrastive cell-text/cell-image models | RNA/image + text | CLIP-style alignment of cells with annotations or images | Enable zero-shot annotation by text queries; inherit the loss of private information (§40.1) | [[P]] |
| Pathology foundation models (UNI, Virchow, Prov-GigaPath, CONCH; 2024) | Whole-slide images (and text for CONCH) | Self-supervised vision transformers trained on hundreds of thousands to millions of slides | Strong features for diagnosis and some molecular-subtype prediction; representations carry strong site and batch signatures (slide-specificity measures were high for several), so shortcut checks are essential | [[S]] features; [[P]] molecular prediction |
| Image-to-expression models | H&E + spatial transcriptomics | Predict expression from histology | Recover spatially varying programs at spot resolution, with low accuracy on individual genes | [[P]] |
| Language-model cells (Cell2Sentence, C2S-Scale) | RNA as text | A text LLM reads a cell | One model-generated hypothesis was validated in vitro (Chapter 38) | [[P]] |
| ESM3 | Sequence, structure, function tokens | A multi-track language model | Prompt in any modality; generation of a novel fluorescent protein (Chapter 34) | [[S]] |
| Clinical multimodal models | Genomes, images, EHR, text | Late fusion or joint models | Gains over unimodal models are modest and sensitive to site shift | [[P]] |

**Multi-scale bridging.** Coarse-graining a molecular model into a cell model loses information that matters unpredictably; ML is used to learn coarse-grained potentials (for proteins, machine-learned coarse-grained force fields reproduce the thermodynamics of small proteins; Majewski et al. 2023) and to learn surrogates of expensive simulations [[P]]. For cells and tissues, *hybrid* models embed learned components in mechanistic scaffolds (Chapter 51, C12). The open problem is the *error propagation across scales*: a small, systematic error in a molecular parameter becomes a large error in a cell-level prediction, and identifiability decreases with each layer.

---

## 40.4 Evaluation pitfalls

1. **Shortcuts from site and batch.** In pathology, the slide's stain, scanner, and hospital are predictable from image features, and may correlate with the label; a multimodal model that fuses images with genomics can learn "which hospital" (Chapter 45). Evaluate on held-out sites and with site-prediction probes.
2. **Modality dominance.** A multimodal model may rely on the more informative modality; ablate each modality to measure the contribution of the other (a model whose performance does not drop when a modality is removed is not using it).
3. **Leakage in pairing.** Paired data often come from the same donors; split by donor, not by cell (Chapter 43).
4. **Retrieval metrics inflated by batch.** If the batch structure differs between modalities, matching can exploit it.
5. **Noise ceilings per modality** (Chapter 25): the cross-modal prediction cannot exceed the shared signal's reliability.

---

## 40.5 Worked research examples

!!! example "Worked Research Example 40.1: A model predicts molecular subtype from H&E images with AUROC 0.89"
    **Situation.** A multimodal paper reports that image features from a pathology foundation model predict a transcriptomic subtype (basal-like versus luminal) of breast cancer from H&E slides with AUROC 0.89, validated on a held-out test set from the same hospitals.

    **Question.** What would you ask for before believing the clinical claim?

    **Reasoning (Expert Chain).**

    1. **L1 Claim.** The images contain information about a molecular subtype, enough for a clinical test.
    2. **L3–L5 Assumptions and failure modes.** (a) *Site and batch*: if the subtype prevalence differs between hospitals, a model that identifies the hospital can reach a high AUROC (check by a site-prediction probe and a site-adjusted evaluation); (b) *Confounding by grade, stage, and treatment*: morphology is strongly related to grade, which is related to subtype; (c) *The label's noise*: the transcriptomic subtype is a thresholded continuous quantity with its own reliability ceiling; (d) *Pairing leakage*: multiple slides per patient across splits.
    3. **L10 Experiments.** (i) Held-out *hospital* evaluation (leave-one-site-out); (ii) *stain normalization and site-stratified* sampling; (iii) *adjustment*: logistic baseline using grade, stage, and age alone (does the image add?); (iv) the *ceiling*: the AUROC of a second RNA assay on the same tissue; (v) *decision curve*: the net benefit at a clinically relevant threshold; (vi) prospective validation in a new cohort.
    4. **L11 Interpretation.** If AUROC stays above 0.8 in held-out sites and adds beyond grade, stage, and age, the image carries subtype information (C2); the clinical claim needs a decision-curve and prospective validation (C3).

    **Expert analysis.** A single in-distribution AUROC cannot separate morphology from site; the held-out-site result is the number that matters.

!!! example "Worked Research Example 40.2: A data plan for a multimodal cell model with 100,000 paired cells. No known answer"
    **Situation.** A consortium can afford 100,000 cells with paired RNA, chromatin, and surface-protein measurements (multiome plus CITE) or, for the same money, 400,000 cells with RNA only and 100,000 cells with chromatin only (unpaired). No one knows which supports a better model of the cell types and states of interest.

    **Question.** How would you decide?

    **Reasoning.**

    1. **Specify the use.** Tasks: (a) cross-modal prediction (predict accessibility from RNA); (b) cell-state classification using private information; (c) rare-state detection; (d) perturbation response (Chapter 39).
    2. **Use the simulation of §40.2 as a prior.** Paired data are required for cross-modal prediction (retrieval improved with pairs; unpaired GW was near chance without shared geometry and matched composition); private information is retained only if the model is trained on each modality's own data (the unpaired design gives more cells per modality and so helps within-modality models).
    3. **Pilot.** Measure 5,000 paired cells in a representative tissue and 20,000 unpaired cells per modality; fit paired and unpaired models; evaluate (a)–(c) on held-out donors; estimate learning curves for each design (Chapter 47) and extrapolate to the budget.
    4. **Decision rule.** If cross-modal prediction (a) is the goal and the paired learning curve is not saturating, buy pairs; if (b)–(c) dominate and the within-modality curves are steeper, buy unpaired cells and a small paired anchor set (about 5% of cells) to calibrate alignment.
    5. **Hedge.** A mixed design: 40% paired, 60% unpaired split across modalities; evaluate the marginal value of each additional 10,000 cells by the pilot's slope.

    **What is not known.** The marginal value of pairs for the consortium's tasks; the pilot estimates it, at about 5% of the cost.

---

## 40.6 Researcher's Notebook

!!! notebook "Researcher's Notebook: before building a multimodal model"
    1. **Decompose** each modality into shared and private information with a simple method (CCA, JIVE) and estimate their sizes.
    2. **Pick the objective for the task**: contrastive for retrieval and shared structure; generative for imputation; supervised for a label; concatenation for private-information tasks.
    3. **Count pairs**: plot a retrieval or prediction learning curve to estimate the pairs needed.
    4. **Test each modality's contribution** by ablation.
    5. **Check for shortcuts**: site, batch, and donor probes.
    6. **Evaluate unpaired alignment** only if shared geometry and composition are plausible; otherwise obtain anchors.
    7. **Reproduce the simulation** with different shared fractions (0.2, 0.5, 0.8) and find where alignment stops being lossy.

    **What it teaches.** A multimodal objective is a decision about which information to keep.

    **An open question to carry forward.** The derivation says the contrastive objective is *permitted* to discard private information; in practice it keeps some. Can the amount of private information retained be *measured* (for example, by the accuracy of decoding modality-specific variables from the aligned embedding) and *controlled* (for example, by adding reconstruction terms with a weight), with a trade-off curve between retrieval accuracy and private-information retention that could guide the choice of weight for each downstream task?

---

## 40.7 Connections

- **Backward:** latent-variable models and CCA (Chapter 8); information and data processing (Chapter 5); representation learning (Chapter 13); measurement operators (Chapter 25); batch and integration (Chapter 30); cell foundation models (Chapter 38); perturbation models (Chapter 39); shortcuts (Chapter 45); design (Chapter 46).
- **Forward:** open problems of cells (Chapter 51, C7); AI scientists (Chapter 54).

!!! takeaways "Key takeaways"
    1. Modalities are different measurement operators on one state; they carry **shared** and **private** information, and the objective decides which a model keeps.
    2. A contrastive objective lower-bounds the *shared* mutual information, $I(u;v)\ge\log N-\mathcal L_\text{NCE}$, and is free to discard private information.
    3. In simulation, cell-type accuracy was 0.80 from RNA-like data, 0.71 from the second modality, **0.92 from both concatenated**, but only **0.57–0.65 from aligned shared embeddings**: alignment discarded a quarter to a third of the classification accuracy.
    4. **Paired data are the scarce resource**: retrieval among 500 cells reached top-1 of 0.058 with 5,000 pairs for a contrastive encoder (CCA 0.034), still improving.
    5. **Unpaired alignment is identifiable only under assumptions**: Gromov–Wasserstein matching reached 0.23 (modality-private structure) and 0.38 (fully shared geometry) at equal composition, falling to 0.08 and 0.20 under a composition shift; 60 anchor pairs raised the shared-geometry case to 0.86 (0.52 under shift).
    6. Real multimodal systems (CITE-seq and multiome integration, pathology foundation models, image-to-expression, text–cell models) have good evidence for features and weaker evidence for downstream clinical or molecular prediction; shortcuts from site, stain, and batch are the main threat.
    7. Plan data collection by the use: pairs for cross-modal tasks, more cells per modality for within-modality tasks, plus a small paired anchor set.

---

## Further reading

- Argelaguet, R. et al. (2020). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. *Genome Biol.* 21, 111. Gayoso, A. et al. (2021). Joint probabilistic modeling of single-cell multi-omic data with totalVI. *Nat. Methods* 18, 272–282. Ashuach, T. et al. (2023). MultiVI: deep generative model for the integration of multimodal data. *Nat. Methods* 20, 1222–1231. Hao, Y. et al. (2021). Integrated analysis of multimodal single-cell data. *Cell* 184, 3573–3587 (WNN).
- Cao, Z.-J. & Gao, G. (2022). Multi-omics single-cell data integration and regulatory inference with graph-linked embedding. *Nat. Biotechnol.* 40, 1458–1466. van den Oord, A., Li, Y. & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748.* Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML.* Peyré, G., Cuturi, M. & Solomon, J. (2016). Gromov-Wasserstein averaging of kernel and distance matrices. *ICML.*
- Chen, R. J. et al. (2024). Towards a general-purpose foundation model for computational pathology (UNI). *Nat. Med.* 30, 850–862. Vorontsov, E. et al. (2024). A foundation model for clinical-grade computational pathology and rare cancers detection (Virchow). *Nat. Med.* 30, 2924–2935. Xu, H. et al. (2024). A whole-slide foundation model for digital pathology from real-world data (Prov-GigaPath). *Nature* 630, 181–188. Lu, M. Y. et al. (2024). A visual-language foundation model for computational pathology (CONCH). *Nat. Med.* 30, 863–874.
- Majewski, M. et al. (2023). Machine learning coarse-grained potentials of protein thermodynamics. *Nat. Commun.* 14, 5739. Kather, J. N. et al. (2020). Pan-cancer image-based detection of clinically actionable genetic alterations. *Nat. Cancer* 1, 789–799. Levine, D. et al. (2023). Cell2Sentence. *bioRxiv.*
