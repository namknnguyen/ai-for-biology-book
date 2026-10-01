# Chapter 38. Single-Cell Foundation Models

!!! abstract "Chapter at a glance"
    **Motivation.** Single-cell atlases now contain more than a hundred million cells, and the success of language models trained on text suggested a path: pretrain a transformer on cells, with genes as tokens, and obtain embeddings, annotations, perturbation predictions, and regulatory networks from one model. Between 2022 and 2025 many such models appeared (scBERT, Geneformer, scGPT, scFoundation, UCE, TranscriptFormer, State, Cell2Sentence). The first independent zero-shot evaluations found that several of them lost to simple baselines (highly variable genes with PCA, scVI, Harmony) on cell-type separation and batch integration, which turned a promise into a research question: *what, exactly, does masked-gene pretraining on expression learn, and when does it beat the data's own principal components?* This chapter defines the tokenizations and objectives, derives what masked-gene prediction can and cannot be expected to learn, and runs a controlled miniature on simulated atlases with known cell types, studies, depth differences, and a novel cell type held out of pretraining. The experiment compares the foundation-model embedding with PCA and a variational autoencoder, separates three kinds of metrics (label transfer, novel-type detection, study mixing), and tests whether a contrastive depth-invariance objective changes the result. It then reviews the evidence, with grades, and gives practical guidance.
    **Prerequisites.** Chapters 8, 12, 13, 17, 25, 30, 43, 45, 46.
    **You will be able to:** (1) describe rank-value, binned-value, continuous-value, and text tokenizations of expression and what each discards; (2) say what masked-gene pretraining learns at the optimum; (3) evaluate an embedding with a *paired* set of metrics that no uninformative embedding can win; (4) interpret the zero-shot literature; (5) decide when to use a foundation model, a classical baseline, or both; (6) design the pretraining study that separates scale, objective, and tokenization.

---

## 38.0 The models, and the claim

**Corpora.** Public single-cell collections (CELLxGENE, the Human Cell Atlas, GEO-derived compendia, and perturbation atlases such as Tahoe-100M and X-Atlas/Orion) provide tens to hundreds of millions of cells. Their *effective* diversity is lower than the count suggests: a few hundred thousand cells per tissue from a handful of laboratories, with strongly shared batch structure, and heavy over-representation of immune cells, tumors, and a few model tissues.

| Model (year) | Tokens | Objective | Notes |
|---|---|---|---|
| scBERT (2022) | Binned expression per gene | Masked expression | Early; cell-type annotation |
| Geneformer (2023) | Gene identities ordered by *rank* of expression (normalized by corpus gene medians) | Masked gene | In-silico perturbation by moving a gene's rank |
| scGPT (2024) | Gene tokens with binned expression values | Generative (autoregressive-like) with attention masks | Multi-task fine-tuning |
| scFoundation (2024) | Continuous expression with read-depth-aware pretraining | Masked expression | ~19k protein-coding genes |
| UCE (2023–25) | Genes represented by protein-language-model embeddings | Contrastive/masked | Zero-shot across species |
| TranscriptFormer (2025), Nicheformer (2025) | Gene tokens; spatial and dissociated data | Generative | Cross-species, spatial |
| State Embedding (Arc, 2025) | Cell sets | Set-level objective | Basis for State Transition (Chapter 39) |
| C2S-Scale (2025) | A cell as a *sentence* of rank-ordered gene names for a text LLM (27B parameters) | Language modeling | A hypothesis from the model was validated in vitro as a conditional amplifier of antigen presentation (silmitasertib with low-dose interferon) |

**The claim.** Pretraining on atlas-scale data teaches *gene–gene regulatory structure* and a *universal cell-state geometry*, so that (i) zero-shot embeddings are better than those computed from the new data alone, (ii) fine-tuning with few labels beats training from scratch, (iii) in-silico perturbation predicts the effect of knockouts, and (iv) attention maps reveal regulatory networks.

!!! lens "Research lens: what information a cell FM uses and ignores"
    *Uses*: the co-expression structure of the pretraining corpus, as a conditional of each gene given the others, in whatever form the tokenization preserves. *Ignores*: anything the tokenization discards (magnitudes beyond rank; genes beyond the top $K$), the batch and depth generating process unless modeled, and any information about *cause*: all data are observational (Chapter 44), so the model cannot know what would happen under intervention.

---

## 38.1 Tokenization and objectives, from first principles

**Rank-value encoding.** For a cell with normalized counts $x_g$, divide each gene by its median nonzero value $m_g$ across the pretraining corpus, sort genes by $x_g/m_g$ decreasing, and keep the top $K$ gene identities as a sequence $(g_{(1)},\dots,g_{(K)})$. This is *invariant to any monotone transformation of the cell's expression* (so it is robust to depth, to a global scale, and to some batch effects), and it *discards* magnitudes, ties, and all genes below rank $K$; the corpus-median normalization lowers the rank of ubiquitously high genes, so cell-specific genes rise.

**Binned values.** Each (gene, bin) pair is a token or the bin is added to the gene embedding; magnitudes are kept at the resolution of the bins, and depth must be handled by normalization.

**Text sentences.** The ordered list of gene *names* is a sentence that a pretrained language model can read, so that the language model's knowledge of gene names enters (potentially useful, potentially a source of literature bias).

!!! math "Derivation: what masked-gene prediction learns at the optimum"
    For masked-token prediction with cross-entropy, the loss is minimized by the *true conditional distribution* of the masked token given the visible ones, $p(g_{(j)}=g\mid\text{context})$. For a rank-encoded cell this is the probability that gene $g$ occupies a given rank position given the other genes: it encodes **co-expression and ordering** (genes that tend to appear together near the top of the list), and *nothing in the objective rewards separating cell types* except insofar as separating them helps predict masked genes. In a data set where a few hundred housekeeping and highly expressed genes dominate every cell's top ranks, masked prediction can be solved by memorizing that shared scaffold; cell identity enters through the minority of tokens that differ. The embedding of a cell (the mean of its token representations) is a *by-product*: it contains cell-type information to the extent that the conditionals do, which is a weaker requirement than being optimized for it. The contrast with a *cell-level* objective (contrastive: two views of the same cell must embed together and apart from others) is that the contrastive objective *requires* the embedding to identify the cell among others, and *invariance* to the augmentation (depth thinning, dropout) is built in. $\square$

**Complexity.** Sequence length $K$ (commonly 2,048 for rank-based models) makes attention $O(K^2)$ per cell; pretraining on $10^8$ cells with $K=2{,}048$ is $10^8\times$ a 2,048-token sequence, comparable to a text corpus of $2\times10^{11}$ tokens, while the *information* per token is far lower than in language (tokens within a cell are an unordered set made into a sequence by sorting).

---

## 38.2 A controlled miniature

**Design.** A simulated atlas with a known truth (Gamma–Poisson counts as in Chapter 25): 1,000 genes; 12 cell types in 4 lineages (lineage-level and type-level expression programs); a **pretraining corpus** of 19,800 cells from 6 studies with different depths (mean UMI about 2,000–4,000) and gene-specific batch effects, containing 11 of the types; and a **new study** of 2,400 cells with its own batch effect, shallower sequencing (mean UMI about 1,260), five known types and one **novel type** never seen in pretraining. Embeddings are evaluated on a pooled set (3,000 labeled corpus cells plus the new study) with four metrics: (i) **label transfer** (k-nearest-neighbor classification of new-study cells by reference labels; chance about 0.2); (ii) **novel-type detection** (AUROC of mean distance to the reference for separating novel-type cells); (iii) the **cell-type silhouette** in the new study; (iv) **study separation within cell type** (absolute silhouette of the study label within each type; 0 means well mixed). The baselines see the new data (PCA on 300 variable genes or on all genes with 30 components; the negative-binomial VAE of Chapter 30 without a batch covariate, trained for 60 epochs). The **mini-FM** is a four-layer transformer (width 64) over the top-96 rank-encoded genes, pretrained *only on the corpus* by masked-gene prediction, used zero-shot (it never sees the new study), at two data sizes (2,000 and 19,800 cells), plus the **same architecture with random weights** as a control for what architecture and encoding alone provide.

```python
--8<-- "code/ch38_sc_fm.py"
```

```text
@@OUTPUT@@
```

**Reading the results.**

1. **PCA is a very strong baseline in this world.** PCA on all genes reaches 0.96 label-transfer accuracy and PCA on 300 variable genes 0.94, with novel-type detection AUROC of 1.00. The NB-VAE without a batch covariate reaches 0.58: its latent space is dominated by the depth and batch differences it was not told about (study separation 0.55).
2. **The mini-FM has not learned cell types.** Zero-shot label transfer is 0.49–0.51 at both pretraining sizes, *the same as the random-weight network (0.50)*: ten times as many cells (19,800 against 2,000) did not help. The cell-type silhouette is 0.04–0.09 against 0.29–0.37 for PCA. The architecture and rank encoding by themselves, with random weights, give the same label transfer as the pretrained model.
3. **What pretraining did change**: novel-type detection rose from 0.60 (random weights) to 0.98–0.995, i.e., pretraining taught the embedding *which cells look like the pretraining corpus*, which is sufficient to flag a new type but not to separate known ones.
4. **Mixing metrics reward uninformative embeddings.** The mini-FM mixes studies better than any other method (study separation 0.035–0.047, against 0.46–0.52 for PCA), but its cell-type silhouette is near zero: an embedding that has discarded most structure mixes studies trivially. *Batch integration scores must always be reported together with a metric that an uninformative embedding fails* (label transfer, silhouette on type).

@@CONTRAST@@

!!! lens "Research lens: assumptions of the experiment"
    A small simulated atlas (1,000 genes; known and simple structure), a four-layer transformer trained for 800 steps with no tuning, a rank encoding of only the top 96 genes, one random seed per condition, and a world in which PCA suffices. A larger model, a longer schedule, or real atlases (with continuous cell-state structure and shared but heterogeneous programs) could change the picture; the experiment is a *template for isolating factors* (data, objective, tokenization, baseline, metric), not a verdict on published models.

---

## 38.3 The evidence on real foundation models

- **Zero-shot embeddings.** Kedzierska et al. (*Genome Biology* 2025) evaluated Geneformer and scGPT without fine-tuning: on cell-type clustering and batch integration they were frequently outperformed by highly-variable-gene selection, Harmony, and scVI, and performance on the same task varied unpredictably with the pretraining data and the data set [[S]]. Similar conclusions about weak gains over simple baselines appear in several other benchmark studies of 2024–2025 (Boiarsky et al.; Liu et al.; PertEval-scFM for perturbation effects) [[S]].
- **Fine-tuning.** With labeled data, fine-tuned models perform competitively for annotation and for specialized tasks; the margin over fine-tuned simple models is modest and data-set dependent [[P]].
- **In-silico perturbation.** Geneformer-style in-silico deletion nominated candidate therapeutic targets that were validated in an iPSC-derived cardiomyocyte model (Theodoris et al., *Nature* 2023), a case where a model's output led to an experiment that succeeded [[P]]; whether in-silico perturbation predicts knockout effects better than co-expression is not established, and the perturbation benchmarks of Chapter 39 point to no advantage over simple baselines on standard metrics [[S]].
- **Cross-species zero-shot.** UCE's use of protein-language-model embeddings as gene tokens allows embedding cells from species not seen in training with useful, if not state-of-the-art, performance [[P]].
- **Language-model "cell sentences".** C2S-Scale (Google/Yale, October 2025) reported a model-derived hypothesis (a kinase inhibitor, silmitasertib, would raise antigen presentation only in an interferon-rich context) that was confirmed in vitro with about a fifty percent increase in a surface marker; this is a single preclinical confirmation of a hypothesis generated by a model, not a measure of how often such hypotheses are right [[P]].
- **Scaling.** No established scaling law relates pretraining data or parameters to downstream cell-FM performance; evidence for and against monotone gains exists [[H]].

**Why might zero-shot cell FMs lag?** The leading hypotheses, ranked in Chapter 58 (Case 1), are: the masked-gene objective rewards co-expression, not identity; rank or binned tokenization discards information; benchmarks saturate with simple baselines; pretraining acquires depth and batch shortcuts; and effective data diversity is far below cell counts.

---

## 38.4 Practical guidance

1. **Always run the baseline** (HVG + PCA; scVI or a batch-aware VAE; Harmony), on the same split, with the same tuning effort, and report the paired metrics (label transfer, rare-type recovery, novel-type detection, silhouette) with a mixing score.
2. **Use a foundation model as a feature extractor plus a simple head**, and fine-tune only with labels; compare against a fine-tuned simple model.
3. **Check for novelty**: the model's confidence on cells unlike its corpus (distance-based or conformal; Chapters 18, 45).
4. **Read depth and batch as nuisance**: test that embeddings do not predict depth or study beyond what biology explains (a probe, Chapter 48).
5. **Record corpus, tokenization, normalization**, because results depend on them.
6. **Do not interpret attention as a regulatory network** without perturbation validation (Chapters 44, 48).
7. **Budget**: a foundation model's cost is justified only if it beats the baseline by a margin larger than its tuning variance on *your* data.

---

## 38.5 Worked research examples

!!! example "Worked Research Example 38.1: Annotating a new tumor dataset"
    **Situation.** A lab has 120,000 cells from 14 tumors profiled in two batches and wishes to annotate cell types, including possible novel malignant states. A collaborator suggests a cell foundation model.

    **Question.** What pipeline would you use, and how would you decide whether the model helps?

    **Reasoning (Expert Chain).**

    1. **L1 Goal.** Annotate known immune and stromal types accurately, flag novel malignant states, and avoid patient- or batch-driven clusters.
    2. **L3–L5 Assumptions and failure modes.** Malignant cells are patient-specific and absent from healthy atlases (a support shift, Chapter 45); an embedding trained on healthy tissues will map them to the nearest healthy type with confidence; batch and patient are confounded.
    3. **L9–L10 Pipeline and experiments.** (a) Baseline: HVG + PCA, Harmony or scVI with patient/batch as covariates (taking the cautions of Chapter 30 on overcorrection), reference-based label transfer from an immune atlas with a rejection threshold. (b) Foundation model: extract zero-shot embeddings and fine-tune with the labeled immune cells. (c) *Evaluation*: hold out one tumor entirely; report label-transfer accuracy per type, rare-type recall, novel-state detection (a flagged fraction in malignant cells versus known types), and cluster-by-patient silhouette. Use cluster-wise manual validation by marker genes for 10% of cells as an independent label.
    4. **L11 Interpretation.** If the foundation model improves rare-type recall and novel-state flagging by a margin beyond the seed-to-seed variance and does not increase patient separation, adopt it as an additional feature; if the margin is within noise, use the baseline.

    **Expert analysis.** The question is not "is the foundation model good?" but "does it add on the metrics that are hard for the baseline (rare and novel states) without adding shortcuts?", which §38.2 shows needs a paired metric set.

!!! example "Worked Research Example 38.2: Should a consortium build its own cell foundation model? No known answer"
    **Situation.** A consortium has 10 million cells from 40 studies and a budget for one pretraining run of about 5 GPU-months. The team is split between building a 300M-parameter model and investing in better baselines and benchmarks.

    **Question.** How would you decide?

    **Reasoning.**

    1. **State the decision and its value.** The value of the model is the improvement over the best baseline on the consortium's *actual downstream tasks*; so list them (annotation, rare-state detection, cross-study integration, perturbation) with weights.
    2. **Pilot cheaply** (about 5% of the budget): the factorial pretraining study of Chapter 58 (Case 1) at 10–50 million parameters, using subsets (1%, 10%, 100% of the data, diverse versus redundant) and three objectives, evaluated on the paired metric set with PCA/scVI as references and ceilings.
    3. **Read the pilot as a scaling forecast**: if downstream metrics improve log-linearly with data and diversity and the extrapolated 300M model beats the baseline by more than the tuning variance on the tasks that matter, proceed; if the objective ablation gives larger gains than a ten-fold data increase, invest in the objective first; if all arms are at the baseline, invest in the benchmark and the data (paired measurements, perturbations).
    4. **Pre-register** the decision thresholds and the tasks; budget the pilot as part of the cost.
    5. **Hedge.** Release the pilot data, code, and benchmark; the consortium's *evaluation suite* is a durable asset even if the model underperforms.

    **What is not known.** The scaling behavior of cell FMs; the pilot is the experiment that estimates it for this consortium's data.

---

## 38.6 Researcher's Notebook

!!! notebook "Researcher's Notebook: a cell-FM evaluation you can run in a day"
    1. **Simulate** an atlas with known types, studies, and a novel type (the script of §38.2) and confirm that your metric suite separates an informative embedding from an uninformative one.
    2. **Add a random-weight control** for any pretrained model.
    3. **Report four numbers** (label transfer, novel-type detection, type silhouette, study separation) and a baseline for each.
    4. **Vary one factor at a time**: data size, objective, tokenization, $K$.
    5. **Repeat with three seeds** and report the variance.
    6. **Replace the simulation with a real held-out study** and compare conclusions.
    7. **Write the null result.**

    **What it teaches.** The model is a hypothesis about what structure matters; the controls say whether the hypothesis is true.

    **An open question to carry forward.** Rank encoding is invariant to monotone transformations, so it is robust to depth; but it also discards the *magnitude* information that distinguishes states with the same ordering and different strength (activation levels, dose responses). Is there a tokenization that preserves rank robustness while retaining magnitude (for example, tokens for rank *and* a quantile-normalized value), and does it change the answer to the question "do cell foundation models learn state strength"? A perturbation-response dataset with dose gradients (Chapter 39) could test it directly.

---

## 38.7 Connections

- **Backward:** transformers (Chapter 12); representation learning and the objective (Chapter 13); scaling (Chapter 17); the measurement model (Chapter 25); classical single-cell methods (Chapter 30); benchmarks and shift (Chapters 43, 45); design (Chapter 46).
- **Forward:** perturbation and virtual-cell models (Chapter 39); multimodal alignment (Chapter 40); mechanistic interpretability (Chapter 48); the open problems of cells (Chapter 51); the case study (Chapter 58, Case 1).

!!! takeaways "Key takeaways"
    1. Single-cell foundation models differ in tokenization (rank, binned, continuous, text) and objective (masked or generative); at the optimum, masked-gene prediction learns conditionals among genes, not cell identity.
    2. Independent zero-shot evaluations found several models behind HVG + PCA, scVI, and Harmony on cell-type separation and batch integration [[S]].
    3. In a controlled miniature, PCA reached label-transfer accuracy 0.94–0.96; a rank-encoding transformer pretrained on 2,000 or 19,800 cells reached 0.49–0.51, *equal to the random-weight network (0.50)*; pretraining improved only novel-type detection (0.98–0.995 against 0.60).
    4. **Batch-mixing scores reward uninformative embeddings**: the mini-FM had the best study mixing (0.04 against 0.46–0.52) and almost no cell-type structure.
    5. @@TAKEAWAY5@@
    6. Fine-tuning with labels, in-silico perturbation, and cross-species transfer have positive but modest and unreplicated evidence; scaling laws are unestablished [[P]]/[[H]].
    7. Practical rule: always run HVG + PCA and scVI/Harmony baselines, use paired metrics with a ceiling, and treat attention maps as hypotheses.

---

## Further reading

- Yang, F. et al. (2022). scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. *Nat. Mach. Intell.* 4, 852–866. Theodoris, C. V. et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. Cui, H. et al. (2024). scGPT. *Nat. Methods* 21, 1470–1480. Hao, M. et al. (2024). Large-scale foundation model on single-cell transcriptomics. *Nat. Methods* 21, 1481–1491.
- Rosen, Y. et al. (2023). Universal cell embeddings: a foundation model for cell biology. *bioRxiv.* Pearce, J. D. et al. (2025). A cross-species generative cell atlas across 1.5 billion years of evolution: the TranscriptFormer single-cell model. *bioRxiv.* Schaar, A. C. et al. (2024). Nicheformer: a foundation model for single-cell and spatial omics. *bioRxiv.* Adduri, A. K. et al. (2025). State (Arc Institute, preprint). Levine, D. et al. (2023). Cell2Sentence: teaching large language models the language of biology. *bioRxiv.*
- Kedzierska, K. Z., Crawford, L., Amini, A. P. & Lu, A. X. (2025). Zero-shot evaluation reveals limitations of single-cell foundation models. *Genome Biol.* Boiarsky, R. et al. (2023). A deep dive into single-cell RNA sequencing foundation models. *bioRxiv.* Wenteler, A. et al. (2024). PertEval-scFM. *arXiv.* Luecken, M. D. et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nat. Methods* 19, 41–50.
- Lopez, R., Regier, J., Cole, M. B., Jordan, M. I. & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nat. Methods* 15, 1053–1058. Korsunsky, I. et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nat. Methods* 16, 1289–1296.
