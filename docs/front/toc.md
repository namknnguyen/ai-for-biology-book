# Complete Table of Contents

Each chapter lists its principal teaching goal. Chapters in Parts VII–X carry the most weight: the book compresses introductory material before it compresses research reasoning.

---

## Front matter

- [How to Use This Book](how-to-use.md)
- **Complete Table of Contents** (this page)
- [Dependency Graph](dependency-graph.md): how the major concepts build on one another
- [Learning Trajectory](learning-trajectory.md): beginner → competent → advanced → independent researcher
- [Map of the Fields](field-map.md): how ML, genomics, protein science, chemistry, evolution, statistics, causality, neuroscience, and AI-for-science connect
- [Research Skills](research-skills.md): the skills the book is designed to build, and where each is trained
- [Notation and Conventions](notation.md)

---

## Part I — Orientation

**[1. The Anatomy of a Research Problem in AI for Biology](../chapters/ch01-expert-chain.md)**
Biology as a latent system seen through measurements; the Four Gaps (measurement, objective, inference, generalization); the twelve-link Expert Chain; a first complete reasoning example; preview of the Ten Attacks.

---

## Part II — Mathematical and Computational Foundations

*Taught efficiently, and only as far as research needs.*

**[2. Linear Algebra as the Language of Representation](../chapters/ch02-linear-algebra.md)**
Vectors, matrices, tensors; eigendecomposition and SVD; low-rank structure; PCA as optimal compression; einsum and tensor shapes; computational cost.

**[3. Calculus and Optimization: How Models Learn](../chapters/ch03-calculus-optimization.md)**
Gradients and Jacobians; the chain rule as a data structure; gradient descent, stochasticity, momentum, Adam; conditioning and loss geometry; why optimization finds solutions that generalize.

**[4. Probability, Statistics, and Bayesian Reasoning](../chapters/ch04-probability-statistics.md)**
Likelihood and maximum likelihood; counts and overdispersion (the distributions of sequencing data); Bayes and hypothesis testing as belief updating; multiple testing and FDR; bootstrap; hierarchical models.

**[5. Information Theory: Entropy, Compression, and Evolution as a Channel](../chapters/ch05-information-theory.md)**
Entropy, cross-entropy, KL, mutual information; data-processing inequality; compression and prediction; information content of genomes; selection as a noisy channel.

**[6. Computational Thinking and Research Implementation](../chapters/ch06-computational-thinking.md)**
Algorithmic thinking and dynamic programming; complexity; genomic data structures; PyTorch research workflow, tensor-shape discipline, reproducibility, and experiment hygiene.

---

## Part III — Core Machine Learning

**[7. Statistical Learning: Generalization, Inductive Bias, and Classical Models](../chapters/ch07-statistical-learning.md)**
Empirical risk minimization; bias–variance; double descent; regularization; linear models, trees, kernels; the $p \gg n$ regime; what "generalize" means when data are not i.i.d.

**[8. Latent-Variable Models and Probabilistic Inference](../chapters/ch08-latent-variable-models.md)**
Mixtures, factor models, HMMs, EM, variational inference; the ELBO; latent-variable thinking as the common structure of single-cell models, phylogenetics, and generative models.

---

## Part IV — Deep Learning

**[9. Neural Networks and Backpropagation](../chapters/ch09-backpropagation.md)**
Full derivation of backpropagation; reverse-mode autodiff built from scratch; initialization, normalization, residual connections; debugging training.

**[10. Convolutional Networks and Sequence Motifs](../chapters/ch10-convolutional-models.md)**
Convolution as motif detection; receptive fields, dilation, pooling; why CNNs became the first successful genomic models.

**[11. Recurrent, State-Space, and Long-Convolution Sequence Models](../chapters/ch11-sequence-models-ssm.md)**
RNN/LSTM; state-space models (S4, Mamba); implicit long convolutions (Hyena); the long-context problem for genomes.

**[12. Attention and the Transformer](../chapters/ch12-attention-transformer.md)**
Attention derived from retrieval; multi-head attention; positional encodings (sinusoidal, rotary); complexity and FlashAttention; implementation and tensor shapes; attention for biological sequences.

**[13. Representation Learning and Self-Supervision](../chapters/ch13-representation-learning.md)**
Autoencoding, masked and autoregressive modeling, contrastive learning (InfoNCE derivation), non-contrastive methods; what makes a representation good; probing; identifiability.

**[14. Generative Models I: Likelihood, VAEs, Flows, GANs, and Energy-Based Models](../chapters/ch14-generative-models-1.md)**
Autoregressive models; VAE derivation; normalizing flows; adversarial and energy-based models; discrete data.

**[15. Generative Models II: Diffusion, Score Matching, and Flow Matching](../chapters/ch15-diffusion-flow-matching.md)**
DDPM derived from variational inference; score matching; SDE and probability-flow views; flow matching; guidance; discrete and manifold diffusion.

**[16. Geometric Deep Learning: Graphs, Symmetry, and Structure](../chapters/ch16-geometric-deep-learning.md)**
Message passing; invariance and equivariance; SO(3)/SE(3) networks; molecular graphs; invariant point attention; structure-aware diffusion.

**[17. Foundation Models: Scaling, Adaptation, and Alignment](../chapters/ch17-foundation-models-scaling.md)**
Scaling laws derived and critiqued; emergence; fine-tuning, LoRA, prompting; reinforcement learning from feedback; the *objective gap* for biological foundation models.

**[18. Interpretability and Evaluation of Deep Models](../chapters/ch18-interpretability-evaluation.md)**
Attribution methods and their failures; probing; sparse autoencoders; calibration and uncertainty; the logic of ablations and controls.

---

## Part V — Biology for Modeling

*Every concept is taught through: what is it → what information → how generated → how measured → how represented → what varies → what ML can learn → what ML cannot observe.*

**[19. The Cell as an Information-Processing System](../chapters/ch19-cell-information-system.md)**
Cell architecture for modelers; the central dogma as information flow; scales and timescales; observable versus latent variables.

**[20. Genomes, Mutation, and Variation](../chapters/ch20-genomes-variation.md)**
Genome structure and content; types and rates of mutation; variation across individuals and species; reference genomes and their biases.

**[21. Population Genetics and Evolutionary Models](../chapters/ch21-population-genetics-evolution.md)**
Drift, selection, mutation, recombination; coalescent theory; linkage disequilibrium; phylogenetics and substitution models; constraint; why evolution is the data-generating process behind sequence models.

**[22. Gene Regulation: From Sequence to Expression](../chapters/ch22-gene-regulation.md)**
Promoters, enhancers, transcription factors, chromatin, 3D genome, RNA processing, translation; the assays and what each observes; the regulatory code as a modeling target.

**[23. Proteins: Sequence, Structure, Function, and Evolution](../chapters/ch23-proteins.md)**
Folding and the energy landscape; structure hierarchy; coevolution; function and fitness; experimental structure determination and deep mutational scanning.

**[24. Small Molecules, Interactions, and the Chemistry of Drug Discovery](../chapters/ch24-molecules-drug-discovery.md)**
Molecular chemistry for modelers; binding thermodynamics and kinetics; pharmacology and ADMET; assays and their noise; the drug-discovery pipeline and its economics.

**[25. Single-Cell, Spatial, and Perturbation Biology](../chapters/ch25-single-cell-spatial-perturbation.md)**
Cell types and states; scRNA-seq, ATAC, multiome; spatial technologies; CRISPR screens and Perturb-seq; measurement models and batch effects.

**[26. Statistical Genetics: From Genotype to Phenotype](../chapters/ch26-statistical-genetics.md)**
Mixed models, heritability, GWAS, linkage disequilibrium, fine-mapping, QTLs, Mendelian randomization; why prediction is not causation.

---

## Part VI — Computational Biology

**[27. Biological Sequences: Alignment, Profiles, and Classical Genomics](../chapters/ch27-sequences-alignment.md)**
Pairwise and multiple alignment; profile HMMs; read mapping; variant calling; genome annotation; what classical pipelines assume.

**[28. Representing Biological Data](../chapters/ch28-representing-biological-data.md)**
Tokenization of sequences; structure, graph, and molecular representations; count matrices; ontologies; datasets and the technologies behind them.

**[29. Classical Models of Regulation, Evolution, and Coevolution](../chapters/ch29-classical-models.md)**
PWMs, gkm-SVM, Potts models and direct-coupling analysis, Rosetta-style energy functions: what deep models subsume, and what they inherited.

**[30. Single-Cell Computational Methods Before Foundation Models](../chapters/ch30-single-cell-methods.md)**
Normalization, scVI, integration, trajectories, RNA velocity, differential expression, early perturbation models.

---

## Part VII — Biological Foundation Models

**[31. Sequence-to-Function Models of the Genome](../chapters/ch31-sequence-to-function.md)**
DeepSEA → Basenji → Enformer → Borzoi → AlphaGenome: architectures, objectives, evaluation, and the limits of predicting from reference sequence.

**[32. Genomic Language Models](../chapters/ch32-genomic-language-models.md)**
DNABERT, Nucleotide Transformer, HyenaDNA, Caduceus, GPN, Evo and Evo 2: tokenization, objectives, likelihood as fitness, benchmark critiques, genome-scale design.

**[33. RNA, Splicing, and Translation Models](../chapters/ch33-rna-splicing-translation.md)**
SpliceAI and successors; RNA language models; translation-efficiency and stability models; mRNA design.

**[34. Protein Language Models and the Evolutionary Signal](../chapters/ch34-protein-language-models.md)**
ESM, ProGen, ProtTrans; why masked-LM log-likelihood predicts fitness; phylogenetic bias; ProteinGym; multimodal protein models.

**[35. Protein Structure Prediction](../chapters/ch35-structure-prediction.md)**
AlphaFold 2 dissected; AlphaFold 3 and open reproductions; what is and is not learned; ensembles, dynamics, and cofolding.

**[36. Generative Protein and Biomolecular Design](../chapters/ch36-protein-design.md)**
Inverse folding, diffusion-based backbone and all-atom design, binder and antibody design, enzymes; design–build–test, hit rates, and biosecurity.

**[37. Molecular Machine Learning and AI Drug Discovery](../chapters/ch37-molecular-ml-drug-discovery.md)**
QSAR to GNNs; docking and cofolding; affinity prediction; generative chemistry; benchmark pathologies; prospective validation; Isomorphic-style integrated engines.

**[38. Single-Cell Foundation Models](../chapters/ch38-single-cell-foundation-models.md)**
scBERT, Geneformer, scGPT, UCE, TranscriptFormer, State embeddings: tokenization, objectives, zero-shot evaluations, and what pretraining learns.

**[39. Perturbation Prediction and the Virtual Cell](../chapters/ch39-perturbation-virtual-cell.md)**
GEARS, CPA, scGen; linear baselines; Virtual Cell Challenge; metric pathologies; causal formulation; large Perturb-seq atlases.

**[40. Multimodal, Spatial, and Multi-Scale Models](../chapters/ch40-multimodal-multiscale.md)**
Multi-omics integration; spatial foundation models; histology–genomics; language–biology interfaces; what "multimodal" buys.

**[41. Genotype-to-Phenotype Models](../chapters/ch41-genotype-phenotype.md)**
Variant-effect predictors, regulatory and coding; polygenic prediction and deep models; personal-genome prediction; clinical use and its constraints.

**[42. Evolutionary Modeling with Deep Learning](../chapters/ch42-evolutionary-modeling.md)**
Fitness landscapes and epistasis; directed evolution; ancestral reconstruction; phylogeny-aware learning; evolution as pretraining signal.

---

## Part VIII — Research Methodology

**[43. Benchmarks, Leakage, and the Science of Evaluation](../chapters/ch43-benchmarks-evaluation.md)**
Split design (homology, species, cell-type, perturbation); baselines and noise ceilings; statistical power; Goodhart's law; prospective evaluation.

**[44. Causal Inference for Biology](../chapters/ch44-causal-inference.md)**
Structural causal models; identification; interventions in perturbation data; invariance; causal representation learning; Mendelian randomization.

**[45. Distribution Shift, Confounding, and Representation Failures](../chapters/ch45-distribution-shift.md)**
Types of shift; batch effects; shortcut learning; collapse; calibration under shift; diagnosing representation failure.

**[46. Experimental Design and Biological Validation](../chapters/ch46-experimental-design.md)**
Assay noise and replicate structure; power; MPRA, DMS, and Perturb-seq design; active learning; lab-in-the-loop; orthogonal validation.

**[47. Scaling, Compute, and Data Economics](../chapters/ch47-scaling-compute-data.md)**
What scaling means when the data are evolutionary; compute accounting; data limits; synthetic data; cost per informative data point.

**[48. Mechanistic Interpretability and Mechanistic Understanding](../chapters/ch48-mechanistic-interpretability.md)**
What it means to understand; sparse autoencoders on biological models; validating features experimentally; from model internals to biological discovery.

---

## Part IX — The Frontier

**[49. Competing Paradigms: A Comparative Map](../chapters/ch49-competing-paradigms.md)**
Representation, scale, architecture, objective, data, compute, generalization, validity, interpretability, failure modes, and appropriate use, compared across paradigms.

**[50. Open Problems I: Genomes, Regulation, and Variants](../chapters/ch50-open-problems-genomes.md)**
Why long-range regulation, cell-type specificity, personal-genome prediction, and noncoding variant interpretation remain hard.

**[51. Open Problems II: Cells, Perturbations, and Virtual Cells](../chapters/ch51-open-problems-cells.md)**
Why predicting unseen perturbations, defining a cell state, and building a "virtual cell" remain open.

**[52. Open Problems III: Proteins, Molecules, and Design](../chapters/ch52-open-problems-proteins.md)**
Dynamics, affinity, function design, out-of-distribution chemistry, and the gap between in silico and in vitro success.

**[53. Brains and Neural Systems](../chapters/ch53-open-problems-brains.md)**
Where computational neuroscience meets biological foundation models: cell types, connectomics, neural population models, NeuroAI.

**[54. AI Scientists and Automated Discovery](../chapters/ch54-ai-scientists.md)**
Agents, hypothesis generation, autonomous labs, evaluation of AI-generated science, and what remains uniquely hard.

---

## Part X — Becoming an Independent Researcher

**[55. Generating Ideas: The Ten Attacks](../chapters/ch55-idea-generation.md)**
Assumption, representation, objective, data, evaluation, scale, cross-domain transfer, reformulation, biological constraint, and biological discovery, each demonstrated on real problems.

**[56. Evaluating Ideas](../chapters/ch56-evaluating-ideas.md)**
Novelty versus superficial modification; scientific meaning; feasibility; kill criteria; Bayesian evidence for ideas; what would change your mind.

**[57. Reading, Reproducing, and Critiquing Papers; Entering a New Field](../chapters/ch57-reading-papers.md)**
A paper-dissection protocol; reproduction as experiment; rapid entry into a new literature.

**[58. Reasoning When No One Knows the Answer](../chapters/ch58-open-ended-case-studies.md)**
Extended case studies where multiple research directions are derived from the same unsolved problem.

**[59. From Idea to Publication: Research Programs, Ethics, and Biosecurity](../chapters/ch59-from-idea-to-publication.md)**
Turning ideas into research programs and papers; collaboration with wet labs; responsible research; capstone program designs.

---

## Appendices

- [A. Mathematical Reference](../appendices/a-math-reference.md)
- [B. Glossary](../appendices/b-glossary.md)
- [C. Code Companion](../appendices/c-code-companion.md)
- [D. Data and Tools](../appendices/d-data-tools.md)
- [E. Timeline of the Field](../appendices/e-timeline.md)
- [F. Annotated Reading Lists](../appendices/f-reading-lists.md)
- [G. Research Templates](../appendices/g-templates.md)
