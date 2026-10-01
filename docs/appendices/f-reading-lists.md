# Appendix F. Annotated Reading Lists

Each chapter ends with its own list; this appendix organizes the literature by *theme*, with one sentence on why each item is worth reading. The lists favor sources that taught the field something durable (a method, a benchmark, a warning) over those that were merely recent. Check the dates: items from 2025–2026 reflect the state of the field in October 2026.

**How to read a paper from this list.** Use the template of Chapter 57: state the problem, the insight, the data, the evaluation, the result, and, before reading the discussion, the assumptions and the strongest baseline the authors did not run.

---

## F.0 If you read only ten

1. Jumper et al. (2021), *Nature* 596, 583. AlphaFold 2: what a well-posed problem, a trusted benchmark (CASP), and the right inductive bias can do.
2. Vaswani et al. (2017), *NeurIPS*. Attention is all you need: the architecture beneath almost everything in Parts VI–VII.
3. Kaplan et al. (2020), *arXiv:2001.08361*, and Hoffmann et al. (2022), *NeurIPS*. Scaling and compute-optimal allocation: the language in which investment decisions are made.
4. Rives et al. (2021), *PNAS* 118, e2016239118, and Lin et al. (2023), *Science* 379, 1123. Protein language models and ESMFold.
5. Avsec et al. (2021), *Nat. Methods* 18, 1196, and Huang et al. (2023), *Nat. Genet.* 55, 2056. A strong sequence-to-function model and the sober evaluation of what it cannot do.
6. Ahlmann-Eltze, Huber & Anders (2025), *Nat. Methods* 22, 1657, and Kedzierska et al. (2025), *Genome Biol.* The 2025 corrections on perturbation and single-cell foundation models.
7. Peters, Janzing & Schölkopf (2017), *Elements of Causal Inference*. What can and cannot be learned from observation.
8. Koh et al. (2021), *ICML*, and Geirhos et al. (2020), *Nat. Mach. Intell.* 2, 665. Distribution shift and shortcut learning.
9. Elhage et al. (2022), *Toy models of superposition*, and Jonas & Kording (2017), *PLoS Comput. Biol.* 13, e1005268. How to interpret, and how to check the interpretation.
10. Gelman & Loken (2014), *Am. Sci.* 102, 460, and Ioannidis (2005), *PLoS Med.* 2, e124. Why a pipeline that tries many things needs a denominator.

---

## F.1 Learning, statistics, and generalization

- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning.* The reference for classical learning: bias–variance, regularization, model assessment.
- Efron, B. & Hastie, T. (2016). *Computer Age Statistical Inference.* Bridges statistics and machine learning; strong on the bootstrap, empirical Bayes, and selective inference.
- Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS* 116, 15849. Double descent; why interpolating models can generalize (Chapter 7).
- Dwork, C. et al. (2015). The reusable holdout: preserving validity in adaptive data analysis. *Science* 349, 636. Why repeated use of a test set invalidates it (Chapter 43).
- Gelman, A. & Loken, E. (2014). The statistical crisis in science. *Am. Sci.* 102, 460. The garden of forking paths (Chapter 54).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B* 57, 289. The multiple-testing procedure behind most omics analyses.

## F.2 Deep learning and sequence models

- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning.* The standard textbook.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS.* The transformer.
- Gu, A., Goel, K. & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR.* The state-space route to long sequences (Chapter 11).
- Dao, T. et al. (2022). FlashAttention. *NeurIPS.* Making attention memory-efficient (Chapter 12).
- Ho, J., Jain, A. & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS.* Lipman, Y. et al. (2023). Flow matching for generative modeling. *ICLR.* The generative models behind structure and design (Chapter 15).
- Bronstein, M. M., Bruna, J., Cohen, T. & Veličković, P. (2021). Geometric deep learning. *arXiv:2104.13478.* Symmetry as design principle (Chapter 16).

## F.3 Biology for modeling

- Alberts, B. et al. *Molecular Biology of the Cell.* The standard cell biology reference.
- Phillips, R., Kondev, J., Theriot, J. & Garcia, H. (2012). *Physical Biology of the Cell.* Quantitative reasoning about cellular scales and thermodynamics (Chapters 19, 22).
- Alon, U. (2019). *An Introduction to Systems Biology.* Network motifs and the logic of regulation.
- Hartl, D. L. & Clark, A. G. (2007). *Principles of Population Genetics.* Nielsen, R. & Slatkin, M. (2013). *An Introduction to Population Genetics.* The evolutionary background of variation (Chapters 20, 21).
- Dill, K. A. & Bromberg, S. (2010). *Molecular Driving Forces.* Statistical thermodynamics for biologists (Chapter 22).

## F.4 Computational biology classics

- Durbin, R., Eddy, S. R., Krogh, A. & Mitchison, G. (1998). *Biological Sequence Analysis.* HMMs, alignment, and phylogeny, from first principles (Chapters 6, 27, 29).
- Eddy, S. R. (2004). What is a hidden Markov model? *Nat. Biotechnol.* 22, 1315. A short, clear introduction.
- Altschul, S. F. et al. (1990). Basic local alignment search tool. *J. Mol. Biol.* 215, 403. Karlin, S. & Altschul, S. F. (1990). Methods for assessing the statistical significance of molecular sequence features. *PNAS* 87, 2264. The statistics of alignment scores (Chapter 27).
- Berg, O. G. & von Hippel, P. H. (1987). Selection of DNA binding sites by regulatory proteins. *J. Mol. Biol.* 193, 723. The energy view of motifs (Chapters 22, 29).
- Morcos, F. et al. (2011). *PNAS* 108, E1293. Direct-coupling analysis; the coevolution route to structure (Chapter 29).

## F.5 Statistical genetics

- Visscher, P. M. et al. (2017). 10 years of GWAS discovery: biology, function, and translation. *Am. J. Hum. Genet.* 101, 5. The map of what GWAS found.
- Bulik-Sullivan, B. K. et al. (2015). LD score regression distinguishes confounding from polygenicity in GWAS. *Nat. Genet.* 47, 291. Yang, J. et al. (2011). GCTA. *Am. J. Hum. Genet.* 88, 76. Mixed models and heritability (Chapter 26).
- Wang, G. et al. (2020). SuSiE. *J. R. Stat. Soc. B* 82, 1273. Weissbrod, O. et al. (2020). PolyFun. *Nat. Genet.* 52, 1355. Fine-mapping with and without priors (Chapter 41).
- Davey Smith, G. & Ebrahim, S. (2003). 'Mendelian randomization'. *Int. J. Epidemiol.* 32, 1. Bowden, J., Davey Smith, G. & Burgess, S. (2015). Mendelian randomization with invalid instruments: MR-Egger. *Int. J. Epidemiol.* 44, 512. Causal inference from genetic instruments and its limits.
- Ding, Y. et al. (2023). *Nature* 618, 774. Polygenic score accuracy across the genetic ancestry continuum.

## F.6 Regulatory genomics and genomic models

- Zhou, J. & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning-based sequence model. *Nat. Methods* 12, 931. Alipanahi, B. et al. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nat. Biotechnol.* 33, 831. The first deep models of regulatory sequence.
- Avsec, Ž. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nat. Methods* 18, 1196. Enformer. Avsec, Ž. et al. (2026). AlphaGenome. *Nature.*
- Karollus, A. et al. (2023), *Genome Biol.* 24, 56; Huang, A. C. et al. (2023), *Nat. Genet.* 55, 2056; Sasse, A. et al. (2023), *Nat. Genet.* 55, 2060. Where the models fail.
- Nguyen, E. et al. (2024). *Science* 386, eado9336 (Evo); Brixi, G. et al. (2026). *Nature* (Evo 2); Tang, Z. et al. (2025). *Genome Biol.*; Patel, A. et al. (2024). DART-Eval. *NeurIPS D&B.* The case for and against DNA language models.
- Avsec, Ž. et al. (2021). Base-resolution models of transcription-factor binding reveal soft motif syntax. *Nat. Genet.* 53, 354. A model designed for interpretability and validated by experiment.
- Jaganathan, K. et al. (2019). *Cell* 176, 535. SpliceAI.

## F.7 Proteins, structure, and design

- Jumper, J. et al. (2021). *Nature* 596, 583; Abramson, J. et al. (2024). *Nature* 630, 493. AlphaFold 2 and 3.
- Rives, A. et al. (2021). *PNAS* 118, e2016239118; Lin, Z. et al. (2023). *Science* 379, 1123; Hayes, T. et al. (2025). *Science* 387, 850. Protein language models to ESM3.
- Notin, P. et al. (2023). ProteinGym. *NeurIPS D&B.* Gordon, C. et al. (2025). Protein language model fitness is a matter of preference. *ICLR.* The benchmark and a careful analysis of why scores vary.
- Watson, J. L. et al. (2023). *Nature* 620, 1089; Dauparas, J. et al. (2022). *Science* 378, 49; Pacesa, M. et al. (2025). *Nature* 646, 483. Design with diffusion, inverse folding, and hallucination.
- Škrinjar, P. et al. (2025). Have protein-ligand co-folding methods moved beyond memorisation? *bioRxiv.* Masters, M. R. et al. (2025). *Nat. Commun.* The evidence on co-folding generalization.
- Lewis, S. et al. (2025). *Science* 389. BioEmu and ensembles.

## F.8 Single cells, perturbations, and virtual cells

- Luecken, M. D. & Theis, F. J. (2019). Current best practices in single-cell RNA-seq analysis: a tutorial. *Mol. Syst. Biol.* 15, e8746. A practical introduction.
- Svensson, V. (2020). Droplet scRNA-seq is not zero-inflated. *Nat. Biotechnol.* 38, 147. Lopez, R. et al. (2018). *Nat. Methods* 15, 1053 (scVI). Measurement models and generative models (Chapters 25, 30).
- Luecken, M. D. et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nat. Methods* 19, 41. Integration benchmarks and their trade-offs.
- Theodoris, C. V. et al. (2023). *Nature* 618, 616 (Geneformer); Cui, H. et al. (2024). *Nat. Methods* 21, 1470 (scGPT). Kedzierska, K. Z. et al. (2025). *Genome Biol.* The models and their zero-shot evaluation.
- Replogle, J. M. et al. (2022). *Cell* 185, 2559; Norman, T. M. et al. (2019). *Science* 365, 786; Ahlmann-Eltze, C. et al. (2025). *Nat. Methods* 22, 1657; Bunne, C. et al. (2024). *Cell* 187, 7045. Perturbation data, baselines, and the virtual cell program.

## F.9 Evaluation, causality, shift, and design

- Pearl, J. (2009). *Causality.* Peters, J., Janzing, D. & Schölkopf, B. (2017). *Elements of Causal Inference.* The two standard references.
- Quiñonero-Candela, J. et al. (2009). *Dataset Shift in Machine Learning.* Koh, P. W. et al. (2021). WILDS. *ICML.* Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nat. Mach. Intell.* 2, 665.
- Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction: a gentle introduction. *Found. Trends Mach. Learn.* 16, 494. Uncertainty with coverage guarantees (Chapters 18, 45).
- Settles, B. (2009). *Active Learning Literature Survey.* Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: a review. *Stat. Sci.* 10, 273. Choosing the next experiment (Chapter 46).
- Chevalley, M. et al. (2025). A large-scale benchmark for network inference from single-cell perturbation data. *Commun. Biol.* Why interventional data did not help real-data network inference (Chapter 44).

## F.10 Interpretability

- Elhage, N. et al. (2022). Toy models of superposition. *Transformer Circuits Thread.* Bricken, T. et al. (2023). Towards monosemanticity. *Transformer Circuits Thread.* The superposition and sparse-autoencoder program.
- Hewitt, J. & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP.* Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS.* How to test an interpretation.
- Jonas, E. & Kording, K. P. (2017). Could a neuroscientist understand a microprocessor? *PLoS Comput. Biol.* 13, e1005268. Validate methods on a system whose mechanism is known.
- Adams, E. et al. (2025). *PNAS* 122; Simon, E. & Zou, J. (2024). InterPLM. *bioRxiv.* Sparse autoencoders in protein language models.

## F.11 Evolution and phylogenetics

- Felsenstein, J. (1981). *J. Mol. Evol.* 17, 368; Felsenstein, J. (1985). *Am. Nat.* 125, 1. Likelihood on trees and the comparative method.
- Yang, Z. (2014). *Molecular Evolution: A Statistical Approach.* The reference for substitution models and ancestral reconstruction.
- Harms, M. J. & Thornton, J. W. (2013). *Nat. Rev. Genet.* 14, 559. Eick, G. N. et al. (2017). *Mol. Biol. Evol.* 34, 247. Resurrecting ancestors and testing the robustness of the conclusion.
- Łuksza, M. & Lässig, M. (2014). *Nature* 507, 57; Thadani, N. N. et al. (2023). *Nature* 622, 818. Forecasting viral evolution.

## F.12 Neuroscience and computation

- Dayan, P. & Abbott, L. F. (2001). *Theoretical Neuroscience.* Gerstner, W. et al. (2014). *Neuronal Dynamics.* The standard texts.
- Dorkenwald, S. et al. (2024), *Nature* 634; Shiu, P. K. et al. (2024), *Nature* 634, 210; Lappalainen, J. K. et al. (2024), *Nature* 634, 1132. The fly connectome and models built from it.
- Wang, E. Y. et al. (2025). *Nature* 640, 470; MICrONS Consortium (2025). *Nature.* A foundation model of mouse visual cortex and functional connectomics.
- Schaeffer, R., Khona, M. & Fiete, I. R. (2022). No free lunch from deep learning in neuroscience. *NeurIPS.* Prediction versus mechanism.

## F.13 AI scientists, reproducibility, and research practice

- Gottweis, J. et al. (2025). Towards an AI co-scientist. *arXiv:2502.18864.* Ghareeb, A. E. et al. (2025). Robin. *arXiv:2505.13400.* Mitchener, L. et al. (2025). Kosmos. *arXiv:2511.02824.* Swanson, K. et al. (2025). The Virtual Lab of AI agents. *Nature.* The 2025 systems and what their evidence supports (Chapter 54).
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Med.* 2, e124. Munafò, M. R. et al. (2017). A manifesto for reproducible science. *Nat. Hum. Behav.* 1, 0021.
- Sandve, G. K. et al. (2013). Ten simple rules for reproducible computational research. *PLoS Comput. Biol.* 9, e1003285. Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLoS Comput. Biol.* 13, e1005510.
- Platt, J. R. (1964). Strong inference. *Science* 146, 347. The oldest and best argument for multiple hypotheses and decisive experiments (Chapters 55, 58).

## F.14 Biosecurity and ethics

- Wittmann, B. J. et al. (2025). Strengthening nucleic acid biosecurity screening against generative protein design tools. *Science.* Baker, D. & Church, G. (2024). Protein design meets biosecurity. *Science* 383, 349. Bloomfield, D. et al. (2024). AI and biosecurity: the need for governance. *Science* 385, 831. Why design capability and risk travel together (Chapters 32, 36, 52, 59).
