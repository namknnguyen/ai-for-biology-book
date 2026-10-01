# Appendix B. Glossary

Terms are listed alphabetically with the chapter where they are first developed (Ch.). Mathematical symbols are in the notation page (front matter) and Appendix A.

**ABC model (Activity-by-Contact).** A model that predicts enhancer–gene regulatory links from an enhancer's activity (accessibility and H3K27ac) and its 3-D contact frequency with the gene's promoter. (Ch. 22)

**Activity cliff.** A pair of structurally similar molecules with a large difference in potency. A characteristic failure case for models that smooth over similar inputs. (Ch. 24)

**ADMET.** Absorption, distribution, metabolism, excretion, and toxicity: the property set that, with potency and selectivity, determines whether a compound can become a drug. (Ch. 24)

**Additive model.** A model in which effects of separate factors (mutations, variants, perturbations) sum on some scale. See *global epistasis* for when additivity on a latent scale looks non-additive on a measured scale. (Chs. 23, 26, 39)

**Alignment (sequence).** A correspondence between residues of two or more sequences, scored by a substitution matrix and gap penalties. (Ch. 27)

**Allele frequency.** The proportion of chromosomes in a population that carry a given variant. (Ch. 21)

**Ambient RNA.** Free mRNA from lysed cells captured in every droplet of a droplet-based single-cell experiment. (Ch. 25)

**Anfinsen hypothesis.** The native structure of a (small) protein is the free-energy minimum determined by its sequence. (Ch. 23)

**Annotation (cell type).** The assignment of cell-type labels to clusters or cells; a model output rather than a measurement. (Chs. 25, 30)

**Attack (A1–A10).** One of ten structured moves for generating research ideas: assumption, representation, objective, data, evaluation, scale, cross-domain transfer, problem reformulation, biological constraint, biological discovery. (Chs. 1, 55)

**Attention.** A mechanism that computes, for each position, a weighted combination of values from all positions, with weights given by a softmax over query–key similarity. (Ch. 12)

**Attribution.** A score assigned to input features indicating their contribution to a model output (gradients, in-silico mutagenesis, DeepLIFT). Not evidence of mechanism by itself. (Ch. 18, 48)

**Backpropagation.** Efficient computation of gradients of a scalar loss with respect to all parameters via the chain rule on a computational graph. (Ch. 9)

**Batch effect.** Systematic differences between groups of samples due to experimental conditions (day, operator, protocol), not biology. (Chs. 25, 30)

**Binding free energy.** $\Delta G^\circ=RT\ln(K_d/c^\circ)$; $1$ log unit of $K_d$ is 1.37 kcal/mol at 298 K. (Ch. 24)

**BLOSUM.** Substitution score matrices derived from conserved blocks of aligned proteins; scores are log-odds ratios. (Ch. 27)

**Burst (transcriptional).** A period of promoter activity during which RNA is produced in a rapid series; produces negative-binomial counts. (Chs. 19, 25)

**Causal inference.** Estimating the effect of interventions from data; requires assumptions beyond association (identification). (Ch. 44)

**ChIP-seq.** Chromatin immunoprecipitation followed by sequencing; measures where a protein or histone mark is associated with DNA. (Ch. 22)

**Claim Ladder (C0–C4).** A hierarchy of claim strength: fit (C0), held-out prediction (C1), prediction under shift (C2), mechanistic claim (C3), intervention/design claim (C4). (Ch. 1)

**Clumping and thresholding (C+T).** A polygenic-score method that retains the lead variant per LD block above a p-value threshold. (Ch. 26)

**Coalescent.** The genealogical process describing the ancestry of a sample of sequences backward in time. (Ch. 21)

**Coevolution.** Correlated evolution of residues (e.g., contacting residues in a protein) detectable in multiple sequence alignments. (Ch. 29)

**Conformal prediction.** A distribution-free method for prediction sets with guaranteed coverage under exchangeability. (Ch. 18)

**Contrastive learning.** Learning representations by pulling together positive pairs and pushing apart negatives, with a bound on mutual information (InfoNCE). (Ch. 13)

**Counterfactual.** A quantity describing what would have happened under a different input or intervention; not directly observed. (Chs. 31, 44)

**CRISPRi / CRISPRa.** Targeted repression/activation of genes using catalytically dead Cas9 fused to repressor/activator domains. (Ch. 25)

**Cryo-EM.** Cryo-electron microscopy: structure determination from images of frozen molecules. (Ch. 23)

**Deep mutational scanning (DMS).** Measuring the functional effects of large libraries of mutations by selection and sequencing. (Ch. 23)

**Denoising (single cell).** Replacing observed counts by model-based smoothed estimates; can induce spurious dependence. (Ch. 30)

**Diffusion model.** A generative model that learns to reverse a gradual noising process. (Ch. 15)

**Direct-coupling analysis (DCA).** Inference of a Potts model from an MSA to separate direct from indirect couplings. (Ch. 29)

**Doublet.** A droplet containing two (or more) cells, producing a hybrid profile. (Ch. 25)

**Dropout (neural network).** Random zeroing of units during training as a regularizer. Not the same as single-cell "dropout". (Ch. 7)

**Dual-use.** Capabilities that can serve both beneficial and harmful purposes. (Ch. 59)

**E-value.** The expected number of alignments with a score at least as high that arise by chance in a search of given size. (Ch. 27)

**Effective sample size ($N_\text{eff}$).** The number of independent observations equivalent to a correlated sample; for MSAs, the weighted count of sequences after reweighting by similarity. (Chs. 23, 29)

**ELBO.** Evidence lower bound used to train latent-variable models by variational inference. (Chs. 8, 14)

**Enhancer.** A distal regulatory DNA element that increases transcription of a target gene. (Ch. 22)

**Epistasis.** Non-additivity of effects of mutations; *specific* (physical interaction) or *global* (nonlinearity of the readout). (Ch. 23)

**Equivariance / invariance.** A function is equivariant if transforming the input transforms the output predictably; invariant if the output is unchanged. (Ch. 16)

**Expert Chain (L1–L12).** The twelve-link reasoning procedure from problem to new directions used throughout the book. (Ch. 1)

**Expected information gain (EIG).** The mutual information between hypotheses and the outcome of an experiment. (Ch. 56)

**FM-index.** A compressed index based on the Burrows–Wheeler transform enabling exact pattern matching in time proportional to pattern length. (Ch. 27)

**Fine-mapping.** Statistical identification of causal variants in an associated locus, producing posterior inclusion probabilities and credible sets. (Ch. 26)

**Flow matching.** Training a vector field that transports a simple distribution to a data distribution by regressing on conditional velocities. (Ch. 15)

**Foundation model.** A model pretrained on broad data at scale and adapted to many tasks. (Chs. 17, 32–40)

**Four Gaps.** Measurement (G-M), objective (G-O), inference/causal (G-I), and generalization (G-G): the places where a model's score can fail to support a biological claim. (Ch. 1)

**Genome-wide association study (GWAS).** A scan of variant–phenotype associations across the genome. (Ch. 26)

**Global epistasis.** Apparent interactions caused by a nonlinear mapping from an additive latent trait to the measured phenotype. (Ch. 23)

**Hardy–Weinberg equilibrium.** Genotype frequencies $p^2,2pq,q^2$ in a randomly mating population without selection, mutation, migration, or drift. (Ch. 21)

**Heritability.** The proportion of phenotypic variance attributable to genetic variance ($H^2$ broad, $h^2$ narrow, $h^2_\text{SNP}$). (Ch. 26)

**Hidden Markov model (HMM).** A probabilistic model with latent Markov states and state-dependent emissions. (Chs. 6, 29)

**Identifiability.** A parameter or effect is identified if it is uniquely determined by the distribution of the observed data. (Chs. 30, 31, 44)

**In-silico mutagenesis (ISM).** Scoring a model's output for every possible single-base change of an input. (Ch. 18)

**Inductive bias.** Assumptions built into a model that guide generalization. (Ch. 7)

**InfoNCE.** A contrastive loss that lower-bounds mutual information. (Ch. 13)

**Instrumental variable.** A variable that affects the exposure and the outcome only through the exposure; the basis of Mendelian randomization. (Ch. 26)

**Isotherm (binding).** The relation between ligand concentration and fraction bound; a sigmoid in log-concentration. (Ch. 24)

**Karlin–Altschul statistics.** The extreme-value theory of local alignment scores: $E=Kmn\,e^{-\lambda S}$. (Ch. 27)

**Leakage.** Information about the test labels entering the training data or model selection. (Chs. 1, 7, 28, 43)

**Ligand efficiency.** Binding free energy per heavy atom. (Ch. 24)

**Linkage disequilibrium (LD).** Non-random association of alleles at different loci. (Chs. 21, 26)

**LD score regression.** A method that distinguishes confounding from polygenicity by regressing association statistics on LD scores. (Ch. 26)

**Linear mixed model (LMM).** A model with fixed effects and Gaussian random effects with a covariance given by genetic relatedness; equivalent to ridge regression on genotypes. (Ch. 26)

**MAPQ.** Mapping quality: a Phred-scaled estimate of the probability that a read's reported location is wrong. (Ch. 27)

**Mappability.** The fraction of the genome in which reads of a given length map uniquely. (Ch. 27)

**Masked language modeling.** Predicting masked tokens from context. (Chs. 12, 28)

**Mendelian randomization (MR).** Using genetic variants as instrumental variables to estimate causal effects of exposures on outcomes. (Ch. 26)

**Message passing.** A graph neural network update in which each node aggregates information from neighbors. (Ch. 16)

**Morgan fingerprint (ECFP).** A hashed count of circular substructures around each atom, equivalent to Weisfeiler–Lehman color refinement. (Ch. 24)

**Multiple sequence alignment (MSA).** An alignment of three or more homologous sequences. (Chs. 23, 27)

**Mutual information.** $I(X;Y)=\KL{p(x,y)}{p(x)p(y)}$; the information shared between variables. (Ch. 5)

**Negative binomial (NB).** A count distribution with variance $\mu+\phi\mu^2$; the Gamma–Poisson mixture. (Chs. 4, 19, 25)

**Noise ceiling.** The maximum achievable performance given measurement noise: $1-\sigma^2/\mathrm{Var}(y)$ for $R^2$. (Ch. 1)

**Pangenome.** A representation of the sequence of many individuals, typically as a graph. (Ch. 27)

**Perturb-seq.** Pooled CRISPR perturbations with single-cell RNA-seq readout. (Ch. 25)

**Phred score.** $Q=-10\log_{10}P(\text{error})$. (Ch. 27)

**Polygenic score (PGS/PRS).** A weighted sum of allele dosages predicting a phenotype. (Ch. 26)

**Position weight matrix (PWM).** A per-position log-likelihood-ratio scoring matrix for a motif. (Ch. 29)

**Potts model.** A maximum-entropy model with pairwise interactions between positions that take $q$ states. (Ch. 29)

**Pseudobulk.** Aggregating counts per sample (donor) to test differences with samples as replicates. (Ch. 25)

**Pseudoreplication.** Treating non-independent observations (cells from one donor) as independent replicates. (Chs. 4, 25)

**Pseudotime.** An ordering of cells along an inferred trajectory; not physical time. (Ch. 30)

**Reference bias.** Preferential alignment of reads to the reference allele. (Ch. 27)

**Reliability (split-half).** The correlation between independent estimates of the same quantity, used to bound achievable metrics. (Chs. 1, 25)

**Residence time.** $1/k_\text{off}$; how long a ligand stays bound. (Ch. 24)

**Reverse complement (RC).** The sequence of the opposite DNA strand read 5′→3′; RC symmetry is a property of genomes averaged over strands. (Ch. 28)

**RNA velocity.** Inference of expression dynamics from unspliced and spliced RNA. (Ch. 30)

**Scaling law.** Empirical power-law relation between loss and model size, data, or compute. (Ch. 17)

**Selection coefficient ($s$).** The relative fitness advantage of an allele. (Ch. 21)

**Single-cell foundation model.** A large pretrained model on single-cell data. (Ch. 38)

**SMILES.** A line notation for molecules; one molecule has many valid SMILES. (Ch. 24)

**Sparsity (single-cell).** The high fraction of zero counts, explained largely by low expression and sampling. (Ch. 25)

**State-space model (SSM).** A sequence model based on a linear recurrence, usable as a long convolution. (Ch. 11)

**Structural rhyme.** The same mathematics appearing in different fields. (Chapters throughout)

**Tokenization.** Mapping raw data to discrete units (nucleotides, $k$-mers, BPE tokens, codons). (Ch. 28)

**UMI.** Unique molecular identifier; a random barcode attached to each captured molecule to collapse duplicates. (Ch. 25)

**Variational autoencoder (VAE).** A latent-variable model trained by maximizing the ELBO with an inference network. (Ch. 14)

**Viterbi algorithm.** Dynamic programming for the most probable hidden-state path. (Ch. 6)

**Weisfeiler–Lehman test.** An iterative color-refinement algorithm; bounds the expressive power of message-passing GNNs. (Chs. 16, 24)

**Zero-shot.** Evaluating a pretrained model without task-specific training. (Chs. 32, 34, 38)
