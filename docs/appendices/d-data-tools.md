# Appendix D. Data and Tools

A map of the public data resources and software that the book's methods run on. Names, versions, and access terms change; the list gives the *kind* of resource, what it is good for, and its main pitfall. Check each resource's own documentation and license before use. For data about people, follow the data-use agreements and the ethics of Chapter 59.

---

## D.1 Data resources

### Sequences, genomes, and variation

| Resource | What it holds | Use in the book | Pitfall |
|---|---|---|---|
| NCBI GenBank/RefSeq, Ensembl, UCSC Genome Browser | Reference genomes and annotations | Chapters 27–29, 32, 33 (the chloroplast genome NC_000932 is a GenBank record) | Annotation versions and assembly builds differ between tools |
| T2T-CHM13; Human Pangenome Reference | A gapless human reference; a pangenome of diverse haplotypes | Chapter 50 (G5) | Most models and annotations still use GRCh38 |
| 1000 Genomes, HGDP, gnomAD | Human variation and allele frequencies across ancestries; constraint metrics | Chapters 20, 21, 26, 41, 50 | Sampling of ancestries is uneven; gnomAD excludes some disease cohorts |
| UK Biobank, All of Us, FinnGen, Biobank Japan | Genotypes, sequences, phenotypes for 10⁵–10⁶ people | Chapters 26, 41 | Access agreements; healthy-volunteer and ancestry biases |
| ClinVar, ClinGen | Clinical variant interpretations | Chapters 41, 50 (G4) | Circularity: labels are influenced by earlier predictors |
| OpenGenome2 (Evo 2 training atlas), Genome Taxonomy Database | Large multi-species genomic corpora | Chapter 32 | Taxonomic over-representation |

### Regulation and expression

| Resource | What it holds | Use | Pitfall |
|---|---|---|---|
| ENCODE, Roadmap Epigenomics, FANTOM, GTEx | Chromatin, expression, QTLs across tissues and cell types | Chapter 31 | Bulk tissues mix cell types; cell-type coverage uneven |
| JASPAR, HOCOMOCO | Transcription-factor motif matrices | Chapters 22, 29 | Motif quality varies by experimental source |
| MPRA and saturation mutagenesis data sets | Quantitative regulatory activity of designed sequences | Chapters 31, 41, 50 | Reporter context differs from the native locus |
| CRISPRi enhancer-screen collections | Enhancer–gene regulation | Chapter 50 (G2) | Few loci and cell types |

### Single cells, perturbations, and spatial data

| Resource | What it holds | Use | Pitfall |
|---|---|---|---|
| CELLxGENE Census, Human Cell Atlas | Hundreds of millions of cells, harmonized | Chapters 30, 38 | Heterogeneous annotations and depth; redundancy |
| Replogle et al. K562/RPE1 Perturb-seq; Norman et al. combinatorial CRISPRa; X-Atlas/Orion; Tahoe-100M; scPerturb | Perturbation transcriptomes at increasing scale | Chapters 39, 45, 46, 51 | Perturbation effects are mostly small; guide efficacy and fitness effects |
| 10x Visium, Xenium, MERFISH, and other spatial platforms | Spatial transcriptomes at different resolutions | Chapter 25 | Resolution–coverage trade-offs |
| DepMap, CCLE, GDSC | Cell-line dependency, expression, drug response | Chapters 37, 51 | Cell-line artifacts; poor transfer to patients |
| CZ CELLxGENE annotation, Cell Ontology | Cell-type vocabulary | Chapter 38 | Annotation noise is a ceiling for label-transfer tasks |

### Proteins, structures, and molecules

| Resource | What it holds | Use | Pitfall |
|---|---|---|---|
| PDB, PDBe | Experimental structures | Chapters 23, 35, 36 | Bias toward crystallizable, soluble, ligand-bound states |
| AlphaFold Database, ESM Atlas | Predicted structures for ≥ 10⁸ sequences | Chapters 34, 35 | Confidence varies; one conformation |
| UniProt/UniRef, Pfam/InterPro, MGnify | Sequences, families, metagenomes | Chapters 29, 34 | Sampling bias across taxa |
| MaveDB, ProteinGym, Atlas of Variant Effects | Deep mutational scans and benchmark suites | Chapters 23, 34, 41 | Each assay selects for something different |
| CASP, CAMEO, PoseBusters, Runs N' Poses | Blind or time-split structure and docking benchmarks | Chapter 35 | Check training-set similarity |
| ChEMBL, PubChem, BindingDB, MoleculeNet (BACE, ESOL, others) | Compound activities, properties | Chapters 24, 37, 47 | Assay heterogeneity; noise ceilings |
| PDBbind and CASF | Protein–ligand affinities | Chapter 37 | Leakage between training and benchmark sets |
| ZINC, Enamine REAL | Purchasable and make-on-demand chemical space | Chapter 37 | Docking score is not activity |

### Neuroscience

| Resource | What it holds | Use | Pitfall |
|---|---|---|---|
| FlyWire; the *C. elegans* connectome | Whole-brain wiring diagrams | Chapter 53 | No synaptic strengths or neuromodulation |
| MICrONS | Co-registered function and electron microscopy of mouse visual cortex | Chapter 53 | One cubic millimetre, one animal |
| Allen Brain Observatory and Brain Cell Atlas; IBL; Neuropixels data repositories | Recordings, cell types | Chapter 53 | Different tasks and preparations |

---

## D.2 Software

### General

| Tool | Purpose |
|---|---|
| NumPy, SciPy, pandas, scikit-learn | Arrays, statistics, classical learning (all chapters) |
| PyTorch (and JAX) | Differentiable programming; every neural model in the book |
| Hugging Face Transformers/Datasets | Model hubs for protein, DNA, and single-cell models |
| MkDocs Material + pymdown-extensions | This book's site |
| Git, DVC, Snakemake, Nextflow | Versioning data and pipelines (Chapter 59) |

### Genomics and statistical genetics

| Tool | Purpose |
|---|---|
| samtools/bcftools, minimap2, BWA-MEM, GATK/DeepVariant | Mapping and variant calling (Chapter 27) |
| PLINK, REGENIE, BOLT-LMM, SAIGE | GWAS and mixed models (Chapter 26) |
| SuSiE, FINEMAP, PolyFun | Fine-mapping with and without functional priors (Chapter 41) |
| LDSC, S-LDSC | Heritability partitioning and confounding (Chapter 26) |
| PRSice, LDpred2, PRS-CS, PRS-CSx | Polygenic scores (Chapter 26) |
| Hail | Large-scale genetics on distributed systems |
| MEME Suite, HOMER, pyjaspar | Motif analysis (Chapter 29) |
| ViennaRNA | RNA folding (Chapter 33) |
| IQ-TREE, RAxML, BEAST, PAML | Phylogenetics, ancestral reconstruction, selection (Chapter 42) |

### Single cell, spatial, perturbation

| Tool | Purpose |
|---|---|
| Scanpy/AnnData, Seurat, scvi-tools, Harmony | Single-cell analysis, integration (Chapters 25, 30, 38) |
| scVelo, CellRank, Waddington-OT, moscot | Dynamics and trajectory (Chapter 51) |
| Squidpy, Cell2location, Tangram | Spatial analysis and deconvolution (Chapter 25) |
| GEARS, scGen/CPA, Geneformer, scGPT, scFoundation, UCE, State | Perturbation and foundation models (Chapters 38, 39) |

### Proteins and molecules

| Tool | Purpose |
|---|---|
| AlphaFold 2/3 (code and server), ColabFold, OpenFold, Boltz-1/2, Chai-1, Protenix, RoseTTAFold | Structure and complex prediction (Chapter 35) |
| ESM-2/ESM3/ESM C, ProGen, ProtT5 | Protein language models (Chapter 34) |
| RFdiffusion, ProteinMPNN, LigandMPNN, BindCraft | Design (Chapter 36) |
| RDKit, Open Babel, Datamol | Cheminformatics (Chapters 24, 37) |
| AutoDock Vina, Glide, DiffDock | Docking (Chapter 37) |
| OpenMM, GROMACS, AMBER, FEP+ and open FEP tools | Molecular dynamics and free-energy methods (Chapters 22, 37) |
| PoseBusters | Physical validity of poses (Chapter 35) |

### Interpretability and evaluation

| Tool | Purpose |
|---|---|
| Captum, SHAP, DeepLIFT, TF-MoDISco | Attribution and motif discovery (Chapters 18, 48) |
| TransformerLens-style hooks, SAE libraries | Activation patching and sparse autoencoders (Chapter 48) |
| MAPIE, conformal prediction libraries | Uncertainty and coverage (Chapters 18, 45) |

---

## D.3 Using public data responsibly

1. **Record the version** (date, accession, release) of every data set; resources change.
2. **Keep splits reproducible**: store the cluster or clade assignments and the seeds.
3. **Read the license and the data-use agreement.** Human data may restrict re-identification, sharing, and commercial use.
4. **Check for circularity** between the data used to train an annotation (a variant-effect predictor, a cell-type annotation, a structure model) and the labels used to evaluate it.
5. **Document exclusions** (samples, genes, perturbations removed) and why.
6. **Share the code that produced each number**, and the exact commands (Appendix C).
