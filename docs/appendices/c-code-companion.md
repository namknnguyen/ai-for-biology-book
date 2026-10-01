# Appendix C. Code Companion

Every number the book reports as a result of a computation comes from a script in the `code/` directory of the repository, run once and pasted into the chapter. This appendix says how to run them, what each does, how long it takes, and what to expect when your numbers differ from the book's.

---

## C.1 Setting up

```bash
git clone <the repository>            # the book's source
cd ai-for-biology-book
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # only needed to build the site (mkdocs)
pip install -r code/requirements-code.txt       # numpy, scipy, scikit-learn, pandas, torch (CPU), rdkit, biopython, tokenizers, pyjaspar, ViennaRNA
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if you prefer the CPU-only build
```

Everything runs on a laptop CPU; there is no GPU requirement. The outputs in the book were generated on a four-core CPU, one script at a time.

```bash
python code/run_all.py --list                  # the table of scripts below, with approximate runtimes
python code/run_all.py --quick                 # all scripts estimated at ten minutes or less
python code/run_all.py --only ch26 ch35        # selected scripts (prefix match)
python code/ch39_perturbation.py               # or run any script directly
```

`run_all.py` writes each script's output to `outputs/<script>.txt` and prints a one-line summary. Scripts that need data download it on first use into `data/` (this directory is not committed): BACE-1 and ESOL from the MoleculeNet mirror (Chapters 24, 37, 47), the *Arabidopsis thaliana* chloroplast genome NC_000932 as a GenBank file (Chapters 28, 29, 32, 33), and PDB entries (Chapter 35).

---

## C.2 Conventions in the code

- **One file, one chapter.** The file name starts with the chapter number. A suffix such as `b` marks a second experiment for the same chapter (`ch32b`, `ch36b`, `ch38b`).
- **Tensor shapes are commented.** Where a model manipulates arrays or tensors, each line states the shape: `B` batch, `L` length, `d` width, `H` heads.
- **Seeds are fixed in the file.** A script prints the same numbers on the same library versions. Results that depend on thread counts (torch training with several threads) can differ in the last digits.
- **A script states its assumptions in its docstring** and prints a table whose columns are those discussed in the chapter. The chapter pastes this output verbatim in a `text` block directly below the code.
- **Verification before use.** Every simulation is accompanied by a check against theory or brute force where one exists (Chapter 6: dynamic programming against exhaustive search; Chapter 9: autograd against finite differences; Chapter 21: drift and coalescent against analytic expectations; Chapter 27: Karlin–Altschul statistics against empirical score distributions).
- **Failures are reported.** When a first version of an experiment failed to show the expected effect, the chapter says so (for example, Chapter 35's random-start optimizer, which failed with all true contacts until a distance-geometry initialization was used; Chapter 38's miniature foundation model, which did not learn cell types).

---

## C.3 The scripts

| Chapter | Script | What it demonstrates | ~min | Needs |
|---|---|---|---|---|
| 1 | `ch01_noise_ceiling.py` | What "$R^2=0.7$" means: noise ceilings and split leakage | 0.2 | |
| 2 | `ch02_linear_algebra.py` | Low-rank structure, PCA, what leading components measure | 0.1 | |
| 3 | `ch03_optimization.py` | Conditioning, momentum, Adam, implicit regularization | 0.1 | |
| 4 | `ch04_statistics.py` | Counts, pseudoreplication, multiple testing, Bayesian bookkeeping | 0.2 | |
| 5 | `ch05_information.py` | Entropy, KL asymmetry, compression, motif information, data processing | 0.1 | |
| 6 | `ch06_dynamic_programming.py` | Alignment and HMM algorithms checked against brute force | 0.1 | |
| 6 | `ch06_training_skeleton.py` | A tested research skeleton for sequence models | 1 | torch |
| 7 | `ch07_generalization.py` | Double descent, genetic architecture versus regularizer, selection leakage | 1 | |
| 8 | `ch08_latent_variables.py` | EM, identifiability, clusters in a continuum | 0.5 | |
| 9 | `ch09_autograd.py` | Reverse-mode autodiff from scratch; batched backprop; initialization | 0.2 | |
| 10 | `ch10_cnn_grammar.py` | Which grammar a CNN can express: spacing, receptive fields, RC weight tying | 3 | torch |
| 11 | `ch11_sequence_models.py` | RNN gradient decay; SSM = convolution; parallel scan; long-context cost | 0.2 | |
| 12 | `ch12_attention.py` | Attention, RoPE, online softmax, linear attention | 0.2 | torch |
| 13 | `ch13_representation.py` | The objective decides what a representation keeps | 3 | torch |
| 14 | `ch14_generative.py` | VAE $\beta$ sweep, flows, GAN discriminator, EBMs, typical set | 3 | torch |
| 15 | `ch15_diffusion.py` | DDPM, flow matching, Tweedie, classifier-free guidance | 4 | torch |
| 16 | `ch16_geometric.py` | Equivariance, 1-WL limits, E(n)-equivariant message passing | 4 | torch |
| 17 | `ch17_scaling.py` | Scaling-law fits, compute-optimal allocation, LoRA, KL-regularized alignment | 0.5 | |
| 18 | `ch18_interpretability.py` | Attributions with ground truth, CKA, sparse autoencoders, conformal coverage | 3 | torch |
| 19 | `ch19_cell_dynamics.py` | Bursting gene expression; bistable circuit | 0.5 | |
| 20 | `ch20_mutation_bias.py` | Mutation bias versus selection | 0.5 | |
| 21 | `ch21_popgen.py` | Wright–Fisher, $F_{ST}$, coalescent, LD decay, Jukes–Cantor | 1 | |
| 22 | `ch22_thermodynamic_model.py` | Thermodynamic model of TF binding and regulatory logic | 0.2 | |
| 23 | `ch23_global_epistasis.py` | Marginal stability and global epistasis | 1 | |
| 24 | `ch24_chemistry.py` | Molecular representation, binding free energy, evaluation on BACE and ESOL | 4 | rdkit, download |
| 25 | `ch25_measurement.py` | Measurement models for single-cell, spatial, perturbation data | 2 | |
| 26 | `ch26_statgen.py` | LDSC, LMM = ridge, fine-mapping, PRS portability, MR | 3 | |
| 27 | `ch27_alignment.py` | Karlin–Altschul statistics, read mapping, variant calling | 2 | |
| 28 | `ch28_representation.py` | DNA representations on a real genome; BPE; RC symmetry | 3 | download |
| 29 | `ch29_classical_models.py` | PWMs, an HMM gene finder, Potts/DCA | 6 | pyjaspar, download |
| 30 | `ch30_single_cell_methods.py`, `ch30_false_correlation.py` | PCA, batch correction, NB-VAE, denoising-induced false correlation | 12 + 3 | torch |
| 31 | `ch31_counterfactual.py` | A sequence-to-function model that predicts well and gets the edit wrong | 8 | torch |
| 32 | `ch32_dna_lm.py`, `ch32b_composition_check.py` | A small DNA LM against Markov baselines; composition confound | 25 + 0.2 | torch, download |
| 33 | `ch33_rna_structure.py` | RNA folding on 37 tRNAs; composition-matched controls | 3 | ViennaRNA, download |
| 34 | `ch34_plm_toy.py` | Cross-family pretraining: when it transfers | 12 | torch |
| 35 | `ch35_structure.py` | Structure reconstruction from contacts on a real protein | 6 | download |
| 36 | `ch36_design_surrogate.py`, `ch36b_local_design.py` | Designing against a surrogate; trust-region phase transition | 20 + 10 | torch |
| 37 | `ch37_molecular_ml.py` | BACE: forest, kNN, GCN; enrichment and uncertainty | 15 | rdkit, torch, download |
| 38 | `ch38_sc_fm.py`, `ch38b_contrastive.py` | A miniature single-cell foundation model; contrastive objective | 12 + 8 | torch |
| 39 | `ch39_perturbation.py` | Perturbation baselines, metrics, graph propagation | 1 | |
| 40 | `ch40_multimodal.py` | Multimodal alignment: pairs, private information, unpaired GW | 15 | torch |
| 41 | `ch41_functional_priors.py` | Annotation priors in fine-mapping and polygenic prediction | 15 | |
| 42 | `ch42_ancestral.py` | Ancestral reconstruction bias; effective sample size on trees | 0.5 | |
| 43 | `ch43_benchmarks.py` | Winner's curse, adaptive reuse, prevalence, resolution | 2 | |
| 44 | `ch44_causal.py` | Observational versus interventional identification | 1 | |
| 45 | `ch45_shift.py` | Covariate, label, concept shift, and shortcuts | 2 | |
| 46 | `ch46_design.py` | Experimental design: perturbation choice, cells vs perturbations, closed loop | 10 | |
| 47 | `ch47_scaling_data.py` | Learning curves, ceilings, diversity versus volume (BACE) | 5 | rdkit, download |
| 48 | `ch48_interp_circuit.py` | Interpretability methods graded on a known circuit | 1 | torch |
| 53 | `ch53_neural_models.py` | Neural noise ceilings; connectome-constrained networks | 8 | torch |
| 54 | `ch54_agent_verification.py` | Forking paths and judge drift in automated research | 0.3 | |
| 56 | `ch56_information_gain.py` | Choosing experiments by expected information gain | 0.2 | |

Chapters not listed (49–52, 55, 57–59 and the front matter) are conceptual or atlas chapters whose worked examples are procedures, not programs.

---

## C.4 When your numbers differ

1. **Library versions.** Differences in the last digits come from library or BLAS versions; larger differences in torch scripts can come from the number of CPU threads. The qualitative conclusions the chapters draw are written to be robust to these differences, and every chapter that draws a conclusion from a simulation says which comparisons are large relative to the noise.
2. **Scripts guarded after the output was produced.** Three scripts (`ch30_single_cell_methods.py`, `ch36_design_surrogate.py`, `ch38_sc_fm.py`) were wrapped in `if __name__ == "__main__":` after their outputs were pasted into the chapters, so that other scripts can import their functions; the guard does not change the numbers.
3. **Stochastic training.** A new seed changes the numbers; the chapters report means over seeds where the effect was small relative to the seed variance, and say when only one seed was run.
4. **Real data versions.** MoleculeNet files and GenBank records are stable but not guaranteed to be immutable.

---

## C.5 Extending the code

A script that tests an idea from a chapter should follow the template of Chapter 6's skeleton: (i) a *planted-signal* data generator with a known truth; (ii) a baseline (the simplest method that could work); (iii) the method; (iv) a metric with its noise ceiling; (v) a table of results over a controlled factor (sample size, noise, shift); (vi) a statement of what would count as success in advance. Chapter 59 describes how to turn such a script into a reproducible research artifact.
