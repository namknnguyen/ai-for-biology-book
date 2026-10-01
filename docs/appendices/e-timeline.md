# Appendix E. A Timeline of the Field

A selective chronology of the ideas, data, and systems on which AI for biology rests, with the chapters that use them. Dates are for first public appearance (preprint or paper); where a journal version appeared later, both are given. Entries from 2025–2026 are the ones this book verified against primary sources as of October 2026; later developments are not included.

---

## E.1 Foundations (to 1999)

| Year | Event | Why it matters | Chapters |
|---|---|---|---|
| 1953 | Structure of DNA (Watson and Crick; Franklin and Wilkins) | The physical basis of sequence as information | 19, 20 |
| 1958–61 | The central dogma; the genetic code (Crick; Nirenberg and colleagues) | Sequence → protein is a code with structure (codon frame) | 19, 28, 33 |
| 1960s | Anfinsen: sequence determines structure | The premise of structure prediction | 35 |
| 1970 | Needleman–Wunsch global alignment | Dynamic programming for sequences | 6, 27 |
| 1977 | DNA sequencing (Sanger; Maxam and Gilbert) | Sequence data become abundant | 27 |
| 1981 | Smith–Waterman local alignment; Felsenstein's likelihood for trees | Alignment statistics; phylogenetic inference | 27, 42 |
| 1982 | Hopfield networks | Attractors as memory; the energy view of networks | 19, 53 |
| 1986 | Backpropagation popularized (Rumelhart, Hinton, Williams) | Training multilayer networks | 9 |
| 1990 | BLAST; the Human Genome Project launched | Fast database search; the reference genome project | 27 |
| 1994 | Hidden Markov models for protein families (Krogh and colleagues) | Probabilistic models of sequence families | 6, 29 |
| 1997 | LSTM (Hochreiter and Schmidhuber) | Long-range dependence in sequences | 11 |

## E.2 Genomics and early machine learning (2000–2016)

| Year | Event | Why it matters | Chapters |
|---|---|---|---|
| 2001–03 | Draft and finished human genome | A reference for everything that follows | 20, 26 |
| 2005–08 | Next-generation sequencing; the first GWAS era | Population-scale data; association without mechanism | 26 |
| 2011 | Direct-coupling analysis predicts protein contacts (Marks; Morcos) | Coevolution gives structure | 29, 35 |
| 2012 | ENCODE; AlexNet; CRISPR–Cas9 as an editing tool (Jinek et al.) | Regulatory maps; deep learning on images; programmable perturbations | 10, 31, 39 |
| 2015 | DeepBind, DeepSEA; Drop-seq and inDrops | Deep learning on regulatory sequence; single-cell RNA at scale | 10, 25, 31 |
| 2016 | Perturb-seq (Dixit et al.; Adamson et al.); Basset | Perturbation with single-cell readout; more deep models of regulatory sequence | 31, 39 |

## E.3 The deep-learning decade (2017–2022)

| Year | Event | Why it matters | Chapters |
|---|---|---|---|
| 2017 | The Transformer (Vaswani et al.); 10x Genomics droplet platform | The architecture of the next decade; single-cell at 10⁴–10⁵ cells | 12, 25 |
| 2018 | BERT; RNA velocity; AlphaFold at CASP13 | Masked pretraining; inferring dynamics from snapshots; deep learning enters structure prediction | 13, 35, 51 |
| 2019 | SpliceAI; the ESM protein language-model preprint (published in *PNAS* in 2021) | A deep splicing code; self-supervised protein models | 33, 34 |
| 2020 | Scaling laws (Kaplan et al.); AlphaFold2 at CASP14; GPT-3 | Predictable improvement with scale; near-experimental structure accuracy | 17, 35 |
| 2021 | AlphaFold2 and the AlphaFold Database; RoseTTAFold; Enformer; ESM-1v | Structure prediction broadly available; long-range sequence-to-function | 31, 34, 35 |
| 2022 | Chinchilla; ESMFold (preprint); ProteinMPNN; first genome-scale Perturb-seq (Replogle et al.); Stable-Diffusion-style models | Compute-optimal training; design pipelines; perturbation data at scale | 17, 35, 36, 39 |

## E.4 The foundation-model era (2023–2026)

| Year | Event | Why it matters | Chapters |
|---|---|---|---|
| 2023 | Geneformer (*Nature*); scGPT and scFoundation preprints; AlphaMissense; RFdiffusion; HyenaDNA; Nucleotide Transformer | Single-cell foundation models; proteome-wide missense scores; backbone diffusion; long-context DNA models | 32, 36, 38, 41 |
| 2024 | AlphaFold 3 (May); ESM3 preprint; Evo (preprint Feb; *Science* Nov); AlphaProteo (Sep); FlyWire connectome (Oct); Nobel Prizes in Chemistry (Baker; Hassabis and Jumper) and Physics (Hopfield and Hinton) | Complexes with diffusion; multimodal protein models; DNA at scale; whole-brain wiring; recognition of the field | 32, 34–36, 53 |
| 2025 | Evo 2 preprint (Feb); Co-Scientist preprint (Feb); MICrONS and a foundation model of mouse visual cortex (Apr); ESM3 in *Science* (Jan); State (Jun); Boltz-2 (Jun); X-Atlas/Orion (Jun); Chai-2 (Jul); Virtual Lab in *Nature* (Jul); Centaur in *Nature* (2025); BindCraft in *Nature* (Aug); Ahlmann-Eltze et al. on perturbation baselines (Aug); Virtual Cell Challenge; C2S-Scale (Oct); Kosmos (Nov); RFdiffusion3 open-sourced (Dec); CASP funding lapses and is bridged | Frontier systems and the first systematic checks against simple baselines; AI-scientist systems; community benchmarks | 32, 34–36, 38, 39, 53, 54 |
| 2026 | AlphaGenome in *Nature* (Jan); IsoDDE technical report (Feb); Evo 2 in *Nature* (Mar); Co-Scientist and Robin in *Nature* (May); rentosertib Phase III announced (Jul); Evo-designed phage genomes in *Science* (Aug) | Regulatory sequence-to-function at 1 Mb; proprietary co-folding for drug discovery; published AI-scientist loops with wet-lab validation; clinical-stage AI-nominated drug; genome-scale generative design | 32, 35, 37, 54 |

---

## E.5 What the timeline suggests

1. **Representation changes preceded capability jumps**: alignment and HMMs (1970s–90s), convolutional filters (2015), attention and masking (2017–18), diffusion and equivariance (2021–23).
2. **Data came before models**: genomes, GWAS, PDB, UniRef, single-cell atlases, and perturbation screens each preceded the models that exploited them by years.
3. **Benchmarks drive progress and are fragile**: CASP made structure prediction measurable; its funding lapse in 2025 is a reminder that benchmarks are infrastructure.
4. **Each wave was followed by a correction**: the 2023–25 zero-shot evaluations of single-cell and DNA models against simple baselines are the most recent.
5. **Open problems have shifted** from *prediction* of molecular objects to *intervention, context, and time* (Chapters 50–52).
