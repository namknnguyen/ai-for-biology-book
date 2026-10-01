# Progress log

Legend: [x] written + built + math-checked; [ ] pending.

## Infrastructure
- [x] mkdocs.yml (nav lists ALL planned chapters; non-existent pages produce warnings until written)
- [x] CSS (custom admonitions), mathjax.js (macros), extra.js ([[E]] badges)
- [x] .github/workflows/pages.yml (build + deploy)
- [x] tools/check_math.mjs (MathJax parse check; needs MJ_NODE_MODULES), tools/render_check.py (Playwright render check)
  - node_modules for tools live in the scratchpad: .../scratchpad/tools/node_modules (npm i mathjax-full@3 jsdom mermaid@11 mathjax@3)
  - Run: `MJ_NODE_MODULES=<scratchpad>/tools/node_modules node tools/check_math.mjs site`

## Front matter
- [x] index, how-to-use, toc, dependency-graph, learning-trajectory, field-map, research-skills, notation

## Chapters
- [x] 1 Expert chain (code/ch01_noise_ceiling.py)
(update as chapters are completed)

## Verified facts (Oct 2026 web checks) to reuse
- Evo 2: Nature 652, 1349-1361, published 4 Mar 2026 (preprint Feb 2025); 40B params, 1 Mb context, >9T nucleotides, 100k+ species; fully open (data, code, weights). First author Garyk Brixi.
- AlphaGenome: Nature, 28 Jan 2026; up to 1 Mb input, base-pair resolution; modalities: expression, splicing, accessibility, TF binding, contacts. (Avsec et al.)
- Evo phage design: King et al., Science 6 Aug 2026 (bioRxiv Sept 2025); ~300 synthesized genomes, 16 viable; phiX174 template; Evo 1 and Evo 2 used.
- Ahlmann-Eltze, Huber, Anders: Nature Methods 22:1657-1661, 4 Aug 2025: 5 foundation models + 2 DL models did not beat simple (additive/mean) baselines for double perturbation; follow-up bioRxiv Oct 2025 argues well-calibrated metrics show DL beats uninformative baselines.
- Kedzierska et al., Genome Biology 2025: zero-shot Geneformer/scGPT lose to HVG, Harmony, scVI.
- Virtual Cell Challenge 2025 (Arc): >5,000 registrants in 114 countries, >1,200 teams submitted, >300 final; two $100k grand prizes; Altos Labs won Generalist prize (flow-matching generative model); winners combined deep learning with classical statistical features; models not consistently beating naive baselines on all metrics.
- State (Arc, bioRxiv June 2025): State Embedding + State Transition; trained on 167M observational cells and >100M perturbed cells across 70 contexts.
- Isomorphic IsoDDE technical report 10 Feb 2026: unified system (AF3-like cofolding, affinity, pocket finding, antibody-antigen); claims ~2x AF3 on hard (dissimilar-to-training) protein-ligand structures; proprietary.
- ESM3: Science Jan 2025, esmGFP 58% identity to closest fluorescent protein.
- Boltz-2 (2025): open-source MIT, affinity module approaching FEP, ~1000x faster; Chai-2 (bioRxiv Jul 2025): ~16% zero-shot antibody hit rate, 52 antigens, ~20 designs per target; RFdiffusion3 open-sourced 3 Dec 2025 (all-atom).
- Tahoe-100M: >100M cells, 50 cancer cell lines, >1,100 small molecules (379 drugs?), published in Cell 2026. X-Atlas/Orion (Xaira): 8M cells, genome-wide Perturb-seq in HCT116 and HEK293T (June 2025).
- Robin (FutureHouse): ripasudil for dry AMD, 2.5 months; Kosmos (Edison Scientific, Nov 2025); Google Co-Scientist.
- DNA LM benchmarks: Tang & Koo (rep power, 2025), DART-Eval (2025), GENEB (2026) — little advantage over one-hot supervised on human regulatory tasks.
