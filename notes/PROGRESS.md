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
All 59 chapters, Part 0 (guide + ten chapters M1-M10, added at the user's request for first-year readers), the front matter, and Appendices A-G exist; `mkdocs build --strict` passes and the MathJax parse check reports 0 errors.
- Ch 1-31, 33, 43-45, 55-57, 59: written in earlier sessions (see git log).
- Ch 32 genomic LMs, 34 protein LMs, 35 structure prediction, 36 design, 37 molecular ML, 38 single-cell FMs, 39 perturbation/virtual cell,
  40 multimodal, 41 genotype-phenotype, 42 evolutionary modeling, 47 scaling and data economics, 48 mechanistic interpretability,
  49 competing paradigms, 50-52 open-problem atlases, 53 brains, 54 AI scientists, 58 open-ended case studies: written in this session
  (each with a tested script in code/ and its output pasted into the chapter).
- Ch 46 experimental design: text complete; the Section 4 table is filled from the final rerun of code/ch46_design.py.
- Appendices C (code companion), D (data and tools), E (timeline), F (reading lists) written.
- code/run_all.py and code/requirements-code.txt added.

## Chapter template actually used (keep consistent)
`!!! abstract "Chapter at a glance"` (Motivation / Prerequisites / You will be able to), numbered sections 'N.M', derivations inline,
`!!! math` for side derivations, `!!! lens`, `!!! rhyme`, `!!! bio` (biology chapters), `!!! paper` dissections, 2 worked examples
(`!!! example "Worked Research Example N.K: ..."` with Situation / Question / Reasoning steps / Expert analysis), `!!! notebook`,
Connections, `!!! takeaways`, Further reading. Code shown via `--8<-- "code/xxx.py"` inside a fenced block, followed by the
actual script output pasted in a ```text block. Evidence badges [[E]] [[S]] [[P]] [[H]] [[X]].
Cross-refs: "Chapter N" plain text (no links needed).

## Verification commands
mkdocs build 2>&1 | grep -iE "ERROR|snippet" ; MJ_NODE_MODULES=<scratchpad>/tools/node_modules node tools/check_math.mjs site
Python deps installed in sandbox: numpy scipy torch scikit-learn mkdocs-material playwright.

## Git/push status
Local commits only; `git push` and MCP writes return 403 (GitHub App lacks write access to namknnguyen/ai-for-biology-book). Repeatedly retried at milestones; still 403.
Retry push at milestones. If still blocked at the end: tell user to install/authorize Claude GitHub App with write access, then push branch
claude/compassionate-archimedes-d1r2q8 and set Settings > Pages > Source: GitHub Actions.

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

- Evo (Nguyen et al., Science 2024, 386:eado9336): 7B params, 131,072-token context, StripedHyena (attention + data-controlled convolutions), OpenGenome ~300B prokaryotic nucleotides, byte-level single-nt tokenizer.
- Evo 2 architecture: StripedHyena 2 = convolutional multi-hybrid with short explicit (SE), medium regularized (MR), long implicit (LI) Hyena operators + attention; pretrain at 8,192 ctx then midtraining extending to 1M; 40B params; >9T nucleotides.
- HyenaDNA (Nguyen et al. 2023): up to 1M-token single-nucleotide context; up to 160x faster than Transformer w/ FlashAttention; 1.6M params vs 2.5B on NT benchmark comparisons.

## Part 0 (mathematical background for a first-year student)
- docs/chapters/m00-part0-guide.md (placement check, paths, dependency map) and m01-m10 (language/proof/counting, functions/logs, calculus 1,
  multivariable calculus/optimization, linear algebra I and II, probability, statistics, discrete structures/algorithms, Python/numerics).
- Each chapter has a tested script code/m01_*.py ... m10_*.py with its real output pasted in; all are in code/run_all.py and Appendix C.
- Integrated into mkdocs nav, index, how-to-use (new "From scratch" path), toc, dependency graph, learning trajectory, notation, README,
  Appendix A pointer, and "If this chapter moves too fast" notes in Chapters 1-9, 12, 16, 21, 26, 27, 42.
- tools/fix_display_math.py: blank lines around $$ fences (a bug that rendered 167 display equations as raw TeX before it was fixed).
- tools/build_pdf.py + mkdocs-pdf.yml: PDF build (Chromium, local MathJax/Mermaid; use --untagged to keep the file under 30 MB).
