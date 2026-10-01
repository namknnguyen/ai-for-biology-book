# Book plan and conventions (working notes; not published on the site)

Title: **AI for Biology: From First Principles to the Research Frontier**
Site: mkdocs-material, docs/ -> GitHub Pages via .github/workflows/pages.yml
Branch: claude/compassionate-archimedes-d1r2q8

## Core devices (use consistently in every chapter)

* **Expert Chain** (12 links, L1-L12): Problem -> Existing approaches -> Assumptions -> Why they might work ->
  Failure modes -> Bottlenecks -> Open questions -> Hypotheses -> Candidate solutions -> Experiments ->
  Interpretation -> New directions. Defined in Ch 1.
* **Four Gaps**: Measurement gap, Objective gap, Inference (causal) gap, Generalization gap. Defined in Ch 1.
* **Ten Attacks** (idea generation, A1-A10): Assumption, Representation, Objective, Data, Evaluation, Scale,
  Cross-domain transfer, Problem reformulation, Biological constraint, Biological discovery. Defined in Ch 1
  (preview), taught fully in Ch 55.
* **Modeling ladder for biology** (7 questions): What is it? What information does it contain? How is it
  generated? How is it measured? How is it represented computationally? What variation exists? What can ML
  learn / what can ML not observe?
* **Evidence grades** (inline badges): [[E]] Established, [[S]] Strong evidence, [[P]] Plausible,
  [[H]] Open hypothesis, [[X]] Speculative.

## Admonition vocabulary

* `!!! example "Worked Research Example N.M: title"`  (worked research reasoning; no textbook exercises)
* `!!! notebook "Researcher's Notebook: title"`
* `!!! rhyme "Structural rhyme: A <-> B"` (same mathematics in different fields)
* `!!! bio "Biology for modeling: ..."`
* `!!! math "Derivation: ..."` (side derivations; main derivations stay inline)
* `!!! lens "Research lens: ..."` (assumptions, information used/ignored, failure modes)
* `!!! paper "Paper dissection: ..."` (fixed template: Problem / Insight / Architecture / Objective / Data /
  Training / Evaluation / Results / Why it worked / Assumptions / Limitations / What followed / Unresolved)
* `!!! openproblem "Open problem: ..."`
* `!!! takeaways "Key takeaways"`

## Notation (see docs/front/notation.md for the reader-facing version)

* Sequence x_{1:L} over alphabet A; batch B; length L; model width d; heads H; head width d_h; vocabulary V.
* Latent biological state s; genotype g; environment e; phenotype y; measurement operator M_k; data x^{(k)}.
* Parameters theta; encoder/representation phi_theta(x)=h; loss L; data distribution p_data; model p_theta.
* Perturbation c (condition); control/treated expression; cell type/state t.
* Macros available in MathJax: \R \E \Var \Cov \Ent \MI \KL{p}{q} \argmax \argmin \softmax \diag \tr \rank \indep \do \Normal

## Code

Tested code lives in code/ and is included in chapters with `--8<-- "code/file.py"` inside a fenced block.
Run all with `python code/run_all.py` (needs numpy, scipy, torch CPU).

## Chapter status

See notes/PROGRESS.md.
