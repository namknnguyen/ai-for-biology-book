# Research Skills This Book Is Designed to Develop

The book trains a specific, testable set of skills. Each is attached to places in the book where it is *practiced*, not merely described.

## The twelve-question capability

By the end you should be able to face any new paper or unsolved problem and answer, in order:

1. **What do we know?** (established vs. plausible vs. speculative)
2. **What don't we know?**
3. **What assumptions are being made?** (explicit and hidden)
4. **Why do existing approaches work?**
5. **Where do they fail?**
6. **What information is missing?**
7. **What alternative explanations exist?**
8. **What hypothesis should we test?**
9. **What experiment would distinguish the hypotheses?**
10. **What result would change our beliefs?**
11. **What new method could address the underlying problem?**
12. **Is the idea novel, meaningful, and feasible?**

These twelve questions are the operational form of the **Expert Chain** (Chapter 1).

## The skill inventory

### A. Understanding

| # | Skill | How it is trained | Primary chapters |
|---|---|---|---|
| A1 | **Read a paper to the objective and the information flow.** Reconstruct the actual loss, what the model sees, and what it cannot see | Paper dissections; reading protocol | 9–18, 31–42, 57 |
| A2 | **Derive rather than memorize.** Re-derive the key equation of a method from its assumptions | Step-by-step derivations; "derive before you read" practice | 3, 9, 12–15, 21, 26, 44 |
| A3 | **Know the biology as a measurement process.** For any dataset: how generated, how measured, what is lost | "Biology for modeling" boxes | 19–26 |
| A4 | **Reproduce and critically evaluate.** Rebuild a result and test whether it is robust | Code companion; reproduction protocol | 6, 43, 57 |

### B. Critique

| # | Skill | How it is trained | Primary chapters |
|---|---|---|---|
| B1 | **Find leakage and confounding.** Identify the route by which a benchmark can be solved without the intended capability | Worked examples on homology leakage, batch effects, shortcut features | 1, 7, 43, 45 |
| B2 | **Separate fundamental limits from implementation problems.** Ask whether more data/compute/engineering could fix a failure, or whether information is absent | Researcher's Notebooks on the *information-theoretic* and *identification* arguments | 5, 13, 44, 47 |
| B3 | **Distinguish correlation from mechanism.** State what an observational model can and cannot say about interventions | Causal inference chapter; perturbation case studies | 26, 39, 44 |
| B4 | **Grade evidence.** Apply established/strong/plausible/hypothesis/speculative honestly | Evidence badges throughout | all |

### C. Generation

| # | Skill | How it is trained | Primary chapters |
|---|---|---|---|
| C1 | **Decompose a vague problem** into well-posed sub-problems with identifiable data requirements | Researcher's Notebooks | 1, 49–54 |
| C2 | **Generate hypotheses** that make different predictions | Worked examples, increasing sophistication | all, esp. 55, 58 |
| C3 | **Propose models and objectives** motivated by the diagnosed failure rather than by fashion | Ten Attacks; objective/representation attacks | 13–17, 55 |
| C4 | **Reformulate problems** so a different tool class applies | Reformulation attack; structural rhymes | 8, 15, 44, 55 |
| C5 | **Transfer ideas across fields** with the mathematical structure intact | Structural rhymes; cross-domain attack | all, esp. 55 |

### D. Testing

| # | Skill | How it is trained | Primary chapters |
|---|---|---|---|
| D1 | **Design discriminating experiments** that separate competing hypotheses | Worked examples; experiment-design chapter | 43, 46, 58 |
| D2 | **Design benchmarks** with baselines, ceilings, leakage controls, power | Benchmark chapter; critical re-analyses | 43 |
| D3 | **Predict outcomes before running** and interpret surprises | Pre-registration templates | 46, 56, Appendix G |
| D4 | **Plan wet-lab validation** realistically (noise, replicates, cost, orthogonal assays) | Experiment-design chapter; assay boxes | 22–25, 46 |

### E. Judgment

| # | Skill | How it is trained | Primary chapters |
|---|---|---|---|
| E1 | **Distinguish novelty from superficial modification** | Novelty tests with worked examples | 56 |
| E2 | **Judge scientific meaning and feasibility** | Value × feasibility × information-gain analysis | 56 |
| E3 | **Identify what evidence would change your mind**, and say it first | Kill criteria; pre-mortems | 56, 58 |
| E4 | **Enter a new area rapidly** | Literature-mapping protocol | 57 |
| E5 | **Turn ideas into publishable research responsibly** (ethics, biosecurity, open science) | Capstone programs | 59 |

## Self-assessment grid

After each Part, rate yourself 0–3 on the relevant skills (0 = cannot, 1 = with the book open, 2 = unaided on familiar problems, 3 = unaided on unfamiliar problems). A reader who reaches the end of Part X with mostly 2s and 3s on B, C, D and E has achieved the book's goal; skills A are prerequisites rather than ends in themselves.

## What the book does *not* train

Wet-lab technique, large-scale distributed systems engineering, and clinical translation are discussed so you can reason about them and collaborate effectively, but they are not taught as skills. Research at this intersection is collaborative; knowing what to ask of your collaborators is itself a skill the book tries to build (Chapters 46 and 59).
