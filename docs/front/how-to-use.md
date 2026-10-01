# How to Use This Book

## What kind of reader this assumes

You can program a little (Python), remember some high-school calculus and probability, and have heard of DNA and neural networks. Nothing else is assumed. The foundations in Part II are intentionally compressed: they are meant to be *sufficient for research*, not a substitute for a full course. If you already know a topic, the chapter's opening "what you must be able to do" box tells you what to check and what to skip.

## Allocation of depth

The book deliberately does **not** spend its pages evenly.

| Topic | Depth | Why |
|---|---|---|
| Basic calculus, basic cell biology | Concise | You need the tools, not the whole subject |
| Backpropagation, attention, variational inference, diffusion | Deep, with derivations | You cannot attack or modify a method you cannot derive |
| Biological representations, measurement technologies, genomic modeling | Deep | The modeling problem *is* the measurement problem |
| Benchmarks, leakage, confounding, causal reasoning | Deep | This is where most published claims quietly fail |
| Frontier, open problems, idea generation | Largest share | The goal is research ability |

## The recurring devices

Every chapter uses a small set of devices. Learn to recognize them.

!!! example "Worked Research Example"
    These replace the usual exercise sets. Each one states a research situation, then **walks through the reasoning** (what does the objective teach? what information is available? what is missing? which alternative explanations exist? which experiments distinguish them?), and only then gives the expert analysis. Early examples teach basic scientific reasoning; late ones resemble real research problems; the last ones have no known answer. *Try to write your own answer before reading the walkthrough.* That is the exercise.

!!! notebook "Researcher's Notebook"
    A demonstration of a *research move*: decomposing a problem, finding a hidden assumption, separating a fundamental limitation from an implementation problem, designing a discriminating experiment, deciding whether an idea is novel or merely a modification.

!!! rhyme "Structural rhyme"
    A call-out that two apparently different problems share the same mathematical structure (for example, Potts-model coevolution and attention; population-genetic drift and stochastic optimization; selection and likelihood).

!!! bio "Biology for modeling"
    Biology is taught through seven questions: *What is it? What information does it contain? How is it generated? How is it measured? How is it represented computationally? What variation exists? What can ML learn, and what can ML not observe?*

!!! lens "Research lens"
    The assumptions, information used, information ignored, and failure modes of a method.

!!! paper "Paper dissection"
    A fixed-template analysis of an important paper: problem, insight, architecture, objective, data, training, evaluation, results, why it worked, assumptions, limitations, what followed, what remains unresolved.

!!! openproblem "Open problem"
    A problem the field has not solved, analyzed with the same diagnostic questions each time: what makes it difficult, what has prevented progress, which assumptions are responsible, what information is missing, what technology could change it, what reformulation might help, and what experiment would reveal the bottleneck.

!!! takeaways "Key takeaways"
    A short list at the end of a chapter. If you cannot reconstruct the argument behind a takeaway, re-read.

## Evidence grades

Claims about science carry different weights. The book marks them:

| Badge | Meaning |
|---|---|
| [[E]] | **Established result.** Proven, replicated, or so thoroughly tested that disagreement would be surprising. |
| [[S]] | **Strong empirical evidence.** Several independent sources agree, but alternative explanations are not fully excluded. |
| [[P]] | **Plausible interpretation.** Consistent with the data; not yet discriminated from rivals. |
| [[H]] | **Open hypothesis.** A specific claim that an experiment could test. |
| [[X]] | **Speculative direction.** Motivated but unproven; may be wrong. |

When you read the literature, practice assigning these grades yourself. Authors rarely do.

## Three reading paths

=== "Full path (recommended)"
    Read in order. Parts II–III can be skimmed if you have a strong ML background, but *do read* Chapters 3 (optimization), 4 (statistics, especially multiple testing and Bayesian reasoning), and 5 (information theory), because later chapters reuse their language constantly.

=== "ML-first (strong ML, new to biology)"
    Chapter 1 → skim Part II → skim Parts III–IV (read the "Biology connection" sections and Chapters 10, 12, 13, 14–15, 18 closely) → **Part V in full** (Chapters 19–26) → Part VI → Part VII → Parts VIII–X.

=== "Biology-first (strong biology, new to ML)"
    Chapter 1 → Part II (all) → Part III → Part IV (all) → skim Part V (read the "What can ML not observe?" sections and Chapters 25–26) → Part VI → Part VII → Parts VIII–X.

=== "Fast track to the frontier"
    Chapter 1 → Chapter 18 (interpretability and evaluation) → Chapter 43 (benchmarks) → Chapter 44 (causality) → the Part VII chapter for your area → Part IX → Chapters 55–58. Return to earlier chapters whenever the notation or an argument is unfamiliar. The [dependency graph](dependency-graph.md) tells you what to revisit.

## How to study a chapter

1. **Read the motivation and the "what you must be able to do" box.** Decide what you already know.
2. **Derive before you read the derivation.** When the text says "derive", close the page and try. You learn the structure of the argument from the places you get stuck.
3. **Run the code.** Code in the book lives in the repository's `code/` directory and is included verbatim; it is tested. Change one thing at a time and predict the outcome before running.
4. **Attempt the worked research examples before reading the analysis.** Write the list of hypotheses and discriminating experiments first.
5. **Keep a research notebook of your own.** After every chapter, write three things: one assumption you had not noticed, one claim whose evidence grade you would downgrade, and one question you could not answer. The templates in [Appendix G](../appendices/g-templates.md) help.

## A note on the organizations and groups named in this book

The book uses research programs from places such as Google DeepMind, Isomorphic Labs, the Arc Institute and Stanford groups behind the Evo models, the Chan Zuckerberg Initiative and its cell-atlas and virtual-cell efforts, EvolutionaryScale, the Baker lab and the Institute for Protein Design, and many academic laboratories as *examples of kinds of problems*, because their published work illustrates the paradigms well. Naming a group is never an endorsement of a claim. The user-supplied phrase "CellType-related research" was interpreted as the broader line of work on cell-type and cell-state modeling (cell atlases, cell-state embeddings, and virtual-cell models); if you meant a specific group, treat that part of the book as the general landscape rather than a description of that group.

## Conventions

- Vectors are columns, bold lowercase; matrices are bold uppercase. Details are in [Notation and Conventions](notation.md).
- *Italic* introduces a term at its definition. **Bold** marks a principle you should be able to restate.
- Citations are given as (Author, Year) with venue in the chapter's reading list. Dates are given because in this field *when* a result appeared often matters for interpreting it.
- Code is Python; deep-learning examples use PyTorch. Tensor shapes are written in comments.
