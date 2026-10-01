# AI for Biology
## From First Principles to the Research Frontier

> *A research-training textbook. Its aim is not to make you knowledgeable about AI and biology. Its aim is to make you able to look at an unfamiliar problem in AI for biology and reason, independently, about what is known, what is not, why the existing methods work, where they break, and what experiment would tell you something new.*

---

## What this book is

Most textbooks teach methods. This one teaches **research reasoning** and uses methods, mathematics, biology, and the literature as the material that reasoning operates on.

You will start from linear algebra and end by designing, attacking, and defending your own research programs. Along the way you will:

- **Derive** the mathematics that modern biological AI rests on (backpropagation, attention, variational inference, diffusion, contrastive learning, statistical genetics, coalescent theory, causal identification), not merely state it.
- **Learn biology for modeling**: for every biological object you will ask what information it contains, how it is generated, how it is measured, how it is represented computationally, what varies, what a model can learn from it, and what a model *cannot* observe.
- **Study the field historically**: sequence-to-function models, genomic language models, protein language models, structure prediction and design, drug discovery, single-cell foundation models, perturbation prediction and the "virtual cell", multimodal and multi-scale models, agents for scientific discovery. Papers are treated as evidence in an argument, not as a list.
- **Practice the expert's chain of thought** on worked research examples that begin with basic scientific reasoning and end with open problems where no one knows the answer.

## The one question the book is built around

> Given a new paper or an unsolved problem, can you say **what we know, what we don't know, what assumptions are being made, why existing approaches work, where they fail, what information is missing, what alternative explanations exist, what hypothesis to test, what experiment would distinguish the hypotheses, what result would change your beliefs, and what new method could address the underlying problem?**

Every chapter is organized so that it contributes to a single skill: moving along the **Expert Chain** (introduced in [Chapter 1](chapters/ch01-expert-chain.md)) from a problem to new research directions.

## How the book is organized

| Part | Theme | Chapters | What it gives you |
|---|---|---|---|
| **I** | Orientation | 1 | The Expert Chain, the Four Gaps, and the measurement view of biology: the spine of everything else |
| **II** | Foundations | 2–6 | Linear algebra, optimization, probability, information theory, and implementation, taught efficiently and only as far as research needs |
| **III** | Core machine learning | 7–8 | Generalization, inductive bias, latent-variable inference |
| **IV** | Deep learning | 9–18 | Backpropagation in depth, CNNs, sequence models, attention, self-supervision, generative models, diffusion, geometric learning, scaling, interpretability |
| **V** | Biology for modeling | 19–26 | The cell, genomes, evolution, gene regulation, proteins, small molecules, single-cell and perturbation biology, statistical genetics |
| **VI** | Computational biology | 27–30 | Alignment, representations, classical models, single-cell methods: what foundation models inherit and what they replace |
| **VII** | Biological foundation models | 31–42 | Sequence-to-function, genomic and RNA models, protein language models, structure, design, drug discovery, single-cell models, the virtual cell, multimodal models, genotype→phenotype, evolutionary modeling |
| **VIII** | Research methodology | 43–48 | Benchmarks and leakage, causal inference, distribution shift, experimental design, scaling economics, mechanistic understanding |
| **IX** | The frontier | 49–54 | Competing paradigms and the open-problem atlas, each problem analyzed with the same diagnostic questions |
| **X** | Independent research | 55–59 | Ten systematic strategies for generating ideas, evaluating them, reading papers, reasoning without answers, and turning ideas into research programs |

Start with the [**complete table of contents**](front/toc.md), the [**dependency graph**](front/dependency-graph.md) that shows how concepts build on one another, the [**learning trajectory**](front/learning-trajectory.md), the [**map of the fields**](front/field-map.md), and the list of [**research skills**](front/research-skills.md) the book is designed to develop. Then read [How to use this book](front/how-to-use.md) and begin [Chapter 1](chapters/ch01-expert-chain.md).

## A note on honesty and dates

This field changes monthly. Claims about recent work were checked against primary sources and publisher pages in **October 2026**, and every substantial claim carries an evidence grade:

[[E]] established result &nbsp; [[S]] strong empirical evidence &nbsp; [[P]] plausible interpretation &nbsp; [[H]] open hypothesis &nbsp; [[X]] speculative research direction

Where the book cannot verify something, it says so. Treat the book's account of the frontier as a dated snapshot and a method for re-deriving the frontier yourself, not as the final word.
