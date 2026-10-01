# AI for Biology: From First Principles to the Research Frontier

A research-training textbook that takes a reader from beginner to independent researcher at the intersection of
deep learning, genomics, protein science, single-cell biology, drug discovery, and AI for science.

**Read it online:** <https://namknnguyen.github.io/ai-for-biology-book/>

The book is organized as ten parts: orientation, mathematical foundations, core machine learning, deep learning,
biology for modeling, computational biology, biological foundation models, research methodology, the frontier,
and independent research. The last third is written as a research training manual: open problems, systematic
idea generation, and reasoning when the answer is not known.

## Building locally

```bash
pip install -r requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs
```

## Running the book's code

Implementations quoted in the text live in `code/` and are included verbatim in the chapters.

```bash
pip install numpy scipy torch
python code/run_all.py
```

## Publishing

`.github/workflows/pages.yml` builds the site and deploys it with GitHub Actions. One-time repository setting:
**Settings → Pages → Build and deployment → Source: GitHub Actions.**

## A note on claims

Each substantive claim about the literature is graded inline (Established, Strong evidence, Plausible, Open
hypothesis, Speculative). Recent results were checked against primary sources and publisher pages in
October 2026; this field moves fast, so check dates and read the primary papers before citing.
