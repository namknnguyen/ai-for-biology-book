# AI for Biology: From First Principles to the Research Frontier

A research-training textbook that takes a reader from beginner to independent researcher at the intersection of
deep learning, genomics, protein science, single-cell biology, drug discovery, and AI for science.

**Read it online:** <https://namknnguyen.github.io/ai-for-biology-book/>  
**PDF edition (462 pages):** [`docs/assets/ai-for-biology-book.pdf`](docs/assets/ai-for-biology-book.pdf)

The book is organized as ten parts, preceded by a **Part 0** that teaches the mathematical and computational background from high-school algebra upward (ten chapters, M1–M10, for a first-year student): orientation, mathematical foundations, core machine learning, deep learning,
biology for modeling, computational biology, biological foundation models, research methodology, the frontier,
and independent research. The last third is written as a research training manual: open problems, systematic
idea generation, and reasoning when the answer is not known.

## Building locally

```bash
pip install -r requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs
```

## Building the PDF

```bash
pip install -r requirements.txt -r tools/requirements-pdf.txt
npm install mathjax@3 mermaid@11            # MathJax and Mermaid are served locally while rendering
python tools/build_pdf.py node_modules docs/assets/ai-for-biology-book.pdf
```

`tools/build_pdf.py` builds a single-page print version (`mkdocs-pdf.yml`), typesets every equation and
diagram in headless Chromium, adds a cover, page numbers, bookmarks, and a contents list with page numbers.
`tools/fix_display_math.py` keeps `$$` display-math fences separated by blank lines, which the Markdown
parser needs in order to recognize them as equations.

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
