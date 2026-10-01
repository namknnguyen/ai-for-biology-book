# Chapter M10. Python and Numerical Computing for the Mathematics

!!! abstract "Chapter at a glance"
    **Motivation.** The mathematics of Chapters M1–M9 becomes useful only when you can run it. Computing is also where mathematics and reality part ways: computers store numbers with finite precision, loops are slower than array operations by orders of magnitude, random numbers are not random, and a program can run without errors and still be wrong. This chapter gives a compact, working introduction to the Python tools used throughout the book (plain Python, NumPy, a first look at PyTorch) and, more importantly, to the *numerical habits* that separate code you can trust from code that merely runs.

    **Prerequisites.** Chapters M1–M9 for the mathematics; no prior programming is *required*, but a first programming course or an online Python tutorial will make this chapter faster. If you have never written a loop, spend a few days on one of the beginner resources in the further reading first.

    **You will be able to:** (1) write and run Python scripts, and set up a reproducible environment; (2) use lists, dictionaries, sets, functions and comprehensions to process biological sequences; (3) use NumPy arrays: shapes, indexing, reductions with `axis`, and broadcasting; (4) recognize silent shape bugs; (5) explain why vectorized code is fast and measure it; (6) describe floating-point numbers, machine epsilon, rounding, overflow, underflow and catastrophic cancellation, and fix them; (7) use random seeds correctly and report variability across seeds; (8) compute a gradient with automatic differentiation; (9) adopt the debugging and testing habits used in the rest of the book.

---

## M10.1 Setting up, and the discipline of reproducibility

**Python** is the language of this book's code. You need Python 3.10 or later and a few packages: `numpy` and `scipy` for numerical computing, `matplotlib` for plots, `pandas` for tables, and `torch` (PyTorch) for deep learning. The recommended setup is an isolated **virtual environment** so that the book's packages do not conflict with others on your machine:

```bash
python -m venv .venv          # create an environment in the folder .venv
source .venv/bin/activate     # activate it (on Windows: .venv\Scripts\activate)
pip install numpy scipy matplotlib pandas torch
python code/m01_language.py   # run a script from this book
```

Two ways of working are common: **scripts** (a `.py` file run from the terminal) and **notebooks** (Jupyter, mixing code and output). Notebooks are excellent for exploration and plotting; scripts are better for anything you want to rerun, test or share, because they run from top to bottom in a fresh process. All code in this book is in scripts (the folder `code/`), each runnable on its own, and the output you see in the chapters is the output of running the script with the stated seed. `python code/run_all.py --list` shows how to run them all.

**Reproducibility** means someone else (or you in six months) can rerun your analysis and get the same numbers. The minimum:

1. **Record versions.** `pip freeze > requirements.txt` lists exact package versions. Numerical libraries change behaviour between versions.
2. **Fix random seeds** (§M10.5) and report them.
3. **Keep the code in version control** (git), with a message for each change. The history is a lab notebook that cannot be forgotten.
4. **Separate raw data, code and outputs.** Never edit raw data by hand; always produce results by running code.

---

## M10.2 Plain Python in one section

Python code is organized by indentation. The ideas you need:

- **Values and types:** integers (`3`), floats (`3.14`), strings (`"ATG"`), booleans (`True`), and the empty value `None`. Integer division is `//`, the remainder is `%`, and powers are `**` (not `^`, which means bitwise XOR).
- **Containers:** a **list** `[1, 2, 3]` is an ordered, changeable sequence; a **tuple** `(1, 2)` is unchangeable; a **dictionary** `{"A": "T", "C": "G"}` maps keys to values (the genetic code of Chapter M1 is a dictionary); a **set** `{1, 2, 3}` holds distinct items with fast membership tests. Dictionaries and sets are hash tables (Chapter M9), so `x in my_dict` is fast regardless of size, whereas `x in my_list` scans the whole list.
- **Control flow:** `for item in container:`, `while condition:`, `if ... elif ... else:`. A **comprehension** builds a container in one line: `[x * x for x in range(10) if x % 2 == 0]`.
- **Functions:** `def f(x, y=1): return x + y`. A function should do one thing, have a clear name, and avoid changing its inputs.
- **Indexing starts at 0**, and slices exclude the end: `s[2:5]` is characters 2, 3, 4. Negative indices count from the end: `s[-1]` is the last.
- **Reading errors:** a *traceback* lists the chain of calls; the **last line** names the error and the line above it shows where it happened. Read from the bottom.

!!! tip "A first useful program: a FASTA parser, a reverse complement, a translation"
    The script (§M10.7) parses a FASTA record, which is a header line starting with `>` followed by sequence lines; computes GC content (the indicator sum of Chapter M1, divided by the length); builds the **reverse complement** (complement each base and reverse the order, because the two DNA strands run in opposite directions); translates the sequence with the genetic-code dictionary of Chapter M1, three bases at a time; and counts 3-mers with `collections.Counter`. For the 39-base toy gene `ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG` it reports GC content $0.564$, the reverse complement `CTATCGGGCACCCTTTCAGCGGCCCATTACAATGGCCAT`, the translation `MAIVMGR*KGAR*` (a stop codon at the 8th codon, so a real gene finder would call the open reading frame $\text{MAIVMGR}$), and the most frequent 3-mer `GCC` (3 times).

    **Notice what the code does *not* do:** it ignores the reading frame (it assumes frame 0), the other strand, and ambiguous bases (`N`), and it would fail on lowercase letters. Every simplification is a modelling decision; production tools such as Biopython handle these cases.

---

## M10.3 NumPy: arrays, shapes, and broadcasting

NumPy provides the **array**, a block of numbers of one type with a **shape**, a tuple of axis lengths: a vector has shape `(d,)`, a matrix `(m, n)`, a batch of matrices `(B, m, n)`. It is the in-memory form of the vectors, matrices and tensors of Chapters M5 and M6. The ideas that matter:

**Creation and type.** `np.zeros((3, 4))`, `np.arange(10)`, `np.linspace(0, 1, 11)`, `rng.normal(size=(100, 5))`. Each array has a `dtype` (`float64` by default; `int64`, `float32`, ...). *Always know your dtype*: integer arrays silently truncate, and `float32` has less precision (§M10.4).

**Indexing and slicing.** `X[2, 3]` is one element; `X[:, 0]` is the first column; `X[X > 0]` selects elements by a **boolean mask**. Slices are **views** (no copy: changing the slice changes the original), whereas boolean and integer-array indexing returns **copies**.

**Reductions and `axis`.** `X.sum()` sums everything; `X.sum(axis=0)` sums *down* the rows (one value per column); `X.sum(axis=1)` sums *across* the columns (one value per row). The axis you name is the axis that **disappears**. `keepdims=True` keeps it with length 1, which is what you need for broadcasting.

**Broadcasting.** NumPy combines arrays of different shapes by *stretching* length-1 axes. The rule: align the shapes from the **right**; at each position the two lengths must be equal or one of them must be 1; the result takes the larger. In the script, a $6\times4$ counts matrix `X` is divided by a $6\times1$ column of per-cell totals (`depth`, computed with `keepdims=True`): the $(6,4)/(6,1)$ division stretches the column across the genes, normalizing every cell to $10^4$ total counts (every row sum prints $10000$). Subtracting a $(1,4)$ row of gene means centers the columns: their means print as $0$ to 12 decimals.

!!! warning "Silent broadcasting bugs"
    Broadcasting also makes mistakes quiet. In the script, adding a vector of shape `(3,)` to a column of shape `(3, 1)` does not raise an error; it produces a $3\times3$ array ($(3,)$ is treated as $(1,3)$ and both axes stretch), where one wanted a vector of length 3. A downstream `mean` then averages nine numbers instead of three, and the program runs happily. **Defences:** write the shape of every array in a comment; `assert x.shape == (n, d)` at function boundaries; use `keepdims=True` and `reshape` explicitly; use `np.einsum` with named axes. (Chapter 2 makes `einsum` a core tool: for example `np.einsum("bld,bkd->blk", Q, K)` contracts the feature axis `d` of two batches of matrices, giving shape $(2,5,4)$ for inputs $(2,5,7)$ and $(2,4,7)$ in the script.)

**Vectorization.** A Python `for` loop executes interpreted code for every element; a NumPy operation runs one compiled loop over the whole array and uses the processor's vector units and optimized libraries (BLAS) for matrix products. Timings measured by the script (hardware dependent): summing the squares of $10^6$ numbers takes $24$ ms in a Python loop and $1.5$ ms in NumPy, about $16\times$ faster; a $300\times300$ matrix product takes about $6.8$ seconds in pure-Python triple loops (extrapolated from a fifth of the rows) and $1.2$ ms with `A @ B`, a factor of several thousand. The matrix product is $O(n^3)$ either way (Chapter M9); the difference is the *constant*, which in practice decides what you can compute. **Rule:** if an array operation exists, use it; write an explicit loop only over things that are genuinely sequential (time steps, epochs, a Markov chain).

---

## M10.4 Floating-point numbers: why computers get arithmetic slightly wrong

A computer stores a real number in a fixed number of bits, as a **floating-point** number: a sign, a significand (the digits) and an exponent, much like scientific notation $\pm d.dddd\times2^{e}$. Standard formats: **float64** (double: 53 significand bits, about 16 decimal digits, range up to $1.8\times10^{308}$), **float32** (24 bits, about 7 digits) and **float16** and **bfloat16** (about 3 digits; used for fast, memory-saving deep learning). Four consequences follow.

1. **Most decimal numbers are not exactly representable.** $0.1$ has no finite binary expansion, so `0.1 + 0.2 == 0.3` is `False` in every language (script: $0.1+0.2=0.30000000000000004$). **Never compare floats with `==`;** use `np.isclose(a, b)` with a tolerance.
2. **Relative precision is limited by machine epsilon**, the gap between 1 and the next float: $2.2\times10^{-16}$ (float64), $1.2\times10^{-7}$ (float32), $9.8\times10^{-4}$ (float16). So `1 + 1e-16 == 1` is `True`, and adding a small number to a large one can lose it entirely: `1e16 + 1 - 1e16` evaluates to $0$, not $1$. float16 can represent integers exactly only up to 2048; the script shows 2049 stored as 2048.
3. **Range limits:** results beyond $1.8\times10^{308}$ **overflow** to infinity (`exp(710)` is `inf`), and tiny ones **underflow** to 0 (`exp(-800)` is `0.0`). This is the failure that Chapter M2's log-sum-exp prevents, and why probabilities are multiplied as sums of logs.
4. **Catastrophic cancellation:** subtracting two nearly equal numbers cancels the leading digits and leaves mostly rounding error. The script computes $(1-\cos x)/x^2$ for small $x$, whose true limit is $0.5$: at $x=10^{-5}$ the naive formula gives $0.5000000414$ (an error in the 8th digit) and at $x=10^{-8}$ it returns exactly $0$ (all digits lost), while the algebraically identical form $2\sin^2(x/2)/x^2$ gives $0.5$ at every $x$. *Mathematically equal expressions are not numerically equal.* The same mechanism underlies the U-shaped error of finite-difference derivatives (Chapter M3).

!!! math "A numerical case study: the variance of data with a large offset"
    The textbook identity $\operatorname{Var}(y)=\mathbb{E}[y^2]-(\mathbb{E}y)^2$ is exact, but it subtracts two enormous, nearly equal numbers when the mean is large compared with the spread. For $10^5$ values with mean $10^9$ and true variance $1$ (script), the naive formula returns $0.0000$ in float64 (the difference is below the rounding error of the terms, which are about $10^{18}$ with absolute error near $10^{2}$), and in float32 it returns $4.8\times10^{11}$: complete garbage. The **two-pass** formula (compute the mean first, then average $(y-\bar y)^2$) and **Welford's streaming algorithm** both give $0.9955$ (the sample variance of that particular sample). Welford updates a running mean and sum of squares one observation at a time, $\delta=y_i-m_{i-1}$, $m_i=m_{i-1}+\delta/i$, $M_i=M_{i-1}+\delta(y_i-m_i)$, using only stable differences, and is the algorithm inside most statistics libraries.

    **Why this matters for biology.** Gene expression, fluorescence and mass-spectrometry intensities often have large baselines with small informative differences; summing large squares of raw values (as a one-line variance does) can erase the signal. Centering the data first is both statistically and numerically wise. In deep learning, **mixed precision** (float16/bfloat16 for speed with float32 for accumulations) works only because a few operations (sums, softmax normalizers, loss values) are kept in higher precision.

**Practical rules.** (i) Work in float64 unless you have a reason not to. (ii) Use library routines (`np.var`, `scipy.special.logsumexp`, `np.linalg.solve`) rather than textbook formulas typed from memory: they are written with these pitfalls in mind. (iii) Compare floats with tolerances. (iv) If a result changes when you change the order of operations or the precision, it is *numerically unstable*, and you should treat it as unreliable until you understand why.

---

## M10.5 Randomness: seeds, reproducibility, and variability

A computer's "random" numbers come from a **pseudorandom number generator**: a deterministic function that, from an initial **seed**, produces a sequence that passes statistical tests of randomness. It is deterministic by design: the same seed yields the same sequence, which is what lets you reproduce an experiment. In NumPy, create a generator with `rng = np.random.default_rng(seed)` and pass it around, rather than using a hidden global state.

Two uses of seeds, and both are needed:

- **Fix the seed to reproduce.** The outputs shown in this book are the outputs for the stated seed.
- **Vary the seed to measure chance.** A single run of a stochastic procedure is *one draw* from a distribution of possible results. In the script, a simple two-group experiment with 20 samples per group and a true effect of $0.3$ gives, over six seeds, estimated effects $0.05,\,0.37,\,0.19,\,0.12,\,0.47,\,0.56$. Over 2,000 seeds the estimate has mean $0.29$ and standard deviation $0.30$, and $16\%$ of runs even have the wrong sign. A result reported from one seed (or one train/test split, one initialization) may be luck.

!!! warning "Single-seed comparisons"
    In machine learning, a difference of one point between two methods, each trained once, is often smaller than the spread across seeds. Honest comparison reports the mean and standard deviation (or an interval) over several seeds, ideally with *paired* runs sharing data splits (Chapters 43 and 47). Wherever this book reports a single-seed result, it says so, because a single seed supports an illustration but not a general claim.

---

## M10.6 Looking at data, and plotting

**Plot before you model.** Summary statistics can hide the structure that matters. **Anscombe's quartet** (1973) is four small datasets with the same mean of $x$ ($9.00$), mean of $y$ ($7.50$), variance of $y$ ($\approx4.12$), correlation ($0.816$) and fitted line ($y=0.50x+3.00$) (script output), yet one is a clean linear relationship, one is a curve, one is linear with a single outlier, and one is a vertical stack with a single high-leverage point controlling the fit. Any analysis that reports only the summary would call them identical. A scatter plot distinguishes them in a second.

The standard tool is **matplotlib** (`import matplotlib.pyplot as plt`). The plots to make by habit: a **histogram** or density of each variable (shape, outliers, zeros); a **scatter** plot of each pair of variables (linearity, clusters); the same on a **log scale** when data span orders of magnitude (Chapter M2); a **residual** plot after fitting a model (structure left over means the model is wrong); and, for time series, the series itself. For high-dimensional data, plot a PCA or UMAP embedding (Chapter 30) *and* colour it by every technical variable (batch, depth, donor) before colouring it by biology.

---

## M10.7 A first look at PyTorch and automatic differentiation

**PyTorch** has the NumPy-like **tensor** (same shapes, broadcasting and indexing) plus two things deep learning needs: computation on a **GPU**, and **automatic differentiation** (autograd). If a tensor is created with `requires_grad=True`, PyTorch records every operation applied to it in a computation graph (a DAG, Chapter M9); calling `.backward()` on a scalar result then applies the chain rule in reverse order (Chapters M3, M4) and stores $\partial\text{result}/\partial\text{tensor}$ in `tensor.grad`.

The script reproduces a gradient from Chapter M4: the loss $-\ln\sigma(\mathbf{w}^\top\mathbf{x})$ for $\mathbf{w}=(0.5,-1,2)$, $\mathbf{x}=(1,2,-0.5)$. Autograd returns the gradient $(-0.92414,\,-1.84828,\,0.46207)$, identical to the hand-derived formula $-(1-\sigma(\mathbf{w}^\top\mathbf{x}))\,\mathbf{x}$ to every printed digit. This is the entire mechanism behind training a neural network with a million parameters: write the forward computation, call `.backward()`, and update the parameters opposite to the gradient (Chapter M4's gradient descent). Chapter 9 explains how autograd works; Chapter 6 gives a training skeleton. Habits that carry over from NumPy: write the shape of each tensor, check `x.shape`, and use `torch.no_grad()` for evaluation so that no graph is recorded.

```python
--8<-- "code/m10_numerics.py"
```

Output (seed 0; the timings depend on the machine):

```text
sequence ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG (length 39), GC content 0.564
reverse complement CTATCGGGCACCCTTTCAGCGGCCCATTACAATGGCCAT
translation (stop = *): MAIVMGR*KGAR*
3-mer counts, top 3: [('GCC', 3), ('ATG', 2), ('TGG', 2)] 

X shape (6, 4) | depth shape (6, 1) | normalized shape (6, 4) | every cell now sums to [10000. 10000. 10000. 10000. 10000. 10000.]
centered columns have mean [-0.  0. -0. -0.] | shape rule: align shapes from the right; each pair of sizes must be equal or 1
(3,1) + (1,4) -> (3, 4) ; an accidental (3,) + (3,1) -> (3, 3) (a silent shape bug: 3 x 3 instead of 3)
einsum 'bld,bkd->blk' style contraction: (2, 5, 4)

sum of squares of 1e6 numbers (best of 5): Python loop 24 ms, NumPy 1.5 ms -> 16x faster (hardware dependent)
300x300 matrix product: Python loops (est.) 6.8 s, NumPy matmul 1.2 ms; same answer on the computed rows: True

0.1 + 0.2 == 0.3 ? False (0.1 + 0.2 = 0.30000000000000004);  np.isclose: True
machine epsilon (double) = 2.220e-16; float32 = 1.192e-07; float16 = 9.766e-04
1 + 1e-16 == 1 ? True  | adding a small number to a large one: 1e16 + 1 - 1e16 = 0.0
largest float64 ~ 1.80e+308 | exp(710) = inf (overflow) | exp(-800) = 0.0 (underflow to 0)
float16 cannot represent 2049: stored as 2048.0 (integers above 2048 are rounded to even)
  x = 1e-02: naive (1 - cos x)/x^2 = 0.4999958333   stable form = 0.4999958333   (true limit 0.5)
  x = 1e-05: naive (1 - cos x)/x^2 = 0.5000000414   stable form = 0.5000000000   (true limit 0.5)
  x = 1e-08: naive (1 - cos x)/x^2 = 0.0000000000   stable form = 0.5000000000   (true limit 0.5)

variance of data with mean 1e9 and true variance 1.0:  naive E[y^2] - E[y]^2 = 0.0000;  two-pass = 0.9955;  Welford = 0.9955
with float32 inputs the naive formula gives 481036337152.0 (garbage, possibly negative)

same seed, same result: True | different seeds: [0.05, 0.37, 0.19, 0.12, 0.47, 0.56]
over 2000 seeds the 'effect' (true 0.3) has mean 0.29 and standard deviation 0.30; fraction of runs with the wrong sign: 0.16

autograd gradient [-0.9241399765014648, -1.8482799530029297, 0.4620699882507324] | hand formula -(1 - sigmoid(w.x)) x = [-0.9241399765014648, -1.8482799530029297, 0.4620699882507324] | tensor shapes: w (3,), loss ()

Anscombe's quartet: mean x, mean y, var y, correlation, fitted slope and intercept
  dataset 1: 9.00  7.50  4.13  r = 0.816  y = 0.50 x + 3.00
  dataset 2: 9.00  7.50  4.13  r = 0.816  y = 0.50 x + 3.00
  dataset 3: 9.00  7.50  4.12  r = 0.816  y = 0.50 x + 3.00
  dataset 4: 9.00  7.50  4.12  r = 0.817  y = 0.50 x + 3.00
  (a straight line, a curve, an outlier, and a single leverage point: identical numbers, completely different data)
```

---

## M10.8 Habits for code you can trust

Most computational errors in research are not syntax errors; they are code that *runs* and is *wrong*. The defences are habits.

1. **Test on a case with a known answer.** Before running on real data, run on a tiny example you can verify by hand, or on simulated data where you know the truth (Chapter M8's simulations; Chapter 30's false-correlation controls).
2. **Assert shapes and invariants.** `assert probs.shape == (n, K)`; `assert np.allclose(probs.sum(1), 1)`; `assert not np.isnan(loss)`. A failing assertion near the cause beats a mysterious number far from it.
3. **Compare two implementations.** A fast version against a slow, obvious one (the script compares Gaussian elimination to `np.linalg.solve`, autograd to the hand formula, finite differences to analytic gradients). Agreement is strong evidence; disagreement locates the bug.
4. **Change one thing at a time**, and keep the old result to compare with.
5. **Print or plot intermediate quantities**: shapes, means, ranges, the first few rows. Most bugs are visible in the data after the first bad step.
6. **Debug by bisection.** If a pipeline of ten steps produces nonsense, check the output of step 5, then 2 or 8, halving the suspect region each time.
7. **Make a minimal reproducible example** of any bug, which usually reveals the cause by itself and is what you need to ask for help.
8. **Use version control and write down what you ran.** A result you cannot reproduce is not a result.
9. **Beware *plausible* wrong answers.** An accuracy of 0.97 is not evidence of correctness; leakage between training and test data (Chapter 43) produces great numbers from broken code.

!!! example "Worked example M10.1: A silent shape bug in a normalization"
    **Situation.** You normalize a counts matrix `X` of shape $(N,G)$ (cells by genes) so that each *gene* has mean 0, but write `Xc = X - X.mean(axis=1)`. The code runs.

    **What happens.** `X.mean(axis=1)` has shape `(N,)`. Broadcasting `(N, G) - (N,)` aligns shapes from the right: `(N,)` is treated as `(1, N)`. If $N=G$ it runs and subtracts the wrong thing; if $N\ne G$ it raises a shape error, which is the lucky outcome. With a square test matrix (often chosen for convenience) the bug is invisible.

    **Fix and defence.** Use `X - X.mean(axis=0, keepdims=True)` (per-gene means, shape `(1, G)`), then `assert np.allclose(Xc.mean(axis=0), 0)`. And test on a *non-square* array with distinct sizes, so that a shape mix-up cannot go unnoticed (use $N=5$, $G=7$, never $5\times5$).

    **Lesson.** Silent broadcasting makes *correct shape* a property you must test for, not one the language guarantees.

!!! example "Worked example M10.2: A likelihood that underflows"
    **Situation.** You compute the likelihood of a sequence of 1,500 independent bases under a model with per-base probabilities between 0.2 and 0.4 as `np.prod(probs)`; the result is `0.0` and you conclude "the sequence is impossible under the model".

    **Reasoning.** Each factor is around $0.3$, so the product is about $0.3^{1500}=10^{-784}$, below the smallest representable float64 ($\approx10^{-308}$; denormals reach about $10^{-324}$). The product underflowed to exactly zero. The *mathematical* likelihood is positive; the computed one is not.

    **Fix.** Work with **log-likelihoods**: `np.sum(np.log(probs))` $\approx-1{,}800$ is perfectly representable. To compare two models, subtract the logs (a log-likelihood ratio). To normalize across models, use log-sum-exp (Chapter M2).

    **Lesson.** Probabilities of long sequences are *always* handled in log space. This is not an optimization; it is correctness.

---

## M10.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The code ran, the loss went down, so it works\""
    **The observation.** A newly written training loop for a sequence model runs without errors, and the training loss falls steadily from 1.38 to 0.9 over a few hundred steps.

    **Tempting conclusion.** "The implementation is correct and the model is learning the task."

    **Decompose.** A falling training loss is the *weakest* evidence of correctness. A model with a bug can still reduce its training loss: by memorizing, by exploiting a **leak** (the target is accidentally present in the input), by learning a trivial feature (sequence length, a padding pattern, the position of a special token) or by learning the *marginal class frequencies* (a loss of $\ln4=1.386$ for four equally likely bases is the starting point of a model that has learned nothing; reaching $1.3$ may reflect only base composition). A loss that starts exactly at $\ln K$ for a $K$-way classification is a good sign about the initialization; the *end* value must be compared with the right baseline (the entropy of the label distribution, a unigram or $k$-mer model, Chapter 32).

    **Hidden assumptions.** (i) The data pipeline aligns inputs and targets correctly (an off-by-one shift is a classic bug). (ii) The train and validation sets share no near-duplicates. (iii) Numerical issues (float16 overflow, `NaN`s) are not being silently clipped.

    **Discriminating experiments.** Overfit a *tiny* subset (10 examples): a correct model drives the loss near zero, and failure indicates a bug rather than a hard task. Shuffle the labels: a correct pipeline's performance should fall to chance. Compare with trivial baselines. Check a few predictions by eye. Verify gradients on a tiny network against finite differences.

    **What this teaches.** "No error" is not "correct", and "better than before" is not "good". The habits of §M10.8 are the experimental design of computational work.

---

## M10.10 Connections

- **Forward:** Chapter 6 develops computational thinking for research: complexity, vectorization, experiment management, reproducible pipelines and a training skeleton; Chapter 9 explains autograd and numerical stability of softmax and cross-entropy; Chapter 28 treats sequence tokenization and file formats (FASTA, FASTQ, VCF, h5ad); Chapter 43 develops leakage and honest evaluation; Appendix C lists every script in the book with run instructions.
- **Backward:** every mathematical chapter of Part 0: the genetic-code dictionary (M1), log-sum-exp and logs (M2), finite differences and Euler's method (M3), gradient checks (M4), linear solves and conditioning (M5), SVD (M6), seeds and Monte Carlo (M7), simulation of statistics (M8), data structures (M9).

!!! takeaways "Key takeaways"
    1. Use a **virtual environment**, record versions, **fix seeds**, and keep code in version control: reproducibility is a baseline, not an extra.
    2. Python dictionaries and sets are hash tables (fast membership); lists are scanned. A FASTA parser, reverse complement and translation take a dozen lines.
    3. NumPy arrays have a **shape** and **dtype**. In reductions the named `axis` is the one that disappears; `keepdims=True` preserves it for broadcasting. Broadcasting aligns shapes from the right and **fails silently** when shapes happen to be compatible: assert shapes, use named `einsum` axes, and test on non-square arrays.
    4. **Vectorize**: NumPy was 16$\times$ faster for a sum of squares and thousands of times faster for a matrix product than the equivalent pure-Python loops (hardware dependent).
    5. **Floating point** is finite: `0.1 + 0.2 != 0.3`, machine epsilon is $2.2\times10^{-16}$ (float64), overflow gives `inf`, underflow gives 0, and subtracting nearly equal numbers destroys digits. Mathematically equal formulas can be numerically unequal; use stable library routines, Welford/two-pass variance, and log-space for probabilities.
    6. A **seed** makes randomness reproducible; **varying the seed** measures it. One seed is one draw (in the example, 16% of runs had the wrong sign for a true effect of 0.3 with $n=20$).
    7. **Look at the data**: Anscombe's four datasets share all summary statistics and differ completely in structure.
    8. **Autograd** applies the chain rule to the recorded computation graph; it matched the hand-derived logistic gradient to every digit. "No errors" and "loss decreasing" are weak evidence: test on tiny cases, assert invariants, compare implementations, and overfit a small subset.

---

## Further reading

- Downey, A. B. *Think Python* (free online). A clear first course in programming.
- VanderPlas, J. *Python Data Science Handbook* (free online). NumPy, pandas and matplotlib in depth.
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585, 357–362. The array model and broadcasting.
- Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys* 23, 5–48. The classic account.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*. SIAM. Welford, cancellation and rounding error, rigorously.
- Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLoS Computational Biology* 13, e1005510. Practical habits for reproducible research code.
- Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician* 27, 17–21.
- Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*.
