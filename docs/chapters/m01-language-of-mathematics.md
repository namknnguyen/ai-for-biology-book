# Chapter M1. The Language of Mathematics: Sets, Functions, Logic, Proof, and Counting

!!! abstract "Chapter at a glance"
    **Motivation.** Every later chapter writes its ideas in mathematical sentences. Before you can use a theorem you must be able to *read* it: to see what kind of object each symbol names, what is being claimed, and what would show the claim false. This chapter teaches the grammar. It is deliberately slow, because reading mathematics is a skill, and it is the skill people most often skip.

    **Prerequisites.** High-school algebra. Nothing else. If you can solve $2x + 3 = 11$ and you know what a fraction and an exponent are, you are ready.

    **You will be able to:** (1) read a formula by identifying its objects, their types, and its quantifiers; (2) use set and function notation, and say when a map loses information; (3) distinguish a statement from its converse and contrapositive, and *necessary* from *sufficient*; (4) prove a simple statement by induction, contradiction, or counting; (5) manipulate sums, products and indicator functions without losing indices; (6) count arrangements, estimate astronomically large numbers, and compute collision probabilities; (7) use big-$O$, $\approx$ and $\sim$ honestly.

---

## M1.1 How to read a formula

A line of mathematics is a very compressed sentence. The compression is the point: a symbol such as $\sum$ lets us say "add up all of these" once instead of writing a hundred terms. The cost is that the reader must *decompress*. Here is a procedure that works for every formula in this book.

**Reading procedure.**

1. **Name the objects and their types.** Is $x$ a number, a vector, a function, a set, a random variable? In this book vectors are bold ($\mathbf{x}$), matrices are bold capitals ($\mathbf{A}$), and sets are usually capitals in a different font or are written in braces. The type tells you which operations make sense. You cannot add a number to a set.
2. **Find the quantifiers.** "For every," "there exists," "for some." A formula with a hidden quantifier is the most common source of misreading (§M1.3).
3. **Find the verb.** Every complete mathematical sentence has one: $=$ (is equal to), $\le$ (is at most), $\in$ (is an element of), $\Rightarrow$ (implies). An expression without a verb, such as $\sum_i x_i^2$, is a noun phrase: it names a number.
4. **Test it on the smallest case.** Put $n = 1$ or $n = 2$, or a $2 \times 2$ matrix, and compute by hand. If you cannot do the smallest case you do not yet understand the formula.
5. **Say it in words and in a picture.** "The squared length of the vector." If you cannot, keep reading the surrounding text; the author owes you that sentence.

Here is the procedure applied to a formula from Chapter 4, the *sample mean*:

$$
\bar{x} \;=\; \frac{1}{n}\sum_{i=1}^{n} x_i .
$$

Objects: $x_1, \dots, x_n$ are $n$ numbers (say, expression of one gene in $n$ cells); $n$ is a whole number; $\bar{x}$ is a number. The verb is "$=$": the left side is *defined* by the right side. The quantifier is implicit: the index $i$ runs over *every* whole number from 1 to $n$. Smallest case: $n = 2$, $x_1 = 3$, $x_2 = 5$, so $\bar{x} = (3+5)/2 = 4$. In words: add up all the measurements and divide by how many there are. Nothing is mysterious; the formula was never harder than that, only shorter.

!!! tip "The Greek alphabet you will meet"
    | Symbol | Name | Typical use in this book |
    |---|---|---|
    | $\alpha, \beta, \gamma$ | alpha, beta, gamma | rates, regression coefficients, hyperparameters |
    | $\delta, \Delta$ | delta | a small change; a difference |
    | $\epsilon, \varepsilon$ | epsilon | a tiny quantity; noise |
    | $\theta, \boldsymbol\theta$ | theta | the parameters of a model |
    | $\lambda$ | lambda | a rate; an eigenvalue; a regularization strength |
    | $\mu, \sigma$ | mu, sigma | a mean; a standard deviation |
    | $\pi$ | pi | 3.14159...; also a probability distribution or a policy |
    | $\rho, \tau$ | rho, tau | a correlation; a time constant or temperature |
    | $\phi, \psi$ | phi, psi | a feature map; an angle; a wavefunction |
    | $\Sigma$ (capital) | Sigma | a sum *or* a covariance matrix; context tells which |
    | $\omega, \Omega$ | omega | a frequency; a sample space |

    Letters are reused across fields. Never assume a symbol keeps its meaning from one chapter to the next: read its definition.

---

## M1.2 Sets and functions

### M1.2.1 Sets

A **set** is a collection of distinct objects, written in braces. The four DNA bases form the set $\{A, C, G, T\}$. The notation $x \in S$ reads "$x$ is an element of $S$", and $S \subseteq T$ reads "every element of $S$ is also in $T$". Order and repetition do not matter: $\{A, C\} = \{C, A\} = \{A, A, C\}$.

The operations you need:

| Operation | Notation | Meaning | Example |
|---|---|---|---|
| Union | $S \cup T$ | elements in $S$ or $T$ (or both) | genes expressed in either of two tissues |
| Intersection | $S \cap T$ | elements in both | genes expressed in both |
| Difference | $S \setminus T$ | in $S$ but not in $T$ | genes expressed in $S$ only |
| Cartesian product | $S \times T$ | all ordered pairs $(s, t)$ | all (gene, tissue) pairs |
| Size (cardinality) | $\lvert S \rvert$ | number of elements | $\lvert\{A,C,G,T\}\rvert = 4$ |
| Empty set | $\varnothing$ | the set with no elements | genes expressed in neither |

Sets of numbers have standard names: $\N = \{0, 1, 2, \dots\}$ (the natural numbers), $\mathbb{Z}$ (the integers), $\mathbb{Q}$ (fractions) and $\R$ (the real numbers, the whole number line with no gaps). The notation $\R^d$ means the set of ordered lists of $d$ real numbers: a point in $d$ dimensions, or a **vector**. A cell described by the expression of 20,000 genes is a point in $\R^{20000}$. A DNA sequence of length $L$ is an element of $\{A,C,G,T\}^L$, the product of $L$ copies of the base set. Set notation is how the book states, precisely, "what kind of thing is this data point?"

Set-builder notation describes a set by a property: $\{x \in \R : x > 0\}$ reads "the set of real $x$ such that $x > 0$". The colon (or a vertical bar) means "such that".

**Inclusion–exclusion** is the first counting rule worth memorizing. For two finite sets,

$$
\lvert S \cup T \rvert \;=\; \lvert S \rvert + \lvert T \rvert - \lvert S \cap T \rvert ,
$$

because elements in the intersection were counted twice. If 6,000 genes are expressed in liver, 5,000 in brain, and 3,500 in both, then $6000 + 5000 - 3500 = 7500$ genes are expressed in at least one of the two. Forgetting the subtraction is how overlapping gene lists produce impossible totals.

### M1.2.2 Functions

A **function** $f : A \to B$ assigns to every element $a$ of the **domain** $A$ exactly one element $f(a)$ of the **codomain** $B$. The word *exactly one* is the whole definition. The set of values actually produced, $\{f(a) : a \in A\}$, is the **image** (or range). A function is a rule, not necessarily a formula; a lookup table is a function.

!!! math "Definitions: injective, surjective, bijective"
    A function $f : A \to B$ is

    - **injective** (one-to-one) if different inputs always give different outputs: $f(a) = f(a') \Rightarrow a = a'$;
    - **surjective** (onto) if every element of $B$ is hit: for every $b \in B$ there is an $a$ with $f(a) = b$;
    - **bijective** if both. A bijection can be *undone*: there is an inverse function $f^{-1}$ with $f^{-1}(f(a)) = a$.

    Injective means *no information is lost* going forward. Surjective means *nothing in $B$ is unreachable*.

!!! bio "Biology for modeling: the genetic code is a function that loses information"
    **What is it?** Translation reads DNA in triplets (codons). The genetic code is the function $\text{code} : \{A,C,G,T\}^3 \to \{\text{20 amino acids}\} \cup \{\text{stop}\}$ from 64 codons to 21 outcomes.

    **Information it contains.** The identity of the next amino acid, or a stop signal.

    **What it discards.** Which of several synonymous codons was used. With 64 inputs and 21 outputs the map cannot be injective, so it cannot be inverted: many codons share an output.

    **Modeling consequence.** A protein sequence does *not* determine the DNA that encoded it. In the code below, the ten-residue peptide MKTAYIAKQR can be encoded by $1\cdot 2\cdot 4\cdot 4\cdot 2\cdot 3\cdot 4\cdot 2\cdot 2\cdot 6 = 18{,}432$ different DNA sequences. A protein language model trained on amino-acid strings never sees which of those was used, so it can say nothing about codon choice, translation speed, or mRNA stability (Chapters 33 and 34 return to this). The asymmetry of *protein to DNA* versus *DNA to protein* is a statement about injectivity.

    **What would change the interpretation.** Different organisms and organelles use slightly different codes: the *function* depends on the species, so a model trained on one genetic code should not be applied to another without checking.

Composition of functions is the other operation to master. If $f : A \to B$ and $g : B \to C$, then $g \circ f : A \to C$ is "first $f$, then $g$": $(g \circ f)(a) = g(f(a))$. A deep neural network is exactly a composition of many simple functions, and the chain rule of calculus (Chapter M3) is the rule for differentiating a composition.

### M1.2.3 Code: seeing the structure of a function

The script below builds the genetic code as a Python dictionary (a function in the plainest sense), computes its *preimages*, and does the same kind of counting for a few other objects used in this chapter. It uses only the standard library and NumPy.

```python
--8<-- "code/m01_language.py"
```

Output (seed 0):

```text
domain size 64, codomain (image) size 21
is the map one-to-one? False | is it onto the 21 symbols? True
preimage sizes (codons per outcome): [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6]
number of DNA sequences that encode the 10-residue peptide MKTAYIAKQR: 18432

checked n(n+1)/2 for n = 1..2000: True;  geometric series identity (r=0.9, N=50): True
the false guess 'sum of first n odd numbers = n^2 + 1' fails at n = 1

number of DNA sequences of length L  (4^L):
  L =  10: 4^L = 1.049e+06
  L =  20: 4^L = 1.100e+12
  L =  30: 4^L = 1.153e+18
  L = 100: 4^L = 1.607e+60
number of proteins of length 100 (20^100): 1.268e+130   (atoms in the observable universe ~ 1e80)
ways to choose 5 mutated sites among 300 positions: C(300,5) = 19,582,837,560
ways to order 8 distinct genes on a chromosome arm: 8! = 40,320
  Stirling n=  5: relative error 1.651%
  Stirling n= 20: relative error 0.416%
  Stirling n=100: relative error 0.083%
length-8 sequences with exactly 3 G/C: brute force 14336, formula C(8,3)*2^8 = 14336

barcode collisions (birthday problem): P(at least one pair of cells shares a barcode)
  B =        365  n =     23: exact 0.5073   exp-approx 0.5000
  B =     65,536  n =    300: exact 0.4961   exp-approx 0.4956
  B = 16,777,216  n =  5,000: exact 0.5253   exp-approx 0.5252
  B = 4,294,967,296  n = 10,000: exact 0.0116   exp-approx 0.0116
  simulation, B = 65,536, n = 300: 0.490 (exact 0.496)
  cells needed for a 50% chance of some collision with 12-base barcodes: about 4,823 (= 1.18 sqrt(B))
```

The preimage sizes show the degeneracy of the code: two outcomes (Met and Trp) have a single codon, Leu, Ser and Arg have six each, the stop signal has three, and the rest sit in between. The 21 outcomes partition the 64 codons: the preimages are disjoint and together cover the domain.

---

## M1.3 Logic: statements, quantifiers, and the structure of a claim

### M1.3.1 Statements and implication

A **statement** is a sentence that is either true or false. "The mean of these four numbers is 4" is a statement. The most important kind is the **implication** $P \Rightarrow Q$ ("if $P$ then $Q$"). Four related statements are constantly confused:

| Name | Form | Equivalent to the original? |
|---|---|---|
| Implication | $P \Rightarrow Q$ | (the original) |
| Converse | $Q \Rightarrow P$ | **No** |
| Contrapositive | $\text{not } Q \Rightarrow \text{not } P$ | **Yes** |
| Negation | $P$ and not $Q$ | the statement that the implication is *false* |

Take $P$ = "this mutation is a frameshift in the coding region of gene $G$" and $Q$ = "the protein of gene $G$ is truncated or non-functional". "If a frameshift, then non-functional" is plausible. Its converse ("if non-functional, then frameshift") is false: a missense mutation can also destroy a protein. Its contrapositive ("if the protein is functional, then there was no frameshift") is true *if the original is*, and it is often the form you can test.

**Necessary and sufficient.** $P \Rightarrow Q$ says $P$ is *sufficient* for $Q$ (having $P$ guarantees $Q$) and $Q$ is *necessary* for $P$ (you cannot have $P$ without $Q$). "$P$ if and only if $Q$" ($P \Leftrightarrow Q$) says both directions hold. These words are the vocabulary of experimental biology:

- A gene is **necessary** for a phenotype if knocking it out removes the phenotype (loss of function).
- A gene is **sufficient** if adding it to a naive cell produces the phenotype (gain of function).
- Many genes are necessary but not sufficient; some are sufficient but not necessary (redundant pathways). A paper that shows necessity and then writes "sufficient" has changed the claim.

### M1.3.2 Quantifiers

Two quantifiers appear everywhere:

- $\forall$ ("for all"): $\forall x \in S,\ P(x)$ says *every* element satisfies $P$.
- $\exists$ ("there exists"): $\exists x \in S,\ P(x)$ says *at least one* does.

Their negations swap:

$$
\text{not}\,(\forall x,\ P(x)) \;\Longleftrightarrow\; \exists x,\ \text{not}\,P(x),
\qquad
\text{not}\,(\exists x,\ P(x)) \;\Longleftrightarrow\; \forall x,\ \text{not}\,P(x).
$$

So to disprove "every gene in the pathway is essential" you need *one* non-essential gene. To disprove "some gene in the pathway is essential" you need to show *all* of them are non-essential, which is much more work. When the order of different quantifiers changes, the meaning changes. Compare:

- $\forall \text{ cells } c,\ \exists \text{ a gene } g$ that is expressed in $c$: every cell expresses something (almost certainly true).
- $\exists \text{ a gene } g,\ \forall \text{ cells } c$: $g$ is expressed in $c$: one gene is expressed in *every* cell (a housekeeping gene; a much stronger and rarer claim).

Mathematicians write these carefully. Biologists often write them loosely ("this gene is expressed in neurons"), and the ambiguity (in *all* neurons? *some*? *at what level*?) is a frequent source of irreproducible claims. The habit to build: whenever you read a general claim, silently insert the quantifier and ask whether it is the one the evidence supports.

!!! warning "The conditional probability trap, previewed"
    "Most patients with the disease test positive" and "most patients who test positive have the disease" are different statements. They have the form $P \Rightarrow Q$ and its converse. Chapter M7 turns that difference into a number (Bayes' rule), and it is one of the most consequential confusions in diagnostics and in genomics screens.

---

## M1.4 Proof: what it buys you, and four ways to do it

A **proof** is an argument that a statement is true in *every* case it covers, using only definitions and previously proved facts. Compare it with an experiment: an experiment samples the world and reports what it found; a proof covers all cases of an abstract object at once. Computer checks, like the one in the script above, are experiments on a mathematical object: they can refute a conjecture (a single counterexample suffices) but a billion successes prove nothing about the next case. Both activities are useful, and they feed each other. In this book proofs appear in *derivations* so that you can see exactly which assumptions a result needs; those assumptions are where the biology can break the model.

### M1.4.1 Direct proof

Assume the hypotheses, and chain definitions and known facts to the conclusion.

**Claim.** The sum of two even integers is even. **Proof.** Even means "equal to $2k$ for some integer $k$." Let the two integers be $2a$ and $2b$. Then $2a + 2b = 2(a + b)$, and $a + b$ is an integer. $\square$

### M1.4.2 Contrapositive and contradiction

To prove $P \Rightarrow Q$ you may instead prove its contrapositive, $\text{not}\,Q \Rightarrow \text{not}\,P$. In **proof by contradiction** you assume the statement is false and derive an absurdity.

**Claim.** If a DNA sequence of length $n$ is partitioned into $m < n$ non-overlapping windows, then some window has length at least 2. **Proof (pigeonhole).** Suppose every window had length at most 1. Then the total length would be at most $m < n$, contradicting that the windows cover all $n$ positions. $\square$

The **pigeonhole principle** (if you place more than $m$ objects into $m$ boxes, some box holds at least two) is the engine of many counting results, including the barcode collisions of §M1.7.

### M1.4.3 Induction

Induction proves a statement $S(n)$ for all integers $n \ge 1$ by two steps: the **base case** (prove $S(1)$) and the **inductive step** (assume $S(n)$ and prove $S(n+1)$). It is a row of dominoes: the base case knocks over the first, and the step guarantees each domino knocks over the next.

!!! math "Worked proof: the sum of the first $n$ integers"
    **Claim.** For every integer $n \ge 1$,

    $$
    1 + 2 + \cdots + n \;=\; \frac{n(n+1)}{2}.
    $$

    **Base case ($n = 1$).** The left side is $1$; the right side is $1\cdot 2/2 = 1$. True.

    **Inductive step.** Assume the formula holds for $n$. Then

    $$
    1 + 2 + \cdots + n + (n+1) \;=\; \frac{n(n+1)}{2} + (n+1) \;=\; \frac{n(n+1) + 2(n+1)}{2} \;=\; \frac{(n+1)(n+2)}{2},
    $$

    which is the formula for $n + 1$. By induction the formula holds for all $n \ge 1$. $\square$

    **What the code adds.** The script verified the formula for $n$ up to 2,000 (a *check*), and showed that a plausible-looking wrong guess ("the sum of the first $n$ odd numbers is $n^2 + 1$") already fails at $n = 1$. Checking the base case first is often the cheapest way to kill a bad conjecture. The true statement, proved the same way, is that the sum of the first $n$ odd numbers is $n^2$.

The same pattern proves the **geometric series** formula that will model mutation accumulation and decay (Chapter M2):

$$
1 + r + r^2 + \cdots + r^N \;=\; \frac{1 - r^{N+1}}{1 - r}\qquad (r \ne 1),
$$

and for $\lvert r \rvert < 1$, letting $N \to \infty$, the infinite sum is $1/(1-r)$. With $r = 0.9$ the infinite sum is $10$: if each generation retains 90% of the previous generation's signal and you add one unit per generation, the total signal saturates at 10 units.

!!! example "Worked example M1.1: a claim, its converse, and what an experiment can show"
    **Situation.** A paper reports: "In every tumor with a *KRAS* G12D mutation we examined, the downstream MAPK pathway was active."

    **Question.** Which of the following can you conclude? (a) A tumor with an active MAPK pathway has a *KRAS* G12D mutation. (b) A tumor in which MAPK is inactive does not have *KRAS* G12D. (c) *KRAS* G12D is sufficient to activate MAPK in a new cell type.

    **Reasoning.** Let $P$ = "tumor has *KRAS* G12D" and $Q$ = "MAPK is active". The paper reports $P \Rightarrow Q$ for the tumors examined.

    - (a) is the converse $Q \Rightarrow P$: not implied. MAPK can be activated by other mutations (*BRAF*, *EGFR*) or by growth-factor signaling.
    - (b) is the contrapositive $\text{not } Q \Rightarrow \text{not } P$: it *is* implied, *for the tumors examined*, and it is the useful form: an inactive MAPK pathway would count as evidence against a *KRAS* G12D driver.
    - (c) is a claim of sufficiency in a different setting. The tumors examined may all share other changes that are also required. An implication observed in a sample does not establish a causal sufficiency claim.

    **Also check the quantifier.** "In every tumor we examined" is $\forall$ over a *finite sample*, not over all tumors. The honest scope is "in these $n$ tumors", and a single counterexample elsewhere would refute the general version.

    **Lesson.** Writing the statement as $P \Rightarrow Q$ takes ten seconds and removes most of the confusion. This is the same discipline the Expert Chain applies when it asks for the *assumptions* behind a claim (Chapter 1).

---

## M1.5 Sums, products, indices, and indicators

Almost every formula in this book contains a sum, so index fluency is worth a section.

**Summation.** $\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n$. The letter $i$ is a **dummy index**: it exists only inside the sum, and renaming it changes nothing: $\sum_i a_i = \sum_j a_j$. Three rules cover most uses:

$$
\sum_i (a_i + b_i) = \sum_i a_i + \sum_i b_i,
\qquad
\sum_i c\,a_i = c\sum_i a_i,
\qquad
\sum_{i}\sum_{j} a_{ij} = \sum_{j}\sum_{i} a_{ij}.
$$

The last, *swapping the order of summation*, is the workhorse behind matrix multiplication: for matrices, $C_{ik} = \sum_j A_{ij}B_{jk}$ (Chapter M5). The product $\prod_{i=1}^{n} a_i = a_1 a_2\cdots a_n$ behaves similarly, and a product of many probabilities underflows a computer's number format, which is why we take logarithms and sum instead (Chapters M2 and M10): $\log \prod_i p_i = \sum_i \log p_i$.

**Indicator and Kronecker delta.** The **indicator function** $\mathbf{1}[\text{condition}]$ equals 1 if the condition is true and 0 otherwise. It turns counting into summing: the number of G or C bases in a sequence $s_1\cdots s_L$ is $\sum_{i=1}^{L}\mathbf{1}[s_i \in \{G, C\}]$, and the GC fraction is that count divided by $L$. The **Kronecker delta** $\delta_{ij}$ is 1 if $i = j$ and 0 otherwise; it is the entries of the identity matrix. A one-hot encoding of a base is a vector $\mathbf{e}_b$ with $(\mathbf{e}_b)_k = \delta_{bk}$.

**Telescoping.** Many sums collapse because adjacent terms cancel:

$$
\sum_{i=1}^{n}\big(a_{i} - a_{i-1}\big) = a_n - a_0 .
$$

This shows that the total change over a time course equals the sum of the step changes, and it is the discrete shadow of the Fundamental Theorem of Calculus (Chapter M3).

**Double-sum hygiene.** When a sum has two indices, write the ranges and keep each dummy index distinct. The classic error is to reuse $i$ for both the outer and inner sum, so that "$\sum_i a_i \sum_i b_i$" silently means something different from what was intended. The correct way to write the product of two sums is $\sum_i a_i \sum_j b_j = \sum_i\sum_j a_i b_j$.

---

## M1.6 Sizes of things: scientific notation, orders of magnitude, and big-$O$

Biology spans many orders of magnitude, and so does computing. A bacterial genome has $\sim 5\times10^{6}$ bases; the human genome $\sim 3\times10^{9}$; a human body has $\sim 3\times10^{13}$ cells; and the number of base pairs of DNA across all of Earth's cells is of order $10^{37}$. You should be able to write and manipulate $a\times 10^{b}$ and estimate *how many orders of magnitude* a quantity is without a calculator. The tool for that is the **Fermi estimate**: break a quantity into factors you can bound, multiply, and quote the result to one significant figure.

!!! example "Worked example M1.2: a Fermi estimate for a screen"
    **Question.** How many single-cell RNA reads does an experiment need so that an average gene in an average cell is seen at least once?

    **Reasoning.** A human cell contains on the order of $10^{5}$ to $10^{6}$ mRNA molecules ($\approx 3\times10^5$ is a common figure) spread over about $10^4$ expressed genes, so an average expressed gene has $\sim 30$ copies. A droplet protocol captures roughly 10% of molecules, giving $\sim 3$ observed molecules per gene per cell, and $\sim 3\times10^{4}$ molecules ($\approx$ UMIs) per cell in total. Gene counts are very uneven (a few genes take a large share), so the *median* gene is seen far less often than the *mean*, and most cells have many genes with a zero count. This is one reason sparsity is a fact of the measurement and not of the biology (Chapter 25).

    **What to notice.** Each step is a rough number you can defend within a factor of 3. The answer is an order of magnitude and a *shape* (heavy-tailed), not a precise value. That is exactly what you need to decide whether an experiment is feasible.

**Asymptotic notation.** To describe how a quantity *grows*, ignoring constants, write:

- $f(n) = O(g(n))$ ("big-O"): for large $n$, $f$ grows no faster than a constant times $g$. Sorting $n$ numbers takes $O(n\log n)$ steps; comparing all pairs of $n$ cells takes $O(n^2)$ steps.
- $f(n) = \Theta(g(n))$: grows at the same rate as $g$ (both an upper and lower bound).
- $f \approx g$: numerically close. $f \sim g$: the ratio tends to 1 as the argument grows (a precise statement in limits); in loose use, "of the order of".

A transformer's self-attention costs $O(L^2)$ in sequence length $L$ (Chapters 2 and 12). That single fact explains why a model that handles 1,000 tokens comfortably struggles with $10^6$: the cost grows by a factor $10^6$, not $10^3$.

!!! warning "Big-$O$ hides constants"
    An $O(n)$ algorithm with a constant of 1,000 loses to an $O(n^2)$ algorithm with a constant of 1 for all $n < 1000$. In practice, constants, memory access patterns and parallelism matter as much as the exponent. Use big-$O$ to see how a method will *scale*, and a timing experiment to see how it *performs*.

---

## M1.7 Counting

Counting underlies probability (Chapter M7) and the size of every search space in this book.

**The multiplication rule.** If a task is a sequence of $k$ independent choices with $n_1, n_2, \dots, n_k$ options, there are $n_1 n_2\cdots n_k$ outcomes. A DNA sequence of length $L$ has $4^L$ possibilities, and a protein of length $L$ has $20^L$. The output above shows how quickly these grow: $4^{30}\approx 10^{18}$, and $20^{100}\approx 10^{130}$, vastly more than the $\sim 10^{80}$ atoms in the observable universe. This is the quantitative content of the statement "protein sequence space cannot be enumerated", and it is why every method in the book must *generalize* from a vanishingly small sample of the space (Chapters 7 and 36).

**Permutations.** The number of ways to order $n$ distinct objects is $n! = n(n-1)\cdots 2\cdot 1$. Ordering 8 genes on a chromosome arm: $8! = 40{,}320$.

**Combinations.** The number of ways to choose $k$ objects from $n$ *without regard to order* is the **binomial coefficient**

$$
\binom{n}{k} \;=\; \frac{n!}{k!\,(n-k)!}.
$$

Choosing 5 mutated sites among 300 positions has $\binom{300}{5} = 19{,}582{,}837{,}560$ possibilities, about $2\times10^{10}$. Why the formula? There are $n!$ orderings of all $n$ objects; ordering the chosen $k$ among themselves and the unchosen $n-k$ among themselves does not change which set was chosen, so divide by $k!\,(n-k)!$. The check in the script confirms the more general fact that the number of length-8 sequences over $\{A,C,G,T\}$ with exactly 3 G or C bases is $\binom{8}{3}\cdot 2^8 = 14{,}336$: choose *which* 3 positions are G/C ($\binom{8}{3}$), then pick one of two bases for each G/C position ($2^3$) and one of two bases for each A/T position ($2^5$), giving $2^8$ in total. That is the *binomial distribution* of Chapter M7 in disguise.

**Stirling's approximation.** Factorials of large numbers overflow every calculator. The approximation

$$
n! \;\approx\; \sqrt{2\pi n}\,\Big(\frac{n}{e}\Big)^{n}
$$

has relative error 1.7% at $n=5$, 0.4% at $n=20$ and 0.08% at $n=100$ (script). Its logarithm, $\ln n! \approx n\ln n - n + \tfrac12\ln(2\pi n)$, appears in the entropy of a multinomial (Chapter 5) and in statistical-mechanics derivations of the Boltzmann distribution (Chapter 22).

### M1.7.1 The birthday problem and barcode collisions

How many people must be in a room before two share a birthday with probability above 50%? Most people guess about 180. The answer is 23, because what grows is the number of *pairs*, which is $\binom{n}{2}\approx n^2/2$, not $n$.

!!! math "Derivation: the birthday problem for $B$ possible values"
    Draw $n$ items independently and uniformly from $B$ equally likely values. The probability that *all are different* is

    $$
    \frac{B}{B}\cdot\frac{B-1}{B}\cdot\frac{B-2}{B}\cdots\frac{B-n+1}{B} \;=\; \prod_{i=0}^{n-1}\Big(1 - \frac{i}{B}\Big).
    $$

    (The first draw is free; the second must avoid 1 value, the third 2 values, and so on.) Using $1 - x \approx e^{-x}$ for small $x$ (a fact justified by the Taylor series of Chapter M3),

    $$
    P(\text{no collision}) \;\approx\; \exp\!\Big(-\sum_{i=0}^{n-1}\frac{i}{B}\Big) \;=\; \exp\!\Big(-\frac{n(n-1)}{2B}\Big),
    $$

    using $\sum_{i=0}^{n-1} i = n(n-1)/2$ (proved above by induction). So $P(\text{collision})\approx 1 - e^{-n(n-1)/2B}$, which reaches $50\%$ when $n(n-1)/2B \approx \ln 2$, that is, $n\approx 1.18\sqrt{B}$.

    **Check.** $B = 365$: $1.18\sqrt{365}\approx 22.5$. The exact value for $n = 23$ is 0.5073 (script output).

!!! bio "Biology for modeling: barcodes, doublets and the square-root law"
    **What is it?** Droplet single-cell methods tag each cell's molecules with a random barcode, and many sequencing and pooled-screen methods tag molecules or cells with random sequences.

    **Information it contains.** A barcode is useful only if it is *unique per cell (or per molecule)*.

    **The square-root law.** With $B = 4^{12}\approx 1.7\times10^7$ possible 12-base barcodes, the chance that *some two* of 5,000 cells share a barcode is already 53%, and a 50% chance of at least one collision is reached at only $\approx 1.18\sqrt{B}\approx 4{,}800$ cells (script output). Doubling the barcode length to 16 bases ($B = 4^{16}\approx 4.3\times10^9$) makes 10,000 cells safe at a 1.2% collision chance. In practice, whitelists of known barcodes and error-correcting designs are used, and collision probability is one of the quantities you compute when you design a pooled experiment.

    **What would change the interpretation.** The formula assumes barcodes are drawn *uniformly*. Real barcode pools are not uniform (synthesis bias, sequencing error, PCR bias), so the true collision rate is higher than the uniform calculation. Non-uniformity always *increases* collision probability.

    **Modeling consequence.** A random-barcode scheme costs the square root of the space. The same square-root law governs hash collisions in computing: with $B$ slots, $n$ keys already produce about $n^2/2B$ colliding pairs, so even a table much larger than $n$ must handle collisions (Chapter M9).

---

## M1.8 Notation habits that prevent errors

A short list to adopt now; the book uses all of them.

1. **State the shape or type of every object** the first time it appears: $\mathbf{X}\in\R^{N\times G}$ is a matrix with $N$ rows (cells) and $G$ columns (genes).
2. **Name the index that runs over each axis** and keep it consistent: $i$ for cells, $g$ for genes, $t$ for time.
3. **Write functions with their domain and codomain** the first time: $f:\R^d\to\R$.
4. **Do not confuse a sample with a population.** $\bar{x}$ is computed from data; $\mu$ is a property of the process. (Chapter M7.)
5. **Mark definitions with $:=$** (or the word "define") so that you can tell a definition from a claim.
6. **When a formula surprises you, test it on $n=1$ or $n=2$** before trusting it.

---

## M1.9 Worked examples

!!! example "Worked example M1.3: How many mutants are in a two-site library?"
    **Situation.** A deep mutational scan changes each of 2 chosen positions of a protein to any of the 20 amino acids (including keeping the wild-type residue).

    **Question.** (a) How many distinct variants exist? (b) How many of them differ from wild-type at *exactly one* of the two positions? (c) At *both*?

    **Reasoning.** (a) By the multiplication rule: $20\times20 = 400$ variants including wild-type. (b) Choose which position changes (2 ways) and choose a non-wild-type residue (19 ways): $2\times19 = 38$ single mutants. (c) Both positions change: $19\times19 = 361$ double mutants. Check: $1 + 38 + 361 = 400$. ✓ (wild-type, singles, doubles partition the set).

    **Generalize.** With $L$ positions and 20 residues, the number of variants with exactly $k$ substitutions is $\binom{L}{k}19^{k}$; the total is $20^L$. For $L = 10$ and $k=2$: $\binom{10}{2}19^2 = 45\cdot361 = 16{,}245$, a tiny fraction of the $20^{10}\approx10^{13}$ variants. *Experiments can cover all singles and some doubles, never all variants*, which is why Chapter 36 treats the design of which variants to test as a research problem in itself.

!!! example "Worked example M1.4: Is the genetic code's degeneracy a reason to distrust a protein model?"
    **Situation.** A colleague says: "The protein language model ignores codon choice, therefore it cannot be useful for predicting expression."

    **Question.** Evaluate the claim using the vocabulary of this chapter.

    **Reasoning.** The map from DNA to protein is not injective, so the protein sequence discards the codon choice. It follows that a model *whose input is only the protein sequence* has *no access* to codon effects. That is a statement about the **information in the input** (the measurement-and-representation gap of Chapter 1), not about the model's quality. Whether this matters for a *given* expression prediction depends on how much of the variation in expression is driven by synonymous codon choice for that gene, an empirical question. The colleague's conclusion is the converse-style leap: "ignores codons $\Rightarrow$ useless for expression" does not follow. A protein model might still predict expression-related properties through amino-acid composition, signal peptides, or stability, which depend on the protein.

    **Lesson.** Notice *what information the input contains*, then ask what part of the target can in principle be explained by it. The Expert Chain does this at link L4 ("why might a single model work?") and L5 ("failure modes").

---

## M1.10 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"The two gene lists overlap, so the pathway is shared\""
    **The observation.** Two experiments each produce a list of differentially expressed genes: 1,200 genes in experiment A, 1,000 in B, and 300 genes appear in both. A figure shows a Venn diagram, and the text says the large overlap shows a shared mechanism.

    **Tempting conclusion.** "300 shared genes is a strong overlap, so the two conditions share a pathway."

    **Decompose.** Use inclusion–exclusion: the union has $1200 + 1000 - 300 = 1{,}900$ genes. Now ask what overlap *chance* alone would give. If the genome has $20{,}000$ expressed genes and the lists were independent random subsets, the expected overlap is $1200\times1000/20000 = 60$ genes. So 300 is five times what chance predicts. That is real enrichment (the quantity to test is how surprising 300 is under chance, the topic of Chapter M8), but note how much hinges on the *background size*: if the two experiments only tested 3,000 genes in common, then chance predicts $1200\times1000/3000 = 400$, and the observed overlap of 300 would be *below* chance.

    **Hidden assumptions.** (i) The background set (which genes could have been called at all) is stated and shared. (ii) The lists are not both driven by a common technical factor, such as gene length or expression level, which makes the same genes easy to call in any experiment. (iii) "Shared genes" implies "shared pathway" only if the genes belong to a common mechanism, not merely to a common class of highly expressed genes.

    **Discriminating experiments.** Recompute the overlap with a *matched null*: random gene sets with the same expression-level distribution; or test whether the overlapping genes are enriched for a specific pathway beyond what expression level alone predicts.

    **What this teaches.** A count means nothing until you say what you would *expect to count under a null*. Counting correctly (inclusion–exclusion, background size) is the first step of research reasoning; the null is the second.

---

## M1.11 Connections

- **Forward:** the sets and functions of this chapter become random variables and distributions (M7); sums and products become matrix multiplication (M5) and log-likelihoods (M8); induction and recursion return in dynamic programming (M9); counting returns as the size of sequence spaces (Chapters 27, 28, 36); big-$O$ returns in the cost tables of Chapters 2 and 12.
- **Backward:** none. This is where the book's mathematics begins.

!!! takeaways "Key takeaways"
    1. To read a formula: name each object and its type, find the quantifiers, find the verb, test the smallest case, and say it in words.
    2. A **function** assigns exactly one output to each input. It is **injective** if it loses no information, **surjective** if it reaches everything. The genetic code is onto but not injective, so a protein does not determine its DNA (18,432 DNA sequences encode one ten-residue peptide).
    3. $P \Rightarrow Q$ is equivalent to its **contrapositive**, not to its **converse**. *Necessary* and *sufficient* are the two directions of an implication; experimental biology is largely a search for which direction holds.
    4. $\forall$ and $\exists$ swap under negation, and changing their order changes the claim. Insert the missing quantifier whenever a general claim is stated loosely.
    5. A computer check can refute a conjecture but never prove it; a proof covers every case of an abstract object. **Induction** needs a base case and an inductive step.
    6. $\sum$, $\prod$, indicators and the Kronecker delta turn counting into algebra; swap the order of summation freely for finite sums, and use $\log\prod = \sum\log$ to avoid underflow.
    7. Sequence spaces explode ($4^{30}\approx10^{18}$, $20^{100}\approx10^{130}$). $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ counts unordered choices; Stirling's approximation is accurate to 0.08% at $n=100$.
    8. The **birthday problem** shows collisions arrive at $\approx1.18\sqrt{B}$ draws, not at $B$. Random barcodes cost the square root of their space, and non-uniform barcodes collide more often.

---

## Further reading

- Velleman, D. J. *How to Prove It: A Structured Approach*. Cambridge University Press. The standard gentle introduction to logic and proof.
- Hammack, R. *Book of Proof* (free online). A short and clear alternative.
- Knuth, D. E., Graham, R. L. & Patashnik, O. *Concrete Mathematics*. Addison-Wesley. Sums, binomial coefficients and asymptotics, with great problems.
- Weinstein, L. & Adam, J. A. *Guesstimation*. Princeton University Press. The art of the Fermi estimate.
- Milo, R. & Phillips, R. *Cell Biology by the Numbers* (free online). The orders of magnitude of cell biology, in the Fermi-estimate spirit.
