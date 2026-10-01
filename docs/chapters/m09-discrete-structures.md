# Chapter M9. Discrete Structures: Graphs, Trees, Algorithms, and Complexity

!!! abstract "Chapter at a glance"
    **Motivation.** Much of biology is not a smooth function of continuous variables: it is a *structure* of discrete things and relationships. Proteins interact in networks; species are related by trees; regulators point at targets; a genome is assembled by finding a path through overlaps; sequences are aligned by choosing among exponentially many alignments. Computer science supplies the vocabulary (graphs, trees, recursion, complexity) and the tools (breadth-first search, dynamic programming, hashing). This chapter introduces them with the biological examples they were created for, and gives you the single most important habit of a computational scientist: *count the size of the search space, then ask how an algorithm avoids visiting it all*.

    **Prerequisites.** Chapters M1 (sets, counting, induction, big-$O$), M5 (matrices), M6 (eigenvalues, for the graph Laplacian). Some programming (Chapter M10 helps).

    **You will be able to:** (1) represent a network as a graph and an adjacency matrix and compute degrees, paths, components and walks; (2) use the graph Laplacian to count connected components; (3) describe trees and DAGs and count labelled trees; (4) build a tree from a distance matrix (UPGMA) and read Newick notation; (5) state running times with big-$O$ and explain why naive recursion can be exponential while dynamic programming is polynomial; (6) compute an edit distance; (7) assemble a genome from $k$-mers as a path in a de Bruijn graph and say when repeats make the answer ambiguous; (8) estimate the number of distinct $k$-mers in a genome and explain hashing.

---

## M9.1 Graphs

A **graph** $G=(V,E)$ consists of a set $V$ of **nodes** (vertices) and a set $E$ of **edges**, each joining two nodes. Edges may be **undirected** (protein A binds protein B) or **directed** (transcription factor A regulates gene B), and may carry **weights** (interaction strength, evolutionary distance). Almost every relational dataset in biology is a graph:

| Biological object | Nodes | Edges |
|---|---|---|
| Protein interaction network | proteins | physical binding |
| Gene regulatory network | genes / transcription factors | A regulates B (directed, signed) |
| Metabolic network | metabolites and reactions | substrate/product of |
| Cell lineage or developmental tree | cell states | differentiation steps |
| Phylogeny | species / sequences | descent, with branch length |
| Molecule | atoms | bonds (Chapters 16, 24, 37) |
| Protein structure | residues | spatial contact (Chapters 16, 35) |
| Knowledge graph | genes, diseases, drugs | known relations |

**Vocabulary.** The **degree** of a node is the number of edges touching it; the *sum of all degrees equals twice the number of edges* (each edge has two ends). In the script's toy network of 12 proteins with 12 edges the degrees sum to $24$. A **path** is a sequence of edges leading from one node to another; its length is the number of edges. A graph is **connected** if a path joins every pair of nodes; otherwise it falls into **connected components**. A **cycle** is a closed path; a **triangle** is a cycle of length 3.

**The adjacency matrix.** Number the nodes $1,\dots,n$. The **adjacency matrix** $\mathbf{A}\in\{0,1\}^{n\times n}$ has $A_{ij}=1$ if there is an edge between $i$ and $j$ (symmetric for undirected graphs). It turns graph questions into linear algebra (Chapters M5 and M6):

- **Degrees** are row sums: $\mathbf{A}\mathbf{1}$.
- **Walks.** $(\mathbf{A}^{k})_{ij}$ counts the walks of length $k$ from $i$ to $j$. (Proof by induction: a walk of length $k+1$ is a walk of length $k$ followed by one edge, $\sum_l(\mathbf{A}^k)_{il}A_{lj}$.) In the script, there is exactly one walk of length 2 from $A$ to $D$ (via $C$).
- **Triangles:** $\operatorname{tr}(\mathbf{A}^3)/6$ counts them (each triangle is counted 6 times: 3 start nodes $\times$ 2 directions). The script finds 3 triangles in the network.

**The Laplacian.** With $\mathbf{D}=\operatorname{diag}(\text{degrees})$, the **graph Laplacian** is $\mathbf{L}=\mathbf{D}-\mathbf{A}$. It is symmetric and positive semi-definite, and for any vector $\mathbf{x}$,

$$
\mathbf{x}^\top\mathbf{L}\mathbf{x}=\sum_{(i,j)\in E}(x_i-x_j)^2,
$$

a measure of how much $\mathbf{x}$ varies *across edges*. Consequently $\mathbf{L}\mathbf{x}=\mathbf{0}$ exactly when $\mathbf{x}$ is constant on each connected component, so **the number of zero eigenvalues of $\mathbf{L}$ equals the number of connected components**. The script's Laplacian has three zero eigenvalues (and then $0.438,1.0,\dots$), matching the three components. The eigenvectors of the smallest nonzero eigenvalues give the best *smooth* functions on the graph and are the basis of **spectral clustering** and of the spectral view of graph neural networks (Chapter 16).

**Searching a graph.** **Breadth-first search (BFS)** explores a graph in rings: first the start node, then its neighbours, then theirs, using a queue. It finds *shortest paths* in unweighted graphs and the connected component of a node. In the script, BFS from $A$ reaches $B,C$ at distance 1, $D$ at 2, $E,F$ at 3, and never reaches the other six nodes: they lie in different components. **Depth-first search (DFS)** follows one path as far as it goes before backtracking; it is natural for cycle detection and for enumerating structures. Both run in $O(|V|+|E|)$ time with adjacency lists.

!!! bio "Biology for modeling: what a network claim is a claim about"
    **What is it?** A protein-interaction network assembled from many experiments; "degree" = the number of known partners.

    **Information it contains.** Which pairs were reported to interact in at least one assay.

    **What it discards.** Condition and cell type (a pair that interacts in one tissue and never meets in another is one edge), strength and kinetics, and the *absence* of interactions that were not tested. Edges are measurements with false positives and false negatives, and the *coverage* is uneven: well-studied proteins have many more reported partners simply because they have been studied more (§M9.8).

    **Modelling consequence.** Graph statistics (degree, shortest path, clustering) describe the *experimental record* as much as the biology. A model that predicts "essential genes are hubs" may be partly learning "popular genes are hubs".

---

## M9.2 Trees and DAGs

A **tree** is a connected graph with no cycles. A tree on $n$ nodes has exactly $n-1$ edges. Trees represent hierarchy and descent: phylogenies, taxonomies, lineage trees, and the dendrograms of hierarchical clustering. A **rooted** tree has a distinguished root (the common ancestor), and its **leaves** are the nodes of degree 1 (the observed species or cells).

**Counting trees: the explosion that shapes phylogenetics.** An **unrooted binary tree** with $n$ labelled leaves has $2n-3$ edges. The number of *distinct* such trees is

$$
(2n-5)!!\;=\;1\cdot3\cdot5\cdots(2n-5),
$$

derived by induction: a tree on $n-1$ leaves has $2n-5$ edges, and the new leaf $n$ can be attached in the middle of any one of them, giving $(2n-5)$ ways. Values from the script: $3$ trees for $n=4$, $15$ for $n=5$, $10{,}395$ for $n=8$ (the script prints $1.04\times10^{4}$), $2.0\times10^{6}$ for $n=10$, $2.2\times10^{20}$ for $n=20$ and $2.8\times10^{74}$ for $n=50$. No computer can enumerate the trees of even 20 species, so **finding the best tree under a likelihood criterion is a search problem over a space that is astronomically large**: it is NP-hard, and real programs use heuristics (greedy additions, local rearrangements, Markov chain Monte Carlo; Chapter 42). This is the first example of a pattern that returns in protein design (Chapter 36), alignment (Chapter 27) and neural architecture search: *exact optimization is hopeless, so the algorithm design is the science*.

**UPGMA: a tree from a distance matrix.** Given pairwise distances between species, **UPGMA** (unweighted pair group method with arithmetic mean) repeatedly merges the two closest clusters, placing the new node at half their distance and defining the distance from the new cluster to the others as the size-weighted average. It assumes a **molecular clock** (equal rates along all lineages), which makes the tree *ultrametric*. In the script, a five-species distance matrix yields

```text
(macaque:10,(orang:7,(gorilla:3,(human:1,chimp:1):2):4):3);
```

in **Newick notation**, the standard text format for trees: nested parentheses group sister taxa, and each name is followed by its branch length. Reading it: human and chimp merge first (distance 2, each 1 from their ancestor), gorilla joins 3 units above the leaf level, then orang at 7, and the macaque (the outgroup) last at 10. If the molecular clock is violated (a fast-evolving lineage), UPGMA can return the wrong topology; neighbour-joining and likelihood methods (Chapter 42) relax the assumption.

**Directed acyclic graphs (DAGs).** A **DAG** is a directed graph with no directed cycle: you can never return to a node by following arrows. It admits a **topological order** (list the nodes so every arrow points forward). DAGs encode "A influences B" without feedback: **causal graphs** (Chapter 44), the dependency structure of a computation (a neural network's computation graph, which is why backpropagation can be run in reverse topological order; Chapter 9), and ancestry in a pedigree. Cyclic regulation, such as feedback loops, is *not* a DAG, a fact that matters when learning regulatory networks from data.

---

## M9.3 Algorithms and complexity

An **algorithm** is a precise procedure that solves a class of problems. We judge algorithms on **correctness** (always gives the right answer?) and **cost**: running time and memory, as functions of the input size $n$, in the big-$O$ notation of Chapter M1. The usual vocabulary:

| Growth | Name | $n=10^{3}$ | $n=10^{6}$ | Typical example |
|---|---|---|---|---|
| $O(1)$ | constant | 1 | 1 | look up a key in a hash table |
| $O(\log n)$ | logarithmic | 10 | 20 | binary search in sorted data |
| $O(n)$ | linear | $10^3$ | $10^6$ | scan a genome once |
| $O(n\log n)$ | quasi-linear | $10^4$ | $2\times10^{7}$ | sorting |
| $O(n^2)$ | quadratic | $10^6$ | $10^{12}$ | compare all pairs; align two sequences of length $n$ |
| $O(n^3)$ | cubic | $10^9$ | $10^{18}$ | naive matrix products; Gaussian elimination |
| $O(2^n)$ | exponential | $10^{301}$ | impossible | enumerate all subsets |

A rough computing budget: a single modern core does about $10^{9}$ simple operations per second. An $O(n^2)$ algorithm is comfortable up to $n\sim10^5$ and out of reach at $n=10^{9}$ (a whole genome), which is why alignment of whole genomes uses indexes and heuristics (Chapter 27). An exponential algorithm is useless beyond $n\approx40$ no matter how fast the machine.

**Hashing.** A **hash table** maps keys to values via a function that computes an array index from the key, giving $O(1)$ average-time lookup and insertion. It is the data structure behind Python's `dict` and `set`. To detect duplicates among $n$ items, comparing all pairs costs $n(n-1)/2$ operations, while inserting each item into a hash set costs $n$. The script's table: for $n=10^{3}$ pairwise $=499{,}500$ vs hashing $=1{,}000$ ($500\times$ cheaper); for $n=10^{5}$, $5\times10^{9}$ vs $10^{5}$ ($50{,}000\times$); for $n=10^{7}$ the ratio is $5\times10^{6}$. By Chapter M1's birthday-problem arithmetic, hash tables have collisions (two keys mapping to the same index; with $B$ slots the expected number of colliding pairs is about $n^2/2B$, so even a mostly empty table has some), which are handled by chaining or probing.

**$P$ versus $NP$ in one paragraph.** Problems with algorithms of polynomial cost are in $P$. Many important problems (the best phylogenetic tree, the shortest route visiting all nodes, the minimum-energy protein conformation on a lattice) are **NP-hard**: no polynomial algorithm is known, and it is widely believed none exists. This does *not* mean they cannot be solved in practice: heuristics and the special structure of real instances often work well; it means no guarantee can be had for the worst case.

---

## M9.4 Recursion and dynamic programming

A **recursive** algorithm solves a problem by solving smaller instances of the same problem (as in the proof by induction of Chapter M1). The danger is that the same subproblem may be solved many times. **Dynamic programming (DP)** removes this repetition by solving each distinct subproblem once and storing the answer in a table. The recipe: (1) define the subproblems; (2) write a **recurrence** expressing a subproblem's answer in terms of smaller ones; (3) fill a table in an order that respects the dependencies.

**The edit distance.** How different are two strings? The **edit distance** (Levenshtein) is the minimum number of single-character insertions, deletions and substitutions needed to turn one into the other. Let $D[i,j]$ be the edit distance between the first $i$ characters of string $a$ and the first $j$ of string $b$. Then

$$
D[i,j]=\min\begin{cases}
D[i-1,j]+1&\text{(delete }a_i)\\
D[i,j-1]+1&\text{(insert }b_j)\\
D[i-1,j-1]+\mathbf{1}[a_i\ne b_j]&\text{(match or substitute)}
\end{cases}
$$

with $D[i,0]=i$ and $D[0,j]=j$. The answer is $D[|a|,|b|]$. The table has $(|a|+1)(|b|+1)$ cells, each filled in $O(1)$ time, so the cost is $O(|a||b|)$. In the script, $\text{GATTACA}\to\text{GCATGCU}$ has edit distance $4$.

**Why DP matters: counting calls.** Computing the same recurrence by naive recursion without storing results makes an exponential number of calls. For random DNA strings of equal length the script counts recursive calls against table cells:

| Length | recursive calls | table cells |
|---|---|---|
| 4 | 481 | 25 |
| 6 | 13,483 | 49 |
| 8 | 398,593 | 81 |
| 9 | 2,193,844 | 100 |

The recursive count grows by a factor of about $5$ each time the length grows by 1; the table grows quadratically. At length 100, extrapolating the trend, the recursion would need more than $10^{60}$ calls, while the table needs 10,201 cells. DP *is* "recursion plus memory".

**Where this goes.** Sequence alignment is edit distance with biologically meaningful scores: the **Needleman–Wunsch** (global) and **Smith–Waterman** (local) algorithms are exactly this table with a substitution matrix and gap penalties in place of 0/1 costs (Chapter 27). The **Viterbi** and **forward** algorithms for hidden Markov models (gene finders, profile HMMs for protein families; Chapters 27 and 29) are dynamic programs over the same kind of table; RNA secondary-structure prediction uses a DP over nested intervals (Chapter 33); and the gradient of a recurrent network over time (backpropagation through time) is computed by a backward sweep with the same table-filling structure (Chapter 11).

```python
--8<-- "code/m09_discrete.py"
```

Output (seed 0):

```text
degrees: {'A': 2, 'B': 2, 'C': 3, 'D': 3, 'E': 2, 'F': 2, 'G': 2, 'H': 2, 'I': 3, 'J': 1, 'K': 1, 'L': 1} | sum of degrees = 24 = 2 x number of edges = 24
BFS from A (shortest path lengths): {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 3, 'F': 3}  -> nodes unreachable from A: ['G', 'H', 'I', 'J', 'K', 'L']
connected components: [['A', 'B', 'C', 'D', 'E', 'F'], ['G', 'H', 'I', 'J'], ['K', 'L']]
walks of length 2 from A to D: 1 (via C);  triangles = trace(A^3)/6 = 3  (ABC, DEF, GHI)
graph Laplacian eigenvalues (smallest 5): [-0.    -0.     0.     0.438  1.   ] -> 3 zero eigenvalues = number of connected components

number of distinct unrooted binary trees with n labelled leaves, (2n-5)!!:
  n =  4: 3.000e+00   (edges in each tree: 5)
  n =  5: 1.500e+01   (edges in each tree: 7)
  n =  8: 1.040e+04   (edges in each tree: 13)
  n = 10: 2.027e+06   (edges in each tree: 17)
  n = 20: 2.216e+20   (edges in each tree: 37)
  n = 50: 2.838e+74   (edges in each tree: 97)
UPGMA tree (Newick, branch lengths in distance units): (macaque:10,(orang:7,(gorilla:3,(human:1,chimp:1):2):4):3);

edit distance(GATTACA, GCATGCU) = 4
naive recursion vs dynamic programming (equal-length random DNA strings):
  length 4: distance 4 (DP agrees: True);  recursive calls       481   table cells 25
  length 6: distance 3 (DP agrees: True);  recursive calls    13,483   table cells 49
  length 8: distance 5 (DP agrees: True);  recursive calls   398,593   table cells 81
  length 9: distance 8 (DP agrees: True);  recursive calls 2,193,844   table cells 100

genome of length 19: ATGCGATGCCGTAATGACG
  k = 3: 10 nodes, 17 edges (k-mers), distinct reconstructions with the right start:  20; original among them: True
  k = 4: 13 nodes, 16 edges (k-mers), distinct reconstructions with the right start:   2; original among them: True
  k = 5: 14 nodes, 15 edges (k-mers), distinct reconstructions with the right start:   1; original among them: True
  k = 6: 14 nodes, 14 edges (k-mers), distinct reconstructions with the right start:   1; original among them: True
  k = 8: 12 nodes, 12 edges (k-mers), distinct reconstructions with the right start:   1; original among them: True

random genome of length 100,000: 99,993 8-mers occurrences, 51,191 distinct (of 65,536 possible); expected distinct B(1 - e^(-n/B)) = 51,285
  most common 8-mers appear 9 times (the average is 1.53); fraction of possible 8-mers never seen: 0.219 (theory e^(-n/B) = 0.217)
duplicate detection among n items: all-pairs comparisons n(n-1)/2 vs hashing (n insertions):
  n =      1,000: pairwise            499,500   hashing        1,000   ratio 500x
  n =    100,000: pairwise      4,999,950,000   hashing      100,000   ratio 50,000x
  n = 10,000,000: pairwise 49,999,995,000,000   hashing   10,000,000   ratio 5,000,000x
```

---

## M9.5 Strings, $k$-mers, and genome assembly as a path problem

A **$k$-mer** is a substring of length $k$. A sequence of length $L$ has $L-k+1$ $k$-mer occurrences, and there are $4^{k}$ possible DNA $k$-mers. $k$-mers are the vocabulary of genomics: they index genomes for fast search, define the "tokens" of DNA language models (Chapters 28, 32), and are the unit of assembly.

**How many distinct $k$-mers?** If $n$ occurrences are drawn uniformly from $B=4^{k}$ possibilities, the expected number of *distinct* $k$-mers is $B(1-e^{-n/B})$ (each $k$-mer is missed with probability $e^{-n/B}$, the Poisson zero probability of Chapter M7). The script checks: a random genome of $10^{5}$ bases has $99{,}993$ 8-mer occurrences among $B=65{,}536$ possibilities; the observed number of distinct 8-mers is $51{,}191$ against the predicted $51{,}285$, and $21.9\%$ of the possible 8-mers never occur (theory $e^{-n/B}=21.7\%$). The most frequent 8-mer appears 9 times even though the average is 1.5: random counts spread. Real genomes depart sharply from this uniform null (repeats, CpG depletion, codon bias), which is exactly what makes $k$-mer statistics informative.

**Genome assembly.** A sequencer reads short fragments; the assembler must reconstruct the genome. A natural formulation (de Bruijn, 1946; applied to assembly in the 2000s) builds a graph whose **nodes are $(k-1)$-mers** and whose **edges are the observed $k$-mers**: the $k$-mer $x_1x_2\cdots x_k$ is an edge from node $x_1\cdots x_{k-1}$ to node $x_2\cdots x_k$. Reading the genome from left to right traces a path that uses every edge exactly once, an **Eulerian path**. (Euler's 1736 solution of the Königsberg bridge problem gives a simple criterion for when such a path exists: the graph is connected and at most two nodes have unbalanced in- and out-degrees. It is *easy*; the superficially similar problem of finding a path that visits every *node* once, a Hamiltonian path, is NP-hard. Reformulating assembly as an Eulerian-path problem was the idea that made large assemblies computationally feasible.)

**Repeats make it ambiguous.** If the genome contains a repeated segment at least $k-1$ bases long, several distinct Eulerian paths may exist, and the $k$-mers alone cannot tell which reconstruction is right. In the script, the genome `ATGCGATGCCGTAATGACG` (19 bases, with the repeats `ATG` and `GATG`) gives:

| $k$ | nodes | edges | distinct reconstructions | original recovered uniquely? |
|---|---|---|---|---|
| 3 | 10 | 17 | **20** | no |
| 4 | 13 | 16 | **2** | no |
| 5 | 14 | 15 | 1 | yes |
| 6 | 14 | 14 | 1 | yes |
| 8 | 12 | 12 | 1 | yes |

With $k=3$ there are twenty different genomes consistent with the set of 3-mers; with $k=4$, two; from $k=5$ on, the repeated segments no longer permit an alternative ordering and the answer is unique. **The trade-off:** a larger $k$ resolves more repeats, but needs reads long enough to contain the $k$-mers and *each sequencing error corrupts $k$ different $k$-mers*, so noisy data favour smaller $k$. Real assemblers choose $k$ (or several) based on read length and error rate, and genomes with long repeats (human centromeres, segmental duplications) remain hard: they were closed only with long-read technologies (Chapter 28).

!!! example "Worked example M9.1: Choosing $k$ for assembly"
    **Situation.** You assemble a bacterial genome from 100-base reads with a per-base error rate of 1%. The genome contains a repeat of length 30.

    **Reasoning.** To be sure of resolving the repeat, the $k$-mers must extend past it: $k-1>30$, i.e. $k\ge32$ is sufficient. Reads of length 100 contain $100-k+1=69$ $k$-mers for $k=32$. The probability that a given $k$-mer is free of sequencing error is $0.99^{32}=0.72$; for $k=64$ it is $0.99^{64}=0.53$. So a larger $k$ resolves the repeat but discards more of the data (28% of 32-mers, 47% of 64-mers contain an error). With coverage of $50\times$ the loss is tolerable, because every genomic position is seen by many overlapping reads, and error-containing $k$-mers (which occur once or twice) can be filtered by count. At $5\times$ coverage the same $k$ would leave many positions with no error-free $k$-mer and break the assembly into pieces.

    **Lesson.** The choice of $k$ is a trade-off between *resolving power* (repeats) and *noise tolerance* (errors): the same structure as the bias–variance trade-off of Chapter M8. The solution depends on coverage, which is why sequencing budgets and algorithm parameters must be designed together.

---

## M9.6 Combinatorial explosion and what to do about it

The common thread of this chapter: the *space of candidate answers* is huge (trees, alignments, assemblies, sequences) and brute force fails. The standard responses, which you will see throughout the book:

1. **Exploit structure to get a polynomial algorithm.** Dynamic programming over alignments, Eulerian paths for assembly, the Viterbi algorithm for HMMs.
2. **Greedy or local search** (heuristics). Neighbour-joining and tree rearrangements; progressive multiple sequence alignment. No guarantee, but fast and often good.
3. **Randomized search** (Markov chain Monte Carlo, simulated annealing, genetic algorithms). Sample the space in proportion to quality (Chapters M6, M7, 34, 36, 42).
4. **Relaxation.** Replace the discrete problem with a continuous one that can be optimized by gradient descent, then round: the standard approach in modern deep learning for discrete objects (Chapters 15, 36).
5. **Learn the heuristic.** Train a model to guess good answers (a neural network proposing protein structures or sequences), then verify them (Chapters 35 and 36).

---

## M9.7 Worked example

!!! example "Worked example M9.2: A network measure and a null model"
    **Situation.** In the 12-protein network of the script, node $C$ has degree 3 and lies on the shortest path between $A$ and the cluster $\{D,E,F\}$. A colleague proposes that $C$ is "a hub that coordinates the two modules".

    **Reasoning.** Removing $C$ disconnects the graph $\{A,B\}\mid\{D,E,F\}$ (it is a **cut vertex**, or articulation point), so $C$ is structurally important *in this graph*. But degree 3 is barely above the average degree of 2 in this network; the colleague's word "hub" implies degree far above typical. To say whether the structure is *surprising*, generate a **null model**: random graphs with the same number of nodes and the same degree sequence, and ask how often a node of this degree is a cut vertex. If the answer is "frequently", then the observation tells you about degree sequences and nothing about coordination.

    **What to check.** (a) Are the edges themselves reliable (which assay)? (b) Is $C$'s importance due to an artefact, such as being a bait protein used in many experiments? (c) Do independent data (knock-out phenotype, co-expression) agree that $C$ is central?

    **Lesson.** Every network statistic needs a null model; the Expert Chain's L2/L5 (existing approaches and failure modes) applies as much to a graph measure as to a neural network.

---

## M9.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: \"Hub proteins are more likely to be essential\""
    **The observation.** Across a published protein-interaction network, proteins with many partners are far more often essential genes. A model that predicts essentiality from degree achieves strong performance.

    **Tempting conclusion.** "Network centrality causes essentiality: highly connected proteins are the keystones of the cell."

    **Decompose.** Three non-causal explanations produce the same correlation. (i) **Study bias:** essential proteins are scientifically important, so they are studied more, so more of their interactions are reported, so they have higher degree. The network measures *attention* as well as biology. (ii) **Expression and abundance:** abundant proteins are easier to detect in assays and, being involved in core processes, are essential more often. (iii) **Conserved, ancient proteins** both interact with many partners and are essential, with both caused by their evolutionary age. A genuinely causal claim (removing a hub is more harmful than removing a peripheral protein) needs an experiment or a causal model.

    **Hidden assumptions.** (i) The network was assembled without regard to the property being predicted. (ii) The edges are equally reliable across proteins. (iii) The correlation will hold on a held-out set of proteins and species.

    **Discriminating experiments.** Compare high- and low-degree proteins *matched on* publication count and expression level; use a network built from a single, systematic assay (for example, a uniform yeast two-hybrid screen) rather than literature curation; test whether the predictor works across species.

    **What this teaches.** A graph is a dataset with a collection process. Before reading biology into its structure, ask how the edges were observed.

---

## M9.9 Connections

- **Forward:** graph neural networks use adjacency matrices and Laplacians (Chapter 16); alignment and HMMs are dynamic programs (Chapters 27, 29); assembly and $k$-mer tokenization appear in Chapter 28 and in DNA language models (Chapter 32); phylogenetic inference returns in Chapter 42; causal DAGs in Chapter 44; computation graphs and backpropagation in Chapter 9; network biology and regulatory graphs in Chapter 19; molecular graphs in Chapters 24 and 37.
- **Backward:** counting and induction (M1); matrices, eigenvalues and the Laplacian (M5, M6); Poisson probabilities for $k$-mer counts (M7).

!!! takeaways "Key takeaways"
    1. A **graph** has nodes and edges (directed, weighted); the **adjacency matrix** makes it linear algebra: $(\mathbf{A}^k)_{ij}$ counts walks, $\operatorname{tr}(\mathbf{A}^3)/6$ counts triangles, and the Laplacian $\mathbf{L}=\mathbf{D}-\mathbf{A}$ has one zero eigenvalue per connected component.
    2. **BFS** finds shortest paths and components in $O(|V|+|E|)$ time. Graph statistics describe how the graph was *measured* as much as the biology.
    3. A **tree** on $n$ nodes has $n-1$ edges; the number of unrooted binary trees on $n$ leaves is $(2n-5)!!$ ($2.8\times10^{74}$ for $n=50$), so tree search is a heuristic business. **UPGMA** builds a tree from distances and assumes a molecular clock; **Newick** is the text format.
    4. A **DAG** has no directed cycles and admits a topological order: causal models, computation graphs, pedigrees.
    5. **Complexity** classes (constant to exponential) tell you whether an algorithm scales: $n^2$ is fine for $10^5$ items, hopeless for $10^9$. **Hashing** gives $O(1)$ lookups and turns $n^2/2$ pairwise comparisons into $n$ insertions.
    6. **Dynamic programming** is recursion plus memory: edit distance fills an $O(nm)$ table, whereas naive recursion needs 2.2 million calls where the table needs 100 cells. Alignment, HMMs and RNA folding are dynamic programs.
    7. **Assembly** is an Eulerian path in a de Bruijn graph; repeats of length $\ge k-1$ can make the reconstruction ambiguous (20 solutions at $k=3$, 2 at $k=4$, 1 at $k\ge5$ in the toy genome). Larger $k$ resolves repeats but loses data to errors.
    8. The number of distinct $k$-mers in a random genome follows $B(1-e^{-n/B})$. When the answer space is huge, use structure, heuristics, sampling, relaxation, or a learned proposal.

---

## Further reading

- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press. The standard reference for graphs, DP and complexity.
- Jones, N. C. & Pevzner, P. A. *An Introduction to Bioinformatics Algorithms*. MIT Press. The algorithms of this chapter with biological problems throughout.
- Durbin, R., Eddy, S., Krogh, A. & Mitchison, G. *Biological Sequence Analysis*. Cambridge University Press. DP, HMMs and phylogeny.
- Compeau, P. E. C., Pevzner, P. A. & Tesler, G. (2011). How to apply de Bruijn graphs to genome assembly. *Nature Biotechnology* 29, 987–991.
- Felsenstein, J. *Inferring Phylogenies*. Sinauer. Tree counting, UPGMA, and likelihood methods.
- Newman, M. E. J. *Networks*. Oxford University Press. Network structure, null models and spectral methods.
