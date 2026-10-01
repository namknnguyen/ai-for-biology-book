# Dependency Graph

How the major concepts build on one another. An arrow $A \to B$ means *B uses ideas from A in an essential way* (not merely that A is mentioned). Dashed arrows mark prerequisites that are taught **just in time**, inside the chapter that first needs them.

## 1. The big picture: parts

```mermaid
flowchart TD
  P0["Part 0<br/>Mathematical background<br/>logic · functions · calculus · linear algebra · probability · statistics · discrete maths · Python"]
  P1["Part I<br/>Orientation<br/>Expert Chain · Four Gaps · Measurement view"]
  P2["Part II<br/>Foundations<br/>linear algebra · optimization · probability · information · implementation"]
  P3["Part III<br/>Core ML<br/>generalization · latent variables"]
  P4["Part IV<br/>Deep learning<br/>backprop · attention · self-supervision · generative · diffusion · scaling"]
  P5["Part V<br/>Biology for modeling<br/>cell · genome · evolution · regulation · protein · molecule · single-cell · genetics"]
  P6["Part VI<br/>Computational biology<br/>alignment · representations · classical models"]
  P7["Part VII<br/>Biological foundation models<br/>DNA · RNA · protein · structure · design · drugs · cells · virtual cell · G→P · evolution"]
  P8["Part VIII<br/>Research methodology<br/>benchmarks · causality · shift · experiments · scaling · mechanism"]
  P9["Part IX<br/>The frontier<br/>paradigms · open-problem atlas · AI scientists"]
  P10["Part X<br/>Independent research<br/>ten attacks · evaluating ideas · case studies · programs"]

  P0 --> P1
  P0 --> P2
  P1 --> P2 --> P3 --> P4
  P1 --> P5
  P2 -.-> P5
  P4 --> P6
  P5 --> P6
  P4 --> P7
  P6 --> P7
  P5 --> P7
  P7 --> P8
  P3 --> P8
  P8 --> P9
  P7 --> P9
  P9 --> P10
  P8 --> P10
  P1 -. "Expert Chain used throughout" .-> P10
```

## 2. Chapter-level dependencies

The next four diagrams zoom in. Colors mark the Part: <span style="color:#00897b">**teal**</span> mathematical foundations and core ML, <span style="color:#3949ab">**indigo**</span> deep learning, <span style="color:#43a047">**green**</span> biology and computational biology, <span style="color:#e65100">**orange**</span> biological foundation models, <span style="color:#8e24aa">**purple**</span> methodology, frontier, and independent research.

### 2a. From foundations to deep learning (Chapters 1–18)

```mermaid
flowchart TD
  C1["1 Expert Chain<br/>& Four Gaps"]
  C2["2 Linear<br/>algebra"]
  C3["3 Calculus &<br/>optimization"]
  C4["4 Probability<br/>& statistics"]
  C5["5 Information<br/>theory"]
  C6["6 Computational<br/>thinking"]
  C7["7 Statistical<br/>learning"]
  C8["8 Latent-variable<br/>models"]
  C9["9 Backprop"]
  C10["10 CNNs"]
  C11["11 RNN / SSM /<br/>long conv"]
  C12["12 Attention &<br/>Transformer"]
  C13["13 Representation<br/>learning"]
  C14["14 Generative I<br/>VAE · flows · GAN"]
  C15["15 Diffusion &<br/>flow matching"]
  C16["16 Geometric<br/>DL"]
  C17["17 Foundation models<br/>& scaling"]
  C18["18 Interpretability<br/>& evaluation"]

  C1 --> C2 & C4
  C2 --> C3
  C3 --> C7
  C4 --> C7
  C4 --> C8
  C7 --> C8
  C3 --> C9
  C6 -.-> C9
  C7 --> C9
  C9 --> C10 --> C11 --> C12
  C2 --> C12
  C12 --> C13
  C5 --> C13
  C8 --> C14
  C5 --> C14
  C13 --> C14
  C14 --> C15
  C9 --> C16
  C12 --> C17
  C13 --> C17
  C9 --> C18
  C7 --> C18

  classDef f fill:#e0f2f1,stroke:#00897b,color:#004d40;
  classDef d fill:#e8eaf6,stroke:#3949ab,color:#1a237e;
  class C1,C2,C3,C4,C5,C6,C7,C8 f;
  class C9,C10,C11,C12,C13,C14,C15,C16,C17,C18 d;
```

### 2b. Biology and computational biology (Chapters 19–30)

```mermaid
flowchart TD
  C19["19 Cell as information<br/>system"]
  C20["20 Genomes &<br/>variation"]
  C21["21 Population genetics<br/>& evolution"]
  C22["22 Gene<br/>regulation"]
  C23["23 Proteins"]
  C24["24 Molecules &<br/>drug chemistry"]
  C25["25 Single-cell, spatial,<br/>perturbation"]
  C26["26 Statistical<br/>genetics"]
  C27["27 Sequences &<br/>alignment"]
  C28["28 Representing<br/>biological data"]
  C29["29 Classical models<br/>(PWM · Potts · Rosetta)"]
  C30["30 Single-cell<br/>methods"]
  K4["4 Probability"]:::ext
  K6["6 Computational thinking"]:::ext
  K8["8 Latent-variable models"]:::ext

  C19 --> C20 --> C21
  C19 --> C22 & C23 & C24 & C25
  K4 --> C26
  C21 --> C26
  C20 --> C27
  K6 -.-> C27
  C27 --> C28
  C21 & C22 & C23 --> C29
  K8 --> C29
  C25 --> C30
  K8 --> C30
  C28 --> C30

  classDef b fill:#e8f5e9,stroke:#43a047,color:#1b5e20;
  classDef ext fill:#f5f5f5,stroke:#9e9e9e,color:#424242,stroke-dasharray: 4 3;
  class C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30 b;
```

### 2c. Biological foundation models (Chapters 31–42) and what they draw on

*Genomes, RNA, and genotype → phenotype:*

```mermaid
flowchart TD
  D10["10 CNNs"]:::ext
  D11["11 SSM / long conv"]:::ext
  D12["12 Transformer"]:::ext
  D13["13 Self-supervision"]:::ext
  D17["17 Scaling"]:::ext
  B22["22 Regulation"]:::ext
  B26["26 Stat. genetics"]:::ext
  B29["29 Classical models"]:::ext
  C31["31 Sequence-to-function"]
  C32["32 Genomic LMs"]
  C33["33 RNA · splicing · translation"]
  C41["41 Genotype → phenotype"]

  D10 & B22 & B29 --> C31
  D11 & D12 & D13 & D17 --> C32
  C31 --> C32
  C31 & C32 --> C33
  B26 & C31 & C32 --> C41

  classDef m fill:#fff3e0,stroke:#e65100,color:#bf360c;
  classDef ext fill:#f5f5f5,stroke:#9e9e9e,color:#424242,stroke-dasharray: 4 3;
  class C31,C32,C33,C41 m;
```

*Proteins, molecules, and evolution:*

```mermaid
flowchart TD
  D13["13 Self-supervision"]:::ext
  D15["15 Diffusion"]:::ext
  D16["16 Geometric DL"]:::ext
  B21["21 Pop. genetics"]:::ext
  B23["23 Proteins"]:::ext
  B24["24 Molecules"]:::ext
  C34["34 Protein LMs"]
  C35["35 Structure prediction"]
  C36["36 Protein & biomolecular design"]
  C37["37 Molecular ML & drug discovery"]
  C42["42 Evolutionary modeling"]

  B21 & B23 & D13 --> C34
  C34 --> C35
  D16 --> C35
  C35 --> C36
  D15 --> C36
  C35 --> C37
  B24 & D15 --> C37
  B21 & C34 --> C42

  classDef m fill:#fff3e0,stroke:#e65100,color:#bf360c;
  classDef ext fill:#f5f5f5,stroke:#9e9e9e,color:#424242,stroke-dasharray: 4 3;
  class C34,C35,C36,C37,C42 m;
```

*Cells:*

```mermaid
flowchart TD
  D13["13 Self-supervision"]:::ext
  B25["25 Single-cell, spatial, perturbation"]:::ext
  B30["30 Single-cell methods"]:::ext
  M44["44 Causal inference"]:::ext
  C38["38 Single-cell foundation models"]
  C39["39 Perturbation & virtual cell"]
  C40["40 Multimodal & multi-scale"]

  B25 & B30 & D13 --> C38
  C38 --> C39
  M44 -.-> C39
  C38 & C39 --> C40

  classDef m fill:#fff3e0,stroke:#e65100,color:#bf360c;
  classDef ext fill:#f5f5f5,stroke:#9e9e9e,color:#424242,stroke-dasharray: 4 3;
  class C38,C39,C40 m;
```

### 2d. Methodology, frontier, and independent research (Chapters 43–59)

```mermaid
flowchart TD
  C43["43 Benchmarks &<br/>evaluation"]
  C44["44 Causal<br/>inference"]
  C45["45 Shift, confounding,<br/>failure"]
  C46["46 Experimental<br/>design"]
  C47["47 Scaling &<br/>data economics"]
  C48["48 Mechanistic<br/>interpretability"]
  PART7["Part VII<br/>(Ch 31–42)"]:::ext
  C49["49 Competing<br/>paradigms"]
  C50["50 Open problems:<br/>genomes"]
  C51["51 Open problems:<br/>cells"]
  C52["52 Open problems:<br/>proteins & design"]
  C53["53 Brains & neural<br/>systems"]
  C54["54 AI scientists"]
  C55["55 Ten attacks"]
  C56["56 Evaluating ideas"]
  C57["57 Reading papers"]
  C58["58 Reasoning without<br/>answers"]
  C59["59 Idea → publication"]

  C43 --> C45
  C44 --> C45
  C43 & C44 --> C46
  PART7 --> C49
  C43 & C44 & C45 & C46 & C47 & C48 --> C49
  C49 --> C50 & C51 & C52 & C53 & C54
  C50 & C51 & C52 --> C55
  C53 & C54 --> C55
  C55 --> C56
  C57 --> C58
  C56 --> C58
  C58 --> C59

  classDef p fill:#f3e5f5,stroke:#8e24aa,color:#4a148c;
  classDef ext fill:#f5f5f5,stroke:#9e9e9e,color:#424242,stroke-dasharray: 4 3;
  class C43,C44,C45,C46,C47,C48,C49,C50,C51,C52,C53,C54,C55,C56,C57,C58,C59 p;
```

## 3. Concept dependencies (the "structural rhymes")

The same mathematical object recurs across chapters. Following these threads is the fastest way to see how the book's ideas are connected.

```mermaid
flowchart TD
  LV["Latent-variable inference<br/>ELBO · EM · posterior"]
  LV --> VAE["VAE · scVI<br/>(Ch 8, 14, 30)"]
  LV --> COAL["Coalescent · phylogenetics<br/>(Ch 21, 42)"]
  LV --> DIFF["Diffusion<br/>(Ch 15, 36)"]
```

```mermaid
flowchart TD
  LR["Low-rank / factor structure<br/>SVD · PCA · attention · LD · coevolution"]
  LR --> ATT["Attention · protein LMs<br/>(Ch 12, 34)"]
  LR --> GWAS["GWAS · LD · polygenic scores<br/>(Ch 26, 41)"]
  LR --> SC["Cell-state manifolds<br/>(Ch 25, 30, 38)"]
```

```mermaid
flowchart TD
  EB["Energy / Boltzmann models<br/>Potts · score · fitness · Rosetta"]
  EB --> PLM["Protein fitness from likelihood<br/>(Ch 29, 34, 42)"]
  EB --> SCORE["Score-based design<br/>(Ch 15, 36)"]
```

```mermaid
flowchart TD
  EQ["Equivariance & convolution<br/>motifs · CNN · Hyena · SE(3)"]
  EQ --> GENO["Sequence-to-function<br/>(Ch 10, 31)"]
  EQ --> STRUCT["Structure models<br/>(Ch 16, 35)"]
```

```mermaid
flowchart TD
  CI["Conditional independence & causality<br/>graphical models · do-calculus · DCA"]
  CI --> PERT["Perturbation prediction<br/>(Ch 39, 44)"]
  CI --> MR["Mendelian randomization<br/>(Ch 26, 44)"]
```

```mermaid
flowchart TD
  CH["Noisy channel & compression<br/>entropy · LMs · evolution · MDL"]
  CH --> LM["Genomic / protein LMs<br/>(Ch 5, 32, 34)"]
  CH --> EVO["Selection as a channel<br/>(Ch 5, 21)"]
```

## 4. Three ways to use the graph

| If you want to... | Follow this path |
|---|---|
| Understand how AlphaFold-style models work | 2 → 3 → 9 → 12 → 16 → 23 → 34 → 35 |
| Understand why single-cell foundation models disappoint (and what might fix them) | 4 → 8 → 13 → 25 → 30 → 38 → 39 → 43 → 44 → 45 |
| Understand genomic language models and their evaluation | 5 → 11 → 12 → 13 → 20 → 21 → 22 → 31 → 32 → 43 |
| Be able to design a good benchmark for a new problem | 4 → 7 → 18 → 43 → 44 → 45 → 46 |
| Generate research ideas in a new subfield | 1 → [the subfield's Part VII chapter] → 49 → [matching Open Problems chapter] → 55 → 56 |
