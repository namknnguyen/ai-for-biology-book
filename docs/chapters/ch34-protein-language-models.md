# Chapter 34. Protein Language Models: What Evolution Teaches a Sequence Model

!!! abstract "Chapter at a glance"
    **Motivation.** Protein language models (pLMs) were the first biological foundation models to work at scale: trained only to fill in masked or next amino acids across hundreds of millions of natural sequences, they produce embeddings that predict structure, function, and the effect of mutations with no labels. They also expose, more cleanly than any other biological model, the central questions of this book: *what has the model learned (physics, motifs, family memory, database bias)?, when does pretraining help a new task?, does scale keep helping?*, and *is a score a measurement?* The chapter connects the pLM objective to the Potts models of Chapter 29 (the masked conditional *is* the Potts conditional), runs a controlled experiment that shows exactly when cross-family pretraining transfers (and when it does not), dissects the scaling and preference evidence on ProteinGym, summarizes what interpretability has found inside the models, and compares pLMs with their alternatives (family-specific generative models, structure-conditioned models, family-conditioned pLMs).
    **Prerequisites.** Chapters 12, 13, 17, 18, 23, 29, 32, 43.
    **You will be able to:** (1) derive the zero-shot mutation score of a masked LM and show that it equals a Potts energy difference when the data are Potts; (2) explain why pretraining on many families helps a new family, and bound the benefit by the shared structure; (3) interpret a ProteinGym-style result with the wild-type likelihood, the assay, and database biases in view; (4) summarize what pLMs have been shown to learn and what remains hypothesis; (5) choose between a pLM, a family model, a structure-conditioned model, and few-shot supervised learning for a variant-effect task; (6) design a fair test of whether scale or tuning helps your protein.

---

## 34.0 Why proteins are different from genomes for language models

A protein sequence is the end product of selection for a function, and a *family* of homologous sequences is a sample from the set of sequences compatible with that function and fold. Three facts make pLMs more successful per parameter than genomic LMs (Chapter 32):

1. **The data are already selected.** Every natural protein sequence has survived selection for stability and function; a genome's bases are mostly neutral or only weakly constrained. The training signal is dense in constraint.
2. **The alphabet is richer and the units are functional.** Twenty letters, with a physicochemical structure (hydrophobicity, charge, size), and a single protein is one functional unit.
3. **Families give a natural population.** Homologous sequences across species are replicates under selection, and the covariation among positions (Chapter 29) encodes contacts.

!!! lens "Research lens: what information does a pLM use and ignore?"
    *Uses*: the statistics of the sequence database it was trained on (UniRef-scale clusters, i.e., hundreds of millions of sequences), hence family membership, conservation, motifs, covariation. *Ignores*: the *measurement* of any property (stability, activity, expression, binding) in any assay; structure unless supplied (ESM3 adds it); the environment (temperature, pH, cellular context) and the *selection pressure of the assay* (a deep mutational scan selects for something the organism never selected for); and the *sampling process of the database* (some taxa, such as model organisms and human pathogens, are over-represented). A pLM score is therefore a statement about *evolutionary plausibility under the database's sampling*, and becomes a prediction of fitness only to the extent that the assay's selection coincides with evolutionary selection.

---

## 34.1 The objective, and its relation to the Potts model

**Masked language modeling (MLM).** Select a fraction of positions $M$ (typically 15%), replace them with a mask token, and minimize
$$
\mathcal L_\text{MLM}(\theta)=-\mathbb E_{x\sim p_\text{data}}\ \mathbb E_{M}\ \sum_{i\in M}\log p_\theta\big(x_i\mid x_{\setminus M}\big).
$$
Examples: ESM-1b (Rives et al., 2021; 650M parameters on UniRef50), ESM-1v (Meier et al., 2021; a variant-effect-oriented ensemble), and ESM-2 (Lin et al., 2023; 8M to 15B parameters). **Autoregressive** pLMs (ProGen, Madani et al.; ProGen2, Nijkamp et al.; ProtGPT2) maximize $\sum_i\log p_\theta(x_i\mid x_{<i})$ and generate by sampling; **encoder-decoder or masked-span** variants and **conditional** models (family-conditioned: PoET, Truong and Bepler 2023, which conditions on an MSA-like set of homologs in context) exist. **Multimodal** pLMs (ESM3, Hayes et al., *Science* 2025) predict tokens of sequence, structure, and function jointly, so a prompt can be partial in any modality.

!!! math "Derivation: the zero-shot mutation score is a Potts energy difference"
    Suppose the data are drawn from a Potts model $p(x)\propto\exp\!\big(\sum_ih_i(x_i)+\sum_{i<j}J_{ij}(x_i,x_j)\big)$ (Chapter 29). Its single-site conditional is exactly a softmax,
    $$
    p(x_i=a\mid x_{\setminus i})=\softmax_a\Big(h_i(a)+\sum_{j\ne i}J_{ij}(a,x_j)\Big).
    $$
    An MLM with infinite capacity and enough data learns precisely this conditional (the true conditionals of the data distribution minimize the cross-entropy). For a single mutation at position $i$ from the wild-type residue $a$ to $b$, define the **masked-marginal log-likelihood ratio**
    $$
    \text{LLR}(a\to b)=\log p_\theta(x_i=b\mid x_{\setminus i})-\log p_\theta(x_i=a\mid x_{\setminus i}).
    $$
    The normalizing constant cancels, giving
    $$
    \text{LLR}(a\to b)=\big[h_i(b)-h_i(a)\big]+\sum_{j\ne i}\big[J_{ij}(b,x_j)-J_{ij}(a,x_j)\big]=\Delta\!\log\tilde p(x),
    $$
    the *exact* change in the unnormalized log-probability of the whole sequence, i.e., minus the Potts energy difference $\Delta E$. The masked conditional therefore recovers the energy change of a *single* substitution *without computing the partition function*; multiple substitutions need the joint and require either the autoregressive likelihood or an approximation (summing masked marginals, which ignores the interactions among the mutations). $\square$

!!! rhyme "Structural rhyme: masked language model ↔ pseudolikelihood ↔ site-wise logistic regression (Chapter 29)"
    A pLM is plmDCA with an enormous nonlinear model per site, fit to a vast number of families at once. The Potts pseudolikelihood fits $L$ logistic regressions to one family; the pLM fits one network to all families, with the family identity implicit in the context.

**Implementation sketch (shapes).**

```python
# masked-marginal zero-shot scoring of all single mutants of a wild type
# tokens: (L,) ints.  model(tokens_masked) -> logits (L, 20)
def score_all(model, tokens):
    scores = torch.zeros(len(tokens), 20)
    for i in range(len(tokens)):                          # L forward passes (or one with a masked-marginal approximation: O(1) passes)
        masked = tokens.clone(); masked[i] = MASK
        logp = F.log_softmax(model(masked[None])[0, i], -1)        # (20,) log p(. | x_{-i})
        scores[i] = logp - logp[tokens[i]]                       # LLR of every substitution at i vs the wild type
    return scores                                                # (L, 20): high = more plausible than wild type
```

**Complexity.** A forward pass is $O(L^2d+Ld^2)$ per layer; all single mutants cost $L$ passes (one per masked site) rather than $19L$. A 15B-parameter model on a 500-residue protein takes seconds per pass on a modern GPU; scoring a proteome is a research-scale compute project.

---

## 34.2 When does pretraining across families help a new family? A controlled experiment

The question every pLM user asks: a model pretrained on families A, B, C… is applied to my family D. When is that better than a model trained on D alone? A Potts model gives a setting where the truth is known (and the zero-shot score *should* equal the true energy change).

**Design.** Families are Potts models on $L=30$ positions with $q=4$ states (about 10% of position pairs coupled). In the **related** condition each family is 80% a shared "fold" (shared fields and couplings) plus 20% family-specific; in the **unrelated** condition there is no sharing. A masked-position MLP is pretrained on 20 families (400 sequences each, no family label; masked-position prediction as above), then evaluated zero-shot on a *new* family by the Spearman correlation between its masked-marginal LLR and the **true** energy change for all single mutants of 25 wild types of that family. It is compared with four baselines on the new family, given $N$ sequences: a site-independent model (conservation only), a Potts pseudolikelihood model (the classical DCA, Chapter 29), the same network trained from scratch, and the pretrained network adapted to the $N$ sequences by gradient descent (a *light* budget: 4 epochs at a low learning rate; and a *full* budget: the same epochs and data as the from-scratch run, starting from the pretrained weights).

```python
--8<-- "code/ch34_plm_toy.py"
```

```text

-- RELATED families (80% of the coupling structure shared): pretraining on 20 families (8,000 sequences); target family adapted with N sequences --
N target     site-independent   Potts (pseudolikelihood)   network from scratch   pretrained, zero-shot   pretrained + light adaptation   pretrained + full-budget adaptation
     20            0.471                  0.513                   0.148                 0.748                   0.764                           0.581
     50            0.489                  0.569                   0.084                 0.748                   0.769                           0.558
    200            0.548                  0.732                   0.330                 0.748                   0.775                           0.726
   1000            0.564                  0.880                   0.523                 0.748                   0.780                           0.784

-- UNRELATED families (no shared structure): pretraining on 20 families (8,000 sequences); target family adapted with N sequences --
N target     site-independent   Potts (pseudolikelihood)   network from scratch   pretrained, zero-shot   pretrained + light adaptation   pretrained + full-budget adaptation
     20            0.432                  0.468                   0.182                -0.029                   0.094                           0.224
     50            0.515                  0.603                   0.140                -0.029                   0.118                           0.256
    200            0.539                  0.741                   0.363                -0.029                   0.119                           0.440
   1000            0.549                  0.882                   0.561                -0.029                   0.124                           0.503
```

**Reading the results.**

1. **Shared structure creates zero-shot ability.** In the related condition the pretrained network, which has never seen the target family, reaches a zero-shot Spearman of 0.75 against the true energy change, far above the site-independent baseline (0.47–0.56) and the Potts model fit to 20–50 target sequences (0.51–0.57). In the unrelated condition the same recipe gives $-0.03$: **zero-shot transfer is a statement about what the families share, and nothing more.**
2. **Pretraining pays most when target data are scarce, and caps at the shared fraction.** With related families, light adaptation moves 0.75 to 0.76–0.78 for any $N$ from 20 to 1,000. The family-specific Potts model catches up at $N\approx200$ (0.73) and wins at $N=1000$ (0.88 against 0.78): the pretrained network cannot recover the 20% of the energy function that is specific to the family without enough target data, and its capacity is spent on the shared 80%. The *crossover size* depends on the shared fraction and on how well the target-specific model class fits; for real proteins neither is known, and the practical remedy is to *measure it*, with a held-out set of the target's labeled variants.
3. **The from-scratch network is poor in this toy**: far below Potts pseudolikelihood at every $N$ (0.52 against 0.88 at $N=1000$ with related families) and below even the site-independent baseline in seven of the eight cells (0.08–0.56 against 0.43–0.56; the exception is $N=1000$ with unrelated families, 0.56 against 0.55). A generic MLP is a poor inductive bias for couplings with few sequences; Potts pseudolikelihood, whose model class matches the truth, is the best at large $N$. A pLM is not automatically better than a model built for the structure of the problem.
4. **Adaptation can be too light or too heavy.** With *related* families, the full-budget adaptation is *worse* than light adaptation at small $N$ (0.58 against 0.76 at $N=20$): fine-tuning on 20 sequences for many steps overwrites the shared structure (catastrophic forgetting; overfitting). With *unrelated* families, light adaptation leaves the model near 0.1 while full-budget adaptation reaches 0.22–0.50, similar to training from scratch (0.18–0.56) and slightly lower at the largest $N$ (0.50 against 0.56): pretraining on unrelated data gives an initialization that does not help and may cost a little. The practical rule: **tune the adaptation budget on held-out target variants; it is a hyperparameter with large effects** (Chapter 17's LoRA and early-stopping discussion).
5. **Evo-tuning in miniature.** Adapting a pretrained pLM on the sequences of the target family (unsupervised "evo-tuning", Alley et al. 2019; Biswas et al. 2021) is the real-world analog of the adaptation columns; Gordon et al. (2025, §34.4) report that it repairs poor zero-shot performance on proteins to which the model assigns low likelihood.

!!! lens "Research lens: assumptions and what the toy cannot say"
    $q=4$, $L=30$, a single shared fold, an MLP rather than a transformer, 400 sequences per pretraining family, and ground truth that *is* an energy function. Real fitness depends on stability, expression, and activity in an assay (Chapter 23), the shared structure between families is not a fixed fraction, and databases are phylogenetically correlated. The experiment demonstrates *the logic of transfer* (benefit bounded by sharing; adaptation as a tuned budget; model-class match), not the numbers a real pLM will show.

---

## 34.3 What do protein language models learn?

Evidence, from the most to the least settled:

- **Contacts and structure** [[E]]. Attention maps and embeddings contain contact information: logistic regression on attention maps predicts contacts (Rao et al., 2021, "Transformer protein language models are unsupervised structure learners"), and ESMFold (Lin et al., 2023) predicts atomic structure from a single sequence by adding a folding trunk to ESM-2 embeddings. The accuracy of language-model-based structure prediction scales with model size and falls for sequences with few homologs [[S]].
- **Evolutionary statistics, partly by motif memory** [[S]]. Zhang et al. (*PNAS* 2024) show that ESM-2 predicts contacts by recalling *motifs of pairwise contacts* seen in training (co-occurring sequence segments), rather than by something like inferring a coupling matrix for the target family from first principles; coevolution is represented as motif-conditioned pairwise statistics. This is a statement about the mechanism of the "unsupervised" contact predictions, and it implies the model's success is bounded by the coverage of the motifs in its training set.
- **Biophysical features** [[P]]. Sparse autoencoders trained on pLM activations find features aligned with secondary structure, binding sites, and domains (Simon and Zou, InterPLM, 2024/2025; Adams et al., *PNAS* 2025), and these features sometimes steer generation. As in Chapter 18, alignment of a feature with an annotation is not evidence that the model *uses* it causally; ablation and patching are needed (Chapter 48).
- **Taxonomic and database bias** [[S]]. Because UniRef-scale databases sample the tree of life unevenly, likelihoods are higher for well-sampled taxa; Ding and Steinhardt (2024) show that pLM likelihoods are biased by unequal sequence sampling across the tree of life and that this affects fitness prediction.
- **Memorization** [[P]]. Large models fit rare sequences; whether a design or a prediction is novel needs a nearest-neighbor check against the training set at the sequence and the structure level.

---

## 34.4 Variant-effect prediction: the evidence, and why scale does not always help

**ProteinGym** (Notin et al., NeurIPS 2023 Datasets and Benchmarks, updated since) standardizes more than 2.5 million variant measurements from over 200 deep mutational scanning (DMS) assays and clinical variant sets, and compares models in the *zero-shot* and *supervised* regimes using Spearman correlation per assay averaged across assays. The ranking of methods has been stable in outline since 2023: **family-specific or family-aware models** (EVE, Tranception with retrieval, ensembles such as TranceptEVE; MSA-conditioned pLMs such as PoET) are at or near the top of the zero-shot ranking, **single-sequence pLMs** are competitive and cheaper, **structure-aware** pLMs improve over sequence-only ones on average, and *supervised* models using even a few hundred labeled variants often outperform all of them. As of the updates reported for ProteinGym v1.3, the best zero-shot ensembles reached an average Spearman of about 0.5, with ESM-2 650M at about 0.41; these numbers move with each benchmark release, so cite the version [[S]].

!!! paper "Paper dissection: 'Protein language model fitness is a matter of preference' (Gordon, Lu & Abbeel; bioRxiv 2024; ICLR 2025)"
    **Problem.** Why does the zero-shot fitness prediction of a pLM vary so much across proteins, and why does it sometimes get *worse* with scale?
    **Insight.** The model's *likelihood of the wild-type sequence* (its implicit preference) predicts how good its mutation scores will be.
    **Data / evaluation.** The wild-type sequences of 217 ProteinGym DMS assays, scored by several pLMs; performance (Spearman) as a function of the wild-type likelihood.
    **Result (as reported).** Both over-preferred (very high likelihood) and under-preferred (very low likelihood) wild types harm performance, with the best performance at intermediate likelihood; unsupervised fine-tuning on homologous sequences (evo-tuning) remedies under-performance on low-likelihood wild types.
    **Why.** A masked-marginal LLR is informative when the model's distribution around the wild type reflects the *constraint pattern*; for over-preferred sequences the model is near-certain at many positions (the wild type is memorized or typical of a dominant clade) and gives saturated scores that cannot rank mutations; for under-preferred sequences the model has little information.
    **Assumptions.** The DMS fitness is correlated with natural-selection plausibility; the likelihood is comparable across proteins.
    **Limitations.** Correlational: a causal intervention on the training distribution (re-training with a changed sampling) would test it more directly.
    **Related finding.** An analysis of ESM-2 models from 8M to 15B parameters on 154 ProteinGym DMS assays found that fitness-prediction performance peaked at 650M parameters and declined for larger models, attributed to a shift of the wild-type likelihood out of the intermediate range (bioRxiv 2025, "Understanding language model scaling on protein fitness prediction") [[S]]. *Perplexity continues to improve with scale; fitness prediction need not.*

The lesson is general. Language modeling loss and variant-effect accuracy measure different things: the loss rewards the model for assigning high probability to sequences like those in the database, while a zero-shot mutation score requires the *shape* of the probability near one sequence to match constraint. Both the scaling law (Chapter 17) and the downstream metric must be reported.

!!! rhyme "Structural rhyme: protein-LM preference ↔ the composition confound of a DNA LM (Chapter 32) ↔ the frequency-versus-selection confound (Chapter 20)"
    In all three, a likelihood is a joint statement about *sampling* (what the database contains), *mutational or compositional bias*, and *selection*; a score that ignores the first two will fail where they dominate. The remedy is to calibrate against a null that contains only the nuisance variable.

---

## 34.5 Generation: from sampling plausible sequences to testing them

Generative pLMs sample sequences: ProGen produced functional lysozymes, some with low identity to natural ones, after fine-tuning on a family (Madani et al., *Nat. Biotechnol.* 2023); ESM3 (Hayes et al., *Science* 2025) produced esmGFP, a fluorescent protein with 58% sequence identity to the nearest known fluorescent protein, which the authors describe as equivalent to many hundreds of millions of years of evolution [[S]]; later models scaled to tens of billions of parameters. The questions for any generation claim are the three of Chapter 36: *what fraction of designs worked* (the denominator), *how different from natural sequences* (novelty by sequence and structure), and *how different from each other* (diversity). A generative model's sampling temperature, the family it is conditioned on, and the filter applied before testing determine all three.

---

## 34.6 Comparing paradigms for variant-effect prediction

| Approach | Information used | Needs | Strengths | Failure modes |
|---|---|---|---|---|
| **Site-independent profile** (conservation) | Family MSA | An MSA | Fast; hard to beat on conserved sites | No epistasis; MSA depth |
| **Potts / EVE (family-specific generative)** | Family MSA, pairwise or latent structure | A deep MSA ($N_\text{eff}\gtrsim L$) | Matches the family's constraints; interpretable | Shallow MSAs; alignment errors; one model per family |
| **Single-sequence pLM** | All of UniRef (family inferred in context) | A sequence | Works for orphans; transfers across families; cheap embeddings | Preference/likelihood effects (§34.4); database bias; opaque |
| **Family-conditioned pLM** (PoET, retrieval-augmented) | Homologs as context | Homologs | Combines family specificity with generalization | Needs retrieval; context cost |
| **Structure-aware / inverse-folding scores** | Structure plus sequence | A structure | Captures stability and burial | Structure errors; ignores function not in the structure |
| **Supervised on assay data** | Labels from the target assay | 50–500 labeled variants | Learns the actual selection pressure | Overfits small data; needs experiments (Chapter 46) |

No row dominates. The best practice, supported by ProteinGym's supervised track and by experimental programs (EVOLVEpro-like active learning with a pLM, Jiang et al., *Science* 2024), is a *hybrid*: zero-shot scores or embeddings as features and priors, a small number of assay measurements to learn the actual objective, chosen by active learning (Chapter 46).

---

## 34.7 Worked research examples

!!! example "Worked Research Example 34.1: A pLM scores a DMS with Spearman 0.30. What is the matter?"
    **Situation.** You score every single mutant of a metabolic enzyme with ESM-2 650M and compare with a deep mutational scan of growth of a complemented strain. Spearman is 0.30, lower than the 0.4–0.5 typical for similar enzymes.

    **Question.** Which explanations would you test, in what order, and what would each imply?

    **Reasoning (Expert Chain).**

    1. **L1–L2 Problem and baselines.** Compute (a) site-independent conservation from an MSA, (b) an MSA-conditioned model (EVE/Potts), (c) a structure-based $\Delta\Delta G$ estimate. Without them, 0.30 is uninterpretable.
    2. **L3 Assumption: the assay measures what evolution selected.** The DMS selects for growth in one condition; natural selection includes many conditions, regulation, and expression level. Check the *noise ceiling* of the DMS (replicate correlation, Chapter 1 and 25) and the fraction of variants near the wild-type level; a saturating phenotype (global epistasis, Chapter 23) compresses the dynamic range so that rank correlation is bounded.
    3. **L4 Failure mode: wild-type likelihood.** Compute the model's wild-type pseudo-perplexity relative to the benchmark distribution (§34.4). If the wild type is over- or under-preferred, expect poor LLR ranking; test evo-tuning on homologs.
    4. **L5 Failure mode: domain structure.** Compute Spearman *per region* (active site, buried core, surface): pLMs may rank core mutations well and surface ones poorly because surface positions are weakly constrained and the assay's noise dominates.
    5. **L10 Experiments.** (a) Replace single-sequence scoring by an MSA-conditioned model; (b) evo-tune on 10,000 homologs; (c) add structural features; (d) fit a supervised model on 100 randomly chosen variants and evaluate on the rest (the *learning curve* indicates how much of the assay's selection is not captured by evolutionary information; Chapter 47).
    6. **L11 Interpretation.** If conservation alone gives 0.3 and a supervised model on 100 variants gives 0.7, the evolutionary prior is incomplete for this assay, and the missing information is the *assay-specific selection*, not model capacity.

    **Expert analysis.** The correlation is the *product* of a model's quality, the assay's reliability, and the match between the assay's selection and evolution's. Diagnose each factor before blaming the model or scaling it.

!!! example "Worked Research Example 34.2: Will a bigger pLM help my protein family? No known answer"
    **Situation.** Your lab engineers a family of glycoside hydrolases and has 150 measured variants (activity) for one member. A colleague proposes switching to the 15B-parameter model; compute is not free.

    **Question.** How would you decide, and what would you do with the 150 measurements?

    **Reasoning.**

    1. **Do not test the size on the 150 measurements alone** (a noisy comparison: the standard error of a Spearman correlation on 150 points is about 0.07 for $\rho\approx0.4$, so differences between models of 0.05 are within noise). Instead use the *distribution across assays* of a benchmark that resembles your family (ProteinGym assays from related enzymes) to estimate whether scale helps *on average* for this kind of protein; given §34.4, expect a plateau or decline beyond 650M for many families.
    2. **Use the 150 measurements for what they are best at**: learning the *assay-specific* correction on top of an evolutionary prior. Fit a small supervised head (ridge or Gaussian process) on embeddings from two or three models and concatenations of zero-shot scores; evaluate with cross-validation *grouped by position* (so the model is tested on unseen positions, which is what design needs; Chapter 43).
    3. **Spend the next experiments by information** (Chapter 46): choose the next 20–50 variants by uncertainty and expected improvement, which typically gives more than a bigger model.
    4. **Decision rule.** Adopt the larger model only if it improves the cross-validated, position-grouped $\rho$ by more than twice its standard error across at least three related assays and the compute is justified; otherwise tune the cheaper model (evo-tuning on homologs).

    **What is not known.** Whether, for your family, evolutionary information saturates below the assay's selection (in which case no pLM will do better than the assay-learned correction), and whether the benchmark's average behavior applies. The first 150 measurements, analyzed this way, give an estimate.

---

## 34.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: using a protein language model responsibly"
    1. **Compute the wild-type likelihood** (pseudo-perplexity) of your protein under the model and compare with benchmark proteins.
    2. **Compare with a conservation baseline and an MSA model** on the same variants; require the pLM to add information.
    3. **Check the assay's noise ceiling** and the dynamic range of the phenotype.
    4. **Report per-position and per-region performance**, not only a global Spearman.
    5. **For generation, report the denominator** (designs tested), novelty (nearest training sequence and structure), and diversity.
    6. **Treat adaptation as a hyperparameter**: tune the budget on held-out variants (§34.2).
    7. **Re-run the toy** of §34.2 with different shared fractions (0.2, 0.5, 0.95) to see where pretraining stops helping.

    **What it teaches.** A pLM is a compressed model of evolutionary plausibility; whether that predicts *your* property is a measurable quantity, and its measurement is the contribution.

    **An open question to carry forward.** The toy shows a crossover in $N$ between a pretrained model (capped by the shared fraction) and a family-specific model. For real families the shared fraction is unknown. Could one *estimate it from the pretraining data alone*, for example by measuring how well a pLM's embedding of a family predicts the family-specific coupling statistics inferred from its MSA, and use it to forecast where evo-tuning or supervised data will start to dominate?

---

## 34.9 Connections

- **Backward:** attention (Chapter 12); representation learning and the objective (Chapter 13); scaling laws and LoRA (Chapter 17); interpretability (Chapter 18); proteins and global epistasis (Chapter 23); Potts models and DCA (Chapter 29); DNA LMs and the likelihood-ratio confound (Chapter 32); benchmarks (Chapter 43).
- **Forward:** structure prediction built on pLM embeddings (Chapter 35); design and generation (Chapter 36); evolutionary models of fitness (Chapter 42); mechanistic interpretability (Chapter 48); open problems in proteins (Chapter 52).

!!! takeaways "Key takeaways"
    1. A masked pLM learns the conditional distribution of each residue given the rest; for Potts-distributed data the masked-marginal log-likelihood ratio *equals* the energy difference of a single mutation (the partition function cancels).
    2. **Zero-shot transfer is bounded by shared structure**: with 80% shared coupling structure across families the pretrained network reached a zero-shot Spearman of 0.75 on a new family; with no sharing it reached $-0.03$.
    3. Pretraining helps most when target data are scarce; a family-specific Potts model overtook the pretrained network at about 200–1,000 sequences (0.88 against 0.78 at $N=1000$).
    4. **Adaptation is a tuned budget**: too heavy at small $N$ (0.58 against 0.76) forgets the shared structure; too light leaves unrelated pretraining useless (0.1).
    5. pLMs encode contact and motif statistics (partly by memorizing pairwise motifs), taxonomic database biases, and features that sparse autoencoders can recover; mechanistic claims need causal tests.
    6. **Scale does not monotonically improve fitness prediction**: wild-type likelihood predicts performance, with both over- and under-preferred sequences doing worse; ESM-2 fitness performance peaked at 650M parameters in one analysis; report both perplexity and the downstream metric.
    7. The best practice is a hybrid of evolutionary priors (zero-shot scores and embeddings) and a small number of assay measurements chosen by active learning.

---

## Further reading

- Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118, e2016239118. Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *NeurIPS.* Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130. Hayes, T. et al. (2025). Simulating 500 million years of evolution with a language model. *Science* 387, 850–858 (ESM3).
- Madani, A. et al. (2023). Large language models generate functional protein sequences across diverse families. *Nat. Biotechnol.* 41, 1099–1106. Nijkamp, E., Ruffolo, J., Weinstein, E. N., Naik, N. & Madani, A. (2022). ProGen2: exploring the boundaries of protein language models. *arXiv:2206.13517.* Truong, T. & Bepler, T. (2023). PoET: a generative model of protein families as sequences-of-sequences. *NeurIPS.*
- Notin, P. et al. (2023). ProteinGym: large-scale benchmarks for protein fitness prediction and design. *NeurIPS Datasets and Benchmarks.* Gordon, C., Lu, A. X. & Abbeel, P. (2025). Protein language model fitness is a matter of preference. *ICLR.* Ding, F. & Steinhardt, J. (2024). Protein language models are biased by unequal sequence sampling across the tree of life. *bioRxiv.*
- Rao, R., Meier, J., Sercu, T., Ovchinnikov, S. & Rives, A. (2021). Transformer protein language models are unsupervised structure learners. *ICLR.* Zhang, Z. et al. (2024). Protein language models learn evolutionary statistics of interacting sequence motifs. *PNAS* 121, e2406285121. Adams, E. et al. (2025). Sparse autoencoders uncover biologically interpretable features in protein language model representations. *PNAS* 122. Simon, E. & Zou, J. (2024). InterPLM: discovering interpretable features in protein language models via sparse autoencoders. *bioRxiv.*
- Alley, E. C. et al. (2019). Unified rational protein engineering with sequence-based deep representation learning. *Nat. Methods* 16, 1315–1322. Biswas, S., Khimulya, G., Alley, E. C., Esvelt, K. M. & Church, G. M. (2021). Low-N protein engineering with data-efficient deep learning. *Nat. Methods* 18, 389–396. Jiang, K. et al. (2024). Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science* 387, eadr6006. Frazer, J. et al. (2021). Disease variant prediction with deep generative models of evolutionary data (EVE). *Nature* 599, 91–95.
