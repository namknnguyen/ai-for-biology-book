# Chapter 47. Scaling, Compute, and the Economics of Biological Data

!!! abstract "Chapter at a glance"
    **Motivation.** In language modeling, loss falls as a smooth power law in parameters, data, and compute, and that regularity organizes a field's investment. Biology invites the same question: *if we gather ten times more cells, structures, sequences, or perturbations, or spend ten times more compute, how much better does a model get, and where does improvement stop?* The answer is more complicated than in text, because biological data are strongly redundant (a hundred million cells carry far fewer independent observations), measurement noise sets an irreducible floor, and what matters is often generalization to *new* chemistry, cell types, families, or individuals, where the error floor is set by distribution shift rather than noise. This chapter derives the form of learning curves with an irreducible term, shows how a power-law fit that is allowed to find its own floor can extrapolate through the measurement noise, and runs three experiments on a real data set (1,513 BACE-1 inhibitors): learning curves under random and cluster (novel-chemistry) splits; the same model class under a noise-ceiling constraint; and a comparison of *diversity* against *volume* at a fixed number of training molecules. It then reviews the scaling evidence for biological foundation models, frames the economics of data collection (value per dollar of more, more diverse, less noisy, or paired data), and gives a compute-accounting standard.
    **Prerequisites.** Chapters 1, 4, 7, 17, 24, 34, 38, 42–46.
    **You will be able to:** (1) write a learning curve with a noise floor and a shift floor and fit it without extrapolating below the measured ceiling; (2) estimate the marginal value of data from a learning curve; (3) compare the value of diversity, volume, quality, and pairing; (4) judge scaling claims in biology by what is held fixed; (5) account for compute and cost in a reproducible way; (6) design a scaling study for a new biological task.

---

## 47.0 What scales, and what is different about biology

**The language-model picture.** For a model with $N$ parameters trained on $D$ tokens, the test loss follows approximately
$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},
$$
with an irreducible term $E$ (the entropy of the data), and fitted exponents $\alpha,\beta\approx0.3$–$0.4$ (Kaplan et al. 2020; Hoffmann et al. 2022); for a fixed compute budget $C\approx6ND$ floating-point operations, the loss-minimizing allocation grows $N$ and $D$ together (Chapter 17). The regularity depends on the data being drawn i.i.d. from one distribution and on the evaluation being drawn from that same distribution.

**What differs in biology.**

1. **Redundancy.** Tokens in text are far less correlated than cells in an atlas or sequences in a database: homologous sequences are related by a tree (Chapter 42: 32 species carried the information of about 2.3 independent observations about a trait mean), and cells share tissue, donor, and batch. The *effective* sample size can be orders of magnitude below the nominal one.
2. **Noise ceilings.** A measurement has reproducibility limits (Chapter 1); a model cannot beat the noise of its labels, so the irreducible term is partly *known* from replicates (pIC$_{50}$ noise SD 0.54 gives an RMSE floor of 0.54 on BACE).
3. **Shift floors.** The deployment distribution differs from the training one (new scaffolds, cell types, ancestries, folds): error does not decay to the noise floor with more data *of the same kind*; it plateaus at a level set by the shift (Chapter 45).
4. **Heterogeneous objectives.** A scaling law for perplexity (a self-supervised loss) need not predict a downstream metric (Chapter 34: fitness prediction peaked at 650M parameters while perplexity kept improving).
5. **Data are costly and not fungible.** A perturbation cell, a crystal structure, and a deep mutational scan cost orders of magnitude different amounts, and they are different *kinds* of information (observation versus intervention, Chapter 44).

!!! lens "Research lens: what a learning curve measures"
    A learning curve $e(n)$ (test error against training-set size) is a *measurement of the data's information about the target, as seen through a model class and a split*. Its asymptote is the sum of the noise floor and the shift floor; its slope is the marginal value of more data of the same kind; its dependence on the split tells which floor binds.

---

## 47.1 Learning curves with floors

Let a model trained on $n$ examples have test error (RMSE or loss)
$$
e(n)=c+a\,n^{-b},\qquad c=c_\text{noise}\oplus c_\text{shift}\oplus c_\text{bias},\quad a>0,\ 0<b\le1,
$$
where $c$ collects the irreducible parts: the measurement noise of the labels, the shift between train and test distributions, and the approximation bias of the model class. The marginal value of one more example is
$$
-\frac{de}{dn}=ab\,n^{-b-1},
$$
which decreases as $n^{-(1+b)}$; the number of examples needed to halve the *reducible* error $e-c$ is $n_{1/2}=n\,2^{1/b}$ (for $b=0.3$, a factor of about 10 per halving).

!!! math "Derivation: fitting with an unknown floor can extrapolate through a known floor"
    Fit $e(n)=c+an^{-b}$ to $K$ noisy points. The parameters $(a,b,c)$ are nearly *non-identifiable* when the data cover less than a decade of $n$ and the curve has not yet bent: for any small $c'$ there are $(a',b')$ fitting the data nearly as well, with the objective surface a long, nearly flat valley. A least-squares fit that is free to choose $c\ge0$ therefore often returns $c=0$ with a small exponent $b$, and *extrapolates linearly in $\log n$ through the true floor*. If the floor is measurable (replicate noise gives $c_\text{noise}$ for an RMSE), **fixing** $c\ge c_\text{noise}$ turns the extrapolation into a bounded one. The practical rule: *never extrapolate a learning curve below a floor you can measure*, and report the extrapolation under a range of floors. $\square$

---

## 47.2 Three experiments on real data

The data are the 1,513 BACE-1 inhibitors of Chapter 24 (pIC$_{50}$, SD 1.34) with random forests on Morgan count fingerprints (the same model class as Chapter 37). **Experiment 1** estimates learning curves at $n=50$–800 under a **random split** (the test set is 400 random molecules; training molecules are drawn from the rest) and a **cluster split** (a quarter of the Butina clusters held out as the test set: novel chemical series), six repetitions each, with power-law fits with a free and with a fixed floor. **Experiment 2** compares *diversity* with *volume*: with the training size fixed at 100 molecules, the molecules are drawn from 1, 2, 5, 10, 20 or 30 different clusters (round-robin), and the model is tested on molecules from held-out clusters (15 draws per row).

```python
--8<-- "code/ch47_scaling_data.py"
```

```text
1513 molecules, 103 Butina clusters (Tanimoto >= 0.4 to the centroid); sd of pIC50 = 1.34

== 1. Learning curves: RMSE (pIC50 units) of a random forest versus training-set size ==
n train    random split RMSE (SE)    cluster split RMSE (SE)
    50     1.161 (0.021)            1.290 (0.089)
   100     1.041 (0.019)            1.290 (0.066)
   200     0.952 (0.009)            1.292 (0.077)
   400     0.871 (0.009)            1.145 (0.038)
   800     0.772 (0.013)            1.116 (0.045)

power-law fits  RMSE = a * n^-b + c
random split  : a = 2.03, b = 0.14, floor c = 0.000;  predicted RMSE at n = 1,200: 0.735; at 12,000: 0.529; at 120,000: 0.381
                with the floor FIXED at the measured noise ceiling 0.54: a = 2.24, b = 0.33;  predicted RMSE at n = 1,200: 0.763; at 12,000: 0.645; at 120,000: 0.590
cluster split : a = 1.66, b = 0.06, floor c = 0.000;  predicted RMSE at n = 1,200: 1.106; at 12,000: 0.970; at 120,000: 0.851
                with the floor FIXED at the measured noise ceiling 0.54: a = 1.16, b = 0.10;  predicted RMSE at n = 1,200: 1.111; at 12,000: 0.993; at 120,000: 0.900
noise ceiling check: a single-measurement noise SD of 0.54 pIC50 (Chapter 24) is an RMSE floor of 0.54 for any model

== 2. The same number of training molecules (100) from few vs many clusters; test = molecules from held-out clusters ==
clusters used   training molecules   test RMSE (SE over 15 draws)
       1             100            1.413 (0.053)   [14 valid draws]
       2             100            1.436 (0.050)   [15 valid draws]
       5             100            1.411 (0.042)   [15 valid draws]
      10             100            1.378 (0.042)   [15 valid draws]
      20             100            1.306 (0.051)   [15 valid draws]
      30             100            1.285 (0.037)   [15 valid draws]
```

**Reading Experiment 1 (learning curves).**

1. **Random split: steady improvement.** RMSE falls from 1.161 ($n=50$) to 0.772 ($n=800$), roughly a 33% reduction over a 16-fold increase in data. An unconstrained power-law fit gives $a=2.03$, $b=0.14$ and a floor of 0 (the lower bound), and *extrapolates to 0.529 at 12,000 molecules and 0.381 at 120,000*. The noise of the labels alone (SD 0.54 pIC$_{50}$) makes an RMSE below 0.54 impossible, so **the free-floor extrapolation is invalid**: the second value lies far below the ceiling. With the floor fixed at the measured noise ceiling the fit has $b=0.33$ and predicts 0.763 at 1,200, 0.645 at 12,000, and 0.590 at 120,000: a 15-fold increase in data from the current size (800 to 12,000 molecules) reduces the error by about 0.13 pIC$_{50}$ (16%), and the next ten-fold (to 120,000) by about 0.05 (8%): *the benefit of more data of the same kind saturates just above the noise floor*.
2. **Cluster split: nearly flat.** RMSE is 1.29 for $n=50$–200 (equal to the standard deviation of the labels, 1.34: the model has barely learned anything transferable at these sizes) and 1.12 at $n=800$. The fitted exponent is 0.06 (free floor) or 0.10 (fixed floor), and the extrapolations at 120,000 molecules are 0.85–0.90. *Under a shift, volume of the same kind buys very little*: for the error to approach the noise floor the training set would have to cover the novel series, not merely be larger.
3. **Consequence.** A model's random-split learning curve says how well it will do on more *of the same*; the cluster-split curve says how well it will do on *new chemistry*. The gap between the two (0.34 at $n=800$) is the shift floor, and it is the quantity that matters for a team that will deploy the model on new scaffolds.

**Reading Experiment 2 (diversity versus volume).** With 100 training molecules, the test RMSE on held-out clusters is 1.413 when all training molecules come from one cluster, 1.436 (2 clusters), 1.411 (5), 1.378 (10), 1.306 (20) and **1.285 (30)**. (Standard errors are 0.04–0.05.) Three observations. (i) *A model trained on one chemical series is worse than predicting the mean on new series*: the label standard deviation is 1.34, and the RMSE is 1.41. (ii) Diversity helps: moving from 1 to 30 clusters at fixed $n=100$ reduces the error by 0.13 pIC$_{50}$. The reference is the cluster-split learning curve of Experiment 1, whose training molecules are drawn at random from the whole pool and so are already diverse: there $n=100$ gives 1.290 (equal to the 30-cluster row) and $n=800$ gives 1.116, an improvement of 0.17 for an eight-fold increase in volume. *A deliberately narrow training set costs about as much as the benefit of a six-fold increase in the volume of a diverse one.* (iii) The improvement is gradual until about 10–20 clusters, so *the number of clusters needed to cover a scaffold space may be large*.

!!! lens "Research lens: assumptions of the experiments"
    One target and one model class (a forest, whose learning curve differs from a neural network's); a short range of $n$ (50–800, one decade) for the curve fits (hence the parameter non-identifiability); clusters defined by a Tanimoto threshold; round-robin sampling within clusters; six or fifteen repetitions. A graph network or a pretrained model might have a steeper curve; the *method* (floors, splits, diversity) is the portable part.

---

## 47.3 Scaling evidence for biological foundation models

| Domain | What is known | Grade |
|---|---|---|
| Protein language models | Perplexity and contact accuracy improve smoothly with scale (ESM-2, 8M to 15B); fitness prediction peaks near 650M parameters in one analysis, with wild-type likelihood explaining the non-monotonicity (Chapter 34) | [[S]] |
| Structure prediction | Accuracy depends on the MSA, the data (PDB plus distillation), and architecture; no simple scaling law; improvements came from architecture and data, and co-folding generalization depends on training similarity (Chapter 35) | [[S]] |
| Genomic language models | Evo reported a scaling study that favored hybrid convolution–attention architectures at matched compute; Evo 2's larger models improve likelihood and several downstream tasks; for regulatory tasks, scale has not consistently beaten supervised baselines (Chapter 32) | [[S]] for likelihood; [[H]] for regulatory transfer |
| Single-cell foundation models | No established scaling law; in a controlled miniature, ten times more pretraining cells gave no gain in label transfer (Chapter 38) | [[H]] |
| Perturbation models | Large data sets exist (X-Atlas/Orion, Tahoe-100M); scaling of unseen-perturbation accuracy with data or parameters is not established (Chapter 39) | [[H]] |
| Molecular ML | Gains from pretraining and scale on property prediction are modest and data-set dependent (Chapter 37) | [[P]] |
| Protein design | Reported hit rates rose between 2020 and 2026 with models and data; a controlled scaling study at fixed targets does not exist (Chapter 58, Case 3) | [[H]] |

**A pattern.** Self-supervised *likelihood* improves with scale in the domains where data are abundant and the data distribution is stable (sequences); *downstream* tasks that require something the likelihood does not measure (fitness, regulatory activity, cell identity, interventions) improve when data of the right *kind* are added, not merely more of the same.

---

## 47.4 The economics of data

The question for a funder or a lab is the *value per dollar* of one more unit of data of kind $k$:
$$
\text{value}_k=\frac{-\Delta e_k}{\text{cost}_k},
$$
the reduction in expected error on the target task divided by cost. Kinds differ.

- **More of the same** (volume): diminishing returns as $n^{-(1+b)}$; the learning curve gives the value.
- **More diverse** (coverage of scaffolds, cell types, families, ancestries): the cluster-split curve and the diversity experiment give it; often the highest value when the target is novelty.
- **Less noisy** (replicates, better assays): reducing the label noise lowers the floor $c_\text{noise}$ and speeds learning; the cost per unit of variance reduction depends on the assay.
- **Of a different kind** (interventions instead of observations; pairs instead of unpaired modalities; within-locus instead of between-gene): can change the *identifiability* of the target (Chapters 31, 39, 40), a category change that no volume of the old kind provides.
- **Compute and modeling** (a better architecture or objective): worth considering when the curve is flat at large $n$ (model bias) or when the objective is mismatched (Chapter 58, Case 1).

**Order of magnitude of costs** (rough, to be re-estimated for your setting): a measured pIC$_{50}$ in a screening assay costs dollars to tens of dollars per compound; a deep mutational scan of one protein $10^4$–$10^5$ dollars; a genome-scale Perturb-seq screen of the order of $10^6$ dollars; a crystal structure of a protein–ligand complex $10^3$–$10^4$ dollars; one large pretraining run of a protein or DNA model $10^5$–$10^7$ dollars of compute. The point of the numbers is the *ratios* and the fact that a fixed budget buys very different numbers of units of different kinds.

!!! rhyme "Structural rhyme: learning curves with floors ↔ the noise ceiling (Chapter 1) ↔ the information bound on cross-modal retrieval (Chapter 40)"
    In each, performance saturates at a level set by information the data cannot supply (noise, private structure, shift). Past the knee, the efficient investment is in a different *kind* of data, not in more of the same.

---

## 47.5 Compute accounting

For transformers, training compute is approximately $C\approx6ND$ floating-point operations ($N$ parameters, $D$ training tokens). For a 40-billion-parameter DNA model trained on $9\times10^{12}$ nucleotides, $C\approx2\times10^{24}$ FLOPs, which at an effective $3\times10^{14}$ FLOP/s per accelerator is about $7\times10^{9}$ accelerator-seconds, roughly two million GPU-hours (about 230 GPU-years): a thousand accelerators for about three months. A small model (the 159k-parameter DNA LM of Chapter 32, trained on $2{,}500\times48\times128\approx1.5\times10^{7}$ tokens: $C\approx6\times1.6\times10^{5}\times1.5\times10^{7}\approx1.4\times10^{13}$ FLOPs) trains in about twenty minutes on a CPU. A reproducible report states:

1. **Parameters, tokens, and FLOPs** (or GPU-hours on a named device).
2. **Data size in effective units** (clusters, donors, studies, perturbations) as well as nominal.
3. **The metric at several scales** (at least three), so that a reader can fit a curve.
4. **Cost in dollars or energy** for the main run and for hyperparameter search (often larger than the final run).
5. **Seeds and variance** at the smallest scale.
6. **What was held fixed** (data, objective, architecture) when scaling.

---

## 47.6 Worked research examples

!!! example "Worked Research Example 47.1: Ten times more data, or ten times more compute?"
    **Situation.** A group has a potency model with a random-split RMSE of 0.77 on 800 training molecules (the setting of §47.2) and a cluster-split RMSE of 1.12. They can either pay for 10,000 more measured compounds (about $10^5$ dollars) or a 10-fold larger model/hyperparameter search (about $10^4$ dollars).

    **Question.** What should they do?

    **Reasoning (Expert Chain).**

    1. **L1 What is the target?** If deployment is on the *same* series, the random-split curve applies; if on new series, the cluster-split curve.
    2. **L3–L5 What does the floor say?** Random-split extrapolation with the noise floor fixed (0.54): about 0.65 at 12,000 molecules, i.e., 0.12 improvement for $10^5$ dollars; cluster-split extrapolation: 0.99 at 12,000, i.e., 0.12 improvement, still far above the noise floor.
    3. **L6 Bottleneck.** In the cluster-split regime the bottleneck is *coverage*, not volume: the diversity experiment shows that for 100 molecules, 30 clusters give 1.285 against 1.413 for a single cluster; extending coverage by selecting the next compounds from *unrepresented clusters* is worth more per compound than random volume.
    4. **L10 Experiments.** (a) Estimate how the cluster-split error depends on the number of clusters covered by the training set; (b) with a 10× larger model, test whether the learning curve changes (compute is cheap: 10% of the cost): if the curve is unchanged, the model is not the bottleneck; (c) choose the 10,000 compounds by a diversity-aware design (Chapter 46) rather than at random.
    5. **L11 Decision.** Buy the compounds, but *designed for coverage*; spend 10% of the budget first on the model test; if the larger model helps on the cluster split by more than 0.1, spend more on modeling.

    **Expert analysis.** The two options are not alternatives of the same kind: the cheap test of the model-size option removes the uncertainty about whether it matters; the data option should be re-specified from "10,000 more" to "10,000 chosen for coverage".

!!! example "Worked Research Example 47.2: A scaling study for a perturbation model, with no known answer"
    **Situation.** A consortium wants to know whether perturbation-response models improve with (i) more perturbations per cell type, (ii) more cell types, (iii) more cells per perturbation, or (iv) larger models. No scaling law is known.

    **Question.** How would you design the study?

    **Reasoning.**

    1. **Outcome.** The ceiling-normalized delta-correlation on held-out perturbations in (a) seen contexts and (b) held-out contexts (Chapter 39), with the nearest-neighbor and additive baselines.
    2. **Factorial design.** Subsample an existing large data set (genome-scale Perturb-seq in several cell lines; Chapter 39) to vary: perturbations $P\in\{100,300,1000,3000\}$; contexts $K\in\{1,2,4,8\}$; cells per perturbation $m\in\{25,50,100,200\}$; model size (three sizes). A fractional factorial design with 24 runs estimates main effects and the key interactions ($P\times K$).
    3. **Model.** A linear model of the metric on $\log P$, $\log K$, $\log m$, $\log N_\text{params}$ with interactions, reported with intervals; check the *residual floors* (noise ceiling by replicates; shift floor in held-out contexts).
    4. **Predictions to pre-register.** If the held-out-context metric depends mainly on $K$ and little on $P$, then diversity of contexts (a data-collection strategy) beats depth; if the metric depends on $\log m$ only through the noise ceiling, then cells per perturbation saturate quickly (Chapter 46's trade-off).
    5. **Cost model.** Convert the fitted slopes into metric gain per dollar for each factor using the cost table; the decision for the next data collection is the factor with the highest gain per dollar *at the current operating point*, not the steepest slope in general.
    6. **Caveat.** Subsampling changes *what is available* but not the experimental protocol; confirm the main prediction on a prospective data set.

    **What is not known.** The exponents, and whether context diversity or perturbation count dominates; the study estimates them.

---

## 47.7 Researcher's Notebook

!!! notebook "Researcher's Notebook: a scaling study you can run"
    1. **Measure the noise ceiling** from replicates before fitting any curve.
    2. **Run learning curves under the splits that match deployment** (random for same-distribution, cluster/clade/donor for novelty).
    3. **Fit with the floor fixed** at the measured ceiling and report a range of floors; never extrapolate below the ceiling.
    4. **Vary diversity at fixed $n$** and compare its effect with the effect of volume.
    5. **Estimate effective sample size** (clusters, trees, donors).
    6. **Report downstream metrics, not only loss**, at three or more scales.
    7. **Convert slopes to value per dollar** and decide the next data purchase.

    **What it teaches.** Scaling in biology is a measurement problem: the floors and the splits matter more than the exponents.

    **An open question to carry forward.** Diversity helped more than volume in the BACE experiment, but the number of clusters needed to reach a target error was not determined. Could one estimate the *effective dimension of chemical, cellular, or protein space* (the number of clusters needed to cover a distribution at a given resolution) and use it to forecast, before collecting data, how many diverse samples a novelty-robust model needs? Such a "coverage number" would play the role for diversity that the Chinchilla coefficient plays for volume.

---

## 47.8 Connections

- **Backward:** noise ceilings (Chapter 1); statistical learning (Chapter 7); scaling laws and compute-optimal allocation (Chapter 17); molecular data (Chapter 24); protein LMs (Chapter 34); cell FMs (Chapter 38); redundancy and effective sample size (Chapter 42); benchmarks, causality, shift, and design (Chapters 43–46).
- **Forward:** competing paradigms (Chapter 49); open problems (Chapters 50–52); funding and publishing (Chapter 59).

!!! takeaways "Key takeaways"
    1. Learning curves are $e(n)=c+an^{-b}$ with a floor $c$ made of label noise, shift, and model bias; the marginal value of data decays as $n^{-(1+b)}$.
    2. **A power-law fit with a free floor extrapolates through the measured noise floor**: on BACE the free-floor fit predicted an RMSE of 0.38 at 120,000 molecules against a noise floor of 0.54; with the floor fixed at 0.54, the prediction was 0.59 (0.645 at 12,000).
    3. **Under a shift (new chemical series) more data of the same kind barely helps**: the cluster-split RMSE went from 1.29 at $n\le200$ to 1.12 at $n=800$ (exponent 0.06–0.10); the gap to the random split (0.34 at $n=800$) is the shift floor.
    4. **Diversity beats volume for novelty**: at 100 training molecules, RMSE on held-out clusters fell from 1.41 (one cluster, worse than predicting the mean, 1.34) to 1.29 (30 clusters).
    5. Scaling evidence in biology: likelihoods improve smoothly with scale where data are abundant (sequences); downstream metrics need data of the right kind (fitness: peak at 650M parameters in one analysis; cell identity: no gain from 10× pretraining in the miniature) [[S]]/[[H]].
    6. Data economics: compare value per dollar of volume, diversity, quality, and different kinds (interventions, pairs, within-locus), at the current operating point.
    7. Report parameters, tokens, FLOPs, effective data size, metrics at three or more scales, cost including search, seeds, and what was held fixed.

---

## Further reading

- Kaplan, J. et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361.* Hoffmann, J. et al. (2022). Training compute-optimal large language models. *NeurIPS.* Hestness, J. et al. (2017). Deep learning scaling is predictable, empirically. *arXiv:1712.00409.* Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S. & Morcos, A. S. (2022). Beyond neural scaling laws: beating power law scaling via data pruning. *NeurIPS.*
- Lin, Z. et al. (2023). *Science* 379, 1123–1130 (ESM-2 scaling). Nguyen, E. et al. (2024). Sequence modeling and design from molecular to genome scale with Evo. *Science* 386, eado9336. Brixi, G. et al. (2026). Evo 2. *Nature.* Understanding language model scaling on protein fitness prediction (2025, *bioRxiv*; see Chapter 34).
- Cortes, C., Jackel, L. D., Solla, S. A., Vapnik, V. & Denker, J. S. (1993). Learning curves: asymptotic values and rate of convergence. *NeurIPS.* Figueroa, R. L., Zeng-Treitler, Q., Kandula, S. & Ngo, L. H. (2012). Predicting sample size required for classification performance. *BMC Med. Inform. Decis. Mak.* 12, 8.
- Kedzierska, K. Z. et al. (2025). *Genome Biol.* (single-cell). Ahlmann-Eltze, C., Huber, W. & Anders, S. (2025). *Nat. Methods* 22, 1657–1661 (perturbation). Raji, I. D. et al. (2021). AI and the everything in the whole wide world benchmark. *NeurIPS Datasets and Benchmarks* (on benchmark validity).
