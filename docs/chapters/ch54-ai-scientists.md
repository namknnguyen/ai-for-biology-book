# Chapter 54. AI Scientists: Agents, Automation, and the Verification Gap

!!! abstract "Chapter at a glance"
    **Motivation.** Since 2024, systems built from large language models have been asked to do what the rest of this book trains you to do: read the literature, state hypotheses, plan analyses and experiments, run them, and write up the result. Some of these systems have produced claims that survived wet-lab tests; others have produced polished text that no one could trust. A researcher in 2026 will use these tools, evaluate them, compete with them, and be asked whether to believe their output. This chapter gives an anatomy of research agents, dissects the main published systems with their evidence graded, and, most importantly, treats the *verification gap*: why an automated loop that generates and selects hypotheses manufactures confidence faster than it manufactures knowledge, and what protocol closes the gap. Two simulations quantify the mechanism: forking paths (a null dataset made to look significant by trying many analyses) and judge drift (a hypothesis-evolution loop whose own score rises far faster than the truth).
    **Prerequisites.** Chapters 1, 4, 17, 18, 43, 45, 46, 55–57.
    **You will be able to:** (1) describe the components of a research agent and where each fails; (2) read an "AI scientist" paper with the claim ladder (C0–C4) and the paper-dissection template; (3) quantify the multiple-comparisons burden of an agent's logged analysis paths and calibrate against it; (4) explain judge drift (Goodhart's law for hypothesis selection) and the conditions under which tournament scores track truth; (5) design a prospective evaluation of an AI-generated hypothesis; (6) decide, for a given task, where to automate and where to keep a human in the loop.

---

## 54.0 What is being claimed

"AI scientist" covers four different things, and most disagreements are about which one is meant.

| Level | What the system does | Example of the claim | Evidence needed |
|---|---|---|---|
| **Tool** | Answers questions, writes code, retrieves papers on request | "It found the relevant papers in seconds" | Spot checks |
| **Assistant** | Executes a multi-step analysis that a human specified | "It reproduced the analysis in a day" | Reproduction against a known result |
| **Collaborator** | Proposes hypotheses and plans, which humans select and test | "Two of three suggested drugs were active in organoids" | A pre-registered, wet-lab test; controls |
| **Autonomous investigator** | Runs the loop (hypothesis → experiment → analysis → revision) with little oversight | "It discovered a mechanism" | Prospective validation, independent replication, cost accounting, an audit of the trajectory |

The claim ladder of Chapter 1 maps onto this table. A system that retrieves and summarizes makes a C0 claim (it ran); one that reproduces a known result makes a C1 claim (the output matches a reference); a hypothesis that survives a wet-lab test makes a C2 or C3 claim *about that hypothesis*, and only a C4 claim (a mechanism or a general method that holds across contexts) justifies the word "discovery". The most common error in reading this literature is to let evidence at one rung stand for a claim at a higher one.

!!! lens "Research lens: three questions for any agent claim"
    (i) **What was the selection?** Out of how many hypotheses, analyses, or runs was the reported one chosen, and by whom (a human, a judge model, the agent itself)? (ii) **Who graded?** A human expert blind to the source, an automated judge, or the same system? (iii) **What would a human with the same tools, time, and budget have done?** Without a baseline of that kind, "the system found X" is as uninformative as "the model reached $R^2=0.7$" without the noise ceiling (Chapter 1).

---

## 54.1 Anatomy of a research agent

Every system in this chapter is an instance of one loop, with the components in different proportions.

```text
state ← {question, literature notes, data handles, hypotheses, results, log}
repeat:
    H ← generate(state)                    # LLM: propose hypotheses / analyses / designs
    H ← critique(H, state)                 # LLM (or another model): find flaws, rank
    plan ← select(H)                       # by judge score, tournament, or human choice
    result ← execute(plan)                 # code in a sandbox | a robotic lab | a simulator | a retrieval call
    state ← update(state, plan, result)    # memory: what was tried, what happened
until budget exhausted or stopping rule met
report ← synthesize(state)                 # a document with figures and citations
```

Each line hides a distinct research problem:

- **generate** is limited by what the base model has read and by the diversity of its samples. Language models are strongest at recombining *published* ideas; whether they can produce hypotheses that depart from the training distribution is an open empirical question [[H]].
- **critique** and **select** are where selection pressure enters. If the critic is a language model, its preferences include style, plausibility-as-text, and agreement with the generator's own priors. This is the loop's weakest link and the subject of §54.4.
- **execute** is the part that grounds the loop in reality, and the most heterogeneous: running Python in a sandbox (cheap, fast, and error-prone in subtle ways: wrong statistical test, silent data-leak), calling structure or design tools (as in the Virtual Lab below), or operating instruments (slow, expensive, and the true bottleneck in biology). Chapter 46's economics apply: a system that generates a thousand hypotheses per day and can test five per month is a *selection* problem, not a generation problem.
- **update** is memory. Long runs fail by losing the thread, by anchoring on early (wrong) results, and by quietly forgetting which analyses were already tried, which makes the multiple-comparisons accounting of §54.3 impossible unless the log is a first-class output.
- **synthesize** is persuasion. A fluent report is evidence of nothing, and in a system selected for human approval it is a force *against* truth (§54.4).

!!! rhyme "Structural rhyme: a research agent ↔ closed-loop Bayesian optimization (Chapter 46)"
    Both alternate *propose* and *evaluate* under a budget, with a surrogate (the agent's beliefs) steering the proposals. Everything Chapter 46 says about *exploitation of a surrogate's errors* applies: the agent's critic is a learned surrogate for "is this true and useful", and an optimizer pushing against a surrogate finds its blind spots. What the research agent adds is that the *search space itself* (hypotheses expressed in natural language) has no metric and no guarantee of coverage.

---

## 54.2 The systems, and what each one's evidence supports

This section dissects the main published systems using the book's template, abbreviated. Figures and dates are from the papers and their journal versions as of October 2026; company-reported numbers are marked as such. Several are described in the grading language of the book: [[E]] established, [[S]] strong, [[P]] plausible, [[H]] open hypothesis.

!!! paper "Paper dissection: Co-Scientist (Gottweis et al., Google; arXiv 2025; *Nature* 2026)"
    **Problem.** Generate novel, testable biomedical hypotheses and research plans from a goal stated in natural language.
    **Insight.** Treat hypothesis generation as a *test-time-compute* problem: spend inference compute on a structured generate, debate, and evolve loop with a tournament to rank candidates, rather than on a single prompt.
    **Architecture.** A supervisor agent coordinates specialized Gemini-based agents: *Generation*, *Reflection* (a peer-reviewer role that filters), *Ranking* (pairwise debates feeding an Elo rating), *Proximity* (similarity clustering to avoid duplicates), *Evolution* (modifies and combines top hypotheses), and *Meta-review* (feeds patterns of critique back into the other agents' prompts). Tools include literature search and, in places, specialized models.
    **Objective.** None in the gradient sense: selection by tournament Elo, a self-referential score whose validity depends on external checks.
    **Evaluation.** Expert preference ratings of outputs; benchmark questions with known answers (to show Elo tracks correctness); and three prospective biomedical applications with wet-lab follow-up: *drug repurposing for acute myeloid leukemia* (candidates inhibiting AML cell lines in vitro), *epigenetic targets for liver fibrosis* (tested in human hepatic organoids; an independent group's report found two of three suggested epigenetic-modifier drugs showed significant anti-fibrotic activity, one a repurposed anti-cancer drug, vorinostat), and *a bacterial gene-transfer mechanism* (capsid-forming phage-inducible chromosomal islands), where the system proposed a mechanism that matched a finding the experimental group had not yet published.
    **Why it worked (as far as shown).** The prospective tests show that the hypotheses produced by the loop are *enriched* for correct ones relative to a null of arbitrary suggestions, and in the third case that the loop can reach a conclusion from the literature that experts had reached from data. Test-time compute raised the Elo and the correctness on benchmark questions.
    **Assumptions.** The tournament's judge approximates scientific value; domain experts supplied the research goals and filtered; the validation experiments chose a handful of the outputs.
    **Limitations.** Three prospective tests are a small sample; no denominators are given that would let a reader compute the *precision at the top of the list* against a baseline of expert-generated candidates evidence of enrichment over arbitrary suggestions is [[S]], evidence of superiority over experts is [[H]]; the Elo is not an external measure (§54.4); the system inherits the literature's biases (well-studied genes and diseases are over-represented).
    **What followed.** Reproductions and open re-implementations; integration into data-aware agents; use inside labs as an idea-generation partner.
    **Unresolved.** Whether tournament-selected hypotheses are *more novel and correct than human-selected ones at equal cost*, and whether any gain survives a blind comparison against domain experts given the same literature access.

!!! paper "Paper dissection: Robin (Ghareeb et al., FutureHouse; arXiv 2025; *Nature* 2026)"
    **Problem.** Close the loop between literature-driven hypothesis generation and *data-driven* analysis, for a disease with no good drug.
    **Insight.** Chain specialized agents (literature search, hypothesis ranking, experiment design, data analysis) with a human performing the wet-lab steps, so that every hypothesis, experimental plan, analysis, and figure in the main text is produced by the system.
    **Application.** Dry age-related macular degeneration: the system proposed enhancing phagocytosis by retinal pigment epithelium as a therapeutic strategy, ranked candidate drugs, and selected ripasudil, a clinically used Rho-kinase inhibitor that, to the authors' knowledge, had not been proposed for this indication. A follow-up RNA-seq experiment designed and analyzed by the system showed upregulation of the lipid-efflux pump ABCA1, proposed as a possible mechanism and target.
    **Evaluation.** In vitro phagocytosis assays in human cells; RNA-seq. No animal or clinical evidence.
    **Why it worked (as far as shown).** The assay was a good *proxy for the hypothesized mechanism*, the candidate set was a library of already-characterized drugs (a space where the right answer is likely to be present), and the human-run experiments kept the loop honest.
    **Limitations.** A single indication; one experiment type; the step from an in-vitro phagocytosis effect to a disease-modifying drug is the long part of drug development (Chapter 37); and as in all drug repurposing, the hit rate of a good in vitro assay must be compared to what a pharmacologist screening the same library would obtain.
    **Unresolved.** How often the loop finds something useful in a *new* disease without a ready mechanistic hypothesis; the cost per validated candidate.

!!! paper "Paper dissection: Kosmos (Mitchener et al., Edison Scientific; arXiv Nov 2025)"
    **Problem.** Run long, data-driven discovery campaigns without losing coherence.
    **Insight.** A *structured world model* shared by a data-analysis agent and a literature agent, updated over cycles, lets a run last up to about 12 hours and 20 cycles.
    **Reported numbers (company and collaborator reported; [[P]]).** A run reads on the order of 1,500 papers and executes about 42,000 lines of analysis code; independent scientists rated about 79% of the report's statements as accurate; seven discoveries are highlighted across metabolomics, materials science, neuroscience, and statistical genetics, three of which independently reproduced findings from preprints or unpublished manuscripts the system could not access, and four of which are presented as novel; collaborators estimated that a run is equivalent to about six months of work and that the number of valuable findings grew roughly linearly with the number of cycles up to 20.
    **How to read it.** The 79% statistic means that roughly one statement in five in a report is *not* accurate, and the reader has no way to tell which. For a human collaborator, an error rate of that order is acceptable when every claim is checked and unacceptable when reports are treated as results; hence the verification protocol of §54.5. "Six months of work" is a subjective estimate, the type of figure that deserves an independent time-and-motion comparison. An independent evaluation of Kosmos in radiation biology exists (arXiv:2511.13825) and is exactly the kind of document to read next to a developer's report.

!!! paper "Paper dissection: Virtual Lab (Swanson, Wu, Bulaong, Pak & Zou; *Nature* 29 July 2025)"
    **Problem.** Interdisciplinary computational research that needs several kinds of expertise (immunology, protein modeling, computational biology) in sequence.
    **Insight.** A *principal-investigator* language-model agent runs meetings with specialist agents and a critic, with a human researcher giving high-level feedback; the agents choose and run tools.
    **Result.** The team designed a nanobody pipeline combining a protein language model (ESM), AlphaFold-Multimer, and Rosetta, producing 92 nanobody candidates against SARS-CoV-2 variants; experimental validation found a range of functional binders, and two had improved binding to the recent JN.1 or KP.3 variants while keeping binding to the ancestral spike.
    **How to read it.** The *pipeline* (Chapters 34–36) rather than the agent supplied the scientific content; the agents' contribution was composing and running it, with human feedback. The success here is a C2/C3-level result about two molecules. The fair comparison is to a human group with the same tools: the tools are the same, and the claimed advantage is speed and breadth of the composition.

!!! paper "Paper dissection: The AI Scientist (Lu et al. 2024) and AI Scientist-v2 (Yamada et al., Sakana AI; arXiv April 2025)"
    **Problem.** End-to-end machine-learning research: idea, code, experiments, manuscript, review.
    **Insight (v2).** An *agentic tree search* over experiments managed by a dedicated agent, with vision-language feedback on figures.
    **Result.** Of three fully AI-generated manuscripts submitted, under a protocol agreed with the organizers of a workshop at ICLR 2025 and with ethics approval, one received review scores that would have cleared the workshop's acceptance threshold (an average of 6.33); the authors withdrew it after review.
    **How to read it.** One acceptance out of three at a workshop (with a much higher acceptance rate than the main conference) is evidence that a *C0–C1-level* paper can pass a light review, not evidence of discovery; the authors themselves list naive ideas, implementation errors, duplicated figures, and hallucinated citations as common failures. For biology the lesson is about the *review* side: reviewers read a paper, not the logs, and a fluent report passes if no one re-runs the code.

!!! bio "Biology for modeling: why the wet lab is the bottleneck"
    **What is it?** An experiment is a chain: reagent preparation, a perturbation, an incubation, a readout, a quality-control step. **How is it measured?** Each step is a hand-off between instruments and people, with tacit knowledge (which batch of antibody works, which cell passage behaves). **What can ML not observe?** The tacit state: the lab's undocumented know-how, and the failures never written up. An agent that learns from the literature sees only published successes. Robotic labs (and the early robot scientists *Adam* and *Eve*, King and colleagues, 2004–2018) automate the *execution* of a narrow class of assays, but each assay needs months of engineering. The practical consequence: in biology, the throughput of an AI scientist is set by assay automation and by experimental cost, not by the speed of the language model, and the main use of the agent is to *choose which experiments to run* (Chapter 46), which makes the quality of its selection the quantity that matters.

---

## 54.3 Forking paths at machine speed

An agent can run in an hour as many analyses as a human analyst runs in a year. Every analysis has choices (which predictor, which subgroup, which transform, which outlier rule, which covariates), and a report that describes only the path that "worked" has the statistical properties of the best of many tests. This is the *garden of forking paths* (Gelman and Loken) and the *p-hacking* of Simmons, Nelson and Simonsohn, now without fatigue to limit it.

**The arithmetic.** If a dataset is null and the agent tries $M$ independent tests at level $\alpha$, the probability of reporting at least one "hit" is

$$
1-(1-\alpha)^M ,
$$

which is 0.23 for $M=5$, 0.64 for $M=20$, and 0.99 for $M=100$. Variants of an analysis are not independent, so the effective number of tests $M_\text{eff}$ is smaller than the number tried: it is set by the correlations among the statistics. Two corrections apply: *Bonferroni* ($p\cdot M$) is valid for any dependence but conservative when variants are correlated; **max-T permutation** is exact: shuffle the outcome, rerun *the entire logged set of analyses*, and record the *smallest* p-value each time; the 5th percentile of those minima is the threshold for "the best of everything tried" to be called significant. The procedure treats the agent's search as the statistic. Its price is power: the same effect must be larger to clear a threshold set by 200 variants than a threshold set by one.

The experiment below uses 60 samples, 20 correlated candidate predictors, and a binary covariate, and defines 200 analysis variants (predictor × five subgroups × two transforms). Half of the simulated datasets are null; in the other half predictor 0 truly affects the outcome ($\beta=0.45$).

```python
--8<-- "code/ch54_agent_verification.py"
```

```text
== 1. One dataset, many analysis variants; the agent reports the variant with the smallest p-value ==
n = 60 samples, 20 candidate predictors (pairwise correlation 0.3), 1 binary covariate; variants = predictor x subgroup (5) x transform (2) = 200
variants tried M   P(report p < 0.05) on NULL data   with Bonferroni over M   fraction of reported hits that are real (data are 50% null / 50% real effect)
           1              0.055                           0.055                    0.713
           5              0.244                           0.049                    0.670
          20              0.565                           0.044                    0.605
          50              0.821                           0.047                    0.543
         100              0.927                           0.052                    0.519
         200              0.983                           0.037                    0.504

min-p calibrated against its own null distribution over all 200 logged variants (threshold 0.00031, not 0.05): false-positive rate 0.053, power for the real effect 0.524
for comparison, ONE pre-specified test (predictor 0, all samples, raw): false-positive rate 0.051, power 0.923

== 2. A hypothesis-evolution loop selected by a judge: the judge's score rises faster than the truth ==
hypothesis i has true value v_i ~ N(0,1) (probability-it-is-true-and-useful on a logit-like scale) and a 'persuasiveness' f_i ~ N(0,1) unrelated to truth;
the judge sees J = v + w*f + noise (noise SD 1.0 per comparison, 20 comparisons per hypothesis); each round, every survivor is mutated 5 ways:
a mutation changes v by N(0, 0.15^2) (truth is hard to move) and f by N(0, 0.6^2) (persuasiveness is easy to move); the top 20% by tournament rank survive and are re-expanded.
judge weight on persuasiveness w   round   mean judge score of the pool   mean TRUE value of the pool   mean persuasiveness
                 0.0                      0                  -0.02                      -0.02              -0.02
                 0.0                      2                   2.02                       2.02               0.05
                 0.0                      4                   2.74                       2.74               0.10
                 0.0                      8                   3.65                       3.65               0.06
                 1.0                      0                  -0.04                      -0.02              -0.02
                 1.0                      2                   3.18                       1.30               1.88
                 1.0                      4                   5.39                       1.59               3.80
                 1.0                      8                   9.50                       1.86               7.64

selecting 3 hypotheses for experiments out of N candidates with the judge (w = 1.0, no evolution): mean TRUE value of the selected three, 400 repetitions
N candidates   judge-selected   random   oracle (best 3 by true value)
        10           0.59        -0.01     1.06
        30           0.99         0.00     1.67
       100           1.26         0.00     2.20
       300           1.49        -0.00     2.60
      1000           1.71        -0.00     3.00
```

**Reading Part 1.** (a) With one pre-specified test the false-positive rate is 0.05 and power is 0.92 for the real effect. (b) With the best of 200 variants the false-positive rate on null data is 0.98; even five variants raise it to 0.24. (c) Reporting the best p-value yields hits that are decreasingly informative: the fraction of hits from datasets that truly contain an effect falls from 0.71 to 0.50 (no better than a coin flip) as variants multiply. (d) Bonferroni and the permutation threshold restore the 5% error rate, and the cost is visible: power for the real effect drops from 0.92 (pre-specified) to 0.52 (search over 200 variants, calibrated). *The agent's search is not free; it must be paid for in the statistics.* That cost cannot be avoided by an agent being clever, only by (i) pre-specifying the analysis, (ii) logging and correcting for every path, or (iii) confirming on **new data**, which is exact and cheap in relative terms: a hit that survives a pre-specified replication on 60 fresh samples was subjected to a single test.

!!! lens "Research lens: what an honest agent log contains"
    For every reported result: the number of analyses *attempted* (not reported), the random seeds, the data versions and filters, the code that produced each figure, the prompts and model versions that generated the analysis, and the choice that led from an exploratory to a confirmatory analysis. A report from an agent that does not output the denominator of attempts has failed the first requirement of Chapter 56.

---

## 54.4 Judge drift: Goodhart's law for hypotheses

The second mechanism concerns selection among hypotheses. Suppose a judge $J$ scores each hypothesis, and the loop keeps what scores well and then mutates it (the *Evolution* agent of Co-Scientist, or any generate-critique-refine loop). Write the judge's score as

$$
J = v + w\,f + \varepsilon,
$$

where $v$ is the hypothesis's true value (probability that it is true and useful), $f$ is a feature the judge likes that is *not* value (persuasiveness, fluency, resemblance to published claims, mention of fashionable mechanisms), $w$ is the weight the judge puts on it, and $\varepsilon$ is noise. Nothing in this model requires the judge to be *bad*: language-model judges and human reviewers have non-zero $w$.

**Selection alone.** If one chooses the top $k$ of $N$ by $J$, the mean true value of the selection grows with $N$ but saturates: the winner's curse of Chapter 43. The best candidates by $J$ are enriched for high $v$, high $f$, and high $\varepsilon$ in proportions set by their variances.

**Selection plus variation: the breeder's equation.** In an evolution loop, each round selects survivors and *re-expands* them with mutations (here: changes to the hypothesis proposed by the generator). The response of a trait to selection is the selection differential times the fraction of the *heritable* variation in $J$ that belongs to that trait. If a mutation changes $v$ with standard deviation $\sigma_v$ and $f$ with $\sigma_f$, the share of each round's gain that is real is approximately

$$
\frac{\Delta \bar v}{\Delta \bar J}\;\approx\;\frac{\sigma_v^{2}}{\sigma_v^{2}+w^{2}\sigma_f^{2}} .
$$

Truth is hard to move (a mutation that improves the hypothesis's *correctness* requires new information about the world); persuasiveness is easy to move (rewording, adding mechanism, citing more). Even with an unbiased-looking judge ($w=1$), if $\sigma_f$ is four times $\sigma_v$ the real share of the gain is $1/(1+16)\approx 6\%$ and the judge's score therefore overstates progress by about $17\times$. The *tournament Elo goes up* because the loop is working; whether *truth* goes up depends on $w\sigma_f/\sigma_v$.

!!! math "Derivation: the share of real gain"
    Let the population have heritable (mutational) variation with $\text{Var}(\Delta v)=\sigma_v^2$ and $\text{Var}(\Delta f)=\sigma_f^2$, independent, and let selection operate on $J = v + wf$. By the breeder's equation (Lande), the per-round response of trait $z$ to selection on $J$ is $\Delta\bar z = \text{Cov}_\text{mut}(z, J)\,\beta$, where $\beta$ is the selection gradient. Then $\text{Cov}_\text{mut}(v,J)=\sigma_v^2$ and $\text{Cov}_\text{mut}(f,J)=w\sigma_f^2$, so $\Delta\bar J=\Delta\bar v+w\Delta\bar f \propto \sigma_v^2 + w^2\sigma_f^2$ and the share of $\Delta\bar J$ due to $v$ is $\sigma_v^2/(\sigma_v^2+w^2\sigma_f^2)$. The noise $\varepsilon$ reduces the *magnitude* of the response (the selection gradient) but not the shares. $\square$

The simulation (Part 2 above) tests this with $\sigma_v=0.15$, $\sigma_f=0.6$: the predicted share is $0.0225/(0.0225+0.36)=0.059$ for $w=1$. Between rounds 2 and 8 the pool's judge score rises by $9.50-3.18=6.32$ and its true value by $1.86-1.30=0.56$, a real share of $0.09$, close to the prediction (the first rounds also draw on the standing variation of the initial pool, which is why the overall share, $1.86/9.50=0.20$, is higher).

**Reading Part 2.** (a) With an aligned judge ($w=0$) the pool's true value rises steadily (to 3.65 in eight rounds, in units where the starting population has SD 1): the loop *works*. (b) With $w=1$ the judge's score rises to 9.5 while the pool's true value rises only to 1.9, and persuasiveness to 7.6: **about 80% of the apparent progress is not progress**. (c) For *selection without evolution*, the selected three of $N$ candidates improve from 0.59 (N=10) to 1.71 (N=1000) against an oracle of 1.06 to 3.00: more candidates help, at a rapidly diminishing rate, and the judge never gets to the oracle.

**What corrects it.** (i) *Grounding*: a judge that sees data or executes a check is not a pure text judge (and $w$ falls). (ii) *External calibration*: periodically spend experiments to measure the correlation between judge score and outcome *within the selected pool*, which is exactly the number that drifts. (iii) *Diversity pressure*: the Proximity agent in Co-Scientist is a diversity mechanism; the larger the pool and the weaker the selection, the slower the drift. (iv) *Adversarial critics trained on past failures*: the critique distribution must be anchored in outcomes, not only in text. (v) *Humans as the final judge*, which does not remove $w$ (humans like persuasive text too) but replaces one bias with another that is at least partly independent.

!!! rhyme "Structural rhyme: judge drift ↔ reward hacking in RLHF (Chapter 17) ↔ the adaptive reuse of a test set (Chapter 43)"
    The same statement in three settings: *an optimizer that selects on a proxy exploits whatever part of the proxy is easiest to move.* In preference tuning the proxy is a reward model; in benchmarks it is a finite test set queried repeatedly; here it is a judge. The remedies in all three are the same: hold out an *independent* check, limit the number of adaptive queries, and measure the proxy-to-truth correlation where the optimizer is looking.

---

## 54.5 Evaluating an AI scientist

A protocol, in increasing order of cost and credibility:

1. **Retrodiction with a date cutoff.** Provide the literature up to a date, ask for hypotheses about an outcome that was published later, and score blind. It has a leak problem: a model trained after the cutoff may have seen the answer; use models and corpora with a verifiably earlier cutoff, and check for contamination (Chapter 43).
2. **Known-answer benchmarks for the parts.** *BixBench* (FutureHouse, 2025) offers 296 open-answer questions from 53 real bioinformatics analysis scenarios; at its release, frontier models with a basic agent harness reached about 17% accuracy on open answers. Such benchmarks measure the *execute* step and are saturating quickly, so cite the date. Prefer benchmarks whose answer keys were rederived and audited (an audit of the benchmark's scoring is as important as the score, Chapter 43).
3. **Blind expert comparison against human-generated alternatives**, with a pre-registered rubric, using hypotheses from both sources shuffled together. This is the missing experiment in most reports.
4. **Prospective experimental validation with a pre-specified denominator.** Choose $k$ hypotheses by a rule *before* looking at outcomes (the top $k$ by the system's own ranking, *plus* $k$ chosen by experts and $k$ random from the system's pool), test them in the same assay with controls, and report all of them. The statistic is the **enrichment of true hits in the top of the ranking over the random and expert baselines**.
5. **Cost accounting.** Total cost (compute, experiments, expert time for verification) *per validated finding*, versus the same for a human-led project. A system whose reports each need a week of expert verification has not saved the week.
6. **Independent replication** of the *discovery* in a different lab or dataset, which turns C3 into C4.

Quantities to demand in any report: the number of hypotheses generated and tested, the fraction of generated statements that were checked and found incorrect, the number of analysis variants run, the identity and calibration of the judge, the share of steps done by humans and the share of those that were corrections, and a full trajectory log.

!!! openproblem "Open problem: measuring the marginal value of an AI collaborator"
    There is no accepted design for estimating how much an AI system adds to a research group's *validated discoveries per unit cost*, over and above what the same group does with ordinary tools. The measurements that exist are mostly demonstrations (an interesting output on a problem chosen by the developers). **Diagnostic questions:** Could one run a randomized controlled trial in which research groups are assigned to use or not to use a system on a matched set of problems (such as rare-variant interpretation tasks or assay design), with outcomes scored by blind experts at 6 and 24 months? What outcomes are measurable (papers? validated hits? experiments avoided?) and which are gameable? Which tasks have already moved from "assistant" to "tool", and how did we notice?

---

## 54.6 Where agents help, where they mislead, and safety

**Where they help now** [[S]]: literature triage and structured summaries with citations checked by a tool; writing and debugging analysis code (reproducibility, pipelines); assembling multi-tool computational pipelines (the Virtual Lab pattern, §54.2); drafting experimental protocols; proposing *candidate lists* in domains where a cheap assay can screen them (drug repurposing against a library; variant prioritization with functional follow-up); and rubber-duck critique of one's own plan (a language model as a weak reviewer, §54.4 notwithstanding).

**Where they mislead** [[S]]: any analysis whose correctness depends on a subtle statistical or biological fact (pseudoreplication, batch confounding, circular selection; Chapters 4, 25, 30), because the agent can run the wrong test confidently and the output looks the same; citations (hallucinated or mismatched) unless every reference is machine-verified against a bibliographic database; novelty claims (a model that has not read a paper will say its idea is new); interpretation of a result in light of a hypothesis that has already been committed to (anchoring across cycles).

**Safety and dual use.** An agent with tools to design sequences or synthesize molecules lowers the cost of capability in both directions. Chapter 59's biosecurity framework applies: restrict tool access by risk class (screen design requests against known hazard databases; keep synthesis ordering behind human review; log every tool call), publish evaluations of *refusal and screening* rather than only capability, and treat an agent's access to a lab as a privilege that scales with its audited reliability. A *bad* system is not only wrong; it is wrong at scale, with authoritative prose, which erodes the literature that future models train on (the pollution problem raised in §54.8).

---

## 54.7 Worked research examples

!!! example "Worked Research Example 54.1: Auditing a claimed autonomous discovery"
    **Situation.** A company releases a report in which an agent, given a public transcriptomic cohort of 600 patients with a neurodegenerative disease, "discovers that gene $G$ is a driver of disease progression", supported by a figure with $p=3\times10^{-5}$ and an enrichment analysis, plus a literature-based mechanism. They say a collaborator confirmed the finding by qPCR in patient samples.

    **Question.** What would you ask for before believing it, and what is the strongest claim the evidence could support?

    **Reasoning (Expert Chain).**

    1. **L1 Problem.** The claim is causal ("driver"); the evidence is an association with progression in observational data, plus a qPCR confirmation of *expression difference*. The ladder: C1 (association reproduced by qPCR) at best; "driver" is C3/C4.
    2. **L3–L5 Assumptions and failure modes.** (a) *Forking paths*: how many genes, covariates, and subgroups were examined? A transcriptome-wide analysis of 20,000 genes has $M_\text{eff}\gg 1000$ before any analysis variants, and $p=3\times10^{-5}$ is *not* significant at the Bonferroni threshold $2.5\times10^{-6}$ for 20,000 genes. (b) *Confounding by cell composition* (Chapter 25: a gene "associated with progression" may report neuronal loss), by batch, by post-mortem interval, by medication. (c) *Reverse causation* (a consequence of degeneration). (d) *The qPCR confirmation tests the wrong thing*: it measures that expression differs in samples, which the discovery data already said.
    3. **L6 Bottleneck.** The step from correlation to cause needs *intervention*: genetic evidence (Mendelian randomization with eQTLs, Chapter 26: an instrument for $G$'s expression), perturbation in a model system (Chapter 39), or natural experiments.
    4. **L10 Experiments to request.** The log of all analyses tried; the Bonferroni/permutation-adjusted p-value; adjustment for estimated cell-type proportions; replication in an independent cohort; MR with a cis-eQTL instrument; a perturbation of $G$ in a relevant cell model with a pre-specified readout; a blinded comparison against a human analyst given the same cohort.
    5. **L11 Interpretation.** If the adjusted result survives and replicates, claim "$G$'s expression is associated with progression independently of composition in two cohorts" (C1 to C2). Add "consistent with a causal role by MR" only if the instrument is valid (C3 candidate). "Driver" requires an intervention.

    **Expert analysis.** The report's selling point (an autonomous discovery) is the part the evidence supports least; the useful part (a candidate gene that survives correction and replication, if it does) is a good, ordinary result.

!!! example "Worked Research Example 54.2: A prospective test of an AI hypothesis generator, with no known answer"
    **Situation.** Your lab can run 24 CRISPRi perturbations with a high-content readout per month. A hypothesis generator suggests 300 genes as regulators of a cellular phenotype of interest. No one knows the base rate of true regulators among them, or whether the system beats your own judgment.

    **Question.** How would you use the 24 experiments to learn *both* about the biology and about the generator?

    **Reasoning.**

    1. **Define arms.** (a) top 8 by the system's rank; (b) 8 genes chosen by two lab experts blind to the system's ranks, from the same 300; (c) 8 random genes from the same 300. A pool restricted to the system's list estimates the *within-list* enrichment; add 4 non-listed positive controls and 4 negative controls if you can afford them (the 24 includes them, reducing arms to 6 each).
    2. **Power.** With 6 per arm, the minimum detectable difference in hit rate is large (if the random hit rate is 10%, even a perfect system's top 6 detects an enrichment of 5/6 vs 0.6/6, i.e., *only a very strong effect*). State this before the experiment, and decide that the result is *screening evidence to be accumulated across months*, not a verdict. Sequential design: a Bayesian model of the hit rate in each arm updated each month (Chapter 46).
    3. **Measurement.** Use the readout's noise ceiling (Chapter 25) to define "hit"; include replicate guides; pre-register the effect-size threshold.
    4. **Selection bias.** The experts see the system's list; to keep their choice independent, give them the list *without* ranks, the order of presentation randomized.
    5. **Decision rule.** Pre-specify: if after 96 experiments the system's top-ranked arm has a hit rate exceeding the random arm's by at least 15 percentage points with a posterior probability above 0.9, adopt the system as a *prioritizer* for the pool, and continue to track its precision. Otherwise use it only for idea generation.

    **What is not known.** Whether any hypothesis generator provides enrichment *beyond* that of an expert using a good database, in a field without a ready-made library of known regulators; and whether this changes when the generator is given the lab's own data. This is the experiment the field needs, and each lab that runs it and reports the denominators contributes to answering it.

---

## 54.8 Researcher's Notebook

!!! notebook "Researcher's Notebook: using an agent without being used by it"
    1. **Write the question and the analysis plan before calling any agent**, then compare its plan to yours: disagreements show where each of you is making an assumption.
    2. **Ask for the denominator**: every analysis attempted, every hypothesis considered. If the agent cannot provide it, treat its reported result as the best of an unknown number of tries (§54.3).
    3. **Re-run the central analysis yourself** with a different method (a permutation test; a held-out split) and check that the effect is of the same size.
    4. **Machine-verify every citation** (DOI resolves, title and authors match, the cited page contains the claim).
    5. **Look for text-only judgment**: if the system ranked hypotheses without running any check, estimate $w$ yourself by taking five of its top-ranked and five of its bottom-ranked hypotheses and assessing them blind.
    6. **Randomize** which of your experiments follow the system's advice, and keep the results (§54.7, Example 2): your own lab's data on the generator's precision is the most valuable thing you can accumulate.
    7. **Record** model name, version, date, prompts, and tools in the log; model behavior changes between versions.

    **What it teaches.** An agent is a very fast, very fluent junior colleague with unusual breadth and no accountability. The scientist's job is to design the accountability.

    **An open question to carry forward.** Pretraining corpora will increasingly contain AI-written scientific text, including incorrect claims in confident prose. Could one measure the *contamination of the literature itself* (the fraction of claims in new papers whose only support is another AI-generated paper) and design retrieval systems that weight claims by the independence of their evidence? What would a citation graph that tracks *independent replication* rather than *mention* look like, and who would build it?

---

## 54.9 Connections

- **Backward:** the claim ladder and Expert Chain (Chapter 1); multiple testing and the arithmetic of selection (Chapter 4); alignment of proxies and reward hacking (Chapter 17); evaluation and winner's curse (Chapter 43); causal claims from observational data (Chapter 44); active learning and design (Chapter 46); idea generation and evaluation (Chapters 55, 56); critique of papers (Chapter 57).
- **Forward:** open-ended case studies in which the reasoning, not the tool, is the point (Chapter 58); the biosecurity framework and publication ethics (Chapter 59).

!!! takeaways "Key takeaways"
    1. "AI scientist" spans tool, assistant, collaborator, and autonomous investigator; the evidence required rises with each level, and claims are routinely made one rung above their evidence.
    2. Published systems (Co-Scientist, Robin, Kosmos, the Virtual Lab, AI Scientist-v2) show that agent loops can *enrich* for correct hypotheses and compose multi-tool pipelines, with wet-lab confirmation in specific cases; none provides a head-to-head, denominator-reporting comparison against expert humans with equal resources [[S]] for enrichment, [[H]] for superiority.
    3. **Forking paths at machine speed**: in simulation the best of 200 analysis variants gave a false-positive rate of 0.98 on null data (0.24 for five variants), and the share of reported hits from datasets with a real effect fell from 0.71 to 0.50; max-T permutation over the logged set restores 5% error and costs power (0.92 → 0.52).
    4. **Judge drift**: in a judge-driven evolution loop with a persuasiveness bias of weight 1, the judge's score rose to 9.5 while the true value rose to 1.9 (an aligned judge reached 3.65); the real share of gain is $\sigma_v^2/(\sigma_v^2+w^2\sigma_f^2)$, which is small whenever truth is harder to move than persuasion.
    5. The remedies are structural: log every attempt, calibrate against the full logged set, ground judges in checks, hold out external tests, and measure judge-to-outcome correlation within the selected pool.
    6. In biology the bottleneck is experiments, not text generation; the agent's most valuable job is *choosing which experiments to run* (Chapter 46), and the quality of that choice should be tested prospectively against expert and random baselines.
    7. Treat every agent output as the best of an unknown number of attempts until the denominator is shown.

---

## Further reading

- Gottweis, J. et al. (2025). Towards an AI co-scientist. *arXiv:2502.18864*; and the *Nature* (2026) version, "Accelerating scientific discovery with Co-Scientist". Guan, Y. et al. (2025). AI-assisted drug re-purposing for human liver fibrosis. *Adv. Sci.* (independent validation in hepatic organoids).
- Ghareeb, A. E. et al. (2025). Robin: a multi-agent system for automating scientific discovery. *arXiv:2505.13400*; *Nature* (2026), "A multi-agent system for automating scientific discovery".
- Mitchener, L. et al. (2025). Kosmos: an AI scientist for autonomous discovery. *arXiv:2511.02824.* For an independent assessment, see the radiation-biology evaluation, *arXiv:2511.13825*.
- Swanson, K., Wu, W., Bulaong, N. L., Pak, J. E. & Zou, J. (2025). The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* (29 July 2025).
- Lu, C. et al. (2024). The AI Scientist: towards fully automated open-ended scientific discovery. *arXiv:2408.06292.* Yamada, Y. et al. (2025). The AI Scientist-v2: workshop-level automated scientific discovery via agentic tree search. *arXiv:2504.08066.*
- Mitchener, L. et al. (2025). BixBench: a comprehensive benchmark for LLM-based agents in computational biology. *arXiv:2503.00096.*
- King, R. D. et al. (2004). Functional genomic hypothesis generation and experimentation by a robot scientist. *Nature* 427, 247–252. King, R. D. et al. (2009). The automation of science. *Science* 324, 85–89.
- Gelman, A. & Loken, E. (2013). The garden of forking paths: why multiple comparisons can be a problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis was posited ahead of time (unpublished manuscript); Gelman, A. & Loken, E. (2014). The statistical crisis in science. *Am. Sci.* 102, 460–465. Simmons, J. P., Nelson, L. D. & Simonsohn, U. (2011). False-positive psychology. *Psychol. Sci.* 22, 1359–1366. Westfall, P. H. & Young, S. S. (1993). *Resampling-Based Multiple Testing.* Wiley (max-T).
- Lande, R. (1979). Quantitative genetic analysis of multivariate evolution, applied to brain:body size allometry. *Evolution* 33, 402–416 (the multivariate breeder's equation). Manheim, D. & Garrabrant, S. (2018). Categorizing variants of Goodhart's law. *arXiv:1803.04585.* Gao, L., Schulman, J. & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML.*
