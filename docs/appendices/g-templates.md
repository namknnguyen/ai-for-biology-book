# Appendix G. Research Templates

Every template below is used in the book. They are meant to be copied, filled in, and kept next to your work. Each is a *working document*: short, honest, and revisable. Where a template is introduced, the chapter is given.

---

## G.1 The Expert Chain worksheet (Chapter 1)

| Link | Prompt | Your entry |
|---|---|---|
| **L1 Problem** | What is the question, in one sentence, with a measurable success criterion? | |
| **L2 Existing approaches** | What are the three strongest current approaches, including the best classical baseline? | |
| **L3 Assumptions** | List at least ten, including independence, stationarity, additivity, identifiability. | |
| **L4 Why they might work** | What structure in the data or biology do the approaches exploit? | |
| **L5 Failure modes** | Where do they fail? Cite the Four Gaps (measurement, objective, inference, generalization). | |
| **L6 Bottlenecks** | Is the limit data, objective, capacity, context, measurement, or identification? | |
| **L7 Open questions** | What is not known? State as questions. | |
| **L8 Hypotheses** | If/then/because statements, with predictions. | |
| **L9 Candidate solutions** | Methods, data, evaluations that would test each hypothesis. | |
| **L10 Experiments** | The cheapest decisive experiment for each, with the kill condition. | |
| **L11 Interpretation** | What would each outcome mean? What would you believe afterwards? | |
| **L12 New directions** | What does a success or failure open up? | |

---

## G.2 The Attack Sheet (Chapter 55)

**Problem sentence and metric:**

**Current best approach and score (with noise ceiling and strongest classical baseline):**

**Four-Gap diagnosis:** G-M: ____  G-O: ____  G-I: ____  G-G: ____

**Assumptions (at least ten; mark each *known / approximate / convenient / unexamined*):**

| # | Assumption | Status | What if false? | Cheapest test |
|---|---|---|---|---|
| 1 | | | | |

**Ideas by attack** (three per relevant attack, each *If [change], then [measurable consequence], because [mechanism]*):

| Attack | Idea | Cheapest decisive experiment | Kill condition |
|---|---|---|---|
| A1 Assumption | | | |
| A2 Representation | | | |
| A3 Objective | | | |
| A4 Data | | | |
| A5 Evaluation | | | |
| A6 Scale | | | |
| A7 Cross-domain | | | |
| A8 Reformulation | | | |
| A9 Biological constraint | | | |
| A10 Biological discovery | | | |

**Crosses (five cells of the $10\times10$ product table):**

---

## G.3 The claim–evidence table (Chapters 1, 57, 59)

| Claim (as it will be written) | Rung (C0–C4) | Evidence | What would have to be true | Baseline ladder and ceiling | Unit of replication and $n$ | Gap | Rung after planned work |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

*Rules.* (i) Abstract verbs must match rungs: *predicts* (C1–C2), *is associated with*, *is consistent with*, *causes*, *designs* (prospective). (ii) Every number in the text must be traceable to a row. (iii) Rows with a non-empty *Gap* column become limitations or planned experiments.

---

## G.4 Triage scoresheet and expected information gain (Chapter 56)

**Idea (If/then/because):**

| Criterion | Score 1–5 | Evidence |
|---|---|---|
| Novelty after prior-art search (three names, five searches; closest paper and its rung on the prior-art ladder) | | |
| Meaning: decision changed; Claim-Ladder reach | | |
| Feasibility: data, compute ($6ND$), expertise, access | | |
| Time to first signal | | |
| Decisiveness: EIG per unit cost | | |
| Downside value: informative null, reusable asset | | |
| Risk: ethics, biosecurity, reproducibility | | |

**Hypotheses and priors:**

| Hypothesis | Prior |
|---|---|
| H1 | |
| H2 | |
| H3 | |

**Experiments, likelihoods, and costs:**

| Experiment | $P(\text{outcome}=1\mid H_1)$ | $\mid H_2$ | $\mid H_3$ | Cost | EIG (bits) | EIG per cost |
|---|---|---|---|---|---|---|

(Compute with `code/ch56_information_gain.py`; perturb likelihoods by $\pm0.1$ and record whether the ranking changes.)

**Decision statement:** *If [result], I will [action]. If [opposite result], I will [other action].*

**Kill criteria and checkpoint dates:**

---

## G.5 Pre-mortem (Chapter 56)

*Imagine the project is finished and failed. The five most likely reasons, each with a cheap mitigation to schedule now:*

| Reason (from: saturated or leaky benchmark; weak baseline; effect within the noise ceiling; key variable unidentified; no biological transfer; resources late; scooped) | Probability | Mitigation | Date |
|---|---|---|---|

---

## G.6 Pre-registration of an analysis (Chapter 59)

1. **Primary question and hypothesis:**
2. **Datasets, versions, and access:**
3. **Locked split** (how produced; near-duplicate removal; both orientations for DNA; family/cluster/donor/ancestry/time):
4. **Baseline ladder and tuning budget for each:**
5. **Noise ceiling** (how estimated):
6. **Primary metric(s) and unit of replication; effective $n$:**
7. **Statistical analysis** (paired bootstrap over units; multiple-comparison rule):
8. **Stratifications to report** (similarity to training, effect size, mappability, depth, ancestry):
9. **Success threshold and decision rule:**
10. **Exploratory analyses** (labeled as such):
11. **Seeds, compute budget, software environment, code and data release:**

---

## G.7 Paper dissection (Chapter 57)

*(Problem; key insight; architecture/method with tensor shapes; objective; data and measurement process; training; evaluation (metrics, baselines, ceiling, unit of replication); results with numbers; why it worked; assumptions; limitations; what followed; unresolved with attacks.)*

**Red flags found** (numbers from the catalog in §57.4):

**Two attacks and their cheapest tests:**

**One question for the authors:**

---

## G.8 Reading log entry (Chapter 57)

| Field | Entry |
|---|---|
| Citation, date, version, code/data link | |
| Headline claim and rung | |
| Objective and information-flow sketch | |
| Split, baselines, ceiling, unit of replication | |
| Red flags | |
| Three key assumptions | |
| Two attacks and tests | |
| Question for the authors | |

---

## G.9 Dataset datasheet (Chapters 25, 43, 59)

*(Motivation; composition (units, classes, sizes); the measurement process (assay, chemistry, pipeline, mappability, noise); collection (consent, ancestry, sites); preprocessing and labels (who or what produced them); splits and leakage checks; uses and limitations; maintenance and versioning; ethical and privacy considerations.)*

---

## G.10 Model card (Chapter 59)

*(Model details and version; intended use and out-of-scope uses; training data (including exclusions and why); training compute; evaluation data and metrics with ceilings and baselines; subgroup and stratified results; known failure modes; calibration; ethical and biosecurity considerations; contact and change log.)*

---

## G.11 Ethics, equity, and dual-use assessment (Chapter 59)

| Question | Answer |
|---|---|
| Is data about people used? Under what consent and agreement? Any re-identification risk? | |
| Which populations are represented? Subgroup performance (continuous genetic distance)? | |
| Is a clinical use proposed? What is the prospective-validation plan, calibration, net benefit? | |
| Could this capability raise the risk of serious harm (uplift versus convenience)? What is the counterfactual? | |
| Which pathogens, toxins, or hazardous functions are touched? Have biosafety/biosecurity experts been consulted? | |
| What controls are proportionate (data exclusion, staged/gated release, withheld artifacts, red-teaming, coordinated disclosure)? | |
| What are the defensive uses? | |
| Who reviewed this, and when will it be revisited? | |

---

## G.12 Capstone proposal (Chapter 59)

**Title and capstone number (C1–C8):**

**Question and the Expert Chain (one line per link):**

**Known-answer toy** (weeks 1–2): *design and success criterion:*

**Real-data analysis** (weeks 3–8): *datasets, ceilings, baselines, locked split, preregistration:*

**Decisive evaluation** (weeks 9–12): *adversarial test; stratified error analysis:*

**Deliverables:** short paper with claim–evidence table, reproducible repository, release and ethics statement.

**Risks and mitigations (pre-mortem):**

**What result would change my belief, and what would I do next?**
