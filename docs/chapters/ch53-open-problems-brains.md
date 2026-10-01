# Chapter 53. Brains and Neural Systems: Computational Neuroscience as a Frontier of AI for Biology

!!! abstract "Chapter at a glance"
    **Motivation.** Neuroscience is the field where the book's two halves meet most directly: the models we build for biology (sequence models, latent-variable models, recurrent dynamics, attention) were inspired by brains, and brains are now measured with data sets that look like those of genomics, with tens of thousands of simultaneously recorded neurons and electron-microscopy reconstructions of every synapse in a cubic millimetre. This chapter treats the nervous system as a modeling target using the same ladder as the rest of the book: what is measured and what is hidden (the measurement gap), what a model can be identified from (the inference gap), what an objective means (the objective gap), and what carries over from one animal, area, or task to another (the generalization gap). It builds the standard models (integrate-and-fire neurons, rate networks, attractors, latent dynamics), then dissects three frontier developments: *digital twins* trained on neural recordings, *connectome-constrained* models, and *foundation models of behavior*. Two simulations quantify the central tensions: a noise ceiling for neural response prediction, and the question "is a wiring diagram enough?".
    **Prerequisites.** Chapters 2, 8, 11, 12, 18, 19, 25, 31, 43–46.
    **You will be able to:** (1) write and simulate leaky integrate-and-fire and rate-network models and relate them to the attractor and bistable models of Chapter 19; (2) compute a noise-ceiling-normalized explained variance for neural prediction; (3) state what a connectome does and does not determine about function, with a quantitative example; (4) explain why task-optimized models can predict responses without identifying mechanism; (5) read the claims about neural and behavioral foundation models with the claim ladder; (6) formulate open problems in the book's diagnostic style.

---

## 53.0 Why this chapter, and what is different

Neuroscience has the same pipeline as the rest of the book: a hidden system (neural circuits), measurement operators (spikes, calcium, field potentials, blood flow, electron micrographs, behavior), and a desire for predictive and mechanistic models. Four features make it distinctive.

1. **Time matters at the millisecond scale, and the state is high-dimensional and fast.** A genome is static; a cortical column's state changes every few milliseconds. The relevant model class is *dynamical* (Chapters 11, 19).
2. **The measurement gap is extreme and multi-axis.** No method gives all of: every neuron, every spike, every synapse, every molecule, in an animal that behaves. Methods trade coverage, temporal resolution, and invasiveness (§53.1).
3. **The objective is behavior and computation, not a molecular readout.** What would it mean to have "understood" a circuit? The answers in the literature (predict its activity; reproduce its behavior; derive it from an optimization principle; reverse-engineer its mechanism) are different objectives, and a model can satisfy one and fail another (§53.4).
4. **Structural rhymes are strong.** Cell types as attractors of a network (Chapter 19); latent dynamical systems (Chapters 8, 11); transformers and associative memory; representation learning and neural coding; credit assignment and backpropagation (§53.7).

!!! rhyme "Structural rhyme: the connectome ↔ the genome as a 'parts list'"
    A genome sequence lists the parts and the regulatory sites, not the dynamics; a connectome lists the neurons and synapses, not the strengths, the neuromodulatory state, the plasticity, or the dynamics. Both are *necessary structure* for a mechanistic model and both leave *parameters* to be inferred from functional data. The question "how much of the function follows from the structure?" is the same question as "how much of expression follows from sequence?" (Chapter 31), with the same answer: some, in a way that must be measured.

---

## 53.1 Biology for modeling: neurons, circuits, and how they are measured

!!! bio "Biology for modeling: the nervous system through the seven-question ladder"
    **What is it?** Neurons (about $8.6\times10^{10}$ in the human brain, about $7\times10^{7}$ in the mouse, about 139,000 in the adult fruit fly brain, 302 in *C. elegans*) are excitable cells connected by synapses (chemical, with a transmitter and a strength that changes with experience; or electrical, via gap junctions); they come in many types (hundreds of transcriptomically defined types in cortex and thousands in the human brain, from single-cell atlases: Chapter 25). **What information does it contain?** The wiring, the cell types and their receptors and channels, and the history-dependent state (synaptic weights, neuromodulators, intrinsic excitability). **How is it generated?** Development (genetically specified cell types and coarse wiring, refined by activity) and learning (plasticity). **How is it measured?** See the table. **How is it represented computationally?** Spike trains (point processes), rates, latent trajectories, graphs (connectomes), or high-dimensional embeddings of behavior. **What variation exists?** Across individuals (even in the fly, wiring differs from animal to animal), across states (sleep, arousal), across areas, across species. **What can ML not observe?** Most of the state: the neuromodulatory milieu, synaptic strengths in vivo, unrecorded neurons, internal states that do not reach behavior.

| Method | What it measures | Coverage | Temporal resolution | Main limits |
|---|---|---|---|---|
| Intracellular / patch recording | Membrane potential of single cells | 1 to a few cells | sub-ms | Very low throughput |
| Extracellular electrophysiology (e.g., Neuropixels) | Spikes of hundreds to thousands of neurons | Along a probe track | sub-ms | Spike sorting errors; sampling bias for large, active cells |
| Two-photon calcium imaging | Fluorescence of an indicator (a slow, nonlinear proxy of spiking) for $10^3$–$10^5$ neurons | A field of view or volume | about 10–100 ms | Slow indicator; deconvolution to spikes is model-dependent |
| fMRI | Blood-oxygenation signal in voxels ($\sim$1 mm) | Whole brain, human | about 1 s | Indirect; averages $10^5$–$10^6$ neurons |
| EEG / MEG | Summed field of populations | Whole scalp | ms | Poor spatial resolution; source ambiguity |
| Electron-microscopy connectomics | Neurons and synapses (anatomy), from serial sections | Whole fly brain; mm$^3$ of mouse cortex | none (a fixed snapshot) | No synaptic strengths, neuromodulation, or dynamics; proofreading cost |
| Spatial / single-cell transcriptomics | Gene expression, cell types | Large | none | Dissociation; no electrical activity |
| Behavior (video, tracking, task logs) | Output | Whole animal | frame rate | Many-to-one mapping from neural state |

The pairs **(anatomy without function)** and **(function without anatomy)** are the two ends of the measurement gap, and *joint* measurements (a calcium-imaged neuron later reconstructed in electron microscopy) are the scarce, valuable ones. The MICrONS project is the largest such data set (§53.5).

---

## 53.2 The standard models, from first principles

**Leaky integrate-and-fire (LIF).** A neuron's membrane potential $V$ integrates input current through a leak:
$$
\tau_m \frac{dV}{dt}=-(V-V_\text{rest})+R\,I(t),\qquad \text{if } V\ge V_\text{th}:\ \text{emit a spike and reset } V\leftarrow V_\text{reset}.
$$
With a constant current $I$ the neuron fires periodically at rate
$$
f(I)=\Big[\tau_m \ln\!\frac{RI+V_\text{rest}-V_\text{reset}}{RI+V_\text{rest}-V_\text{th}}\Big]^{-1}\quad\text{for } RI>V_\text{th}-V_\text{rest},
$$
a threshold-linear curve at moderate currents. The model has two parameters per neuron that matter for rates ($\tau_m$, the threshold distance) and no ion-channel detail; its virtue is that circuits of $10^5$ LIF neurons can be simulated (Shiu et al., §53.5).

**Hodgkin–Huxley and beyond.** Voltage-gated conductances ($g_\text{Na},g_\text{K}$ with gating variables) explain the shape of the spike and the diversity of firing patterns; each additional channel adds parameters, and *many different combinations of conductances give the same behavior* (degeneracy, §53.4).

**Rate networks.** Averaging over spikes gives a firing-rate model for $N$ neurons,
$$
\tau\,\dot r=-r+\phi(Wr+u(t)),
$$
with $W\in\mathbb R^{N\times N}$ the synaptic weight matrix (Dale's law: each column has one sign, excitatory or inhibitory), $u$ the external input, and $\phi$ a saturating or rectifying nonlinearity. The simulation of §53.5 uses $\phi=\tanh$.

!!! math "Derivation: stability and the spectral radius"
    Linearize $\tau\dot r=-r+Wr$ around $0$: $r(t)=e^{(W-I)t/\tau}r(0)$, which decays if and only if every eigenvalue $\lambda$ of $W$ has $\text{Re}\,\lambda<1$. For a random matrix with i.i.d. entries of variance $g^2/N$ (a "random network"), the eigenvalues fill a disc of radius $g$ (Girko's circular law), so $g<1$ gives a quiet network, and for $g>1$ the network is chaotic (Sompolinsky, Crisanti and Sommers 1988). Real cortex has structured $W$: strong excitation balanced by inhibition produces fast, stable, asynchronous dynamics (the balanced network of van Vreeswijk and Sompolinsky), in which individual inputs are large but cancel on average, so the response is a *difference of large numbers*: a source of both computational power and fragility to small errors in $W$. $\square$

**Attractors.** Networks with symmetric excitatory structure have fixed points representing stored patterns (Hopfield 1982); ring-shaped connectivity gives a continuous attractor for heading direction in the fly's central complex, a rare case where the wiring diagram was confirmed to implement a recognizable computational motif. This is the same mathematics as the bistable gene circuit of Chapter 19: *cell states and memories are attractors*.

**Low-rank structure.** Population activity is low-dimensional (Chapter 2: the participation ratio of neural population activity in a task is usually in the tens even for thousands of neurons) and a network with $W=\sum_{k\le R}m_kn_k^\top$ of rank $R$ has dynamics confined to the span of the $m_k$; modern models fit rank-$R$ networks to behavior and neural data.

---

## 53.3 Encoding models, digital twins, and the noise ceiling

**Encoding models** predict neural responses from stimuli or behavior: linear-nonlinear-Poisson models, generalized linear models with spike-history terms, and, since about 2014, *task-optimized deep networks* whose internal units predict responses in visual cortex (Yamins and DiCarlo; Cadieu et al.). The logic is that a network trained for object recognition develops internal representations that predict ventral-stream activity better than hand-designed models. The recent version is the **digital twin**: a network trained directly on neural recordings from an animal, then used as a surrogate for *in silico* experiments.

!!! paper "Paper dissection: A foundation model of mouse visual cortex (Wang et al., *Nature* 640, April 2025)"
    **Problem.** Build a predictive model of the activity of thousands of neurons across visual areas that transfers to new animals and new stimulus classes.
    **Insight.** Train one *core* network on recordings from many mice watching natural movies, then fit a light animal-specific *readout* for each new mouse, so that the core is a foundation model and each animal's digital twin is a transfer.
    **Architecture / training.** A shared core of convolutional and recurrent components (taking the visual stimulus, the animal's behavioral state such as pupil and running), per-neuron readouts, trained on responses of many thousands of neurons from multiple mice.
    **Evaluation.** Prediction of held-out natural movies; *out-of-distribution* generalization to new stimulus classes (noise patterns and static images) after training only on movies; transfer of the core to the MICrONS mouse, whose digital twin predicts neuronal responses, and a learned per-neuron *functional barcode* that predicts the neuron's anatomical location, cell class, and connectivity, verified against the electron-microscopy reconstruction of the same animal.
    **Why it worked.** Large, diverse recordings; the inclusion of behavioral state as an input (a major driver of variance); the transfer structure (shared core, individual readouts); and the existence of an independent anatomical ground truth in the same animal.
    **Assumptions.** Responses are determined by the recent stimulus history and behavioral state; a shared core suffices across animals; the readout captures individual differences.
    **Limitations.** Prediction is not mechanism (§53.4); the generalization to new stimulus types is partial; the model learns from head-fixed mice in a limited stimulus space; trial-to-trial variability sets a ceiling (below).
    **Unresolved.** Whether *in silico* optimal stimuli found with the twin reproduce in the animal for stimuli unlike the training set, and whether the twin's internal structure corresponds to anything in the circuit.

**A noise ceiling for neural prediction (Chapter 1 again).** Spiking is stochastic; the same stimulus produces different spike counts on different trials. A model that predicts the *mean* response perfectly cannot predict single trials. Single-trial $R^2$ is therefore capped, and the field's convention is to report the fraction of **explainable** variance (the variance of the trial-averaged response beyond noise), estimated from repeats. The simulation below uses 120 neurons with Poisson spiking and a shared gain fluctuation across neurons (trial-to-trial arousal-like noise) and a flexible stimulus-driven model.

```python
--8<-- "code/ch53_neural_models.py"
```

```text
== 1. Predicting single-trial responses: 120 neurons, 300 stimuli, 8 repeats, Poisson spiking with shared gain fluctuations ==
single-trial R^2 of the model: 0.368;   reliability ceiling from the other 7 trials: 0.399;   model / ceiling = 0.92
R^2 of the model against the TRUE mean response (which the experimenter cannot see): 0.763;   single-trial R^2 of the true mean itself: 0.477
noise correlation between neurons (shared gain): mean pairwise correlation of residuals = 0.124
```

**Reading Part 1.** The stimulus-driven model reaches a single-trial $R^2$ of 0.37, which looks poor, while a ceiling estimated from the other seven repeats of the same stimulus is 0.40 (the model reaches 92% of it) and the *oracle* (the true mean rate, which no experimenter sees) reaches 0.48 on single trials (the model reaches 77% of it). Two lessons: (1) an unnormalized $R^2$ near 0.4 can be an excellent model; (2) **the ceiling estimated from a finite number of repeats is itself noisy and biased low** (a held-out trial is predicted by the mean of 7 trials, which carries its own noise), so normalized scores near or above 1 occur for the wrong reasons; report the number of repeats and the estimator. The shared gain produces *noise correlations* between neurons (mean residual correlation 0.12), which a stimulus-only model cannot capture; a model with a shared latent factor can (Chapter 8) and this is one reason behavior and arousal are inputs to the twins.

---

## 53.4 Prediction is not mechanism: objective, degeneracy, identifiability

Three reasons a model can predict neural activity and still be the wrong explanation.

1. **Many architectures predict equally** (the inference gap). For feed-forward visual encoding, networks with different architectures and objectives reach similar predictivity of the same areas; *rankings among them are small and sensitive to how the readout is fit* (the regression from network units to neurons has many degrees of freedom: a flexible readout can make almost any rich representation look predictive, the same issue as Chapter 18's probes). Predictivity measures *linear decodability of a representation*, not identity of mechanism [[S]].
2. **Task optimization does not select a unique solution** (the objective gap). Schaeffer, Khona and Fiete (2022) showed that grid-cell-like units emerge in task-optimized recurrent networks only under specific, fragile choices (the readout, the position code, the periodic structure of the target), and not from the task alone [[S]]: the "emergent" representation is partly a property of the objective's encoding.
3. **Circuit degeneracy**. In the crustacean stomatogastric ganglion, very different combinations of ion-channel conductances and synaptic strengths produce the same rhythmic output (Prinz, Bucher and Marder 2004) and animals vary widely in their parameters while performing the same function [[E]]. If the same function arises from many parameter sets, *fitting a model to function cannot recover the biological parameters*; at best it recovers equivalence classes.

**Latent dynamics have rotation ambiguity.** For a latent linear system $z_{t+1}=Az_t+\epsilon$ observed via $x=Cz$, the pair $(A,C)$ is identifiable only up to an invertible change of coordinates $(TAT^{-1},CT^{-1})$ (Chapter 8). The eigenvalues of $A$ (decay rates and rotation frequencies) are *invariants*, but the individual axes are not; claims about "the dynamics along axis 3" are claims about an arbitrary coordinate unless the axis is tied to an intervention or a behavior. The rotational dynamics of motor cortex during reaching (Churchland et al. 2012) are an example of an *invariant* claim (eigenvalues with imaginary parts) that has held up.

!!! lens "Research lens: the Jonas–Kording test"
    Jonas and Kording (2017) applied the standard analysis toolbox of neuroscience (lesions, tuning curves, correlations, dimensionality reduction) to a simulated microprocessor whose full wiring and behavior were known, and showed that the methods produced plausible-looking but wrong or uninformative accounts of how it works [[S]]. The point is not that neuroscience is hopeless; it is that *every method should be validated on a system whose mechanism is known*, as this book does with simulations (Chapters 18, 31, 44) before applying a method to the real data.

---

## 53.5 Connectomics: the wiring diagram, and whether it is enough

**What exists (as of October 2026).**

- ***C. elegans***: the full hermaphrodite connectome, 302 neurons (White et al. 1986; updated by Cook et al. 2019).
- ***Drosophila* adult brain (FlyWire)**: the first whole-brain connectome of an adult fly, with 139,255 proofread neurons and more than 50 million synaptic connections (Dorkenwald et al., Schlegel et al., *Nature*, October 2024), with systematic cell-type annotation and network analyses of the wiring statistics.
- **Mouse visual cortex (MICrONS)**: one cubic millimetre, with calcium imaging of about 75,000 neurons co-registered with an electron-microscopy reconstruction of more than 200,000 cells and about 0.5 billion synapses (MICrONS consortium, *Nature*, April 2025); one early finding is that neurons with similar response properties preferentially connect, within and across areas.
- **Human**: only small samples so far, such as one cubic millimetre of temporal cortex reconstructed at nanoscale resolution (Shapson-Coe et al., *Science* 2024; of the order of 10⁵ cells and 10⁸ synapses).

**From wiring to activity.** Shiu et al. (*Nature* 634, 2024) built a leaky integrate-and-fire model of the *entire* fly brain from the connectome: each neuron is a LIF unit, synapse counts set weights (scaled by one global constant), and neurotransmitter identity (predicted from the electron-microscopy images) sets the sign. The model, with no fitting to activity, correctly predicted responses of known circuits for feeding and grooming upon activation of different gustatory and mechanosensory neuron types, and generated testable predictions. Lappalainen et al. (*Nature* 2024) took the opposite route for the fly visual system: the connectome constrains a network whose *neuron and synapse parameters* are then trained to perform visual tasks, and the resulting model predicted tuning properties of identified cell types before they had been measured. These are the two ends of the spectrum from **counts-only** to **connectome as constraint, parameters from data**; the question of this section is *when each suffices*.

The simulation below constructs a 60-neuron rate network with known connectivity, synapse counts, and signs, and unknown synaptic strengths drawn log-normally with spread $\sigma$ per connection (the real analog is that synapse size, receptor content, and release probability vary). Three models see the same stimuli and responses: **A** uses counts and signs with a single global scale (the Shiu et al. recipe), **B** uses the connectome only as a *mask* and fits all weights on the known connections from data, and **C** fits all $N^2$ weights from data, ignoring the connectome. Models are scored on held-out stimuli and on a *causal* test: the predicted change in other neurons' responses when one neuron is silenced, compared with the true change (a counterfactual that is never in the training data).

```text
== 2. Is the wiring diagram enough? 60-neuron rate network; connectome known (who connects, synapse counts, signs); synaptic strengths unknown (log-normal, spread sigma) ==
models: A = counts x sign (one global scale fit); B = connectome-constrained: weights fit freely on the known connections only, from small random values; C = unconstrained: all N^2 weights fit freely from small random values
quantities: held-out response correlation r (new stimuli) and the correlation of predicted vs true changes in responses when one neuron is silenced ('ablation')
sigma  stimuli   A: counts only        B: connectome-constrained     C: unconstrained        (response r / ablation r)
 0.0      30    1.000 /  1.000         0.998 /  0.989              0.870 /  0.342
 0.0     150    1.000 /  1.000         1.000 /  0.999              0.997 /  0.980
 0.0     600    1.000 /  1.000         1.000 /  1.000              1.000 /  0.998
 0.5      30    0.972 /  0.864         0.998 /  0.988              0.867 /  0.299
 0.5     150    0.972 /  0.833         1.000 /  0.999              0.995 /  0.973
 0.5     600    0.972 /  0.826         1.000 /  1.000              1.000 /  0.997
 1.0      30    0.924 /  0.599         0.994 /  0.937              0.848 /  0.211
 1.0     150    0.921 /  0.570         0.999 /  0.987              0.983 /  0.857
 1.0     600    0.922 /  0.538         0.999 /  0.997              0.998 /  0.943
```

**Reading Part 2.**

1. **Counts alone predict responses well and perturbations badly, and more data does not help.** With a single global scale, model A reproduces held-out responses with $r=0.92$–$0.97$ at every heterogeneity level, yet its prediction of the effect of silencing a neuron falls from 1.00 ($\sigma=0$, where counts are exactly proportional to strengths) to 0.83–0.86 ($\sigma=0.5$) and 0.54–0.60 ($\sigma=1.0$). It has nothing to learn from additional stimuli. *Held-out response accuracy is a poor test of a wiring-based model*; the **ablation test** is the discriminating one (the same lesson as Chapter 31, where good prediction concealed a wrong counterfactual).
2. **The connectome as a mask is worth a great deal.** Model B, which fits the weights on the known connections only, reaches ablation $r\ge0.94$ at every heterogeneity level and every data size, including 30 stimuli (fewer than the 60 neurons). The unconstrained model C has 3,600 free weights against $K	imes60$ measured responses: with $K=30$ the data cannot determine them, and C predicts held-out responses moderately ($r\approx0.85$–$0.87$) while its ablation prediction is poor ($r=0.21$–$0.34$): *a model can fit responses without identifying the wiring that produces them*. Only with $K=150$–$600$ does C approach B, and at $\sigma=1.0$ it is still behind (ablation $r=0.86$ at $K=150$ and $0.94$ at $K=600$, against B's $0.94$ at $K=30$): in this toy the connectome mask is worth roughly a 20-fold increase in the number of stimulus presentations.
3. **Why**: the mask reduces the parameters from $N^2=3{,}600$ to about 430 (the 12% of pairs that are connected) and fixes the sign of each, and that is enough to make the problem identifiable at small $K$. The value of the mask scales with the sparsity and with how well the remaining model class matches reality.
4. **What the toy assumes**, each of which makes B look better than it would in real data: a *complete and error-free* connectome (real reconstructions have false and missed synapses and uncertain neurotransmitter assignments); every neuron recorded (real recordings sample a fraction of the neurons); the correct nonlinearity and no unmodeled neuromodulation or plasticity; and noise at 5% of the signal. The qualitative result (mask-and-fit beats both counts-only and unconstrained fits, and the response test hides what the ablation test reveals) is the part to carry forward; the factor of 20 is a property of this network.

!!! lens "Research lens: what the connectome contributes, in information terms"
    The connectome reduces the hypothesis space of $W$ from $N^2$ free parameters to $|E|$ ($|E|\approx0.12N^2$ here, and about $10^{-5}$ of the possible pairs in a fly brain); it supplies the signs; it supplies *counts* that may be informative about strengths (here, by construction, proportional to them up to the heterogeneity $\sigma$). It does not supply the strengths (hence the value of functional data), neuromodulation, intrinsic properties, or the dynamics. How much each missing piece costs is an empirical question per circuit, which is why connectome-constrained modeling is run as *fit, then test on held-out stimuli and held-out perturbations*.

---

## 53.6 Foundation models for neural activity and behavior

- **Neural activity.** Beyond the mouse visual model of §53.3, large pretrained transformers over spike trains (for example POYO and related models) treat spikes as tokens and aim to transfer across sessions, animals, and tasks (brain-machine interfaces, decoding). The data are heterogeneous (different electrodes, different neurons in each session), which is the same *unaligned feature space* problem as single-cell atlases (Chapters 30, 38): there is no shared "gene list".
- **Behavior and cognition.** *Centaur* (Binz et al., *Nature* 2025) fine-tuned a language model on *Psych-101*, trial-by-trial data from 160 psychological experiments (60,092 participants, 10,681,650 choices), and reports that it predicts held-out participants' choices better than domain-specific cognitive models and generalizes to some held-out tasks. It drew substantial criticism: that the model's apparent success may reflect overfitting to the structure of the training experiments (one study found it continued to give "correct-looking" answers when the task instruction was replaced by an irrelevant one), that 160 experiments are a small sample of cognition, and that prediction of behavior is not explanation of cognition (an analog and a digital clock agree on the time with entirely different internals) [[S]] for the criticisms, [[H]] for whether such models will yield theory.
- **Language models and brains.** Large language models' internal states predict fMRI and electrocorticography responses to language, with predictivity correlating with next-word prediction quality (Schrimpf et al. 2021; Caucheteux and King 2022; Goldstein et al. 2022) [[S]]. As in §53.4, this is evidence that the *representations are linearly aligned*, not that the brain implements the same algorithm. Non-invasive "semantic decoders" (Tang et al. 2023) reconstruct the gist of continuous language from fMRI after hours of per-subject training and fail across subjects and when the subject resists, which bounds both capability and the privacy concern [[S]].

!!! openproblem "Open problem: what would count as understanding a circuit?"
    A circuit is "understood" at the level of (i) prediction of activity, (ii) prediction of behavior under perturbation (lesions, optogenetics), (iii) a compact causal model that generalizes to new conditions, or (iv) derivation from an objective plus constraints. **Diagnostic questions:** Which of these can be tested with the current tools? In the fly, where wiring and many cell-type functions are known, what *held-out perturbation test* would a connectome-constrained model have to pass (silencing a cell type; rewiring in silico and comparing with a genetic manipulation) to count as mechanistic rather than predictive? Could a *benchmark of perturbation predictions in the fly* (a registered set of silencing/activation experiments, results sealed until models are submitted) play the role CASP played for structure (Chapter 35)?

---

## 53.7 Backpropagation and the brain; learning in biological and artificial networks

Backpropagation (Chapter 9) requires *symmetric weights* (the backward pass uses the transpose of the forward weights) and a separate backward phase carrying error signals: both seem biologically implausible, the "weight-transport problem". Proposals (feedback alignment, Lillicrap et al. 2016: random fixed feedback weights suffice because forward weights *learn to align* with them; predictive-coding and energy-based formulations that approximate backprop with local updates, Whittington and Bogacz 2017; dendritic-compartment models of error signals; and three-factor rules with neuromodulators as the third factor) offer algorithms that use only local information. It remains an open question which of these, if any, the brain uses, and the measurement needed (the *plasticity rule*, per synapse, in vivo) is exactly what current tools cannot supply at scale.

!!! rhyme "Structural rhyme: credit assignment ↔ identifiability of a causal graph (Chapter 44)"
    Both ask how a downstream error or effect is attributed to an upstream unit when many paths exist. In causal inference the answer needs interventions; in the brain, neuromodulatory and local signals appear to approximate an intervention-free estimate. In both, an observational "correlation of activity with error" is insufficient without extra structure.

---

## 53.8 Worked research examples

!!! example "Worked Research Example 53.1: Interpreting a digital-twin result"
    **Situation.** A paper reports that a digital twin of mouse V1 predicts held-out responses with a correlation of 0.65 per neuron on average, and that "optimal stimuli" found with the twin drive neurons in the real animal 40% more than the best natural image. The authors claim the twin "captures the computation of V1".

    **Question.** What does the evidence support?

    **Reasoning.**

    1. **Normalize by the ceiling.** Correlation 0.65 is meaningful only relative to the repeat-based reliability of each neuron; estimate $\rho_\text{ceiling}$ per neuron (using the number of repeats reported) and report the fraction of explainable variance; a neuron-wise table, not one average. (§53.3: ceiling estimates are noisy for few repeats.)
    2. **In-distribution versus out-of-distribution.** The 0.65 is on held-out movies from the training distribution. The claim about optimal stimuli is *out of distribution*; the 40% boost is a prospective test [[S]] for the claim that the twin extrapolates in the direction that drives a neuron. It does not show that the twin's internal units correspond to circuit elements.
    3. **What would show "captures the computation"?** Predictions under *interventions* the twin was not trained on: silencing a cell type (optogenetics) and predicting the change in other neurons' tuning; testing a mechanistic prediction from the twin's structure (for example that a given neuron's selectivity arises from a specific input combination) with the connectome of the same animal (the MICrONS design).
    4. **Controls.** A simpler baseline with the same inputs (stimulus, behavior) to show what the deep core adds; shuffled-neuron and shuffled-animal readouts to test transfer; a model-class comparison (several architectures with matched predictivity) to estimate how much of the *structure* is constrained by the data (§53.4).
    5. **Claim.** "A network trained on recordings predicts V1 responses to held-out natural movies at X% of explainable variance and finds stimuli that drive neurons more strongly in vivo" (C1 to C2). "Captures the computation" is C4 and is not licensed.

    **Expert analysis.** The strongest part of the claim is the prospective in vivo test; the weakest is the interpretation of internal units. The accurate summary is that the twin is a good *surrogate for experiments*, not yet a theory.

!!! example "Worked Research Example 53.2: Does a task-optimized network explain an area? No known answer"
    **Situation.** A group trains recurrent networks to perform the navigation task a rodent performs and finds that units resemble entorhinal grid cells; they propose that grid cells emerge from path integration under an efficient-coding objective.

    **Question.** What would you test to decide whether this is an explanation, and what is not known?

    **Reasoning (Expert Chain).**

    1. **L1 Problem.** Explanatory claim ("grid cells arise because of the objective") from a model that reproduces one *observation* (periodic tuning).
    2. **L3–L5 Assumptions and failure modes.** The result may depend on choices unrelated to the biology (output encoding, nonnegativity, regularizers); different choices give different or no grids (Schaeffer et al., §53.4). The existence of a model that reproduces grid-like units does not show that the brain uses the objective.
    3. **L7–L8 Hypotheses.** H1: grid-like tuning is the unique efficient solution to path integration. H2: it is a consequence of specific architecture and output choices. H3: both objective and inductive biases are needed.
    4. **L9–L10 Experiments.** (a) *Robustness*: vary output encodings, nonlinearities, noise levels, environment shape, and regularization; report the fraction of runs producing grid cells and their *module structure*. (b) *Predictions beyond the training observation*: how do grids deform in non-square environments, and what happens to the network's units when the environment is rescaled; test against recordings. (c) *Quantitative comparison* with the neural data across several statistics (spacing ratios between modules, the effect of boundaries), using a held-out set of statistics. (d) *Causal tests in the animal* of the mechanism the model proposes (for instance, whether a manipulation that should disrupt path-integration inputs changes grids in the predicted way).
    5. **L11 Interpretation.** If (a) shows high fragility and (b) predicts nothing new, the model is an existence proof. If grid module ratios and deformations follow, it is a candidate explanation; interventions in the animal elevate it.
    6. **L12 New directions.** *Model comparison with prospective, pre-registered predictions* between competing normative accounts is the missing practice.

    **What is not known.** Whether the entorhinal code is the solution to a computational problem or a developmental product of attractor network self-organization; the two accounts make partly overlapping predictions, which is why the discriminating experiments of step 4 are the research.

---

## 53.9 Researcher's Notebook

!!! notebook "Researcher's Notebook: from a neural data set to a defensible claim"
    1. **State the objective** (predict, reproduce behavior, explain mechanism) and the claim-ladder rung.
    2. **Estimate the noise ceiling** from repeats, with the estimator, number of repeats, and a confidence interval; report normalized scores *and* raw ones.
    3. **Include behavioral and arousal variables** as inputs and report their contribution; they are often the largest single predictor.
    4. **Compare with a simple baseline** (linear-nonlinear, GLM, PCA-based decoder) and with several architectures at matched predictivity.
    5. **Validate the method on a system with known mechanism** (a simulated network) before interpreting internal units.
    6. **Test out of distribution**: new stimulus class, new animal, new area, held-out perturbation.
    7. **Use the invariants** (eigenvalues, geometry) rather than the axes of a fitted latent model.
    8. **When a connectome is available, test the model on perturbations** (silencing, activating) as well as responses; compare counts-only, mask-and-fit, and free models.

    **What it teaches.** The scarce resource is not predictive accuracy but *causal tests*; the rest is bookkeeping about what the data can identify.

    **An open question to carry forward.** The simulation of §53.5 made the strength heterogeneity $\sigma$ a controlled parameter. For a real connectome, can $\sigma$ (and the relation between synapse counts and strengths per cell type) be *estimated* from a small amount of paired functional and structural data (for instance, electrophysiology of tens of synapses in the same animal), and how many such measurements per cell-type pair would make the counts-only model reliable enough for prediction at the scale of a whole brain?

---

## 53.10 Connections

- **Backward:** dynamical models (Chapters 11, 19); latent variables and identifiability (Chapters 2, 8); measurement and noise ceilings (Chapters 1, 25); counterfactuals and identifiability (Chapters 31, 44); foundation-model transfer (Chapters 17, 38); interpretability and probing (Chapter 18).
- **Forward:** mechanistic interpretability as a method shared with neuroscience (Chapter 48); open-problem atlases (Chapters 50–52); AI scientists in neuroscience (Chapter 54); open-ended case studies (Chapter 58).

!!! takeaways "Key takeaways"
    1. Neural systems are measured through different operators with a severe coverage-versus-resolution tradeoff; joint structure-and-function data (MICrONS) are the scarce, informative ones.
    2. **Noise ceilings are essential**: a single-trial $R^2$ of 0.37 was 92% of the repeat-based ceiling and 77% of the oracle; ceiling estimates from few repeats are biased low and noisy.
    3. Prediction is not mechanism: many models predict equally (the readout absorbs differences), task optimization yields grid-like units only under fragile choices, and circuits are degenerate; use invariants and causal tests.
    4. The fly connectome (139,255 neurons, over 50 million synapses) and MICrONS (about 200,000 cells, 0.5 billion synapses) make connectome-constrained modeling possible; Shiu et al. (counts and signs only) and Lappalainen et al. (connectome plus trained parameters) are the two ends of a spectrum.
    5. **Is the wiring diagram enough?** In a simulated 60-neuron network, counts-only models predicted held-out responses ($r=0.92$–$0.97$) but not the effect of silencing a neuron (0.54–0.60 at strength heterogeneity $\sigma=1.0$); using the connectome as a *mask* with weights fit from data restored ablation prediction ($r\ge0.94$) even with fewer stimuli than neurons, where an unconstrained model failed ($r=0.21$–$0.34$); the mask was worth about a 20-fold increase in stimuli. Test models on perturbations.
    6. Foundation models of neural activity and behavior (Wang et al. 2025; Centaur) raise the same questions as elsewhere: out-of-distribution generalization, shortcuts, and whether prediction yields theory.
    7. Validate every analysis method on a system whose mechanism is known (Jonas and Kording), and test models on perturbations, not only on held-out stimuli.

---

## Further reading

- Dayan, P. & Abbott, L. F. (2001). *Theoretical Neuroscience.* MIT Press. Gerstner, W., Kistler, W. M., Naud, R. & Paninski, L. (2014). *Neuronal Dynamics.* Cambridge University Press. Marder, E. & Goaillard, J.-M. (2006). Variability, compensation and homeostasis in neuron and network function. *Nat. Rev. Neurosci.* 7, 563–574. Prinz, A. A., Bucher, D. & Marder, E. (2004). Similar network activity from disparate circuit parameters. *Nat. Neurosci.* 7, 1345–1352.
- Sompolinsky, H., Crisanti, A. & Sommers, H. J. (1988). Chaos in random neural networks. *Phys. Rev. Lett.* 61, 259–262. van Vreeswijk, C. & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. *Science* 274, 1724–1726. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS* 79, 2554–2558.
- Yamins, D. L. K. et al. (2014). Performance-optimized hierarchical models predict neural responses in higher visual cortex. *PNAS* 111, 8619–8624. Schrimpf, M. et al. (2021). The neural architecture of language: integrative modeling converges on predictive processing. *PNAS* 118, e2105646118. Schaeffer, R., Khona, M. & Fiete, I. R. (2022). No free lunch from deep learning in neuroscience: a case study through models of the entorhinal-hippocampal circuit. *NeurIPS.* Jonas, E. & Kording, K. P. (2017). Could a neuroscientist understand a microprocessor? *PLoS Comput. Biol.* 13, e1005268.
- Wang, E. Y. et al. (2025). Foundation model of neural activity predicts response to new stimulus types. *Nature* 640, 470–477. MICrONS Consortium (2025). Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* (April 2025). Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634. Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of Drosophila. *Nature* 634. Shiu, P. K. et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634, 210–219. Lappalainen, J. K. et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140.
- Kim, S. S., Rouault, H., Druckmann, S. & Jayaraman, V. (2017). Ring attractor dynamics in the Drosophila central brain. *Science* 356, 849–853. Hulse, B. K. et al. (2021). A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection. *eLife* 10, e66039. Azabou, M. et al. (2023). A unified, scalable framework for neural population decoding (POYO). *NeurIPS.*
- Binz, M. et al. (2025). A foundation model to predict and capture human cognition. *Nature.* Churchland, M. M. et al. (2012). Neural population dynamics during reaching. *Nature* 487, 51–56. Pandarinath, C. et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nat. Methods* 15, 805–815. Tang, J., LeBel, A., Jain, S. & Huth, A. G. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings. *Nat. Neurosci.* 26, 858–866.
- Lillicrap, T. P., Cownden, D., Tweed, D. B. & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nat. Commun.* 7, 13276. Whittington, J. C. R. & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Comput.* 29, 1229–1262. Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J. & Hinton, G. (2020). Backpropagation and the brain. *Nat. Rev. Neurosci.* 21, 335–346.
