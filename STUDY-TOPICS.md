# Biomimetic Architecture — Study Topics

*Topics to master using the Feynman Technique: study it, then explain it simply. If you can't explain it simply, you don't understand it yet.*

**How to use this:** For each topic, study it, then write a plain-language explanation in the "My explanation" section. If you get stuck, that's where the gaps are.

---

## 1. Neuron Fundamentals

### 1.1 Membrane Potential & Ion Channels
*How a neuron maintains a resting voltage and what happens when it changes.*
- Resting potential, depolarization, hyperpolarization
- Voltage-gated sodium/potassium channels
- The Nernst equation (you probably remember this)

**My explanation:**
> *(fill in)*

### 1.2 Action Potentials
*The all-or-nothing firing event.*
- Threshold, rising phase, falling phase, undershoot
- Why it's all-or-nothing (positive feedback loop)
- Propagation along the axon

**My explanation:**
> *(fill in)*

### 1.3 Refractory Periods
*Why a neuron can't fire again immediately.*
- Absolute refractory period (sodium channel inactivation)
- Relative refractory period (hyperpolarization, needs stronger stimulus)
- Functional significance: prevents backward propagation, limits firing rate

**My explanation:**
> *(fill in)*

### 1.4 Neuron Models (Computational)
*Mathematical abstractions of biological neurons.*
- Hodgkin-Huxley (biophysically detailed, expensive)
- Leaky Integrate-and-Fire (LIF) — closest to our design
- Izhikevich model (computationally cheap, biologically rich)
- Adaptive Exponential (AdEx)
- Tradeoffs: biological realism vs. computational cost

**My explanation:**
> *(fill in)*

---

## 2. Synapses & Connections

### 2.1 Synaptic Transmission
*How one neuron talks to another.*
- Chemical synapses: neurotransmitter release, receptor binding
- Excitatory vs. inhibitory post-synaptic potentials (EPSPs/IPSPs)
- Temporal summation (rapid successive inputs) vs. spatial summation (multiple inputs at once)
- Synaptic delay and why it matters for timing

**My explanation:**
> *(fill in)*

### 2.2 Synaptogenesis
*How new connections form.*
- Activity-dependent synaptogenesis (use creates connections)
- Growth cones, axon guidance, target recognition
- Critical periods in development
- Can we replicate this? When should new connections form in our model?

**My explanation:**
> *(fill in)*

### 2.3 Synaptic Pruning
*How unused connections are removed.*
- "Use it or lose it" — inactive synapses weaken and disappear
- Pruning during sleep (synaptic homeostasis hypothesis)
- Developmental pruning vs. ongoing maintenance
- Relevance: prevents unbounded connection growth in our model

**My explanation:**
> *(fill in)*

---

## 3. Learning & Plasticity

### 3.1 Hebbian Learning
*"Neurons that fire together wire together."*
- Hebb's postulate (1949)
- Strengths: simple, local, biologically plausible
- Weaknesses: only strengthens (no weakening), unstable without constraints
- Why it's necessary but not sufficient

**My explanation:**
> *(fill in)*

### 3.2 Long-Term Potentiation (LTP)
*The biological mechanism for strengthening connections.*
- NMDA receptors as coincidence detectors
- Early LTP (minutes to hours) vs. late LTP (hours to days, requires protein synthesis)
- The role of calcium influx
- Connection to memory formation

**My explanation:**
> *(fill in)*

### 3.3 Long-Term Depression (LTD)
*The biological mechanism for weakening connections.*
- Low-frequency stimulation → weakening
- The inverse of LTP (same machinery, different calcium dynamics)
- Essential for preventing saturation — without forgetting, you can't learn new things

**My explanation:**
> *(fill in)*

### 3.4 Spike-Timing-Dependent Plasticity (STDP) ⭐
*The timing-based learning rule. Our most likely candidate.*
- Pre-before-post (causal) → strengthen (LTP)
- Post-before-pre (anti-causal) → weaken (LTD)
- The timing window (~20ms each direction)
- Why temporal order matters: it captures causation
- STDP as a unification of Hebbian learning, LTP, and LTD
- Variants: symmetric STDP, triplet STDP, reward-modulated STDP

**My explanation:**
> *(fill in)*

### 3.5 Homeostatic Plasticity
*How the network keeps itself stable.*
- Synaptic scaling: global adjustment of all synaptic strengths
- Intrinsic plasticity: adjusting a neuron's own excitability
- Why this matters: STDP alone is unstable — weights can blow up or die
- The thermostat analogy: local learning rules need a global regulation mechanism

**My explanation:**
> *(fill in)*

---

## 4. Neural Coding (The Output Problem)

### 4.1 Rate Coding
*Information encoded in firing frequency.*
- Simple: count spikes per time window
- Works well for slow signals
- Limitation: throws away temporal information

**My explanation:**
> *(fill in)*

### 4.2 Temporal Coding
*Information encoded in precise spike timing.*
- Phase coding, first-spike coding, interspike intervals
- Higher information capacity than rate coding
- Harder to decode reliably

**My explanation:**
> *(fill in)*

### 4.3 Population Coding ⭐
*Information encoded in the collective activity of a group of neurons.*
- Georgopoulos' motor cortex work: individual neurons are noisy, populations are precise
- Each neuron has a "preferred stimulus" — population vector = weighted average
- Probably our best bet for output

**My explanation:**
> *(fill in)*

### 4.4 Sparse Distributed Representations (SDRs)
*Numenta's encoding scheme.*
- Large bit vectors with a small percentage of active bits
- Properties: similarity = overlap, capacity is enormous, robust to noise
- How HTM uses SDRs for both input encoding and internal representations

**My explanation:**
> *(fill in)*

### 4.5 Attractor Networks
*Networks that settle into stable states.*
- Hopfield networks: energy minimization, pattern completion
- Point attractors (stable states) vs. limit cycles (oscillations)
- Content-addressable memory: input a partial pattern, get the whole thing back
- Could our network "settle" into an answer?

**My explanation:**
> *(fill in)*

---

## 5. Network Topology

### 5.1 Small-World Networks
*Most nodes are locally clustered but globally reachable in few steps.*
- Watts & Strogatz (1998)
- The brain has small-world properties
- Efficient: high clustering + short path lengths

**My explanation:**
> *(fill in)*

### 5.2 Scale-Free Networks
*A few highly connected hubs, many sparsely connected nodes.*
- Power-law degree distribution
- Robust to random failure, vulnerable to targeted attack on hubs
- Some evidence the brain has scale-free properties

**My explanation:**
> *(fill in)*

### 5.3 Cortical Columns & Modularity
*The brain's organizational unit.*
- Minicolumns (~80-120 neurons) and macrocolumns (~100 minicolumns)
- Each column processes a specific receptive field
- Hawkins' "Thousand Brains" theory: each column builds a complete model
- Connection to Mithen's cognitive modularity

**My explanation:**
> *(fill in)*

---

## 6. The Big Integration Questions

### 6.1 Credit Assignment Problem
*How does a neuron deep in the network know it contributed to an error?*
- Backpropagation solves this in traditional NNs but isn't biologically plausible
- Alternatives: feedback alignment, target propagation, predictive coding
- The brain solves this somehow — we just don't know how yet

**My explanation:**
> *(fill in)*

### 6.2 Binding Problem
*How does the brain combine separate features into a unified perception?*
- You see color, shape, and motion in different brain areas — how are they combined?
- Temporal synchrony hypothesis: features bound by synchronized firing
- Relevant to our cross-module communication question

**My explanation:**
> *(fill in)*

### 6.3 Convergence & Halting
*When has the network "finished" processing?*
- Biology doesn't have this problem — it never stops
- For a computational system we need: when does the network's response to an input stabilize?
- Attractor settling? Firing rate equilibrium? Fixed time window?

**My explanation:**
> *(fill in)*

---

## Progress Tracker

| # | Topic | Studied | Explained | Confident |
|---|-------|---------|-----------|-----------|
| 1.1 | Membrane Potential | ⬜ | ⬜ | ⬜ |
| 1.2 | Action Potentials | ⬜ | ⬜ | ⬜ |
| 1.3 | Refractory Periods | ⬜ | ⬜ | ⬜ |
| 1.4 | Neuron Models | ⬜ | ⬜ | ⬜ |
| 2.1 | Synaptic Transmission | ⬜ | ⬜ | ⬜ |
| 2.2 | Synaptogenesis | ⬜ | ⬜ | ⬜ |
| 2.3 | Synaptic Pruning | ⬜ | ⬜ | ⬜ |
| 3.1 | Hebbian Learning | ⬜ | ⬜ | ⬜ |
| 3.2 | LTP | ⬜ | ⬜ | ⬜ |
| 3.3 | LTD | ⬜ | ⬜ | ⬜ |
| 3.4 | STDP | ⬜ | ⬜ | ⬜ |
| 3.5 | Homeostatic Plasticity | ⬜ | ⬜ | ⬜ |
| 4.1 | Rate Coding | ⬜ | ⬜ | ⬜ |
| 4.2 | Temporal Coding | ⬜ | ⬜ | ⬜ |
| 4.3 | Population Coding | ⬜ | ⬜ | ⬜ |
| 4.4 | SDRs | ⬜ | ⬜ | ⬜ |
| 4.5 | Attractor Networks | ⬜ | ⬜ | ⬜ |
| 5.1 | Small-World Networks | ⬜ | ⬜ | ⬜ |
| 5.2 | Scale-Free Networks | ⬜ | ⬜ | ⬜ |
| 5.3 | Cortical Columns | ⬜ | ⬜ | ⬜ |
| 6.1 | Credit Assignment | ⬜ | ⬜ | ⬜ |
| 6.2 | Binding Problem | ⬜ | ⬜ | ⬜ |
| 6.3 | Convergence & Halting | ⬜ | ⬜ | ⬜ |

---

*Feynman rule: If you can't explain it to a 12-year-old, you don't understand it. Fill in the "My explanation" sections in your own words. No jargon allowed.*
