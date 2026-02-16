# Biomimetic Neural Architecture

*A graph-based neural computation framework that replaces dense matrix multiplication with sparse, event-driven signal propagation — inspired by biological neural systems, built for machine learning.*

**Author:** Matt Titmus & Dross
**Status:** Living document / Thought experiment
**Started:** 2026-02-15

---

## What This Is

This is an alternative compute architecture for neural networks. Instead of dense matrix multiplication, computation happens through sparse, event-driven message passing between simple units arranged in a graph.

**The goal is not to simulate biology.** The goal is to replace `Y = WX + b` with something that scales with *activity* rather than *network size* — and to do it using ideas stolen from the only system that demonstrably solves intelligence at 20 watts.

We borrow from neuroscience when it solves a computational problem. We ignore it when it doesn't. This is engineering, not biology.

### What This Is Not

- **Not a spiking neural network simulator.** We don't model ion channels, membrane potentials, or neurotransmitter dynamics. Tools like NEST and Brian2 serve computational neuroscience. We're building a machine learning architecture.
- **Not a neuroscience research project.** We don't care whether our model is biologically *accurate*. We care whether it's computationally *useful*.
- **Not an incremental optimization.** Sparse matrix libraries and pruning techniques make traditional NNs more efficient. We're proposing a different computational primitive entirely — replacing matrix multiplication with graph-based signal propagation.

### Design Philosophy

1. **Biology is a parts catalog, not a blueprint.** Take what works (sparse activation, event-driven compute, local learning rules, temporal dynamics). Leave what doesn't (exact biophysics, evolutionary baggage, biological constraints that don't apply to silicon).
2. **Justify every biological feature computationally.** If we include something because "the brain does it," we need to articulate *why* it helps computation. Refractory periods aren't here for biological fidelity — they're here because they prevent runaway cascading and create useful temporal dynamics.
3. **Matrix multiplication is the benchmark.** Every design decision must be evaluated against the question: "Does this let us do something matrices can't, or do it more efficiently?" If neither, cut it.
4. **Practicality over elegance.** If a hybrid approach works (e.g., a conventional readout layer on top of biomimetic internals), that's fine. Purity is not a goal.

## Motivation

Modern neural networks rely on dense matrix multiplication — every forward pass computes across the entire weight matrix regardless of how much of that computation is relevant to the input. This works, but it's brute force.

The fundamental problem is that **computation scales with network size, not with input complexity.** A 10-billion-parameter model does the same amount of work whether you ask it "what's 2+2" or "explain quantum gravity." Every weight participates in every forward pass. Zero times a weight is still a multiply.

Biological neural systems solved this problem differently:

- **Sparse:** The vast majority of neurons are not firing at any given moment (~1-5% active)
- **Event-driven:** Computation only happens when a neuron is stimulated
- **Locally connected:** Most neurons connect to a small neighborhood, not the entire network
- **Energy-efficient:** ~20 watts for the human brain vs. megawatts for large model training

These aren't incidental properties of biology — they're engineering solutions to the same scaling problem we face. The brain can't afford O(n²) per thought either.

**Core question:** Can we build a neural computation framework where cost scales with *activity* (the number of neurons that actually contribute to a given computation) rather than *network size* (the total number of parameters)?

## The Biomimetic Neuron

Each neuron is a data structure in memory (a struct, not a float in a 2D array). It is heavier per-unit than a matrix element, but the hypothesis is that sparse activation compensates.

### Data Structure

```
BioNeuron {
    activation_level: int16       // Current activation state (clamped, signed)
    baseline: int16               // Resting activation (same for all neurons)
    firing_threshold: int16       // Activation level that triggers propagation
    last_interaction: uint32      // Counter: when this neuron was last touched
    refractory_until: uint32      // Counter: cannot fire again until this time
    connections: []Connection     // Outgoing "dendritic" connections
}

Connection {
    target: uint32                // Index into neuron array (supports >4B neurons)
    weight: int16                 // Signed: negative = inhibitory, positive = excitatory
                                  // Clamped to [-32768, +32767], no overflow
}
// Connection = 6 bytes (8 with padding). Pointer alternative would be 10+.
```

### Design Decision: Array Indices Over Pointers

Neurons live in a single contiguous array. Connections reference targets by uint32 index, not pointer.

**Rationale:**
- **Memory:** uint32 = 4 bytes vs pointer = 8 bytes on 64-bit systems. With thousands of connections per neuron, this halves the connection overhead.
- **Cache locality:** Contiguous array = cache-friendly sequential access. Pointer-chasing = cache misses.
- **Serialization:** Array + indices can be dumped/loaded trivially. Pointer graphs require reconstruction.
- **Parallelism:** Indices are easy to partition across cores or machines. Pointers are process-local.
- **Capacity:** uint32 supports >4 billion neuron indices — more than sufficient.
```

### Design Decision: Integer Arithmetic

All weights and activation levels use fixed-width signed integers (int16), not floats.

**Rationale:**
- **Memory:** int16 = 2 bytes vs float64 = 8 bytes. 4x savings per weight. At billions of connections, this is the difference between fitting in RAM and not.
- **Compute:** Integer addition/subtraction/comparison are the cheapest CPU operations. No FPU required, no IEEE 754 overhead. The entire activation cycle is three integer operations: one subtract (decay), one add (weight), one compare (threshold).
- **Hardware:** This could run on devices that can't touch traditional NNs — edge hardware, microcontrollers, embedded systems.
- **Biological fidelity:** Ion channels are binary (open/closed). Neurotransmitters release in discrete quanta. The analog character of neural activity emerges from many discrete events summed. Integer math may actually be *more* biologically faithful than floats.

**Clamping:** All values are clamped to their type range — no overflow/wraparound. A massively inhibited neuron saturates at min, not wraps to max. Same for weights. This mirrors biological saturation (finite receptor density, finite vesicle pools).
```

### The Activation Cycle

When a neuron receives input (is "poked" by an upstream neuron firing):

#### Step 1: Decay
Calculate how far activation has drifted back toward baseline since `last_interaction`.

```
elapsed = now - last_interaction
activation_level = baseline + (activation_level - baseline) * decay_function(elapsed)
last_interaction = now
```

**Key insight:** Idle neurons cost *zero* compute. There is no global clock tick updating millions of quiet neurons. Decay is calculated lazily, only when the neuron is next stimulated. This is what makes the architecture event-driven rather than clock-driven.

The counter-based approach (vs. wall-clock timestamps) makes the computation time-agnostic and deterministic.

#### Step 2: Summation
Apply incoming weight directly to the current activation level:

```
activation_level += incoming_weight
```

This is deliberately simple — no matrix multiply, no activation function applied across a layer. Just addition.

#### Step 3: Threshold Check & Propagation
If activation exceeds the firing threshold, fire:

```
if activation_level >= firing_threshold AND now >= refractory_until:
    for conn in connections:
        stimulate(conn.target, conn.weight)  // Recursive propagation
    refractory_until = now + refractory_period
    activation_level = post_fire_reset       // Could reset to baseline or below
```

#### Step 4: Refractory Period
After firing, the neuron is temporarily unresponsive. This:
- Prevents runaway cascading (the biological equivalent of an infinite loop)
- Acts as a natural rate limiter
- Creates temporal dynamics (a neuron can't just fire continuously)

In biology, there's both an *absolute* refractory period (cannot fire at all) and a *relative* one (can fire but needs stronger stimulus). Worth considering both.

## Properties

### How This Compares

| Property | Biology | This Model | Traditional NN |
|---|---|---|---|
| Sparse activation | ✅ ~1-5% active | ✅ Only active paths compute | ❌ Full matrix every pass |
| Event-driven | ✅ | ✅ Lazy decay | ❌ Clock-driven |
| Excitatory/inhibitory | ✅ | ✅ Signed weights | ⚠️ Implicit in matrix |
| Refractory period | ✅ | ✅ | ❌ |
| Temporal dynamics | ✅ | ✅ Via decay + counters | ❌ Static per pass |
| Energy proportional to activity | ✅ | ✅ (hypothetically) | ❌ |

### What's Different from Standard NNs

- **No layers.** The topology is a graph, not a stack. Signals propagate through the graph freely.
- **No global forward pass.** Computation is local and cascading.
- **Time is a first-class citizen.** The decay function means the *same* input can produce different outputs depending on recent history.
- **Computation cost scales with activity, not network size.** A 10-billion-neuron network where 0.1% is active costs the same as a 10-million-neuron fully-active network.

## The Energy Argument

This may be the strongest practical case for the architecture. AI energy consumption is becoming a crisis:

- Training a frontier model costs millions in electricity
- Inference at scale is worse — it runs 24/7 and grows with users
- A single H100 GPU: ~700W. A human brain: ~20W. The gap is orders of magnitude.

This model's efficiency gains are structural, not incremental:

| | Traditional NN | Biomimetic |
|---|---|---|
| Idle neurons | Still computed (zero × weight = still a multiply) | Zero cost — not touched at all |
| Weak signals | Computed, contribute near-zero | Naturally filtered by threshold — stop propagating |
| Cost scales with... | Network size (every pass, full matrix) | Activity (only active signal paths) |
| Theoretical floor | O(n²) per layer per pass | O(k) where k = active neurons << n |

The counterargument — "GPUs are optimized for matrices so this would be slower in practice" — is a hardware argument, not an algorithmic one. If the architecture is sound, hardware follows. That's literally why neuromorphic chips exist.

## Open Questions

### 1. Learning Rule ⭐ THE hard problem
How do connections form, strengthen, weaken, and prune? This could occupy an entire PhD.

#### Primary Candidate: Reward-Modulated STDP (R-STDP)

After surveying the neuromorphic literature (see `research/neuromorphic-landscape.md`), **R-STDP is our primary learning rule candidate.** It bridges the gap between local plasticity and goal-directed behavior:

**How it works (three-phase):**

1. **STDP creates eligibility traces.** When pre-synaptic neuron fires before post-synaptic (causal timing), the synapse is *marked* as a candidate for strengthening. Reverse timing marks it for weakening. These are just candidates — no weight change yet.

```
If pre fires before post (Δt > 0):  eligibility += A+ × exp(-Δt / τ+)
If post fires before pre (Δt < 0):  eligibility -= A- × exp(Δt / τ-)
```

2. **Eligibility traces decay over time.** If no reward signal arrives, the candidate changes fade away. This is a temporal credit window — "did this firing pattern lead to something good within the next N ticks?"

3. **Global reward/punishment signal consolidates changes.** When a reward signal arrives (analogous to dopamine in biology), all outstanding eligibility traces are consolidated into actual weight changes:

```
ΔW = reward_signal × eligibility_trace
```

Positive reward + positive eligibility = strengthen. Positive reward + negative eligibility = weaken. Negative reward inverts both.

**Why R-STDP fits our architecture:**
- **Local:** Only needs spike timing (which we track via `last_interaction` and `refractory_until`) + a global scalar reward signal
- **Temporal:** Naturally uses the timing dynamics built into our neuron model
- **Integer-friendly:** Eligibility traces and decay can be computed with the same lazy-decay approach we use for activation
- **Hardware-proven:** Loihi 2 implements three-factor learning rules (STDP + reward modulation) natively
- **Solves credit assignment:** The reward signal provides direction; STDP provides local structure

**Implementation note:** The learning rule is implemented behind the `LearningRule` interface, allowing algorithms to be swapped at runtime. Three implementations exist:
- **`learning/stdp`** — Pure STDP (unsupervised Hebbian). Weight changes applied directly from spike timing. No reward signal needed.
- **`learning/rstdp`** — Reward-modulated STDP (three-factor). Spike timing creates eligibility traces; reward consolidates them into weight changes.
- **`learning/predictive`** — Predictive learning (Saponati & Vinck 2023). Self-supervised; STDP-like behavior emerges from prediction error minimization.

#### Other Biological Mechanisms (for future consideration)
- **Hebbian plasticity** ("fire together, wire together") — basic association, subsumed by STDP
- **Long-term potentiation/depression (LTP/LTD)** — memory consolidation, threshold-based permanence
- **Pruning during sleep** — the brain cleans up connections offline. Could we do periodic pruning passes?
- **Neurogenesis & dendritic growth** — new connections forming over time
- **Myelination** — changing signal propagation speed (connection "priority")

#### The Credit Assignment Problem
Backprop solves credit assignment by propagating error gradients backward through layers. Our architecture has no layers. R-STDP addresses this differently: STDP handles the *where* (which synapses were active in the right pattern) and the reward signal handles the *what* (was the outcome good or bad). The eligibility trace window handles the *when* (how recently did the activity occur relative to the reward).

This is less precise than backprop — it won't find the mathematically optimal gradient. But it's also more robust, more biologically plausible, and naturally supports continual learning without catastrophic forgetting.

#### 🔬 Predictive Learning Rule — IMPLEMENTED ✅
Based on Saponati & Vinck 2023 (Nature Communications): "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule."

**The key insight:** STDP is not the learning rule — it's a *side effect*. When neurons optimize a simpler objective (predict their own future inputs), STDP timing curves emerge automatically. The predictive rule is more fundamental than STDP.

**Implementation:** `PredictiveRule` in `learning/predictive/`. Implements the `LearningRule` interface and can be swapped with STDP or R-STDP at runtime.

**Advantages over R-STDP:**
- **Fully self-supervised** — no external reward signal needed
- **STDP emerges automatically** — not manually programmed
- **Self-stabilizing** — heterosynaptic competition prevents runaway weights
- **Sequence learning** — neurons naturally learn to anticipate temporal patterns
- **Principled foundation** — derived from an optimization objective, not heuristics

**How it works:**
1. Each neuron predicts its next input: `prediction = activation × weight`
2. Prediction error drives learning: `error = actual_input - prediction`
3. Weights update to reduce prediction error, using both a per-synapse correlation term and a global heterosynaptic term
4. Eligibility traces provide temporal context (input history)

**Status:** IMPLEMENTED ✅ — `learning/predictive/` package. Integer arithmetic (int16), 13 tests passing. Currently alongside pure STDP and R-STDP. Needs head-to-head benchmarking before choosing a default.

See `research/predictive-learning-rule.md` for the full analysis.

### 2. Information Output ⭐ Hard but tractable
Input is straightforward — anything can be linearized into stimulation patterns. Output is harder.

**Biological approaches:**
- **Population coding** — read from a group of output neurons, interpret the pattern of activity (how the motor cortex works)
- **Rate coding** — firing frequency encodes the value (simpler, well-understood)
- **Attractor states** — network settles into a stable pattern that *is* the answer (Hopfield network style)

**Hybrid/pragmatic approaches:**
- **Readout layer** — a thin conventional layer on top that translates biomimetic activity into usable output. Inelegant but practical.
- **Temporal coding** — information encoded in *when* neurons fire, not just *whether* they fire

Matt plans to take inspiration from both the natural world and existing computational solutions — hybrid approach.

### 3. Representational Equivalence ⭐ Needs empirical evidence
**Can this learn representations equivalent to matrix transforms?**

**Theoretical argument (yes):** The universal approximation theorem says any continuous function can be approximated by a sufficiently wide network of simple units with nonlinear activation. Biomimetic neurons *are* nonlinear (threshold + fire = step function). Theoretical equivalence seems assured.

**Practical argument (unknown):** Theoretical possibility ≠ learnable in practice. The real question is whether the learning rule (#1) can find good representations in reasonable time.

**Pragmatic argument:** Full equivalence may not be necessary. If this architecture excels at specific problem classes — temporal/streaming data, anomaly detection, low-power edge inference — that's a viable niche independent of general-purpose equivalence.

### 4. Implementation Challenges
- **Memory overhead:** Each neuron is much larger than a matrix element. But if only 1% are active...
- **Hardware mismatch:** GPUs are absurdly optimized for matrix math. This architecture wants something more like a message-passing system. Neuromorphic chips (Intel Loihi, IBM TrueNorth) are closer.
- **Parallelism:** The cascading nature could be tricky to parallelize. But neurons with no dependencies *can* fire simultaneously.
- **Convergence:** How do we know when the network has "finished" processing an input? (Biology doesn't have this problem because it never stops.)

### 5. Connection Topology
- How many connections per neuron? (Biological neurons: 1,000-10,000)
- Random initial connectivity? Structured? Small-world?
- Do connections grow/prune over time?

## Architectural Insight: Modularity

Reference: *The Prehistory of the Mind* by Steven Mithen (1996)

Mithen argues the human mind evolved as a series of specialized, semi-independent cognitive modules — social intelligence, natural history intelligence, technical intelligence, language — each shaped by different evolutionary pressures. These modules only became interconnected later ("cognitive fluidity"), and it was the *cross-talk between modules* that produced uniquely human capabilities like art, religion, and science.

**Implication for this architecture:** The network topology shouldn't be one homogeneous mesh. It should be **clustered** — semi-independent modules with dense internal connections but sparse cross-connections. This mirrors both evolutionary neuroscience and Mithen's archaeological evidence.

Benefits:
- **Learning becomes more tractable** — each module can specialize with simpler local learning rules
- **Inter-module connections can evolve separately** — potentially a different learning rule for cross-module links
- **Emergent capabilities** — novel behavior arises from cross-module interaction, not from any single module
- **Fault tolerance** — damage to one module doesn't destroy the whole network
- **Biological fidelity** — the brain really is organized this way (visual cortex, motor cortex, Broca's area, etc.)

This is the "duct-taped evolutionary solution" — messy, modular, and more powerful than any clean unified design.

## Related Work

Research in the neighborhood of this idea, and how we differ:

| Approach | Relationship | Key Difference |
|---|---|---|
| **Spiking Neural Networks (SNNs)** | Closest existing paradigm | SNNs often aim for biological accuracy. We aim for ML utility. We use simpler dynamics (no differential equations) and prioritize learnability over biophysical realism. |
| **Neuromorphic hardware** (Intel Loihi, IBM TrueNorth, BrainScaleS) | Potential deployment target | These chips are designed for SNNs. Our architecture could map well to them, but we design software-first — it should run efficiently on commodity hardware too. |
| **Numenta / HTM** | Similar motivation, different architecture | HTM uses sparse distributed representations and cortical column theory. We use simpler neuron models and focus on graph-based signal propagation rather than columnar structure. |
| **Sparse matrix / pruning techniques** | Optimization of existing paradigm | These make matrix multiplication more efficient. We replace it entirely with a different computational primitive. |
| **Graph Neural Networks (GNNs)** | Structural similarity | GNNs pass messages on graphs but still use matrix ops at each node and require synchronous forward passes. Our propagation is asynchronous and event-driven. |
| **Leaky Integrate-and-Fire (LIF)** | Our neuron model is a simplified LIF | We use lazy decay instead of continuous integration — no ODE solving, just a calculation at interaction time. |
| **Liquid State Machines** | Reservoir computing with spiking neurons | LSMs use a fixed random reservoir. We want the topology itself to be learnable. |
| **Neural ODEs** | Continuous-time neural computation | Different approach to a similar motivation (computation as a continuous process). Much heavier mathematically. |

## Notes

- Matt's background is in molecular & cellular biology (4 years of PhD program), not CS. This design comes from understanding the actual biological substrate, not abstracting from existing ML. The advantage: seeing neural computation as it actually works in nature, not through the lens of how we've historically implemented it in software.
- The "focus computation where it's needed" property might make this particularly suited for real-time, streaming, or anomaly-detection tasks — problems where most of the input is uninteresting and only occasional signals matter.
- Even if this never becomes a practical training architecture, it could be valuable as an *inference* architecture — train with matrices, deploy as biomimetic network. The conversion from trained weights to a sparse graph topology is a research question worth pursuing.
- **The C. elegans connectome demo** is a validation tool, not the product. It proves the engine can handle real sparse topologies and produce coherent behavior. The real test is whether the architecture can *learn* — solve ML benchmarks through local plasticity rules rather than backpropagation.

---

*This document is a living design. We'll iterate on it as we dig deeper into the literature and work through the hard questions.*
