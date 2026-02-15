# Biomimetic Neural Architecture

*A speculative design for a neural computation model using sparse, event-driven, independently functional units instead of dense matrix transforms.*

**Author:** Matt Titmus & Dross
**Status:** Living document / Thought experiment
**Started:** 2026-02-15

---

## Motivation

Modern neural networks rely on dense matrix multiplication — every forward pass computes across the entire weight matrix regardless of how much of that computation is relevant to the input. This works, but it's brute force. Biological neural systems, by contrast, are:

- **Sparse:** The vast majority of neurons are not firing at any given moment
- **Event-driven:** Computation only happens when a neuron is stimulated
- **Locally connected:** Most neurons connect to a small neighborhood, not the entire network
- **Energy-efficient:** ~20 watts for the human brain vs. megawatts for large model training

**Core question:** Can we replicate the functional output of dense matrix transforms using a large array of mathematically simple, independently functional units — and gain efficiency by focusing computation only where it's needed?

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

### What This Gets Right (Compared to Biology)

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

**Biological inspiration (multi-layered):**
- **Hebbian plasticity** ("fire together, wire together") — basic association, but too simple alone
- **Spike-timing-dependent plasticity (STDP)** — *most promising starting point.* Local (no global error signal), temporal (fits our architecture), well-studied. Weight change depends on relative timing of pre/post-synaptic firing.
- **Long-term potentiation/depression (LTP/LTD)** — memory consolidation, threshold-based permanence
- **Pruning during sleep** — the brain cleans up connections offline. Could we do periodic pruning passes?
- **Neurogenesis & dendritic growth** — new connections forming over time
- **Myelination** — changing signal propagation speed (connection "priority")

**Computational approaches:**
- **Backpropagation equivalent?** — Can error signals propagate backward through this graph? Unclear — the lack of layers makes this non-obvious.
- **Evolutionary/genetic approach** — optimize topology and weights through selection pressure
- **Reward-modulated STDP** — a hybrid: STDP for local updates, global reward signal for direction

**The core tension:** Backprop works because it has a clear error gradient to follow. Biology doesn't have backprop and manages fine — but it has billions of years of evolutionary optimization baked into the architecture. We need to find the minimal viable learning rule.

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

Research in the neighborhood of this idea:
- **Spiking Neural Networks (SNNs)** — closest existing paradigm
- **Neuromorphic computing** — Intel Loihi, IBM TrueNorth, BrainScaleS
- **Numenta / Hierarchical Temporal Memory (HTM)** — Jeff Hawkins' biologically-inspired approach
- **Leaky Integrate-and-Fire (LIF) models** — computational neuroscience standard
- **Sparse Distributed Representations** — Numenta's encoding scheme
- **Liquid State Machines** — reservoir computing with spiking neurons
- **Neural ODEs** — continuous-time neural computation (different approach, similar motivation)

## Notes

- Matt's background is in molecular & cellular biology (4 years of PhD program), not CS. This design comes from understanding the actual biological substrate, not abstracting from existing ML.
- The "focus computation where it's needed" property might make this particularly suited for real-time, streaming, or anomaly-detection tasks — problems where most of the input is uninteresting and only occasional signals matter.
- Even if this never becomes a practical training architecture, it could be valuable as an *inference* architecture — train with matrices, deploy as biomimetic network?

---

*This document is a living design. We'll iterate on it as we dig deeper into the literature and work through the hard questions.*
