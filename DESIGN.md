# SparkSNN — Design Document

*An energy-efficient spiking neural network framework using integer arithmetic, sparse connectivity, and event-driven computation.*

**Authors:** Matt Titmus & Dross
**Status:** Living document
**Started:** 2026-02-15
**Last updated:** 2026-03-01

---

## What This Is

SparkSNN is an alternative compute architecture for neural networks. Instead of dense floating-point matrix multiplication, computation happens through sparse, event-driven signal propagation between integer LIF neurons arranged in a graph.

**The goal:** Replace `Y = WX + b` with something that scales with *activity* rather than *network size* — and do it using ideas from the only system that demonstrably solves intelligence at 20 watts.

### Design Philosophy

1. **Biology is a parts catalog, not a blueprint.** Take what works (sparse activation, event-driven compute, mixed excitatory/inhibitory connectivity). Leave what doesn't (exact biophysics, evolutionary baggage).
2. **Justify every biological feature computationally.** Refractory periods aren't here for biological fidelity — they're here to prevent runaway cascading and create temporal dynamics.
3. **Matrix multiplication is the benchmark.** Every decision must answer: "Does this do something matrices can't, or do it more efficiently?"
4. **Practicality over elegance.** Hybrid approaches are fine. Purity is not a goal.

### What This Is Not

- **Not a spiking neural network simulator.** We don't model ion channels or neurotransmitter dynamics. NEST and Brian2 serve neuroscience. We're building an ML architecture.
- **Not an incremental optimization.** Sparse matrix libraries make traditional NNs more efficient. We're proposing a different computational primitive — graph-based signal propagation.

---

## Architecture

### Integer LIF Neurons

Each neuron is a struct in a contiguous array. All arithmetic is integer.

```
Neuron {
    Activation: int32         // Current membrane potential
    Baseline: int32           // Resting activation
    Threshold: int32          // Firing threshold
    DecayRate: uint16         // Fixed-point fraction of 65536
    LastFired: uint32         // Counter tick of last spike (0 = never)
    LastInteraction: uint32   // Counter tick of last input
    Connections: []Connection // Outgoing synapses
}

Connection {
    Target: uint32            // Index into neuron array
    Weight: int32             // Signed: negative = inhibitory, positive = excitatory
    Eligibility: int32        // STDP eligibility trace
}
```

### Design Decisions

**Array indices over pointers.** Neurons live in a contiguous array; connections reference targets by uint32 index. This gives cache locality, trivial serialization, and easy partitioning across cores.

**Integer arithmetic.** Int32 weights and activations, no floats at runtime. Integer add is ~0.1 pJ vs FP32 multiply at ~3.7 pJ (Horowitz 2014). The entire activation cycle is: one multiply + shift (decay), one add (accumulate), one compare (threshold).

**Lazy decay.** Idle neurons cost zero compute. Decay is calculated on-demand when a neuron next receives input, not on a global clock tick. This is what makes the architecture event-driven.

**Accumulate-then-fire.** All stimulations arriving in the same tick are summed per neuron before evaluation (spatial summation). Results are deterministic regardless of processing order.

### The Activation Cycle

When a neuron receives input:

1. **Decay:** `activation = baseline + (activation - baseline) * decay_function(elapsed)`
2. **Accumulate:** Sum all pending stimulations → `activation += total`
3. **Threshold check:** If `activation >= threshold` and refractory period has elapsed → **fire**
4. **Fire:** Reset membrane, queue weight additions to all connected targets for next tick

**Signal propagation uses a 1-tick delay.** Downstream targets are processed on the next `Tick()`. This prevents infinite cascading and creates useful temporal dynamics.

### Sparse Event-Driven Computation

| Property | Traditional NN | SparkSNN |
|:---|:---|:---|
| Idle neurons | Still computed (0 × weight = still a multiply) | Zero cost — not touched |
| Cost scales with | Network size (full matrix every pass) | Activity (only active paths) |
| Theoretical complexity | O(n²) per layer | O(k) where k = active neurons ≪ n |

A 10-billion-neuron network where 0.1% is active costs the same as a 10-million-neuron fully-active network.

---

## Training: Dual-Domain Approach

### The Problem

Integer arithmetic is not differentiable, and the spike function (Heaviside step) has zero gradient almost everywhere. Biological learning rules (STDP, reward modulation, arbiter neurons) can solve simple tasks but hit a ceiling at multi-layer credit assignment.

### The Solution: Surrogate Gradient Training

We maintain two parallel representations:

1. **Float64 domain (training):** Shadow weights, continuous membrane simulation, surrogate gradient BPTT, Adam optimizer
2. **Int32 domain (inference):** Quantized weights, integer LIF dynamics, event-driven propagation

A configurable **weight scale factor** α converts between domains: `w_int32 = round(α × w_float64)`

**Key components:**
- **Fast sigmoid surrogate** (slope=25): Smooth approximation of the Heaviside derivative during backward pass
- **Spike count cross-entropy loss:** Total spikes per output neuron as logits
- **Adam optimizer** with standard hyperparameters
- **Quantization parity:** Verified zero prediction degradation between float64 and int32 inference on Iris (0/30 mismatches)

See `research/surrogate-gradient-training.md` for the full design.

### Results

| Benchmark | Architecture | Accuracy | Training Time |
|:---|:---|---:|:---|
| Iris | 40 → 20 → 3 | **100%** | 1.9s (CPU) |
| MNIST | 784 → 256 → 10 (20% sparse) | **95.8%** | ~13 min (CPU) |
| MNIST (deep) | 784 → 256 → 128 → 10 | 93.7% | ~19 min (CPU) |

### Energy Efficiency

Per-inference comparison (MNIST, Horowitz 2014 at 45nm):

| | Dense MLP (FP32) | SparkSNN (int32) | Ratio |
|:---|---:|---:|:---|
| Energy | 935 nJ | 107 nJ | **8.7× more efficient** |

The SNN performs more total operations, but they're overwhelmingly cheap int32 additions (0.1 pJ) rather than expensive FP32 MACs (4.6 pJ). Combined with spike-driven sparsity, this yields nearly an order of magnitude energy reduction. See the [paper draft](../research/integer-snn-paper-draft.md) for the full analysis.

### The Road from Biology to Backpropagation

Before arriving at surrogate gradients, we implemented and tested over ten variants of biologically-inspired learning rules:

| Approach | Best Result | Why It Failed |
|:---|---:|:---|
| Pure STDP | Chance | No credit assignment |
| Reward-modulated STDP | Chance | Global reward too diffuse |
| Weight perturbation | 98% (XOR only) | Too slow for larger problems |
| Arbiter neurons | 100% (Iris) | No hidden-layer credit assignment |
| Three-phase training (10 variants) | 86.7% (Iris) | Every improvement converged toward reimplementing backprop |

**Key finding:** Each improvement to our heuristic approach moved it closer to computing exact gradients. Output-targeted correction → output layer gradient. Causal constraints → pre × post activation. Layer-by-layer propagation → backpropagation. By the tenth iteration, we had reimplemented backprop with noisy heuristics. The 86.7% ceiling was the cost of that noise.

**Lesson:** Borrow biology for architecture, use mathematics for training.

---

## Learning Rules

The `LearningRule` interface allows runtime-swappable learning algorithms:

| Package | Rule | Type | Status |
|:---|:---|:---|:---|
| `learning/surrogate` | Surrogate gradient BPTT | Gradient-based | ✅ **Primary** — 100% Iris, 95.8% MNIST |
| `learning/arbiter` | Arbiter neurons + three-phase | Heuristic | ✅ 96.7-100% Iris |
| `learning/stdp` | Pure STDP | Hebbian | ✅ Implemented, chance-level on tasks |
| `learning/rstdp` | Reward-modulated STDP | Three-factor | ✅ Implemented |
| `learning/predictive` | Predictive rule (Saponati & Vinck 2023) | Self-supervised | ✅ Implemented |
| `learning/perturbation` | Weight perturbation | Gradient-free | ✅ First to solve XOR |

---

## Initialization

**Critical finding:** Sparse mixed-sign initialization is essential for spiking network learning.

With full connectivity and uniformly positive weights, all hidden neurons receive identical input and fire in unison (spike rate 20/20). This eliminates representational diversity and makes all learning rules ineffective.

**Our approach:**
- Input → Hidden: 20-50% connection probability, weights uniform in [-w_max, +w_max]
- Hidden → Output: 50-70% connection probability, weights uniform in [-w_max, +w_max]

Each neuron gets a unique receptive field from the start. **Diversity is a prerequisite for learning, not an outcome of it.**

---

## Future Work

### Near-term
- [ ] Larger benchmarks: Fashion-MNIST, CIFAR-10, temporal datasets (SHD, SSC)
- [ ] Deeper architectures with gradient management (residual connections, normalization)
- [ ] Learning rate scheduling, weight decay
- [ ] Int32 inference verification on MNIST (done for Iris)
- [ ] Adaptive weight scale factor (auto-calculate from max fan-in)

### The Vision: Continuous Activity and Persistent State

Everything above is feedforward classification — present a sample, run N timesteps, read output. The architecture was designed for something bigger.

The long-term goal is a network that maintains **ongoing dynamics** — not processing discrete samples, but continuously active, with memory and state emerging from persistent activation patterns rather than external storage. This means:

- **No "forward pass."** The network is always running. Input modulates ongoing activity, not initiates it.
- **Memory from dynamics.** Attractor states, recurrent loops, and sustained activity patterns encode information without a separate memory mechanism.
- **Temporal processing.** The decay function and refractory periods give the architecture inherent temporal dynamics. Signals that arrive at different times produce different results.
- **Structural plasticity.** Connections grow and prune based on activity (see `research/structural-plasticity.md`). The topology itself becomes learned.

This is the hard problem. It's also the interesting one.

### Modularity

The network topology should be **clustered** — semi-independent modules with dense internal connections but sparse cross-connections (inspired by Mithen 1996, cortical column theory, and Hawkins). Benefits:
- Each module specializes with simpler local learning
- Novel capabilities emerge from cross-module interaction
- Fault tolerance — damage to one module doesn't destroy the network

---

## Related Work

| Approach | Relationship | Key Difference |
|:---|:---|:---|
| **Spiking Neural Networks** | Closest paradigm | We prioritize ML utility over biological accuracy. Simpler dynamics, integer math. |
| **Neuromorphic hardware** (Loihi, TrueNorth) | Deployment target | We design software-first for commodity hardware, but map naturally to neuromorphic chips. |
| **Numenta / HTM** | Similar motivation | Different architecture. We use simpler neuron models and graph-based propagation. |
| **Graph Neural Networks** | Structural similarity | GNNs still use matrix ops at each node with synchronous passes. We're asynchronous and event-driven. |
| **snntorch / SpyTorch** | Training methodology | They train SNNs in PyTorch. We implement everything in Go with native integer inference. |

## References

- Horowitz, M. (2014). Computing's energy problem (and what we can do about it). *IEEE ISSCC*.
- Neftci, E. O., et al. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*.
- Saponati, M. & Vinck, M. (2023). Sequence anticipation and STDP emerge from a predictive learning rule. *Nature Communications*.
- Mithen, S. (1996). *The Prehistory of the Mind.* Thames & Hudson.

---

*This document is a living design, updated as the project evolves.*
