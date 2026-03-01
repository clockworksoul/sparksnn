# Surrogate Gradient Training for Biomimetic Networks

**Author:** Matt Titmus & Dross  
**Date:** 2026-03-01  
**Status:** Design proposal

---

## Context

We've spent several sessions building learning rules for the biomimetic
network: pure STDP, R-STDP, arbiter neurons, and most recently a
three-phase training methodology (reinforce → correct → reset). Each
approach hit a ceiling around 73-87% on Iris classification, while the
original arbiter approach reaches 100%.

The core problem: we've been hand-rolling credit assignment with
heuristics (wrong/correct score ratios, layer-by-layer backtracking)
instead of computing it properly. As we iterated, each improvement
brought us closer to reimplementing backpropagation — but with more
complexity and fewer mathematical guarantees.

**Decision:** Implement proper backpropagation adapted for our spiking
architecture using surrogate gradients. This aligns with our design
philosophy: "Biology is a parts catalog, not a blueprint."

## What We Keep

The following are sound and should not change:

- **LIF neuron model** — integer math, event-driven, lazy decay
- **Sparse connectivity** — mixed-sign, partial connectivity
- **Three-phase training cycle** — forward pass → backward pass → reset
- **Network topology** — graph-based, sparse, supports arbitrary structure
- **Integer inference** — all-integer forward pass for runtime efficiency

## What Changes

Replace the arbiter/heuristic learning rules with a proper gradient
computation during training. Training uses float64; inference stays
integer.

## The Dead Neuron Problem

Spiking neurons have a non-differentiable activation function. The spike
is a Heaviside step:

```
S[t] = 1 if U[t] >= threshold, else 0
```

The derivative of the Heaviside function is the Dirac delta — zero
everywhere except at the threshold, where it's infinity. This kills
gradient flow: ∂S/∂U = 0 almost everywhere, so no learning occurs.

## Surrogate Gradients

The standard solution (Neftci et al., 2019): keep the Heaviside during
the forward pass, but substitute a smooth approximation during the
backward pass.

**Forward pass:** Normal spiking behavior. `S = Θ(U - threshold)`

**Backward pass:** Replace ∂S/∂U with a smooth surrogate:

### Option A: Fast Sigmoid (recommended)
```
∂S̃/∂U = slope / (1 + |slope * (U - threshold)|)²
```
Where `slope` controls sharpness. Higher slope = closer to true
Heaviside but harder to train. Typical: slope = 25.

### Option B: Arctangent
```
∂S̃/∂U = 1/π * 1/(1 + (π * (U - threshold))²)
```

### Option C: Boxcar (simplest)
```
∂S̃/∂U = 1/width if |U - threshold| < width/2, else 0
```

**Recommendation:** Start with fast sigmoid. It's the most common in the
literature and works well across a range of network sizes.

## Architecture

### Training Loop (per sample)

```
Phase 1: Forward Pass
  - Present input to network
  - Run for T timesteps
  - At each timestep, record:
    - Membrane potentials U[t] for all neurons (float64)
    - Spike outputs S[t] for all neurons
    - Input currents I[t] for all neurons
  - Compute loss from output spike counts vs target

Phase 2: Backward Pass
  - Compute ∂L/∂S for output neurons (from loss function)
  - Backpropagate through time (BPTT):
    For t = T down to 1:
      For each layer (output → hidden → input):
        - ∂L/∂U[t] = ∂L/∂S[t] * surrogate(U[t])
        - ∂L/∂W += ∂L/∂U[t] * S_pre[t]  (pre-synaptic spikes)
        - Propagate: ∂L/∂S_pre[t] += W^T * ∂L/∂U[t]
        - Temporal: ∂L/∂U[t-1] += β * ∂L/∂U[t]  (decay factor)
  - Update weights: W -= learning_rate * ∂L/∂W

Phase 3: Reset
  - Clear membrane potentials to baseline
  - Clear recorded traces
```

### Loss Function

**Cross-entropy on membrane potential** (following snntorch approach):
- At each timestep, compute softmax over output membrane potentials
- Cross-entropy loss against one-hot target
- Sum loss across all timesteps

This encourages the correct output neuron's membrane to stay high
(firing frequently) while suppressing incorrect neurons.

**Alternative: spike count loss** (simpler, may work fine for us):
- Count output spikes over all timesteps
- Cross-entropy on the spike count vector
- Single loss per sample, not per timestep

Recommend starting with spike count loss — it's closer to how we
currently evaluate and simpler to implement.

## Implementation Plan

### New Types

```go
// SurrogateGradient computes the surrogate derivative for backprop.
// Forward: Heaviside. Backward: smooth approximation.
type SurrogateGradient interface {
    // Forward returns 1.0 if u >= threshold, else 0.0
    Forward(u, threshold float64) float64
    // Backward returns the surrogate derivative ∂S̃/∂U
    Backward(u, threshold float64) float64
}

// Trace holds the recorded state of a neuron at one timestep.
// Used during backpropagation through time.
type Trace struct {
    U     float64   // membrane potential (pre-spike)
    S     float64   // spike output (0 or 1)
    I     float64   // total input current
}

// TrainingState holds all recorded traces for a sample presentation.
type TrainingState struct {
    // Traces[neuronIdx][timestep]
    Traces [][]Trace
    // Gradients accumulated for each connection
    WeightGrad map[ConnectionID]float64
}

// Trainer implements surrogate gradient training.
type Trainer struct {
    Net       *Network
    Surrogate SurrogateGradient
    LR        float64       // learning rate
    Beta      float64       // decay factor (float64 version of DecayRate)
    NumSteps  int           // timesteps per sample
}
```

### Key Design Decisions

1. **Float64 training, integer inference.** The `Trainer` maintains a
   parallel float64 representation of weights during training. After
   training completes (or periodically), float64 weights are quantized
   back to int32 for the network. This is standard practice in
   quantization-aware training.

2. **BPTT depth = NumSteps.** We unroll through all timesteps. For our
   current 40-step presentations this is manageable. For deeper
   networks, truncated BPTT (e.g., 10-step windows) may be needed.

3. **Batch size = 1 initially.** Process one sample at a time, matching
   our current approach. Mini-batching can be added later by
   accumulating gradients.

4. **Optimizer: SGD with momentum, then Adam.** Start simple, upgrade
   if needed.

5. **Reset mechanism detached from gradient.** Following the standard
   approach: the post-fire reset is included in the forward pass but
   excluded from the backward pass. The surrogate gradient only
   approximates ∂S/∂U, not ∂R/∂U.

### Integration with Existing Code

The `Trainer` operates alongside the existing `Network`:

- **Forward pass** uses the existing `Stimulate`/`Tick` machinery but
  records float64 traces as a side effect
- **Backward pass** is entirely new code, operating on the traces
- **Weight update** writes back to `Connection.Weight` (int32) after
  quantizing from float64
- The existing `LearningRule` interface is bypassed during training
  (set to `NoOpLearning`)

### File Layout

```
learning/
  surrogate/
    surrogate.go       -- SurrogateGradient interface + implementations
    trainer.go         -- Trainer: forward recording, BPTT, weight update
    trainer_test.go    -- Unit tests
    loss.go            -- Loss functions (spike count CE, membrane CE)
benchmark/
  iris/
    iris_surrogate_test.go  -- Iris benchmark with surrogate training
```

## What This Gives Us

1. **Proper credit assignment.** Gradients flow through the entire
   network, not just heuristic layer-by-layer corrections.

2. **Scales to deeper networks.** Adding hidden layers "just works" —
   gradients propagate through all of them automatically.

3. **Well-understood training dynamics.** Learning rate schedules,
   optimizers, regularization — the entire ML training toolkit applies.

4. **Our architecture advantages remain.** Sparse, event-driven, integer
   inference. The training is conventional; the inference is not.

## What We Lose

1. **Biological plausibility of learning.** Backprop is not how the
   brain learns. But our design philosophy explicitly says we don't care
   about that: "Biology is a parts catalog, not a blueprint."

2. **Online/local learning.** The current STDP/arbiter approaches can
   learn continuously and locally. Surrogate gradients require storing
   traces and doing a full backward pass. For training this is fine;
   for online adaptation we may want to keep a simple Hebbian rule.

3. **Pure integer training.** Training will use float64. This is a
   pragmatic tradeoff — we get reliable convergence in exchange for
   float math during training only.

## Success Criteria

- **Iris:** ≥ 96% accuracy (matching our best arbiter result)
- **MNIST:** ≥ 95% accuracy (proving the approach scales)
- **Training time:** Comparable to or better than current approach
- **Integer inference:** Post-training accuracy within 1% of float
  accuracy after weight quantization

## References

- Neftci, E.O., Mostafa, H., Zenke, F. (2019). "Surrogate Gradient
  Learning in Spiking Neural Networks." IEEE Signal Processing Magazine.
  arXiv:1901.09948

- Zenke, F., Ganguli, S. (2018). "SuperSpike: Supervised Learning in
  Multilayer Spiking Neural Networks." Neural Computation 30(6).

- snntorch documentation: snntorch.readthedocs.io — practical PyTorch
  implementation of surrogate gradient training for LIF neurons.

- spytorch (github.com/fzenke/spytorch) — minimal tutorial
  implementation by Zenke.

## Next Steps

1. Implement `SurrogateGradient` interface with fast sigmoid
2. Implement `Trainer` with trace recording and BPTT
3. Implement spike count cross-entropy loss
4. Test on Iris
5. Test on MNIST
6. Tune: learning rate, surrogate slope, number of timesteps
