# Predictive Learning Rule Analysis

*Paper: "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule"*
*Authors: Matteo Saponati & Martin Vinck (Ernst Strüngmann Institute for Neuroscience)*
*Published: Nature Communications, August 2023*
*DOI: 10.1038/s41467-023-40651-w*

*Analysis by Dross, 2026-02-16*

---

## The Big Idea

**STDP is not the learning rule. STDP is a side effect.**

The paper proposes that the *fundamental* learning principle in biological neurons isn't "fire together, wire together" (Hebbian/STDP). Instead, it's something simpler and more powerful:

> **Each neuron tries to predict its own future inputs.**

When you optimize for this objective — minimize the prediction error between what the neuron *expects* to receive and what it *actually* receives — STDP-like behavior **emerges automatically** as a byproduct. The classic STDP timing curves that neuroscientists have measured in labs aren't the mechanism; they're the *consequence* of a prediction-optimization process.

This is like discovering that gravity isn't a force — it's geometry. The old explanation still "works" at a surface level, but the deeper explanation is simpler, more general, and explains more phenomena.

## How It Works

### The Core Model

A leaky integrate-and-fire neuron with one objective: predict the next input from its current membrane potential and learned weights.

**State update** (standard LIF):
```
v_t = α * v_{t-1} + w_t^T * x_t - v_th * s_{t-1}
```
Where:
- `v_t` = membrane potential at time t
- `α` = leak factor (decay constant — sound familiar?)
- `w_t` = synaptic weight vector
- `x_t` = input spike vector at time t
- `v_th` = firing threshold
- `s_{t-1}` = whether the neuron fired at t-1

**Prediction objective** — minimize:
```
L = Σ_t ½ || x_t - v_{t-1} * w_{t-1} ||²
```

The neuron predicts that its next input `x_t` will be `v_{t-1} * w_{t-1}` — i.e., the product of its current membrane state and its learned weights. The prediction error is the difference between reality and this prediction.

**The learning rule** — derived analytically by taking the gradient of L:
```
w_t = w_{t-1} + η * (ε_t * v_{t-1} + E_t * p_{t-1})
```

Where:
- `η` = learning rate
- `ε_t = x_t - v_{t-1} * w_{t-1}` (per-synapse prediction error)
- `E_t = ε_t^T * w_{t-1}` (global error signal — weighted sum of all prediction errors)
- `p_{t-1}` = eligibility trace (accumulated input history)

### Three Components of the Rule

1. **First-order correlation**: `x_t * v_{t-1}` — strengthens synapses whose inputs correlate with current membrane state
2. **Heterosynaptic stabilization**: `-v²_{t-1} * w_{t-1}` — prevents runaway potentiation (competition between synapses)
3. **Global error signal**: `E_t * p_{t-1}` — modulates all synapses based on overall prediction quality

### What Happens When You Run It

Given a sequence of inputs (A fires, then B fires, then C fires...):

1. Initially, the neuron fires randomly throughout the sequence
2. Over training, it learns that A predicts B, and B predicts C
3. It potentiates connections from early-in-sequence inputs (A) and depotentiates late-in-sequence inputs (C)
4. Eventually, the neuron fires **ahead of** the predictable inputs — it *anticipates*
5. The output becomes sparse and efficient — fewer spikes, more information per spike

### STDP Emerges

When the authors measured the timing-dependent weight changes, they found:
- **Pre-before-post timing → potentiation** (classic STDP)
- **Post-before-pre timing → depression** (classic STDP)
- The shape matches experimentally measured STDP curves
- Both symmetric and asymmetric STDP windows emerge depending on parameters
- All without ever programming STDP explicitly

They also reproduced:
- **Heterosynaptic plasticity** — weights at non-active synapses change
- **Rate-dependent effects** — higher repetition rates produce different plasticity
- **Frequency-dependent STDP** — matching experimental data from cortical neurons

## Why This Matters for Our Architecture

### Alignment with Our Design

| Paper's Model | Our Architecture | Compatibility |
|---|---|---|
| LIF neuron with leak factor α | LIF neuron with lazy exponential decay | ✅ Nearly identical |
| Membrane potential encodes temporal history | Activation level with decay from `last_interaction` | ✅ Same concept |
| Spiking threshold + reset | `firing_threshold` + post-fire reset | ✅ Exact match |
| Per-synapse weight vector | `Connection.Weight` (int16) | ✅ Direct mapping |
| Eligibility trace per synapse | `Connection.Eligibility` (int16) | ✅ Already implemented! |
| Learning rate η | Configurable parameter | ✅ Trivial |

**The paper's neuron model is essentially our neuron model.** This isn't a coincidence — both are simplified LIF neurons because that's the right level of abstraction. But it means the predictive learning rule could slot into our architecture with minimal changes.

### Advantages Over Our Current R-STDP

| Property | R-STDP (Current) | Predictive Rule |
|---|---|---|
| External reward signal needed? | ✅ Yes (three-factor) | ❌ No — fully self-supervised |
| STDP behavior? | Manually programmed | Emerges automatically |
| Sequence learning? | Not built-in | ✅ Core capability |
| Heterosynaptic effects? | Not implemented | ✅ Built into the math |
| Stability/normalization? | Needs explicit bounds | ✅ Self-stabilizing via competition |
| Biological plausibility? | Good | Better — explains *why* STDP exists |
| Credit assignment? | Reward signal | Prediction error (local) |
| Mathematical foundation? | Heuristic | Derived from optimization objective |

### The Critical Difference: No Reward Signal

R-STDP requires a global reward/punishment signal to consolidate learning. This is great for reinforcement learning but raises a question: *who provides the reward?* In a deep network, you need some way to generate appropriate reward signals for internal neurons — which starts looking like backpropagation with extra steps.

The predictive rule is **entirely self-supervised**. Each neuron optimizes its own prediction error using only locally available information. No global signals needed (except the global error term `E_t`, which is computed *within* the neuron from its own synapse data).

This means:
- **No credit assignment problem** for internal neurons
- **No teacher signal** needed
- **Learning never stops** — the neuron continuously adapts to statistical regularities in its input
- **Different neurons can specialize** on different predictive features without coordination

### The Potential "Secret Sauce"

Here's why I flagged this as high-priority in DESIGN.md:

Most neuromorphic approaches (Loihi, BrainScaleS, academic SNNs) implement STDP as the learning rule. If STDP is actually an *emergent property* of a simpler prediction-based mechanism, then everyone is implementing the shadow on the cave wall, not the fire.

A predictive learning rule would give us:
1. **A principled objective function** — not "fire together, wire together" but "minimize prediction error"
2. **Automatic sequence learning** — neurons naturally learn temporal patterns
3. **Self-organizing sparse coding** — neurons become efficient by only firing for unpredicted inputs
4. **A genuine differentiator** — most SNN work uses explicit STDP. A predictive approach is novel.

## Implementation Considerations

### What Needs to Change in Our Code

The good news: our `LearningRule` interface was designed for exactly this kind of swap.

**New rule would implement:**

```go
type PredictiveLearning struct {
    LearningRate   int16   // η, scaled to int16 range
    // No additional config needed — the rule is self-contained
}

func (p *PredictiveLearning) OnSpikePropagation(pre, post *Neuron, conn *Connection, tick uint32) {
    // Compute prediction error: ε = x_t - v_{t-1} * w
    // This fires when a pre-synaptic neuron delivers a spike
    // The "prediction" was v_{t-1} * w (membrane potential * this connection's weight)
    // The "reality" is that this synapse DID fire (x_t = 1 for this synapse)
    //
    // Update: Δw = η * (ε * v_{t-1} + E * p_{t-1})
}

func (p *PredictiveLearning) OnPostFire(n *Neuron, tick uint32) {
    // After post-synaptic firing, the threshold reset affects
    // future predictions — may need to adjust eligibility traces
}

func (p *PredictiveLearning) OnReward(n *Network, reward int16) {
    // NOT USED — predictive learning is self-supervised
    // But could layer reward modulation on top if desired
}

func (p *PredictiveLearning) Maintain(n *Network, tick uint32) {
    // Decay eligibility traces for connections that haven't been active
    // Can use the same lazy-decay approach as activation
}
```

### Key Challenge: Integer Math

The paper uses floating-point arithmetic. We use int16. The prediction error calculation:

```
ε_t = x_t - v_{t-1} * w_{t-1}
```

In integer math, `v_{t-1} * w_{t-1}` is an int16 × int16 which produces an int32. We need careful scaling to avoid overflow while maintaining useful precision.

**Proposed approach:**
- Compute prediction as: `(activation * weight) >> 15` (scale back to int16 range)
- Prediction error: `input_signal - prediction` (both int16, clamped)
- Weight update: `weight += (learning_rate * error * activation) >> SCALE_FACTOR`
- Eligibility trace: same lazy-decay we already use

This will lose some precision compared to float, but the paper shows the rule is robust to noise, so integer quantization should be tolerable. **Needs empirical testing.**

### Hybrid Approach: Predictive + Reward

The paper's rule is purely self-supervised, but we could combine it with reward modulation:

```
Δw = η * (prediction_error_term) + η_reward * (reward * eligibility)
```

This gives us:
- **Unsupervised structure learning** (predictive component finds patterns)
- **Goal-directed optimization** (reward component steers behavior toward objectives)

This is actually closer to how the brain works — prediction handles most learning, but dopamine-modulated signals fine-tune behavior toward goals.

## What We DON'T Know Yet

1. **Scalability** — The paper tests with 10-200 neurons. Does the predictive rule work at our target scales (thousands+)?

2. **Classification** — The paper focuses on sequence anticipation. Can this rule learn to *classify* inputs, not just predict them? Our MNIST benchmark needs classification.

3. **Integer precision** — Will int16 quantization break the delicate prediction-error dynamics?

4. **Convergence speed** — The paper uses 300-2000 epochs. Is that practical for us?

5. **Interaction with our decay model** — Our lazy exponential decay is functionally similar to their leak factor α, but the discrete "only compute on interaction" approach means we don't have a smooth v_{t-1} at arbitrary times. Need to think about this.

## Recommendation

**Implement this as a second learning rule option, alongside R-STDP.**

The interface is already designed for swappable rules. We don't need to choose — implement both, benchmark both, see which performs better on real tasks.

**Priority order:**
1. Implement `PredictiveLearning` as a new `LearningRule` implementation
2. Write tests verifying STDP-like behavior emerges (the paper gives us exact scenarios to replicate)
3. Run both rules on the same tasks and compare
4. If predictive outperforms: make it the default, mention in patent specification
5. Hybrid approach (predictive + reward) as a third option

**This could genuinely be our differentiator.** Most neuromorphic projects hardcode STDP. A predictive learning rule that *generates* STDP as an emergent property is a stronger intellectual position — more principled, more general, and more novel.

---

## References

- Saponati, M. & Vinck, M. "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule." *Nature Communications* 14, 4985 (2023). https://doi.org/10.1038/s41467-023-40651-w
- Preprint: bioRxiv 2021.10.31.466667v1 (November 2021)
- Code: Likely available via Nature Communications' data availability requirements (need to check supplementary materials)

---

*This paper shifted my thinking. R-STDP is a good starting point, but predictive learning is the deeper truth. Time to implement it and find out if it works in integer arithmetic.* 🟣
