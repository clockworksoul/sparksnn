# Arbiter Neurons — Biological Backpropagation

*Hypothesis developed 2026-02-21 in conversation with Matt.*

## The Problem

Matrix-based neural networks use backpropagation to adjust weights during training. But in a network of independent spiking neurons, there's no global loss function and no gradient to propagate. How do individual neurons learn whether they contributed to a correct or incorrect outcome?

## The Hypothesis

**Dedicated backward-pointing "arbiter neurons"** form parallel layers alongside forward computation layers, providing error signals that enable training. This is biological backpropagation.

## Why Negative Signals, Not Positive

Key insight (Matt): STDP already handles the positive case. Neurons that fire together wire together — repeated successful sequences strengthen themselves automatically just by being used. The positive training signal is *free*.

What's missing is the brake. Without a negative signal, the network just strengthens whatever patterns fire most frequently, regardless of correctness. Pure Hebbian learning = strong, confident pathways that are confidently wrong.

Therefore, what's needed is a **negative error signal** to work counter to repetition-based reinforcement.

## Biological Precedent

### Cerebellar Climbing Fibers
The clearest biological example. The cerebellum has:
- **Purkinje cells** — forward computation neurons
- **Climbing fibers** (from the inferior olive) — wrap around Purkinje cells and fire on *errors*

Climbing fibers fire at low baseline rates (~1-4 Hz). Silence means "carry on, you're fine." A burst means "that was wrong, adjust." It's an exception-based system — only speak up when there's a problem.

This is energy efficient: if 90% of what the network does is correct, why waste spikes confirming it? Just signal the 10% that needs fixing.

### Cortical Feedback Connections
- Layers 5/6 project back to layers 1/2/3
- Predictive coding theory: forward connections carry data up, backward connections carry *prediction errors* down
- "Here's what I expected. You sent something different. Adjust."

### Computational Neuroscience
- **Feedback Alignment** (Lillicrap et al. 2016) — even *random* backward weights work for training
- **Target Propagation** — each layer receives a "target" from the layer above
- **Saponati & Vinck predictive rule** — already implemented in our `learning/predictive/` package

## Not All Circuits Learn Equally

Critical insight: biological neural circuits have different plasticity profiles.
- **Retinal neurons** are relatively fixed — seeing more stuff doesn't make them better at seeing
- **Cerebellar circuits** are *specifically architected* for learning (climbing fibers = dedicated error infrastructure)

This means our network shouldn't have uniform plasticity. Some regions should be stable processors; others should be learning-optimized with full arbiter infrastructure.

## Proposed Mechanism

```
Forward neuron fires → connection strengthened via STDP (automatic, free)

Arbiter neuron detects error → fires → SUPPRESSES/WEAKENS the connections
that just activated (anti-STDP)

Default state: arbiter is quiet (normal STDP training proceeds)
Error state: arbiter fires (recent pathway is depressed, not potentiated)
```

A well-timed arbiter spike *depresses* rather than potentiates the recent pathway. This is clean, implementable, and biologically grounded.

## Architecture

Each trainable hidden layer gets a parallel **arbiter layer**:
- Arbiter neurons point backward (toward the layer they supervise)
- They receive signals from downstream (outcome/error information)
- They fire to suppress incorrect patterns in the forward layer
- They are quiet when the forward layer is performing correctly

## Open Questions

1. **How do arbiters know what's wrong?** They need some error signal themselves. In the cerebellum, the inferior olive computes errors. What's our equivalent? Output comparison? Prediction mismatch?
2. **Do arbiter neurons learn?** In cerebellum, climbing fibers are relatively hardwired. In cortex, feedback connections are plastic. Which model do we want?
3. **Timing:** How quickly must the arbiter fire after the forward activation to correctly depress the right connections? (Relates to STDP timing windows)
4. **Scope:** Does each arbiter neuron supervise one forward neuron, a group, or an entire layer?
5. **Connection to layers conversation:** How does this interact with layer topology? (Discussion planned for 2026-02-22)

## Naming

**Arbiter neurons / arbiter layer** — they *judge* whether forward neurons were correct.

Considered alternatives:
- "Echo neurons" — describes mechanism (signal echoes back) but not purpose
- "Sentinel neurons" — implies protection, not teaching
- "Critic neurons" — good (actor-critic precedent) but less distinctive
- "Watcher neurons" — original working name

"Arbiter" captures the purpose (judgment/evaluation) rather than the mechanism, which is more durable as the implementation evolves.

## References to Explore

- Marr (1969) — original cerebellar learning theory
- Albus (1971) — cerebellar model articulation controller
- Lillicrap et al. (2016) — Feedback Alignment
- Saponati & Vinck (2023) — predictive learning rule
- Rao & Ballard (1999) — predictive coding in visual cortex
