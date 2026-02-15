# Biomimetic Neural Network

A speculative architecture for neural computation using sparse, event-driven, independently functional units instead of dense matrix transforms.

## What Is This?

Modern neural networks rely on brute-force matrix multiplication — every forward pass computes across the entire weight matrix regardless of relevance. Biological neural systems don't work this way. They're sparse, event-driven, and energy-efficient.

This project explores whether we can build a computational neural architecture that follows biological principles more faithfully:

- **Biomimetic neurons** — simple data structures with activation levels, decay, and firing thresholds
- **Event-driven computation** — idle neurons cost zero; only active signal paths consume resources
- **Integer arithmetic** — int16 weights and activations, not floating point. Cheaper per operation, smaller per unit.
- **Array-indexed connections** — cache-friendly, serializable, partitionable
- **Modular topology** — inspired by cognitive modularity (Mithen) and cortical columns (Hawkins)

## Why?

Because a human brain runs on ~20 watts and a single H100 GPU pulls 700. The efficiency gap isn't incremental — it's orders of magnitude. We think the architecture is part of the reason.

## Status

🧪 **Early-stage thought experiment.** We're working through the theory, building a reading list, and figuring out the hard problems (learning rules, output encoding, representational equivalence) before writing code.

## Documents

- **[DESIGN.md](DESIGN.md)** — Architecture design and decisions
- **[READING-LIST.md](READING-LIST.md)** — Papers, textbooks, and references organized by topic
- **[STUDY-TOPICS.md](STUDY-TOPICS.md)** — Feynman Technique study guide for prerequisite concepts

## Key Open Questions

1. **Learning** — How do connections form, strengthen, and prune? STDP is the leading candidate, but this is the hard problem.
2. **Output** — How do we read information out of the network? Population coding, rate coding, and attractor states are all on the table.
3. **Equivalence** — Can this architecture learn representations equivalent to what matrix transforms produce?

## License

All rights reserved.
