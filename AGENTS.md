# AGENTS.md — Biomimetic Neural Network

Quick orientation for AI agents working on this codebase.

## What This Is

A sparse, event-driven spiking neural network in pure Go. All integer math (int16 weights/activation, uint16 decay rates). No floats, no external dependencies. Biologically inspired: leaky integrate-and-fire neurons, spike-timing-dependent plasticity, lateral inhibition.

**Goal:** Prove that local learning rules (no backpropagation) can learn useful representations on commodity hardware.

## Go Module

- **Module:** `github.com/clockworksoul/biomimetic-network`
- **Go version:** 1.26+
- **Binary:** `/usr/local/go/bin/go` (not on PATH)
- **Dependencies:** None (stdlib only)

## Package Structure

```
biomimetic-network/
│
├── *.go                    # Root package: core types and engine
│   ├── neuron.go           # Neuron, Connection structs; ClampAdd; decay; Stimulate
│   ├── network.go          # Network: Tick, Connect, Stimulate, Reward, fireIdx
│   ├── learning.go         # LearningRule interface, IncomingConnection, NoOpLearning
│   └── serialize.go        # JSON save/load
│
├── learning/               # Swappable learning rule implementations
│   ├── stdp/               # Pure STDP — unsupervised Hebbian, timing → weight directly
│   ├── rstdp/              # Reward-modulated STDP — three-factor (timing × reward)
│   └── predictive/         # Predictive learning (Saponati & Vinck 2023) — self-supervised
│
├── celegans/               # C. elegans connectome loader + escape response demo
│   └── data loaded from data/celegans_connectome.csv
│
├── benchmark/              # Benchmark harness for testing learning rules
│   ├── benchmark.go        # Task interface, Sample, Tracker, convergence detection
│   ├── xor/                # XOR classification benchmark
│   └── cmd/xor/            # CLI: runs all rules against XOR, prints comparison
│
├── cmd/visualize/          # Network visualization tool
├── data/                   # Static data files (connectome CSV)
├── research/               # Design docs and paper analyses
│   ├── structural-plasticity.md   # Design for connection growth/pruning (next to implement)
│   ├── neuromorphic-landscape.md  # Survey of Loihi 2, TrueNorth, HTM, STDP
│   └── predictive-learning-rule.md # Analysis of Saponati & Vinck 2023
└── patent/                 # Provisional patent materials
```

## Core Types (root package)

**Neuron** — Leaky integrate-and-fire. Fields: `Activation`, `Baseline`, `Threshold`, `DecayRate` (uint16 fixed-point), `LastFired`, `LastInteraction`, `Connections []Connection`.

**Connection** — Directed synapse. Fields: `Target` (uint32 index), `Weight` (int16), `Eligibility` (int16 trace for learning).

**Network** — Neuron array + tick-based simulation. Key methods: `Stimulate()`, `Tick()`, `TickN()`, `Connect()`, `Reward()`. Signals propagate with 1-tick delay per hop.

**LearningRule** — Interface with 4 methods: `OnSpikePropagation`, `OnPostFire`, `OnReward`, `Maintain`. Implementations in `learning/` subpackages.

## Key Design Decisions

- **All integer math.** Weights are int16 (-32768 to 32767). Activation is int16. Decay is uint16 fixed-point (fraction of 65536). No floating point anywhere in the hot path.
- **`ClampAdd`** for overflow-safe int16 addition (exported for subpackage use).
- **Lazy decay.** Neurons only compute decay when next stimulated. Idle neurons cost zero.
- **Counter starts at 1** so `LastFired == 0` unambiguously means "never fired."
- **Incoming index built lazily**, invalidated on `Connect()` or structural changes.
- **Learning rules import root as `bio`** to avoid circular deps.

## Running Tests

```bash
/usr/local/go/bin/go test ./...
```

Currently ~70 tests across 6 packages. All should pass.

## Running Benchmarks

```bash
/usr/local/go/bin/go run ./benchmark/cmd/xor
```

## Key Docs

- **DESIGN.md** — Architecture, neuron model, learning rules, rationale
- **TODO.md** — Current priorities and roadmap
- **research/structural-plasticity.md** — Next major feature design

## ⚠️ Important Rules

- **NO CODEX.** This is personal IP, not Yext work. Do not use any Yext-paid tools (Codex, Copilot via Yext license, etc.). Edit code directly.
- **No floats in core.** All neuron/connection/learning math must be integer. Research code and benchmarks can use floats for reporting/analysis.
- **Patent pending.** Don't make the repo public or post to arXiv until the provisional patent is filed.
