# AGENTS.md — SparkSNN

Quick orientation for AI agents working on this codebase.

## What This Is

An energy-efficient spiking neural network framework in pure Go. All integer arithmetic for inference (int32 weights/activation, uint16 fixed-point decay). Event-driven, sparse connectivity, lazy decay (idle neurons cost zero computation).

**Current status:** Surrogate gradient training works. MNIST 97.21%, Iris 100%, with verified zero quantization degradation between float64 training and int32 inference. Paper draft in progress.

## Go Module

- **Module:** `github.com/clockworksoul/sparksnn`
- **Go version:** 1.26+
- **Binary:** `/usr/local/go/bin/go` (not on PATH)
- **Dependencies:** None (stdlib only)

## Package Structure

```
sparksnn/
│
├── *.go                    # Root package: core types and engine
│   ├── neuron.go           # Neuron, Connection structs; ClampAdd32; decay; Stimulate
│   ├── network.go          # Network: Tick, Connect, Stimulate, Reward, fireIdx
│   ├── learning.go         # LearningRule interface, IncomingConnection, NoOpLearning
│   ├── serialize.go        # JSON save/load
│   ├── plasticity.go       # Structural plasticity (connection growth/pruning)
│   ├── spatial_plasticity.go # Spatial-aware plasticity
│   └── development.go      # Developmental growth rules
│
├── learning/               # Swappable learning rule implementations
│   ├── stdp/               # Pure STDP — unsupervised Hebbian, timing → weight directly
│   ├── rstdp/              # Reward-modulated STDP — three-factor (timing × reward)
│   ├── predictive/         # Predictive learning (Saponati & Vinck 2023) — self-supervised
│   ├── arbiter/            # Arbiter neurons — biological error signals for credit assignment
│   ├── hybrid/             # Hybrid learning combining multiple rules
│   ├── perturbation/       # Weight perturbation-based learning
│   └── surrogate/          # ⭐ Surrogate gradient BPTT — the primary training method
│       ├── surrogate.go    # Surrogate gradient functions (FastSigmoid, Boxcar)
│       ├── loss.go         # Spike count and membrane cross-entropy loss
│       └── trainer.go      # BPTT trainer with Adam optimizer and int32 quantization
│
├── celegans/               # C. elegans connectome loader + escape response demo
│   └── data loaded from data/celegans_connectome.csv
│
├── benchmark/              # Benchmark harness for testing learning rules
│   ├── benchmark.go        # Task interface, Sample, Tracker, convergence detection
│   ├── xor/                # XOR classification benchmark
│   ├── iris/               # Iris dataset benchmarks (arbiter, surrogate, three-phase)
│   │   ├── analysis.go     # Result analysis and comparison tools
│   │   └── data.go         # Iris dataset with population coding
│   ├── mnist/              # MNIST benchmarks (surrogate, tuned, deep, activity measurement)
│   │   └── mnist.go        # MNIST data loading and encoding
│   └── cmd/xor/            # CLI: runs all rules against XOR, prints comparison
│
├── cmd/visualize/          # Network visualization tool
├── data/                   # Static data files (connectome CSV, MNIST)
├── DESIGN.md               # Architecture, neuron model, learning rules, rationale
├── TODO.md                 # Current priorities and roadmap
└── README.md               # Project overview
```

## Core Types (root package)

**Neuron** — Leaky integrate-and-fire. Fields: `Activation` (int32), `Baseline` (int32), `Threshold` (int32), `DecayRate` (uint16 fixed-point fraction of 65536), `LastFired`, `LastInteraction`, `Connections []Connection`.

**Connection** — Directed synapse. Fields: `Target` (uint32 index), `Weight` (int32), `Eligibility` (int32 trace for learning).

**Network** — Neuron array + tick-based simulation. Key methods: `Stimulate()`, `Tick()`, `TickN()`, `Connect()`, `Reward()`. Signals propagate with 1-tick delay per hop.

**LearningRule** — Interface with 4 methods: `OnSpikePropagation`, `OnPostFire`, `OnReward`, `Maintain`. Implementations in `learning/` subpackages.

## Training (Dual-Domain Approach)

The primary training method is **surrogate gradient BPTT** (`learning/surrogate/`):

1. **Float64 domain:** Shadow weights in float64, continuous membrane simulation, surrogate gradients, Adam optimizer
2. **Int32 domain:** Quantized weights for inference via configurable weight scale factor α

A `Trainer` manages both domains. Key methods: `Train()`, `Predict()`, `EnableAdam()`. After each weight update, float64 shadows are quantized and written to the int32 network.

**Weight scale factor (α):** Converts between domains: `w_int32 = round(α × w_float64)`. Use 2^16 for small networks, 2^20 for MNIST-scale.

## Key Design Decisions

- **All integer inference.** Weights are int32. Activation is int32. Decay is uint16 fixed-point (fraction of 65536). No floating point in the inference path.
- **`ClampAdd32`** for overflow-safe int32 addition (exported for subpackage use).
- **Lazy decay.** Neurons only compute decay when next stimulated. Idle neurons cost zero.
- **Counter starts at 1** so `LastFired == 0` unambiguously means "never fired."
- **Incoming index built lazily**, invalidated on `Connect()` or structural changes.
- **Learning rules import root as `bio`** to avoid circular deps.
- **Sparse mixed-sign initialization** is essential — uniform positive weights cause all neurons to fire identically.

## Running Tests

```bash
/usr/local/go/bin/go test ./...
```

**Note:** MNIST tests download the dataset on first run and are slow (~minutes). Iris and XOR tests are fast.

## Running Benchmarks

```bash
# XOR comparison across learning rules
/usr/local/go/bin/go run ./benchmark/cmd/xor

# MNIST surrogate gradient (in test form)
/usr/local/go/bin/go test ./benchmark/mnist/ -run TestMNISTSurrogate -v -timeout 30m
```

## Key Results

| Benchmark | Architecture | Accuracy | Method |
|:---|:---|---:|:---|
| Iris | 40→20→3 | 100% | Surrogate gradient |
| MNIST | 784→512→10 (30% sparse) | 97.21% | Surrogate + Adam + LR decay |
| Int32/Float64 parity | Iris | 0 mismatches | Perfect quantization |

## Key Docs

- **DESIGN.md** — Architecture, neuron model, learning rules, rationale
- **TODO.md** — Current priorities and roadmap
- **README.md** — Project overview and quick start

## ⚠️ Important Rules

- **NO CODEX.** This is personal IP, not Yext work. Do not use any Yext-paid tools (Codex, Copilot via Yext license, etc.). Edit code directly.
- **No floats in inference.** All neuron/connection math at inference time must be integer. Training uses float64 (the dual-domain approach). Research code and benchmarks can use floats for reporting/analysis.
- **Paper in progress.** Targeting conference submission (draft maintained separately).
