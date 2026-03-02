# SparkSNN

An energy-efficient spiking neural network architecture using integer arithmetic, sparse connectivity, and event-driven computation.

## What Is This?

SparkSNN is a spiking neural network framework written in Go. It performs inference entirely in **integer arithmetic** — no floating-point operations — making it suitable for deployment on resource-constrained hardware without FPUs.

**Key properties:**

- **Integer inference** — int32 weights, int32 membrane potentials, fixed-point decay. No floats at runtime.
- **Event-driven** — only spiking neurons trigger downstream computation. Cost scales with *activity*, not network size.
- **Lazy decay** — idle neurons cost zero computation. Membrane decay is calculated on-demand, not per-tick. Two-thirds of neurons are idle at any given timestep.
- **Sparse connectivity** — 10-30% connection density, further reducing computation and memory.
- **Dual-domain training** — train in float64 with surrogate gradient BPTT, deploy in int32 with zero prediction degradation.
- **~11× more energy efficient** than an equivalent dense FP32 MLP at inference.

## Results

| Benchmark | Architecture | Accuracy | Notes |
|:---|:---|---:|:---|
| Iris | 40 → 20 → 3 | **100%** | Population coded, SGD, 300 epochs |
| MNIST | 784 → 256 → 10 (20% sparse) | **95.8%** | Adam, 15 epochs, ~13 min on CPU |
| MNIST (tuned) | 784 → 512 → 10 (30% sparse) | **97.21%** | Adam + LR decay, 42 epochs |
| MNIST (deep) | 784 → 256 → 128 → 10 | 93.7% | Naive depth doesn't help for MNIST |

Int32 inference produces **identical predictions** to float64 training on all Iris test samples (verified).

## Architecture

The neuron model is Leaky Integrate-and-Fire (LIF), implemented in integer math:

```
membrane = (membrane * decayRate) >> 16   // fixed-point decay (1 shift)
membrane += incoming_current               // int32 add per spike
if membrane >= threshold:                  // int32 compare
    spike → propagate to connected neurons
    membrane -= threshold                  // soft reset
```

When a spike arrives at a synapse, the operation is a **single int32 addition** (add the weight to the target membrane). No multiply needed — the spike is binary.

Idle neurons perform no computation whatsoever — not even decay. Decay is calculated lazily when a neuron next receives input, collapsing multiple timesteps into a single operation. In our trained MNIST network, **66.3% of neurons are idle per timestep**, meaning nearly two-thirds of the network costs nothing on any given tick.

## Training

SparkSNN uses surrogate gradient backpropagation through time (BPTT) with a fast sigmoid surrogate (Neftci et al., 2019). Training maintains float64 shadow weights that are quantized to int32 after each update.

```go
trainer := surrogate.NewTrainer(net, cfg, weightScale)
trainer.EnableAdam()

for _, sample := range trainingSamples {
    loss := trainer.TrainSample(inputValues, label)
}
// Weights are automatically synced to int32 network after each update
```

## Project Structure

```
neuron.go                    # Integer LIF neuron model
network.go                   # Event-driven network simulation
learning.go                  # LearningRule interface
learning/surrogate/          # Surrogate gradient training
  surrogate.go               #   Gradient interface (FastSigmoid, Boxcar)
  loss.go                    #   Cross-entropy loss functions
  trainer.go                 #   BPTT trainer with Adam + int32 quantization
learning/arbiter/            # Arbiter neuron learning (biological approach)
learning/stdp/               # Reward-modulated STDP
learning/predictive/         # Predictive learning rule
benchmark/iris/              # Iris dataset + benchmarks
benchmark/mnist/             # MNIST dataset + benchmarks
celegans/                    # C. elegans connectome demo (302 neurons)
```

## Quick Start

```bash
# Run Iris benchmark
go test -v -run TestIrisSurrogate ./benchmark/iris/

# Run MNIST benchmark (requires data files in data/mnist/)
go test -v -run TestMNISTSurrogate -timeout 30m ./benchmark/mnist/

# Run tuned MNIST benchmark
go test -v -run TestMNISTTuned -timeout 120m ./benchmark/mnist/

# Run all tests
go test ./...
```

## Energy Efficiency

Per-inference energy comparison for MNIST (Horowitz 2014 estimates at 45nm):

| | Dense MLP (FP32) | SparkSNN (conservative) | SparkSNN (with lazy decay) |
|:---|---:|---:|---:|
| **Energy** | 935 nJ | 107 nJ | **85 nJ** |
| **vs. MLP** | — | 8.7× | **11.0×** |

The SNN performs *more* total operations, but they're overwhelmingly cheap int32 additions (0.1 pJ) rather than expensive FP32 multiply-accumulates (4.6 pJ). Combined with spike-driven sparsity and lazy decay, this yields an **order-of-magnitude energy reduction**.

The network naturally adapts its computational effort to input complexity: digit "1" (25.5% active) uses half the compute of digit "0" (37.9% active). A conventional MLP does the same work regardless of input.

## References

- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51-63.
- Horowitz, M. (2014). Computing's energy problem (and what we can do about it). *IEEE ISSCC*, 10-14.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
