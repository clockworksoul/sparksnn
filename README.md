# SparkSNN

An energy-efficient spiking neural network architecture using integer arithmetic, sparse connectivity, and event-driven computation.

## What Is This?

SparkSNN is a spiking neural network framework written in Go. It performs inference entirely in **integer arithmetic** — no floating-point operations — making it suitable for deployment on resource-constrained hardware without FPUs.

**Key properties:**

- **Integer inference** — int32 weights, int32 membrane potentials, fixed-point decay. No floats at runtime.
- **Event-driven** — only spiking neurons trigger downstream computation. Cost scales with *activity*, not network size.
- **Sparse connectivity** — 10-20% connection density, further reducing computation and memory.
- **Dual-domain training** — train in float64 with surrogate gradient BPTT, deploy in int32 with zero prediction degradation.
- **~8.7× more energy efficient** than an equivalent dense FP32 MLP at inference (see our [paper draft](research/surrogate-gradient-training.md) for the full analysis).

## Results

| Benchmark | Architecture | Accuracy | Notes |
|:---|:---|---:|:---|
| Iris | 40 → 20 → 3 | **100%** | Population coded, SGD, 300 epochs |
| MNIST | 784 → 256 → 10 (20% sparse) | **95.8%** | Adam, 15 epochs, ~13 min on CPU |
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
research/                    # Design docs and paper drafts
```

## Quick Start

```bash
# Run Iris benchmark
go test -v -run TestIrisSurrogate ./benchmark/iris/

# Run MNIST benchmark (requires data files in data/mnist/)
go test -v -run TestMNISTSurrogate -timeout 30m ./benchmark/mnist/

# Run all tests
go test ./...
```

## Energy Efficiency

Per-inference energy comparison (MNIST, Horowitz 2014 estimates at 45nm):

| | Dense MLP (FP32) | SparkSNN (int32) |
|:---|---:|---:|
| **Energy** | 935 nJ | 107 nJ |
| Multiply ops | 203,264 | 25,620 |
| Add ops | ~266 | 257,190 |

The SNN performs *more* total operations, but they're overwhelmingly cheap int32 additions (0.1 pJ) rather than expensive FP32 multiply-accumulates (4.6 pJ). Combined with spike-driven sparsity, this yields an **8.7× energy reduction**.

## References

- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51-63.
- Horowitz, M. (2014). Computing's energy problem (and what we can do about it). *IEEE ISSCC*, 10-14.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
