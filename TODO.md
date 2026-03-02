# TODO — SparkSNN

*Updated 2026-03-01*

---

## 🔴 Next Up

- [ ] **Paper submission** — Draft in progress. Needs review pass, figures, and venue selection.
- [ ] **Fashion-MNIST benchmark** — Harder than MNIST, same input shape. Good next test of generalization.
- [ ] **Adaptive weight scale** — Auto-calculate intScale from max fan-in × max weight to avoid overflow while maximizing precision.
- [ ] **Int32 inference verification on MNIST** — Done for Iris (0 mismatches). Should verify on MNIST too.

## 🧪 Accuracy Improvements

- [ ] **Convolutional structure** — Weight sharing / local receptive fields for spatial feature detection. Likely needed for 99%+ MNIST or any image task beyond it.
- [ ] **Deeper networks** — Naive depth hurts (93.7% vs 95.8%). Investigate residual connections, layer normalization, or gradient clipping for deeper SNNs.
- [ ] **Dropout / regularization** — May help with the 97.2% plateau.
- [ ] **Larger benchmarks** — CIFAR-10, temporal datasets (SHD, SSC).

## ⚡ Performance & Efficiency

- [ ] **Parallelism** — Tick-level goroutine parallelism. Neurons within a tick are independent.
- [ ] **Per-layer activity instrumentation** — Current activity test tracks total neurons. Break down by layer for more precise energy estimates.
- [ ] **Profile at scale** — 10K, 100K, 1M neuron networks for memory and CPU benchmarks.
- [ ] **Neuromorphic hardware deployment** — Test on Intel Loihi or FPGA where integer constraint gives direct hardware benefits.

## 🧠 The Vision: Continuous Activity

- [ ] **Persistent state** — Networks that maintain ongoing dynamics, not just process discrete samples.
- [ ] **Recurrent connections** — Enable temporal pattern recognition and working memory.
- [ ] **Structural plasticity** — Dynamic connection growth/pruning.
- [ ] **Attractor states** — Memory emerging from stable activation patterns.
- [ ] **Spontaneous firing** — Pacemaker neurons that fire without input.
- [ ] **Neuromodulation** — Global signals (dopamine-like) that modulate entire regions.

## 🐛 C. elegans

- [ ] Per-neuron thresholds (sensory vs hub neurons)
- [ ] Neuromuscular connections (565 connections from NeuronsToMuscle sheet)
- [ ] Compare to OpenWorm simulation

## 📊 Tooling

- [ ] **Visualization** — Watch network activity over time
- [ ] **Metrics export** — Fire rates, activation distributions, cascade depth
- [ ] **Network statistics** — Degree distribution, clustering coefficient

---

## ✅ Done

- [x] Core neuron + network engine (integer LIF, event-driven, lazy decay)
- [x] LearningRule interface with 6 implementations (surrogate, arbiter, STDP, R-STDP, predictive, perturbation)
- [x] Surrogate gradient BPTT trainer with Adam optimizer
- [x] Dual-domain training (float64 train → int32 deploy)
- [x] Int32/float64 inference parity verified (Iris: 0/30 mismatches)
- [x] Iris benchmark: **100%** (surrogate gradient, 300 epochs)
- [x] MNIST benchmark: **97.21%** (512 hidden, Adam + LR decay, 42 epochs)
- [x] MNIST deep benchmark: 93.7% (784→256→128→10, useful negative result)
- [x] Activity instrumentation: 66.3% neurons idle per timestep
- [x] Energy analysis: **11× more efficient** than dense FP32 MLP (with lazy decay)
- [x] Arbiter neurons: 96.7-100% on Iris (biological approach)
- [x] Three-phase training: 10 variants explored, peaked at 86.7%
- [x] Weight perturbation: first rule to solve XOR
- [x] C. elegans connectome demo (299 neurons, 3363 connections)
- [x] Serialization
- [x] Apache 2.0 license
- [x] Paper draft with computational cost analysis
