# TODO — Biomimetic Neural Network

*Tracked next steps, open questions, and ideas. Updated 2026-02-16.*

---

## 🔴 Prove It Learns — THE Priority

- [ ] **Simple classification benchmark (MNIST)** — This is the "not a toy" proof. We have two learning rules but zero evidence they can learn useful representations. Until we can classify *something*, we're speculating.
  - [ ] Design input encoding (pixel values → spike trains)
  - [ ] Design network topology (input layer → hidden → output population)
  - [ ] Design output decoding (population coding from 10 output groups)
  - [ ] Implement training loop with both learning rules
  - [ ] Benchmark R-STDP vs Predictive head-to-head
  - [ ] Compare against a simple baseline (e.g., logistic regression)
- [ ] **Continual learning / catastrophic forgetting** — potentially our strongest differentiator. Train on task A, then task B, measure retention of A. Traditional NNs struggle here; local learning rules handle it naturally.

## 🔬 C. elegans Model

- [ ] **Per-neuron thresholds:** The single biggest improvement available. Sensory relay neurons (PVC) should be more excitable than hub neurons (AVAL). Would fix forward escape directional bias and reduce overall activation spread. Biology has diverse neuron types — this is the next step.
- [ ] **Forward escape directional bias:** Posterior touch currently produces A-class ≥ B-class instead of B > A. Root cause: PLMR connects directly to AVAL (4 synapses). Real worms achieve forward bias via neuromodulation. Per-neuron thresholds would help; full fix needs neuromodulation.
- [ ] **Add neuromuscular connections:** `NeuronsToMuscle` sheet has 565 connections. Motor neurons → muscle cells would complete the circuit.
- [ ] **Add sensory neuron metadata:** `Sensory` sheet has neuron types, neurotransmitter, and function info.
- [ ] **Compare to OpenWorm's simulation**

## 🧠 Architecture & Core

- [ ] **Relative refractory period:** Currently absolute only. A relative window (can fire with stronger input) would add richer temporal dynamics.
- [ ] **Connection delay:** Axon propagation delay proportional to length.
- [ ] **Spontaneous firing:** Pacemaker neurons fire without input.
- [ ] **Neuromodulation:** Dopamine, serotonin, etc. modulate entire regions, not individual synapses.

## 📊 Observability & Tooling

- [ ] **Visualization:** Watch network activity over time (even terminal-based)
- [ ] **Metrics export:** Fire rate per neuron, activation distribution, cascade depth
- [ ] **Network graph statistics:** Degree distribution, clustering coefficient, path length

## 🏗️ Infrastructure

- [ ] **Benchmarks:** Profile 10K, 100K, 1M neuron networks for memory and CPU
- [ ] **Parallelism:** Tick-level goroutine parallelism — neurons within a tick are independent
- [ ] **Synthetic test networks:** Small-world, scale-free, random topologies for scaling tests

## 💡 Output / Interface Problem

- [ ] **Population coding PoC:** Output neuron group → interpret firing pattern as classification
- [ ] **Rate coding:** Read values from firing frequency over a time window
- [ ] **Readout layer:** Thin conventional layer translating biomimetic activity to output

## 📚 Research & Reading

- [ ] **STDP implementations** in SNN frameworks (Brian2, NEST, Norse)
- [ ] **Leaky Integrate-and-Fire tuning** — decades of computational neuroscience parameter work
- [ ] **Numenta/HTM papers** — similar problems, different solutions

## 📏 Benchmarking Strategy

- [ ] **Define benchmark suite** — what tasks, what metrics, what baselines?
- [ ] **Energy/compute efficiency** — ops-per-inference comparison vs. traditional forward pass
- [ ] **Temporal / streaming data** — anomaly detection, sequence prediction (native temporal dynamics)
- [ ] **Scaling curves** — inference cost vs. network size vs. input complexity

## 🔵 IP Protection

- [x] Email sent to Yext legal requesting Covenants Agreement (2026-02-14)
- [ ] Talk to Adam (VP Eng) about side project IP — **scheduled for 2026-02-17**
- [ ] Review invention assignment clause when received
- [ ] File provisional patent ($130 small entity) — see `patent/PROVISIONAL-PATENT-GUIDE.md`
- [ ] After filing: publish (arXiv, blog, make repo public)

## ❓ Open Questions

- **Convergence:** How do we know processing is "done"? Fixed tick budget? Activity threshold?
- **Representational capacity:** Can 299 neurons learn anything beyond hardwired reflexes, or must we scale first?
- **Sleep/consolidation:** Offline pruning and restructuring — worth modeling?
- **Matrix equivalence:** Formal proof that this can represent same functions as feedforward nets?
- **Integer precision:** Will int16 quantization limit the predictive rule's learning dynamics?

## ✅ Done

- [x] Core neuron + network implementation (neuron.go, network.go)
- [x] Serialization (serialize.go)
- [x] C. elegans connectome loader (299 neurons, 3363 connections)
- [x] First stimulation PoC — touch receptor cascade through real connectome
- [x] DESIGN.md with full architectural rationale
- [x] Inhibitory balance tuning (GABAergic neuron map, gap junction attenuation, hyperpolarization)
- [x] Backward escape response — anterior touch → A-class > B-class motors ✅
- [x] Forward escape circuit activation — posterior touch → B-class motors fire ✅
- [x] CelegansParams struct — clean API with DefaultCelegansParams()
- [x] **LearningRule interface** — swappable learning algorithms (2026-02-16)
- [x] **R-STDP implementation** — three-factor learning rule in `learning/stdp/` (2026-02-16)
- [x] **Predictive learning rule** — self-supervised, STDP emerges from prediction error. In `learning/predictive/` (2026-02-16)
- [x] **Repo restructure** — learning rules in subpackages under `learning/` (2026-02-16)
- [x] **Neuromorphic landscape research** — `research/neuromorphic-landscape.md` (2026-02-16)
- [x] **Predictive learning paper analysis** — `research/predictive-learning-rule.md` (2026-02-16)
- [x] 51 tests passing across 5 packages

---

*Add items freely. Check things off. Keep it honest.*
