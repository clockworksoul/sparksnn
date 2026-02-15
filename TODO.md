# TODO — Biomimetic Neural Network

*Tracked next steps, open questions, and ideas. Updated 2026-02-15.*

---

## 🔬 C. elegans Model

- [ ] **Per-neuron thresholds:** The single biggest improvement available. Sensory relay neurons (PVC) should be more excitable than hub neurons (AVAL). Would fix forward escape directional bias and reduce overall activation spread. Biology has diverse neuron types — this is the next step.
- [ ] **Forward escape directional bias:** Posterior touch currently produces A-class ≥ B-class instead of B > A. Root cause: PLMR connects directly to AVAL (4 synapses). Real worms achieve forward bias via neuromodulation (NMR-1 receptor expression), reciprocal AVA↔AVB inhibition, and state-dependent gating. Per-neuron thresholds would help; full fix needs neuromodulation.
- [ ] **Add neuromuscular connections:** `NeuronsToMuscle` sheet has 565 connections. Motor neurons → muscle cells would let us observe simulated movement output and complete the circuit.
- [ ] **Add sensory neuron metadata:** `Sensory` sheet has neuron types, neurotransmitter, and function info. Use it to set per-neuron properties (ties into per-neuron thresholds above).
- [ ] **Compare to OpenWorm's simulation:** They model C. elegans with different methods. How do our activation patterns and circuit behaviors compare?

## 🧠 Architecture & Core

- [ ] **Learning rule (THE hard problem):** Start with STDP (spike-timing-dependent plasticity):
  - Track pre/post-synaptic fire times per connection
  - Weight update based on temporal correlation
  - Then reward-modulated STDP for supervised direction
- [ ] **Relative refractory period:** Currently absolute only. A relative window (can fire with stronger input) would add biological realism and richer temporal dynamics.
- [ ] **Connection delay:** Axon propagation delay proportional to length. A `delay` field on Connection could produce more realistic temporal patterns.
- [ ] **Spontaneous firing:** Pacemaker neurons fire without input. Mechanism for spontaneous baseline activity.
- [ ] **Neuromodulation:** Dopamine, serotonin, etc. modulate entire regions, not individual synapses. Key missing piece for directional escape bias and behavioral state.

## 📊 Observability & Tooling

- [ ] **Visualization:** Watch network activity over time. Even terminal-based:
  - Per-tick firing count histogram
  - Signal propagation tracer (stimulate X, show cascade)
  - Most-active-neurons leaderboard
- [ ] **Metrics export:** Fire rate per neuron, activation distribution, cascade depth from stimulus
- [ ] **Network graph statistics:** Degree distribution, clustering coefficient, path length — compare C. elegans model to known small-world properties

## 🏗️ Infrastructure

- [ ] **Benchmarks:** Profile 10K, 100K, 1M neuron networks for memory and CPU
- [ ] **Parallelism:** Tick-level goroutine parallelism — neurons within a tick are independent. Partition by graph connectivity.
- [ ] **Synthetic test networks:** Small-world, scale-free, random topologies for scaling tests independent of biology

## 💡 Output / Interface Problem

- [ ] **Population coding PoC:** Output neuron group → interpret firing pattern as classification. Test: which sensory group was stimulated?
- [ ] **Rate coding:** Read values from firing frequency over a time window
- [ ] **Readout layer:** Thin conventional layer translating biomimetic activity to output — pragmatic hybrid

## 📚 Research & Reading

- [ ] **STDP implementations** in SNN frameworks (Brian2, NEST, Norse)
- [ ] **Leaky Integrate-and-Fire tuning** — decades of computational neuroscience parameter work
- [ ] **Numenta/HTM papers** — similar problems, different solutions
- [ ] **Intel Loihi papers** — neuromorphic hardware for this exact computation model

## ❓ Open Questions

- **Convergence:** How do we know processing is "done"? Fixed tick budget? Activity threshold?
- **Training signal:** STDP is unsupervised. How does error reach the right connections for supervised tasks?
- **Representational capacity:** Can 299 neurons learn anything beyond hardwired reflexes, or must we scale first?
- **Sleep/consolidation:** Offline pruning and restructuring — worth modeling?
- **Matrix equivalence:** Formal proof that this can represent same functions as feedforward nets?
- **Minimal interesting demo:** What shows this isn't a toy? Reflex validation ✅ done. Next: learning a classification? Anomaly detection?

## ✅ Done

- [x] Core neuron + network implementation (neuron.go, network.go)
- [x] Serialization (serialize.go)
- [x] C. elegans connectome loader from OpenWorm data (299 neurons, 3363 connections)
- [x] First stimulation PoC — touch receptor cascade through real connectome
- [x] DESIGN.md with full architectural rationale
- [x] **Inhibitory balance tuning** — GABAergic neuron map (DD, VD, RME classes), 5x inhibitory scaling, gap junction attenuation (40%), post-fire hyperpolarization (-200), aggressive decay (61% retention)
- [x] **Backward escape response** — anterior touch (ALM/AVM) → A-class > B-class motors, activity self-terminates in ~5 ticks, 48% activation. Biologically correct. ✅
- [x] **Forward escape circuit activation** — posterior touch (PLM) → AVAL → PVC → AVB → B-class motors fire in correct temporal sequence. 6-tick oscillation cycle emerges from refractory period (central pattern generator behavior). ✅
- [x] **CelegansParams struct** — clean API for tunable network construction with DefaultCelegansParams()
- [x] 27 tests all passing

---

*Add items freely. Check things off. Keep it honest.*
