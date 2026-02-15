# TODO — Biomimetic Neural Network

*Tracked next steps, open questions, and ideas. Updated 2026-02-15.*

---

## 🔬 C. elegans Model

- [ ] **Parameter tuning:** Current params (weightScale=200, threshold=300) produce runaway activation (277/299 neurons active). Real worms don't seize when you poke them. Investigate:
  - Stronger inhibitory scaling (GABA connections may need higher relative weight)
  - Separate thresholds per neuron class (sensory vs interneuron vs motor)
  - Post-fire reset below baseline (hyperpolarization)
- [ ] **Add neuromuscular connections:** We loaded `Connectome` sheet but not `NeuronsToMuscle` (565 connections). Motor neurons → muscle cells would let us observe simulated movement output.
- [ ] **Add sensory neuron metadata:** The `Sensory` sheet has neuron types and neurotransmitter info. Use it to set per-neuron properties.
- [ ] **Validate against known circuits:** The tap withdrawal reflex (touch → backward movement) is well-characterized. Can we stimulate PLM neurons and see the correct motor neurons activate?
- [ ] **Compare gap junctions vs chemical synapses:** Are gap junctions behaving differently enough? They should be faster/more reliable. Consider whether they need different weight scaling.
- [ ] **Inhibitory balance:** Count GABA vs excitatory connections. The 277/299 activation suggests insufficient inhibition — biological networks maintain ~20-30% inhibitory neurons.

## 🧠 Architecture & Core

- [ ] **Learning rule (THE hard problem):** Start with STDP (spike-timing-dependent plasticity). Needs:
  - Track pre-synaptic and post-synaptic fire times per connection
  - Implement weight update based on temporal correlation
  - Consider reward-modulated STDP as a second step
- [ ] **Post-fire reset behavior:** Currently unclear what happens to activation after firing. Should reset to baseline? Below baseline (hyperpolarization)? This matters a lot for oscillation dynamics.
- [ ] **Relative refractory period:** Currently we have absolute refractory only. Add a relative refractory window where the neuron *can* fire but needs stronger input.
- [ ] **Per-neuron parameters:** Right now baseline, threshold, and decay are global. Biology has diverse neuron types — some fire easily, some rarely. Support per-neuron overrides.
- [ ] **Connection delay:** Biological axons have propagation delay proportional to length. Adding a `delay` field to Connection could produce more realistic temporal dynamics.
- [ ] **Spontaneous firing:** Some biological neurons fire without input (pacemaker neurons). Consider a mechanism for spontaneous baseline activity.
- [ ] **Neuromodulation:** Dopamine, serotonin, etc. modulate entire regions, not individual synapses. How to model global modulatory signals?

## 📊 Observability & Tooling

- [ ] **Visualization:** Build a simple tool to watch network activity over time. Even a terminal-based heatmap would help. Consider:
  - Per-tick firing count histogram
  - Most-active-neurons leaderboard
  - Signal propagation tracer (stimulate X, show cascade)
- [ ] **Metrics export:** Fire rate per neuron, average activation, connection weight distribution, cascade depth from stimulus
- [ ] **Network statistics:** Degree distribution, clustering coefficient, path length — compare our C. elegans model to known small-world network properties
- [ ] **Snapshot diff:** Compare two network states to see what changed during learning (when we have learning)

## 🏗️ Infrastructure

- [ ] **Benchmarks:** How fast can we tick a 10K, 100K, 1M neuron network? Profile memory and CPU.
- [ ] **Parallelism:** Neuron updates within a tick are independent — parallelize with goroutines. Partition by graph connectivity for minimal cross-partition traffic.
- [ ] **Larger test networks:** Build synthetic networks with known topology (small-world, scale-free, random) to test scaling properties independent of biology.
- [ ] **Serialization round-trip test:** Verify save/load preserves exact state including activation levels and pending queue.

## 💡 Output / Interface Problem

- [ ] **Population coding PoC:** Designate a group of "output neurons" and interpret their firing pattern as a classification. Test on something trivial (e.g., which sensory neuron group was stimulated?).
- [ ] **Rate coding:** Can we read a value from a neuron's firing frequency over a time window?
- [ ] **Readout layer:** Pragmatic hybrid — thin conventional layer on top translating biomimetic activity to useful output. Worth prototyping even if inelegant.

## 📚 Research & Reading

- [ ] **Study STDP implementations** in SNN frameworks (Brian2, NEST, Norse) for practical learning rule ideas
- [ ] **Compare to OpenWorm's approach** — they also simulate C. elegans but with different methods. How do our results compare?
- [ ] **Read up on Leaky Integrate-and-Fire** tuning — standard computational neuroscience has decades of parameter-fitting work
- [ ] **Numenta's HTM papers** — they solved some similar problems differently
- [ ] **Intel Loihi papers** — neuromorphic hardware designed for exactly this kind of computation

## ❓ Open Questions

- **Convergence criterion:** How do we know when the network has "finished" processing an input? Biology never stops. Do we need a fixed tick budget? Activity-level threshold?
- **Training signal:** STDP is unsupervised. For supervised tasks, how does error information reach the right connections? Reward-modulated STDP? Something else?
- **Representational capacity:** Can a 300-neuron C. elegans-topology network learn *anything* useful beyond its hardwired reflexes? Or do we need to scale up first?
- **Sleep/consolidation:** The brain prunes and consolidates during sleep. Should we have an offline mode that restructures connections?
- **Matrix equivalence:** Is there a formal proof (or disproof) that this architecture can represent the same functions as a standard feedforward net of equivalent size?
- **What's the minimal interesting demo?** Something that shows this isn't just a toy — C. elegans reflex validation? Learning a simple classification? Anomaly detection in a stream?

## ✅ Done

- [x] Core neuron + network implementation (neuron.go, network.go)
- [x] Serialization (serialize.go)
- [x] C. elegans connectome loader from OpenWorm data
- [x] First stimulation PoC — touch receptor cascade through real connectome
- [x] DESIGN.md with full architectural rationale

---

*Add items freely. Check things off. Keep it honest.*
