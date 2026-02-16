# Neuromorphic Computing Landscape — Research Notes

*Dross's deep dive, 2026-02-16. Fulfilling a DROSS-TODO item.*

## Purpose

Understand the existing neuromorphic computing ecosystem — what's been built, what works, what doesn't — so I can contribute meaningfully to our biomimetic architecture design rather than just taking notes.

---

## Intel Loihi 2

### Architecture
- **128 neuromorphic cores** per chip, each simulating spiking neurons
- **1 million neurons, 120 million synapses** per chip
- 6 embedded x86 microprocessor cores for management/I/O
- Fabricated on **Intel 4 process** (first Intel product on this node)
- Asynchronous, event-driven — cores communicate via spike messages only
- **10x faster spike processing** than Loihi 1

### Key Innovations
- **Programmable neuron models** — not limited to leaky integrate-and-fire. Custom models via microcode assembly. This is huge: it means you could theoretically implement *our* neuron model on Loihi 2 hardware.
- **Graded spikes** — up to 32-bit payloads, not just binary 1/0. Biology-inspired but computationally richer. Our int16 weights could map directly to this.
- **Three-factor learning rules** — supports STDP and reward-modulated variants natively in hardware. This is exactly the learning rule space we identified as most promising.
- **Lava framework** — open-source Python SDK for writing neuromorphic apps. Deploy to CPU or Loihi 2.

### Scale: Hala Point System (2024-2025)
- Intel's largest neuromorphic system
- Multiple Loihi 2 chips networked together
- Targeting "sustainable AI" — efficiency claims vs. GPU inference

### Relevance to Our Project
**High.** Loihi 2 is basically the hardware our architecture was designed for — even though we designed software-first. Key alignment points:
- Event-driven, asynchronous (✅ matches our model)
- Programmable neuron models (✅ we could implement our lazy-decay LIF)
- Graded spikes (✅ our int16 weights could work as spike payloads)
- On-chip STDP (✅ our most promising learning rule candidate)

**Gaps:** Loihi 2 is research-access only (Intel Neuromorphic Research Community). Not commodity hardware. Our software-first approach on commodity CPUs remains the right call for now, but Loihi 2 is a compelling deployment target.

---

## IBM TrueNorth

### Architecture
- **4,096 neurosynaptic cores** in a 2D grid
- **1 million neurons, 256 million synapses**
- 5.4 billion transistors, 28nm CMOS
- **65 milliwatts** power consumption — this is remarkable
- Globally Asynchronous, Locally Synchronous (GALS) — same paradigm as Loihi but with a 1ms global time step

### Design Principles (Seven Pillars)
1. **Purely event-driven** — asynchronous interconnect, cores only active when spikes arrive
2. **Low power** — low-power CMOS process, minimize static power
3. **Massive parallelism** — 4,096 cores working independently
4. **Real-time operation** — 1ms global synchronization tick
5. **Scalable** — no global clock to distribute
6. **Error tolerant** — redundancy in memory circuits
7. **1:1 software-hardware correspondence** — simulation matches silicon exactly

### Key Design Insight: In-Memory Computing
TrueNorth is NOT a Von Neumann machine. Each core co-locates 256 neurons *with* their synaptic weights in local memory. No data bus bottleneck — processing happens where the data lives.

**This maps directly to our architecture.** Our neurons are structs that carry their own connection data. The entire point is that computation is local to the neuron, not centralized.

### Power Density Comparison
- Modern digital processor: ~100W/cm²
- Human brain: ~10mW/cm²
- TrueNorth: ~20mW/cm² (getting closer to brain territory)

### Relevance to Our Project
**Medium-High.** TrueNorth validates several of our core assumptions:
- Event-driven computation works in silicon ✅
- Co-located memory + compute = massive power savings ✅
- 1M neurons at 65mW proves the energy argument ✅

**Limitations for us:** TrueNorth's neuron model is less flexible than Loihi 2 — fixed leaky integrate-and-fire, no custom models. Also, the 1ms global tick is a compromise — our model is truly asynchronous (no global clock at all).

---

## Numenta / Hierarchical Temporal Memory (HTM)

### The Thousand Brains Theory
Jeff Hawkins' framework proposes that:
- Every **cortical column** learns complete models of objects (not just the top of a hierarchy)
- Thousands of models distributed across the neocortex work together via long-range connections
- Intelligence emerges from **compositional, location-based, temporal** representations

### HTM Algorithm
Three core properties:
1. **Sequence learning** — temporal patterns, "what comes next?"
2. **Continual learning** — no catastrophic forgetting, learns incrementally
3. **Sparse distributed representations (SDRs)** — only ~2% of bits active at any time

### How HTM Compares to Our Architecture

| Feature | HTM | Our Model |
|---------|-----|-----------|
| Neuron model | Complex (multiple dendrite segments, predictive states) | Simpler (activation + threshold + decay) |
| Learning | Hebbian (local, unsupervised) | TBD — STDP is our leading candidate (also local) |
| Sparsity | ~2% active (explicit design goal) | Emergent from threshold dynamics |
| Topology | Columnar (cortical column structure) | Graph-based (modular but not columnar) |
| Weights | Binary (0/1 connections) | Signed int16 (graded weights) |
| Temporal | Yes — sequence memory is core | Yes — lazy decay gives temporal dynamics |
| Backprop | No | No (probably) |

### Key Insight from HTM: Sparse Distributed Representations
SDRs are *extremely* powerful for certain operations:
- **Union/intersection** — bitwise OR/AND on sparse binary vectors
- **Noise tolerance** — if only 2% of bits are active, corrupting a few doesn't destroy the representation
- **Massive capacity** — combinatorial explosion of possible 2%-active patterns in a large vector

This is relevant to our **output problem** (Open Question #2 in DESIGN.md). Population coding from our output neurons could be interpreted as a kind of SDR — the pattern of which neurons are active *is* the representation.

### HTM Status (2026)
- HTM is now **legacy research** — Numenta has moved on from maintaining it
- The Thousand Brains Theory continues as Numenta's core framework
- Open-source implementations exist but community has shrunk
- Core ideas (sparsity, temporal, local learning) remain influential

### Relevance to Our Project
**Medium.** We share HTM's philosophical DNA — local learning, sparsity, temporal dynamics — but our architectural approach is different. HTM's cortical column structure is more biologically constrained than we want to be. However, their work on SDRs and the mathematical properties of sparse representations is directly useful.

**Key takeaway:** HTM proved that local Hebbian learning + sparse representations can do meaningful computation without backprop. That's encouraging for our STDP-based learning hypothesis.

---

## STDP: The Learning Rule Deep Dive

Since DESIGN.md identifies this as "THE hard problem," I dug deeper here.

### What STDP Is
Spike-Timing-Dependent Plasticity: the weight change between two connected neurons depends on the **relative timing** of their firing.

- **Pre fires before post** (causal) → strengthen connection (LTP)
- **Post fires before pre** (anti-causal) → weaken connection (LTD)
- **Time window:** typically ~20-40ms in biology

The magnitude of change decays exponentially with time difference:
```
If Δt > 0 (pre before post):  ΔW = A+ × exp(-Δt / τ+)
If Δt < 0 (post before pre):  ΔW = -A- × exp(Δt / τ-)
```

### Why STDP Fits Our Architecture Perfectly
1. **Local** — only needs information available at the synapse (pre/post spike times). No global error signal needed.
2. **Temporal** — inherently uses timing, which our architecture tracks via counters (last_interaction, refractory_until).
3. **Causal** — strengthens connections that predict future activity. This is literally what we want for sequence learning.
4. **Hardware-friendly** — Loihi 2 implements it natively. Simple enough for integer arithmetic.

### Variants Worth Considering
- **Reward-modulated STDP (R-STDP):** STDP sets up "eligibility traces" (candidate weight changes), but they only consolidate when a global reward/punishment signal arrives. This bridges the gap between local learning and goal-directed behavior. **Most promising hybrid approach.**
- **Triplet STDP:** Considers triplets of spikes rather than pairs. More biologically accurate, captures frequency effects.
- **Voltage-dependent STDP:** Weight change also depends on membrane potential. Richer dynamics but more complex.

### The Credit Assignment Problem
The elephant in the room. Backprop solves credit assignment by propagating error gradients backward through layers. STDP has no such mechanism — it only knows about local spike timing.

**Possible solutions:**
1. **R-STDP** — global reward signal provides direction, STDP provides local updates
2. **Eligibility traces + neuromodulation** — dopamine-like signals modulate which recent STDP changes persist
3. **Evolutionary optimization** — use genetic algorithms to find good initial topologies, then STDP for fine-tuning
4. **Hierarchical credit assignment** — each module (per our modularity insight) gets its own error signal from downstream modules

### Recent Research (2023-2025)
- **Nature Comms 2023:** "Sequence anticipation and STDP emerge from a predictive learning rule" — showed that a general predictive learning rule in neurons naturally produces STDP-like behavior. This suggests STDP might be a *consequence* of a more fundamental prediction-based rule, not the root mechanism.
- **Continual learning on Loihi 2 (2025):** CLP-SNN demonstrated online continual learning without catastrophic forgetting using spiking networks on Loihi 2. Power-constrained settings. This is directly relevant — our architecture should also support continual learning.
- **Modulated STDP review (2024):** ScienceDirect review of reward-modulated STDP variants. Confirms R-STDP as the most practical bridge between local plasticity and task-directed learning.

---

## Synthesis: What This Means for Our Design

### Validated Assumptions
1. **Event-driven sparse computation works in silicon.** TrueNorth proves it at 65mW. Loihi 2 proves it with programmable models.
2. **Local learning rules (STDP) can produce meaningful computation.** HTM and Loihi 2 research both demonstrate this.
3. **Our neuron model is in the right design space.** Simplified LIF with lazy decay is computationally simpler than what Loihi 2 supports — we're not asking for more than hardware can deliver.
4. **Modularity is the right topology.** Every successful system (brain, HTM columns, TrueNorth cores) uses modular, clustered architecture.

### New Ideas for DESIGN.md
1. **Reward-modulated STDP should be our primary learning rule candidate.** Pure STDP for local structure + global reward signal for task direction. This is the consensus "best bet" from the literature.
2. **Consider SDR-style output.** Instead of a conventional readout layer, interpret output neuron activity as a sparse distributed representation. Mathematical properties of SDRs (noise tolerance, capacity, easy union/intersection) could be powerful.
3. **Eligibility traces are essential.** STDP alone is too local — we need some mechanism to hold candidate weight changes until a reward signal confirms them. This is biologically real (dopamine modulation) and computationally necessary.
4. **Our model could literally run on Loihi 2.** If we ever want to prove the energy argument with real hardware numbers, Loihi 2 is the target. The Lava framework would let us prototype.

### Open Questions Refined
- **Learning rule:** R-STDP with eligibility traces is the starting point, not pure STDP. The Nature 2023 paper suggesting STDP emerges from a predictive rule is intriguing — could we implement the *predictive* rule instead?
- **Output:** SDR interpretation is worth exploring alongside population/rate coding
- **Topology:** Should we implement cortical-column-like structure within modules? HTM's columnar model is complex but well-studied.
- **Benchmarking:** The continual learning / catastrophic forgetting angle might be our strongest selling point vs. traditional NNs. STDP-based networks handle this naturally.

---

## References

- Intel Loihi 2: [Open Neuromorphic overview](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/)
- Intel Hala Point: [Intel Newsroom, Jan 2025](https://newsroom.intel.com/artificial-intelligence/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai)
- TrueNorth deep dive: [Open Neuromorphic](https://open-neuromorphic.org/blog/truenorth-deep-dive-ibm-neuromorphic-chip-design/)
- TrueNorth paper: Akopyan et al., "TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip" (IEEE TCADIS, 2015)
- HTM guide: [Numenta blog](https://www.numenta.com/blog/2019/10/24/machine-learning-guide-to-htm/)
- STDP predictive rule: "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule" (Nature Communications, 2023)
- Continual learning on Loihi 2: "Real-time Continual Learning on Intel Loihi 2" (arXiv, Nov 2025)
- R-STDP review: "Modulated spike-time dependent plasticity based learning for spiking neural networks" (ScienceDirect, Dec 2024)
- Kanerva SDM: *Sparse Distributed Memory* (MIT Press, 1988)
