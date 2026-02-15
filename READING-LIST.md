# Biomimetic Architecture — Reading List

*Organized by topic. Mix of textbooks, review articles, and primary literature.*
*📖 = Book | 📄 = Paper/Review | 🔗 = Online resource*

---

## 🧠 Foundations: Computational Neuroscience

These give you the vocabulary and models — the bridge between the biology you know and the computation we're building.

- 📖 **Gerstner et al. — *Spiking Neuron Models: Single Neurons, Populations, Plasticity*** (Cambridge, 2002)
  The standard textbook. Covers leaky integrate-and-fire, Hodgkin-Huxley, STDP, network dynamics. Written for physicists/mathematicians but very accessible with a biology background. Free extracts at [lcnwww.epfl.ch](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/SpikingNeuronM-extracts.pdf).

- 📖 **Gerstner et al. — *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*** (Cambridge, 2014)
  Updated, broader scope. Full text free online at [neuronaldynamics.epfl.ch](https://neuronaldynamics.epfl.ch/). **Start here** — it's the most modern and accessible entry point.

- 📖 **Dayan & Abbott — *Theoretical Neuroscience*** (MIT Press, 2001)
  The other classic. More math-heavy but comprehensive. Good reference for neural coding (rate codes, population codes, temporal codes) — directly relevant to our output problem.

## ⚡ Spiking Neural Networks

The existing paradigm closest to what we're building.

- 📄 **Yamazaki et al. — "Spiking Neural Networks and Their Applications: A Review"** (Brain Sciences, 2022)
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/) — Comprehensive review covering neuron models, synapse models, training methods, and applications. Good overview of the field.

- 📄 **Pfeiffer & Pfeil — "Deep Learning With Spiking Neurons: Opportunities and Challenges"** (Frontiers in Neuroscience, 2018)
  How deep learning concepts map (or don't) onto spiking networks. Directly relevant to the representational equivalence question.

- 📄 **Tavanaei et al. — "Deep Learning in Spiking Neural Networks"** (Neural Networks, 2019)
  Another good survey bridging conventional deep learning and SNNs.

## 🔄 Learning Rules: STDP and Beyond

The hard problem. How does a biomimetic network learn?

- 📄 **Dan & Poo — "Spike Timing-Dependent Plasticity of Neural Circuits"** (Neuron, 2004)
  Foundational review of STDP — how the relative timing of pre- and post-synaptic spikes determines weight changes. The biological basis for our most likely learning rule.

- 📄 **Markram et al. — "A History of Spike-Timing-Dependent Plasticity"** (Frontiers in Synaptic Neuroscience, 2011)
  Historical perspective plus experimental evidence. Markram's lab discovered STDP.

- 📄 **Bengio et al. — "Towards Biologically Plausible Deep Learning"** (arXiv:1502.04156)
  Can we get backprop-like learning without backprop? Explores biologically plausible alternatives.

- 📄 **Lillicrap et al. — "Backpropagation and the Brain"** (Nature Reviews Neuroscience, 2020)
  How the brain might approximate error-driven learning. Reviews feedback alignment, target propagation, and other alternatives to backprop.

## 🖥️ Neuromorphic Computing

Hardware that's built for architectures like ours.

- 📄 **Exploring Neuromorphic Computing Based on Spiking Neural Networks: Algorithms to Hardware** (ACM Computing Surveys, 2023)
  [ACM](https://dl.acm.org/doi/full/10.1145/3571155) — Comprehensive survey from algorithms to chips.

- 📄 **Davies et al. — "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning"** (IEEE Micro, 2018)
  Intel's neuromorphic chip. Implements on-chip STDP. Closest existing hardware to what we'd need.

- 🔗 **"Neuromorphic Computing 2025: Current State of the Art"**
  [humanunsupervised.com](https://humanunsupervised.com/papers/neuromorphic_landscape.html) — Recent landscape review covering Loihi, TrueNorth, SpiNNaker, memristors, and more.

## 🧩 Alternative Architectures & Theories

People thinking along similar lines from different angles.

- 📖 **Jeff Hawkins — *On Intelligence*** (2004) + ***A Thousand Brains*** (2021)
  Hawkins has been arguing for biologically-inspired computing for 20 years. HTM (Hierarchical Temporal Memory) shares DNA with our approach — sparse distributed representations, temporal sequences, cortical columns. *A Thousand Brains* proposes every cortical column builds a complete model of its input. Very relevant to our modularity discussion.

- 📖 **Numenta research papers** — [numenta.com/resources/research-publications](https://www.numenta.com/resources/research-publications/)
  Their work on sparsity, SDRs, and cortical columns is directly applicable.

- 📄 **Maass — "Networks of Spiking Neurons: The Third Generation of Neural Network Models"** (Neural Networks, 1997)
  The paper that defined SNNs as the "third generation." Shows they're theoretically more powerful than sigmoid networks.

- 📄 **Chen et al. — "Neural Ordinary Differential Equations"** (NeurIPS, 2018)
  Different approach to continuous-time computation. Worth knowing about as a contrast.

## 🌳 Evolution, Modularity & the Messy Brain

The philosophical/biological grounding for why messy works.

- 📖 **Steven Mithen — *The Prehistory of the Mind*** (1996)
  Already discussed. Cognitive modularity and the emergence of cross-module "cognitive fluidity."

- 📖 **Gary Marcus — *Kluge: The Haphazard Evolution of the Human Mind*** (2008)
  Argues the mind is a "kluge" — a clumsy, inelegant solution that works anyway. Complementary to Mithen.

- 📖 **Sebastian Seung — *Connectome: How the Brain's Wiring Makes Us Who We Are*** (2012)
  The connectome project — mapping every neural connection. Relevant to topology and connection patterns.

## 📊 Neural Coding & the Output Problem

Specifically for solving "how do we get information out?"

- 📄 **Georgopoulos et al. — "Neuronal Population Coding of Movement Direction"** (Science, 1986)
  The classic population coding paper. Showed that motor cortex neurons collectively encode direction even though individual neurons are noisy. This might be our output mechanism.

- 📄 **Rieke et al. — *Spikes: Exploring the Neural Code*** (MIT Press, 1999)
  Book-length treatment of how information is encoded in spike trains. Rate coding vs. temporal coding debate.

---

## Suggested Reading Order

1. **Neuronal Dynamics** (free online textbook) — get the foundations
2. **Yamazaki et al. SNN review** — landscape of existing work
3. **Hawkins — *A Thousand Brains*** — the philosophical alignment
4. **Dan & Poo on STDP** — the learning rule we'll probably start with
5. **Georgopoulos on population coding** — the output problem
6. **Loihi paper** — what hardware exists for this
7. **Mithen — *Prehistory of the Mind*** — you already know this one, but re-read with the architecture in mind
8. Then dive into whatever thread pulls you hardest

---

*Living document. We'll add to this as we go deeper.*
