# Sparse Coding & Neuron Specialization in Biological Networks

*Research notes from Dross Hour, 2026-02-23*

## The Problem We're Stuck On

With many hidden neurons, all outputs receive identical total spike counts regardless of input.
Hidden neurons don't specialize — they all fire for everything, or none fire at all.
The arbiter correction signal is noise relative to the sum of hundreds of inputs.

## Key Papers

### 1. SAILnet — Synaptically Local Sparse Coding (Zylberberg et al., 2011)
**"A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons"**
- PLoS Computational Biology, PMC3203062

**Architecture:** Spiking LIF neurons, lateral inhibitory connections, local plasticity only.

**Key insight: THREE learning rules working together:**
1. **Feedforward STDP** — strengthens connections from inputs that predict the neuron's firing
2. **Lateral inhibition plasticity** — inhibitory connections between hidden neurons GROW when two neurons fire together (anti-Hebbian). This forces decorrelation.
3. **Homeostatic threshold adjustment** — each neuron has a target firing rate. If it fires too much, threshold goes UP. Too little, threshold goes DOWN.

**Why it works:** The combination of lateral inhibition learning + homeostasis forces neurons apart. If two neurons respond to similar inputs, their mutual inhibition grows until one of them stops firing for those inputs. Over time, each neuron carves out its own unique receptive field.

**Mathematical proof:** Sparseness + decorrelation together are sufficient for local plasticity to learn optimal representations. Neither alone is enough.

### 2. Biologically Grounded V1 Model (bioRxiv, Dec 2024)
**"Emergence of Sparse Coding, Balance and Decorrelation"**

**Architecture:**
- 400 excitatory + 100 inhibitory LIF neurons (Dale's Law: separate populations)
- 512 LGN input neurons (256 ON + 256 OFF — like our population coding!)
- All-to-all lateral connectivity
- All synapses plastic via STDP
- Homeostatic rule adjusts weights AND thresholds per neuron

**Key insight: Excitation/Inhibition Balance**
- Increased excitatory input is automatically balanced by increased inhibition
- This balance causes decorrelated firing → population sparseness
- Population sparseness is both the CAUSE and RESULT of diverse receptive fields

**Emergent properties (not explicitly coded):**
- Temporal sparseness (neurons fire for few stimuli)
- Population sparseness (few neurons fire per stimulus)
- Gabor-like receptive fields
- Excitation-inhibition balance at multiple timescales

## What This Means for Our MNIST Problem

### Why our hidden neurons don't specialize:
1. **No lateral inhibition between hidden neurons** — we removed it because it caused winner lock-in with the OUTPUT layer. But the HIDDEN layer needs it.
2. **No homeostatic firing rate targets** — neurons that fire too much should become harder to excite. Our adaptive thresholds were a step in this direction but too simple.
3. **No anti-Hebbian lateral learning** — when two hidden neurons co-fire, nothing pushes them apart. They remain redundant copies.

### The biological recipe for specialization:
1. **Sparse random initial connectivity** ✓ (we had this)
2. **Feedforward Hebbian/STDP learning** ✓ (our arbiter strengthening)
3. **Lateral inhibition BETWEEN HIDDEN NEURONS that grows with co-firing** ✗ (we never had this)
4. **Homeostatic threshold/weight adjustment with target firing rates** ~ (our adaptive thresholds were a start)
5. **Separate excitatory and inhibitory populations (Dale's Law)** ✗ (we violate this — neurons can have both positive and negative output weights)

### The key missing piece: COMPETITIVE LATERAL INHIBITION IN HIDDEN LAYER

Our arbiter system handles error correction (feedback from output mistakes).
But we have NO mechanism for hidden neurons to compete with each other.
Without competition, all hidden neurons learn the same thing.

### Proposed approach:
1. Add lateral inhibitory connections between hidden neurons
2. Make lateral inhibition LEARNABLE — strengthen when two hidden neurons co-fire (anti-Hebbian)
3. Add homeostatic target firing rate per hidden neuron
4. Keep the arbiter error signal for supervised feedback from output layer

This is essentially: **unsupervised feature learning in hidden layer** (sparse coding via competition) + **supervised error correction at output layer** (arbiter neurons).

The hidden layer self-organizes diverse features. The output layer learns to read those features for classification. Two separate learning dynamics working at different layers.

## Connection to Our Arbiter Architecture

The arbiter system is about **error-driven learning** — knowing the answer was wrong and correcting.
Sparse coding is about **self-organized competition** — neurons fighting for the right to represent each input.

These aren't competing approaches. Biology uses BOTH:
- Visual cortex uses sparse coding + lateral inhibition for unsupervised feature extraction
- Higher cortical areas use error signals (reward prediction error, etc.) for goal-directed learning

Our architecture should do the same:
- Hidden layer: sparse coding dynamics (lateral inhibition + homeostasis)
- Output layer: arbiter-driven error correction

## References
- Zylberberg et al. (2011). "A Sparse Coding Model with Synaptically Local Plasticity and Spiking Neurons." PLoS Comp Bio.
- bioRxiv 2024.12.05.627100. "Emergence of Sparse Coding, Balance and Decorrelation from a Biologically-Grounded Spiking Neural Network Model."
- Olshausen & Field (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." Nature.
- Földiák (1990). "Forming sparse representations by local anti-Hebbian learning." Biological Cybernetics.
