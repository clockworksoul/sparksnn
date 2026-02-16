# MNIST Benchmark Design

*The "prove it learns" test for the biomimetic architecture.*
*Drafted 2026-02-16 by Matt & Dross*

---

## Goal

Demonstrate that our biomimetic architecture can learn useful representations through local plasticity rules — no backpropagation, no gradient descent, no global error signal (for the predictive rule).

**Target:** Classify MNIST handwritten digits (10 classes, 28×28 grayscale images).

**Success criteria:**
- ≥80% accuracy = "it works" (proves the architecture can learn)
- ≥90% accuracy = "competitive" (matches other SNN/STDP approaches)
- Any accuracy > random (10%) with the predictive rule = "STDP really does emerge"

**Context:** State-of-the-art SNNs with STDP achieve 90-95% on MNIST. Traditional NNs get 99%+. We're not trying to beat traditional NNs — we're proving our architecture can learn at all.

---

## Architecture

### Network Topology

```
Input Layer (784 neurons)
    ↓ fully connected
Hidden Layer (400 neurons, excitatory)
    ↔ lateral inhibition (400 inhibitory neurons, 1:1 paired)
    ↓ readout
Output (10 classes via population coding)
```

**Why 400 hidden neurons?** This is the standard size used in Diehl & Cook (2015), the seminal STDP-MNIST paper. It achieves ~95% accuracy. We start here for comparability and can scale up/down later.

### Layer Details

**Input layer (784 neurons):**
- One neuron per pixel (28×28 = 784)
- These are not learning neurons — they're input encoders
- Each connects to all 400 excitatory hidden neurons

**Excitatory hidden layer (400 neurons):**
- Receive input from all 784 input neurons (fully connected = 313,600 connections)
- Each paired 1:1 with an inhibitory neuron
- Learning happens on the input→excitatory connections
- These neurons develop digit-selective receptive fields through learning

**Inhibitory hidden layer (400 neurons):**
- One per excitatory neuron
- When excitatory neuron E_i fires, inhibitory neuron I_i fires
- I_i inhibits ALL other excitatory neurons (lateral inhibition / winner-take-all)
- Fixed weights — no learning on inhibitory connections
- Purpose: competition. Forces different neurons to specialize on different patterns

**Output (population coding):**
- After training, each excitatory neuron develops a preference for a digit class
- During evaluation, present an image, count spikes per excitatory neuron
- Assign each neuron to its most-responsive digit class
- Classify by which neuron class fires most

---

## Input Encoding: Rate Coding

Convert pixel intensity to spike probability per tick.

```
For each pixel (value 0-255):
  spike_probability = pixel_value / 255
  At each tick: spike if random() < spike_probability
```

**In our integer math:**
```go
// pixel is 0-255, stimWeight is the base weight for input connections
if rand.Intn(256) < pixelValue {
    net.Stimulate(inputNeuronIdx, stimWeight)
}
```

**Presentation time:** 350 ticks per image (standard in SNN literature).
**Rest period:** 150 ticks between images (let activity settle, reset).

**Why rate coding?** Simplest, well-understood, and what most STDP-MNIST papers use. Latency coding is more efficient but harder to implement correctly. We can optimize to latency coding later.

---

## Output Decoding: Population Assignment

After training, assign each excitatory neuron to a digit class:

1. Present all training images (or a subset)
2. For each image, record which excitatory neurons fired
3. For each neuron, tally: "how many times did I fire for each digit?"
4. Assign the neuron to the digit it fired most for

**During inference:**
1. Present test image for 350 ticks
2. Count spikes from each excitatory neuron
3. Sum spikes by assigned class
4. Predicted digit = class with most total spikes

This is standard for unsupervised SNN classification (Diehl & Cook 2015).

---

## Learning Rules

We benchmark both:

### R-STDP (learning/stdp)
- Spike timing creates eligibility traces
- **Question:** What provides the reward signal? For unsupervised MNIST, we can use a simplified approach:
  - No explicit reward — rely on lateral inhibition to create competition
  - Alternative: use the "winner fired" signal as implicit reward
- May need to extend R-STDP to support unsupervised mode (eligibility directly modifies weights without reward gating)

### Predictive (learning/predictive)
- Self-supervised — no reward needed
- Each neuron learns to predict its inputs
- Synapses that carry predictive information get strengthened
- **This is the exciting test:** Does the predictive rule develop digit-selective neurons purely from prediction error optimization?

### Comparison baseline
- **Random weights** (no learning) — establishes floor
- **Logistic regression** on raw pixels — establishes a simple ML baseline (~92%)

---

## Implementation Plan

### Package Structure

```
benchmark/
├── DESIGN.md          ← this file
├── mnist/
│   ├── mnist.go       ← MNIST data loader (IDX format)
│   ├── mnist_test.go
│   ├── encode.go      ← rate coding / spike generation
│   └── encode_test.go
├── classify/
│   ├── network.go     ← build the MNIST network topology
│   ├── train.go       ← training loop
│   ├── eval.go        ← evaluation / accuracy calculation
│   └── classify_test.go
└── cmd/
    └── mnist/
        └── main.go    ← CLI: train and evaluate
```

### Step 1: MNIST Data Loader
- Parse IDX file format (the raw MNIST binary format)
- Download from http://yann.lecun.com/exdb/mnist/ or use a Go package
- Return []Image{pixels []byte, label byte}

### Step 2: Input Encoding
- `RateEncode(image []byte, net *bio.Network, inputNeurons []uint32, numTicks int)`
- For each tick: for each pixel, probabilistically stimulate the input neuron

### Step 3: Network Builder
- `BuildMNISTNetwork(hiddenSize int, rule bio.LearningRule) *bio.Network`
- Creates the full topology: 784 input + N excitatory + N inhibitory
- Sets up connections with random initial weights (small, positive for input→excitatory)
- Sets up fixed inhibitory connections

### Step 4: Training Loop
- For each training image:
  1. Rate-encode and present for 350 ticks
  2. Record which excitatory neurons fired
  3. Rest for 150 ticks (just tick with no input)
  4. Learning rule updates weights continuously during presentation

### Step 5: Evaluation
- After training, assign neurons to classes (population assignment)
- Present test images, classify by population vote
- Report accuracy

### Step 6: CLI
- `go run ./benchmark/cmd/mnist --rule=predictive --hidden=400 --epochs=1`
- Prints training progress and final accuracy

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Input neurons | 784 | One per pixel |
| Hidden excitatory | 400 | Standard, tunable |
| Hidden inhibitory | 400 | 1:1 with excitatory |
| Ticks per image | 350 | Presentation time |
| Rest ticks | 150 | Between images |
| Input weight range | [0, 500] | Random initial, tunable |
| Inhibitory weight | -1000 | Strong, fixed |
| Excit→Inhib weight | 500 | Fixed, ensures inhibitory fires when excitatory fires |
| Baseline | 0 | Standard |
| Threshold | 500 | Tunable — key parameter |
| Decay rate | 58982 | ~90% retention |
| Refractory period | 5 | Prevents single-neuron domination |

---

## Challenges & Risks

1. **Scale:** 784 × 400 = 313,600 input connections + inhibitory connections. This is much larger than our C. elegans model (4,447 connections). Performance may be an issue.

2. **Parameter tuning:** SNN performance is notoriously sensitive to threshold, weight initialization, learning rate, and presentation time. Expect iteration.

3. **STDP without reward:** Pure STDP (without reward modulation) learns features but needs lateral inhibition to avoid all neurons converging to the same pattern. Our R-STDP is designed for reward — we may need to add a pure STDP mode.

4. **Integer precision:** The predictive rule does delicate prediction-error math in int16. This is uncharted territory — no one has run predictive plasticity in integer arithmetic before.

5. **Training time:** SNN training is slower than traditional NN training. 60,000 images × 350 ticks = 21 million ticks. With 313K connections... this might take a while. May need to start with a subset.

---

## Success Milestones

| Milestone | What it proves |
|---|---|
| Network builds and runs without crashing | Infrastructure works at scale |
| Neurons develop different receptive fields | Learning rules create specialization |
| >10% accuracy (random baseline) | Network learned *something* |
| >50% accuracy | Respectable unsupervised learning |
| >80% accuracy | Architecture genuinely works |
| >90% accuracy | Competitive with published SNN results |
| Predictive ≥ STDP accuracy | Our "secret sauce" is real |

---

## References

- Diehl, P.U. & Cook, M. "Unsupervised learning of digit recognition using spike-timing-dependent plasticity." Frontiers in Computational Neuroscience 9, 99 (2015). — The standard STDP-MNIST benchmark.
- snntorch Tutorial 1 — Spike encoding methods: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html
- Saponati & Vinck (2023) — Our predictive learning rule source paper.

---

*This is the most important thing we build next. If this works, we have a real project. If it doesn't, we have a research question about why.*
