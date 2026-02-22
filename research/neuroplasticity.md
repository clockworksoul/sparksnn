# Neuroplasticity Research for Biomimetic Neural Architecture

*Research compiled 2026-02-22 by Dross*
*Context: biomimetic-network project — see DESIGN.md for architecture, research/structural-plasticity.md for structural plasticity design, research/arbiter-neurons.md for error signaling hypothesis*

---

## Table of Contents

1. [Synaptic Plasticity Beyond STDP](#1-synaptic-plasticity-beyond-stdp)
2. [Structural Plasticity](#2-structural-plasticity)
3. [Cortical Remapping](#3-cortical-remapping)
4. [Critical Periods](#4-critical-periods)
5. [Biological Error Signaling — Analogues to Backpropagation](#5-biological-error-signaling--analogues-to-backpropagation)
6. [Regional Plasticity Profiles](#6-regional-plasticity-profiles)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Synaptic Plasticity Beyond STDP

Our architecture currently implements pure STDP, R-STDP, predictive learning (Saponati & Vinck), and weight perturbation. Biology has several additional plasticity mechanisms that could improve learning.

### 1.1 Long-Term Potentiation / Long-Term Depression (LTP/LTD)

**What it is:** Persistent strengthening (LTP) or weakening (LTD) of synapses based on stimulation patterns. Unlike STDP (which cares about spike timing), LTP/LTD are more about stimulation *intensity and frequency*.

**Key mechanisms:**
- **LTP induction:** High-frequency stimulation (tetanic) → large calcium influx through NMDA receptors → CaMKII activation → AMPA receptor insertion → synapse strengthened
- **LTD induction:** Low-frequency stimulation → modest calcium influx → phosphatase activation → AMPA receptor removal → synapse weakened
- **The calcium hypothesis:** The *level* of postsynaptic calcium determines direction — high calcium → LTP, moderate calcium → LTD, low calcium → no change

**Relevance to our architecture:** We already have something resembling this via STDP. But the calcium-level model suggests a simpler, more fundamental mechanism: **accumulation magnitude determines plasticity direction**.

**Implementation idea — Accumulation-Gated Plasticity:**
```
After a neuron fires:
  total_input = sum of all incoming stimulations that contributed
  
  if total_input > LTP_threshold:
    strengthen active incoming connections (large input = reliable pattern)
  elif total_input > LTD_threshold:
    weaken active incoming connections (moderate input = noisy/uncertain)
  else:
    no change (sub-threshold input = irrelevant)
```
This could complement STDP: STDP handles temporal patterns, accumulation-gating handles magnitude patterns. Could be added to the `LearningRule` interface as an alternative `OnPostFire` behavior.

### 1.2 Metaplasticity (The BCM Theory)

**What it is:** "Plasticity of plasticity" — the rules governing synaptic modification themselves change based on the neuron's recent history. The Bienenstock-Cooper-Munro (BCM) theory (1982) is the canonical model.

**The BCM sliding threshold:**
- Each neuron has a modification threshold (θ_M) that determines whether activity leads to LTP or LTD
- θ_M **slides** based on the neuron's recent average activity:
  - If the neuron has been very active → θ_M increases → harder to get LTP, easier to get LTD
  - If the neuron has been quiet → θ_M decreases → easier to get LTP, harder to get LTD
- This prevents runaway potentiation AND runaway depression — it's a homeostatic mechanism operating on the learning rule itself

**Why it matters:** Without metaplasticity, Hebbian-type learning is unstable. Neurons that fire a lot get stronger, fire more, get stronger... positive feedback loop. BCM's sliding threshold is an elegant solution: success raises the bar.

**Implementation idea — Sliding Plasticity Threshold:**
```go
type Neuron struct {
    // ... existing fields ...
    PlasticityThreshold int32  // BCM θ_M — slides with activity history
}

// During learning rule application:
// If neuron's recent firing rate > PlasticityThreshold:
//   This activation caused LTD (weaken recent inputs)
// If neuron's recent firing rate < PlasticityThreshold:
//   This activation caused LTP (strengthen recent inputs)
// 
// After each remodel period:
//   PlasticityThreshold = f(recent_average_firing_rate)
```

This is highly compatible with our existing architecture. The `Maintain()` hook on `LearningRule` could adjust `PlasticityThreshold` periodically. It would stabilize all our learning rules without requiring careful manual tuning of learning rates.

**Key reference:** Abraham & Bear (1996), "Metaplasticity: the plasticity of synaptic plasticity." *Trends in Neurosciences* 19(4), 126-130.

### 1.3 Homeostatic Synaptic Scaling

**What it is:** A global gain-control mechanism where neurons multiplicatively scale ALL their incoming synaptic weights up or down to maintain a target firing rate (Turrigiano et al., 1998; Turrigiano 2008).

**Crucially different from threshold adjustment:** Our structural plasticity already does threshold homeostasis (lowering threshold for dead neurons). Synaptic scaling is different — it multiplies all incoming weights by a common factor, preserving their *relative* strengths while adjusting overall drive.

**Why this matters:**
- **Preserves learned information.** Threshold adjustment changes excitability but doesn't care about weight ratios. Scaling preserves the pattern the neuron has learned while adjusting its overall responsiveness.
- **Prevents catastrophic forgetting.** When you scale weights rather than modifying them individually, the relationships between inputs (what the neuron has "learned") are maintained.
- **Operates on a slow timescale** (hours in biology) — complementary to fast Hebbian learning (seconds).

**The biology:** Turrigiano showed that blocking activity in cultured cortical networks causes a multiplicative upscaling of all excitatory synapses (mEPSC amplitudes increase proportionally). Increasing activity causes downscaling. The scaling factor is proportional to the neuron's deviation from its target firing rate.

**Implementation idea — Multiplicative Weight Scaling:**
```go
// In Remodel() or a new HomeostasisPass():
func ScaleWeights(neuron *Neuron, targetRate, actualRate float64) {
    if actualRate == 0 {
        return // handled by threshold homeostasis
    }
    scaleFactor := targetRate / actualRate  // > 1 if too quiet, < 1 if too active
    scaleFactor = clamp(scaleFactor, 0.9, 1.1)  // conservative per-pass
    
    for i := range neuron.IncomingConnections {
        neuron.IncomingConnections[i].Weight = int32(float64(weight) * scaleFactor)
    }
}
```

**Challenge for our architecture:** We currently track outgoing connections, not incoming. Scaling requires knowing all connections *into* a neuron. We have the `incomingIndex` cache for this, but it's invalidated after structural changes. Might want a more persistent incoming connection tracking mechanism.

**Priority: HIGH.** This is probably the single most impactful plasticity mechanism we're missing. It would stabilize learning across all our rules.

### 1.4 Synaptic Tagging and Capture (STC)

**What it is:** Explains how short-term synaptic changes become permanent memories. A two-step process:
1. **Tagging:** Synaptic activity sets a local "tag" at the synapse (lasts ~1-2 hours)
2. **Capture:** Plasticity-related proteins (PRPs), synthesized at the cell body, are "captured" by tagged synapses → converts temporary change to long-term

**The clever part:** A strongly stimulated synapse produces both a tag AND triggers PRP synthesis. A weakly stimulated synapse on the *same neuron* only sets a tag. But if the strong stimulus happens first, the PRPs it triggered get captured by the weak synapse's tag too — transforming a weak, transient change into a strong, permanent one.

**Implementation idea — Two-Phase Weight Consolidation:**
```
Connection {
    Weight: int32           // Permanent weight
    PendingDelta: int32     // Temporary weight change (the "tag")
    TagExpiry: uint32       // Tick when the tag expires
}

// Learning rules write to PendingDelta instead of Weight directly.
// A consolidation signal (analogous to PRP synthesis) converts
// PendingDelta → Weight for all tagged synapses on that neuron.
// Without consolidation, PendingDelta decays to zero.
```

This maps beautifully to our R-STDP eligibility trace mechanism. In fact, **eligibility traces ARE synaptic tags** — and the reward signal IS the consolidation trigger. R-STDP already implements a simplified version of STC. The extension would be: eligibility traces on one synapse can benefit from reward signals triggered by activity at a different synapse on the same neuron.

**Priority: MEDIUM.** Interesting, but R-STDP already captures the core idea. Worth revisiting when we tackle memory consolidation.

### 1.5 Heterosynaptic Plasticity

**What it is:** Weight changes at inactive synapses, driven by activity at *other* synapses on the same neuron. When some synapses undergo LTP, neighboring inactive synapses undergo LTD (and vice versa). This is a normalization mechanism.

**Why it matters:** Prevents total synaptic weight from growing unboundedly. Maintains competition between inputs — for one to strengthen, others must weaken. This is feature selection at the synaptic level.

**Implementation idea:** Already partially present in our predictive learning rule (which has a heterosynaptic competition term). Could be generalized: after any weight increase, proportionally decrease all other incoming weights to maintain a constant total incoming weight per neuron.

```go
// After strengthening connection i:
totalWeight := sum(all incoming weights)
if totalWeight > TargetTotalWeight {
    excess := totalWeight - TargetTotalWeight
    // Distribute reduction across all OTHER incoming connections proportionally
}
```

**Priority: MEDIUM.** The predictive rule already does this. For other rules, it would be a useful regularizer.

---

## 2. Structural Plasticity

*See `research/structural-plasticity.md` for the detailed design. This section covers mechanisms beyond what's already documented.*

### 2.1 Synaptogenesis — Beyond Co-Activity

Our structural plasticity grows connections between co-active neurons. Biology has additional signals:

**Neurotrophic factor signaling (BDNF):**
- Active neurons release BDNF (brain-derived neurotrophic factor)
- Dendrites grow toward BDNF sources
- Computationally: neurons that fire a lot become "attractive" — other neurons grow connections *to* them, not just between co-active pairs

**Implementation idea — Activity-Attractiveness:**
```go
// During growth phase, prioritize connecting TO highly active neurons
// (not just between co-active pairs)
// This creates hub neurons that aggregate successful signals
attractiveness[neuron] = firingRate * (MaxConnections - currentConnections)
```

This would naturally create the kind of hub-and-spoke topology seen in biological neural circuits (and small-world networks).

### 2.2 Dendritic Growth and Retraction

**What it is:** Dendrites physically extend and retract based on activity. Not just forming/removing individual synapses — entire dendritic branches grow toward active regions.

**Implementation idea — Connection Clusters:**
Instead of independent connections, group connections into "branches." Growing a branch means adding several connections to nearby neurons at once. Retracting a branch removes them all. This:
- Reduces the search space for growth (grow a branch toward a region, not individual connections)
- Creates correlated connectivity patterns (neurons in the same branch share inputs)
- Matches biological dendritic structure

**Priority: LOW for now.** Our connection model is flat (no grouping). This would require architectural changes.

### 2.3 Myelination-Like Priority

**What it is:** In biology, frequently-used axons become myelinated → faster signal propagation. More use = faster communication.

**Implementation idea — Variable Propagation Delay:**
```go
type Connection struct {
    // ... existing fields ...
    Delay uint8  // Propagation delay in ticks (default 1)
}

// Frequently-used connections get Delay reduced (faster)
// Rarely-used connections get Delay increased (slower, saves bandwidth)
```

This could create interesting temporal dynamics where well-learned pathways respond faster than novel ones. Currently we have uniform 1-tick delay everywhere.

**Priority: MEDIUM.** Easy to implement, interesting dynamics, but not clear if it helps learning.

---

## 3. Cortical Remapping

**What it is:** Large-scale reorganization of cortical maps after injury or altered input. The classic example: after arm amputation, the cortical area that processed arm sensation gets "invaded" by neighboring face/trunk representations.

**Key mechanisms:**
1. **Unmasking of existing connections:** Silent synapses that existed but were suppressed by inhibition become active when the dominant input disappears
2. **Competitive Hebbian dynamics:** Active inputs strengthen into vacated territory because there's no competition
3. **Structural plasticity:** Over weeks-months, actual new connections grow into the reorganized area

**Relevance to our architecture:** This is emergent behavior from the plasticity mechanisms we already have (or plan to have):
- Homeostatic scaling would make deprived neurons more excitable
- Structural plasticity would grow new connections from neighboring active neurons
- STDP would strengthen whatever new inputs arrive

**Implementation idea — No special mechanism needed!** Cortical remapping should emerge naturally if we have:
1. ✅ Homeostatic threshold adjustment (already in structural plasticity)
2. ✅ Co-activity-based connection growth (already in structural plasticity)
3. ⬜ Synaptic scaling (proposed in §1.3 above)
4. ⬜ Competitive inhibition between neuron groups

**Test case:** Create a network trained on two input regions. "Amputate" one region (stop sending input). Verify that the deprived neurons remap to process the remaining input.

**Priority: LOW (as a feature), HIGH (as a validation test).** If our plasticity mechanisms are right, remapping should be free.

---

## 4. Critical Periods

**What it is:** Time windows during development when neural circuits are maximally plastic and require specific input for proper wiring. After the critical period closes, the same circuits become resistant to change.

### 4.1 Biological Mechanisms

**Opening the critical period:**
- Maturation of GABAergic inhibition, particularly **parvalbumin-positive (PV+) fast-spiking interneurons**
- Counterintuitively, *inhibition enables plasticity* — it creates signal-to-noise ratio that makes Hebbian learning effective

**Closing the critical period:**
- **Perineuronal nets (PNNs):** Extracellular matrix structures that physically encase PV+ interneurons, stabilizing their synapses and preventing further rewiring
- **Myelin-associated inhibitors:** Myelination of axons prevents structural remodeling
- **Increased inhibition:** The same PV+ interneurons that opened the critical period now suppress plasticity by creating too-strong inhibition

**Reopening critical periods:**
- Dissolving PNNs (chondroitinase treatment) reopens plasticity in adults
- Reducing PV+ interneuron activity (chemogenetic silencing) reinstates critical period plasticity
- Certain drugs (e.g., valproic acid) can reopen critical periods

### 4.2 Implementation Ideas

**Training phases with decreasing plasticity:**
```go
type PlasticitySchedule struct {
    Phase           string   // "critical", "consolidation", "mature"
    LearningRate    float64  // Multiplier on weight changes
    StructuralRate  float64  // Multiplier on growth/pruning
    PruneThreshold  int32    // Increases as network matures
}

// Critical period: high learning rate, high structural plasticity, low prune threshold
// Consolidation: moderate learning, reduced structural changes
// Mature: low learning rate, minimal structural changes, high prune threshold
```

**Why this matters for our architecture:**
- **Training efficiency:** Early training should be aggressive (big weight changes, lots of rewiring). Later training should fine-tune existing structure.
- **Catastrophic forgetting prevention:** Once a network has learned good representations, reducing plasticity prevents new inputs from destroying them.
- **Mimics curriculum learning:** Train on easy examples first (during critical period), then gradually increase difficulty.

**PV+ interneuron analogue — Inhibitory Stabilization:**
The biological finding that *inhibition* enables and then closes critical periods suggests:
- During training: moderate inhibitory connections create competition → good feature selection
- After training: strengthen inhibitory connections → stabilize learned representations
- To "reopen" learning: temporarily weaken inhibition (lower inhibitory weights)

```go
// Per-module plasticity control
type Module struct {
    Neurons         []uint32
    PlasticityPhase Phase
    InhibitionLevel int32     // Higher = more stable, less plastic
}

func (m *Module) ReopenCriticalPeriod() {
    m.PlasticityPhase = Critical
    m.InhibitionLevel /= 2    // Reduce inhibition to enable plasticity
}
```

This connects directly to the modularity insight in DESIGN.md — different modules should have independent plasticity schedules.

**Priority: HIGH for training schedules, MEDIUM for the inhibitory stabilization mechanism.**

---

## 5. Biological Error Signaling — Analogues to Backpropagation

*This section expands on the arbiter neuron hypothesis documented in `research/arbiter-neurons.md`.*

### 5.1 Cerebellar Climbing Fibers — The Clearest Biological Error Circuit

**The circuit:**
```
Mossy fibers → Granule cells → Parallel fibers → Purkinje cells → Deep cerebellar nuclei
                                                       ↑
                                    Climbing fibers (from inferior olive)
                                    = DEDICATED ERROR SIGNAL
```

**Key properties:**
- Each Purkinje cell receives input from ~200,000 parallel fibers (forward computation) but only ONE climbing fiber (error signal)
- Climbing fibers fire at very low rates (1-4 Hz baseline) — silence = "you're doing fine"
- A climbing fiber burst causes **massive calcium influx** in the Purkinje cell → triggers **long-term depression (LTD)** at recently-active parallel fiber synapses
- This is supervised learning: the climbing fiber carries the error, parallel fiber activity identifies which synapses to blame

**Recent findings (2024):**
- Climbing fiber signals are **graded**, not all-or-nothing (Frontiers in Neural Circuits, 2013; Neuron, 2024)
- The magnitude of the climbing fiber response encodes error magnitude, not just error presence
- Granule cells can track long time intervals when combined with climbing fiber error signals (Neuron, June 2024)
- "Climbing fibers provide essential instructive signals for associative learning" — Nature Neuroscience, April 2024, confirming their role is genuinely instructive, not merely modulatory

**Implementation mapping to our architecture:**
```
Our architecture                    Cerebellum
─────────────────                   ──────────
Forward neurons                  →  Granule → Parallel fiber → Purkinje
Arbiter neurons                  →  Climbing fibers (inferior olive)
STDP (auto-strengthening)        →  Parallel fiber LTP (when no error)
Arbiter-triggered depression     →  Climbing fiber-triggered parallel fiber LTD
Reward signal (R-STDP)           →  Neuromodulatory context (dopamine, etc.)
```

**Concrete implementation — Graded Error Signal:**
The arbiter-neurons doc proposes binary error (fire = error, silence = OK). The climbing fiber literature suggests we should support **graded** error:
```go
type ArbiterNeuron struct {
    Neuron
    // ErrorMagnitude replaces binary fire/don't-fire
    // Higher activation = larger error = stronger depression of supervised synapses
}

// When arbiter fires with activation A:
// Depression applied to recently-active forward synapses =
//   A * DepressionRate (proportional to error magnitude)
```

### 5.2 Cortical Feedback Connections and Predictive Coding

**The anatomy:**
- Cortical layers 5/6 send **massive** feedback projections to layers 1/2/3 of lower areas
- There are MORE top-down (feedback) connections than bottom-up (feedforward) in cortex
- Feedforward: layers 2/3 → layer 4 of next area (driving input)
- Feedback: layers 5/6 → layers 1/2/3 of previous area (predictions/errors)

**Predictive coding theory (Rao & Ballard 1999; Friston 2005):**
```
Higher area sends PREDICTION down (via feedback connections)
Lower area computes: ERROR = actual_input - prediction
Lower area sends ERROR up (via feedforward connections)
Higher area updates its model to reduce error
```

This is **hierarchical error minimization** — conceptually similar to backpropagation, but computed locally at each level.

**Dendritic error computation (Urbanczik & Senn 2014; Sacramento et al. 2018):**

The most exciting recent development: **errors may be computed within single neurons via dendritic compartments**, not in separate error neurons.

Sacramento, Costa et al. (2018) — "Dendritic cortical microcircuits approximate the backpropagation algorithm":
- Pyramidal neurons have two dendritic compartments: **basal** (receives feedforward input) and **apical** (receives feedback/error signals)
- The voltage difference between somatic and apical compartments approximates the error gradient
- Learning rules based on this difference approximate backpropagation
- The model can be trained on classification and regression tasks with performance approaching backprop

**Experimental confirmation (2023-2025):**
- Bhaya-Grossman & Bhatt (2025, bioRxiv → Nature pending): Recorded from L5 pyramidal neuron somas AND their distal apical dendrites simultaneously during a brain-computer interface task. Found that **dendritic signals contain vectorized instructive signals** (error + reward information) that are spatially segregated from somatic computation.
- This is direct experimental evidence for the dendritic error computation model.

**Implementation idea — Dendritic Compartment Model:**
```go
type DendriticNeuron struct {
    Neuron
    ApicalActivation  int32  // Feedback/error signals accumulate here
    BasalActivation   int32  // Feedforward signals accumulate here
    // Somatic activation = BasalActivation (existing Activation field)
    
    // Learning signal = ApicalActivation - f(BasalActivation)
    // i.e., what feedback says minus what feedforward predicts
}

// Forward connections target BasalActivation
// Feedback (arbiter/error) connections target ApicalActivation
// Weight update ∝ (ApicalActivation - BasalActivation) * presynaptic_activity
```

This is more biologically accurate than separate arbiter neurons AND has a mathematical proof that it approximates backprop (Sacramento et al. 2018). **It could replace or augment the arbiter neuron concept** — instead of separate error neurons, the error is computed *within* forward neurons via their apical compartment.

**Priority: VERY HIGH.** This may be the most promising path to effective supervised learning in our architecture.

### 5.3 Feedback Alignment (Lillicrap et al. 2016)

**The key insight:** You don't need symmetric (transpose) weights for backprop to work. **Even random, fixed backward weights** can transmit useful error signals for learning.

**How it works:**
- Standard backprop: error at layer L is propagated to layer L-1 using W^T (transpose of forward weights)
- Feedback alignment: error at layer L is propagated to layer L-1 using B (random fixed matrix)
- Amazingly, the forward weights W gradually align with B during training, so the learning signal becomes increasingly accurate

**Why this matters for us:** It means arbiter/feedback connections DON'T need to have specific, carefully-tuned weights. Random backward connections can work. This is hugely encouraging because:
1. We can initialize arbiter connections randomly
2. They don't need to mirror forward connection topology exactly
3. The system self-organizes to make the error signals useful

**Implementation:** Our arbiter neurons could use random, fixed-weight backward connections. No need to learn arbiter connection weights — just forward weights.

```go
// When creating an arbiter layer:
for each arbiterNeuron {
    // Connect to random subset of supervised forward neurons
    // with random weights — they DON'T need to match forward topology
    for _, target := range randomSubset(forwardLayerNeurons) {
        net.Connect(arbiterNeuron, target, randomWeight())
    }
    // These weights stay FIXED. Only forward weights learn.
}
```

### 5.4 Target Propagation (Bengio 2014; Lee et al. 2015)

**The idea:** Instead of propagating error gradients backward, propagate **targets** — "this is what your output should have been."

**How it works:**
- Each layer has a paired "inverse" function (approximate decoder)
- The output error is converted to a target for the top hidden layer
- That target is propagated backward through inverse functions to generate targets for each layer
- Each layer learns locally: minimize difference between actual output and target

**Difference Target Propagation (Lee et al. 2015):** Adds a linear correction for imperfect inverse functions. Achieves performance comparable to backprop on deep networks.

**Relevance to our architecture:** Target propagation is naturally layer-wise local. Each layer only needs to know its target (from the layer above) and its actual output. This maps well to our graph-based architecture where there's no global gradient computation.

**Implementation challenge:** Requires paired encoder/decoder for each layer. In our architecture, we'd need bidirectional connections that can compute "what input would have produced this desired output" — this is close to the predictive coding model in §5.2.

### 5.5 The Saponati & Vinck Predictive Rule (Already Implemented)

Our `learning/predictive/` package implements the Saponati & Vinck (2023) rule. Worth noting how it connects to the error signaling discussion:

- **Prediction error IS the learning signal.** Each neuron predicts its next input; the error between prediction and actual input drives weight updates.
- **This is a form of self-supervised error signaling.** No external arbiter needed — each neuron is its own error computer.
- **STDP emerges from prediction error minimization.** The timing-dependent weight changes we see in STDP are a *consequence* of error minimization, not a fundamental rule.

**Connection to dendritic model:** The predictive rule could be enhanced by computing prediction error in a dendritic compartment (apical = prediction from feedback, basal = actual feedforward input). This would unify the predictive rule with the dendritic error computation model.

### 5.6 "Error Neurons" and "Teaching Signals" in Computational Neuroscience

**Key papers and concepts:**

1. **Lillicrap et al. (2020) — "Backpropagation and the brain"** (Nature Reviews Neuroscience):
   Major review arguing the brain DOES implement something functionally equivalent to backprop, through:
   - Feedback connections carrying error-like signals
   - Dendritic segregation of forward/error signals
   - Neuromodulatory gating of learning
   - The "weight transport problem" (how does the brain compute W^T?) solved by feedback alignment

2. **Kappel et al. (2015, 2017) — Synaptic Sampling:**
   Framework where stochastic synaptic weight fluctuations (noise) combined with reward modulation performs Bayesian inference over network configurations. **Directly relevant to our weight perturbation rule** — our perturbation approach is a simplified version of synaptic sampling.

3. **Marr-Albus cerebellar model (1969/1971):**
   The original computational model of cerebellar learning. Predicted climbing fiber error signals decades before experimental confirmation. Key insight: the cerebellum is a **pattern associator** trained by error signals, not a general-purpose computer.

4. **Actor-Critic models:**
   In reinforcement learning, the critic computes a value estimate and the temporal difference error. Biologically:
   - **Basal ganglia** = critic (dopamine signals encode prediction error)
   - **Cortex/cerebellum** = actor (performs actions, updated by error)
   - Our arbiter neurons are closest to the critic role

5. **Interneuron-mediated error signals:**
   Multiple groups have proposed that specific interneuron subtypes (SST+, VIP+) carry error or surprise signals:
   - **SST+ (somatostatin) interneurons:** inhibit pyramidal cell dendrites, could gate prediction errors
   - **VIP+ interneurons:** disinhibitory — they inhibit SST+ cells, which disinhibits pyramidal cells. This creates a "learning gate" — VIP+ activation = "pay attention, learn from this"

### 5.7 Synthesis: Error Signaling Architecture for Our Network

Combining all the above, here's a unified proposal:

**Option A: Separate Arbiter Neurons (simpler, current proposal)**
```
Forward layer: A₁ → A₂ → A₃ → Output
                ↑     ↑     ↑
Arbiter layer:  E₁ ← E₂ ← E₃ ← Error
```
- Arbiter neurons are separate units with backward connections
- Random fixed weights (feedback alignment)
- Fire to depress recently-active forward synapses
- Pro: Simple, clean separation of concerns
- Con: Doubles neuron count, arbiter neurons are single-purpose

**Option B: Dendritic Compartments (more biological, more powerful)**
```
Each forward neuron has:
  - Basal dendrite: receives feedforward input (normal connections)
  - Apical dendrite: receives feedback/error signals
  - Learning signal = apical - f(basal)

Layer N+1 sends feedback connections to Layer N's apical compartments
```
- No separate error neurons needed
- Each neuron computes its own error via dendritic voltage difference
- Proven to approximate backprop (Sacramento et al. 2018)
- Pro: No neuron count increase, mathematically grounded
- Con: Requires extending Neuron struct, more complex activation

**Option C: Hybrid (pragmatic)**
- Use dendritic compartments for within-module error computation
- Use arbiter neurons for cross-module error signals (like climbing fibers between cerebellum regions)
- Predictive learning rule as the default (self-supervised, no external error needed)
- Arbiter/dendritic error for supervised tasks that need external teaching signals

**Recommendation: Start with Option A (arbiter neurons) for simplicity, plan migration to Option B/C.** The arbiter neuron model is implementable today within our existing architecture. The dendritic compartment model is more powerful but requires extending the Neuron struct with a second accumulation compartment — a significant change.

---

## 6. Regional Plasticity Profiles

**The biological insight:** Not all brain circuits are equally plastic. The brain has a spectrum from almost-fixed (retina, brainstem reflexes) to maximally-plastic (hippocampus, cerebellum).

| Brain Region | Plasticity Level | Function | Error Infrastructure |
|---|---|---|---|
| Retina | Very low | Feature extraction (edges, colors) | None — hardwired |
| V1 (primary visual) | Low (adult) | Basic visual processing | Limited — mostly during critical period |
| Hippocampus | Very high | Memory formation | Massive — rapid synaptic modification |
| Cerebellum | High (specific) | Motor learning, timing | Dedicated climbing fiber error system |
| Prefrontal cortex | Moderate | Decision-making, planning | Dopaminergic modulation |
| Basal ganglia | Moderate | Action selection, reward learning | Dopamine error signals |

**Key insight for our architecture:** Different modules should have different plasticity profiles AND different error infrastructure:

```go
type ModulePlasticityProfile struct {
    LearningRule        LearningRule        // Which rule to use
    StructuralPlasticity StructuralPlasticity // Growth/pruning config
    PlasticityRate      float64              // Global learning rate multiplier
    HasArbiters         bool                 // Whether this module has error neurons
    CriticalPeriod      *CriticalPeriod      // Schedule for plasticity changes
}

// Example configurations:
var (
    SensoryProfile = ModulePlasticityProfile{
        LearningRule:   PredictiveRule,  // Self-supervised
        PlasticityRate: 0.1,             // Low — mostly fixed
        HasArbiters:    false,           // No external error signals
    }
    
    LearningProfile = ModulePlasticityProfile{
        LearningRule:   HybridRule,      // Predictive + R-STDP
        PlasticityRate: 1.0,             // Full plasticity
        HasArbiters:    true,            // Dedicated error infrastructure
        CriticalPeriod: &CriticalPeriod{...},
    }
    
    MemoryProfile = ModulePlasticityProfile{
        LearningRule:   RSTDP,           // Reward-modulated
        PlasticityRate: 2.0,             // Very high — rapid learning
        HasArbiters:    false,           // Uses reward signal instead
    }
)
```

**Priority: MEDIUM.** Requires the module/layer system to be implemented first (the layers discussion planned for 2026-02-22).

---

## 7. Implementation Roadmap

Ordered by impact and dependency:

### Phase 1: Stabilize Learning (Next)
1. **Homeostatic synaptic scaling** (§1.3) — multiplicative weight scaling to maintain target firing rates. HIGH priority. Stabilizes all learning rules.
2. **Metaplasticity / BCM sliding threshold** (§1.2) — per-neuron plasticity threshold that adapts to activity history. HIGH priority. Prevents runaway potentiation.

### Phase 2: Error Signaling (After Layers)
3. **Arbiter neurons v1** (§5.1, §5.3) — separate error neurons with random fixed backward weights. HIGH priority. First supervised learning mechanism beyond weight perturbation.
4. **Graded error signals** — arbiter activation magnitude encodes error magnitude (not binary). Builds on climbing fiber literature.

### Phase 3: Advanced Plasticity
5. **Critical period schedules** (§4.2) — decreasing plasticity over training. MEDIUM priority. Prevents catastrophic forgetting.
6. **Regional plasticity profiles** (§6) — different modules get different plasticity configurations.
7. **Dendritic compartment model** (§5.2) — extend Neuron struct with apical/basal compartments. Most powerful error signaling mechanism, but biggest architectural change.

### Phase 4: Refinements
8. **Synaptic tagging and capture** (§1.4) — two-phase weight consolidation.
9. **Heterosynaptic normalization** (§1.5) — weight competition within neurons.
10. **Variable propagation delay** (§2.3) — myelination-like priority for frequently-used connections.
11. **Cortical remapping validation** (§3) — test that plasticity mechanisms produce emergent remapping.

---

## Key References

### Synaptic Plasticity
- Bienenstock, Cooper & Munro (1982). "Theory for the development of neuron selectivity." *Journal of Neuroscience* 2(1), 32-48. — BCM theory
- Abraham & Bear (1996). "Metaplasticity: the plasticity of synaptic plasticity." *Trends in Neurosciences* 19(4), 126-130.
- Turrigiano et al. (1998). "Activity-dependent scaling of quantal amplitude in neocortical neurons." *Nature* 391, 892-896. — Synaptic scaling discovery
- Turrigiano (2008). "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell* 135(3), 422-435. — Comprehensive scaling review
- Frey & Morris (1997). "Synaptic tagging and long-term potentiation." *Nature* 385, 533-536. — STC model

### Error Signaling & Biological Backpropagation
- Marr (1969). "A theory of cerebellar cortex." *Journal of Physiology* 202(2), 437-470. — Original cerebellar learning model
- Albus (1971). "A theory of cerebellar function." *Mathematical Biosciences* 10(1-2), 25-61.
- Lillicrap et al. (2016). "Random synaptic feedback weights support error backpropagation for deep learning." *Nature Communications* 7, 13276. — Feedback alignment
- Lee, Zhang, Fischer & Bengio (2015). "Difference Target Propagation." *ECML/PKDD*.
- Sacramento, Costa et al. (2018). "Dendritic cortical microcircuits approximate the backpropagation algorithm." *arXiv:1810.11393*. — Dendritic error backprop
- Lillicrap, Santoro et al. (2020). "Backpropagation and the brain." *Nature Reviews Neuroscience* 21, 335-346. — Major review
- Saponati & Vinck (2023). "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule." *Nature Communications*.
- Rao & Ballard (1999). "Predictive coding in the visual cortex." *Nature Neuroscience* 2(1), 79-87. — Predictive coding
- Kappel, Habenschuss, Legenstein & Maass (2015). "Network Plasticity as Bayesian Inference." *PLoS Computational Biology*. — Synaptic sampling
- Bhaya-Grossman & Bhatt et al. (2025). "Vectorized instructive signals in cortical dendrites during a brain-computer interface task." *bioRxiv/Nature*. — Experimental evidence for dendritic error signals

### Critical Periods
- Hensch (2005). "Critical period plasticity in local cortical circuits." *Nature Reviews Neuroscience* 6, 877-888.
- Kuhlman et al. (2013). "A disinhibitory microcircuit initiates critical period plasticity." *Nature* 501, 543-546.

### Structural Plasticity
- Butz, Wörgötter & van Ooyen (2009). "Activity-dependent structural plasticity." *Brain Research Reviews* 60(2), 287-305.
- Holtmaat & Svoboda (2009). "Experience-dependent structural synaptic plasticity in the mammalian brain." *Nature Reviews Neuroscience* 10, 647-658.

---

*This document is a living reference. Update as we implement and discover what works.*
