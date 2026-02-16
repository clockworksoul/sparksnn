# Structural Plasticity Design

*Growing and pruning connections in a spiking network.*
*Drafted 2026-02-16 by Matt & Dross*

---

## The Problem

Our network topology is fixed at construction time. Every connection that will ever exist must be created via `Connect()` before training begins. Learning rules can only adjust weights on existing connections — they can't discover that a connection *should* exist, or that one is useless and should be removed.

This is biologically unrealistic and practically limiting:

- **XOR failure mode:** Hidden neurons start connected to both inputs with random weights. They can't *specialize* by dropping one input — they can only shift weight magnitudes. With 2 inputs and 20 hidden neurons, there's no topological diversity for lateral inhibition to exploit.
- **MNIST inefficiency:** 784 × 400 = 313K connections, most carrying noise. The brain doesn't wire every input to every neuron — it grows connections where they're useful.
- **No feature discovery:** A neuron can't "reach out" to a useful input it wasn't initially connected to. The topology is a lottery — if the random initialization didn't wire the right things together, no amount of weight tuning fixes it.

---

## What Biology Does

### Synaptogenesis (Connection Growth)
- Dendrites physically extend toward active axons
- Driven by molecular signals: BDNF (brain-derived neurotrophic factor), glutamate spillover, calcium concentration
- **Computationally:** Neurons that are co-active but not connected develop connections. "Fire together, wire together" at the structural level.
- Timescale: hours to days (slow relative to synaptic weight changes)

### Synaptic Pruning
- Unused or weak synapses are eliminated
- Activity-dependent: "use it or lose it"
- Happens massively during development, continues throughout life
- Complement of long-term depression (LTD) — extreme LTD leads to pruning
- **Computationally:** Connections whose weight stays near zero get removed.
- Timescale: hours to weeks

### Neurogenesis (New Neurons)
- New neurons born in hippocampus (dentate gyrus) and olfactory bulb
- Controversial whether it happens elsewhere in adults
- New neurons integrate into existing circuits, form connections, specialize
- **Computationally:** Add new neurons to the network when capacity is exhausted.
- Timescale: weeks to months

### Homeostatic Plasticity
- Neurons regulate their own excitability to maintain a target firing rate
- Too much activity → raise threshold or scale down all incoming weights
- Too little activity → lower threshold or scale up all incoming weights
- Prevents runaway excitation and silent neurons
- **Computationally:** Adjust per-neuron threshold based on recent firing rate.
- Timescale: hours

---

## Proposed Design

### Scope

For the first implementation, I propose we tackle:

1. ✅ **Pruning** — remove weak/unused connections
2. ✅ **Growth** — form new connections between co-active neurons  
3. ✅ **Homeostatic scaling** — prevent dead and saturated neurons
4. ❌ **Neurogenesis** — defer (adds complexity, network size changes, index invalidation)

### Interface

```go
// StructuralPlasticity controls network topology changes:
// growing new connections and pruning unused ones.
// Called less frequently than LearningRule — typically once
// per sample or every N ticks, not every tick.
type StructuralPlasticity interface {
    // Remodel evaluates the network for structural changes:
    // pruning weak connections and growing new ones.
    // Called periodically (not every tick). Returns the number
    // of connections pruned and grown.
    Remodel(net *Network, tick uint32) (pruned, grown int)
}
```

Single method rather than separate Prune/Grow because:
- Pruning and growth are coupled (prune first to make room, then grow)
- Easier to maintain invariants (max connections per neuron, etc.)
- Called at the same frequency anyway

### Why Not Fold Into LearningRule?

Different timescales. Weight changes happen every tick. Structural changes should happen less often — every K ticks, or once per training sample. Separating them keeps the per-tick hot path fast and makes the calling frequency independently tunable.

### Pruning

#### Algorithm

```
For each neuron:
  For each outgoing connection:
    If |weight| < PruneThreshold for PrunePatience consecutive remodel calls:
      Remove the connection
```

#### Key Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| PruneThreshold | int16 | Weight magnitude below which a connection is "weak" | 10 |
| PrunePatience | uint16 | How many remodel calls a connection must stay weak before removal | 5 |

#### Data Structure Impact

We need to track how long a connection has been weak. Options:

**Option A: Add field to Connection**
```go
type Connection struct {
    Target      uint32
    Weight      int16
    Eligibility int16
    WeakCount   uint16  // +2 bytes, consecutive remodel calls below threshold
}
```
Pro: Simple. Con: Every connection pays the memory cost.

**Option B: External map**
```go
type pruneTracker struct {
    weakCounts map[connectionKey]uint16
}
```
Pro: Only weak connections pay. Con: Map overhead, cache-unfriendly.

**Recommendation: Option A.** 2 bytes per connection is cheap, and we avoid pointer chasing. Connection struct goes from 8 bytes to 10 (or 12 with alignment). We can pack it if needed — `Eligibility` and `WeakCount` could share a field since pruning and learning operate at different timescales, but that's premature optimization.

Actually, even simpler: **don't track duration at all for v1.** Just prune connections where `|weight| < PruneThreshold`. The learning rule already decays weights toward zero — if a weight has been pushed below the threshold, it's had its chance. This avoids any struct changes.

**v1 Pruning: prune if |weight| < PruneThreshold. No patience counter.**

#### Connection Removal Mechanics

Removing a connection from the middle of a `[]Connection` slice is O(n). Options:

1. **Swap-remove:** O(1) but changes ordering. Fine — connection order doesn't matter.
2. **Mark-and-sweep:** Set a tombstone flag, compact periodically. More complex.
3. **Rebuild slice:** Filter into new slice. O(n) per neuron but simple.

**Recommendation: Option 3 (rebuild).** Remodeling is infrequent. O(n) per neuron is fine when called every 100+ ticks. Simpler than swap-remove (no index confusion) and avoids tombstone complexity.

⚠️ **Important:** Removing connections invalidates the incoming index. Must call `net.incomingIndex = nil` after any structural change.

### Growth

This is the hard part.

#### The Core Question: Who Connects To Whom?

Biology: neurons grow connections to neighbors that are co-active. The signal is "I fire a lot, and that neuron over there fires a lot around the same time — maybe I should connect to it."

We don't have spatial locality (no physical position), so "neighbors" doesn't apply. We need a purely activity-based criterion.

#### Candidate Selection

**The naive approach** — check all possible pairs — is O(n²) per remodel call. For 400 hidden neurons, that's 160K pairs. For 1000 neurons, it's 1M. Not great.

**Better: activity-based candidates.** Only consider growth between neurons that have both been recently active. Track which neurons have fired recently and only evaluate pairs from that active set.

```go
// During Remodel:
// 1. Identify recently active neurons (fired within last K ticks)
// 2. For each active neuron, consider growth to other active neurons
//    that it's NOT already connected to
// 3. Score candidates and grow the top N
```

#### Growth Score

What makes a good new connection? The candidate should:

1. **Be co-active** — both neurons fire in the same time window
2. **Have causal timing** — source fires shortly before target (pre→post)
3. **Not duplicate** — no existing connection between them
4. **Respect locality** — prefer connections within the same layer or to adjacent layers

We could compute a **co-activity score** similar to STDP timing:
```
score = f(Δt) where Δt = target.LastFired - source.LastFired
```
Positive Δt (source before target) → good candidate for excitatory connection.
Negative Δt (target before source) → feedback connection (could be useful but risky).

#### Growth Budget

Can't grow unlimited connections — the network would become fully connected. Need constraints:

| Parameter | Type | Description | Default |
|---|---|---|---|
| MaxConnectionsPerNeuron | int | Cap on outgoing connections per neuron | 50 |
| GrowthRate | int | Max new connections per remodel call | 5 |
| MinCoActivityWindow | uint32 | Both neurons must have fired within this many ticks | 20 |
| InitialWeight | int16 | Starting weight for new connections | 50 |

#### Growth Algorithm (v1)

```
1. Collect recently active neurons (fired within MinCoActivityWindow ticks)
2. For each active neuron S (potential source):
   a. Skip if S has >= MaxConnectionsPerNeuron connections
   b. Build set of S's existing targets
   c. For each other active neuron T (potential target):
      - Skip if S == T
      - Skip if S already connects to T
      - Skip if S.LastFired >= T.LastFired (want causal: S fires first)
      - Compute score = f(T.LastFired - S.LastFired)
      - Add to candidate list
3. Sort candidates by score (descending)
4. Grow top GrowthRate candidates
```

#### What About Layer Constraints?

Should we only allow growth within certain layer boundaries? E.g., input→hidden is OK, but hidden→input is not?

**Arguments for layer constraints:**
- Prevents pathological feedback loops
- Keeps the network architecture interpretable
- Most biological circuits have laminar structure

**Arguments against:**
- Biology has massive feedback connections (cortex has more top-down than bottom-up)
- Recurrence might be exactly what we need for XOR
- Constraining growth is anti-discovery

**Recommendation:** Make it configurable. Default to unconstrained, but allow the caller to pass a `func(source, target uint32) bool` filter.

```go
// GrowthFilter is an optional function that restricts which
// neuron pairs can form new connections. Return true to allow.
type GrowthFilter func(source, target uint32) bool
```

### Homeostatic Scaling

Neither pruning nor growth solves dead neurons directly. If a neuron never fires, it can't be co-active with anything — growth ignores it. If all its incoming weights are low, they'll get pruned, making things worse.

**Homeostatic plasticity** closes this loop: neurons that fire too little become more excitable; neurons that fire too much become less excitable.

#### Algorithm

```
For each neuron:
  firingRate = (times fired in last W ticks) / W
  
  if firingRate < TargetRateLow:
    neuron.Threshold -= HomeostaticStep
    if neuron.Threshold < MinThreshold:
      neuron.Threshold = MinThreshold
      
  if firingRate > TargetRateHigh:
    neuron.Threshold += HomeostaticStep
    if neuron.Threshold > MaxThreshold:
      neuron.Threshold = MaxThreshold
```

#### Tracking Firing Rate

We don't currently track firing rate — only `LastFired`. Options:

**Option A: Rolling counter per neuron**
Add `FireCount uint32` to Neuron, increment on each fire. Compute rate as `FireCount / elapsed` over a window.

Problem: We'd need either a windowed counter (ring buffer per neuron — expensive) or a global reset point.

**Option B: Exponential moving average**
Add `FiringRate uint16` to Neuron (fixed-point). On each tick:
```
if neuron fired this tick:
    FiringRate = FiringRate * decay + (1 - decay) * MAX
else:
    FiringRate = FiringRate * decay
```

This is cheap (one multiply per neuron per tick), requires only 2 bytes, and automatically windows over time.

**Option C: Compute from LastFired during remodel only**
Don't track rate at all. During remodel, look at `LastFired`: if it's very old, the neuron is effectively dead. If it's very recent, it's very active.

This is crude — it doesn't distinguish "fired once 5 ticks ago" from "fires every tick." But it's zero-cost during normal operation.

**Recommendation: Option B** for accuracy, but **Option C for v1** (no struct changes, simple, good enough to rescue dead neurons).

#### Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| HomeostaticEnabled | bool | Whether to adjust thresholds | true |
| DeadTickThreshold | uint32 | Ticks without firing to be "dead" | 200 |
| HomeostaticStep | int16 | How much to adjust threshold per remodel | 10 |
| MinThreshold | int16 | Floor for threshold adjustment | 50 |
| MaxThreshold | int16 | Ceiling for threshold adjustment | 1000 |

---

## Combined Algorithm: Remodel()

```
func Remodel(net *Network, tick uint32):
    
    // Phase 1: Homeostatic threshold adjustment
    for each neuron:
        if tick - neuron.LastFired > DeadTickThreshold:
            neuron.Threshold -= HomeostaticStep  // make more excitable
            clamp to MinThreshold
        if neuron fired very recently (multiple times in window):
            neuron.Threshold += HomeostaticStep  // make less excitable
            clamp to MaxThreshold
    
    // Phase 2: Prune weak connections
    pruned = 0
    for each neuron:
        filter connections: keep only |weight| >= PruneThreshold
        pruned += removed count
    
    // Phase 3: Grow new connections
    grown = 0
    activeNeurons = [neurons where tick - LastFired < MinCoActivityWindow]
    candidates = score all valid (source, target) pairs from active set
    sort by score descending
    for top GrowthRate candidates:
        net.Connect(source, target, InitialWeight)
        grown++
    
    // Phase 4: Invalidate cached structures
    if pruned > 0 || grown > 0:
        net.incomingIndex = nil
    
    return pruned, grown
```

---

## Open Questions

### 1. Growth Direction
Should new connections be excitatory (positive weight) or should the sign be inferred from timing? Starting with small positive weight seems safest — if it should be inhibitory, the learning rule will push it negative.

### 2. Self-Connections
Should a neuron be able to connect to itself? Biology has autapses (self-synapses) but they're rare. Probably not useful for us — exclude by default.

### 3. Interaction With Learning Rules
When a new connection is grown with InitialWeight=50, should the learning rule immediately start adjusting it? Yes — that's the whole point. The structural plasticity creates the *opportunity*; the learning rule determines if it's useful.

But there's a subtlety: if we prune at |weight| < 10 and the learning rule pushes a new connection (initial=50) downward, it could get pruned before it has a chance to prove useful. The patience counter (deferred to v2) would help here. For v1: make InitialWeight large enough relative to PruneThreshold that there's a reasonable learning window.

### 4. Remodel Frequency
How often to call Remodel? Options:
- Every N ticks (e.g., every 200 ticks)
- Once per training sample (after rest period)
- When triggered by the caller

**Recommendation:** Expose as a caller decision, not built into the engine. The benchmark harness calls `Remodel()` at appropriate points. Suggest once per training sample as the default cadence.

### 5. Computational Cost
Growth candidate evaluation is the expensive part. For N active neurons, we evaluate O(N²) pairs. Mitigations:
- Only consider recently active neurons (reduces N dramatically)
- Cap GrowthRate to limit actual connection creation
- Random sampling: instead of scoring all pairs, sample K random pairs and grow the best

For v1, the active set will be small enough (XOR: ~10-15 neurons) that O(N²) is fine. For MNIST (400+ active neurons), we'll need the random sampling optimization.

### 6. Inhibitory Growth
Should structural plasticity grow inhibitory connections? Biology does — inhibitory interneurons extend axons and form new synapses. But for v1, growing only excitatory connections (positive InitialWeight) is simpler. Inhibitory growth is a v2 feature.

### 7. Directed vs Undirected Growth
Our connections are directed (source → target). Should growth be bidirectional? If A and B are co-active, do we grow A→B, B→A, or both?

**Recommendation:** Grow only in the causal direction (source fired before target → source→target). If the reverse timing also occurs, a separate growth event will catch it. This naturally discovers feedforward structure.

---

## Impact on Existing Code

### Network struct
```go
type Network struct {
    // ... existing fields ...
    
    // StructuralPlasticity controls connection growth and pruning.
    // If nil, topology is static (current behavior).
    StructuralPlasticity StructuralPlasticity
}
```

### Neuron struct
No changes for v1 (homeostasis uses LastFired, no firing rate tracker).

### Connection struct  
No changes for v1 (no patience counter).

### Incoming index
Must be invalidated after Remodel if any connections changed. Already handled by setting `net.incomingIndex = nil`.

### Serialization
StructuralPlasticity (like LearningRule) is a runtime behavior, not serialized state. No changes to Save/Load.

---

## Milestone Plan

**v1 (this PR):** Basic structural plasticity
- Pruning (simple threshold)
- Growth (co-activity based, causal timing preference)
- Homeostatic threshold adjustment (dead neuron rescue)
- `StructuralPlasticity` interface + default implementation
- Wire into XOR benchmark, test whether it helps

**v2:** Refinements
- Patience counter for pruning (don't prune too eagerly)
- Exponential moving average firing rate (better homeostasis)
- Growth filter function (layer constraints)
- Random sampling for growth candidates (MNIST scale)
- Inhibitory connection growth

**v3:** Advanced
- Neurogenesis (add neurons dynamically)
- Activity-dependent decay rate adjustment
- Connection-age-dependent pruning (older = harder to prune)

---

## References

- Butz, M., Wörgötter, F., & van Ooyen, A. (2009). "Activity-dependent structural plasticity." *Brain Research Reviews*, 60(2), 287-305.
- Lamprecht, R., & LeDoux, J. (2004). "Structural plasticity and memory." *Nature Reviews Neuroscience*, 5(1), 45-54.
- Turrigiano, G.G. (2008). "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell*, 135(3), 422-435.
- Diehl, P.U. & Cook, M. (2015). "Unsupervised learning of digit recognition using STDP." — Uses homeostatic mechanisms for SNN training.

---

*This is potentially the most important feature we add. Weight tuning finds the best version of a fixed topology. Structural plasticity finds the right topology in the first place.*
