# Spatial Wiring & Developmental Scaffolding

*Design doc — February 17, 2026*

## Problem

Neurons in our network are currently just indices in an array with no
concept of spatial relationship. Connections are created fully-connected
or randomly, which doesn't scale (O(n²) connections) and doesn't produce
biologically plausible topology.

Real brains wire up using spatial proximity, chemical gradients, and
activity-dependent refinement. We need a lightweight mechanism that:

1. Guides initial connection formation using spatial rules
2. Supports structural plasticity (grow/prune connections)
3. Adds zero permanent cost to the trained network
4. Scales to millions of neurons

## Core Insight

**Spatial coordinates are developmental scaffolding.** They guide wiring
during construction and training, then get thrown away. Just like
biological axon guidance molecules that are expressed during development
and downregulated in the adult brain.

The trained network is just neurons, connections, and weights — no
coordinates needed at runtime.

## Design

### Development Harness

A new `Development` struct (or `Harness`, `Builder` — name TBD) manages
the construction and developmental phase of a network. It owns the
temporary spatial data and provides methods for wiring, training with
structural plasticity, and finalizing.

```
type Development struct {
    Net        *Network
    Positions  []Position   // temporary, per-neuron
    Layers     []Layer      // logical groupings
    Params     DevParams    // wiring rules, probabilities
}

type Position struct {
    X, Y float32
}

type Layer struct {
    Name       string
    StartIdx   uint32
    EndIdx     uint32       // exclusive
    Role       LayerRole    // Input, Hidden, Output, Inhibitory
}

type LayerRole int
const (
    RoleInput LayerRole = iota
    RoleHidden
    RoleOutput
    RoleInhibitory
)
```

After development is complete, call `Finalize()` to return the bare
`*Network` and discard all scaffolding.

### Connection Formation: Peter's Rule

Connection probability between two neurons decays with distance:

```
P(connect) = P_base * exp(-d² / (2 * σ²))
```

Where:
- `d` = Euclidean distance between neuron positions
- `σ` = connectivity radius (tunable per layer pair)
- `P_base` = maximum connection probability at distance 0

This gives sparse, local connectivity naturally. A neuron with σ=0.1
in a [0,1]² space connects mostly to its immediate neighbors.

**Inter-layer wiring** uses the same formula but with configurable σ
per layer pair. Input→Hidden might have a wide σ (broad receptive
fields), while Hidden→Hidden might have a narrow σ (local recurrence).

**Connection rules** specify which layer pairs can connect and with
what parameters:

```
type ConnectionRule struct {
    FromLayer    string
    ToLayer      string
    Sigma        float32    // spatial spread
    PBase        float32    // max connection probability
    InitWeight   WeightInit // how to initialize weights
}

type WeightInit struct {
    Min, Max int32
}
```

### Neuron Placement

Neurons within a layer are placed in 2D space. Several strategies:

- **Grid**: Regular grid positions. Simple, predictable.
- **Random uniform**: Random positions within a bounding box.
- **Gaussian clusters**: Neurons clustered around centers. Models
  cortical columns or nuclei.

For v1, random uniform within a per-layer bounding box is fine.

```
type PlacementStrategy int
const (
    PlaceGrid PlacementStrategy = iota
    PlaceRandom
    PlaceGaussian
)
```

### Overproduction & Pruning

Following the biological model:

1. **Overproduce**: Create connections at ~2x target density during
   initial wiring. Use a higher P_base than final desired density.

2. **Activity-dependent pruning**: After initial training, remove
   connections that:
   - Have near-zero absolute weight (|w| < threshold)
   - Were never active (neither pre nor post neuron fired during
     recent evaluation)

3. **Compensatory growth**: When a neuron loses too many connections
   (below a minimum), grow new ones to nearby active neurons. This
   prevents dead neurons from becoming permanently isolated.

This maps to the existing `StructuralPlasticity` interface:

```go
type StructuralPlasticity interface {
    Remodel(net *Network, tick uint32) (pruned, grown int)
}
```

A spatial-aware implementation would use the Development harness's
position data during training, then stop remodeling after finalization.

### Development Lifecycle

```
1. Create Development harness
2. Define layers (with placement strategy + bounding boxes)
3. Place neurons (assigns temporary positions)
4. Define connection rules (layer pairs + σ + P_base)
5. Wire network (creates initial connections per Peter's Rule)
6. Attach learning rule + structural plasticity
7. Train (present data, perturbation + STDP + remodel cycles)
8. Finalize → returns bare *Network, discards positions/layers
```

### Future Extensions

The harness design naturally supports future features without
changing the core Network:

- **Chemical gradients**: A function `f(position) → float32` that
  biases connection probability or initial weight in a region.
  E.g., a gradient that makes left-side neurons prefer excitatory
  connections and right-side prefer inhibitory.

- **Neuron type differentiation**: During development, assign neuron
  types (excitatory/inhibitory/modulatory) based on position +
  gradient signals. Models how cortical layers differentiate.

- **Topographic maps**: Input neurons arranged to match input feature
  topology (e.g., 2D pixel grid for vision). Spatial wiring
  preserves this topology through the hidden layers.

- **Critical periods**: Time-limited windows where structural
  plasticity is aggressive, followed by consolidation where
  topology becomes more fixed. Just toggle the remodeling rate.

- **Multi-region networks**: Multiple development harnesses, each
  building a region, then connected via long-range inter-region
  rules with wide σ.

## Scaling Considerations

At 1M neurons with ~100 connections each:
- Positions: 1M × 8 bytes = 8 MB (temporary)
- Connections: 100M × 8 bytes = 800 MB (permanent, this is the real cost)
- Neighbor lookup during structural plasticity: needs spatial index
  (k-d tree or grid hash) for efficient "find nearby neurons" queries.
  Only exists during development.

For the neighbor index, a simple 2D grid hash (divide space into cells,
bucket neurons by cell) gives O(1) amortized lookup and is trivial to
implement. No need for fancy data structures at our scale.

## Relationship to Existing Code

- **Network**: Unchanged. Still just neurons + connections + counter.
- **LearningRule**: Unchanged. Still per-spike and per-reward hooks.
- **StructuralPlasticity**: Existing interface works. New spatial-aware
  implementation uses positions from Development harness.
- **Benchmark tasks**: Updated to use Development harness for network
  construction instead of manual `Connect()` calls.

## Open Questions

1. **What σ values work?** Need experimentation. Start with σ that
   gives ~10-20% connectivity density for same-layer, ~30-50% for
   adjacent layers.

2. **How often to remodel?** Currently `Remodel()` is called per
   sample. May need to be less frequent for larger networks.

3. **Pruning threshold**: What weight magnitude counts as "near-zero"?
   Probably relative to the weight distribution, not absolute.

4. **Does 2D suffice or do we need 3D?** Start with 2D. Only add a
   third dimension if we find we can't represent the topology we need.

## First Implementation Plan

1. Define `Position`, `Layer`, `Development` structs
2. Implement neuron placement (random uniform)
3. Implement Peter's Rule wiring
4. Implement spatial-aware `StructuralPlasticity` (prune weak, grow
   to nearby active)
5. Implement `Finalize()` (strip scaffolding, return bare network)
6. Re-run Iris benchmark using Development harness
7. Compare against fully-connected baseline

## References

- Peters & Feldman (1976) — "Peter's Rule": connection probability
  proportional to axon/dendrite overlap
- Kappel et al. (2015) — "Network Plasticity as Bayesian Inference":
  synaptic stochasticity enables sampling from posterior distribution
  of network configurations
- Shatz (1996) — Activity-dependent refinement of visual system wiring
- PNAS (2020) — "How synaptic pruning shapes neural wiring during
  development": overproduction + selective elimination
- Egger et al. (2014) — Morpho-anatomical connection strategy for
  full-scale point-neuron microcircuits
