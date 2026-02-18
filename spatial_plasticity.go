package biomimetic

import (
	"math"
	"math/rand/v2"
	"sort"
)

// SpatialPlasticityConfig extends PlasticityConfig with spatial
// awareness. New connections prefer nearby neurons, matching
// biological distance-dependent synaptogenesis.
type SpatialPlasticityConfig struct {
	PlasticityConfig

	// GrowthSigma controls distance bias for new connections.
	// Growth probability decays as exp(-d²/2σ²). Larger sigma =
	// more willingness to grow long-range connections.
	// 0 = no distance bias (fall back to base behavior).
	GrowthSigma float32

	// ForwardBias controls how strongly new connections prefer the
	// forward (increasing-X) direction. The X coordinate is treated
	// as network depth: inputs should be at low X, outputs at high X.
	//
	// For a candidate connection source→target, the directional
	// score is: max(targetX - sourceX, epsilon). Forward connections
	// (positive ΔX) get a score proportional to the depth gap;
	// backward connections get only epsilon (~0.01), making them
	// rare but not impossible.
	//
	// The score is raised to ForwardBias as a power:
	//   dirScore^ForwardBias
	// So ForwardBias=0 disables (all directions equal),
	// ForwardBias=1 is linear preference, ForwardBias=2 is strong.
	// Default 0 (disabled).
	ForwardBias float32

	// Positions is the neuron position slice from the Development
	// harness. Must be set before use and must outlive training
	// (not yet finalized).
	Positions []Position
}

// SpatialPlasticity implements structural plasticity with distance-
// dependent connection growth. Pruning and homeostasis are identical
// to DefaultPlasticity; only growth is spatially biased.
type SpatialPlasticity struct {
	Config SpatialPlasticityConfig
	base   DefaultPlasticity
}

// NewSpatialPlasticity creates a spatial structural plasticity
// controller.
func NewSpatialPlasticity(config SpatialPlasticityConfig) *SpatialPlasticity {
	return &SpatialPlasticity{
		Config: config,
		base: DefaultPlasticity{
			Config: config.PlasticityConfig,
		},
	}
}

// Remodel performs homeostasis, pruning, and spatially-biased growth.
func (sp *SpatialPlasticity) Remodel(net *Network, tick uint32) (pruned, grown int) {
	cfg := sp.Config.PlasticityConfig

	// Phase 1: Homeostatic threshold adjustment (same as base)
	if cfg.HomeostaticEnabled {
		sp.base.homeostasis(net, tick)
	}

	// Phase 2: Prune weak connections (same as base)
	pruned = sp.base.prune(net)

	// Phase 3: Spatially-biased growth
	grown = sp.grow(net, tick)

	// Phase 4: Spatially-biased exploratory growth for dead neurons
	if cfg.ExploratoryGrowth {
		grown += sp.explore(net, tick)
	}

	if pruned > 0 || grown > 0 {
		net.incomingIndex = nil
	}

	return pruned, grown
}

// distanceScore returns a [0,1] score based on distance, where
// closer neurons score higher. Uses Gaussian decay.
func (sp *SpatialPlasticity) distanceScore(a, b uint32) float64 {
	if sp.Config.GrowthSigma <= 0 || sp.Config.Positions == nil {
		return 1.0 // no spatial bias
	}
	if int(a) >= len(sp.Config.Positions) || int(b) >= len(sp.Config.Positions) {
		return 1.0
	}

	dist := sp.Config.Positions[a].distance(sp.Config.Positions[b])
	sigma := sp.Config.GrowthSigma
	return math.Exp(-float64(dist*dist) / float64(2*sigma*sigma))
}

// forwardScore returns a directional bias score for a connection
// from source to target, using the X coordinate as network depth.
// Forward connections (target.X > source.X) score high; backward
// connections score near-zero but not zero.
func (sp *SpatialPlasticity) forwardScore(source, target uint32) float64 {
	if sp.Config.ForwardBias <= 0 || sp.Config.Positions == nil {
		return 1.0 // disabled
	}
	if int(source) >= len(sp.Config.Positions) || int(target) >= len(sp.Config.Positions) {
		return 1.0
	}

	dx := float64(sp.Config.Positions[target].X - sp.Config.Positions[source].X)

	const epsilon = 0.01
	score := math.Max(dx, epsilon)

	return math.Pow(score, float64(sp.Config.ForwardBias))
}

type spatialCandidate struct {
	source uint32
	target uint32
	score  float64
}

// grow creates new connections between co-active neurons, biased
// toward spatially close pairs.
func (sp *SpatialPlasticity) grow(net *Network, tick uint32) int {
	cfg := sp.Config.PlasticityConfig
	if cfg.GrowthRate <= 0 {
		return 0
	}

	// Collect recently active neurons
	var active []uint32
	for i := range net.Neurons {
		n := &net.Neurons[i]
		if n.LastFired == 0 {
			continue
		}
		if tick-n.LastFired <= cfg.MinCoActivityWindow {
			active = append(active, uint32(i))
		}
	}

	if len(active) < 2 {
		return 0
	}

	// Build existing connection set
	existing := make(map[uint64]bool)
	for i := range net.Neurons {
		for _, conn := range net.Neurons[i].Connections {
			existing[uint64(i)<<32|uint64(conn.Target)] = true
		}
	}

	// Score candidates: timing score × distance score
	var candidates []spatialCandidate

	evalPair := func(s, t uint32) {
		if s == t {
			return
		}
		if cfg.Filter != nil && !cfg.Filter(s, t) {
			return
		}

		key := uint64(s)<<32 | uint64(t)
		if existing[key] {
			return
		}

		sLast := net.Neurons[s].LastFired
		tLast := net.Neurons[t].LastFired

		var timingScore float64
		if tLast == 0 {
			timingScore = 0.1 // target never fired — low priority
		} else if sLast >= tLast {
			return // anti-causal
		} else {
			dt := tLast - sLast
			timingScore = float64(cfg.MinCoActivityWindow-dt) / float64(cfg.MinCoActivityWindow)
			if timingScore <= 0 {
				return
			}
		}

		distScore := sp.distanceScore(s, t)
		fwdScore := sp.forwardScore(s, t)
		finalScore := timingScore * distScore * fwdScore

		if finalScore > 0.001 {
			candidates = append(candidates, spatialCandidate{
				source: s, target: t, score: finalScore,
			})
		}
	}

	if cfg.GrowthCandidates > 0 && len(active)*len(active) > cfg.GrowthCandidates {
		// Random sampling
		for i := 0; i < cfg.GrowthCandidates; i++ {
			s := active[rand.IntN(len(active))]
			t := active[rand.IntN(len(active))]
			evalPair(s, t)
		}
	} else {
		for _, s := range active {
			for _, t := range active {
				evalPair(s, t)
			}
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	grown := 0
	for _, c := range candidates {
		if grown >= cfg.GrowthRate {
			break
		}
		if cfg.MaxConnectionsPerNeuron > 0 &&
			len(net.Neurons[c.source].Connections) >= cfg.MaxConnectionsPerNeuron {
			continue
		}

		net.Neurons[c.source].Connections = append(
			net.Neurons[c.source].Connections,
			Connection{Target: c.target, Weight: cfg.InitialWeight},
		)
		grown++
	}

	return grown
}

// explore grows random incoming connections to dead neurons,
// biased toward nearby active neurons.
func (sp *SpatialPlasticity) explore(net *Network, tick uint32) int {
	cfg := sp.Config.PlasticityConfig
	if cfg.ExploratoryRate <= 0 {
		return 0
	}

	window := cfg.DeadThreshold
	if cfg.MinCoActivityWindow > window {
		window = cfg.MinCoActivityWindow
	}

	var active []uint32
	for i := range net.Neurons {
		n := &net.Neurons[i]
		if n.LastFired > 0 && tick-n.LastFired <= window {
			active = append(active, uint32(i))
		}
	}
	if len(active) == 0 {
		return 0
	}

	existing := make(map[uint64]bool)
	for i := range net.Neurons {
		for _, conn := range net.Neurons[i].Connections {
			existing[uint64(i)<<32|uint64(conn.Target)] = true
		}
	}

	grown := 0
	for i := range net.Neurons {
		n := &net.Neurons[i]
		idx := uint32(i)

		ticksSinceFire := tick - n.LastFired
		if n.LastFired == 0 {
			ticksSinceFire = tick
		}
		if ticksSinceFire <= cfg.DeadThreshold {
			continue
		}

		// Build distance-weighted candidates for this dead neuron
		type sourceCandidate struct {
			idx   uint32
			score float64
		}
		var sources []sourceCandidate

		for _, a := range active {
			if a == idx {
				continue
			}
			if cfg.Filter != nil && !cfg.Filter(a, idx) {
				continue
			}
			key := uint64(a)<<32 | uint64(idx)
			if existing[key] {
				continue
			}
			if cfg.MaxConnectionsPerNeuron > 0 &&
				len(net.Neurons[a].Connections) >= cfg.MaxConnectionsPerNeuron {
				continue
			}

			score := sp.distanceScore(a, idx) * sp.forwardScore(a, idx)
			if score > 0.001 {
				sources = append(sources, sourceCandidate{a, score})
			}
		}

		if len(sources) == 0 {
			continue
		}

		// Sort by distance score and pick top candidates
		sort.Slice(sources, func(i, j int) bool {
			return sources[i].score > sources[j].score
		})

		added := 0
		for _, src := range sources {
			if added >= cfg.ExploratoryRate {
				break
			}

			net.Neurons[src.idx].Connections = append(
				net.Neurons[src.idx].Connections,
				Connection{Target: idx, Weight: cfg.InitialWeight},
			)
			key := uint64(src.idx)<<32 | uint64(idx)
			existing[key] = true
			added++
			grown++
		}
	}

	return grown
}
