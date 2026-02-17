package biomimetic

import (
	"math"
	"math/rand/v2"
	"sort"
)

// StructuralPlasticity controls network topology changes: growing
// new connections and pruning unused ones. Called less frequently
// than LearningRule — typically once per training sample, not every
// tick.
//
// If Network.StructuralPlasticity is nil, topology is static
// (current default behavior).
type StructuralPlasticity interface {
	// Remodel evaluates the network for structural changes:
	// pruning weak connections, growing new ones, and adjusting
	// neuron excitability. Returns the number of connections
	// pruned and grown.
	Remodel(net *Network, tick uint32) (pruned, grown int)
}

// GrowthFilter is an optional function that restricts which neuron
// pairs can form new connections. Return true to allow the
// connection from source to target. Use this to enforce layer
// constraints or prevent feedback loops.
type GrowthFilter func(source, target uint32) bool

// PlasticityConfig holds parameters for the default structural
// plasticity implementation.
type PlasticityConfig struct {
	// --- Pruning ---

	// PruneThreshold is the weight magnitude below which a
	// connection is considered weak and eligible for removal.
	PruneThreshold int32

	// --- Growth ---

	// GrowthRate is the maximum number of new connections created
	// per Remodel call.
	GrowthRate int

	// MaxConnectionsPerNeuron caps outgoing connections per neuron.
	// 0 = unlimited.
	MaxConnectionsPerNeuron int

	// MinCoActivityWindow is the maximum tick gap between a
	// neuron's last fire and the current tick for it to be
	// considered "recently active." Should be >= the total
	// presentation + rest period per sample so that neurons
	// are still considered active when Remodel is called.
	MinCoActivityWindow uint32

	// InitialWeight is the starting weight for newly grown
	// connections.
	InitialWeight int32

	// GrowthCandidates is how many random (source, target) pairs
	// to evaluate per Remodel call. Limits computational cost for
	// large networks. 0 = evaluate all active pairs (fine for
	// small networks, expensive for large ones).
	GrowthCandidates int

	// Filter is an optional function to restrict which neuron
	// pairs can form connections. Nil = no restrictions.
	Filter GrowthFilter

	// --- Exploratory Growth ---

	// ExploratoryGrowth enables random connection growth to dead
	// neurons. When a neuron hasn't fired in DeadThreshold ticks,
	// in addition to lowering its threshold, we grow random
	// incoming connections from recently active neurons. This
	// solves the chicken-and-egg problem: dead neurons can't
	// participate in co-activity-based growth because they never
	// fire, but they can't fire because they have no connections.
	ExploratoryGrowth bool

	// ExploratoryRate is the number of random incoming connections
	// to grow per dead neuron per Remodel call.
	ExploratoryRate int

	// --- Homeostasis ---

	// HomeostaticEnabled controls whether neuron thresholds are
	// adjusted based on firing activity.
	HomeostaticEnabled bool

	// DeadThreshold is how many ticks without firing before a
	// neuron is considered "dead" and its threshold is lowered.
	DeadThreshold uint32

	// OveractiveWindow is how many ticks to look back when
	// determining if a neuron is overactive. A neuron that has
	// fired more than OveractiveFires times in this window gets
	// its threshold raised.
	OveractiveWindow uint32

	// OveractiveFires is the maximum fires within OveractiveWindow
	// before a neuron is considered overactive.
	OveractiveFires int

	// HomeostaticStep is how much to adjust the threshold per
	// Remodel call for dead or overactive neurons.
	HomeostaticStep int32

	// MinThreshold is the floor for threshold adjustment.
	MinThreshold int32

	// MaxThreshold is the ceiling for threshold adjustment.
	MaxThreshold int32
}

// DefaultPlasticityConfig returns reasonable defaults for structural
// plasticity.
func DefaultPlasticityConfig() PlasticityConfig {
	return PlasticityConfig{
		// Pruning
		PruneThreshold: 10,

		// Growth
		GrowthRate:              5,
		MaxConnectionsPerNeuron: 50,
		MinCoActivityWindow:     200,
		InitialWeight:           50,
		GrowthCandidates:        0, // evaluate all (fine for small networks)

		// Exploratory growth
		ExploratoryGrowth: true,
		ExploratoryRate:   2,

		// Homeostasis
		HomeostaticEnabled: true,
		DeadThreshold:      200,
		OveractiveWindow:   0, // disabled for v1 (no firing rate tracking)
		HomeostaticStep:    10,
		MinThreshold:       50,
		MaxThreshold:       1000,
	}
}

// DefaultPlasticity implements structural plasticity with pruning,
// growth, and homeostatic threshold adjustment.
type DefaultPlasticity struct {
	Config PlasticityConfig
}

// NewPlasticity creates a structural plasticity controller with
// the given config.
func NewPlasticity(config PlasticityConfig) *DefaultPlasticity {
	return &DefaultPlasticity{Config: config}
}

// Remodel performs three phases: homeostatic adjustment, pruning,
// and growth. Returns the number of connections pruned and grown.
func (p *DefaultPlasticity) Remodel(net *Network, tick uint32) (pruned, grown int) {
	// Phase 1: Homeostatic threshold adjustment
	if p.Config.HomeostaticEnabled {
		p.homeostasis(net, tick)
	}

	// Phase 2: Prune weak connections
	pruned = p.prune(net)

	// Phase 3: Grow new connections (co-activity based)
	grown = p.grow(net, tick)

	// Phase 4: Exploratory growth for dead neurons
	if p.Config.ExploratoryGrowth {
		grown += p.explore(net, tick)
	}

	// Invalidate incoming index if topology changed
	if pruned > 0 || grown > 0 {
		net.incomingIndex = nil
	}

	return pruned, grown
}

// homeostasis adjusts neuron thresholds based on recent activity.
// Dead neurons (haven't fired in a long time) get more excitable.
// For v1, we only rescue dead neurons — overactive detection
// requires firing rate tracking (deferred to v2).
func (p *DefaultPlasticity) homeostasis(net *Network, tick uint32) {
	cfg := p.Config

	for i := range net.Neurons {
		n := &net.Neurons[i]

		// Dead neuron check: hasn't fired in DeadThreshold ticks
		ticksSinceFire := tick - n.LastFired
		if n.LastFired == 0 {
			// Never fired — definitely dead
			ticksSinceFire = tick
		}

		if ticksSinceFire > cfg.DeadThreshold {
			n.Threshold -= cfg.HomeostaticStep
			if n.Threshold < cfg.MinThreshold {
				n.Threshold = cfg.MinThreshold
			}
		}
	}
}

// prune removes connections whose weight magnitude is below the
// prune threshold. Returns the number of connections removed.
func (p *DefaultPlasticity) prune(net *Network) int {
	threshold := p.Config.PruneThreshold
	pruned := 0

	for i := range net.Neurons {
		conns := net.Neurons[i].Connections
		if len(conns) == 0 {
			continue
		}

		// Filter in place: keep connections above threshold
		kept := conns[:0]
		for _, conn := range conns {
			w := conn.Weight
			if w < 0 {
				w = -w
			}
			if w == math.MinInt32 {
				// Special case: abs(MinInt32) overflows
				kept = append(kept, conn)
				continue
			}
			if w >= threshold {
				kept = append(kept, conn)
			} else {
				pruned++
			}
		}
		net.Neurons[i].Connections = kept
	}

	return pruned
}

// growthCandidate represents a potential new connection scored by
// causal timing.
type growthCandidate struct {
	source uint32
	target uint32
	score  int32 // higher = better candidate
}

// grow creates new connections between co-active neurons that
// aren't already connected. Returns the number of connections
// grown.
func (p *DefaultPlasticity) grow(net *Network, tick uint32) int {
	cfg := p.Config
	if cfg.GrowthRate <= 0 {
		return 0
	}

	// Collect recently active neurons
	active := make([]uint32, 0)
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

	// Build existing connection set for fast lookup
	existingConns := make(map[uint64]bool)
	for i := range net.Neurons {
		for _, conn := range net.Neurons[i].Connections {
			key := uint64(i)<<32 | uint64(conn.Target)
			existingConns[key] = true
		}
	}

	// Score candidates
	var candidates []growthCandidate

	if cfg.GrowthCandidates > 0 && len(active)*len(active) > cfg.GrowthCandidates {
		// Random sampling for large networks
		candidates = p.sampleCandidates(net, active, existingConns, tick)
	} else {
		// Evaluate all pairs for small networks
		candidates = p.allCandidates(net, active, existingConns, tick)
	}

	// Sort by score descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	// Grow top candidates
	grown := 0
	for _, c := range candidates {
		if grown >= cfg.GrowthRate {
			break
		}

		// Check connection cap
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

// allCandidates evaluates all active neuron pairs for growth.
func (p *DefaultPlasticity) allCandidates(net *Network, active []uint32, existing map[uint64]bool, tick uint32) []growthCandidate {
	var candidates []growthCandidate

	for _, s := range active {
		for _, t := range active {
			if s == t {
				continue
			}

			// Check filter
			if p.Config.Filter != nil && !p.Config.Filter(s, t) {
				continue
			}

			// Skip if already connected
			key := uint64(s)<<32 | uint64(t)
			if existing[key] {
				continue
			}

			// Score: prefer causal timing (source fired before target)
			sLastFired := net.Neurons[s].LastFired
			tLastFired := net.Neurons[t].LastFired

			var score int32

			if tLastFired == 0 {
				// Target never fired — valid growth target (exploratory).
				// Give it a moderate score so co-active pairs rank higher
				// but dead targets still get connections.
				score = 1
			} else if sLastFired >= tLastFired {
				// Anti-causal or simultaneous — skip
				continue
			} else {
				dt := tLastFired - sLastFired
				// Score inversely proportional to time gap
				// Closer timing = higher score
				score = int32(p.Config.MinCoActivityWindow) - int32(dt)
				if score <= 0 {
					continue
				}
			}

			candidates = append(candidates, growthCandidate{
				source: s,
				target: t,
				score:  score,
			})
		}
	}

	return candidates
}

// sampleCandidates randomly samples pairs from the active set.
func (p *DefaultPlasticity) sampleCandidates(net *Network, active []uint32, existing map[uint64]bool, tick uint32) []growthCandidate {
	var candidates []growthCandidate
	n := len(active)

	for i := 0; i < p.Config.GrowthCandidates; i++ {
		si := rand.IntN(n)
		ti := rand.IntN(n)
		if si == ti {
			continue
		}

		s := active[si]
		t := active[ti]

		if p.Config.Filter != nil && !p.Config.Filter(s, t) {
			continue
		}

		key := uint64(s)<<32 | uint64(t)
		if existing[key] {
			continue
		}

		sLastFired := net.Neurons[s].LastFired
		tLastFired := net.Neurons[t].LastFired

		var score int32
		if tLastFired == 0 {
			score = 1
		} else if sLastFired >= tLastFired {
			continue
		} else {
			dt := tLastFired - sLastFired
			score = int32(p.Config.MinCoActivityWindow) - int32(dt)
			if score <= 0 {
				continue
			}
		}

		candidates = append(candidates, growthCandidate{
			source: s,
			target: t,
			score:  score,
		})
	}

	return candidates
}

// explore grows random incoming connections to dead neurons from
// recently active neurons. This is "dendritic exploration" — dead
// neurons reach out randomly to find useful inputs.
// Uses DeadThreshold as the activity window (wider than
// MinCoActivityWindow) because we need to find ANY active neuron,
// not just recently co-active ones.
func (p *DefaultPlasticity) explore(net *Network, tick uint32) int {
	cfg := p.Config
	if cfg.ExploratoryRate <= 0 {
		return 0
	}

	// Find active neurons (potential sources) — use wider window
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

	// Build existing connection set
	existingConns := make(map[uint64]bool)
	for i := range net.Neurons {
		for _, conn := range net.Neurons[i].Connections {
			key := uint64(i)<<32 | uint64(conn.Target)
			existingConns[key] = true
		}
	}

	grown := 0
	for i := range net.Neurons {
		n := &net.Neurons[i]
		idx := uint32(i)

		// Is this neuron dead?
		ticksSinceFire := tick - n.LastFired
		if n.LastFired == 0 {
			ticksSinceFire = tick
		}
		if ticksSinceFire <= cfg.DeadThreshold {
			continue // not dead
		}

		// Grow random incoming connections from active neurons
		added := 0
		attempts := cfg.ExploratoryRate * 3 // try a few times to find valid sources
		for a := 0; a < attempts && added < cfg.ExploratoryRate; a++ {
			source := active[rand.IntN(len(active))]
			if source == idx {
				continue
			}

			// Check filter
			if cfg.Filter != nil && !cfg.Filter(source, idx) {
				continue
			}

			// Check not already connected
			key := uint64(source)<<32 | uint64(idx)
			if existingConns[key] {
				continue
			}

			// Check max connections
			if cfg.MaxConnectionsPerNeuron > 0 &&
				len(net.Neurons[source].Connections) >= cfg.MaxConnectionsPerNeuron {
				continue
			}

			net.Neurons[source].Connections = append(
				net.Neurons[source].Connections,
				Connection{Target: idx, Weight: cfg.InitialWeight},
			)
			existingConns[key] = true
			added++
			grown++
		}
	}

	return grown
}

// Disconnect removes all connections from source to target.
// Returns true if any connections were removed.
func (net *Network) Disconnect(source, target uint32) bool {
	if source >= uint32(len(net.Neurons)) {
		return false
	}

	conns := net.Neurons[source].Connections
	kept := conns[:0]
	removed := false

	for _, conn := range conns {
		if conn.Target == target {
			removed = true
		} else {
			kept = append(kept, conn)
		}
	}

	net.Neurons[source].Connections = kept
	if removed {
		net.incomingIndex = nil
	}
	return removed
}
