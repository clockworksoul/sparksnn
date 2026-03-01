// Package perturbation implements a weight perturbation learning rule
// for biomimetic spiking networks.
//
// Weight perturbation is a gradient-free optimization method:
//
//  1. Randomly perturb one synaptic weight
//  2. Evaluate network performance (via reward signal)
//  3. Keep the perturbation if performance improved, revert if worse
//
// This provides per-connection credit assignment without backpropagation
// or gradient computation. Related to biological "synaptic sampling"
// observed in cortex, where synapses stochastically explore weight
// space and stabilize at configurations that correlate with reward.
//
// The rule operates through the OnReward hook: each reward signal
// triggers evaluation of the previous perturbation and selection of
// the next one. Callers should deliver a reward signal after each
// training sample with a value proportional to performance (e.g.,
// number of correct classifications).
//
// OnSpikePropagation, OnPostFire, and Maintain are no-ops — this
// rule does not use spike timing information.
package perturbation

import (
	"math/rand/v2"

	bio "github.com/clockworksoul/sparksnn"
)

// Config holds tunable parameters for weight perturbation learning.
type Config struct {
	// PerturbSize is the initial maximum magnitude of random weight
	// perturbations. The actual perturbation is uniform in
	// [-PerturbSize, +PerturbSize].
	PerturbSize int32

	// MaxPerturbSize caps adaptive perturbation growth.
	MaxPerturbSize int32

	// AdaptAfter doubles PerturbSize after this many consecutive
	// steps with no improvement. 0 disables adaptation.
	AdaptAfter int

	// KeepEqualProb is the probability (0.0-1.0) of keeping a
	// perturbation that produces equal reward (exploration).
	// 0.5 is a good default.
	KeepEqualProb float64

	// MaxWeightMagnitude caps the absolute value of weights.
	// 0 = no cap (use bio.MaxWeight).
	MaxWeightMagnitude int32

	// BatchSize is the number of OnReward calls to accumulate
	// before evaluating a perturbation. Weight perturbation needs
	// the full-batch signal (all patterns) to make good decisions.
	// Set to the number of training patterns (e.g., 4 for XOR).
	// 1 = evaluate per sample (noisy). 0 defaults to 1.
	BatchSize int
}

// DefaultConfig returns reasonable defaults for weight perturbation.
func DefaultConfig() Config {
	return Config{
		PerturbSize:        200,
		MaxPerturbSize:     2000,
		AdaptAfter:         200,
		KeepEqualProb:      0.5,
		MaxWeightMagnitude: 0,
	}
}

// Rule implements weight perturbation learning.
//
// State tracks the current perturbation being evaluated. After each
// OnReward call, the previous perturbation is accepted or reverted
// and a new one is applied.
type Rule struct {
	Config Config

	// rng is the random source for perturbation selection.
	rng *rand.Rand

	// State for the pending perturbation
	lastReward     int32
	hasLastReward  bool
	pendingNeuron  int // neuron index of perturbed connection
	pendingConn    int // connection index within that neuron
	oldWeight      int32
	hasPending     bool
	noImproveCount int
	perturbSize    int32

	// Batch accumulation
	batchReward int32
	batchCount  int

	// Cached list of learnable connection locations.
	// Rebuilt when network topology changes.
	connList     []connRef
	connListSize int // len(net.Neurons) when list was built
}

type connRef struct {
	neuronIdx int
	connIdx   int
}

// NewRule creates a weight perturbation learning rule.
func NewRule(config Config) *Rule {
	return &Rule{
		Config:      config,
		rng:         rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
		perturbSize: config.PerturbSize,
	}
}

// ensureConnList builds or refreshes the cached connection list.
func (r *Rule) ensureConnList(net *bio.Network) {
	if len(r.connList) > 0 && r.connListSize == len(net.Neurons) {
		return
	}
	r.connList = r.connList[:0]
	for i := range net.Neurons {
		for j := range net.Neurons[i].Connections {
			r.connList = append(r.connList, connRef{i, j})
		}
	}
	r.connListSize = len(net.Neurons)
}

// OnSpikePropagation is a no-op for weight perturbation.
func (r *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
}

// OnPostFire is a no-op for weight perturbation.
func (r *Rule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
}

// OnReward accumulates reward signals and evaluates perturbations
// at batch boundaries.
//
// Call once per training sample with reward > 0 for correct, <= 0
// for incorrect. After BatchSize calls, the accumulated reward is
// compared against the previous batch to decide whether to keep
// or revert the current weight perturbation.
func (r *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {
	r.ensureConnList(net)
	if len(r.connList) == 0 {
		return
	}

	batchSize := r.Config.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}

	// Accumulate reward within batch
	r.batchReward += reward
	r.batchCount++
	if r.batchCount < batchSize {
		return // not yet a full batch
	}

	// Full batch reached — evaluate and perturb
	batchTotal := r.batchReward
	r.batchReward = 0
	r.batchCount = 0

	maxMag := r.Config.MaxWeightMagnitude
	if maxMag == 0 {
		maxMag = bio.MaxWeight
	}

	clamp := func(w int32) int32 {
		if maxMag > 0 && maxMag < bio.MaxWeight {
			if w > maxMag {
				return maxMag
			}
			if w < -maxMag {
				return -maxMag
			}
		}
		return w
	}

	// Phase 1: Evaluate previous perturbation
	if r.hasPending && r.hasLastReward {
		conn := &net.Neurons[r.pendingNeuron].Connections[r.pendingConn]

		if batchTotal > r.lastReward {
			// Improvement — keep perturbation
			r.noImproveCount = 0
		} else if batchTotal < r.lastReward {
			// Worse — revert
			conn.Weight = r.oldWeight
			r.noImproveCount++
		} else {
			// Equal — keep with some probability (exploration)
			if r.rng.Float64() >= r.Config.KeepEqualProb {
				conn.Weight = r.oldWeight
			}
			r.noImproveCount++
		}

		// Adaptive perturbation size
		if r.Config.AdaptAfter > 0 && r.noImproveCount >= r.Config.AdaptAfter {
			r.perturbSize = min(r.perturbSize*2, r.Config.MaxPerturbSize)
			r.noImproveCount = 0
		}
	}

	// Phase 2: Apply new perturbation
	ref := r.connList[r.rng.IntN(len(r.connList))]
	conn := &net.Neurons[ref.neuronIdx].Connections[ref.connIdx]

	r.pendingNeuron = ref.neuronIdx
	r.pendingConn = ref.connIdx
	r.oldWeight = conn.Weight
	r.hasPending = true

	delta := r.rng.Int32N(r.perturbSize*2+1) - r.perturbSize
	conn.Weight = clamp(bio.ClampAdd(conn.Weight, delta))

	r.lastReward = batchTotal
	r.hasLastReward = true
}

// Maintain is a no-op for weight perturbation.
func (r *Rule) Maintain(net *bio.Network, tick uint32) {
}
