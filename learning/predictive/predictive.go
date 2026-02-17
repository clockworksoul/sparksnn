// Package predictive implements the predictive learning rule from
// Saponati & Vinck (2023) for biomimetic networks.
//
// The core idea: each neuron tries to predict its own future inputs.
// STDP-like behavior emerges as a byproduct of this prediction
// optimization. No external reward signal is needed — learning is
// entirely self-supervised.
//
// Reference: "Sequence anticipation and spike-timing-dependent
// plasticity emerge from a predictive learning rule"
// Nature Communications 14, 4985 (2023)
package predictive

import (
	bio "github.com/clockworksoul/biomimetic-network"
)

// Config holds tunable parameters for the predictive learning rule.
type Config struct {
	// LearningRate controls the magnitude of weight updates.
	// Expressed as a fixed-point fraction of 65536:
	//   655 = ~1% learning rate
	//   3277 = ~5%
	//   6554 = ~10%
	// Smaller values = slower, more stable learning.
	LearningRate uint16

	// EligibilityDecayRate controls how quickly eligibility traces
	// (input history) fade. Same semantics as neuron DecayRate:
	//   58982 = ~90% retention per tick
	//   52429 = ~80% retention per tick
	//   32768 = 50% retention per tick
	EligibilityDecayRate uint16

	// MaxWeightMagnitude caps the absolute value of weights after
	// updates. 0 = no cap (use MaxWeight).
	MaxWeightMagnitude int32

	// PredictionScale controls how the prediction v*w is scaled
	// back into range. The raw product of activation (int32)
	// times weight (int32) is int64. We right-shift by this amount.
	// Default: 15 (divides by 32768). Adjust based on the magnitude
	// of weights in your network — larger weights need larger scale.
	PredictionScale uint8
}

// DefaultConfig returns reasonable starting parameters.
func DefaultConfig() Config {
	return Config{
		LearningRate:         655,   // ~1%
		EligibilityDecayRate: 52429, // ~80% retention per tick
		MaxWeightMagnitude:   0,
		PredictionScale:      15,
	}
}

// Rule implements the predictive learning rule. Instead of explicitly
// programming STDP, each neuron optimizes a prediction objective:
// minimize the error between predicted and actual incoming signals.
//
// STDP emerges because:
//   - Pre-before-post: the pre-synaptic input predicts the post
//     firing → gets credit (potentiation).
//   - Post-before-pre: the input is redundant/predicted by other
//     inputs → loses credit (depression).
type Rule struct {
	Config Config
}

// NewRule creates a predictive learning rule with the given config.
func NewRule(config Config) *Rule {
	return &Rule{Config: config}
}

// predict computes the predicted input for a connection given the
// post-synaptic neuron's current activation.
func (p *Rule) predict(postActivation, weight int32) int32 {
	product := int64(postActivation) * int64(weight)
	shifted := product >> p.Config.PredictionScale
	if shifted > int64(bio.MaxWeight) {
		return bio.MaxWeight
	}
	if shifted < int64(bio.MinWeight) {
		return bio.MinWeight
	}
	return int32(shifted)
}

// OnSpikePropagation is called when a pre-synaptic neuron fires
// and delivers a signal through a connection. Accumulates into
// the eligibility trace to record that this synapse was active.
func (p *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	conn.Eligibility = bio.ClampAdd(conn.Eligibility, conn.Weight)
}

// OnPostFire is called when a post-synaptic neuron fires.
// This is where the predictive weight update happens: computing
// prediction errors and updating weights to reduce them.
//
// The update rule (adapted to integer math):
//   w_t = w_{t-1} + η * (ε * v_{t-1} + E * p_{t-1})
func (p *Rule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
	if len(incoming) == 0 {
		return
	}

	lr := int64(p.Config.LearningRate)
	scale := p.Config.PredictionScale
	maxMag := p.Config.MaxWeightMagnitude
	if maxMag == 0 {
		maxMag = bio.MaxWeight
	}

	// Phase 1: Compute per-synapse prediction errors and global error.
	var globalError int64

	type synapseError struct {
		error int32
		conn  *bio.Connection
	}
	errors := make([]synapseError, 0, len(incoming))

	for _, in := range incoming {
		if in.Conn == nil {
			continue
		}

		actual := int64(in.Conn.Eligibility)
		predicted := (actual * int64(in.Conn.Weight)) >> scale
		err := actual - predicted

		if err > int64(bio.MaxWeight) {
			err = int64(bio.MaxWeight)
		}
		if err < int64(bio.MinWeight) {
			err = int64(bio.MinWeight)
		}

		errors = append(errors, synapseError{error: int32(err), conn: in.Conn})

		globalError += err * int64(in.Conn.Weight) >> scale
	}

	// Phase 2: Apply weight updates.
	for _, se := range errors {
		term1 := int64(se.error) * int64(se.conn.Eligibility) >> scale
		term2 := globalError * int64(se.conn.Eligibility) >> scale

		delta := (lr * (term1 + term2)) >> 16
		if delta == 0 && (term1+term2) != 0 {
			if term1+term2 > 0 {
				delta = 1
			} else {
				delta = -1
			}
		}

		if delta > int64(bio.MaxWeight) {
			delta = int64(bio.MaxWeight)
		}
		if delta < int64(bio.MinWeight) {
			delta = int64(bio.MinWeight)
		}

		se.conn.Weight = bio.ClampAdd(se.conn.Weight, int32(delta))

		if maxMag > 0 && maxMag < bio.MaxWeight {
			if se.conn.Weight > maxMag {
				se.conn.Weight = maxMag
			}
			if se.conn.Weight < -maxMag {
				se.conn.Weight = -maxMag
			}
		}
	}
}

// OnReward is a no-op for predictive learning. The rule is fully
// self-supervised — no external reward signal needed.
func (p *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {}

// Maintain decays eligibility traces across the network.
func (p *Rule) Maintain(net *bio.Network, tick uint32) {
	rate := p.Config.EligibilityDecayRate
	if rate == 0 {
		return
	}

	for i := range net.Neurons {
		for j := range net.Neurons[i].Connections {
			conn := &net.Neurons[i].Connections[j]
			if conn.Eligibility == 0 {
				continue
			}

			decayed := (int64(conn.Eligibility) * int64(rate)) >> 16
			conn.Eligibility = int32(decayed)
		}
	}
}
