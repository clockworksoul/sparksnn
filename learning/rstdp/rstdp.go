// Package rstdp implements reward-modulated spike-timing-dependent
// plasticity (R-STDP) as a learning rule for biomimetic networks.
//
// R-STDP operates in three phases:
//
//  1. Spike timing creates eligibility traces on connections
//     (OnSpikePropagation / OnPostFire).
//  2. Eligibility traces decay each tick (Maintain).
//  3. A reward signal consolidates traces into weight changes
//     (OnReward).
//
// Without reward signals, weights never change — STDP only marks
// candidates. This is the "three-factor" learning rule: pre-timing ×
// post-timing × reward.
//
// For pure STDP (no reward gating), see learning/stdp instead.
package rstdp

import (
	"math"

	bio "github.com/clockworksoul/sparksnn"
)

// Config holds tunable parameters for the R-STDP learning rule.
type Config struct {
	// APlus is the maximum eligibility increase for causal timing
	// (pre fires before post).
	APlus int32

	// AMinus is the maximum eligibility decrease for anti-causal
	// timing (post fires before pre).
	AMinus int32

	// TauPlus is the time constant (in ticks) for the causal
	// exponential decay window. Larger = wider window.
	TauPlus uint32

	// TauMinus is the time constant (in ticks) for the anti-causal
	// exponential decay window.
	TauMinus uint32

	// EligibilityDecayRate controls how quickly eligibility traces
	// fade per tick. Same fixed-point fraction as neuron DecayRate:
	// 58982 = ~90% retention per tick, 32768 = 50%.
	EligibilityDecayRate uint16

	// MaxWeightMagnitude caps the absolute value of weights after
	// learning. Prevents runaway weight growth. 0 = use MaxWeight.
	MaxWeightMagnitude int32

	// MultiplicativeReward switches from additive to multiplicative
	// reward consolidation. When true, the weight change is a
	// percentage of the current weight rather than a fixed delta:
	//
	//   delta = sign(reward * eligibility) * |weight| * RewardRate
	//
	// This makes strong connections resilient (hard to destroy) and
	// weak connections volatile (easy to reshape). Prevents the
	// neuron death caused by additive punishment on small weights.
	MultiplicativeReward bool

	// RewardRate is the fractional weight change per eligible
	// reward event when MultiplicativeReward is true. Expressed
	// as a fraction (e.g. 0.10 = 10% of current weight).
	// Ignored when MultiplicativeReward is false. Default: 0.10.
	RewardRate float64
}

// DefaultConfig returns reasonable default R-STDP parameters.
func DefaultConfig() Config {
	return Config{
		APlus:                100,
		AMinus:               100,
		TauPlus:              5,
		TauMinus:             5,
		EligibilityDecayRate: 52429, // ~80% retention per tick
		MaxWeightMagnitude:   0,     // no cap (use MaxWeight)
	}
}

// Rule implements reward-modulated spike-timing-dependent plasticity.
type Rule struct {
	Config Config
}

// NewRule creates an R-STDP learning rule with the given config.
func NewRule(config Config) *Rule {
	return &Rule{Config: config}
}

// Window calculates the exponentially-decayed magnitude for a
// given time difference and time constant. Returns 0 if dt exceeds
// a reasonable window (6× tau).
func Window(dt uint32, amplitude int32, tau uint32) int32 {
	if tau == 0 || dt > tau*6 {
		return 0
	}

	// Fixed-point retention: exp(-1/tau) * 65536
	retention := uint32(math.Round(math.Exp(-1.0/float64(tau)) * 65536))

	result := int64(amplitude)
	for i := uint32(0); i < dt; i++ {
		result = (result * int64(retention)) >> 16
		if result == 0 {
			return 0
		}
	}

	return int32(result)
}

// OnSpikePropagation evaluates pre-before-post timing. If the
// post-synaptic neuron has fired recently (within the STDP window),
// this is anti-causal: post fired before pre → depression.
func (s *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	if postLastFired == 0 || postLastFired >= preFiredAt {
		return
	}

	dt := preFiredAt - postLastFired
	delta := Window(dt, s.Config.AMinus, s.Config.TauMinus)
	if delta != 0 {
		conn.Eligibility = bio.ClampAdd(conn.Eligibility, -delta)
	}
}

// OnPostFire evaluates post-before-pre timing for all incoming
// connections. For each incoming connection whose source has fired
// recently (within the STDP window), this is causal: pre fired
// before post → strengthen.
func (s *Rule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
	for _, in := range incoming {
		if in.Conn == nil {
			continue
		}

		encodedFiredAt := in.SourceIndex
		if encodedFiredAt == 0 {
			continue
		}
		preFiredAt := encodedFiredAt - 1

		if preFiredAt >= postFiredAt {
			continue
		}

		dt := postFiredAt - preFiredAt
		delta := Window(dt, s.Config.APlus, s.Config.TauPlus)
		if delta != 0 {
			in.Conn.Eligibility = bio.ClampAdd(in.Conn.Eligibility, delta)
		}
	}
}

// OnReward consolidates eligibility traces into actual weight
// changes. Positive reward + positive eligibility = strengthen.
func (s *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {
	if reward == 0 {
		return
	}

	maxMag := s.Config.MaxWeightMagnitude
	if maxMag == 0 {
		maxMag = bio.MaxWeight
	}

	for i := range net.Neurons {
		for j := range net.Neurons[i].Connections {
			conn := &net.Neurons[i].Connections[j]
			if conn.Eligibility == 0 {
				continue
			}

			var delta int64

			if s.Config.MultiplicativeReward {
				// Multiplicative: delta proportional to current weight.
				// Direction from sign(reward * eligibility).
				// Magnitude from |weight| * rate.
				rate := s.Config.RewardRate
				if rate <= 0 {
					rate = 0.10
				}

				sign := int64(1)
				if (reward > 0) != (conn.Eligibility > 0) {
					sign = -1
				}

				absWeight := int64(conn.Weight)
				if absWeight < 0 {
					absWeight = -absWeight
				}
				// Minimum magnitude so zero/tiny weights can still move
				if absWeight < 50 {
					absWeight = 50
				}

				delta = sign * int64(float64(absWeight)*rate)
			} else {
				// Additive (original behavior)
				delta = (int64(reward) * int64(conn.Eligibility)) >> 8
			}

			if delta > int64(bio.MaxWeight) {
				delta = int64(bio.MaxWeight)
			}
			if delta < int64(bio.MinWeight) {
				delta = int64(bio.MinWeight)
			}

			conn.Weight = bio.ClampAdd(conn.Weight, int32(delta))

			if maxMag > 0 && maxMag < bio.MaxWeight {
				if conn.Weight > maxMag {
					conn.Weight = maxMag
				}
				if conn.Weight < -maxMag {
					conn.Weight = -maxMag
				}
			}

			conn.Eligibility = 0
		}
	}
}

// Maintain decays all eligibility traces across the network.
func (s *Rule) Maintain(net *bio.Network, tick uint32) {
	rate := s.Config.EligibilityDecayRate
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
