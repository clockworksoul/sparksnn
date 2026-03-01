// Package stdp implements pure spike-timing-dependent plasticity
// as a learning rule for biomimetic networks.
//
// Pure STDP is unsupervised Hebbian learning based on spike timing:
//
//   - Pre fires before post (causal) → potentiation (strengthen).
//   - Post fires before pre (anti-causal) → depression (weaken).
//
// Weight changes are applied immediately based on timing — no reward
// signal is needed. The Eligibility field is used as a temporary
// accumulator within a tick; it is zeroed after being applied.
//
// For reward-modulated STDP (three-factor), see learning/rstdp.
package stdp

import (
	"math"

	bio "github.com/clockworksoul/sparksnn"
)

// Config holds tunable parameters for the pure STDP learning rule.
type Config struct {
	// APlus is the maximum weight increase for causal timing
	// (pre fires before post).
	APlus int32

	// AMinus is the maximum weight decrease for anti-causal
	// timing (post fires before pre).
	AMinus int32

	// TauPlus is the time constant (in ticks) for the causal
	// exponential decay window. Larger = wider window.
	TauPlus uint32

	// TauMinus is the time constant (in ticks) for the anti-causal
	// exponential decay window.
	TauMinus uint32

	// MaxWeightMagnitude caps the absolute value of weights after
	// learning. Prevents runaway weight growth. 0 = use MaxWeight.
	MaxWeightMagnitude int32
}

// DefaultConfig returns reasonable default pure STDP parameters.
func DefaultConfig() Config {
	return Config{
		APlus:              100,
		AMinus:             100,
		TauPlus:            5,
		TauMinus:           5,
		MaxWeightMagnitude: 0, // no cap (use MaxWeight)
	}
}

// Rule implements pure spike-timing-dependent plasticity.
type Rule struct {
	Config Config
}

// NewRule creates a pure STDP learning rule with the given config.
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

// clampWeight enforces MaxWeightMagnitude on a connection.
func (s *Rule) clampWeight(conn *bio.Connection) {
	maxMag := s.Config.MaxWeightMagnitude
	if maxMag > 0 && maxMag < bio.MaxWeight {
		if conn.Weight > maxMag {
			conn.Weight = maxMag
		}
		if conn.Weight < -maxMag {
			conn.Weight = -maxMag
		}
	}
}

// OnSpikePropagation evaluates pre-before-post timing. If the
// post-synaptic neuron has fired recently (within the STDP window),
// this is anti-causal: post fired before pre → weaken the weight
// directly.
func (s *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	if postLastFired == 0 || postLastFired >= preFiredAt {
		return
	}

	dt := preFiredAt - postLastFired
	delta := Window(dt, s.Config.AMinus, s.Config.TauMinus)
	if delta != 0 {
		conn.Weight = bio.ClampAdd(conn.Weight, -delta)
		s.clampWeight(conn)
	}
}

// OnPostFire evaluates post-before-pre timing for all incoming
// connections. For each incoming connection whose source has fired
// recently (within the STDP window), this is causal: pre fired
// before post → strengthen the weight directly.
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
			in.Conn.Weight = bio.ClampAdd(in.Conn.Weight, delta)
			s.clampWeight(in.Conn)
		}
	}
}

// OnReward is a no-op for pure STDP. Weight changes happen
// directly from spike timing — no reward signal is needed.
func (s *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {}

// Maintain is a no-op for pure STDP. There are no eligibility
// traces to decay — weight changes are applied immediately.
func (s *Rule) Maintain(net *bio.Network, tick uint32) {}
