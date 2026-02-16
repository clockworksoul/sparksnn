package stdp

import (
	bio "github.com/clockworksoul/biomimetic-network"
)

// PureConfig holds tunable parameters for pure (non-reward-modulated)
// STDP. Weight changes happen immediately on spike timing — no
// eligibility traces, no reward gating.
//
// This is the "classic" STDP rule used in most unsupervised SNN
// learning benchmarks (e.g., Diehl & Cook 2015 MNIST).
type PureConfig struct {
	// APlus is the maximum weight increase for causal timing
	// (pre fires before post). Expressed as int16.
	APlus int16

	// AMinus is the maximum weight decrease for anti-causal timing
	// (post fires before pre). Expressed as int16.
	AMinus int16

	// TauPlus is the time constant (in ticks) for the causal
	// exponential decay window.
	TauPlus uint32

	// TauMinus is the time constant (in ticks) for the anti-causal
	// exponential decay window.
	TauMinus uint32

	// MaxWeightMagnitude caps the absolute value of weights.
	// 0 = use MaxWeight.
	MaxWeightMagnitude int16
}

// DefaultPureConfig returns reasonable defaults for pure STDP.
func DefaultPureConfig() PureConfig {
	return PureConfig{
		APlus:              10,  // Smaller than R-STDP — direct weight changes add up fast
		AMinus:             10,
		TauPlus:            5,
		TauMinus:           5,
		MaxWeightMagnitude: 0,
	}
}

// PureRule implements classic spike-timing-dependent plasticity.
// Weight changes are applied directly based on spike timing:
//
//   - Pre fires before post (causal) → potentiation (Δw > 0)
//   - Post fires before pre (anti-causal) → depression (Δw < 0)
//
// The magnitude decays exponentially with the time difference,
// controlled by tau parameters.
//
// Unlike the reward-modulated Rule, PureRule does not use eligibility
// traces or reward signals. This makes it suitable for unsupervised
// feature learning where no external feedback is available.
type PureRule struct {
	Config PureConfig
}

// NewPureRule creates a pure STDP learning rule.
func NewPureRule(config PureConfig) *PureRule {
	return &PureRule{Config: config}
}

// clampWeight applies the MaxWeightMagnitude cap if configured.
func (p *PureRule) clampWeight(conn *bio.Connection) {
	maxMag := p.Config.MaxWeightMagnitude
	if maxMag <= 0 || maxMag >= bio.MaxWeight {
		return
	}
	if conn.Weight > maxMag {
		conn.Weight = maxMag
	}
	if conn.Weight < -maxMag {
		conn.Weight = -maxMag
	}
}

// OnSpikePropagation is called when a pre-synaptic neuron fires.
// If the post-synaptic neuron fired recently (post before pre),
// this is anti-causal → depression. Apply weight change immediately.
func (p *PureRule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	if postLastFired == 0 || postLastFired >= preFiredAt {
		return
	}

	// Anti-causal: post fired before pre → weaken
	dt := preFiredAt - postLastFired
	delta := Window(dt, p.Config.AMinus, p.Config.TauMinus)
	if delta != 0 {
		conn.Weight = bio.ClampAdd(conn.Weight, -delta)
		p.clampWeight(conn)
	}
}

// OnPostFire is called when a post-synaptic neuron fires. For each
// incoming connection whose source fired recently (pre before post),
// this is causal → potentiation. Apply weight change immediately.
func (p *PureRule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
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

		// Causal: pre fired before post → strengthen
		dt := postFiredAt - preFiredAt
		delta := Window(dt, p.Config.APlus, p.Config.TauPlus)
		if delta != 0 {
			in.Conn.Weight = bio.ClampAdd(in.Conn.Weight, delta)
			p.clampWeight(in.Conn)
		}
	}
}

// OnReward is a no-op for pure STDP. No reward signal is used.
func (p *PureRule) OnReward(net *bio.Network, reward int16, tick uint32) {}

// Maintain is a no-op for pure STDP. No eligibility traces to decay.
func (p *PureRule) Maintain(net *bio.Network, tick uint32) {}
