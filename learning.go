package biomimetic

import "math"

// LearningRule defines the interface for synaptic plasticity mechanisms.
// Implementations are called by the Network at key moments during
// signal propagation. This allows different learning algorithms to
// be swapped in without changing the core engine.
//
// All methods receive the current network counter tick for
// time-dependent calculations.
type LearningRule interface {
	// OnSpikePropagation is called when a pre-synaptic neuron fires
	// and sends a signal through a connection. It receives the
	// connection, the tick when the pre-synaptic neuron fired, and
	// the tick when the post-synaptic neuron last fired (0 if never).
	//
	// For STDP: this is where pre-before-post timing is evaluated.
	OnSpikePropagation(conn *Connection, preFiredAt, postLastFired uint32)

	// OnPostFire is called when a post-synaptic neuron fires.
	// It receives the neuron's incoming connections so the rule
	// can evaluate post-before-pre timing for recent inputs.
	OnPostFire(incoming []IncomingConnection, postFiredAt uint32)

	// OnReward is called when a global reward or punishment signal
	// is delivered to the network. The signal is a signed value:
	// positive = reward, negative = punishment.
	//
	// For R-STDP: this consolidates eligibility traces into actual
	// weight changes.
	OnReward(net *Network, reward int16, tick uint32)

	// Maintain is called once per tick for housekeeping: decaying
	// eligibility traces, pruning dead connections, etc.
	Maintain(net *Network, tick uint32)
}

// IncomingConnection represents an inbound connection to a neuron,
// referencing the source neuron and the specific connection struct.
// Used by OnPostFire to evaluate post-synaptic timing.
type IncomingConnection struct {
	SourceIndex uint32
	Conn        *Connection
}

// NoOpLearning is a learning rule that does nothing. Use it when
// you want a static network with no plasticity (e.g., the C. elegans
// connectome with fixed weights).
type NoOpLearning struct{}

func (NoOpLearning) OnSpikePropagation(*Connection, uint32, uint32) {}
func (NoOpLearning) OnPostFire([]IncomingConnection, uint32)        {}
func (NoOpLearning) OnReward(*Network, int16, uint32)               {}
func (NoOpLearning) Maintain(*Network, uint32)                      {}

// STDPConfig holds tunable parameters for the STDP learning rule.
type STDPConfig struct {
	// APlus is the maximum eligibility increase for causal timing
	// (pre fires before post). Expressed as int16.
	APlus int16

	// AMinus is the maximum eligibility decrease for anti-causal
	// timing (post fires before pre). Expressed as int16.
	AMinus int16

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
	MaxWeightMagnitude int16
}

// DefaultSTDPConfig returns reasonable default STDP parameters.
func DefaultSTDPConfig() STDPConfig {
	return STDPConfig{
		APlus:                100,
		AMinus:               100,
		TauPlus:              5,
		TauMinus:             5,
		EligibilityDecayRate: 52429, // ~80% retention per tick
		MaxWeightMagnitude:   0,     // no cap (use MaxWeight)
	}
}

// STDPRule implements reward-modulated spike-timing-dependent
// plasticity. It operates in three phases:
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
type STDPRule struct {
	Config STDPConfig
}

// NewSTDPRule creates an STDP learning rule with the given config.
func NewSTDPRule(config STDPConfig) *STDPRule {
	return &STDPRule{Config: config}
}

// stdpWindow calculates the exponentially-decayed magnitude for a
// given time difference and time constant. Returns 0 if dt exceeds
// a reasonable window (6× tau).
func stdpWindow(dt uint32, amplitude int16, tau uint32) int16 {
	if tau == 0 || dt > tau*6 {
		return 0
	}

	// exp(-dt/tau) approximated with integer math.
	// We compute in int32 to avoid overflow.
	// Using the identity: exp(-dt/tau) ≈ (tau-1)^dt / tau^dt
	// But for simplicity and accuracy, we use a lookup-style
	// approach: multiply by retention fraction per tick.
	//
	// retention = exp(-1/tau) ≈ (tau-1)/tau in fixed point
	// For tau=5: retention per tick ≈ 0.8187 ≈ 53674/65536
	if tau == 0 {
		return 0
	}

	// Fixed-point retention: exp(-1/tau) * 65536
	retention := uint32(math.Round(math.Exp(-1.0/float64(tau)) * 65536))

	result := int32(amplitude)
	for i := uint32(0); i < dt; i++ {
		result = (result * int32(retention)) >> 16
		if result == 0 {
			return 0
		}
	}

	return int16(result)
}

// OnSpikePropagation evaluates pre-before-post timing. If the
// post-synaptic neuron has fired recently (within the STDP window),
// this is anti-causal: post fired before pre, so we add negative
// eligibility (candidate for weakening).
//
// Wait — that's backwards. Let me be precise:
// - Pre fires now, post fired at postLastFired.
// - If postLastFired < preFiredAt: post fired before pre → anti-causal → weaken
// - dt = preFiredAt - postLastFired
func (s *STDPRule) OnSpikePropagation(conn *Connection, preFiredAt, postLastFired uint32) {
	if postLastFired == 0 || postLastFired >= preFiredAt {
		return // Post hasn't fired, or post fired after pre (handled in OnPostFire)
	}

	// Anti-causal: post fired before pre → depression
	dt := preFiredAt - postLastFired
	delta := stdpWindow(dt, s.Config.AMinus, s.Config.TauMinus)
	if delta != 0 {
		conn.Eligibility = clampAdd(conn.Eligibility, -delta)
	}
}

// OnPostFire evaluates post-before-pre timing for all incoming
// connections. For each incoming connection whose source has fired
// recently (within the STDP window), this is causal: pre fired
// before post → strengthen.
func (s *STDPRule) OnPostFire(incoming []IncomingConnection, postFiredAt uint32) {
	for _, in := range incoming {
		if in.Conn == nil {
			continue
		}

		// SourceIndex has been repurposed to hold the source neuron's
		// LastFired tick (see Network.getIncomingConnections).
		// We use a special sentinel: if the source has HasFired=false
		// (LastFired==0 and we haven't set it), we skip. But tick 0 is
		// a valid fire time, so we use the convention that
		// getIncomingConnections adds 1 to LastFired, and we subtract
		// it here. See getIncomingConnections.
		encodedFiredAt := in.SourceIndex
		if encodedFiredAt == 0 {
			continue // Source has never fired
		}
		preFiredAt := encodedFiredAt - 1 // Decode: subtract the +1 offset

		if preFiredAt >= postFiredAt {
			continue // Not causal
		}

		dt := postFiredAt - preFiredAt
		delta := stdpWindow(dt, s.Config.APlus, s.Config.TauPlus)
		if delta != 0 {
			in.Conn.Eligibility = clampAdd(in.Conn.Eligibility, delta)
		}
	}
}

// OnReward consolidates eligibility traces into actual weight
// changes. Called when a global reward/punishment signal arrives.
//
// ΔW = reward × eligibility (scaled down to prevent huge jumps)
func (s *STDPRule) OnReward(net *Network, reward int16, tick uint32) {
	if reward == 0 {
		return
	}

	maxMag := s.Config.MaxWeightMagnitude
	if maxMag == 0 {
		maxMag = MaxWeight
	}

	for i := range net.Neurons {
		for j := range net.Neurons[i].Connections {
			conn := &net.Neurons[i].Connections[j]
			if conn.Eligibility == 0 {
				continue
			}

			// Scale: (reward * eligibility) >> 8 to keep changes small
			delta := (int32(reward) * int32(conn.Eligibility)) >> 8
			if delta > int32(MaxWeight) {
				delta = int32(MaxWeight)
			}
			if delta < int32(MinWeight) {
				delta = int32(MinWeight)
			}

			conn.Weight = clampAdd(conn.Weight, int16(delta))

			// Clamp to max magnitude if configured
			if maxMag > 0 && maxMag < MaxWeight {
				if conn.Weight > maxMag {
					conn.Weight = maxMag
				}
				if conn.Weight < -maxMag {
					conn.Weight = -maxMag
				}
			}

			// Clear eligibility after consolidation
			conn.Eligibility = 0
		}
	}
}

// Maintain decays all eligibility traces across the network.
// Called once per tick.
func (s *STDPRule) Maintain(net *Network, tick uint32) {
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

			// Same decay math as neuron activation decay
			decayed := (int32(conn.Eligibility) * int32(rate)) >> 16
			conn.Eligibility = int16(decayed)
		}
	}
}
