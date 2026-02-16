package biomimetic

// PredictiveConfig holds tunable parameters for the predictive
// learning rule based on Saponati & Vinck (2023).
//
// The core idea: each neuron tries to predict its own future inputs.
// STDP-like behavior emerges as a byproduct of this prediction
// optimization. No external reward signal is needed — learning is
// entirely self-supervised.
//
// Reference: "Sequence anticipation and spike-timing-dependent
// plasticity emerge from a predictive learning rule"
// Nature Communications 14, 4985 (2023)
type PredictiveConfig struct {
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
	MaxWeightMagnitude int16

	// PredictionScale controls how the prediction v*w is scaled
	// back into int16 range. The raw product of activation (int16)
	// times weight (int16) is int32. We right-shift by this amount.
	// Default: 15 (divides by 32768, mapping the full int16×int16
	// product range back to int16).
	PredictionScale uint8
}

// DefaultPredictiveConfig returns reasonable starting parameters.
func DefaultPredictiveConfig() PredictiveConfig {
	return PredictiveConfig{
		LearningRate:         655, // ~1%
		EligibilityDecayRate: 52429, // ~80% retention per tick
		MaxWeightMagnitude:   0,
		PredictionScale:      15,
	}
}

// PredictiveRule implements the predictive learning rule from
// Saponati & Vinck (2023). Instead of explicitly programming STDP,
// each neuron optimizes a prediction objective: minimize the error
// between predicted and actual incoming signals.
//
// The rule operates in these steps:
//
//  1. When a spike arrives at a synapse, the prediction error is
//     computed: error = actual_input - prediction.
//     The prediction is derived from the post-synaptic neuron's
//     membrane potential and the connection weight.
//
//  2. The weight is updated proportionally to the prediction error
//     scaled by the post-synaptic activation (correlation term)
//     and a global error signal scaled by the eligibility trace
//     (heterosynaptic term).
//
//  3. Eligibility traces accumulate input history and decay over
//     time, providing temporal context for learning.
//
// STDP emerges because:
//   - Pre-before-post: the pre-synaptic input predicts the post
//     firing → gets credit (potentiation).
//   - Post-before-pre: the input is redundant/predicted by other
//     inputs → loses credit (depression).
type PredictiveRule struct {
	Config PredictiveConfig
}

// NewPredictiveRule creates a predictive learning rule with the
// given configuration.
func NewPredictiveRule(config PredictiveConfig) *PredictiveRule {
	return &PredictiveRule{Config: config}
}

// predict computes the predicted input for a connection given the
// post-synaptic neuron's current activation. Returns int16.
//
// prediction = (activation * weight) >> PredictionScale
//
// This maps the int16×int16 product back into int16 range.
func (p *PredictiveRule) predict(postActivation, weight int16) int16 {
	product := int32(postActivation) * int32(weight)
	shifted := product >> p.Config.PredictionScale
	// Clamp to int16
	if shifted > int32(MaxWeight) {
		return MaxWeight
	}
	if shifted < int32(MinWeight) {
		return MinWeight
	}
	return int16(shifted)
}

// OnSpikePropagation is called when a pre-synaptic neuron fires
// and delivers a signal through a connection.
//
// This is where the core predictive update happens:
//   1. Compute prediction error: actual spike arrived (magnitude =
//      weight, since that's the signal strength), but the neuron
//      predicted v_{t-1} * w. The error is the difference.
//   2. Compute correlation term: error * v_{t-1}
//   3. Update weight based on prediction error.
//
// The connection's Eligibility field is repurposed as the input-
// specific eligibility trace p_{t-1} from the paper.
func (p *PredictiveRule) OnSpikePropagation(conn *Connection, preFiredAt, postLastFired uint32) {
	// We need the post-synaptic neuron's activation to compute the
	// prediction. However, our interface only provides timing info.
	//
	// DESIGN NOTE: The predictive rule fundamentally needs access to
	// the post-neuron's activation (membrane potential). Our current
	// interface doesn't provide this. We use the Eligibility trace
	// to accumulate a proxy: each spike marks the connection as
	// recently active, and the correlation with post-synaptic state
	// is captured in OnPostFire where we have access to incoming
	// connections and their context.
	//
	// For now: mark the connection as "spike arrived" by adding to
	// the eligibility trace. The magnitude encodes spike recency.
	// OnPostFire will use this to compute the actual weight update.

	// Accumulate into eligibility: this spike just propagated
	// through this connection. Use a fixed contribution that
	// subsequent OnPostFire / Maintain calls will modulate.
	conn.Eligibility = clampAdd(conn.Eligibility, conn.Weight)
}

// OnPostFire is called when a post-synaptic neuron fires.
// This is where we compute the predictive weight update, because
// we have access to all incoming connections and can compute the
// global error signal.
//
// The update rule (from the paper, adapted to integer math):
//   w_t = w_{t-1} + η * (ε * v_{t-1} + E * p_{t-1})
//
// Where:
//   ε = per-synapse prediction error
//   E = global error signal (sum of all prediction errors weighted
//       by current weights)
//   p_{t-1} = eligibility trace (input history for this synapse)
//   v_{t-1} = post-synaptic activation at time of firing
func (p *PredictiveRule) OnPostFire(incoming []IncomingConnection, postFiredAt uint32) {
	if len(incoming) == 0 {
		return
	}

	lr := int32(p.Config.LearningRate)
	scale := p.Config.PredictionScale
	maxMag := p.Config.MaxWeightMagnitude
	if maxMag == 0 {
		maxMag = MaxWeight
	}

	// Phase 1: Compute per-synapse prediction errors and global error.
	//
	// For each incoming connection, the "actual input" at firing time
	// is reflected by whether the synapse was recently active
	// (captured in the eligibility trace). The "prediction" is what
	// the neuron expected from that synapse based on its membrane
	// state and the weight.
	//
	// Since we fire threshold-style (neuron fires when activation
	// crosses threshold), at the moment of firing the activation
	// approximates v_{t-1} + sum(inputs) ≈ threshold. We use the
	// connection's weight as a proxy for "what this synapse delivered"
	// and the eligibility as "accumulated recent input strength."

	// Compute global error signal E = Σ(ε_i * w_i)
	var globalError int32

	type synapseError struct {
		error int16
		conn  *Connection
	}
	errors := make([]synapseError, 0, len(incoming))

	for _, in := range incoming {
		if in.Conn == nil {
			continue
		}

		// Per-synapse prediction error:
		// If this synapse was recently active (eligibility != 0),
		// the actual input was "present". The prediction was based
		// on the weight. Error = actual - predicted.
		//
		// Active synapse (eligibility > 0): contributed to firing
		// Inactive synapse (eligibility == 0): did not contribute
		//
		// For active: error = weight - predicted ≈ weight - (weight * weight >> scale)
		//   Simplify: the key signal is eligibility (actual input proxy)
		//   minus the self-prediction term.
		//
		// Adaptation for integer math:
		//   actual = eligibility (accumulated weighted input)
		//   predicted = (eligibility * weight) >> scale
		//   error = actual - predicted

		actual := int32(in.Conn.Eligibility)
		predicted := (actual * int32(in.Conn.Weight)) >> scale
		err := actual - predicted

		// Clamp error to int16
		if err > int32(MaxWeight) {
			err = int32(MaxWeight)
		}
		if err < int32(MinWeight) {
			err = int32(MinWeight)
		}

		errors = append(errors, synapseError{error: int16(err), conn: in.Conn})

		// Global error += error * weight
		globalError += err * int32(in.Conn.Weight) >> scale
	}

	// Phase 2: Apply weight updates.
	// Δw = η * (ε * v_{t-1} + E * p_{t-1})
	//
	// We approximate v_{t-1} with the eligibility trace magnitude
	// (which encodes recent post-synaptic input history), and p_{t-1}
	// is the eligibility trace itself.

	for _, se := range errors {
		// Term 1: correlation term = ε * eligibility
		// (prediction error × input history ≈ ε × v_{t-1})
		term1 := int32(se.error) * int32(se.conn.Eligibility) >> scale

		// Term 2: heterosynaptic term = E * eligibility
		// (global error × input history ≈ E × p_{t-1})
		term2 := globalError * int32(se.conn.Eligibility) >> scale

		// Combined update, scaled by learning rate
		delta := (lr * (term1 + term2)) >> 16
		if delta == 0 && (term1+term2) != 0 {
			// Preserve sign for small updates
			if term1+term2 > 0 {
				delta = 1
			} else {
				delta = -1
			}
		}

		// Clamp delta
		if delta > int32(MaxWeight) {
			delta = int32(MaxWeight)
		}
		if delta < int32(MinWeight) {
			delta = int32(MinWeight)
		}

		se.conn.Weight = clampAdd(se.conn.Weight, int16(delta))

		// Clamp to max magnitude
		if maxMag > 0 && maxMag < MaxWeight {
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
// self-supervised — no external reward signal needed. However, we
// keep the method to satisfy the LearningRule interface, and a
// future hybrid rule could layer reward modulation on top.
func (p *PredictiveRule) OnReward(net *Network, reward int16, tick uint32) {
	// Intentionally empty. Predictive learning doesn't use reward.
	// A hybrid PredictiveRewardRule could combine both mechanisms.
}

// Maintain decays eligibility traces across the network, similar
// to STDP's Maintain. The eligibility trace serves as the input-
// specific history p_{t-1} from the paper.
func (p *PredictiveRule) Maintain(net *Network, tick uint32) {
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

			// Same lazy decay as activation
			decayed := (int32(conn.Eligibility) * int32(rate)) >> 16
			conn.Eligibility = int16(decayed)
		}
	}
}
