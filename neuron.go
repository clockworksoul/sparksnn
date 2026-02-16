// Package biomimetic implements a sparse, event-driven neural network
// architecture using biologically-inspired neurons with integer arithmetic.
package biomimetic

import "math"

const (
	// MaxActivation is the maximum activation level a neuron can reach.
	MaxActivation int16 = math.MaxInt16 // 32767

	// MinActivation is the minimum activation level a neuron can reach.
	MinActivation int16 = math.MinInt16 // -32768

	// MaxWeight is the maximum connection weight.
	MaxWeight int16 = math.MaxInt16

	// MinWeight is the minimum connection weight.
	MinWeight int16 = math.MinInt16
)

// Connection represents a directed synaptic connection from one neuron
// to another, with a signed weight (positive = excitatory, negative =
// inhibitory). Weights are clamped to int16 range — no overflow.
type Connection struct {
	// Target is the index of the target neuron in the network's neuron
	// array. Using uint32 indices instead of pointers for cache locality,
	// serialization, and memory savings (4 bytes vs 8).
	Target uint32

	// Weight is the signed connection strength. Positive values are
	// excitatory; negative values are inhibitory. Clamped to
	// [MinWeight, MaxWeight].
	Weight int16

	// Eligibility is the current eligibility trace for this connection.
	// Set by STDP timing rules, decayed each tick, and consolidated
	// into weight changes when a reward signal arrives. Positive =
	// candidate for strengthening, negative = candidate for weakening.
	// Zero when no learning activity is pending.
	Eligibility int16
}

// Neuron represents a single biomimetic neuron. It is a simple data
// structure — not a goroutine or an object with methods that manage
// its own lifecycle. The Network drives all neuron interactions.
//
// This is a leaky integrate-and-fire model: activation decays toward
// baseline over time, input weights are summed directly into the
// activation level, and firing occurs when activation exceeds the
// threshold (subject to refractory period).
type Neuron struct {
	// Activation is the current activation level. Clamped to
	// [MinActivation, MaxActivation]. Decays toward Baseline when
	// the neuron is not being stimulated.
	Activation int16

	// Baseline is the resting activation level. Activation decays
	// toward this value over time. Same for all neurons in a network
	// (but stored per-neuron for flexibility).
	Baseline int16

	// Threshold is the activation level that triggers firing.
	// When Activation >= Threshold and the refractory period has
	// elapsed, the neuron fires and propagates to its connections.
	Threshold int16

	// LastInteraction is the counter value when this neuron was last
	// stimulated. Used for lazy decay calculation — idle neurons cost
	// zero compute.
	LastInteraction uint32

	// DecayRate controls how quickly activation decays toward baseline.
	// Expressed as a fixed-point fraction of 65536:
	//   64000 = ~97.7% retention per tick (slow decay)
	//   58982 = ~90% retention per tick
	//   32768 = 50% retention per tick (fast decay)
	// Initialized to a network-wide default but can be tuned per-neuron
	// to model different neuron types (e.g., fast-spiking interneurons
	// vs slow pyramidal cells).
	DecayRate uint16

	// RefractoryUntil is the counter value until which the neuron
	// cannot fire. Prevents runaway cascading and models the
	// biological refractory period.
	RefractoryUntil uint32

	// LastFired is the counter value when this neuron last fired.
	// Used by learning rules (e.g., STDP) to calculate spike timing
	// between pre- and post-synaptic neurons. Zero if never fired.
	LastFired uint32

	// Connections are the outgoing synaptic connections to other
	// neurons. Each connection has a target index and a signed weight.
	Connections []Connection
}

// clampAdd adds a (possibly negative) int16 value to a base int16,
// clamping the result to [MinActivation, MaxActivation] instead of
// wrapping on overflow.
func clampAdd(base, delta int16) int16 {
	sum := int32(base) + int32(delta)
	if sum > int32(MaxActivation) {
		return MaxActivation
	}
	if sum < int32(MinActivation) {
		return MinActivation
	}
	return int16(sum)
}

// decay calculates how far activation has moved back toward baseline
// since the last interaction. Uses an integer approximation of
// exponential decay: each elapsed tick, the distance from baseline
// shrinks by a fraction determined by the decay rate.
//
// The decay is lazy — it's only calculated when the neuron is next
// stimulated, so idle neurons cost nothing.
func (n *Neuron) decay(now uint32) {
	elapsed := now - n.LastInteraction
	if elapsed == 0 {
		return
	}

	distance := int32(n.Activation) - int32(n.Baseline)
	if distance == 0 {
		return
	}

	// Integer exponential decay approximation.
	// For each elapsed tick, multiply distance by (decayRate / 65536).
	// A decayRate of 64000 means ~97.7% retention per tick (slow decay).
	// A decayRate of 32768 means 50% retention per tick (fast decay).
	//
	// For large elapsed values, we cap to avoid excessive looping.
	// After ~20 ticks at 50% retention, the value is negligible.
	if elapsed > 32 {
		n.Activation = n.Baseline
		n.LastInteraction = now
		return
	}

	for i := uint32(0); i < elapsed; i++ {
		distance = (distance * int32(n.DecayRate)) >> 16
		if distance == 0 {
			break
		}
	}

	n.Activation = int16(int32(n.Baseline) + distance)
	n.LastInteraction = now
}

// Stimulate applies an incoming signal to the neuron at the given
// counter time. It performs the full activation cycle:
//
//  1. Decay — calculate activation drift toward baseline since last
//     interaction (lazy; idle neurons cost nothing).
//  2. Summation — add the incoming weight to the activation level.
//  3. Threshold check — if activation >= threshold and refractory
//     period has elapsed, fire.
//
// Returns true if the neuron fired, false otherwise.
func (n *Neuron) Stimulate(weight int16, now uint32) bool {
	// Step 1: Decay
	n.decay(now)

	// Step 2: Summation (clamped)
	n.Activation = clampAdd(n.Activation, weight)

	// Step 3: Threshold check + refractory period
	if n.Activation >= n.Threshold && now >= n.RefractoryUntil {
		return true
	}

	return false
}
