// Package arbiter implements a biologically-inspired error-signaling
// learning rule using dedicated "arbiter neurons" — analogous to
// cerebellar climbing fibers that fire on errors to depress recently
// active pathways.
//
// The mechanism:
//   - Forward neurons learn via STDP: pre-before-post timing
//     strengthens connections automatically (free positive signal).
//   - Arbiter neurons fire when the network's output is WRONG.
//   - An arbiter spike DEPRESSES (weakens) recently-active connections
//     in the layer it supervises — anti-Hebbian error correction.
//   - When arbiters are silent, STDP proceeds normally.
//
// This is biological backpropagation: the error signal travels
// backward through dedicated infrastructure, not through the same
// connections that carry forward computation.
//
// Reference: research/arbiter-neurons.md
package arbiter

import (
	bio "github.com/clockworksoul/sparksnn"
)

// LayerSpec describes a single layer in the network and its
// associated arbiter neurons.
type LayerSpec struct {
	// Start is the index of the first neuron in this layer.
	Start uint32

	// End is one past the last neuron in this layer (exclusive).
	End uint32

	// ArbiterStart is the index of the first arbiter neuron for
	// this layer. Zero means no arbiters (e.g., input layer).
	ArbiterStart uint32

	// ArbiterEnd is one past the last arbiter neuron (exclusive).
	// Zero means no arbiters.
	ArbiterEnd uint32
}

// Config holds tunable parameters for the arbiter learning rule.
type Config struct {
	// --- STDP (positive reinforcement) ---

	// APlus is the maximum weight increase for causal timing
	// (pre fires before post). Applied directly to weights.
	APlus int32

	// AMinus is the maximum weight decrease for anti-causal
	// timing (post fires before pre). Applied directly to weights.
	AMinus int32

	// TauPlus is the time constant (in ticks) for the causal
	// STDP window.
	TauPlus uint32

	// TauMinus is the time constant for the anti-causal window.
	TauMinus uint32

	// --- Arbiter (error suppression) ---

	// DepressionStrength controls how much an arbiter spike
	// weakens recently-active connections. Higher = stronger
	// error correction.
	DepressionStrength int32

	// ArbiterWindow is how many ticks back from an arbiter spike
	// to look for recently-active connections to depress.
	ArbiterWindow uint32

	// --- Strengthening ---

	// StrengtheningRatio is how much stronger the "strengthen
	// correct" signal is relative to DepressionStrength.
	// E.g., 2.0 means correct outputs get 2x the weight delta
	// that wrong outputs lose. This compensates for the fact that
	// there are typically more wrong outputs than correct ones.
	// 0 = same as depression strength (1.0).
	StrengtheningRatio float64

	// --- Multiplicative mode ---

	// Multiplicative switches from fixed deltas to proportional
	// weight changes. When true, DepressionStrength and
	// StrengtheningRatio are ignored, and CorrectionRate is used
	// instead. Weight changes are a fraction of current weight:
	//
	//   depression:    weight -= |weight| * CorrectionRate
	//   strengthening: weight += |weight| * CorrectionRate * StrengtheningRatio
	//
	// This makes strong connections resilient and weak connections
	// volatile. Weights asymptotically approach zero but never
	// reach it — no need for MinWeightMagnitude.
	Multiplicative bool

	// CorrectionRate is the fractional weight change per error
	// when Multiplicative is true. E.g., 0.20 = 20% of current
	// weight magnitude. Ignored when Multiplicative is false.
	CorrectionRate float64

	// --- Weight bounds ---

	// MaxWeightMagnitude caps the absolute value of weights.
	// 0 = no cap.
	MaxWeightMagnitude int32

	// MinWeightMagnitude is the floor for weight magnitude after
	// depression. Prevents weights from being driven to zero and
	// killed. 0 = no floor.
	MinWeightMagnitude int32
}

// DefaultConfig returns reasonable starting parameters.
func DefaultConfig() Config {
	return Config{
		APlus:              50,
		AMinus:             25,
		TauPlus:            8,
		TauMinus:           8,
		DepressionStrength:  80,
		ArbiterWindow:       15,
		StrengtheningRatio:  2.0,
		MaxWeightMagnitude:  0,
		MinWeightMagnitude:  0,
	}
}

// Rule implements the arbiter learning rule.
type Rule struct {
	Config Config

	// Layers describes the network topology. The rule uses this
	// to know which neurons are forward neurons vs arbiters, and
	// which arbiter layer supervises which forward layer.
	Layers []LayerSpec

	// outputStart/outputEnd define the output layer for error
	// computation.
	outputStart uint32
	outputEnd   uint32

	// pendingError is set by SignalError and consumed by Maintain.
	// It contains the indices of arbiter neurons that should fire.
	pendingError []uint32
}

// NewRule creates an arbiter learning rule.
//
// layers describes the forward layers (input, hidden..., output).
// Each layer except the input layer can have arbiter neurons.
// The output layer is always the last entry in layers.
func NewRule(config Config, layers []LayerSpec) *Rule {
	r := &Rule{
		Config: config,
		Layers: layers,
	}

	if len(layers) > 0 {
		last := layers[len(layers)-1]
		r.outputStart = last.Start
		r.outputEnd = last.End
	}

	return r
}

// SignalError is called after presenting a sample to indicate which
// output was correct. It applies targeted error correction:
//
//   - Connections to WRONG output neurons that fired are DEPRESSED
//   - Connections to the CORRECT output neuron are STRENGTHENED
//     (proportional to how little it fired vs the winner)
//   - Hidden neurons that fired for wrong outputs get their incoming
//     connections depressed
//
// This is the key training signal. Call it once per sample, after
// PresentSample and before the next sample.
//
// correctClass is the index of the correct output neuron within the
// output layer (0-indexed relative to the output layer start).
// spikeCounts are the spike counts for each output neuron.
func (r *Rule) SignalError(net *bio.Network, correctClass int, spikeCounts []int) {
	// Determine if the network was wrong
	predicted := -1
	bestCount := 0
	for i, c := range spikeCounts {
		if c > bestCount {
			bestCount = c
			predicted = i
		}
	}

	if predicted == correctClass && bestCount > 0 {
		// Correct! Arbiters stay silent.
		return
	}

	// Apply targeted error correction
	r.applyTargetedCorrection(net, correctClass, spikeCounts)
}

// StrengthenActive strengthens connections that led to the correct
// output. This is Phase 1 of three-phase training: targeted
// reinforcement of pathways to the correct class.
//
// Only modulates connections where BOTH source and target fired
// (causal Hebbian constraint). Connections that didn't result in
// activation are left untouched.
//
// Two layers of strengthening:
//  1. Hidden → correct output: strengthen if hidden neuron fired
//     AND the correct output fired.
//  2. Input → helpful hidden: strengthen if input neuron fired AND
//     the hidden neuron fired, and the hidden neuron is biased
//     toward the correct output.
//
// correctClass is the index of the correct output neuron within the
// output layer. spikeCounts are the spike counts for each output.
// Returns true if strengthening was applied (correct prediction).
func (r *Rule) StrengthenActive(net *bio.Network, correctClass int, spikeCounts []int) bool {
	// Only strengthen on correct predictions
	predicted := -1
	bestCount := 0
	for i, c := range spikeCounts {
		if c > bestCount {
			bestCount = c
			predicted = i
		}
	}

	if predicted != correctClass || bestCount == 0 {
		return false
	}

	tick := net.Counter
	window := r.Config.ArbiterWindow
	correctOutput := r.outputStart + uint32(correctClass)

	strength := r.Config.DepressionStrength
	ratio := r.Config.StrengtheningRatio
	if ratio <= 0 {
		ratio = 1.0
	}
	strengthenDelta := int32(float64(strength) * ratio)
	if strengthenDelta == 0 {
		strengthenDelta = 1
	}

	mult := r.Config.Multiplicative
	rate := r.Config.CorrectionRate

	// Correct output must have fired for any strengthening to apply
	correctNeuron := &net.Neurons[correctOutput]
	if correctNeuron.LastFired == 0 || tick-correctNeuron.LastFired > window {
		return true // correct prediction but output didn't fire in window — odd, skip
	}

	// Layer 1: Hidden → correct output
	// Only strengthen if the source (hidden) neuron also fired.
	for _, layer := range r.Layers {
		for idx := layer.Start; idx < layer.End; idx++ {
			if idx >= r.outputStart && idx < r.outputEnd {
				continue
			}

			n := &net.Neurons[idx]
			if n.LastFired == 0 || tick-n.LastFired > window {
				continue // source didn't fire — skip
			}

			for j := range n.Connections {
				conn := &n.Connections[j]
				if conn.Target != correctOutput {
					continue
				}

				// Both source and target fired — causal connection
				if mult {
					delta := r.multDelta(conn.Weight, rate*ratio)
					conn.Weight = bio.ClampAdd(conn.Weight, delta)
				} else {
					conn.Weight = bio.ClampAdd(conn.Weight, strengthenDelta)
				}
				r.clampWeight(conn)
			}
		}
	}

	// Layer 2: Input → Hidden
	// Only strengthen if: input fired, hidden fired, AND hidden is
	// biased toward the correct output.
	for _, layer := range r.Layers {
		if layer.ArbiterStart == 0 && layer.ArbiterEnd == 0 {
			continue
		}

		for hidIdx := layer.Start; hidIdx < layer.End; hidIdx++ {
			hidNeuron := &net.Neurons[hidIdx]
			if hidNeuron.LastFired == 0 || tick-hidNeuron.LastFired > window {
				continue // hidden didn't fire — skip
			}

			var wrongScore, correctScore int64
			for _, conn := range hidNeuron.Connections {
				if conn.Target < r.outputStart || conn.Target >= r.outputEnd {
					continue
				}
				if conn.Target == correctOutput {
					correctScore += int64(conn.Weight)
				} else {
					wrongScore += int64(conn.Weight)
				}
			}

			if correctScore <= wrongScore {
				continue
			}

			thirdStrength := strengthenDelta / 3
			if thirdStrength == 0 {
				thirdStrength = 1
			}
			upstreamRate := rate * ratio / 3.0

			for i := range net.Neurons {
				srcNeuron := &net.Neurons[i]
				srcIdx := uint32(i)

				if srcIdx >= layer.Start && srcIdx < layer.End {
					continue
				}
				// Source must have fired
				if srcNeuron.LastFired == 0 || tick-srcNeuron.LastFired > window {
					continue
				}

				for j := range srcNeuron.Connections {
					conn := &srcNeuron.Connections[j]
					if conn.Target == hidIdx {
						// Both source (input) and target (hidden) fired
						if mult {
							delta := r.multDelta(conn.Weight, upstreamRate)
							conn.Weight = bio.ClampAdd(conn.Weight, delta)
						} else {
							conn.Weight = bio.ClampAdd(conn.Weight, thirdStrength)
						}
						r.clampWeight(conn)
					}
				}
			}
		}
	}

	return true
}

// CorrectErrors weakens connections that contributed to incorrect
// outputs AND strengthens connections to the correct output. This is
// Phase 2 of three-phase training: arbiter-driven error correction.
//
// Both signals are applied on every wrong prediction so that the
// correct pathway gets reinforcement even when the network is wrong.
// Returns true if corrections were applied (i.e., prediction was wrong).
func (r *Rule) CorrectErrors(net *bio.Network, correctClass int, spikeCounts []int) bool {
	// If correct, nothing to correct
	predicted := -1
	bestCount := 0
	for i, c := range spikeCounts {
		if c > bestCount {
			bestCount = c
			predicted = i
		}
	}

	if predicted == correctClass && bestCount > 0 {
		return false
	}

	r.applyErrorDepression(net, correctClass, spikeCounts)
	r.strengthenCorrectOnError(net, correctClass)
	return true
}

// strengthenCorrectOnError strengthens connections leading to the
// correct output even when the network predicted wrong. This ensures
// the correct pathway gets positive signal on every sample, not just
// correct predictions.
//
// Only modulates connections where both source and target fired
// (causal Hebbian constraint). Note: the correct output may not
// have fired on a wrong prediction — if so, we strengthen connections
// from active hidden neurons anyway (they tried to activate it).
func (r *Rule) strengthenCorrectOnError(net *bio.Network, correctClass int) {
	tick := net.Counter
	window := r.Config.ArbiterWindow
	correctOutput := r.outputStart + uint32(correctClass)

	strength := r.Config.DepressionStrength
	ratio := r.Config.StrengtheningRatio
	if ratio <= 0 {
		ratio = 1.0
	}
	strengthenDelta := int32(float64(strength) * ratio)
	if strengthenDelta == 0 {
		strengthenDelta = 1
	}

	mult := r.Config.Multiplicative
	rate := r.Config.CorrectionRate

	// Check if the correct output fired — if it did, only strengthen
	// causal connections. If it didn't fire at all, strengthen from
	// any active source (to encourage it to fire next time).
	correctFired := false
	correctNeuron := &net.Neurons[correctOutput]
	if correctNeuron.LastFired > 0 && tick-correctNeuron.LastFired <= window {
		correctFired = true
	}

	// Strengthen Hidden → correct output connections
	for _, layer := range r.Layers {
		for idx := layer.Start; idx < layer.End; idx++ {
			if idx >= r.outputStart && idx < r.outputEnd {
				continue
			}

			n := &net.Neurons[idx]
			if n.LastFired == 0 || tick-n.LastFired > window {
				continue // source didn't fire — can't be causal
			}

			// If correct output fired, only strengthen from neurons
			// that also fired (causal). If it didn't fire, strengthen
			// from any active neuron (bootstrap signal).
			if correctFired {
				// Both fired — causal, full strength
			}
			// If !correctFired, source fired but target didn't —
			// still strengthen to encourage correct output to fire

			for j := range n.Connections {
				conn := &n.Connections[j]
				if conn.Target != correctOutput {
					continue
				}

				if mult {
					delta := r.multDelta(conn.Weight, rate*ratio)
					conn.Weight = bio.ClampAdd(conn.Weight, delta)
				} else {
					conn.Weight = bio.ClampAdd(conn.Weight, strengthenDelta)
				}
				r.clampWeight(conn)
			}
		}
	}
}

// isArbiter returns true if the neuron index belongs to an arbiter layer.
func (r *Rule) isArbiter(idx uint32) bool {
	for _, layer := range r.Layers {
		if layer.ArbiterStart > 0 && idx >= layer.ArbiterStart && idx < layer.ArbiterEnd {
			return true
		}
	}
	return false
}

// applyErrorDepression depresses connections that led to wrong outputs,
// WITHOUT strengthening correct ones. Used by CorrectErrors (Phase 2).
//
// Only modulates connections where both source and target fired
// (causal Hebbian constraint). A connection from a silent neuron
// didn't contribute to the wrong output, so it's left alone.
func (r *Rule) applyErrorDepression(net *bio.Network, correctClass int, spikeCounts []int) {
	tick := net.Counter
	window := r.Config.ArbiterWindow
	strength := r.Config.DepressionStrength
	correctOutput := r.outputStart + uint32(correctClass)

	mult := r.Config.Multiplicative
	rate := r.Config.CorrectionRate

	// Hidden → Output: depress connections where source fired AND
	// wrong output target fired.
	for _, layer := range r.Layers {
		for idx := layer.Start; idx < layer.End; idx++ {
			if idx >= r.outputStart && idx < r.outputEnd {
				continue
			}

			n := &net.Neurons[idx]
			// Source must have fired
			if n.LastFired == 0 || tick-n.LastFired > window {
				continue
			}

			for j := range n.Connections {
				conn := &n.Connections[j]
				if conn.Target < r.outputStart || conn.Target >= r.outputEnd {
					continue
				}

				outputIdx := int(conn.Target - r.outputStart)

				// Target must be a wrong output that actually fired
				if conn.Target == correctOutput || spikeCounts[outputIdx] == 0 {
					continue
				}

				// Verify target neuron fired within window (causal check)
				targetNeuron := &net.Neurons[conn.Target]
				if targetNeuron.LastFired == 0 || tick-targetNeuron.LastFired > window {
					continue
				}

				// Both source and target fired — this connection
				// causally contributed to the wrong output
				if mult {
					delta := r.multDelta(conn.Weight, rate)
					conn.Weight = bio.ClampAdd(conn.Weight, -delta)
				} else {
					conn.Weight = bio.ClampAdd(conn.Weight, -strength)
				}
				r.clampWeight(conn)
				r.floorWeight(conn)
			}
		}
	}

	// Input → Hidden: depress connections where input fired AND
	// hidden neuron fired AND hidden is biased toward wrong outputs.
	for _, layer := range r.Layers {
		if layer.ArbiterStart == 0 && layer.ArbiterEnd == 0 {
			continue
		}

		for hidIdx := layer.Start; hidIdx < layer.End; hidIdx++ {
			hidNeuron := &net.Neurons[hidIdx]
			// Target (hidden) must have fired
			if hidNeuron.LastFired == 0 || tick-hidNeuron.LastFired > window {
				continue
			}

			var wrongScore, correctScore int64
			for _, conn := range hidNeuron.Connections {
				if conn.Target < r.outputStart || conn.Target >= r.outputEnd {
					continue
				}
				if conn.Target == correctOutput {
					correctScore += int64(conn.Weight)
				} else {
					wrongScore += int64(conn.Weight)
				}
			}

			if wrongScore <= correctScore {
				continue
			}

			halfStrength := strength / 3
			if halfStrength == 0 {
				halfStrength = 1
			}
			upstreamRate := rate / 3.0

			for i := range net.Neurons {
				srcNeuron := &net.Neurons[i]
				srcIdx := uint32(i)

				if srcIdx >= layer.Start && srcIdx < layer.End {
					continue
				}
				// Source must have fired
				if srcNeuron.LastFired == 0 || tick-srcNeuron.LastFired > window {
					continue
				}

				for j := range srcNeuron.Connections {
					conn := &srcNeuron.Connections[j]
					if conn.Target == hidIdx {
						// Both source (input) and target (hidden) fired
						if mult {
							delta := r.multDelta(conn.Weight, upstreamRate)
							conn.Weight = bio.ClampAdd(conn.Weight, -delta)
						} else {
							conn.Weight = bio.ClampAdd(conn.Weight, -halfStrength)
						}
						r.clampWeight(conn)
						r.floorWeight(conn)
					}
				}
			}
		}
	}
}

// applyTargetedCorrection applies class-selective error correction:
//
//  1. Hidden→Output: depress connections to wrong outputs that fired,
//     strengthen connections to the correct output.
//  2. Input→Hidden: depress connections to hidden neurons that were
//     most active for wrong outputs (using connection weights as proxy).
//
// This is more selective than blanket depression — it targets the
// specific pathways that led to the wrong answer.
func (r *Rule) applyTargetedCorrection(net *bio.Network, correctClass int, spikeCounts []int) {
	tick := net.Counter
	window := r.Config.ArbiterWindow
	strength := r.Config.DepressionStrength
	correctOutput := r.outputStart + uint32(correctClass)

	// Strengthening is scaled relative to depression
	ratio := r.Config.StrengtheningRatio
	if ratio <= 0 {
		ratio = 1.0
	}
	strengthenDelta := int32(float64(strength) * ratio)
	if strengthenDelta == 0 {
		strengthenDelta = strength
	}

	// Multiplicative mode: compute deltas as fraction of weight
	mult := r.Config.Multiplicative
	rate := r.Config.CorrectionRate

	// Phase 1: Hidden → Output connections
	// Depress connections to wrong outputs, strengthen to correct output
	for _, layer := range r.Layers {
		// Find the layer that feeds the output (last non-output layer)
		for idx := layer.Start; idx < layer.End; idx++ {
			// Skip if this IS the output layer
			if idx >= r.outputStart && idx < r.outputEnd {
				continue
			}

			n := &net.Neurons[idx]

			// Was this neuron recently active?
			if n.LastFired == 0 || tick-n.LastFired > window {
				continue
			}

			for j := range n.Connections {
				conn := &n.Connections[j]
				if conn.Target < r.outputStart || conn.Target >= r.outputEnd {
					continue
				}

				outputIdx := int(conn.Target - r.outputStart)

				if conn.Target == correctOutput {
					// Strengthen connection to correct output
					if mult {
						delta := r.multDelta(conn.Weight, rate*ratio)
						conn.Weight = bio.ClampAdd(conn.Weight, delta)
					} else {
						conn.Weight = bio.ClampAdd(conn.Weight, strengthenDelta)
					}
				} else if spikeCounts[outputIdx] > 0 {
					// Depress connection to wrong output that fired
					if mult {
						delta := r.multDelta(conn.Weight, rate)
						conn.Weight = bio.ClampAdd(conn.Weight, -delta)
					} else {
						conn.Weight = bio.ClampAdd(conn.Weight, -strength)
					}
				}

				r.clampWeight(conn)
				r.floorWeight(conn)
			}
		}
	}

	// Phase 2: Input → Hidden connections
	// For each hidden neuron that fired, check if it predominantly
	// connects to wrong outputs. If so, depress its incoming connections.
	for _, layer := range r.Layers {
		if layer.ArbiterStart == 0 && layer.ArbiterEnd == 0 {
			continue
		}

		for hidIdx := layer.Start; hidIdx < layer.End; hidIdx++ {
			hidNeuron := &net.Neurons[hidIdx]
			if hidNeuron.LastFired == 0 || tick-hidNeuron.LastFired > window {
				continue
			}

			// Score this hidden neuron: does it connect more strongly
			// to wrong outputs or the correct output?
			var wrongScore, correctScore int64
			for _, conn := range hidNeuron.Connections {
				if conn.Target < r.outputStart || conn.Target >= r.outputEnd {
					continue
				}
				if conn.Target == correctOutput {
					correctScore += int64(conn.Weight)
				} else {
					wrongScore += int64(conn.Weight)
				}
			}

			if wrongScore <= correctScore {
				continue // This neuron is fine, skip
			}

			// This hidden neuron is biased toward wrong outputs.
			// Depress incoming connections (scaled down — gentler than
			// the output correction).
			halfStrength := strength / 3
			if halfStrength == 0 {
				halfStrength = 1
			}
			upstreamRate := rate / 3.0

			// Find neurons that connect TO this hidden neuron
			for i := range net.Neurons {
				srcNeuron := &net.Neurons[i]
				srcIdx := uint32(i)

				// Only depress from the layer before this one
				if srcIdx >= layer.Start && srcIdx < layer.End {
					continue
				}
				if srcNeuron.LastFired == 0 || tick-srcNeuron.LastFired > window {
					continue
				}

				for j := range srcNeuron.Connections {
					conn := &srcNeuron.Connections[j]
					if conn.Target == hidIdx {
						if mult {
							delta := r.multDelta(conn.Weight, upstreamRate)
							conn.Weight = bio.ClampAdd(conn.Weight, -delta)
						} else {
							conn.Weight = bio.ClampAdd(conn.Weight, -halfStrength)
						}
						r.clampWeight(conn)
						r.floorWeight(conn)
					}
				}
			}
		}
	}
}

// multDelta computes a proportional weight change: |weight| * rate,
// with a minimum of 1 so that even tiny weights can still move.
// Always returns a positive value — caller decides the sign.
func (r *Rule) multDelta(weight int32, rate float64) int32 {
	absW := int64(weight)
	if absW < 0 {
		absW = -absW
	}
	// Minimum magnitude so near-zero weights can still grow
	if absW < 10 {
		absW = 10
	}
	delta := int32(float64(absW) * rate)
	if delta < 1 {
		delta = 1
	}
	return delta
}

// floorWeight enforces MinWeightMagnitude bounds.
func (r *Rule) floorWeight(conn *bio.Connection) {
	minMag := r.Config.MinWeightMagnitude
	if minMag <= 0 {
		return
	}
	if conn.Weight > 0 && conn.Weight < minMag {
		conn.Weight = minMag
	} else if conn.Weight < 0 && conn.Weight > -minMag {
		conn.Weight = -minMag
	}
}

// stdpWindow calculates the exponentially-decayed magnitude for a
// given time difference and time constant.
func stdpWindow(dt uint32, amplitude int32, tau uint32) int32 {
	if tau == 0 || dt > tau*6 {
		return 0
	}

	// Simple integer approximation: amplitude * (tau-dt) / tau
	// for dt <= tau, 0 otherwise. This is a linear approximation
	// that avoids the exponential loop.
	if dt >= tau {
		return 0
	}

	return int32(int64(amplitude) * int64(tau-dt) / int64(tau))
}

// OnSpikePropagation handles pre-before-post STDP timing.
// Anti-causal: post fired recently before pre → weaken.
func (r *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	if postLastFired == 0 || postLastFired >= preFiredAt {
		return
	}

	dt := preFiredAt - postLastFired
	delta := stdpWindow(dt, r.Config.AMinus, r.Config.TauMinus)
	if delta != 0 {
		conn.Weight = bio.ClampAdd(conn.Weight, -delta)
		r.clampWeight(conn)
	}
}

// OnPostFire handles post-after-pre STDP timing.
// Causal: pre fired recently before post → strengthen.
func (r *Rule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
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
		delta := stdpWindow(dt, r.Config.APlus, r.Config.TauPlus)
		if delta != 0 {
			in.Conn.Weight = bio.ClampAdd(in.Conn.Weight, delta)
			r.clampWeight(in.Conn)
		}
	}
}

// OnReward is a no-op for the arbiter rule. Error signals are
// delivered through SignalError, not the global reward channel.
func (r *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {}

// Maintain is called once per tick. Currently a no-op — the arbiter
// rule applies changes immediately rather than accumulating traces.
func (r *Rule) Maintain(net *bio.Network, tick uint32) {}

// clampWeight enforces MaxWeightMagnitude bounds.
func (r *Rule) clampWeight(conn *bio.Connection) {
	maxMag := r.Config.MaxWeightMagnitude
	if maxMag <= 0 {
		return
	}
	if conn.Weight > maxMag {
		conn.Weight = maxMag
	}
	if conn.Weight < -maxMag {
		conn.Weight = -maxMag
	}
}
