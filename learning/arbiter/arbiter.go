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
	bio "github.com/clockworksoul/biomimetic-network"
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
					conn.Weight = bio.ClampAdd(conn.Weight, strengthenDelta)
				} else if spikeCounts[outputIdx] > 0 {
					// Depress connection to wrong output that fired
					conn.Weight = bio.ClampAdd(conn.Weight, -strength)
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
						conn.Weight = bio.ClampAdd(conn.Weight, -halfStrength)
						r.clampWeight(conn)
						r.floorWeight(conn)
					}
				}
			}
		}
	}
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
