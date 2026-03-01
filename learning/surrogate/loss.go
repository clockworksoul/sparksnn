package surrogate

import "math"

// SpikeCountCrossEntropy computes cross-entropy loss from output spike
// counts. This is the simplest loss for classification: count spikes
// per output neuron over the simulation, apply softmax, compute CE
// against the one-hot target.
//
// Returns the loss value and the gradient ∂L/∂counts (one per output).
func SpikeCountCrossEntropy(spikeCounts []float64, correctClass int) (loss float64, grad []float64) {
	n := len(spikeCounts)
	grad = make([]float64, n)

	// Softmax over spike counts
	// First find max for numerical stability
	maxCount := spikeCounts[0]
	for _, c := range spikeCounts[1:] {
		if c > maxCount {
			maxCount = c
		}
	}

	expSum := 0.0
	exps := make([]float64, n)
	for i, c := range spikeCounts {
		exps[i] = math.Exp(c - maxCount)
		expSum += exps[i]
	}

	probs := make([]float64, n)
	for i := range probs {
		probs[i] = exps[i] / expSum
	}

	// Cross-entropy loss: -log(p[correctClass])
	loss = -math.Log(probs[correctClass] + 1e-10)

	// Gradient of CE w.r.t. spike counts (softmax + CE combined):
	// ∂L/∂count_i = p_i - y_i
	// where y is one-hot at correctClass
	for i := range grad {
		grad[i] = probs[i]
	}
	grad[correctClass] -= 1.0

	return loss, grad
}

// MembraneCrossEntropy computes cross-entropy loss from output membrane
// potentials at a single timestep. This is the snntorch approach:
// softmax over membrane potentials, CE against target.
//
// Returns loss and gradient ∂L/∂U for each output neuron.
func MembraneCrossEntropy(membranes []float64, correctClass int) (loss float64, grad []float64) {
	n := len(membranes)
	grad = make([]float64, n)

	maxMem := membranes[0]
	for _, m := range membranes[1:] {
		if m > maxMem {
			maxMem = m
		}
	}

	expSum := 0.0
	exps := make([]float64, n)
	for i, m := range membranes {
		exps[i] = math.Exp(m - maxMem)
		expSum += exps[i]
	}

	probs := make([]float64, n)
	for i := range probs {
		probs[i] = exps[i] / expSum
	}

	loss = -math.Log(probs[correctClass] + 1e-10)

	for i := range grad {
		grad[i] = probs[i]
	}
	grad[correctClass] -= 1.0

	return loss, grad
}
