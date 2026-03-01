package surrogate

import (
	"math"

	bio "github.com/clockworksoul/biomimetic-network"
)

// LayerSpec describes a contiguous range of neuron indices in a layer.
type LayerSpec struct {
	Start uint32
	End   uint32
}

// Config holds training hyperparameters.
type Config struct {
	// LearningRate is the step size for weight updates.
	LearningRate float64

	// NumSteps is the number of simulation timesteps per sample.
	NumSteps int

	// Surrogate is the surrogate gradient function.
	Surrogate Gradient

	// Layers describes the network topology from input to output.
	// Must be in order: [input, hidden..., output].
	Layers []LayerSpec

	// Beta is the membrane decay factor as a float64 in [0, 1).
	// Derived from the network's DecayRate: beta = DecayRate / 65536.
	Beta float64

	// InputWeight is the stimulation weight for input encoding.
	InputWeight float64

	// Threshold is the firing threshold as float64.
	Threshold float64
}

// Trainer implements surrogate gradient training for a biomimetic
// network. It maintains float64 shadow weights and performs BPTT
// with surrogate gradients to train the spiking network.
type Trainer struct {
	Config Config
	Net    *bio.Network

	// weights maps (srcIdx, connIdx) to the float64 shadow weight.
	// These are the "true" weights; int32 network weights are quantized
	// from these after each update.
	weights [][]float64

	// connections stores the topology: connections[srcIdx][connIdx] = targetIdx
	connections [][]uint32

	// weightScale converts float64 weights to int32: int32 = float64 * weightScale
	weightScale float64
}

// NewTrainer creates a trainer for the given network.
// It initializes float64 shadow weights from the network's int32 weights,
// dividing by weightScale to convert from int32 to float64 domain.
// Use weightScale=1.0 if the network already uses float-friendly weights.
func NewTrainer(net *bio.Network, cfg Config, weightScale float64) *Trainer {
	if weightScale == 0 {
		weightScale = 1.0
	}

	t := &Trainer{
		Config:      cfg,
		Net:         net,
		weights:     make([][]float64, len(net.Neurons)),
		connections: make([][]uint32, len(net.Neurons)),
		weightScale: weightScale,
	}

	for i := range net.Neurons {
		n := &net.Neurons[i]
		t.weights[i] = make([]float64, len(n.Connections))
		t.connections[i] = make([]uint32, len(n.Connections))
		for j, conn := range n.Connections {
			t.weights[i][j] = float64(conn.Weight) / weightScale
			t.connections[i][j] = conn.Target
		}
	}

	return t
}

// WeightScale is the factor used to convert between float64 training
// weights and int32 network weights. Set by NewTrainer.
func (t *Trainer) SetWeightScale(s float64) { t.weightScale = s }

// syncWeightsToNetwork quantizes float64 shadow weights back to int32
// and writes them to the network, multiplying by weightScale.
func (t *Trainer) syncWeightsToNetwork() {
	scale := t.weightScale
	if scale == 0 {
		scale = 1.0
	}
	for i := range t.Net.Neurons {
		for j := range t.Net.Neurons[i].Connections {
			w := t.weights[i][j] * scale
			if w > math.MaxInt32 {
				w = math.MaxInt32
			} else if w < math.MinInt32 {
				w = math.MinInt32
			}
			t.Net.Neurons[i].Connections[j].Weight = int32(math.Round(w))
		}
	}
}

// neuronTrace holds the recorded state of a single neuron across time.
type neuronTrace struct {
	u []float64 // membrane potential before spike check
	s []float64 // spike output (0 or 1)
}

// TrainSample performs one training step on a single sample.
// Returns the loss value.
//
// inputValues are the float64 input activations (one per input neuron).
// These are presented at every timestep (deterministic input).
func (t *Trainer) TrainSample(inputValues []float64, correctClass int) float64 {
	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	// ===== FORWARD PASS =====
	// Simulate the network in float64, recording traces.

	// Initialize membrane potentials
	mem := make([]float64, numNeurons)
	// All start at baseline (0)

	// Traces for BPTT
	traces := make([]neuronTrace, numNeurons)
	for i := range traces {
		traces[i].u = make([]float64, numSteps)
		traces[i].s = make([]float64, numSteps)
	}

	// Spike counts for output neurons (for loss)
	spikeCounts := make([]float64, numOutputs)

	for step := 0; step < numSteps; step++ {
		// Compute input currents for all neurons
		current := make([]float64, numNeurons)

		// External input: stimulate input neurons
		inputLayer := cfg.Layers[0]
		for i := inputLayer.Start; i < inputLayer.End; i++ {
			idx := int(i - inputLayer.Start)
			if idx < len(inputValues) {
				current[i] += inputValues[idx] * cfg.InputWeight
			}
		}

		// Synaptic input: spikes from previous step propagate
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traces[src].s[step-1] == 0 {
					continue // no spike from this neuron last step
				}
				for j, tgt := range t.connections[src] {
					current[tgt] += t.weights[src][j]
				}
			}
		}

		// Update membrane potentials and check for spikes
		for i := 0; i < numNeurons; i++ {
			// LIF dynamics: decay + input
			mem[i] = cfg.Beta*mem[i] + current[i]

			// Record pre-spike membrane potential
			traces[i].u[step] = mem[i]

			// Spike check (Heaviside)
			if mem[i] >= cfg.Threshold {
				traces[i].s[step] = 1.0

				// Track output spikes
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}

				// Reset: subtract threshold (soft reset)
				mem[i] -= cfg.Threshold
			} else {
				traces[i].s[step] = 0.0
			}
		}
	}

	// ===== COMPUTE LOSS =====
	loss, dLdCounts := SpikeCountCrossEntropy(spikeCounts, correctClass)

	// ===== BACKWARD PASS (BPTT) =====
	// We need gradients for weights. The chain:
	// L depends on spikeCounts
	// spikeCounts = sum over t of S_output[t]
	// S[t] = Heaviside(U[t] - threshold) → surrogate: dS/dU
	// U[t] = beta * U[t-1] + sum(W * S_pre[t-1]) + I_ext
	//
	// So: dL/dW_ij = sum_t dL/dS_j[t] * dS_j/dU_j[t] * S_i[t-1]
	//   + temporal terms from U[t] depending on U[t-1]

	// dL/dU[t] for each neuron at each timestep
	dLdU := make([][]float64, numNeurons)
	for i := range dLdU {
		dLdU[i] = make([]float64, numSteps)
	}

	// Weight gradients
	dLdW := make([][]float64, len(t.weights))
	for i := range dLdW {
		dLdW[i] = make([]float64, len(t.weights[i]))
	}

	// Backward through time
	for step := numSteps - 1; step >= 0; step-- {

		// For output neurons: dL/dS comes from the spike count loss
		for oi := 0; oi < numOutputs; oi++ {
			nIdx := int(outputStart) + oi
			// Each spike at any timestep contributes equally to the count
			// So dL/dS[t] = dL/dCount for the output neuron
			surr := cfg.Surrogate.Derivative(traces[nIdx].u[step], cfg.Threshold)
			dLdU[nIdx][step] += dLdCounts[oi] * surr
		}

		// Temporal gradient: dL/dU[t] propagates to dL/dU[t-1] via beta
		// U[t] = beta * U[t-1] + ... → dU[t]/dU[t-1] = beta
		// But only if the neuron didn't spike (reset breaks the chain)
		if step > 0 {
			for i := 0; i < numNeurons; i++ {
				if traces[i].s[step] == 0 {
					// No spike → membrane carries forward → gradient flows back
					dLdU[i][step-1] += cfg.Beta * dLdU[i][step]
				}
				// If spiked: reset breaks the temporal chain (detached)
			}
		}

		// Synaptic gradient: dL/dW_ij += dL/dU_j[t] * S_i[t-1]
		// (pre-synaptic spike at t-1 arrives as current at t)
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traces[src].s[step-1] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					dLdW[src][j] += dLdU[tgt][step]
				}
			}
		}

		// Propagate gradient through synapses to pre-synaptic neurons:
		// dL/dS_i[t-1] += sum_j W_ij * dL/dU_j[t]
		// Then: dL/dU_i[t-1] += dL/dS_i[t-1] * surrogate(U_i[t-1])
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traces[src].s[step-1] == 0 {
					continue // only need gradient if neuron spiked
				}
				var dLdS float64
				for j, tgt := range t.connections[src] {
					dLdS += t.weights[src][j] * dLdU[tgt][step]
				}
				surr := cfg.Surrogate.Derivative(traces[src].u[step-1], cfg.Threshold)
				dLdU[src][step-1] += dLdS * surr
			}
		}
	}

	// Also accumulate gradients from external input at step 0
	// (input neurons don't have incoming learned connections,
	// so nothing to do here for weights)

	// ===== WEIGHT UPDATE (SGD) =====
	lr := cfg.LearningRate
	for src := range t.weights {
		for j := range t.weights[src] {
			t.weights[src][j] -= lr * dLdW[src][j]
		}
	}

	// Sync float64 weights back to the int32 network
	t.syncWeightsToNetwork()

	// Reset network activations for next sample
	t.Net.ResetActivation()

	return loss
}
