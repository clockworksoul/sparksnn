package surrogate

import (
	"math"

	bio "github.com/clockworksoul/sparksnn"
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

	// Adam optimizer state
	useAdam bool
	adam    adamState

	// Pre-allocated buffers for TrainSample, reused across calls to
	// avoid per-sample heap allocations. Allocated once in NewTrainer
	// and zeroed between samples via resetBuffers.
	buf trainBuffers
}

// trainBuffers holds pre-allocated working memory for TrainSample.
// All slices are allocated once and zeroed between samples.
type trainBuffers struct {
	initialized bool

	// Forward pass
	mem         []float64   // [numNeurons] membrane potentials
	traceU      []float64   // [numNeurons * numSteps] flat: trace[i][t] = traceU[i*numSteps+t]
	traceS      []float64   // [numNeurons * numSteps] flat: spike[i][t] = traceS[i*numSteps+t]
	spikeCounts []float64   // [numOutputs]
	current     []float64   // [numNeurons] reused each timestep

	// Backward pass
	dLdU []float64   // [numNeurons * numSteps] flat
	dLdW [][]float64 // [numSrc][numConns] mirrors weights topology

	// Dimensions (cached for bounds)
	numNeurons int
	numSteps   int
	numOutputs int
}

// adamState holds per-weight Adam optimizer moments.
type adamState struct {
	m [][]float64 // first moment (mean of gradients)
	v [][]float64 // second moment (mean of squared gradients)
	t int         // timestep counter

	beta1   float64
	beta2   float64
	epsilon float64
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

	t.initBuffers()

	return t
}

// initBuffers allocates the working buffers used by TrainSample.
// Called once from NewTrainer; buffers are reused across all samples.
func (t *Trainer) initBuffers() {
	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)

	t.buf = trainBuffers{
		initialized: true,
		mem:         make([]float64, numNeurons),
		traceU:      make([]float64, numNeurons*numSteps),
		traceS:      make([]float64, numNeurons*numSteps),
		spikeCounts: make([]float64, numOutputs),
		current:     make([]float64, numNeurons),
		dLdU:        make([]float64, numNeurons*numSteps),
		dLdW:        make([][]float64, len(t.weights)),
		numNeurons:  numNeurons,
		numSteps:    numSteps,
		numOutputs:  numOutputs,
	}
	for i := range t.weights {
		t.buf.dLdW[i] = make([]float64, len(t.weights[i]))
	}
}

// resetBuffers zeroes all pre-allocated buffers between samples.
func (t *Trainer) resetBuffers() {
	clear(t.buf.mem)
	clear(t.buf.traceU)
	clear(t.buf.traceS)
	clear(t.buf.spikeCounts)
	clear(t.buf.current)
	clear(t.buf.dLdU)
	for i := range t.buf.dLdW {
		clear(t.buf.dLdW[i])
	}
}

// EnableAdam activates the Adam optimizer with standard defaults
// (beta1=0.9, beta2=0.999, epsilon=1e-8).
func (t *Trainer) EnableAdam() {
	t.useAdam = true
	t.adam = adamState{
		m:       make([][]float64, len(t.weights)),
		v:       make([][]float64, len(t.weights)),
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-8,
	}
	for i := range t.weights {
		t.adam.m[i] = make([]float64, len(t.weights[i]))
		t.adam.v[i] = make([]float64, len(t.weights[i]))
	}
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

// TrainSample performs one training step on a single sample.
// Returns the loss value.
//
// inputValues are the float64 input activations (one per input neuron).
// These are presented at every timestep (deterministic input).
//
// All working memory is pre-allocated on the Trainer and reused across
// calls, eliminating per-sample heap allocations.
func (t *Trainer) TrainSample(inputValues []float64, correctClass int) float64 {
	cfg := t.Config
	numNeurons := t.buf.numNeurons
	numSteps := t.buf.numSteps
	numOutputs := t.buf.numOutputs
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	// Zero all buffers from previous sample
	t.resetBuffers()

	// Aliases for readability
	mem := t.buf.mem
	traceU := t.buf.traceU
	traceS := t.buf.traceS
	spikeCounts := t.buf.spikeCounts
	current := t.buf.current
	dLdU := t.buf.dLdU
	dLdW := t.buf.dLdW

	// ===== FORWARD PASS =====
	// Simulate the network in float64, recording traces.

	for step := 0; step < numSteps; step++ {
		// Zero current for this timestep
		clear(current)

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
				if traceS[src*numSteps+step-1] == 0 {
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
			traceU[i*numSteps+step] = mem[i]

			// Spike check (Heaviside)
			if mem[i] >= cfg.Threshold {
				traceS[i*numSteps+step] = 1.0

				// Track output spikes
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}

				// Reset: subtract threshold (soft reset)
				mem[i] -= cfg.Threshold
			}
			// traceS already zeroed by resetBuffers
		}
	}

	// ===== COMPUTE LOSS =====
	loss, dLdCounts := SpikeCountCrossEntropy(spikeCounts, correctClass)

	// ===== BACKWARD PASS (BPTT) =====
	// dLdU and dLdW already zeroed by resetBuffers

	// Backward through time
	for step := numSteps - 1; step >= 0; step-- {

		// For output neurons: dL/dS comes from the spike count loss
		for oi := 0; oi < numOutputs; oi++ {
			nIdx := int(outputStart) + oi
			surr := cfg.Surrogate.Derivative(traceU[nIdx*numSteps+step], cfg.Threshold)
			dLdU[nIdx*numSteps+step] += dLdCounts[oi] * surr
		}

		// Temporal gradient: dL/dU[t] propagates to dL/dU[t-1] via beta
		if step > 0 {
			for i := 0; i < numNeurons; i++ {
				if traceS[i*numSteps+step] == 0 {
					// No spike → membrane carries forward → gradient flows back
					dLdU[i*numSteps+step-1] += cfg.Beta * dLdU[i*numSteps+step]
				}
				// If spiked: reset breaks the temporal chain (detached)
			}
		}

		// Synaptic gradient: dL/dW_ij += dL/dU_j[t] * S_i[t-1]
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traceS[src*numSteps+step-1] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					dLdW[src][j] += dLdU[int(tgt)*numSteps+step]
				}
			}
		}

		// Propagate gradient through synapses to pre-synaptic neurons
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traceS[src*numSteps+step-1] == 0 {
					continue
				}
				var dLdS float64
				for j, tgt := range t.connections[src] {
					dLdS += t.weights[src][j] * dLdU[int(tgt)*numSteps+step]
				}
				surr := cfg.Surrogate.Derivative(traceU[src*numSteps+step-1], cfg.Threshold)
				dLdU[src*numSteps+step-1] += dLdS * surr
			}
		}
	}

	// ===== WEIGHT UPDATE =====
	lr := cfg.LearningRate
	if t.useAdam {
		t.adam.t++
		tt := float64(t.adam.t)
		b1, b2, eps := t.adam.beta1, t.adam.beta2, t.adam.epsilon
		bc1 := 1.0 - math.Pow(b1, tt) // bias correction
		bc2 := 1.0 - math.Pow(b2, tt)
		for src := range t.weights {
			for j := range t.weights[src] {
				g := dLdW[src][j]
				t.adam.m[src][j] = b1*t.adam.m[src][j] + (1-b1)*g
				t.adam.v[src][j] = b2*t.adam.v[src][j] + (1-b2)*g*g
				mHat := t.adam.m[src][j] / bc1
				vHat := t.adam.v[src][j] / bc2
				t.weights[src][j] -= lr * mHat / (math.Sqrt(vHat) + eps)
			}
		}
	} else {
		for src := range t.weights {
			for j := range t.weights[src] {
				t.weights[src][j] -= lr * dLdW[src][j]
			}
		}
	}

	// Sync float64 weights back to the int32 network
	t.syncWeightsToNetwork()

	// Reset network activations for next sample
	t.Net.ResetActivation()

	return loss
}

// ActivityStats holds per-timestep activity measurements from a forward pass.
type ActivityStats struct {
	// ActivePerStep[t] is the number of neurons that were active at timestep t
	// (received input or fired).
	ActivePerStep []int

	// SpikedPerStep[t] is the number of neurons that fired at timestep t.
	SpikedPerStep []int

	// TotalNeurons is the total neuron count.
	TotalNeurons int

	// NumSteps is the number of timesteps.
	NumSteps int
}

// MeanActivityRate returns the average fraction of neurons active per timestep.
func (s *ActivityStats) MeanActivityRate() float64 {
	if s.NumSteps == 0 || s.TotalNeurons == 0 {
		return 0
	}
	total := 0
	for _, a := range s.ActivePerStep {
		total += a
	}
	return float64(total) / float64(s.NumSteps) / float64(s.TotalNeurons)
}

// MeanSpikeRate returns the average fraction of neurons spiking per timestep.
func (s *ActivityStats) MeanSpikeRate() float64 {
	if s.NumSteps == 0 || s.TotalNeurons == 0 {
		return 0
	}
	total := 0
	for _, a := range s.SpikedPerStep {
		total += a
	}
	return float64(total) / float64(s.NumSteps) / float64(s.TotalNeurons)
}

// PredictWithStats runs a forward pass and returns the predicted class
// plus detailed activity statistics for energy analysis.
func (t *Trainer) PredictWithStats(inputValues []float64) (int, ActivityStats) {
	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	mem := make([]float64, numNeurons)
	spikes := make([]float64, numNeurons)
	spikeCounts := make([]float64, numOutputs)

	stats := ActivityStats{
		ActivePerStep: make([]int, numSteps),
		SpikedPerStep: make([]int, numSteps),
		TotalNeurons:  numNeurons,
		NumSteps:      numSteps,
	}

	for step := 0; step < numSteps; step++ {
		// Track which neurons are touched this step (received input or fired)
		active := make([]bool, numNeurons)
		current := make([]float64, numNeurons)

		// External input
		inputLayer := cfg.Layers[0]
		for i := inputLayer.Start; i < inputLayer.End; i++ {
			idx := int(i - inputLayer.Start)
			if idx < len(inputValues) && inputValues[idx] > 0.01 {
				current[i] += inputValues[idx] * cfg.InputWeight
				active[i] = true
			}
		}

		// Synaptic input from previous step's spikes
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if spikes[src] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					current[tgt] += t.weights[src][j]
					active[tgt] = true
				}
			}
		}

		// Reset spikes for this step
		for i := range spikes {
			spikes[i] = 0
		}

		// Update membrane and check spikes
		for i := 0; i < numNeurons; i++ {
			mem[i] = cfg.Beta*mem[i] + current[i]
			if mem[i] >= cfg.Threshold {
				spikes[i] = 1.0
				active[i] = true
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}
				mem[i] -= cfg.Threshold
			}
		}

		// Count activity
		activeCount := 0
		spikeCount := 0
		for i := 0; i < numNeurons; i++ {
			if active[i] {
				activeCount++
			}
			if spikes[i] == 1.0 {
				spikeCount++
			}
		}
		stats.ActivePerStep[step] = activeCount
		stats.SpikedPerStep[step] = spikeCount
	}

	bestClass := 0
	bestCount := spikeCounts[0]
	for i := 1; i < numOutputs; i++ {
		if spikeCounts[i] > bestCount {
			bestCount = spikeCounts[i]
			bestClass = i
		}
	}
	return bestClass, stats
}

// Predict runs a forward pass without training and returns the
// predicted class (highest spike count). Uses float64 simulation.
func (t *Trainer) Predict(inputValues []float64) int {
	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	mem := make([]float64, numNeurons)
	spikes := make([]float64, numNeurons) // current step spikes
	spikeCounts := make([]float64, numOutputs)

	for step := 0; step < numSteps; step++ {
		current := make([]float64, numNeurons)

		// External input
		inputLayer := cfg.Layers[0]
		for i := inputLayer.Start; i < inputLayer.End; i++ {
			idx := int(i - inputLayer.Start)
			if idx < len(inputValues) {
				current[i] += inputValues[idx] * cfg.InputWeight
			}
		}

		// Synaptic input from previous step
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if spikes[src] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					current[tgt] += t.weights[src][j]
				}
			}
		}

		// Reset spikes for this step
		for i := range spikes {
			spikes[i] = 0
		}

		// Update membrane and check spikes
		for i := 0; i < numNeurons; i++ {
			mem[i] = cfg.Beta*mem[i] + current[i]
			if mem[i] >= cfg.Threshold {
				spikes[i] = 1.0
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}
				mem[i] -= cfg.Threshold
			}
		}
	}

	// Return class with highest spike count
	bestClass := 0
	bestCount := spikeCounts[0]
	for i := 1; i < numOutputs; i++ {
		if spikeCounts[i] > bestCount {
			bestCount = spikeCounts[i]
			bestClass = i
		}
	}
	return bestClass
}

// PredictPerTimestep runs inference and returns the predicted class at each
// timestep. The returned slice has length NumSteps, where result[t] is the
// class with the highest cumulative spike count after timestep t.
// This is used for early-exit analysis: determining the minimum number of
// timesteps needed for correct classification.
func (t *Trainer) PredictPerTimestep(inputValues []float64) []int {
	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	mem := make([]float64, numNeurons)
	spikes := make([]float64, numNeurons)
	spikeCounts := make([]float64, numOutputs)
	predictions := make([]int, numSteps)

	for step := 0; step < numSteps; step++ {
		current := make([]float64, numNeurons)

		// External input
		inputLayer := cfg.Layers[0]
		for i := inputLayer.Start; i < inputLayer.End; i++ {
			idx := int(i - inputLayer.Start)
			if idx < len(inputValues) {
				current[i] += inputValues[idx] * cfg.InputWeight
			}
		}

		// Synaptic input from previous step
		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if spikes[src] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					current[tgt] += t.weights[src][j]
				}
			}
		}

		// Reset spikes for this step
		for i := range spikes {
			spikes[i] = 0
		}

		// Update membrane and check spikes
		for i := 0; i < numNeurons; i++ {
			mem[i] = cfg.Beta*mem[i] + current[i]
			if mem[i] >= cfg.Threshold {
				spikes[i] = 1.0
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}
				mem[i] -= cfg.Threshold
			}
		}

		// Record prediction at this timestep
		bestClass := 0
		bestCount := spikeCounts[0]
		for c := 1; c < numOutputs; c++ {
			if spikeCounts[c] > bestCount {
				bestCount = spikeCounts[c]
				bestClass = c
			}
		}
		predictions[step] = bestClass
	}

	return predictions
}
