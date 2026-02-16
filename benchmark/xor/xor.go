// Package xor provides an XOR classification task for benchmarking
// learning rules. XOR is the simplest non-linearly-separable problem:
// a network must learn that (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.
//
// This serves as a smoke test — if a learning rule can't learn XOR,
// it won't learn MNIST.
package xor

import (
	"fmt"
	"math/rand/v2"
	"os"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/benchmark"
)

// Task implements the XOR benchmark.
type Task struct{}

func (Task) Name() string       { return "XOR" }
func (Task) NumInputs() int     { return 2 }
func (Task) NumClasses() int    { return 2 }

// TrainingSamples returns the 4 XOR patterns, repeated to provide
// enough training signal. Each pattern appears 250 times (1000 total).
func (Task) TrainingSamples() []benchmark.Sample {
	base := xorPatterns()
	samples := make([]benchmark.Sample, 0, len(base)*250)
	for i := 0; i < 250; i++ {
		samples = append(samples, base...)
	}
	return samples
}

// TestSamples returns the 4 XOR patterns (no repetition needed
// for evaluation).
func (Task) TestSamples() []benchmark.Sample {
	return xorPatterns()
}

func xorPatterns() []benchmark.Sample {
	return []benchmark.Sample{
		{Inputs: []byte{0, 0}, Label: 0},
		{Inputs: []byte{0, 255}, Label: 1},
		{Inputs: []byte{255, 0}, Label: 1},
		{Inputs: []byte{255, 255}, Label: 0},
	}
}

// NetworkConfig holds the XOR network topology parameters.
type NetworkConfig struct {
	// HiddenSize is the number of excitatory hidden neurons.
	HiddenSize int

	// TicksPerSample is how many ticks to present each input.
	TicksPerSample int

	// RestTicks is how many idle ticks between samples.
	RestTicks int

	// InputWeight is the stimulation weight for active inputs.
	InputWeight int16

	// InhibWeight is the lateral inhibition weight (negative).
	InhibWeight int16

	// Threshold for hidden and output neurons.
	Threshold int16

	// DecayRate for all neurons (fixed-point, /65536).
	DecayRate uint16

	// RefractoryPeriod in ticks.
	RefractoryPeriod uint32

	// InitWeightMax is the upper bound for random initial weights
	// on learnable connections.
	InitWeightMax int16
}

// DefaultConfig returns reasonable defaults for XOR.
func DefaultConfig() NetworkConfig {
	return NetworkConfig{
		HiddenSize:       20,
		TicksPerSample:   100,
		RestTicks:        50,
		InputWeight:      300,
		InhibWeight:      -2000,
		Threshold:        150,
		DecayRate:        45000, // ~69% retention — faster decay
		RefractoryPeriod: 5,
		InitWeightMax:    150,
	}
}

// Layout describes the neuron index ranges in the network.
type Layout struct {
	InputStart  uint32 // First input neuron index
	InputEnd    uint32 // One past last input neuron
	HiddenStart uint32 // First excitatory hidden neuron
	HiddenEnd   uint32 // One past last excitatory hidden
	InhibStart  uint32 // First inhibitory neuron
	InhibEnd    uint32 // One past last inhibitory
	OutputStart uint32 // First output neuron
	OutputEnd   uint32 // One past last output neuron
}

// BuildNetwork creates an XOR network with the given learning rule.
//
// Topology:
//
//	2 input neurons
//	  ↓ fully connected (learnable weights)
//	N excitatory hidden neurons
//	  ↔ N inhibitory neurons (lateral inhibition, fixed weights)
//	  ↓ fully connected (learnable weights)
//	2 output neurons (class 0, class 1)
func BuildNetwork(cfg NetworkConfig, rule bio.LearningRule) (*bio.Network, Layout) {
	numInput := 2
	numHidden := cfg.HiddenSize
	numInhib := cfg.HiddenSize
	numOutput := 2
	total := numInput + numHidden + numInhib + numOutput

	net := bio.NewNetwork(uint32(total), 0, cfg.Threshold, cfg.DecayRate, cfg.RefractoryPeriod)
	net.LearningRule = rule

	layout := Layout{
		InputStart:  0,
		InputEnd:    uint32(numInput),
		HiddenStart: uint32(numInput),
		HiddenEnd:   uint32(numInput + numHidden),
		InhibStart:  uint32(numInput + numHidden),
		InhibEnd:    uint32(numInput + numHidden + numInhib),
		OutputStart: uint32(numInput + numHidden + numInhib),
		OutputEnd:   uint32(total),
	}

	// Input → Hidden (learnable, random positive weights)
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			w := int16(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(i, h, w)
		}
	}

	// Hidden excitatory → paired inhibitory (fixed, strong)
	for i := 0; i < numHidden; i++ {
		excit := layout.HiddenStart + uint32(i)
		inhib := layout.InhibStart + uint32(i)
		net.Connect(excit, inhib, 500) // strong enough to always fire
	}

	// Inhibitory → all OTHER excitatory (lateral inhibition, fixed)
	for i := 0; i < numInhib; i++ {
		inhib := layout.InhibStart + uint32(i)
		for j := 0; j < numHidden; j++ {
			if i == j {
				continue // don't inhibit own pair
			}
			excit := layout.HiddenStart + uint32(j)
			net.Connect(inhib, excit, cfg.InhibWeight)
		}
	}

	// Hidden → Output (learnable, random positive weights)
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			w := int16(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(h, o, w)
		}
	}

	return net, layout
}

// PresentSample rate-encodes a sample and presents it to the network.
// Returns the spike counts for each output neuron.
func PresentSample(net *bio.Network, layout Layout, sample benchmark.Sample, cfg NetworkConfig) []int {
	numOutputs := int(layout.OutputEnd - layout.OutputStart)
	spikeCounts := make([]int, numOutputs)

	// Present phase: rate-encode inputs over TicksPerSample ticks
	for tick := 0; tick < cfg.TicksPerSample; tick++ {
		// Stimulate input neurons probabilistically
		for i, val := range sample.Inputs {
			if val > 0 && rand.IntN(256) < int(val) {
				net.Stimulate(layout.InputStart+uint32(i), cfg.InputWeight)
			}
		}

		net.Tick()

		// Count output spikes
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			if net.Neurons[o].LastFired == net.Counter {
				spikeCounts[o-layout.OutputStart]++
			}
		}
	}

	// Rest phase: let activity die down
	net.TickN(uint32(cfg.RestTicks))

	return spikeCounts
}

// Classify returns the predicted class based on output spike counts.
// Returns -1 if no output neuron fired (tie at zero).
func Classify(spikeCounts []int) int {
	bestClass := -1
	bestCount := 0
	for i, count := range spikeCounts {
		if count > bestCount {
			bestCount = count
			bestClass = i
		}
	}
	return bestClass
}

// EvalTrials is the number of times each test pattern is presented
// during evaluation. Averaging over multiple trials reduces noise
// from stochastic rate coding.
const EvalTrials = 5

// Evaluate runs all test samples and returns the accuracy (0.0 to 1.0).
// Each test sample is presented EvalTrials times; spike counts are
// summed across trials before classification.
// Also returns the number of hidden neurons that never fired (dead)
// and the mean spike rate per hidden neuron per sample.
func Evaluate(net *bio.Network, layout Layout, task benchmark.Task, cfg NetworkConfig) (accuracy float64, dead int, spikeRate float64) {
	testSamples := task.TestSamples()
	correct := 0
	totalHiddenSpikes := 0
	hiddenFired := make(map[uint32]bool)
	numOutputs := int(layout.OutputEnd - layout.OutputStart)

	for _, sample := range testSamples {
		// Aggregate spike counts over multiple trials
		aggSpikes := make([]int, numOutputs)

		for trial := 0; trial < EvalTrials; trial++ {
			// Record hidden LastFired before
			hiddenLastFired := make([]uint32, layout.HiddenEnd-layout.HiddenStart)
			for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
				hiddenLastFired[h-layout.HiddenStart] = net.Neurons[h].LastFired
			}

			spikeCounts := PresentSample(net, layout, sample, cfg)

			for i, c := range spikeCounts {
				aggSpikes[i] += c
			}

			// Count hidden neuron spikes
			for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
				if net.Neurons[h].LastFired > hiddenLastFired[h-layout.HiddenStart] {
					hiddenFired[h] = true
					totalHiddenSpikes++
				}
			}
		}

		predicted := Classify(aggSpikes)
		if predicted == sample.Label {
			correct++
		}
	}

	numHidden := int(layout.HiddenEnd - layout.HiddenStart)
	dead = numHidden - len(hiddenFired)

	totalTrials := len(testSamples) * EvalTrials
	accuracy = float64(correct) / float64(len(testSamples))
	if totalTrials > 0 && numHidden > 0 {
		spikeRate = float64(totalHiddenSpikes) / float64(totalTrials)
	}

	return accuracy, dead, spikeRate
}

// CollectWeights gathers all learnable connection weights from
// the network (input→hidden and hidden→output connections).
func CollectWeights(net *bio.Network, layout Layout) []int16 {
	var weights []int16

	// Input → Hidden weights
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for _, conn := range net.Neurons[i].Connections {
			if conn.Target >= layout.HiddenStart && conn.Target < layout.HiddenEnd {
				weights = append(weights, conn.Weight)
			}
		}
	}

	// Hidden → Output weights
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for _, conn := range net.Neurons[h].Connections {
			if conn.Target >= layout.OutputStart && conn.Target < layout.OutputEnd {
				weights = append(weights, conn.Weight)
			}
		}
	}

	return weights
}

// Run executes the full XOR benchmark for a given learning rule.
// It trains the network, periodically evaluates, and reports results.
func Run(rule bio.LearningRule, ruleName string, cfg NetworkConfig) *benchmark.Tracker {
	task := Task{}
	net, layout := BuildNetwork(cfg, rule)
	trainSamples := task.TrainingSamples()
	tracker := benchmark.NewTracker(10) // converge after 10 stale checkpoints

	// Evaluate before training (baseline)
	acc, dead, sr := Evaluate(net, layout, task, cfg)
	weights := CollectWeights(net, layout)
	wm, ws := benchmark.WeightStats(weights)
	tracker.Record(benchmark.Checkpoint{
		SamplesProcessed: 0,
		Accuracy:         acc,
		WeightMean:       wm,
		WeightStdDev:     ws,
		DeadNeurons:      dead,
		SpikeRate:        sr,
	})

	checkEvery := 100 // evaluate every 100 training samples

	for i, sample := range trainSamples {
		PresentSample(net, layout, sample, cfg)

		if (i+1)%checkEvery == 0 {
			acc, dead, sr := Evaluate(net, layout, task, cfg)
			weights := CollectWeights(net, layout)
			wm, ws := benchmark.WeightStats(weights)
			converged := tracker.Record(benchmark.Checkpoint{
				SamplesProcessed: i + 1,
				Accuracy:         acc,
				WeightMean:       wm,
				WeightStdDev:     ws,
				DeadNeurons:      dead,
				SpikeRate:        sr,
			})

			if converged {
				fmt.Fprintf(os.Stderr, "[%s] Converged at sample %d (acc=%.1f%%)\n",
					ruleName, i+1, acc*100)
				break
			}
		}
	}

	tracker.PrintReport(os.Stdout, task.Name(), ruleName)
	return tracker
}
