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
	samples := make([]benchmark.Sample, 0, len(base)*1000)
	for i := 0; i < 1000; i++ {
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
	InputWeight int32

	// InhibWeight is the lateral inhibition weight (negative).
	InhibWeight int32

	// Threshold for hidden and output neurons.
	Threshold int32

	// DecayRate for all neurons (fixed-point, /65536).
	DecayRate uint16

	// RefractoryPeriod in ticks.
	RefractoryPeriod uint32

	// InitWeightMax is the upper bound for random initial weights
	// on learnable connections.
	InitWeightMax int32

	// InitialDensity controls what fraction of possible learnable
	// connections are created at startup. 1.0 = fully connected,
	// 0.25 = each possible connection has 25% chance of existing.
	// Lower values leave room for structural plasticity to discover
	// the right topology.
	InitialDensity float64

	// UseInhibition enables lateral inhibition (excitatory→inhibitory→
	// all-other-excitatory). When false, the network is simpler:
	// just input→hidden→output with no inhibitory interneurons.
	UseInhibition bool

	// NoiseProbability is the per-tick probability that a hidden
	// neuron receives a random excitatory "miniature EPSP" (spontaneous
	// background activity). Prevents dead neurons, enables exploration.
	// 0 = no noise, 0.05 = 5% chance per neuron per tick.
	NoiseProbability float64

	// NoiseWeight is the stimulation weight for noise events.
	NoiseWeight int32

	// DeterministicInput uses fixed stimulation every tick instead
	// of probabilistic rate coding. Cleaner signal for small problems.
	DeterministicInput bool
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
		InitWeightMax:    500,  // avg ~250, well above threshold — single inputs can fire hidden neurons
		InitialDensity:   1.0,  // fully connected by default (backward compat)
		UseInhibition:    true, // lateral inhibition by default (backward compat)
		NoiseProbability: 0.02, // 2% chance per hidden neuron per tick — subtle background activity
		NoiseWeight:      100,  // sub-threshold nudge, needs accumulation to fire
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
	numInhib := 0
	if cfg.UseInhibition {
		numInhib = cfg.HiddenSize
	}
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

	// Input → Hidden (learnable, random positive weights, density-controlled)
	// Start positive so hidden neurons are alive. Learning rules will
	// push some weights negative to create the inhibitory connections
	// needed for XOR (e.g., "A AND NOT B" requires B→H1 to go negative).
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			if cfg.InitialDensity < 1.0 && rand.Float64() > cfg.InitialDensity {
				continue // skip this connection
			}
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(i, h, w)
		}
	}

	// Lateral inhibition (optional)
	if cfg.UseInhibition {
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
	}

	// Hidden → Output (learnable, random positive weights, density-controlled)
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			if cfg.InitialDensity < 1.0 && rand.Float64() > cfg.InitialDensity {
				continue
			}
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
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
		// Stimulate input neurons
		for i, val := range sample.Inputs {
			if cfg.DeterministicInput {
				if val > 0 {
					net.Stimulate(layout.InputStart+uint32(i), cfg.InputWeight)
				}
			} else {
				if val > 0 && rand.IntN(256) < int(val) {
					net.Stimulate(layout.InputStart+uint32(i), cfg.InputWeight)
				}
			}
		}

		// Spontaneous background noise on hidden neurons (miniature EPSPs)
		if cfg.NoiseProbability > 0 {
			for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
				if rand.Float64() < cfg.NoiseProbability {
					net.Stimulate(h, cfg.NoiseWeight)
				}
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
func CollectWeights(net *bio.Network, layout Layout) []int32 {
	var weights []int32

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
	totalPruned, totalGrown := 0, 0

	for i, sample := range trainSamples {
		PresentSample(net, layout, sample, cfg)

		// Trigger structural plasticity after each sample
		p, g := net.Remodel()
		totalPruned += p
		totalGrown += g

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
	fmt.Fprintf(os.Stderr, "[%s] Structural changes: %d pruned, %d grown\n",
		ruleName, totalPruned, totalGrown)
	return tracker
}
