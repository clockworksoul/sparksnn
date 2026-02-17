// Package iris provides the Fisher Iris classification task for
// benchmarking learning rules. Iris has 4 continuous features
// (sepal length, sepal width, petal length, petal width) and 3
// classes (setosa, versicolor, virginica).
//
// This is a step up from XOR: more features, more classes, and
// continuous (not binary) inputs. A good test of whether a learning
// rule can generalize beyond toy boolean problems.
//
// The dataset is embedded in the binary — no external files needed.
package iris

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/benchmark"
)

// Task implements the Iris benchmark.
type Task struct {
	train []benchmark.Sample
	test  []benchmark.Sample
}

func (t *Task) Name() string       { return "Iris" }
func (t *Task) NumInputs() int     { return 4 }
func (t *Task) NumClasses() int    { return 3 }

func (t *Task) TrainingSamples() []benchmark.Sample { return t.train }
func (t *Task) TestSamples() []benchmark.Sample     { return t.test }

// NewTask creates an Iris task with a stratified 120/30 train/test
// split. The split is deterministic for a given seed.
func NewTask(seed uint64) *Task {
	samples := loadAll()

	// Group by class
	byClass := make(map[int][]benchmark.Sample)
	for _, s := range samples {
		byClass[s.Label] = append(byClass[s.Label], s)
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef))

	var train, test []benchmark.Sample
	for cls := 0; cls < 3; cls++ {
		group := byClass[cls]
		rng.Shuffle(len(group), func(i, j int) {
			group[i], group[j] = group[j], group[i]
		})
		// 40 train, 10 test per class
		train = append(train, group[:40]...)
		test = append(test, group[40:]...)
	}

	// Shuffle train set
	rng.Shuffle(len(train), func(i, j int) {
		train[i], train[j] = train[j], train[i]
	})

	return &Task{train: train, test: test}
}

// loadAll parses the embedded dataset and returns normalized samples.
// Features are min-max normalized to [0, 255] across the full dataset.
func loadAll() []benchmark.Sample {
	lines := strings.Split(strings.TrimSpace(irisCSV), "\n")

	type raw struct {
		features [4]float64
		label    int
	}

	classMap := map[string]int{
		"Iris-setosa":     0,
		"Iris-versicolor": 1,
		"Iris-virginica":  2,
	}

	var data []raw
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}

		var r raw
		for i := 0; i < 4; i++ {
			v, err := strconv.ParseFloat(parts[i], 64)
			if err != nil {
				continue
			}
			r.features[i] = v
		}

		cls, ok := classMap[parts[4]]
		if !ok {
			continue
		}
		r.label = cls
		data = append(data, r)
	}

	// Find min/max per feature
	var mins, maxs [4]float64
	for i := 0; i < 4; i++ {
		mins[i] = math.MaxFloat64
		maxs[i] = -math.MaxFloat64
	}
	for _, d := range data {
		for i := 0; i < 4; i++ {
			if d.features[i] < mins[i] {
				mins[i] = d.features[i]
			}
			if d.features[i] > maxs[i] {
				maxs[i] = d.features[i]
			}
		}
	}

	// Normalize to [0, 255]
	samples := make([]benchmark.Sample, len(data))
	for i, d := range data {
		inputs := make([]byte, 4)
		for j := 0; j < 4; j++ {
			span := maxs[j] - mins[j]
			if span == 0 {
				inputs[j] = 128
			} else {
				norm := (d.features[j] - mins[j]) / span
				inputs[j] = byte(norm * 255)
			}
		}
		samples[i] = benchmark.Sample{
			Inputs: inputs,
			Label:  d.label,
		}
	}

	return samples
}

// NetworkConfig holds the Iris network topology parameters.
type NetworkConfig struct {
	HiddenSize         int
	TicksPerSample     int
	RestTicks          int
	InputWeight        int32
	Threshold          int32
	DecayRate          uint16
	RefractoryPeriod   uint32
	InitWeightMax      int32
	NoiseProbability   float64
	NoiseWeight        int32
	DeterministicInput bool
}

// DefaultConfig returns reasonable defaults for Iris.
func DefaultConfig() NetworkConfig {
	return NetworkConfig{
		HiddenSize:         16,
		TicksPerSample:     50,
		RestTicks:          20,
		InputWeight:        500,
		Threshold:          150,
		DecayRate:          45000,
		RefractoryPeriod:   5,
		InitWeightMax:      500,
		NoiseProbability:   0.02,
		NoiseWeight:        100,
		DeterministicInput: true,
	}
}

// Layout describes the neuron index ranges in the network.
type Layout struct {
	InputStart  uint32
	InputEnd    uint32
	HiddenStart uint32
	HiddenEnd   uint32
	OutputStart uint32
	OutputEnd   uint32
}

// BuildNetwork creates an Iris network.
//
// Topology:
//
//	4 input neurons
//	  ↓ fully connected (learnable weights)
//	N hidden neurons
//	  ↓ fully connected (learnable weights)
//	3 output neurons (setosa, versicolor, virginica)
//
// No lateral inhibition for now — keep it simple and let
// perturbation do the work. We can add inhibition later if needed.
func BuildNetwork(cfg NetworkConfig, rule bio.LearningRule) (*bio.Network, Layout) {
	numInput := 4
	numHidden := cfg.HiddenSize
	numOutput := 3
	total := numInput + numHidden + numOutput

	net := bio.NewNetwork(uint32(total), 0, cfg.Threshold, cfg.DecayRate, cfg.RefractoryPeriod)
	net.LearningRule = rule

	layout := Layout{
		InputStart:  0,
		InputEnd:    uint32(numInput),
		HiddenStart: uint32(numInput),
		HiddenEnd:   uint32(numInput + numHidden),
		OutputStart: uint32(numInput + numHidden),
		OutputEnd:   uint32(total),
	}

	// Input → Hidden (learnable, random positive weights)
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(i, h, w)
		}
	}

	// Hidden → Output (learnable, random positive weights)
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(h, o, w)
		}
	}

	return net, layout
}

// PresentSample encodes a sample and presents it to the network.
// Returns spike counts for each output neuron.
func PresentSample(net *bio.Network, layout Layout, sample benchmark.Sample, cfg NetworkConfig) []int {
	numOutputs := int(layout.OutputEnd - layout.OutputStart)
	spikeCounts := make([]int, numOutputs)

	for tick := 0; tick < cfg.TicksPerSample; tick++ {
		// Stimulate input neurons proportional to feature value
		for i, val := range sample.Inputs {
			if cfg.DeterministicInput {
				// Scale stimulation weight by feature value
				if val > 0 {
					scaledWeight := int32(float64(cfg.InputWeight) * float64(val) / 255.0)
					if scaledWeight > 0 {
						net.Stimulate(layout.InputStart+uint32(i), scaledWeight)
					}
				}
			} else {
				// Rate coding: fire probability proportional to value
				if val > 0 && rand.IntN(256) < int(val) {
					net.Stimulate(layout.InputStart+uint32(i), cfg.InputWeight)
				}
			}
		}

		// Background noise on hidden neurons
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

	// Rest phase
	net.TickN(uint32(cfg.RestTicks))

	return spikeCounts
}

// Classify returns the predicted class from output spike counts.
// Returns -1 if no output neuron fired.
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

// EvalTrials is the number of presentations per test sample during
// evaluation. Averaging reduces noise.
const EvalTrials = 3

// Evaluate runs all test samples and returns accuracy, dead neuron
// count, and mean spike rate.
func Evaluate(net *bio.Network, layout Layout, task *Task, cfg NetworkConfig) (accuracy float64, dead int, spikeRate float64) {
	testSamples := task.TestSamples()
	correct := 0
	totalHiddenSpikes := 0
	hiddenFired := make(map[uint32]bool)
	numOutputs := int(layout.OutputEnd - layout.OutputStart)

	for _, sample := range testSamples {
		aggSpikes := make([]int, numOutputs)

		for trial := 0; trial < EvalTrials; trial++ {
			hiddenLastFired := make([]uint32, layout.HiddenEnd-layout.HiddenStart)
			for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
				hiddenLastFired[h-layout.HiddenStart] = net.Neurons[h].LastFired
			}

			spikeCounts := PresentSample(net, layout, sample, cfg)

			for i, c := range spikeCounts {
				aggSpikes[i] += c
			}

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

// CollectWeights gathers all learnable weights (input→hidden and
// hidden→output).
func CollectWeights(net *bio.Network, layout Layout) []int32 {
	var weights []int32

	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for _, conn := range net.Neurons[i].Connections {
			if conn.Target >= layout.HiddenStart && conn.Target < layout.HiddenEnd {
				weights = append(weights, conn.Weight)
			}
		}
	}

	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for _, conn := range net.Neurons[h].Connections {
			if conn.Target >= layout.OutputStart && conn.Target < layout.OutputEnd {
				weights = append(weights, conn.Weight)
			}
		}
	}

	return weights
}

// Run executes the full Iris benchmark with weight perturbation.
func Run(rule bio.LearningRule, ruleName string, cfg NetworkConfig) *benchmark.Tracker {
	task := NewTask(42)
	net, layout := BuildNetwork(cfg, rule)
	tracker := benchmark.NewTracker(20) // more patience for harder problem

	// Repeat training set to give perturbation enough steps
	trainSamples := task.TrainingSamples()
	var expandedTrain []benchmark.Sample
	for rep := 0; rep < 200; rep++ {
		expandedTrain = append(expandedTrain, trainSamples...)
	}

	// Baseline evaluation
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

	checkEvery := 120 // evaluate every epoch (120 training samples)

	for i, sample := range expandedTrain {
		// Present sample and evaluate for reward signal.
		// The perturbation rule accumulates reward across BatchSize
		// calls to OnReward.
		evalSpikes := PresentSample(net, layout, sample, cfg)
		predicted := Classify(evalSpikes)
		reward := int32(-1)
		if predicted == sample.Label {
			reward = 1
		}
		net.Reward(reward)

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
