package xor

import (
	"os"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/benchmark"
	"github.com/clockworksoul/biomimetic-network/learning/predictive"
	"github.com/clockworksoul/biomimetic-network/learning/rstdp"
	"github.com/clockworksoul/biomimetic-network/learning/stdp"
)

func TestTaskInterface(t *testing.T) {
	var _ benchmark.Task = Task{}
}

func TestTaskBasics(t *testing.T) {
	task := Task{}
	if task.NumInputs() != 2 {
		t.Errorf("NumInputs() = %d, want 2", task.NumInputs())
	}
	if task.NumClasses() != 2 {
		t.Errorf("NumClasses() = %d, want 2", task.NumClasses())
	}
	if len(task.TestSamples()) != 4 {
		t.Errorf("TestSamples() has %d samples, want 4", len(task.TestSamples()))
	}
	train := task.TrainingSamples()
	if len(train) == 0 || len(train)%4 != 0 {
		t.Errorf("TrainingSamples() has %d samples, want multiple of 4", len(train))
	}
}

func TestBuildNetwork(t *testing.T) {
	cfg := DefaultConfig()
	net, layout := BuildNetwork(cfg, bio.NoOpLearning{})

	totalNeurons := 2 + cfg.HiddenSize + cfg.HiddenSize + 2
	if len(net.Neurons) != totalNeurons {
		t.Errorf("network has %d neurons, want %d", len(net.Neurons), totalNeurons)
	}

	// Check input neurons have connections to all hidden
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		if len(net.Neurons[i].Connections) != cfg.HiddenSize {
			t.Errorf("input neuron %d has %d connections, want %d",
				i, len(net.Neurons[i].Connections), cfg.HiddenSize)
		}
	}

	// Check hidden neurons have connections to paired inhibitory + outputs
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		// 1 inhibitory + 2 output = 3 connections
		expected := 1 + 2
		if len(net.Neurons[h].Connections) != expected {
			t.Errorf("hidden neuron %d has %d connections, want %d",
				h, len(net.Neurons[h].Connections), expected)
		}
	}

	// Check inhibitory neurons have connections to all other excitatory
	for i := layout.InhibStart; i < layout.InhibEnd; i++ {
		expected := cfg.HiddenSize - 1 // all except own pair
		if len(net.Neurons[i].Connections) != expected {
			t.Errorf("inhibitory neuron %d has %d connections, want %d",
				i, len(net.Neurons[i].Connections), expected)
		}
	}
}

func TestPresentSample(t *testing.T) {
	cfg := DefaultConfig()
	net, layout := BuildNetwork(cfg, bio.NoOpLearning{})

	sample := benchmark.Sample{Inputs: []byte{255, 0}, Label: 1}
	spikeCounts := PresentSample(net, layout, sample, cfg)

	if len(spikeCounts) != 2 {
		t.Errorf("got %d spike counts, want 2", len(spikeCounts))
	}

	// With strong input, at least some output should fire
	total := 0
	for _, c := range spikeCounts {
		total += c
	}
	t.Logf("Output spikes for [255,0]: class0=%d, class1=%d", spikeCounts[0], spikeCounts[1])
}

func TestClassify(t *testing.T) {
	if Classify([]int{5, 3}) != 0 {
		t.Error("expected class 0 for [5,3]")
	}
	if Classify([]int{2, 7}) != 1 {
		t.Error("expected class 1 for [2,7]")
	}
	if Classify([]int{0, 0}) != -1 {
		t.Error("expected -1 for [0,0]")
	}
}

func TestEvaluateNoLearning(t *testing.T) {
	cfg := DefaultConfig()
	net, layout := BuildNetwork(cfg, bio.NoOpLearning{})

	acc, dead, sr := Evaluate(net, layout, Task{}, cfg)
	t.Logf("No-learning baseline: accuracy=%.1f%%, dead=%d, spikeRate=%.2f",
		acc*100, dead, sr)
	// No assertions on accuracy — random weights give random results
}

func TestRunPureSTDP(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark in short mode")
	}

	cfg := DefaultConfig()
	stdpCfg := stdp.DefaultConfig()
	stdpCfg.APlus = 5
	stdpCfg.AMinus = 5
	stdpCfg.TauPlus = 8
	stdpCfg.TauMinus = 8
	stdpCfg.MaxWeightMagnitude = 500
	rule := stdp.NewRule(stdpCfg)

	tracker := Run(rule, "Pure STDP", cfg)

	t.Logf("Pure STDP best accuracy: %.1f%%", tracker.BestAccuracy()*100)
}

func TestRunRSTDP(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark in short mode")
	}

	cfg := DefaultConfig()
	rstdpCfg := rstdp.DefaultConfig()
	rule := rstdp.NewRule(rstdpCfg)

	// For R-STDP we need to inject reward after each sample.
	// Run manually instead of using Run() so we can call Reward().
	task := Task{}
	net, layout := BuildNetwork(cfg, rule)
	tracker := benchmark.NewTracker(10)

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

	trainSamples := task.TrainingSamples()
	for i, sample := range trainSamples {
		spikeCounts := PresentSample(net, layout, sample, cfg)
		predicted := Classify(spikeCounts)

		// Reward correct, punish incorrect
		if predicted == sample.Label {
			net.Reward(50)
		} else {
			net.Reward(-25)
		}

		if (i+1)%100 == 0 {
			acc, dead, sr := Evaluate(net, layout, task, cfg)
			weights := CollectWeights(net, layout)
			wm, ws := benchmark.WeightStats(weights)
			tracker.Record(benchmark.Checkpoint{
				SamplesProcessed: i + 1,
				Accuracy:         acc,
				WeightMean:       wm,
				WeightStdDev:     ws,
				DeadNeurons:      dead,
				SpikeRate:        sr,
			})
		}
	}

	t.Logf("R-STDP best accuracy: %.1f%%", tracker.BestAccuracy()*100)
	tracker.PrintReport(os.Stdout, task.Name(), "R-STDP")
}

func TestRunPredictive(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark in short mode")
	}

	cfg := DefaultConfig()
	predCfg := predictive.DefaultConfig()
	predCfg.LearningRate = 328 // ~0.5%
	predCfg.MaxWeightMagnitude = 500
	rule := predictive.NewRule(predCfg)

	tracker := Run(rule, "Predictive", cfg)

	t.Logf("Predictive best accuracy: %.1f%%", tracker.BestAccuracy()*100)
}

func TestWeightStats(t *testing.T) {
	weights := []int16{100, 200, 300, 400, 500}
	mean, stddev := benchmark.WeightStats(weights)

	if mean != 300 {
		t.Errorf("mean = %f, want 300", mean)
	}
	if stddev < 140 || stddev > 142 {
		t.Errorf("stddev = %f, want ~141.4", stddev)
	}
}

func TestTrackerConvergence(t *testing.T) {
	tracker := benchmark.NewTracker(3)

	// Improving
	if tracker.Record(benchmark.Checkpoint{Accuracy: 0.25}) {
		t.Error("should not converge after 1 checkpoint")
	}
	if tracker.Record(benchmark.Checkpoint{Accuracy: 0.50}) {
		t.Error("should not converge while improving")
	}
	if tracker.Record(benchmark.Checkpoint{Accuracy: 0.75}) {
		t.Error("should not converge while improving")
	}

	// Plateau — 3 checkpoints worse than best
	if tracker.Record(benchmark.Checkpoint{Accuracy: 0.70}) {
		t.Error("should not converge yet (only 1 stale)")
	}
	if tracker.Record(benchmark.Checkpoint{Accuracy: 0.71}) {
		t.Error("should not converge yet (only 2 stale)")
	}
	if !tracker.Record(benchmark.Checkpoint{Accuracy: 0.72}) {
		t.Error("should converge after 3 stale checkpoints")
	}
}
