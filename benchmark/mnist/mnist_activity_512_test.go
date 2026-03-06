package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTActivity512 measures per-timestep neuron activity rates
// on the tuned MNIST network (784→512→10, 30% sparse, 42 epochs with
// LR scheduling) for energy analysis. This matches the headline 97.21%
// configuration from the paper.
func TestMNISTActivity512(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	numInput := 784
	numHidden := 512
	numOutput := 10
	total := numInput + numHidden + numOutput

	threshold := 1.0
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0
	inputWeight := 0.5
	initWeightMax := 0.2 // paper appendix: [-0.2, +0.2]

	intScale := float64(1 << 20)

	intThreshold := int32(threshold * intScale)
	net := bio.NewNetwork(uint32(total), 0, intThreshold, decayRate, 3)
	net.LearningRule = bio.NoOpLearning{}

	inputStart := uint32(0)
	inputEnd := uint32(numInput)
	hiddenStart := uint32(numInput)
	hiddenEnd := uint32(numInput + numHidden)
	outputStart := uint32(numInput + numHidden)
	outputEnd := uint32(total)

	rng := rand.New(rand.NewPCG(42, 42^0xbeef))

	// Input→Hidden: 30% sparse
	for i := inputStart; i < inputEnd; i++ {
		for h := hiddenStart; h < hiddenEnd; h++ {
			if rng.Float64() > 0.30 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(i, h, w)
		}
	}

	// Hidden→Output: 60% sparse
	for h := hiddenStart; h < hiddenEnd; h++ {
		for o := outputStart; o < outputEnd; o++ {
			if rng.Float64() > 0.60 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h, o, w)
		}
	}

	numSteps := 40 // tuned config uses 40 timesteps

	cfg := surrogate.Config{
		LearningRate: 0.001,
		NumSteps:     numSteps,
		Surrogate:    surrogate.DefaultFastSigmoid(),
		Layers: []surrogate.LayerSpec{
			{Start: inputStart, End: inputEnd},
			{Start: hiddenStart, End: hiddenEnd},
			{Start: outputStart, End: outputEnd},
		},
		Beta:        beta,
		InputWeight: inputWeight,
		Threshold:   threshold,
	}

	trainer := surrogate.NewTrainer(net, cfg, intScale)
	trainer.EnableAdam()

	trainSamples := task.TrainingSamples()
	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	// Train 42 epochs with LR scheduling (halve every 15 epochs)
	totalEpochs := 42
	for epoch := 0; epoch < totalEpochs; epoch++ {
		// LR scheduling: 0.001 for 1-15, 0.0005 for 16-30, 0.00025 for 31+
		if epoch == 15 {
			trainer.Config.LearningRate = 0.0005
			t.Log("LR → 0.0005")
		} else if epoch == 30 {
			trainer.Config.LearningRate = 0.00025
			t.Log("LR → 0.00025")
		}

		perm := rng.Perm(len(trainSamples))
		for _, pi := range perm {
			sample := trainSamples[pi]
			trainer.TrainSample(encodeInput(sample.Inputs), sample.Label)
		}

		if (epoch+1)%5 == 0 || epoch == totalEpochs-1 {
			t.Logf("Epoch %d complete", epoch+1)
		}
	}
	t.Log("Training complete. Measuring activity...")

	// Run full test set with instrumentation
	testSamples := task.TestSamples()

	totalActivePerStep := make([]float64, numSteps)
	totalSpikedPerStep := make([]float64, numSteps)

	digitActiveRate := make([]float64, 10)
	digitSpikeRate := make([]float64, 10)
	digitCount := make([]int, 10)

	correct := 0
	nSamples := len(testSamples)

	for _, sample := range testSamples {
		inputValues := encodeInput(sample.Inputs)
		predicted, stats := trainer.PredictWithStats(inputValues)

		if predicted == sample.Label {
			correct++
		}

		for step := 0; step < numSteps; step++ {
			totalActivePerStep[step] += float64(stats.ActivePerStep[step])
			totalSpikedPerStep[step] += float64(stats.SpikedPerStep[step])
		}

		digitActiveRate[sample.Label] += stats.MeanActivityRate()
		digitSpikeRate[sample.Label] += stats.MeanSpikeRate()
		digitCount[sample.Label]++
	}

	acc := float64(correct) / float64(nSamples) * 100
	t.Logf("Test accuracy: %.2f%%", acc)

	// Overall activity rates
	t.Log("")
	t.Log("=== Overall Activity Rates (fraction of neurons) ===")
	var overallActive, overallSpiked float64
	for step := 0; step < numSteps; step++ {
		avgActive := totalActivePerStep[step] / float64(nSamples) / float64(total)
		avgSpiked := totalSpikedPerStep[step] / float64(nSamples) / float64(total)
		overallActive += avgActive
		overallSpiked += avgSpiked
		if step%5 == 0 || step == numSteps-1 {
			t.Logf("  Step %2d: active=%.1f%%, spiking=%.1f%%",
				step, avgActive*100, avgSpiked*100)
		}
	}
	meanActive := overallActive / float64(numSteps)
	meanSpiked := overallSpiked / float64(numSteps)
	t.Logf("  Mean:    active=%.1f%%, spiking=%.1f%%", meanActive*100, meanSpiked*100)
	t.Logf("  Idle:    %.1f%% of neurons skip decay per timestep (lazy decay savings)",
		(1-meanActive)*100)

	// Per-digit activity
	t.Log("")
	t.Log("=== Activity Rate by Digit ===")
	for d := 0; d < 10; d++ {
		if digitCount[d] > 0 {
			avgActive := digitActiveRate[d] / float64(digitCount[d])
			avgSpike := digitSpikeRate[d] / float64(digitCount[d])
			t.Logf("  Digit %d: active=%.1f%%, spiking=%.1f%% (n=%d)",
				d, avgActive*100, avgSpike*100, digitCount[d])
		}
	}

	// Energy analysis
	t.Log("")
	t.Log("=== Energy Implications ===")
	neuronsPerStep := float64(total)
	activeNeuronsPerStep := meanActive * neuronsPerStep
	idleNeuronsPerStep := neuronsPerStep - activeNeuronsPerStep

	conservativeDecayOps := neuronsPerStep * float64(numSteps)
	lazyDecayOps := activeNeuronsPerStep * float64(numSteps)

	savedOps := conservativeDecayOps - lazyDecayOps
	t.Logf("  Total neurons: %d (784 input + %d hidden + 10 output)", total, numHidden)
	t.Logf("  Timesteps: %d", numSteps)
	t.Logf("  Decay ops (conservative): %.0f multiply+shift per inference", conservativeDecayOps)
	t.Logf("  Decay ops (lazy decay):   %.0f multiply+shift per inference", lazyDecayOps)
	t.Logf("  Saved: %.0f ops (%.1f%% reduction in decay computation)",
		savedOps, savedOps/conservativeDecayOps*100)
	t.Logf("  Idle neurons per step: %.0f of %.0f (%.1f%%)",
		idleNeuronsPerStep, neuronsPerStep, idleNeuronsPerStep/neuronsPerStep*100)
}
