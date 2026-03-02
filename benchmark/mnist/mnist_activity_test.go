package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTActivity measures per-timestep neuron activity rates
// on a trained MNIST network for energy analysis.
//
// Uses the baseline config (784→256→10, 20% sparse) trained for
// 15 epochs, then runs the full test set with instrumentation.
func TestMNISTActivity(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	numInput := 784
	numHidden := 256
	numOutput := 10
	total := numInput + numHidden + numOutput

	threshold := 1.0
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0
	inputWeight := 0.5
	initWeightMax := 0.3

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

	for i := inputStart; i < inputEnd; i++ {
		for h := hiddenStart; h < hiddenEnd; h++ {
			if rng.Float64() > 0.20 {
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

	for h := hiddenStart; h < hiddenEnd; h++ {
		for o := outputStart; o < outputEnd; o++ {
			if rng.Float64() > 0.50 {
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

	numSteps := 30

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

	// Train for 15 epochs
	trainSamples := task.TrainingSamples()
	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	t.Log("Training 15 epochs...")
	for epoch := 0; epoch < 15; epoch++ {
		perm := rng.Perm(len(trainSamples))
		for _, pi := range perm {
			sample := trainSamples[pi]
			trainer.TrainSample(encodeInput(sample.Inputs), sample.Label)
		}
	}
	t.Log("Training complete. Measuring activity...")

	// Run full test set with instrumentation
	testSamples := task.TestSamples()

	// Accumulators
	totalActivePerStep := make([]float64, numSteps)
	totalSpikedPerStep := make([]float64, numSteps)

	// Per-layer accumulators
	type layerActivity struct {
		name       string
		start, end uint32
	}
	layers := []layerActivity{
		{"input", inputStart, inputEnd},
		{"hidden", hiddenStart, hiddenEnd},
		{"output", outputStart, outputEnd},
	}
	layerActiveTotal := make([][]float64, len(layers))
	layerSpikedTotal := make([][]float64, len(layers))
	for l := range layers {
		layerActiveTotal[l] = make([]float64, numSteps)
		layerSpikedTotal[l] = make([]float64, numSteps)
	}

	// Per-digit accumulators
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
	t.Logf("Test accuracy: %.1f%%", acc)

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

	// Per-layer breakdown
	t.Log("")
	t.Log("=== Per-Layer Activity (mean across all steps and samples) ===")
	// We need per-layer stats from the ActivityStats — but we only have
	// total neuron counts. Let me compute from the per-step data using
	// the knowledge that input neurons are indices 0..783, etc.
	// Actually, ActivityStats tracks total active neurons, not per-layer.
	// Let me just report what we have and note the limitation.
	t.Logf("  (Per-layer breakdown requires per-layer instrumentation — see below)")

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

	// Conservative estimate (all neurons decay every step)
	conservativeDecayOps := neuronsPerStep * float64(numSteps)
	// Lazy decay estimate (only active neurons decay)
	lazyDecayOps := activeNeuronsPerStep * float64(numSteps)

	savedOps := conservativeDecayOps - lazyDecayOps
	t.Logf("  Decay ops (conservative): %.0f multiply+shift per inference", conservativeDecayOps)
	t.Logf("  Decay ops (lazy decay):   %.0f multiply+shift per inference", lazyDecayOps)
	t.Logf("  Saved: %.0f ops (%.1f%% reduction in decay computation)",
		savedOps, savedOps/conservativeDecayOps*100)
	t.Logf("  Idle neurons per step: %.0f of %.0f (%.1f%%)",
		idleNeuronsPerStep, neuronsPerStep, idleNeuronsPerStep/neuronsPerStep*100)
}
