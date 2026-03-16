package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTInt32Verification trains a network with surrogate gradients
// (float64 domain), then compares predictions from the float64 trainer
// path vs the native int32 network on all 10,000 MNIST test samples.
//
// This validates the dual-domain claim: train in float64, deploy in int32,
// zero (or near-zero) prediction divergence.
//
// Architecture matches TestMNISTTuned: 784 → 512 (30% sparse) → 10
func TestMNISTInt32Verification(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	numInput := 784
	numHidden := 512
	numOutput := 10
	total := numInput + numHidden + numOutput

	threshold := 1.0
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0
	inputWeight := 0.5
	initWeightMax := 0.2

	intScale := float64(1 << 20)

	intThreshold := int64(threshold * intScale)
	net := bio.NewNetwork(uint32(total), 0, intThreshold, decayRate, 3)
	net.LearningRule = bio.NoOpLearning{}

	inputStart := uint32(0)
	inputEnd := uint32(numInput)
	hiddenStart := uint32(numInput)
	hiddenEnd := uint32(numInput + numHidden)
	outputStart := uint32(numInput + numHidden)
	outputEnd := uint32(total)

	rng := rand.New(rand.NewPCG(42, 42^0xbeef))

	// 30% input→hidden connectivity
	for i := inputStart; i < inputEnd; i++ {
		for h := hiddenStart; h < hiddenEnd; h++ {
			if rng.Float64() > 0.30 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int64(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(i, h, w)
		}
	}

	// 60% hidden→output connectivity
	for h := hiddenStart; h < hiddenEnd; h++ {
		for o := outputStart; o < outputEnd; o++ {
			if rng.Float64() > 0.60 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int64(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h, o, w)
		}
	}

	totalConns := 0
	for i := range net.Neurons {
		totalConns += len(net.Neurons[i].Connections)
	}
	t.Logf("Network: %d neurons, %d connections", total, totalConns)

	baseLR := 0.001

	cfg := surrogate.Config{
		LearningRate: baseLR,
		NumSteps:     40,
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
	testSamples := task.TestSamples()

	// Train for 30 epochs (enough to converge, don't need absolute best)
	epochs := 30
	bestAcc := 0.0

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	for epoch := 0; epoch < epochs; epoch++ {
		lr := baseLR * math.Pow(0.5, float64(epoch/15))
		trainer.Config.LearningRate = lr

		perm := rng.Perm(len(trainSamples))

		totalLoss := 0.0
		for _, pi := range perm {
			sample := trainSamples[pi]
			inputValues := encodeInput(sample.Inputs)
			loss := trainer.TrainSample(inputValues, sample.Label)
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainSamples))

		// Quick accuracy check using float64 path
		correct := 0
		for _, sample := range testSamples {
			inputValues := encodeInput(sample.Inputs)
			predicted := trainer.Predict(inputValues)
			if predicted == sample.Label {
				correct++
			}
		}

		acc := float64(correct) / float64(len(testSamples))
		if acc > bestAcc {
			bestAcc = acc
		}

		t.Logf("Epoch %d: acc=%.2f%% (best=%.2f%%), avgLoss=%.4f, lr=%.6f",
			epoch+1, acc*100, bestAcc*100, avgLoss, lr)
	}

	t.Logf("\nTraining complete. Best float64 accuracy: %.2f%%", bestAcc*100)

	// ===== INT32 vs FLOAT64 INFERENCE COMPARISON =====
	t.Log("")
	t.Log("=== Int32 vs Float64 Inference Comparison (10,000 samples) ===")

	float64Correct := 0
	int32Correct := 0
	mismatches := 0
	intInputWeight := inputWeight * intScale

	for si, sample := range testSamples {
		inputValues := encodeInput(sample.Inputs)

		// Float64 path (trainer.Predict — uses float64 shadow weights)
		f64Pred := trainer.Predict(inputValues)

		// Int32 path (native network — uses quantized int32 weights)
		net.ResetActivation()
		spikeCounts := make([]int, numOutput)

		for step := 0; step < cfg.NumSteps; step++ {
			for i := 0; i < numInput; i++ {
				if inputValues[i] > 0.01 {
					w := int64(inputValues[i] * intInputWeight)
					if w > 0 {
						net.Stimulate(inputStart+uint32(i), w)
					}
				}
			}
			net.Tick()

			for o := outputStart; o < outputEnd; o++ {
				if net.Neurons[o].LastFired == net.Counter {
					spikeCounts[o-outputStart]++
				}
			}
		}

		i32Pred := -1
		bestCount := 0
		for i, c := range spikeCounts {
			if c > bestCount {
				bestCount = c
				i32Pred = i
			}
		}

		if f64Pred == sample.Label {
			float64Correct++
		}
		if i32Pred == sample.Label {
			int32Correct++
		}
		if f64Pred != i32Pred {
			mismatches++
			if mismatches <= 20 { // Log first 20 mismatches
				t.Logf("  Mismatch sample %d: float64=%d, int32=%d, label=%d, spikes=%v",
					si, f64Pred, i32Pred, sample.Label, spikeCounts)
			}
		}
	}

	f64Acc := float64(float64Correct) / float64(len(testSamples)) * 100
	i32Acc := float64(int32Correct) / float64(len(testSamples)) * 100

	t.Log("")
	t.Logf("Float64 accuracy: %.2f%% (%d/%d)", f64Acc, float64Correct, len(testSamples))
	t.Logf("Int32   accuracy: %.2f%% (%d/%d)", i32Acc, int32Correct, len(testSamples))
	t.Logf("Mismatches: %d/%d samples (%.2f%%)",
		mismatches, len(testSamples),
		float64(mismatches)/float64(len(testSamples))*100)

	if mismatches == 0 {
		t.Log("🏆 Perfect match — int32 inference is identical to float64!")
	} else if mismatches <= 10 {
		t.Logf("✓ %d mismatches — excellent quantization fidelity", mismatches)
	} else if float64(mismatches)/float64(len(testSamples)) < 0.01 {
		t.Logf("✓ %d mismatches (%.2f%%) — good quantization fidelity",
			mismatches, float64(mismatches)/float64(len(testSamples))*100)
	} else if float64(mismatches)/float64(len(testSamples)) < 0.05 {
		t.Logf("⚠ %d mismatches (%.2f%%) — minor quantization differences",
			mismatches, float64(mismatches)/float64(len(testSamples))*100)
	} else {
		t.Errorf("✗ %d mismatches (%.2f%%) — significant quantization degradation",
			mismatches, float64(mismatches)/float64(len(testSamples))*100)
	}
}
