package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTEarlyExit trains the tuned 512-neuron MNIST network, then
// measures test accuracy at each timestep (1..40) during inference.
// This determines the minimum timesteps needed for inference without
// accuracy loss, directly informing the energy analysis in §7.3.
//
// If inference accuracy plateaus at timestep T < 40, the energy cost
// per sample can be reduced by a factor of 40/T.
func TestMNISTEarlyExit(t *testing.T) {
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

	// 30% input→hidden connectivity
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

	// 60% hidden→output connectivity
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

	totalConns := 0
	for i := range net.Neurons {
		totalConns += len(net.Neurons[i].Connections)
	}
	t.Logf("Network: %d neurons, %d connections", total, totalConns)

	maxSteps := 40
	baseLR := 0.001

	cfg := surrogate.Config{
		LearningRate: baseLR,
		NumSteps:     maxSteps,
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

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	// ─── Phase 1: Train ────────────────────────────────────────
	epochs := 42
	bestAcc := 0.0
	patience := 0

	t.Log("=== TRAINING ===")

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
			patience = 0
		} else {
			patience++
		}

		t.Logf("Epoch %d: acc=%.2f%% (best=%.2f%%), loss=%.4f, lr=%.6f",
			epoch+1, acc*100, bestAcc*100, avgLoss, lr)

		if patience >= 10 {
			t.Logf("Early stopping at epoch %d", epoch+1)
			break
		}
	}

	t.Logf("Training complete. Best accuracy: %.2f%%", bestAcc*100)

	// ─── Phase 2: Per-timestep inference accuracy ──────────────
	t.Log("")
	t.Log("=== EARLY EXIT ANALYSIS ===")
	t.Logf("Running inference on %d test samples, recording accuracy at each timestep 1..%d",
		len(testSamples), maxSteps)

	correctAtStep := make([]int, maxSteps)

	for _, sample := range testSamples {
		inputValues := encodeInput(sample.Inputs)
		predictions := trainer.PredictPerTimestep(inputValues)

		for step := 0; step < maxSteps; step++ {
			if predictions[step] == sample.Label {
				correctAtStep[step]++
			}
		}
	}

	// ─── Results ───────────────────────────────────────────────
	t.Log("")
	t.Log("Timestep | Accuracy | Δ from t=40")
	t.Log("---------+----------+-----------")

	nTest := float64(len(testSamples))
	acc40 := float64(correctAtStep[maxSteps-1]) / nTest

	for step := 0; step < maxSteps; step++ {
		acc := float64(correctAtStep[step]) / nTest
		delta := acc - acc40
		marker := ""
		if step == maxSteps-1 {
			marker = " ← training timesteps"
		} else if math.Abs(delta) < 0.001 {
			marker = " ← matches t=40"
		}
		t.Logf("  t=%-3d  | %6.2f%%  | %+.2f%%%s",
			step+1, acc*100, delta*100, marker)
	}

	// Find earliest timestep within 0.1% of t=40
	t.Log("")
	for step := 0; step < maxSteps; step++ {
		acc := float64(correctAtStep[step]) / nTest
		if acc >= acc40-0.001 {
			savings := float64(maxSteps-step-1) / float64(maxSteps) * 100
			t.Logf("🎯 Earliest timestep within 0.1%% of full accuracy: t=%d (%.2f%% vs %.2f%%)",
				step+1, acc*100, acc40*100)
			t.Logf("   Potential energy savings: %.0f%% fewer timesteps (%d → %d)",
				savings, maxSteps, step+1)
			break
		}
	}

	// Find earliest timestep within 0.5% of t=40
	for step := 0; step < maxSteps; step++ {
		acc := float64(correctAtStep[step]) / nTest
		if acc >= acc40-0.005 {
			savings := float64(maxSteps-step-1) / float64(maxSteps) * 100
			t.Logf("📊 Earliest timestep within 0.5%% of full accuracy: t=%d (%.2f%% vs %.2f%%)",
				step+1, acc*100, acc40*100)
			t.Logf("   Potential energy savings: %.0f%% fewer timesteps (%d → %d)",
				savings, maxSteps, step+1)
			break
		}
	}
}
