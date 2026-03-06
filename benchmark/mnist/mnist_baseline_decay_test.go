package mnist

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTBaselineWithDecay is an apples-to-apples comparison run.
//
// Architecture: identical to TestMNISTSurrogate (baseline)
//   - 784 input → 256 hidden → 10 output
//   - 20% input→hidden, 50% hidden→output connectivity
//   - initWeightMax = 0.3, 30 timesteps
//
// Training schedule: identical to TestMNISTTuned
//   - Up to 50 epochs (with early stopping, patience=10)
//   - LR 0.001 halved every 15 epochs
//   - Adam optimizer
//
// Purpose: isolate the effect of architecture/hyperparameter tuning from
// the training schedule. The original baseline ran only 15 epochs with
// flat LR, making it unclear how much of the tuned config's improvement
// came from longer training + LR decay vs. architectural changes.
func TestMNISTBaselineWithDecay(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	// === Architecture: same as TestMNISTSurrogate (baseline) ===
	numInput := 784
	numHidden := 256
	numOutput := 10
	total := numInput + numHidden + numOutput

	threshold := 1.0
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0
	inputWeight := 0.5
	initWeightMax := 0.3 // baseline value

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

	// Sparse connectivity: 20% input→hidden (baseline)
	inputConnProb := 0.20
	for i := inputStart; i < inputEnd; i++ {
		for h := hiddenStart; h < hiddenEnd; h++ {
			if rng.Float64() > inputConnProb {
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

	// Hidden→Output: 50% connectivity (baseline)
	hiddenConnProb := 0.50
	for h := hiddenStart; h < hiddenEnd; h++ {
		for o := outputStart; o < outputEnd; o++ {
			if rng.Float64() > hiddenConnProb {
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
	t.Logf("Network: %d neurons, %d connections (input→hidden %.0f%%, hidden→output %.0f%%)",
		total, totalConns, inputConnProb*100, hiddenConnProb*100)

	// === Training schedule: same as TestMNISTTuned ===
	baseLR := 0.001

	cfg := surrogate.Config{
		LearningRate: baseLR,
		NumSteps:     30, // baseline timesteps
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

	epochs := 50
	bestAcc := 0.0
	patience := 0

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// Learning rate decay: halve every 15 epochs (same as tuned)
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

		// Force GC between epochs to prevent OOM on memory-constrained machines
		runtime.GC()

		t.Logf("Epoch %d: acc=%.2f%% (best=%.2f%%), avgLoss=%.4f, lr=%.6f",
			epoch+1, acc*100, bestAcc*100, avgLoss, lr)

		// Early stopping if no improvement for 10 epochs
		if patience >= 10 {
			t.Logf("Early stopping at epoch %d (no improvement for 10 epochs)", epoch+1)
			break
		}
	}

	t.Logf("Final best accuracy: %.2f%%", bestAcc*100)

	if bestAcc >= 0.97 {
		t.Logf("✓ Achieved %.2f%% — excellent for baseline architecture!", bestAcc*100)
	} else if bestAcc >= 0.95 {
		t.Logf("✓ Achieved %.2f%% — strong result", bestAcc*100)
	} else if bestAcc >= 0.90 {
		t.Logf("⚠ Achieved %.2f%% — good, training schedule helped", bestAcc*100)
	} else {
		t.Logf("⚠ Achieved %.2f%% — architecture is the bottleneck", bestAcc*100)
	}
}
