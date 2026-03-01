package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/learning/surrogate"
)

// TestMNISTSurrogate tests surrogate gradient training on MNIST.
//
// Architecture: 784 input → 256 hidden → 10 output
// No population coding — raw pixel values as input.
// Sparse connectivity to keep it tractable.
func TestMNISTSurrogate(t *testing.T) {
	task, err := NewTask(0, 0) // full dataset: 60k train, 10k test
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	numInput := 784
	numHidden := 256
	numOutput := 10
	total := numInput + numHidden + numOutput

	// Float64 training domain
	threshold := 1.0
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0 // ~0.763
	inputWeight := 0.5                    // scaled by pixel value
	initWeightMax := 0.3

	intScale := float64(1 << 14) // 16384 — smaller scale for MNIST range

	intThreshold := int32(threshold * intScale)
	net := bio.NewNetwork(uint32(total), 0, intThreshold, decayRate, 3)
	net.LearningRule = bio.NoOpLearning{}

	inputStart := uint32(0)
	inputEnd := uint32(numInput)
	hiddenStart := uint32(numInput)
	hiddenEnd := uint32(numInput + numHidden)
	outputStart := uint32(numInput + numHidden)
	outputEnd := uint32(total)

	// Sparse connectivity: 20% input→hidden
	rng := rand.New(rand.NewPCG(42, 42^0xbeef))
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

	// Hidden→Output: 50% connectivity
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

	t.Logf("Network: %d neurons, input→hidden %.0f%%, hidden→output %.0f%%",
		total, inputConnProb*100, hiddenConnProb*100)

	// Count connections
	totalConns := 0
	for i := range net.Neurons {
		totalConns += len(net.Neurons[i].Connections)
	}
	t.Logf("Total connections: %d", totalConns)

	cfg := surrogate.Config{
		LearningRate: 0.001,
		NumSteps:     30,
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

	epochs := 15
	bestAcc := 0.0

	// Precompute normalized input encoding
	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle training data
		perm := rng.Perm(len(trainSamples))

		totalLoss := 0.0
		for _, pi := range perm {
			sample := trainSamples[pi]
			inputValues := encodeInput(sample.Inputs)
			loss := trainer.TrainSample(inputValues, sample.Label)
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainSamples))

		// Evaluate on test set using the surrogate trainer's forward pass
		// (faster than running the int32 network for evaluation during training)
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

		t.Logf("Epoch %d: acc=%.1f%% (best=%.1f%%), avgLoss=%.4f",
			epoch+1, acc*100, bestAcc*100, avgLoss)
	}

	t.Logf("Final best accuracy: %.1f%%", bestAcc*100)

	if bestAcc >= 0.95 {
		t.Logf("✓ Achieved %.1f%% — target met!", bestAcc*100)
	} else if bestAcc >= 0.90 {
		t.Logf("✓ Achieved %.1f%% — strong result, close to target", bestAcc*100)
	} else if bestAcc >= 0.80 {
		t.Logf("⚠ Achieved %.1f%% — learning well, needs tuning", bestAcc*100)
	} else if bestAcc >= 0.50 {
		t.Logf("⚠ Achieved %.1f%% — learning but needs work", bestAcc*100)
	} else {
		t.Errorf("Best accuracy %.1f%% — not working", bestAcc*100)
	}
}
