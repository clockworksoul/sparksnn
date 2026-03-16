package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTDeepDense replicates the PocketNN architecture
// (Song & Lin, 2022) using our surrogate gradient training.
//
// Architecture: 784 → 100 → 50 → 10, fully dense
// PocketNN achieves 96.98% with integer-only DFA training.
// With proper surrogate gradient BPTT, we should match or exceed this.
//
// Total connections: 784*100 + 100*50 + 50*10 = 83,900
// Compare to our tuned single-layer: 784→512 (30% sparse) = 123,600
func TestMNISTDeepDense(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	numInput := 784
	numHidden1 := 100
	numHidden2 := 50
	numOutput := 10
	total := numInput + numHidden1 + numHidden2 + numOutput

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
	hidden1Start := uint32(numInput)
	hidden1End := uint32(numInput + numHidden1)
	hidden2Start := uint32(numInput + numHidden1)
	hidden2End := uint32(numInput + numHidden1 + numHidden2)
	outputStart := uint32(numInput + numHidden1 + numHidden2)
	outputEnd := uint32(total)

	rng := rand.New(rand.NewPCG(42, 42^0xbeef))

	// 100% input→hidden1 connectivity (fully dense)
	for i := inputStart; i < inputEnd; i++ {
		for h := hidden1Start; h < hidden1End; h++ {
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int64(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(i, h, w)
		}
	}

	// 100% hidden1→hidden2 connectivity (fully dense)
	for h1 := hidden1Start; h1 < hidden1End; h1++ {
		for h2 := hidden2Start; h2 < hidden2End; h2++ {
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int64(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h1, h2, w)
		}
	}

	// 100% hidden2→output connectivity (fully dense)
	for h2 := hidden2Start; h2 < hidden2End; h2++ {
		for o := outputStart; o < outputEnd; o++ {
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int64(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h2, o, w)
		}
	}

	totalConns := 0
	for i := range net.Neurons {
		totalConns += len(net.Neurons[i].Connections)
	}
	t.Logf("Network: %d neurons, %d connections (expected 83,900)", total, totalConns)
	t.Logf("Architecture: %d → %d → %d → %d (fully dense)",
		numInput, numHidden1, numHidden2, numOutput)

	baseLR := 0.001

	cfg := surrogate.Config{
		LearningRate: baseLR,
		NumSteps:     40,
		Surrogate:    surrogate.DefaultFastSigmoid(),
		Layers: []surrogate.LayerSpec{
			{Start: inputStart, End: inputEnd},
			{Start: hidden1Start, End: hidden1End},
			{Start: hidden2Start, End: hidden2End},
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
		// Learning rate decay: halve every 15 epochs
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

		t.Logf("Epoch %d: acc=%.2f%% (best=%.2f%%), avgLoss=%.4f, lr=%.6f",
			epoch+1, acc*100, bestAcc*100, avgLoss, lr)

		// Early stopping if no improvement for 10 epochs
		if patience >= 10 {
			t.Logf("Early stopping at epoch %d (no improvement for 10 epochs)", epoch+1)
			break
		}
	}

	t.Logf("\nFinal best accuracy: %.2f%%", bestAcc*100)
	t.Logf("PocketNN reference: 96.98%% (DFA, integer-only training, 3 epochs)")
	t.Logf("SparkSNN tuned reference: 97.21%% (784→512 sparse, surrogate grad, 42 epochs)")

	if bestAcc >= 0.97 {
		t.Logf("🏆 Exceeded PocketNN with smaller network!")
	} else if bestAcc >= 0.9698 {
		t.Logf("✓ Matched PocketNN")
	} else {
		t.Logf("⚠ Below PocketNN's 96.98%% — depth gradient attenuation?")
	}
}
