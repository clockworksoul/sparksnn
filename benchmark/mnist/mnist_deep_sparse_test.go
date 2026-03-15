package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTDeepSparse matches the Fashion-MNIST deep architecture
// for apples-to-apples depth comparison across both datasets.
//
// Architecture: 784 → 256 (30% sparse) → 128 (30% sparse) → 10 (60%)
// Same as TestFashionMNISTDeep in benchmark/fashionmnist.
//
// Compare to:
//   - MNIST tuned single-layer: 784→512 (30% sparse) = 97.21%
//   - MNIST deep dense (PocketNN): 784→100→50→10 = 96.18%
//   - Fashion-MNIST deep sparse: same arch = 79.13%
func TestMNISTDeepSparse(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	numInput := 784
	numHidden1 := 256
	numHidden2 := 128
	numOutput := 10
	total := numInput + numHidden1 + numHidden2 + numOutput

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
	hidden1Start := uint32(numInput)
	hidden1End := uint32(numInput + numHidden1)
	hidden2Start := uint32(numInput + numHidden1)
	hidden2End := uint32(numInput + numHidden1 + numHidden2)
	outputStart := uint32(numInput + numHidden1 + numHidden2)
	outputEnd := uint32(total)

	rng := rand.New(rand.NewPCG(42, 42^0xbeef))

	// 30% input→hidden1 connectivity
	for i := inputStart; i < inputEnd; i++ {
		for h := hidden1Start; h < hidden1End; h++ {
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

	// 30% hidden1→hidden2 connectivity
	for h1 := hidden1Start; h1 < hidden1End; h1++ {
		for h2 := hidden2Start; h2 < hidden2End; h2++ {
			if rng.Float64() > 0.30 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h1, h2, w)
		}
	}

	// 60% hidden2→output connectivity
	for h2 := hidden2Start; h2 < hidden2End; h2++ {
		for o := outputStart; o < outputEnd; o++ {
			if rng.Float64() > 0.60 {
				continue
			}
			wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
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
	t.Logf("Network: %d neurons, %d connections", total, totalConns)
	t.Logf("Architecture: %d → %d → %d → %d (30%% sparse hidden layers)",
		numInput, numHidden1, numHidden2, numOutput)

	baseLR := 0.0001

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

	epochs := 60
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
		perClass := make([]int, 10)
		perClassTotal := make([]int, 10)
		for _, sample := range testSamples {
			inputValues := encodeInput(sample.Inputs)
			predicted := trainer.Predict(inputValues)
			perClassTotal[sample.Label]++
			if predicted == sample.Label {
				correct++
				perClass[sample.Label]++
			}
		}

		acc := float64(correct) / float64(len(testSamples))
		if acc > bestAcc {
			bestAcc = acc
			patience = 0
			t.Logf("  Per-class accuracy:")
			for c := 0; c < 10; c++ {
				classAcc := float64(perClass[c]) / float64(perClassTotal[c]) * 100
				t.Logf("    %d: %.1f%% (%d/%d)",
					c, classAcc, perClass[c], perClassTotal[c])
			}
		} else {
			patience++
		}

		t.Logf("Epoch %d: acc=%.2f%% (best=%.2f%%), avgLoss=%.4f, lr=%.6f",
			epoch+1, acc*100, bestAcc*100, avgLoss, lr)

		if patience >= 10 {
			t.Logf("Early stopping at epoch %d (no improvement for 10 epochs)", epoch+1)
			break
		}
	}

	t.Logf("\nFinal best accuracy: %.2f%%", bestAcc*100)
	t.Logf("MNIST tuned single-layer reference: 97.21%% (784→512, 30%% sparse)")
	t.Logf("MNIST deep dense reference: 96.18%% (784→100→50→10, fully dense)")
	t.Logf("Fashion-MNIST same arch reference: 79.13%% (784→256→128→10, 30%% sparse)")

	if bestAcc >= 0.9721 {
		t.Logf("🏆 Deep sparse matches or beats single-layer tuned!")
	} else if bestAcc >= 0.9618 {
		t.Logf("✓ Beats deep dense PocketNN arch (%.2f%% vs 96.18%%)", bestAcc*100)
	} else {
		t.Logf("⚠ Below deep dense — sparsity + depth may compound gradient issues")
	}
}
