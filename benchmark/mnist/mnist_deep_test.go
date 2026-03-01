package mnist

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/learning/surrogate"
)

// TestMNISTDeep tests surrogate gradient training on MNIST with two
// hidden layers: 784 → 256 → 128 → 10.
func TestMNISTDeep(t *testing.T) {
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
	initWeightMax := 0.3

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

	connectSparse := func(srcStart, srcEnd, dstStart, dstEnd uint32, prob float64) int {
		count := 0
		for i := srcStart; i < srcEnd; i++ {
			for j := dstStart; j < dstEnd; j++ {
				if rng.Float64() > prob {
					continue
				}
				wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
				w := int32(math.Round(wf * intScale))
				if w == 0 {
					w = 1
				}
				net.Connect(i, j, w)
				count++
			}
		}
		return count
	}

	// Input → Hidden1: 20% sparse
	c1 := connectSparse(inputStart, inputEnd, hidden1Start, hidden1End, 0.20)
	// Hidden1 → Hidden2: 30% sparse
	c2 := connectSparse(hidden1Start, hidden1End, hidden2Start, hidden2End, 0.30)
	// Hidden2 → Output: 70% sparse
	c3 := connectSparse(hidden2Start, hidden2End, outputStart, outputEnd, 0.70)

	t.Logf("Architecture: %d → %d → %d → %d", numInput, numHidden1, numHidden2, numOutput)
	t.Logf("Connections: input→h1=%d, h1→h2=%d, h2→out=%d, total=%d",
		c1, c2, c3, c1+c2+c3)

	cfg := surrogate.Config{
		LearningRate: 0.001,
		NumSteps:     30,
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

	epochs := 15
	bestAcc := 0.0

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	for epoch := 0; epoch < epochs; epoch++ {
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
		}

		t.Logf("Epoch %d: acc=%.1f%% (best=%.1f%%), avgLoss=%.4f",
			epoch+1, acc*100, bestAcc*100, avgLoss)
	}

	t.Logf("Final best accuracy: %.1f%%", bestAcc*100)

	if bestAcc >= 0.97 {
		t.Logf("✓ Achieved %.1f%% — excellent!", bestAcc*100)
	} else if bestAcc >= 0.95 {
		t.Logf("✓ Achieved %.1f%% — target met!", bestAcc*100)
	} else if bestAcc >= 0.90 {
		t.Logf("✓ Achieved %.1f%% — strong result", bestAcc*100)
	} else if bestAcc >= 0.80 {
		t.Logf("⚠ Achieved %.1f%% — learning, needs tuning", bestAcc*100)
	} else {
		t.Errorf("Best accuracy %.1f%% — not working", bestAcc*100)
	}
}
