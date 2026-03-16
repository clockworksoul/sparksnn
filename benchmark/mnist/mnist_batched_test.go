package mnist

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTBatched runs the tuned MNIST configuration with mini-batch
// parallel training.
//
// Same architecture and hyperparameters as TestMNISTTuned (784 → 512
// hidden, 30% sparse, Adam + LR decay) but with mini-batch gradient
// averaging and parallel gradient computation.
//
// Compare results to TestMNISTTuned baseline (97.21%) to validate
// that batched training preserves accuracy while improving throughput.
func TestMNISTBatched(t *testing.T) {
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
	net := sparksnn.NewNetwork(uint32(total), 0, intThreshold, decayRate, 3)
	net.LearningRule = sparksnn.NoOpLearning{}

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

	batchSize := 16
	numWorkers := runtime.NumCPU()
	if numWorkers > batchSize {
		numWorkers = batchSize
	}

	t.Logf("Network: %d neurons, %d connections", total, totalConns)
	t.Logf("Mini-batch: size=%d, workers=%d (of %d CPUs)", batchSize, numWorkers, runtime.NumCPU())

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

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	epochs := 50
	bestAcc := 0.0
	patience := 0

	for epoch := 0; epoch < epochs; epoch++ {
		// Learning rate decay: halve every 15 epochs
		lr := baseLR * math.Pow(0.5, float64(epoch/15))
		trainer.Config.LearningRate = lr

		perm := rng.Perm(len(trainSamples))

		totalLoss := 0.0
		numBatches := 0

		for batchStart := 0; batchStart < len(perm); batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > len(perm) {
				batchEnd = len(perm)
			}

			batch := make([]surrogate.BatchSample, batchEnd-batchStart)
			for i, pi := range perm[batchStart:batchEnd] {
				sample := trainSamples[pi]
				batch[i] = surrogate.BatchSample{
					InputValues:  encodeInput(sample.Inputs),
					CorrectClass: sample.Label,
				}
			}

			loss := trainer.TrainBatch(batch, numWorkers)
			totalLoss += loss
			numBatches++
		}
		avgLoss := totalLoss / float64(numBatches)

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

		if patience >= 10 {
			t.Logf("Early stopping at epoch %d (no improvement for 10 epochs)", epoch+1)
			break
		}
	}

	t.Logf("Final best accuracy: %.2f%% (baseline: 97.21%%)", bestAcc*100)
}
