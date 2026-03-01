package iris

import (
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/learning/surrogate"
)

// TestIrisSurrogate tests surrogate gradient training on Iris.
//
// This uses proper backpropagation through time with a surrogate
// gradient to overcome the non-differentiability of spikes.
// Float64 training, integer inference.
func TestIrisSurrogate(t *testing.T) {
	task := NewTask(42)

	// Network parameters
	popSize := 10
	numInput := 4 * popSize // 40 input neurons (population coded)
	numHidden := 20
	numOutput := 3
	total := numInput + numHidden + numOutput

	// For surrogate gradient training, we use small float-friendly values.
	// The int32 network is just for inference evaluation — the trainer
	// simulates entirely in float64 with its own scale.
	threshold := float64(1.0)
	decayRate := uint16(50000)
	beta := float64(decayRate) / 65536.0 // ~0.763
	inputWeight := float64(1.0)
	initWeightMax := float64(0.5)
	refractoryPeriod := uint32(3)

	// Int32 scale factor: multiply float64 weights by this to get int32
	intScale := float64(1 << 16) // 65536

	// Network uses scaled int32 weights for inference
	intThreshold := int32(threshold * intScale)
	net := bio.NewNetwork(uint32(total), 0, intThreshold, decayRate, refractoryPeriod)
	net.LearningRule = bio.NoOpLearning{}

	inputStart := uint32(0)
	inputEnd := uint32(numInput)
	hiddenStart := uint32(numInput)
	hiddenEnd := uint32(numInput + numHidden)
	outputStart := uint32(numInput + numHidden)
	outputEnd := uint32(total)

	// Sparse mixed-sign connectivity
	for i := inputStart; i < inputEnd; i++ {
		for h := hiddenStart; h < hiddenEnd; h++ {
			if rand.Float64() > 0.5 {
				continue
			}
			wf := (rand.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(i, h, w)
		}
	}

	for h := hiddenStart; h < hiddenEnd; h++ {
		for o := outputStart; o < outputEnd; o++ {
			if rand.Float64() > 0.7 {
				continue
			}
			wf := (rand.Float64()*2.0 - 1.0) * initWeightMax
			w := int32(math.Round(wf * intScale))
			if w == 0 {
				w = 1
			}
			net.Connect(h, o, w)
		}
	}

	// No lateral inhibition for now — keep it simple for first test

	// Trainer config
	cfg := surrogate.Config{
		LearningRate: 0.001,
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

	// Training
	trainSamples := task.TrainingSamples()
	testSamples := task.TestSamples()

	epochs := 300
	bestAcc := 0.0

	// Precompute population-coded input values for each sample
	populationSigma := 255.0 / float64(popSize-1)

	encodeInput := func(sample []byte) []float64 {
		values := make([]float64, numInput)
		for f := 0; f < 4; f++ {
			val := float64(sample[f])
			for k := 0; k < popSize; k++ {
				center := 255.0 * float64(k) / float64(popSize-1)
				d := val - center
				response := math.Exp(-(d * d) / (2 * populationSigma * populationSigma))
				values[f*popSize+k] = response
			}
		}
		return values
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle training data
		rng := rand.New(rand.NewPCG(uint64(epoch), uint64(epoch)^0xcafe))
		perm := rng.Perm(len(trainSamples))

		totalLoss := 0.0
		for _, pi := range perm {
			sample := trainSamples[pi]
			inputValues := encodeInput(sample.Inputs)
			loss := trainer.TrainSample(inputValues, sample.Label)
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainSamples))

		// Evaluate on test set
		correct := 0
		for _, sample := range testSamples {
			inputValues := encodeInput(sample.Inputs)

			// Forward pass only (no training) using the int32 network
			// Reset first
			net.ResetActivation()

			spikeCounts := make([]int, numOutput)
			for step := 0; step < cfg.NumSteps; step++ {
				// Stimulate input neurons (scale input weight to int32 domain)
				intInputWeight := inputWeight * intScale
				for i := 0; i < numInput; i++ {
					if inputValues[i] > 0.01 {
						w := int32(inputValues[i] * intInputWeight)
						if w > 0 {
							net.Stimulate(inputStart+uint32(i), w)
						}
					}
				}
				net.Tick()

				// Count output spikes
				for o := outputStart; o < outputEnd; o++ {
					if net.Neurons[o].LastFired == net.Counter {
						spikeCounts[o-outputStart]++
					}
				}
			}

			predicted := -1
			bestCount := 0
			for i, c := range spikeCounts {
				if c > bestCount {
					bestCount = c
					predicted = i
				}
			}
			if predicted == sample.Label {
				correct++
			}
		}

		acc := float64(correct) / float64(len(testSamples))
		if acc > bestAcc {
			bestAcc = acc
		}

		if (epoch+1)%10 == 0 {
			t.Logf("Epoch %d: acc=%.1f%% (best=%.1f%%), avgLoss=%.4f",
				epoch+1, acc*100, bestAcc*100, avgLoss)
		}
	}

	t.Logf("Final best accuracy: %.1f%%", bestAcc*100)

	if bestAcc >= 0.96 {
		t.Logf("✓ Achieved %.1f%% — matches arbiter baseline!", bestAcc*100)
	} else if bestAcc >= 0.80 {
		t.Logf("✓ Achieved %.1f%% — solid, room to tune", bestAcc*100)
	} else if bestAcc >= 0.50 {
		t.Logf("⚠ Achieved %.1f%% — learning but needs work", bestAcc*100)
	} else {
		t.Errorf("Best accuracy %.1f%% — surrogate gradient training not working", bestAcc*100)
	}
}
