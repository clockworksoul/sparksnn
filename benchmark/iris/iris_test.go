package iris

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/clockworksoul/biomimetic-network/learning/perturbation"
)

// TestIrisDataLoad verifies the dataset loads and splits correctly.
func TestIrisDataLoad(t *testing.T) {
	task := NewTask(42)

	train := task.TrainingSamples()
	test := task.TestSamples()

	if len(train) != 120 {
		t.Fatalf("expected 120 training samples, got %d", len(train))
	}
	if len(test) != 30 {
		t.Fatalf("expected 30 test samples, got %d", len(test))
	}

	// Verify stratification: 40 train, 10 test per class
	trainCounts := map[int]int{}
	testCounts := map[int]int{}
	for _, s := range train {
		trainCounts[s.Label]++
	}
	for _, s := range test {
		testCounts[s.Label]++
	}

	for cls := 0; cls < 3; cls++ {
		if trainCounts[cls] != 40 {
			t.Errorf("class %d: expected 40 train, got %d", cls, trainCounts[cls])
		}
		if testCounts[cls] != 10 {
			t.Errorf("class %d: expected 10 test, got %d", cls, testCounts[cls])
		}
	}

	t.Logf("Train: %v, Test: %v", trainCounts, testCounts)
}

// TestIrisPerturbation tests weight perturbation learning on Iris.
// This is the key scalability test: can perturbation learn a real
// (non-binary) classification task with 4 continuous features and
// 3 classes?
func TestIrisPerturbation(t *testing.T) {
	for _, hiddenSize := range []int{16, 32} {
		t.Run(fmt.Sprintf("hidden=%d", hiddenSize), func(t *testing.T) {
			task := NewTask(42)
			cfg := DefaultConfig()
			cfg.HiddenSize = hiddenSize

			numTrials := 10
			var bestAcc float64
			var totalAcc float64

			for trial := 0; trial < numTrials; trial++ {
				perturbCfg := perturbation.Config{
					PerturbSize:        300,
					MaxPerturbSize:     3000,
					AdaptAfter:         300,
					KeepEqualProb:      0.5,
					MaxWeightMagnitude: 5000,
					BatchSize:          120, // full epoch
				}
				rule := perturbation.NewRule(perturbCfg)

				net, layout := BuildNetwork(cfg, rule)

				// Train: present all training samples, deliver reward
				trainSamples := task.TrainingSamples()
				maxEpochs := 500

				for epoch := 0; epoch < maxEpochs; epoch++ {
					// Shuffle each epoch
					rand.Shuffle(len(trainSamples), func(i, j int) {
						trainSamples[i], trainSamples[j] = trainSamples[j], trainSamples[i]
					})

					epochCorrect := 0
					for _, sample := range trainSamples {
						spikeCounts := PresentSample(net, layout, sample, cfg)
						predicted := Classify(spikeCounts)
						reward := int32(-1)
						if predicted == sample.Label {
							reward = 1
							epochCorrect++
						}
						net.Reward(reward)
					}

					// Early exit if training accuracy is high
					if float64(epochCorrect)/float64(len(trainSamples)) > 0.95 {
						break
					}
				}

				// Evaluate on test set
				acc, dead, sr := Evaluate(net, layout, task, cfg)
				totalAcc += acc

				if acc > bestAcc {
					bestAcc = acc
				}

				if trial == 0 || acc == bestAcc {
					t.Logf("Trial %d: acc=%.1f%% dead=%d spikeRate=%.2f",
						trial, acc*100, dead, sr)
				}
			}

			avgAcc := totalAcc / float64(numTrials)
			t.Logf("Hidden=%d: best=%.1f%% avg=%.1f%% over %d trials",
				hiddenSize, bestAcc*100, avgAcc*100, numTrials)

			// We expect better than random (33.3%)
			if bestAcc <= 0.333 {
				t.Errorf("Best accuracy %.1f%% is no better than random", bestAcc*100)
			}
		})
	}
}

// TestIrisPerturbationQuick is a shorter version for quick iteration.
// Single trial, fewer epochs, just to verify the pipeline works.
func TestIrisPerturbationQuick(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.HiddenSize = 16

	// Smaller batch size = more perturbation trials per epoch.
	// With BatchSize=12, we get 10 perturbation evaluations per epoch.
	perturbCfg := perturbation.Config{
		PerturbSize:        300,
		MaxPerturbSize:     3000,
		AdaptAfter:         100,
		KeepEqualProb:      0.5,
		MaxWeightMagnitude: 5000,
		BatchSize:          12,
	}
	rule := perturbation.NewRule(perturbCfg)
	net, layout := BuildNetwork(cfg, rule)

	trainSamples := task.TrainingSamples()

	// Baseline
	acc0, _, _ := Evaluate(net, layout, task, cfg)
	t.Logf("Baseline accuracy: %.1f%%", acc0*100)

	// Train for 200 epochs
	for epoch := 0; epoch < 200; epoch++ {
		rand.Shuffle(len(trainSamples), func(i, j int) {
			trainSamples[i], trainSamples[j] = trainSamples[j], trainSamples[i]
		})

		for _, sample := range trainSamples {
			spikeCounts := PresentSample(net, layout, sample, cfg)
			predicted := Classify(spikeCounts)
			reward := int32(-1)
			if predicted == sample.Label {
				reward = 1
			}
			net.Reward(reward)
		}

		if (epoch+1)%25 == 0 {
			acc, dead, _ := Evaluate(net, layout, task, cfg)
			t.Logf("Epoch %d: acc=%.1f%% dead=%d", epoch+1, acc*100, dead)
		}
	}

	accFinal, _, _ := Evaluate(net, layout, task, cfg)
	t.Logf("Final accuracy: %.1f%% (baseline was %.1f%%)", accFinal*100, acc0*100)
}
