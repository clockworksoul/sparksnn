package iris

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestIrisAdHocPerturbation uses the same ad-hoc weight perturbation
// approach that achieved 82-98% on XOR: evaluate full dataset per
// perturbation step, keep/revert based on total accuracy.
//
// This bypasses the LearningRule interface and directly manipulates
// weights, giving the clearest picture of whether perturbation can
// solve Iris at all.
func TestIrisAdHocPerturbation(t *testing.T) {
	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	for _, hiddenSize := range []int{16, 32} {
		t.Run(fmt.Sprintf("hidden=%d", hiddenSize), func(t *testing.T) {
			task := NewTask(42)
			cfg := DefaultConfig()
			cfg.HiddenSize = hiddenSize

			numTrials := 5
			var bestTrialAcc float64
			var totalAcc float64

			for trial := 0; trial < numTrials; trial++ {
				net, layout := BuildNetwork(cfg, nil) // no learning rule

				// Collect learnable connections
				var conns []connRef
				for i := int(layout.InputStart); i < int(layout.HiddenEnd); i++ {
					for j := range net.Neurons[i].Connections {
						target := net.Neurons[i].Connections[j].Target
						// Only learnable connections (input→hidden, hidden→output)
						if (target >= layout.HiddenStart && target < layout.HiddenEnd) ||
							(target >= layout.OutputStart && target < layout.OutputEnd) {
							conns = append(conns, connRef{i, j})
						}
					}
				}

				trainSamples := task.TrainingSamples()

				// Evaluate: present all training samples, count correct
				evaluate := func() int {
					correct := 0
					for _, sample := range trainSamples {
						spikeCounts := PresentSample(net, layout, sample, cfg)
						predicted := Classify(spikeCounts)
						if predicted == sample.Label {
							correct++
						}
					}
					return correct
				}

				bestScore := evaluate()
				perturbSize := int32(300)
				noImproveCount := 0
				maxSteps := 20000

				for step := 0; step < maxSteps; step++ {
					// Pick a random connection and perturb
					ref := conns[rand.IntN(len(conns))]
					conn := &net.Neurons[ref.neuronIdx].Connections[ref.connIdx]

					oldWeight := conn.Weight
					delta := int32(rand.IntN(int(perturbSize)*2+1)) - perturbSize
					newWeight := int64(conn.Weight) + int64(delta)
					if newWeight > 5000 {
						newWeight = 5000
					}
					if newWeight < -5000 {
						newWeight = -5000
					}
					conn.Weight = int32(newWeight)

					score := evaluate()

					if score > bestScore {
						bestScore = score
						noImproveCount = 0
					} else if score < bestScore {
						conn.Weight = oldWeight
						noImproveCount++
					} else {
						if rand.IntN(2) == 0 {
							conn.Weight = oldWeight
						}
						noImproveCount++
					}

					// Adaptive perturbation size
					if noImproveCount > 300 {
						perturbSize = min(perturbSize*2, 3000)
						noImproveCount = 0
					}

					if (step+1)%5000 == 0 {
						trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
						testAcc, dead, _ := Evaluate(net, layout, task, cfg)
						t.Logf("  Trial %d, step %d: train=%.1f%% test=%.1f%% dead=%d pertSize=%d",
							trial, step+1, trainAcc, testAcc*100, dead, perturbSize)
					}
				}

				// Final test accuracy
				testAcc, dead, sr := Evaluate(net, layout, task, cfg)
				trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
				totalAcc += testAcc

				if testAcc > bestTrialAcc {
					bestTrialAcc = testAcc
				}

				t.Logf("Trial %d: train=%.1f%% test=%.1f%% dead=%d spikeRate=%.2f",
					trial, trainAcc, testAcc*100, dead, sr)
			}

			avgAcc := totalAcc / float64(numTrials)
			t.Logf("Hidden=%d: best=%.1f%% avg=%.1f%% over %d trials",
				hiddenSize, bestTrialAcc*100, avgAcc*100, numTrials)

			if bestTrialAcc <= 0.333 {
				t.Errorf("Best accuracy %.1f%% is no better than random", bestTrialAcc*100)
			}
		})
	}
}
