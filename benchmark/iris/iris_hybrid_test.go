package iris

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/clockworksoul/sparksnn/learning/hybrid"
	"github.com/clockworksoul/sparksnn/learning/perturbation"
	"github.com/clockworksoul/sparksnn/learning/rstdp"
)

// TestIrisHybridAdHoc uses ad-hoc perturbation (full-dataset eval)
// with R-STDP applied in a separate training pass.
//
// The protocol per perturbation step:
//   1. Perturb one weight
//   2. Evaluate on full dataset (LearningRule disabled — clean eval)
//   3. Keep/revert based on accuracy
//   4. Run one R-STDP training pass (LearningRule enabled)
//
// This keeps perturbation's evaluation clean while still letting
// R-STDP refine timing-dependent patterns between perturbation steps.
func TestIrisHybridAdHoc(t *testing.T) {
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
				rstdpRule := rstdp.NewRule(rstdp.Config{
					APlus:                10,
					AMinus:               10,
					TauPlus:              5,
					TauMinus:             5,
					EligibilityDecayRate: 45000,
					MaxWeightMagnitude:   5000,
				})

				// Start with no learning rule — we toggle it manually
				net, layout := BuildNetwork(cfg, nil)

				var conns []connRef
				for i := int(layout.InputStart); i < int(layout.HiddenEnd); i++ {
					for j := range net.Neurons[i].Connections {
						target := net.Neurons[i].Connections[j].Target
						if (target >= layout.HiddenStart && target < layout.HiddenEnd) ||
							(target >= layout.OutputStart && target < layout.OutputEnd) {
							conns = append(conns, connRef{i, j})
						}
					}
				}

				trainSamples := task.TrainingSamples()

				// Clean evaluation: no learning rule active
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

				// R-STDP training pass: learning rule active
				rstdpPass := func() {
					net.LearningRule = rstdpRule
					for _, sample := range trainSamples {
						spikeCounts := PresentSample(net, layout, sample, cfg)
						predicted := Classify(spikeCounts)
						if predicted == sample.Label {
							net.Reward(1)
						} else {
							net.Reward(-1)
						}
					}
					net.LearningRule = nil
				}

				bestScore := evaluate()
				perturbSize := int32(300)
				noImproveCount := 0
				maxSteps := 20000
				rstdpEvery := 200 // run R-STDP pass every N perturbation steps

				for step := 0; step < maxSteps; step++ {
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

					if noImproveCount > 300 {
						perturbSize = min(perturbSize*2, 3000)
						noImproveCount = 0
					}

					// Periodic R-STDP refinement
					if (step+1)%rstdpEvery == 0 {
						rstdpPass()
						// Re-evaluate after R-STDP (it may have improved things)
						newScore := evaluate()
						if newScore > bestScore {
							bestScore = newScore
						}
					}

					if (step+1)%5000 == 0 {
						trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
						testAcc, dead, _ := Evaluate(net, layout, task, cfg)
						t.Logf("  Trial %d, step %d: train=%.1f%% test=%.1f%% dead=%d",
							trial, step+1, trainAcc, testAcc*100, dead)
					}
				}

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

// TestIrisCompare runs perturbation-only vs hybrid side by side.
func TestIrisCompare(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.HiddenSize = 16
	maxSteps := 15000

	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	t.Run("perturbation-only", func(t *testing.T) {
		net, layout := BuildNetwork(cfg, nil)

		var conns []connRef
		for i := int(layout.InputStart); i < int(layout.HiddenEnd); i++ {
			for j := range net.Neurons[i].Connections {
				target := net.Neurons[i].Connections[j].Target
				if (target >= layout.HiddenStart && target < layout.HiddenEnd) ||
					(target >= layout.OutputStart && target < layout.OutputEnd) {
					conns = append(conns, connRef{i, j})
				}
			}
		}

		trainSamples := task.TrainingSamples()
		evaluate := func() int {
			correct := 0
			for _, sample := range trainSamples {
				spikeCounts := PresentSample(net, layout, sample, cfg)
				if Classify(spikeCounts) == sample.Label {
					correct++
				}
			}
			return correct
		}

		bestScore := evaluate()
		perturbSize := int32(300)
		noImproveCount := 0

		for step := 0; step < maxSteps; step++ {
			ref := conns[rand.IntN(len(conns))]
			conn := &net.Neurons[ref.neuronIdx].Connections[ref.connIdx]
			oldWeight := conn.Weight
			delta := int32(rand.IntN(int(perturbSize)*2+1)) - perturbSize
			nw := int64(conn.Weight) + int64(delta)
			if nw > 5000 { nw = 5000 }
			if nw < -5000 { nw = -5000 }
			conn.Weight = int32(nw)

			score := evaluate()
			if score > bestScore {
				bestScore = score
				noImproveCount = 0
			} else if score < bestScore {
				conn.Weight = oldWeight
				noImproveCount++
			} else {
				if rand.IntN(2) == 0 { conn.Weight = oldWeight }
				noImproveCount++
			}
			if noImproveCount > 300 {
				perturbSize = min(perturbSize*2, 3000)
				noImproveCount = 0
			}
		}

		testAcc, dead, _ := Evaluate(net, layout, task, cfg)
		t.Logf("perturbation-only: train=%.1f%% test=%.1f%% dead=%d",
			float64(bestScore)/float64(len(trainSamples))*100, testAcc*100, dead)
	})

	t.Run("hybrid", func(t *testing.T) {
		rstdpRule := rstdp.NewRule(rstdp.Config{
			APlus: 10, AMinus: 10,
			TauPlus: 5, TauMinus: 5,
			EligibilityDecayRate: 45000,
			MaxWeightMagnitude:   5000,
		})
		net, layout := BuildNetwork(cfg, nil)

		var conns []connRef
		for i := int(layout.InputStart); i < int(layout.HiddenEnd); i++ {
			for j := range net.Neurons[i].Connections {
				target := net.Neurons[i].Connections[j].Target
				if (target >= layout.HiddenStart && target < layout.HiddenEnd) ||
					(target >= layout.OutputStart && target < layout.OutputEnd) {
					conns = append(conns, connRef{i, j})
				}
			}
		}

		trainSamples := task.TrainingSamples()
		evaluate := func() int {
			correct := 0
			for _, sample := range trainSamples {
				spikeCounts := PresentSample(net, layout, sample, cfg)
				if Classify(spikeCounts) == sample.Label {
					correct++
				}
			}
			return correct
		}

		rstdpPass := func() {
			net.LearningRule = rstdpRule
			for _, sample := range trainSamples {
				spikeCounts := PresentSample(net, layout, sample, cfg)
				predicted := Classify(spikeCounts)
				if predicted == sample.Label {
					net.Reward(1)   // reinforce correct patterns
				} else {
					net.Reward(-1)  // weaken incorrect patterns
				}
			}
			net.LearningRule = nil
		}

		bestScore := evaluate()
		perturbSize := int32(300)
		noImproveCount := 0

		for step := 0; step < maxSteps; step++ {
			ref := conns[rand.IntN(len(conns))]
			conn := &net.Neurons[ref.neuronIdx].Connections[ref.connIdx]
			oldWeight := conn.Weight
			delta := int32(rand.IntN(int(perturbSize)*2+1)) - perturbSize
			nw := int64(conn.Weight) + int64(delta)
			if nw > 5000 { nw = 5000 }
			if nw < -5000 { nw = -5000 }
			conn.Weight = int32(nw)

			score := evaluate()
			if score > bestScore {
				bestScore = score
				noImproveCount = 0
			} else if score < bestScore {
				conn.Weight = oldWeight
				noImproveCount++
			} else {
				if rand.IntN(2) == 0 { conn.Weight = oldWeight }
				noImproveCount++
			}
			if noImproveCount > 300 {
				perturbSize = min(perturbSize*2, 3000)
				noImproveCount = 0
			}

			// R-STDP refinement pass every 200 steps
			if (step+1)%200 == 0 {
				rstdpPass()
				newScore := evaluate()
				if newScore > bestScore {
					bestScore = newScore
				}
			}
		}

		testAcc, dead, _ := Evaluate(net, layout, task, cfg)
		t.Logf("hybrid: train=%.1f%% test=%.1f%% dead=%d",
			float64(bestScore)/float64(len(trainSamples))*100, testAcc*100, dead)
	})
}

// Keep the LearningRule-based comparison for reference, but this
// is mainly for future use once the perturbation LearningRule is
// improved.
func TestIrisLearningRuleCompare(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.HiddenSize = 16

	maxEpochs := 200
	trainSamples := task.TrainingSamples()

	t.Run("perturbation-only", func(t *testing.T) {
		perturbCfg := perturbation.Config{
			PerturbSize:        300,
			MaxPerturbSize:     3000,
			AdaptAfter:         200,
			KeepEqualProb:      0.5,
			MaxWeightMagnitude: 5000,
			BatchSize:          12,
		}
		rule := perturbation.NewRule(perturbCfg)
		net, layout := BuildNetwork(cfg, rule)

		for epoch := 0; epoch < maxEpochs; epoch++ {
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
		}

		acc, _, _ := Evaluate(net, layout, task, cfg)
		t.Logf("Perturbation-only (LR): %.1f%%", acc*100)
	})

	t.Run("hybrid", func(t *testing.T) {
		hybridCfg := hybrid.Config{
			RSTDP: rstdp.Config{
				APlus:                10,
				AMinus:               10,
				TauPlus:              5,
				TauMinus:             5,
				EligibilityDecayRate: 45000,
				MaxWeightMagnitude:   5000,
			},
			Perturbation: perturbation.Config{
				PerturbSize:        300,
				MaxPerturbSize:     3000,
				AdaptAfter:         200,
				KeepEqualProb:      0.5,
				MaxWeightMagnitude: 5000,
				BatchSize:          12,
			},
		}
		rule := hybrid.NewRule(hybridCfg)
		net, layout := BuildNetwork(cfg, rule)

		for epoch := 0; epoch < maxEpochs; epoch++ {
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
		}

		acc, _, _ := Evaluate(net, layout, task, cfg)
		t.Logf("Hybrid (LR): %.1f%%", acc*100)
	})
}
