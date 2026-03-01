package iris

import (
	"fmt"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
)

// TestIrisSpatialWiring uses the Development harness to create
// a spatially-wired network for Iris classification, then trains
// with ad-hoc perturbation.
//
// This tests whether distance-dependent connectivity (Peter's Rule)
// can match or beat fully-connected networks on Iris.
func TestIrisSpatialWiring(t *testing.T) {
	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	for _, sigma := range []float32{0.3, 0.5, 0.8, 1.5} {
		t.Run(fmt.Sprintf("sigma=%.1f", sigma), func(t *testing.T) {
			task := NewTask(42)
			cfg := DefaultConfig()

			numTrials := 3
			var bestTrialAcc float64
			var totalAcc float64

			for trial := 0; trial < numTrials; trial++ {
				dev := bio.NewDevelopmentSeeded(bio.DevParams{}, uint64(trial*1000+42))

				// Place layers in overlapping 2D space
				// Input and output at edges, hidden in the middle
				dev.AddLayer(bio.LayerSpec{
					Name: "input", Role: bio.RoleInput, Size: 4,
					Placement: bio.PlaceGrid,
					OriginX: 0, OriginY: 0.4, Width: 1, Height: 0.2,
				})
				dev.AddLayer(bio.LayerSpec{
					Name: "hidden", Role: bio.RoleHidden, Size: 16,
					Placement: bio.PlaceRandom,
					OriginX: 0, OriginY: 0, Width: 1, Height: 1,
				})
				dev.AddLayer(bio.LayerSpec{
					Name: "output", Role: bio.RoleOutput, Size: 3,
					Placement: bio.PlaceGrid,
					OriginX: 0, OriginY: 0.4, Width: 1, Height: 0.2,
				})

				dev.Build(bio.DevParams{
					Baseline: 0, Threshold: int32(cfg.Threshold),
					DecayRate: cfg.DecayRate, RefractoryPeriod: cfg.RefractoryPeriod,
				})

				// Wire with Peter's Rule
				inToHid := dev.Wire(bio.ConnectionRule{
					FromLayer: "input", ToLayer: "hidden",
					Sigma: sigma, PBase: 1.0,
					InitWeightMin: 1, InitWeightMax: int32(cfg.InitWeightMax),
				})
				hidToOut := dev.Wire(bio.ConnectionRule{
					FromLayer: "hidden", ToLayer: "output",
					Sigma: sigma, PBase: 1.0,
					InitWeightMin: 1, InitWeightMax: int32(cfg.InitWeightMax),
				})

				inDensity := dev.ConnectionDensity("input", "hidden")
				outDensity := dev.ConnectionDensity("hidden", "output")

				if trial == 0 {
					t.Logf("  σ=%.1f: in→hid=%d (%.0f%%), hid→out=%d (%.0f%%)",
						sigma, inToHid, inDensity*100, hidToOut, outDensity*100)
				}

				inputLayer := dev.GetLayer("input")
				hiddenLayer := dev.GetLayer("hidden")
				outputLayer := dev.GetLayer("output")

				// Build layout for PresentSample compatibility
				layout := Layout{
					InputStart:  inputLayer.StartIdx,
					InputEnd:    inputLayer.EndIdx,
					HiddenStart: hiddenLayer.StartIdx,
					HiddenEnd:   hiddenLayer.EndIdx,
					OutputStart: outputLayer.StartIdx,
					OutputEnd:   outputLayer.EndIdx,
				}

				net := dev.Finalize()

				// Collect learnable connections
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

				if len(conns) == 0 {
					t.Logf("  Trial %d: no learnable connections, skipping", trial)
					continue
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
				maxSteps := 15000

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

				testAcc, dead, sr := Evaluate(net, layout, task, cfg)
				trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
				totalAcc += testAcc

				if testAcc > bestTrialAcc {
					bestTrialAcc = testAcc
				}

				t.Logf("  Trial %d: train=%.1f%% test=%.1f%% dead=%d spikeRate=%.2f conns=%d",
					trial, trainAcc, testAcc*100, dead, sr, len(conns))
			}

			avgAcc := totalAcc / float64(numTrials)
			t.Logf("σ=%.1f: best=%.1f%% avg=%.1f%%", sigma, bestTrialAcc*100, avgAcc*100)
		})
	}
}
