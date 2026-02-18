package iris

import (
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

// TestIrisSpatialPlasticity starts with a sparse, spatially-wired
// network and lets structural plasticity grow/prune connections
// during perturbation training.
//
// The protocol:
//   1. Build sparse network via Development harness (σ=0.3, ~30% density)
//   2. Train with ad-hoc perturbation
//   3. Every N perturbation steps, run Remodel() to prune weak
//      connections and grow new ones toward nearby active neurons
//   4. Finalize and evaluate
func TestIrisSpatialPlasticity(t *testing.T) {
	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	task := NewTask(42)
	cfg := DefaultConfig()

	numTrials := 5
	var bestTrialAcc float64
	var totalAcc float64

	for trial := 0; trial < numTrials; trial++ {
		dev := bio.NewDevelopmentSeeded(bio.DevParams{}, uint64(trial*1000+42))

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

		inputLayer := dev.GetLayer("input")
		hiddenLayer := dev.GetLayer("hidden")
		outputLayer := dev.GetLayer("output")

		// Wire with narrow sigma for sparse initial connectivity
		dev.Wire(bio.ConnectionRule{
			FromLayer: "input", ToLayer: "hidden",
			Sigma: 0.3, PBase: 1.0,
			InitWeightMin: 1, InitWeightMax: int32(cfg.InitWeightMax),
		})
		dev.Wire(bio.ConnectionRule{
			FromLayer: "hidden", ToLayer: "output",
			Sigma: 0.3, PBase: 1.0,
			InitWeightMin: 1, InitWeightMax: int32(cfg.InitWeightMax),
		})

		initialConns := dev.ConnectionCount()

		layout := Layout{
			InputStart:  inputLayer.StartIdx,
			InputEnd:    inputLayer.EndIdx,
			HiddenStart: hiddenLayer.StartIdx,
			HiddenEnd:   hiddenLayer.EndIdx,
			OutputStart: outputLayer.StartIdx,
			OutputEnd:   outputLayer.EndIdx,
		}

		// Set up spatial structural plasticity
		// Filter: only allow input→hidden and hidden→output
		filter := func(source, target uint32) bool {
			// input → hidden
			if source >= layout.InputStart && source < layout.InputEnd &&
				target >= layout.HiddenStart && target < layout.HiddenEnd {
				return true
			}
			// hidden → output
			if source >= layout.HiddenStart && source < layout.HiddenEnd &&
				target >= layout.OutputStart && target < layout.OutputEnd {
				return true
			}
			return false
		}

		plasticity := bio.NewSpatialPlasticity(bio.SpatialPlasticityConfig{
			PlasticityConfig: bio.PlasticityConfig{
				PruneThreshold:          50,  // aggressive pruning
				GrowthRate:              1,   // conservative growth
				MaxConnectionsPerNeuron: 12,  // cap total connections
				MinCoActivityWindow:     uint32(cfg.TicksPerSample + cfg.RestTicks),
				InitialWeight:           100,
				GrowthCandidates:        0,
				Filter:                  filter,
				ExploratoryGrowth:       true,
				ExploratoryRate:         1,
				HomeostaticEnabled:      true,
				DeadThreshold:           uint32(cfg.TicksPerSample * 5),
				HomeostaticStep:         10,
				MinThreshold:            50,
				MaxThreshold:            500,
			},
			GrowthSigma: 0.5,
			Positions:   dev.Positions,
		})

		net := dev.Net // don't finalize yet — plasticity needs positions

		// Collect initial learnable connections
		collectConns := func() []connRef {
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
			return conns
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

		conns := collectConns()
		if len(conns) == 0 {
			t.Logf("  Trial %d: no initial connections, skipping", trial)
			continue
		}

		bestScore := evaluate()
		perturbSize := int32(300)
		noImproveCount := 0
		maxSteps := 15000
		remodelEvery := 1000
		totalPruned, totalGrown := 0, 0

		for step := 0; step < maxSteps; step++ {
			// Perturbation step
			conns = collectConns() // refresh after potential remodel
			if len(conns) == 0 {
				break
			}

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

			// Structural plasticity
			if (step+1)%remodelEvery == 0 {
				pruned, grown := plasticity.Remodel(net, net.Counter)
				totalPruned += pruned
				totalGrown += grown
			}

			if (step+1)%5000 == 0 {
				currentConns := collectConns()
				trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
				testAcc, dead, _ := Evaluate(net, layout, task, cfg)
				t.Logf("  Trial %d, step %d: train=%.1f%% test=%.1f%% dead=%d conns=%d pruned=%d grown=%d",
					trial, step+1, trainAcc, testAcc*100, dead, len(currentConns), totalPruned, totalGrown)
			}
		}

		// Final evaluation
		finalConns := collectConns()
		testAcc, dead, sr := Evaluate(net, layout, task, cfg)
		trainAcc := float64(bestScore) / float64(len(trainSamples)) * 100
		totalAcc += testAcc

		if testAcc > bestTrialAcc {
			bestTrialAcc = testAcc
		}

		t.Logf("Trial %d: train=%.1f%% test=%.1f%% dead=%d spikeRate=%.2f conns=%d→%d pruned=%d grown=%d",
			trial, trainAcc, testAcc*100, dead, sr,
			initialConns, len(finalConns), totalPruned, totalGrown)
	}

	avgAcc := totalAcc / float64(numTrials)
	t.Logf("RESULT: best=%.1f%% avg=%.1f%% over %d trials", bestTrialAcc*100, avgAcc*100, numTrials)

	if bestTrialAcc <= 0.333 {
		t.Errorf("Best accuracy %.1f%% is no better than random", bestTrialAcc*100)
	}
}
