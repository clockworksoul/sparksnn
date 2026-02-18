package iris

import (
	"math/rand/v2"
	"os"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

// TestIrisAnalysis trains a network with ad-hoc perturbation, then
// dumps a full structural and functional analysis. This is for
// understanding WHAT the network learned, not benchmarking.
func TestIrisAnalysis(t *testing.T) {
	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.PopulationSize = 5 // 5 neurons per feature = 20 input neurons

	type networkState struct {
		net    *bio.Network
		layout Layout
	}

	numTrials := 3
	var bestNet *networkState
	var bestTestAcc float64

	for trial := 0; trial < numTrials; trial++ {
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

		for step := 0; step < 20000; step++ {
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
		t.Logf("Trial %d: train=%.1f%% test=%.1f%% dead=%d spikeRate=%.2f",
			trial, trainAcc, testAcc*100, dead, sr)

		if testAcc > bestTestAcc {
			bestTestAcc = testAcc
			bestNet = &networkState{net: net, layout: layout}
		}
	}

	t.Logf("\nBest trial: test=%.1f%%\n", bestTestAcc*100)
	t.Logf("Running full analysis on best network...\n")

	analysis := Analyze(bestNet.net, bestNet.layout, task, cfg)
	analysis.PrintReport(os.Stdout)
}
