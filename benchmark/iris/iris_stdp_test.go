package iris

import (
	"math/rand/v2"
	"os"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/learning/stdp"
)

// TestIrisSTDP tests pure (unsupervised) STDP on Iris with
// population coding. STDP has no reward signal — it only
// strengthens causal timing (pre→post) and weakens anti-causal.
// The question: can Hebbian learning alone organize weights into
// something that separates classes?
func TestIrisSTDP(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.PopulationSize = 5

	numTrials := 3
	var bestTestAcc float64
	var totalAcc float64

	type networkState struct {
		net    *bio.Network
		layout Layout
	}
	var bestNet *networkState

	for trial := 0; trial < numTrials; trial++ {
		stdpCfg := stdp.Config{
			APlus:              30,
			AMinus:             30,
			TauPlus:            10,
			TauMinus:           10,
			MaxWeightMagnitude: 5000,
		}
		rule := stdp.NewRule(stdpCfg)

		net, layout := BuildNetwork(cfg, rule)

		trainSamples := task.TrainingSamples()
		maxEpochs := 200

		for epoch := 0; epoch < maxEpochs; epoch++ {
			rand.Shuffle(len(trainSamples), func(i, j int) {
				trainSamples[i], trainSamples[j] = trainSamples[j], trainSamples[i]
			})

			for _, sample := range trainSamples {
				PresentSample(net, layout, sample, cfg)
				// No reward — pure unsupervised
			}

			if (epoch+1)%25 == 0 {
				testAcc, dead, sr := Evaluate(net, layout, task, cfg)
				t.Logf("  Trial %d, epoch %d: test=%.1f%% dead=%d spikeRate=%.2f",
					trial, epoch+1, testAcc*100, dead, sr)
			}
		}

		testAcc, dead, sr := Evaluate(net, layout, task, cfg)
		totalAcc += testAcc
		t.Logf("Trial %d: test=%.1f%% dead=%d spikeRate=%.2f",
			trial, testAcc*100, dead, sr)

		if testAcc > bestTestAcc {
			bestTestAcc = testAcc
			bestNet = &networkState{net: net, layout: layout}
		}
	}

	avgAcc := totalAcc / float64(numTrials)
	t.Logf("\nSTDP RESULT: best=%.1f%% avg=%.1f%% over %d trials",
		bestTestAcc*100, avgAcc*100, numTrials)

	if bestNet != nil {
		t.Logf("\nAnalysis of best network:")
		analysis := Analyze(bestNet.net, bestNet.layout, task, cfg)
		analysis.PrintReport(os.Stdout)
	}
}
