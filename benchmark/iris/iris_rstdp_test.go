package iris

import (
	"math/rand/v2"
	"os"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/rstdp"
)

// TestIrisRSTDP tests reward-modulated STDP on Iris with population
// coding. Previous attempts with 4 inputs failed — R-STDP couldn't
// learn because the input encoding was too coarse. Population coding
// should give local timing correlations meaningful information.
func TestIrisRSTDP(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.PopulationSize = 5
	cfg.LateralInhibition = -2000 // winner-take-all in output layer

	numTrials := 3
	var bestTestAcc float64
	var totalAcc float64

	type networkState struct {
		net    *bio.Network
		layout Layout
	}
	var bestNet *networkState

	for trial := 0; trial < numTrials; trial++ {
		rstdpCfg := rstdp.Config{
			APlus:                50,
			AMinus:               50,
			TauPlus:              10,
			TauMinus:             10,
			EligibilityDecayRate: 58982, // ~90% retention
			MaxWeightMagnitude:   5000,
			MultiplicativeReward: true,
			RewardRate:           0.10, // 10% of current weight
		}
		rule := rstdp.NewRule(rstdpCfg)

		net, layout := BuildNetwork(cfg, rule)

		trainSamples := task.TrainingSamples()
		maxEpochs := 200

		for epoch := 0; epoch < maxEpochs; epoch++ {
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

			trainAcc := float64(epochCorrect) / float64(len(trainSamples)) * 100

			if (epoch+1)%25 == 0 {
				testAcc, dead, sr := Evaluate(net, layout, task, cfg)
				t.Logf("  Trial %d, epoch %d: train=%.1f%% test=%.1f%% dead=%d spikeRate=%.2f",
					trial, epoch+1, trainAcc, testAcc*100, dead, sr)
			}

			if trainAcc > 95 {
				break
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
	t.Logf("\nRSTDP RESULT: best=%.1f%% avg=%.1f%% over %d trials",
		bestTestAcc*100, avgAcc*100, numTrials)

	// Run analysis on best network
	if bestNet != nil {
		t.Logf("\nAnalysis of best network:")
		analysis := Analyze(bestNet.net, bestNet.layout, task, cfg)
		analysis.PrintReport(os.Stdout)
	}

	if bestTestAcc <= 0.333 {
		t.Errorf("Best accuracy %.1f%% is no better than random", bestTestAcc*100)
	}
}
