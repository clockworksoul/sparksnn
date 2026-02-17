// Command xor runs the XOR benchmark against all three learning rules
// and prints a comparative report.
package main

import (
	"fmt"
	"os"

	"github.com/clockworksoul/biomimetic-network/benchmark"
	"github.com/clockworksoul/biomimetic-network/benchmark/xor"
	"github.com/clockworksoul/biomimetic-network/learning/predictive"
	"github.com/clockworksoul/biomimetic-network/learning/rstdp"
	"github.com/clockworksoul/biomimetic-network/learning/stdp"

	bio "github.com/clockworksoul/biomimetic-network"
)

func main() {
	cfg := xor.DefaultConfig()

	type trial struct {
		name    string
		rule    bio.LearningRule
		reward  bool // whether to inject reward signals
		tracker *benchmark.Tracker
	}

	stdpCfg := stdp.DefaultConfig()
	stdpCfg.APlus = 5
	stdpCfg.AMinus = 5
	stdpCfg.TauPlus = 8
	stdpCfg.TauMinus = 8
	stdpCfg.MaxWeightMagnitude = 500

	rstdpCfg := rstdp.DefaultConfig()

	predCfg := predictive.DefaultConfig()
	predCfg.LearningRate = 328
	predCfg.MaxWeightMagnitude = 500

	trials := []trial{
		{name: "No Learning", rule: bio.NoOpLearning{}},
		{name: "Pure STDP", rule: stdp.NewRule(stdpCfg)},
		{name: "R-STDP", rule: rstdp.NewRule(rstdpCfg), reward: true},
		{name: "Predictive", rule: predictive.NewRule(predCfg)},
	}

	for i := range trials {
		tr := &trials[i]
		fmt.Fprintf(os.Stderr, "\nRunning %s...\n", tr.name)

		if tr.reward {
			// R-STDP needs manual reward injection
			tr.tracker = runWithReward(tr.rule, tr.name, cfg)
		} else {
			tr.tracker = xor.Run(tr.rule, tr.name, cfg)
		}
	}

	// Summary comparison
	fmt.Println("\n========================================")
	fmt.Println("         XOR BENCHMARK SUMMARY")
	fmt.Println("========================================")
	fmt.Printf("%-15s %-12s %-12s\n", "Rule", "Best Acc", "Final Acc")
	fmt.Println("----------------------------------------")

	for _, tr := range trials {
		fmt.Printf("%-15s %-12.1f%% %-12.1f%%\n",
			tr.name,
			tr.tracker.BestAccuracy()*100,
			tr.tracker.LastAccuracy()*100,
		)
	}
	fmt.Println("========================================")
}

func runWithReward(rule bio.LearningRule, name string, cfg xor.NetworkConfig) *benchmark.Tracker {
	task := xor.Task{}
	net, layout := xor.BuildNetwork(cfg, rule)
	tracker := benchmark.NewTracker(10)

	// Baseline
	acc, dead, sr := xor.Evaluate(net, layout, task, cfg)
	weights := xor.CollectWeights(net, layout)
	wm, ws := benchmark.WeightStats(weights)
	tracker.Record(benchmark.Checkpoint{
		SamplesProcessed: 0,
		Accuracy:         acc,
		WeightMean:       wm,
		WeightStdDev:     ws,
		DeadNeurons:      dead,
		SpikeRate:        sr,
	})

	trainSamples := task.TrainingSamples()
	for i, sample := range trainSamples {
		spikeCounts := xor.PresentSample(net, layout, sample, cfg)
		predicted := xor.Classify(spikeCounts)

		if predicted == sample.Label {
			net.Reward(500)
		} else {
			net.Reward(-300)
		}

		if (i+1)%100 == 0 {
			acc, dead, sr := xor.Evaluate(net, layout, task, cfg)
			weights := xor.CollectWeights(net, layout)
			wm, ws := benchmark.WeightStats(weights)
			converged := tracker.Record(benchmark.Checkpoint{
				SamplesProcessed: i + 1,
				Accuracy:         acc,
				WeightMean:       wm,
				WeightStdDev:     ws,
				DeadNeurons:      dead,
				SpikeRate:        sr,
			})

			if converged {
				fmt.Fprintf(os.Stderr, "[%s] Converged at sample %d (acc=%.1f%%)\n",
					name, i+1, acc*100)
				break
			}
		}
	}

	tracker.PrintReport(os.Stdout, task.Name(), name)
	return tracker
}
