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
	cfg.HiddenSize = 2            // minimal — matches proven handwired/learned topology
	cfg.UseInhibition = false     // no lateral inhibition — learning creates competition
	cfg.DeterministicInput = true // clean signal, no rate coding noise
	cfg.TicksPerSample = 50      // match minimal test
	cfg.RestTicks = 20

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
		{name: "R-STDP+Asym", rule: rstdp.NewRule(rstdpCfg), reward: true},
		{name: "Asymmetric", rule: bio.NoOpLearning{}, reward: true},
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

		// Standard R-STDP reward
		if predicted == sample.Label {
			net.Reward(500)
		} else {
			net.Reward(-300)
		}

		// Asymmetric activity-reward update: the symmetry-breaking
		// key to learning inhibitory connections. For each learnable
		// connection, if source fired but target didn't, push weight
		// opposite to reward. This creates differentiation.
		reward := int32(300)
		if predicted != sample.Label {
			reward = -200
		}
		activityWindow := uint32(cfg.TicksPerSample + cfg.RestTicks + 10)
		applyAsymmetricUpdate(net, layout, reward, activityWindow)

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

// applyAsymmetricUpdate applies the symmetry-breaking activity-reward
// learning rule to all learnable connections (input→hidden, hidden→output).
// When source fired but target didn't, weight moves OPPOSITE to reward.
// This is the key mechanism for discovering inhibitory connections.
func applyAsymmetricUpdate(net *bio.Network, layout xor.Layout, reward int32, window uint32) {
	now := net.Counter
	maxW := int32(5000) // cap weight magnitude

	isRecentlyActive := func(idx uint32) bool {
		n := &net.Neurons[idx]
		return n.LastFired > 0 && now-n.LastFired < window
	}

	clampWeight := func(w int32) int32 {
		if w > maxW {
			return maxW
		}
		if w < -maxW {
			return -maxW
		}
		return w
	}

	// Update input→hidden connections
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		srcActive := isRecentlyActive(i)
		for j := range net.Neurons[i].Connections {
			conn := &net.Neurons[i].Connections[j]
			if conn.Target < layout.HiddenStart || conn.Target >= layout.HiddenEnd {
				continue
			}
			tgtActive := isRecentlyActive(conn.Target)

			if srcActive && tgtActive {
				conn.Weight = clampWeight(bio.ClampAdd(conn.Weight, reward/4))
			} else if srcActive && !tgtActive {
				conn.Weight = clampWeight(bio.ClampAdd(conn.Weight, -reward/8))
			}
		}
	}

	// Update hidden→output connections
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		srcActive := isRecentlyActive(h)
		for j := range net.Neurons[h].Connections {
			conn := &net.Neurons[h].Connections[j]
			if conn.Target < layout.OutputStart || conn.Target >= layout.OutputEnd {
				continue
			}
			tgtActive := isRecentlyActive(conn.Target)

			if srcActive && tgtActive {
				conn.Weight = clampWeight(bio.ClampAdd(conn.Weight, reward/4))
			} else if srcActive && !tgtActive {
				conn.Weight = clampWeight(bio.ClampAdd(conn.Weight, -reward/8))
			}
		}
	}
}
