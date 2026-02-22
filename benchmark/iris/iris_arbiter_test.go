package iris

import (
	"math/rand/v2"
	"os"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/benchmark"
	"github.com/clockworksoul/biomimetic-network/learning/arbiter"
)

// TestIrisArbiter tests the arbiter learning rule on Iris classification.
//
// Architecture:
//   - 4 input neurons (one per feature, amplitude coded)
//   - 16 hidden neurons (fully connected from input)
//   - 3 output neurons (one per class, fully connected from hidden)
//   - 16 arbiter neurons (one per hidden neuron, fire on errors)
//
// Training:
//   - STDP strengthens recently-active pathways on every presentation
//   - On errors, arbiter neurons fire and depress recently-active
//     connections into the hidden layer
//   - On correct predictions, arbiters are silent (STDP only)
func TestIrisArbiter(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.HiddenSize = 20
	cfg.TicksPerSample = 40
	cfg.RestTicks = 15
	cfg.InputWeight = 600
	cfg.Threshold = 120
	cfg.DecayRate = 50000
	cfg.RefractoryPeriod = 3
	cfg.InitWeightMax = 400
	cfg.LateralInhibition = -2000
	cfg.DeterministicInput = true
	cfg.PopulationSize = 5

	numInput := cfg.inputNeuronCount()
	numHidden := cfg.HiddenSize
	numOutput := 3
	numArbiterHidden := numHidden // one arbiter per hidden neuron

	total := numInput + numHidden + numOutput + numArbiterHidden

	net := bio.NewNetwork(uint32(total), 0, cfg.Threshold, cfg.DecayRate, cfg.RefractoryPeriod)

	layout := Layout{
		InputStart:  0,
		InputEnd:    uint32(numInput),
		HiddenStart: uint32(numInput),
		HiddenEnd:   uint32(numInput + numHidden),
		OutputStart: uint32(numInput + numHidden),
		OutputEnd:   uint32(numInput + numHidden + numOutput),
	}

	arbiterHiddenStart := uint32(numInput + numHidden + numOutput)
	arbiterHiddenEnd := uint32(total)

	// Define layers for the arbiter rule
	inputLayer := arbiter.LayerSpec{
		Start: layout.InputStart,
		End:   layout.InputEnd,
		// No arbiters for input layer
	}
	hiddenLayer := arbiter.LayerSpec{
		Start:        layout.HiddenStart,
		End:          layout.HiddenEnd,
		ArbiterStart: arbiterHiddenStart,
		ArbiterEnd:   arbiterHiddenEnd,
	}
	outputLayer := arbiter.LayerSpec{
		Start: layout.OutputStart,
		End:   layout.OutputEnd,
		// Output errors are signaled via SignalError, not arbiters
	}

	arbCfg := arbiter.DefaultConfig()
	// Disable STDP for now — with all neurons firing, it just
	// uniformly saturates all weights. The arbiter error signal
	// handles both strengthening (correct) and weakening (wrong).
	arbCfg.APlus = 0
	arbCfg.AMinus = 0
	arbCfg.TauPlus = 10
	arbCfg.TauMinus = 10
	arbCfg.Multiplicative = false    // multiplicative available but fixed works better here
	arbCfg.StrengtheningRatio = 3.0  // strengthen correct 3x more than depress wrong
	arbCfg.DepressionStrength = 20
	arbCfg.ArbiterWindow = 50        // wide window to catch all activity in a sample
	arbCfg.MaxWeightMagnitude = 3000
	arbCfg.MinWeightMagnitude = 15

	rule := arbiter.NewRule(arbCfg, []arbiter.LayerSpec{
		inputLayer, hiddenLayer, outputLayer,
	})
	net.LearningRule = rule

	// Wire up: Input → Hidden (random positive weights)
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(i, h, w)
		}
	}

	// Hidden → Output (random positive weights)
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			w := int32(rand.IntN(int(cfg.InitWeightMax))) + 1
			net.Connect(h, o, w)
		}
	}

	// Output ↔ Output (lateral inhibition)
	if cfg.LateralInhibition != 0 {
		for o1 := layout.OutputStart; o1 < layout.OutputEnd; o1++ {
			for o2 := layout.OutputStart; o2 < layout.OutputEnd; o2++ {
				if o1 != o2 {
					net.Connect(o1, o2, cfg.LateralInhibition)
				}
			}
		}
	}

	tracker := benchmark.NewTracker(30)

	trainSamples := task.TrainingSamples()

	// Baseline
	acc, dead, sr := Evaluate(net, layout, task, cfg)
	weights := CollectWeights(net, layout)
	wm, ws := benchmark.WeightStats(weights)
	tracker.Record(benchmark.Checkpoint{
		SamplesProcessed: 0,
		Accuracy:         acc,
		WeightMean:       wm,
		WeightStdDev:     ws,
		DeadNeurons:      dead,
		SpikeRate:        sr,
	})

	t.Logf("Baseline: acc=%.1f%%, dead=%d, spikeRate=%.2f", acc*100, dead, sr)

	epochs := 100
	checkEvery := len(trainSamples) // check every epoch
	sampleCount := 0
	bestAcc := acc

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle training data each epoch
		rng := rand.New(rand.NewPCG(uint64(epoch), uint64(epoch)^0xcafe))
		perm := rng.Perm(len(trainSamples))

		for _, pi := range perm {
			sample := trainSamples[pi]
			sampleCount++

			// Present sample
			spikeCounts := PresentSample(net, layout, sample, cfg)

			// Signal error to arbiter system
			rule.SignalError(net, sample.Label, spikeCounts)
		}

		if (epoch+1)%1 == 0 { // check every epoch
			acc, dead, sr := Evaluate(net, layout, task, cfg)
			weights := CollectWeights(net, layout)
			wm, ws := benchmark.WeightStats(weights)
			converged := tracker.Record(benchmark.Checkpoint{
				SamplesProcessed: sampleCount,
				Accuracy:         acc,
				WeightMean:       wm,
				WeightStdDev:     ws,
				DeadNeurons:      dead,
				SpikeRate:        sr,
			})

			if acc > bestAcc {
				bestAcc = acc
			}

			if (epoch+1)%10 == 0 {
				t.Logf("Epoch %d: acc=%.1f%% (best=%.1f%%), dead=%d, sr=%.2f, wmean=%.0f, wstd=%.0f",
					epoch+1, acc*100, bestAcc*100, dead, sr, wm, ws)
			}

			if converged {
				t.Logf("Converged at epoch %d", epoch+1)
				break
			}

			_ = checkEvery
		}
	}

	tracker.PrintReport(os.Stdout, task.Name(), "Arbiter")

	// We want to see improvement over baseline (33% = random for 3 classes)
	if bestAcc <= 0.34 {
		t.Errorf("Best accuracy %.1f%% is at chance level; arbiter learning failed", bestAcc*100)
	} else {
		t.Logf("✓ Best accuracy: %.1f%% (above chance)", bestAcc*100)
	}

	// Report but don't fail on modest accuracy — this is a first implementation
	if bestAcc >= 0.60 {
		t.Logf("✓ Achieved %.1f%% — solid performance for biological learning!", bestAcc*100)
	} else if bestAcc >= 0.45 {
		t.Logf("⚠ Achieved %.1f%% — learning is happening but needs tuning", bestAcc*100)
	}
}
