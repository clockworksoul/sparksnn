package iris

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/benchmark"
	"github.com/clockworksoul/biomimetic-network/learning/arbiter"
)

// TestIrisThreePhase tests the three-phase training methodology:
//
//	Phase 1: Strengthen all firing connections (Hebbian reinforcement)
//	Phase 2: Arbiter corrects misfired connections (error depression)
//	Phase 3: Hard reset all neuron activations to baseline
//
// This separates positive reinforcement from error correction and
// prevents compounding excitation across samples.
func TestIrisThreePhase(t *testing.T) {
	task := NewTask(42)
	cfg := DefaultConfig()
	cfg.HiddenSize = 20
	cfg.TicksPerSample = 40
	cfg.RestTicks = 15
	cfg.InputWeight = 600 << 8       // 153,600
	cfg.Threshold = 120 << 8         // 30,720
	cfg.DecayRate = 50000
	cfg.RefractoryPeriod = 3
	cfg.InitWeightMax = 400 << 8     // 102,400
	cfg.LateralInhibition = -2000 << 8 // -512,000
	cfg.DeterministicInput = true
	cfg.PopulationSize = 10
	cfg.NoiseWeight = 100 << 8       // 25,600 — scale noise with everything else

	numInput := cfg.inputNeuronCount()
	numHidden := cfg.HiddenSize
	numOutput := 3
	numArbiterHidden := numHidden

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

	inputLayer := arbiter.LayerSpec{
		Start: layout.InputStart,
		End:   layout.InputEnd,
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
	}

	arbCfg := arbiter.DefaultConfig()
	// Disable STDP — three-phase handles reinforcement explicitly
	arbCfg.APlus = 0
	arbCfg.AMinus = 0
	arbCfg.TauPlus = 10
	arbCfg.TauMinus = 10
	arbCfg.Multiplicative = true
	arbCfg.CorrectionRate = 0.15
	arbCfg.StrengtheningRatio = 1.0 // equal magnitude; frequency will differ
	arbCfg.DepressionStrength = 20 << 8 // 5,120 — fallback for non-multiplicative
	arbCfg.ArbiterWindow = 50
	arbCfg.MaxWeightMagnitude = 3000 << 8 // 768,000
	arbCfg.MinWeightMagnitude = 0

	rule := arbiter.NewRule(arbCfg, []arbiter.LayerSpec{
		inputLayer, hiddenLayer, outputLayer,
	})
	net.LearningRule = rule

	// Wire up: Input → Hidden
	// Sparse (50%), mixed sign — gives each hidden neuron a unique receptive field.
	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			if rand.Float64() > 0.5 {
				continue
			}
			w := int32(rand.IntN(int(cfg.InitWeightMax)*2)) - cfg.InitWeightMax
			if w == 0 {
				w = 1
			}
			net.Connect(i, h, w)
		}
	}

	// Hidden → Output
	// Sparse (70%), mixed sign.
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		for o := layout.OutputStart; o < layout.OutputEnd; o++ {
			if rand.Float64() > 0.7 {
				continue
			}
			w := int32(rand.IntN(int(cfg.InitWeightMax)*2)) - cfg.InitWeightMax
			if w == 0 {
				w = 1
			}
			net.Connect(h, o, w)
		}
	}

	// Lateral inhibition
	if cfg.LateralInhibition != 0 {
		for o1 := layout.OutputStart; o1 < layout.OutputEnd; o1++ {
			for o2 := layout.OutputStart; o2 < layout.OutputEnd; o2++ {
				if o1 != o2 {
					net.Connect(o1, o2, cfg.LateralInhibition)
				}
			}
		}
	}

	tracker := benchmark.NewTracker(80) // more patience — we're decaying the rate
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

	epochs := 200
	sampleCount := 0
	bestAcc := acc

	// Learning rate schedule: exponential decay from initial to min.
	// Big steps early (explore), small steps later (fine-tune).
	// Analogous to biological critical periods.
	initialRate := arbCfg.CorrectionRate // 0.15
	minRate := 0.005                      // floor — never stop learning entirely
	// Decay factor per epoch: we want to reach ~minRate around epoch 100.
	// decay^100 = minRate/initialRate → decay = (minRate/initialRate)^(1/100)
	decayFactor := math.Pow(minRate/initialRate, 1.0/100.0)

	for epoch := 0; epoch < epochs; epoch++ {
		// Decay the correction rate
		currentRate := initialRate * math.Pow(decayFactor, float64(epoch))
		if currentRate < minRate {
			currentRate = minRate
		}
		rule.Config.CorrectionRate = currentRate

		rng := rand.New(rand.NewPCG(uint64(epoch), uint64(epoch)^0xcafe))
		perm := rng.Perm(len(trainSamples))

		for _, pi := range perm {
			sample := trainSamples[pi]
			sampleCount++

			// === THREE-PHASE TRAINING ===

			// Present sample
			spikeCounts := PresentSample(net, layout, sample, cfg)

			// Phase 1: Strengthen active connections (only on correct predictions)
			rule.StrengthenActive(net, sample.Label, spikeCounts)

			// Phase 2: Arbiter corrects misfired connections (only on errors)
			rule.CorrectErrors(net, sample.Label, spikeCounts)

			// Phase 3: Hard reset all activations to baseline
			net.ResetActivation()
		}

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
			t.Logf("Epoch %d: acc=%.1f%% (best=%.1f%%), dead=%d, sr=%.2f, wmean=%.0f, wstd=%.0f, rate=%.4f",
				epoch+1, acc*100, bestAcc*100, dead, sr, wm, ws, currentRate)
		}

		if converged {
			t.Logf("Converged at epoch %d", epoch+1)
			break
		}
	}

	tracker.PrintReport(os.Stdout, task.Name(), "ThreePhase")

	if bestAcc <= 0.34 {
		t.Errorf("Best accuracy %.1f%% is at chance level; three-phase learning failed", bestAcc*100)
	} else {
		t.Logf("✓ Best accuracy: %.1f%% (above chance)", bestAcc*100)
	}

	if bestAcc >= 0.60 {
		t.Logf("✓ Achieved %.1f%% — solid performance!", bestAcc*100)
	} else if bestAcc >= 0.45 {
		t.Logf("⚠ Achieved %.1f%% — learning is happening but needs tuning", bestAcc*100)
	}
}
