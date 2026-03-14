package mnist

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/surrogate"
)

// TestMNISTDecaySweep runs the tuned MNIST architecture with different
// membrane decay rates to find the optimal β value.
//
// Architecture: 784 → 512 hidden (30% sparse) → 10 output
// Fixed 20 epochs per decay value for comparison.
//
// Current default: β = 50000/65536 ≈ 0.763 (chosen by intuition).
// Lower β = leakier = stronger coincidence detection.
// Higher β = more integration = temporal averaging.
func TestMNISTDecaySweep(t *testing.T) {
	task, err := NewTask(0, 0)
	if err != nil {
		t.Fatalf("Failed to load MNIST: %v", err)
	}

	t.Logf("Loaded %d training, %d test samples",
		len(task.TrainingSamples()), len(task.TestSamples()))

	type decayConfig struct {
		name      string
		decayRate uint16
		beta      float64
	}

	configs := []decayConfig{
		{"β=0.50 (very leaky)", 32768, 32768.0 / 65536.0},
		{"β=0.61 (leaky)", 40000, 40000.0 / 65536.0},
		{"β=0.76 (current default)", 50000, 50000.0 / 65536.0},
		{"β=0.85 (moderate)", 55706, 55706.0 / 65536.0},
		{"β=0.95 (slow decay)", 62259, 62259.0 / 65536.0},
	}

	trainSamples := task.TrainingSamples()
	testSamples := task.TestSamples()

	encodeInput := func(pixels []byte) []float64 {
		values := make([]float64, len(pixels))
		for i, p := range pixels {
			values[i] = float64(p) / 255.0
		}
		return values
	}

	t.Logf("\n========== DECAY RATE SWEEP ==========")
	t.Logf("Architecture: 784 → 512 (30%% sparse) → 10")
	t.Logf("Fixed 20 epochs per config, Adam + LR decay every 15")
	t.Logf("======================================\n")

	type result struct {
		name    string
		beta    float64
		bestAcc float64
		accAt5  float64
		accAt10 float64
		accAt15 float64
		accAt20 float64
	}
	var results []result

	for _, dc := range configs {
		t.Logf("\n--- Running: %s (uint16=%d) ---", dc.name, dc.decayRate)

		numInput := 784
		numHidden := 512
		numOutput := 10
		total := numInput + numHidden + numOutput

		threshold := 1.0
		inputWeight := 0.5
		initWeightMax := 0.2
		intScale := float64(1 << 20)

		intThreshold := int32(threshold * intScale)
		net := bio.NewNetwork(uint32(total), 0, intThreshold, dc.decayRate, 3)
		net.LearningRule = bio.NoOpLearning{}

		inputStart := uint32(0)
		inputEnd := uint32(numInput)
		hiddenStart := uint32(numInput)
		hiddenEnd := uint32(numInput + numHidden)
		outputStart := uint32(numInput + numHidden)
		outputEnd := uint32(total)

		// Same seed for every config so topology is identical
		rng := rand.New(rand.NewPCG(42, 42^0xbeef))

		for i := inputStart; i < inputEnd; i++ {
			for h := hiddenStart; h < hiddenEnd; h++ {
				if rng.Float64() > 0.30 {
					continue
				}
				wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
				w := int32(math.Round(wf * intScale))
				if w == 0 {
					w = 1
				}
				net.Connect(i, h, w)
			}
		}

		for h := hiddenStart; h < hiddenEnd; h++ {
			for o := outputStart; o < outputEnd; o++ {
				if rng.Float64() > 0.60 {
					continue
				}
				wf := (rng.Float64()*2.0 - 1.0) * initWeightMax
				w := int32(math.Round(wf * intScale))
				if w == 0 {
					w = 1
				}
				net.Connect(h, o, w)
			}
		}

		baseLR := 0.001

		cfg := surrogate.Config{
			LearningRate: baseLR,
			NumSteps:     40,
			Surrogate:    surrogate.DefaultFastSigmoid(),
			Layers: []surrogate.LayerSpec{
				{Start: inputStart, End: inputEnd},
				{Start: hiddenStart, End: hiddenEnd},
				{Start: outputStart, End: outputEnd},
			},
			Beta:        dc.beta,
			InputWeight: inputWeight,
			Threshold:   threshold,
		}

		trainer := surrogate.NewTrainer(net, cfg, intScale)
		trainer.EnableAdam()

		r := result{name: dc.name, beta: dc.beta}
		epochs := 20

		for epoch := 0; epoch < epochs; epoch++ {
			lr := baseLR * math.Pow(0.5, float64(epoch/15))
			trainer.Config.LearningRate = lr

			perm := rng.Perm(len(trainSamples))

			totalLoss := 0.0
			for _, pi := range perm {
				sample := trainSamples[pi]
				inputValues := encodeInput(sample.Inputs)
				loss := trainer.TrainSample(inputValues, sample.Label)
				totalLoss += loss
			}
			avgLoss := totalLoss / float64(len(trainSamples))

			correct := 0
			for _, sample := range testSamples {
				inputValues := encodeInput(sample.Inputs)
				predicted := trainer.Predict(inputValues)
				if predicted == sample.Label {
					correct++
				}
			}

			acc := float64(correct) / float64(len(testSamples))
			if acc > r.bestAcc {
				r.bestAcc = acc
			}

			switch epoch + 1 {
			case 5:
				r.accAt5 = acc
			case 10:
				r.accAt10 = acc
			case 15:
				r.accAt15 = acc
			case 20:
				r.accAt20 = acc
			}

			t.Logf("  Epoch %d: acc=%.2f%% (best=%.2f%%), avgLoss=%.4f, lr=%.6f",
				epoch+1, acc*100, r.bestAcc*100, avgLoss, lr)
		}

		results = append(results, r)
		t.Logf("--- %s: best=%.2f%% ---\n", dc.name, r.bestAcc*100)
	}

	// Summary table
	t.Logf("\n========== SUMMARY ==========")
	t.Logf("%-30s  %7s  %7s  %7s  %7s  %7s",
		"Config", "Ep5", "Ep10", "Ep15", "Ep20", "Best")
	t.Logf("%-30s  %7s  %7s  %7s  %7s  %7s",
		"------", "---", "----", "----", "----", "----")
	bestOverall := 0.0
	bestName := ""
	for _, r := range results {
		t.Logf("%-30s  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%",
			r.name, r.accAt5*100, r.accAt10*100, r.accAt15*100, r.accAt20*100, r.bestAcc*100)
		if r.bestAcc > bestOverall {
			bestOverall = r.bestAcc
			bestName = r.name
		}
	}
	t.Logf("\nWinner: %s (%.2f%%)", bestName, bestOverall*100)
	t.Logf("Current default (β=0.76): %.2f%%",
		results[2].bestAcc*100)
	diff := bestOverall - results[2].bestAcc
	t.Logf("Improvement over default: %+.2f%%", diff*100)

	if diff > 0.005 {
		t.Logf("🏆 Found a better decay rate! Re-run full training with %s", bestName)
	} else if diff > 0 {
		t.Logf("✓ Marginal improvement — may not be worth re-running")
	} else {
		t.Logf("✓ Current default is optimal (or tied)")
	}

	// Also log as CSV for easy parsing
	t.Logf("\n--- CSV ---")
	t.Logf("beta,ep5,ep10,ep15,ep20,best")
	for _, r := range results {
		t.Logf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
			r.beta, r.accAt5, r.accAt10, r.accAt15, r.accAt20, r.bestAcc)
	}
	_ = fmt.Sprintf("") // avoid unused import
}
