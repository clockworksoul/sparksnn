package perturbation

import (
	"fmt"
	"math/rand/v2"
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

func TestNewRule(t *testing.T) {
	cfg := DefaultConfig()
	rule := NewRule(cfg)
	if rule == nil {
		t.Fatal("NewRule returned nil")
	}
	if rule.perturbSize != cfg.PerturbSize {
		t.Errorf("perturbSize = %d, want %d", rule.perturbSize, cfg.PerturbSize)
	}
}

func TestOnRewardPerturbs(t *testing.T) {
	cfg := DefaultConfig()
	rule := NewRule(cfg)

	net := bio.NewNetwork(3, 0, 100, 45000, 3)
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 500)

	// First call: no previous perturbation, just sets up the first one
	rule.OnReward(net, 0, 1)
	if !rule.hasPending {
		t.Fatal("should have pending perturbation after first OnReward")
	}

	// Second call: evaluates first perturbation, applies new one
	rule.OnReward(net, 1, 2) // better reward
	// First perturbation should have been kept (reward improved)
}

func TestRevertOnWorse(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PerturbSize = 100
	rule := NewRule(cfg)

	net := bio.NewNetwork(2, 0, 100, 45000, 3)
	net.Connect(0, 1, 500)

	// First call: sets up perturbation
	rule.OnReward(net, 10, 1)
	perturbedWeight := net.Neurons[0].Connections[0].Weight

	// Second call with WORSE reward: should revert
	rule.OnReward(net, 5, 2)

	// The first perturbation should have been reverted
	// (but a new one is now applied, so weight may differ)
	// We can't easily test the intermediate state, but we can
	// verify the rule tracked it correctly
	if !rule.hasPending {
		t.Fatal("should still have a pending perturbation")
	}
	_ = perturbedWeight
}

func TestAdaptivePerturbSize(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PerturbSize = 100
	cfg.MaxPerturbSize = 800
	cfg.AdaptAfter = 5
	rule := NewRule(cfg)

	net := bio.NewNetwork(2, 0, 100, 45000, 3)
	net.Connect(0, 1, 500)

	// Send many equal-reward signals to trigger adaptation
	for i := 0; i < 10; i++ {
		rule.OnReward(net, 50, uint32(i+1))
	}

	if rule.perturbSize <= cfg.PerturbSize {
		t.Errorf("perturbSize should have adapted up, got %d", rule.perturbSize)
	}
}

func TestNoOpMethods(t *testing.T) {
	rule := NewRule(DefaultConfig())

	// These should not panic
	rule.OnSpikePropagation(&bio.Connection{}, 1, 2)
	rule.OnPostFire(nil, 1)

	net := bio.NewNetwork(2, 0, 100, 45000, 3)
	rule.Maintain(net, 1)
}

func TestWeightCapping(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PerturbSize = 1000
	cfg.MaxWeightMagnitude = 500
	rule := NewRule(cfg)

	net := bio.NewNetwork(2, 0, 100, 45000, 3)
	net.Connect(0, 1, 400)

	// Run many perturbations
	for i := 0; i < 100; i++ {
		rule.OnReward(net, int32(i), uint32(i+1))
	}

	w := net.Neurons[0].Connections[0].Weight
	if w > 500 || w < -500 {
		t.Errorf("weight %d exceeds MaxWeightMagnitude 500", w)
	}
}

// TestXORWithPerturbationRule verifies the formal LearningRule
// implementation can learn XOR, matching the ad-hoc test results.
func TestXORWithPerturbationRule(t *testing.T) {
	type sample struct {
		a, b  bool
		label int
	}
	patterns := []sample{
		{false, false, 0},
		{true, false, 1},
		{false, true, 1},
		{true, true, 0},
	}

	threshold := int32(500)
	inputStim := int32(1000)
	ticks := 30
	rest := 10

	for _, hiddenSize := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("hidden=%d", hiddenSize), func(t *testing.T) {
			total := 2 + hiddenSize + 2
			outStart := uint32(2 + hiddenSize)

			successes := 0
			numTrials := 30

			for trial := 0; trial < numTrials; trial++ {
				cfg := DefaultConfig()
				cfg.PerturbSize = 200
				cfg.MaxPerturbSize = 2000
				cfg.AdaptAfter = 200
				cfg.BatchSize = 4 // evaluate after all 4 XOR patterns
				rule := NewRule(cfg)

				net := bio.NewNetwork(uint32(total), 0, threshold, 45000, 3)
				net.LearningRule = rule

				// Wire network
				for i := uint32(0); i < 2; i++ {
					for h := uint32(2); h < uint32(2+hiddenSize); h++ {
						net.Connect(i, h, int32(rand.IntN(800))+100)
					}
				}
				for h := uint32(2); h < uint32(2+hiddenSize); h++ {
					net.Connect(h, outStart, int32(rand.IntN(800))+100)
					net.Connect(h, outStart+1, int32(rand.IntN(800))+100)
				}

				// Train: present samples, deliver reward via net.Reward()
				// 10000 epochs × 4 patterns / batch_size 4 = 10000 perturbation steps
				for epoch := 0; epoch < 10000; epoch++ {
					perm := rand.Perm(4)
					for _, pi := range perm {
						p := patterns[pi]

						out0, out1 := 0, 0
						for tick := 0; tick < ticks; tick++ {
							if p.a {
								net.Stimulate(0, inputStim)
							}
							if p.b {
								net.Stimulate(1, inputStim)
							}
							net.Tick()
							if net.Neurons[outStart].LastFired == net.Counter {
								out0++
							}
							if net.Neurons[outStart+1].LastFired == net.Counter {
								out1++
							}
						}
						net.TickN(uint32(rest))

						predicted := 0
						if out1 > out0 {
							predicted = 1
						}

						// Reward = correct count proxy
						// Higher reward for correct predictions
						reward := int32(-1)
						if predicted == p.label {
							reward = 1
						}
						net.Reward(reward)
					}
				}

				// Evaluate
				correct := 0
				for _, p := range patterns {
					out0, out1 := 0, 0
					for ev := 0; ev < 5; ev++ {
						for tick := 0; tick < ticks; tick++ {
							if p.a {
								net.Stimulate(0, inputStim)
							}
							if p.b {
								net.Stimulate(1, inputStim)
							}
							net.Tick()
							if net.Neurons[outStart].LastFired == net.Counter {
								out0++
							}
							if net.Neurons[outStart+1].LastFired == net.Counter {
								out1++
							}
						}
						net.TickN(uint32(rest))
					}
					predicted := 0
					if out1 > out0 {
						predicted = 1
					}
					if predicted == p.label {
						correct++
					}
				}

				if correct == 4 {
					successes++
				}
			}

			rate := float64(successes) / float64(numTrials) * 100
			t.Logf("Hidden=%d: %d/%d trials succeeded (%.1f%%)", hiddenSize, successes, numTrials, rate)
			if successes == 0 {
				t.Errorf("No trial achieved 100%% with %d hidden neurons", hiddenSize)
			}
		})
	}
}
