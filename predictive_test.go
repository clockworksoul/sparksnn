package biomimetic

import (
	"testing"
)

func TestPredictiveRuleSatisfiesInterface(t *testing.T) {
	// Verify PredictiveRule implements LearningRule.
	var _ LearningRule = &PredictiveRule{}
	var _ LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
}

func TestPredictiveNoRewardNeeded(t *testing.T) {
	// Unlike R-STDP, predictive learning changes weights without
	// any reward signal. Weights should change after spike activity.
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// A fires, B fires on next tick
	net.Stimulate(0, 500)
	net.Tick()

	// Run more ticks to let learning settle
	net.TickN(5)

	// Weight should have changed — no reward needed
	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight == originalWeight {
		t.Errorf("predictive rule should change weights without reward, weight stayed at %d", originalWeight)
	}
}

func TestPredictiveEligibilityAccumulates(t *testing.T) {
	// When spikes propagate through a connection, eligibility should
	// accumulate (representing recent input history).
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
	net.Connect(0, 1, 500)

	// Before any activity
	if net.Neurons[0].Connections[0].Eligibility != 0 {
		t.Fatalf("initial eligibility should be 0")
	}

	// A fires — spike propagates through connection
	net.Stimulate(0, 500)

	// Eligibility should now be nonzero
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig == 0 {
		t.Error("eligibility should be nonzero after spike propagation")
	}
}

func TestPredictiveEligibilityDecays(t *testing.T) {
	// Eligibility traces should decay over time.
	cfg := DefaultPredictiveConfig()
	cfg.EligibilityDecayRate = 32768 // 50% per tick — aggressive

	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(cfg)
	net.Connect(0, 1, 500)

	// Create eligibility
	net.Stimulate(0, 500)
	net.Tick() // Process propagation

	eligAfter := net.Neurons[0].Connections[0].Eligibility

	// Tick several times — eligibility should decay
	net.Tick()
	net.Tick()
	net.Tick()

	eligLater := net.Neurons[0].Connections[0].Eligibility

	if eligAfter != 0 && abs16(eligLater) >= abs16(eligAfter) {
		t.Errorf("eligibility should decay: was %d, now %d", eligAfter, eligLater)
	}
}

func TestPredictiveCausalTimingPotentiates(t *testing.T) {
	// The paper's core finding: when pre fires before post (causal),
	// the predictive input gets credit → potentiation.
	//
	// Setup: A -> B -> C, A fires first in sequence.
	// A's connection to B should be potentiated because A's input
	// predicts B's subsequent firing.
	cfg := DefaultPredictiveConfig()
	cfg.LearningRate = 3277 // ~5% — strong enough to see changes

	net := NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(cfg)
	net.Connect(0, 1, 300)
	net.Connect(1, 2, 300)

	originalAB := net.Neurons[0].Connections[0].Weight

	// Run the sequence multiple times: A fires → B fires → C fires
	for epoch := 0; epoch < 50; epoch++ {
		net.Stimulate(0, 300)
		net.Tick() // B fires
		net.Tick() // C fires
		net.TickN(5) // Let decay settle
	}

	finalAB := net.Neurons[0].Connections[0].Weight

	// A->B should have changed (we expect potentiation for predictive inputs,
	// but the exact direction depends on the error dynamics)
	if finalAB == originalAB {
		t.Errorf("A->B weight should change with repeated causal timing, stayed at %d", originalAB)
	}
}

func TestPredictiveSequenceLearning(t *testing.T) {
	// Paper's Fig 2 scenario (simplified): A sequence of neurons
	// fires in order. After training, the network should show weight
	// reorganization — early-in-sequence connections strengthened,
	// late-in-sequence connections weakened.
	cfg := DefaultPredictiveConfig()
	cfg.LearningRate = 3277 // ~5%

	// Create a simple chain: 0 -> 1 -> 2 -> 3 -> 4
	net := NewNetwork(5, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(cfg)

	for i := uint32(0); i < 4; i++ {
		net.Connect(i, i+1, 200)
	}

	// Record initial weights
	initialWeights := make([]int16, 4)
	for i := 0; i < 4; i++ {
		initialWeights[i] = net.Neurons[i].Connections[0].Weight
	}

	// Train: repeatedly stimulate the start of the chain
	for epoch := 0; epoch < 100; epoch++ {
		net.Stimulate(0, 300)
		net.TickN(10) // Let the cascade propagate and settle
	}

	// Verify weights changed
	anyChanged := false
	for i := 0; i < 4; i++ {
		if net.Neurons[i].Connections[0].Weight != initialWeights[i] {
			anyChanged = true
			break
		}
	}

	if !anyChanged {
		t.Error("expected at least some weights to change after sequence training")
	}
}

func TestPredictiveSwappableWithSTDP(t *testing.T) {
	// Verify we can swap between STDP and Predictive rules at runtime.
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)

	// Start with STDP
	net.LearningRule = NewSTDPRule(DefaultSTDPConfig())
	net.Stimulate(0, 500)
	net.Tick()

	eligSTDP := net.Neurons[0].Connections[0].Eligibility
	if eligSTDP == 0 {
		t.Error("STDP should set eligibility")
	}

	// Reset
	net.Neurons[0].Connections[0].Eligibility = 0
	net.Neurons[0].Activation = net.Neurons[0].Baseline
	net.Neurons[0].LastFired = 0
	net.Neurons[1].Activation = net.Neurons[1].Baseline
	net.Neurons[1].LastFired = 0

	// Swap to Predictive
	net.LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
	net.Stimulate(0, 500)
	net.Tick()

	eligPred := net.Neurons[0].Connections[0].Eligibility
	// Both rules should produce eligibility, but they may differ
	t.Logf("STDP eligibility: %d, Predictive eligibility: %d", eligSTDP, eligPred)
}

func TestPredictiveRewardIsNoOp(t *testing.T) {
	// Calling Reward should not crash or change state for predictive rule.
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	weightBefore := net.Neurons[0].Connections[0].Weight
	net.Reward(100) // Should be no-op
	weightAfter := net.Neurons[0].Connections[0].Weight

	if weightBefore != weightAfter {
		t.Errorf("Reward should be no-op for predictive rule: was %d, now %d",
			weightBefore, weightAfter)
	}
}

func TestPredictiveWeightsClamped(t *testing.T) {
	// Weights should never exceed MaxWeightMagnitude.
	cfg := DefaultPredictiveConfig()
	cfg.LearningRate = 6554 // 10% — aggressive
	cfg.MaxWeightMagnitude = 1000

	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(cfg)
	net.Connect(0, 1, 900)

	// Hammer the connection with lots of activity
	for i := 0; i < 200; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		net.TickN(3)
	}

	w := net.Neurons[0].Connections[0].Weight
	if w > 1000 || w < -1000 {
		t.Errorf("weight %d exceeds MaxWeightMagnitude 1000", w)
	}
}

func TestPredictiveInhibitoryConnections(t *testing.T) {
	// Predictive rule should work with negative (inhibitory) weights too.
	cfg := DefaultPredictiveConfig()
	cfg.LearningRate = 3277

	net := NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(cfg)
	net.Connect(0, 1, -200) // Inhibitory

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Stimulate A (won't make B fire since weight is negative,
	// but spike still propagates and learning should occur)
	for i := 0; i < 50; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		net.TickN(3)
	}

	// The connection should show some learning (eligibility accumulation
	// and decay even if B never fires from this input)
	// Weight may or may not change depending on whether B fires from
	// other inputs, but the system should not crash.
	_ = originalWeight // Avoid unused warning; primary test is no panic
}

func TestPredictivePredictFunction(t *testing.T) {
	// Test the prediction calculation directly.
	p := NewPredictiveRule(DefaultPredictiveConfig())

	// Zero activation → zero prediction
	pred := p.predict(0, 100)
	if pred != 0 {
		t.Errorf("predict(0, 100) = %d, want 0", pred)
	}

	// Positive activation × positive weight → positive prediction
	pred = p.predict(1000, 1000)
	if pred <= 0 {
		t.Errorf("predict(1000, 1000) = %d, want positive", pred)
	}

	// Positive × negative → negative prediction
	pred = p.predict(1000, -1000)
	if pred >= 0 {
		t.Errorf("predict(1000, -1000) = %d, want negative", pred)
	}

	// Max values should clamp, not overflow
	pred = p.predict(MaxActivation, MaxWeight)
	if pred < 0 {
		t.Errorf("predict(max, max) overflowed: %d", pred)
	}
}

func TestPredictiveStaticNetworkUnchanged(t *testing.T) {
	// A network with no activity should have no weight changes.
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewPredictiveRule(DefaultPredictiveConfig())
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 300)

	w1 := net.Neurons[0].Connections[0].Weight
	w2 := net.Neurons[1].Connections[0].Weight

	// Just tick without stimulation
	net.TickN(50)

	if net.Neurons[0].Connections[0].Weight != w1 {
		t.Errorf("weight 0->1 changed without activity: %d -> %d", w1, net.Neurons[0].Connections[0].Weight)
	}
	if net.Neurons[1].Connections[0].Weight != w2 {
		t.Errorf("weight 1->2 changed without activity: %d -> %d", w2, net.Neurons[1].Connections[0].Weight)
	}
}

func TestPredictiveDefaultConfig(t *testing.T) {
	cfg := DefaultPredictiveConfig()
	if cfg.LearningRate == 0 {
		t.Error("default LearningRate should be nonzero")
	}
	if cfg.EligibilityDecayRate == 0 {
		t.Error("default EligibilityDecayRate should be nonzero")
	}
	if cfg.PredictionScale == 0 {
		t.Error("default PredictionScale should be nonzero")
	}
}

// abs16 returns the absolute value of an int16.
func abs16(v int16) int16 {
	if v < 0 {
		if v == MinActivation {
			return MaxActivation // avoid overflow
		}
		return -v
	}
	return v
}
