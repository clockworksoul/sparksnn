package predictive

import (
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
	"github.com/clockworksoul/biomimetic-network/learning/rstdp"
)

func TestRuleSatisfiesInterface(t *testing.T) {
	var _ bio.LearningRule = &Rule{}
	var _ bio.LearningRule = NewRule(DefaultConfig())
}

func TestNoRewardNeeded(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	net.Stimulate(0, 500)
	net.Tick()
	net.TickN(5)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight == originalWeight {
		t.Errorf("predictive rule should change weights without reward, weight stayed at %d", originalWeight)
	}
}

func TestEligibilityAccumulates(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	if net.Neurons[0].Connections[0].Eligibility != 0 {
		t.Fatalf("initial eligibility should be 0")
	}

	net.Stimulate(0, 500)

	elig := net.Neurons[0].Connections[0].Eligibility
	if elig == 0 {
		t.Error("eligibility should be nonzero after spike propagation")
	}
}

func TestEligibilityDecays(t *testing.T) {
	cfg := DefaultConfig()
	cfg.EligibilityDecayRate = 32768 // 50% per tick

	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	eligAfter := net.Neurons[0].Connections[0].Eligibility

	net.Tick()
	net.Tick()
	net.Tick()

	eligLater := net.Neurons[0].Connections[0].Eligibility

	if eligAfter != 0 && abs16(eligLater) >= abs16(eligAfter) {
		t.Errorf("eligibility should decay: was %d, now %d", eligAfter, eligLater)
	}
}

func TestCausalTimingChangesWeights(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = 3277 // ~5%

	net := bio.NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 300)
	net.Connect(1, 2, 300)

	originalAB := net.Neurons[0].Connections[0].Weight

	for epoch := 0; epoch < 50; epoch++ {
		net.Stimulate(0, 300)
		net.Tick()
		net.Tick()
		net.TickN(5)
	}

	finalAB := net.Neurons[0].Connections[0].Weight
	if finalAB == originalAB {
		t.Errorf("A->B weight should change with repeated causal timing, stayed at %d", originalAB)
	}
}

func TestSequenceLearning(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = 3277

	net := bio.NewNetwork(5, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)

	for i := uint32(0); i < 4; i++ {
		net.Connect(i, i+1, 200)
	}

	initialWeights := make([]int16, 4)
	for i := 0; i < 4; i++ {
		initialWeights[i] = net.Neurons[i].Connections[0].Weight
	}

	for epoch := 0; epoch < 100; epoch++ {
		net.Stimulate(0, 300)
		net.TickN(10)
	}

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

func TestSwappableWithSTDP(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)

	// Start with R-STDP
	net.LearningRule = rstdp.NewRule(rstdp.DefaultConfig())
	net.Stimulate(0, 500)
	net.Tick()

	eligSTDP := net.Neurons[0].Connections[0].Eligibility
	if eligSTDP == 0 {
		t.Error("R-STDP should set eligibility")
	}

	// Reset
	net.Neurons[0].Connections[0].Eligibility = 0
	net.Neurons[0].Activation = net.Neurons[0].Baseline
	net.Neurons[0].LastFired = 0
	net.Neurons[1].Activation = net.Neurons[1].Baseline
	net.Neurons[1].LastFired = 0

	// Swap to Predictive
	net.LearningRule = NewRule(DefaultConfig())
	net.Stimulate(0, 500)
	net.Tick()

	eligPred := net.Neurons[0].Connections[0].Eligibility
	t.Logf("R-STDP eligibility: %d, Predictive eligibility: %d", eligSTDP, eligPred)
}

func TestRewardIsNoOp(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	weightBefore := net.Neurons[0].Connections[0].Weight
	net.Reward(100)
	weightAfter := net.Neurons[0].Connections[0].Weight

	if weightBefore != weightAfter {
		t.Errorf("Reward should be no-op for predictive rule: was %d, now %d",
			weightBefore, weightAfter)
	}
}

func TestWeightsClamped(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = 6554
	cfg.MaxWeightMagnitude = 1000

	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 900)

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

func TestInhibitoryConnections(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate = 3277

	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, -200)

	for i := 0; i < 50; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		net.TickN(3)
	}
	// Primary test: no panic with inhibitory connections
}

func TestPredictFunction(t *testing.T) {
	p := NewRule(DefaultConfig())

	pred := p.predict(0, 100)
	if pred != 0 {
		t.Errorf("predict(0, 100) = %d, want 0", pred)
	}

	pred = p.predict(1000, 1000)
	if pred <= 0 {
		t.Errorf("predict(1000, 1000) = %d, want positive", pred)
	}

	pred = p.predict(1000, -1000)
	if pred >= 0 {
		t.Errorf("predict(1000, -1000) = %d, want negative", pred)
	}

	pred = p.predict(bio.MaxActivation, bio.MaxWeight)
	if pred < 0 {
		t.Errorf("predict(max, max) overflowed: %d", pred)
	}
}

func TestStaticNetworkUnchanged(t *testing.T) {
	net := bio.NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 300)

	w1 := net.Neurons[0].Connections[0].Weight
	w2 := net.Neurons[1].Connections[0].Weight

	net.TickN(50)

	if net.Neurons[0].Connections[0].Weight != w1 {
		t.Errorf("weight 0->1 changed without activity: %d -> %d", w1, net.Neurons[0].Connections[0].Weight)
	}
	if net.Neurons[1].Connections[0].Weight != w2 {
		t.Errorf("weight 1->2 changed without activity: %d -> %d", w2, net.Neurons[1].Connections[0].Weight)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
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

func abs16(v int16) int16 {
	if v < 0 {
		if v == bio.MinActivation {
			return bio.MaxActivation
		}
		return -v
	}
	return v
}
