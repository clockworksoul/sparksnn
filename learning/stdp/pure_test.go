package stdp

import (
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

func TestPureRuleSatisfiesInterface(t *testing.T) {
	var _ bio.LearningRule = &PureRule{}
	var _ bio.LearningRule = NewPureRule(DefaultPureConfig())
}

func TestPureCausalPotentiation(t *testing.T) {
	// Pre fires before post → immediate weight increase (no reward needed).
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(DefaultPureConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// A fires, then B fires on next tick (causal)
	net.Stimulate(0, 500)
	net.Tick()

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight <= originalWeight {
		t.Errorf("causal timing should increase weight immediately: was %d, now %d",
			originalWeight, newWeight)
	}
}

func TestPureAntiCausalDepression(t *testing.T) {
	// Post fires before pre → immediate weight decrease.
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(DefaultPureConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Force B to fire first
	net.Tick() // tick 1
	net.Stimulate(1, 500) // B fires at tick 1

	// Advance past refractory
	net.Tick() // tick 2
	net.Tick() // tick 3

	// Now A fires — anti-causal (B fired before A)
	net.Stimulate(0, 500) // A fires at tick 3

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight >= originalWeight {
		t.Errorf("anti-causal timing should decrease weight immediately: was %d, now %d",
			originalWeight, newWeight)
	}
}

func TestPureNoEligibilityTraces(t *testing.T) {
	// Pure STDP should not use eligibility traces at all.
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(DefaultPureConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	// Eligibility should remain 0 — pure STDP modifies weights directly
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig != 0 {
		t.Errorf("pure STDP should not set eligibility traces, got %d", elig)
	}
}

func TestPureRewardIsNoOp(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(DefaultPureConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	weightBefore := net.Neurons[0].Connections[0].Weight
	net.Reward(100) // Should be no-op
	weightAfter := net.Neurons[0].Connections[0].Weight

	if weightBefore != weightAfter {
		t.Errorf("Reward should be no-op: was %d, now %d", weightBefore, weightAfter)
	}
}

func TestPureWeightsClamped(t *testing.T) {
	cfg := DefaultPureConfig()
	cfg.APlus = 100 // Big updates
	cfg.MaxWeightMagnitude = 600

	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(cfg)
	net.Connect(0, 1, 550)

	// Repeated causal firing should hit the cap
	for i := 0; i < 50; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		// Reset neurons for next round
		net.Neurons[0].Activation = net.Neurons[0].Baseline
		net.Neurons[0].LastFired = 0
		net.Neurons[1].Activation = net.Neurons[1].Baseline
		net.Neurons[1].LastFired = 0
	}

	w := net.Neurons[0].Connections[0].Weight
	if w > 600 {
		t.Errorf("weight %d exceeds MaxWeightMagnitude 600", w)
	}
}

func TestPureRepeatedCausalStrengthens(t *testing.T) {
	// Repeated causal firing should progressively strengthen the connection.
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultPureConfig()
	cfg.APlus = 50
	net.LearningRule = NewPureRule(cfg)
	net.Connect(0, 1, 200)

	weights := make([]int16, 0, 5)
	weights = append(weights, net.Neurons[0].Connections[0].Weight)

	for i := 0; i < 4; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		weights = append(weights, net.Neurons[0].Connections[0].Weight)

		// Reset for clean next iteration
		net.Neurons[0].Activation = net.Neurons[0].Baseline
		net.Neurons[0].LastFired = 0
		net.Neurons[1].Activation = net.Neurons[1].Baseline
		net.Neurons[1].LastFired = 0
		net.TickN(5) // Let things settle
	}

	// Weights should be monotonically increasing
	for i := 1; i < len(weights); i++ {
		if weights[i] <= weights[i-1] {
			t.Errorf("weights should increase with repeated causal firing: %v", weights)
			break
		}
	}
}

func TestPureStaticNetworkUnchanged(t *testing.T) {
	net := bio.NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(DefaultPureConfig())
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 300)

	w1 := net.Neurons[0].Connections[0].Weight
	w2 := net.Neurons[1].Connections[0].Weight

	net.TickN(50)

	if net.Neurons[0].Connections[0].Weight != w1 {
		t.Errorf("weight changed without activity: %d -> %d", w1, net.Neurons[0].Connections[0].Weight)
	}
	if net.Neurons[1].Connections[0].Weight != w2 {
		t.Errorf("weight changed without activity: %d -> %d", w2, net.Neurons[1].Connections[0].Weight)
	}
}

func TestPureChainLearning(t *testing.T) {
	// In a chain A→B→C, repeated firing should strengthen
	// both A→B and B→C connections.
	cfg := DefaultPureConfig()
	cfg.APlus = 30

	net := bio.NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NewPureRule(cfg)
	net.Connect(0, 1, 300)
	net.Connect(1, 2, 300)

	origAB := net.Neurons[0].Connections[0].Weight
	origBC := net.Neurons[1].Connections[0].Weight

	for i := 0; i < 20; i++ {
		net.Stimulate(0, 300)
		net.Tick() // B fires
		net.Tick() // C fires
		net.TickN(5)
	}

	finalAB := net.Neurons[0].Connections[0].Weight
	finalBC := net.Neurons[1].Connections[0].Weight

	if finalAB <= origAB {
		t.Errorf("A→B should strengthen: was %d, now %d", origAB, finalAB)
	}
	if finalBC <= origBC {
		t.Errorf("B→C should strengthen: was %d, now %d", origBC, finalBC)
	}
}
