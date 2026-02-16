package rstdp

import (
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

func TestRuleSatisfiesInterface(t *testing.T) {
	var _ bio.LearningRule = &Rule{}
	var _ bio.LearningRule = NewRule(DefaultConfig())
}

func TestCausalEligibility(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	elig := net.Neurons[0].Connections[0].Eligibility
	if elig <= 0 {
		t.Errorf("causal STDP should produce positive eligibility, got %d", elig)
	}
}

func TestAntiCausalEligibility(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	net.Tick()
	net.Stimulate(1, 500)
	net.Tick()
	net.Tick()

	net.Stimulate(0, 500)

	elig := net.Neurons[0].Connections[0].Eligibility
	if elig >= 0 {
		t.Errorf("anti-causal STDP should produce negative eligibility, got %d", elig)
	}
}

func TestRewardConsolidatesWeights(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	net.Stimulate(0, 500)
	net.Tick()

	elig := net.Neurons[0].Connections[0].Eligibility
	if elig <= 0 {
		t.Fatalf("expected positive eligibility before reward, got %d", elig)
	}

	net.Reward(100)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight <= originalWeight {
		t.Errorf("positive reward + positive eligibility should increase weight: was %d, now %d",
			originalWeight, newWeight)
	}

	if net.Neurons[0].Connections[0].Eligibility != 0 {
		t.Errorf("eligibility should be 0 after reward, got %d",
			net.Neurons[0].Connections[0].Eligibility)
	}
}

func TestPunishmentWeakensWeight(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	net.Stimulate(0, 500)
	net.Tick()
	net.Reward(-100)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight >= originalWeight {
		t.Errorf("negative reward + positive eligibility should decrease weight: was %d, now %d",
			originalWeight, newWeight)
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

	eligAfterFire := net.Neurons[0].Connections[0].Eligibility
	if eligAfterFire <= 0 {
		t.Fatalf("expected positive eligibility, got %d", eligAfterFire)
	}

	net.Tick()
	net.Tick()
	net.Tick()

	eligAfterDecay := net.Neurons[0].Connections[0].Eligibility
	if eligAfterDecay >= eligAfterFire {
		t.Errorf("eligibility should decay over time: was %d, now %d",
			eligAfterFire, eligAfterDecay)
	}
}

func TestNoRewardNoWeightChange(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	net.Stimulate(0, 500)
	net.Tick()
	net.TickN(20)

	if net.Neurons[0].Connections[0].Weight != originalWeight {
		t.Errorf("weight changed without reward: was %d, now %d",
			originalWeight, net.Neurons[0].Connections[0].Weight)
	}
}

func TestWindowExpiry(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 64000, 2)
	cfg := DefaultConfig()
	cfg.TauPlus = 3
	cfg.TauMinus = 3
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 100)

	net.Neurons[0].LastFired = 1

	for i := 0; i < 30; i++ {
		net.Tick()
	}

	net.Stimulate(1, 500)

	elig := net.Neurons[0].Connections[0].Eligibility
	if elig != 0 {
		t.Errorf("eligibility should be 0 for expired timing window, got %d", elig)
	}
}

func TestWindow(t *testing.T) {
	tests := []struct {
		name string
		dt   uint32
		amp  int16
		tau  uint32
		want bool
	}{
		{"dt=0", 0, 100, 5, true},
		{"dt=1, tau=5", 1, 100, 5, true},
		{"dt=30, tau=5 (expired)", 30, 100, 5, false},
		{"dt=1, tau=0", 1, 100, 0, false},
		{"dt=5, tau=5", 5, 100, 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Window(tt.dt, tt.amp, tt.tau)
			if tt.want && got == 0 {
				t.Errorf("expected nonzero, got 0")
			}
			if !tt.want && got != 0 {
				t.Errorf("expected 0, got %d", got)
			}
		})
	}
}
