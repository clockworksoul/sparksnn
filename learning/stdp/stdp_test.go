package stdp

import (
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

func TestRuleSatisfiesInterface(t *testing.T) {
	var _ bio.LearningRule = &Rule{}
	var _ bio.LearningRule = NewRule(DefaultConfig())
}

func TestCausalTimingStrengthensWeight(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Pre fires, then post fires — causal → potentiation
	net.Stimulate(0, 500)
	net.Tick()

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight <= originalWeight {
		t.Errorf("causal STDP should increase weight: was %d, now %d",
			originalWeight, newWeight)
	}
}

func TestAntiCausalTimingWeakensWeight(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Post fires first
	net.Tick()
	net.Stimulate(1, 500)
	net.Tick()
	net.Tick()

	// Then pre fires — anti-causal → depression
	net.Stimulate(0, 500)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight >= originalWeight {
		t.Errorf("anti-causal STDP should decrease weight: was %d, now %d",
			originalWeight, newWeight)
	}
}

func TestNoRewardNeeded(t *testing.T) {
	// Pure STDP changes weights without any reward signal
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	net.Stimulate(0, 500)
	net.Tick()

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight == originalWeight {
		t.Errorf("pure STDP should change weights without reward, stayed at %d", originalWeight)
	}
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
		t.Errorf("Reward should be no-op for pure STDP: was %d, now %d",
			weightBefore, weightAfter)
	}
}

func TestMaintainIsNoOp(t *testing.T) {
	// Pure STDP has no eligibility decay — verify Maintain does nothing
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500)
	net.Tick()

	weightAfterFire := net.Neurons[0].Connections[0].Weight

	// Many ticks with no activity — weight shouldn't drift
	net.TickN(50)

	weightAfterMaintain := net.Neurons[0].Connections[0].Weight
	if weightAfterFire != weightAfterMaintain {
		t.Errorf("weight should not change during Maintain: was %d, now %d",
			weightAfterFire, weightAfterMaintain)
	}
}

func TestWindowExpiry(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 64000, 2)
	cfg := DefaultConfig()
	cfg.TauPlus = 3
	cfg.TauMinus = 3
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 100)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Neuron 0 fires at tick 1
	net.Neurons[0].LastFired = 1

	// Wait way past the timing window
	for i := 0; i < 30; i++ {
		net.Tick()
	}

	// Neuron 1 fires — but too late for causal pairing
	net.Stimulate(1, 500)

	if net.Neurons[0].Connections[0].Weight != originalWeight {
		t.Errorf("weight should not change for expired timing window: was %d, now %d",
			originalWeight, net.Neurons[0].Connections[0].Weight)
	}
}

func TestMaxWeightMagnitude(t *testing.T) {
	cfg := DefaultConfig()
	cfg.APlus = 500
	cfg.MaxWeightMagnitude = 600

	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(cfg)
	net.Connect(0, 1, 550)

	// Repeated causal firing should hit the cap
	for i := 0; i < 20; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		net.TickN(3)
	}

	w := net.Neurons[0].Connections[0].Weight
	if w > 600 {
		t.Errorf("weight %d exceeds MaxWeightMagnitude 600", w)
	}
}

func TestRepeatedCausalPotentiation(t *testing.T) {
	net := bio.NewNetwork(2, 0, 100, 58982, 2)
	net.LearningRule = NewRule(DefaultConfig())
	net.Connect(0, 1, 200)

	initial := net.Neurons[0].Connections[0].Weight

	for i := 0; i < 20; i++ {
		net.Stimulate(0, 500)
		net.Tick()
		net.TickN(3)
	}

	final := net.Neurons[0].Connections[0].Weight
	if final <= initial {
		t.Errorf("repeated causal timing should strengthen weight: was %d, now %d",
			initial, final)
	}
}

func TestWindow(t *testing.T) {
	tests := []struct {
		name string
		dt   uint32
		amp  int32
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
