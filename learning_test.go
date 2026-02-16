package biomimetic

import "testing"

func TestNoOpLearningDoesNothing(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.LearningRule = NoOpLearning{}
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 500)

	net.Stimulate(0, 500) // A fires
	net.Tick()            // B fires
	net.Tick()            // C fires

	// Weights should be unchanged
	if net.Neurons[0].Connections[0].Weight != 500 {
		t.Errorf("A->B weight changed: got %d, want 500", net.Neurons[0].Connections[0].Weight)
	}
	if net.Neurons[1].Connections[0].Weight != 500 {
		t.Errorf("B->C weight changed: got %d, want 500", net.Neurons[1].Connections[0].Weight)
	}
}

func TestSTDPCausalEligibility(t *testing.T) {
	// Pre fires before post → positive eligibility (potentiation candidate)
	// Chain: A -> B. A fires, then B fires on next tick.
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	net.Stimulate(0, 500) // A fires at tick 0
	net.Tick()            // tick 1: B receives signal, fires

	// B fired at tick 1, A fired at tick 0 → causal (pre before post)
	// OnPostFire should have set positive eligibility on A->B connection
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig <= 0 {
		t.Errorf("causal STDP should produce positive eligibility, got %d", elig)
	}
}

func TestSTDPAntiCausalEligibility(t *testing.T) {
	// Post fires before pre → negative eligibility (depression candidate)
	// B fires first (externally stimulated), then A fires and sends signal to B.
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	// Force B to fire first at tick 1
	net.Tick() // tick 1
	net.Stimulate(1, 500) // B fires at tick 1
	// B is now in refractory, LastFired = 1

	// Advance past B's refractory
	net.Tick() // tick 2
	net.Tick() // tick 3

	// Now A fires at tick 3 — A is pre, B is post, B fired at tick 1 (before A at tick 3)
	net.Stimulate(0, 500) // A fires at tick 3

	// OnSpikePropagation should detect: preFiredAt=3, postLastFired=1
	// That's anti-causal: post fired before pre → negative eligibility
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig >= 0 {
		t.Errorf("anti-causal STDP should produce negative eligibility, got %d", elig)
	}
}

func TestSTDPRewardConsolidatesWeights(t *testing.T) {
	// Positive eligibility + positive reward → weight increase
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Create positive eligibility via causal timing
	net.Stimulate(0, 500) // A fires
	net.Tick()            // B fires (causal: pre before post)

	// Verify eligibility exists
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig <= 0 {
		t.Fatalf("expected positive eligibility before reward, got %d", elig)
	}

	// Deliver positive reward
	net.Reward(100)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight <= originalWeight {
		t.Errorf("positive reward + positive eligibility should increase weight: was %d, now %d",
			originalWeight, newWeight)
	}

	// Eligibility should be cleared after reward
	if net.Neurons[0].Connections[0].Eligibility != 0 {
		t.Errorf("eligibility should be 0 after reward, got %d",
			net.Neurons[0].Connections[0].Eligibility)
	}
}

func TestSTDPPunishmentWeakensWeight(t *testing.T) {
	// Positive eligibility + negative reward → weight decrease
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Create positive eligibility via causal timing
	net.Stimulate(0, 500) // A fires
	net.Tick()            // B fires

	// Deliver negative reward (punishment)
	net.Reward(-100)

	newWeight := net.Neurons[0].Connections[0].Weight
	if newWeight >= originalWeight {
		t.Errorf("negative reward + positive eligibility should decrease weight: was %d, now %d",
			originalWeight, newWeight)
	}
}

func TestSTDPEligibilityDecays(t *testing.T) {
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	cfg.EligibilityDecayRate = 32768 // 50% per tick — aggressive decay
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	// Create eligibility
	net.Stimulate(0, 500)
	net.Tick() // B fires, eligibility set

	eligAfterFire := net.Neurons[0].Connections[0].Eligibility
	if eligAfterFire <= 0 {
		t.Fatalf("expected positive eligibility, got %d", eligAfterFire)
	}

	// Tick several times — eligibility should decay
	net.Tick()
	net.Tick()
	net.Tick()

	eligAfterDecay := net.Neurons[0].Connections[0].Eligibility
	if eligAfterDecay >= eligAfterFire {
		t.Errorf("eligibility should decay over time: was %d, now %d",
			eligAfterFire, eligAfterDecay)
	}
}

func TestSTDPNoRewardNoWeightChange(t *testing.T) {
	// Without reward, eligibility traces should exist but weights
	// should not change (three-factor: needs the reward signal).
	net := NewNetwork(2, 0, 100, 58982, 2)
	cfg := DefaultSTDPConfig()
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 500)

	originalWeight := net.Neurons[0].Connections[0].Weight

	// Create eligibility
	net.Stimulate(0, 500)
	net.Tick()

	// Run many ticks without reward
	net.TickN(20)

	// Weight should be unchanged (eligibility decayed to 0, never consolidated)
	if net.Neurons[0].Connections[0].Weight != originalWeight {
		t.Errorf("weight changed without reward: was %d, now %d",
			originalWeight, net.Neurons[0].Connections[0].Weight)
	}
}

func TestSTDPWindowExpiry(t *testing.T) {
	// If too much time passes between pre and post firing,
	// no eligibility should be generated (outside the STDP window).
	net := NewNetwork(2, 0, 100, 64000, 2) // slow decay so neuron stays active
	cfg := DefaultSTDPConfig()
	cfg.TauPlus = 3  // narrow window
	cfg.TauMinus = 3 // narrow window
	net.LearningRule = NewSTDPRule(cfg)
	net.Connect(0, 1, 100)

	// A fires at tick 0
	net.Neurons[0].LastFired = 1

	// Wait a long time (well beyond 6*tau = 18 ticks)
	for i := 0; i < 30; i++ {
		net.Tick()
	}

	// Now stimulate B to fire — the timing gap should be too large
	net.Stimulate(1, 500)

	// Check: B's OnPostFire should find A's LastFired too far back
	elig := net.Neurons[0].Connections[0].Eligibility
	if elig != 0 {
		t.Errorf("eligibility should be 0 for expired timing window, got %d", elig)
	}
}

func TestLearningRuleSwappable(t *testing.T) {
	// Verify we can swap learning rules at runtime
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)

	// Start with NoOp
	net.LearningRule = NoOpLearning{}
	net.Stimulate(0, 500)
	net.Tick()
	if net.Neurons[0].Connections[0].Eligibility != 0 {
		t.Error("NoOp should not set eligibility")
	}

	// Swap to STDP
	net.LearningRule = NewSTDPRule(DefaultSTDPConfig())

	// Reset neurons for clean test
	net.Neurons[0].Activation = net.Neurons[0].Baseline
	net.Neurons[0].HasFired = false
	net.Neurons[1].Activation = net.Neurons[1].Baseline
	net.Neurons[1].HasFired = false

	net.Stimulate(0, 500)
	net.Tick()

	// STDP should now be active
	if net.Neurons[0].Connections[0].Eligibility == 0 {
		t.Error("STDP should set eligibility after swap")
	}
}

func TestStdpWindow(t *testing.T) {
	// Test the exponential decay function directly
	tests := []struct {
		name string
		dt   uint32
		amp  int16
		tau  uint32
		want bool // true = nonzero expected
	}{
		{"dt=0", 0, 100, 5, true},
		{"dt=1, tau=5", 1, 100, 5, true},
		{"dt=30, tau=5 (expired)", 30, 100, 5, false},
		{"dt=1, tau=0", 1, 100, 0, false},
		{"dt=5, tau=5", 5, 100, 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stdpWindow(tt.dt, tt.amp, tt.tau)
			if tt.want && got == 0 {
				t.Errorf("expected nonzero, got 0")
			}
			if !tt.want && got != 0 {
				t.Errorf("expected 0, got %d", got)
			}
		})
	}
}

func TestLastFiredTracking(t *testing.T) {
	// Verify that LastFired is set correctly when neurons fire
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)

	if net.Neurons[0].LastFired != 0 {
		t.Errorf("initial LastFired should be 0, got %d", net.Neurons[0].LastFired)
	}

	net.Stimulate(0, 500) // A fires at tick 0
	if net.Neurons[0].LastFired != 0 {
		t.Errorf("A LastFired should be 0 (current counter), got %d", net.Neurons[0].LastFired)
	}

	net.Tick() // tick 1: B fires
	if net.Neurons[1].LastFired != 1 {
		t.Errorf("B LastFired should be 1, got %d", net.Neurons[1].LastFired)
	}
}
