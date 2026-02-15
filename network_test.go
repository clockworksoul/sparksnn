package biomimetic

import "testing"

func TestNetworkBasicPropagation(t *testing.T) {
	// Create a simple 3-neuron chain: A -> B -> C
	// With tick-driven propagation, A fires on Stimulate,
	// B fires on Tick 1, C fires on Tick 2.
	net := NewNetwork(3, 0, 100, 58982, 2) // 90% decay, 2-tick refractory

	net.Connect(0, 1, 500) // A -> B
	net.Connect(1, 2, 500) // B -> C

	// Stimulate A strongly enough to fire
	net.Stimulate(0, 500)

	// A should have fired and reset to baseline
	if net.Neurons[0].Activation != 0 {
		t.Errorf("neuron A activation: got %d, want 0 (baseline)", net.Neurons[0].Activation)
	}

	// B's stimulation is pending, not yet delivered
	if net.Neurons[1].Activation != 0 {
		t.Errorf("neuron B should not be stimulated yet: got %d, want 0", net.Neurons[1].Activation)
	}

	if net.Pending() != 1 {
		t.Fatalf("expected 1 pending stimulation, got %d", net.Pending())
	}

	// Tick 1: B receives signal from A, fires, queues C
	fired := net.Tick()
	if fired != 1 {
		t.Errorf("tick 1: expected 1 neuron fired, got %d", fired)
	}
	if net.Neurons[1].Activation != 0 {
		t.Errorf("neuron B should have fired and reset: got %d", net.Neurons[1].Activation)
	}

	// Tick 2: C receives signal from B, fires
	fired = net.Tick()
	if fired != 1 {
		t.Errorf("tick 2: expected 1 neuron fired, got %d", fired)
	}
	if net.Neurons[2].Activation != 0 {
		t.Errorf("neuron C should have fired and reset: got %d", net.Neurons[2].Activation)
	}

	// Tick 3: nothing pending
	fired = net.Tick()
	if fired != 0 {
		t.Errorf("tick 3: expected 0 fired, got %d", fired)
	}
}

func TestNetworkInhibitionBlocksPropagation(t *testing.T) {
	// A -> B (excitatory), C -> B (inhibitory)
	net := NewNetwork(3, 0, 400, 58982, 2)

	net.Connect(0, 1, 300)  // A -> B: not enough alone
	net.Connect(2, 1, -200) // C -> B: inhibitory

	// Both A and C fire, sending signals to B
	net.Stimulate(2, 500) // C fires, queues -200 to B
	net.Stimulate(0, 500) // A fires, queues +300 to B

	// Tick: B receives both signals
	fired := net.Tick()
	if fired != 0 {
		t.Errorf("B should not fire: -200 + 300 = 100, below threshold 400")
	}

	// B receives -200 then +300 in same tick. The -200 is applied first,
	// then when +300 arrives, decay has already been calculated for that
	// tick, giving us -180 + 300 = 120 (not 200, because decay is applied).
	// Still below threshold of 400.
	if net.Neurons[1].Activation != 120 {
		t.Errorf("neuron B activation: got %d, want 120", net.Neurons[1].Activation)
	}
}

func TestNetworkDecayOverTime(t *testing.T) {
	net := NewNetwork(2, 0, 1000, 32768, 2) // 50% decay

	// Stimulate neuron 1 directly (below threshold)
	net.Stimulate(1, 800)

	// 800 < 1000, no fire. Activation should be 800
	if net.Neurons[1].Activation != 800 {
		t.Errorf("neuron 1 activation: got %d, want 800", net.Neurons[1].Activation)
	}

	// Advance time
	net.Tick()

	// Stimulate again with small amount — should decay first
	// 800 * 50% = 400, then +100 = 500
	net.Stimulate(1, 100)
	if net.Neurons[1].Activation != 500 {
		t.Errorf("neuron 1 activation after decay+stim: got %d, want 500", net.Neurons[1].Activation)
	}
}

func TestNetworkTemporalPropagation(t *testing.T) {
	// Chain of 5 neurons. Signal should take 4 ticks to reach the end.
	net := NewNetwork(5, 0, 50, 58982, 2)

	for i := uint32(0); i < 4; i++ {
		net.Connect(i, i+1, 500)
	}

	net.Stimulate(0, 500) // Fires neuron 0

	// Each tick should fire exactly one neuron down the chain
	for tick := 1; tick <= 4; tick++ {
		fired := net.Tick()
		if fired != 1 {
			t.Errorf("tick %d: expected 1 fired, got %d", tick, fired)
		}
		// The neuron at index `tick` should have fired (reset to baseline)
		if net.Neurons[tick].Activation != 0 {
			t.Errorf("tick %d: neuron %d should have fired and reset", tick, tick)
		}
	}

	// Tick 5: nothing left
	if fired := net.Tick(); fired != 0 {
		t.Errorf("tick 5: expected 0 fired, got %d", fired)
	}
}

func TestNetworkRefractoryPreventsRefire(t *testing.T) {
	// Two neurons in a loop: A -> B -> A
	// Refractory period should prevent infinite cycling
	net := NewNetwork(2, 0, 50, 58982, 3) // 3-tick refractory

	net.Connect(0, 1, 500)
	net.Connect(1, 0, 500)

	net.Stimulate(0, 500) // A fires

	// Tick 1: B fires (receives from A)
	fired := net.Tick()
	if fired != 1 {
		t.Errorf("tick 1: expected 1 (B), got %d", fired)
	}

	// Tick 2: A receives from B, but A is in refractory — should NOT fire
	fired = net.Tick()
	if fired != 0 {
		t.Errorf("tick 2: expected 0 (A in refractory), got %d", fired)
	}

	// Network should settle
	if net.Pending() != 0 {
		t.Errorf("expected no pending stimulations, got %d", net.Pending())
	}
}

func TestNetworkActiveNeurons(t *testing.T) {
	net := NewNetwork(5, 0, 1000, 58982, 2)

	net.Neurons[1].Activation = 500
	net.Neurons[3].Activation = 800

	active := net.ActiveNeurons(400)
	if len(active) != 2 {
		t.Fatalf("expected 2 active neurons, got %d", len(active))
	}
	if active[0] != 1 || active[1] != 3 {
		t.Errorf("expected neurons [1, 3], got %v", active)
	}
}

func TestNetworkTick(t *testing.T) {
	net := NewNetwork(1, 0, 100, 58982, 2)

	if net.Counter != 0 {
		t.Fatalf("initial counter should be 0")
	}

	net.Tick()
	if net.Counter != 1 {
		t.Errorf("counter after Tick: got %d, want 1", net.Counter)
	}

	net.TickN(10)
	if net.Counter != 11 {
		t.Errorf("counter after TickN(10): got %d, want 11", net.Counter)
	}
}
