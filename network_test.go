package biomimetic

import "testing"

func TestNetworkBasicPropagation(t *testing.T) {
	// Create a simple 3-neuron chain: A -> B -> C
	net := NewNetwork(3, 0, 100, 58982, 2) // 90% decay, 2-tick refractory

	// A -> B (strong excitatory)
	net.Connect(0, 1, 500)
	// B -> C (strong excitatory)
	net.Connect(1, 2, 500)

	// Stimulate A strongly enough to fire
	net.Stimulate(0, 500)

	// A should have fired and reset to baseline
	if net.Neurons[0].Activation != 0 {
		t.Errorf("neuron A activation: got %d, want 0 (baseline)", net.Neurons[0].Activation)
	}

	// B should have received 500 from A, fired, and reset
	if net.Neurons[1].Activation != 0 {
		t.Errorf("neuron B activation: got %d, want 0 (baseline after firing)", net.Neurons[1].Activation)
	}

	// C should have received 500 from B, fired, and reset
	if net.Neurons[2].Activation != 0 {
		t.Errorf("neuron C activation: got %d, want 0 (baseline after firing)", net.Neurons[2].Activation)
	}

	// All three should be in refractory
	if net.Neurons[0].RefractoryUntil != 2 {
		t.Errorf("neuron A refractory: got %d, want 2", net.Neurons[0].RefractoryUntil)
	}
}

func TestNetworkInhibitionBlocksPropagation(t *testing.T) {
	// A -> B (excitatory), C -> B (inhibitory)
	// If C fires first, B shouldn't reach threshold from A alone
	net := NewNetwork(3, 0, 400, 58982, 2)

	net.Connect(0, 1, 300) // A -> B: not enough alone
	net.Connect(2, 1, -200) // C -> B: inhibitory

	// First, inhibit B via C
	net.Stimulate(2, 500) // C fires, sends -200 to B

	// Now stimulate A
	net.Stimulate(0, 500) // A fires, sends +300 to B

	// B should have: -200 + 300 = 100, below threshold of 400
	if net.Neurons[1].Activation != 100 {
		t.Errorf("neuron B activation: got %d, want 100", net.Neurons[1].Activation)
	}
}

func TestNetworkDecayOverTime(t *testing.T) {
	net := NewNetwork(2, 0, 1000, 32768, 2) // 50% decay

	net.Connect(0, 1, 500)

	// Stimulate neuron 1 directly (below threshold)
	net.Stimulate(1, 800)

	// 800 < 1000, no fire. Activation should be 800
	if net.Neurons[1].Activation != 800 {
		t.Errorf("neuron 1 activation: got %d, want 800", net.Neurons[1].Activation)
	}

	// Advance time
	net.TickN(1)

	// Stimulate again with small amount — should decay first
	// 800 * 50% = 400, then +100 = 500
	net.Stimulate(1, 100)
	if net.Neurons[1].Activation != 500 {
		t.Errorf("neuron 1 activation after decay+stim: got %d, want 500", net.Neurons[1].Activation)
	}
}

func TestNetworkMaxPropagationDepth(t *testing.T) {
	// Long chain of 10 neurons, but limit propagation to 3
	net := NewNetwork(10, 0, 50, 58982, 2)
	net.MaxPropagationDepth = 3

	for i := uint32(0); i < 9; i++ {
		net.Connect(i, i+1, 500)
	}

	net.Stimulate(0, 500)

	// Neurons 0, 1, 2 should have fired (depth 0, 1, 2)
	// Neuron 3 should NOT have fired (depth 3 = limit)
	for i := uint32(0); i < 3; i++ {
		if net.Neurons[i].RefractoryUntil == 0 {
			t.Errorf("neuron %d should have fired", i)
		}
	}
	if net.Neurons[3].RefractoryUntil != 0 {
		t.Errorf("neuron 3 should NOT have fired (depth limit)")
	}
}

func TestNetworkActiveNeurons(t *testing.T) {
	net := NewNetwork(5, 0, 1000, 58982, 2)

	// Manually set some activations
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
