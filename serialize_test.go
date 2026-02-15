package biomimetic

import (
	"bytes"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	// Build a small network with connections
	net := NewNetwork(3, 0, 1000, 58982, 2)
	net.Connect(0, 1, 500)
	net.Connect(0, 2, -300)
	net.Connect(1, 2, 700)

	// Stimulate to change some state
	net.Stimulate(0, 1500) // Should fire neuron 0
	net.Tick()
	net.Tick()

	// Save
	var buf bytes.Buffer
	if err := net.Save(&buf); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load
	loaded, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Verify network properties
	if loaded.Counter != net.Counter {
		t.Errorf("Counter: got %d, want %d", loaded.Counter, net.Counter)
	}
	if loaded.DefaultDecayRate != net.DefaultDecayRate {
		t.Errorf("DefaultDecayRate: got %d, want %d", loaded.DefaultDecayRate, net.DefaultDecayRate)
	}
	if loaded.RefractoryPeriod != net.RefractoryPeriod {
		t.Errorf("RefractoryPeriod: got %d, want %d", loaded.RefractoryPeriod, net.RefractoryPeriod)
	}
	if len(loaded.Neurons) != len(net.Neurons) {
		t.Fatalf("Neuron count: got %d, want %d", len(loaded.Neurons), len(net.Neurons))
	}

	// Verify each neuron
	for i := range net.Neurons {
		got := loaded.Neurons[i]
		want := net.Neurons[i]

		if got.Activation != want.Activation {
			t.Errorf("Neuron[%d].Activation: got %d, want %d", i, got.Activation, want.Activation)
		}
		if got.Baseline != want.Baseline {
			t.Errorf("Neuron[%d].Baseline: got %d, want %d", i, got.Baseline, want.Baseline)
		}
		if got.Threshold != want.Threshold {
			t.Errorf("Neuron[%d].Threshold: got %d, want %d", i, got.Threshold, want.Threshold)
		}
		if got.DecayRate != want.DecayRate {
			t.Errorf("Neuron[%d].DecayRate: got %d, want %d", i, got.DecayRate, want.DecayRate)
		}
		if got.RefractoryUntil != want.RefractoryUntil {
			t.Errorf("Neuron[%d].RefractoryUntil: got %d, want %d", i, got.RefractoryUntil, want.RefractoryUntil)
		}
		if len(got.Connections) != len(want.Connections) {
			t.Errorf("Neuron[%d].Connections: got %d, want %d", i, len(got.Connections), len(want.Connections))
			continue
		}
		for j := range want.Connections {
			if got.Connections[j] != want.Connections[j] {
				t.Errorf("Neuron[%d].Connection[%d]: got %+v, want %+v",
					i, j, got.Connections[j], want.Connections[j])
			}
		}
	}

	// Verify pending queues are empty on load
	if loaded.Pending() != 0 {
		t.Errorf("Loaded network should have empty pending queue, got %d", loaded.Pending())
	}
}

func TestSaveLoadEmpty(t *testing.T) {
	net := NewNetwork(0, 0, 1000, 32768, 1)

	var buf bytes.Buffer
	if err := net.Save(&buf); err != nil {
		t.Fatalf("Save empty network failed: %v", err)
	}

	loaded, err := Load(&buf)
	if err != nil {
		t.Fatalf("Load empty network failed: %v", err)
	}

	if len(loaded.Neurons) != 0 {
		t.Errorf("Expected 0 neurons, got %d", len(loaded.Neurons))
	}
}
