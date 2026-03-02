package sparksnn

import (
	"testing"
)

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

func TestLastFiredTracking(t *testing.T) {
	// Verify that LastFired is set correctly when neurons fire
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)

	if net.Neurons[0].LastFired != 0 {
		t.Errorf("initial LastFired should be 0, got %d", net.Neurons[0].LastFired)
	}

	net.Stimulate(0, 500)
	if net.Neurons[0].LastFired != 1 {
		t.Errorf("A LastFired should be 1 (current counter), got %d", net.Neurons[0].LastFired)
	}

	net.Tick()
	if net.Neurons[1].LastFired != 2 {
		t.Errorf("B LastFired should be 2, got %d", net.Neurons[1].LastFired)
	}
}
