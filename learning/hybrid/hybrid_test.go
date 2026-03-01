package hybrid

import (
	"testing"

	bio "github.com/clockworksoul/sparksnn"
)

// TestHybridImplementsLearningRule verifies the interface is satisfied.
func TestHybridImplementsLearningRule(t *testing.T) {
	var _ bio.LearningRule = (*Rule)(nil)
}

// TestHybridDelegation verifies that both sub-rules are called.
func TestHybridDelegation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Perturbation.BatchSize = 1 // immediate evaluation
	rule := NewRule(cfg)

	// Build a tiny network
	net := bio.NewNetwork(4, 0, 100, 45000, 3)
	net.LearningRule = rule
	net.Connect(0, 2, 200)
	net.Connect(1, 2, 200)
	net.Connect(2, 3, 200)

	// Stimulate to create spike activity
	net.Stimulate(0, 200)
	net.Tick()
	net.Stimulate(1, 200)
	net.Tick()

	// Deliver reward — should trigger both R-STDP and perturbation
	initialWeights := make([]int32, 0)
	for i := range net.Neurons {
		for _, c := range net.Neurons[i].Connections {
			initialWeights = append(initialWeights, c.Weight)
		}
	}

	net.Reward(100)

	// At least one weight should have changed (from perturbation if not STDP)
	changed := false
	idx := 0
	for i := range net.Neurons {
		for _, c := range net.Neurons[i].Connections {
			if c.Weight != initialWeights[idx] {
				changed = true
			}
			idx++
		}
	}

	if !changed {
		t.Error("No weights changed after reward — delegation may be broken")
	}
}
