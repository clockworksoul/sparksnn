package sparksnn

import (
	"testing"
)

func TestPlasticitySatisfiesInterface(t *testing.T) {
	var _ StructuralPlasticity = &DefaultPlasticity{}
	var _ StructuralPlasticity = NewPlasticity(DefaultPlasticityConfig())
}

func TestNilPlasticityIsNoOp(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)
	net.Connect(1, 2, 500)

	pruned, grown := net.Remodel()
	if pruned != 0 || grown != 0 {
		t.Errorf("nil plasticity should be no-op, got pruned=%d grown=%d", pruned, grown)
	}
}

func TestPruneWeakConnections(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Connect(0, 1, 500) // strong — keep
	net.Connect(0, 2, 5)   // weak — prune

	cfg := DefaultPlasticityConfig()
	cfg.PruneThreshold = 10
	cfg.GrowthRate = 0           // disable growth
	cfg.HomeostaticEnabled = false // disable homeostasis
	net.StructuralPlasticity = NewPlasticity(cfg)

	pruned, _ := net.Remodel()
	if pruned != 1 {
		t.Errorf("expected 1 pruned, got %d", pruned)
	}

	if len(net.Neurons[0].Connections) != 1 {
		t.Errorf("expected 1 connection remaining, got %d", len(net.Neurons[0].Connections))
	}

	if net.Neurons[0].Connections[0].Target != 1 {
		t.Errorf("wrong connection survived: target=%d, want 1", net.Neurons[0].Connections[0].Target)
	}
}

func TestPruneNegativeWeights(t *testing.T) {
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, -5) // weak negative — prune

	cfg := DefaultPlasticityConfig()
	cfg.PruneThreshold = 10
	cfg.GrowthRate = 0
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	pruned, _ := net.Remodel()
	if pruned != 1 {
		t.Errorf("expected 1 pruned, got %d", pruned)
	}
}

func TestPruneKeepsStrongNegative(t *testing.T) {
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Connect(0, 1, -500) // strong negative — keep

	cfg := DefaultPlasticityConfig()
	cfg.PruneThreshold = 10
	cfg.GrowthRate = 0
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	pruned, _ := net.Remodel()
	if pruned != 0 {
		t.Errorf("strong negative should not be pruned, got %d pruned", pruned)
	}
}

func TestHomeostasisRescuesDeadNeurons(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)

	// Advance the clock so neurons look dead
	net.Counter = 500

	originalThreshold := net.Neurons[0].Threshold

	cfg := DefaultPlasticityConfig()
	cfg.DeadThreshold = 200
	cfg.HomeostaticStep = 15
	cfg.MinThreshold = 20
	cfg.GrowthRate = 0
	net.StructuralPlasticity = NewPlasticity(cfg)

	net.Remodel()

	if net.Neurons[0].Threshold >= originalThreshold {
		t.Errorf("dead neuron threshold should decrease: was %d, now %d",
			originalThreshold, net.Neurons[0].Threshold)
	}

	expected := originalThreshold - cfg.HomeostaticStep
	if net.Neurons[0].Threshold != expected {
		t.Errorf("threshold should be %d, got %d", expected, net.Neurons[0].Threshold)
	}
}

func TestHomeostasisRespectsMinThreshold(t *testing.T) {
	net := NewNetwork(1, 0, 60, 58982, 2)
	net.Counter = 1000

	cfg := DefaultPlasticityConfig()
	cfg.DeadThreshold = 200
	cfg.HomeostaticStep = 50
	cfg.MinThreshold = 30
	cfg.GrowthRate = 0
	net.StructuralPlasticity = NewPlasticity(cfg)

	// First remodel: 60 - 50 = 10, but clamped to 30
	net.Remodel()
	if net.Neurons[0].Threshold != 30 {
		t.Errorf("threshold should be clamped to MinThreshold 30, got %d",
			net.Neurons[0].Threshold)
	}
}

func TestHomeostasisSkipsActiveNeurons(t *testing.T) {
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Counter = 500

	// Neuron 0 fired recently
	net.Neurons[0].LastFired = 490

	cfg := DefaultPlasticityConfig()
	cfg.DeadThreshold = 200
	cfg.HomeostaticStep = 15
	cfg.GrowthRate = 0
	net.StructuralPlasticity = NewPlasticity(cfg)

	net.Remodel()

	if net.Neurons[0].Threshold != 100 {
		t.Errorf("active neuron threshold should not change: got %d", net.Neurons[0].Threshold)
	}

	// Neuron 1 never fired — should be adjusted
	if net.Neurons[1].Threshold >= 100 {
		t.Errorf("dead neuron 1 threshold should decrease: got %d", net.Neurons[1].Threshold)
	}
}

func TestGrowthBetweenCoActiveNeurons(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Counter = 100

	// Neuron 0 fired at tick 90, neuron 2 fired at tick 95
	// Causal: 0 → 2 should be a growth candidate
	net.Neurons[0].LastFired = 90
	net.Neurons[2].LastFired = 95

	// No existing connection between 0 and 2
	cfg := DefaultPlasticityConfig()
	cfg.GrowthRate = 5
	cfg.MinCoActivityWindow = 20
	cfg.InitialWeight = 75
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	_, grown := net.Remodel()
	if grown != 1 {
		t.Errorf("expected 1 grown connection, got %d", grown)
	}

	// Check the connection was created
	found := false
	for _, conn := range net.Neurons[0].Connections {
		if conn.Target == 2 && conn.Weight == 75 {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected connection 0→2 with weight 75")
	}
}

func TestGrowthRespectsFilter(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Counter = 100
	net.Neurons[0].LastFired = 90
	net.Neurons[2].LastFired = 95

	cfg := DefaultPlasticityConfig()
	cfg.GrowthRate = 5
	cfg.MinCoActivityWindow = 20
	cfg.HomeostaticEnabled = false
	// Filter: block all connections
	cfg.Filter = func(source, target uint32) bool { return false }
	net.StructuralPlasticity = NewPlasticity(cfg)

	_, grown := net.Remodel()
	if grown != 0 {
		t.Errorf("filter should block all growth, got %d grown", grown)
	}
}

func TestGrowthSkipsExistingConnections(t *testing.T) {
	net := NewNetwork(2, 0, 100, 58982, 2)
	net.Counter = 100
	net.Neurons[0].LastFired = 90
	net.Neurons[1].LastFired = 95
	net.Connect(0, 1, 500) // already connected

	cfg := DefaultPlasticityConfig()
	cfg.GrowthRate = 5
	cfg.MinCoActivityWindow = 20
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	_, grown := net.Remodel()
	if grown != 0 {
		t.Errorf("should not grow duplicate connections, got %d grown", grown)
	}
}

func TestGrowthRespectsMaxConnections(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Counter = 100
	net.Neurons[0].LastFired = 90
	net.Neurons[2].LastFired = 95
	net.Connect(0, 1, 500) // existing connection

	cfg := DefaultPlasticityConfig()
	cfg.GrowthRate = 5
	cfg.MaxConnectionsPerNeuron = 1 // already at cap
	cfg.MinCoActivityWindow = 20
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	_, grown := net.Remodel()
	if grown != 0 {
		t.Errorf("should respect max connections cap, got %d grown", grown)
	}
}

func TestGrowthPrefersCausalTiming(t *testing.T) {
	net := NewNetwork(4, 0, 100, 58982, 2)
	net.Counter = 100

	// All recently active
	net.Neurons[0].LastFired = 90  // source
	net.Neurons[1].LastFired = 91  // close target (dt=1, score=19)
	net.Neurons[2].LastFired = 99  // far target (dt=9, score=11)
	net.Neurons[3].LastFired = 85  // anti-causal (fired BEFORE source) — skip

	cfg := DefaultPlasticityConfig()
	cfg.GrowthRate = 1 // only grow best candidate
	cfg.MinCoActivityWindow = 20
	cfg.InitialWeight = 75
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	_, grown := net.Remodel()
	if grown != 1 {
		t.Fatalf("expected 1 grown, got %d", grown)
	}

	// Should have picked 0→1 (highest score)
	found := false
	for _, conn := range net.Neurons[0].Connections {
		if conn.Target == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected growth to pick closest causal pair (0→1)")
	}
}

func TestDisconnect(t *testing.T) {
	net := NewNetwork(3, 0, 100, 58982, 2)
	net.Connect(0, 1, 500)
	net.Connect(0, 2, 300)

	removed := net.Disconnect(0, 1)
	if !removed {
		t.Error("Disconnect should return true when connection exists")
	}

	if len(net.Neurons[0].Connections) != 1 {
		t.Errorf("expected 1 connection after disconnect, got %d",
			len(net.Neurons[0].Connections))
	}

	if net.Neurons[0].Connections[0].Target != 2 {
		t.Errorf("wrong connection survived disconnect")
	}

	// Disconnect non-existent
	removed = net.Disconnect(0, 1)
	if removed {
		t.Error("Disconnect should return false for non-existent connection")
	}
}

func TestPruneAndGrowCombined(t *testing.T) {
	net := NewNetwork(4, 0, 100, 58982, 2)
	net.Counter = 100

	net.Connect(0, 1, 3)   // weak — will be pruned
	net.Connect(0, 2, 500) // strong — keep

	// Neurons 2 and 3 are co-active
	net.Neurons[2].LastFired = 90
	net.Neurons[3].LastFired = 95

	cfg := DefaultPlasticityConfig()
	cfg.PruneThreshold = 10
	cfg.GrowthRate = 5
	cfg.MinCoActivityWindow = 20
	cfg.InitialWeight = 75
	cfg.HomeostaticEnabled = false
	net.StructuralPlasticity = NewPlasticity(cfg)

	pruned, grown := net.Remodel()
	if pruned != 1 {
		t.Errorf("expected 1 pruned, got %d", pruned)
	}
	if grown != 1 {
		t.Errorf("expected 1 grown (2→3), got %d", grown)
	}
}

func TestDefaultPlasticityConfig(t *testing.T) {
	cfg := DefaultPlasticityConfig()
	if cfg.PruneThreshold == 0 {
		t.Error("PruneThreshold should be nonzero")
	}
	if cfg.GrowthRate == 0 {
		t.Error("GrowthRate should be nonzero")
	}
	if !cfg.HomeostaticEnabled {
		t.Error("HomeostaticEnabled should be true by default")
	}
	if cfg.MinThreshold >= cfg.MaxThreshold {
		t.Error("MinThreshold should be less than MaxThreshold")
	}
}
