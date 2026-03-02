package sparksnn

import (
	"math"
	"testing"
)

func TestDevelopmentBasic(t *testing.T) {
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	dev.AddLayer(LayerSpec{
		Name: "input", Role: RoleInput, Size: 4,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})
	dev.AddLayer(LayerSpec{
		Name: "hidden", Role: RoleHidden, Size: 16,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})
	dev.AddLayer(LayerSpec{
		Name: "output", Role: RoleOutput, Size: 3,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})

	if len(dev.Positions) != 23 {
		t.Fatalf("expected 23 positions, got %d", len(dev.Positions))
	}

	// Verify layer indices
	input := dev.GetLayer("input")
	hidden := dev.GetLayer("hidden")
	output := dev.GetLayer("output")

	if input.StartIdx != 0 || input.EndIdx != 4 {
		t.Errorf("input layer: got [%d, %d), want [0, 4)", input.StartIdx, input.EndIdx)
	}
	if hidden.StartIdx != 4 || hidden.EndIdx != 20 {
		t.Errorf("hidden layer: got [%d, %d), want [4, 20)", hidden.StartIdx, hidden.EndIdx)
	}
	if output.StartIdx != 20 || output.EndIdx != 23 {
		t.Errorf("output layer: got [%d, %d), want [20, 23)", output.StartIdx, output.EndIdx)
	}
}

func TestDevelopmentWiring(t *testing.T) {
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	// Place input and hidden in the same 1x1 space
	dev.AddLayer(LayerSpec{
		Name: "input", Role: RoleInput, Size: 4,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})
	dev.AddLayer(LayerSpec{
		Name: "hidden", Role: RoleHidden, Size: 16,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})

	dev.Build(DevParams{
		Baseline: 0, Threshold: 150,
		DecayRate: 45000, RefractoryPeriod: 5,
	})

	// Wide sigma = nearly full connectivity
	created := dev.Wire(ConnectionRule{
		FromLayer: "input", ToLayer: "hidden",
		Sigma: 10.0, PBase: 1.0,
		InitWeightMin: 100, InitWeightMax: 500,
	})

	density := dev.ConnectionDensity("input", "hidden")
	t.Logf("Wide sigma: %d connections, %.1f%% density", created, density*100)

	if density < 0.8 {
		t.Errorf("expected >80%% density with wide sigma, got %.1f%%", density*100)
	}
}

func TestDevelopmentNarrowSigma(t *testing.T) {
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	// Spread neurons across a wide space
	dev.AddLayer(LayerSpec{
		Name: "hidden", Role: RoleHidden, Size: 100,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 10, Height: 10,
	})

	dev.Build(DevParams{
		Baseline: 0, Threshold: 150,
		DecayRate: 45000, RefractoryPeriod: 5,
	})

	// Narrow sigma = sparse, local connectivity
	created := dev.Wire(ConnectionRule{
		FromLayer: "hidden", ToLayer: "hidden",
		Sigma: 0.5, PBase: 1.0,
		InitWeightMin: 100, InitWeightMax: 500,
	})

	density := dev.ConnectionDensity("hidden", "hidden")
	t.Logf("Narrow sigma: %d connections out of %d possible, %.1f%% density",
		created, 100*99, density*100)

	// Should be significantly less than full
	if density > 0.3 {
		t.Errorf("expected sparse connectivity with narrow sigma, got %.1f%%", density*100)
	}
	if created == 0 {
		t.Error("expected some connections even with narrow sigma")
	}
}

func TestDevelopmentDistanceDependence(t *testing.T) {
	// Verify that nearby neurons actually connect more than distant ones
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	dev.AddLayer(LayerSpec{
		Name: "neurons", Role: RoleHidden, Size: 200,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 10, Height: 10,
	})

	dev.Build(DevParams{
		Baseline: 0, Threshold: 150,
		DecayRate: 45000, RefractoryPeriod: 5,
	})

	dev.Wire(ConnectionRule{
		FromLayer: "neurons", ToLayer: "neurons",
		Sigma: 1.0, PBase: 1.0,
		InitWeightMin: 100, InitWeightMax: 500,
	})

	// Measure average connection distance vs average all-pairs distance
	var connDistSum float64
	var connCount int
	var allDistSum float64
	var allCount int

	for i := uint32(0); i < 200; i++ {
		for _, conn := range dev.Net.Neurons[i].Connections {
			d := dev.Positions[i].distance(dev.Positions[conn.Target])
			connDistSum += float64(d)
			connCount++
		}
		for j := uint32(0); j < 200; j++ {
			if i != j {
				d := dev.Positions[i].distance(dev.Positions[j])
				allDistSum += float64(d)
				allCount++
			}
		}
	}

	avgConnDist := connDistSum / float64(connCount)
	avgAllDist := allDistSum / float64(allCount)

	t.Logf("Avg connection distance: %.2f", avgConnDist)
	t.Logf("Avg all-pairs distance:  %.2f", avgAllDist)
	t.Logf("Ratio: %.2f", avgConnDist/avgAllDist)

	if avgConnDist >= avgAllDist {
		t.Error("connected neurons should be closer than average pair")
	}
}

func TestDevelopmentFinalize(t *testing.T) {
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	dev.AddLayer(LayerSpec{
		Name: "input", Role: RoleInput, Size: 4,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})
	dev.AddLayer(LayerSpec{
		Name: "output", Role: RoleOutput, Size: 2,
		Placement: PlaceRandom,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})

	dev.Build(DevParams{
		Baseline: 0, Threshold: 150,
		DecayRate: 45000, RefractoryPeriod: 5,
	})

	dev.Wire(ConnectionRule{
		FromLayer: "input", ToLayer: "output",
		Sigma: 10.0, PBase: 1.0,
		InitWeightMin: 100, InitWeightMax: 500,
	})

	net := dev.Finalize()

	// Network should still work
	if len(net.Neurons) != 6 {
		t.Fatalf("expected 6 neurons, got %d", len(net.Neurons))
	}

	// Scaffolding should be gone
	if dev.Positions != nil {
		t.Error("positions should be nil after finalize")
	}
	if dev.Net != nil {
		t.Error("net should be nil after finalize")
	}

	// Network should still function
	net.Stimulate(0, 1000)
	net.Tick()
}

func TestDevelopmentGridPlacement(t *testing.T) {
	dev := NewDevelopmentSeeded(DevParams{}, 42)

	dev.AddLayer(LayerSpec{
		Name: "grid", Role: RoleHidden, Size: 9,
		Placement: PlaceGrid,
		OriginX: 0, OriginY: 0, Width: 1, Height: 1,
	})

	// 9 neurons on a 3x3 grid
	// Verify they're roughly evenly spaced
	for i, pos := range dev.Positions {
		t.Logf("Neuron %d: (%.3f, %.3f)", i, pos.X, pos.Y)
	}

	// Check corners are near expected positions
	checkNear := func(idx int, expX, expY float32) {
		pos := dev.Positions[idx]
		dist := float32(math.Sqrt(float64((pos.X-expX)*(pos.X-expX) + (pos.Y-expY)*(pos.Y-expY))))
		if dist > 0.01 {
			t.Errorf("neuron %d at (%.3f,%.3f), expected near (%.3f,%.3f)",
				idx, pos.X, pos.Y, expX, expY)
		}
	}

	checkNear(0, 0.0, 0.0)       // top-left
	checkNear(2, 0.667, 0.0)     // top-right-ish
	checkNear(6, 0.0, 0.667)     // bottom-left-ish
}
