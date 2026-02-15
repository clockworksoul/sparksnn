package biomimetic

import (
	"fmt"
	"testing"
)

func TestLoadCelegansCSV(t *testing.T) {
	records, err := LoadCelegansCSV("data/celegans_connectome.csv")
	if err != nil {
		t.Fatalf("LoadCelegansCSV: %v", err)
	}

	if len(records) == 0 {
		t.Fatal("no records loaded")
	}

	t.Logf("Loaded %d connections", len(records))

	// Count types
	types := make(map[string]int)
	for _, r := range records {
		types[r.Type]++
	}
	t.Logf("Connection types: %v", types)
}

func TestCelegansNetwork(t *testing.T) {
	records, err := LoadCelegansCSV("data/celegans_connectome.csv")
	if err != nil {
		t.Fatalf("LoadCelegansCSV: %v", err)
	}

	net, nameMap := CelegansNetwork(records, 100, 0, 1000, 58982, 3)

	t.Logf("Network: %d neurons, counter=%d", len(net.Neurons), net.Counter)
	t.Logf("Name map entries: %d", len(nameMap))

	// Count total connections
	totalConns := 0
	maxConns := 0
	for _, n := range net.Neurons {
		totalConns += len(n.Connections)
		if len(n.Connections) > maxConns {
			maxConns = len(n.Connections)
		}
	}
	t.Logf("Total connections in network: %d", totalConns)
	t.Logf("Max connections per neuron: %d", maxConns)

	// Verify some known neurons exist
	knownNeurons := []string{"AVAL", "AVAR", "AVBL", "AVBR", "ADAL", "ADAR"}
	for _, name := range knownNeurons {
		if _, ok := nameMap[name]; !ok {
			t.Errorf("expected neuron %s not found", name)
		}
	}
}

func TestCelegansStimulation(t *testing.T) {
	records, err := LoadCelegansCSV("data/celegans_connectome.csv")
	if err != nil {
		t.Fatalf("LoadCelegansCSV: %v", err)
	}

	// Threshold tuned so a few strong synapses can trigger firing
	net, nameMap := CelegansNetwork(records, 200, 0, 300, 58982, 3)

	// Stimulate a touch receptor neuron (PLML - posterior lateral
	// mechanosensory left)
	plml, ok := nameMap["PLML"]
	if !ok {
		t.Fatal("PLML not found")
	}

	// Strong stimulation to the touch receptor
	net.Stimulate(plml, 5000)

	// Run for several ticks and observe propagation
	totalFired := 0
	for tick := 0; tick < 20; tick++ {
		fired := net.Tick()
		totalFired += fired
		if fired > 0 {
			t.Logf("Tick %d: %d neurons fired, %d pending",
				tick+1, fired, net.Pending())
		}
	}

	t.Logf("Total neurons fired across 20 ticks: %d", totalFired)

	if totalFired == 0 {
		t.Error("expected at least some neurons to fire after stimulating PLML")
	}

	// Check which neurons are active after stimulation
	active := net.ActiveNeurons(0)
	t.Logf("Neurons with above-baseline activation: %d / %d",
		len(active), len(net.Neurons))

	// Build reverse map for readable output
	indexToName := make(map[uint32]string, len(nameMap))
	for name, idx := range nameMap {
		indexToName[idx] = name
	}

	// Print some active neurons
	shown := 0
	for _, idx := range active {
		if shown >= 10 {
			fmt.Printf("  ... and %d more\n", len(active)-shown)
			break
		}
		n := net.Neurons[idx]
		t.Logf("  Active: %s (activation=%d)", indexToName[idx], n.Activation)
		shown++
	}
}
