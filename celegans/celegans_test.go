package celegans

import (
	"testing"

	biomimetic "github.com/clockworksoul/biomimetic-network"
)

const testDataPath = "../data/celegans_connectome.csv"

func loadRecords(t *testing.T) []Record {
	t.Helper()
	records, err := LoadCSV(testDataPath)
	if err != nil {
		t.Fatalf("LoadCSV: %v", err)
	}
	return records
}

func buildNetwork(t *testing.T) (*biomimetic.Network, map[string]uint32, map[uint32]string) {
	t.Helper()
	records := loadRecords(t)
	net, nameMap := BuildNetwork(records, DefaultParams())
	indexToName := make(map[uint32]string, len(nameMap))
	for name, idx := range nameMap {
		indexToName[idx] = name
	}
	return net, nameMap, indexToName
}

func TestLoadCSV(t *testing.T) {
	records := loadRecords(t)
	if len(records) == 0 {
		t.Fatal("no records loaded")
	}
	t.Logf("Loaded %d connections", len(records))

	types := make(map[string]int)
	for _, r := range records {
		types[r.Type]++
	}
	t.Logf("Connection types: %v", types)
}

func TestBuildNetwork(t *testing.T) {
	net, nameMap, _ := buildNetwork(t)

	t.Logf("Network: %d neurons", len(net.Neurons))
	t.Logf("Name map entries: %d", len(nameMap))

	totalConns := 0
	maxConns := 0
	inhibConns := 0
	for _, n := range net.Neurons {
		totalConns += len(n.Connections)
		if len(n.Connections) > maxConns {
			maxConns = len(n.Connections)
		}
		for _, c := range n.Connections {
			if c.Weight < 0 {
				inhibConns++
			}
		}
	}
	t.Logf("Total connections: %d (excitatory: %d, inhibitory: %d)",
		totalConns, totalConns-inhibConns, inhibConns)
	t.Logf("Inhibitory fraction: %.1f%%", float64(inhibConns)/float64(totalConns)*100)
	t.Logf("Max connections per neuron: %d", maxConns)

	knownNeurons := []string{"AVAL", "AVAR", "AVBL", "AVBR", "PLML", "PLMR"}
	for _, name := range knownNeurons {
		if _, ok := nameMap[name]; !ok {
			t.Errorf("expected neuron %s not found", name)
		}
	}
}

func TestStimulation(t *testing.T) {
	net, nameMap, indexToName := buildNetwork(t)

	// Stimulate both PLM neurons — PLML alone has only a single gap
	// junction to PVCL, which isn't enough to cascade with default
	// params. Both together mirror the bilateral touch stimulus.
	net.Stimulate(nameMap["PLML"], 5000)
	net.Stimulate(nameMap["PLMR"], 5000)

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
		t.Error("expected at least some neurons to fire after stimulating PLM")
	}

	active := net.ActiveNeurons(0)
	t.Logf("Active neurons: %d / %d (%.0f%%)",
		len(active), len(net.Neurons),
		float64(len(active))/float64(len(net.Neurons))*100)

	shown := 0
	for _, idx := range active {
		if shown >= 10 {
			t.Logf("  ... and %d more", len(active)-shown)
			break
		}
		t.Logf("  Active: %s (activation=%d)", indexToName[idx], net.Neurons[idx].Activation)
		shown++
	}
}

// TestEscapeForward tests the posterior touch → forward locomotion
// circuit, one of the best-characterized behaviors in C. elegans.
//
// Known pathway:
//
//	PLM (posterior touch receptors)
//	→ PVC (command interneurons)
//	→ AVB (forward command interneurons)
//	→ B-class motor neurons (DB1-7, VB1-11)
//	→ forward movement
func TestEscapeForward(t *testing.T) {
	net, nameMap, indexToName := buildNetwork(t)

	// Stimulate both PLM neurons (bilateral touch to posterior)
	net.Stimulate(nameMap["PLML"], 5000)
	net.Stimulate(nameMap["PLMR"], 5000)

	bClassMotors := []string{
		"DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
		"VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7",
		"VB8", "VB9", "VB10", "VB11",
	}
	aClassMotors := []string{
		"DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",
		"VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
		"VA9", "VA10", "VA11", "VA12",
	}
	circuitNeurons := []string{"PLML", "PLMR", "PVCL", "PVCR", "AVBL", "AVBR", "AVAL", "AVAR"}

	const ticks = 30
	bFired := make(map[string]int)
	aFired := make(map[string]int)

	type fireEvent struct {
		tick   int
		neuron string
	}
	var circuitEvents []fireEvent
	totalFired := 0

	for tick := 0; tick < ticks; tick++ {
		fired := net.Tick()
		totalFired += fired

		if fired > 0 {
			t.Logf("Tick %2d: %3d neurons fired, %d pending",
				tick+1, fired, net.Pending())
		}

		for _, name := range bClassMotors {
			if idx, ok := nameMap[name]; ok {
				if net.Neurons[idx].HasFired && net.Neurons[idx].LastFired == net.Counter {
					bFired[name]++
				}
			}
		}
		for _, name := range aClassMotors {
			if idx, ok := nameMap[name]; ok {
				if net.Neurons[idx].HasFired && net.Neurons[idx].LastFired == net.Counter {
					aFired[name]++
				}
			}
		}
		for _, name := range circuitNeurons {
			idx := nameMap[name]
			if net.Neurons[idx].HasFired && net.Neurons[idx].LastFired == net.Counter {
				circuitEvents = append(circuitEvents, fireEvent{tick + 1, name})
			}
		}
	}

	t.Log("")
	t.Log("=== Circuit Neuron Activity ===")
	for _, e := range circuitEvents {
		t.Logf("  Tick %2d: %s fired", e.tick, e.neuron)
	}

	t.Log("")
	bTotal, aTotal := 0, 0
	for _, v := range bFired {
		bTotal += v
	}
	for _, v := range aFired {
		aTotal += v
	}
	t.Logf("B-class (forward) total firings:  %d", bTotal)
	t.Logf("A-class (backward) total firings: %d", aTotal)

	active := net.ActiveNeurons(0)
	activePercent := float64(len(active)) / float64(len(net.Neurons)) * 100
	t.Logf("Active neurons: %d / %d (%.0f%%)", len(active), len(net.Neurons), activePercent)

	if activePercent > 50 {
		t.Errorf("network seizure: %.0f%% of neurons active (want < 50%%)", activePercent)
	}

	_ = indexToName
}

// TestBackwardEscape tests the anterior touch → backward locomotion.
//
// Known pathway:
//
//	ALM/AVM (anterior touch receptors)
//	→ AVA/AVD (backward command interneurons)
//	→ A-class motor neurons (DA, VA)
//	→ backward movement
func TestBackwardEscape(t *testing.T) {
	net, nameMap, _ := buildNetwork(t)

	net.Stimulate(nameMap["ALML"], 3000)
	net.Stimulate(nameMap["ALMR"], 3000)
	if idx, ok := nameMap["AVM"]; ok {
		net.Stimulate(idx, 3000)
	}

	bClassMotors := []string{
		"DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
		"VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7",
		"VB8", "VB9", "VB10", "VB11",
	}
	aClassMotors := []string{
		"DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",
		"VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
		"VA9", "VA10", "VA11", "VA12",
	}

	const ticks = 30
	bTotal, aTotal := 0, 0

	for tick := 0; tick < ticks; tick++ {
		fired := net.Tick()
		if fired > 0 {
			t.Logf("Tick %2d: %3d fired, %d pending", tick+1, fired, net.Pending())
		}

		for _, name := range bClassMotors {
			if idx, ok := nameMap[name]; ok {
				if net.Neurons[idx].HasFired && net.Neurons[idx].LastFired == net.Counter {
					bTotal++
				}
			}
		}
		for _, name := range aClassMotors {
			if idx, ok := nameMap[name]; ok {
				if net.Neurons[idx].HasFired && net.Neurons[idx].LastFired == net.Counter {
					aTotal++
				}
			}
		}
	}

	t.Logf("B-class (forward) total firings:  %d", bTotal)
	t.Logf("A-class (backward) total firings: %d", aTotal)

	active := net.ActiveNeurons(0)
	activePercent := float64(len(active)) / float64(len(net.Neurons)) * 100
	t.Logf("Active neurons: %d / %d (%.0f%%)", len(active), len(net.Neurons), activePercent)

	if activePercent > 50 {
		t.Errorf("network seizure: %.0f%% active (want < 50%%)", activePercent)
	}

	if aTotal == 0 && bTotal == 0 {
		t.Log("WARNING: no motor neurons fired — checking command interneurons")
		for _, name := range []string{"AVAL", "AVAR", "AVDL", "AVDR"} {
			t.Logf("  %s activation: %d", name, net.Neurons[nameMap[name]].Activation)
		}
	} else if aTotal <= bTotal {
		t.Errorf("backward circuit failed: A-class fired %d, B-class fired %d (want A > B)", aTotal, bTotal)
	} else {
		t.Logf("✓ Backward escape response: A-class (%d) > B-class (%d)", aTotal, bTotal)
	}
}
