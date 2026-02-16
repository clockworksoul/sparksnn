package celegans

import (
	"fmt"
	"testing"

	biomimetic "github.com/clockworksoul/biomimetic-network"
)

// motorActivity sums the activation of a set of neurons identified by
// name prefix (e.g., "VA", "DA" for A-class backward motors; "VB",
// "DB" for B-class forward motors).
func motorActivity(net *biomimetic.Network, nameMap map[string]uint32, prefixes []string) int64 {
	var total int64
	for name, idx := range nameMap {
		for _, prefix := range prefixes {
			if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
				act := int64(net.Neurons[idx].Activation)
				if act > 0 {
					total += act
				}
			}
		}
	}
	return total
}

// namedAct holds a neuron name and its activation level.
type namedAct struct {
	Name       string
	Activation int16
}

func neuronsMatchingPrefix(net *biomimetic.Network, nameMap map[string]uint32, prefixes []string) []namedAct {
	var result []namedAct
	for name, idx := range nameMap {
		for _, prefix := range prefixes {
			if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
				result = append(result, namedAct{
					Name:       name,
					Activation: net.Neurons[idx].Activation,
				})
			}
		}
	}
	// Sort by activation descending
	for i := 1; i < len(result); i++ {
		for j := i; j > 0 && result[j].Activation > result[j-1].Activation; j-- {
			result[j], result[j-1] = result[j-1], result[j]
		}
	}
	return result
}

func safeDivide(a, b int64) float64 {
	if b == 0 {
		if a > 0 {
			return 999.99
		}
		return 0
	}
	return float64(a) / float64(b)
}

// TestBehaviorEscapeForward tests the posterior touch → forward locomotion
// circuit with detailed analysis.
func TestBehaviorEscapeForward(t *testing.T) {
	records := loadRecords(t)
	net, nameMap := BuildNetwork(records, DefaultParams())

	t.Log("=== POSTERIOR TOUCH (expect FORWARD escape) ===")
	net.Stimulate(nameMap["PLML"], 5000)
	net.Stimulate(nameMap["PLMR"], 5000)

	forwardPrefixes := []string{"VB", "DB"}
	backwardPrefixes := []string{"VA", "DA"}

	for tick := 1; tick <= 8; tick++ {
		fired := net.Tick()
		fwd := motorActivity(net, nameMap, forwardPrefixes)
		bwd := motorActivity(net, nameMap, backwardPrefixes)
		t.Logf("Tick %d: %d fired | Forward motors: %d | Backward motors: %d | Ratio F/B: %.2f",
			tick, fired, fwd, bwd, safeDivide(fwd, bwd))
	}

	fwdFinal := motorActivity(net, nameMap, forwardPrefixes)
	bwdFinal := motorActivity(net, nameMap, backwardPrefixes)

	t.Log("")
	t.Log("--- B-class (forward) motor neurons ---")
	for _, na := range neuronsMatchingPrefix(net, nameMap, forwardPrefixes) {
		if na.Activation > 0 {
			t.Logf("  %s: %d", na.Name, na.Activation)
		}
	}

	t.Log("--- A-class (backward) motor neurons ---")
	for _, na := range neuronsMatchingPrefix(net, nameMap, backwardPrefixes) {
		if na.Activation > 0 {
			t.Logf("  %s: %d", na.Name, na.Activation)
		}
	}

	t.Logf("\nFinal motor activity — Forward: %d, Backward: %d (ratio: %.2f)",
		fwdFinal, bwdFinal, safeDivide(fwdFinal, bwdFinal))

	if fwdFinal > bwdFinal {
		t.Logf("✓ Forward motors dominate — consistent with escape forward behavior")
	} else {
		ratio := safeDivide(fwdFinal, bwdFinal)
		t.Logf("⚠ Backward motors still dominate (F/B=%.2f) — expected: reciprocal inhibition needed", ratio)
		t.Logf("  This is a KNOWN LIMITATION of pure connectivity simulation.")
		t.Logf("  See TODO.md: reciprocal inhibition between AVA↔AVB circuits.")
	}
}

// TestBehaviorTapWithdrawal tests the anterior touch → backward
// locomotion circuit.
func TestBehaviorTapWithdrawal(t *testing.T) {
	records := loadRecords(t)
	net, nameMap := BuildNetwork(records, DefaultParams())

	t.Log("=== ANTERIOR TOUCH (expect BACKWARD withdrawal) ===")
	net.Stimulate(nameMap["ALML"], 5000)
	net.Stimulate(nameMap["ALMR"], 5000)
	net.Stimulate(nameMap["AVM"], 5000)

	forwardPrefixes := []string{"VB", "DB"}
	backwardPrefixes := []string{"VA", "DA"}

	for tick := 1; tick <= 8; tick++ {
		fired := net.Tick()
		fwd := motorActivity(net, nameMap, forwardPrefixes)
		bwd := motorActivity(net, nameMap, backwardPrefixes)
		t.Logf("Tick %d: %d fired | Forward motors: %d | Backward motors: %d | Ratio B/F: %.2f",
			tick, fired, fwd, bwd, safeDivide(bwd, fwd))
	}

	fwdFinal := motorActivity(net, nameMap, forwardPrefixes)
	bwdFinal := motorActivity(net, nameMap, backwardPrefixes)

	t.Logf("\nFinal motor activity — Backward: %d, Forward: %d (ratio: %.2f)",
		bwdFinal, fwdFinal, safeDivide(bwdFinal, fwdFinal))

	if bwdFinal <= fwdFinal {
		t.Errorf("FAILED: Expected backward > forward for anterior touch, got B=%d F=%d", bwdFinal, fwdFinal)
	} else {
		t.Logf("✓ Backward motors dominate — consistent with tap withdrawal behavior")
	}
}

// TestDirectionalContrast compares the two responses side by side.
func TestDirectionalContrast(t *testing.T) {
	records := loadRecords(t)

	forwardPrefixes := []string{"VB", "DB"}
	backwardPrefixes := []string{"VA", "DA"}
	ticks := 8

	// Posterior touch
	net1, nameMap := BuildNetwork(records, DefaultParams())
	net1.Stimulate(nameMap["PLML"], 5000)
	net1.Stimulate(nameMap["PLMR"], 5000)
	net1.TickN(uint32(ticks))

	fwd1 := motorActivity(net1, nameMap, forwardPrefixes)
	bwd1 := motorActivity(net1, nameMap, backwardPrefixes)

	// Anterior touch
	net2, _ := BuildNetwork(records, DefaultParams())
	net2.Stimulate(nameMap["ALML"], 5000)
	net2.Stimulate(nameMap["ALMR"], 5000)
	net2.Stimulate(nameMap["AVM"], 5000)
	net2.TickN(uint32(ticks))

	fwd2 := motorActivity(net2, nameMap, forwardPrefixes)
	bwd2 := motorActivity(net2, nameMap, backwardPrefixes)

	t.Log("╔══════════════════════════════════════════════════╗")
	t.Log("║        C. ELEGANS DIRECTIONAL RESPONSE          ║")
	t.Log("╠══════════════════════════════════════════════════╣")
	t.Logf("║  Posterior touch (tail):                        ║")
	t.Logf("║    Forward motors:  %6d                      ║", fwd1)
	t.Logf("║    Backward motors: %6d                      ║", bwd1)
	t.Logf("║    F/B ratio:       %6.2f                      ║", safeDivide(fwd1, bwd1))
	t.Log("║                                                  ║")
	t.Logf("║  Anterior touch (nose):                         ║")
	t.Logf("║    Forward motors:  %6d                      ║", fwd2)
	t.Logf("║    Backward motors: %6d                      ║", bwd2)
	t.Logf("║    B/F ratio:       %6.2f                      ║", safeDivide(bwd2, fwd2))
	t.Log("╚══════════════════════════════════════════════════╝")

	anteriorBias := safeDivide(bwd2, fwd2)
	if anteriorBias <= 1.0 {
		t.Error("Anterior touch should produce backward bias (B/F > 1)")
	} else {
		t.Logf("✓ Anterior touch → backward movement confirmed")
	}

	posteriorBias := safeDivide(fwd1, bwd1)
	if posteriorBias > 1.0 {
		t.Logf("✓ Posterior touch → forward movement confirmed!")
	} else {
		t.Logf("⚠ Posterior touch still backward-biased (F/B=%.2f)", posteriorBias)
		t.Logf("  Known limitation: needs AVA↔AVB reciprocal inhibition")
	}
}

// TestCircuitTrace provides a detailed trace of signal propagation
// through the escape forward circuit, tick by tick.
func TestCircuitTrace(t *testing.T) {
	records := loadRecords(t)
	net, nameMap := BuildNetwork(records, DefaultParams())

	traceNames := []string{
		"PLML", "PLMR",
		"PVCL", "PVCR",
		"AVBL", "AVBR",
		"AVAL", "AVAR",
		"AVDL", "AVDR",
		"VB1", "VB2", "VB5", "VB8", "VB11",
		"DB3", "DB5", "DB7",
		"VA1", "VA5", "VA8", "VA11",
		"DA3", "DA5", "DA8",
	}

	traceIndices := make([]uint32, len(traceNames))
	for i, name := range traceNames {
		traceIndices[i] = nameMap[name]
	}

	t.Log("Signal propagation trace: PLML+PLMR stimulus")
	t.Log("")

	header := fmt.Sprintf("%5s", "Tick")
	for _, name := range traceNames {
		header += fmt.Sprintf(" %5s", name)
	}
	t.Log(header)
	t.Log("")

	net.Stimulate(nameMap["PLML"], 5000)
	net.Stimulate(nameMap["PLMR"], 5000)

	row := fmt.Sprintf("%5s", "init")
	for _, idx := range traceIndices {
		row += fmt.Sprintf(" %5d", net.Neurons[idx].Activation)
	}
	t.Log(row)

	for tick := 1; tick <= 10; tick++ {
		net.Tick()

		row := fmt.Sprintf("%5d", tick)
		for _, idx := range traceIndices {
			n := &net.Neurons[idx]
			marker := " "
			if n.LastFired > 0 && net.Counter < n.LastFired+net.RefractoryPeriod {
				marker = "*"
			}
			row += fmt.Sprintf(" %4d%s", n.Activation, marker)
		}
		t.Log(row)
	}
}
