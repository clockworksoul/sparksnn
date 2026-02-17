package biomimetic

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestMinimalXORLearning tries to learn XOR with R-STDP on the
// smallest possible network: 2 inputs, 2 hidden, 2 outputs.
// Same topology as the handwired test, but with random initial
// weights and reward-modulated learning.
func TestMinimalXORLearning(t *testing.T) {
	// Network: 0=InA, 1=InB, 2=H1, 3=H2, 4=Out0, 5=Out1
	// Target: H1 = "A AND NOT B", H2 = "NOT A AND B"
	// Out1 = H1 OR H2 (class 1), Out0 = neither (class 0)

	type sample struct {
		a, b  bool
		label int // 0 or 1
	}
	patterns := []sample{
		{false, false, 0},
		{true, false, 1},
		{false, true, 1},
		{true, true, 0},
	}

	threshold := int32(500)
	decayRate := uint16(45000)
	refractoryPeriod := uint32(3)
	inputStim := int32(1000) // strong input

	// Simple reward-modulated STDP inline (no separate package dependency)
	// We'll manually track what we need.

	best := 0
	bestWeights := ""

	for trial := 0; trial < 50; trial++ {
		net := NewNetwork(6, 0, threshold, decayRate, refractoryPeriod)

		// Random initial weights: input → hidden (4 connections)
		// Hidden → output (4 connections)
		initW := func() int32 { return int32(rand.IntN(800)) + 100 }

		net.Connect(0, 2, initW()) // A→H1
		net.Connect(1, 2, initW()) // B→H1
		net.Connect(0, 3, initW()) // A→H2
		net.Connect(1, 3, initW()) // B→H2
		net.Connect(2, 4, initW()) // H1→Out0
		net.Connect(2, 5, initW()) // H1→Out1
		net.Connect(3, 4, initW()) // H2→Out0
		net.Connect(3, 5, initW()) // H2→Out1

		ticksPerSample := 50
		restTicks := 20

		for epoch := 0; epoch < 200; epoch++ {
			// Shuffle patterns each epoch
			perm := rand.Perm(4)
			for _, pi := range perm {
				p := patterns[pi]

				// Present sample
				out0Spikes, out1Spikes := 0, 0
				for tick := 0; tick < ticksPerSample; tick++ {
					if p.a {
						net.Stimulate(0, inputStim)
					}
					if p.b {
						net.Stimulate(1, inputStim)
					}
					net.Tick()

					if net.Neurons[4].LastFired == net.Counter {
						out0Spikes++
					}
					if net.Neurons[5].LastFired == net.Counter {
						out1Spikes++
					}
				}

				// Rest
				net.TickN(uint32(restTicks))

				// Classify
				predicted := 0
				if out1Spikes > out0Spikes {
					predicted = 1
				}

				// Reward signal: apply weight perturbation
				reward := int32(-200)
				if predicted == p.label {
					reward = 300
				}

				// Simple reward-modulated weight update:
				// For each learnable connection, adjust based on
				// recent activity correlation × reward.
				// This is a simplified "node perturbation" approach.
				for i := 0; i < 4; i++ { // neurons 0-3 have learnable connections
					for j := range net.Neurons[i].Connections {
						conn := &net.Neurons[i].Connections[j]
						src := &net.Neurons[i]
						tgt := &net.Neurons[conn.Target]

						// Both source and target were active recently?
						srcActive := src.LastFired > 0 && net.Counter-src.LastFired < 60
						tgtActive := tgt.LastFired > 0 && net.Counter-tgt.LastFired < 60

						if srcActive && tgtActive {
							// Strengthen/weaken based on reward
							delta := reward / 4
							conn.Weight = ClampAdd(conn.Weight, delta)
						} else if srcActive && !tgtActive {
							// Source fired but target didn't — small opposite push
							delta := -reward / 8
							conn.Weight = ClampAdd(conn.Weight, delta)
						}
					}
				}
			}
		}

		// Evaluate
		correct := 0
		for _, p := range patterns {
			out0Total, out1Total := 0, 0
			for trial := 0; trial < 5; trial++ {
				for tick := 0; tick < ticksPerSample; tick++ {
					if p.a {
						net.Stimulate(0, inputStim)
					}
					if p.b {
						net.Stimulate(1, inputStim)
					}
					net.Tick()

					if net.Neurons[4].LastFired == net.Counter {
						out0Total++
					}
					if net.Neurons[5].LastFired == net.Counter {
						out1Total++
					}
				}
				net.TickN(uint32(restTicks))
			}

			predicted := 0
			if out1Total > out0Total {
				predicted = 1
			}
			if predicted == p.label {
				correct++
			}
		}

		if correct > best {
			best = correct
			weights := ""
			for i := 0; i < 4; i++ {
				for _, c := range net.Neurons[i].Connections {
					weights += fmt.Sprintf("  %d→%d: %d\n", i, c.Target, c.Weight)
				}
			}
			bestWeights = weights
		}

		if correct == 4 {
			t.Logf("✅ Trial %d: 4/4 correct!", trial)
			t.Logf("Weights:\n%s", bestWeights)
			return
		}
	}

	t.Logf("Best: %d/4 correct", best)
	t.Logf("Best weights:\n%s", bestWeights)
	if best < 3 {
		t.Errorf("Expected at least 3/4 correct, got %d/4", best)
	}
}
