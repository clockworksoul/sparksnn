package biomimetic

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestXORPerturbation uses weight perturbation learning:
// 1. Evaluate current accuracy
// 2. Randomly perturb a weight
// 3. Re-evaluate
// 4. Keep perturbation if accuracy improves, revert if worse
//
// This provides proper credit assignment per connection.
func TestXORPerturbation(t *testing.T) {
	type sample struct {
		a, b  bool
		label int
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
	inputStim := int32(1000)
	ticksPerSample := 30
	restTicks := 10

	// Evaluate all 4 patterns, return number correct
	evaluate := func(net *Network) int {
		correct := 0
		outStart := uint32(len(net.Neurons) - 2)
		for _, p := range patterns {
			out0, out1 := 0, 0
			for tick := 0; tick < ticksPerSample; tick++ {
				if p.a {
					net.Stimulate(0, inputStim)
				}
				if p.b {
					net.Stimulate(1, inputStim)
				}
				net.Tick()
				if net.Neurons[outStart].LastFired == net.Counter {
					out0++
				}
				if net.Neurons[outStart+1].LastFired == net.Counter {
					out1++
				}
			}
			net.TickN(uint32(restTicks))

			predicted := 0
			if out1 > out0 {
				predicted = 1
			}
			if predicted == p.label {
				correct++
			}
		}
		return correct
	}

	// Collect all learnable connection pointers
	type connRef struct {
		neuronIdx int
		connIdx   int
	}

	for _, hiddenSize := range []int{2, 4, 8, 16} {
		t.Run(fmt.Sprintf("hidden=%d", hiddenSize), func(t *testing.T) {
			total := 2 + hiddenSize + 2
			outStart := uint32(2 + hiddenSize)
			_ = outStart

			successes := 0
			numTrials := 50

			for trial := 0; trial < numTrials; trial++ {
				net := NewNetwork(uint32(total), 0, threshold, decayRate, refractoryPeriod)

				initW := func() int32 { return int32(rand.IntN(800)) + 100 }

				// Wire: input→hidden, hidden→output
				for i := uint32(0); i < 2; i++ {
					for h := uint32(2); h < uint32(2+hiddenSize); h++ {
						net.Connect(i, h, initW())
					}
				}
				for h := uint32(2); h < uint32(2+hiddenSize); h++ {
					net.Connect(h, uint32(2+hiddenSize), initW())
					net.Connect(h, uint32(2+hiddenSize+1), initW())
				}

				// Collect learnable connections
				var conns []connRef
				for i := 0; i < 2+hiddenSize; i++ {
					for j := range net.Neurons[i].Connections {
						conns = append(conns, connRef{i, j})
					}
				}

				bestScore := evaluate(net)
				perturbSize := int32(200)
				noImproveCount := 0

				for step := 0; step < 10000 && bestScore < 4; step++ {
					// Pick a random connection
					ref := conns[rand.IntN(len(conns))]
					conn := &net.Neurons[ref.neuronIdx].Connections[ref.connIdx]

					// Save and perturb
					oldWeight := conn.Weight
					delta := int32(rand.IntN(int(perturbSize)*2+1)) - perturbSize
					conn.Weight = ClampAdd(conn.Weight, delta)

					score := evaluate(net)

					if score > bestScore {
						bestScore = score
						noImproveCount = 0
					} else if score < bestScore {
						// Revert
						conn.Weight = oldWeight
						noImproveCount++
					} else {
						// Same score — keep with 50% probability (exploration)
						if rand.IntN(2) == 0 {
							conn.Weight = oldWeight
						}
						noImproveCount++
					}

					// Adaptive perturbation size
					if noImproveCount > 200 {
						perturbSize = min(perturbSize*2, 2000)
						noImproveCount = 0
					}
				}

				if bestScore == 4 {
					successes++
				}
			}

			rate := float64(successes) / float64(numTrials) * 100
			t.Logf("Hidden=%d: %d/%d trials succeeded (%.1f%%)", hiddenSize, successes, numTrials, rate)
			if successes == 0 {
				t.Errorf("No trial achieved 100%% accuracy with %d hidden neurons", hiddenSize)
			}
		})
	}
}
