package biomimetic

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestXORScaling tests whether the asymmetric reward learning rule
// scales from 2 to 4 to 8 to 16 hidden neurons on XOR.
// Uses the exact same approach as TestMinimalXORLearning (which works).
func TestXORScaling(t *testing.T) {
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

	for _, hiddenSize := range []int{2, 4, 8, 16} {
		t.Run(fmt.Sprintf("hidden=%d", hiddenSize), func(t *testing.T) {
			threshold := int32(500)
			decayRate := uint16(45000)
			refractoryPeriod := uint32(3)
			inputStim := int32(1000)
			ticksPerSample := 50
			restTicks := 20
			numTrials := 100
			epochs := 500

			// Network layout:
			// [0, 1]           = inputs
			// [2, 2+hidden)    = hidden
			// [2+hidden]       = output 0
			// [2+hidden+1]     = output 1
			total := 2 + hiddenSize + 2
			outStart := uint32(2 + hiddenSize)

			successes := 0

			for trial := 0; trial < numTrials; trial++ {
				net := NewNetwork(uint32(total), 0, threshold, decayRate, refractoryPeriod)

				initW := func() int32 { return int32(rand.IntN(800)) + 100 }

				// Input → Hidden
				for i := uint32(0); i < 2; i++ {
					for h := uint32(2); h < uint32(2+hiddenSize); h++ {
						net.Connect(i, h, initW())
					}
				}
				// Hidden → Output
				for h := uint32(2); h < uint32(2+hiddenSize); h++ {
					net.Connect(h, outStart, initW())
					net.Connect(h, outStart+1, initW())
				}

				for epoch := 0; epoch < epochs; epoch++ {
					perm := rand.Perm(4)
					for _, pi := range perm {
						p := patterns[pi]

						out0Spikes, out1Spikes := 0, 0
						for tick := 0; tick < ticksPerSample; tick++ {
							if p.a {
								net.Stimulate(0, inputStim)
							}
							if p.b {
								net.Stimulate(1, inputStim)
							}
							net.Tick()

							if net.Neurons[outStart].LastFired == net.Counter {
								out0Spikes++
							}
							if net.Neurons[outStart+1].LastFired == net.Counter {
								out1Spikes++
							}
						}
						net.TickN(uint32(restTicks))

						predicted := 0
						if out1Spikes > out0Spikes {
							predicted = 1
						}

						reward := int32(-200)
						if predicted == p.label {
							reward = 300
						}

						// Asymmetric activity-reward update
						for i := uint32(0); i < uint32(2+hiddenSize); i++ {
							src := &net.Neurons[i]
							srcActive := src.LastFired > 0 && net.Counter-src.LastFired < 60
							for j := range src.Connections {
								conn := &src.Connections[j]
								tgt := &net.Neurons[conn.Target]
								tgtActive := tgt.LastFired > 0 && net.Counter-tgt.LastFired < 60

								if srcActive && tgtActive {
									conn.Weight = ClampAdd(conn.Weight, reward/4)
								} else if srcActive && !tgtActive {
									conn.Weight = ClampAdd(conn.Weight, -reward/8)
								}
							}
						}
					}
				}

				// Evaluate
				correct := 0
				for _, p := range patterns {
					out0Total, out1Total := 0, 0
					for ev := 0; ev < 5; ev++ {
						for tick := 0; tick < ticksPerSample; tick++ {
							if p.a {
								net.Stimulate(0, inputStim)
							}
							if p.b {
								net.Stimulate(1, inputStim)
							}
							net.Tick()
							if net.Neurons[outStart].LastFired == net.Counter {
								out0Total++
							}
							if net.Neurons[outStart+1].LastFired == net.Counter {
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

				if correct == 4 {
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
