package biomimetic

import "testing"

// TestHandwiredXOR constructs an XOR circuit by hand to prove the
// architecture can represent XOR. No learning — just manually chosen
// weights. If this fails, the problem is in the neuron model, not
// the learning rules.
//
// Circuit:
//   Input A (0) ──(+excite)──→ H1 (2)  "A AND NOT B"
//   Input B (1) ──(-inhib)───→ H1 (2)
//   Input A (0) ──(-inhib)───→ H2 (3)  "NOT A AND B"
//   Input B (1) ──(+excite)──→ H2 (3)
//   H1 (2) ──(+excite)──→ Output (4)
//   H2 (3) ──(+excite)──→ Output (4)
//
func TestHandwiredXOR(t *testing.T) {
	// Neurons: 0=InputA, 1=InputB, 2=H1, 3=H2, 4=Output
	// Use threshold that requires strong excitation to fire.
	baseline := int32(0)
	threshold := int32(1000)
	decayRate := uint16(0) // no decay within a trial
	refractoryPeriod := uint32(0)

	cases := []struct {
		name   string
		a, b   bool
		expect bool // should output fire?
	}{
		{"0 XOR 0 = 0", false, false, false},
		{"1 XOR 0 = 1", true, false, true},
		{"0 XOR 1 = 1", false, true, true},
		{"1 XOR 1 = 0", true, true, false},
	}

	excite := int32(2000)  // well above threshold
	inhib := int32(-3000)  // strong enough to cancel excitation

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			net := NewNetwork(5, baseline, threshold, decayRate, refractoryPeriod)

			// Wire the circuit
			net.Connect(0, 2, excite) // A → H1 (excitatory)
			net.Connect(1, 2, inhib)  // B → H1 (inhibitory)
			net.Connect(0, 3, inhib)  // A → H2 (inhibitory)
			net.Connect(1, 3, excite) // B → H2 (excitatory)
			net.Connect(2, 4, excite) // H1 → Output
			net.Connect(3, 4, excite) // H2 → Output

			// Stimulate inputs
			if tc.a {
				net.Stimulate(0, excite)
			}
			if tc.b {
				net.Stimulate(1, excite)
			}

			// Tick 1: input signals propagate to hidden layer
			net.Tick()

			// Tick 2: hidden layer signals propagate to output
			net.Tick()

			outputFired := net.Neurons[4].LastFired > 0

			if outputFired != tc.expect {
				t.Errorf("got outputFired=%v, want %v (output activation=%d, H1.LastFired=%d, H2.LastFired=%d)",
					outputFired, tc.expect,
					net.Neurons[4].Activation,
					net.Neurons[2].LastFired,
					net.Neurons[3].LastFired)
			}
		})
	}
}
