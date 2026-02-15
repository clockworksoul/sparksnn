package biomimetic

import "testing"

func TestClampAdd(t *testing.T) {
	tests := []struct {
		name     string
		base     int16
		delta    int16
		expected int16
	}{
		{"normal positive", 100, 50, 150},
		{"normal negative", 100, -50, 50},
		{"zero", 0, 0, 0},
		{"overflow clamped", MaxActivation - 10, 100, MaxActivation},
		{"underflow clamped", MinActivation + 10, -100, MinActivation},
		{"max + positive", MaxActivation, 1, MaxActivation},
		{"min + negative", MinActivation, -1, MinActivation},
		{"cross zero positive", -50, 100, 50},
		{"cross zero negative", 50, -100, -50},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampAdd(tt.base, tt.delta)
			if got != tt.expected {
				t.Errorf("clampAdd(%d, %d) = %d, want %d",
					tt.base, tt.delta, got, tt.expected)
			}
		})
	}
}

func TestNeuronDecay(t *testing.T) {
	n := Neuron{
		Activation:      1000,
		Baseline:        0,
		Threshold:       500,
		LastInteraction:  0,
		RefractoryUntil: 0,
	}

	// 50% retention per tick, 1 tick elapsed
	n.decay(1, 32768)

	// 1000 * 32768/65536 = 500
	if n.Activation != 500 {
		t.Errorf("after 1 tick at 50%% decay: got %d, want 500", n.Activation)
	}

	// Another tick: 500 * 0.5 = 250
	n.decay(2, 32768)
	if n.Activation != 250 {
		t.Errorf("after 2nd tick at 50%% decay: got %d, want 250", n.Activation)
	}
}

func TestNeuronDecayLargeElapsed(t *testing.T) {
	n := Neuron{
		Activation:      1000,
		Baseline:        0,
		Threshold:       500,
		LastInteraction:  0,
		RefractoryUntil: 0,
	}

	// Large elapsed should decay fully to baseline
	n.decay(100, 32768)
	if n.Activation != 0 {
		t.Errorf("after large elapsed: got %d, want 0 (baseline)", n.Activation)
	}
}

func TestNeuronStimulateNoFire(t *testing.T) {
	n := Neuron{
		Activation:      0,
		Baseline:        0,
		Threshold:       1000,
		LastInteraction:  0,
		RefractoryUntil: 0,
	}

	// Weight of 500 shouldn't reach threshold of 1000
	fired := n.Stimulate(500, 0, 32768)
	if fired {
		t.Error("neuron should not have fired")
	}
	if n.Activation != 500 {
		t.Errorf("activation should be 500, got %d", n.Activation)
	}
}

func TestNeuronStimulateFires(t *testing.T) {
	n := Neuron{
		Activation:      900,
		Baseline:        0,
		Threshold:       1000,
		LastInteraction:  0,
		RefractoryUntil: 0,
	}

	// 900 + 200 = 1100 > 1000 threshold
	fired := n.Stimulate(200, 0, 32768)
	if !fired {
		t.Error("neuron should have fired")
	}
}

func TestNeuronRefractoryPeriod(t *testing.T) {
	n := Neuron{
		Activation:      2000,
		Baseline:        0,
		Threshold:       1000,
		LastInteraction:  5,
		RefractoryUntil: 10, // Can't fire until counter >= 10
	}

	// Above threshold but in refractory period
	fired := n.Stimulate(100, 5, 32768)
	if fired {
		t.Error("neuron should not fire during refractory period")
	}

	// Now past refractory period — stimulate heavily to ensure above threshold after decay
	fired = n.Stimulate(5000, 10, 32768)
	if !fired {
		t.Error("neuron should fire after refractory period")
	}
}

func TestNeuronInhibition(t *testing.T) {
	n := Neuron{
		Activation:      500,
		Baseline:        0,
		Threshold:       1000,
		LastInteraction:  0,
		RefractoryUntil: 0,
	}

	// Inhibitory signal
	fired := n.Stimulate(-300, 0, 32768)
	if fired {
		t.Error("neuron should not fire with inhibitory signal")
	}
	if n.Activation != 200 {
		t.Errorf("activation should be 200 after inhibition, got %d", n.Activation)
	}
}
