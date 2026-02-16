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
		DecayRate:       32768, // 50% retention per tick
		LastInteraction: 0,
	}

	// 50% retention per tick, 1 tick elapsed
	n.decay(1)

	// 1000 * 32768/65536 = 500
	if n.Activation != 500 {
		t.Errorf("after 1 tick at 50%% decay: got %d, want 500", n.Activation)
	}

	// Another tick: 500 * 0.5 = 250
	n.decay(2)
	if n.Activation != 250 {
		t.Errorf("after 2nd tick at 50%% decay: got %d, want 250", n.Activation)
	}
}

func TestNeuronDecayLargeElapsed(t *testing.T) {
	n := Neuron{
		Activation:      1000,
		Baseline:        0,
		Threshold:       500,
		DecayRate:       32768,
		LastInteraction: 0,
	}

	// Large elapsed should decay fully to baseline
	n.decay(100)
	if n.Activation != 0 {
		t.Errorf("after large elapsed: got %d, want 0 (baseline)", n.Activation)
	}
}

func TestNeuronStimulateNoFire(t *testing.T) {
	n := Neuron{
		Activation:      0,
		Baseline:        0,
		Threshold:       1000,
		DecayRate:       32768,
		LastInteraction: 0,
	}

	// Weight of 500 shouldn't reach threshold of 1000
	fired := n.Stimulate(500, 0, 2)
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
		DecayRate:       32768,
		LastInteraction: 0,
	}

	// 900 + 200 = 1100 > 1000 threshold
	fired := n.Stimulate(200, 0, 2)
	if !fired {
		t.Error("neuron should have fired")
	}
}

func TestNeuronRefractoryPeriod(t *testing.T) {
	n := Neuron{
		Activation:      2000,
		Baseline:        0,
		Threshold:       1000,
		DecayRate:       32768,
		LastInteraction: 5,
		LastFired:       5,
		HasFired:        true,
	}

	// Above threshold but in refractory period (fired at 5, refractory=5, so blocked until 10)
	fired := n.Stimulate(100, 7, 5)
	if fired {
		t.Error("neuron should not fire during refractory period")
	}

	// Now past refractory period — stimulate heavily to ensure above threshold after decay
	fired = n.Stimulate(5000, 10, 5)
	if !fired {
		t.Error("neuron should fire after refractory period")
	}
}

func TestNeuronRefractoryFirstFire(t *testing.T) {
	// A neuron that has never fired should not be blocked by refractory
	n := Neuron{
		Activation:      0,
		Baseline:        0,
		Threshold:       100,
		DecayRate:       58982,
		LastInteraction: 0,
	}

	// Even at tick 0 with refractory=5, a never-fired neuron should fire
	fired := n.Stimulate(500, 0, 5)
	if !fired {
		t.Error("never-fired neuron should not be blocked by refractory period")
	}
}

func TestNeuronInhibition(t *testing.T) {
	n := Neuron{
		Activation:      500,
		Baseline:        0,
		Threshold:       1000,
		DecayRate:       32768,
		LastInteraction: 0,
	}

	// Inhibitory signal
	fired := n.Stimulate(-300, 0, 2)
	if fired {
		t.Error("neuron should not fire with inhibitory signal")
	}
	if n.Activation != 200 {
		t.Errorf("activation should be 200 after inhibition, got %d", n.Activation)
	}
}
