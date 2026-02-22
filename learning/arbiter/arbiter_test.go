package arbiter

import (
	"testing"

	bio "github.com/clockworksoul/biomimetic-network"
)

// TestStdpWindow verifies the linear STDP window approximation.
func TestStdpWindow(t *testing.T) {
	// At dt=0, should return full amplitude
	got := stdpWindow(0, 100, 10)
	if got != 100 {
		t.Errorf("stdpWindow(0, 100, 10) = %d, want 100", got)
	}

	// At dt=tau, should return 0
	got = stdpWindow(10, 100, 10)
	if got != 0 {
		t.Errorf("stdpWindow(10, 100, 10) = %d, want 0", got)
	}

	// At dt=5 with tau=10, should return 50
	got = stdpWindow(5, 100, 10)
	if got != 50 {
		t.Errorf("stdpWindow(5, 100, 10) = %d, want 50", got)
	}

	// Beyond 6*tau, should return 0
	got = stdpWindow(61, 100, 10)
	if got != 0 {
		t.Errorf("stdpWindow(61, 100, 10) = %d, want 0", got)
	}
}

// TestCausalSTDP tests that pre-before-post timing strengthens
// connections.
func TestCausalSTDP(t *testing.T) {
	cfg := DefaultConfig()
	cfg.APlus = 100
	cfg.TauPlus = 10

	layers := []LayerSpec{
		{Start: 0, End: 2},
		{Start: 2, End: 4},
	}
	rule := NewRule(cfg, layers)

	conn := &bio.Connection{Target: 2, Weight: 500}

	// Source fired at tick 5 (encoded as 6), post fires at tick 7
	incoming := []bio.IncomingConnection{
		{SourceIndex: 6, Conn: conn},
	}
	rule.OnPostFire(incoming, 7)

	// dt=2, tau=10: delta = 100 * (10-2)/10 = 80
	if conn.Weight != 580 {
		t.Errorf("after causal STDP, weight = %d, want 580", conn.Weight)
	}
}

// TestAntiCausalSTDP tests that post-before-pre timing weakens
// connections.
func TestAntiCausalSTDP(t *testing.T) {
	cfg := DefaultConfig()
	cfg.AMinus = 100
	cfg.TauMinus = 10

	layers := []LayerSpec{
		{Start: 0, End: 2},
		{Start: 2, End: 4},
	}
	rule := NewRule(cfg, layers)

	conn := &bio.Connection{Target: 2, Weight: 500}

	// Post fired at tick 5, pre fires at tick 7
	rule.OnSpikePropagation(conn, 7, 5)

	// dt=2, tau=10: delta = 100 * (10-2)/10 = 80, applied as -80
	if conn.Weight != 420 {
		t.Errorf("after anti-causal STDP, weight = %d, want 420", conn.Weight)
	}
}

// TestSignalErrorCorrect verifies that correct predictions don't
// trigger depression.
func TestSignalErrorCorrect(t *testing.T) {
	net := bio.NewNetwork(6, 0, 100, 58982, 3)
	// Connect neuron 0 -> neuron 2 with weight 500
	net.Connect(0, 2, 500)

	layers := []LayerSpec{
		{Start: 0, End: 2},
		{Start: 2, End: 4, ArbiterStart: 4, ArbiterEnd: 6},
	}

	cfg := DefaultConfig()
	rule := NewRule(cfg, layers)
	net.LearningRule = rule

	// Fire neuron 0 so it's recently active
	net.Neurons[0].LastFired = net.Counter

	// Correct prediction: class 1 with highest spikes
	rule.SignalError(net, 1, []int{0, 5, 0})

	// Weight should be unchanged (no depression)
	if net.Neurons[0].Connections[0].Weight != 500 {
		t.Errorf("weight after correct = %d, want 500",
			net.Neurons[0].Connections[0].Weight)
	}
}

// TestSignalErrorWrong verifies that incorrect predictions trigger
// targeted correction of hidden→output connections.
func TestSignalErrorWrong(t *testing.T) {
	// Build a small network:
	// 2 input (0,1), 2 hidden (2,3), 3 output (4,5,6), 2 arbiter (7,8)
	net := bio.NewNetwork(9, 0, 100, 58982, 3)

	// Hidden→Output connections
	net.Connect(2, 4, 500) // hidden 0 → output 0
	net.Connect(2, 5, 500) // hidden 0 → output 1
	net.Connect(2, 6, 500) // hidden 0 → output 2

	layers := []LayerSpec{
		{Start: 0, End: 2},                              // input
		{Start: 2, End: 4, ArbiterStart: 7, ArbiterEnd: 9}, // hidden
		{Start: 4, End: 7},                              // output
	}

	cfg := DefaultConfig()
	cfg.DepressionStrength = 100
	rule := NewRule(cfg, layers)
	net.LearningRule = rule

	// Fire hidden neuron 2 so it's recently active
	net.Neurons[2].LastFired = net.Counter

	// Wrong prediction: correct is 0, but output 1 fired most
	rule.SignalError(net, 0, []int{1, 5, 0})

	// Connection to correct output (4) should be STRENGTHENED
	if net.Neurons[2].Connections[0].Weight <= 500 {
		t.Errorf("weight to correct output = %d, want > 500",
			net.Neurons[2].Connections[0].Weight)
	}

	// Connection to wrong output that fired (5) should be DEPRESSED
	if net.Neurons[2].Connections[1].Weight >= 500 {
		t.Errorf("weight to wrong output = %d, want < 500",
			net.Neurons[2].Connections[1].Weight)
	}
}

// TestWeightClamping verifies MaxWeightMagnitude is enforced.
func TestWeightClamping(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxWeightMagnitude = 1000
	cfg.APlus = 500
	cfg.TauPlus = 10

	layers := []LayerSpec{{Start: 0, End: 2}}
	rule := NewRule(cfg, layers)

	conn := &bio.Connection{Target: 1, Weight: 900}
	incoming := []bio.IncomingConnection{
		{SourceIndex: 2, Conn: conn}, // source fired at tick 1
	}
	rule.OnPostFire(incoming, 2) // dt=1

	if conn.Weight > 1000 {
		t.Errorf("weight %d exceeds max magnitude 1000", conn.Weight)
	}
}
