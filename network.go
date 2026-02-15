package biomimetic

// Network is a collection of neurons stored in a contiguous array.
// All neuron references use uint32 indices into this array for cache
// locality and memory efficiency.
type Network struct {
	// Neurons is the contiguous array of all neurons in the network.
	Neurons []Neuron

	// Counter is the global time counter. Incremented by the caller
	// to represent the passage of time. Used for decay calculations
	// and refractory period tracking.
	Counter uint32

	// DecayRate controls how quickly activation decays toward baseline.
	// Expressed as a fixed-point fraction of 65536:
	//   64000 = ~97.7% retention per tick (slow decay)
	//   58982 = ~90% retention per tick
	//   32768 = 50% retention per tick (fast decay)
	DecayRate uint16

	// RefractoryPeriod is the number of counter ticks after firing
	// during which a neuron cannot fire again.
	RefractoryPeriod uint32

	// MaxPropagationDepth limits the depth of cascading propagation
	// to prevent runaway chains. A value of 0 means no limit (use
	// refractory periods alone for cycle prevention).
	MaxPropagationDepth uint32
}

// NewNetwork creates a network with the given number of neurons.
// All neurons are initialized to the same baseline, threshold, and
// zero activation.
func NewNetwork(size uint32, baseline, threshold int16, decayRate uint16, refractoryPeriod uint32) *Network {
	neurons := make([]Neuron, size)
	for i := range neurons {
		neurons[i] = Neuron{
			Activation: baseline,
			Baseline:   baseline,
			Threshold:  threshold,
		}
	}

	return &Network{
		Neurons:          neurons,
		DecayRate:        decayRate,
		RefractoryPeriod: refractoryPeriod,
	}
}

// Connect adds a directed connection from neuron at index `from` to
// neuron at index `to` with the given weight.
func (net *Network) Connect(from, to uint32, weight int16) {
	net.Neurons[from].Connections = append(net.Neurons[from].Connections, Connection{
		Target: to,
		Weight: weight,
	})
}

// Stimulate sends an external signal to a specific neuron and
// propagates any resulting cascade through the network.
func (net *Network) Stimulate(index uint32, weight int16) {
	net.stimulate(index, weight, 0)
}

// stimulate is the internal recursive stimulation function with
// depth tracking.
func (net *Network) stimulate(index uint32, weight int16, depth uint32) {
	if index >= uint32(len(net.Neurons)) {
		return
	}

	if net.MaxPropagationDepth > 0 && depth >= net.MaxPropagationDepth {
		return
	}

	neuron := &net.Neurons[index]
	fired := neuron.Stimulate(weight, net.Counter, net.DecayRate)

	if fired {
		// Set refractory period
		neuron.RefractoryUntil = net.Counter + net.RefractoryPeriod

		// Reset activation after firing (to baseline)
		neuron.Activation = neuron.Baseline

		// Propagate to all outgoing connections
		for _, conn := range neuron.Connections {
			net.stimulate(conn.Target, conn.Weight, depth+1)
		}
	}
}

// Tick advances the network's counter by one step. This represents
// the passage of time. No computation happens — decay is lazy and
// only calculated when a neuron is next stimulated.
func (net *Network) Tick() {
	net.Counter++
}

// TickN advances the counter by n steps.
func (net *Network) TickN(n uint32) {
	net.Counter += n
}

// ActiveNeurons returns the indices of all neurons whose activation
// is above a given threshold. Useful for reading output from the
// network (population coding).
//
// Note: this does NOT apply decay. It reads raw stored activation
// values. For accurate readings, consider calling at a consistent
// point relative to stimulation.
func (net *Network) ActiveNeurons(above int16) []uint32 {
	var active []uint32
	for i, n := range net.Neurons {
		if n.Activation > above {
			active = append(active, uint32(i))
		}
	}
	return active
}
