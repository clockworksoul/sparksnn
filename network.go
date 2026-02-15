package biomimetic

// PendingStimulation represents a signal scheduled to arrive at a
// neuron on the next tick. Created when a neuron fires — its
// downstream targets receive stimulation one tick later, modeling
// biological signal propagation delay.
type PendingStimulation struct {
	Target uint32
	Weight int16
}

// Network is a collection of neurons stored in a contiguous array.
// All neuron references use uint32 indices into this array for cache
// locality and memory efficiency.
//
// Signals propagate with a fixed 1-tick delay per hop. When a neuron
// fires, its downstream targets are added to a pending queue and
// processed on the next Tick(). This replaces recursive instant
// propagation with tick-driven temporal propagation.
type Network struct {
	// Neurons is the contiguous array of all neurons in the network.
	Neurons []Neuron

	// Counter is the global time counter. Incremented on each Tick().
	// Used for decay calculations and refractory period tracking.
	Counter uint32

	// DefaultDecayRate is the decay rate assigned to new neurons.
	// Stored for reference; individual neurons may diverge.
	DefaultDecayRate uint16

	// RefractoryPeriod is the number of counter ticks after firing
	// during which a neuron cannot fire again.
	RefractoryPeriod uint32

	// pending holds stimulations to process during the current Tick().
	pending []PendingStimulation

	// nextPending collects stimulations generated during the current
	// Tick(), to be processed on the next Tick().
	nextPending []PendingStimulation
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
			DecayRate:  decayRate,
		}
	}

	return &Network{
		Neurons:          neurons,
		DefaultDecayRate: decayRate,
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

// Stimulate sends an external signal to a specific neuron. If the
// neuron fires, its downstream targets are queued for the next tick.
// This is the entry point for injecting input into the network.
func (net *Network) Stimulate(index uint32, weight int16) {
	if index >= uint32(len(net.Neurons)) {
		return
	}

	neuron := &net.Neurons[index]
	fired := neuron.Stimulate(weight, net.Counter)

	if fired {
		net.fire(neuron)
	}
}

// fire handles a neuron that has exceeded its threshold: sets the
// refractory period, resets activation, and queues downstream
// stimulations for the next tick.
func (net *Network) fire(neuron *Neuron) {
	neuron.RefractoryUntil = net.Counter + net.RefractoryPeriod
	neuron.Activation = neuron.Baseline

	for _, conn := range neuron.Connections {
		net.nextPending = append(net.nextPending, PendingStimulation{
			Target: conn.Target,
			Weight: conn.Weight,
		})
	}
}

// Tick advances the counter by one step and processes all pending
// stimulations from the previous tick. Any neurons that fire during
// processing queue their downstream targets for the *next* tick.
//
// Returns the number of neurons that fired during this tick.
func (net *Network) Tick() int {
	net.Counter++

	// Swap: pending becomes current, nextPending becomes the new
	// accumulator. Reuse backing arrays to reduce allocations.
	net.pending, net.nextPending = net.nextPending, net.pending[:0]

	fired := 0
	for _, stim := range net.pending {
		if stim.Target >= uint32(len(net.Neurons)) {
			continue
		}

		neuron := &net.Neurons[stim.Target]
		if neuron.Stimulate(stim.Weight, net.Counter) {
			net.fire(neuron)
			fired++
		}
	}

	return fired
}

// TickN advances the counter by n steps, processing pending
// stimulations at each tick. Returns total neurons fired across
// all ticks.
func (net *Network) TickN(n uint32) int {
	total := 0
	for i := uint32(0); i < n; i++ {
		total += net.Tick()
	}
	return total
}

// Pending returns the number of stimulations queued for the next tick.
func (net *Network) Pending() int {
	return len(net.nextPending)
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
