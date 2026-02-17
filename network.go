package biomimetic

// PendingStimulation represents a signal scheduled to arrive at a
// neuron on the next tick. Created when a neuron fires — its
// downstream targets receive stimulation one tick later, modeling
// biological signal propagation delay.
type PendingStimulation struct {
	Target uint32
	Weight int32
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

	// PostFireReset is the activation level a neuron is set to after
	// firing. Models hyperpolarization when negative. Only used when
	// UsePostFireReset is true; otherwise neurons reset to Baseline.
	PostFireReset int32

	// UsePostFireReset enables the PostFireReset value instead of
	// resetting to Baseline after firing.
	UsePostFireReset bool

	// LearningRule is the active plasticity mechanism. If nil,
	// no learning occurs (equivalent to NoOpLearning).
	LearningRule LearningRule

	// StructuralPlasticity controls connection growth and pruning.
	// If nil, topology is static. Call Remodel() to trigger
	// structural changes — it is NOT called automatically per tick.
	StructuralPlasticity StructuralPlasticity

	// incomingIndex maps each neuron index to its incoming connections.
	// Built lazily by buildIncomingIndex(). Used by learning rules
	// that need to evaluate post-synaptic timing across all inputs.
	incomingIndex [][]IncomingConnection

	// pending holds stimulations to process during the current Tick().
	pending []PendingStimulation

	// nextPending collects stimulations generated during the current
	// Tick(), to be processed on the next Tick().
	nextPending []PendingStimulation
}

// NewNetwork creates a network with the given number of neurons.
// All neurons are initialized to the same baseline, threshold, and
// zero activation.
func NewNetwork(size uint32, baseline, threshold int32, decayRate uint16, refractoryPeriod uint32) *Network {
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
		Counter:          1, // Start at 1 so LastFired=0 always means "never fired"
		DefaultDecayRate: decayRate,
		RefractoryPeriod: refractoryPeriod,
	}
}

// Connect adds a directed connection from neuron at index `from` to
// neuron at index `to` with the given weight.
func (net *Network) Connect(from, to uint32, weight int32) {
	net.Neurons[from].Connections = append(net.Neurons[from].Connections, Connection{
		Target: to,
		Weight: weight,
	})
	// Invalidate incoming index so it's rebuilt on next use
	net.incomingIndex = nil
}

// buildIncomingIndex creates a reverse mapping from each neuron to
// its incoming connections. Called lazily when learning rules need it.
func (net *Network) buildIncomingIndex() {
	net.incomingIndex = make([][]IncomingConnection, len(net.Neurons))
	for i := range net.Neurons {
		for j := range net.Neurons[i].Connections {
			conn := &net.Neurons[i].Connections[j]
			target := conn.Target
			if target < uint32(len(net.Neurons)) {
				net.incomingIndex[target] = append(net.incomingIndex[target], IncomingConnection{
					SourceIndex: uint32(i),
					Conn:        conn,
				})
			}
		}
	}
}

// getIncomingConnections returns all incoming connections to a neuron,
// with SourceIndex set to the source neuron's LastFired tick + 1 (for
// STDP timing evaluation). The +1 offset allows distinguishing
// "fired at tick 0" from "never fired" (both would be 0 otherwise).
// The learning rule must subtract 1 to get the actual tick.
func (net *Network) getIncomingConnections(neuronIdx uint32) []IncomingConnection {
	if net.incomingIndex == nil {
		net.buildIncomingIndex()
	}
	incoming := net.incomingIndex[neuronIdx]

	result := make([]IncomingConnection, len(incoming))
	for i, in := range incoming {
		sourceNeuron := &net.Neurons[in.SourceIndex]
		var encoded uint32
		if sourceNeuron.LastFired > 0 {
			// Neuron has fired at some point. Encode as LastFired + 1.
			encoded = sourceNeuron.LastFired + 1
		}
		// else: encoded stays 0 = never fired

		result[i] = IncomingConnection{
			SourceIndex: encoded,
			Conn:        in.Conn,
		}
	}
	return result
}

// Reward delivers a global reward or punishment signal to the network.
// The learning rule uses this to consolidate eligibility traces into
// actual weight changes. Positive = reward, negative = punishment.
func (net *Network) Reward(signal int32) {
	if net.LearningRule != nil {
		net.LearningRule.OnReward(net, signal, net.Counter)
	}
}

// Stimulate sends an external signal to a specific neuron. If the
// neuron fires, its downstream targets are queued for the next tick.
// This is the entry point for injecting input into the network.
func (net *Network) Stimulate(index uint32, weight int32) {
	if index >= uint32(len(net.Neurons)) {
		return
	}

	neuron := &net.Neurons[index]
	fired := neuron.Stimulate(weight, net.Counter, net.RefractoryPeriod)

	if fired {
		net.fireIdx(index)
	}
}

// fireIdx handles a neuron that has exceeded its threshold: sets the
// refractory period, resets activation, records the fire time, calls
// learning rule hooks, and queues downstream stimulations for the
// next tick.
func (net *Network) fireIdx(idx uint32) {
	neuron := &net.Neurons[idx]
	neuron.LastFired = net.Counter

	if net.UsePostFireReset {
		neuron.Activation = net.PostFireReset
	} else {
		neuron.Activation = neuron.Baseline
	}

	// Learning: notify post-synaptic firing for incoming connections
	if net.LearningRule != nil {
		incoming := net.getIncomingConnections(idx)
		net.LearningRule.OnPostFire(incoming, net.Counter)
	}

	for i := range neuron.Connections {
		conn := &neuron.Connections[i]

		// Learning: notify pre-synaptic spike propagation
		if net.LearningRule != nil && conn.Target < uint32(len(net.Neurons)) {
			postNeuron := &net.Neurons[conn.Target]
			net.LearningRule.OnSpikePropagation(conn, net.Counter, postNeuron.LastFired)
		}

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
// Stimulations are accumulated per neuron before evaluating fire
// thresholds, modeling biological spatial summation at the soma.
// This ensures the result is independent of stimulation order
// within a tick.
//
// Returns the number of neurons that fired during this tick.
func (net *Network) Tick() int {
	net.Counter++

	// Swap: pending becomes current, nextPending becomes the new
	// accumulator. Reuse backing arrays to reduce allocations.
	net.pending, net.nextPending = net.nextPending, net.pending[:0]

	// Phase 1: Accumulate all stimulations per target neuron.
	// We reuse a map to sum weights. For small networks this is fine;
	// for large networks we could use a dense array keyed by index.
	accumulated := make(map[uint32]int64)
	for _, stim := range net.pending {
		if stim.Target < uint32(len(net.Neurons)) {
			accumulated[stim.Target] += int64(stim.Weight)
		}
	}

	// Phase 2: Apply accumulated stimulation and evaluate firing.
	fired := 0
	for target, totalWeight := range accumulated {
		// Clamp accumulated weight to int32 range
		w := totalWeight
		if w > int64(MaxActivation) {
			w = int64(MaxActivation)
		}
		if w < int64(MinActivation) {
			w = int64(MinActivation)
		}

		neuron := &net.Neurons[target]
		if neuron.Stimulate(int32(w), net.Counter, net.RefractoryPeriod) {
			net.fireIdx(target)
			fired++
		}
	}

	// Learning: maintain eligibility traces (decay, cleanup)
	if net.LearningRule != nil {
		net.LearningRule.Maintain(net, net.Counter)
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

// Remodel triggers structural plasticity: pruning weak connections,
// growing new ones, and adjusting neuron excitability. Returns the
// number of connections pruned and grown. No-op if
// StructuralPlasticity is nil.
//
// Call this periodically — typically once per training sample, not
// every tick. Structural changes are expensive relative to normal
// tick processing.
func (net *Network) Remodel() (pruned, grown int) {
	if net.StructuralPlasticity == nil {
		return 0, 0
	}
	return net.StructuralPlasticity.Remodel(net, net.Counter)
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
func (net *Network) ActiveNeurons(above int32) []uint32 {
	var active []uint32
	for i, n := range net.Neurons {
		if n.Activation > above {
			active = append(active, uint32(i))
		}
	}
	return active
}
