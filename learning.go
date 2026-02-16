package biomimetic

// LearningRule defines the interface for synaptic plasticity mechanisms.
// Implementations are called by the Network at key moments during
// signal propagation. This allows different learning algorithms to
// be swapped in without changing the core engine.
//
// Implementations live in subpackages under learning/:
//   - learning/stdp:       Reward-modulated STDP (three-factor)
//   - learning/predictive: Predictive learning (self-supervised)
//
// All methods receive the current network counter tick for
// time-dependent calculations.
type LearningRule interface {
	// OnSpikePropagation is called when a pre-synaptic neuron fires
	// and sends a signal through a connection. It receives the
	// connection, the tick when the pre-synaptic neuron fired, and
	// the tick when the post-synaptic neuron last fired (0 if never).
	//
	// For STDP: this is where pre-before-post timing is evaluated.
	OnSpikePropagation(conn *Connection, preFiredAt, postLastFired uint32)

	// OnPostFire is called when a post-synaptic neuron fires.
	// It receives the neuron's incoming connections so the rule
	// can evaluate post-before-pre timing for recent inputs.
	OnPostFire(incoming []IncomingConnection, postFiredAt uint32)

	// OnReward is called when a global reward or punishment signal
	// is delivered to the network. The signal is a signed value:
	// positive = reward, negative = punishment.
	//
	// For R-STDP: this consolidates eligibility traces into actual
	// weight changes. For predictive learning: this is a no-op.
	OnReward(net *Network, reward int16, tick uint32)

	// Maintain is called once per tick for housekeeping: decaying
	// eligibility traces, pruning dead connections, etc.
	Maintain(net *Network, tick uint32)
}

// IncomingConnection represents an inbound connection to a neuron,
// referencing the source neuron and the specific connection struct.
// Used by OnPostFire to evaluate post-synaptic timing.
type IncomingConnection struct {
	SourceIndex uint32
	Conn        *Connection
}

// NoOpLearning is a learning rule that does nothing. Use it when
// you want a static network with no plasticity (e.g., the C. elegans
// connectome with fixed weights).
type NoOpLearning struct{}

func (NoOpLearning) OnSpikePropagation(*Connection, uint32, uint32) {}
func (NoOpLearning) OnPostFire([]IncomingConnection, uint32)        {}
func (NoOpLearning) OnReward(*Network, int16, uint32)               {}
func (NoOpLearning) Maintain(*Network, uint32)                      {}
