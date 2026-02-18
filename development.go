package biomimetic

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Position represents a neuron's temporary 2D location during
// network development. Discarded after Finalize().
type Position struct {
	X, Y float32
}

// distance returns the Euclidean distance between two positions.
func (a Position) distance(b Position) float32 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return float32(math.Sqrt(float64(dx*dx + dy*dy)))
}

// LayerRole describes the functional role of a neuron layer.
type LayerRole int

const (
	RoleInput LayerRole = iota
	RoleHidden
	RoleOutput
	RoleInhibitory
)

// Layer describes a logical group of neurons with spatial placement.
type Layer struct {
	Name     string
	Role     LayerRole
	StartIdx uint32
	EndIdx   uint32 // exclusive
}

// Size returns the number of neurons in this layer.
func (l Layer) Size() uint32 {
	return l.EndIdx - l.StartIdx
}

// PlacementStrategy controls how neurons are arranged in 2D space.
type PlacementStrategy int

const (
	// PlaceRandom distributes neurons uniformly within a bounding box.
	PlaceRandom PlacementStrategy = iota

	// PlaceGrid arranges neurons on a regular grid.
	PlaceGrid
)

// LayerSpec defines a layer to be created during development.
type LayerSpec struct {
	Name      string
	Role      LayerRole
	Size      uint32
	Placement PlacementStrategy

	// Bounding box for neuron placement. Positions are assigned
	// within [OriginX, OriginX+Width] × [OriginY, OriginY+Height].
	OriginX, OriginY float32
	Width, Height    float32
}

// ConnectionRule specifies how two layers should be wired.
type ConnectionRule struct {
	FromLayer string
	ToLayer   string

	// Sigma controls the spatial spread of connections.
	// P(connect) = PBase * exp(-d² / (2 * sigma²)).
	// Larger sigma = more distant connections allowed.
	Sigma float32

	// PBase is the maximum connection probability at distance 0.
	// 1.0 = guaranteed connection at same position.
	PBase float32

	// InitWeightMin and InitWeightMax define the range for random
	// initial weights on created connections.
	InitWeightMin int32
	InitWeightMax int32
}

// DevParams holds global parameters for the development phase.
type DevParams struct {
	Baseline         int32
	Threshold        int32
	DecayRate        uint16
	RefractoryPeriod uint32
}

// Development manages the construction and developmental phase of
// a network. It owns temporary spatial data (neuron positions,
// layer definitions) and provides methods for wiring based on
// spatial proximity.
//
// After development is complete, call Finalize() to return the
// bare Network and discard all scaffolding.
type Development struct {
	Net       *Network
	Positions []Position
	Layers    []Layer
	layerMap  map[string]*Layer
	rng       *rand.Rand
}

// NewDevelopment creates a development harness with global network
// parameters.
func NewDevelopment(params DevParams) *Development {
	return &Development{
		layerMap: make(map[string]*Layer),
		rng:      rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
	}
}

// NewDevelopmentSeeded creates a development harness with a fixed
// seed for reproducible experiments.
func NewDevelopmentSeeded(params DevParams, seed uint64) *Development {
	return &Development{
		layerMap: make(map[string]*Layer),
		rng:      rand.New(rand.NewPCG(seed, seed^0xcafef00d)),
	}
}

// AddLayer creates a layer of neurons with the given spec.
// Layers must be added in order before Build() is called.
// Returns the layer for chaining.
func (d *Development) AddLayer(spec LayerSpec) *Layer {
	if _, exists := d.layerMap[spec.Name]; exists {
		panic(fmt.Sprintf("duplicate layer name: %q", spec.Name))
	}

	startIdx := uint32(len(d.Positions))
	endIdx := startIdx + spec.Size

	// Place neurons
	for i := uint32(0); i < spec.Size; i++ {
		var pos Position
		switch spec.Placement {
		case PlaceRandom:
			pos = Position{
				X: spec.OriginX + d.rng.Float32()*spec.Width,
				Y: spec.OriginY + d.rng.Float32()*spec.Height,
			}
		case PlaceGrid:
			cols := uint32(math.Ceil(math.Sqrt(float64(spec.Size))))
			row := i / cols
			col := i % cols
			pos = Position{
				X: spec.OriginX + float32(col)/float32(cols)*spec.Width,
				Y: spec.OriginY + float32(row)/float32(cols)*spec.Height,
			}
		}
		d.Positions = append(d.Positions, pos)
	}

	layer := Layer{
		Name:     spec.Name,
		Role:     spec.Role,
		StartIdx: startIdx,
		EndIdx:   endIdx,
	}
	d.Layers = append(d.Layers, layer)
	d.layerMap[spec.Name] = &d.Layers[len(d.Layers)-1]

	return &d.Layers[len(d.Layers)-1]
}

// Build creates the Network from all added layers. Must be called
// after all AddLayer calls and before Wire or any training.
func (d *Development) Build(params DevParams) {
	totalNeurons := uint32(len(d.Positions))
	d.Net = NewNetwork(totalNeurons, params.Baseline, params.Threshold,
		params.DecayRate, params.RefractoryPeriod)
}

// Wire creates connections between two layers using Peter's Rule:
// P(connect) = PBase * exp(-d² / (2σ²)).
//
// Returns the number of connections created.
func (d *Development) Wire(rule ConnectionRule) int {
	if d.Net == nil {
		panic("must call Build() before Wire()")
	}

	from, ok := d.layerMap[rule.FromLayer]
	if !ok {
		panic(fmt.Sprintf("unknown layer: %q", rule.FromLayer))
	}
	to, ok := d.layerMap[rule.ToLayer]
	if !ok {
		panic(fmt.Sprintf("unknown layer: %q", rule.ToLayer))
	}

	twoSigmaSq := float64(2 * rule.Sigma * rule.Sigma)
	created := 0

	for i := from.StartIdx; i < from.EndIdx; i++ {
		for j := to.StartIdx; j < to.EndIdx; j++ {
			if i == j {
				continue // no self-connections
			}

			dist := d.Positions[i].distance(d.Positions[j])
			prob := float64(rule.PBase) * math.Exp(-float64(dist*dist)/twoSigmaSq)

			if d.rng.Float64() < prob {
				weightRange := rule.InitWeightMax - rule.InitWeightMin
				w := rule.InitWeightMin
				if weightRange > 0 {
					w += int32(d.rng.IntN(int(weightRange + 1)))
				}
				d.Net.Connect(i, j, w)
				created++
			}
		}
	}

	return created
}

// Layer returns the named layer, or nil if not found.
func (d *Development) GetLayer(name string) *Layer {
	return d.layerMap[name]
}

// ConnectionCount returns the total number of connections in the
// network.
func (d *Development) ConnectionCount() int {
	count := 0
	for i := range d.Net.Neurons {
		count += len(d.Net.Neurons[i].Connections)
	}
	return count
}

// ConnectionDensity returns the fraction of possible connections
// that actually exist between two layers.
func (d *Development) ConnectionDensity(fromLayer, toLayer string) float64 {
	from := d.layerMap[fromLayer]
	to := d.layerMap[toLayer]
	if from == nil || to == nil {
		return 0
	}

	possible := int(from.Size()) * int(to.Size())
	if from.Name == to.Name {
		possible = int(from.Size()) * (int(from.Size()) - 1) // no self-connections
	}
	if possible == 0 {
		return 0
	}

	actual := 0
	for i := from.StartIdx; i < from.EndIdx; i++ {
		for _, conn := range d.Net.Neurons[i].Connections {
			if conn.Target >= to.StartIdx && conn.Target < to.EndIdx {
				actual++
			}
		}
	}

	return float64(actual) / float64(possible)
}

// Finalize strips all developmental scaffolding and returns the
// bare Network. The Development struct should not be used after
// this call.
func (d *Development) Finalize() *Network {
	net := d.Net
	d.Net = nil
	d.Positions = nil
	d.Layers = nil
	d.layerMap = nil
	return net
}
