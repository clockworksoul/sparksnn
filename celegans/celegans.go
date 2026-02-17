// Package celegans provides tools for loading, building, and analyzing
// a biomimetic neural network based on the C. elegans connectome — the
// only organism whose complete nervous system has been fully mapped.
//
// Data source: OpenWorm project (CElegansNeuroML).
package celegans

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	biomimetic "github.com/clockworksoul/biomimetic-network"
)

// Record represents a single row from the C. elegans connectome
// dataset (OpenWorm/CElegansNeuroML).
type Record struct {
	Origin           string
	Target           string
	Type             string // "Send" (chemical synapse) or "GapJunction"
	NumConnections   int
	Neurotransmitter string
}

// LoadCSV reads the C. elegans connectome from a CSV file and returns
// the parsed records.
func LoadCSV(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	return ParseCSV(f)
}

// ParseCSV reads C. elegans connectome records from a reader.
func ParseCSV(r io.Reader) ([]Record, error) {
	reader := csv.NewReader(r)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var records []Record
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row: %w", err)
		}
		if len(row) < 4 {
			continue
		}

		numConn, err := strconv.ParseFloat(strings.TrimSpace(row[3]), 64)
		if err != nil {
			return nil, fmt.Errorf("parse connections %q: %w", row[3], err)
		}

		nt := ""
		if len(row) > 4 {
			nt = strings.TrimSpace(row[4])
		}

		records = append(records, Record{
			Origin:           strings.TrimSpace(row[0]),
			Target:           strings.TrimSpace(row[1]),
			Type:             strings.TrimSpace(row[2]),
			NumConnections:   int(numConn),
			Neurotransmitter: nt,
		})
	}

	return records, nil
}

// Params holds tunable parameters for building a C. elegans network.
// Separate from the constructor to keep the API clean as we add more
// knobs.
type Params struct {
	// WeightScale multiplies synapse count to get connection weight.
	// e.g., 100 means 3 synapses → weight 300.
	WeightScale int

	// InhibitoryScale is an additional multiplier applied to
	// inhibitory (GABA) connection weights. In biology, inhibitory
	// interneurons have outsized influence — a single GABAergic
	// neuron can suppress many excitatory ones. Values > 1 amplify
	// inhibition relative to excitation.
	//
	// Biology note: C. elegans has ~6% GABA connections but ~30%
	// inhibitory influence. A scale of 3-5x compensates.
	InhibitoryScale int

	// GapJunctionScale scales gap junction weights relative to
	// chemical synapses. Gap junctions are typically weaker per-
	// synapse (electrical coupling vs vesicle release) but faster.
	// Values < WeightScale attenuate them. 0 means use WeightScale.
	GapJunctionScale int

	Baseline         int32
	Threshold        int32
	DecayRate        uint16
	RefractoryPeriod uint32

	// PostFireReset is the activation level after firing. If 0,
	// defaults to Baseline. Negative values model hyperpolarization
	// (the neuron goes below resting potential after firing), which
	// is biologically accurate and critical for preventing runaway
	// excitation.
	PostFireReset int32

	// UsePostFireReset indicates whether PostFireReset should be
	// used (since 0 is a valid reset value).
	UsePostFireReset bool
}

// DefaultParams returns sensible defaults for C. elegans network
// construction. These are tuned to produce biologically plausible
// activation patterns — not seizure-like runaway excitation.
func DefaultParams() Params {
	return Params{
		WeightScale:      100,
		InhibitoryScale:  5,      // GABA neurons punch well above their weight
		GapJunctionScale: 40,     // Gap junctions weaker per-synapse
		Baseline:         0,
		Threshold:        500,
		DecayRate:        45000,  // ~69% retention — aggressive decay fights runaway
		RefractoryPeriod: 5,
		PostFireReset:    -150,   // Hyperpolarization after firing
		UsePostFireReset: true,
	}
}

// IsGABANeuron returns true if the named neuron is a known GABAergic
// (inhibitory) neuron in C. elegans.
//
// Known inhibitory neuron classes:
//   - DD1-6, VD1-13 (dorsal/ventral D-type, cross-inhibitory)
//   - RME neurons (head muscle inhibition)
//   - AVL, DVB (enteric muscles)
func IsGABANeuron(name string) bool {
	return gabaMotorNeurons[name]
}

var gabaMotorNeurons = map[string]bool{
	"DD1": true, "DD2": true, "DD3": true,
	"DD4": true, "DD5": true, "DD6": true,
	"VD1": true, "VD2": true, "VD3": true, "VD4": true,
	"VD5": true, "VD6": true, "VD7": true, "VD8": true,
	"VD9": true, "VD10": true, "VD11": true, "VD12": true,
	"VD13": true,
	"RMED": true, "RMEL": true, "RMER": true, "RMEV": true,
	"AVL": true, "DVB": true,
}

// BuildNetwork constructs a biomimetic Network from C. elegans
// connectome data. Returns the network and a map of neuron names
// to their indices in the network.
//
// Gap junctions are modeled as bidirectional connections with
// attenuated weight (GapJunctionScale). Chemical synapses ("Send")
// are unidirectional.
//
// Inhibitory classification uses both neurotransmitter type (GABA)
// and known inhibitory neuron identity (D-type motor neurons, RME
// neurons). This is more biologically accurate than neurotransmitter
// alone, since C. elegans has inhibitory neurons that aren't always
// annotated as GABA in the dataset.
func BuildNetwork(
	records []Record,
	params Params,
) (*biomimetic.Network, map[string]uint32) {

	// First pass: collect unique neuron names
	nameSet := make(map[string]bool)
	for _, r := range records {
		nameSet[r.Origin] = true
		nameSet[r.Target] = true
	}

	// Assign indices (sorted for determinism)
	nameToIndex := make(map[string]uint32, len(nameSet))
	names := make([]string, 0, len(nameSet))
	for name := range nameSet {
		names = append(names, name)
	}
	sortStrings(names)
	for i, name := range names {
		nameToIndex[name] = uint32(i)
	}

	// Create network
	net := biomimetic.NewNetwork(uint32(len(names)), params.Baseline, params.Threshold,
		params.DecayRate, params.RefractoryPeriod)

	// Set post-fire reset if configured
	if params.UsePostFireReset {
		net.PostFireReset = params.PostFireReset
		net.UsePostFireReset = true
	}

	gjScale := params.GapJunctionScale
	if gjScale == 0 {
		gjScale = params.WeightScale
	}

	inhibScale := params.InhibitoryScale
	if inhibScale == 0 {
		inhibScale = 1
	}

	// Second pass: add connections
	for _, r := range records {
		fromIdx := nameToIndex[r.Origin]
		toIdx := nameToIndex[r.Target]

		// Choose base scale based on connection type
		scale := int32(params.WeightScale)
		if r.Type == "GapJunction" {
			scale = int32(gjScale)
		}

		w := int32(r.NumConnections) * scale

		// Determine if inhibitory: explicit GABA neurotransmitter
		// OR the origin is a known GABAergic neuron
		inhibitory := r.Neurotransmitter == "GABA" || gabaMotorNeurons[r.Origin]
		if inhibitory {
			w = -w * int32(inhibScale)
		}

		// Clamp to int32 range
		if w > math.MaxInt32 {
			w = math.MaxInt32
		}
		if w < math.MinInt32 {
			w = math.MinInt32
		}

		net.Connect(fromIdx, toIdx, int32(w))

		// Gap junctions are bidirectional
		if r.Type == "GapJunction" {
			net.Connect(toIdx, fromIdx, int32(w))
		}
	}

	return net, nameToIndex
}

// sortStrings is a simple insertion sort for small slices.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
