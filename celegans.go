package biomimetic

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// CelegansRecord represents a single row from the C. elegans
// connectome dataset (OpenWorm/CElegansNeuroML).
type CelegansRecord struct {
	Origin         string
	Target         string
	Type           string // "Send" (chemical synapse) or "GapJunction"
	NumConnections int
	Neurotransmitter string
}

// LoadCelegansCSV reads the C. elegans connectome from a CSV file
// and returns the parsed records.
func LoadCelegansCSV(path string) ([]CelegansRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	return ParseCelegansCSV(f)
}

// ParseCelegansCSV reads C. elegans connectome records from a reader.
func ParseCelegansCSV(r io.Reader) ([]CelegansRecord, error) {
	reader := csv.NewReader(r)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var records []CelegansRecord
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

		records = append(records, CelegansRecord{
			Origin:           strings.TrimSpace(row[0]),
			Target:           strings.TrimSpace(row[1]),
			Type:             strings.TrimSpace(row[2]),
			NumConnections:   int(numConn),
			Neurotransmitter: nt,
		})
	}

	return records, nil
}

// CelegansParams holds tunable parameters for building a C. elegans
// network. Separate from the constructor to keep the API clean as
// we add more knobs.
type CelegansParams struct {
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

	Baseline         int16
	Threshold        int16
	DecayRate        uint16
	RefractoryPeriod uint32

	// PostFireReset is the activation level after firing. If 0,
	// defaults to Baseline. Negative values model hyperpolarization
	// (the neuron goes below resting potential after firing), which
	// is biologically accurate and critical for preventing runaway
	// excitation.
	PostFireReset int16

	// UsePostFireReset indicates whether PostFireReset should be
	// used (since 0 is a valid reset value).
	UsePostFireReset bool
}

// DefaultCelegansParams returns sensible defaults for C. elegans
// network construction. These are tuned to produce biologically
// plausible activation patterns — not seizure-like runaway excitation.
func DefaultCelegansParams() CelegansParams {
	return CelegansParams{
		WeightScale:      100,
		InhibitoryScale:  5,     // GABA neurons punch well above their weight
		GapJunctionScale: 40,    // Gap junctions weaker per-synapse
		Baseline:         0,
		Threshold:        350,   // Low enough for 4-synapse connections to trigger
		DecayRate:        40000,  // ~61% retention — strong decay fights runaway
		RefractoryPeriod: 6,     // Longer refractory helps damp oscillation
		PostFireReset:    -200,   // Strong hyperpolarization after firing
		UsePostFireReset: true,
	}
}

// isInhibitory returns true if the neurotransmitter is known to be
// inhibitory in C. elegans. This includes GABA (classic inhibitory)
// and some glutamate connections that act on GluCl channels — a
// well-known nematode-specific quirk where glutamate can be
// inhibitory via glutamate-gated chloride channels.
//
// Known inhibitory neuron classes in C. elegans (GABA motor neurons):
// DD1-6, VD1-13 (dorsal/ventral D-type, cross-inhibitory)
// RME neurons (head muscle inhibition)
// AVL, DVB (enteric muscles)
// IsGABANeuron returns true if the named neuron is a known GABAergic
// (inhibitory) neuron in C. elegans.
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

// CelegansNetwork builds a biomimetic Network from C. elegans
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
func CelegansNetwork(
	records []CelegansRecord,
	params CelegansParams,
) (*Network, map[string]uint32) {

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
	net := NewNetwork(uint32(len(names)), params.Baseline, params.Threshold,
		params.DecayRate, params.RefractoryPeriod)

	// Set post-fire reset if configured
	if params.UsePostFireReset {
		for i := range net.Neurons {
			// Store post-fire reset in baseline for now.
			// TODO: Add dedicated PostFireReset field to Neuron.
			_ = i // placeholder — we'll handle this in fire()
		}
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

		// Clamp to int16 range
		if w > math.MaxInt16 {
			w = math.MaxInt16
		}
		if w < math.MinInt16 {
			w = math.MinInt16
		}

		net.Connect(fromIdx, toIdx, int16(w))

		// Gap junctions are bidirectional
		if r.Type == "GapJunction" {
			net.Connect(toIdx, fromIdx, int16(w))
		}
	}

	return net, nameToIndex
}

// sortStrings is a simple insertion sort for small slices.
// Avoids importing "sort" for this one use.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
