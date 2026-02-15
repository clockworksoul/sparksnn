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

// CelegansNetwork builds a biomimetic Network from C. elegans
// connectome data. Returns the network and a map of neuron names
// to their indices in the network.
//
// Parameters:
//   - records: parsed connectome data from LoadCelegansCSV
//   - weightScale: multiplier to convert synapse count to int16 weight
//     (e.g., 100 means 3 synapses = weight 300)
//   - baseline, threshold: neuron parameters
//   - decayRate: default decay rate for all neurons
//   - refractoryPeriod: ticks after firing before neuron can fire again
//
// Gap junctions are modeled as bidirectional excitatory connections.
// Chemical synapses ("Send") are unidirectional. GABA neurotransmitter
// connections are inhibitory (negative weight); all others are excitatory.
func CelegansNetwork(
	records []CelegansRecord,
	weightScale int,
	baseline, threshold int16,
	decayRate uint16,
	refractoryPeriod uint32,
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
	// Sort for deterministic ordering
	sortStrings(names)
	for i, name := range names {
		nameToIndex[name] = uint32(i)
	}

	// Create network
	net := NewNetwork(uint32(len(names)), baseline, threshold, decayRate, refractoryPeriod)

	// Second pass: add connections
	for _, r := range records {
		fromIdx := nameToIndex[r.Origin]
		toIdx := nameToIndex[r.Target]

		// Calculate weight: synapse count * scale factor
		w := int32(r.NumConnections) * int32(weightScale)

		// GABA is inhibitory; everything else is excitatory
		if r.Neurotransmitter == "GABA" {
			w = -w
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
