package sparksnn

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// networkJSON is the JSON-serializable representation of a Network.
// Exported fields only; pending stimulation queues are transient
// runtime state and are not persisted.
type networkJSON struct {
	Neurons          []Neuron `json:"neurons"`
	Counter          uint32   `json:"counter"`
	DefaultDecayRate uint16   `json:"defaultDecayRate"`
	RefractoryPeriod uint32   `json:"refractoryPeriod"`
}

// Save writes the network to the given writer as JSON.
// Pending stimulation queues are not saved — they are transient
// runtime state. A loaded network starts with empty queues.
func (net *Network) Save(w io.Writer) error {
	data := networkJSON{
		Neurons:          net.Neurons,
		Counter:          net.Counter,
		DefaultDecayRate: net.DefaultDecayRate,
		RefractoryPeriod: net.RefractoryPeriod,
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(data)
}

// SaveFile writes the network to a file as JSON.
func (net *Network) SaveFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	if err := net.Save(f); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// Load reads a network from the given reader (JSON format).
// Pending stimulation queues are initialized empty.
func Load(r io.Reader) (*Network, error) {
	var data networkJSON
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return nil, fmt.Errorf("decode network: %w", err)
	}

	return &Network{
		Neurons:          data.Neurons,
		Counter:          data.Counter,
		DefaultDecayRate: data.DefaultDecayRate,
		RefractoryPeriod: data.RefractoryPeriod,
	}, nil
}

// LoadFile reads a network from a JSON file.
func LoadFile(path string) (*Network, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	return Load(f)
}
