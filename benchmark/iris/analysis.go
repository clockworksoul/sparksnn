package iris

import (
	"fmt"
	"io"
	"math"
	"sort"

	bio "github.com/clockworksoul/biomimetic-network"
)

// NetworkAnalysis captures the full structural and functional state
// of a trained Iris network. Use Analyze() to populate.
type NetworkAnalysis struct {
	// Structural
	TotalConnections   int
	InputHiddenConns   int
	HiddenOutputConns  int
	HiddenHiddenConns  int // lateral (shouldn't exist in standard topology)
	BackwardConns      int // output→hidden or hidden→input
	WeightDistribution WeightDist

	// Per-layer weight stats
	InputHiddenWeights  WeightDist
	HiddenOutputWeights WeightDist

	// Per-neuron analysis
	Neurons []NeuronProfile

	// Per-class firing patterns (hidden neuron spike counts)
	// ClassProfiles[class][hiddenIdx] = avg spike count
	ClassProfiles [3][]float64

	// Output neuron spike counts per class
	// OutputProfiles[class][outputIdx] = avg spike count
	OutputProfiles [3][]float64

	// Strongest paths: input→hidden→output chains
	StrongestPaths []Path

	// Classification results
	ConfusionMatrix [3][3]int // [actual][predicted]
	Errors          []ClassificationError
}

// WeightDist summarizes a weight distribution.
type WeightDist struct {
	Count    int
	Min      int32
	Max      int32
	Mean     float64
	StdDev   float64
	Median   int32
	Positive int // count > 0
	Negative int // count < 0
	Zero     int // count == 0
}

// NeuronProfile captures per-neuron structural and functional info.
type NeuronProfile struct {
	Index         uint32
	Layer         string // "input", "hidden", "output"
	InDegree      int    // incoming connections
	OutDegree     int    // outgoing connections
	TotalSpikes   int    // total spikes across all class presentations
	SpikesPerClass [3]int
	IsDead        bool
	// For hidden neurons: which class they fire most for
	PreferredClass int // -1 if dead or no preference
	Selectivity    float64 // how much they prefer one class (0=uniform, 1=exclusive)
}

// ClassificationError records a single misclassified sample.
type ClassificationError struct {
	SampleIdx    int
	TrueLabel    int
	PredLabel    int
	Features     []byte   // raw input features [0-255]
	SpikeCounts  []int    // output neuron spike counts
	Confidence   float64  // margin between top two outputs
}

// Path represents an input→hidden→output signal chain.
type Path struct {
	InputIdx      uint32
	HiddenIdx     uint32
	OutputIdx     uint32
	InputHiddenW  int32
	HiddenOutputW int32
	CombinedScore float64 // product of absolute weights
}

// Analyze runs a full structural and functional analysis of a
// trained Iris network. Presents all training samples to observe
// firing patterns.
func Analyze(net *bio.Network, layout Layout, task *Task, cfg NetworkConfig) *NetworkAnalysis {
	a := &NetworkAnalysis{}

	// --- Structural analysis ---
	incomingCount := make(map[uint32]int)

	for i := uint32(0); i < uint32(len(net.Neurons)); i++ {
		for _, conn := range net.Neurons[i].Connections {
			a.TotalConnections++
			incomingCount[conn.Target]++

			srcLayer := neuronLayer(i, layout)
			tgtLayer := neuronLayer(conn.Target, layout)

			switch {
			case srcLayer == "input" && tgtLayer == "hidden":
				a.InputHiddenConns++
			case srcLayer == "hidden" && tgtLayer == "output":
				a.HiddenOutputConns++
			case srcLayer == "hidden" && tgtLayer == "hidden":
				a.HiddenHiddenConns++
			case srcLayer == "output" || (srcLayer == "hidden" && tgtLayer == "input"):
				a.BackwardConns++
			}
		}
	}

	// Weight distributions
	a.WeightDistribution = collectWeightDist(net, layout, "all")
	a.InputHiddenWeights = collectWeightDist(net, layout, "input-hidden")
	a.HiddenOutputWeights = collectWeightDist(net, layout, "hidden-output")

	// --- Functional analysis: present samples and record spikes ---
	numHidden := int(layout.HiddenEnd - layout.HiddenStart)
	numOutput := int(layout.OutputEnd - layout.OutputStart)

	for c := 0; c < 3; c++ {
		a.ClassProfiles[c] = make([]float64, numHidden)
		a.OutputProfiles[c] = make([]float64, numOutput)
	}

	classCounts := [3]int{}
	trainSamples := task.TrainingSamples()

	// Per-neuron profiles
	totalNeurons := len(net.Neurons)
	a.Neurons = make([]NeuronProfile, totalNeurons)
	for i := range a.Neurons {
		a.Neurons[i] = NeuronProfile{
			Index:          uint32(i),
			Layer:          neuronLayer(uint32(i), layout),
			OutDegree:      len(net.Neurons[i].Connections),
			InDegree:       incomingCount[uint32(i)],
			PreferredClass: -1,
		}
	}

	for _, sample := range trainSamples {
		cls := sample.Label
		classCounts[cls]++

		// Record LastFired before presentation
		preFired := make([]uint32, totalNeurons)
		for i := range net.Neurons {
			preFired[i] = net.Neurons[i].LastFired
		}

		spikeCounts := PresentSample(net, layout, sample, cfg)

		// Record output spikes
		for o := 0; o < numOutput; o++ {
			a.OutputProfiles[cls][o] += float64(spikeCounts[o])
		}

		// Check which hidden neurons fired
		for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
			hi := int(h - layout.HiddenStart)
			if net.Neurons[h].LastFired > preFired[h] {
				a.ClassProfiles[cls][hi]++
				a.Neurons[h].TotalSpikes++
				a.Neurons[h].SpikesPerClass[cls]++
			}
		}

		// Check input neurons too
		for inp := layout.InputStart; inp < layout.InputEnd; inp++ {
			if net.Neurons[inp].LastFired > preFired[inp] {
				a.Neurons[inp].TotalSpikes++
				a.Neurons[inp].SpikesPerClass[cls]++
			}
		}
	}

	// Normalize class profiles to averages
	for c := 0; c < 3; c++ {
		if classCounts[c] > 0 {
			for i := range a.ClassProfiles[c] {
				a.ClassProfiles[c][i] /= float64(classCounts[c])
			}
			for i := range a.OutputProfiles[c] {
				a.OutputProfiles[c][i] /= float64(classCounts[c])
			}
		}
	}

	// Compute selectivity for hidden neurons
	for h := layout.HiddenStart; h < layout.HiddenEnd; h++ {
		np := &a.Neurons[h]
		total := np.TotalSpikes
		if total == 0 {
			np.IsDead = true
			continue
		}

		maxClass := 0
		maxSpikes := 0
		for c := 0; c < 3; c++ {
			if np.SpikesPerClass[c] > maxSpikes {
				maxSpikes = np.SpikesPerClass[c]
				maxClass = c
			}
		}
		np.PreferredClass = maxClass

		// Selectivity: 1 - (entropy / max_entropy)
		// High = neuron is class-selective; Low = fires uniformly
		maxEntropy := math.Log(3.0)
		entropy := 0.0
		for c := 0; c < 3; c++ {
			p := float64(np.SpikesPerClass[c]) / float64(total)
			if p > 0 {
				entropy -= p * math.Log(p)
			}
		}
		np.Selectivity = 1.0 - entropy/maxEntropy
	}

	// --- Classification: confusion matrix + error analysis ---
	// Run on test set with multiple trials per sample for stability
	testSamples := task.TestSamples()

	for sIdx, sample := range testSamples {
		aggSpikes := make([]int, numOutput)
		for trial := 0; trial < EvalTrials; trial++ {
			spikes := PresentSample(net, layout, sample, cfg)
			for o := range spikes {
				aggSpikes[o] += spikes[o]
			}
		}

		predicted := Classify(aggSpikes)
		if predicted < 0 {
			predicted = 0 // no spikes — default to class 0
		}
		a.ConfusionMatrix[sample.Label][predicted]++

		if predicted != sample.Label {
			// Compute confidence: margin between top two
			sorted := make([]int, len(aggSpikes))
			copy(sorted, aggSpikes)
			sort.Sort(sort.Reverse(sort.IntSlice(sorted)))
			confidence := 0.0
			if sorted[0] > 0 {
				confidence = float64(sorted[0]-sorted[1]) / float64(sorted[0])
			}

			a.Errors = append(a.Errors, ClassificationError{
				SampleIdx:   sIdx,
				TrueLabel:   sample.Label,
				PredLabel:   predicted,
				Features:    sample.Inputs,
				SpikeCounts: aggSpikes,
				Confidence:  confidence,
			})
		}
	}

	// --- Strongest paths ---
	a.StrongestPaths = findStrongestPaths(net, layout, 10)

	return a
}

// PrintReport writes a human-readable analysis to w.
func (a *NetworkAnalysis) PrintReport(w io.Writer) {
	classNames := [3]string{"setosa", "versicolor", "virginica"}

	fmt.Fprintf(w, "=== NETWORK STRUCTURE ===\n")
	fmt.Fprintf(w, "Total connections:   %d\n", a.TotalConnections)
	fmt.Fprintf(w, "  Input → Hidden:    %d\n", a.InputHiddenConns)
	fmt.Fprintf(w, "  Hidden → Output:   %d\n", a.HiddenOutputConns)
	fmt.Fprintf(w, "  Hidden → Hidden:   %d\n", a.HiddenHiddenConns)
	fmt.Fprintf(w, "  Backward:          %d\n", a.BackwardConns)

	printWeightDist := func(name string, wd WeightDist) {
		fmt.Fprintf(w, "\n--- %s Weights (n=%d) ---\n", name, wd.Count)
		fmt.Fprintf(w, "  Range:    [%d, %d]\n", wd.Min, wd.Max)
		fmt.Fprintf(w, "  Mean:     %.1f\n", wd.Mean)
		fmt.Fprintf(w, "  StdDev:   %.1f\n", wd.StdDev)
		fmt.Fprintf(w, "  Median:   %d\n", wd.Median)
		fmt.Fprintf(w, "  Positive: %d  Negative: %d  Zero: %d\n", wd.Positive, wd.Negative, wd.Zero)
	}

	printWeightDist("All", a.WeightDistribution)
	printWeightDist("Input→Hidden", a.InputHiddenWeights)
	printWeightDist("Hidden→Output", a.HiddenOutputWeights)

	fmt.Fprintf(w, "\n=== HIDDEN NEURON PROFILES ===\n")
	fmt.Fprintf(w, "%-6s %-6s %-6s %-8s %-12s %-12s %-12s %-10s %-6s\n",
		"Idx", "InDeg", "OutDeg", "Spikes", "Setosa", "Versicolor", "Virginica", "Preferred", "Select")

	for _, np := range a.Neurons {
		if np.Layer != "hidden" {
			continue
		}
		prefName := "DEAD"
		if !np.IsDead {
			prefName = classNames[np.PreferredClass]
		}
		fmt.Fprintf(w, "%-6d %-6d %-6d %-8d %-12d %-12d %-12d %-10s %.3f\n",
			np.Index, np.InDegree, np.OutDegree, np.TotalSpikes,
			np.SpikesPerClass[0], np.SpikesPerClass[1], np.SpikesPerClass[2],
			prefName, np.Selectivity)
	}

	fmt.Fprintf(w, "\n=== OUTPUT NEURON AVG SPIKES BY CLASS ===\n")
	fmt.Fprintf(w, "%-12s %-10s %-10s %-10s\n", "Class", "Output 0", "Output 1", "Output 2")
	for c := 0; c < 3; c++ {
		fmt.Fprintf(w, "%-12s", classNames[c])
		for o := range a.OutputProfiles[c] {
			fmt.Fprintf(w, " %-10.2f", a.OutputProfiles[c][o])
		}
		fmt.Fprintf(w, "\n")
	}

	fmt.Fprintf(w, "\n=== HIDDEN NEURON AVG FIRING RATE BY CLASS ===\n")
	fmt.Fprintf(w, "%-6s %-10s %-10s %-10s\n", "Idx", "Setosa", "Versicolor", "Virginica")
	// Find the hidden start index from neuron profiles
	var hiddenStart uint32
	for _, np := range a.Neurons {
		if np.Layer == "hidden" {
			hiddenStart = np.Index
			break
		}
	}
	for _, np := range a.Neurons {
		if np.Layer != "hidden" {
			continue
		}
		hi := int(np.Index - hiddenStart)
		fmt.Fprintf(w, "%-6d", np.Index)
		for c := 0; c < 3; c++ {
			fmt.Fprintf(w, " %-10.3f", a.ClassProfiles[c][hi])
		}
		fmt.Fprintf(w, "\n")
	}

	fmt.Fprintf(w, "\n=== CONFUSION MATRIX (rows=actual, cols=predicted) ===\n")
	fmt.Fprintf(w, "%-12s %-10s %-10s %-10s %-8s\n", "", "Setosa", "Versicolor", "Virginica", "Total")
	for actual := 0; actual < 3; actual++ {
		rowTotal := 0
		for p := 0; p < 3; p++ {
			rowTotal += a.ConfusionMatrix[actual][p]
		}
		fmt.Fprintf(w, "%-12s %-10d %-10d %-10d %-8d\n",
			classNames[actual],
			a.ConfusionMatrix[actual][0],
			a.ConfusionMatrix[actual][1],
			a.ConfusionMatrix[actual][2],
			rowTotal)
	}

	// Per-class precision/recall
	fmt.Fprintf(w, "\n%-12s %-10s %-10s %-10s\n", "Class", "Precision", "Recall", "F1")
	for c := 0; c < 3; c++ {
		tp := a.ConfusionMatrix[c][c]
		// Precision: tp / sum of column c
		colSum := 0
		for r := 0; r < 3; r++ {
			colSum += a.ConfusionMatrix[r][c]
		}
		// Recall: tp / sum of row c
		rowSum := 0
		for p := 0; p < 3; p++ {
			rowSum += a.ConfusionMatrix[c][p]
		}
		precision := 0.0
		if colSum > 0 {
			precision = float64(tp) / float64(colSum)
		}
		recall := 0.0
		if rowSum > 0 {
			recall = float64(tp) / float64(rowSum)
		}
		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}
		fmt.Fprintf(w, "%-12s %-10.3f %-10.3f %-10.3f\n", classNames[c], precision, recall, f1)
	}

	if len(a.Errors) > 0 {
		featureNames := [4]string{"sepal_len", "sepal_wid", "petal_len", "petal_wid"}
		fmt.Fprintf(w, "\n=== MISCLASSIFIED SAMPLES (%d errors) ===\n", len(a.Errors))
		for _, e := range a.Errors {
			fmt.Fprintf(w, "  Sample %d: true=%s predicted=%s spikes=%v confidence=%.2f\n",
				e.SampleIdx, classNames[e.TrueLabel], classNames[e.PredLabel],
				e.SpikeCounts, e.Confidence)
			fmt.Fprintf(w, "    features: ")
			for f := 0; f < len(e.Features) && f < 4; f++ {
				fmt.Fprintf(w, "%s=%d ", featureNames[f], e.Features[f])
			}
			fmt.Fprintf(w, "\n")
		}
	} else {
		fmt.Fprintf(w, "\n=== NO MISCLASSIFIED SAMPLES ===\n")
	}

	fmt.Fprintf(w, "\n=== TOP %d STRONGEST PATHS (Input→Hidden→Output) ===\n", len(a.StrongestPaths))
	fmt.Fprintf(w, "%-8s %-8s %-8s %-10s %-10s %-12s\n",
		"Input", "Hidden", "Output", "W(i→h)", "W(h→o)", "Score")
	for _, p := range a.StrongestPaths {
		fmt.Fprintf(w, "%-8d %-8d %-8d %-10d %-10d %-12.0f\n",
			p.InputIdx, p.HiddenIdx, p.OutputIdx,
			p.InputHiddenW, p.HiddenOutputW, p.CombinedScore)
	}
}

// --- helpers ---

func neuronLayer(idx uint32, layout Layout) string {
	switch {
	case idx >= layout.InputStart && idx < layout.InputEnd:
		return "input"
	case idx >= layout.HiddenStart && idx < layout.HiddenEnd:
		return "hidden"
	case idx >= layout.OutputStart && idx < layout.OutputEnd:
		return "output"
	default:
		return "unknown"
	}
}

func collectWeightDist(net *bio.Network, layout Layout, which string) WeightDist {
	var weights []int32

	for i := uint32(0); i < uint32(len(net.Neurons)); i++ {
		srcLayer := neuronLayer(i, layout)
		for _, conn := range net.Neurons[i].Connections {
			tgtLayer := neuronLayer(conn.Target, layout)

			include := false
			switch which {
			case "all":
				include = true
			case "input-hidden":
				include = srcLayer == "input" && tgtLayer == "hidden"
			case "hidden-output":
				include = srcLayer == "hidden" && tgtLayer == "output"
			}
			if include {
				weights = append(weights, conn.Weight)
			}
		}
	}

	if len(weights) == 0 {
		return WeightDist{}
	}

	sort.Slice(weights, func(i, j int) bool { return weights[i] < weights[j] })

	wd := WeightDist{
		Count:  len(weights),
		Min:    weights[0],
		Max:    weights[len(weights)-1],
		Median: weights[len(weights)/2],
	}

	var sum float64
	for _, w := range weights {
		sum += float64(w)
		if w > 0 {
			wd.Positive++
		} else if w < 0 {
			wd.Negative++
		} else {
			wd.Zero++
		}
	}
	wd.Mean = sum / float64(len(weights))

	var variance float64
	for _, w := range weights {
		d := float64(w) - wd.Mean
		variance += d * d
	}
	wd.StdDev = math.Sqrt(variance / float64(len(weights)))

	return wd
}

func findStrongestPaths(net *bio.Network, layout Layout, topN int) []Path {
	var paths []Path

	for i := layout.InputStart; i < layout.InputEnd; i++ {
		for _, ihConn := range net.Neurons[i].Connections {
			if ihConn.Target < layout.HiddenStart || ihConn.Target >= layout.HiddenEnd {
				continue
			}
			h := ihConn.Target

			for _, hoConn := range net.Neurons[h].Connections {
				if hoConn.Target < layout.OutputStart || hoConn.Target >= layout.OutputEnd {
					continue
				}

				// Score: product of weights (both positive = excitatory path)
				score := float64(ihConn.Weight) * float64(hoConn.Weight)

				paths = append(paths, Path{
					InputIdx:      i,
					HiddenIdx:     h,
					OutputIdx:     hoConn.Target,
					InputHiddenW:  ihConn.Weight,
					HiddenOutputW: hoConn.Weight,
					CombinedScore: score,
				})
			}
		}
	}

	// Sort by absolute score descending
	sort.Slice(paths, func(i, j int) bool {
		return math.Abs(paths[i].CombinedScore) > math.Abs(paths[j].CombinedScore)
	})

	if len(paths) > topN {
		paths = paths[:topN]
	}
	return paths
}
