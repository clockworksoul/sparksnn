// Package benchmark provides a framework for testing learning rules
// against classification tasks. It defines the Task interface and
// convergence tracking utilities.
package benchmark

import (
	"fmt"
	"io"
	"math"
	"strings"
)

// Sample is a single labeled data point for classification.
type Sample struct {
	// Inputs are the feature values, normalized to [0, 255].
	// 0 = no signal, 255 = maximum signal.
	Inputs []byte

	// Label is the correct class (0-indexed).
	Label int
}

// Task defines a classification benchmark.
type Task interface {
	// Name returns a human-readable name for the task.
	Name() string

	// NumInputs returns the number of input features.
	NumInputs() int

	// NumClasses returns the number of output classes.
	NumClasses() int

	// TrainingSamples returns the training dataset.
	TrainingSamples() []Sample

	// TestSamples returns the test dataset.
	TestSamples() []Sample
}

// Checkpoint records accuracy and network statistics at a point
// during training.
type Checkpoint struct {
	// SamplesProcessed is how many training samples have been
	// presented so far.
	SamplesProcessed int

	// Accuracy is the classification accuracy on the test set
	// at this point (0.0 to 1.0).
	Accuracy float64

	// WeightMean is the mean weight across all learning connections.
	WeightMean float64

	// WeightStdDev is the standard deviation of weights.
	WeightStdDev float64

	// DeadNeurons is the number of hidden neurons that didn't fire
	// during the last evaluation.
	DeadNeurons int

	// SpikeRate is the average number of spikes per hidden neuron
	// per sample during the last evaluation.
	SpikeRate float64
}

// Tracker accumulates checkpoints and detects convergence.
type Tracker struct {
	// Checkpoints is the ordered list of recorded checkpoints.
	Checkpoints []Checkpoint

	// PatienceCheckpoints is how many consecutive checkpoints
	// without improvement before declaring convergence.
	PatienceCheckpoints int

	// MinImprovement is the minimum accuracy improvement (absolute)
	// to count as "improved." Default: 0.001 (0.1%).
	MinImprovement float64
}

// NewTracker creates a convergence tracker.
// patience is how many checkpoints without improvement before
// convergence is declared. Set to 0 to disable convergence detection.
func NewTracker(patience int) *Tracker {
	return &Tracker{
		PatienceCheckpoints: patience,
		MinImprovement:      0.001,
	}
}

// Record adds a checkpoint and returns true if training has converged.
func (t *Tracker) Record(cp Checkpoint) bool {
	t.Checkpoints = append(t.Checkpoints, cp)

	if t.PatienceCheckpoints <= 0 || len(t.Checkpoints) < t.PatienceCheckpoints+1 {
		return false
	}

	best := t.BestAccuracy()
	recent := t.Checkpoints[len(t.Checkpoints)-t.PatienceCheckpoints:]
	for _, r := range recent {
		if best-r.Accuracy < t.MinImprovement {
			// At least one recent checkpoint is near best
			return false
		}
	}

	// All recent checkpoints are worse than best by > MinImprovement
	return true
}

// BestAccuracy returns the highest accuracy seen so far.
func (t *Tracker) BestAccuracy() float64 {
	best := 0.0
	for _, cp := range t.Checkpoints {
		if cp.Accuracy > best {
			best = cp.Accuracy
		}
	}
	return best
}

// LastAccuracy returns the most recent accuracy, or 0 if no
// checkpoints recorded.
func (t *Tracker) LastAccuracy() float64 {
	if len(t.Checkpoints) == 0 {
		return 0
	}
	return t.Checkpoints[len(t.Checkpoints)-1].Accuracy
}

// PrintReport writes a summary of training progress.
func (t *Tracker) PrintReport(w io.Writer, taskName, ruleName string) {
	fmt.Fprintf(w, "\n=== %s / %s ===\n", taskName, ruleName)

	if len(t.Checkpoints) == 0 {
		fmt.Fprintln(w, "No checkpoints recorded.")
		return
	}

	// Header
	fmt.Fprintf(w, "%-10s %-10s %-12s %-12s %-8s %-10s\n",
		"Samples", "Accuracy", "WeightMean", "WeightStd", "Dead", "SpikeRate")
	fmt.Fprintln(w, strings.Repeat("-", 66))

	for _, cp := range t.Checkpoints {
		fmt.Fprintf(w, "%-10d %-10.2f%% %-12.1f %-12.1f %-8d %-10.2f\n",
			cp.SamplesProcessed,
			cp.Accuracy*100,
			cp.WeightMean,
			cp.WeightStdDev,
			cp.DeadNeurons,
			cp.SpikeRate,
		)
	}

	fmt.Fprintf(w, "\nBest accuracy: %.2f%%\n", t.BestAccuracy()*100)
}

// WeightStats computes mean and standard deviation for a slice of
// int16 weights.
func WeightStats(weights []int16) (mean, stddev float64) {
	if len(weights) == 0 {
		return 0, 0
	}

	var sum float64
	for _, w := range weights {
		sum += float64(w)
	}
	mean = sum / float64(len(weights))

	var variance float64
	for _, w := range weights {
		d := float64(w) - mean
		variance += d * d
	}
	variance /= float64(len(weights))
	stddev = math.Sqrt(variance)

	return mean, stddev
}
