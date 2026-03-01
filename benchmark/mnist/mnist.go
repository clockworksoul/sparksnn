// Package mnist provides the MNIST handwritten digit classification
// task for benchmarking learning rules. MNIST has 784 features
// (28x28 grayscale pixels) and 10 classes (digits 0-9).
//
// Data is loaded from gzipped IDX files in data/mnist/.
package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/clockworksoul/sparksnn/benchmark"
)

// Task implements the MNIST benchmark.
type Task struct {
	train []benchmark.Sample
	test  []benchmark.Sample
}

func (t *Task) Name() string       { return "MNIST" }
func (t *Task) NumInputs() int     { return 784 }
func (t *Task) NumClasses() int    { return 10 }

func (t *Task) TrainingSamples() []benchmark.Sample { return t.train }
func (t *Task) TestSamples() []benchmark.Sample     { return t.test }

// NewTask loads MNIST from the data/mnist/ directory.
// maxTrain and maxTest limit the number of samples loaded (0 = all).
func NewTask(maxTrain, maxTest int) (*Task, error) {
	dataDir := findDataDir()

	train, err := loadSamples(
		filepath.Join(dataDir, "train-images-idx3-ubyte.gz"),
		filepath.Join(dataDir, "train-labels-idx1-ubyte.gz"),
		maxTrain,
	)
	if err != nil {
		return nil, fmt.Errorf("loading training data: %w", err)
	}

	test, err := loadSamples(
		filepath.Join(dataDir, "t10k-images-idx3-ubyte.gz"),
		filepath.Join(dataDir, "t10k-labels-idx1-ubyte.gz"),
		maxTest,
	)
	if err != nil {
		return nil, fmt.Errorf("loading test data: %w", err)
	}

	return &Task{train: train, test: test}, nil
}

// findDataDir locates the data/mnist/ directory relative to this source file.
func findDataDir() string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "data", "mnist")
}

// loadSamples reads gzipped IDX image and label files.
func loadSamples(imagesPath, labelsPath string, max int) ([]benchmark.Sample, error) {
	images, rows, cols, err := loadImages(imagesPath)
	if err != nil {
		return nil, err
	}

	labels, err := loadLabels(labelsPath)
	if err != nil {
		return nil, err
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("image count %d != label count %d", len(images), len(labels))
	}

	n := len(images)
	if max > 0 && max < n {
		n = max
	}

	pixelsPerImage := rows * cols
	samples := make([]benchmark.Sample, n)
	for i := 0; i < n; i++ {
		inputs := make([]byte, pixelsPerImage)
		copy(inputs, images[i])
		samples[i] = benchmark.Sample{
			Inputs: inputs,
			Label:  int(labels[i]),
		}
	}

	return samples, nil
}

// loadImages reads a gzipped IDX3 image file.
func loadImages(path string) (images [][]byte, rows, cols int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, 0, 0, err
	}
	defer gz.Close()

	var header struct {
		Magic    int32
		NumItems int32
		Rows     int32
		Cols     int32
	}
	if err := binary.Read(gz, binary.BigEndian, &header); err != nil {
		return nil, 0, 0, fmt.Errorf("reading image header: %w", err)
	}

	if header.Magic != 2051 {
		return nil, 0, 0, fmt.Errorf("bad image magic: %d", header.Magic)
	}

	rows = int(header.Rows)
	cols = int(header.Cols)
	n := int(header.NumItems)
	pixelsPerImage := rows * cols

	allPixels := make([]byte, n*pixelsPerImage)
	if _, err := readFull(gz, allPixels); err != nil {
		return nil, 0, 0, fmt.Errorf("reading image data: %w", err)
	}

	images = make([][]byte, n)
	for i := 0; i < n; i++ {
		images[i] = allPixels[i*pixelsPerImage : (i+1)*pixelsPerImage]
	}

	return images, rows, cols, nil
}

// loadLabels reads a gzipped IDX1 label file.
func loadLabels(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var header struct {
		Magic    int32
		NumItems int32
	}
	if err := binary.Read(gz, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("reading label header: %w", err)
	}

	if header.Magic != 2049 {
		return nil, fmt.Errorf("bad label magic: %d", header.Magic)
	}

	labels := make([]byte, header.NumItems)
	if _, err := readFull(gz, labels); err != nil {
		return nil, fmt.Errorf("reading label data: %w", err)
	}

	return labels, nil
}

// readFull reads exactly len(buf) bytes, handling partial reads from gzip.
func readFull(r *gzip.Reader, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := r.Read(buf[total:])
		total += n
		if err != nil {
			if total == len(buf) {
				return total, nil
			}
			return total, err
		}
	}
	return total, nil
}
