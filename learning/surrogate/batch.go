package surrogate

import (
	"math"
	"sync"
)

// BatchSample holds a single training sample for batch processing.
type BatchSample struct {
	InputValues  []float64
	CorrectClass int
}

// workerBuffers holds per-goroutine working memory for gradient computation.
// Mirrors trainBuffers but independent per worker.
type workerBuffers struct {
	mem         []float64
	traceU      []float64
	traceS      []float64
	spikeCounts []float64
	current     []float64
	dLdU        []float64
	dLdW        [][]float64

	numNeurons int
	numSteps   int
	numOutputs int
}

func newWorkerBuffers(numNeurons, numSteps, numOutputs int, weightTopology []int) *workerBuffers {
	wb := &workerBuffers{
		mem:         make([]float64, numNeurons),
		traceU:      make([]float64, numNeurons*numSteps),
		traceS:      make([]float64, numNeurons*numSteps),
		spikeCounts: make([]float64, numOutputs),
		current:     make([]float64, numNeurons),
		dLdU:        make([]float64, numNeurons*numSteps),
		dLdW:        make([][]float64, len(weightTopology)),
		numNeurons:  numNeurons,
		numSteps:    numSteps,
		numOutputs:  numOutputs,
	}
	for i, n := range weightTopology {
		wb.dLdW[i] = make([]float64, n)
	}
	return wb
}

func (wb *workerBuffers) reset() {
	clear(wb.mem)
	clear(wb.traceU)
	clear(wb.traceS)
	clear(wb.spikeCounts)
	clear(wb.current)
	clear(wb.dLdU)
	for i := range wb.dLdW {
		clear(wb.dLdW[i])
	}
}

// computeGradients runs a forward+backward pass for a single sample,
// writing gradients into wb.dLdW. Returns the loss.
// This is the core of TrainSample, extracted for parallel use.
// It reads (but does not modify) the trainer's weights and connections.
func (t *Trainer) computeGradients(wb *workerBuffers, inputValues []float64, correctClass int) float64 {
	cfg := t.Config
	numNeurons := wb.numNeurons
	numSteps := wb.numSteps
	numOutputs := wb.numOutputs
	outputStart := cfg.Layers[len(cfg.Layers)-1].Start

	wb.reset()

	mem := wb.mem
	traceU := wb.traceU
	traceS := wb.traceS
	spikeCounts := wb.spikeCounts
	current := wb.current
	dLdU := wb.dLdU
	dLdW := wb.dLdW

	// ===== FORWARD PASS =====
	for step := 0; step < numSteps; step++ {
		clear(current)

		inputLayer := cfg.Layers[0]
		for i := inputLayer.Start; i < inputLayer.End; i++ {
			idx := int(i - inputLayer.Start)
			if idx < len(inputValues) {
				current[i] += inputValues[idx] * cfg.InputWeight
			}
		}

		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traceS[src*numSteps+step-1] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					current[tgt] += t.weights[src][j]
				}
			}
		}

		for i := 0; i < numNeurons; i++ {
			mem[i] = cfg.Beta*mem[i] + current[i]
			traceU[i*numSteps+step] = mem[i]
			if mem[i] >= cfg.Threshold {
				traceS[i*numSteps+step] = 1.0
				if uint32(i) >= outputStart && uint32(i) < outputStart+uint32(numOutputs) {
					spikeCounts[uint32(i)-outputStart] += 1.0
				}
				mem[i] -= cfg.Threshold
			}
		}
	}

	// ===== COMPUTE LOSS =====
	loss, dLdCounts := SpikeCountCrossEntropy(spikeCounts, correctClass)

	// ===== BACKWARD PASS (BPTT) =====
	for step := numSteps - 1; step >= 0; step-- {
		for oi := 0; oi < numOutputs; oi++ {
			nIdx := int(outputStart) + oi
			surr := cfg.Surrogate.Derivative(traceU[nIdx*numSteps+step], cfg.Threshold)
			dLdU[nIdx*numSteps+step] += dLdCounts[oi] * surr
		}

		if step > 0 {
			for i := 0; i < numNeurons; i++ {
				if traceS[i*numSteps+step] == 0 {
					dLdU[i*numSteps+step-1] += cfg.Beta * dLdU[i*numSteps+step]
				}
			}
		}

		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traceS[src*numSteps+step-1] == 0 {
					continue
				}
				for j, tgt := range t.connections[src] {
					dLdW[src][j] += dLdU[int(tgt)*numSteps+step]
				}
			}
		}

		if step > 0 {
			for src := 0; src < numNeurons; src++ {
				if traceS[src*numSteps+step-1] == 0 {
					continue
				}
				var dLdS float64
				for j, tgt := range t.connections[src] {
					dLdS += t.weights[src][j] * dLdU[int(tgt)*numSteps+step]
				}
				surr := cfg.Surrogate.Derivative(traceU[src*numSteps+step-1], cfg.Threshold)
				dLdU[src*numSteps+step-1] += dLdS * surr
			}
		}
	}

	return loss
}

// TrainBatch trains on a mini-batch of samples in parallel.
// Gradients are computed concurrently across numWorkers goroutines,
// then averaged and applied as a single weight update.
// Returns the average loss across the batch.
func (t *Trainer) TrainBatch(samples []BatchSample, numWorkers int) float64 {
	if len(samples) == 0 {
		return 0
	}
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > len(samples) {
		numWorkers = len(samples)
	}

	cfg := t.Config
	numNeurons := len(t.Net.Neurons)
	numSteps := cfg.NumSteps
	numOutputs := int(cfg.Layers[len(cfg.Layers)-1].End - cfg.Layers[len(cfg.Layers)-1].Start)

	// Build weight topology (number of connections per source neuron)
	weightTopo := make([]int, len(t.weights))
	for i := range t.weights {
		weightTopo[i] = len(t.weights[i])
	}

	// Pre-allocate worker buffer pool
	bufPool := make([]*workerBuffers, numWorkers)
	for w := 0; w < numWorkers; w++ {
		bufPool[w] = newWorkerBuffers(numNeurons, numSteps, numOutputs, weightTopo)
	}

	// Accumulated gradients and loss (per worker, to avoid contention)
	type workerResult struct {
		dLdW     [][]float64
		lossSum  float64
		count    int
	}
	results := make([]workerResult, numWorkers)
	for w := 0; w < numWorkers; w++ {
		results[w].dLdW = make([][]float64, len(t.weights))
		for i := range t.weights {
			results[w].dLdW[i] = make([]float64, len(t.weights[i]))
		}
	}

	// Distribute samples across workers
	var wg sync.WaitGroup
	samplesPerWorker := (len(samples) + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * samplesPerWorker
		end := start + samplesPerWorker
		if end > len(samples) {
			end = len(samples)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(workerID int, batch []BatchSample) {
			defer wg.Done()
			wb := bufPool[workerID]
			res := &results[workerID]

			for _, s := range batch {
				loss := t.computeGradients(wb, s.InputValues, s.CorrectClass)
				res.lossSum += loss
				res.count++

				// Accumulate gradients
				for src := range wb.dLdW {
					for j := range wb.dLdW[src] {
						res.dLdW[src][j] += wb.dLdW[src][j]
					}
				}
			}
		}(w, samples[start:end])
	}

	wg.Wait()

	// Merge results: average gradients across all samples
	batchSize := float64(len(samples))
	totalLoss := 0.0

	// Use the trainer's existing dLdW buffer for the merged result
	for i := range t.buf.dLdW {
		clear(t.buf.dLdW[i])
	}

	for w := 0; w < numWorkers; w++ {
		totalLoss += results[w].lossSum
		for src := range results[w].dLdW {
			for j := range results[w].dLdW[src] {
				t.buf.dLdW[src][j] += results[w].dLdW[src][j]
			}
		}
	}

	// Average the gradients
	for src := range t.buf.dLdW {
		for j := range t.buf.dLdW[src] {
			t.buf.dLdW[src][j] /= batchSize
		}
	}

	// ===== WEIGHT UPDATE (single-threaded) =====
	lr := cfg.LearningRate
	dLdW := t.buf.dLdW

	if t.useAdam {
		t.adam.t++
		tt := float64(t.adam.t)
		b1, b2, eps := t.adam.beta1, t.adam.beta2, t.adam.epsilon
		bc1 := 1.0 - math.Pow(b1, tt)
		bc2 := 1.0 - math.Pow(b2, tt)
		for src := range t.weights {
			for j := range t.weights[src] {
				g := dLdW[src][j]
				t.adam.m[src][j] = b1*t.adam.m[src][j] + (1-b1)*g
				t.adam.v[src][j] = b2*t.adam.v[src][j] + (1-b2)*g*g
				mHat := t.adam.m[src][j] / bc1
				vHat := t.adam.v[src][j] / bc2
				t.weights[src][j] -= lr * mHat / (math.Sqrt(vHat) + eps)
			}
		}
	} else {
		for src := range t.weights {
			for j := range t.weights[src] {
				t.weights[src][j] -= lr * dLdW[src][j]
			}
		}
	}

	// Sync to int32 network
	t.syncWeightsToNetwork()
	t.Net.ResetActivation()

	return totalLoss / batchSize
}
