package fixedpoint

import (
	"math"
	"math/rand/v2"
	"testing"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// TestSigmoidLUT_PointwiseAccuracy validates LUT accuracy across the full range.
func TestSigmoidLUT_PointwiseAccuracy(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)

	var maxRelErr, maxAbsErr float64
	var sumRelErr float64
	var worstX float64
	const nSamples = 100_000

	for i := range nSamples {
		x := -8.0 + 16.0*float64(i)/float64(nSamples-1)
		expected := sigmoid(x)
		got := lut.EvalFloat(x)

		absErr := math.Abs(got - expected)
		if absErr > maxAbsErr {
			maxAbsErr = absErr
		}

		// Use absolute error for values near 0 or 1 (relative is misleading).
		if expected > 0.01 && expected < 0.99 {
			relErr := absErr / expected
			if relErr > maxRelErr {
				maxRelErr = relErr
				worstX = x
			}
			sumRelErr += relErr
		}
	}

	t.Logf("Sigmoid point-wise accuracy over %d samples:", nSamples)
	t.Logf("  Max relative error (core 0.01-0.99): %.4f%% (at x=%.3f)", maxRelErr*100, worstX)
	t.Logf("  Max absolute error (full range):     %.6f", maxAbsErr)

	if maxRelErr > 0.005 { // 0.5%
		t.Errorf("max relative error %.4f%% exceeds 0.5%% threshold", maxRelErr*100)
	}
	if maxAbsErr > 0.001 {
		t.Errorf("max absolute error %.6f exceeds 0.001 threshold", maxAbsErr)
	}
}

// TestSigmoidLUT_KnownValues checks specific sigmoid values.
func TestSigmoidLUT_KnownValues(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)

	tests := []struct {
		input    float64
		expected float64
		tol      float64
	}{
		{0.0, 0.5, 0.001},
		{1.0, sigmoid(1.0), 0.001},   // ~0.7311
		{-1.0, sigmoid(-1.0), 0.001}, // ~0.2689
		{2.0, sigmoid(2.0), 0.001},   // ~0.8808
		{-2.0, sigmoid(-2.0), 0.001}, // ~0.1192
		{5.0, sigmoid(5.0), 0.001},   // ~0.9933
		{-5.0, sigmoid(-5.0), 0.001}, // ~0.0067
		{8.0, sigmoid(8.0), 0.001},   // ~0.9997
		{-8.0, sigmoid(-8.0), 0.001}, // ~0.0003
	}

	for _, tt := range tests {
		got := lut.EvalFloat(tt.input)
		if math.Abs(got-tt.expected) > tt.tol {
			t.Errorf("Sigmoid(%.1f) = %.6f, want %.6f (tol %.4f)",
				tt.input, got, tt.expected, tt.tol)
		}
	}
}

// TestSigmoidLUT_Symmetry verifies σ(x) + σ(-x) ≈ 1.
func TestSigmoidLUT_Symmetry(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)

	var maxAsymmetry float64
	const nSamples = 10_000

	for i := range nSamples {
		x := 8.0 * float64(i) / float64(nSamples-1) // [0, 8]
		pos := lut.EvalFloat(x)
		neg := lut.EvalFloat(-x)
		sum := pos + neg
		asymmetry := math.Abs(sum - 1.0)
		if asymmetry > maxAsymmetry {
			maxAsymmetry = asymmetry
		}
	}

	t.Logf("Symmetry: max |σ(x) + σ(-x) - 1| = %.6f", maxAsymmetry)

	if maxAsymmetry > 0.001 {
		t.Errorf("max asymmetry %.6f exceeds 0.001 threshold", maxAsymmetry)
	}
}

// TestSigmoidLUT_Monotonicity verifies that the LUT preserves monotonicity.
func TestSigmoidLUT_Monotonicity(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)

	const nSamples = 50_000
	prev := lut.EvalFloat(-8.0)

	for i := 1; i <= nSamples; i++ {
		x := -8.0 + 16.0*float64(i)/float64(nSamples)
		curr := lut.EvalFloat(x)

		if curr < prev {
			t.Errorf("monotonicity violated: σ(%.6f)=%.6f < σ(prev)=%.6f",
				x, curr, prev)
		}
		prev = curr
	}
}

// TestSigmoidLUT_Clamping verifies graceful clamping at extremes.
func TestSigmoidLUT_Clamping(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)

	// Below range → clamp to σ(-8) ≈ 0
	belowMin := lut.EvalFloat(-100.0)
	atMin := lut.EvalFloat(-8.0)
	if belowMin != atMin {
		t.Errorf("expected clamping at min: got %.6f, want %.6f", belowMin, atMin)
	}

	// Above range → clamp to σ(8) ≈ 1
	aboveMax := lut.EvalFloat(100.0)
	atMax := lut.EvalFloat(8.0)
	if aboveMax != atMax {
		t.Errorf("expected clamping at max: got %.6f, want %.6f", aboveMax, atMax)
	}
}

// TestSigmoidLUT_GatingAccuracy simulates sigmoid gating (the RWKV use case):
// output = sigmoid(r) * wkv. Measures error vs float64 reference.
func TestSigmoidLUT_GatingAccuracy(t *testing.T) {
	lut := NewSigmoidLUT(-8, 8)
	rng := rand.New(rand.NewPCG(42, 0))

	const nSamples = 10_000
	var maxRelErr float64

	for range nSamples {
		// r from linear projection: typically [-5, 5]
		r := (rng.Float64() - 0.5) * 10
		// wkv: weighted average of v, typically [-2, 2]
		wkv := (rng.Float64() - 0.5) * 4

		expected := sigmoid(r) * wkv
		got := lut.EvalFloat(r) * wkv

		if math.Abs(expected) < 1e-6 {
			continue
		}

		relErr := math.Abs(got-expected) / math.Abs(expected)
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}

	t.Logf("Gating accuracy (sigmoid(r) * wkv, %d samples):", nSamples)
	t.Logf("  Max relative error: %.4f%%", maxRelErr*100)

	if maxRelErr > 0.005 { // 0.5%
		t.Errorf("gating max relative error %.4f%% exceeds 0.5%%", maxRelErr*100)
	}
}

// BenchmarkSigmoidLUT_Eval measures throughput.
func BenchmarkSigmoidLUT_Eval(b *testing.B) {
	lut := NewSigmoidLUT(-8, 8)

	rng := rand.New(rand.NewPCG(42, 0))
	inputs := make([]int32, 1024)
	for i := range inputs {
		x := (rng.Float64() - 0.5) * 16
		inputs[i] = int32(x * q16f)
	}

	b.ResetTimer()
	var sink int32
	for i := range b.N {
		sink = lut.Eval(inputs[i%len(inputs)])
	}
	_ = sink
}

// BenchmarkSigmoidFloat_Reference provides a baseline.
func BenchmarkSigmoidFloat_Reference(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	inputs := make([]float64, 1024)
	for i := range inputs {
		inputs[i] = (rng.Float64() - 0.5) * 16
	}

	b.ResetTimer()
	var sink float64
	for i := range b.N {
		sink = sigmoid(inputs[i%len(inputs)])
	}
	_ = sink
}
