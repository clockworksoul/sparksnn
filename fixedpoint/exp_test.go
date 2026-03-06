package fixedpoint

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExpLUT_PointwiseAccuracy validates LUT accuracy across the full input
// range. We measure three regions reflecting Q16 precision characteristics:
//
//  1. Core range [-4, 8]: exp(x) ≥ 0.018 → ≥1200 Q16 levels. Interpolation
//     error dominates; should be <0.1%.
//  2. Low precision [-8, -4): exp(x) < 0.018 → <1200 Q16 levels. Relative
//     error is dominated by quantization noise (up to ~6%), but these values
//     are negligibly small in the WKV accumulator (see TestExpLUT_WKVAccumulatedError).
//
// The -4 boundary is chosen because exp(-4) ≈ 0.018 → 1200 Q16 levels,
// giving ~0.08% quantization granularity — below our interpolation error.
func TestExpLUT_PointwiseAccuracy(t *testing.T) {
	lut := NewExpLUT(-8, 8)

	var coreMaxRelErr, coreSum float64
	var lowMaxRelErr, lowSum float64
	var coreSamples, lowSamples int
	const nSamples = 100_000

	for i := range nSamples {
		x := -8.0 + 16.0*float64(i)/float64(nSamples-1)
		expected := math.Exp(x)
		got := lut.EvalFloat(x)

		if expected == 0 {
			continue
		}

		relErr := math.Abs(got-expected) / expected

		if x < -4.0 {
			lowSamples++
			if relErr > lowMaxRelErr {
				lowMaxRelErr = relErr
			}
			lowSum += relErr
		} else {
			coreSamples++
			if relErr > coreMaxRelErr {
				coreMaxRelErr = relErr
			}
			coreSum += relErr
		}
	}

	coreMean := coreSum / float64(coreSamples)
	lowMean := lowSum / float64(lowSamples)

	t.Logf("Core range [-4, 8] (%d samples, ≥1200 Q16 levels):", coreSamples)
	t.Logf("  Max relative error:  %.4f%%", coreMaxRelErr*100)
	t.Logf("  Mean relative error: %.4f%%", coreMean*100)
	t.Logf("Low precision [-8, -4) (%d samples, <1200 Q16 levels):", lowSamples)
	t.Logf("  Max relative error:  %.4f%% (Q16 quantization dominated)", lowMaxRelErr*100)
	t.Logf("  Mean relative error: %.4f%%", lowMean*100)

	// Core range: interpolation error dominates, should be tight.
	// Max ~0.15% occurs near x=-4 boundary where Q16 levels are ~1200.
	if coreMaxRelErr > 0.002 { // 0.2%
		t.Errorf("core max relative error %.4f%% exceeds 0.2%% threshold", coreMaxRelErr*100)
	}
	if coreMean > 0.0005 { // 0.05%
		t.Errorf("core mean relative error %.4f%% exceeds 0.05%% threshold", coreMean*100)
	}

	// Low precision range: Q16 quantization noise dominates, but bounded.
	if lowMaxRelErr > 0.10 { // 10%
		t.Errorf("low-precision max relative error %.4f%% exceeds 10%% bound", lowMaxRelErr*100)
	}
}

// TestExpLUT_KnownValues checks specific exp() values for sanity.
func TestExpLUT_KnownValues(t *testing.T) {
	lut := NewExpLUT(-8, 8)

	tests := []struct {
		input    float64
		expected float64
		tol      float64 // absolute tolerance
	}{
		{0.0, 1.0, 0.001},
		{1.0, math.E, 0.01},
		{-1.0, 1.0 / math.E, 0.001},
		{2.0, math.Exp(2), 0.02},
		{-2.0, math.Exp(-2), 0.001},
		{8.0, math.Exp(8), 1.0},  // large values get more absolute error
		{-8.0, math.Exp(-8), 0.001},
	}

	for _, tt := range tests {
		got := lut.EvalFloat(tt.input)
		if math.Abs(got-tt.expected) > tt.tol {
			t.Errorf("Eval(%.1f) = %.6f, want %.6f (tol %.4f)",
				tt.input, got, tt.expected, tt.tol)
		}
	}
}

// TestExpLUT_Monotonicity verifies that the LUT preserves the monotonic
// property of exp(): if x1 < x2, then exp(x1) < exp(x2).
func TestExpLUT_Monotonicity(t *testing.T) {
	lut := NewExpLUT(-8, 8)

	const nSamples = 50_000
	prev := lut.EvalFloat(-8.0)

	for i := 1; i <= nSamples; i++ {
		x := -8.0 + 16.0*float64(i)/float64(nSamples)
		curr := lut.EvalFloat(x)

		if curr < prev {
			t.Errorf("monotonicity violated: exp(%.6f)=%.6f < exp(prev)=%.6f",
				x, curr, prev)
		}
		prev = curr
	}
}

// TestExpLUT_Clamping verifies that out-of-range inputs are clamped gracefully.
func TestExpLUT_Clamping(t *testing.T) {
	lut := NewExpLUT(-8, 8)

	// Values beyond range should clamp to boundary values.
	atMin := lut.EvalFloat(-8.0)
	belowMin := lut.EvalFloat(-100.0)
	if belowMin != atMin {
		t.Errorf("expected clamping at min: got %.6f, want %.6f", belowMin, atMin)
	}

	atMax := lut.EvalFloat(8.0)
	aboveMax := lut.EvalFloat(100.0)
	if aboveMax != atMax {
		t.Errorf("expected clamping at max: got %.6f, want %.6f", aboveMax, atMax)
	}
}

// TestExpLUT_Q16Precision verifies that Q16 fixed-point has sufficient
// precision across the range, particularly that exp(-8) doesn't collapse
// to zero.
func TestExpLUT_Q16Precision(t *testing.T) {
	lut := NewExpLUT(-8, 8)

	// exp(-8) ≈ 0.000335 → in Q16: ~22. Small but nonzero.
	q16Val := lut.Eval(int32(-8 * q16f))
	t.Logf("exp(-8) in Q16: %d (expected ~22)", q16Val)
	if q16Val <= 0 {
		t.Error("exp(-8) collapsed to zero in Q16")
	}

	// exp(0) = 1.0 → in Q16: 65536.
	q16Val = lut.Eval(0)
	t.Logf("exp(0) in Q16: %d (expected 65536)", q16Val)
	if math.Abs(float64(q16Val)-q16f) > 10 {
		t.Errorf("exp(0) in Q16 = %d, expected ~65536", q16Val)
	}

	// exp(8) ≈ 2981 → in Q16: ~195,360,063. Must fit int32.
	q16Val = lut.Eval(int32(8 * q16f))
	t.Logf("exp(8) in Q16: %d (expected ~195,360,063, max int32 = %d)", q16Val, math.MaxInt32)
	if q16Val <= 0 {
		t.Error("exp(8) overflowed to negative in Q16")
	}
}

// TestExpLUT_WKVAccumulatedError simulates the RWKV WKV recurrence over
// 512 timesteps and measures accumulated error vs float64 reference.
// This is the critical test: does LUT error compound unacceptably?
func TestExpLUT_WKVAccumulatedError(t *testing.T) {
	lut := NewExpLUT(-8, 8)
	rng := rand.New(rand.NewPCG(42, 0))

	const (
		channels  = 64
		timesteps = 512
		decay     = 0.95
	)

	// Generate random k and v sequences in realistic ranges.
	// k values: [-4, 4] (typical after time_mix interpolation)
	// v values: [-2, 2] (typical projection output)
	k := make([][]float64, timesteps)
	v := make([][]float64, timesteps)
	for t := range timesteps {
		k[t] = make([]float64, channels)
		v[t] = make([]float64, channels)
		for c := range channels {
			k[t][c] = (rng.Float64() - 0.5) * 8 // [-4, 4]
			v[t][c] = (rng.Float64() - 0.5) * 4  // [-2, 2]
		}
	}

	// Run WKV with float64 reference.
	refA := make([]float64, channels)
	for ts := range timesteps {
		for c := range channels {
			refA[c] = decay*refA[c] + math.Exp(k[ts][c])*v[ts][c]
		}
	}

	// Run WKV with LUT approximation.
	lutA := make([]float64, channels)
	for ts := range timesteps {
		for c := range channels {
			expK := lut.EvalFloat(k[ts][c])
			lutA[c] = decay*lutA[c] + expK*v[ts][c]
		}
	}

	// Measure per-channel relative error and cosine similarity.
	var maxRelErr float64
	var sumRelErr float64
	var dotProduct, refNorm, lutNorm float64
	validChannels := 0

	for c := range channels {
		dotProduct += refA[c] * lutA[c]
		refNorm += refA[c] * refA[c]
		lutNorm += lutA[c] * lutA[c]

		if math.Abs(refA[c]) < 1e-10 {
			continue // skip near-zero channels
		}
		validChannels++

		relErr := math.Abs(lutA[c]-refA[c]) / math.Abs(refA[c])
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		sumRelErr += relErr
	}

	cosineSim := dotProduct / (math.Sqrt(refNorm) * math.Sqrt(lutNorm))
	meanRelErr := sumRelErr / float64(validChannels)

	t.Logf("WKV accumulated error (%d channels × %d timesteps):", channels, timesteps)
	t.Logf("  Max relative error:  %.3f%%", maxRelErr*100)
	t.Logf("  Mean relative error: %.3f%%", meanRelErr*100)
	t.Logf("  Cosine similarity:   %.10f", cosineSim)

	if maxRelErr > 0.01 {
		t.Errorf("max accumulated error %.3f%% exceeds 1%% threshold", maxRelErr*100)
	}
	if cosineSim < 0.999999 {
		t.Errorf("cosine similarity %.10f below 0.999999 threshold", cosineSim)
	}
}

// BenchmarkExpLUT_Eval measures the throughput of the integer LUT evaluation.
func BenchmarkExpLUT_Eval(b *testing.B) {
	lut := NewExpLUT(-8, 8)

	// Pre-generate random Q16 inputs in [-8, 8].
	rng := rand.New(rand.NewPCG(42, 0))
	inputs := make([]int32, 1024)
	for i := range inputs {
		x := (rng.Float64() - 0.5) * 16 // [-8, 8]
		inputs[i] = int32(x * q16f)
	}

	b.ResetTimer()
	var sink int32
	for i := range b.N {
		sink = lut.Eval(inputs[i%len(inputs)])
	}
	_ = sink
}

// BenchmarkExpFloat_Reference provides a baseline: how fast is math.Exp?
func BenchmarkExpFloat_Reference(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	inputs := make([]float64, 1024)
	for i := range inputs {
		inputs[i] = (rng.Float64() - 0.5) * 16
	}

	b.ResetTimer()
	var sink float64
	for i := range b.N {
		sink = math.Exp(inputs[i%len(inputs)])
	}
	_ = sink
}
