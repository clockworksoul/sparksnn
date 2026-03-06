package fixedpoint

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExpLUT64_PointwiseAccuracy validates Q32 LUT accuracy across the full
// range, with special attention to the low-precision region where Q16 struggles.
func TestExpLUT64_PointwiseAccuracy(t *testing.T) {
	lut64 := NewExpLUT64(-8, 8)
	lut16 := NewExpLUT(-8, 8)

	const nSamples = 100_000

	// Track Q32 and Q16 errors separately for comparison.
	var q32CoreMax, q32LowMax float64
	var q16CoreMax, q16LowMax float64
	var q32CoreSum, q16CoreSum float64
	var coreSamples, lowSamples int

	for i := range nSamples {
		x := -8.0 + 16.0*float64(i)/float64(nSamples-1)
		expected := math.Exp(x)
		if expected == 0 {
			continue
		}

		got64 := lut64.EvalFloat(x)
		got16 := lut16.EvalFloat(x)
		err64 := math.Abs(got64-expected) / expected
		err16 := math.Abs(got16-expected) / expected

		if x < -4.0 {
			lowSamples++
			if err64 > q32LowMax {
				q32LowMax = err64
			}
			if err16 > q16LowMax {
				q16LowMax = err16
			}
		} else {
			coreSamples++
			q32CoreSum += err64
			q16CoreSum += err16
			if err64 > q32CoreMax {
				q32CoreMax = err64
			}
			if err16 > q16CoreMax {
				q16CoreMax = err16
			}
		}
	}

	t.Logf("=== Q32 vs Q16 Point-wise Comparison ===")
	t.Logf("")
	t.Logf("Core range [-4, 8] (%d samples):", coreSamples)
	t.Logf("  Q16: max=%.4f%%  mean=%.4f%%", q16CoreMax*100, q16CoreSum/float64(coreSamples)*100)
	t.Logf("  Q32: max=%.4f%%  mean=%.4f%%", q32CoreMax*100, q32CoreSum/float64(coreSamples)*100)
	t.Logf("  Improvement: %.1fx max, %.1fx mean",
		q16CoreMax/q32CoreMax, (q16CoreSum/float64(coreSamples))/(q32CoreSum/float64(coreSamples)))
	t.Logf("")
	t.Logf("Low precision [-8, -4) (%d samples):", lowSamples)
	t.Logf("  Q16: max=%.4f%%", q16LowMax*100)
	t.Logf("  Q32: max=%.4f%%", q32LowMax*100)
	t.Logf("  Improvement: %.1fx", q16LowMax/q32LowMax)

	// Q32 should be meaningfully better than Q16 in the low range.
	if q32LowMax > 0.01 { // 1% — Q16 hits 6% here
		t.Errorf("Q32 low-range max error %.4f%% exceeds 1%%", q32LowMax*100)
	}
}

// TestExpLUT64_Q32Precision checks that Q32 eliminates the precision cliff.
func TestExpLUT64_Q32Precision(t *testing.T) {
	lut := NewExpLUT64(-8, 8)

	// exp(-8) ≈ 0.000335 → Q32: ~1,440,487. Compare to Q16: ~22.
	q32Val := lut.Eval(int32(-8 * q16f))
	t.Logf("exp(-8) in Q32: %d (vs Q16: ~22)", q32Val)
	if q32Val < 1_000_000 {
		t.Errorf("exp(-8) in Q32 = %d, expected ~1.4M", q32Val)
	}

	// exp(0) = 1.0 → Q32: 4,294,967,296
	q32Val = lut.Eval(0)
	t.Logf("exp(0) in Q32: %d (expected ~4,294,967,296)", q32Val)

	// exp(8) ≈ 2981 → Q32: ~12.8 trillion. Must fit int64.
	q32Val = lut.Eval(int32(8 * q16f))
	t.Logf("exp(8) in Q32: %d (max int64 = %d)", q32Val, int64(math.MaxInt64))
	if q32Val <= 0 {
		t.Error("exp(8) overflowed to negative in Q32")
	}
}

// TestExpLUT64_WKVAccumulatedError is the critical ablation: does Q32
// meaningfully reduce accumulated error in the WKV recurrence vs Q16?
//
// This directly answers the reviewer question: "Why not use higher precision?"
func TestExpLUT64_WKVAccumulatedError(t *testing.T) {
	lut16 := NewExpLUT(-8, 8)
	lut64 := NewExpLUT64(-8, 8)
	rng := rand.New(rand.NewPCG(42, 0))

	const (
		channels  = 64
		timesteps = 512
		decayReal = 0.95
	)

	// Generate random k and v sequences.
	k := make([][]float64, timesteps)
	v := make([][]float64, timesteps)
	for ts := range timesteps {
		k[ts] = make([]float64, channels)
		v[ts] = make([]float64, channels)
		for c := range channels {
			k[ts][c] = (rng.Float64() - 0.5) * 8
			v[ts][c] = (rng.Float64() - 0.5) * 4
		}
	}

	// Float64 reference.
	refA := make([]float64, channels)
	for ts := range timesteps {
		for c := range channels {
			refA[c] = decayReal*refA[c] + math.Exp(k[ts][c])*v[ts][c]
		}
	}

	// Q16 LUT (existing approach).
	q16A := make([]float64, channels)
	for ts := range timesteps {
		for c := range channels {
			expK := lut16.EvalFloat(k[ts][c])
			q16A[c] = decayReal*q16A[c] + expK*v[ts][c]
		}
	}

	// Q32 LUT (new approach).
	q32A := make([]float64, channels)
	for ts := range timesteps {
		for c := range channels {
			expK := lut64.EvalFloat(k[ts][c])
			q32A[c] = decayReal*q32A[c] + expK*v[ts][c]
		}
	}

	// Measure errors.
	var q16MaxErr, q32MaxErr float64
	var q16SumErr, q32SumErr float64
	var q16Dot, q32Dot, refNorm, q16Norm, q32Norm float64
	valid := 0

	for c := range channels {
		q16Dot += refA[c] * q16A[c]
		q32Dot += refA[c] * q32A[c]
		refNorm += refA[c] * refA[c]
		q16Norm += q16A[c] * q16A[c]
		q32Norm += q32A[c] * q32A[c]

		if math.Abs(refA[c]) < 1e-10 {
			continue
		}
		valid++

		e16 := math.Abs(q16A[c]-refA[c]) / math.Abs(refA[c])
		e32 := math.Abs(q32A[c]-refA[c]) / math.Abs(refA[c])

		if e16 > q16MaxErr {
			q16MaxErr = e16
		}
		if e32 > q32MaxErr {
			q32MaxErr = e32
		}
		q16SumErr += e16
		q32SumErr += e32
	}

	q16Cos := q16Dot / (math.Sqrt(refNorm) * math.Sqrt(q16Norm))
	q32Cos := q32Dot / (math.Sqrt(refNorm) * math.Sqrt(q32Norm))

	t.Logf("=== Q32 vs Q16 WKV Accumulated Error (%d ch × %d steps) ===", channels, timesteps)
	t.Logf("")
	t.Logf("          Max Rel Err    Mean Rel Err    Cosine Sim")
	t.Logf("  Q16:    %.4f%%         %.4f%%          %.10f",
		q16MaxErr*100, q16SumErr/float64(valid)*100, q16Cos)
	t.Logf("  Q32:    %.4f%%         %.4f%%          %.10f",
		q32MaxErr*100, q32SumErr/float64(valid)*100, q32Cos)
	t.Logf("")

	if q32MaxErr < q16MaxErr {
		t.Logf("  Q32 improvement: %.1fx max error reduction", q16MaxErr/q32MaxErr)
	} else {
		t.Logf("  Q32 did NOT improve max error (Q16 was already sufficient)")
	}
}

// TestExpLUT64_WKVExtendedSequence tests with a longer sequence (2048 tokens)
// to see if Q32 advantages emerge with more accumulation steps.
func TestExpLUT64_WKVExtendedSequence(t *testing.T) {
	lut16 := NewExpLUT(-8, 8)
	lut64 := NewExpLUT64(-8, 8)
	rng := rand.New(rand.NewPCG(99, 0))

	const (
		channels  = 64
		timesteps = 2048
		decayReal = 0.95
	)

	k := make([][]float64, timesteps)
	v := make([][]float64, timesteps)
	for ts := range timesteps {
		k[ts] = make([]float64, channels)
		v[ts] = make([]float64, channels)
		for c := range channels {
			k[ts][c] = (rng.Float64() - 0.5) * 8
			v[ts][c] = (rng.Float64() - 0.5) * 4
		}
	}

	refA := make([]float64, channels)
	q16A := make([]float64, channels)
	q32A := make([]float64, channels)

	for ts := range timesteps {
		for c := range channels {
			expKReal := math.Exp(k[ts][c])
			expK16 := lut16.EvalFloat(k[ts][c])
			expK32 := lut64.EvalFloat(k[ts][c])

			refA[c] = decayReal*refA[c] + expKReal*v[ts][c]
			q16A[c] = decayReal*q16A[c] + expK16*v[ts][c]
			q32A[c] = decayReal*q32A[c] + expK32*v[ts][c]
		}
	}

	var q16MaxErr, q32MaxErr float64
	valid := 0

	for c := range channels {
		if math.Abs(refA[c]) < 1e-10 {
			continue
		}
		valid++
		e16 := math.Abs(q16A[c]-refA[c]) / math.Abs(refA[c])
		e32 := math.Abs(q32A[c]-refA[c]) / math.Abs(refA[c])
		if e16 > q16MaxErr {
			q16MaxErr = e16
		}
		if e32 > q32MaxErr {
			q32MaxErr = e32
		}
	}

	t.Logf("=== Extended Sequence (2048 tokens) ===")
	t.Logf("  Q16 max error: %.4f%%", q16MaxErr*100)
	t.Logf("  Q32 max error: %.4f%%", q32MaxErr*100)
	if q32MaxErr < q16MaxErr {
		t.Logf("  Q32 improvement: %.1fx", q16MaxErr/q32MaxErr)
	} else {
		t.Logf("  No meaningful difference")
	}
}

// BenchmarkExpLUT64_Eval measures Q32 LUT throughput.
func BenchmarkExpLUT64_Eval(b *testing.B) {
	lut := NewExpLUT64(-8, 8)

	rng := rand.New(rand.NewPCG(42, 0))
	inputs := make([]int32, 1024)
	for i := range inputs {
		x := (rng.Float64() - 0.5) * 16
		inputs[i] = int32(x * q16f)
	}

	b.ResetTimer()
	var sink int64
	for i := range b.N {
		sink = lut.Eval(inputs[i%len(inputs)])
	}
	_ = sink
}
