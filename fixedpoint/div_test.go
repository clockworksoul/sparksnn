package fixedpoint

import (
	"math"
	"testing"
)

// TestDiv_KnownValues checks basic division correctness.
func TestDiv_KnownValues(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int32
		expected float64
		tol      float64
	}{
		{"1/1", Q16, Q16, 1.0, 0.001},
		{"2/1", 2 * Q16, Q16, 2.0, 0.001},
		{"1/2", Q16, 2 * Q16, 0.5, 0.001},
		{"3/4", 3 * Q16, 4 * Q16, 0.75, 0.001},
		{"1/3", Q16, 3 * Q16, 1.0 / 3.0, 0.001},
		{"10/7", 10 * Q16, 7 * Q16, 10.0 / 7.0, 0.001},
		{"-1/2", -Q16, 2 * Q16, -0.5, 0.001},
		{"-3/-4", -3 * Q16, -4 * Q16, 0.75, 0.001},
		{"1/-3", Q16, -3 * Q16, -1.0 / 3.0, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Div(tt.a, tt.b)
			got := float64(result) / q16f
			if math.Abs(got-tt.expected) > tt.tol {
				t.Errorf("Div(%d, %d) = %.6f, want %.6f", tt.a, tt.b, got, tt.expected)
			}
		})
	}
}

// TestDiv_Precision verifies that Q16 division truncation error is bounded.
func TestDiv_Precision(t *testing.T) {
	// The worst-case truncation error for Q16 division is ±1 in the result,
	// which is 1/65536 ≈ 0.0015% relative error for values near 1.0.
	var maxRelErr float64
	count := 0

	for a := int32(1); a <= 100; a++ {
		for b := int32(1); b <= 100; b++ {
			aQ16 := a * Q16
			bQ16 := b * Q16
			expected := float64(a) / float64(b)
			got := float64(Div(aQ16, bQ16)) / q16f

			if expected == 0 {
				continue
			}
			count++

			relErr := math.Abs(got-expected) / expected
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	t.Logf("Division precision over %d pairs:", count)
	t.Logf("  Max relative error: %.6f%%", maxRelErr*100)

	// Q16 truncation gives at most 1/65536 relative error for values >= 1.
	// For smaller quotients the relative error is slightly larger.
	if maxRelErr > 0.002 { // 0.2%
		t.Errorf("max relative error %.6f%% exceeds 0.2%% threshold", maxRelErr*100)
	}
}

// TestDiv64_WKVSimulation simulates the WKV accumulator with int64 and
// verifies division results match float64 reference.
func TestDiv64_WKVSimulation(t *testing.T) {
	// Simulate: accumulator builds up exp(k)*v over multiple tokens,
	// then we divide num/den.
	var numF, denF float64
	var numI, denI int64

	decay := 0.95
	decayQ16 := int64(decay * q16f)

	// Feed 100 tokens with random-ish k and v values.
	for i := range 100 {
		// k in [-4, 4], v in [-1, 1]
		kReal := -4.0 + 8.0*float64(i)/99.0
		vReal := -1.0 + 2.0*float64(i%7)/6.0

		expK := math.Exp(kReal)

		// Float reference
		numF = decay*numF + expK*vReal
		denF = decay*denF + expK

		// Integer simulation
		expKQ16 := int64(expK * q16f)
		vQ16 := int64(vReal * q16f)

		numI = (decayQ16*numI>>16 + expKQ16*vQ16>>16)
		denI = (decayQ16*denI>>16 + expKQ16)
	}

	// Compute wkv both ways.
	wkvFloat := numF / denF
	wkvInt := float64(Div64(numI, denI)) / q16f

	relErr := math.Abs(wkvInt-wkvFloat) / math.Abs(wkvFloat)

	t.Logf("WKV simulation (100 tokens):")
	t.Logf("  Float reference:  %.6f", wkvFloat)
	t.Logf("  Integer Div64:    %.6f", wkvInt)
	t.Logf("  Relative error:   %.4f%%", relErr*100)

	if relErr > 0.01 { // 1% — generous, accounts for accumulated Q16 rounding
		t.Errorf("WKV relative error %.4f%% exceeds 1%% threshold", relErr*100)
	}
}

// TestDiv_SignPreservation verifies correct sign handling.
func TestDiv_SignPreservation(t *testing.T) {
	pos := int32(3 * Q16)
	neg := int32(-3 * Q16)
	den := int32(2 * Q16)

	if Div(pos, den) <= 0 {
		t.Error("positive/positive should be positive")
	}
	if Div(neg, den) >= 0 {
		t.Error("negative/positive should be negative")
	}
	if Div(pos, -den) >= 0 {
		t.Error("positive/negative should be negative")
	}
	if Div(neg, -den) <= 0 {
		t.Error("negative/negative should be positive")
	}
}

// BenchmarkDiv measures Q16 division throughput.
func BenchmarkDiv(b *testing.B) {
	a := int32(42 * Q16)
	d := int32(7 * Q16)
	var sink int32
	for range b.N {
		sink = Div(a, d)
	}
	_ = sink
}

// BenchmarkDiv64 measures int64 division throughput.
func BenchmarkDiv64(b *testing.B) {
	a := int64(42 * int64(Q16))
	d := int64(7 * int64(Q16))
	var sink int32
	for range b.N {
		sink = Div64(a, d)
	}
	_ = sink
}
