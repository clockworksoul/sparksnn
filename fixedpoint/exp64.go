package fixedpoint

import "math"

const (
	// Q32 is the fixed-point scaling factor for int64: 1.0 = 2^32 = 4294967296.
	Q32 int64 = 1 << 32

	// q32f is the float64 equivalent for precomputation.
	q32f = float64(Q32)
)

// ExpLUT64 approximates exp(x) using a precomputed lookup table with linear
// interpolation. Inputs are Q16 fixed-point (int32), outputs are Q32
// fixed-point (int64). This "mixed-width" design follows hardware conventions:
// narrow inputs for cache-friendly storage, wide outputs for precision.
//
// Compared to ExpLUT (Q16 output):
//   - exp(-8) has ~1.4M levels instead of 22 → eliminates precision cliff
//   - exp(0) = 4,294,967,296 instead of 65,536 → 65536× more precision
//   - Table is 8 KB instead of 4 KB (1024 × int64), still fits L1 cache
//
// The Q32 output is designed for use with int64 WKV accumulators, where
// the extra precision compounds beneficially over hundreds of timesteps.
type ExpLUT64 struct {
	table    [expTableSize]int64
	xMin     int32 // minimum input in Q16 (same as ExpLUT)
	xMax     int32 // maximum input in Q16
	rangeQ16 int64
}

// NewExpLUT64 creates an ExpLUT64 covering the input range [minVal, maxVal].
//
// For RWKV key values, use NewExpLUT64(-8, 8).
func NewExpLUT64(minVal, maxVal float64) *ExpLUT64 {
	e := &ExpLUT64{
		xMin: int32(minVal * q16f),
		xMax: int32(maxVal * q16f),
	}
	e.rangeQ16 = int64(e.xMax) - int64(e.xMin)

	for i := range expTableSize {
		realX := minVal + (maxVal-minVal)*float64(i)/float64(expTableSize-1)
		e.table[i] = int64(math.Exp(realX) * q32f)
	}

	return e
}

// Eval computes an approximation of exp(x) where x is Q16 fixed-point (int32).
// Returns the result in Q32 fixed-point (int64).
//
// Values outside the LUT range are clamped to the boundary values.
func (e *ExpLUT64) Eval(x int32) int64 {
	if x <= e.xMin {
		return e.table[0]
	}
	if x >= e.xMax {
		return e.table[expTableSize-1]
	}

	offset := int64(x - e.xMin)
	scaledQ16 := offset * int64(expTableSize-1) << 16 / e.rangeQ16

	idx := int(scaledQ16 >> 16)
	frac := scaledQ16 & 0xFFFF

	if idx >= expTableSize-1 {
		return e.table[expTableSize-1]
	}

	lo := e.table[idx]
	hi := e.table[idx+1]

	// Interpolation: lo + (hi-lo)*frac/65536
	// (hi-lo) can be large for int64, but frac is at most 65535,
	// so (hi-lo)*frac fits int64 as long as |hi-lo| < 2^48.
	// max |hi-lo| ≈ exp(8)*Q32/1024 ≈ 12.5B, well under 2^48.
	return lo + (hi-lo)*frac>>16
}

// EvalFloat is a convenience method for testing.
func (e *ExpLUT64) EvalFloat(x float64) float64 {
	q16Input := int32(x * q16f)
	q32Output := e.Eval(q16Input)
	return float64(q32Output) / q32f
}

// ToQ16 converts a Q32 result to Q16 by right-shifting 16 bits.
// Useful when feeding into Q16-based downstream operations.
func ToQ16(q32 int64) int32 {
	return int32(q32 >> 16)
}
