package fixedpoint

import "math"

const (
	// sigmoidTableSize is the number of entries in the sigmoid lookup table.
	// 256 entries at 4 bytes each = 1 KB. Sigmoid is very smooth, so fewer
	// entries are needed compared to exp (which needs 1024).
	sigmoidTableSize = 256
)

// SigmoidLUT approximates σ(x) = 1/(1+exp(-x)) using a precomputed lookup
// table with linear interpolation. All inputs and outputs are Q16 fixed-point.
//
// Error characteristics (input range [-8, 8]):
//   - Point-wise max relative error: <0.05% (predicted)
//   - Output range: [0, 65536] in Q16 (representing [0.0, 1.0])
//
// Sigmoid is much easier than exp because:
//   - Bounded output [0, 1] → full Q16 range utilized efficiently
//   - Very smooth (small second derivative) → interpolation is accurate
//   - No precision cliff at extremes (near-0 and near-1 are fine for gating)
//
// Memory: 1 KB (256 × int32). Fits in L1 cache alongside the ExpLUT.
type SigmoidLUT struct {
	table    [sigmoidTableSize]int32
	xMin     int32 // minimum input in Q16
	xMax     int32 // maximum input in Q16
	rangeQ16 int64 // xMax - xMin as int64
}

// NewSigmoidLUT creates a SigmoidLUT covering the input range [minVal, maxVal],
// where minVal and maxVal are real-valued bounds.
//
// For RWKV receptance gating, use NewSigmoidLUT(-8, 8). Values outside the
// range are clamped: σ(x) ≈ 0 for x << 0, σ(x) ≈ 1 for x >> 0.
func NewSigmoidLUT(minVal, maxVal float64) *SigmoidLUT {
	s := &SigmoidLUT{
		xMin: int32(minVal * q16f),
		xMax: int32(maxVal * q16f),
	}
	s.rangeQ16 = int64(s.xMax) - int64(s.xMin)

	for i := range sigmoidTableSize {
		realX := minVal + (maxVal-minVal)*float64(i)/float64(sigmoidTableSize-1)
		sigVal := 1.0 / (1.0 + math.Exp(-realX))
		s.table[i] = int32(sigVal * q16f)
	}

	return s
}

// Eval computes an approximation of σ(x) where x is in Q16 fixed-point.
// Returns the result in Q16 fixed-point, in the range [0, 65536].
//
// Values outside the LUT range are clamped: returns ~0 for very negative x,
// returns ~65536 for very positive x.
func (s *SigmoidLUT) Eval(x int32) int32 {
	if x <= s.xMin {
		return s.table[0]
	}
	if x >= s.xMax {
		return s.table[sigmoidTableSize-1]
	}

	offset := int64(x - s.xMin)
	scaledQ16 := offset * int64(sigmoidTableSize-1) << 16 / s.rangeQ16

	idx := int(scaledQ16 >> 16)
	frac := scaledQ16 & 0xFFFF

	if idx >= sigmoidTableSize-1 {
		return s.table[sigmoidTableSize-1]
	}

	lo := int64(s.table[idx])
	hi := int64(s.table[idx+1])

	return int32(lo + (hi-lo)*frac>>16)
}

// EvalFloat is a convenience method that takes a real-valued float64 input
// and returns a real-valued float64 output. Useful for testing.
func (s *SigmoidLUT) EvalFloat(x float64) float64 {
	q16Input := int32(x * q16f)
	q16Output := s.Eval(q16Input)
	return float64(q16Output) / q16f
}
