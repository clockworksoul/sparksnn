package fixedpoint

// Div computes a/b in Q16 fixed-point arithmetic, where both a and b are
// Q16 values (real_value * 65536). Returns the quotient in Q16.
//
// Uses int64 intermediate to prevent overflow during the shift.
// Division is exact within ±1 integer truncation (~0.0015% relative error).
//
// Panics if b is zero. For RWKV's WKV normalization, b (the denominator)
// is a sum of exp(k) values and is always positive after the first token.
func Div(a, b int32) int32 {
	return int32((int64(a) << 16) / int64(b))
}

// Div64 computes a/b where both a and b are int64, returning a Q16 int32.
//
// This is the primary division function for RWKV's WKV normalization, where
// the numerator and denominator accumulators must be int64 to avoid overflow:
//
//	num[t] = exp(-w) * num[t-1] + exp(k[t]) * v[t]   // can exceed int32 max
//	den[t] = exp(-w) * den[t-1] + exp(k[t])           // can exceed int32 max
//	wkv[t] = Div64(num[t], den[t])                     // result fits Q16 int32
//
// The result fits int32 because wkv is a weighted average of v values,
// which are themselves bounded Q16 values.
//
// Panics if b is zero.
func Div64(a, b int64) int32 {
	return int32((a << 16) / b)
}
