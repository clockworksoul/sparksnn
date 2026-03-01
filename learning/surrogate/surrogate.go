// Package surrogate implements surrogate gradient training for spiking
// neural networks. During the forward pass, neurons spike normally
// (Heaviside step function). During the backward pass, the
// non-differentiable spike is replaced with a smooth surrogate whose
// gradient can flow through the network.
//
// Reference: Neftci et al. (2019), "Surrogate Gradient Learning in
// Spiking Neural Networks." arXiv:1901.09948
package surrogate

import "math"

// Gradient computes the surrogate derivative for backpropagation.
// The forward pass uses the true Heaviside; the backward pass uses
// a smooth approximation.
type Gradient interface {
	// Derivative returns the surrogate ∂S̃/∂U evaluated at the given
	// membrane potential and threshold. This replaces the Dirac delta
	// (derivative of the Heaviside) with a smooth function that
	// allows gradient flow.
	Derivative(u, threshold float64) float64
}

// FastSigmoid implements the fast sigmoid surrogate gradient:
//
//	∂S̃/∂U = slope / (1 + |slope * (U - threshold)|)²
//
// This is the most common surrogate in the literature. The slope
// parameter controls sharpness: higher = closer to true Heaviside
// but harder to train. Typical value: 25.
type FastSigmoid struct {
	Slope float64
}

// DefaultFastSigmoid returns a fast sigmoid with slope=25.
func DefaultFastSigmoid() *FastSigmoid {
	return &FastSigmoid{Slope: 25.0}
}

func (fs *FastSigmoid) Derivative(u, threshold float64) float64 {
	x := fs.Slope * (u - threshold)
	denom := 1.0 + math.Abs(x)
	return fs.Slope / (denom * denom)
}

// Boxcar implements the simplest possible surrogate gradient:
// a rectangular window around the threshold.
//
//	∂S̃/∂U = 1/width if |U - threshold| < width/2, else 0
type Boxcar struct {
	Width float64
}

func (b *Boxcar) Derivative(u, threshold float64) float64 {
	if math.Abs(u-threshold) < b.Width/2.0 {
		return 1.0 / b.Width
	}
	return 0.0
}
