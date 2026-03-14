# Iris — Surrogate Gradient Results

## Configuration
- **Architecture:** 40 input (population coded, 10 per feature) → 20 hidden → 3 output
- **Connectivity:** Full
- **Connections:** ~940
- **Training:** 300 epochs, SGD, LR 0.001, 40 timesteps
- **Weight scale (α):** 2^16 (65,536)
- **Seed:** 42

## Results
| Domain | Accuracy | Mismatches |
|---|---|---|
| Float64 | 100% (30/30) | — |
| Int32 | 100% (30/30) | 0/30 |

- **Quantization degradation:** Zero (perfect parity)
- **Training time:** ~1.9 seconds (Apple M-series CPU, single core)

## Key Finding
Every test sample produces identical predictions through both the float64 and int32 inference paths, demonstrating zero quantization degradation.
