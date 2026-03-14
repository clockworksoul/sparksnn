# MNIST — Surrogate Gradient Results

## Tuned Configuration (Best)
- **Architecture:** 784 → 512 hidden (30% sparse) → 10 output
- **Connections:** ~123,600
- **Training:** Adam + LR decay (halved every 15 epochs), 40 timesteps, early stopped at epoch 42
- **Weight scale (α):** 2^20 (1,048,576)
- **Seed:** 42

### Results
| Metric | Value |
|---|---|
| **Best accuracy** | **97.21%** |
| Final training loss | 0.0005 |
| Training time | ~89 min (Apple M-series CPU, single core) |

### LR Schedule Effect
| LR Phase | Learning Rate | Best Accuracy |
|---|---|---|
| Epochs 1–15 | 0.001 | 96.46% |
| Epochs 16–30 | 0.0005 | 97.07% |
| Epochs 31–42 | 0.00025 | 97.21% |

## Baseline Configuration
- **Architecture:** 784 → 256 hidden (20% sparse) → 10 output
- **Connections:** ~41,500
- **Training:** Adam, LR 0.001, 30 timesteps, 15 epochs

| Variant | Accuracy |
|---|---|
| Baseline (Adam, flat LR) | 95.8% |
| + LR decay | 96.39% |

## Ablations
| Configuration | Connections | Best Accuracy | Training Time |
|---|---|---|---|
| 256 hidden, SGD, 10k subset | 41,500 | 87.7% | ~3 min |
| 256 hidden, Adam, 60k full | 41,500 | 95.8% | ~13 min |
| 256 hidden, Adam + LR decay | 41,500 | 96.39% | ~36 min |
| 512 hidden, Adam + LR decay | 123,600 | **97.21%** | ~89 min |
| 256→128 deep, Adam | 50,840 | 93.7% | ~19 min |

## Activity Rates (Tuned, Full Test Set)
| Metric | Rate |
|---|---|
| Active neurons/timestep | 44.4% |
| Spiking neurons/timestep | 10.8% |
| Idle neurons/timestep | 55.6% |

## Energy Analysis
| | Dense MLP (FP32) | SparkSNN (conservative) | SparkSNN (lazy decay) |
|---|---|---|---|
| Energy | 1,870 nJ | ~316 nJ | ~210 nJ |
| vs. MLP | — | ~5.9× | ~8.9× |

## Context
- Single-layer perceptron: ~92% on MNIST
- Simple two-layer MLP: ~97%
- Our sparse spiking network matches the MLP with 30% connectivity and integer inference
