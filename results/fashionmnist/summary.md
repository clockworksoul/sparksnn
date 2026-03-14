# Fashion-MNIST — Surrogate Gradient Results

## Configuration
- **Architecture:** 784 → 512 hidden (30% sparse) → 10 output
- **Connections:** ~123,600
- **Training:** Adam + LR decay (halved every 15 epochs), 40 timesteps, max 60 epochs, patience 10
- **Weight scale (α):** 2^20 (1,048,576)
- **Seed:** 42
- **Same hyperparameters as MNIST tuned config** (no task-specific tuning)

## Results
*Run in progress — started 2026-03-14 ~11:13 ET*

## Classes
| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Context
- Simple dense MLP: ~88% on Fashion-MNIST
- State-of-the-art single hidden layer: ~89-90%
- Expected SNN result: 85-88% (harder than MNIST, same architecture)
