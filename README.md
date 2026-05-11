# 2048 — N-tuple TD-afterstate agent

A reinforcement-learning agent that plays the game **2048** using an **N-tuple network** trained with **TD-afterstate learning** (Szubert & Jaśkowski). This is the established SOTA family for 2048: a linear function approximator over patterns of board cells indexed by their log2 values, updated by on-policy temporal-difference learning on afterstates.

No GPU. The hot path is millions of small lookups, which the CPU + Numba JIT handles much faster than a GPU.

## Demo

<video src="https://github.com/Aaronc123814/2048/releases/download/v0.1/gameplay.mp4" controls width="480"></video>

If the embed doesn't render in your viewer, [download or watch the clip directly](https://github.com/Aaronc123814/2048/releases/download/v0.1/gameplay.mp4).

## Results

Trained for ~400k self-play episodes on a single CPU.

| Episodes | ≥ 2048 | ≥ 4096 | ≥ 8192 | Avg score |
|---------:|-------:|-------:|-------:|----------:|
| 50,000   |  85.0% |  44.5% |   0.5% |    47,899 |
| 100,000  |  90.5% |  69.0% |  10.5% |    66,758 |
| 200,000  |  91.5% |  82.0% |  44.0% |    97,383 |
| 400,000  |  92.5% |  83.5% |  57.0% |   111,278 |
| 700,000  |  98.5% |  93.5% |  71.5% |   131,263 |

Evaluations are 200 greedy games (no learning) at the listed episode count. Full per-episode training history is in [`training_log.csv`](./training_log.csv).

## Method

- **Representation.** Each cell holds the log2 of its tile (0 for empty), so the whole 4×4 board fits in 16 nybbles.
- **Engine.** Every 16-bit packed row is precomputed for the move-left result and score gained, giving O(1) moves. Wrapped in Numba-`njit` functions for the rest.
- **N-tuple network.** Four base 6-tuples (an L-shape, a shifted L, a 3×2 corner block, a diagonal zigzag), each closed under the D4 symmetry group (4 rotations × {identity, hflip}) → 32 features sharing 4 weight tables via parameter tying. Each table has `16^6 ≈ 16.8M` entries → 256 MB of weights total.
- **Learning rule.** 1-ply greedy action selection over `r + V(afterstate)`, with on-line TD-afterstate updates:
  ```
  target ← r' + V(s'')
  V(s') ← V(s') + α · (target − V(s')) / N_FEATURES
  ```
  α = 0.1, spread per-weight by `N_FEATURES` so the effective learning rate stays at α.
- **Checkpointing.** Weights saved every 10k episodes; periodic 200-game greedy evaluations with learning disabled.

## Files

| File | Description |
| --- | --- |
| [`2048_ntuple_td.ipynb`](./2048_ntuple_td.ipynb) | Full implementation: engine, features, training loop, evaluation |
| [`training_log.csv`](./training_log.csv) | Per-200-episode log: max tile, score, rolling 2048/4096 rates, ep/s |
| `ntuple_weights.npy` (release asset) | Trained N-tuple weights — `(4, 16^6)` `float32`, ~256 MB |
| `gameplay.mp4` (release asset) | Recorded gameplay from the trained checkpoint |

## Loading the trained weights

The weights are too large for git (256 MB), so they ship as a release asset. Download from [release v0.1](https://github.com/Aaronc123814/2048/releases/tag/v0.1) or with `curl`:

```bash
curl -LO https://github.com/Aaronc123814/2048/releases/download/v0.1/ntuple_weights.npy
```

Then resume training, or run pure greedy evaluation, by dropping the file next to the notebook — the training loop already auto-resumes from `ntuple_weights.npy` when it exists. For inference only:

```python
import numpy as np
weights = np.load("ntuple_weights.npy")          # shape (4, 16777216), float32
# pass weights into select_best / evaluate_episode from the notebook
```

## References

- Szubert, M., & Jaśkowski, W. (2014). *Temporal Difference Learning of N-Tuple Networks for the Game 2048.* IEEE CIG.
- Wu, I.-C. et al. (2014). *Multi-Stage Temporal Difference Learning for 2048.*
