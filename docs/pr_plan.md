# PR Plan

## Architecture Constraint

All model training/evaluation changes must run on both:
- Apple Silicon (`mps`)
- NVIDIA GPU hosts (`cuda`, tested target: GTX 3050 4GB)

Required behavior:
- `train.device=auto` picks `cuda -> mps -> cpu`.
- Explicit `cuda`/`mps` gracefully falls back to `cpu` if unavailable.
- Non-training commands (`fetch`, `bpe`, `tokenize`) must work without PyTorch installed.
- Default model/training settings remain small enough for 4GB VRAM environments.

## Sequence

1. PR3: Train Stage
- Add config section `train`.
- Add `train` CLI command.
- Implement tiny GPT training from `data/encoded/...` and write outputs to `runs/<exp>/<lang>/`.

2. PR4: Eval Stage
- Add config section `eval`.
- Add `eval` CLI command.
- Aggregate per-language training metrics into `runs/<exp>/eval/summary.json`.

3. PR5: Full Experiment Orchestration
- Add `experiment` CLI command chaining:
  `fetch -> bpe -> tokenize -> train -> eval`.
- Preserve hybrid artifact contract:
  canonical datasets/tokenizers/encodings in `data/`,
  experiment-specific outputs in `runs/`.

