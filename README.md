# lang-regularity-study

Study of language regularity using Wikipedia text corpora for controlled cross-language experiments.

## Overview

This project streams Wikipedia text from Hugging Face datasets, normalizes whitespace, and builds fixed-size corpora per language for BPE and small-model comparison experiments.

The project follows a hybrid artifact layout:
- Canonical reusable artifacts live in `data/`.
- Experiment-specific model/eval outputs live in `runs/`.

## Installation

```bash
# Base pipeline (fetch/bpe/tokenize)
uv pip install -e .

# Include training dependency
uv pip install -e ".[train]"
```

## Cross-Architecture Runtime (M1 + GTX 3050)

Training device is controlled by `train.device` in config:
- `auto`: prefer `cuda`, then `mps`, else `cpu`
- `cuda`, `mps`, `cpu`: explicit target with safe fallback to `cpu` if unavailable

This allows one config to run on both Apple Silicon and NVIDIA machines.

## Usage

### Fetch all experiment languages

```bash
make fetch
# or
python -m lang_regularity fetch --config configs/latin_tight.yaml
```

### Convenience alias

```bash
make fetch-all
# or
python -m lang_regularity fetch-all --config configs/latin_tight.yaml
```

### Force re-download

```bash
python -m lang_regularity fetch --config configs/latin_tight.yaml --force
```

### Train BPE tokenizers

```bash
make bpe
# or
python -m lang_regularity bpe --config configs/latin_tight.yaml
```

### Tokenize corpora for model training

```bash
make tokenize
# or
python -m lang_regularity tokenize --config configs/latin_tight.yaml
```

### Train small language models

```bash
make train
# or
python -m lang_regularity train --config configs/latin_tight.yaml
```

### Evaluate and compare languages

```bash
make eval
# or
python -m lang_regularity eval --config configs/latin_tight.yaml
```

### Run full data pipeline (through tokenize)

```bash
make pipeline
# force all stages
make pipeline FORCE=1
# or
python -m lang_regularity pipeline --config configs/latin_tight.yaml --force
```

## Output

- **Raw data**: `data/raw/<lang>/wiki.txt` - Extracted text corpus
- **Metadata**: `data/raw/<lang>/wiki.txt.meta.json` - Source info, checksums, sizes, timestamps
- **Work directory**: `data/.work/<lang>/` - Reserved for run artifacts
- **BPE artifacts**: `data/tokenizers/<experiment>/<lang>/` - `tokenizer.json`, vocab/merges, metadata
- **Encoded tokens**: `data/encoded/<experiment>/<lang>/` - `train.bin`, `val.bin`, tokenization metadata
- **Training outputs**: `runs/<experiment>/<lang>/` - `model.pt`, `train.log`, `metrics.json`, config snapshot
- **Eval summary**: `runs/<experiment>/eval/summary.json` - cross-language ranking and aggregate stats

## Configuration

Edit the experiment config file in `configs/` to customize:
- `languages`: List of language codes to fetch (en, eo, fi, fr, tr, etc.)
- `hf_dataset`: Hugging Face dataset id (default: `wikimedia/wikipedia`)
- `hf_date`: Snapshot date to pin reproducible content (example: `20231101`)
- `hf_split`: Dataset split (typically `train`)
- `hf_text_field`: Field containing article text (typically `text`)
- `output_dir`: Where to write extracted text
- `work_dir`: Where to store intermediate files
- `max_size_mb`: Shared maximum output text size in MB per language
- `force`: Whether to rebuild even if output already exists
- `bpe`: Tokenizer training settings
- `tokenize`: Text-to-token-id encoding settings
- `train`: Small GPT-style training settings
- `eval`: Evaluation summary output settings

`bpe` settings:
- `experiment_name`: Namespace for tokenizer outputs
- `output_root`: Root directory for BPE artifacts
- `vocab_size`: BPE vocabulary size
- `min_frequency`: Minimum pair frequency to merge
- `limit_alphabet`: Max initial alphabet size
- `train_sample_mb`: Training text sample size per language
- `shuffle_seed`: Deterministic shuffle seed for sampled training texts
- `special_tokens`: Reserved tokens included in vocab
- `unk_token`: Unknown-token symbol (must be in `special_tokens`)

`tokenize` settings:
- `experiment_name`: Namespace for encoded outputs
- `output_root`: Root directory for encoded data artifacts
- `val_ratio`: Fraction of documents routed to validation split
- `seed`: Deterministic split seed
- `add_bos`: Whether to prepend `<bos>` token id per document
- `add_eos`: Whether to append `<eos>` token id per document
- `dtype`: Output id dtype (`auto`, `uint16`, `uint32`)
- `max_tokens`: Optional cap on total written tokens per language

`train` settings:
- `experiment_name`: Namespace under `runs/`
- `output_root`: Root directory for training outputs
- `device`: `auto|cuda|mps|cpu`
- `block_size`: Context length in tokens
- `batch_size`: Batch size per step
- `max_steps`: Total optimization steps
- `eval_interval`: Steps between validation checks
- `eval_batches`: Batches used per evaluation pass
- `learning_rate`: AdamW learning rate
- `weight_decay`: AdamW weight decay
- `grad_clip`: Gradient clipping max norm
- `n_embd`, `n_head`, `n_layer`, `dropout`: Transformer shape
- `seed`: Random seed for reproducibility

`eval` settings:
- `output_subdir`: Subdirectory under `runs/<train_experiment>/`

Skip/overwrite behavior:
- `fetch` skips languages with existing `wiki.txt` + `.meta.json` unless `--force` or `force: true`
- `bpe` skips languages when tokenizer artifacts exist and both corpus checksum and BPE config hash match
- `bpe --force` always retrains and overwrites tokenizer artifacts
- `tokenize` skips languages with existing `train.bin`, `val.bin`, and metadata unless `--force`
- `train` skips languages with existing `model.pt` + `metrics.json` unless `--force`
- `eval` skips existing summary output unless `--force`

## Planned PR Sequence

- `PR2` complete: tokenize stage (`data/encoded/...`)
- `PR3`: train stage with cross-architecture device fallback (`cuda`/`mps`/`cpu`)
- `PR4`: evaluation summary/comparison stage (`runs/<exp>/eval/summary.json`)
- `PR5`: full experiment command (`fetch -> bpe -> tokenize -> train -> eval`)

Detailed plan: `docs/pr_plan.md`.
