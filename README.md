# lang-regularity-study

Study of language regularity using Wikipedia text corpora for controlled cross-language experiments.

## Overview

This project streams Wikipedia text from Hugging Face datasets, normalizes whitespace, and builds fixed-size corpora per language for BPE and small-model comparison experiments.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

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

### Run full data + BPE pipeline

```bash
make pipeline
```

## Output

- **Raw data**: `data/raw/<lang>/wiki.txt` - Extracted text corpus
- **Metadata**: `data/raw/<lang>/wiki.txt.meta.json` - Source info, checksums, sizes, timestamps
- **Work directory**: `data/.work/<lang>/` - Reserved for run artifacts
- **BPE artifacts**: `data/processed/bpe/<experiment>/<lang>/` - `tokenizer.json`, vocab/merges, metadata

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

Skip/overwrite behavior:
- `fetch` skips languages with existing `wiki.txt` + `.meta.json` unless `--force` or `force: true`
- `bpe` skips languages when tokenizer artifacts exist and both corpus checksum and BPE config hash match
- `bpe --force` always retrains and overwrites tokenizer artifacts
