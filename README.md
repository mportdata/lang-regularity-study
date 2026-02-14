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

## Output

- **Raw data**: `data/raw/<lang>/wiki.txt` - Extracted text corpus
- **Metadata**: `data/raw/<lang>/wiki.txt.meta.json` - Source info, checksums, sizes, timestamps
- **Work directory**: `data/.work/<lang>/` - Reserved for run artifacts

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
