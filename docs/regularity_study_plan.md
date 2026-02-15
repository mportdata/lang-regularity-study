# Language Regularity Study Plan

## Goal

Measure how language regularity affects:
- BPE behavior (tokenization efficiency and structure)
- downstream language model learning under fixed budgets

## Core Principle

Hold all training/tokenization settings fixed across languages.
Only language should vary within each experiment row.

## Experiment Matrix (Latin Languages)

Languages:
- `en`, `eo`, `fi`, `fr`, `tr`

Controlled axes:
- `data_budget_mb`: e.g. `50`, `500`, `1000`
- `bpe_vocab_size`: e.g. `16000`, `24000`, `32000`
- `model_scale`: e.g. `small`, `large`, `xlarge`
- seed: fixed (`42`) for the primary pass, optional multi-seed follow-up

Suggested matrix:
1. `latin_tight` baseline (smaller budgets)
2. `latin_large`
3. `latin_xlarge`

## Primary Outcome Metrics

Tokenizer-level:
- `tokens_per_byte` = `(tokens_train + tokens_val) / corpus_bytes`
- `tokens_per_doc` = `(tokens_train + tokens_val) / docs_total`

Model-level:
- `val_loss`
- `val_ppl`

Stability:
- optional multi-seed mean/std

## Required Metadata per Run

For every (experiment, language):
- corpus path + corpus bytes + corpus checksum
- tokenizer config + vocab size
- tokenize stats (docs, tokens, dtype)
- train config + final metrics
- eval summary row

## Analysis Outputs

Use `scripts/build_results_table.py` to generate:
- `analysis/results_table.csv` (one row per experiment+language)

Then run statistical analysis on this table:
- correlations between regularity proxies and `tokens_per_byte`, `val_loss`
- plots by language and experiment

## Notes on Re-Training

Re-training is needed when comparing new settings (data budget, vocab size, model size, etc.).
If you only aggregate/report already completed runs, no retraining is needed.

