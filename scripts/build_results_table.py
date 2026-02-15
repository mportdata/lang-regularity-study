#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read_json(path)


def build_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(runs_root.glob("*/*/metrics.json")):
        metrics = _read_json(metrics_path)
        experiment = metrics_path.parent.parent.name
        language = metrics.get("language_code")

        train_bin_path = Path(str(metrics.get("train_bin_path")))
        val_bin_path = Path(str(metrics.get("val_bin_path")))
        tokenize_meta_path = train_bin_path.parent / "tokenize.meta.json"
        tokenize_meta = _maybe_read_json(tokenize_meta_path) or {}

        tokenizer_path_raw = metrics.get("tokenizer_path") or tokenize_meta.get("tokenizer_path")
        bpe_meta = None
        corpus_meta = None
        corpus_bytes = None
        tokens_total = None
        tokens_per_byte = None
        if isinstance(tokenizer_path_raw, str) and tokenizer_path_raw:
            tokenizer_path = Path(tokenizer_path_raw)
            bpe_meta = _maybe_read_json(tokenizer_path.parent / "bpe.meta.json") or {}
            corpus_path_raw = bpe_meta.get("input_corpus_path") or tokenize_meta.get("corpus_path")
            if isinstance(corpus_path_raw, str):
                corpus_path = Path(corpus_path_raw)
                if corpus_path.exists():
                    corpus_bytes = corpus_path.stat().st_size
                corpus_meta = _maybe_read_json(corpus_path.with_suffix(corpus_path.suffix + ".meta.json"))

        tokens_train = int(tokenize_meta.get("tokens_train", 0) or 0)
        tokens_val = int(tokenize_meta.get("tokens_val", 0) or 0)
        if tokens_train or tokens_val:
            tokens_total = tokens_train + tokens_val
        if tokens_total is not None and corpus_bytes:
            tokens_per_byte = tokens_total / corpus_bytes

        row = {
            "experiment": experiment,
            "language": language,
            "device_used": metrics.get("device_used"),
            "max_steps": metrics.get("max_steps"),
            "batch_size": metrics.get("batch_size"),
            "block_size": metrics.get("block_size"),
            "n_embd": metrics.get("n_embd"),
            "n_head": metrics.get("n_head"),
            "n_layer": metrics.get("n_layer"),
            "learning_rate": metrics.get("learning_rate"),
            "train_loss": metrics.get("train_loss"),
            "val_loss": metrics.get("val_loss"),
            "val_ppl": metrics.get("val_ppl"),
            "tokenizer_path": tokenizer_path_raw,
            "bpe_vocab_size": (bpe_meta or {}).get("bpe_config", {}).get("vocab_size"),
            "bpe_train_sample_mb": (bpe_meta or {}).get("bpe_config", {}).get("train_sample_mb"),
            "corpus_path": (bpe_meta or {}).get("input_corpus_path") or tokenize_meta.get("corpus_path"),
            "corpus_bytes": corpus_bytes,
            "max_size_mb": (corpus_meta or {}).get("max_size_mb"),
            "docs_total": tokenize_meta.get("docs_total"),
            "tokens_train": tokens_train,
            "tokens_val": tokens_val,
            "tokens_total": tokens_total,
            "tokens_per_byte": tokens_per_byte,
            "updated_at_utc": metrics.get("updated_at_utc"),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "language",
        "device_used",
        "max_steps",
        "batch_size",
        "block_size",
        "n_embd",
        "n_head",
        "n_layer",
        "learning_rate",
        "train_loss",
        "val_loss",
        "val_ppl",
        "tokenizer_path",
        "bpe_vocab_size",
        "bpe_train_sample_mb",
        "corpus_path",
        "corpus_bytes",
        "max_size_mb",
        "docs_total",
        "tokens_train",
        "tokens_val",
        "tokens_total",
        "tokens_per_byte",
        "updated_at_utc",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a normalized analysis table from run/tokenizer/tokenization artifacts."
    )
    parser.add_argument("--runs-root", default="runs", type=Path)
    parser.add_argument("--out", default=Path("analysis/results_table.csv"), type=Path)
    args = parser.parse_args()

    rows = build_rows(args.runs_root)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

