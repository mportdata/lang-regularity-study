from __future__ import annotations

from array import array
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import typer
from tokenizers import Tokenizer

from ..config import TokenizeConfig, load_config
from ..data.paths import resolve_input_corpus_path


def _iter_documents(corpus_path: Path):
    current: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line:
                current.append(line)
                continue
            if current:
                yield "\n".join(current)
                current = []
    if current:
        yield "\n".join(current)


def _resolve_dtype(tokenize_cfg: TokenizeConfig, vocab_size: int) -> tuple[str, str]:
    dtype = tokenize_cfg.dtype
    if dtype == "auto":
        resolved = "uint16" if vocab_size <= 65535 else "uint32"
    else:
        resolved = dtype

    if resolved == "uint16" and vocab_size > 65535:
        raise ValueError(
            "tokenize.dtype=uint16 cannot represent tokenizer vocab size "
            f"{vocab_size}; use uint32 or auto."
        )

    typecode = "H" if resolved == "uint16" else "I"
    return resolved, typecode


def _get_special_token_ids(
    tokenizer: Tokenizer, add_bos: bool, add_eos: bool
) -> tuple[int | None, int | None]:
    bos_id = tokenizer.token_to_id("<bos>") if add_bos else None
    eos_id = tokenizer.token_to_id("<eos>") if add_eos else None

    if add_bos and bos_id is None:
        raise ValueError("Config requested add_bos but tokenizer has no '<bos>' token.")
    if add_eos and eos_id is None:
        raise ValueError("Config requested add_eos but tokenizer has no '<eos>' token.")

    return bos_id, eos_id


def _write_ids(handle, ids: list[int], typecode: str) -> int:
    if not ids:
        return 0
    arr = array(typecode, ids)
    arr.tofile(handle)
    return len(arr)


def _tokenize_language(
    language: str,
    corpus_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    tokenize_cfg: TokenizeConfig,
    force: bool,
) -> None:
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    metadata_path = output_dir / "tokenize.meta.json"

    if train_path.exists() and val_path.exists() and metadata_path.exists() and not force:
        typer.echo(f"Tokenized data already exists for '{language}'. Skipping (use --force).")
        return

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus for '{language}': {corpus_path}. Run fetch first.")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer for '{language}': {tokenizer_path}. Run bpe first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    dtype_name, typecode = _resolve_dtype(tokenize_cfg, vocab_size=vocab_size)
    bos_id, eos_id = _get_special_token_ids(
        tokenizer=tokenizer,
        add_bos=tokenize_cfg.add_bos,
        add_eos=tokenize_cfg.add_eos,
    )

    rng = random.Random(tokenize_cfg.seed)
    docs_total = 0
    docs_train = 0
    docs_val = 0
    tokens_train = 0
    tokens_val = 0

    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for doc in _iter_documents(corpus_path):
            encoded = tokenizer.encode(doc)
            ids = encoded.ids
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
            if not ids:
                continue

            docs_total += 1
            is_val = rng.random() < tokenize_cfg.val_ratio
            if is_val:
                docs_val += 1
                tokens_val += _write_ids(val_f, ids, typecode=typecode)
            else:
                docs_train += 1
                tokens_train += _write_ids(train_f, ids, typecode=typecode)

            if tokenize_cfg.max_tokens is not None and (tokens_train + tokens_val) >= tokenize_cfg.max_tokens:
                break

    if tokens_train == 0:
        raise ValueError(
            f"No train tokens were written for '{language}'. "
            "Increase corpus size, adjust val_ratio, or disable max_tokens cap."
        )

    now = datetime.now(UTC).isoformat()
    metadata = {
        "language_code": language,
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(tokenizer_path),
        "output_dir": str(output_dir),
        "dtype": dtype_name,
        "vocab_size": vocab_size,
        "train_bin_path": str(train_path),
        "val_bin_path": str(val_path),
        "docs_total": docs_total,
        "docs_train": docs_train,
        "docs_val": docs_val,
        "tokens_train": tokens_train,
        "tokens_val": tokens_val,
        "val_ratio": tokenize_cfg.val_ratio,
        "seed": tokenize_cfg.seed,
        "add_bos": tokenize_cfg.add_bos,
        "add_eos": tokenize_cfg.add_eos,
        "max_tokens": tokenize_cfg.max_tokens,
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Wrote tokenized train: {train_path}")
    typer.echo(f"Wrote tokenized val: {val_path}")
    typer.echo(f"Wrote tokenization metadata: {metadata_path}")


def run_tokenize(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)
    if cfg.bpe is None:
        raise ValueError(f"Config '{config_path}' is missing required 'bpe' section.")
    if cfg.tokenize is None:
        raise ValueError(f"Config '{config_path}' is missing required 'tokenize' section.")

    effective_force = force or cfg.force
    for language in cfg.languages:
        corpus_path = resolve_input_corpus_path(cfg.output_dir / language, cfg.max_size_mb)
        tokenizer_path = cfg.bpe.output_root / cfg.bpe.experiment_name / language / "tokenizer.json"
        output_dir = cfg.tokenize.output_root / cfg.tokenize.experiment_name / language
        _tokenize_language(
            language=language,
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            tokenize_cfg=cfg.tokenize,
            force=effective_force,
        )
