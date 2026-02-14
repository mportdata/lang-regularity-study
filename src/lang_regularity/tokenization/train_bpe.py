from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import typer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from ..config import BPEConfig, load_config


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_config_hash(language: str, bpe_cfg: BPEConfig) -> str:
    payload = {
        "language": language,
        "experiment_name": bpe_cfg.experiment_name,
        "vocab_size": bpe_cfg.vocab_size,
        "min_frequency": bpe_cfg.min_frequency,
        "limit_alphabet": bpe_cfg.limit_alphabet,
        "train_sample_mb": bpe_cfg.train_sample_mb,
        "shuffle_seed": bpe_cfg.shuffle_seed,
        "special_tokens": bpe_cfg.special_tokens,
        "unk_token": bpe_cfg.unk_token,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _bpe_config_snapshot(bpe_cfg: BPEConfig) -> dict[str, object]:
    return {
        "experiment_name": bpe_cfg.experiment_name,
        "vocab_size": bpe_cfg.vocab_size,
        "min_frequency": bpe_cfg.min_frequency,
        "limit_alphabet": bpe_cfg.limit_alphabet,
        "train_sample_mb": bpe_cfg.train_sample_mb,
        "shuffle_seed": bpe_cfg.shuffle_seed,
        "special_tokens": bpe_cfg.special_tokens,
        "unk_token": bpe_cfg.unk_token,
    }


def _collect_training_texts(corpus_path: Path, sample_mb: float, seed: int) -> list[str]:
    target_bytes = int(sample_mb * 1024 * 1024)
    texts: list[str] = []
    current_doc: list[str] = []
    bytes_collected = 0

    with corpus_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line:
                current_doc.append(line)
                continue

            if not current_doc:
                continue

            doc = "\n".join(current_doc)
            doc_bytes = len(doc.encode("utf-8"))
            if bytes_collected + doc_bytes > target_bytes and texts:
                break

            texts.append(doc)
            bytes_collected += doc_bytes
            current_doc = []

            if bytes_collected >= target_bytes:
                break

    if current_doc and (bytes_collected < target_bytes or not texts):
        doc = "\n".join(current_doc)
        texts.append(doc)

    if not texts:
        raise ValueError(f"No training text found in corpus: {corpus_path}")

    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts


def _train_tokenizer(texts: Iterable[str], bpe_cfg: BPEConfig) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=bpe_cfg.unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=bpe_cfg.vocab_size,
        min_frequency=bpe_cfg.min_frequency,
        special_tokens=bpe_cfg.special_tokens,
        limit_alphabet=bpe_cfg.limit_alphabet,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def _should_skip(
    metadata_path: Path,
    tokenizer_json_path: Path,
    expected_config_hash: str,
    expected_bpe_config: dict[str, object],
    input_sha256: str,
    force: bool,
) -> bool:
    if force:
        return False
    if not metadata_path.exists() or not tokenizer_json_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if metadata.get("input_text_sha256") != input_sha256:
        return False
    if metadata.get("config_hash") == expected_config_hash:
        return True
    return metadata.get("bpe_config") == expected_bpe_config


def _train_for_language(
    input_root: Path, language: str, bpe_cfg: BPEConfig, force: bool
) -> None:
    input_corpus_path = input_root / language / "wiki.txt"
    if not input_corpus_path.exists():
        raise FileNotFoundError(
            f"Missing corpus for language '{language}': {input_corpus_path}. Run fetch first."
        )

    output_dir = bpe_cfg.output_root / bpe_cfg.experiment_name / language
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_json_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "bpe.meta.json"

    input_sha256 = _sha256_for_file(input_corpus_path)
    config_hash = _stable_config_hash(language=language, bpe_cfg=bpe_cfg)
    bpe_config_snapshot = _bpe_config_snapshot(bpe_cfg=bpe_cfg)

    if _should_skip(
        metadata_path=metadata_path,
        tokenizer_json_path=tokenizer_json_path,
        expected_config_hash=config_hash,
        expected_bpe_config=bpe_config_snapshot,
        input_sha256=input_sha256,
        force=force,
    ):
        typer.echo(f"BPE already up to date for '{language}'. Skipping (use --force to retrain).")
        return

    typer.echo(f"Training BPE for language '{language}'...")
    texts = _collect_training_texts(
        corpus_path=input_corpus_path,
        sample_mb=bpe_cfg.train_sample_mb,
        seed=bpe_cfg.shuffle_seed,
    )
    tokenizer = _train_tokenizer(texts=texts, bpe_cfg=bpe_cfg)

    tokenizer.save(str(tokenizer_json_path))
    saved_files = tokenizer.model.save(str(output_dir), prefix="bpe")

    now = datetime.now(UTC).isoformat()
    metadata = {
        "language_code": language,
        "input_corpus_path": str(input_corpus_path),
        "input_text_sha256": input_sha256,
        "input_text_size_bytes": input_corpus_path.stat().st_size,
        "output_dir": str(output_dir),
        "tokenizer_json_path": str(tokenizer_json_path),
        "model_files": [str(output_dir / p) for p in saved_files],
        "config_hash": config_hash,
        "bpe_config": bpe_config_snapshot,
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Wrote tokenizer: {tokenizer_json_path}")
    typer.echo(f"Wrote metadata: {metadata_path}")


def run_bpe(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)
    if cfg.bpe is None:
        raise ValueError(f"Config '{config_path}' is missing required 'bpe' section.")

    effective_force = force or cfg.force
    for language in cfg.languages:
        _train_for_language(
            input_root=cfg.output_dir,
            language=language,
            bpe_cfg=cfg.bpe,
            force=effective_force,
        )

