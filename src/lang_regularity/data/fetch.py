from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import typer
from datasets import load_dataset

from ..config import load_config


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _stream_hf_to_text(
    dataset_name: str,
    language_code: str,
    dataset_date: str,
    split: str,
    text_field: str,
    output_path: Path,
    max_size_mb: float | None,
) -> dict[str, int | bool]:
    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb is not None else None
    bytes_written = 0
    articles_written = 0
    truncated = False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = f"{dataset_date}.{language_code}"
    stream = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
    )

    with output_path.open("w", encoding="utf-8") as out:
        for row in stream:
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            normalized = _normalize_text(text)
            if not normalized:
                continue

            article_blob = normalized + "\n\n"
            encoded = article_blob.encode("utf-8")

            if max_bytes is not None and bytes_written + len(encoded) > max_bytes:
                truncated = True
                break

            out.write(article_blob)
            bytes_written += len(encoded)
            articles_written += 1

    return {
        "bytes_written": bytes_written,
        "articles_written": articles_written,
        "truncated": truncated,
    }


def _fetch_language(cfg, lang: str, force: bool) -> None:
    effective_force = force or cfg.force

    raw_lang_dir = cfg.output_dir / lang
    work_lang_dir = cfg.work_dir / lang
    output_text_path = raw_lang_dir / "wiki.txt"
    metadata_path = raw_lang_dir / "wiki.txt.meta.json"

    raw_lang_dir.mkdir(parents=True, exist_ok=True)
    work_lang_dir.mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    if output_text_path.exists() and metadata_path.exists() and not effective_force:
        typer.echo(
            f"Output already exists at '{output_text_path}'. Skipping (use --force to rebuild)."
        )
        return

    typer.echo(
        "Streaming from Hugging Face dataset "
        f"'{cfg.hf_dataset}' (language={lang}, date={cfg.hf_date})..."
    )

    stats = _stream_hf_to_text(
        dataset_name=cfg.hf_dataset,
        language_code=lang,
        dataset_date=cfg.hf_date,
        split=cfg.hf_split,
        text_field=cfg.hf_text_field,
        output_path=output_text_path,
        max_size_mb=cfg.max_size_mb,
    )

    now = datetime.now(UTC).isoformat()
    text_sha256 = _sha256_for_file(output_text_path)
    hf_dataset_url = f"https://huggingface.co/datasets/{cfg.hf_dataset}"

    metadata = {
        "language_code": lang,
        "source_type": "huggingface_stream",
        "source_url": hf_dataset_url,
        "hf_dataset": cfg.hf_dataset,
        "hf_date": cfg.hf_date,
        "hf_split": cfg.hf_split,
        "hf_text_field": cfg.hf_text_field,
        "output_text_path": str(output_text_path),
        "work_dir": str(work_lang_dir),
        "output_text_sha256": text_sha256,
        "output_text_size_bytes": output_text_path.stat().st_size,
        "articles_written": stats["articles_written"],
        "truncated_by_max_size": stats["truncated"],
        "max_size_mb": cfg.max_size_mb,
        "created_at_utc": now,
        "updated_at_utc": now,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    typer.echo(f"Wrote text: {output_text_path}")
    typer.echo(f"Wrote metadata: {metadata_path}")


def fetch(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)
    for lang in cfg.languages:
        _fetch_language(cfg=cfg, lang=lang, force=force)
