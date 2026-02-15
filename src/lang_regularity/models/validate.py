from __future__ import annotations

from pathlib import Path

import torch
import typer

from .catalog import ModelEntry, discover_models, select_model


def _resolve_tokenizer_path(entry: ModelEntry, checkpoint: dict, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if entry.tokenizer_path is not None:
        return entry.tokenizer_path
    raw = checkpoint.get("tokenizer_path")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return None


def _validate_entry(entry: ModelEntry, tokenizer: Path | None = None) -> tuple[bool, str]:
    try:
        checkpoint = torch.load(entry.model_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot load checkpoint: {exc}"

    required_keys = [
        "model_state_dict",
        "vocab_size",
        "block_size",
        "n_embd",
        "n_head",
        "n_layer",
        "dropout",
    ]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        return False, f"checkpoint missing keys: {', '.join(missing)}"

    tokenizer_path = _resolve_tokenizer_path(entry, checkpoint, tokenizer)
    if tokenizer_path is None:
        return False, "tokenizer path is not resolvable"
    if not tokenizer_path.exists():
        return False, f"tokenizer file not found: {tokenizer_path}"

    return True, f"ok (tokenizer={tokenizer_path})"


def run_validate_model(
    runs_root: Path,
    run_dir: Path | None,
    experiment: str | None,
    language: str | None,
    latest: bool,
    all_models: bool,
    tokenizer: Path | None = None,
) -> None:
    if all_models:
        entries = discover_models(runs_root)
        if experiment is not None:
            entries = [e for e in entries if e.experiment == experiment]
        if language is not None:
            entries = [e for e in entries if e.language == language]
        if not entries:
            raise FileNotFoundError(f"No models found under {runs_root} for requested filters.")
    else:
        entries = [
            select_model(
                runs_root=runs_root,
                run_dir=run_dir,
                experiment=experiment,
                language=language,
                prefer_latest=latest,
            )
        ]

    failures = 0
    for entry in entries:
        ok, detail = _validate_entry(entry, tokenizer=tokenizer)
        status = "OK" if ok else "FAIL"
        typer.echo(f"[{status}] {entry.experiment}/{entry.language} {entry.model_path} :: {detail}")
        if not ok:
            failures += 1

    if failures:
        raise typer.Exit(code=1)

