from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    experiment: str
    language: str
    model_path: Path
    metrics_path: Path | None
    tokenizer_path: Path | None
    updated_at_utc: str | None


def discover_models(runs_root: Path) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    if not runs_root.exists():
        return entries

    for model_path in sorted(runs_root.glob("*/*/model.pt")):
        language = model_path.parent.name
        experiment = model_path.parent.parent.name
        metrics_path = model_path.parent / "metrics.json"
        tokenizer_path: Path | None = None
        updated_at_utc: str | None = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                tokenizer_raw = metrics.get("tokenizer_path")
                if isinstance(tokenizer_raw, str) and tokenizer_raw:
                    tokenizer_path = Path(tokenizer_raw)
                if tokenizer_path is None:
                    train_bin_raw = metrics.get("train_bin_path")
                    if isinstance(train_bin_raw, str) and train_bin_raw:
                        tokenized_dir = Path(train_bin_raw).parent
                        tokenize_meta_path = tokenized_dir / "tokenize.meta.json"
                        if tokenize_meta_path.exists():
                            tokenize_meta = json.loads(
                                tokenize_meta_path.read_text(encoding="utf-8")
                            )
                            tok_meta_raw = tokenize_meta.get("tokenizer_path")
                            if isinstance(tok_meta_raw, str) and tok_meta_raw:
                                tokenizer_path = Path(tok_meta_raw)
                updated_raw = metrics.get("updated_at_utc")
                if isinstance(updated_raw, str) and updated_raw:
                    updated_at_utc = updated_raw
            except json.JSONDecodeError:
                pass
        entries.append(
            ModelEntry(
                experiment=experiment,
                language=language,
                model_path=model_path,
                metrics_path=metrics_path if metrics_path.exists() else None,
                tokenizer_path=tokenizer_path,
                updated_at_utc=updated_at_utc,
            )
        )
    return entries


def select_model(
    runs_root: Path,
    run_dir: Path | None,
    experiment: str | None,
    language: str | None,
    prefer_latest: bool,
) -> ModelEntry:
    if run_dir is not None:
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        entries = discover_models(runs_root)
        for entry in entries:
            if entry.model_path == model_path:
                return entry
        return ModelEntry(
            experiment=run_dir.parent.name,
            language=run_dir.name,
            model_path=model_path,
            metrics_path=(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None,
            tokenizer_path=None,
            updated_at_utc=None,
        )

    if not experiment or not language:
        raise ValueError(
            "Provide either --run-dir or both --experiment and --language to select a model."
        )

    candidates = [
        e
        for e in discover_models(runs_root)
        if e.experiment == experiment and e.language == language
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No model found for experiment='{experiment}', language='{language}' under {runs_root}."
        )
    if len(candidates) == 1:
        return candidates[0]

    if prefer_latest:
        candidates.sort(key=lambda e: e.model_path.stat().st_mtime, reverse=True)
        return candidates[0]

    locations = "\n".join(str(c.model_path) for c in candidates)
    raise ValueError(
        "Multiple models match your selector. Pass --latest or choose --run-dir explicitly.\n"
        f"{locations}"
    )
