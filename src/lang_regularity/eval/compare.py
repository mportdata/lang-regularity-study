from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from ..config import load_config
from .robustness import coefficient_of_variation


def run_eval(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)
    if cfg.train is None:
        raise ValueError(f"Config '{config_path}' is missing required 'train' section.")
    if cfg.eval is None:
        raise ValueError(f"Config '{config_path}' is missing required 'eval' section.")

    experiment_root = cfg.train.output_root / cfg.train.experiment_name
    out_dir = experiment_root / cfg.eval.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    if summary_path.exists() and not (force or cfg.force):
        typer.echo(f"Eval summary already exists at '{summary_path}'. Skipping (use --force).")
        return

    rows: list[dict[str, object]] = []
    val_ppls: list[float] = []
    for language in cfg.languages:
        metrics_path = experiment_root / language / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing training metrics for '{language}': {metrics_path}. Run train first."
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "language_code": language,
            "val_ppl": metrics["val_ppl"],
            "val_loss": metrics["val_loss"],
            "train_loss": metrics["train_loss"],
            "device_used": metrics["device_used"],
            "model_path": metrics["model_path"],
        }
        rows.append(row)
        val_ppls.append(float(metrics["val_ppl"]))

    sorted_rows = sorted(rows, key=lambda r: float(r["val_ppl"]))
    now = datetime.now(UTC).isoformat()
    summary = {
        "experiment_name": cfg.train.experiment_name,
        "rows": sorted_rows,
        "mean_val_ppl": sum(val_ppls) / len(val_ppls) if val_ppls else 0.0,
        "val_ppl_cv": coefficient_of_variation(val_ppls),
        "best_language": sorted_rows[0]["language_code"] if sorted_rows else None,
        "worst_language": sorted_rows[-1]["language_code"] if sorted_rows else None,
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote eval summary: {summary_path}")

