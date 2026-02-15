from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path

import typer

from ..config import ExperimentConfig, TrainConfig, load_config
from ..utils.device import resolve_torch_device


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_eval_loss(model, dataset: PackedTokenDataset, cfg: TrainConfig, device: str) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            xb, yb = dataset.sample_batch(cfg.batch_size, device=device)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Expected loss in eval, got None.")
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _run_language_training(cfg: ExperimentConfig, language: str, force: bool) -> None:
    import torch
    from .dataset import PackedTokenDataset, load_token_ids
    from .gpt import TinyGPT

    if cfg.tokenize is None:
        raise ValueError("Missing tokenize config; run tokenization first.")
    if cfg.train is None:
        raise ValueError("Missing train config.")

    tokenize_root = cfg.tokenize.output_root / cfg.tokenize.experiment_name / language
    meta_path = tokenize_root / "tokenize.meta.json"
    train_bin = tokenize_root / "train.bin"
    val_bin = tokenize_root / "val.bin"
    if not meta_path.exists() or not train_bin.exists() or not val_bin.exists():
        raise FileNotFoundError(
            f"Missing encoded artifacts for '{language}' in {tokenize_root}. Run tokenize first."
        )

    tokenize_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dtype_name = str(tokenize_meta["dtype"])
    vocab_size = int(tokenize_meta["vocab_size"])
    tokenizer_path = (
        cfg.bpe.output_root / cfg.bpe.experiment_name / language / "tokenizer.json"
        if cfg.bpe is not None
        else None
    )

    device_info = resolve_torch_device(cfg.train.device)
    if device_info.fallback_reason:
        typer.echo(f"{device_info.fallback_reason} Falling back to CPU.")
    typer.echo(f"Training '{language}' on device={device_info.selected}")

    run_dir = cfg.train.output_root / cfg.train.experiment_name / language
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"
    train_log_path = run_dir / "train.log"
    cfg_snapshot_path = run_dir / "config.snapshot.json"

    if model_path.exists() and metrics_path.exists() and not force:
        typer.echo(
            f"Training outputs already exist for '{language}' at {run_dir}. Skipping (use --force)."
        )
        return

    train_tokens = load_token_ids(train_bin, dtype_name=dtype_name)
    val_tokens = load_token_ids(val_bin, dtype_name=dtype_name)
    train_ds = PackedTokenDataset(train_tokens, block_size=cfg.train.block_size)
    val_ds = PackedTokenDataset(val_tokens, block_size=cfg.train.block_size)

    _set_seed(cfg.train.seed)
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=cfg.train.block_size,
        n_embd=cfg.train.n_embd,
        n_head=cfg.train.n_head,
        n_layer=cfg.train.n_layer,
        dropout=cfg.train.dropout,
    ).to(device_info.selected)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    with train_log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, cfg.train.max_steps + 1):
            xb, yb = train_ds.sample_batch(cfg.train.batch_size, device=device_info.selected)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Expected train loss, got None.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
            optimizer.step()

            if step == 1 or step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
                train_loss = _mean_eval_loss(model, train_ds, cfg.train, device_info.selected)
                val_loss = _mean_eval_loss(model, val_ds, cfg.train, device_info.selected)
                val_ppl = math.exp(min(20.0, val_loss))
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                }
                history.append(row)
                log_f.write(
                    f"step={step} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}\n"
                )
                log_f.flush()
                typer.echo(
                    f"[{language}] step={step} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}"
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": cfg.train.block_size,
            "n_embd": cfg.train.n_embd,
            "n_head": cfg.train.n_head,
            "n_layer": cfg.train.n_layer,
            "dropout": cfg.train.dropout,
            "language": language,
            "tokenize_experiment": cfg.tokenize.experiment_name,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        },
        model_path,
    )

    latest = history[-1]
    now = datetime.now(UTC).isoformat()
    metrics = {
        "language_code": language,
        "device_requested": cfg.train.device,
        "device_used": device_info.selected,
        "train_bin_path": str(train_bin),
        "val_bin_path": str(val_bin),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "model_path": str(model_path),
        "train_log_path": str(train_log_path),
        "max_steps": cfg.train.max_steps,
        "batch_size": cfg.train.batch_size,
        "block_size": cfg.train.block_size,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip": cfg.train.grad_clip,
        "n_embd": cfg.train.n_embd,
        "n_head": cfg.train.n_head,
        "n_layer": cfg.train.n_layer,
        "dropout": cfg.train.dropout,
        "seed": cfg.train.seed,
        "final_step": latest["step"],
        "train_loss": latest["train_loss"],
        "val_loss": latest["val_loss"],
        "val_ppl": latest["val_ppl"],
        "history": history,
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    config_snapshot = {
        "languages": cfg.languages,
        "tokenize_experiment": cfg.tokenize.experiment_name,
        "train": {
            "experiment_name": cfg.train.experiment_name,
            "output_root": str(cfg.train.output_root),
            "device": cfg.train.device,
            "block_size": cfg.train.block_size,
            "batch_size": cfg.train.batch_size,
            "max_steps": cfg.train.max_steps,
            "eval_interval": cfg.train.eval_interval,
            "eval_batches": cfg.train.eval_batches,
            "learning_rate": cfg.train.learning_rate,
            "weight_decay": cfg.train.weight_decay,
            "grad_clip": cfg.train.grad_clip,
            "n_embd": cfg.train.n_embd,
            "n_head": cfg.train.n_head,
            "n_layer": cfg.train.n_layer,
            "dropout": cfg.train.dropout,
            "seed": cfg.train.seed,
        },
        "created_at_utc": now,
    }
    cfg_snapshot_path.write_text(json.dumps(config_snapshot, indent=2) + "\n", encoding="utf-8")

    typer.echo(f"Wrote model: {model_path}")
    typer.echo(f"Wrote train metrics: {metrics_path}")


def run_train(config_path: Path, force: bool = False) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch. Install with one of:\n"
            "  uv pip install '.[train]'        # CPU default\n"
            "  pip install torch                 # choose platform wheel as needed"
        ) from exc

    cfg = load_config(config_path)
    if cfg.train is None:
        raise ValueError(f"Config '{config_path}' is missing required 'train' section.")
    if cfg.tokenize is None:
        raise ValueError(f"Config '{config_path}' is missing required 'tokenize' section.")

    effective_force = force or cfg.force
    for language in cfg.languages:
        _run_language_training(cfg=cfg, language=language, force=effective_force)
