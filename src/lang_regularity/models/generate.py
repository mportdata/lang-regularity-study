from __future__ import annotations

from pathlib import Path
import re
import sys

import torch
import typer
from tokenizers import Tokenizer, decoders

from ..utils.device import resolve_torch_device
from .catalog import discover_models, select_model
from .gpt import TinyGPT


def _interactive_select_model(entries) -> object:
    import curses

    labels = [
        f"{e.experiment}/{e.language} [{e.updated_at_utc or '-'}] {e.model_path}" for e in entries
    ]

    def _run(stdscr):
        curses.curs_set(0)
        selected = 0
        top = 0

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            max_rows = max(1, height - 3)
            if selected < top:
                top = selected
            if selected >= top + max_rows:
                top = selected - max_rows + 1

            stdscr.addstr(0, 0, "Select a model (Up/Down, Enter):")
            view = labels[top : top + max_rows]
            for row, label in enumerate(view, start=0):
                i = top + row
                line = label[: max(1, width - 1)]
                if i == selected:
                    stdscr.addstr(row + 2, 0, line, curses.A_REVERSE)
                else:
                    stdscr.addstr(row + 2, 0, line)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(labels) - 1, selected + 1)
            elif key in (10, 13, curses.KEY_ENTER):
                return selected

    idx = curses.wrapper(_run)
    return entries[idx]


def _decode_text(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    raw_tokens = [tokenizer.id_to_token(i) for i in token_ids]
    raw_tokens = [t for t in raw_tokens if isinstance(t, str)]
    text = decoders.ByteLevel().decode(raw_tokens) if raw_tokens else ""
    if not text:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    # Backward-compatible fallback for older tokenizer artifacts without byte-level decoder.
    if "Ġ" in text or "Ċ" in text:
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
        text = re.sub(r"[ ]+([,.;:!?])", r"\1", text)
        text = re.sub(r"\n[ ]+", "\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
    return text


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.shape[-1])
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        vals = vals / vals.sum()
        choice = torch.multinomial(vals, num_samples=1)
        return int(idx[choice].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def run_models_list(
    runs_root: Path,
    experiment: str | None = None,
    language: str | None = None,
) -> None:
    entries = discover_models(runs_root)
    if experiment:
        entries = [e for e in entries if e.experiment == experiment]
    if language:
        entries = [e for e in entries if e.language == language]

    if not entries:
        typer.echo("No models found.")
        return

    typer.echo("experiment\tlanguage\tupdated_at_utc\tmodel_path")
    for entry in entries:
        typer.echo(
            f"{entry.experiment}\t{entry.language}\t"
            f"{entry.updated_at_utc or '-'}\t{entry.model_path}"
        )


def run_generate(
    prompt: str | None,
    runs_root: Path,
    run_dir: Path | None,
    experiment: str | None,
    language: str | None,
    tokenizer_path: Path | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
    latest: bool,
) -> None:
    selectors_given = run_dir is not None or (experiment is not None and language is not None)
    if selectors_given:
        entry = select_model(
            runs_root=runs_root,
            run_dir=run_dir,
            experiment=experiment,
            language=language,
            prefer_latest=latest,
        )
    else:
        entries = discover_models(runs_root)
        if not entries:
            raise FileNotFoundError(f"No models found under {runs_root}.")
        entries.sort(key=lambda e: e.model_path.stat().st_mtime, reverse=True)

        if not sys.stdin.isatty():
            raise ValueError(
                "No model selector args were provided and interactive selection is unavailable "
                "(non-TTY). Pass --run-dir or --experiment + --language."
            )

        if len(entries) == 1:
            entry = entries[0]
            typer.echo(
                "Using only discovered model: "
                f"{entry.experiment}/{entry.language} ({entry.model_path})"
            )
        else:
            entry = _interactive_select_model(entries)

    prompt_text = prompt
    if prompt_text is None or not prompt_text.strip():
        if not sys.stdin.isatty():
            raise ValueError(
                "No prompt provided and interactive input is unavailable (non-TTY). "
                "Pass --prompt explicitly."
            )
        prompt_text = typer.prompt("Prompt")
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("Prompt cannot be empty.")

    device_info = resolve_torch_device(device)
    if device_info.fallback_reason:
        typer.echo(f"{device_info.fallback_reason} Falling back to CPU.")

    checkpoint = torch.load(entry.model_path, map_location=device_info.selected)
    tokenizer_from_checkpoint = checkpoint.get("tokenizer_path")
    tokenizer_file = tokenizer_path or entry.tokenizer_path
    if tokenizer_file is None and isinstance(tokenizer_from_checkpoint, str):
        tokenizer_file = Path(tokenizer_from_checkpoint)
    if tokenizer_file is None:
        raise ValueError(
            "Tokenizer path could not be resolved from run metadata or checkpoint. "
            "Pass --tokenizer explicitly."
        )
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")

    model = TinyGPT(
        vocab_size=int(checkpoint["vocab_size"]),
        block_size=int(checkpoint["block_size"]),
        n_embd=int(checkpoint["n_embd"]),
        n_head=int(checkpoint["n_head"]),
        n_layer=int(checkpoint["n_layer"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device_info.selected)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    prompt_ids = tokenizer.encode(prompt_text).ids
    if not prompt_ids:
        raise ValueError("Prompt produced zero tokens; provide non-empty prompt text.")

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device_info.selected)[None, :]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size :]
            logits, _ = model(idx_cond)
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
            )
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device_info.selected)
            idx = torch.cat([idx, next_tensor], dim=1)

    out_ids = idx[0].tolist()
    text = _decode_text(tokenizer, out_ids)
    typer.echo(text)
