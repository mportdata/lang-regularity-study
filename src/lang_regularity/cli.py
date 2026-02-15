from pathlib import Path
import sys

import typer

app = typer.Typer(help="Language regularity study CLI")


def _interactive_select_config(config_dir: Path = Path("configs")) -> Path:
    import curses

    options = sorted(config_dir.glob("*.yaml"))
    if not options:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")
    if len(options) == 1:
        return options[0]

    labels = [str(p) for p in options]

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

            stdscr.addstr(0, 0, "Select pipeline config (Up/Down, Enter):")
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
    return options[idx]


@app.command("fetch")
def fetch_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if exists"
    ),
) -> None:
    from .data.fetch import fetch

    fetch(config_path=config, force=force)


@app.command("fetch-all")
def fetch_all_command(
    config: Path = typer.Option(
        Path("configs/latin_tight.yaml"),
        "--config",
        "-c",
        help="Path to experiment YAML config file",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if exists"
    ),
) -> None:
    from .data.fetch import fetch

    fetch(config_path=config, force=force)


@app.command("bpe")
def bpe_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    force: bool = typer.Option(False, "--force", "-f", help="Force retrain even if up to date"),
) -> None:
    from .tokenization.train_bpe import run_bpe

    run_bpe(config_path=config, force=force)


@app.command("tokenize")
def tokenize_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if outputs exist"),
) -> None:
    from .tokenization.encode import run_tokenize

    run_tokenize(config_path=config, force=force)


@app.command("pipeline")
def pipeline_command(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to experiment YAML config (if omitted, choose interactively)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild for all pipeline stages"
    ),
) -> None:
    from .data.fetch import fetch
    from .tokenization.encode import run_tokenize
    from .tokenization.train_bpe import run_bpe

    selected_config = config
    if selected_config is None:
        if not sys.stdin.isatty():
            raise ValueError("No --config provided and interactive selection is unavailable.")
        selected_config = _interactive_select_config()

    fetch(config_path=selected_config, force=force)
    run_bpe(config_path=selected_config, force=force)
    run_tokenize(config_path=selected_config, force=force)


@app.command("train")
def train_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    force: bool = typer.Option(False, "--force", "-f", help="Force retrain if outputs exist"),
) -> None:
    from .models.trainer import run_train

    run_train(config_path=config, force=force)


@app.command("models")
def models_command(
    runs_root: Path = typer.Option(
        Path("runs"),
        "--runs-root",
        help="Root directory containing run artifacts",
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Filter by train experiment name"
    ),
    language: str | None = typer.Option(None, "--language", help="Filter by language code"),
) -> None:
    from .models.generate import run_models_list

    run_models_list(
        runs_root=runs_root,
        experiment=experiment,
        language=language,
    )


@app.command("generate")
def generate_command(
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt text to continue (if omitted, you'll be prompted interactively)",
    ),
    runs_root: Path = typer.Option(
        Path("runs"),
        "--runs-root",
        help="Root directory containing run artifacts",
    ),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Explicit run directory (<runs>/<experiment>/<language>)"
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Train experiment name (used with --language)"
    ),
    language: str | None = typer.Option(None, "--language", help="Language code"),
    tokenizer: Path | None = typer.Option(
        None, "--tokenizer", help="Override tokenizer.json path"
    ),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens", help="Tokens to generate"),
    temperature: float = typer.Option(0.8, "--temperature", help="Sampling temperature"),
    top_k: int | None = typer.Option(50, "--top-k", help="Top-k sampling cutoff"),
    device: str = typer.Option("auto", "--device", help="auto|cuda|mps|cpu"),
    latest: bool = typer.Option(
        True,
        "--latest/--no-latest",
        help="Choose latest matching checkpoint when selector is ambiguous",
    ),
) -> None:
    from .models.generate import run_generate

    run_generate(
        prompt=prompt,
        runs_root=runs_root,
        run_dir=run_dir,
        experiment=experiment,
        language=language,
        tokenizer_path=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
        latest=latest,
    )


@app.command("validate-model")
def validate_model_command(
    runs_root: Path = typer.Option(
        Path("runs"),
        "--runs-root",
        help="Root directory containing run artifacts",
    ),
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Explicit run directory (<runs>/<experiment>/<language>)"
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Train experiment name (used with --language)"
    ),
    language: str | None = typer.Option(None, "--language", help="Language code"),
    latest: bool = typer.Option(
        True,
        "--latest/--no-latest",
        help="Choose latest matching checkpoint when selector is ambiguous",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Validate all discovered models (optionally filtered by --experiment/--language)",
    ),
    tokenizer: Path | None = typer.Option(
        None, "--tokenizer", help="Override tokenizer.json path"
    ),
) -> None:
    from .models.validate import run_validate_model

    run_validate_model(
        runs_root=runs_root,
        run_dir=run_dir,
        experiment=experiment,
        language=language,
        latest=latest,
        all_models=all_models,
        tokenizer=tokenizer,
    )


@app.command("eval")
def eval_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recompute eval outputs"),
) -> None:
    from .eval.compare import run_eval

    run_eval(config_path=config, force=force)


@app.command("experiment")
def experiment_command(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to experiment YAML config (if omitted, choose interactively)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild/retrain for all pipeline stages"
    ),
) -> None:
    from .data.fetch import fetch
    from .eval.compare import run_eval
    from .models.trainer import run_train
    from .tokenization.encode import run_tokenize
    from .tokenization.train_bpe import run_bpe

    selected_config = config
    if selected_config is None:
        if not sys.stdin.isatty():
            raise ValueError("No --config provided and interactive selection is unavailable.")
        selected_config = _interactive_select_config()

    fetch(config_path=selected_config, force=force)
    run_bpe(config_path=selected_config, force=force)
    run_tokenize(config_path=selected_config, force=force)
    run_train(config_path=selected_config, force=force)
    run_eval(config_path=selected_config, force=force)


def main() -> None:
    app()
