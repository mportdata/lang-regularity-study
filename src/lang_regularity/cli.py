from pathlib import Path

import typer

from .data.fetch import fetch

app = typer.Typer(help="Language regularity study CLI")


@app.command("fetch")
def fetch_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if exists"
    ),
) -> None:
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
    fetch(config_path=config, force=force)


def main() -> None:
    app()
