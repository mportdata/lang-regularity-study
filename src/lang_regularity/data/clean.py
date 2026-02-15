from __future__ import annotations

from pathlib import Path

import typer


def run_clean(input_path: Path, output_path: Path) -> None:
    """Placeholder for optional corpus cleaning/normalization variants."""
    typer.echo(
        "Clean stage scaffold is in place. "
        f"Implement transformations from '{input_path}' -> '{output_path}'."
    )

