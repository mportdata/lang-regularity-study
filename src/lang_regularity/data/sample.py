from __future__ import annotations

from pathlib import Path

import typer


def run_sample(input_path: Path, output_path: Path, max_size_mb: float) -> None:
    """Placeholder for deterministic corpus sampling utilities."""
    typer.echo(
        "Sample stage scaffold is in place. "
        f"Implement sampling from '{input_path}' -> '{output_path}' (max_size_mb={max_size_mb})."
    )

