from __future__ import annotations

from pathlib import Path


def _size_token(max_size_mb: float | None) -> str:
    if max_size_mb is None:
        return "full"
    if float(max_size_mb).is_integer():
        return str(int(max_size_mb))
    return str(max_size_mb).replace(".", "p")


def corpus_filename(max_size_mb: float | None) -> str:
    return f"wiki_{_size_token(max_size_mb)}mb.txt" if max_size_mb is not None else "wiki_full.txt"


def corpus_metadata_filename(max_size_mb: float | None) -> str:
    return f"{corpus_filename(max_size_mb)}.meta.json"


def corpus_paths(raw_lang_dir: Path, max_size_mb: float | None) -> tuple[Path, Path]:
    corpus = raw_lang_dir / corpus_filename(max_size_mb)
    metadata = raw_lang_dir / corpus_metadata_filename(max_size_mb)
    return corpus, metadata


def resolve_input_corpus_path(raw_lang_dir: Path, max_size_mb: float | None) -> Path:
    expected, _ = corpus_paths(raw_lang_dir, max_size_mb)
    if expected.exists():
        return expected

    # Backward compatibility with older layout.
    legacy = raw_lang_dir / "wiki.txt"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"Missing corpus: expected '{expected}' (or legacy '{legacy}'). Run fetch first."
    )

