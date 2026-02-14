from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    languages: list[str]
    hf_dataset: str
    hf_date: str
    hf_split: str
    hf_text_field: str
    output_dir: Path
    work_dir: Path
    max_size_mb: float | None = None
    force: bool = False


def _require_mapping(data: object, config_path: Path) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{config_path}' must be a YAML mapping.")
    return data


def _require_str(data: dict, key: str, config_path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config key '{key}' in '{config_path}' must be a non-empty string.")
    return value.strip()


def _require_str_list(data: dict, key: str, config_path: Path) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config key '{key}' in '{config_path}' must be a non-empty list.")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Config key '{key}' in '{config_path}' has invalid item at index {i}; "
                "expected non-empty string."
            )
        out.append(item.strip())
    return out


def load_config(config_path: Path) -> ExperimentConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = _require_mapping(raw, config_path)
    languages = _require_str_list(data, "languages", config_path)
    hf_dataset = _require_str(data, "hf_dataset", config_path)
    hf_date = _require_str(data, "hf_date", config_path)
    hf_split = _require_str(data, "hf_split", config_path)
    hf_text_field = _require_str(data, "hf_text_field", config_path)
    output_dir = Path(_require_str(data, "output_dir", config_path))
    work_dir = Path(_require_str(data, "work_dir", config_path))

    max_size_mb = data.get("max_size_mb")
    if max_size_mb is not None:
        if not isinstance(max_size_mb, (int, float)) or max_size_mb <= 0:
            raise ValueError(
                f"Config key 'max_size_mb' in '{config_path}' must be a positive number."
            )
        max_size_mb = float(max_size_mb)

    force = data.get("force", False)
    if not isinstance(force, bool):
        raise ValueError(f"Config key 'force' in '{config_path}' must be a boolean.")

    return ExperimentConfig(
        languages=languages,
        hf_dataset=hf_dataset,
        hf_date=hf_date,
        hf_split=hf_split,
        hf_text_field=hf_text_field,
        output_dir=output_dir,
        work_dir=work_dir,
        max_size_mb=max_size_mb,
        force=force,
    )
