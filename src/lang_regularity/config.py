from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BPEConfig:
    experiment_name: str
    output_root: Path
    vocab_size: int
    min_frequency: int
    limit_alphabet: int
    train_sample_mb: float
    shuffle_seed: int
    special_tokens: list[str]
    unk_token: str


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
    bpe: BPEConfig | None = None


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


def _require_int(data: dict, key: str, config_path: Path) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config key '{key}' in '{config_path}' must be an integer.")
    return value


def _load_bpe_config(data: dict, config_path: Path) -> BPEConfig | None:
    bpe_raw = data.get("bpe")
    if bpe_raw is None:
        return None
    if not isinstance(bpe_raw, dict):
        raise ValueError(f"Config key 'bpe' in '{config_path}' must be a mapping.")

    experiment_name = _require_str(bpe_raw, "experiment_name", config_path)
    output_root = Path(_require_str(bpe_raw, "output_root", config_path))

    vocab_size = _require_int(bpe_raw, "vocab_size", config_path)
    min_frequency = _require_int(bpe_raw, "min_frequency", config_path)
    limit_alphabet = _require_int(bpe_raw, "limit_alphabet", config_path)
    shuffle_seed = _require_int(bpe_raw, "shuffle_seed", config_path)
    train_sample_mb_raw = bpe_raw.get("train_sample_mb")
    if not isinstance(train_sample_mb_raw, (int, float)) or train_sample_mb_raw <= 0:
        raise ValueError(
            f"Config key 'bpe.train_sample_mb' in '{config_path}' must be a positive number."
        )
    train_sample_mb = float(train_sample_mb_raw)

    special_tokens = _require_str_list(bpe_raw, "special_tokens", config_path)
    unk_token = _require_str(bpe_raw, "unk_token", config_path)

    if unk_token not in special_tokens:
        raise ValueError(
            f"Config key 'bpe.unk_token' in '{config_path}' must exist in 'bpe.special_tokens'."
        )
    if vocab_size <= len(special_tokens):
        raise ValueError(
            f"Config key 'bpe.vocab_size' in '{config_path}' must exceed special token count."
        )
    if min_frequency < 1:
        raise ValueError(f"Config key 'bpe.min_frequency' in '{config_path}' must be >= 1.")
    if limit_alphabet < 1:
        raise ValueError(f"Config key 'bpe.limit_alphabet' in '{config_path}' must be >= 1.")

    return BPEConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        limit_alphabet=limit_alphabet,
        train_sample_mb=train_sample_mb,
        shuffle_seed=shuffle_seed,
        special_tokens=special_tokens,
        unk_token=unk_token,
    )


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

    bpe = _load_bpe_config(data, config_path)

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
        bpe=bpe,
    )
