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
class TokenizeConfig:
    experiment_name: str
    output_root: Path
    val_ratio: float
    seed: int
    add_bos: bool
    add_eos: bool
    dtype: str
    max_tokens: int | None = None


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    output_root: Path
    device: str
    block_size: int
    batch_size: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    seed: int


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
    tokenize: TokenizeConfig | None = None
    train: TrainConfig | None = None


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


def _load_tokenize_config(data: dict, config_path: Path) -> TokenizeConfig | None:
    tokenize_raw = data.get("tokenize")
    if tokenize_raw is None:
        return None
    if not isinstance(tokenize_raw, dict):
        raise ValueError(f"Config key 'tokenize' in '{config_path}' must be a mapping.")

    experiment_name = _require_str(tokenize_raw, "experiment_name", config_path)
    output_root = Path(_require_str(tokenize_raw, "output_root", config_path))
    seed = _require_int(tokenize_raw, "seed", config_path)

    val_ratio_raw = tokenize_raw.get("val_ratio")
    if not isinstance(val_ratio_raw, (int, float)) or not (0 < float(val_ratio_raw) < 1):
        raise ValueError(
            f"Config key 'tokenize.val_ratio' in '{config_path}' must be between 0 and 1."
        )
    val_ratio = float(val_ratio_raw)

    add_bos = tokenize_raw.get("add_bos", False)
    add_eos = tokenize_raw.get("add_eos", True)
    if not isinstance(add_bos, bool):
        raise ValueError(f"Config key 'tokenize.add_bos' in '{config_path}' must be a boolean.")
    if not isinstance(add_eos, bool):
        raise ValueError(f"Config key 'tokenize.add_eos' in '{config_path}' must be a boolean.")

    dtype = _require_str(tokenize_raw, "dtype", config_path).lower()
    if dtype not in {"auto", "uint16", "uint32"}:
        raise ValueError(
            f"Config key 'tokenize.dtype' in '{config_path}' must be one of auto|uint16|uint32."
        )

    max_tokens_raw = tokenize_raw.get("max_tokens")
    max_tokens: int | None = None
    if max_tokens_raw is not None:
        if not isinstance(max_tokens_raw, int) or max_tokens_raw <= 0:
            raise ValueError(
                f"Config key 'tokenize.max_tokens' in '{config_path}' must be a positive integer."
            )
        max_tokens = max_tokens_raw

    return TokenizeConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        val_ratio=val_ratio,
        seed=seed,
        add_bos=add_bos,
        add_eos=add_eos,
        dtype=dtype,
        max_tokens=max_tokens,
    )


def _load_train_config(data: dict, config_path: Path) -> TrainConfig | None:
    train_raw = data.get("train")
    if train_raw is None:
        return None
    if not isinstance(train_raw, dict):
        raise ValueError(f"Config key 'train' in '{config_path}' must be a mapping.")

    experiment_name = _require_str(train_raw, "experiment_name", config_path)
    output_root = Path(_require_str(train_raw, "output_root", config_path))
    device = _require_str(train_raw, "device", config_path).lower()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Config key 'train.device' in '{config_path}' must be one of auto|cpu|cuda|mps."
        )

    block_size = _require_int(train_raw, "block_size", config_path)
    batch_size = _require_int(train_raw, "batch_size", config_path)
    max_steps = _require_int(train_raw, "max_steps", config_path)
    eval_interval = _require_int(train_raw, "eval_interval", config_path)
    eval_batches = _require_int(train_raw, "eval_batches", config_path)
    n_embd = _require_int(train_raw, "n_embd", config_path)
    n_head = _require_int(train_raw, "n_head", config_path)
    n_layer = _require_int(train_raw, "n_layer", config_path)
    seed = _require_int(train_raw, "seed", config_path)

    if block_size <= 1:
        raise ValueError(f"Config key 'train.block_size' in '{config_path}' must be > 1.")
    if batch_size <= 0:
        raise ValueError(f"Config key 'train.batch_size' in '{config_path}' must be > 0.")
    if max_steps <= 0:
        raise ValueError(f"Config key 'train.max_steps' in '{config_path}' must be > 0.")
    if eval_interval <= 0:
        raise ValueError(f"Config key 'train.eval_interval' in '{config_path}' must be > 0.")
    if eval_batches <= 0:
        raise ValueError(f"Config key 'train.eval_batches' in '{config_path}' must be > 0.")
    if n_embd <= 0 or n_head <= 0 or n_layer <= 0:
        raise ValueError(
            f"Config keys 'train.n_embd|n_head|n_layer' in '{config_path}' must be > 0."
        )
    if n_embd % n_head != 0:
        raise ValueError(
            f"Config key 'train.n_embd' in '{config_path}' must be divisible by train.n_head."
        )

    learning_rate_raw = train_raw.get("learning_rate")
    weight_decay_raw = train_raw.get("weight_decay")
    grad_clip_raw = train_raw.get("grad_clip")
    dropout_raw = train_raw.get("dropout")
    if not isinstance(learning_rate_raw, (int, float)) or learning_rate_raw <= 0:
        raise ValueError(
            f"Config key 'train.learning_rate' in '{config_path}' must be a positive number."
        )
    if not isinstance(weight_decay_raw, (int, float)) or weight_decay_raw < 0:
        raise ValueError(
            f"Config key 'train.weight_decay' in '{config_path}' must be >= 0."
        )
    if not isinstance(grad_clip_raw, (int, float)) or grad_clip_raw <= 0:
        raise ValueError(f"Config key 'train.grad_clip' in '{config_path}' must be > 0.")
    if not isinstance(dropout_raw, (int, float)) or not (0 <= float(dropout_raw) < 1):
        raise ValueError(f"Config key 'train.dropout' in '{config_path}' must be in [0, 1).")

    return TrainConfig(
        experiment_name=experiment_name,
        output_root=output_root,
        device=device,
        block_size=block_size,
        batch_size=batch_size,
        max_steps=max_steps,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        learning_rate=float(learning_rate_raw),
        weight_decay=float(weight_decay_raw),
        grad_clip=float(grad_clip_raw),
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=float(dropout_raw),
        seed=seed,
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
    tokenize = _load_tokenize_config(data, config_path)
    train = _load_train_config(data, config_path)

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
        tokenize=tokenize,
        train=train,
    )
