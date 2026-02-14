from __future__ import annotations

from array import array
from pathlib import Path


def load_token_ids(bin_path: Path, dtype_name: str):
    import torch

    if dtype_name == "uint16":
        typecode = "H"
    elif dtype_name == "uint32":
        typecode = "I"
    else:
        raise ValueError(f"Unsupported token dtype: {dtype_name}")

    raw = array(typecode)
    with bin_path.open("rb") as f:
        raw.fromfile(f, bin_path.stat().st_size // raw.itemsize)
    return torch.tensor(raw, dtype=torch.long)


class PackedTokenDataset:
    def __init__(self, tokens, block_size: int) -> None:
        if tokens.numel() <= block_size:
            raise ValueError(
                f"Token count {tokens.numel()} must exceed block_size {block_size}."
            )
        self.tokens = tokens
        self.block_size = block_size
        self.max_start = tokens.numel() - block_size - 1

    def sample_batch(self, batch_size: int, device: str):
        import torch

        starts = torch.randint(low=0, high=self.max_start + 1, size=(batch_size,))
        x = torch.stack([self.tokens[s : s + self.block_size] for s in starts])
        y = torch.stack([self.tokens[s + 1 : s + 1 + self.block_size] for s in starts])
        return x.to(device), y.to(device)
