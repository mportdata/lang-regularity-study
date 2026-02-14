from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bsz, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}.")

        pos = torch.arange(0, t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]

        causal_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1
        )
        x = self.blocks(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(bsz * t, -1), targets.reshape(bsz * t))
        return logits, loss

