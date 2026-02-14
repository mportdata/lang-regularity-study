from __future__ import annotations

import math


def perplexity_from_loss(loss: float) -> float:
    return math.exp(min(20.0, loss))

