"""ALS edge-classification model for v1.

This is a minimal logistic model that maps four features:
[Semantic_Similarity, Emotional_Alignment, Time_Closeness, Target_Intensity]
into a single edge "link" probability via a linear layer
followed by a sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ALSModel(nn.Module):
    def __init__(self, input_dim: int = 4, ignore_feature: int | None = None) -> None:
        super().__init__()
        # Order of inputs varies by config; default is 3 for legacy
        self.slp = nn.Linear(input_dim, 1, bias=True)
        # If set to an index, that feature is zeroed in forward.
        self.ignore_feature = ignore_feature
        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize weights to a small positive value (similarity -> link)
        nn.init.constant_(self.slp.weight, 0.1)
        # Initialize bias to class distribution prior (~1/3 links -> logit ~= -0.69)
        nn.init.constant_(self.slp.bias, -0.69)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_feature is not None:
            # Zero out the selected feature dimension across the batch.
            x = x.clone()
            x[:, self.ignore_feature] = 0.0
        return torch.sigmoid(self.slp(x))
