import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Low-Rank Adaptation for the Query (Q) and Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        alpha: int = 1,
        r: int = 3,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.scale = alpha / r  # Scaling factor for LoRA updates

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components with scaling
        new_q = self.scale * self.linear_b_q(self.linear_a_q(x))  # LoRA update for query
        new_v = self.scale * self.linear_b_v(self.linear_a_v(x))  # LoRA update for value

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q  # Update query
        qkv[:, :, -self.dim :] += new_v  # Update value

        return qkv

