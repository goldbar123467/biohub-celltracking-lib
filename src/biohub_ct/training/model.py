from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 12

    def __post_init__(self) -> None:
        if self.base_channels < 4 or self.base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four, at least four")


def _block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


def _downsample(image: torch.Tensor) -> torch.Tensor:
    """Nonoverlapping block means with deterministic CUDA autograd."""
    b, c, z, y, x = image.shape
    cropped = image[:, :, : z // 2 * 2, : y // 2 * 2, : x // 2 * 2]
    return cropped.reshape(b, c, z // 2, 2, y // 2, 2, x // 2, 2).mean(dim=(3, 5, 7))


def _upsample(image: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Nearest replication; concatenate edge slices to support odd skip shapes."""
    for axis in (2, 3, 4):
        image = image.repeat_interleave(2, dim=axis)
        if image.shape[axis] < shape[axis - 2]:
            edge = [slice(None)] * 5
            edge[axis] = slice(-1, None)
            image = torch.cat((image, image[tuple(edge)]), dim=axis)
    return image


class PointDetector3D(nn.Module):
    """Two-level U-Net, B1ZYX normalized intensity to B1ZYX center logits."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        c = config.base_channels
        self.encoder0 = _block(1, c)
        self.encoder1 = _block(c, c * 2)
        self.bottom = _block(c * 2, c * 4)
        self.decoder1 = _block(c * 6, c * 2)
        self.decoder0 = _block(c * 3, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1 or min(image.shape[2:]) < 4:
            raise ValueError("Expected B1ZYX input with spatial dimensions >= 4")
        e0 = self.encoder0(image)
        e1 = self.encoder1(_downsample(e0))
        bottom = self.bottom(_downsample(e1))
        d1 = self.decoder1(torch.cat((_upsample(bottom, e1.shape[2:]), e1), 1))
        d0 = self.decoder0(torch.cat((_upsample(d1, e0.shape[2:]), e0), 1))
        return self.head(d0)


def masked_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    normalizer: float | None = None,
) -> torch.Tensor:
    """Weighted sigmoid MSE. Unknown voxels have weight zero and zero gradient.

    Inputs must all be finite, even outside supervision; corruption fails loudly.
    """
    if logits.shape != target.shape or logits.shape != weight.shape or logits.ndim != 5:
        raise ValueError("Loss inputs must have the same B1ZYX shape")
    if logits.shape[1] != 1:
        raise ValueError("Loss requires one heatmap channel")
    if not all(bool(torch.isfinite(x).all()) for x in (logits, target, weight)):
        raise ValueError("Nonfinite heatmap loss input")
    if bool(((target < 0) | (target > 1)).any()) or bool((weight < 0).any()):
        raise ValueError("Targets must be in [0,1] and weights nonnegative")
    denominator = weight.float().sum()
    if not bool(denominator > 0):
        raise ValueError("No supervised voxels in this batch")
    if normalizer is not None:
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Loss normalizer must be finite and positive")
        denominator = normalizer
    return (
        (logits.float().sigmoid() - target.float()).square() * weight.float()
    ).sum() / denominator
