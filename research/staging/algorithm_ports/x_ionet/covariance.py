"""Declared full-SPD reconstruction and original-weight Huber/Gaussian objective.

No network, registry or uncertainty-metric bridge. Cholesky parameterization,
physical diagonal floor and batch reduction are explicit engineering contracts,
not recovered author settings. Covariance remains full, in displacement m².
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def full_spd_from_lower(raw: torch.Tensor, *, minimum_std_m: float) -> dict:
    """Six entries [L00,L10,L11,L20,L21,L22]; diagonal=Softplus(raw)+given floor."""
    if not math.isfinite(minimum_std_m) or minimum_std_m <= 0:
        raise ValueError("an explicit positive standard-deviation floor in meters is required")
    if (not isinstance(raw, torch.Tensor) or raw.ndim < 2 or raw.shape[-1] != 6
            or not raw.is_floating_point() or not torch.isfinite(raw).all()):
        raise ValueError("raw lower factor must be finite floating (...,6)")
    zero = torch.zeros_like(raw[..., 0])
    d0 = F.softplus(raw[..., 0]) + minimum_std_m
    d1 = F.softplus(raw[..., 2]) + minimum_std_m
    d2 = F.softplus(raw[..., 5]) + minimum_std_m
    lower = torch.stack((torch.stack((d0, zero, zero), -1),
                         torch.stack((raw[..., 1], d1, zero), -1),
                         torch.stack((raw[..., 3], raw[..., 4], d2), -1)), -2)
    covariance = lower @ lower.transpose(-1, -2)
    if (not (lower.diagonal(dim1=-2, dim2=-1) > 0).all()
            or not torch.isfinite(covariance).all()):
        raise ValueError("factor floor/entries are not representable in the chosen floating dtype")
    _, info = torch.linalg.cholesky_ex(covariance)
    if (info != 0).any():
        raise ValueError("reconstructed covariance lost numeric SPD; no fake repair jitter")
    return {"factor": lower, "covariance": covariance}


def huber_gaussian_loss(prediction: torch.Tensor, target: torch.Tensor,
                        covariance: torch.Tensor, *, reduction: str) -> dict:
    """Full matrix equation-5--7 objective, delta=.005 and Gaussian weight=1e-4.

    Coordinates are summed per sample. Batch SUM or mean_samples must be named;
    neither is presented as a recovered author reduction. No phase switch, angle,
    diagonal logstd, hidden inverse or covariance jitter is substituted.
    """
    if (prediction.ndim < 2 or prediction.shape[-1] != 3 or not prediction.numel()
            or target.shape != prediction.shape
            or covariance.shape != (*prediction.shape[:-1], 3, 3)):
        raise ValueError("expected matching displacement (...,3) and covariance (...,3,3)")
    if reduction not in ("sum", "mean_samples"):
        raise ValueError("explicit reduction must be sum or mean_samples")
    if any(not t.is_floating_point() or t.dtype != prediction.dtype
           or t.device != prediction.device or not torch.isfinite(t).all()
           for t in (prediction, target, covariance)):
        raise ValueError("loss arrays must share finite floating dtype/device")
    if not torch.allclose(covariance, covariance.transpose(-1, -2), rtol=0, atol=1e-10):
        raise ValueError("full covariance must be symmetric")
    lower, info = torch.linalg.cholesky_ex(covariance)
    if (info != 0).any():
        raise ValueError("full covariance must be strictly positive definite; no fake jitter")
    residual = prediction - target
    magnitude = residual.abs()
    huber = torch.where(magnitude <= 0.005, 0.5 * residual.square(),
                        0.005 * magnitude - 0.5 * 0.005 ** 2).sum(-1)
    whitened = torch.linalg.solve_triangular(lower, residual.unsqueeze(-1), upper=False)
    gaussian = (0.5 * whitened.square().sum((-2, -1))
                + torch.log(lower.diagonal(dim1=-2, dim2=-1)).sum(-1))
    reduce = torch.sum if reduction == "sum" else torch.mean
    h, g = reduce(huber), reduce(gaussian)
    return {"loss": h + 1e-4 * g, "huber": h, "gaussian": g,
            "huber_per_sample": huber, "gaussian_per_sample": gaussian}
