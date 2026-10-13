"""Staging-only released aggregate-recording training step; no NIO wiring.

Released Adam groups, loss threshold and gradient clipping are explicit here.
Recording/chunk selection, train-only normalization and real KITTI acquisition
remain caller/data qualifications, not invented inside a forward method.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .filter import State
from .model import AIIMUDR, augment_released
from .relative_pose import build_pairs, relative_pose_sum


@dataclass(frozen=True)
class SupervisedRecording:
    timestamps_ns: Tensor
    body_imu: Tensor
    initial: State
    reference_R: Tensor
    reference_p: Tensor


def released_optimizer(model: AIIMUDR) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": model.initprocesscov_net.parameters(), "lr": 1e-4, "weight_decay": 0.},
        {"params": model.mes_net.cov_net.parameters(), "lr": 1e-4, "weight_decay": 1e-8},
        {"params": model.mes_net.cov_lin.parameters(), "lr": 1e-4, "weight_decay": 1e-8},
    ])


def released_epoch_step(model: AIIMUDR, optimizer: torch.optim.Optimizer,
                        recordings: tuple[SupervisedRecording, ...], *,
                        augmentation: bool) -> dict:
    """One aggregate Adam update, or an explicit skipped-update receipt.

    NoEligiblePairs is an admission error and is never converted into a pass.
    The caller must explicitly choose source training augmentation. This is
    the released recipe, not the formal benchmark's common recipe.
    """
    if not recordings:
        raise ValueError("non-empty supervised recording list required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accepted, rejected, losses = 0, [], []
    for index, recording in enumerate(recordings):
        if recording.body_imu.ndim != 3 or recording.body_imu.shape[0] != 1:
            raise ValueError("released per-recording loss gate requires B1; pass independent recordings")
        pairs = build_pairs(recording.reference_R, recording.reference_p)
        imu = augment_released(recording.body_imu) if augmentation else recording.body_imu
        output = model(recording.timestamps_ns, imu, recording.initial)
        loss = relative_pose_sum(output.R, output.p, pairs)
        if not bool(torch.isfinite(loss)) or bool(loss > 20):
            rejected.append({"recording_index": index, "reason": "nonfinite_or_loss_gt20"})
        else:
            losses.append(loss)
            accepted += 1
    if not losses:
        return {"optimizer_updates": 0, "accepted_recordings": 0, "rejected": rejected,
                "reason": "no_accepted_losses"}
    total = torch.stack(losses).sum()
    if bool(total == 0):
        return {"optimizer_updates": 0, "accepted_recordings": accepted, "rejected": rejected,
                "reason": "zero_aggregate_loss"}
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not bool(torch.isfinite(norm)) or bool(norm > 3):
        optimizer.zero_grad(set_to_none=True)
        return {"optimizer_updates": 0, "accepted_recordings": accepted, "rejected": rejected,
                "reason": "nonfinite_or_preclip_grad_norm_gt3", "loss": float(total.detach())}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"optimizer_updates": 1, "accepted_recordings": accepted, "rejected": rejected,
            "loss": float(total.detach()), "preclip_grad_norm": float(norm.detach())}
