"""Released 100..800 m pair rule and normalized relative-pose MSE SUM.

Only supervised training/evaluation may call this module with reference pose.
The filter forward has no such arguments. Sampling constants assume the
released raw 100 Hz clock; nominal-grid/jitter provenance is a data contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class NoEligiblePairs(ValueError):
    """Short/slow recording cannot exercise the released distance objective."""


@dataclass(frozen=True)
class RelativePosePairs:
    start: Tensor  # original raw 100 Hz indices, each divisible by ten
    end: Tensor
    target_local: Tensor
    path_lengths: Tensor  # requested 100..800 m, NOT the loss denominator


def build_pairs(reference_R: Tensor, reference_p: Tensor) -> tuple[RelativePosePairs, ...]:
    if reference_R.ndim != 4 or reference_p.shape != (*reference_R.shape[:2], 3):
        raise ValueError("reference R/p require (B,T,3,3)/(B,T,3)")
    if reference_R.shape[-2:] != (3, 3) or not bool(torch.isfinite(reference_R).all()):
        raise ValueError("finite reference rotations required")
    if not bool(torch.isfinite(reference_p).all()):
        raise ValueError("finite reference position required; split invalid gaps first")
    if reference_p.dtype != torch.float64 or reference_R.dtype != torch.float64:
        raise ValueError("float64 supervised pose required")
    if reference_p.device != reference_R.device:
        raise ValueError("reference pose device mismatch")
    result = []
    for R, p in zip(reference_R.detach(), reference_p.detach()):
        R, p = R[::10], p[::10]
        distances = torch.cat((p.new_zeros(1), torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1).cumsum(0)))
        starts, ends, lengths = [], [], []
        # Precisely source floor(N_down/10)-1, not all possible starts.
        for k in range(max(0, len(p) // 10 - 1)):
            first = 10 * k
            for length in range(100, 801, 100):
                threshold = distances[first] + length
                if bool(threshold > distances[-1]):
                    continue
                last = int(torch.searchsorted(distances[first:], threshold)) + first
                starts.append(first)
                ends.append(last)
                lengths.append(length)
        if not starts:
            raise NoEligiblePairs("recording has no released 100..800 m pairs; do not replace the objective")
        first = torch.tensor(starts, dtype=torch.long, device=p.device)
        last = torch.tensor(ends, dtype=torch.long, device=p.device)
        target = (R[first].mT @ (p[last] - p[first])[..., None]).squeeze(-1)
        if not bool((torch.linalg.vector_norm(target, dim=-1) > 0).all()):
            raise ValueError("zero GT chord cannot normalize released loss; no artificial epsilon")
        result.append(RelativePosePairs(first * 10, last * 10, target,
                                       p.new_tensor(lengths)))
    return tuple(result)


def relative_pose_sum(predicted_R: Tensor, predicted_p: Tensor,
                      pairs: tuple[RelativePosePairs, ...]) -> Tensor:
    if predicted_R.ndim != 4 or predicted_p.shape != (*predicted_R.shape[:2], 3):
        raise ValueError("predicted R/p layout mismatch")
    if len(pairs) != predicted_p.shape[0] or not pairs:
        raise NoEligiblePairs("non-empty per-recording pairs required")
    loss = predicted_p.new_zeros(())
    for R, p, item in zip(predicted_R, predicted_p, pairs):
        if not item.start.numel():
            raise NoEligiblePairs("empty pair set")
        if bool((item.start < 0).any()) or bool((item.end >= len(p)).any()):
            raise ValueError("pair not inside this supervised chunk")
        target = item.target_local.detach()
        denominator = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        if not bool((denominator > 0).all()):
            raise ValueError("zero chord denominator")
        predicted = (R[item.start].mT @ (p[item.end] - p[item.start])[..., None]).squeeze(-1)
        loss = loss + ((predicted / denominator - target / denominator).square()).sum()
    return loss
