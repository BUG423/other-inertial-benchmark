"""Supervised released-grid pair cache and crop offset, not a training loop.

No model, IMU, clock or filesystem is read here. Reference labels belong only
on the supervised side. Float64 labels retain the accepted port's disclosed
precision adaptation; no complete legacy bit-exact reproduction is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
import numpy as np

DECIMATION = 10
START_STEP_DECIMATED = 10
DISTANCES_METRES = tuple(range(100, 801, 100))


class NoCachedPairs(ValueError):
    """No admitted released distance pair; never substitute a different loss."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} needs an explicit integer >= {minimum}")
    return value


def _owned(value: Any, shape: tuple, dtype: str, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} requires {shape} {dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    result = array.copy(); result.setflags(write=False)
    return result


def _span_admitted(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    # Every original sample in the closed pair span is checked, including
    # non-decimated samples. Endpoint-only validity would hide gaps.
    invalid = np.concatenate((np.zeros(1, dtype=np.int64), (~mask).cumsum(dtype=np.int64)))
    return invalid[ends + 1] == invalid[starts]


@dataclass(frozen=True)
class CachedReferencePairs:
    raw_index_origin: int
    source_length: int
    start_raw: np.ndarray
    end_raw: np.ndarray
    target_local: np.ndarray
    requested_distances_m: np.ndarray
    valid_reference: np.ndarray

    def __post_init__(self) -> None:
        _integer(self.raw_index_origin, "raw_index_origin")
        _integer(self.source_length, "source_length", 2)
        if self.raw_index_origin > np.iinfo(np.int64).max - self.source_length + 1:
            raise ValueError("source raw-index extent overflows int64")
        n = len(np.asarray(self.start_raw))
        if n == 0:
            raise NoCachedPairs("no admitted 100..800m source-cache pairs")
        for name, shape, dtype in (("start_raw", (n,), "int64"), ("end_raw", (n,), "int64"),
                ("target_local", (n, 3), "float64"), ("requested_distances_m", (n,), "float64"),
                ("valid_reference", (self.source_length,), "bool")):
            object.__setattr__(self, name, _owned(getattr(self, name), shape, dtype, name))
        start = self.start_raw - self.raw_index_origin
        end = self.end_raw - self.raw_index_origin
        if ((start < 0).any() or (end >= self.source_length).any() or (end <= start).any()
                or (start % 100 != 0).any() or (end % DECIMATION != 0).any()):
            raise ValueError("cache indices must retain the released recording-anchored grid")
        if not np.isin(self.requested_distances_m, DISTANCES_METRES).all():
            raise ValueError("only source requested distances100..800m are allowed")
        if not _span_admitted(self.valid_reference, start, end).all():
            raise ValueError("cached pair crosses an unadmitted original reference sample")
        if not (np.linalg.norm(self.target_local, axis=1) > 0).all():
            raise ValueError("zero reference chord cannot normalize released loss")


@dataclass(frozen=True)
class CroppedReferencePairs:
    crop_start_raw: int
    crop_length: int
    source_pair_indices: np.ndarray
    start: np.ndarray
    end: np.ndarray
    target_local: np.ndarray
    requested_distances_m: np.ndarray

    def __post_init__(self) -> None:
        _integer(self.crop_start_raw, "crop_start_raw")
        _integer(self.crop_length, "crop_length", 10)
        n = len(np.asarray(self.start))
        if n == 0:
            raise NoCachedPairs("crop has no admitted source-cache pair; objective unavailable")
        for name, shape, dtype in (("source_pair_indices", (n,), "int64"), ("start", (n,), "int64"),
                ("end", (n,), "int64"), ("target_local", (n, 3), "float64"),
                ("requested_distances_m", (n,), "float64")):
            object.__setattr__(self, name, _owned(getattr(self, name), shape, dtype, name))
        # Retain precompute_lost's floor(N/10) endpoint bound, even when N is not
        # divisible by ten. It is stricter than merely end_raw < crop_end.
        limit = DECIMATION * (self.crop_length // DECIMATION)
        if ((self.source_pair_indices < 0).any() or (self.start < 0).any()
                or (self.end >= limit).any() or (self.end <= self.start).any()
                or (self.start % DECIMATION != 0).any() or (self.end % DECIMATION != 0).any()):
            raise ValueError("crop indices differ from released decimated-endpoint bounds")
        if not np.isin(self.requested_distances_m, DISTANCES_METRES).all():
            raise ValueError("invalid source requested distance")
        if not (np.linalg.norm(self.target_local, axis=1) > 0).all():
            raise ValueError("zero chord cannot normalize released loss")


def cache_reference_pairs(reference_R: Any, reference_p: Any, *, raw_index_origin: int = 0,
                          valid_reference: Optional[Any] = None) -> CachedReferencePairs:
    """Build once on a whole admitted recording/continuous-segment reference.

    Grid anchors are this source's origin and do not restart for later crops or
    validity gaps. The optional mask only excludes existing pairs; this is an
    explicit quality adaptation since released source has no such mask.
    """
    origin = _integer(raw_index_origin, "raw_index_origin")
    p, R = np.asarray(reference_p), np.asarray(reference_R)
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 2 or p.dtype != np.dtype("float64"):
        raise ValueError("whole-source finite float64 XYZ reference required")
    n = len(p)
    if origin > np.iinfo(np.int64).max - n + 1:
        raise ValueError("source raw-index extent overflows int64")
    p = _owned(p, (n, 3), "float64", "reference_p")
    R = _owned(R, (n, 3, 3), "float64", "reference_R")
    if (not np.allclose(R.swapaxes(-2, -1) @ R, np.eye(3), rtol=0, atol=1e-8)
            or not np.allclose(np.linalg.det(R), 1., rtol=0, atol=1e-8)):
        raise ValueError("reference rotations must already be SO(3); no projection")
    mask = np.ones(n, dtype=bool) if valid_reference is None else _owned(
        valid_reference, (n,), "bool", "valid_reference")
    if not mask.any():
        raise NoCachedPairs("reference admission mask is entirely invalid")
    decimated = p[::DECIMATION]
    distances = np.concatenate((np.zeros(1), np.linalg.norm(np.diff(decimated, axis=0), axis=1).cumsum()))
    firsts, lasts, requested = [], [], []
    for k in range(max(0, len(decimated) // START_STEP_DECIMATED - 1)):
        first = START_STEP_DECIMATED * k
        for length in DISTANCES_METRES:
            if distances[first] + length > distances[-1]:
                continue
            last = first + int(np.searchsorted(distances[first:], distances[first] + length, side="left"))
            firsts.append(first * DECIMATION); lasts.append(last * DECIMATION); requested.append(length)
    if not firsts:
        raise NoCachedPairs("whole source has no released100..800m pair")
    starts, ends = np.array(firsts, dtype=np.int64), np.array(lasts, dtype=np.int64)
    keep = _span_admitted(mask, starts, ends)
    if not keep.any():
        raise NoCachedPairs("all released source-cache pairs cross reference-quality gaps")
    starts, ends = starts[keep], ends[keep]
    target = np.einsum("nji,nj->ni", R[starts], p[ends] - p[starts])
    return CachedReferencePairs(origin, n, starts + origin, ends + origin,
                                target, np.asarray(requested, dtype=np.float64)[keep], mask)


def crop_cached_pairs(cache: CachedReferencePairs, *, crop_start_raw: int, crop_length: int,
                      valid_pair_samples: Optional[Any] = None) -> CroppedReferencePairs:
    """Select existing global pairs, subtract the source raw-index crop offset.

    Labels remain cached; never rebuild a one-second start grid in this crop.
    valid_pair_samples is supervised pair eligibility only, not a model input.
    """
    if not isinstance(cache, CachedReferencePairs):
        raise TypeError("typed source cache required")
    first = _integer(crop_start_raw, "crop_start_raw")
    length = _integer(crop_length, "crop_length", 10)
    relative = first - cache.raw_index_origin
    if relative < 0 or relative % DECIMATION or relative + length > cache.source_length:
        raise ValueError("crop must be inside source and anchored at source N0=10*k")
    starts, ends = cache.start_raw - first, cache.end_raw - first
    limit = DECIMATION * (length // DECIMATION)
    keep = (starts >= 0) & (ends < limit)
    selected = np.flatnonzero(keep)
    if valid_pair_samples is not None and len(selected):
        mask = _owned(valid_pair_samples, (length,), "bool", "valid_pair_samples")
        selected = selected[_span_admitted(mask, starts[selected], ends[selected])]
    elif valid_pair_samples is not None:
        _owned(valid_pair_samples, (length,), "bool", "valid_pair_samples")
    if not len(selected):
        raise NoCachedPairs("crop has no admitted cached distance pair")
    return CroppedReferencePairs(first, length, selected.astype(np.int64), starts[selected], ends[selected],
                                 cache.target_local[selected], cache.requested_distances_m[selected])


def as_loss_pairs(crops: Sequence[CroppedReferencePairs], *, device: Any = "cpu") -> tuple:
    """Labels-only conversion for accepted relative_pose_sum; no loss rewrite."""
    import torch
    from .relative_pose import RelativePosePairs
    if not crops or any(not isinstance(c, CroppedReferencePairs) for c in crops):
        raise NoCachedPairs("nonempty typed per-recording crop labels required")
    return tuple(RelativePosePairs(
        torch.tensor(c.start.copy(), dtype=torch.long, device=device),
        torch.tensor(c.end.copy(), dtype=torch.long, device=device),
        torch.tensor(c.target_local.copy(), dtype=torch.float64, device=device),
        torch.tensor(c.requested_distances_m.copy(), dtype=torch.float64, device=device)) for c in crops)
