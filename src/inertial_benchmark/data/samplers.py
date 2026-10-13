"""Committed-only sampling for a future unpaired D/G training protocol.

Reserve a pair, load its inputs, complete both optimizer steps, then commit the
pair. Iterator yield/prefetch is never a commit. Checkpoints contain committed
state only; an outstanding reservation can be replayed after failure. This leaf
does not roll back mutated model/optimizer state: a half-pair failure must abort
and restore the last complete-pair training checkpoint. Initial exact-resume
integration supports ``workers=0`` only, with one pending pair at a time.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import torch
from torch.utils.data import Sampler


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def independent_stream_seed(seed: int, stream: str, epoch: int, cycle: int = 0) -> int:
    """Stable stream seed; no Python hash/global RNG or seed-record mutation."""
    _integer(seed, "seed")
    _integer(epoch, "epoch")
    _integer(cycle, "cycle")
    if not isinstance(stream, str) or not stream or "\x00" in stream:
        raise ValueError("stream must be a nonempty name without NUL")
    payload = f"ipb-committed-sampler-v1\0{seed}\0{stream}\0{epoch}\0{cycle}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class BatchReservation:
    """Indices and augmentation coordinates for one uncommitted batch."""

    stream: str
    epoch: int
    cycle: int
    start: int
    stop: int
    indices: tuple
    token: str


class CommittedBatchSampler(Sampler):
    """One deterministic permutation/cycle whose cursor advances only on commit.

    A ``cycle=True`` reservation may prepare the next target permutation, but the
    new cycle/generator state remains pending until commit. ``state_dict`` never
    serializes pending state. Independent CPU generators leave global Torch RNG
    untouched. Iterable use deliberately rejects uncommitted advancement instead
    of silently supporting DataLoader prefetch; a paired protocol should prefer
    ``UnpairedBatchStream.reserve_pair`` and direct indexed loading.
    """

    def __init__(
        self,
        size: int,
        batch_size: int,
        *,
        seed: int,
        stream: str,
        epoch: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        workers: int = 0,
    ) -> None:
        self.size = _integer(size, "size", 1)
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.seed = _integer(seed, "seed")
        self.epoch = _integer(epoch, "epoch")
        self.stream = stream
        independent_stream_seed(seed, stream, epoch)
        if workers != 0:
            raise ValueError("committed exact-resume sampling currently requires workers=0")
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise ValueError("shuffle/drop_last must be booleans")
        if drop_last and size < batch_size:
            raise ValueError("drop_last would remove every batch")
        self.shuffle, self.drop_last = shuffle, drop_last
        self.cycle, self.cursor = 0, 0
        self._generator = torch.Generator(device="cpu")
        self._permutation, state = self._new_cycle(self.epoch, self.cycle)
        self._generator.set_state(state)
        self._pending: Optional[BatchReservation] = None
        self._pending_permutation: Optional[torch.Tensor] = None
        self._pending_generator_state: Optional[torch.Tensor] = None

    @property
    def limit(self) -> int:
        return (self.size // self.batch_size) * self.batch_size if self.drop_last else self.size

    @property
    def exhausted(self) -> bool:
        return self.cursor == self.limit

    def _new_cycle(self, epoch: int, cycle: int) -> tuple:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(independent_stream_seed(self.seed, self.stream, epoch, cycle))
        order = (torch.randperm(self.size, generator=generator, device="cpu") if self.shuffle
                 else torch.arange(self.size, dtype=torch.int64, device="cpu"))
        return order, generator.get_state().clone()

    def reserve(self, *, cycle: bool = False) -> Optional[BatchReservation]:
        """Return the same pending batch until committed/rolled back; no state advance."""
        if not isinstance(cycle, bool):
            raise ValueError("cycle must be a boolean")
        if self._pending is not None:
            return self._pending
        if self.exhausted and not cycle:
            return None
        number = self.cycle + 1 if self.exhausted else self.cycle
        start = 0 if self.exhausted else self.cursor
        if self.exhausted:
            order, generator_state = self._new_cycle(self.epoch, number)
        else:
            order = self._permutation
            generator_state = self._generator.get_state()
        stop = min(start + self.batch_size, self.limit)
        indices = tuple(order[start:stop].tolist())
        payload = repr((self.seed, self.stream, self.epoch, number, start, stop, indices)).encode()
        reservation = BatchReservation(self.stream, self.epoch, number, start, stop, indices,
                                       hashlib.sha256(payload).hexdigest())
        self._pending = reservation
        self._pending_permutation = order
        self._pending_generator_state = generator_state
        return reservation

    def _validate_commit(self, reservation: BatchReservation) -> None:
        if not isinstance(reservation, BatchReservation) or self._pending != reservation:
            raise ValueError("commit requires this sampler's current reservation")

    def _apply_commit(self) -> None:
        reservation = self._pending
        # Called only after validation (for BOTH streams in an atomic pair).
        self._permutation = self._pending_permutation
        self._generator.set_state(self._pending_generator_state)
        self.cycle, self.cursor = reservation.cycle, reservation.stop
        self.rollback()

    def commit(self, reservation: BatchReservation) -> None:
        """Single-stream primitive; paired training must use commit_pair instead."""
        self._validate_commit(reservation)
        self._apply_commit()

    def rollback(self) -> None:
        """Discard reservation only; weights/optimizers require training-checkpoint restore."""
        self._pending = self._pending_permutation = self._pending_generator_state = None

    def start_epoch(self, epoch: int) -> None:
        """Advance a complete epoch once; partial-resume must load state instead."""
        _integer(epoch, "epoch")
        if not self.exhausted or self._pending is not None or epoch != self.epoch + 1:
            raise ValueError("start_epoch requires a fully committed previous epoch and epoch+1")
        order, state = self._new_cycle(epoch, 0)
        self._permutation = order
        self._generator.set_state(state)
        self.epoch, self.cycle, self.cursor = epoch, 0, 0

    def state_dict(self) -> dict:
        """Basic types + CPU tensors compatible with weights_only checkpoint loading."""
        return {"schema": "ipb-committed-batch-v1", "size": self.size,
                "batch_size": self.batch_size, "seed": self.seed, "stream": self.stream,
                "shuffle": self.shuffle, "drop_last": self.drop_last,
                "epoch": self.epoch, "cycle": self.cycle, "cursor": self.cursor,
                "permutation": self._permutation.clone(),
                "generator_state": self._generator.get_state().clone()}

    def _validated_state(self, state: Mapping[str, Any]) -> tuple:
        if not isinstance(state, Mapping) or state.get("schema") != "ipb-committed-batch-v1":
            raise ValueError("unknown committed sampler checkpoint schema")
        for key in ("size", "batch_size", "seed", "stream", "shuffle", "drop_last"):
            if (type(state.get(key)) is not type(getattr(self, key))
                    or state[key] != getattr(self, key)):
                raise ValueError(f"sampler checkpoint contract differs: {key}")
        epoch = _integer(state.get("epoch"), "epoch")
        cycle = _integer(state.get("cycle"), "cycle")
        cursor = _integer(state.get("cursor"), "cursor")
        if cursor > self.limit or (cursor != self.limit and cursor % self.batch_size):
            raise ValueError("checkpoint cursor is not a committed batch boundary")
        order, rng = state.get("permutation"), state.get("generator_state")
        if (not isinstance(order, torch.Tensor) or order.device.type != "cpu"
                or order.dtype != torch.int64 or tuple(order.shape) != (self.size,)
                or not torch.equal(torch.sort(order).values,
                                   torch.arange(self.size, dtype=torch.int64, device="cpu"))):
            raise ValueError("checkpoint permutation is not a full CPU index permutation")
        if (not isinstance(rng, torch.Tensor) or rng.device.type != "cpu"
                or rng.dtype != torch.uint8):
            raise ValueError("checkpoint generator state must be a CPU byte tensor")
        expected_order, expected_rng = self._new_cycle(epoch, cycle)
        if not torch.equal(order, expected_order) or not torch.equal(rng, expected_rng):
            raise ValueError("sampler state does not match its independent seed/epoch/cycle")
        return epoch, cycle, cursor, order.clone(), rng.clone()

    def _apply_state(self, values: tuple) -> None:
        epoch, cycle, cursor, order, rng = values
        self._generator.set_state(rng)
        self.epoch, self.cycle, self.cursor, self._permutation = epoch, cycle, cursor, order
        self.rollback()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._apply_state(self._validated_state(state))

    def __len__(self) -> int:
        return math.ceil((self.limit - self.cursor) / self.batch_size)

    def __iter__(self) -> Iterator[list]:
        while not self.exhausted:
            ticket = self.reserve()
            yield list(ticket.indices)
            if self._pending is not None:
                raise RuntimeError(
                    "batch was prefetched/advanced before optimizer commit; use workers=0")


@dataclass(frozen=True)
class DomainPairReservation:
    """Independent batches; indices need not have equal counts or temporal pairing."""

    source: BatchReservation
    target: BatchReservation
    pair: int


class UnpairedBatchStream:
    """Atomically commit source/target cursors after a complete 1D+1G update pair.

    Real optimizer counters are supplied by the training protocol. These checks
    enforce the contract but cannot prove a dishonest caller performed a step.
    No global RNG/model/optimizer state is saved by this sampler leaf.
    """

    def __init__(self, source: CommittedBatchSampler, target: CommittedBatchSampler) -> None:
        if source is target or source.stream == target.stream:
            raise ValueError("source and target need distinct independent stream names")
        if source.seed != target.seed or source.epoch != target.epoch:
            raise ValueError("both domains must share the persisted experiment seed and epoch")
        if (source.cursor or target.cursor or source.cycle or target.cycle
                or source._pending is not None or target._pending is not None):
            raise ValueError("construct from fresh samplers, then load the paired checkpoint")
        self.source, self.target = source, target
        self.pairs = self.generator_steps = self.discriminator_steps = 0
        self._pending: Optional[DomainPairReservation] = None

    def reserve_pair(self) -> Optional[DomainPairReservation]:
        if self._pending is not None:
            return self._pending
        source = self.source.reserve()
        if source is None:
            return None
        target = self.target.reserve(cycle=True)
        self._pending = DomainPairReservation(source, target, self.pairs)
        return self._pending

    def commit_pair(self, pair: DomainPairReservation, *, generator_steps: int,
                    discriminator_steps: int) -> None:
        """Validate BOTH reservations/counters before changing either committed cursor."""
        _integer(generator_steps, "generator_steps")
        _integer(discriminator_steps, "discriminator_steps")
        if (not isinstance(pair, DomainPairReservation) or pair != self._pending
                or pair.pair != self.pairs):
            raise ValueError("commit requires the current unpaired reservation")
        if (generator_steps != self.generator_steps + 1
                or discriminator_steps != self.discriminator_steps + 1):
            raise ValueError("both G and D must complete exactly one real step before commit")
        self.source._validate_commit(pair.source)
        self.target._validate_commit(pair.target)
        self.source._apply_commit()
        self.target._apply_commit()
        self.pairs += 1
        self.generator_steps, self.discriminator_steps = generator_steps, discriminator_steps
        self._pending = None

    def rollback(self) -> None:
        self.source.rollback()
        self.target.rollback()
        self._pending = None

    def start_epoch(self, epoch: int) -> None:
        """Source exhausts once; target starts fresh regardless of its partial cycle."""
        if (self._pending is not None or self.source._pending is not None
                or self.target._pending is not None or not self.source.exhausted):
            raise ValueError("cannot advance an uncommitted/partial source epoch")
        _integer(epoch, "epoch")
        if epoch != self.source.epoch + 1 or epoch != self.target.epoch + 1:
            raise ValueError("paired epoch must advance exactly once")
        source_order, source_rng = self.source._new_cycle(epoch, 0)
        target_order, target_rng = self.target._new_cycle(epoch, 0)
        self.source._apply_state((epoch, 0, 0, source_order, source_rng))
        self.target._apply_state((epoch, 0, 0, target_order, target_rng))

    def state_dict(self) -> dict:
        return {"schema": "ipb-unpaired-stream-v1", "pairs": self.pairs,
                "generator_steps": self.generator_steps,
                "discriminator_steps": self.discriminator_steps,
                "source": self.source.state_dict(), "target": self.target.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "ipb-unpaired-stream-v1":
            raise ValueError("unknown unpaired stream checkpoint schema")
        pairs = _integer(state.get("pairs"), "pairs")
        g = _integer(state.get("generator_steps"), "generator_steps")
        d = _integer(state.get("discriminator_steps"), "discriminator_steps")
        if g != pairs or d != pairs:
            raise ValueError("checkpoint cannot represent a half-completed D/G pair")
        source = self.source._validated_state(state.get("source"))
        target = self.target._validated_state(state.get("target"))
        if source[0] != target[0]:
            raise ValueError("source/target checkpoint epochs differ")
        if source[1] != 0:
            raise ValueError("source must consume one pass per epoch without target-style cycling")
        # A pair consumes one source batch. Both stream exposure counts must agree
        # within this epoch, including a short final batch in every target cycle.
        source_batches = math.ceil(source[2] / self.source.batch_size)
        target_batches = (target[1] * math.ceil(self.target.limit / self.target.batch_size)
                          + math.ceil(target[2] / self.target.batch_size))
        if source_batches != target_batches or pairs < source_batches:
            raise ValueError("checkpoint source/target committed exposure counts disagree")
        self.source._apply_state(source)
        self.target._apply_state(target)
        self.pairs, self.generator_steps, self.discriminator_steps = pairs, g, d
        self._pending = None
