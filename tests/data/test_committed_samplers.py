"""Scientific cursor/RNG isolation and checkpoint tests; run only on the server."""

import io
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inertial_benchmark.data.samplers import (  # noqa: E402
    CommittedBatchSampler,
    UnpairedBatchStream,
    independent_stream_seed,
)


def sampler(size=7, batch=2, *, stream="source", seed=1735231569, **kw):
    return CommittedBatchSampler(size, batch, seed=seed, stream=stream, **kw)


def pair_stream(source_size=7, target_size=4, source_batch=2, target_batch=3, **kw):
    return UnpairedBatchStream(
        sampler(source_size, source_batch, stream="source", **kw),
        sampler(target_size, target_batch, stream="target", **kw),
    )


def equal_state(a, b):
    assert set(a) == set(b)
    for key, value in a.items():
        if isinstance(value, dict):
            equal_state(value, b[key])
        elif isinstance(value, torch.Tensor):
            assert torch.equal(value, b[key]), key
        else:
            assert value == b[key], key


def commit(stream, pair):
    stream.commit_pair(pair, generator_steps=stream.generator_steps + 1,
                       discriminator_steps=stream.discriminator_steps + 1)


def test_independent_streams_do_not_touch_global_python_numpy_torch_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    stream = pair_stream(source_size=100, target_size=100)
    pair = stream.reserve_pair()
    assert pair.source.indices != pair.target.indices
    commit(stream, pair)
    stream.state_dict()
    stream.reserve_pair()
    assert random.getstate() == python_state
    current_np = np.random.get_state()
    assert current_np[0] == numpy_state[0]
    np.testing.assert_array_equal(current_np[1], numpy_state[1])
    assert current_np[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert independent_stream_seed(1735231569, "source", 0) != independent_stream_seed(
        1735231569, "target", 0)
    assert independent_stream_seed(1735231569, "source", 0) != independent_stream_seed(
        1735231569, "source", 1)


def test_reservation_is_idempotent_and_checkpoint_contains_only_committed_state():
    stream = pair_stream()
    before = stream.state_dict()
    pair = stream.reserve_pair()
    assert stream.reserve_pair() == pair
    assert stream.source.cursor == stream.target.cursor == stream.pairs == 0
    equal_state(before, stream.state_dict())
    stream.rollback()
    assert stream.reserve_pair() == pair
    equal_state(before, stream.state_dict())


@pytest.mark.parametrize("g,d", [(0, 1), (1, 0), (0, 0), (2, 1), (1, 2), (True, 1)])
def test_half_pair_or_unproven_step_counts_cannot_commit(g, d):
    stream = pair_stream()
    before = stream.state_dict()
    pair = stream.reserve_pair()
    with pytest.raises(ValueError):
        stream.commit_pair(pair, generator_steps=g, discriminator_steps=d)
    equal_state(before, stream.state_dict())


def test_commit_both_cursors_once_and_stale_tickets_rejected():
    stream = pair_stream()
    pair = stream.reserve_pair()
    commit(stream, pair)
    assert stream.source.cursor == 2 and stream.target.cursor == 3
    assert stream.pairs == stream.generator_steps == stream.discriminator_steps == 1
    before = stream.state_dict()
    with pytest.raises(ValueError):
        stream.commit_pair(pair, generator_steps=2, discriminator_steps=2)
    equal_state(before, stream.state_dict())


def test_failed_second_reservation_validation_does_not_partially_commit_source():
    stream = pair_stream()
    pair = stream.reserve_pair()
    stream.target.rollback()  # wrong protocol use; should fail before *either* commit
    before = stream.state_dict()
    with pytest.raises(ValueError, match="reservation"):
        commit(stream, pair)
    equal_state(before, stream.state_dict())


def test_target_cycle_preparation_is_not_committed_and_resume_replays_it():
    stream = pair_stream(source_size=9, target_size=2, source_batch=2, target_batch=3)
    commit(stream, stream.reserve_pair())
    assert stream.target.exhausted and stream.target.cycle == 0
    checkpoint = stream.state_dict()
    pending = stream.reserve_pair()
    assert pending.target.cycle == 1 and stream.target.cycle == 0
    equal_state(checkpoint, stream.state_dict())
    resumed = pair_stream(source_size=9, target_size=2, source_batch=2, target_batch=3)
    resumed.load_state_dict(checkpoint)
    assert resumed.reserve_pair() == pending
    commit(stream, pending)
    commit(resumed, resumed.reserve_pair())
    equal_state(stream.state_dict(), resumed.state_dict())


def test_partial_epoch_resume_all_next_pairs_and_next_generator_draw_exact():
    uninterrupted = pair_stream(source_size=11, target_size=5)
    commit(uninterrupted, uninterrupted.reserve_pair())
    commit(uninterrupted, uninterrupted.reserve_pair())
    state = uninterrupted.state_dict()
    resumed = pair_stream(source_size=11, target_size=5)
    torch_rng = torch.get_rng_state().clone()
    resumed.load_state_dict(state)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    source_seen = []
    while (pair := uninterrupted.reserve_pair()) is not None:
        restored = resumed.reserve_pair()
        assert pair == restored
        source_seen.extend(pair.source.indices)
        commit(uninterrupted, pair)
        commit(resumed, restored)
        equal_state(uninterrupted.state_dict(), resumed.state_dict())
    assert resumed.reserve_pair() is None
    assert len(source_seen) == 7 and len(set(source_seen)) == 7
    for original, restored in ((uninterrupted.source, resumed.source),
                               (uninterrupted.target, resumed.target)):
        a, b = torch.Generator(), torch.Generator()
        a.set_state(original.state_dict()["generator_state"])
        b.set_state(restored.state_dict()["generator_state"])
        assert torch.equal(torch.rand(8, generator=a), torch.rand(8, generator=b))


def test_epoch_advance_preserves_global_step_counts_and_resets_target_cycle():
    stream = pair_stream(source_size=5, target_size=2)
    while (pair := stream.reserve_pair()) is not None:
        commit(stream, pair)
    assert stream.pairs == 3 and stream.target.cycle == 2
    stream.start_epoch(1)
    assert stream.source.epoch == stream.target.epoch == 1
    assert stream.source.cursor == stream.target.cursor == stream.target.cycle == 0
    assert stream.pairs == stream.generator_steps == stream.discriminator_steps == 3
    state = stream.state_dict()
    restored = pair_stream(source_size=5, target_size=2)
    restored.load_state_dict(state)
    assert restored.reserve_pair() == stream.reserve_pair()
    commit(stream, stream.reserve_pair())
    commit(restored, restored.reserve_pair())
    equal_state(stream.state_dict(), restored.state_dict())


def test_epoch_advance_refuses_partial_pending_and_skipped_epoch():
    stream = pair_stream(source_size=2, target_size=3)
    with pytest.raises(ValueError):
        stream.start_epoch(1)
    pending = stream.reserve_pair()
    with pytest.raises(ValueError):
        stream.start_epoch(1)
    commit(stream, pending)
    with pytest.raises(ValueError):
        stream.start_epoch(2)
    stream.start_epoch(1)


def test_weights_only_checkpoint_roundtrip_and_pending_reservation_not_serialized():
    stream = pair_stream()
    commit(stream, stream.reserve_pair())
    next_pair = stream.reserve_pair()
    output = io.BytesIO()
    torch.save(stream.state_dict(), output)
    output.seek(0)
    state = torch.load(output, map_location="cpu", weights_only=True)
    restored = pair_stream()
    restored.load_state_dict(state)
    assert restored.reserve_pair() == next_pair
    equal_state(stream.state_dict(), restored.state_dict())


def test_returned_state_is_detached_from_live_sampler():
    stream = pair_stream()
    before = stream.state_dict()
    state = stream.state_dict()
    state["source"]["permutation"].zero_()
    state["target"]["generator_state"].zero_()
    equal_state(before, stream.state_dict())


@pytest.mark.parametrize("mutation", [
    "contract", "cursor", "order", "rng", "half_pair", "exposure",
])
def test_checkpoint_corruption_refused_without_mutating_either_sampler(mutation):
    stream = pair_stream()
    commit(stream, stream.reserve_pair())
    before = stream.state_dict()
    state = deepcopy(before)
    if mutation == "contract":
        state["target"]["seed"] += 1
    elif mutation == "cursor":
        state["target"]["cursor"] = 2  # batch size 3; not a committed boundary
    elif mutation == "order":
        state["target"]["permutation"].zero_()
    elif mutation == "rng":
        state["target"]["generator_state"].zero_()
    elif mutation == "half_pair":
        state["discriminator_steps"] += 1
    else:
        state["target"]["cursor"] = 4  # valid boundary but implies two target batches
    with pytest.raises(ValueError):
        stream.load_state_dict(state)
    equal_state(before, stream.state_dict())


def test_iterator_yield_does_not_commit_and_prefetch_is_explicit_error():
    sample = sampler()
    iterator = iter(sample)
    first = next(iterator)
    assert len(first) == 2 and sample.cursor == 0
    with pytest.raises(RuntimeError, match="prefetched"):
        next(iterator)
    sample.rollback()
    for batch in sample:
        assert batch == list(sample.reserve().indices)
        sample.commit(sample.reserve())
    assert sample.exhausted


def test_drop_last_and_short_final_target_batches_are_explicit():
    stream = pair_stream(source_size=7, target_size=4)
    lengths = []
    while (pair := stream.reserve_pair()) is not None:
        lengths.append((len(pair.source.indices), len(pair.target.indices)))
        commit(stream, pair)
    assert lengths == [(2, 3), (2, 1), (2, 3), (1, 1)]
    sample = sampler(size=7, batch=2, drop_last=True)
    seen = []
    while (ticket := sample.reserve()) is not None:
        seen.extend(ticket.indices)
        sample.commit(ticket)
    assert len(seen) == 6 and sample.cursor == sample.limit == 6


@pytest.mark.parametrize("kwargs", [{"workers": 1}, {"size": 0}, {"batch_size": 0},
                                   {"seed": -1}, {"stream": ""},
                                   {"size": 1, "batch_size": 2, "drop_last": True}])
def test_invalid_or_unqualified_sampler_contract_refused(kwargs):
    args = dict(size=7, batch_size=2, seed=1735231569, stream="source")
    args.update(kwargs)
    with pytest.raises(ValueError):
        CommittedBatchSampler(**args)


def test_unpaired_wrapper_requires_distinct_streams_and_same_experiment_seed():
    with pytest.raises(ValueError, match="distinct"):
        UnpairedBatchStream(sampler(), sampler())
    with pytest.raises(ValueError, match="persisted"):
        UnpairedBatchStream(sampler(), sampler(stream="target", seed=843628571))
    sample = sampler()
    sample.reserve()
    with pytest.raises(ValueError, match="fresh"):
        UnpairedBatchStream(sample, sampler(stream="target"))


def test_wrong_reservation_and_none_are_rejected_as_contract_errors():
    stream = pair_stream()
    pair = stream.reserve_pair()
    bad = replace(pair, source=replace(pair.source, indices=(0,)))
    for wrong in (None, bad):
        with pytest.raises(ValueError):
            stream.commit_pair(wrong, generator_steps=1, discriminator_steps=1)
    assert stream.source.cursor == stream.target.cursor == 0
