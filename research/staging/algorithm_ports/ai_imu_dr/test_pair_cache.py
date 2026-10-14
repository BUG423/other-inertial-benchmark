"""Server-only supervised indexing science; no raw data, Trainer or registration."""
from __future__ import annotations
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .pair_cache import (CachedReferencePairs, NoCachedPairs, as_loss_pairs,
                         cache_reference_pairs, crop_cached_pairs)
from .relative_pose import build_pairs, relative_pose_sum


def reference(length=3201, speed=.25, incline=.125):
    k = np.arange(length, dtype=np.float64)
    p = np.stack((speed * k, np.zeros(length), incline * k), -1)
    R = np.repeat(np.eye(3)[None], length, axis=0)
    return R, p


def numpy_oracle(R, p):
    """Independent raw index enumeration, float64 chord labels, no crop reset."""
    points = list(range(0, len(p), 10)); distance = [0.]
    for a, b in zip(points[:-1], points[1:]):
        distance.append(distance[-1] + float(np.sqrt(sum(float(x)**2 for x in p[b]-p[a]))))
    out = []
    for k in range(max(0, len(points)//10 - 1)):
        start = k*10
        for length in range(100,801,100):
            threshold = distance[start] + length
            if threshold > distance[-1]:continue
            end = next(j for j in range(start,len(points)) if distance[j] >= threshold)
            a,b=points[start],points[end]
            out.append((a,b,length,R[a].T @ (p[b]-p[a])))
    return out


def official_functions():
    root = os.environ.get("AI_IMU_DR_OFFICIAL_SOURCE")
    if root is None:
        raise RuntimeError("explicit source-only operator oracle path required; no silent skip")
    path = Path(root) / "src/train_torch_filter.py"
    expected = "92c3e01bc2602a544754103a5fc8be68c85cb08b31a6a663b50d4b69ccd5e4f9"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("pinned released train_torch_filter.py bytes differ")
    tree = ast.parse(path.read_text())
    names = {"compute_delta_p", "precompute_lost"}
    selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names
    env = {"np":np,"torch":torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),env)
    return SimpleNamespace(**{name:env[name] for name in names})


def test_cache_float64_reference_matches_independent_numpy_geometry_and_requested_lengths():
    R,p=reference()
    # Nontrivial reference rotation, not always an identity-frame fixture.
    theta=np.linspace(-.3,.4,len(p));c,s=np.cos(theta),np.sin(theta)
    R[:,0,0]=R[:,1,1]=c;R[:,0,1]=-s;R[:,1,0]=s
    expected=numpy_oracle(R,p);cache=cache_reference_pairs(R,p)
    np.testing.assert_array_equal(cache.start_raw,[x[0] for x in expected])
    np.testing.assert_array_equal(cache.end_raw,[x[1] for x in expected])
    np.testing.assert_array_equal(cache.requested_distances_m,[x[2] for x in expected])
    np.testing.assert_allclose(cache.target_local,[x[3] for x in expected],rtol=0,atol=2e-12)
    assert set(cache.requested_distances_m)==set(range(100,801,100))


@pytest.mark.parametrize("offset",[0,10,30,100])
def test_pinned_source_cache_and_crop_oracle_offsets_preserve_exact_pair_grid(offset):
    R,p=reference(3201,speed=.25,incline=0.)
    source=official_functions()
    # Released cache uses float32 R/delta labels; binary-coordinate fixture
    # makes these values exactly representable and allows exact numeric parity.
    raw=source.compute_delta_p(torch.from_numpy(R).float(),torch.from_numpy(p))
    cache=cache_reference_pairs(R,p);crop=crop_cached_pairs(cache,crop_start_raw=offset,crop_length=2000)
    np.testing.assert_array_equal(cache.start_raw,np.array(raw[0])*10)
    np.testing.assert_array_equal(cache.end_raw,np.array(raw[1])*10)
    np.testing.assert_array_equal(cache.target_local,raw[2].double().numpy())
    Rpred=torch.from_numpy(R[offset:offset+2000]).double()
    ppred=torch.from_numpy(p[offset:offset+2000]-p[offset]).double()
    got,target=source.precompute_lost(Rpred,ppred,raw,offset)
    pairs=as_loss_pairs((crop,))
    # Selected source normalized labels and predictions are exact in this case.
    denominator=np.linalg.norm(crop.target_local,axis=1,keepdims=True)
    np.testing.assert_array_equal(got.numpy(),crop.target_local/denominator)
    np.testing.assert_array_equal(target.numpy(),crop.target_local/denominator)
    assert relative_pose_sum(Rpred[None],ppred[None],pairs).item()==0.


@pytest.mark.parametrize("offset",[10,30,90])
def test_rebuilding_each_crop_is_a_real_nonsecond_grid_counterexample(offset):
    R,p=reference();cache=cache_reference_pairs(R,p)
    crop=crop_cached_pairs(cache,crop_start_raw=offset,crop_length=2000)
    rebuilt=build_pairs(torch.from_numpy(R[offset:offset+2000])[None],
                        torch.from_numpy(p[offset:offset+2000]-p[offset])[None])[0]
    assert int(crop.start[0]) == 100-offset
    assert int(rebuilt.start[0]) == 0
    assert not np.array_equal(crop.start,rebuilt.start.numpy())
    np.testing.assert_array_equal(crop.start+offset,cache.start_raw[crop.source_pair_indices])


def test_aligned_second_grid_rebuild_can_match_but_does_not_justify_nonsecond_reset():
    R,p=reference();crop=crop_cached_pairs(cache_reference_pairs(R,p),crop_start_raw=100,crop_length=2000)
    rebuilt=build_pairs(torch.from_numpy(R[100:2100])[None],torch.from_numpy(p[100:2100]-p[100])[None])[0]
    np.testing.assert_array_equal(crop.start,rebuilt.start.numpy())
    np.testing.assert_array_equal(crop.end,rebuilt.end.numpy())


@pytest.mark.parametrize("length",[200,201,209,210])
def test_source_floor_decimated_endpoint_rule_at_partial_crop_boundary(length):
    R,p=reference(301,speed=.5,incline=0.)
    cache=cache_reference_pairs(R,p)
    # Source cache has start0/end200. It is inside N201 raw values but original
    # precompute_lost excludes endpoint200 until floor(N/10) permits index20.
    if length < 210:
        with pytest.raises(NoCachedPairs):crop_cached_pairs(cache,crop_start_raw=0,crop_length=length)
    else:
        got=crop_cached_pairs(cache,crop_start_raw=0,crop_length=length)
        np.testing.assert_array_equal(got.start,[0]);np.testing.assert_array_equal(got.end,[200])


def test_source_single_pair_squeeze_fragility_is_disclosed_not_silently_copied():
    R,p=reference(211,speed=.5,incline=0.)
    source=official_functions();raw=source.compute_delta_p(torch.from_numpy(R).float(),torch.from_numpy(p))
    assert raw[2].shape == (3,)  # Unconditional squeeze destroys the K=1 axis.
    with pytest.raises((IndexError,RuntimeError)):
        source.precompute_lost(torch.from_numpy(R),torch.from_numpy(p),raw,0)
    crop=crop_cached_pairs(cache_reference_pairs(R,p),crop_start_raw=0,crop_length=210)
    assert crop.target_local.shape==(1,3)
    assert as_loss_pairs((crop,))[0].target_local.shape==(1,3)


def test_nondecimated_mask_gap_excludes_entire_pair_span_without_restart():
    R,p=reference();mask=np.ones(len(p),dtype=bool);mask[5]=False
    base=cache_reference_pairs(R,p);masked=cache_reference_pairs(R,p,valid_reference=mask)
    keep=np.array([mask[a:b+1].all() for a,b in zip(base.start_raw,base.end_raw)])
    np.testing.assert_array_equal(masked.start_raw,base.start_raw[keep])
    np.testing.assert_array_equal(masked.end_raw,base.end_raw[keep])
    assert not np.any(masked.start_raw==0) and masked.start_raw[0]==100
    crop=crop_cached_pairs(base,crop_start_raw=0,crop_length=2000,valid_pair_samples=mask[:2000])
    assert not np.any(crop.start==0) and crop.start[0]==100


@pytest.mark.parametrize("location",["cache","crop"])
def test_all_invalid_supervision_never_becomes_pass_or_alternative_loss(location):
    R,p=reference()
    with pytest.raises(NoCachedPairs):
        if location=="cache":cache_reference_pairs(R,p,valid_reference=np.zeros(len(p),dtype=bool))
        else:crop_cached_pairs(cache_reference_pairs(R,p),crop_start_raw=0,crop_length=2000,
                               valid_pair_samples=np.zeros(2000,dtype=bool))


def test_no_100m_path_and_zero_chord_are_explicit_admission_rejections():
    R,p=reference(301,speed=.01,incline=0.)
    with pytest.raises(NoCachedPairs):cache_reference_pairs(R,p)
    k=np.arange(1201);phase=k%400
    p=np.zeros((1201,3));p[:,0]=np.where(phase<=200,phase/4,(400-phase)/4)
    R=np.repeat(np.eye(3)[None],len(p),axis=0)
    with pytest.raises(ValueError,match="zero reference chord"):
        cache_reference_pairs(R,p)


@pytest.mark.parametrize("change",["position_float32","rotation_shape","rotation_reflection","nan","mask_integer"])
def test_source_geometry_and_quality_mask_contracts_refuse_invalid_inputs(change):
    R,p=reference();kw={}
    if change=="position_float32":p=p.astype(np.float32)
    elif change=="rotation_shape":R=R[:,:2,:2]
    elif change=="rotation_reflection":R[:,0,0]=-1
    elif change=="nan":p[5,0]=np.nan
    else:kw["valid_reference"]=np.ones(len(p),dtype=np.uint8)
    with pytest.raises(ValueError):cache_reference_pairs(R,p,**kw)


@pytest.mark.parametrize("first,length",[(True,2000),(-10,2000),(5,2000),(0,9),(0,4000),(0,2000.)])
def test_crop_integer_alignment_and_source_bounds_are_enforced(first,length):
    R,p=reference()
    with pytest.raises(ValueError):crop_cached_pairs(cache_reference_pairs(R,p),crop_start_raw=first,crop_length=length)


def test_nonzero_source_raw_origin_and_translation_do_not_reset_global_crop_labels():
    R,p=reference();a=cache_reference_pairs(R,p)
    b=cache_reference_pairs(R,p+np.array([4096.,2048.,1024.]),raw_index_origin=123000)
    ac=crop_cached_pairs(a,crop_start_raw=30,crop_length=2000)
    bc=crop_cached_pairs(b,crop_start_raw=123030,crop_length=2000)
    np.testing.assert_array_equal(ac.start,bc.start);np.testing.assert_array_equal(ac.end,bc.end)
    np.testing.assert_array_equal(ac.target_local,bc.target_local)


def test_cache_crop_labels_are_readonly_owned_and_do_not_consume_rng():
    R,p=reference();np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
    cache=cache_reference_pairs(R,p);crop=crop_cached_pairs(cache,crop_start_raw=10,crop_length=2000)
    saved=cache.target_local.copy();p[:]=0
    np.testing.assert_array_equal(cache.target_local,saved)
    for value in [cache.start_raw,cache.target_local,cache.valid_reference,crop.start,crop.target_local]:
        assert not value.flags.writeable
        with pytest.raises(ValueError):value.flat[0]=0
    np_after=np.random.get_state();assert np_before[0]==np_after[0]
    np.testing.assert_array_equal(np_before[1],np_after[1]);assert np_before[2:]==np_after[2:]
    assert torch.equal(torch_before,torch.get_rng_state())


def test_B2_label_conversion_keeps_chord_SUM_and_real_XYZ_gradients():
    R,p=reference();cache=cache_reference_pairs(R,p)
    crops=[crop_cached_pairs(cache,crop_start_raw=k,crop_length=2000) for k in [10,30]]
    labels=as_loss_pairs(crops);assert len(labels)==2 and all(not x.target_local.requires_grad for x in labels)
    Rpred=torch.stack([torch.from_numpy(R[k:k+2000]) for k in [10,30]])
    truth=torch.stack([torch.from_numpy(p[k:k+2000]-p[k]) for k in [10,30]])
    factor=torch.tensor([.97,1.04,.91],dtype=torch.float64,requires_grad=True)
    predicted=truth*factor
    got=relative_pose_sum(Rpred,predicted,labels)
    independent=got.new_zeros(())
    for b,c in enumerate(crops):
        target=torch.from_numpy(c.target_local.copy());chord=target.norm(dim=1,keepdim=True)
        estimate=(Rpred[b,c.start].mT @ (predicted[b,c.end]-predicted[b,c.start])[...,None]).squeeze(-1)
        independent+=((estimate/chord-target/chord)**2).sum()
    assert torch.equal(got,independent) and got>0
    assert not torch.equal(got,independent/sum(len(c.start) for c in crops))
    got.backward();assert torch.isfinite(factor.grad).all() and factor.grad[0]!=0 and factor.grad[2]!=0
    assert np.any(crops[0].target_local[:,2]!=0)


def test_source_raw_index_extent_never_wraps_int64():
    R,p=reference()
    with pytest.raises(ValueError,match="overflows int64"):
        cache_reference_pairs(R,p,raw_index_origin=int(np.iinfo(np.int64).max))
