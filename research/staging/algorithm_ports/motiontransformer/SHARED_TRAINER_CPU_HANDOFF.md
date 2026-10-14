# Shared Trainer CPU handoff — A2

The opt-in [typed binding](../../../../src/inertial_benchmark/engine/protocol_binding.py)
and [Motion factory](trainer_binding.py) use the existing
[Trainer](../../../../src/inertial_benchmark/engine/trainer.py). The programmatic
`protocol_factory` argument enables complete D→G transactions. The normal path
retains its single optimizer, epoch seeding, sequence calibration and checkpoint
handling. The protocol path creates two stage optimizers/schedulers and no unused
legacy loader or optimizer over all parameters. Source-only validation reuses
Validator/Predictor with an explicitly named polar-velocity diagnostic.

One isolated AutoDL CPU run actually passed **27 collected tests: 24 new protocol
nodes and 3 existing Trainer resume/loss-phase regressions**, with no failures,
errors or skips. The controller finished in 38.2083 seconds and released its
independent CPU lease. The actual receipt is `722b721319ac0cc9f3cbb5c4c17b898ccee8698a33c40669003f8b6a771a414e`;
JUnit is `7c14a48970e737aa83b8923fb6a9df07602b0d22e2bf2bd053e53f3d80181d65`.
The tested 465-file source manifest is
`bed8bbc5ef7bad82a2290ecd7de4fd8a93cd6c640eefdaea247b8aa0c57f194b`.
Root accepted the selected-native CPU engineering scope after verifying the
original receipts and guarded bytes. Root review:
`86b21dd430d069b1d1213d601b9cd8a4195088c05929db0f281548732f2b9680`.
This acceptance does not register a model or grant GPU/benchmark admission.

The separate real-native probe used RIDI `dan_bag1`/`dan_body1` for source training,
canonical source-val `hao_bag1`, and RoNIN `a000_1` for unlabelled target inputs.
It had 119 source and 310 target windows. The default engineering structure
(`hidden=128`, one recurrent layer) preserved the native 400-sample/200 Hz window,
pair averaging into the paper 200-sample/100 Hz span, and all five objectives.
Source batch 2 and target batch 3 recorded three committed D→G pairs in the
resumed training trajectory: six stage optimizer steps. The first two pairs
already changed 30 generator and 12 discriminator parameter tensors.
An atomic checkpoint after pair 2 resumed pair 3 exactly against uninterrupted
execution: model, both optimizers/schedulers, sampler/cycle state, partial-epoch
aggregates, modes, gradient flags, all RNG states and next draws matched. This
default-width proof is separate from the earlier reduced-width resume test.
Native-probe receipt:
`4143707a6339ccc4fb9405c5272b986d13ddc9991c5d8e0d2a133e3c5caff57f`.

The input proof binds only those two source recordings and one target recording.
RoNIN attitude uses privileged initial Tango alignment **and the native branch's
upstream whole-recording reference-quality selection**. Target position,
velocity, pose-validity arrays and target held-out recordings are not training
inputs. No permission for all 71 RoNIN train recordings is implied. Source native
XYZ trajectory/metrics and XY from that same prediction were checked; target
checks were input-only raw forward outputs. Typed inference payloads round-tripped
through `weights_only=True` with matching outputs/RNG. This is not the public
`NIO.export` route, TorchScript or ONNX support.

The preceding run remains recorded as **25 passes and 2 failures**. Its returned
checkpoint dictionaries aliased live permission/sampling identities, allowing
the two negative tests to modify both sides of a comparison. The sole subsequent
fix snapshots those dictionaries with `immutable_copy`; the original tests and
failed receipts remain intact. Lifecycle/identity/RNG checks precede live state
restoration. Arbitrary Torch backend or optimizer-slot corruption can still
poison a load and requires restoration from the last valid atomic checkpoint.

All earlier 14 scientific files, protected source versions, 13 native datasets,
formal policy/seeds/capacity and the existing 2,067-cell matrix retained their
guarded bytes. Runtime used two Torch threads, one interop thread and no CUDA;
observed CPU use stayed below 47.03%, with no new throttling. Engineering seed
1043, batch 2/3 and LR `1e-4` are recorded separately from the formal protocol.
No complete real source epoch, GPU/common-batch-128 capacity, public NIO workflow,
active registration, all-target permission or paper-effect reproduction is
claimed. The original body/polar-displacement Phase B remains separate work.
