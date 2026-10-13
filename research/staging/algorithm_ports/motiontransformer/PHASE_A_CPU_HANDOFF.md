# MotionTransformer Phase A CPU handoff

Phase A is a standalone, unregistered training protocol. Its focused server CPU
qualification completed on **2026-10-14**: **21 collected / 21 passed / 0 failed /
0 errors / 0 skipped**. Total bounded-worker time was 18.519 s; the focused pytest
run took 8.020 s. Root independently verified the original evidence and accepted
this focused qualification. This adds no model, profile, dataset or run to the
frozen formal matrix.

## Actual qualification

- The default hidden width **128** used source batch **2** and unpaired target
  batch **3** for two complete **discriminator → generator** pairs. Both
  optimizer counters advanced twice, both parameter partitions changed, and the
  original five generator objectives stayed finite.
- Scientific input timing stayed **400 samples at 200 Hz → pair average → 200
  samples at 100 Hz**. The explicitly adapted supervised labels are world
  average velocity expressed as planar speed/heading plus learned vertical
  velocity for XYZ. This differs from the paper's original polar displacement.
- A separate reduced-width mathematical fixture saved after pair 2 with
  `weights_only=True`, restored, and reproduced pair 3 exactly: model weights,
  both optimizers and schedulers, committed samplers, objective aggregates and
  Python/NumPy/Torch RNG state. It is not a default-width NIO lifecycle claim.
- Tests exercised source-only labels, canonical train/held-out identities,
  actual native-format manifests and `DatasetSpec`/`InertialDataset` APIs.
  Target numeric position, velocity and pose-validity were poisoned and never
  read; target held-out files stayed unopened. Allowed reference quaternion and
  timestamp bytes were bound to a separate coverage receipt. Changing quaternion
  bytes while retaining the same IMU rejected admission.
- A partial D/G pair poisons further training, evaluation, export and state
  serialization until a committed checkpoint is restored. Mode flags and
  discriminator gradient flags are preserved at checkpoint/evaluation boundaries.
- The typed **research inference payload** excludes optimizers, schedulers,
  samplers, RNG and raw target/test data. Its serialization/output checks passed;
  it is not the public `NIO.export` route, TorchScript or ONNX support.

All domain data in this batch were explicitly **synthetic engineering fixtures**.
The native file ABI and permission guards were exercised; these results do not
establish real cross-dataset adaptation or scientific accuracy. Phase A permits
only explicitly admitted **reference-attitude-assisted** target input. Finite
quaternions alone do not establish permission or valid attitude coverage.

## Evidence and protected scope

The isolated source had **462 managed files**: qualified parent 452, three new
protocol/test files, and seven unchanged, previously qualified Motion source
dependencies. The three tested files are:

| File | SHA-256 |
|---|---|
| `src/inertial_benchmark/engine/training_protocol.py` | `99d3f1ed3b79d5cb22ed442edee1887aa572c4553e941bd0ab05c8875034aa4b` |
| `phase_a_protocol.py` | `652e25ea1b9e13e7d1e29ae9a6729f7edcb879c191ab3c471c7cad0b0111e00d` |
| `test_phase_a_protocol.py` | `0cc8d3953c1329d059b9b46acba954f75c10cf84c61e37d3b7beb4a018b7917d` |

- Source manifest: `b39d8c8d0259e136385b261372fbb3f9c548340e450081a342735e98fdde3094`.
- Original server receipt: `e7b4d56913dc749a68a9dacbf0fc6f673dc5fc999cc49daef4b633155906abec`.
- Actual JUnit: `0d89a23dadef280207234c281eb1484539d0a61ba776c7fa0dfc22edaec17557`.
- Root independent review: `39485ba4c0ae941f1d6f0558ee5f8dc160363fa638bbb5c8c0212883402f89dd`.

The worker used two Torch threads, one interop thread and an independent,
nonblocking supplemental CPU lease. CUDA was never initialized. Protected formal,
export, integration and prior research sources, the 13 formal native datasets,
original Motion/data leaves, seeds, policy and capacity receipts stayed byte
identical before and after. Formal parent/child processes were not signalled.

The earlier missing-path preflight failure and the subsequent **3 passed / 18
failed** fixture-metadata run remain preserved with their original receipts,
logs and JUnit. The final correction completed synthetic metadata in the test
factory; it did not weaken validation or change protocol/model mathematics.

## Remaining integration

Phase A has no current `Trainer` hook, registered profile, real two-domain NIO
train/validation/prediction lifecycle, public native export bridge or GPU
admission. A reviewed single-Trainer hook and real source/target permission
certificates must precede those steps. Original body-IMU/polar-displacement
**Phase B** remains separate work. Existing formal source and coverage stay frozen.
