# AutoDL v1 operations

**Current live execution (2026-10-24):** the formal campaign runs on the `ps`
Agent through Alibaba WSS. Its authoritative matrix is
`/home/ps/RAL/cloud-import-20261023/runtime/primary-local/matrix/main`.
This workstation only edits source, sends control commands and publishes reports.
The AutoDL SSH paths and hardware receipts below describe earlier deployment
procedures; they are not evidence of the current Agent's status or capacity.

These tools use the `dev/v1` registry, converters, `NIO` / `Trainer` lifecycle and
public `ipb` interface. The current project has migrated to a **4090D 24 GiB,
16 CPU, approximately 62 GiB RAM** AutoDL instance. Read the actual device and
cgroup limits on that instance before choosing resources; historical receipts
from the former CPU instance do not describe its current capacity.

Run framework imports, tests, conversion, capacity probes and training **on
AutoDL**. Local work is source editing, static review and the SSH client. The
CPU helper checks the server checkout/snapshot path; the full-matrix helper
requires a verified immutable server snapshot and verifies the imported package
path. Learned GPU execution additionally requires actual CUDA and the explicit
`--gpu-authorized` flag; it has no automatic CPU fallback. These flags represent
an operator's authorized action, not a claim that a campaign has already run.

The current source inventory has **33 registered architectures, 57 profiles and
23 independent paper-method sources (21 published + 2 preprints)**, plus two repository-defined physical
classics. Architecture, backbone/profile and paper-method counts are different.
The accepted-native registry currently lists 13 independent sources, with 1,376
accepted sequences and 114.306383333 hours; raw partials
and engineering excerpts do not count. Both inventories are dynamic. See
[algorithm provenance](../../docs/models/provenance.json) for exclusions, authors,
source versions and implementation fidelity. This procedure does not certify
that every profile has passed CPU/GPU acceptance or completed formal training.

## New library methods and the unchanged formal campaign

The active 2026 common-benchmark-v1 campaign remains bound to manifest
`1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949`,
425 managed files,31 architectures / 53 profiles,XYZ 31 / XY 22 and 2,067 cells. Do not
regenerate or overwrite its coverage/runstates from a changed live registry.

Current HEAD adds M2EIT/DiffusionIMU with four XY/XYZ profiles, giving 33
architectures, 57 profiles, 23 paper methods (21 published + 2 preprints) + 2 classics;
53 profiles are neural. XYZ 33 = 32 neural + RIDI3D; XY 24 = 21 neural + RIDI2D + 2 classics.
CPU admission combines old 53 receipts with a separately source-bound new 4
batch; it is not a single 57-profile acceptance batch.

The new batch ran 31 scientific nodes (all passed, zero failed/skipped) and all
four default-width, T200 @ 200Hz, B2 NIO lifecycles with two actual updates,
distinct accepted RIDI train/val/test excerpts,checkpoint/native-export
roundtrip and RNG replay in 70.083 s. Its 443-file manifest is
`b639c69773b92ea11c6e999e9d7c5f88702c5bd9df140adc5ce67f95779f6b79`;
parent receipt is `ee3f064e0f8377e9d5299b8da081b8064b6206b83bb8fedb7e667a698e10f66b`
and JUnit is `ff4d30cc48cfca2d31c9aed74c9155562f57b213b7a891058148e92e8b05e293`.

Old 49 neural CUDA acceptance and old 53 native-export proofs retain their scope.
New 4 native export/import/predict passed independently; new 4 CUDA and GPU batch 128
capacity remain pending. DiffusionIMU retains Gaussian evaluation noise and
requires the same RNG/batching/timestep policy; its auxiliary u is not sigma,
logstd or calibrated covariance. Neither method currently has a reviewed
TorchScript/ONNX adapter. Reconstruction assumptions remain in the method cards.

A potential 57 × 13 × 3 = 2,223-cell inventory needs a separately accepted source,
capacity/taskview/policy/output scope. It is not a started formal matrix and
must not append profiles to the running 2,067-cell queue. Full-paper accuracy
is not established by this finite engineering acceptance.

## Connection and source deployment

### Relay-authoritative formal-matrix reporting

The live formal campaign runs on the Agent host, not on this workstation and
not on an old SSH snapshot. `tools/ops_v1/live_progress_reporter.py` therefore
opens a short-lived browser session through the `app-remote-terminal` WSS relay,
reads the Agent's `coverage.json`, `run-states` and controller process list,
then deletes the temporary PTY. The relay APP pairing profile is read from
`REMOTE_TERMINAL_APP_PROFILE` (defaulting to the sibling
`app-remote-terminal/pairing/20261023/remote-pc-1.app.json`) and must be mode
`0600`; its token is never written to the repository or the Issue body.

The generic transport lives in the sibling repository at
`app-remote-terminal/scripts/relay-exec.js`; the benchmark repository no longer
keeps a second implementation. Install the sibling relay's existing Node `ws`
dependency before running the reporter. A command is accepted only with remote
exit code zero and an acknowledgement that its temporary terminal was deleted.
Do not install or launch a local Agent for this workflow.

Use a read-only dry run to inspect the same evidence that the scheduled report
will publish:

```bash
python3 tools/ops_v1/live_progress_reporter.py --dry-run
```

The user-level `other-inertial-benchmark-issue-reporter.timer` runs this report
every five hours and mentions `@BUG423 @RicardoCDUT @Sarah-123886` in
`BUG423/other-inertial-benchmark` Issue #1. Its service needs `node` and `gh`
in its own `PATH`, plus the existing GitHub CLI authentication; an interactive
shell's PATH is not inherited automatically. The installed user service has a
`cli-paths.conf` drop-in for those executable directories. Check scheduling with
`systemctl --user list-timers other-inertial-benchmark-issue-reporter.timer`.
The reporter fails closed if the Agent is offline, the pairing profile
is missing, the formal scope is not exactly 2,067 cells, or the live run-state
evidence cannot be read. The reporter never starts, stops, retries, downloads,
or edits a benchmark run.

The Agent also runs an independent, low-priority diagnostic supervisor from
`research/staging/trajectory_shortening_20261024/research_supervisor.py`.
It refreshes completed-result length statistics every five hours, for at most
34 cycles or until there are no qualified/running formal cells. Outputs are
new leaves under `/home/ps/RAL/research-20261024/length-refresh`; the small
`research-status.json` carries its own observation time. The Issue reporter
reads that timestamped summary and the independent candidate's CPU qualification.
It does not imply that the diagnostic snapshot covers the latest live count,
or that CPU qualification proves accuracy. This supervisor never starts GPU
training. Source-bound CUDA capacity and the 63-cell candidate pilot remain
pending resource handover after prior authorized queues; see the
[candidate plan](../../research/staging/coherent_spectral_io_20261024/README.md).

The separate `cause_decomposition.py` audit reads the same completed-result
metadata and writes `cause-status.json` on the Agent. It compares the oracle
reference ratio with model-side `plr_oracle`, separates speed bias from direction
error, and checks train/test speed-ratio gaps. The reporter includes this
timestamped cause snapshot when available. It is descriptive evidence on the
observed cells; target and loss comparisons remain confounded until the fixed
architecture interventions in the trajectory research plan are run.

Install the optional SSH client dependencies locally with
`python3 -m pip install -e '.[ops]'`. Copy
[ssh-config.example.json](ssh-config.example.json) to `.local/autodl-v1.json`,
fill in the **current** private connection fields and use `chmod 600` on that
file. Do not commit credentials or print their contents. The migrated instance
uses port **33160**; the old port is not part of this workflow.

```bash
python3 tools/ops_v1/remote.py status
python3 tools/ops_v1/remote.py sync
python3 tools/ops_v1/remote.py exec 'cd /root/autodl-tmp/other-inertial-benchmark-v1 && .venv/bin/ipb info'
```

`sync` admits tracked and nonignored source files, excludes symlinks, data, runs,
weights, external references and private configuration, and records SHA-256
hashes. It stages the archive before acquiring the shared CPU/GPU locks. Exit
75 means the lock was busy and the staged source was **not installed**; resume
after the active task releases the lock. Full deployment removes only omitted
previously managed files whose old hashes still match. Data, source archives and
results remain separate.

Install the server environment once; do not reinstall Torch or dependencies
between accepted receipts and formal runs without new environment acceptance:

```bash
cd /root/autodl-tmp/other-inertial-benchmark-v1
/root/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --cache-dir .local/pip-cache -e '.[train,test,ops]'
```

Use the same interpreter and a fixed source snapshot throughout a campaign.
`source_manifest.json` records managed source, while private receipts record the
actual Python/Torch/core dependency versions. A live editable checkout can
change during development; formal runs must use an immutable snapshot under
`.local/source-snapshots/<deployment-id>/repo`, not that live import.

The following server-only standard-library command creates a new snapshot from
an installed, verified manifest. Run it while source deployment is idle; it
refuses to overwrite an existing snapshot and checks every copied source hash.
An existing coordinator-provided snapshot may be reused instead.

```bash
.venv/bin/python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
root = Path('/root/autodl-tmp/other-inertial-benchmark-v1')
manifest_path = root / 'source_manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
files = {**manifest.get('files', {}), **manifest.get('operational_acceptance_helpers', {})}
assert files, 'Installed source inventory is empty'
snapshot = root / '.local/source-snapshots' / manifest['deployment_id'] / 'repo'
snapshot.mkdir(parents=True, exist_ok=False)
for name, expected in files.items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    source, destination = root / relative, snapshot / relative
    assert not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
assert manifest_path.read_bytes() == manifest_bytes, 'Deployment changed during snapshot copy'
(snapshot / 'source_manifest.json').write_bytes(manifest_bytes)
print(snapshot)
PY
```

Do not edit a snapshot used by running jobs. A failed or interrupted snapshot
copy lacks a complete verified manifest and must not be admitted. Preserve its
failure evidence and create a separately identified valid snapshot.

## Generate three random seeds once, then reuse the record

Each independent benchmark chooses three distinct OS-random seeds **once**,
persists the record, and reuses the same three values across every model and
dataset, including retries and checkpoint resumes. Do not draw seeds in the
training loop, select the best seed, or replace the record mid-campaign.

This executable standard-library example runs on AutoDL. If a record already
exists, it validates and reuses it rather than drawing again. Exclusive creation
protects an existing record from accidental overwrite.

```bash
.venv/bin/python - <<'PY'
import json, os, secrets
from datetime import datetime, timezone
from pathlib import Path
path = Path('.local/plans/random-seeds.json')
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    record = json.loads(path.read_text())
else:
    seeds = []
    while len(seeds) < 3:
        value = secrets.randbelow(2 ** 31)
        if value not in seeds:
            seeds.append(value)
    record = {'schema': 'ib.benchmark.random-seeds.v1',
              'entropy_source': 'operating_system_random_via_python_secrets',
              'sample_once': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'seeds': seeds}
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
seeds = record.get('seeds', [])
assert record.get('schema') == 'ib.benchmark.random-seeds.v1'
assert record.get('entropy_source') == 'operating_system_random_via_python_secrets'
assert record.get('sample_once') is True
assert len(seeds) == len(set(seeds)) == 3
assert all(type(value) is int and 0 <= value < 2 ** 31 for value in seeds)
print(path, seeds)
PY
```

This project's values are already stored in the ignored
`.local/plans/post-migration/random-seeds.json`; deploy that existing record as
`.local/plans/random-seeds.json` on the server and keep its bytes/hash unchanged.
Other users can create their own record using the example. Engineering helpers
may use a fixed diagnostic seed; those short probes are not research runs and
do not replace this formal record.

## Server variables and all-profile CPU acceptance

Set `<deployment-id>` to an actually verified snapshot and `<campaign-id>` to a
new receipt/output scope. Keep datasets and outputs outside the source snapshot.

```bash
IPB_V1_ROOT=/root/autodl-tmp/other-inertial-benchmark-v1
IPB_SNAPSHOT_ROOT="$IPB_V1_ROOT/.local/source-snapshots/<deployment-id>/repo"
IPB_OPS_PYTHON="$IPB_V1_ROOT/.venv/bin/python"
IPB_MATRIX_OUTPUT="$IPB_V1_ROOT/.local/matrix/<campaign-id>"
IPB_CPU_PROOFS="$IPB_V1_ROOT/.local/reports/all-profile-cpu-<campaign-id>"
export PYTHONPATH="$IPB_SNAPSHOT_ROOT/src"
export IPB_DATASETS="$IPB_V1_ROOT/datasets/ipb-v1"
test -f "$IPB_SNAPSHOT_ROOT/source_manifest.json"
```

```bash
"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/all_model_cpu.py" cpu \
  --root "$IPB_SNAPSHOT_ROOT" --data-root "$IPB_V1_ROOT" \
  --output "$IPB_CPU_PROOFS" --dataset ridi \
  --batch 2 --threads 2 --jobs 4 --seconds 600
```

`all_model_cpu.py` enumerates every active model YAML, including backbone
variants and XY reference profiles. Each subprocess uses the real scientific
input span/history/output layout, augmentation and loss, and independent real
train/val/test excerpts. The standard `NIO`/`Trainer` lifecycle checks actual
updates where applicable, validation, prediction and checkpoint reload. A short
MSE-stage lifecycle alone does not prove the later NLL/covariance phase;
phase-switch profiles also need their scientific phase/backward receipts.

Sequence models fit or calibrate on **train only**. RIDI performs learned
SVM/SVR fitting; PDR and constant-heading are physical calibration references.
None should claim invented neural optimizer updates. The CPU helper hides CUDA,
uses its own locks, retains per-profile failures/timeouts, and never treats a
skip as a passed model. `--profiles` is a selected retest scope, not full-matrix
acceptance. Historical passing counts do not certify changed source.

CPU receipts can be reused on new snapshots only when scientific source and
core dependencies are unchanged, or through an exact approved
`ib.scientific-source-bridge.v1` record. A bridge lists changed scientific file
hashes, actual targeted test receipts and their hashes, the reason and both
source-manifest hashes. Editing a receipt's source hash is not a bridge.

## Explicit LieEvents optimisation equivalence acceptance

`tests/test_lie_events_equivalence.py` in this tools directory is an operator
acceptance suite, outside the repository's default `testpaths = ["tests"]`.
It compares the optimised implementation with the unmodified historical repo
file; the regular public model tests do not depend on an ignored oracle.
Run this suite explicitly on AutoDL for both CPU and CUDA. A missing oracle or
requested CUDA device fails rather than silently skipping the comparison.

Use an authorized repository checkout containing commit
`b9e7886ffcc51c1c4e2d494ac9be3faf2ec81375`, or the already archived identical
private file. Recover only
`src/inertial_benchmark/nn/modules/lie_events.py`; its required SHA256 is
`c8158b700566d8cc7015741099097b6e3f480fcd71383eff13ce4fec10c11518`.
This is repository-owned historical code, separate from the upstream author's
reference repository. Keep the oracle in an ignored directory, never in the
active Python package, and preserve its original bytes.

```bash
export IPB_HISTORY_REPO="$IPB_REMOTE_ROOT"
export IPB_LIE_EVENTS_ORACLE_PATH="$IPB_REMOTE_ROOT/.local/acceptance-oracles/lie_events-before-sync.py"
"$IPB_OPS_PYTHON" - <<'PY'
import hashlib, os, pathlib, subprocess
revision = 'b9e7886ffcc51c1c4e2d494ac9be3faf2ec81375'
data = subprocess.check_output([
    'git', '-C', os.environ['IPB_HISTORY_REPO'], 'show',
    revision + ':src/inertial_benchmark/nn/modules/lie_events.py'])
expected = 'c8158b700566d8cc7015741099097b6e3f480fcd71383eff13ce4fec10c11518'
assert hashlib.sha256(data).hexdigest() == expected
path = pathlib.Path(os.environ['IPB_LIE_EVENTS_ORACLE_PATH'])
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    assert path.read_bytes() == data, 'refuse to replace a different oracle'
else:
    with path.open('xb') as handle:
        handle.write(data)
PY
IPB_LIE_EVENTS_TEST_DEVICE=cpu "$IPB_OPS_PYTHON" -m pytest -q \
  "$IPB_SNAPSHOT_ROOT/tools/ops_v1/tests/test_lie_events_equivalence.py"
IPB_LIE_EVENTS_TEST_DEVICE=cuda "$IPB_OPS_PYTHON" -m pytest -q \
  "$IPB_SNAPSHOT_ROOT/tools/ops_v1/tests/test_lie_events_equivalence.py"
```

Each device runs 19 exact-equality cases, including events/poses, boundaries,
sorting and interpolation, training noise, the next random draw, loss and
gradients of all three LieEvents profiles. These checks are followed by fresh
batch-128 pipeline/capacity measurements; equivalence alone is not a speedup or
formal-training result. The optimisation and this acceptance are pending their
server receipts until those commands actually pass.

## Inventory, qualify, measure, freeze, run

Use one output scope across these stages. Helpers acquire their own resource
locks; do not wrap them in a second `flock` for the same lock.

```bash
"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/full_matrix.py" inventory \
  --source-root "$IPB_SNAPSHOT_ROOT" --output "$IPB_MATRIX_OUTPUT" \
  --registry "$IPB_V1_ROOT/.local/reports/formal-native-registry.json" \
  --seeds "$IPB_V1_ROOT/.local/plans/random-seeds.json"

"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/full_matrix.py" qualify \
  --source-root "$IPB_SNAPSHOT_ROOT" --output "$IPB_MATRIX_OUTPUT" \
  --cpu-receipts "$IPB_CPU_PROOFS"

"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/full_matrix.py" capacity \
  --source-root "$IPB_SNAPSHOT_ROOT" --output "$IPB_MATRIX_OUTPUT" \
  --cpu-receipts "$IPB_CPU_PROOFS" --gpu-authorized
```

Supply `--science-bridges <approved-bridge.json>` alongside `--cpu-receipts` when
reusing qualified historical CPU evidence. `--profiles` and `--datasets` let
qualification run in finite batches while keeping all other coverage rows
visible. They are filters, not permission to drop unaccepted combinations from
the delivery matrix.

| Stage | Evidence and boundary |
|---|---|
| `inventory` | Every active profile × accepted independent native × the same three seeds; records config/input/data/split identities and dimensional/input tracks. A downloaded archive alone is not accepted native. |
| `qualify` | Actual valid native task windows and XYZ→XY target slicing, split availability and CPU-model proof. Missing orientation/modalities, missing validation and conversion failures remain explicit pending/unsupported rows. |
| `capacity` | Real native DataLoader/augmentation/transfer plus forward, scientific loss, backward and optimizer steps; then common-batch `NIO` train/val/predict/reload and four-mode MC=8 diagnostics. Engineering fixtures are not convergence experiments. A capacity JSON's presence does not imply `formal_ready`. |
| Freeze | Review measured common train/val batch, full lifecycle/diagnostic memory headroom, throughput, source/environment identities, actual task-window counts, whole-matrix cost and checkpoint storage. Persist a separately hashed frozen policy tied to those receipts. |
| `run` | Executes accepted rows through the existing benchmark engine, persists identities and status; incomplete rows and declared checkpoint transfers remain visible. It does not download or convert data. |

The common primary recipe passes the same optimizer, LR, scheduler, weight decay,
train/val batch, epoch cap, minimum epochs, validation/early-stopping rules and
resource settings across learning profiles. Model windows, target type, output
layout, scientific augmentation and loss/phase are preserved; unified does not
mean replacing every loss with MSE. The fixed-epoch primary policy uses
`train_windows_budget=0`. An equal-window/compute experiment is a separately
named protocol; integer epoch conversion is not an exact equal-window guarantee.
Ensure the actual training reaches each profile's scientific phase; report both
last and selected checkpoint epoch. A pre-switch selected checkpoint cannot be
advertised as a trained covariance result.

Do not turn the candidate policy into `status=frozen` merely to bypass a gate.
The source currently has additional admission/retention/phase/state corrections
under review; actual targeted server tests and an approved source snapshot must
verify them before formal freezing. Required acceptance includes capacity-policy
resource/source binding, full lifecycle headroom, actual small-budget gap support,
checkpoint retention/storage accounting, checkpoint-aware final loss phase,
resume identity and current public-CLI transfer state. This README does not mark
those changes passed or assert formal training has completed.

Once the reviewed policy is frozen, the two execution lanes are:

```bash
"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/full_matrix.py" run \
  --source-root "$IPB_SNAPSHOT_ROOT" --output "$IPB_MATRIX_OUTPUT" \
  --seeds "$IPB_V1_ROOT/.local/plans/random-seeds.json" \
  --cpu-receipts "$IPB_CPU_PROOFS" \
  --policy "$IPB_V1_ROOT/.local/plans/frozen-main-policy.json" \
  --lane learned --gpu-authorized

"$IPB_OPS_PYTHON" "$IPB_SNAPSHOT_ROOT/tools/ops_v1/full_matrix.py" run \
  --source-root "$IPB_SNAPSHOT_ROOT" --output "$IPB_MATRIX_OUTPUT" \
  --seeds "$IPB_V1_ROOT/.local/plans/random-seeds.json" \
  --cpu-receipts "$IPB_CPU_PROOFS" \
  --policy "$IPB_V1_ROOT/.local/plans/frozen-main-policy.json" \
  --lane classic
```

The `classic` CLI lane means **non-gradient sequence fitting/calibration**, and
includes RIDI; it is not a count of independent physical classics. Partial
filters and successful stages are not evidence that all matrix cells completed.
Read per-row status, actual source-qualified receipts and final
`coverage_completed`, not only a process exit code.

## Public `ipb` interface and scientific tracks

The operational queue orchestrates the same public framework, not a second
training stack. On the server, use the snapshot import above:

```bash
export IPB_NATIVE_REGISTRY="$IPB_V1_ROOT/.local/reports/formal-native-registry.json"
export IPB_RANDOM_SEED_RECORD="$IPB_V1_ROOT/.local/plans/random-seeds.json"
export IPB_FROZEN_POLICY="$IPB_V1_ROOT/.local/plans/frozen-main-policy.json"
export IPB_MATRIX_ADMISSION="$IPB_MATRIX_OUTPUT/coverage.json"
"$IPB_V1_ROOT/.venv/bin/ipb" benchmark cfg=main dry_run=true report=false
```

After public admission/state integration is accepted, `ipb benchmark cfg=main`
executes the dynamic public matrix. Until then use the qualified operations
runner for its persisted run-state/transfer handling; do not interpret a CLI
pending transfer as a completed evaluation. Standalone `ipb train`, `val`,
`predict`, `export`, `report` remain the model-facing API, but a manual run does
not inherit a campaign's scientific identity just by using the same model name.

### Inference export acceptance

The new `NIO.export()` / `ipb export` implementation passed a separate
server CPU admission gate; it is not part of the already running immutable
formal snapshot. Native packages preserve scientific InputSpec, loss/phase and
numeric trained or calibrated state, with a verified member inventory. They omit
optimizer state and explicitly reject resume; use the original training
checkpoint for continuation or `pretrained` for a new run.

```bash
PYTHONPATH="$IPB_EXPORT_SNAPSHOT/src" "$IPB_OPS_PYTHON" -m inertial_benchmark export \
  model="$IPB_TRAINING_CHECKPOINT" format=native path="$IPB_EXPORT_OUTPUT/native"
PYTHONPATH="$IPB_EXPORT_SNAPSHOT/src" "$IPB_OPS_PYTHON" -m inertial_benchmark predict \
  model="$IPB_EXPORT_OUTPUT/native" source="$IPB_ACCEPTED_TEST_SEQUENCE" device=cpu save=false
```

Bind `IPB_EXPORT_SNAPSHOT` to the separately reviewed export snapshot and use a
new output directory. Do not replace the source or environment of active jobs.
TorchScript/ONNX are enabled only for the reviewed RoNIN ResNet and IMUNet tensor
adapters, with the original scientific span and dynamic batch. Other models have
an explicit unsupported graph status; no trace fallback is accepted. Every
published graph executes and compares raw outputs before publication.

`NIO(model.onnx)` / `NIO(model.torchscript)` verifies that bundle and reloads its
native `model.pt` for the IPB Predictor. It **does not execute the tensor backend**.
Use an independent ONNX Runtime/TorchScript session for raw tensor inference;
see [export contracts and examples](../../docs/EXPORT.md).

Actual server acceptance collected and passed **45 targeted nodes**, including
both reviewed XYZ adapters through TorchScript and ONNX Runtime at batches 1/2/5,
native real-record/CLI roundtrips, resume refusal, and three ordinary checkpoint
resume/phase regressions. The RoNIN tensor test uses a smaller engineering
structure; IMUNet uses its original model. Missing optional dependencies are not
an ONNX pass. Install
`onnx`/`onnxruntime` into a private `--target` directory using bounded server-only
acquisition, without upgrading the base training Torch/numpy. Run only in the
coordinated nonblocking CPU lease; record actual threads, collection, formats,
checks and hashes. No GPU or formal run is needed for these inference checks.
An additional 70.16 s native-only batch reused all **53 accepted engineering
checkpoints and their original RIDI excerpts**, without training: 53/53 state,
InputSpec, loss/phase/evidence, raw/trajectory/target outputs, metrics and RNG
roundtrips were exact. RIDI XYZ also traversed actual CLI export/predict/NPZ.
Torch2/interop1 and ORT2/1 were recorded; CUDA stayed uninitialized. New429 source,
formal425 source,13 native metadata and477 original checkpoint/proof/fixture
files stayed unchanged. The 45-node and 53-native batches used a distinct,
authorized supplemental two-thread CPU lease; the classic CPU/GPU queues continued.

Source manifest: `d58706ac0feb287685a5f48eb4184d9855ff12c64e76efaa24c759cb8ce0f484`.
Targeted receipt SHA: `cb9f9600b795839fb59ff291e2aae113b8d154b9257af43b598e7cc72d3b598f`.
Native53 receipt SHA: `0b02b27e2773776571fcf6bdefbdd1bc074e71627f3150f7e0dda1fa6ee0697b`.
Receipts are retained in `.local/reports/export-supplemental-next-d587/` and
`.local/reports/native53-export-d587/`; tested snapshot bytes remain unchanged
when later documentation is published. This makes no all-53-format tensor export
or formal accuracy claim.

XYZ-capable output is a dimensional capability, not proof that the original
paper was three-dimensional. Preserve `3d_adapted`, original XY reference,
paper-only reconstruction and partial-source labels. Requested XYZ evaluation
requires actual predicted XYZ; a planar method cannot enter by adding zero or
reference Z. The same XYZ prediction supplies XY auxiliary metrics by slicing,
without retraining, reinference or second integration. Reference pose/orientation
inputs and true initial-velocity inputs are privileged tracks, distinct from
available-device inputs. Human, UAV, wheeled and legged domains stay identifiable.

A test-only native, such as Blackbird, needs a declared compatible training
checkpoint: same profile/input track/seed, scientific input specification,
source/policy identities and verified checkpoint hash. The current declared
Blackbird origin is EuRoC. This is transfer evaluation, not training on Blackbird.
Source/dataset/split/policy/seed changes need a new run identity, not implicit
checkpoint resume or score merging. Test is final evaluation after val selection;
never use test to choose variants, thresholds, hyperparameters or seed.

## Data/reference acquisition and other acceptance tools

Download and process new datasets **directly on the current AutoDL server**,
using official sources, bounded retries/timeouts and resumable independent raw
directories. Preserve original archives, provenance, byte counts and hashes;
check archive integrity before extraction and retain at least 30 GiB free for
ongoing work. Do not change canonical/native files to fit a model. Source-specific
converters must report axes, SI units, clock handling, reference quality, splits
and rejects. Add only genuinely verified nonempty native sources to the accepted
registry; aliases, duplicate raw archives and a dataset containing another source
do not inflate independent coverage.

If AutoDL network acceleration is needed, scope it to the download subprocess:
`source /etc/network_turbo >/dev/null 2>&1`. Do not print its environment or store
its secret values in repository files. Failure to download, gated terms/login,
missing actual modalities or scientific-reference defects remain explicit
pending/blocking items. Raw source and official algorithm-reference code belong
in ignored server directories with pinned versions/licensing evidence; obtaining
a wrapper repository does not prove the network source exists or authorize
unlicensed code copying. Do not download raw/weights locally for this workflow.

`server_acceptance.py` is a historical small engineering conversion/lifecycle
scope, not all-profile scientific qualification. `run_checks.py` records pytest,
Ruff, compilation, package/import location, config and dry-run checks;
`cpu_pytest.py` controls test resource environment and records original nodeids.
File isolation preserves per-module logs/JUnit when needed; selected test paths
remain selected evidence. Dataset-dependent skipped tests are visible, never
public-data results. Keep historical receipt bytes and failed attempts; update
scientific source through new deployments and evidence, not overwritten reports.
