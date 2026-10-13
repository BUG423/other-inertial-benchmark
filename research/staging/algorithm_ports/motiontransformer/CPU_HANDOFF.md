# MotionTransformer CPU scientific handoff

**Accepted scope:** actual AutoDL CPU scientific contracts and a synthetic two-domain
optimizer probe. The method remains unregistered. A standard two-domain NIO
lifecycle, real-dataset domain training and CUDA acceptance remain pending.

The original seven MotionTransformer files and two A2DIO evidence files are
unchanged. Their [source card](source-card.md) and [provenance](provenance.json)
retain the pre-runtime state; this new record updates engineering evidence only.
The [model](model.py) is an independent paper-only reconstruction with declared
recurrent architecture, reduction and optimizer-scheduling assumptions. CPU
execution does not establish author-parameter equality or paper accuracy.

On 2026-10-13 UTC, the server actually collected and passed **28 scientific test
nodes**, with zero failures/errors/skips. The successful v2 parent completed in
**24.518 s**, including collection, tests and a separate default-width probe.
Root independently verified the original receipt/JUnit/probe bytes and frozen
staging hashes; this document is not a local runtime result.

The [scientific tests](test_scientific_contract.py) cover all five joint objectives,
both generator directions and both discriminators, shared encoder/predictor,
disjoint optimizer ownership, real parameter updates, unpaired batch sizes,
source-only label access, noise/RNG replay and numeric state reload. The original
planar `(Δlength, Δheading)` integration accumulates predicted heading increments
without reading later ground-truth headings. The separate gravity-world velocity
adaptation verifies native 400 samples at 200 Hz are explicitly pair-averaged to
200 steps at 100 Hz, with correct channel order and a learned, nonzero-gradient
vertical velocity. Its XYZ run derives XY from the same XYZ slice. These changed
velocity/frame labels are not the original paper's polar-displacement task.

## Default-width two-domain probe

The additional probe used synthetic engineering measurements, not accepted
dataset sequences or an accuracy experiment. It retained **hidden width 128,
200 six-axis steps at 100 Hz (2 s)**, source batch **B2**, and an unpaired target
batch **B3**. Source labels entered supervision; target-domain labels were never
read. Adam at `lr=1e-4` was an explicit engineering recipe assumption, not a
recovered paper optimizer or the formal benchmark recipe.

Both the joint generator optimizer and discriminator optimizer completed **two
actual updates**. All seven module groups changed: shared encoder, source and
target generators, shared decoder, shared predictor, and both discriminators.
The objectives were finite and retained GAN, reconstruction, prediction, cycle
and perceptual terms, with paper trade-off weights `.01 / 100 / .1 / 1`.

After those two updates, the probe restored a checkpoint containing model state,
both optimizer states and RNG state. Its **third update**, losses, updated weights
and next random draw exactly matched uninterrupted execution. Source/target
inference, restored state and RNG were also exact. This is a method-specific
two-optimizer engineering proof; it is not a completed standard NIO Trainer
train/validation/prediction lifecycle.

## Evidence and preserved first attempt

Private operator artifacts are retained under
`.local/reports/motiontransformer-staging-science-f0a49602-v2/`.

| Original artifact | SHA256 |
| --- | --- |
| `receipt.json` | `14f797a80ead0c421522d2ad4c8dff83a15d63702cc136ecd2dbeb8aa6b50d56` |
| `junit.xml` | `a26c1f8bfa59a9e54776f81e8b0ef8dab6820dafdee61c36c01f2bcf14b5a01b` |
| `domain-probe.json` | `4fd2f75b8d86498f8e4138f2ccef717e5292f801991ca0a828384680b69bfaa2` |

The v2 private runner SHA is
`853b0764bae62891d03e03b8412e850e3f3a97ef20a7e5f0d800380e628b0f57`.
Root's separate review is retained at
`.local/reports/root-motiontransformer-actual-review.json`.

The first attempt is preserved unchanged under
`.local/reports/motiontransformer-staging-science-f0a49602/`. Its actual pytest
run passed the same 28 nodes, but the private parent misparsed the collection
log as zero nodes and failed its count assertion before running the additional
probe. Its parent receipt remains **failed**, not relabelled as success. The v2
helper corrected only that private parser and completed the full stated scope;
the seven model/config/test/card files did not change between attempts.

## Resource and integration boundary

Execution used the approved supplemental CPU lease, Torch threads **2** and
interop threads **1**. CUDA was not initialized; no GPU lease or original CPU
training lease was acquired. The v2 worker exited and released its lease.
Protected source snapshots, thirteen native metadata groups, fixture files and
staging files remained unchanged. The dependency source manifest was
`d58706ac0feb287685a5f48eb4184d9855ff12c64e76efaa24c759cb8ce0f484`;
the running formal 425-file manifest remained
`1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949`.

The current formal **2,067 rows**, registered profiles, capacity policy and
persisted three random benchmark seeds were unchanged. The synthetic probe's
seed `1043` is an engineering RNG fixture, not an additional formal seed.
This handoff adds no registered architecture, profile or accepted formal method;
it does not establish a 33-architecture/57-profile matrix.

Next integration requires a reviewed source/target train-split policy, unpaired
target-IMU loading without target-label leakage, two-optimizer Trainer stages
and checkpoint/sampler/RNG resume, and explicit validation diagnostics. Original
body-frame polar targets and stateful integration remain a separate protocol.
The adapter's ordinary single-domain loss refusal stays in place. Only after a
true standard dual-domain lifecycle and separate capacity qualification may a
new source snapshot register the method and extend the formal matrix.
