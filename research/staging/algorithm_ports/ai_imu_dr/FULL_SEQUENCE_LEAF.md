# Full-sequence output leaf — static research contract

This is a **new unregistered transport leaf**, separate from the eleven already
CPU-accepted AI-IMU-DR files. No shared NIO/Trainer/Predictor/Results source is
modified. The new leaf/tests have only standard-library AST/static review;
actual server collection and execution are pending. The prior 51-pass evidence
does not qualify these newly added files.

[full_sequence_prediction.py](full_sequence_prediction.py) represents one
recording's original int64 nanosecond clock, corrected world p/v XYZ, predicted
body-to-world R, both biases, estimated Rci/lever arm, full 21D error P and two
car lateral/up observation variances. Values are copied into readonly owned
float64 arrays, checked for shape/finiteness/rotation/covariance validity and
never projected, repaired, padded, frame-rotated or reintegrated. Same-XYZ metric
views use only `[:,:2]` for XY; valid-input gaps remain explicit masks and are
not interpolated. No reference position/velocity or reference join occurs here.

The contract declares fixed SI units: p and lever arm in metres, v in metres
per second, gyro bias in radians per second, accelerometer bias in metres per
second squared, rotations dimensionless, original timestamps in nanoseconds,
and the two lateral/up car velocity variances in squared metres per squared
second. Error P has **mixed coordinate-product units**, with 3-axis blocks ordered
(theta rad, xiv m/s, xip m, bg rad/s, ba m/s², thetaci rad, tci m). It is not assigned
one Cartesian unit. The NPZ loader rejects missing or altered unit declarations.
These declarations record the accepted filter's input/output convention; they
cannot detect physically mis-scaled raw inputs or prove acquisition provenance.
A future admitted reader must verify original units before supplying the filter.

The batched filter adapter selects one explicit batch item, detaches/transfers
its already computed outputs and verifies original values' dimensions. This
is output transport, not a training CPU fallback or an inference backend.
No covariance mapping, additional network forward or random draw is performed.

Metadata preserves the released approximate Taylor3/Gdt/no-reset convention,
left-SE₂(3) tangent order, source code identity, reference-assisted initial R/v
and the CNN's eight-future-sample/80 ms lookahead at nominal100Hz. Int64 ordering
is checked before differences, but tensor metadata cannot prove the physical
acquisition clock. Caller-provided raw/clock/initialization receipt hashes are
recorded, **not verified by this codec**. A separate admitted recording reader
must validate those actual files and permissions before model execution.
The nominal100Hz declaration does not independently prove rate/jitter origin.

Cartesian uncertainty remains explicitly **unsupported/NA**. The two N outputs
are pseudo-measurement variances. P[6:9,6:9] is a group-tangent block; proper
position covariance requires `Jp=[−skew(p),0,I,…]`, all P cross terms, independent
finite differences/Monte Carlo and separate statistical qualification. The
leaf raises for direct Cartesian covariance requests instead of inventing sigma.
P and N themselves remain available unchanged for diagnostics and later work.

The NPZ codec uses non-pickle numeric arrays/JSON, rejects invented capabilities
and extra sigma fields, and refuses overwrite. It is an **experimental research
transport**, not public NIO native export, resumable training, legacy Results
serialization or a benchmark-ready artifact. It neither changes old NPZ fields
nor claims compatibility with their loaders. A compression/write failure may
leave a failed output file; callers retain it or use a new path, never silently
overwrite it. Existing core export paths and all prior file bytes remain intact.

[test_full_sequence_prediction.py](test_full_sequence_prediction.py) proposes
exact p/v/state preservation despite velocities inconsistent with p integration,
same-XYZ XY views, raw integer jitter/offset/overflow guards, readonly ownership,
explicit gap masks, invalid state/clock/covariance/unit rejection, a tangent-block
counterexample, NA uncertainty, metadata/NPZ replay, false capability rejection
and actual accepted-filter tensor-output transport with unchanged RNG. All
fixtures are synthetic; the one filter probe does no training and does not
qualify real KITTI, NIO, CUDA, full6000 capacity or the future covariance mapping.
Only actual finite AutoDL CPU testing after review can accept this leaf.
