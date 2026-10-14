# Full-sequence transport CPU handoff

The three-file full-sequence leaf passed one authorized, isolated AutoDL CPU
qualification. Actual collection and JUnit both contain **39 tests: 39 passed,
0 failed, 0 errors, 0 skipped**. Runner elapsed time was 17.582 s, including a
5 s resource probe and 6.023 s collection; focused testing took 6.277 s. Torch
used 2 threads and 1 interop thread, CUDA was not initialized, and the
supplemental CPU lease was released. No formal process or GPU was controlled.

The immutable [implementation](full_sequence_prediction.py),
[scientific tests](test_full_sequence_prediction.py) and
[leaf specification](FULL_SEQUENCE_LEAF.md) were tested without edits. The
specification retains its pre-execution static status for source identity;
this separate handoff records the later actual result. The existing eleven
AI-IMU-DR scientific files, including their earlier CPU handoff, are unchanged.
Their 51 tests were not rerun.

Tests verify direct corrected world p/v and all 21D filter-state fields,
readonly ownership, the same XYZ prediction sliced for XY, explicit input-gap
masks, integer nanosecond jitter and overflow guards, fixed SI metadata and
rejection of invented units. P retains mixed error-coordinate product units;
the two car lateral/up N variances retain m²/s² units. Both are preserved as
state diagnostics, while Cartesian uncertainty remains unsupported/NA.
Research NPZ state/metadata replay, overwrite refusal and capability rejection
also passed.

One test evaluates the previously accepted default-width filter on a synthetic
**B2, T13** input, selects one actual returned batch item and checks every
transported value and the unchanged next RNG state exactly. This is genuine
filter-output transport, not another training run or a real recording test.
All fixture clocks and motion in this batch are synthetic. The codec records
caller acquisition/initialization receipt identifiers; it does not validate
those actual files, permissions, physical units or the acquisition clock.

The snapshot contains **472 files = accepted 469 parent + 3 new leaf files**.
Formal 425 source bytes, accepted parent 469 bytes, native-13 metadata, the
prior 51-test receipt/JUnit and policy/seed/capacity byte pins matched before,
after and at termination. The three tested leaf files and the original eleven
local files also retain their frozen SHA256 values. Actual collection item
identities equal JUnit and environment records, and all 20 mirrored original
small evidence files passed SHA256 checks.

Evidence identifiers (private operator receipts, no data or weights):

- Output directory: `.local/reports/ai-imu-dr-full-sequence-707bb7b52a87a5b6/`.
- Actual receipt SHA256: `c0b54687b36013c001ce480fc6d45ab19e7d607bb7c6ace4a122792d68fccf80`.
- JUnit SHA256: `af472c21639467e5bdcd5f7871ee5632607a0c8ea34ae5669c8c370dc7ae2b8a`.
- Actual 472-file source-manifest SHA256: `a4a2fce1449bf159494b453184090356246cbd38ee473a68adae0c91527e1f85`.
- Frozen three-file v2 manifest SHA256: `f73eebcfbb98c57e4acfc924d4e20e9395d6d21b5c5349eb03e0730a5f98ad5b`.

No real KITTI/raw-100Hz admission, 100–800 m training-pair qualification,
standard NIO/Trainer/Predictor lifecycle, public native export or resumable
training, Cartesian covariance mapping, T6000 capacity or CUDA result is
claimed by these tests. This leaf remains unregistered; active algorithm and
formal matrix counts and the three persisted benchmark seeds are unchanged.

Root independently matched all 20 original mirrored files to the server, checked
actual 39-node JUnit/filter-output evidence and source/native/policy guards, and
accepted **synthetic CPU output transport only**. Root review SHA256:
`fe20a4866c9325dcedd9331503be57e9480dbd3cd167ef4ef8ec8d2370a6006e`.
