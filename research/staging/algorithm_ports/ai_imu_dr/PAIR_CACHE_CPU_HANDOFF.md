# Released pair-cache / crop-offset CPU handoff

On 2026-10-14, the three-file pair-cache leaf passed a bounded server CPU
qualification: **33 tests collected and 33 passed**, with zero failures,
errors or skips. Collection, JUnit, the runtime receipt and the independent
root review agree on every node. The run completed in 16.828 seconds with two
Torch/OMP threads and one interop thread; CUDA remained uninitialized.

The accepted files retain the released whole-recording distance-pair grid,
100 Hz raw indexing, decimation by ten, starts every 100 raw samples,
100--800 metre endpoint search, crop offset `N0=10*k`, the released
`floor(N/10)` endpoint rule, reference-chord normalization and SUM loss. The
actual test set includes pinned-source function oracles, nonsecond crop
counterexamples, quality-mask exclusions, the single-pair shape boundary,
readonly/no-RNG checks, and a B2 XYZ gradient comparison against the unchanged
SUM loss.

The run created a new immutable 475-file evidence snapshot from the previously
qualified 472-file AI-IMU-DR source plus exactly these three files. Protected
425- and 472-file sources, native metadata, prior staging evidence, policy,
seed and capacity records matched before, after and at terminal cleanup. The
supplemental CPU exclusive lease was released. Nineteen nonblocking warnings
come from the pinned upstream oracle's legacy integer comparison and uint8
masking, plus a readonly NumPy index used without mutation in one test.

This qualification is deliberately narrow. It accepts synthetic supervised
pair caching/cropping and the B2 original SUM/XYZ-gradient adapter. It does not
qualify a real KITTI recording, raw clock or vehicle permissions; a standard
NIO lifecycle; a 6000-sample training pipeline; CUDA; the complete
AI-IMU-DR system; or paper accuracy. It adds no registered model or formal
matrix row, and it did not rerun the earlier 51 core or 39 full-sequence tests.

The pre-execution [leaf note](PAIR_CACHE_LEAF.md) remains byte-identical to the
accepted snapshot and therefore still records its original pending state.
This handoff is the later runtime status.

| Evidence | SHA-256 |
| --- | --- |
| Actual CPU receipt | `e17d6d08565acc68848b837eeac3e627de542ad7f58b7ee48f37f1c44e8dff05` |
| 33-node JUnit | `0ec9a9b392fcd20a94fd2064b74c89b3467d679597543d2a3fdd8712d80966b4` |
| Collection log | `70ff61c351863dd34d6bb9082d0fae80d116185dd7b52ebc6ad82bf73320cfa4` |
| 475-file source manifest | `389e3f42f45e8462698cddfa41626fc3d66527bdef637558e01e43395cdbf34a` |
| Root actual review | `8e165b34e6f1094f39a6d9be25d86c00aa20b6d97385219e5140b07fa029b70c` |
