# KITTI generic 200 Hz candidate: CPU handoff

The lossless KITTI OXTS sidecar was converted through the shared dev/v1 200 Hz
resampling path in an isolated AutoDL CPU scope. Ten actual scientific checks
passed with no failure, error or skip. Full native array and file-hash checks
passed, Torch was not imported, CUDA was not requested, and the before, after
and terminal guards were byte-identical. The supplemental CPU lock was
released normally.

The conversion admitted six clock-contiguous native sequences and explicitly
rejected two branches using the standard gravity hard threshold. It did not
sort timestamps, bridge clock backjumps, interpolate categorical flags or make
long source gaps valid. One accepted sequence has a marginal-gravity warning:
its 4.30 degree tilt is below the 5 degree rejection threshold but above the
2.5 degree warning threshold. The warning is retained rather than hidden.

| Engineering split | Original recording group | Accepted clock branches |
| --- | --- | ---: |
| Train | `2011_10_03_drive_0042_extract` | 1 |
| Validation | `2011_09_30_drive_0033_extract` | 3 |
| Test | `2011_09_30_drive_0034_extract` | 2 |

Every accepted sequence has a valid one-second view, finite XYZ targets and an
XY target that is the exact slice of the same XYZ prediction. Original
recording groups remain disjoint across splits. Source endpoint indices and
civil-clock provenance are retained; categorical flags use exact source
endpoints. Missing deployment device orientation is rejected instead of being
fabricated.

This is an unregistered generic estimated-reference candidate. It does not add
a fourteenth formal dataset, change the frozen 2,067-cell matrix, establish an
independent ground-truth claim, approve the AI-IMU-DR actual-100-Hz vehicle
input path, or complete the released paper protocol. The original 100 Hz
sidecar remains separate and immutable.

Actual evidence identities:

- receipt: `5c82dfb3b9aff8364c786a148fd1d0bc1010ca8861a33c9218c6489dd33b1fe2`
- JUnit: `8cbdc5750c78819e8578823bf6d4b8b13986191716bc6ab2e7edb3233ef367d5`
- quality manifest: `38c675388d74c96829c3dbe67650ff4725fe446229afd00a065f47472a88edf0`
- native full/hash check: `221d3cf3a61392fcce958c48b514fdba76103b679fddf5fdf5dd65cb8bb1e68b`
- native dataset: `0dd8ca195d4281fb10ba26788ff93b17cad8b595265238f2ef23ff96ee0ab4c0`
- conversion report: `33d1c24980864d5340193c79b622606e015338f54635e8bab5b5140581f50ad3`
- before/after/finally guard: `fd28a70bfe030c2d546f5c14801791c48d88a37af003e21a54d81015df7e0991`
- native fingerprint: `2955e2845151fc6fa9c38f5e6a4d7156ac2209554c1b6810ee10339c8c7b9355`

The first derived scope was rejected before snapshot creation or scientific
execution because the raw JUnit SHA had been transcribed incorrectly. Its
failed receipt is preserved as
`2fe34f5f3aa3f197839741316379b201927d0fe3320806d7d9c01445d57c0621`.
The accepted result used a new scope bound to the JUnit bytes recorded by the
raw receipt, mirror manifest, local mirror and server original.
