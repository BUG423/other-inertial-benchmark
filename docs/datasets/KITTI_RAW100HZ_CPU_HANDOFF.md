# KITTI original 100 Hz OXTS: lossless candidate CPU handoff

The server-only whitelist contains 41,547 OXTS packet files, three original
nanosecond clock files and six calibration files. A finite CPU audit completed
in 16.866 seconds: 12 actual NumPy/h5py scientific checks passed; all 41,556
original members retained their acquisition SHA. HDF5 readback preserved every
packet value, original text, source index, clock and quality flag. Torch and
CUDA were not imported or requested. Existing 13 formal datasets and the 2,067
cell experiment plan were unchanged. Root review accepted this only as an
unregistered lossless candidate. The separately reviewed generic 200 Hz result
is recorded in [KITTI_200HZ_CPU_HANDOFF.md](KITTI_200HZ_CPU_HANDOFF.md).

| Preregistered recording split | Original packets | Longest clock-contiguous run | Clock backjumps / gaps over 50 ms | Whole-run 800 m pairs |
| --- | ---: | ---: | ---: | ---: |
| Train: `2011_10_03_drive_0042_extract` | 12,194 | 9,991 | 1 / 1 | 63 |
| Validation: `2011_09_30_drive_0033_extract` | 16,599 | 10,575 | 3 / 3 | 30 |
| Test: `2011_09_30_drive_0034_extract` | 12,754 | 8,740 | 1 / 1 | 0 |

This is a preregistered three-recording engineering split, not the complete
released paper protocol. Contiguous segments use original source-index order;
the five backjumps and five gaps remain in the sidecar. There is no timestamp
sorting, clock repair, local raw download or uniform-rate conversion in this
audit. The civil clock's timezone is unknown: its integer nanoseconds do not
establish an absolute UTC timestamp.

Distance supervision uses the released rule: downsample by ten, one nominal
second between permitted starts, and endpoints at available cumulative path
lengths from 100 to 800 metres. A positive local chord supplies the loss
denominator. In 6,000-original-sample windows spaced by 100 source rows, train
has 40 windows containing an 800 m pair; validation has no 700/800 m windows
but 34 containing a 600 m pair; test has no 600/700/800 m windows and three
containing a 500 m pair. Missing distance bins are coverage **not available**,
not grounds to reject the dataset or replace the released objective. Available
shorter-distance pairs remain usable under their declared reference protocol.

The converter retains thirty source fields. Gyro is zero-based `17:20` in
rad/s, acceleration `11:14` in m/s², and velocity is `[ve, vn, vu]`. Position
uses translated Mercator east/north/up; body-to-world quaternion is `wxyz` from
`Rz(yaw) Ry(pitch) Rx(roll)`, as in the released AI parser. This preserves the
ENU direction instead of applying the official MATLAB demonstration's
first-pose rotation. Source velocity's upward component is perpendicular to
the Earth surface; altitude datum and the physical output anchor remain
source metadata uncertainties. No vehicle-to-IMU translation or mounting
correction is guessed from IMU-to-Velodyne calibration.

OXTS reference is a GPS/INS estimate sharing the IMU input, with privileged
orientation and initial velocity; it is not independent camera/mocap truth.
Navigation status and all three GNSS modes are retained. The manufacturer's
mode zero means the primary GNSS cannot make that measurement, so it is
reported separately from an invalid locked-INS solution. The all-modes-positive
sensitivity mask contains 3,269 / 7,453 / 5,139 rows. Negative modes identify
upstream interpolation and are excluded from candidate input; none occurred
in these three recordings. World-rotated acceleration median vertical values
are 9.82166 / 9.80822 / 9.80837 m/s². Same-INS position/velocity consistency is
diagnostic evidence, not an independent accuracy result.

The underscore-prefixed parser is excluded from automatic converter discovery.
Its HDF5 candidate schema is separate from formal IPB native sequences. The
subsequent generic 200 Hz derived track requires its own resampling, geometry,
mask and native-format acceptance. The original 100 Hz sidecar remains for a
future actual-dt full-sequence filter. Vehicle/NHC frame and lever-arm permission
remain an AI-specific admission question; their uncertainty does not alone
invalidate a generic estimated-reference native track.

Actual receipt SHA256:
`cefe15a9e3e8550189204f69202f83b0ded2535976ff4dcb9702c99939e81af4`.
Scientific JUnit SHA256:
`4f7c7e863afaf8127cf395f5836335cacf41b4b51c8a710b5b4f00c3b1fd733f`.
Quality manifest:
`d5f3edb4a505db2dc4aae7c8f91022bab6467f9cd3e51006c67285998eabd653`.
Before/after protected-source, native-manifest, split and experiment-identity
guard SHA:
`e8d69acab82531bbcd30b1432a5a99e8beb941ac2e6279004efb44fb401ddaac`.
The first nonblocking attempt returned resource-pending 75 before any numerical
execution; its original receipt remains preserved separately.

Primary evidence: [KITTI IJRR data paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf),
[official raw-data development kit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip)
(ZIP SHA256 `5fe511ba02b9588c4b18ad154e6d31d716e16b6ad115d1ac00d13d6d5bbcb6bc`),
[OXTS NCOM documentation](https://www.oxts.com/software/navsuite/documentation/manuals/NCOM_man.pdf)
(current manufacturer semantics, not a certificate of KITTI's 2011 firmware),
and [released AI parser](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/main_kitti.py).
