# AI-IMU-DR — official-source-derived scientific staging

This directory is an **unregistered, unexecuted engineering port**. Its code
and proposed scientific tests have only undergone local standard-library AST
review. No native KITTI recording, framework result, CUDA result, NIO lifecycle,
publication accuracy or additional formal benchmark row is claimed. The
currently running immutable campaign is unchanged.

The independent method is [AI-IMU Dead-Reckoning](https://doi.org/10.1109/TIV.2020.2980758),
IEEE Transactions on Intelligent Vehicles, 2020, by **Martin Brossard, Axel
Barrau and Silvère Bonnabel**. Full authors were checked against the
[author manuscript](https://arxiv.org/html/1904.06064v1) and
[author repository](https://github.com/mbrossar/ai-imu-dr); no author matches
excluded Shanshan Zhang. The journal Version of Record has not been obtained.
The arXiv manuscript is not asserted to be byte-identical to that version.

## Source actually obtained

The complete source-only release is pinned to
`32967812139af82cf12654c93438bf9065b7a241` (Git tree
`168fba39be752b0fd4612258efdf0736849eaf15`). AutoDL obtained seven `src/*.py`
files, README and MIT LICENSE, **nine files / 85,169 B**, with official Git
blob verification; four previous files were reused. No weights, raw data,
Git history or pretrained checkpoint was fetched. The manuscript HTML/PDF
was acquired on AutoDL; only small source text/receipts were mirrored.
Source-derived parts retain the upstream [MIT notice](LICENSE.upstream).

Network/initialization evidence is
[utils_torch_filter.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/utils_torch_filter.py).
Propagation and the selected negative gyro-bias observation Jacobian are in
[utils_numpy_filter.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/utils_numpy_filter.py).
Loss/recipe evidence is
[train_torch_filter.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/train_torch_filter.py),
physical constants and OXTS fields are in
[main_kitti.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/main_kitti.py),
and augmentation is in
[dataset.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/dataset.py).
These source facts do not upgrade the port to an exact paper reproduction.

## Preserved released scientific contract

The six sensor-body inputs are **gyro rad/s, then accelerometer specific force
m/s²**, originally from raw 100 Hz OXTS. The caller supplies original int64
nanosecond timestamps. Ordering is checked before subtraction, including
signed-int64 overflow boundaries, then differences are converted to seconds.
The type/ordering checks cannot prove original acquisition provenance.
Nominal-rate reconstructed timestamps or 10 Hz OXTS repeated/interpolated to
100 Hz are not qualified released inputs.

The initial state is an explicit `State`: IMU-to-world R, world velocity and
position, gyro/accelerometer biases, car-to-IMU Rci and lever arm tci. Released
initial R and velocity are reference-assisted, local position is zero, biases
and tci start zero and Rci starts identity. The caller must disclose this
initial privilege. No later reference pose/velocity is an inference argument.
Phone, drone, legged and underwater sequences do not automatically satisfy
the wheeled nonholonomic assumptions.

The covariance adapter retains two width-32 Conv1d layers with kernel 5 and
dilations 1/3, replication padding of four samples **on both sides after each
convolution**, ReLU and dropout .5, and a 32→2 tanh head. There are **6210 CNN
parameters plus 12 learned P0/Q scale parameters**. The released interior
receptive field is **17 samples, offsets −8..+8**, requiring 80 ms lookahead at
100 Hz. This is an offline/delayed track, not a recovered causal 15-sample net.
N is two positive **car lateral/up pseudo-measurement variances** with scales
10^(3 tanh), not an XYZ position/velocity uncertainty head. P0/Q have six
10^(tanh) scales each. Explicit train-only mean/std and all base constants are
persisted registered buffers.

The 21D tangent order is `(theta, xi_v, xi_p, bg, ba, theta_ci, tci)`.
R/v/p use SE₂(3) left group multiplication: R←Exp(theta)R,
v←Exp(theta)v+J_left(theta)xi_v, p←Exp(theta)p+J_left(theta)xi_p.
Biases/tci are additive and Rci receives a left rotation. Velocity/position
increments are **not** world-additive X-IONet coordinates. Full covariance
and cross-covariance are maintained; no 15D renamed filter is used.

Nominal propagation preserves the released right-end `u[i]`, R←R Exp(omega dt),
v←v+a dt and p←p+v_old dt+0.5 a dt². Covariance preserves exactly the released
F/G structure, `Gdt`, `Phi=I+Fdt+(Fdt)²/2+(Fdt)³/6`, and
`Pnew=Phi(P+Gdt Q Gdtᵀ)Phiᵀ`. This is a declared approximation, not an exact
discrete Jacobian or continuous spectral-density discretization. Released Q
parameters are code-step rate/noise-group parameters; physical units/scaling
must be bound by a future data/recipe contract, not relabelled as PSD.

Observation preserves `h_full=RciᵀRᵀv+skew(tci)(gyro-bg)`, selecting car
lateral/up components and residual `0-h`. The additive bg derivative is
`−skew(tci)`, matching the official **NumPy** backend. The official Torch
backend has the opposite sign. This choice is explicitly an engineering
consistency change; it is not a claim that the paper contains a typo. The
paper lever-arm naming/frame mapping remains a source-card ambiguity.
Joseph update and symmetrization are retained, with **no post-injection
tangent reset**, as in the release. A consistent reset would be a separately
named later ablation; it is not silently enabled here. Cholesky solve replaces
obsolete `torch.gesv`. No artificial jitter or PSD projection is added.

World vertical velocity remains learnable/physical: car-up≈0 is not world-Z=0.
The world output contains R/v/p, both biases, estimated Rci/tci, P and N.
Cartesian uncertainty metrics cannot treat the two N outputs as XYZ sigma.

## Supervision and recipe boundary

[relative_pose.py](relative_pose.py) preserves source decimation by ten,
start indices every ten decimated samples, and requested path lengths
100, 200, …, 800 m. At raw 100 Hz these are 10 Hz samples and one-second
start spacing. Endpoints use cumulative **reference path** length. Reference
and predicted relative displacements each use their respective anchor Rᵀ.
The denominator is the **reference local displacement chord norm**, not the
requested path length. The objective is squared error **SUM**, including
across batch/recordings, not mean reduction or the paper's tabulated t_rel.
Labels are detached. Empty pairs or a zero chord reject admission; short
records do not get a fabricated alternative loss. All pose inputs belong
only to this supervised loss path.

[training.py](training.py) exposes released Adam parameter groups (1e−4,
initial/process weight decay zero, CNN/head 1e−8), one aggregate-recording
update, per-recording finite/loss≤20 gate, clip norm 1 and preclip norm≤3 gate.
Zero/skipped updates are reported as zero. The caller explicitly chooses
augmentation: per-sample gyro/acc std 1e−4 and constant-per-recording bias std
1e−5/1e−4. Dropout follows train/eval mode. This is an isolated released recipe,
not the current formal common-recipe claim. Source 6000-sample chunk sampling,
all recording identities, optimizer/sampler checkpoint resume, full NIO
training and split-bound normalization remain unintegrated.

## Declared adaptations and unresolved boundaries

1. Gradient-safe stack-based skew, stable SO(3)/SE₂(3) formulas, modern solve
   and float64 tensors replace legacy detached scalar construction. Very-small
   angle formulas are more accurate than release first-order branches; no
   bit-exact complete-release run is claimed. Tests use actual pinned B1
   source classes for selected CNN and NumPy step parity.
2. `(B,T,32)`→`(B,T,2)` repairs the source B1-only transpose/squeeze layout.
   This permits independent B2 trajectories without replacing the network.
3. Scientific deterministic evaluation requires explicit `.eval()`; the
   released demonstration never called eval and retained inference dropout.
   Explicit MC training-mode inference is separate and must replay RNG.
4. Physics/normalization buffers are persisted; original weight-only saves
   did not bind all numeric state. Staging RNG/state replay tests do not imply
   a full multi-recording resumable NIO training checkpoint.
5. Source Torch uses no periodic SVD renormalization; source NumPy normalizes
   R every 100 and Rci every 1000 steps. This port follows the Torch convention,
   and selected B1 NumPy oracle checks are one step, below those boundaries.
6. The manuscript's Euler position propagation, causal-history description
   and reported drift statistic differ from the released code. This port
   names its released-code contract, not journal-paper accuracy fidelity.
7. Float64 GT pair construction avoids source float intermediate casts.
   Data-native 200 Hz uniform grids already lose original clock jitter and
   cannot establish this raw-clock contract by themselves.
8. Current NIO Predictor normally reintegrates returned velocity and cannot
   yet preserve filter-corrected p/R/P as a native result. A future typed
   full-sequence result/protocol hook is required; it is **not wired here**.

## Proposed server evidence — none executed yet

[test_scientific_contract.py](test_scientific_contract.py) proposes independent
NumPy/SciPy expm recursion and 21-column finite differences, nonzero lever arm,
PSD/cross-covariance, original approximate-Phi convergence diagnostics, B1
official-source parity/B2 shapes, RF/lookahead, uphill world Z, true integer
clock guards, short-record rejection, SUM/chord loss, full CNN→filter→loss
backprop with two updates, P0/Q/N gradients, numeric state and RNG replay.
Synthetic high-speed references only exercise long-distance equations;
they are not real driving recordings or raw-clock acquisition proof.
Local AST predicts 51 nodes; **actual collection and all passes are pending**.

The operator must set `AI_IMU_DR_OFFICIAL_SOURCE` to the hash-verified acquired
source-only directory. Missing oracle raises a failure, never a silent skip.
Only selected pinned class definitions are executed; no upstream demo, data
loader, package installation, pretrained weights or raw data is executed.
All execution is reserved for finite server-only CPU acceptance after review.
