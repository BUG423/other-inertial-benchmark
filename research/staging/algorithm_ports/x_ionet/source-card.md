# X-IONet: independent geometry and filter staging core

Status: **source implemented; static review only; server CPU tests pending**.
This folder has an independent mathematical core, not a trained X-IONet model,
an active registry entry, an accepted benchmark result or a complete original
system. It adds no architecture, profile, paper or formal queue row to the
running benchmark. No NIO/Trainer/Validator/export integration is supplied.

[implementation-plan.md](implementation-plan.md) and
[provenance.json](provenance.json) are the unchanged historical **design** record.
Their design-only and pending-approval fields describe that earlier review;
this card records the subsequently approved engineering core. They are not
silently rewritten to claim implementation or test acceptance.

## Source, authors and unresolved original implementation

The primary paper authors are **Dehan Shen and Changhao Chen**; neither is
Shanshan Zhang. The versions of arXiv 2511.08277 constitute one method.
Sources: [complete author manuscript, v2](https://arxiv.org/html/2511.08277v2),
[PDF](https://arxiv.org/pdf/2511.08277v2),
[RA-L publication](https://doi.org/10.1109/LRA.2026.3692056), and
[author publication page](https://changhao-chen.github.io/).

The paper combines a platform classifier, two separately trained experts and
an EKF; its networks predict displacement and full covariance. The recovered
error state contains 15 current coordinates and six per historical pose. The
objective specifies Huber delta .005 and Gaussian weight 1e-4. The estimated
attitude is part of its input transformation. These recovered ingredients do
not establish the unreported numerical network parameters or filter settings.

Official author network source has not been verified. Classifier channels,
kernels/pooling/FC widths, expert depth/heads/router/readout and covariance-head
parameterization remain unresolved. This implementation supplies none of them;
it contains no generic neural architecture renamed X-IONet. Paper noise and
initialization values, clone timing, local measurement frame and reset policy
are also unresolved. No original code or weights are copied.

The actual official HTML/PDF/figure acquisition and hashes remain in the
historical provenance. The PDF remains on AutoDL. This core implementation
does not download data/weights or change accepted native assets.

## Exact engineering conventions implemented here

[geometry.py](geometry.py) uses float64 NumPy SO(3) operations. Inputs must
already be proper rotations; no silent normalization is applied. Small-angle
Exp/Log series and a near-pi axis reconstruction are included. At exactly pi
the largest-magnitude axis component is chosen positive; the axis sign is
mathematically ambiguous. The filter's correction chart is strictly below pi.

[filter.py](filter.py) declares, rather than infers, these conventions:

- Body-to-world rotation `R`; left perturbation `R_true = Exp(theta) R_hat`.
  World-additive velocity and position; sensor-coordinate additive biases.
  Current tangent order is `[theta,dv,dp,dbg,dba]`, followed by
  `[theta_i,dp_i]` for each historical clone: dimension `15 + 6*n`.
- Constructor requires initial state/P0/clock, world gravity, all four variance
  spectral densities, discretization, frame, reset, maximum interval and
  rotation correction limit. **There are no original-paper numeric defaults.**
- Named discretization is `piecewise_constant_measurement_average`. Nominal
  propagation holds the old attitude and specific force through the interval,
  uses Euler velocity and trapezoid position, and holds mean biases constant.
  Bias random walks enter covariance, not invented random mean perturbations.
- Four independent isotropic variance PSDs have units `rad²/s`, `m²/s³`,
  `rad²/s³` and `m²/s⁵` for gyro, acceleration, gyro-bias walk and accel-bias
  walk. These are neither standard deviations nor per-sample variances.
  Measurement noise is modeled as an interval-average error with integrated
  covariance `PSD*dt`; bias increments occur at the interval endpoint.
- The discrete transition is derived from this integrator. In particular,
  `F_theta,bg = -R_old J_left((omega-bg)*dt)*dt`, not body-frame `-I*dt`.
  Acceleration orientation coupling is `-skew(R_old*(acc-ba))*dt`.
  Position coupling uses the corresponding half-dt² factor. The specified
  held average-noise approximation gives position variance `.25*q_a*dt³`;
  it is **not** a continuous-white-noise/Van Loan discretization with `1/3`.
- Clock values are explicit Python signed-int64 nanoseconds. Ordering and
  representable positive differences are checked before floating conversion.
  A nominal rate is never used to manufacture propagation time. The caller
  remains responsible for actual acquisition-clock provenance.
- `clone_current()` copies current attitude/position tangents with
  `P_aug = J_aug P J_aug.T`, retaining all cross-correlations. Exact duplicate
  coordinates can produce singular PSD covariance; no fake clone jitter is
  added. Caller-selected `marginalize_clone(timestamp)` keeps the corresponding
  principal submatrix. Clone clocks are explicit and unique; there is no
  invented original window/retention schedule.

The measurement contract is explicitly **full body anchor**:

`h = R_i.T*(p_current-p_i)`

Its analytic blocks are `H_theta_i = R_i.T*skew(p_current-p_i)`,
`H_p_i = -R_i.T` and `H_p_current = R_i.T`; current attitude has no direct
measurement column. Both displacement and covariance must refer to that same
historical body frame, in meters and meters squared. This is an engineering
test contract, **not a resolution of the paper's gravity/yaw-local frame**.
The code rejects an unqualified gravity-yaw frame instead of silently pairing
unmatched outputs with this observation function.

Operational update uses `K*(z-h)` with positive measurement derivative and
plus injection. Kalman gain uses Cholesky solves, covariance uses Joseph form,
and error-coordinate reset follows injection. The rotation reset block is
`J_left(delta_theta)`, derived from
`Log(Exp(delta_theta+eta) Exp(-delta_theta))`; its small-angle first term is
`+0.5*skew(delta_theta)`. Every current/clone rotation block receives its own
reset; other declared world-additive errors keep identity reset.

`paper_literal_sign_diagnostic()` exposes the printed `K*(h-z)` using the
**same** positive-H/plus convention without mutating a filter. The unit example
predicts 1, measures 0 and gives gain .5: consistent injection gives .5,
while the printed expression under that convention gives 1.5. This establishes
a convention ambiguity; it neither proves the authors made a typo nor claims
the consistent reconstruction is their exact implementation.

State covariance accepts symmetric PSD matrices, including singular clones;
measurement and innovation covariance must be SPD. Numerical symmetry/PSD
checks use a stated scaled 1e-10 tolerance without clipping eigenvalues or
adding noise. Negative/singular measurement covariance is rejected. Updates
compute and validate candidates before committing state. JSON `state_dict()`
includes all numeric state, clones, clock, P, units/configuration parameters
and schema; reload supplies no missing scientific defaults. The core has no
RNG, GT input, neural model or hidden training phase.

## Full covariance objective: declared parameterization, recovered weights

[covariance.py](covariance.py) exposes a differentiable Torch helper with six
lower-triangular entries ordered `[L00,L10,L11,L20,L21,L22]`. Diagonals use
Softplus plus a **caller-required** standard-deviation floor in meters;
`Sigma = L L.T` retains off-diagonal terms. This positive-diagonal head is an
explicit reconstruction assumption, not a recovered original network head.
The physical floor is not extra jitter applied to a singular covariance.
Numerical loss of SPD is rejected rather than repaired.

Loss requires raw displacement prediction/target in the same XYZ local frame,
meters, and full covariance in m². Coordinate Huber terms use delta .005 and
are summed per sample. Gaussian terms use a triangular solve for the full
Mahalanobis term and log-diagonal factor sum for half-logdet. The Gaussian
weight remains 1e-4. A caller must name `sum` or `mean_samples`; original
reduction is unknown. There is no invented MSE/NLL phase switch, diagonal
sigma substitution, detached head or 2D target padded with fake Z.

## Proposed finite scientific server acceptance

[test_scientific_contract.py](test_scientific_contract.py) loads the leaves
in an isolated namespace, never the active model registry. All fixtures are
synthetic engineering values. SciPy Rotation is an independent numerical
reference for finite differences, rather than using the implementation's Exp
to test its own derivative. The proposed checks cover:

- Exp/Log at zero, small angle and pi; canonical pi ambiguity; left reset
  derivative and positive-half-skew sign.
- All 15 propagation columns and all 12 declared noise-increment columns;
  explicit PSD units/discretization; tilted stationary specific force and
  known linear acceleration.
- Exact clone augmentation/cross-covariance and singular PSD; propagation
  through clones; principal-submatrix marginalization; observation derivative
  over all current/clone coordinates.
- Positive-H/plus residual counterexample; zero-innovation state preservation;
  full correlated Joseph update followed by independent left-reset finite
  differences for current and two clones.
- Exact JSON scientific state replay; clock/interval/frame/reset/noise failures
  without mutation; no fake SPD repairs.
- Off-diagonal full Gaussian oracle against NumPy solve/logdet, real third-axis
  prediction gradients, six-factor gradients and central-difference gradcheck;
  explicit reduction; invalid covariance rejection without mutation.

No local framework import, pytest, native data processing or model execution
is performed. Static AST checks and estimated node counts are not actual
collection or PASS evidence. After root review, the execution owner can run:

```bash
python -m pytest research/staging/algorithm_ports/x_ionet/test_scientific_contract.py -q
```

That command is intended only inside an isolated AutoDL source snapshot under
an approved bounded supplemental CPU lease (two Torch threads, interop one,
CUDA disabled). Parent/framework threads must be recorded. CPU qualification
must preserve formal source/native/accepted staging guards, actual collection,
JUnit, logs and terminal hashes. A busy lease is pending, not an algorithm pass.

## Remaining gates to a complete method

This is not a full selector/expert implementation, real-data closed loop,
standard NIO lifecycle, native export or GPU capacity qualification. Unknown
network settings and the exact local frame need primary clarification or an
approved assumptions track. Proper independent robot training references,
extrinsics and screened sequence identities are also required. An available
single ADIS/prism GrandTour test mission cannot stand in for the paper's CPT7
training subset. Reference attitude is a privileged named adaptation, not
the original estimated-attitude system. Initialization yaw remains explicit
because six-axis IMU cannot determine arbitrary absolute yaw. None of these
gates are waived by a successful synthetic core test.
