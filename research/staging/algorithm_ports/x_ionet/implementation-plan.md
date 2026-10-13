# X-IONet scientific staging implementation plan

Status: **primary-source review and implementation design only**. There is no
X-IONet model code, runtime acceptance, registration or benchmark increment in
this folder. The proposed core work needs a separate review before implementation.
The running source, queue, accepted Motion leaves and Abol scalar files stay fixed.

## Primary source and recovered scope

The authors are **Dehan Shen and Changhao Chen**; neither matches excluded
Shanshan Zhang. The original and revised titles belong to one arXiv record and
must count once. Sources: [author-submitted v2](https://arxiv.org/html/2511.08277v2),
[complete PDF](https://arxiv.org/pdf/2511.08277v2),
[RA-L DOI](https://doi.org/10.1109/LRA.2026.3692056), and
[corresponding author's publication page](https://changhao-chen.github.io/).
No official method code has been verified through these sources and bounded
exact-title searches; this does not prove code never exists.

The recovered system uses a platform CNN, separately trained pedestrian/robot
experts and an EKF. Inputs are six axes, 200 samples at 200 Hz. Expert embedding
uses 10-sample segments and width 256, temporal then inter-axis attention, a
router, hierarchical merging and multiscale decoding. Outputs are displacement
and a full 3×3 covariance. The objective combines Huber delta .005 with Gaussian
error/logdet weighted 1e−4. The filter has 15 current error coordinates plus six
per historical pose. These are recovered paper facts; the derivations and
implementation choices below are ours.

The PDF (5,416,626 bytes), full HTML (232,776 bytes) and original Fig. 3 PNG
(1,227,280 bytes) were actually acquired on AutoDL using official URLs and bounded
requests. The PDF remains on the server; only small receipts and a figure mirror
were used for this review. No raw dataset, weights or framework tests were
downloaded or run. Asset hashes and unresolved fields are in [provenance.json](provenance.json).

## Network details that remain unresolved

Direct inspection of official Fig. 3 reveals **no numeric channel widths, kernel
sizes, strides, padding, pool sizes or hidden FC width**. Its pipeline shows three
convolution blocks, global average pooling, FC1, ReLU, FC2 and softmax. Section
III-A gives Conv/BN/ReLU/MaxPool and describes a final classifier FC without
specifying that intermediate FC. Preserve this text/figure distinction; do not
invent defaults and call them author settings.

For the experts, segment count 20 is **not evidence of 20 encoder layers**.
Attention heads, actual encoder/decoder depth, router capacity, MLP width,
dropout, odd-segment merge/padding, final readout and covariance parameterization
are not resolved. A Crossformer reference does not establish X-IONet's settings.
Independent expert parameters/checkpoints and hard platform routing are required;
a generic shared Transformer renamed X-IONet would omit the method.

Train the classifier using platform labels from training data only; train the
two experts on their respective supervised training domains. At inference the
selector reads IMU, not dataset name or a true platform label. Record the human/
robot class mapping, confidence and routed expert for every window. Confidence
gating/hysteresis is unreported and cannot be silently added. Network widths must
either be recovered or explicitly approved as reconstruction assumptions.

## Error-state and clone contract

Represent the nominal current state as `(R,v,p,bg,ba)` with body-to-world R.
The proposed tangent state is ordered `(theta,dv,dp,dbg,dba)`, length 15.
Each clone contains only historical `(R_i,p_i)`, with `(theta_i,dp_i)` length six.
For n clones the covariance dimension is **15+6n**. This does not mean a fixed
13-dimensional state or a fixed number of six clones. The paper does not supply
the clone-retention count or scheduling policy.

Use explicit left perturbation for the recovered rotation error:
`R_true = Exp(theta) R_est`. A clone augmentation matrix copies current rotation
and position error coordinates and retains all clone/current cross-covariances:
`P_aug = J_aug P J_aug.T`. Removing an old clone selects the retained rows and
columns; it does not zero cross-covariances or reset current state.

Exact deterministic cloning can make `P_aug` singular. State covariance must be
symmetric **positive semidefinite**, not arbitrarily forced to strict SPD by
adding independent fake clone noise. Prediction covariance Sigma and innovation
covariance `H P H.T + Sigma` must be SPD. A Cholesky solve is therefore appropriate
for the innovation covariance, not an unjustified inversion of cloned P.

The nominal propagation follows the recovered R update, acceleration update and
trapezoid position rule, while mean biases are held constant. Interpreting bias
random walks through process covariance rather than sampling arbitrary noise
into the estimated mean is an explicit standard error-filter interpretation.
Original noise densities, initialization covariance, discretization and reset
Jacobian are unreported. Derive F/G from the chosen left-error propagation and
validate them with finite differences; do not transplant a right-error matrix
without converting its convention.

## Eq. 17: a real sign ambiguity, not a silent fix

Under ordinary additive position injection and `H = derivative(h)`, the printed
positive `K*(h-d)` update increases a simple position residual. Our independent
one-dimensional counterexample has predicted displacement 1, measurement 0,
P=1 and Sigma=1, giving K=1/2. Literal addition produces 1.5 and a larger error;
the consistent `K*(d-h)` update produces .5 and a smaller error.

Changing H to a negative Jacobian or defining negative boxplus could account for
the printed form, but those definitions are not supplied. Rotation error in the
text is left-sided; the additive position-error definition is not fully spelled
out. Do not quietly negate the residual and claim exact implementation.

Proposed leaf contract: document both a **paper-literal diagnostic** and an
explicit **consistent left-error EKF reconstruction**, with the latter pending
review. The diagnostic reproduces the counterexample and is never presented as
a qualified operational filter. The reconstruction injects `K*(d-h)` with its
derived Jacobian and Joseph covariance update, then applies the correctly derived
error-coordinate reset. If an author clarification specifies another convention,
it must receive its own source/provenance revision and tests.

For a declared full-body anchor frame `h = R_i.T*(p_j-p_i)`, our left-error
Jacobian blocks are `H_theta_i = R_i.T*skew(p_j-p_i)`,
`H_p_i = -R_i.T` and `H_p_j = R_i.T`. This is a mathematical test contract,
**not a resolution of the paper's actual local frame**.

## Local-frame and clock alignment are full-system gates

The paper's diagram specifies gravity-aligned local input built using **estimated
EKF attitude**. Eq. 15 does not fully identify which R/local anchor is used.
A full body rotation `R_i` and a yaw-only gravity-local rotation are different
measurement frames and have different orientation Jacobians. Before a full port,
pin the exact local reference, define IMU rotation and displacement/covariance
in that same frame, and derive/test the corresponding Jacobian. Never pair
gravity-local predictions with a body-frame measurement function by name alone.

Window/clone endpoints must match actual timestamps. Two hundred readings at
200 Hz contain 199 intervals between their first and last samples; a nominal
"one-second" label cannot silently replace that span. Update stride, clone
timestamps, history, startup latency and alignment are not recovered.

Initial yaw/attitude, biases and covariance also need an explicit policy.
Six-axis IMU alone does not reveal arbitrary absolute yaw. A true initialization
pose, continuous reference quaternion, stationary calibration or estimated
attitude are distinct privileges/assumptions and must be recorded. Using reference
attitude through the current labelled task view creates a named privileged
network-only adaptation, not the original closed-loop IMU-only system.

## Full covariance proposal, not an author parameter claim

A reliably implementable leaf can expose a **declared Cholesky reconstruction**:
six lower-triangular head values, positive diagonals through Softplus plus an
explicit numerical floor, `Sigma = L L.T`. The floor, head parameterization and
loss reduction are assumptions requiring approval; no author numbers are guessed.
Compute Mahalanobis error with triangular solves and logdet from L, avoiding an
explicit inverse. Keep m² covariance paired with the same raw local displacement
in meters. Retain off-diagonal terms; diagonal logstd cannot replace this full
matrix. Rotate covariance with the same frame map as displacement.

Original Huber and Gaussian weights stay separate. A batch reduction rule must
be named, and no undocumented NLL phase switch is introduced. Future uncertainty
metrics need a reviewed full-covariance contract; present diagonal-only helpers
cannot be fed a fabricated world-diagonal sigma.

## Reviewable implementation order

1. **Pure geometry/filter leaf first:** proposed independent `geometry.py`,
   `filter.py`, `covariance.py` and focused tests inside this staging folder.
   Explicit state layout, clone augmentation/marginalization, timestamp schedule,
   left retraction, propagation and measurement interfaces; no active engine.
   Noise/P0/frame/reset/sign choices must be constructor contracts, not claimed
   paper defaults. This work is waiting for root approval.
2. **Scientific CPU gates:** SO3 Exp/Log/small-angle and retraction tests;
   finite-difference propagation and measurement Jacobians; clone cross-covariance
   and PSD tests; stationary specific-force/gravity and known-motion propagation;
   zero residual/no change; printed-sign counterexample; consistent update reduces
   known residual; correlated SPD likelihood oracle; Joseph symmetry/PSD and
   manifold reset; singular/bad-frame/clock rejection; exact state serialization.
   Tests execute only on AutoDL after static review, with a bounded CPU budget.
3. **Selector/expert reconstruction:** recover missing numeric specs or approve
   explicit assumptions. Preserve the full two-stage routing and both independently
   trained experts. Test classifier CE, hard route/no true-platform inference input,
   temporal/inter-axis attention, hierarchical merge/decoder and real XYZ/full-Sigma
   gradients. Arbitrary default widths are not inserted in this plan.
4. **Real datasets and loop:** the paper uses RoNIN and a screened CPT7/odometry
   GrandTour training subset. The current ADIS/prism single GrandTour test mission
   cannot supply robot training/selection labels without leakage. Acquire/admit
   proper train/val/test robot data, extrinsics and reference fidelity first.
   Closed-loop windows consume only IMU and estimated state; poison reference
   positions/velocities/quaternions after allowed initialization to prove isolation.
5. **Future unified integration:** supervised selector/expert stages and complete
   multi-checkpoint provenance, standard validation/export contracts, then the
   estimated-attitude filter sequence protocol. Present SequenceModel velocity
   integration cannot silently replace a filter's already corrected pose history.
   Shared engine/data changes require a separate owner-reviewed design.

There is no new registry entry, model alias, training job or GPU action here.
Network-only privileged ablations and a complete original-system reconstruction
are distinct tracks of one paper. Neither is accepted or counted until its actual
source/data/lifecycle/capacity tests pass.
