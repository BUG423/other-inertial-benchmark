# ModernTCN velocity architecture port — isolated PoC

Status: unregistered research draft. No framework execution was performed locally.
Source-bound AutoDL CPU scientific and native lifecycle acceptance has passed;
see [CPU receipt handoff](../CPU_HANDOFF.md). CUDA capacity and benchmark accuracy
remain untested. This directory adds zero accepted methods.

## Source and permitted reuse

- Paper: Donghao Luo and Xue Wang, *ModernTCN: A Modern Pure Convolution Structure
  for General Time Series Analysis*, [ICLR 2024 Spotlight](https://openreview.net/forum?id=vpJMJerXHU).
- Official repository: [`luodhhh/ModernTCN@56a9a2c018385cd5acef015378cae7f084d1b11c`](https://github.com/luodhhh/ModernTCN/tree/56a9a2c018385cd5acef015378cae7f084d1b11c).
- Behavioral source: [classification model](https://github.com/luodhhh/ModernTCN/blob/56a9a2c018385cd5acef015378cae7f084d1b11c/ModernTCN-classification/models/ModernTCN.py),
  especially `ReparamLargeKernelConv`, `Block.forward`, `forward_feature` and `classification`.
- Default backbone anchor: the first (EthanolConcentration) command in
  [classification.sh](https://github.com/luodhhh/ModernTCN/blob/56a9a2c018385cd5acef015378cae7f084d1b11c/ModernTCN-classification/scripts/classification.sh).
  This is a released dataset-specific architecture, not a universal paper default.
  Its downsample ratio 2 and unmerged kernels come from `run.py` defaults.
- Upstream code is [MIT licensed](https://github.com/luodhhh/ModernTCN/blob/56a9a2c018385cd5acef015378cae7f084d1b11c/LICENSE),
  copyright (c) 2024 luodhhh. Its complete notice is preserved in `LICENSE.upstream`.
  The port refactors the reviewed classifier computation rather than vendoring
  upstream modules. The license does not imply permissions for arbitrary datasets
  or external pretrained checkpoints; neither was downloaded.
- `provenance.json` records SHA-256 of raw upstream response bytes, preserving
  original CRLF and trailing newlines. Those hashes are not normalized-text hashes.

## First-principles hypothesis

Inertial sequences contain local dynamics and interactions between six ordered
sensor channels. ModernTCN separates time mixing, feature mixing within each
variable, and variable mixing within each feature. The experiment asks whether
this decomposition offers useful accuracy, stability or efficiency under the same
inertial task as other architectures. A 200-sample window may limit the gains
associated with larger effective receptive fields; this remains a testable
hypothesis. No accuracy claim follows from a shape or gradient check.

## Keep / replace / omit

| Component | Official classification path | This port and reason |
| --- | --- | --- |
| Layout | `B,M,D,N` preserves variables and embedded features | Same layout; `M=6`, fixed gyro-then-accelerometer order |
| Patch stem | Shared `Conv1d(1,D,k=P,s=S) -> BN`; fold variables into batch | Preserved; `P=8,S=4`; end-repeat `P-S` samples |
| Stage reduction | `BN -> Conv1d(D_i,D_next,k=2,s=2)`; copy trailing token when odd | Preserved; two stages with widths 32/64 and depths 1/1 |
| Time mixing | Large 13 and small 5 bias-free depthwise Conv-BN branches summed | Preserved unmerged training graph; groups=`M*D`, symmetric zero padding |
| Normalization | `BN(D)` after reshaping to `B*M,D,N` | Preserved, including shared statistics over variables |
| Feature FFN | Pointwise grouped convolutions with groups=`M`, GELU and two dropouts | Preserved; expansion ratio 1, dropout 0.1 |
| Variable FFN | Permute to `B,D,M,N`, then pointwise convolutions with groups=`D` | Preserved; permutation is required to mix variables rather than features |
| Residual | One residual around time mixing and both FFNs | Preserved; no extra FFN residuals added |
| Head | GELU, class dropout, flatten `M*D*N`, linear to classes | Replaced by temporal mean, ordered `M*D` flatten, one linear XY/XYZ velocity head |
| Supervision | Dataset classification loss and original UEA recipes | Shared average-velocity MSE and future unified inertial recipe |
| RevIN | Constructed but never called by classifier feature/classification path | Omitted unused module; no sensor-statistic denormalization into velocity units |
| Forecasting facilities | Forecast head instantiated but not used by classification | Omitted unused forecast head, calendar features, decomposition and forecast machinery |
| Reparameterization | Optional Conv-BN/small-branch fusion for inference | Deferred; PoC executes unmerged graph and makes no fused-latency claim |
| Wrappers | `c_in` tuple convention, task wrapper and unused `stem_ratio`/`dw_dims` | Replaced by BaseModel/InputSpec; unused constructor settings are not fabricated into operators |

The default XYZ port has **89,251 trainable parameters**; XY has **88,866**.
These are analytically derived port counts, pinned by server tests, not original
paper parameter counts. T200 produces 50 stem tokens and 25 final tokens. The
task-head reduction makes parameter count independent of T within a profile.

## Scientific task contract

`ModernTCNVelocity` receives exactly `B x 6 x T`; supplied drafts fix T=200,
200 Hz, channels `[gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z]`, `gravity_world`,
reference orientation and retained gravity. It rejects channel reordering,
history, extra inputs, alternate frames/targets/rates and gravity removal.
An explicit device-orientation InputSpec remains a separate scientific track.
The input transformation is performed by the shared benchmark view, not inside
this model. There are no position labels, initial velocity or external context.

The primary task returns `{"vel": B x 3}` in m/s and learns actual Z with a
third linear output row and gradients into the actual gyro-Z/acc-Z channels.
Its XY evaluation takes the same prediction's `[:, :2]`. The XY YAML is a
declared secondary 2D reference run; it cannot supply XY rows for an XYZ run.
There is no uncertainty head. Initialization is supervised from scratch.

The model is window-noncausal: symmetric convolutions and temporal pooling see
the whole supplied window. They cannot read outside that window. Changing T
requires a separately identified history/window comparison shared by controls.
Small T/width choices used in structure tests are not capacity receipts for the
default profile. Downsample ratio is fixed at two; other padding semantics need
their own reviewed profile.

## Verification and remaining assumptions

`test_scientific_contract.py` checks the official grouping and permutations,
feature-versus-variable dependency separation, temporal channel isolation,
single residual, patch/end padding, parameter counts, deterministic head,
shared MSE, real Z gradients/updates, input rejection and unregistered drafts.
Root also supplies a shared server native-lifecycle test for real fixture
training, validation, prediction and checkpoint reload. Neither test is run
locally; the operator executes them on AutoDL under a source-bound CPU receipt.

Remaining engineering assumptions are explicit: choosing the released Ethanol
backbone for inertial windows, temporal mean instead of classification flatten,
the common reference-orientation gravity-world view, and unified inertial
training hyperparameters. They require pilot evidence. Structural reparameterization,
paper-result reproduction, CUDA capacity, export adapters, three-seed training
and formal admission remain pending.
