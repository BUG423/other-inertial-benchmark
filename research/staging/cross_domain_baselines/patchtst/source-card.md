# PatchTST channel-independent encoder with ordered velocity adapter PoC

Status: unregistered, staging-only, and outside the formal 2067-row matrix. It
adds zero accepted methods and carries no paper-accuracy, GPU or active-registry
claim. Source-bound AutoDL CPU scientific and native lifecycle acceptance is
recorded in the [shared receipt handoff](../CPU_HANDOFF.md).

## Qualified primary sources

The primary paper is [A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers](https://arxiv.org/abs/2211.14730), Yuqi Nie, Nam H. Nguyen,
Phanwadee Sinthong, and Jayant Kalagnanam, accepted at ICLR 2023. The official
implementation is [`yuqinie98/PatchTST` at commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9),
verified at `main` on 2026-10-14. The repository license at that commit is
Apache-2.0. Exact source-file hashes are recorded in `provenance.json`; no
upstream file or weight is vendored here.

The paper and official code agree on the identity-defining mechanics:

- each univariate channel is divided into overlapping length-`P` patches;
- the end is padded by repeating the last value for one stride;
- patches are linearly embedded and receive a learnable position embedding;
- channels are folded into the batch, so attention and feed-forward operations
  do not mix channels while patch projection, position encoding, Transformer,
  and the original non-individual forecasting head share weights; train-mode
  BatchNorm pools statistics over the folded batch and patch positions;
- the supervised path uses MSE. Self-supervised masked reconstruction is a
  separate training path, not a prerequisite for supervised PatchTST.

## Fidelity identity and inertial adaptation

`patchtst_ci_ordered_velocity` / `PatchTSTVelocity` retains patch tokens,
residual-score multi-head attention, feed-forward residual blocks, temporal
BatchNorm, both upstream projection and attention-residual dropout operations,
and the shared channel-independent Transformer. The encoder returns upstream's
`(B,C,D_model,N_patch)` layout. Position embeddings use upstream's learnable
`pe='zeros'` initialization, uniform in `[-0.02,0.02]`, rather than zero values.

Upstream forecasting preserves the identity of each output channel even when
its head weights are shared. A velocity vector requires a task adapter. This
draft flattens encoder features in ordered `C,D_model,N_patch` order, applies
optional adapter input dropout, then a single learned linear map to `(B,D)`
velocity. That linear head can weight gyro/accelerometer units and x/y/z axes
separately. Its learned channel combination is an explicit inertial adaptation;
the source's channel-independence claim applies to the encoder. No encoder
attention, convolution, or gate mixes channels.

The earlier shared-head-plus-channel-mean draft was invariant to all six-channel
permutations. That head erased physical sensor and axis identities and is
superseded; it is not the primary inertial adaptation or current acceptance
evidence. The ordered adapter has a separately named identity so it is not
confused with the original forecasting model.

Causality is explicitly `window_noncausal`: unmasked attention sees all patches
of the observed window. Average velocity is emitted at the benchmark window
center, so streaming latency includes the remainder of that observed window;
this draft makes no zero-lookahead or causal-sequence claim.

In evaluation mode the encoder is channel-permutation equivariant, and changing
one channel leaves the other encoder outputs unchanged. In training mode
BatchNorm pools statistics across `B*C` channel sequences and patch positions,
exactly as upstream does. CI describes shared channel-wise attention/FFN
paths; it does not imply statistical independence during training.

The benchmark-specific contract is six channels in native IPB order
`[gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]`, a 200-sample gravity-world window
at 200 Hz with reference orientation and gravity retained, and raw `(B,D)`
window-average velocity in m/s. The constructor enforces exact channel order,
200 Hz, these view permissions, and no extra inputs, including privileged
initial velocity. Window length is configurable for geometry checks; formal
drafts fix it to 200. YAML records the fixed channel names in top-level
`input_channels` metadata because IPB's model `input` schema does not accept a
`channels` override. Both 2D and real learned 3D heads are drafted. For an
XYZ run, the paper-facing XY slice must come from the same XYZ prediction; the
2D YAML is a separate reference run, not a substitute for that slice.

RevIN is not enabled. Upstream RevIN denormalizes each forecast back into its
corresponding input-channel units; that operation is not dimensionally defined
when six gyro/accelerometer channels are adapted to a 2D/3D velocity target.
This omission is explicit rather than silently applying sensor statistics to
velocity. Series decomposition is also outside this first identity.

## Training boundary

The only declared training mode is ordinary supervised learning from random
initialization with the benchmark's MSE loss. `pretrained=true` fails closed.
There is no mask-reconstruction objective, pretraining checkpoint loader,
fine-tuning mode, downloaded dataset, or downloaded weight in this directory.

The YAML files are architecture drafts only. Their location and `registered:
false` marker are intentional; they are not discoverable by the active registry
and must not be copied into `src/inertial_benchmark/cfg/models` without a new
review, isolated server lifecycle evidence, and a new matrix decision.

## Scientific checks

The isolated CPU tests cover 2D/3D shapes, patch geometry/end padding, loss,
default-width XYZ backward and Z gradients, eval encoder channel isolation and
permutation equivariance, ordered adapter flatten mapping and axis/sensor
sensitivity, source attention residual scores/dropout, position initialization,
upstream output layout, BatchNorm pooling semantics, exact channel/rate/view
contracts, extra-input rejection, scratch-only enforcement and registry absence.

The old local 12-test observation applies to the superseded mean-head source and
was not bound to a source manifest. It is retained in provenance as historical
evidence, withdrawn from current acceptance. Revised tests and the shared native
lifecycle passed on AutoDL; see the source-qualified receipt above. All future
framework imports/tests must run on AutoDL according to
[`tools/ops_v1/README.md`](../../../../tools/ops_v1/README.md).

Root must deploy an isolated, hash-bound source and run on AutoDL CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<isolated-source>/src \
<server-venv>/bin/python -m pytest -p no:cacheprovider -q \
  research/staging/cross_domain_baselines/patchtst/test_scientific_contract.py
```

Passing this small CPU contract is source-qualified PoC evidence only. It is not
formal method acceptance, a dataset result, default Trainer/predict/reload
evidence, capacity qualification, or permission to mutate the frozen matrix.

