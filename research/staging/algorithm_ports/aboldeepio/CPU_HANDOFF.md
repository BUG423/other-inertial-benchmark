# AbolDeepIO original-scalar CPU scientific handoff

Accepted engineering scope: **actual AutoDL CPU collection and 36/36 scientific
tests**, with zero failures, errors or skips. All inputs and clocks in this batch
were **synthetic fixtures**. This is not admission of an original acquisition
clock, standard NIO lifecycle, XYZ trajectory, GPU capacity or journal accuracy.
The method remains unregistered and contributes no active architecture, profile,
paper-method or formal-matrix increment.

The five original [model](model.py), [tests](test_scientific_contract.py),
[source card](source-card.md), [provenance](provenance.json) and package files
remain byte-identical to the reviewed static v2 manifest. Their pre-runtime
status fields are preserved. This separate document records the subsequent
engineering evidence without rewriting the tested source identity.

## Actual server evidence

The bounded supplemental CPU worker ran on 2026-10-13 UTC, from
21:04:41.349304 to 21:04:58.333040, with parent wall time **16.984 seconds**.
Actual collection and the scientific pytest process both returned exit code 0.
The pytest batch took **5.766 seconds**; collection separately took **5.765
seconds**. The operator checked collected node IDs against the JUnit cases.

The independently composed immutable source contained **452 managed files**:
the previously qualified 447-file leaf snapshot plus these five scalar staging
files. Its manifest SHA256 was
`5e9bafdcdcf85ed6f2acd34aee07b0fe81023566953a55d54174e26812a44f83`.
The server source root was
`.local/source-snapshots/31ae82d206bdda2043fa7595577b183f817c4b33f50e7d377b0fc529769df406/repo`.
Its parent 447-file manifest was
`3cf84aed2cc2d41ce22355548ec21efedcae792f0f9915140a795327d4615677`.

Private original receipts and logs are retained at
`.local/reports/abol-scalar-science-31ae82d206bdda20/`:

| Artifact | SHA256 |
| --- | --- |
| `receipt.json` | `cf316968cff23f767a34e36862958239935b4710e2ebe4eecd5717b09c36954a` |
| `focused_1.xml` | `7bff23fc89cd4d3091f4348c8ecb7186fb33f04768510cbd81d3fbd7587a809a` |
| `actual_collection.log` | `6f833b3ae1c87ecaad9b54ca2ec37959587f3e4dfd2942e58bb87606132931c0` |
| `focused_1.log` | `659364c99a6d80a5c959f9799d51a243c0eed7eb37d4ccb50bcd583f0812c872` |

The private runner SHA was
`fa1e24e7f7ec04dc3b906fa3d3dc776fed075605a2ae7a4ba7e0b57400a26811`;
the execution request SHA was
`ef58d9bb1c93f2ba9efb9ef22d5053fe96122975023f0d9ca95d2275d1892166`.
The static v2 manifest remains
`.local/research/algorithm-primary/expansion-aboldeepio-20261014/static-ready-v2.json`,
SHA `145b29c25dfed0aa27ff129fa67000b1b20eb1b0cf24359b404c0b0912177e68`.
The original v1 manifest is archived unchanged and was not silently relabelled
as runtime-qualified.

Execution used Torch **2.12.1+cu130**, Torch threads **2**, interop threads **1**
and OMP threads **2**. CUDA was not initialized. Only the approved supplemental
CPU lease was used; the original CPU training lease and GPU lease were not
acquired. The worker exited and released its lease. Protected source snapshots,
native metadata, Motion staging/leaf files and formal policy/capacity records
remained unchanged. The formal 425-file source and its 2,067 rows were not altered.

## What the 36 scientific cases establish

- The full reported AbolDeepIO1 widths execute for N=10 and N=20: separate
  acceleration/gyro 100→150→100 streams, width-100 time stream and
  150→100→50 feature fusion, dropout 0.25 and two scalar sequences.
- Synthetic nonuniform int64 nanosecond clocks retain their exact adjacent
  differences before conversion to the network dtype, including small differences
  on a large epoch clock. The declared N-target/N+1-reading convention has actual
  one-sample lookahead; missing previous clock context or future readings fail.
- Both int64 overflow directions are rejected: `[max,min]` cannot become a valid
  increasing clock just because subtraction wraps to +1, and `[min,max]` cannot
  accept a true positive interval that wraps to −1. Original timestamp order is
  checked before subtraction, then difference finiteness and positivity.
- Training-only Gaussian noise uses **sqrt(variance)** for variances 0.2 and 0.02.
  Evaluation draws no noise/dropout RNG. Training and evaluation state reloads,
  replayed RNG, scalar predictions and the next draw are exact.
- Prefix Euclidean path length differs from signed displacement or endpoint
  norm. Quaternion **R4 chord sums** lose axis information and are not SO(3)
  geodesic angles. Raw q→−q sign flips and the explicitly separate hemisphere-
  continuity helper produce different, checked labels.
- The loss is weighted absolute-error **SUM**, with correct many-to-many and
  final-supervised-row many-to-one semantics. Full-width losses and gradients
  are finite. Finite, possibly zero raw-nanosecond time-gate gradients do not
  establish time-branch conditioning or accuracy on real data.
- Inference accepts only body acceleration, gyro and dt; reference pose is not
  an inference argument. No ground-truth direction is used to assemble a trajectory.

## Scientific boundary and remaining work

The complete recovered description is the first author's official thesis §6.3,
not an acquired journal version of record. Equality with journal parameters and
reported accuracy remains unverified. N versus N+1 indexing, Keras gate/dropout/
initialization details, linear output heads, lambda defaults 1/1, batch SUM,
quaternion-sign convention and training recipe retain the explicit assumptions
in the source card. CPU tests do not resolve missing paper details.

Tensor dtype/order checks cannot distinguish a real uniform acquisition clock
from artificial nominal-rate timestamps. Original-clock provenance must be
admitted separately by the caller; this synthetic batch did not do that.
Uniform native 200 Hz timestamps cannot recover discarded original jitter.
No hidden time normalization was introduced to repair raw-nanosecond saturation.

The two scalar outputs do not identify signed XYZ displacement or full attitude.
The next scalar lifecycle needs original-clock preservation, validated scalar
labels/metrics and a reviewed recipe. A future XYZ adaptation needs an independently
learned vector/direction head and explicit target/loss/integration contract.
Neither has been accepted here. This document records engineering progress only;
it does not authorize registration or extend the running benchmark.
