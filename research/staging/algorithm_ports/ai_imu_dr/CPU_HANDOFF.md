# AI-IMU-DR scientific CPU handoff

AutoDL independently executed the frozen official-source-derived staging:
**51 actual collected nodes, 51 passed, zero failures/errors/skips**, return
code zero. The scientific parent took **26.360 s**, including actual collection
6.022 s and the focused test process 14.804 s. No local framework test was run.
This is synthetic scientific/engineering acceptance and selected official
source parity, **not** real KITTI accuracy, raw-clock admission, complete NIO
training, CUDA qualification or an active benchmark method/profile increment.
The running formal campaign and its persisted random seeds are unchanged.

The ten original source/test/license/card/provenance files remain byte-identical
to their reviewed static manifest. This handoff is the only new file after the
CPU run. [source-card.md](source-card.md) records the historical pre-execution
contract and fidelity differences; this handoff adds actual evidence without
rewriting the tested source identity. Local AST estimates were not substituted
for pytest collection or JUnit results.

## Actual scope

[test_scientific_contract.py](test_scientific_contract.py) actually exercised:

- The original width-32/dilation-1/3 network, **6210 CNN plus 12 P0/Q parameters**;
  actual pinned-source B1 initialization/RNG/eval output parity, full B2 layout,
  RF17 with eight future samples, declared train dropout and deterministic eval.
- The complete 21D SE₂(3) group filter, both biases and estimated Rci/tci, full
  covariance/cross-covariance, nonzero lever-arm observation, independent NumPy
  recursion, all 21 measurement-Jacobian finite-difference columns, Joseph
  update and PSD guards without invented jitter or covariance repair.
- Released nominal half-acceleration position integration, **Taylor3 Phi,
  Gdt and Phi(P+Gdt Q Gdtᵀ)Phiᵀ**, and the explicitly retained omission of
  post-injection error-coordinate reset. Approximation differences from exact
  held-input propagation were tested as finite-difference convergence diagnostics,
  not falsely required to equal the exact discrete Jacobian.
- The released NumPy negative gyro-bias measurement-Jacobian sign for training
  and inference, with the opposing released Torch sign retained as a tested
  counterexample. This is a disclosed engineering choice, not an author-typo
  claim or bit-exact full-release reproduction.
- Integer-nanosecond difference/jitter and overflow checks; physically nonzero
  uphill world Z; no later reference labels accepted by inference; rejection
  of short recordings lacking original 100–800 m loss pairs.
- Original path-selected pairs, reference local **chord** denominator and MSE
  **SUM**. Complete CNN→P0/Q/N→21D filter→loss backprop produced finite, nonzero
  gradients including all twelve P0/Q scale weights, two real Adam updates and
  changed parameters. A separate original recording loss/clip gate also made
  two real Adam updates under the source recipe.
- Source augmentation standard deviations, training/eval behavior, persisted
  numeric model/normalization/physics state, same-RNG filter/checkpoint output
  and next-draw replay. This is scientific state replay, not yet a complete
  recording-sampler/optimizer-resumable unified NIO training checkpoint.

The high-speed references, nonuniform nanosecond clocks and other tensors in
these tests are **synthetic**. Their long-distance loss pairs establish formulas
and differentiability, not actual driving dynamics, acquisition provenance,
reference accuracy or original publication scores. No raw/native H5 file or
pretrained weights were opened in this qualification.

## Source and receipt identity

Official reference source was already actually acquired on AutoDL at commit
`32967812139af82cf12654c93438bf9065b7a241` under MIT. The oracle tests verify
the SHA256 of the selected original source files before executing only their
class definitions. No upstream top-level demo, data loader or package
installation is executed. Missing or altered oracle source is a failure,
never a silent skip.

The isolated qualification source is composition
`7288f1525a183f26b740987d8d97bcab231ca04e3e201ac1d3637b4b54802086`,
**469 managed files**: already qualified 459 plus exactly these ten new staging
files. Its actual manifest SHA256 is
`d35d2e498ea00e9a4d1bbabe121e0e4ee55619e4e6c28aaf5f163e08397e5acc`.
The original ten-file static ready SHA256 is
`af69bab9fee7177934a9303c4b06975c43c926173fa8cadec9983c33b3df5d0f`.
The qualification did not modify or install over the running formal source.

Ignored operator evidence directory:
`.local/reports/ai-imu-dr-stage2-science-7288f1525a183f26/`.
All **20 original small receipt/log/XML/guard mirrors** match their source
hashes; actual JUnit names equal actual collected pytest node IDs.

| Artifact | SHA256 |
|---|---|
| `receipt.json` | `850d2fefa60641c1669e480583cb67c6cb1a5856ac18b7fd903d0c3843d081cd` |
| `focused_1.xml` | `fc930b99584ca0bb714a89d52e61dca6df3060cf8700b4e45e40b1eeaf41a735` |
| `mirror-receipt.json` | `914474b2e11126cd6294c744a538d28231c18f56daaee8f3380846d440e47273` |
| `independent-final-review.json` | `e2d84cff6726cf0eacc91bce527d520f6d63c608d8fede1188acbb5c0bf8ef59` |
| Root `root-ai-imu-dr-stage2-actual-review.json` | `7209f7604a018b5c09c08c781ac07beebb1f7289e4dfad21fc530c580e2a3b13` |

## Resources and unchanged inputs

The once-only real-file coordinator PID70277 launched timeout PID70279; both
naturally terminated with return code zero. The supplemental CPU exclusive
lease was explicitly released. Threads were actually **Torch2 / interop1**,
with CUDA hidden and **never initialized**. No formal CPU/GPU lease or process
was signalled. The active deadline was 180 s, with eight seconds cleanup grace;
there was no retry or overwritten failed-run evidence.

The shared instance's highest observed one-second CPU window was
7.372/16 effective cores, with zero additional throttling. Shared cgroup peak
was 34,606,043,136 B; this includes other running work and is not this model's
exclusive memory measurement. Preflight/live CPU and RAM limits, disk reserve
and byte-pinned policy/seed/capacity guards were enforced.

Protected source snapshots, all 13 native metadata sets, previous Motion/data
staging, the pinned official source oracle and frozen policy/seed/capacity
files matched before and after. The original ten new scientific files also
remained unchanged. No dataset, model registration, configuration, formal row
or seed count was added by this acceptance.

## Remaining integration gates

1. Acquire actual compatible wheeled KITTI raw **100 Hz** OXTS, original clock,
   quality flags and calibrations on AutoDL. A synchronized 10 Hz subset cannot
   be repeated/interpolated into a source-faithful 100 Hz input. Original clock
   provenance remains a caller/data qualification, not a tensor-type proof.
2. Bind train-only normalization and distinct recording splits; demonstrate
   real eligible 100–800 m pairs and disclosed reference-assisted initial R/v.
3. Integrate the differentiable full-sequence protocol into existing NIO/Trainer,
   keeping filter-corrected world p/R/P instead of silently reintegrating only
   velocity. Add sampler/optimizer/RNG exact resume and native export state.
4. Measure full 6000-sample graph capacity and actual CUDA lifecycle before
   formal training eligibility; preserve official-source versus common-recipe
   tracks. The two car-frame N outputs are not Cartesian prediction sigma.
5. Resolve further source/paper distinctions honestly. Journal Version of
   Record and original publication accuracy remain unqualified.

The method remains **staging, unregistered and outside the formal matrix**.
