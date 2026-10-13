# X-IONet independent filter core: CPU handoff

**Actual AutoDL scientific qualification: 39 collected, 39 passed, zero
failures/errors/skips.** This accepts the independent engineering geometry,
filter and covariance leaves under their stated conventions. It does not
accept an original X-IONet neural network, standard NIO lifecycle, real-data
closed loop, native export, CUDA capacity or paper accuracy reproduction.
There is no registry entry, profile or benchmark-count increment; the running
2067-row formal matrix and its random seed record were not changed.

The accepted implementation is [geometry.py](geometry.py),
[filter.py](filter.py), [covariance.py](covariance.py) and
[test_scientific_contract.py](test_scientific_contract.py).
[source-card.md](source-card.md) records the explicit assumptions and remaining
full-system gates. [implementation-plan.md](implementation-plan.md) and
[provenance.json](provenance.json) retain the original design review unchanged;
their design-only fields describe that earlier stage. This handoff is a new
document written after the actual CPU run, not a member retroactively inserted
into its tested source snapshot.

## What actually ran

One authorized recovery executed an actual uploaded `launch-once.py` file,
then the unchanged finite CPU runner. Actual pytest collection log, collection
environment, test environment and JUnit names/counts agree on all 39 unique
nodes. Every JUnit testcase passed; no node was skipped or replaced by a
static estimate. The scientific command was:

```bash
python -m pytest research/staging/algorithm_ports/x_ionet/test_scientific_contract.py -q
```

The operator harness used a nonblocking independent supplemental CPU EX lease,
Torch two threads, interop one, OMP/BLAS two and hidden CUDA. Recorded thread
observations confirm that policy, and CUDA was not initialized before or after
collection/tests. A five-second CPU probe showed two-core headroom and zero
new throttling; child observations also recorded zero new throttling. Shared
instance memory remained below its 75% gate, and the 30 GiB disk reserve held.
The original formal CPU/GPU leases and processes were not signalled.

The science parent finished in **16.029 seconds**, including the resource
probe and subprocess startup. The outer file coordinator took **16.310
seconds**, exited 0 and recorded supplemental-lease release. These are
engineering harness timings, not model throughput, latency or accuracy.
No raw/native numeric sequence, dataset, reference trajectory or weights were
transferred to the local machine; only small receipts/logs/XML/guards were
mirrored. All scientific fixtures here are **synthetic**.

## Tested scientific scope

- Float64 SO(3) Exp/Log at zero, small angle and pi, including the explicitly
  canonical ambiguous pi axis; left-reset finite differences and the positive
  `+0.5*hat(theta)` small-angle sign.
- Independent SciPy finite differences for all current-state propagation and
  declared noise-increment columns, including world-rotation gyro-bias
  coupling. Tilted stationary specific force and known linear acceleration
  agree with the declared old-attitude/trapezoid integrator.
- Full `15+6*n` covariance augmentation and clone cross-correlations, singular
  PSD without fake jitter, propagation and principal-submatrix removal;
  measurement derivatives over current/clone coordinates.
- Explicit full-body-anchor measurement frame, consistent `z-h` injection,
  zero-residual state preservation, Joseph covariance and independently
  checked left resets for current state and two historical clones.
- The printed `h-z` expression is a same-convention diagnostic counterexample,
  not an operational original-paper implementation or evidence of an author
  typo. Original frame/sign conventions remain unresolved.
- Off-diagonal full-SPD Gaussian loss checked against a separate NumPy
  solve/logdet oracle, genuine third-axis prediction gradients, lower-factor
  gradients and central-difference gradcheck. Huber weights are retained;
  head parameterization/floor and reduction are declared assumptions.
- Exact JSON state continuation; clock, overflow, covariance, frame, reset and
  noise rejection without state mutation or repair jitter.

Successful synthetic tests do not establish original acquisition-clock
provenance, selector/expert training, initialization/yaw observability,
gravity-yaw-local alignment or real-data trajectory accuracy.

## Immutable proof and protected scope

The actual source is an independent **459-file snapshot**, formed from the
qualified 452-file parent plus the seven unchanged X-IONet design/core files.
Its source composition identifier is
`e74cbb980107950b138b81049f8f5f1373a9165fe19181210cdc1dfc2d4d794d`;
the actual generated source-manifest SHA is listed below. Do not confuse a
composition identifier with that generated manifest's byte hash.

| Evidence | SHA-256 |
| --- | --- |
| Parent 452-file manifest | `5e9bafdcdcf85ed6f2acd34aee07b0fe81023566953a55d54174e26812a44f83` |
| Actual 459-file manifest | `ae1faa5ccc0066f8a45f7f942ade7eb281d3fee84ba28e464f2deddecff6e188` |
| Scientific CPU receipt | `98a93ddae704d87f50de41240e8681965049ce080b1df225248a477b7f9d7532` |
| Actual 39-node JUnit | `27d3afcfc99a1c2809420bce82c633df48a854df892112c96c65d3e86b31560a` |
| Metadata-only mirror receipt | `85e5c86f7cf36da420af0af9c1da01170d1179e3500c4ae99167582d374dd600` |
| Root independent actual review | `4059a1d2845c5b8c5bb737a536fabe4a4eadfe33540e06cf29b7bb417731c8aa` |

Operator evidence is kept outside public Git in
`.local/reports/xionet-core-science-e74cbb980107950b/`; it includes collection,
environment records, JUnit, logs, before/after guards and the full receipt.
Root's independent review is
`.local/reports/root-X_IONet-core-actual-review.json`.

All 20 original mirrored evidence files were hash-checked. All seven tested
X-IONet files and the three frozen runner/request/wrapper payloads retain their
reviewed hashes. The 425/429/443 source snapshots, all three historical 447
attempts and qualified 452 snapshot passed complete managed-file guards before
and after the run. Existing server Motion seven files, accepted data four
leaves, all 13 native metadata inventories and the policy/seed/capacity record
also stayed equal. Native-metadata guarding does not claim to revalidate every
raw/H5 data byte in this experiment. The A2 two source cards were separately
checked locally; they were not misrepresented as pre-existing server assets.

## Preserved first launch failure

The initial inline coordinator failed Python parsing at line 9 because a
newline escaped through nested strings. The frozen scientific runner never
started on that attempt: no collection, framework execution or CPU lease was
acquired. That event is neither a CPU scientific failure nor a pass. Its
original launch metadata, malformed coordinator text and 244-byte error log
were retained; the error-log SHA is
`cdfccdffcb217f52c26c0e08c0f56cc5bb269917c61c8232f92afca85dbcd90f`.

A separately reviewed and authorized recovery used a real file launcher with
new v2 coordinator/worker logs, rather than modifying the frozen runner or
silently overwriting failed evidence. No automatic retry was performed. The
accepted 39-node result belongs to that later actual file-launched execution.

Missing neural topology and original full-loop conventions remain explicit
gates. This handoff supplies a tested engineering leaf for future integration,
not a completed independent baseline entry.
