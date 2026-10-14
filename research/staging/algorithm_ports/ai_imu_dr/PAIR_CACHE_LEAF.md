# Released pair-cache / crop-offset leaf

This unregistered supervised indexing leaf complements the already accepted
AI-IMU-DR core. It does not alter those eleven files, the full-sequence output
transport, training loops, loss, model configuration or any formal row. The
new files have only local standard-library AST review; actual server collection
and CPU execution remain pending.

The official source is Martin Brossard, Axel Barrau and Silvère Bonnabel's
[AI-IMU-DR repository, pinned revision](https://github.com/mbrossar/ai-imu-dr/tree/32967812139af82cf12654c93438bf9065b7a241).
The relevant released source is
[src/train_torch_filter.py](https://github.com/mbrossar/ai-imu-dr/blob/32967812139af82cf12654c93438bf9065b7a241/src/train_torch_filter.py):
`compute_delta_p`, `prepare_loss_data`, `prepare_data_filter`,
`get_start_and_end` and `precompute_lost`. Its criterion is normalized reference
chord MSE with **SUM** reduction. No author matches the excluded Shanshan Zhang.
Journal accuracy or complete original training reproduction is not claimed.

[implementation](pair_cache.py) caches the requested100,200,…,800 m pairs on
one whole admitted recording or explicitly declared continuous segment. It
retains raw decimation10, recording-anchored starts every100 raw samples,
reference-path endpoint search and R-start-transposed local chord labels.
`crop_cached_pairs` selects cached global pairs then subtracts N0=10*k. It
never rebuilds the one-second start grid inside a crop. For N0=10, a global
start100 becomes crop start90; rebuilding a crop would incorrectly introduce
start0. Cache identity retains an explicit original raw-index origin.

The released crop endpoint condition uses `floor(N/10)`. A partial-decimation
crop can therefore exclude an endpoint that is physically inside its raw
sample extent. This leaf preserves that rule, including explicit boundary
counterexamples; it does not repair it silently. Released unconditional
`squeeze()` loses the K=1 label dimension and can fail in `precompute_lost`.
This port keeps K×3 labels, a disclosed shape repair. The source float32
rotation/displacement cache differs from the accepted modern float64 label
contract; this leaf retains the latter adaptation. Exact numeric source parity
uses representable binary fixtures, while independent NumPy float64 geometry
covers nonidentity rotations. It does not claim all-reference bit equality.

An optional boolean quality mask excludes existing pairs if **any original
sample** in their closed span is invalid, including non-decimated samples.
It does not reset start grids, manufacture new pairs or interpolate labels.
This is an explicit quality-admission adaptation; released code has no mask.
Finite reference values are still required throughout the source. A future
raw reader decides and certifies physical quality/segment policy separately.
A cache is not a raw-data, reference-anchor, permission or clock certificate.
All-invalid/short/zero-chord/no-crop-pair cases reject this objective.

The labels-only `as_loss_pairs` adapter feeds the unchanged accepted
`relative_pose_sum`; it does not replace the loss with velocity MSE, average
reduction or requested-path-length normalization. References are never fed to
a filter by this module. Readonly owned arrays, integer crop/raw-index guards
and no-RNG-consumption behavior support later sampler integration. There is
no random crop sampler, optimizer, data loader, filesystem reader or model
registration here. Original random-start selection/recipe and real100Hz
vehicle admission remain separate work.

[scientific tests](test_pair_cache.py) propose actual pinned-source function
oracles (AST extraction of the two source functions, no full source imports),
independent NumPy enumeration, nonsecond offset counterexamples, source end
bounds, quality masks, zero supervision, shape/precision boundaries, readonly
ownership and B2 unchanged SUM-loss XYZ gradients. All reference fixtures are
synthetic. The oracle tests require an operator-provided hash-verified official
source-only path; missing source is an error, never an implicit skip. None of
these tests qualify KITTI, a full6000 training pipeline, NIO, CUDA or a new
active algorithm entry.
