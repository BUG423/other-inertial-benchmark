# AbolDeepIO original-scalar staging

Status: independent thesis-based reconstruction, **static only, unregistered and
not CPU/GPU accepted**. It adds zero active methods, profiles or formal rows.
Many-to-many and many-to-one are supervision variants of the same paper, not two
independent algorithms. This folder supplies a scalar scientific core, not the
standard NIO train/val/predict lifecycle or an XYZ trajectory implementation.

The publication is *AbolDeepIO: A Novel Deep Inertial Odometry Network for
Autonomous Vehicles*, by **Mahdi Abolfazli Esfahani, Han Wang, Keyu Wu and Shenghai
Yuan**. It appeared online in 2019 and in IEEE Transactions on Intelligent
Transportation Systems 21(5), 1941–1950 (2020). No author matches the excluded
Shanshan Zhang. Primary metadata: [journal DOI](https://doi.org/10.1109/TITS.2019.2909064)
and [official NTU publication record](https://hdl.handle.net/10356/155329).

The recovered complete method description comes from the first author's
[official doctoral thesis](https://hdl.handle.net/10356/152280), Chapter 6 §6.3,
particularly §6.3.1–6.3.2, equations 6.6–6.10 and Table 6.2. The official
[repository text bitstream](https://dr.ntu.edu.sg/server/api/core/bitstreams/7684b0ca-52df-480c-9e9f-a16be870baf6/content)
was acquired on AutoDL and verified against its repository MD5. Its named method
and bibliography identify the journal article, but **journal VoR/accepted
manuscript equivalence has not been checked**. OriNet in §6.4 is another method
and supplies none of this draft's missing settings. No official author network
code or software redistribution license has been verified; this code is an
independent reconstruction, not copied from an unlicensed implementation.

## Recovered scientific core

Body acceleration and gyroscope enter separate 100→150→100 LSTM streams. Actual
elapsed nanoseconds enter a separate width-100 LSTM. Their learned features fuse
through 150→100→50 LSTMs. Dropout is 0.25. Training-only independent zero-mean
Gaussian noise has acceleration **variance 0.2** and gyroscope **variance 0.02**;
the implementation draws with standard deviations `sqrt(0.2)` and `sqrt(0.02)`.
Time receives no added noise. The core accepts only these three streams at
inference. [Implementation](model.py) and [scientific tests](test_scientific_contract.py)
are independent of the active engine/registry.

The first target is the within-window prefix sum of Euclidean position-step
lengths. The second is the prefix sum of Euclidean **R4 quaternion-coordinate
chords**. It is dimensionless, not a rotation angle in radians. The weighted
objective sums absolute errors over supervised time. It does not square the
errors or average them. Reference positions/quaternions appear only in the
explicit training-label helper.

## Frozen reconstruction assumptions

- **Indexing and latency:** let `N=10` or `20` denote N supervised increments.
  Target readings are `s .. s+N-1`. Inputs contain N+1 actual IMU readings
  `s .. s+N`; a separate previous timestamp at `s-1` supplies the true backward
  dt for the first reading. Fused feature row `i+1` predicts target row `i`, so
  the last target uses one actual future IMU reading. Labels use the preceding
  reference pose plus N target poses, with no future reference pose. This is an
  explicit resolution of the recovered text's inconsistent N, N+1 and inclusive
  `[0,N]` descriptions. It is not verified to match the printed figure or VoR.
- **Cell/dropout:** independent unidirectional PyTorch LSTMs, with element-wise
  output dropout after every LSTM layer. Keras gate/input/recurrent dropout masks
  and initialization are unresolved. PyTorch default cell gates and parameter
  initialization are not claimed identical to historical Keras defaults.
- **Heads:** one linear 50→2 readout at every observed timestep; no undocumented
  Softplus, absolute-value transform or cumulative prediction layer. Untrained
  outputs can be negative. The recovered text does not specify the output head.
- **Loss:** lambda values default to 1/1 as unreported tuning parameters. The
  staging loss uses SUM over batch as well as supervised time; batch reduction is
  another unreported choice. Many-to-one selects the last supervised output;
  many-to-many supervises all N outputs. Neither is a new method.
- **Quaternion signs:** `hemisphere="raw"` preserves literal coordinate
  differences, including a stationary q→−q chord of 2. The optional
  `continuous` helper is a named hemisphere-continuity adaptation. It is never
  applied automatically, and neither branch silently normalizes quaternions.
- **Noise/RNG:** acceleration noise is drawn before gyro noise, then dropout is
  applied acc→gyro→time→fusion. This ordering is a repeatability choice, not a
  recovered TensorFlow RNG sequence. Evaluation uses no noise or dropout draws.
- **Clock numerics:** int64 nanosecond clocks are differenced before conversion
  to network floating dtype; float32 clock inputs, nonmonotonic original
  timestamps and overflowing differences are refused. Original clock provenance
  and the prohibition on nominal-rate fill require separate **caller admission**:
  a tensor cannot distinguish a genuine uniform acquisition clock from fabricated
  nominal timestamps. No seconds conversion or clock standardization is hidden. Raw ~5e6 ns
  values can saturate time-LSTM gates; finite zero time gradients are possible.
  A normalized-time model would be a separately declared adaptation. Uniform
  native IPB timestamps have already lost original sensor jitter: original raw
  clocks or an explicitly preserved clock field are needed for this protocol.
- **Training recipe:** optimizer, learning rate, batch, epochs, initialization
  and validation selection are not recovered. No settings are borrowed from
  OriNet, TinyOdom or a third-party nine-axis reconstruction. They stay unresolved
  for any future training study.

## Scalar information boundary and next gate

Our counterexamples show that opposite signed position steps have the same
length, a closed path can have positive length and zero endpoint displacement,
and equal-angle rotations about different axes have the same quaternion chord.
These are mathematical deductions from the target definitions, not author
accuracy claims. Two scalar outputs do not identify an XYZ trajectory or complete
attitude evolution. No GT direction/orientation is used to manufacture one.

The tests cover full widths, true unequal-clock mapping and one-sample lookahead,
noise variance/RNG behavior, prefix/chord/sign semantics, weighted-L1 sums,
finite gradients, and state/RNG replay. They have **not been run or collected**;
all actual framework execution must be separately authorized on AutoDL. Static
test counts do not constitute CPU acceptance.

After actual server scientific qualification, a scalar-only lifecycle still
needs raw-clock preservation, original-task data labels/metrics, recipe decisions
and journal comparison. XYZ needs a separately learned signed-vector or direction
head, a new target/loss/integration contract and real acceptance; it cannot be
obtained by zero padding, GT heading, or renaming these scalars as velocity.
