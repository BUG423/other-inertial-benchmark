# MotionTransformer (AAAI 2019): unregistered reconstruction

**Status:** complete domain-objective draft; server tests not run. Ordinary NIO
training needs a domain-batch and optimizer-stage extension. This method is not
registered, CPU accepted, or included in the current 2067-row campaign.

Primary sources: [AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/4802),
[complete publisher paper](https://ojs.aaai.org/index.php/AAAI/article/view/4802/4680),
[Oxford author copy](https://www.cs.ox.ac.uk/files/10408/motiontransformer-transferring-neural%20%283%29.pdf).
Full authors are Changhao Chen, Yishu Miao, Chris Xiaoxuan Lu, Linhai Xie, Phil
Blunsom, Andrew Markham and Niki Trigoni; no Shanshan Zhang match.
The bounded author/project search verified no official method network source.
This is an independent paper reconstruction, not a port of author code.

## Primary scientific contract

The method translates unpaired source/target sequences using recurrent networks,
not self-attention. The paper input is 200 six-axis samples at 100 Hz; original
labels are `(Δlength, Δheading)`. A shared domain-conditioned encoder feeds two
domain generators, a noisy shared reconstruction decoder and shared predictor.
Two domain discriminators supply the adversarial objective.

| Paper equations | Draft operation |
| --- | --- |
| 2–6 | `encode`, `translate`, `reconstruct`, `predict` |
| 8–9 | Both discriminator losses and minimax generator objective |
| 10 | Noisy source and target reconstruction |
| 11 | Source-label regression on original and translated source |
| 12 | Both translation cycles, L1 |
| 13 | Both latent perceptual constraints, squared L2 |
| 7 | GAN + `.01 AE + 100 prediction + .1 cycle + 1 perceptual` |
| 14 | `integrate_polar`: predicted heading increments accumulate |

Target-domain labels never enter the objectives. Only train-split target IMU may
enter adaptation; val/test reference labels are evaluation-only.

## Assumptions that prevent an exact-reproduction claim

The eight-page paper does not provide a complete parameterized network or
optimizer curriculum. The draft assumes unidirectional, one-layer LSTMs, width
128, linear readouts, last-state window readout, fixed two-way one-hot domain
vectors and no undocumented input normalization. These are configurable or
explicit design choices, not recovered author parameters.

Squared/L1 norms sum over non-batch coordinates and average over samples.
Reduction conventions are unreported and must be reviewed with the loss weights.
The printed saturating minimax GAN objective is retained. A non-saturating GAN
variant is not silently substituted. Each method update uses one discriminator
step followed by one joint encoder/generator/decoder/predictor step; exact step
ratios, optimizer type, learning rate, initialization and any warm-up remain
recipe assumptions. No paper accuracy is claimed.

## Separate IPB adaptations

`MotionTransformerCore` accepts body-frame `[acc_xyz, gyro_xyz]` and outputs
the original planar polar labels. `integrate_polar` requires an initial position
and heading, then uses only predicted increments. It has no native world-velocity
Predictor bridge. Exact body/polar targets and sequence-level integration require
a future task contract; `InputSpec(frame=body,dims=2)` is currently invalid.

The YAML drafts instantiate `MotionTransformerAdaptation`: native 200 Hz/400
samples are explicitly pair-averaged to 100 Hz/200 steps and permuted from IPB
`[gyro,acc]` to `[acc,gyro]`. Gravity-world input and average world velocity labels
replace the paper body input and displacement/heading-increment labels. The
horizontal head predicts speed/absolute direction; the XYZ profile adds a
separately learned vertical velocity. This is a changed scientific task, named
as an adaptation. XYZ evaluation derives XY from the same XYZ output slice;
the dedicated 2D profile is a separate reference run. No zero-filled Z, fictitious
orientation prediction or original 3D paper claim is permitted.

The adapter refuses a single-domain loss call. Its full joint objective requires
`target_domain_imu`, and discriminator updates remain necessary. The current
Trainer supplies one labelled batch and one optimizer, so these YAML files are
not yet runnable public profiles. This refusal is an integration blocker, not
an unsupported case that can be counted as a successful training run.

## Minimal future integration contract

1. A reviewed protocol chooses one labelled source train split and one different,
   unlabelled target **train** split. Persist both fingerprints and domain IDs;
   allow unequal batch sizes, without temporal pairing or target-label reads.
2. A method-stage hook provides both train batches and disjoint optimizer states.
   Reuse existing Trainer logging, checkpoint, stopping and validation machinery;
   do not create a second benchmark engine. Save/restore both optimizers, sampling
   positions and RNG state. Common-recipe fairness must account for two updates
   and target-domain access, not equate this with ordinary supervised training.
3. Validation/prediction only uses the selected inference-domain encoder/predictor.
   A future supervised validation-loss hook must be explicitly marked as a
   diagnostic, rather than silently replacing five training objectives.
4. After domain policy and assumptions are reviewed, run scientific tests and
   true default-width, original-span multi-optimizer train/val/predict/reload on
   AutoDL. Only a new independently qualified snapshot may register the method.

## Proposed acceptance, not execution evidence

Run the explicit file on AutoDL under an approved separate two-thread CPU lease.
Actual collection establishes the test count; no local framework run occurred.

```bash
PYTHONPATH="$STAGING_SNAPSHOT/src:$STAGING_SNAPSHOT" \
python -m pytest -q research/staging/algorithm_ports/motiontransformer/test_scientific_contract.py
```

The file checks printed objectives, discriminator detach and optimizer ownership,
all generator branches' gradients, unequal unpaired batches, true updates,
latent-noise/RNG replay, original polar path integration, original 200-step span,
unit/channel adaptation, learned Z and explicit single-domain refusal. Small
width fixtures do not establish the default-profile lifecycle or GPU capacity.
