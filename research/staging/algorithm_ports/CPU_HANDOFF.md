# Paper reconstruction CPU handoff

The original fifteen draft files remain unchanged. Their source cards and parent README capture the
pre-runtime draft state; this separate record updates **engineering acceptance** only. Scientific
assumptions remain as stated in [M2EIT](m2eit/source-card.md) and
[DiffusionIMU](diffusionimu/source-card.md). Both are paper-only reconstructions: official author
network source has not been verified, and CPU execution is not proof of paper fidelity or accuracy.

On 2026-10-13 UTC, the new AutoDL server actually collected and passed **21 scientific test nodes**
(11 M2EIT, 10 DiffusionIMU), with no failures/errors/skips. Separate process-local registration then
ran both default-width, original-window XYZ models through the standard **NIO → Trainer → Validator
→ Predictor → checkpoint reload** pipeline on three distinct accepted RIDI sequence excerpts.
Each model completed two AdamW optimizer updates, changed parameters, and produced finite loss.
Reloaded XYZ positions, velocity predictions and targets were exactly equal when replaying the same
RNG. DiffusionIMU kept fresh inference noise; fresh draws changed predictions. Its positive auxiliary
head is a regularizer, not a Gaussian sigma.

The real fixture is a short engineering crop with small nonzero XYZ targets, not a representative
accuracy experiment. Training was one engineering epoch with four train and two validation/test
windows at window 200/B2. The standard gap used its naturally tiny fixture, all four modes and MC8;
this does not measure full-dataset gap capacity. Neither method initialized CUDA or ran GPU work.

| Evidence | Actual result | SHA256 of recorded receipt |
| --- | --- | --- |
| Scientific contracts | 21 passed, CPU only | `e6b33420dca3149baff6ee99e6b7ebdbd0e4b58f11416833ea50445c830b5d68` |
| M2EIT native lifecycle | Passed, 6.30 s | `1868f551acf07f3eb6ea9607af69be091aa7cf380902a1710a141f84232a7792` |
| DiffusionIMU native lifecycle | Passed, 2.50 s | `f1c23cd90a04cbc9f71326b4e48ed2d0839450a9ee4b0acb99bef670612557ef` |

Private operator receipts are retained under `.local/research/algorithm-primary/expansion-20261014/`
and `.local/reports/staging-native-natural-handoff/`. The lifecycle runner hash is
`20823d44bd27ce59e1ebd52e0672ced091a50f6da460acd155da5ca392fcebc0`.
All 425 formal source files, thirteen native metadata groups, and accepted fixture files remained
unchanged before/after execution. The frozen formal manifest hash was
`1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949`.

Neither draft is registered in the active framework, included in its 53 profiles, or appended to the
current 2,067 formal rows. Active counts remain **31 architectures / 53 profiles / 21 independent
paper methods**, plus two physical classic definitions. A future separately reviewed source snapshot
must resolve/declare paper ambiguities, register model/loss/config contracts, qualify actual device
capacity and extend source-qualified matrix identities before these become formal benchmark methods.
