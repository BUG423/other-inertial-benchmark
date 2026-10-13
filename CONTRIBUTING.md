# Contributing

Contributions to other-inertial-benchmark should support common data contracts, faithful algorithm interfaces, comparable evaluation and auditable experiments.

## Scope and conventions

- Use dev/v1, NIO/ipb and native IPB v1 model/converter interfaces for all new integrations.
- Keep code in src/inertial_benchmark, tests in tests, specifications/cards in docs and proposals/studies in research.
- Android source, builds, APKs, capture and device testing belong to [BUG423/app-inertial-lab](https://github.com/BUG423/app-inertial-lab). This repository tests exchanged data.
- Use README.md for Chinese/default and README_EN.md for English. Keep capabilities, status and commands aligned; use tables/Mermaid instead of directory trees.
- Record third-party terms, official sources, independent implementations, assumptions and adaptations separately.
- Save splits, input permissions, budgets, seeds, evaluation definitions and source versions. Implementation or CPU checks do not establish reproduced paper accuracy.
- Do not commit raw datasets, checkpoints, generated artifacts or credentials. Update affected links, cards and commands with changes.

## Add a dataset

All new integrations use the native dev/v1 pipeline. Follow the [converter contract](src/inertial_benchmark/data/converters/base.py), [converter loader](src/inertial_benchmark/data/converters/__init__.py), [common conversion pipeline](src/inertial_benchmark/data/convert.py) and [IPB v1 design](docs/DESIGN.md). An existing native example is [EuRoC](src/inertial_benchmark/data/converters/euroc.py).

| Step | Change and acceptance |
| --- | --- |
| 1. Audit the source | Add a [dataset card](docs/datasets/) with terms, source/version, units, clocks, coordinate frames, reference provenance and allowed input track. Mark unresolved fields explicitly. |
| 2. Parse native samples | Add a module in [converters/](src/inertial_benchmark/data/converters/) providing `NAME`, `VERSION`, `LICENSE`, `official_splits()` and `iter_raw_sequences()`. Yield `RawSequence` on source clocks with required attributes, masks and documented calibration; keep reference and device attitude distinct. The common pipeline performs resampling, validation and writing. |
| 3. Preserve split provenance | Set recording/session `group_id`; use existing official, grouped and extra split hooks where applicable. Return an explicit rejected sequence when a physical contract cannot be established. Audit-only intake does not create trainable data. |
| 4. Configure and accept | Add a [dataset YAML](src/inertial_benchmark/cfg/datasets/) using a path relative to `IPB_DATASETS`. Run `ipb convert dataset=<name> source=<raw>` and `ipb check data=<name> full=true hash=true`; record accepted/rejected counts, warnings, hashes and receipt. Test frame/time/mask boundaries using the existing native contract. |

## Add an algorithm

Use [BaseModel / SequenceModel / InputSpec](src/inertial_benchmark/nn/base.py), the [registry](src/inertial_benchmark/nn/registry.py), existing [model packages](src/inertial_benchmark/models/) and [model YAMLs](src/inertial_benchmark/cfg/models/). The [model interface](docs/DESIGN.md) and [algorithm catalogue](docs/ALGORITHMS.md) define provenance and protocol requirements.

| Step | Change and acceptance |
| --- | --- |
| 1. Establish the method contract | Add or update the [algorithm card](docs/algorithms/) and catalogue with paper, official commit, terms, source-based or paper-only status, assumptions and justified adaptations. Record input permissions, time/layout semantics and losses. |
| 2. Implement the native interface | Derive from `BaseModel`, construct with `InputSpec`, return the documented `vel` layout and declare any extra inputs/outputs. Keep training/validation loss batch semantics consistent. Stateful methods use `SequenceModel.predict_sequence(seq, view, starts)` and train-only `calibrate(views, split="train")`. |
| 3. Register and configure | Use `@register_model("<name>")` and import the package through [models/__init__.py](src/inertial_benchmark/models/__init__.py), or load an external plugin through `IPB_PLUGINS`. Add a model YAML containing `arch`, `args`, `input`, provenance and applicable official/unified recipes. |
| 4. Validate before comparing | Check analytical behavior and source/fixture fidelity where evidence is available. Verify NIO train/val/predict and checkpoint/results on the target server. Freeze data, permissions, recipes and evaluation before sufficient training; structural checks and CPU smoke runs are separate from paper-accuracy reproduction. |

## Validation

```bash
python -m pip install -e ".[train,test]" ruff
ruff check src tests
pytest -q
```

Python CI retains v1 training dependencies and its Python matrix. Android checks run in the independent APP repository. Changed data contracts need physical/analytical boundary checks; benchmark claims also need sufficient training.

## License and evidence

Repository-owned material follows [CC BY-NC-ND 4.0](LICENSE); third-party components and datasets retain their own terms. Proposals remain labeled until code, fixed configuration, splits and traceable results are available.

Read [layout](docs/REPOSITORY_LAYOUT.md), [current status](docs/DEV_V1_INTEGRATION.md) and the relevant data/algorithm cards.
