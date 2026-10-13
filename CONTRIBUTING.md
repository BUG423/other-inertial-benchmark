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
