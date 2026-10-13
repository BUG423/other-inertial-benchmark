# Pending algorithm ports

These independent paper reconstructions are outside the active registry and model configuration directory. Neither method is counted as accepted or reproducible benchmark output. No local framework runtime was used while preparing them.

| Method | Primary venue | Code provenance | Current boundary |
| --- | --- | --- | --- |
| [M2EIT](m2eit/source-card.md) | ICCV 2025 main | Paper-only; author network code not verified | Three experts/router drafted; Haar/residual and embedding ambiguities need review |
| [DiffusionIMU](diffusionimu/source-card.md) | IJCAI 2025 main | Paper-only; author network code not verified | Method-specific equations drafted; encoder/inference assumptions; positive head is not calibrated uncertainty |

Each directory contains a BaseModel/InputSpec draft, two dimension configs, explicit scientific tests and provenance. Imports have no model/loss-registration decorators. The configs deliberately refer to unregistered architecture names and are not runnable public profiles yet. The normal Predictor and velocity integration contract will be reused; there is no parallel training framework.

After root coordinates a finite CPU slot on the new AutoDL instance, explicit staging tests can be run there with `PYTHONPATH` pointing at the agreed immutable source. Reduced-width scientific tests do not substitute for the default-profile lifecycle, original-window/data-view checks and subsequent capacity qualification.

```bash
python -m pytest -q research/staging/algorithm_ports/m2eit/test_scientific_contract.py research/staging/algorithm_ports/diffusionimu/test_scientific_contract.py
```

Future integration order: resolve documented scientific ambiguities; execute targeted server tests; run each original/default-width configuration through real train/validation/predict/reload; review paper fidelity and orientation track; explicitly add model/loss registration in a separately approved source snapshot; extend CPU and capacity qualification; finally append method identity and both dimension profiles to the dynamic matrix. Keep all failed or pending stages visible. No GPU work is armed by these files.
