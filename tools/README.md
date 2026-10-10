# Tools

Supporting tools remain outside the Python core because they have independent runtimes and release lifecycles.

| Component | Repository / Location | Runtime | Responsibility |
| --- | --- | --- | --- |
| [Inertial Positioning Lab](https://github.com/BUG423/inertial-lab) | [BUG423/inertial-lab](https://github.com/BUG423/inertial-lab) (Independent) | Android/Kotlin + Python | High-frequency sensor capture, ARCore VIO reference, `.iplab` archives, and on-device inference |
| [PaperFlow IO](paperflow-io/) | In-tree (`tools/paperflow-io/`) | Codex skills + Python | IO-specific ideation, feasibility review, and experiment planning |

Each tool owns its dependencies, tests, and usage documentation. Shared canonical data behavior belongs in `src/inertial_benchmark/`.
