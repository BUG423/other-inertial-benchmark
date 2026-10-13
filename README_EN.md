# 🧭 other-inertial-benchmark

[![CI](https://github.com/BUG423/other-inertial-benchmark/actions/workflows/ci.yml/badge.svg?branch=dev%2Fv1)](https://github.com/BUG423/other-inertial-benchmark/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Android Lab](https://img.shields.io/badge/Android%20Lab-app--inertial--lab-3DDC84?logo=android&logoColor=white)](https://github.com/BUG423/app-inertial-lab)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-red.svg)](LICENSE)

[**🇨🇳 中文**](README.md) · **🇬🇧 English**

> **other-inertial-benchmark** provides unified data contracts, algorithm interfaces, comparable evaluation and auditable experiments for inertial positioning research.

---

## 🌟 Introduction

The deliverable is an extensible benchmark: dataset-specific converters produce a common storage contract; algorithms retain required windows, outputs and losses through unified interfaces; results use explicit splits, input permissions and evaluation protocols, with configuration and provenance preserved.

**Use `dev/v1` throughout**: IPB v1 data, the `NIO` API, `ipb` CLI and existing model/converter interfaces define new integrations. Implementation does not establish reproduced paper accuracy. Official sources, independent implementations, paper assumptions and adaptations are recorded in [algorithm cards](docs/ALGORITHMS.md).

## ✨ Capabilities and Components

| Component | Current capability | Entry and scope |
| :--- | :--- | :--- |
| **Data layer** | IPB v1 HDF5, 200 Hz, units/frames, provenance, masks, grouped splits and checks | [Design](docs/DESIGN.md) · [Dataset catalog](docs/DATASETS.md) |
| **Core converters** | Eight existing converters: RoNIN, RIDI, OxIOD, TLIO, IDOL, RNIN, IMUNet and PedLocData; full RIDI/RoNIN/IDOL/RNIN/TLIO/OxIOD data accepted and ready for training | [Cards](docs/datasets/) · [Code](src/inertial_benchmark/data/converters/); splits, rejections and fingerprints in [current status](docs/DEV_V1_INTEGRATION.md) |
| **Algorithms** | 29 registered architectures, 34 model configurations (including three `3d_adapted` profiles) and 21 packages; RoNIN, TLIO, RNIN, IMUNet, EqNIO, AirIO, DIVE and classical baselines | [Specifications/fidelity](docs/ALGORITHMS.md) · [Configurations](src/inertial_benchmark/cfg/models/) |
| **Training/prediction** | `NIO` / `ipb`, shared Trainer/Validator/Predictor; window/frame/multi-step/sequence interfaces | [Commands](docs/CLI.md); retain required layouts and losses; an export API is still pending |
| **Evaluation/reports** | ATE, separate aligned diagnostics, time/distance RTE, endpoint drift, length ratios, velocity/direction, efficiency; CSV/Markdown/LaTeX, statistics and plots | [Definitions](docs/METRICS.md); scaling, oracle and aligned diagnostics have distinct meanings |
| **New data intake** | EuRoC 11 recordings and Blackbird 20 published segments converted to native v1 and passed full/hash checks; Pegasus/ADVIO audited and rejected; a TartanIMU 2025 source card added | [Current status](docs/DEV_V1_INTEGRATION.md) · [Cards](docs/datasets/); Blackbird uses route groups and a test-only audit list |
| **Independent APP** | Android capture and on-device evaluation | [BUG423/app-inertial-lab](https://github.com/BUG423/app-inertial-lab) · [Exchange contract](docs/TOOLS.md); ARCore is a VIO reference |
| **Research/provenance** | Trajectory shrinkage, frame studies, proposals and IO research tools | [Research](research/README.md) · [Tools](tools/README.md); proposals and validated experiments are distinct |

### Data and Experiment Flow

```mermaid
flowchart LR
    Raw["Public datasets / independent APP"] --> Audit["Provenance, units, clocks and frame audit"]
    Audit --> Convert["Dataset-specific converter"]
    Convert --> Data["IPB v1 HDF5, masks, splits and fingerprints"]
    Data --> View["Task views: windows, targets and input permissions"]
    View --> Train["NIO / ipb: train, validate and predict"]
    Train --> Eval["Fixed-protocol metrics, trajectories and efficiency"]
    Eval --> Report["Tables, statistics, plots and experiment records"]
    Report --> Study["Shrinkage and generalization studies"]
```

## 🛠️ Technology and Entry Points

| Technology / location | Purpose |
| :--- | :--- |
| Python ≥ 3.9, NumPy, SciPy, h5py, PyYAML | Storage, resampling, geometry and configuration |
| PyTorch ≥ 2.0, pandas, matplotlib | Training, reporting and scientific plots |
| [src/inertial_benchmark/](src/inertial_benchmark/) | `cfg`, `data`, `models`, `nn`, `engine`, `metrics` and `utils` |
| [tests/](tests/) · [.github/workflows/ci.yml](.github/workflows/ci.yml) | Analytical, model/data contract, engine and documentation checks |
| [docs/](docs/README.md) · [research/](research/README.md) | Specifications, source cards, status and research evidence |

See [repository layout](docs/REPOSITORY_LAYOUT.md) and [Ops v1](tools/ops_v1/README.md) for isolated server deployment and resource profiles.

## 🚀 Quick Start

### 1. Install and inspect registered components

```bash
git clone --branch dev/v1 https://github.com/BUG423/other-inertial-benchmark.git
cd other-inertial-benchmark
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[train,test]"
ipb info
ipb cfg model=ronin_resnet18_3d metric_dims=3 recipe=unified
```

Use `pip install -e .` for data only. IDOL Feather sources require `pip install -e ".[idol]"`. Select the appropriate PyTorch CPU/CUDA build.

### 2. Convert and check

Acquire files following the [RIDI card](docs/datasets/ridi.md), then set `source` to their actual location:

```bash
export IPB_DATASETS="$PWD/datasets"
ipb convert dataset=ridi source=raw/ridi workers=1
ipb check data=ridi full=true hash=true
```

Outputs include `sequences/*.h5`, `splits/*.txt`, `dataset.json` and `conversion_report.json`. Configuration lives in [cfg/datasets/](src/inertial_benchmark/cfg/datasets/). Retain official seen/unseen labels without imposing them on every dataset.

The AutoDL checkout accepted in this round contains full RIDI/RoNIN/IDOL/RNIN/TLIO/OxIOD under `datasets/ipb-v1/full-core/`. From that server checkout root, use:

```bash
export IPB_DATASETS="$PWD/datasets/ipb-v1/full-core"
ipb check data=ridi full=true hash=true
ipb cfg model=ronin_resnet18_3d data=ridi metric_dims=3 recipe=unified
```

### 3. CPU workflow example

These strides and training budgets are for engineering checks:

```bash
ipb train model=ronin_resnet18_3d data=ridi metric_dims=3 device=cpu epochs=1 batch=2 val_batch=2 workers=0 cache=false stride=200 eval_stride=200 project=runs/cpu-smoke name=ridi-ronin
ipb val model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt data=ridi split=val metric_dims=3 device=cpu val_batch=2 workers=0 cache=false eval_stride=200 project=runs/cpu-smoke
ipb predict model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt source="$IPB_DATASETS/ridi/sequences" metric_dims=3 device=cpu val_batch=2 workers=0 eval_stride=200
```

Equivalent API:

```python
from inertial_benchmark import NIO

model = NIO("ronin_resnet18_3d", metric_dims=3)
model.train(data="ridi", device="cpu", epochs=1, batch=2, val_batch=2, workers=0, cache=False)
```

The main benchmark saves **learned XYZ predictions**, uses `metric_dims=3` for primary 3D metrics and derives `xy_` metrics from the same predicted XY coordinates. `ronin_resnet18_3d`, `ronin_lstm_3d` and `imunet_3d` are labeled `3d_adapted`: their output heads extend the original 2D methods, without claiming original-paper accuracy or 2D checkpoint equivalence. TLIO retains native 3D output. Original 2D configurations remain a separate reference protocol (`metric_dims=2`); PDR and planar constant-heading variants reject 3D requests rather than supplying fabricated Z predictions.

Full GPU experiments require frozen splits, documented recipes and sufficient training. Record unified/official recipe differences. A one-epoch CPU check does not establish rankings or paper accuracy.

### 4. Matrix and reports

```bash
ipb benchmark cfg=main dry_run=true
ipb report runs=runs/cpu-smoke out=reports/cpu-smoke
pytest -q -m "not slow"
```

The example [main.yaml](src/inertial_benchmark/cfg/benchmarks/main.yaml) uses `ronin_resnet18_3d` × eight dataset templates × seed 0; templates do not establish that data was downloaded. Select accepted data/models and check resource budgets before execution. Complete commands are in [CLI.md](docs/CLI.md).

APP source, APKs, `.iplab` conversion and builds belong to [app-inertial-lab](https://github.com/BUG423/app-inertial-lab). Legacy readability and v1 training admission are separate; see [exchange limitations](docs/TOOLS.md).

## 🔬 Research and Validation Status

The current AutoDL server has accepted **94 RIDI / 151 RoNIN / 130 IDOL / 300 RNIN / 354 TLIO / 126 OxIOD** sequences: six core datasets with **1,155 native v1 HDF5 files, 99.40744861 hours and 725 training sequences**. Full/hash checks report zero errors and all six have `train_ready=true`. The newly accepted RNIN/TLIO/OxIOD contribute **780 sequences and 53.695518 hours**; rejections, official group overlaps and physical warnings remain recorded. RNIN has mixed references: 75 sequences from VICON or other external positioning systems and 225 from BVIO, with all nine test sequences using VIO. TLIO uses a VIO reference, while OxIOD uses an external Vicon reference; its Tango, syn and standalone test directories are outside this intake. EuRoC has 11 accepted native recordings; Blackbird has 20 published segments for test-only auditing, counted separately from the six core datasets. IMUNet awaits a download through the user's signed-in browser; the slow PedLocData download left only unverified partial files. See [project status](docs/DEV_V1_INTEGRATION.md).

The storage inventory at 20:02 Beijing time on 2026-10-13 found **71.288 GB (66.3925 GiB)** of server archives, extracted raw data, native outputs and historical copies, with about **130.83 GiB** free. Eight nonempty current native datasets contain **1,186 sequences, 99.99289 hours and 730 training sequences**, including EuRoC and test-only Blackbird; this does not mean all eight support training. Current native-format data occupies about **4.15 GB**. The **5.8723 GB** of three local archives is excluded from server totals. These sizes cover acquired versions or subsets, without claiming that every public full dataset has been downloaded. See the storage breakdown in [project status](docs/DEV_V1_INTEGRATION.md).

Targeted server CPU acceptance for the new 3D protocol and capacity admission at `dev/v1@2c1f57f` passed **144 tests / 0 skipped / 0 failed**. Real XYZ task views and XY slicing were also accepted separately for RNIN/TLIO/OxIOD. These engineering checks do not establish full-training accuracy and are not added to the historical test totals below. The actual GPU probe has not run, formal budgets are not frozen and the training queue is not armed.

Earlier short RIDI/RoNIN CPU NIO runs completed the engineering loop (one epoch / two updates each); full training and rankings await the GPU stage. Unique coverage of the 719 frozen test nodes is **665 passed / 52 skipped / 2 resource_unverified**; the two unverified cases are CTIN end-to-end train/validation tests. Separate targeted acceptance of `6d44573` passed **all 81 tests**, including five tests for dimension-consistent time/CDF plots and paging beyond eight models; this count is not added to the original 719. The **705 tests**, scores and AMP/dropout observations in the 2026-09-24 handoff are historical results.

Shrinkage should be tested through target/integration oracles, time/length resolution, speed/direction, train/eval mode differences, precision and generalization. Existing length decomposition and mode guards provide a starting point; conclusions require sufficiently trained algorithms and controlled experiments.

| Research entry | Evidence status |
| :--- | :--- |
| [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/) | Historical frame studies and assets |
| [PostDiffIO](research/methods/postdiffio/) · [ModeMoEIO](research/methods/moe-io/) | Proposals without reproducible results from this benchmark round |
| [PaperFlow IO](tools/paperflow-io/) | IO ideation, literature review and experiment planning |
| [FAST-LIVO setup](docs/setup/fast-livo.md) | Historical ROS Noetic reproduction notes |

## 🛣️ Roadmap

| Stage | Deliverable and acceptance |
| :--- | :--- |
| **Existing foundation** | v1 sequences/views, eight core converters, algorithm interfaces, train/validate/predict, metrics and reports |
| **Current CPU stage** | Six full core datasets ready for training; EuRoC intake and Blackbird test audit accepted; 144 new 3D targeted tests passed, with historical suite scope and limits recorded separately |
| **Evaluation increments** | Dimension-consistent time/CDF plots and model paging accepted; continue permission tracks, NA/coverage, 3D trajectories, six panels and static HTML |
| **GPU stage** | One seed to validate recipes/full training, then expand the matrix; at most three seeds for final key experiments |
| **Research delivery** | Auditable benchmark, shrinkage mechanisms and controlled experiments, reproducible release; ICRA / IEEE RA-L target |

See [project status](docs/DEV_V1_INTEGRATION.md) for outstanding work.

## 📄 License and Intellectual Property

This project uses **[CC BY-NC-ND 4.0](LICENSE)** (Attribution–NonCommercial–NoDerivatives). Follow the repository license for use, credit and sharing. Commercial use and distribution of modified versions require permission.

Datasets, third-party code, independent tools and weights retain their own terms and provenance; the repository license does not replace third-party authorization. Cite original papers and record adaptations. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

---

<div align="center">

If this project helps your research, please ⭐ Star and follow future reproducible benchmark releases.

</div>
