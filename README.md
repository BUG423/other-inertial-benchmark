# 🧭 other-inertial-benchmark

[![CI](https://github.com/BUG423/other-inertial-benchmark/actions/workflows/ci.yml/badge.svg?branch=dev%2Fv1)](https://github.com/BUG423/other-inertial-benchmark/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Android Lab](https://img.shields.io/badge/Android%20Lab-app--inertial--lab-3DDC84?logo=android&logoColor=white)](https://github.com/BUG423/app-inertial-lab)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-red.svg)](LICENSE)

**🇨🇳 中文** · [🇬🇧 English](README_EN.md)

> **other-inertial-benchmark** 为惯性定位研究提供统一数据契约、算法接口、可比评价和可审计实验，连接原始数据转换、模型训练、轨迹预测与结果比较。

---

## 🌟 项目简介

最终交付是一套可持续扩展的惯性定位 benchmark：原始数据各有格式，转换后遵循统一存储契约；算法保留自己的窗口、输出与损失逻辑，通过统一接口训练和预测；结果在明确的数据划分、输入权限和评价协议下比较，并保存配置、来源和实验记录。

**一切以 `dev/v1` 为主线**：使用 IPB v1 数据格式、`NIO` Python API、`ipb` CLI 和现有 model/converter 接口接入新数据与算法。模型实现不等于已复现论文精度；官方来源、独立重实现、论文假设和协议适配见[算法规格卡](docs/ALGORITHMS.md)。

## ✨ 核心能力与组件

| 组件 | 当前能力 | 入口与边界 |
| :--- | :--- | :--- |
| **统一数据层** | IPB v1 HDF5、200 Hz 时间轴、单位/坐标系、来源、有效掩码、分组划分与检查 | [设计规范](docs/DESIGN.md) · [数据集目录](docs/DATASETS.md) |
| **核心转换器** | 已有 RoNIN、RIDI、OxIOD、TLIO、IDOL、RNIN、IMUNet、PedLocData 八个转换器 | [数据卡](docs/datasets/) · [代码](src/inertial_benchmark/data/converters/)；实收与验收状态分别记录 |
| **算法库** | 29 个注册架构、31 份模型配置、21 个算法包；含 RoNIN、TLIO、RNIN、IMUNet、EqNIO、AirIO、DIVE 和经典基线 | [规格与忠实性](docs/ALGORITHMS.md) · [模型配置](src/inertial_benchmark/cfg/models/) |
| **训练与推理** | `NIO` / `ipb`，统一 Trainer、Validator、Predictor；窗口、逐帧、多步和序列级接口 | [命令与配置](docs/CLI.md)；统一流程保留算法必要的布局和损失；导出 API 尚待建设 |
| **评价与报告** | ATE、独立对齐诊断、时间/距离 RTE、终点漂移、长度比、速度/方向、效率；CSV/Markdown/LaTeX、统计与图表 | [指标定义](docs/METRICS.md)；短序列换算、oracle、对齐诊断按各自定义解释 |
| **新增数据接入** | EuRoC 11、Blackbird 20 个发布片段已转换为原生 v1，full/hash 检查通过；Pegasus/ADVIO 审计拒收；TartanIMU 2025 来源卡 | [本轮状态](docs/DEV_V1_INTEGRATION.md) · [数据卡](docs/datasets/)；Blackbird 按 route 分组、仅 test 审计，不作训练/验证集 |
| **独立移动端** | Android 采集与端侧评测工程 | [BUG423/app-inertial-lab](https://github.com/BUG423/app-inertial-lab) · [交换契约](docs/TOOLS.md)；ARCore 是 VIO 参考源 |
| **研究与追溯** | 轨迹变短、坐标系研究、方法提案、IO 专用研究工具 | [研究索引](research/README.md) · [工具索引](tools/README.md)；提案与正式实验分开标注 |

### 数据与实验流程

```mermaid
flowchart LR
    Raw["公开数据集 / 独立采集 APP"] --> Audit["来源、单位、时钟与坐标系审计"]
    Audit --> Convert["数据集专用转换器"]
    Convert --> Data["IPB v1 HDF5、掩码、划分与指纹"]
    Data --> View["任务视图：窗口、目标与输入权限"]
    View --> Train["NIO / ipb：训练、验证与预测"]
    Train --> Eval["固定协议指标、轨迹与效率"]
    Eval --> Report["表格、统计、可视化与实验记录"]
    Report --> Study["路径变短机制与泛化研究"]
```

## 🛠️ 技术栈与入口

| 技术 / 位置 | 用途 |
| :--- | :--- |
| Python ≥ 3.9、NumPy、SciPy、h5py、PyYAML | 存储、重采样、几何计算与配置 |
| PyTorch ≥ 2.0、pandas、matplotlib | 训练、结果汇总与科学绘图 |
| [src/inertial_benchmark/](src/inertial_benchmark/) | `cfg`、`data`、`models`、`nn`、`engine`、`metrics` 与 `utils` |
| [tests/](tests/) · [.github/workflows/ci.yml](.github/workflows/ci.yml) | 解析解、模型/数据契约、引擎与文档检查 |
| [docs/](docs/README.md) · [research/](research/README.md) | 规范、来源卡、状态与研究证据 |

完整组件职责见[仓库布局](docs/REPOSITORY_LAYOUT.md)。服务器隔离部署与资源 profile 见 [Ops v1](tools/ops_v1/README.md)。

## 🚀 快速上手

### 1. 安装并查看实际注册项

```bash
git clone --branch dev/v1 https://github.com/BUG423/other-inertial-benchmark.git
cd other-inertial-benchmark
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[train,test]"
ipb info
ipb cfg model=ronin_resnet18 recipe=unified
```

仅使用数据层时安装 `pip install -e .`；读取 IDOL Feather 原始文件需 `pip install -e ".[idol]"`。PyTorch CPU/CUDA 构建按环境选择。

### 2. 转换与检查

先按[RIDI 数据卡](docs/datasets/ridi.md)取得原始文件，并将 `source` 改为实际位置：

```bash
export IPB_DATASETS="$PWD/datasets"
ipb convert dataset=ridi source=raw/ridi workers=1
ipb check data=ridi full=true hash=true
```

输出包含 `sequences/*.h5`、`splits/*.txt`、`dataset.json` 和 `conversion_report.json`。配置在 [cfg/datasets/](src/inertial_benchmark/cfg/datasets/)。官方 seen/unseen 标签可以保留，不强制其他数据集使用同一标签。

### 3. CPU 流程示例

下列配置用于检查数据、训练和结果输出；步长与训练预算仅用于工程检查：

```bash
ipb train model=ronin_resnet18 data=ridi device=cpu epochs=1 batch=2 workers=0 cache=false stride=200 eval_stride=200 project=runs/cpu-smoke name=ridi-ronin
ipb val model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt data=ridi split=val device=cpu workers=0 cache=false eval_stride=200 project=runs/cpu-smoke
ipb predict model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt source=datasets/ridi/sequences device=cpu workers=0 eval_stride=200
```

等价 API：

```python
from inertial_benchmark import NIO

model = NIO("ronin_resnet18")
model.train(data="ridi", device="cpu", epochs=1, batch=2, workers=0, cache=False)
```

正式 GPU 实验应冻结数据划分、配方与充分训练预算，记录统一/官方配方的差异。CPU 一轮结果用于工程验收，不能作为算法排名或论文精度结论。

### 4. 矩阵与报告

```bash
ipb benchmark cfg=main dry_run=true
ipb report runs=runs/cpu-smoke out=reports/cpu-smoke
pytest -q -m "not slow"
```

示例矩阵见 [main.yaml](src/inertial_benchmark/cfg/benchmarks/main.yaml)。正式执行前选择已验收的数据/算法并检查资源预算；完整命令见 [CLI.md](docs/CLI.md)。

APP 源码、APK、`.iplab` 转换与构建统一见 [app-inertial-lab](https://github.com/BUG423/app-inertial-lab)。旧文件读取兼容与 v1 训练准入不同，限制见[交换契约](docs/TOOLS.md)。

## 🔬 研究与验证状态

当前 AutoDL 已将 RIDI、RoNIN 各三条录制转换为原生 v1，并完成代表短段的 CPU NIO 训练、验证、预测与 checkpoint 重载一致性检查（各 1 epoch / 2 updates）。新增 31 份原生 v1 文件已验收；完整测试计数以本轮记录为准，之后开展 GPU 充分训练。2026-09-24 交接中的 **705 项测试**、旧机器成绩及 AMP/dropout 观察属于历史记录，不代表当前服务器复验；本轮以[项目状态](docs/DEV_V1_INTEGRATION.md)和验收记录为准。

路径普遍偏短需要分别检验目标/积分 oracle、时间与长度分辨率、速度幅值/方向、训练和推理模式失配、精度设置与泛化。已有长度分解和模式守卫提供起点，结论须由充分训练的多算法实验支持。

| 研究入口 | 证据状态 |
| :--- | :--- |
| [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/) | 历史坐标系研究与资源 |
| [PostDiffIO](research/methods/postdiffio/) · [ModeMoEIO](research/methods/moe-io/) | 方法提案，尚无本轮可复现 benchmark 成绩 |
| [PaperFlow IO](tools/paperflow-io/) | IO 选题、文献复核与实验规划 |
| [FAST-LIVO 环境记录](docs/setup/fast-livo.md) | 历史 ROS Noetic 复现说明 |

## 🛣️ 路线图

| 阶段 | 交付与验收 |
| :--- | :--- |
| **已有主干** | v1 序列/任务视图、八个核心转换器、多算法接口、训练/验证/预测、指标与报告 |
| **当前 CPU 阶段** | EuRoC/Blackbird 原生转换与数据检查已验收；完成当前服务器完整 suite 和执行环境验收 |
| **评价增量** | 输入权限 track、NA/覆盖、3D 图一致性、超过八模型的分页/分面、六面板与静态 HTML |
| **GPU 阶段** | 单 seed 验证配方与充分训练，再扩展矩阵；最终关键实验最多三个 seed |
| **研究交付** | 可审计 benchmark、收缩机制与受控实验、可复现发布；论文目标 ICRA / IEEE RA-L |

详细待办见[项目状态](docs/DEV_V1_INTEGRATION.md)。

## 📄 知识产权与开源协议

本项目采用 **[CC BY-NC-ND 4.0](LICENSE)**（署名—非商业性使用—禁止演绎 4.0 国际许可）。按仓库许可证使用、署名与共享；未经许可不得用于商业目的或分发修改后的版本。

数据集、第三方代码、独立工具与权重保留各自来源及条款，仓库许可证不替代第三方授权。复用算法请引用对应论文并记录来源/适配。提交前阅读[贡献规范](CONTRIBUTING.md)。

---

<div align="center">

如果本项目对你的研究有帮助，欢迎 ⭐ Star，并关注可复现 benchmark 的后续发布。

</div>
