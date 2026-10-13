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
| **数据接入** | 13 个独立来源的已取得版本/子集通过原生 XYZ 准入，共 1,376 条、114.30638333 小时；9 个有 train/val，4 个仅 test | [数据卡](docs/datasets/) · [转换器](src/inertial_benchmark/data/converters/)；来源、拒收与划分见[本轮状态](docs/DEV_V1_INTEGRATION.md) |
| **算法库** | 33 个活跃架构、57 份配置；23 个论文方法（21 个已发表、2 个预印本）及 2 个经典定义 | [规格与忠实性](docs/ALGORITHMS.md) · [模型配置](src/inertial_benchmark/cfg/models/)；骨干和维度变体不增加独立论文数量 |
| **训练与推理** | `NIO` / `ipb`，统一 Trainer、Validator、Predictor；窗口、逐帧、多步和序列级接口 | [命令与配置](docs/CLI.md)；统一流程保留算法必要的布局和损失；`NIO.export()` / `ipb export` 提供原生科学推理包及已审查的 RoNIN ResNet、IMUNet 张量适配器（TorchScript/ONNX），见[导出说明](docs/EXPORT.md) |
| **评价与报告** | 56 个默认 XYZ 轨迹/速度指标输出键，同预测 XY 派生；ATE/RTE/RPE、误差分位数、漂移、长度/方向、覆盖与效率；原生任务高斯不确定性 | [指标定义](docs/METRICS.md)；原生任务不确定性 25 节点已通过；无预测姿态或协方差时明确不可用 |
| **跨域数据** | EuRoC、Blackbird、GrandTour、SubT-MRS、UZH-FPV 已按真实参考契约接入；后三者取得代表子集 | [本轮状态](docs/DEV_V1_INTEGRATION.md) · [数据卡](docs/datasets/)；仅 test 来源采用预注册 checkpoint 迁移，人体到机器人单列诊断 |
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

### 新增方法与冻结实验范围

当前代码库为 **33 架构 / 57 配置 / 23 论文方法（21 已发表、2 预印本）+ 2 经典定义**：XYZ 33（32 neural + RIDI3D），XY 24（21 neural + RIDI2D + 2 classics）。新接入 [M2EIT](docs/algorithms/m2eit.md) 与 [DiffusionIMU](docs/algorithms/diffusionimu.md) 各两配置，已在独立 443 文件源码上通过 **31 个科学测试和四份默认宽度配置的真实 CPU / 原生导出闭环**。57 份 CPU 准入是原 53 与新 4 各自 exact-source 收据的组合，不是假称同一批 57 个测试；详见[新方法验收](docs/DEV_V1_INTEGRATION.md#新增两方法实际验收与旧正式范围)。新四配置的 CUDA、GPU batch 128 容量和正式训练仍 pending。

运行中的 [common-benchmark-v1](docs/protocols/common-benchmark-v1.json) 保持 **31 架构 / 53 配置 / 2,067 单元**、旧 source `1897c9cc…0949`、XYZ 31 / XY 22。当前新增库配置没有中途替换其 checkpoint、recipe 或覆盖身份。57×13×3=2,223 只是未来 inventory，尚未冻结或启动新正式矩阵。旧 49 neural CUDA 与旧 53 native export 证明仍保留独立来源，新四 native export 另批验收；TorchScript/ONNX 不自动扩展到新方法。

两个新方法均为公开假设的论文独立重实现，未核验到作者可运行网络源码。M2EIT 的 Haar/embedding/scan 假设、DiffusionIMU 的随机评估和 u 非 σ 边界仍在规格卡保留；CPU/导出通过不等于原论文精度复现。

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
python -m pip install -e ".[train,test,ridi,grandtour,idol]"
ipb info
ipb cfg model=ronin_resnet18_3d metric_dims=3 recipe=unified
```

上例安装完整 benchmark 的可选依赖：`ridi` 提供 scikit-learn，`grandtour` 提供 numcodecs，`idol` 提供 Feather/Arrow 读取。仅使用基础数据层可安装 `pip install -e .`，按数据卡选择所需 extras；PyTorch CPU/CUDA 构建按环境选择。

### 2. 转换与检查

先按[RIDI 数据卡](docs/datasets/ridi.md)取得原始文件，并将 `source` 改为实际位置：

```bash
export IPB_DATASETS="$PWD/datasets"
ipb convert dataset=ridi source=raw/ridi workers=1
ipb check data=ridi full=true hash=true
```

输出包含 `sequences/*.h5`、`splits/*.txt`、`dataset.json` 和 `conversion_report.json`。配置在 [cfg/datasets/](src/inertial_benchmark/cfg/datasets/)。官方 seen/unseen 标签可以保留，不强制其他数据集使用同一标签。

已完成本轮验收的 AutoDL 仓库中，当前已验收的 RIDI/RoNIN/IDOL/RNIN/TLIO/OxIOD/IMUNet 版本位于 `datasets/ipb-v1/full-core/`。在该服务器仓库根目录使用：

```bash
export IPB_DATASETS="$PWD/datasets/ipb-v1/full-core"
ipb check data=ridi full=true hash=true
ipb cfg model=ronin_resnet18_3d data=ridi metric_dims=3 recipe=unified
```

### 3. CPU 流程示例

CPU 验收统一在 AutoDL 执行。下列配置用于检查数据、训练和结果输出；步长与训练预算仅用于工程检查：

```bash
ipb train model=ronin_resnet18_3d data=ridi metric_dims=3 device=cpu epochs=1 batch=2 val_batch=2 workers=0 cache=false stride=200 eval_stride=200 project=runs/cpu-smoke name=ridi-ronin
ipb val model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt data=ridi split=val metric_dims=3 device=cpu val_batch=2 workers=0 cache=false eval_stride=200 project=runs/cpu-smoke
ipb predict model=runs/cpu-smoke/train/ridi-ronin/weights/best.pt source="$IPB_DATASETS/ridi/sequences" metric_dims=3 device=cpu val_batch=2 workers=0 eval_stride=200
```

等价 API：

```python
from inertial_benchmark import NIO

model = NIO("ronin_resnet18_3d", metric_dims=3)
model.train(data="ridi", device="cpu", epochs=1, batch=2, val_batch=2, workers=0, cache=False)
```

统一原生数据保留 **XYZ 三维**。具备真实 XYZ 输出的配置使用 `metric_dims=3`，并从同一预测的 XY 分量派生 `xy_` 指标；原文二维配置留在独立 XY 参考协议。`3d_adapted` 配置明确记录输出头及必要目标适配，不宣称原论文精度或二维 checkpoint 等价。33 个架构覆盖 57 份活跃配置，维度与输入能力逐配置准入；缺少预测 Z 不以补零或参考真值代替。TLIO 当前范围为网络，不含论文 EKF。

参考姿态或初始状态作为输入时记录 `reference/oracle` 权限，与可部署的传感器输入分 track；同设备 VIO 标签按参考性质解释。旧正式源的 49 个神经配置已在 4090D 完成 GPU 工程生命周期；当前 57 份配置的 CPU 准入由旧 53 与新 4 独立源码收据组成，新 4 的 GPU 容量待验。旧统一正式配方已冻结并开始 GPU 实验；各模型的科学窗口、目标、损失与阶段保持原定义，完整矩阵仍在运行。

### 4. 矩阵与报告

`cfg=main` 是 AutoDL 部署验收和正式配方冻结后的完整矩阵入口，依赖已生成的数据 registry、随机种子记录、冻结 policy 和资源 admission。首次 clone 需先完成这些准入；本轮已冻结的参数见[公开统一协议](docs/protocols/common-benchmark-v1.json)，它不包含服务器私有运行配置。上面的单模型流程可先用于检查安装和数据接口。

```bash
ipb benchmark cfg=main dry_run=true
ipb report runs=runs/cpu-smoke out=reports/cpu-smoke
pytest -q -m "not slow"
```

上面的 [main.yaml](src/inertial_benchmark/cfg/benchmarks/main.yaml) 是 CLI 配置示例。正式矩阵由 [full_matrix.py](tools/ops_v1/full_matrix.py) 从活跃配置、已验收数据 registry 和一次抽取的三随机种子动态构建：运行中的冻结源码仍为 53 × 13 × 3 = **2,067 个覆盖单元**；当前库 57 份配置的 2,223 行 inventory 仅为后续候选。不适用、待 checkpoint 与失败都有明确状态；矩阵枚举不代表已经训练。完整命令见 [CLI.md](docs/CLI.md) 和 [Ops v1](tools/ops_v1/README.md)。

APP 源码、APK、`.iplab` 转换与构建统一见 [app-inertial-lab](https://github.com/BUG423/app-inertial-lab)。旧文件读取兼容与 v1 训练准入不同，限制见[交换契约](docs/TOOLS.md)。

## 🔬 研究与验证状态

当前已验收 **13 个独立来源、1,376 条原生序列、114.30638333 小时**：RIDI 94、RoNIN 151、IDOL 130、RNIN 300、TLIO 354、OxIOD 126、IMUNet 113、EuRoC 11、Blackbird 20、GrandTour 1、SubT-MRS 28、UZH-FPV 1、PedLocData Demo 47。八个人体来源加 EuRoC 共九个可训练，827 条 train；Blackbird、GrandTour、SubT-MRS、UZH-FPV 当前只含 test。这是已取得版本或代表子集的验收，不代表所有公开全集已收齐。参考可能来自外部定位、VIO、融合估计或 vision-based pose，逐集保真边界与残余 warnings 见[项目状态](docs/DEV_V1_INTEGRATION.md)。

新服务器存储快照（**2026-10-14 03:02 北京时间 / 2026-10-13 19:02 UTC**）：所有保留的归档、解压 raw、原生、历史副本、工程夹具及 partial 按 inode 去重占 **84.448 GB / 78.648 GiB**；13 个正式 native 占 **4.721 GB**；当时可用 **167.593 GB**。审计 119 个数据根、8,937 个唯一 inode，0 errors。后续后台下载与实验输出会改变占盘。正式序列统计不加入 pilot、历史副本、TartanIMU 混合再包装或重叠辅助 test 视图。本机既有原始归档独立记账，不与服务器总量相加。

当前 57 份配置均有相应 CPU 准入：**旧 53 与新 4 分别在各自冻结源码验收**；旧 **49/49 神经配置 GPU 生命周期通过**，新 4 CUDA / batch128 待验，包括真实训练更新、验证、预测与 checkpoint 重载；经典或 CPU-only 方法按其真实标定/回归逻辑验收。硬件为 RTX 4090D 24 GiB。短工程截段和少量更新不构成论文精度复现。**统一协议正式 GPU 实验已运行，完整矩阵尚未完成**，已有实际单元完成 train、best checkpoint 和 test。框架目标 49 unique 节点最终通过，原生任务不确定性 25 节点已验；当前冻结源码的锁、来源/seed 身份与 dispatcher 定向节点 8/8 通过，历史 suite 计数分别保留。

旧正式源码的 13 个来源采样资格覆盖 **2,067 行：2,019 通过、48 输入不支持、0 待验**；48 行均是 `mean_speed_heading`/`pdr` 所需设备姿态缺失，保留真实原因，不用参考姿态补齐。正式配方为 **30 epochs、min 15、patience 10、每 5 epoch 验证、batch 128、train stride 200 / eval stride 10、AdamW lr 1e-3**；完整参数与科学例外见[统一协议](docs/protocols/common-benchmark-v1.json)。资格通过不等于训练完成，其他数据来源的计时仍为容量测量投影，不承诺整矩阵完工时间或收敛。

PedLocData 本轮只接入已完整下载和官方 MD5 校验的 Demo：47 接收 / 0 拒收、grouped 32/9/6、录制组 9/2/2 不交叉，full/hash 与全部真实 XYZ/XY B2 通过。主包的有限下载于 2026-10-13 21:31 UTC 因反复提前断流、部分分块耗尽 40 次尝试而失败，服务器保留 **10,236,096,133 / 15,138,253,808 B**，尚未完整校验或转换；公开全集的历史统计不算成本轮已取得范围。

全矩阵统一使用一次从操作系统随机源抽取的种子 **1735231569、843628571、951150512**。四个 test-only 来源的 checkpoint 选择预注册：Blackbird/UZH-FPV 使用同配置同种子的 EuRoC checkpoint；GrandTour/SubT-MRS 使用 RIDI checkpoint，人体到机器人属于跨域诊断，不进入人体同域训练榜。每个单元保留来源、数据指纹、科学输入契约、配方和 checkpoint 身份。

算法计数分层报告：**33 架构 / 57 配置 / 23 论文方法（21 已发表 + 2 预印本）/ 2 经典定义**。16 个官方仓库已取得固定提交的参考源码，其中 CTIN、iMoT、MEIO 属 `source_partial`，独立适配与论文重实现各自标注。按仓库拥有者的作者排除规则，张珊珊（Shanshan Zhang）相关被排除方法不进入活跃队列，见[排除记录](docs/models/excluded-methods.json)。开源代码存在、接口通过和原论文精度复现是分别记录的证据。

路径普遍偏短需要检验目标/积分 oracle、时间与长度分辨率、速度幅值/方向、训练和推理模式、精度设置与泛化。已准备统一诊断，结论须由完整多算法实验支持。

| 研究入口 | 证据状态 |
| :--- | :--- |
| [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/) | 历史坐标系研究与资源 |
| [PostDiffIO](research/methods/postdiffio/) · [ModeMoEIO](research/methods/moe-io/) | 方法提案，尚无本轮可复现 benchmark 成绩 |
| [PaperFlow IO](tools/paperflow-io/) | IO 选题、文献复核与实验规划 |
| [FAST-LIVO 环境记录](docs/setup/fast-livo.md) | 历史 ROS Noetic 复现说明 |

## 🛣️ 路线图

| 阶段 | 交付与验收 |
| :--- | :--- |
| **数据准入** | 当前 13 独立来源通过原生 XYZ、完整性与科学契约检查；持续扩展真实数据源，缺参考/坐标契约明确阻碍项 |
| **全部算法准备** | 33 架构 / 57 配置各有 source 绑定 CPU 准入（旧 53 + 新 4）；旧 49 神经配置通过 GPU，新 4 GPU 待验；保留原生目标、布局和损失 |
| **正式矩阵执行** | 统一配方已冻结、2,067 行采样资格已核验；正式 GPU 单元正在运行，持续保存实际覆盖/失败/依赖与成本 |
| **评价与研究** | 同预测 XYZ/XY、长度收缩分解、覆盖/NA、参考权限和跨域 track，充分训练后形成主表和受控消融 |
| **研究交付** | 可审计 benchmark、可复现发布和收缩机制证据；目标 ICRA / IEEE RA-L |

详细待办见[项目状态](docs/DEV_V1_INTEGRATION.md)。

## 📄 知识产权与开源协议

本项目采用 **[CC BY-NC-ND 4.0](LICENSE)**（署名—非商业性使用—禁止演绎 4.0 国际许可）。按仓库许可证使用、署名与共享；未经许可不得用于商业目的或分发修改后的版本。

数据集、第三方代码、独立工具与权重保留各自来源及条款，仓库许可证不替代第三方授权。复用算法请引用对应论文并记录来源/适配。提交前阅读[贡献规范](CONTRIBUTING.md)。

---

<div align="center">

如果本项目对你的研究有帮助，欢迎 ⭐ Star，并关注可复现 benchmark 的后续发布。

</div>
