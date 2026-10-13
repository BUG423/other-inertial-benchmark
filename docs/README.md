# 📚 文档索引 / Documentation

[中文入口](../README.md) · [English entry](../README_EN.md)

交付边界是统一数据契约、算法接口、可比评价与可审计实验；以 `dev/v1` 的 NIO/ipb/IPB v1 为准。建议阅读当前状态 → 设计 → 来源卡 → 命令 → 指标。

| 文档 | 用途与证据边界 |
| :--- | :--- |
| [DEV_V1_INTEGRATION.md](DEV_V1_INTEGRATION.md) | 本轮 CPU/GPU 阶段、数据接入与未完成增量 |
| [DESIGN.md](DESIGN.md) | 架构、v1 数据契约、任务视图、推理与评价 |
| [CLI.md](CLI.md) | 实际命令、配置、矩阵与输出文件 |
| [METRICS.md](METRICS.md) | 指标公式、单位、支持条件、统计与已有边界行为 |
| [DATASETS.md](DATASETS.md) · [datasets/](datasets/) | 数据集目录和来源、字段、许可与转换卡 |
| [DATASETS_V1.md](DATASETS_V1.md) | 既有转换/划分记录；历史数量按原环境解释 |
| [ALGORITHMS.md](ALGORITHMS.md) · [algorithms/index.yaml](algorithms/index.yaml) | 算法规格、来源、适配、假设和夹具；代码存在不等于精度复现 |
| [PAPERS.md](PAPERS.md) | 代表性论文与发布状态 |
| [FORMAT.md](FORMAT.md) | 旧 v0.1 格式；读取兼容与 v1 训练准入不同 |
| [TOOLS.md](TOOLS.md) | 独立 APP、交换契约与本仓库工具边界 |
| [Ops v1](../tools/ops_v1/README.md) | 独立环境安装、服务器验收、资源 profile 与本地私有配置 |
| [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md) | 源码、配置、工具、研究与实验产物职责 |
| [HANDOFF.md](HANDOFF.md) | 2026-09-24 历史交接与收缩观察，不替代当前服务器验收 |
| [setup/fast-livo.md](setup/fast-livo.md) | 历史 ROS Noetic 依赖与编译说明 |

区分已实现代码、当前环境验收、历史结果、方法提案与未来计划。第三方资料保留独立条款；公开文档不保存服务器凭据或内部运维路径。
