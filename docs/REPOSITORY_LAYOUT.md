# 🗂️ 仓库布局 / Repository Layout

以 `dev/v1` 的 NIO/ipb/IPB v1 为唯一框架主线，目录职责用表格说明。

| 位置 | 职责 | 准入要求 |
| :--- | :--- | :--- |
| `src/inertial_benchmark/data/` | v1 序列、重采样、转换流水线、划分/任务视图；legacy 为读取兼容 | 单位、坐标系、mask、来源与有效范围明确，不制造 GT |
| `src/inertial_benchmark/data/converters/` | 原始数据解析与 v1 适配 | 来源卡、独立条款、物理自检、转换报告与边界测试 |
| `src/inertial_benchmark/cfg/datasets/` | 数据、转换器与划分配置 | 路径可审计，不保存 raw 或服务器凭据 |
| `src/inertial_benchmark/models/` · `nn/` | 算法包、注册架构、公共网络/损失 | 来源和论文假设明确，保持必要输入/输出语义 |
| `src/inertial_benchmark/cfg/models/` · `cfg/benchmarks/` | 模型和实验配方 | 划分、预算、seed、输入权限与评价协议可追溯 |
| `src/inertial_benchmark/engine/` | NIO、训练、验证、预测、矩阵 | 使用统一视图并保存实际运行证据 |
| `src/inertial_benchmark/metrics/` · `utils/` | 指标、效率、统计、结果与图 | 单位、定义、能力和支持域明确，诊断不混主排名 |
| `tests/` | 数据/模型契约、解析解、引擎和文档 | 有意义且可重复，CPU 检查不替代充分训练 |
| `docs/` | 规范、状态、来源卡和环境记录 | 历史环境/日期与当前验收分开 |
| `tools/paperflow-io/` | IO 专用研究工作流 | 独立说明、依赖、验证与第三方条款 |
| `tools/ops_v1/` | 隔离环境部署、资源配置与服务器验收 | 说明实际已验行为；凭据只保存在忽略的本地配置 |
| `research/methods/` · `research/studies/` | 提案与研究记录 | 标注证据状态，保存配置、来源与限制 |
| `.github/workflows/` | Python 核心 CI 与文档检查 | 保留 v1 训练依赖和测试矩阵；APP 构建在独立仓库 |
| `datasets/`、`runs/`、`reports/`、`external/` | 数据、checkpoint、结果和第三方资产 | 不提交生成产物或凭据，记录实际指纹 |

## 组件边界

Android 源码、构建、APK、转换和端侧测试由独立 [BUG423/app-inertial-lab](https://github.com/BUG423/app-inertial-lab) 维护。本仓库保留交换契约与独立数据测试，不内嵌 Android 项目。可复用转换行为归 data/converters。

环境记录归 docs/setup，注明系统、依赖、来源版本与限制。研究提案具备实现、冻结数据/配方、来源版本、日志和验收后，才进入正式 benchmark。

## 命名与维护

- 仓库名 `other-inertial-benchmark`，Python import `inertial_benchmark`，CLI `ipb`。
- 开发/安装入口 `dev/v1`；公开仓库链接使用 `BUG423/other-inertial-benchmark`。
- README.md 为中文/default，README_EN.md 为 English；组件入口使用表格与 Mermaid。
- 库内优先相对链接，由 CI 检查；外部工具指向独立仓库。
- 路径/契约/配置变更同时更新命令、代码、测试和最近的文档入口。
- 机器配置与凭据只保存于被忽略的本地配置，不进入公开文档或实验附件。
