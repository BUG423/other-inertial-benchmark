# 🛠️ 工具索引 / Tools

| 组件 | 入口 | 职责与维护边界 |
| :--- | :--- | :--- |
| Inertial Positioning Lab | [BUG423/app-inertial-lab](https://github.com/BUG423/app-inertial-lab) | 独立 Android 采集、转换、端侧评测、构建、APK 与测试 |
| [PaperFlow IO](paperflow-io/) | 本仓库 IO 研究工具 | 选题、可行性、实验规划；有独立说明和条款 |
| [Ops v1](ops_v1/README.md) | 隔离服务器部署与验收 | 私有 SSH 配置、资源 profile、CPU 闭环；GPU profile 需要实际硬件验收后启用 |
| Benchmark conversion / reporting | [ipb CLI](../docs/CLI.md) | 可复用转换与汇总代码位于 [Python core](../src/inertial_benchmark/) |

移动端内嵌代码已解耦，本仓库保留[交换契约](../docs/TOOLS.md)和数据契约测试。新工具应有独立说明、依赖和验证路径；机器凭据与生成产物不进入版本库。
