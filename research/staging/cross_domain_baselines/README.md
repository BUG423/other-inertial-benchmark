# 跨领域惯性速度 baseline：隔离研究区

本目录交付三个可追溯的架构适配，以及服务器 CPU 科学/生命周期证据，
不交付尚未运行的精度结果。最终研究问题和阶段判据见
[跨域研究计划](../../../docs/CROSS_DOMAIN_BASELINES_PLAN.md)。

| 架构适配 | 来源规格与实现边界 | 独立服务器科学测试 | 原生生命周期 |
| :--- | :--- | ---: | :--- |
| ModernTCN | [source card](moderntcn/source-card.md) | 31 passed | XY / XYZ 均通过 |
| PatchTST | [source card](patchtst/source-card.md)；CI 编码器 + 有序速度头 | 38 passed | XY / XYZ 均通过 |
| S4D-Lin/ZOH | [source card](s4d/source-card.md)；纯 PyTorch 对角 SSM | 33 passed | XY / XYZ 均通过 |

另有 6 项共享 [NIO 生命周期测试](test_native_lifecycle.py)，共 **108 passed / 0 failed / 0 skipped**。
确切源码、fixture 和收据身份见 [CPU 验收记录](CPU_HANDOFF.md)。
全部框架测试在 AutoDL 完成，本机仅做源文件检查和远端编排。

共同输入为 200 Hz、200 点、固定 `[gyro_xyz, acc_xyz]` 的六轴窗口，
`gravity_world` / `reference` / 保留重力；从头监督训练，输出窗口平均速度，使用 MSE。
XYZ 真实学习 Z；XY 主诊断由同一 XYZ 输出切片得到，二维 YAML 是另一个参考实验。
所有架构仅观察给定窗口，均不声称零延迟流式推理。

模型/配置未加入公开 registry 和 `cfg/models`，当前正式 **2,067 cells**、
惯性领域论文方法 **23**、正式来源 **13** 的口径全部不变。
G1/G2 工程验证已通过，CUDA 容量、收敛/精度、三 seed pilot、完整数据评估和下一版准入均未完成。
进入后续阶段的依赖见 [下一阶段交接](NEXT_STAGE_HANDOFF.md)。
