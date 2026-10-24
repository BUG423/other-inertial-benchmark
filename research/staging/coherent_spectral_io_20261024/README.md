# Coherent Spectral IO：独立候选与验证计划

当前状态：Agent CPU 工程闭环通过；完整数据精度、CUDA 容量、论文创新性尚未确立。
本候选为独立实现，参考 CVPR 2026 的
[PFGNet](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.html)
与 [Spectral Scalpel](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Spectral_Scalpel_Amplifying_Adjacent_Action_Discrepancy_via_Frequency-Selective_Filtering_for_CVPR_2026_paper.html)
的多尺度外围门控与频域建模思想。两篇官方主会页面在 2026-10-24 重新读取，分别为 HTTP 200。
它不是二者的官方惯导复现，也不替代既有十方法 tail 队列。

## 任务与模块

目标是相同 IMU 输入、训练预算和部署预算下改善未见录制的定位精度。
轨迹长度是诊断量；本网络允许速度增大、减小或保持，不预设所有结果都应放大。
输入为 B×6×200、200 Hz，按 gyro_xyz/acc_xyz 排列；输出为真实学习的 B×3 平均速度。
首轮沿用 gravity_world/reference 输入，因此属于 reference_oracle，不能称为可部署姿态方案。

| 模块 | 机制与可检验目的 | 反例/风险 |
| --- | --- | --- |
| 多尺度时序主干 | 5/15/31 时序深度卷积、中心抑制、输入门控和残差 | 单位、噪声和训练配方可能比主干更关键 |
| DC 保留频域门控 | 直流增益固定 1，其余频率正实增益限制在 0.75–1.25；初始为恒等映射 | 只保留潜特征均值，不保证物理速度无偏或精度提高 |
| 跨传感器延迟交互 | 当前窗内 gyro/acc 潜特征在 -2/0/+2 样本延迟上作双线性交互，无循环回绕 | 没有旋转等变保证；时钟偏差和噪声可能伤害 |
| 时序矩残差读出 | 均值、变化幅度和差分能量产生有符号 XYZ 残差；零初始化对齐 GAP 控制 | 容量增加可能解释收益，因此保留等参数 mean-only MLP 控制 |

频域门控位于非线性主干之前：若直接接 GAP+Linear，DC 保留会让该模块在数学上被消去。
新增检查明确验证 spectral-only 的梯度路径，避免一个结构上无效的消融。
没有沿用早期 SAGE 方向×速度×gain 的冗余参数化，也没有把修改损失混入 MSE 消融。

## 已实际完成

[cpu_evidence.json](cpu_evidence.json) 绑定四个执行源码、JUnit 与远端叶目录。
Agent 上用 Torch 2.12.1+cu130、两线程、CUDA 屏蔽、只读归档 namespace 执行：

- 6 项数学、梯度、跨窗隔离、等参数控制、checkpoint 检查通过；
- 真实 RIDI 三条不同录制的独立工程片段，train 4 windows、val 2 windows，实际 2 次更新；
- NIO 训练→验证→checkpoint 重载→native export 重载，预测逐位一致；
- 参数 65,262，CUDA 未初始化，原始三条数据文件哈希前后相同。

两次更新的模型尚未充分训练，工程验证确实出现 speed_ratio 过大的 guard 警告。
本轮不拿这份极短片段的误差或长度比作精度证据，也不按它调整网络。
首个导出尝试发现冻结正式源码不含 NIO.export；最终用已有独立 `598646e7…` 工程源码快照验证，
没有热更新正式 `6c973dd2…` 源码。历史失败叶和日志保留。

## 独立 pilot 与退出条件

[pilot_plan.json](pilot_plan.json) 定义 7 配置 × RIDI/RoNIN/IDOL × 3 固定 seed = 63 个训练单元：
同主干 GAP、spectral-only、cross-only、moments-only、等参数 mean-only MLP、完整候选、RoNIN ResNet18 XYZ。
完整候选与等参数 MLP 控制使用相同主干和模块参数量，后者的读出只接均值特征。
另对 GAP 控制建立仅 validation 拟合、冻结后用于 test 的标量 gain 参考，不在 test 上拟合。

统一为 30 epochs、B128、FP32、AdamW、MSE、train stride200/eval stride10；
关闭早停以匹配曝光次数，每 5 epoch 以 validation mean ATE 选 checkpoint。
这些是新 pilot 的配方，不会改现有正式矩阵的早停政策。
至少两来源 ATE 相对改善 5%，任何来源 ATE 恶化不超过 2%，10s RTE 不恶化超过 2%，
同时报告 subject-group 配对区间、全部 seed、失败、P95 风险和计算成本。
只改善 PLR、输给标量 gain/等参数 MLP，或只在一个 seed 有效，就不把对应模块列为主要贡献。
这些是预先声明的工程筛选阈值，不能作为保证发表的标准。

当前 execution_enabled=false：先完成现有及先前排队任务的资源交接，再做绑定新源码的 CUDA B128 容量验收。
数据 manifest/split 指纹和容量收据也必须先绑定，然后独立启动；不能复用旧正式模型的结果冒充同配方控制。
已有 CVPR 十方法继续保持后置，本候选不绕过它们的收集守卫或修改冻结 2,067-cell 范围。

## 科学边界

MSE 最优估计为 E[v|IMU]。Jensen 不等式给出 ||E[v|IMU]|| ≤ E[||v|| | IMU]；
方向不确定时向量相消可导致幅值减小，但这不是有限模型必然缩短的定理。
强行恢复长度可能增加向量误差和轨迹漂移；要改善的是可辨识运动信息和定位风险。
当前模块组合只是可证伪的候选，仍需系统文献查重、强 baseline 和跨来源验证，不能声称已证明新颖或可发表。
