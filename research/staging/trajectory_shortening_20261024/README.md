# 轨迹长度偏差与 SAGE-IO 保底候选

本目录记录 2026-10-24 对正式矩阵当前结果的只读统计，以及一个隔离的新版本候选。
它不注册模型、不改变 `53 profiles × 13 sources × 3 seeds = 2,067` 的冻结矩阵，也不继承旧结果的身份。

## 当前能支持的结论

远端正式矩阵观察到 981 个已完成主 `test` split 的单元。以指标文档定义的
`plr = L_1s(prediction) / L_1s(reference)` 计算：

| 分层 | 单元数 | `plr < 1` | 中位数 PLR | 解释 |
|---|---:|---:|---:|---|
| 全部已完成主 test | 981 | 850（86.65%） | 0.8506 | 混合了同域、跨域、test-only 和不同输入契约 |
| 人体同域数据（RIDI/RoNIN/OxIOD/IDOL/IMUNet/TLIO/RNIN） | 405 | 392（96.79%） | 0.9084 | 当前最强的条件性证据 |
| UAV/机器人及其他 test-only | 576 | 458（79.51%） | 0.4267 | 域外参考和 checkpoint 迁移不能并入人体主结论 |
| 同一 profile/dataset 的配对组 | 290 组 | 86.21% 的组均值 `< 1` | — | 其中 256 组拥有至少三个 seed |

这已经足以把“同域人体结果经常出现长度变短”列为可复核现象，但还不足以写成“所有算法都如此”或“90% 普遍规律”。单元共享算法、数据集和序列，不能把 981 个单元当作 981 个独立受试者；当前汇总也还没有置信区间、逐序列 group bootstrap 或预注册的效应阈值。

更关键的是指标分解恒等式：

```text
plr_dense = oracle_ratio × plr_oracle
```

`oracle_ratio` 是窗口平均速度和 1 秒折线采样带来的协议平滑效应，`plr_oracle` 才更接近模型输出幅度效应。因此只把 `plr` 拉回 1 不能证明定位更好。已有结果也包含 seed、域外和算法级反例；本目录的 `claims_proven` 保持为空。

## 第一性原理问题

我们真正要解决的是：在固定 IMU 窗口、训练预算和推理预算下，模型是否系统性低估了速度幅度，导致累计位移偏差；如果存在，这个偏差能否在未见录制上降低 ATE/RTE，而不是只让 PLR 更接近 1。

因此研究顺序固定为：

1. 用 `plr_oracle`、`oracle_ratio`、`speed_ratio`、`along_bias` 和方向误差区分协议平滑、幅度误差和方向错误。
2. 按录制分组、按 seed 配对，使用 group bootstrap/置换检验；不把模型 profile 当作独立样本。
3. 先与恒等映射、一个全局 gain、固定因果滤波和固定 stride 比较。
4. 只有在 ATE/RTE 或固定时域漂移同时改善时，才保留可学习模块；只改善 PLR 就停止。
5. 对保留的模块使用新 profile、新 recipe、新 coverage 身份，不能回填当前正式结果。

## SAGE-IO 候选

`SAGE-IO`（Scale-Aware Gated Estimator for Inertial Odometry）是一个保底候选，组合了成熟的多尺度残差时序算子和针对当前现象的幅度/方向解耦读出：

- `MultiScaleTemporalBlock`：三个 1D depthwise temporal branches（kernel/dilation 为 3/1、7/2、15/4），用残差和输入门控融合。它借鉴 ResNet/ConvNeXt/TCN/MS-Temba 的可解释时序归纳偏置，但不是任一论文的完整复现。
- `ScaleGuardHead`：把窗口速度分解为单位方向、正速度幅度和有界 gain；gain 零初始化，初始行为接近普通速度头，避免一开始强行放大。
- `scale_guard_loss`：在原向量损失之外，可在新 recipe 中加入幅度 Smooth-L1 和非零目标方向损失。该辅助损失只属于新 recipe，不能混入冻结矩阵。

候选输入仍是 `B×6×200`、200 Hz、`[gyro_xyz, acc_xyz]`，输出是窗口平均 XYZ 速度。`ScaleGuardHead` 不读取 test 标签、历史窗口或参考轨迹；训练外的轨迹长度诊断只读结果文件。

候选的最小 pilot 是 `ronin_resnet18_3d`/`rio_3d` 对照，在 RIDI、RoNIN、IDOL 三个来源和三个固定 seed 上运行新 profile。先做 CPU/native lifecycle 和单卡容量，再决定是否启动 27 个候选单元。未通过下列任一条件就停止：

- PLR 接近 1 但 ATE/RTE 不改善；
- 一个全局 gain 或固定因果规则同样有效；
- 只在单一来源、单一 seed 或 privileged reference track 有效；
- 方向误差、`train_eval_gap` 或跨域尾部风险恶化；
- 模块收益来自改变数据划分、窗口、AMP、checkpoint 或 test 拟合。

代码见 `scale_guard.py`，静态契约见 `test_scale_guard.py`。当前状态是 `proposal_only`，尚未 import、训练、容量验收或注册。

## 研究与正式范围分离

- 13 个正式来源已在服务器完成归档和校验；4 个 test-only 来源（Blackbird、UZH-FPV、GrandTour、SubT-MRS）仍按预注册 checkpoint 迁移规则单独报告。
- 当前库有 33 个架构、57 个 profile；冻结矩阵只包含旧 53 个 profile。M2EIT/DiffusionIMU 四个 profile 已有独立 CPU/native 证据，但 CUDA、batch128 容量和正式训练仍 pending。
- CVPR 2026 十个架构草稿仍是 `static_draft`，没有运行或正式注册。SAGE-IO 是另一个针对现象的隔离候选，不增加论文方法计数。
