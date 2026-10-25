# 轨迹长度偏差研究

本目录记录 2026-10-24 对正式矩阵当前结果的只读统计，以及一个隔离的新版本候选。
它不注册模型、不改变 `53 profiles × 13 sources × 3 seeds = 2,067` 的冻结矩阵，也不继承旧结果的身份。

## 2026-10-24 的 Agent 新证据

[observed_999_subject_clusters.json](observed_999_subject_clusters.json) 是阿里云 WSS 连接到
`ps` Agent 后，在其实时 run-states 上读取的只读探索性统计；不使用旧 SSH 快照。
当时 999 completed、2 failed、1 running。999 个主 test cell 中 868 个 PLR<1（86.89%），
但混合样本的算术均值为 1.159，说明少数扩张结果很大；不能只报告发生比例。

进一步排除二维、经典方法、test-only 迁移，只看人体 XYZ 可训练神经配置：465 个 cell，
446 个 PLR<1（95.91%），中位数 0.911；与下方旧 981 汇总口径不同，不直接作时间趋势比较。
94 个 profile/source 组合具有全部三个 seed；折叠重复 seed/profile 后有 266 条录制、92 个既有 group_id。
group_id 按数据集既有身份使用，不保证跨来源都是等价的独立受试者。

限定 T200/reference/平均速度的子组，再按来源等权、来源内受试者分组 bootstrap（2,000 次）：

| 指标 | 条件性均值估计 | 95% 分组 bootstrap 区间 |
| --- | ---: | --- |
| PLR | 0.9073 | [0.8906, 0.9232] |
| plr_oracle | 0.9163 | [0.8990, 0.9310] |
| speed_ratio | 0.9262 | [0.9118, 0.9392] |

这支持“已观察人体配置存在条件性幅值/长度低估”，尚不能归结为纯协议平滑。
区间条件于已观察算法/来源，不覆盖算法总体、不证明因果，也不代替逐结果全链验收。
TinyOdom/IDOL、VeloBins/TLIO、NIO-Lie-Events/RNIN 等扩张反例全部保留。
去除 oracle ATE 大于模型 ATE 的片段及多有效段录制的敏感性分析也随原件保存；该筛选本身会改变来源构成。

`audit_recordings.py` 只读 completed 的 metrics.json/sequences.csv，验证身份和逐录制分解恒等式，
记录输入哈希；3 项 Agent 单测验证重复 seed 不膨胀受试者数、subject grouping 和坏结果原子拒收。
`research_supervisor.py` 已在 Agent 新研究目录运行，每 5 小时刷新；最多 34 次，或矩阵不再有 qualified/running 时结束。
Issue reporter 读取带独立时间戳的 research-status.json，不把上一轮研究摘要当成当下全矩阵计数。

解决路线先比较 identity、validation-only 标量 gain、等参数读出和显式新模块；
ATE、10s RTE、方向及 P95 风险必须一起改善。新的独立网络及工程证据见
[Coherent Spectral IO](../coherent_spectral_io_20261024/README.md)。

## 原因分解：2026-10-25 Agent 只读结果

用户提出的核心问题是“为什么会缩短”，而不是强行把长度乘回 1。
[cause_decomposition.py](cause_decomposition.py) 在 Agent 上读取 519 个已完成的人体 XYZ 可训练单元，
并在其中 220 个 T200、`avg_velocity`、`reference_oracle` 单元上分解参考协议、模型幅值、方向误差和训练/测试模式差异。
完整小收据见 [cause_observation_20261025.json](cause_observation_20261025.json)，远端执行叶和源码绑定写在其中。

| 可能原因 | 远端观察 | 当前判断 |
|---|---|---|
| 1 秒参考采样/Oracle 平滑 | `oracle_ratio` 中位数 **0.9808** | 只解释约 2% 的中位差异，不能解释主要缩短 |
| 模型输出幅值低估 | `plr_oracle` 中位数 **0.9164**；`speed_ratio` 中位数 **0.9246** | 主要现象在模型输出中已经存在 |
| 纯方向错误导致向量相消 | 方向误差小于 20° 的 192 个单元中，`speed_ratio` 中位数 **0.9324**，speed/along bias 均为负 | 存在独立的径向幅值偏差；高方向误差尾部会进一步放大缩短，14 个单元中位 `speed_ratio=0.0190` |
| Dropout 或 train/eval 模式 | train/test `speed_ratio` 差的中位数约 **3.4×10⁻⁶** | 不是当前主因；仍需按模型族检查异常单元 |

`speed_ratio` 与方向误差的相关系数为 **−0.946**，说明方向不确定性会放大幅值塌缩；但低方向误差分层仍有约 6.8% 的中位速度低估，因此不能把全部现象归结为方向相消。`plr_oracle` 与 `speed_ratio` 的相关系数为 **0.998**，支持“轨迹变短主要来自输出速度幅值变小”。

这些数字是已观察算法/来源上的条件性证据，不是随机化因果效应。目标和损失本身在正式矩阵中与架构绑定，不能直接比较后称为原因。下一步按 [cause experiment plan](cause_experiment_plan.json) 做固定架构、固定数据、固定 seed 的干预：向量 MSE 对比幅值+方向损失，`avg_velocity` 对比 `velocity_at_end`，并保留 validation-only 标量 gain 作为控制。只有 ATE、10 秒 RTE 和速度/方向误差共同不恶化，才接受对应机制。

## 历史快照：981 completed（保留原始口径）

本节及下方 SAGE-IO 是本轮较早的记录；当前统计与区间以上方 999-cell 分组分析及
Agent 定时刷新为准。SAGE-IO 保留为未执行草稿，当前推进的独立候选为 Coherent Spectral IO。

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

## 历史 SAGE-IO 草稿（未执行）

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
