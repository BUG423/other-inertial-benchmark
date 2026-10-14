# 交付与执行总览

> 状态基线：`dev/v1@8b7ce3f`。本文建立于 2026-10-14，服务于当前冻结实验、结果验收和论文交付。
> 它不是实时监控面板，也不替代[冻结共同协议](protocols/common-benchmark-v1.json)、
> [数据/实验状态](DEV_V1_INTEGRATION.md)或逐项机器收据。

## 1. 如何阅读本文

本文强制区分三类内容：

- **事实**：已有 Git、不可变清单、服务器收据或只读审计支持；动态事实必须带观测时间。
- **计划**：尚待执行的工作和依赖，不代表已经开始或通过。
- **待决策**：需要项目 Owner 或 Legal 明确批准，执行人员不得自行扩大范围。

任何 `qualified`、`running`、工程 smoke、CPU/CUDA 准入、候选数据或研究 staging 都不等于正式结果。
`completed` 只是执行状态；只有结果文件通过身份、非空、协议、split、维度和哈希验收后，才是可统计结果。

## 2. 第一性原理：最终要交付什么

最终目的不是“把很多任务跑完”，也不是得到一张数值最多的排行榜，而是交付一条可复核的科学证据链：

1. 在来源、输入权限、数据划分、随机种子、训练预算和评价协议明确且冻结的前提下，公平比较惯性定位方法；
2. 让每个论文数字能回溯到确切源码、数据指纹、配置、checkpoint、split 和原始小型结果文件；
3. 以真实 XYZ 为主结果，从同一次 XYZ 预测派生 XY 切片，另行保留原生二维参考协议；
4. 对缺失能力、N/A、失败、跨域迁移和参考特权如实分层，不补零、不造标签、不把不适用当零分；
5. 在完整 baseline 之后检验“轨迹缩短”的可证伪机制与修正是否同时改善定位误差，而不是只让长度比接近 1；
6. 最终形成可复现的代码/协议/结果包、论文表图、限制说明和经许可审查的发布材料。

因此，训练只是中间步骤。没有闭合的结果契约、三 seed 统计、反例和许可边界，即使所有 GPU 进程退出也不构成交付完成。

## 3. 五层交付物与 Definition of Done

| 层 | 必须交付 | Definition of Done（全部满足才完成） |
| :--- | :--- | :--- |
| **L1 冻结治理与身份层** | source、policy、coverage、seed、native 数据与 split 的不可变身份；范围变更日志 | 2,067 行逐一具有唯一身份；425 文件冻结 source、13 个 native 来源、三 seed、policy 与 coverage 哈希可复核；任何修复或扩展使用新版本/新清单，不静默改写旧证据；候选数据和 staging 不进入正式范围 |
| **L2 数据、方法与执行层** | 53 个冻结 profile 在 13 来源上的 taskview、输入权限、训练/标定/迁移和 checkpoint 生命周期 | 2,019 个支持单元均有终态和可复核 checkpoint/result 身份；48 个不支持单元保留精确原因；test-only checkpoint 来源、同 seed/track/config 约束成立；无 running、伪训练或未解释失败被写成完成 |
| **L3 结果与证据层** | 每个适用 split 的 `metrics.json`、`sequences.csv`、availability/NA 台账、内容哈希与聚合清单 | 只收录已完成且身份一致、`num_sequences>0` 的结果；running/failed/残留目录不入表；同一 row/split 不重复；N/A 与 invalid 单列并绑定原因和已有小文件哈希；汇总输入、生成表格和 manifest 构成可独立复核的证据包 |
| **L4 统计与科学结论层** | XYZ 主表、同预测 XY 切片、二维参考表、跨域表、三 seed 统计、轨迹缩短诊断与反例 | 相同统计层协议一致；按 dataset/split/model/input track/dimension 分层；逐序列与 group-aware 统计、seed 离散度/置信区间、样本数和 NA 完整；跨域不混入同域榜；`claims_proven` 仅在预注册判据真实满足后更新 |
| **L5 论文与可复现发布层** | 论文正文、附录、机器表图、复现实验说明、来源/许可清单、限制与失败证据 | 所有正文数字由固定脚本从 L3/L4 生成；代码、数据、权重和图表的可分发性经 Owner/Legal 批准；论文方法计数与正式矩阵范围不混淆；提交包通过独立复核，RA-L 或其他 venue 的实际提交另有明确记录 |

依赖关系是：`L1 → L2 → L3 → L4 → L5`。许可盘点可与 L2/L3 并行，但未获批准时会阻断 L5。
后续方法或数据扩展必须创建下一版 L1，不能在当前执行中插队。

## 4. 当前正式范围：不可变约束

以下为**事实**，也是当前正式结果的硬边界：

- 冻结矩阵为 **53 profiles × 13 个已验收独立来源 × 3 seeds = 2,067 cells**；冻结源码为
  425 个受管文件，source manifest SHA256 为
  `1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949`。
- taskview 已核验全部 2,067 行：**2,019 passed / 48 unsupported / 0 pending qualification**。
  48 行继续留在 coverage 中，不能删除或用 reference orientation 补成支持。
- 正式随机种子固定为 **1735231569、843628571、951150512**。所有可比较方法共用这三个一次抽取并持久化的 seed；工程 seed 不算正式重复。
- 主协议是原生 **XYZ**。`xy_*` 必须由同一个 XYZ prediction 的 `[:, :2]` 计算；不得重跑二维模型后称为“同预测 XY”。原论文二维输出属于独立 reference-XY track。
- 缺预测 Z 不得用 0 或参考真值填充；缺预测姿态时 rotation metric 为 N/A；reference/oracle 输入和 deployable 输入必须分 track。
- Blackbird/UZH-FPV 的 test-only 单元使用同 profile/track/seed 的 EuRoC 训练 checkpoint；GrandTour/SubT-MRS 使用 RIDI 来源。人体到机器人仅是跨域诊断，不进入人体同域榜。
- 冻结共同配方、科学例外和证据哈希以[共同协议](protocols/common-benchmark-v1.json)为准；公开指标语义以[指标文档](METRICS.md)为准。

以下均为**正式范围之外的事实**：

- 当前代码库的 57 profiles 包含 M2EIT/DiffusionIMU 新四配置；它们已有独立 CPU 证据，但未进入冻结 53，也未开始其正式矩阵训练。
- KITTI raw 100 Hz 与派生 200 Hz 是已审计候选；它们没有注册进正式 13 来源，不得改变 2,067/2,019/48。
- PedLocData 正式范围仅含已验收 Demo 47；未完整取得的主包 partial 不计入正式数据。
- MotionTransformer、X-IONet、AbolDeepIO、AI-IMU-DR 等研究实现/核心验收不自动增加正式方法数。

## 5. 动态训练快照

**事实，观测时间 2026-10-14 18:36:25 CST（UTC+08:00）**：

| 状态 | 单元数 | 解释 |
| :--- | ---: | :--- |
| completed | 658 | 执行状态为完成；不能直接等同 658 个全链验收或可发表结果 |
| running | 4 | 瞬时状态，下一次读取可变化 |
| qualified | 1,356 | 已通过 taskview，但尚未进入终态 |
| failed | 1 | CTIN/TLIO/seed 1735231569 的数值/科学失败；保留证据，未自动重试 |
| unsupported_taskview | 48 | 冻结的不支持输入单元，保留为明确 N/A/unsupported |
| **合计** | **2,067** | 与冻结 coverage 一致 |

这个快照只回答“当时状态文件写了什么”。它不证明矩阵完成、模型收敛、结果有效或服务器随后仍保持相同计数。
当前 completed 按 seed 为 282 / 189 / 187；按执行类型为 275 个训练、7 个校准、376 个仅评测；
其中 652 个标记为 primary XYZ。冻结 source/policy/seed 身份无错误。
原字节 live-status 镜像 SHA256：`21dd6341b4f3e00113809ae539d7c30be0bea75709c21a7cb31eb72ffd253713`。

**独立 validation 元信息诊断**（不是上述状态快照）：
历史摘要曾提示空/无效 validation 风险，但尚未定位到确切单元和原始证据；不能将其写成已确认 invalid。
本轮独立投影观察 656 completed，其中 280 个训练/标定单元的最终验证结果均非空；probe 未记录采样 UTC，不能补造。
这不是完整 validation 验收。
因此仍不能把 `completed` 数直接写入论文证据完整覆盖数。
L3 matrix-report 验收测试小结果，validation、checkpoint 与 transfer 仍须全链只读验收；两者不能互相替代。

**首份 partial 结果验收事实**：17:58:50–17:59:01 CST 的独立采样保留全部 2,067 cells；
649 个 completed 至少各有一个 accepted 测试 split，共 **1,251 accepted 测试 splits / 23,788 逐序列行 / 0 rejected**。
569 个汇总 strata 中 341 有三 seed、228 仅一 seed；18:33 CST 的只读审计确认 341 的序列集合与 group ID 一致。
这是 seed/sequence 覆盖验收，不是全部论文统计或方法间配对；全部 accepted 属于 reference_oracle。
这不覆盖 deployable 主表、train/val、checkpoint 或 transfer 全链。
精确收据、可用性台账、时间区间和限制见[partial report 交接](FORMAL_PARTIAL_RESULTS_HANDOFF.md)。

## 6. 当前 Git 里程碑

以下为**已经提交并位于 `dev/v1` 的事实**；短测试或 staging 的能力边界仍以各提交说明和收据为准：

| 里程碑 | 提交 | 已形成的基础 |
| :--- | :--- | :--- |
| XYZ 主协议与同预测 XY 诊断 | `2c1f57f` | 统一主三维评价，二维派生与原二维参考分离 |
| 13 来源、指标和动态矩阵编排 | `8ed4857` | 扩展 native、评价与 matrix orchestration |
| 随机种子与全矩阵 readiness 冻结 | `9ea9b13` | 一次抽取三 seed、正式范围与准入证据 |
| 新方法隔离验收 | `50cfed9`、`55f620e` | M2EIT/DiffusionIMU CPU 证据与公开规格；未进入冻结矩阵 |
| MotionTransformer 阶段协议 | `d71201e`、`f0f1aa6`、`3b4ca27` | 标签隔离、Phase A 与 shared Trainer 配对协议；仍属研究扩展 |
| AI-IMU-DR 分阶段移植 | `e5d66e7`、`93c2acc`、`05087ec` | 固定来源核心、全序列输出与 pair-cache；尚非正式矩阵方法 |
| 评估别名复用 | `e795f9d` | exact CPU 等价复用证据；不等于正式 GPU 加速声明 |
| KITTI 候选链 | `b25b5d9`、`92edf25` | raw/derived converter 与候选证据；保持正式 13 来源不变 |
| 完成态与矩阵报告加固 | `8c1d818` | completed-state fail-closed 验收、matrix-aware report 与 178 项组合回归；尚未热部署到冻结训练源 |
| 跨领域 baseline 隔离研究 | `4fdcb0c` | ModernTCN/PatchTST/S4D、108 项 AutoDL CPU 验收和阶段交接；未注册、未进正式矩阵 |
| 首份只读 partial report 交接 | `8b7ce3f` | 2,067-cell 台账与 1,251 测试 splits/23,788 行小结果证据；不代表 checkpoint 全链或矩阵完成 |

completed-state/result-report 加固已通过本地独立审查；是否已经提交以 Git 记录为准，且在产生服务器 preflight receipt 前，
**不能写成已部署到当前冻结矩阵**。

## 7. 阶段化 backlog、依赖与责任角色

### P0：提交或继续消费结果前必须完成

| 顺序 | 任务 | 依赖 | 责任角色 | 完成判据 |
| :--- | :--- | :--- | :--- | :--- |
| P0.1 | 保护冻结矩阵和运行资源 | L1 | 执行/运维 Owner | source、coverage、policy、seed、native guards 无漂移；监控不占用唯一训练锁；异常保留原收据 |
| P0.2 | 完成态结果闸门 | P0.1 | 引擎 Owner + 独立审计 | 适用 split 非空、身份/维度/数量/哈希一致；N/A 不创建伪 eval；篡改或缺失使单元非成功；正负测试通过 |
| P0.3 | 安全矩阵结果聚合 | P0.2 | 结果工具 Owner | 不收 running/failed；拒绝路径穿越、重复 row/split、state 改写科学身份与跨 seed 协议混合；invalid 进入台账而不污染结果；原 `ipb report` 回归通过 |
| P0.4 | 首份可复核 partial snapshot（已通过，18:00 CST） | P0.3 | 结果工具 Owner + 统计 Owner | 首份全范围台账、1,251 accepted 测试 splits、原字节哈希与分层汇总闭合；仍不完整、不含 checkpoint/val 全链，不当作最终结论 |
| P0.5 | 发布许可风险清单 | 可并行 | 来源/许可 Owner + Legal | 对仓库代码、第三方源码、数据、权重、图表逐项给出可分发/仅本地/需授权/禁止四态，未决项有负责人和截止条件 |

### P1：P0 通过后推进的正式交付

| 阶段 | 任务 | 依赖 | 责任角色 | 完成判据 |
| :--- | :--- | :--- | :--- | :--- |
| P1.1 | 完成或明确终止冻结矩阵 | P0.1/P0.2 | 执行/运维 Owner | 2,019 支持单元均为可验收 completed 或经 Owner 批准的明确终止；48 unsupported 保持原原因；无失联 running |
| P1.2 | 论文级表格/图机器化 | P0.3、足够终态覆盖 | 统计/制图 Owner | XYZ 主表、同预测 XY、二维 reference、同域/跨域、三 seed、NA、效率与失败表可由固定命令重建，表图携带输入 manifest |
| P1.3 | 轨迹缩短研究 | P1.2 baseline | 科学研究 Owner | 按预注册依次检验 oracle/采样、幅值/方向、时基、参考来源、模式、loss phase 和域偏移；train/val 选方案、test 一次冻结验证；报告反例与多重比较 |
| P1.4 | 文档真值同步 | P0/P1 实际证据 | 文档/发布 Owner | README、CLI、Handoff、指标、数据/算法卡和运行说明不再引用过时快照；动态进度与冻结协议分开 |
| P1.5 | 论文和复现包 | P1.2、P1.3、P0.5 | Paper Owner + Reproducibility Reviewer | 主张—表图—原始结果—代码/配置双向可追溯；限制、N/A、跨域和许可完整；提交前独立复核通过 |

### P2：不得回填当前冻结矩阵的后续版本

- 用户更新的跨域执行目标为 **CVPR 2026 主会十个独立开源算法**，全部排在现有实验及先前排队实验之后。
  现在仅做来源核验、接口/源码静态准备；本批 CPU/GPU 模型运行均后置。旧三架构工程成果保留但不计入十方法，
  旧三方法 pilot 草案不再作为当前执行清单。完整先后约束见[跨域计划](CROSS_DOMAIN_BASELINES_PLAN.md#当前范围与执行顺序)。
- 对“30 个独立论文方法”的口径和必要性作 Owner 决策；若继续扩展，单独建立 source/data/capacity/seed/version，不修改当前 2,067 行。
- 已有 ModernTCN、PatchTST、S4D 三种架构的隔离工程成果：
  2026-10-14 17:36 CST 完成独立服务器 CPU 科学/原生生命周期验收（108 passed）。
  它们不满足新要求的 CVPR 2026 来源条件，只保留研究参考，不列入本次十方法执行范围；
  仍无 CUDA 容量或精度/三 seed 结果，新增正式方法为 0。
- 候选数据或新增方法只有在来源、许可、真实输入/标签、完整 CPU 生命周期、CUDA 容量和科学协议全部通过后，才可进入下一版 inventory。
- 下一版矩阵应先完成成本/价值评审；不能仅因已有候选 converter、网络核心或短 smoke 就启动付费全矩阵。

## 8. 已知阻塞与证据缺口

### 8.1 P0 结果链加固状态

**事实（当前版本实现，尚未部署到远端冻结矩阵）**：matrix-aware report 与 completed-state gate 已完成作者测试、反例测试和独立复审，
P0/P1 组合回归为 **178 passed / 0 failed**（另有 4 条既有 multiprocessing fork 弃用警告）。两条实现均获独立 GO：

- 只有 completed 且身份、split、维度、逐序列 ID、聚合统计和哈希闭合的结果进入正式表；空/矛盾 completed 进入 invalid ledger，
  running/failed 残留只进入 ignored hash 证据；
- N/A 必须由冻结 taskview 与 native split 的缺失/空状态共同证明；路径穿越、symlink、重复 ID/header、非有限数、state 科学字段覆盖、
  跨 seed 协议混合和 transfer provenance 不一致均 fail closed；
- checkpoint 使用流式快照，结果/identity/split 的解析与哈希使用同一受控字节视图；report 持久化 manifest 根哈希；
- 现有旧 completed state 没有 `result_artifacts` 时仍可走严格的只读现场重验，不自动回填或改写历史 state。

独立服务器分析 snapshot 已生成并验收[首份 partial report](FORMAL_PARTIAL_RESULTS_HANDOFF.md)，
其测试小结果链已闭合，不表示远端所有 checkpoint/validation/transfer 已验收。新实现未热部署到当前运行源码；
部署前仍须完成旧 checkpoint/result 全链只读 preflight；若实际查出 invalid completed，其处置或重跑由 Owner 决定。
下一阶段的 identity、canonical val、captured read-view、checkpoint 预算与 transfer 依赖编排见
[全链 preflight 计划](FORMAL_FULL_CHAIN_PREFLIGHT_PLAN.md)；该完整验收尚未执行。

### 8.2 CTIN/TLIO 正式失败

**事实**：`ctin_3d / tlio / seed 1735231569` 在第 14 轮 batch 47 的梯度范数变为非有限值。
loss 在此前从约 -7.8 单调发散至约 `-1.02e18`；checkpoint 张量仍有限，但 `u_c` 已降至约 -20.6，
optimizer 二阶矩达到约 `3.98e36`。输入/taskview/数据指纹完整，GPU、内存、磁盘和 OOM 计数正常；
根因与 CTIN 的无界 uncertainty-weighted multitask loss 一致，属于算法/数值科学失败，而非资源瞬态。

在 Owner 决策前，该 cell 必须保持 failed，不能自动 resume 后写成普通成功。相同 frozen identity 的隔离复现需 Owner 明确授权；
loss clamp、权重、学习率、grad clip、optimizer reset 或从 best 重训都会改变科学配方，必须使用新 recipe/profile identity，
并评估 CTIN 全部相关 rows 是否需要一致性重跑。

### 8.3 距离 30 个方法仍缺 7 个实际准入

**事实**：当前库统计为 23 个独立论文方法；“到 30”还差 7 个。候选池为 MotionTransformer、X-IONet、
AbolDeepIO、A2DIO、D2IO、AI-IMU-DR、RINS-W。它们分别存在完整 NIO 接口、真实领域数据、论文全文/合法来源、
方向可识别目标、滤波/外参、paper-source 差异或许可等缺口。已有的核心/staging/短 CPU 证据不是 7 个已准入方法。

这 7 个是扩展候选，不是当前论文必须强行纳入的正式单元。若目标改成“30 个已发表方法”，当前 21 个已发表方法到 30
实际差 9 个；Owner 必须先冻结计数口径。

### 8.4 轨迹缩短尚无可发表机制结论

**事实**：`.local/plans/trajectory-shortening-preregistration.json` 的 `claims_proven` 仍为空数组。
现有 partial PLR 只能描述已完成子集，不能证明所有模型普遍缩短、不能归因，也不能证明长度校正改善 ATE/RPE。
完整执行和配对三 seed baseline 是 P1.3 的前置条件。

### 8.5 论文工作尚未启动

**事实**：RA-L 路线和证据需求已有规划，但正文、图表定稿、复现附录、作者审批和实际投稿尚未启动。
“目标 RA-L”不是“正在投稿”，更不是录用承诺。

### 8.6 许可与再分发风险

**事实**：仓库当前采用 CC BY-NC-ND 4.0；第三方方法存在 GPL/AGPL、无 LICENSE、paper-only 重实现、
权重无明确再分发授权等多种状态；数据集和候选下载也保留各自使用条款。论文可引用不等于代码、权重、raw 或派生数据可再分发。
在 Legal 决定之前，只能发布已明确获准的材料；DeepILS 等无明确权重许可的资产继续仅作本地核对。

### 8.7 文档状态有时间漂移

**事实**：[Handoff](HANDOFF.md) 顶部仍是 2026-09-24、705 tests 和旧 `batchA` 语境；若干状态段落记录的是历史服务器快照。
它们可作为历史证据，但不能代表 2026-10-14 的实时执行。冻结协议本身不是 stale，不应用动态训练数覆盖；文档同步应在
P0/P1 事实落定后统一完成。

## 9. 必须由 Owner 或 Legal 决定的事项

| 决策 | 决策者 | 决定前禁止的行为 |
| :--- | :--- | :--- |
| 当前论文是等待完整 2,019 支持单元，还是采用预先定义、非结果导向的证据完整子集 | Project/Paper Owner | 不得按测试结果好坏挑 cell、seed、数据集或方法 |
| 方法目标是当前冻结 21、当前库 23、主表约 10，还是下一版 30 个独立方法 | Project/Paper Owner | 不得把配置变体、staging 核心或候选凑作独立论文方法 |
| 若全链验收发现 invalid completed：仅作失败证据、新身份重跑，或从明确版本重建 | Experiment Owner | 不得把未定位风险当确证，不得手改指标、空 CSV、state 或覆盖旧失败收据 |
| CTIN/TLIO 失败是保留为正式失败、隔离复现，还是以新 recipe 修正 loss 后重跑 CTIN 全体 | Experiment/Scientific Owner | 不得自动 resume；不得在旧 identity 下改 loss、LR、grad clip 或 optimizer |
| 是否投入剩余 GPU 时间以及何时停止当前矩阵 | Project Owner | 不得由执行人员根据中间排名自行提前停止或扩容 |
| 下一版是否纳入 M2EIT/DiffusionIMU、KITTI 或 7 个候选方法 | Project Owner + Scientific Owner | 不得改变当前 source/coverage/seed 身份 |
| 仓库 CC BY-NC-ND 与第三方 GPL/AGPL/无许可源码、数据、权重、图表如何组合和分发 | Legal + Copyright Owner | 不得公开未审资产或把论文引用当作再分发授权 |
| RA-L 的作者、主张范围、投稿时间与 go/no-go | Paper Owner + 全体作者 | 不得宣称已启动投稿、已提交或可转投某会议 |

## 10. 每次状态汇报的最小模板

后续汇报只更新动态事实，不改写冻结事实：

1. `observed_at`（UTC 与 CST）、读取来源与 receipt/hash；
2. 2,067 行状态分布，以及 2,019 支持/48 unsupported 是否守恒；
3. completed 中通过 L3 验收的数量、invalid 数量与原因，不直接复用 state 数；
4. completed training/calibration/eval-only、三 seed 和 dataset/profile 覆盖；
5. 资源、失败、重试、新增证据及 frozen guards；
6. 本周期完成的 DoD、下一依赖、Owner/Legal 阻塞；
7. 明确写出“完整矩阵未完成”和“`claims_proven` 是否仍为空”。

只有在五层 DoD 全部关闭后，项目状态才可从“执行中”改为“可提交/可发布”。
