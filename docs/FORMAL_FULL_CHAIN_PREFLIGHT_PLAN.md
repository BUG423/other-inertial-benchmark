# 正式结果全链只读验收：下一阶段编排

> 2026-10-14。状态：边界和执行顺序已完成静态审查，完整 preflight **尚未执行**。
> [首份 partial report](FORMAL_PARTIAL_RESULTS_HANDOFF.md)只验收测试小结果，不能代替本阶段。

## 本阶段要交付什么

交付逐 cell 的只读证据包，证明论文结果来自对应冻结实验、合法 validation 选模、确切 checkpoint 和预注册迁移来源。
只接受不变的原始证据；缺失、锁忙、超预算、证据变化与科学矛盾分别记录，不修改历史 state 或指标，不自动重跑。

必须区分两个 source 身份：实验为 **425 files / `1897c9cc…0949`**，
独立验证器为 **247 files / `4c482569…8b0b0`**。验证器版本不能替换实验 source。
当前 53 profiles × 13 sources × 3 seeds 的 2,067 行保持不变。

## 已做的 validation 元信息诊断，不是验收

限定原 coverage 中 completed 的只读投影观察到 656 个 completed；
其中 280 个 `trainable=true` 的训练/标定单元，其 canonical `train/metrics.json` 均为
`mode=train, split=val, num_sequences>0`，`train/sequences.csv` 均有非空、未 skipped 的行。
本次 `review_candidates=[]`，没有找到空 validation 的具体单元；历史摘要疑点仍未定位，不能写成已确认 invalid。

诊断实际读取 62,634,617 bytes，用时 1.059 秒；不读取 checkpoint、HDF5 或 prediction，不输出分数，不导入框架，不写远端文件。
probe 未记录观测 UTC，不能补造。`18:23:45 CST` 是证据保存记录时间，**不是采样时间**。

私有证据位于 `.local/matrix-chain-preflight/`：

| 文件 | SHA256 |
| :--- | :--- |
| readonly-metadata-projection-probe.py | `7813e44b8341f1ff946c27497b5511d9f892a236527a8764e6a7069a12c59206` |
| readonly-metadata-projection-output.json | `b9b5a792e9042e7ede6c6231850ca445cef1e93c7bf5409c2e71541dbe34ec7a` |
| readonly-metadata-projection-provenance.json | `bef51cc40ddecdc901496b405e7d19992e8ea373e404f3f9d3eb37d0a1875b8e` |
| readonly-frozen-source-excerpt-output.json | `207e3e5632d98a007d7ce5cdb6a45dd1a3588b1c8986890627db06a4c60fda2c` |

限制：coverage/policy 前后 hash 和 token 不变；cell state 不是最终一步复核，逐 cell 原始投影列表也未保存，
不能从输出的 projection digest 独立重建。租约取得与 fd 关闭由实际执行分支和 exit 0 支持，没有额外自由锁探针。
本次没有核对精确 val ID 集、协议、聚合、best 选模或 checkpoint，不能据“非空”发全链 GO。

## 为什么不能直接调用现有 gate

- 新 `verify_completed_state` 先哈希 checkpoint，只有流式实现，没有单文件/总量/时长上限；它不是轻量 stat 探针。
- 它验收非 official test splits，不包括训练目录内的 canonical full-sequence val；
  val 的 `mode=train` 不能直接当作普通 test 的 `mode=val` 契约处理。
- `verify_completed_state` 自身不完成整个 run identity 对比；需重建旧 source、policy、seed 和原 overrides 的确切请求。
- legacy state 没有新 `result_artifacts` 很正常。只读现场重验通过，不代表历史封存 binding 原本存在，不能回填收据。
- partial capture 的原 state 保留原绝对路径。直接对复制目录调用 gate 会绕过 capture 重读 live 文件，不能声称不可变输入验收。
- frozen Trainer 在 best 缺失时有 last fallback；val 非空不足以证明正常完成有限 fitness 的 best 选模。

可复用接口为 [benchmark.py](../src/inertial_benchmark/engine/benchmark.py) 的
`scientific_run_identity`、`_verify_transfer_provenance` 和严格结果契约，以及
[full_matrix.py](../tools/ops_v1/full_matrix.py) 的 `verify_current_completed` 身份对比原则。
不得调用执行、dispatch、`run_item` 或 `retain_completed_checkpoint`；后者会写 state，并可能删除 last。

## 分阶段任务与责任

| 顺序 | 执行/复审角色 | 交付与通过条件 |
| :--- | :--- | :--- |
| 1 | Root / Identity Reviewer | 固定 request/schema、旧 source/policy/seed/coverage、全部原 overrides，拒绝科学字段覆盖 |
| 2 | Inventory Agent / Root | completed 和 completed origin 的依赖清单；只 stat 唯一 checkpoint 的大小、路径、锁和总读取预算，不读内容 |
| 3 | Capture Agent / Evidence Reviewer | 原字节 state/identity/taskview/val/test 捕获、路径映射、逐 cell 起止时间和 manifest；解析与 hash 共用受控字节视图 |
| 4 | Checkpoint Agent / Resource Reviewer | 预算内 single-fd 流式 SHA、regular/namespace/token 验证；相同 origin checkpoint 去重复；不复制或反序列化权重 |
| 5 | Transfer Reviewer / Root | 源训练 validation 与 checkpoint 通过，目标同 profile/track/seed/config/input；严格来源对象、SHA、路径及跨域标志闭合 |
| 6 | Independent Reviewer / Root | 每 cell accepted/deferred/invalid 和原因，原始证据及输出根 hash 可复核；不变 guards 和实际 lease release；未覆盖能力明确列出 |

validation 必须单独检查已准入 split/窗口、精确序列 ID、模式/来源/seed/维度、聚合、weights 指向以及 best/epoch/fitness 证据。
静态经典标定和神经训练使用各自真实生命周期，不虚构 optimizer updates。
test 继续核对严格 split 适用性、顺序、N/A、数量/维度/ID/聚合/hash。
仅 hash 成功不证明权重 tensor 有限、模型可重载或收敛；如需要，另设有预算的内容验收阶段。

## 隔离、锁和预算

- 不读 running 的 checkpoint。仅尝试既有 `run-locks/<ID>.lock` 的 `LOCK_SH|LOCK_NB`，不创建锁；
  源/目标按固定 ID 顺序获取，与 worker 整个 cell 的 exclusive lock 配合。锁忙记 deferred，不等待或停止训练。
- 每个路径逐组件 `O_NOFOLLOW`；读前后 dev/inode/mode/size/mtime/ctime/nlink 与 namespace 一致；
  **cell state 最后复核**，至多有限次重新采样。观察区间不能称为全局瞬时快照。
- private read-view adapter 显式绑定捕获路径和 bounded checkpoint context，不改原 row/identity，
  不做全局 monkeypatch 或热部署；旧 source 与新验证器差异及桥接源码必须独立审查、hash 固定。
- 候选预算沿用 supplemental 单 CPU、nice +10、CUDA 隐藏、线程 1、cgroup 75% 门槛、6 GiB reserve、
  30 GiB 磁盘 reserve、1,200 秒工作 deadline；小结果总 256 MiB、单 metrics 4 MiB、CSV 8 MiB。
- checkpoint 单文件 2 GiB、每批累计 8 GiB、1 MiB 流式块仅是**待 inventory 后确定的预算候选**，尚未测量或冻结。
  超限、锁忙和变化属于 deferred/pending，不是科学 invalid；不得无上限扫描全部权重。

当前可继续准备 request、依赖清单、read-view 设计和 stat inventory。
只有真实不变的证据才发逐 cell GO；CTIN 修配方、invalid 的处置、新训练或任何范围变更仍由 Owner 决定。
