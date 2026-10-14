# 首份只读正式矩阵 partial report

## 结论与统计单位

2026-10-14 的首份独立小结果验收已通过。
输入采样区间为 **17:58:50.742–17:59:01.219 CST**
（UTC `09:58:50.742547–09:59:01.219214`），报告在 **18:00:03 CST** 生成。
这是逐行一致采样的区间，不是整个矩阵同一瞬间的事务快照。

| 单位 | 已观察或验收的数量 | 不能等同于 |
| :--- | ---: | :--- |
| 保留的 matrix cells | 2,067 | 支持/训练成功数 |
| 采样 completed cells | 649 | checkpoint/验证/迁移全链已验收数 |
| accepted 测试评测 splits | 1,251 | 1,251 个模型、训练单元或独立数据集 |
| accepted 逐序列行 | 23,788 | 23,788 条独立原生录制；含跨方法/seed/split 重复 |
| 汇总 strata | 569 | 全部已完成三 seed 的比较 |

649 个 completed cells 均至少有一个 accepted 测试 split；XYZ 643、原生二维参考 6。
它们按 seed 为 273 / 189 / 187。测试范围 **0 invalid / 0 rejected**，
但本工具不消费 train/val 结果，也不检查 checkpoint 或 transfer provenance；
所以不能据此取消历史空 validation、checkpoint 身份或 completed-state 全链验收风险。

569 个汇总 strata 中，**341 有三 seed、228 只有一个 seed**，没有两 seed stratum。
这些是 dataset/split/model/input track/dimension/domain/training source/privilege 分层，
不是 341 个独立方法或完整实验；三 seed 数量齐全尚不证明逐序列 ID 和 group ID 已精确配对。
全部 accepted 结果属于 **`reference_oracle`**，不能称为 deployable 纯 IMU 主表。
partial 覆盖不可用于最终全域排名、挑选种子或提前停止模型。

## 可用性台账

采样的 2,067 cells 状态为：649 completed / 1,365 qualified / 4 running /
1 failed / 48 unsupported_taskview。全部 ID 原样保留，没有按结果删模型或 seed。

`availability.csv` 的 **4,728 条 split 台账**分布如下：

| availability | 数量 | 处理 |
| :--- | ---: | :--- |
| available | 1,251 | completed 且小结果/身份/split/维度/聚合统计/哈希闭合，进入逐序列表 |
| pending | 3,263 | 未完成，不进入结果表 |
| not_applicable | 153 | native split 缺失/空的证据与 frozen taskview 一致，不造零分 |
| unsupported | 48 | 缺所需输入权限，保留原因与 ID |
| running | 12 | 正在运行单元的 split，不进入结果表 |
| failed | 1 | CTIN/TLIO/seed 1735231569，保留科学失败，不自动重试 |

N/A 与 cell 状态正交，不能与 completed/qualified 混成同一种计数。
其中 15 个 N/A 属于 completed，138 个属于 qualified；15 个 N/A 的残留结果只作为 ignored 哈希证据。
这 153 个 N/A 均属于 PedLocData，原字节空 split 已绑定哈希；不能据此把 completed 单元直接归为 invalid。

## 输入、工具与输出身份

运行中的正式源码未部署或修改。独立分析 snapshot 的 `reports.py` 与 `cli.py`
使用 `8c1d818` 的加固版本；跨域模型未注册、未用于这份正式报告。

| 项 | SHA256 |
| :--- | :--- |
| 正式实验 source manifest（425 files） | `1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949` |
| 独立分析 source manifest（247 files） | `4c482569ef8b3a90fb22d67f44c1fa18b3aa648af37dd7fc9e47d257aae8b0b0` |
| 原字节 coverage | `302225cac104c49a84f8bf122f7cefb21f03b4a8c44254b5e2baa2f0e1ac814f` |
| supervisor runner | `572a514e1629eb93593acb10b3d77ef4d2351f24e0793bdfab22a457045eb969` |
| request | `c910e79eb62d2cf00c5ea2413d59703021a24608f17bb0236e3a241866a8349a` |
| 原始 supervisor receipt | `6b0559a4f55b5cca6f70eeb43da99d3dd636017d2987e6b6924910a1e0c8e007` |
| capture-manifest.json | `b56477533704b64376c0d39b8224e62ed7373cc036db60497a9419e2bfad10f1` |
| child.json | `c0b4e65d4374bea5d12e722cff3c154efd548e8461657ebbf82bd8789e335213` |
| report.json | `fcb71a925aafe99d60a1e2ba285f8f4413e814d5502690df0d332faeb0dfd503` |
| artifacts-sha256.json（report manifest root） | `6e324b276af4bffc5c1c62d46502ac457ac44626adb0485933cd1bbd93af5097` |
| availability.csv | `8ae18c53f98cafa21d70ada1b742eb67a5811a936c6a545d85bbac501264a5ff` |
| summary.csv | `47e63a8a1463ef47f785a3e21409474ea19b65db5cd3990aa890220370414b6a` |
| per_sequence.csv | `6989f24ed7d61620eb68a8699c0712e0ae69c9e463166a8037b3d8cda0152de8` |

服务器 artifact 根目录（相对于实际部署项目根）：
`.local/reports/matrix-partial-20261014T095843Z/`。
`matrix-input/` 保存原字节 coverage/state/taskview/测试小结果；`report/` 保存上述五个报告文件。
runner/request 原件位于 `.local/research/matrix-report-preflight/572a514e1629-20261014T095843Z/`。
本机 `.local/reports/matrix-partial-20261014T095843Z/` 镜像 receipt、capture、child、report、
availability、summary 和 report manifest；85,844,899-byte 的逐序列表仍保存在服务器，
其内容哈希由 supervisor 与 report manifest 独立核对。私有数据、凭据和权重不入 Git。

原始 receipt 的 `captured_rows=2000` 是每百行刷新的中间进度字段，最后未刷新；
不修改旧收据补数。完整 **2,067** 行由 capture manifest 的唯一 row samples、
原字节 coverage、report 和 availability ID 集合共同证明。

## 运行隔离和实际资源

- Parent 使用 stdlib，只解析治理元数据；结果 JSON/CSV 在 capture 时作为 opaque bytes，随后由原 CLI 验收。
- 不读取 HDF5、checkpoint 或 prediction；不改旧 state，不重试训练，不替换冻结 coverage。
- 先拒绝 symlink、非普通文件、路径逃逸和科学字段冲突；每行 state 及结果变化至多采样三次，否则保留失败证据。
- 独占 supplemental CPU lease，单 CPU affinity、nice +10、数值库单线程；Torch 未导入，CUDA 未初始化。
- 总 wall **78.77 秒**，监测峰值 combined RSS **535,318,528 bytes**。
  6 GiB 为 250 ms RSS 采样终止阈值，不是内核硬限；总 deadline 1,200 秒。
- 输入副本为 154,018,645 bytes；预估 dataframe 2.22 GB，保留正式磁盘 reserve 30 GiB。
- 正式 425 source、分析 247 source、policy、capacity、seed、coverage、13 native manifest/split 前后哈希全部相同。
  supplementary lease 实际获取、关闭及释放均有最终收据。

CLI 等价入口为：

```bash
PYTHONPATH=<immutable-analysis>/src CUDA_VISIBLE_DEVICES='' \
<server-venv>/bin/python -m inertial_benchmark matrix-report \
  matrix=<captured-matrix-input> out=<fresh-report-directory>
```

复验仍须外层预算、租约、采样与 source guards，不能直接对正在写入的 live 目录复用此命令，
也不能省略 `out` 后把报告写回正式矩阵。

## 已关闭和未关闭的交付判据

本次关闭[交付总览](DELIVERY_EXECUTION_PLAN.md)的 **P0.4 首份可复核 partial snapshot**：
全范围 availability、accepted/ignored/N/A 台账、原始小结果哈希、逐序列数据、分层汇总和 report manifest 已闭合。
它不是 completed-state gate 已热部署，也不是 L2/L3/L4 全部完成或论文可发布。

下一步可直接推进：按冻结协议编排论文级 XYZ / 同预测 XY / 原二维参考 / 同域-跨域 / seed-coverage 表，
继续只读监控并在新 snapshot 中消费新增结果；三 seed 未齐的 strata 必须明确标记不完整。
`summary.csv` 目前只汇总八项 MAIN_METRICS，`xy_*` 保留于逐序列表；完整 XY/置信区间/统计表属于 P1.2。

仍需完成 checkpoint/transfer/validation 全链只读验收、剩余正式训练、trajectory-shortening 预注册研究、
许可清单和论文/复现包。CTIN 修复、invalid completed 的重跑、提前停止或改变矩阵范围须 Owner 决定；
本次没有启动这些变更，也没有更新 `claims_proven`。
