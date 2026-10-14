# 下一阶段：容量验收与受控 pilot

> 2026-10-14 编排草案。G1/G2 已有 [CPU 收据](CPU_HANDOFF.md)；G3/G4 未开始。
> 下列范围、预算和配置是待冻结准备项，不是已经启动的正式矩阵或完整预注册。

## 执行顺序与责任

| 顺序 | 可执行任务 | 执行/复审角色 | 依赖与完成判据 |
| :--- | :--- | :--- | :--- |
| 1 | 独立 pilot source、临时注册桥、profile inventory 与小 coverage | Integration Agent / Root | 不修改 active registry；桥绑定三个 model.py、YAML 和自身哈希；实际 import 路径正确 |
| 2 | RIDI/EuRoC 的 CPU taskview 与标准按 profile lifecycle 准入 | Admission Agent / Independent Reviewer | 完整来源 manifest/split、实际 stride/window/dims/view 哈希闭合；不能用本次聚合 CPU receipt 冒充旧格式 |
| 3 | 修订隔离 capacity helper 的测量窗口数、fixture 和 recipe 绑定 | Capacity Agent / Root | 真正测试指定 batch/stride/增强/DataLoader/optimizer；无 silent fallback；测量次数写入 receipt |
| 4 | 单卡 native capacity、GPU 生命周期、效率测量 | Capacity Agent / Independent Reviewer | 等正式 GPU lease 释放或明确授权的资源交接；默认宽度 T200；峰值显存/时间/失败边界有原始收据 |
| 5 | 冻结 pilot 候选、控制、recipe、seeds、调参/停止规则与身份 | Root / Scientific Owner | 在看 pilot test 前固定全部判据，预注册不留空；source/coverage/policy/capacity/seed 互相绑定 |
| 6 | 三 seed 训练、validation 选模与冻结后的 test | Execution Agent / Results Reviewer | 无漏 seed、无挑数据/排名删方法；结果走 matrix-aware report；失败亦入账 |

前 1–3 项可在现有正式实验继续时准备和做有预算的 CPU 验证。
第 4–6 项不能因为瞬时 GPU 显存空闲就并发启动；正式 dispatcher 会继续分配工作，
GPU 空闲采样不等于训练锁或未来容量已交接。

## 最小研究范围

优先以 **RIDI 行人源 + EuRoC 机器人源**作为两个可训练同域来源。
本库已验收 RIDI 主 train/val/test 为 45/4/23 条（独立辅助测试分开），
EuRoC 为 5/1/5 条。test-only 的 SubT-MRS、GrandTour、UZH-FPV 不能当作第二个同域训练源。
在新 source 的 taskview 实际生成前，这只是范围准备，不是候选已经准入。

最小候选训练范围为 **3 个 XYZ ports × 2 来源 × 3 seeds = 18 cells**。
若加入 `ronin_resnet18_3d` 与 `imunet_3d` 两个控制，共 30 cells；
控制结果只有在 source/scientific recipe/input/sampling/selection 等身份完全一致时才能复用旧实验，
不能仅看同一 seed 就拼入比较。数量是训练单元，不是论文方法数或结果 split 数。
来源选择必须在看新候选 test 前冻结，不根据早期排名改换。

共同主任务保持六轴 T200 / 200 Hz、`gravity_world`、reference orientation、保留重力、
平均 XYZ 速度、MSE、从头训练。同预测 XY 仍然由 XYZ 切片，二维工程配置不进入主比较。
继承同一个 seed record 的 1735231569 / 843628571 / 951150512，
不得重新抽 seed、换掉弱 seed 或只报最佳 seed。

现正式冻结 recipe 实际为 30 epochs / min 15 / patience 10 / val every 5 / batch 128 /
train stride 200 / eval stride 10；历史 100-epoch 候选草案不是正式事实。
pilot 是否完全继承该 recipe，还是为公平配对消融关闭 early stop、固定更新上限，
须在 G4 预注册时明确，并对所有适用控制一致执行。
当前不宣称 pilot 的 budget/policy 已冻结，也不在尚无 capacity 时承诺开始时间。

## 不能直接复用旧工具的四个接口缺口

主要可复用逻辑在 [full_matrix.py](../../../tools/ops_v1/full_matrix.py) 的
`inventory`、`cpu_proofs`、`_gpu_measure`、`_gpu_lifecycle`、`capacity`，以及
`engine/benchmark.py` 的 capacity policy 验证。新扩展应使用独立源码，而不是热补丁到现运行副本。

1. **注册与源码绑定**：`inventory` 枚举 `src/cfg/models`，当前 staging 不会自动出现；
   `cpu_proofs` 的科学源码 reuse 集合不含 staging/plugin。隔离桥必须额外绑定 model/YAML/plugin SHA，
   原生 checkpoint reload 也需同一桥。108-pass 聚合收据不是旧工具按 profile 需要的标准 receipt。
2. **capacity 全范围**：`--profiles` 只是执行筛选，`formal_ready` 仍按整个 coverage 的 expected profiles 判断。
   不能筛当前 57 配置中的 3 个就声称 pilot ready；必须建立自己的小 coverage。
3. **真实采样与短探针**：stride 200 下已验收 taskview 给出 RIDI train 4,956 / EuRoC train 618 windows；
   `_gpu_measure` 若要求 B128 的 6 个独立 full batches，共 768 windows，会在 EuRoC 用尽 iterator。
   这是探针设计问题，不是 OOM。不得重复回绕后假装独立数据或静默改 stride；
   应让 probe 的有效 warmup/measurement 次数与真实可用 batch 匹配，并原样记录。
4. **生命周期 fixture 与 policy**：现 `_gpu_lifecycle` 固定 RIDI 单条训练 recording，
   `make_fixture` 要求 `2*B+1` windows。B128 + stride 200 需约 257 秒，超过 RIDI 单条最长约 201.7 秒；
   应按原 split 使用多条训练 recording 构造合法 fixture、保持 val/test 独立，明确注入 row sampling_spec。
   `capacity()` 当前产物未写 policy 强制检查的 `common_training_stride/common_eval_stride`；
   新 helper 必须在测量时记录实际值并验证，而不是事后编辑旧 receipt。

首先保留 B128 主比较设计并修订测量/fixture；只有真实容量仍不支持时，才提出统一 B64 的新 identity
和对应对照预算。B64 现在未测，不能自动记为可运行或把某个候选单独降 batch 后混表。
OOM、dependency、iterator exhaustion 和科学输入拒绝应使用不同失败分类。

## 归因、统计和停止规则

预注册必须补齐并固定以下字段后才能启动 G4：

- 全部模型、控制及 matched-pooling 的 profile/source/hash；固定数据 manifest 与 split hash；
- 同額超参搜索空间和次数，或一个不搜索的共同 recipe，二者不能混成一个口径；
- 数据曝光与更新上限、batch/采样/增强、early stop 与 validation 选模规则；
- 资源上限、数值失败/失联/OOM 的终止规则与是否重跑的 identity 边界；
- validation 可训练性、稳定性、资源接受阈值，以及最低有效改进/非劣阈值的数值和依据；
- blind test 的一次冻结策略、逐序列配对与 group-aware 统计、三 seed 波动、N/A 和失败记录。

PatchTST flatten 头比另两个头更大，须报告 backbone/head 分解；
matched-pooling 只新增 PatchTST 的 GAP profile，固定 encoder，配对初始骨干权重并保存初始化哈希。
该消融匹配读出算子而非相等特征宽度，不能宣称只剩归纳偏置一种变量。
算法层不提供数字的阈值现在仍属未决，不伪造一份看似完整的预注册。

## 与主线并行

原 2,067-cell 矩阵保持独立运行和监控；结果工具用独立 analysis snapshot 做只读 partial 验收。
CTIN failed、invalid completed 的处置及任何配方修改仍由 Owner 决定。
跨域候选即使 G3/G4 通过，也只有在 next-version source/coverage/capacity/policy 和 Owner 准入后
才计入下一版；当前正式方法、来源和范围不变。
