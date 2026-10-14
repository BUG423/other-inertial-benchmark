# 十方法静态准备交接（不是 CPU / 训练验收）

> 交接时间：2026-10-15 CST。论文/源码来源审计开始于 2026-10-14；范围规划提交为 `bf55d24`。
> 用户要求：十个 CVPR 2026 主会来源，全部新模型运行留到现有及先前排队任务之后。

## 当前事实

- 十篇独立主会论文的完整原 PDF、作者代码、固定 commit、许可证原字节与真实核心已核。
  去重清单和逐项 primary-source 链接见 [README](README.md)，不计 Workshops/Findings/仅 README/无许可候选。
  BinaryAttention 的 Apache 文件版权附录保留上游未消解冲突标记，另列 notice-review HOLD，不推导法律有效性。
- 十个隔离目录各有模型、XYZ 配置草案、来源/适配元数据、许可、待执行科学单测，共 **50 个文件**。
  都标为 `static_draft`，没有公开 registry 注册、正式 coverage enrollment 或原 source 热部署。
- 三个子 agent 交叉审查：时序三项由结构组复核，视觉五项由时序组复核，结构两项由视觉组和 Root 复核。
  结论只覆盖 source/AST/数学表达/配置/许可，**不构成 G1 数值/梯度或 G2 生命周期通过**。
- Root 的纯标准库检查 `check_static_manifest.py --require-adapters` 通过：
  十个 source identity、23 个 Python AST、JSON、类名/生命周期映射、配置 `INPUT_KEYS` 和精确 artifact SHA。
  不导入任何模型/框架，也不收集 pytest。
  准确命令、采样 UTC、plan SHA 和有限范围声明见 [静态元数据收据](static_metadata_receipt.json)。
- [tail queue](tail_queue.json) 绑定 **50 个适配文件 + 3 个共享 Python 文件**，每项含 SHA256；
  原正式来源绑定仍为 `1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949`。

| 阶段 | 本批实际状态 |
| :--- | :--- |
| 论文来源与隔离源码 | 十项已准备；保留明确算法核心和 keep/replace |
| 有限静态独立审查 | 已完成；修正点与剩余科学边界如下 |
| 模型 import / CPU 科学测试 / 原生生命周期 | **0 次执行，未通过** |
| 参数、MACs、显存、延迟与 CUDA 容量 | **未测、未准入** |
| 三 seed 训练 / validation / test / 精度结果 | **0 个启动、0 个结果** |
| 自动调度 / 正式方法或矩阵新增 | **未部署、0 增量** |

旧 ModernTCN/PatchTST/S4D 的 108 项 CPU 证据仍是旧三项的工程验收，不能继承给新十项。
本批静态身份不改变正式 **53 × 13 × 3 = 2,067**，不改变 23 个惯性来源论文方法的计数。

## 已修正的审查问题

1. BinaryAttention 原 sign STE 在 `x>1` 或 `x<-1` 截断，梯度区间为闭 `[-1,+1]`；
   草稿曾误写为单侧，现已对齐 fixed source。V 的量化反向另为开 `(-2,+2)`，不混淆两者。
2. YAML/直接 `model_cfg.input` 不能放 `channels`（现引擎 `INPUT_KEYS` 不接受）；
   通道顺序改作顶层元数据，模型仍严格检查 canonical `[gyro_xyz,acc_xyz]`。
3. 十项统一限定 mean-overlap 和 plain MSE；非 MSE/隐藏 kwargs、额外输入、预训练或不同任务契约拒绝。
4. 共享生命周期草案逐一要求全部可训练参数有有限梯度；验收旧固定三序列夹具、4 个 train/2 个 val 窗口和实际两次更新。
   不再过滤 `grad=None` 后把剩余梯度当全模型验收。
5. 公共 collection gate 把收据绑定到 tail-plan 和实际适配/共享文件字节；
   metadata 单件/总量限额、夹具路径包含性与读取限额已加。守卫仍只是**未运行的代码草稿**。
6. Git 的 staged 检查定位出 BinaryAttention 原样许可文件中的上游版权附录冲突标记；
   保留原字节、不替作者选版权分支，单列 notice-review HOLD。
   `git diff --check` 仅对这一确切上游证据文件作显式路径排除，其他新增代码/元数据不能豁免。

## 不得抹去的适配边界

- 共同输入是 `B×6×200`、200 Hz、`gravity_world/reference`、保留重力；主输出是真实 XYZ 窗口平均速度。
  输入 track 是 `reference_oracle`，不是 deployable；XY 仅从同一 XYZ 预测切片。
- TESSERA 仅选双流 encoder/GRU-conditioned attention pool/有序 concat/reducer 架构，SSL omitted。
  不继承条件 SSL 提案的 BT、view、mix 表；v1 默认 512/8/4096 与论文差异及未测容量保留。
- MS-Temba / MixerCSeg 的纯 Torch 选择性递推是来源方程参考，不是 fused CUDA；没有内核速度或数值/梯度等价验收。
  MS scan 的 FP32 累积不证明 residual_in_fp32 / AMP parity；先以 `amp=False` 做科学验证，低精度配方尚未冻结。
- BinaryAttention 是浮点 fake-quant 参考，不是 bit-packed CUDA；不能套用论文加速或低比特成本结论。
- UCAN 是 sharing/receiving + dual fusion + selective LKSA 核心；HPA/MBConv/SR 包装省去，
  时域 RF13、Fourier signed-epsilon 和 eye 初始化均是明确变更，不冒称完整网络或原最终初始化。
- PFGNet 的 Conv 初始化与作者显式 Kaiming 不同；ViT³ 的 2D→1D RoPE、3×3→3tap 和邻域尺度改变明确披露。
- Edge-RecViT/BinaryAttention 保留 CLS readout，其他 GAP 或 GRU pooling 不等价。
  宽深、token 化、初始化和 readout 仍是 profile 假设；横向结果不自动构成纯归纳偏置因果证明。
- 已编写 oracle、有限梯度、真实 Z、轴顺序和 reload 测试不等于已通过；部分梯度检查也不等于全源码梯度数值 parity。

## 下一步（全部留到末尾）

1. 原有实验及此前已授权排队任务的终态/失败/N/A 处置闭合；共享 CPU/GPU 资源交接并生成小型审计收据。
2. 先关闭 BinaryAttention 的版权声明复核，或固定作者修正来源/替换候选并重新审查；
   不把原样 Apache 文本存在当作完整授权结论。
   冻结独立新源码与数据身份，取得有界 CPU 资源；先验 collection guard，再做科学与原生生命周期。
   默认大配置的 CPU 预算须重新核，不沿用旧三项 2 GiB/420s 限额当作本批容量 GO。
3. 通过 CPU 和 CUDA 容量后，冻结新 profile/policy/coverage/seed、预算、对照与停止/选模规则；不静默缩小模型/改 batch。
4. 十方法 × RIDI/EuRoC × 三 seed 的 60 个候选单元仍是提案，尚未注册或启动；对照与完整 13 来源扩展未冻结。
5. 最后做 checkpoint、val/test、非空小结果及哈希、同预测 XY 和三 seed 统计闭环，才能进入结果表。

`execution_enabled=false`、`active_queue_enrolled=false`、`tail_barrier.satisfied=false`、`readiness_receipt=null` 保持不变。
没有新收据、运行或容量证据时，不发启动授权；不停止、重启、抢占原任务，也不自动重试 CTIN。
许可文件和来源审计不是 Owner/Legal 对整个项目、派生代码、数据或权重再分发的批准。
