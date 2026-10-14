# 跨领域架构 baseline 研究计划

> 建立时间：2026-10-14。状态：下一版研究扩展；不属于当前冻结的 2,067-cell 正式矩阵。
> 本计划借鉴 IMUNet 的“迁移架构蓝图、替换任务接口、在惯性任务上从头训练”做法，
> 不把它误称为预训练迁移学习，也不以新增模型数量本身作为目标。

## 1. 第一性原理：研究问题与交付物

要回答的科学问题是：

> 在输入、坐标系、监督目标、数据划分、训练预算、随机种子和评价协议完全一致时，
> 来自其他领域的不同序列建模归纳偏置，能否成为准确、稳定、高效且可复核的惯性定位参考 baseline？

因此，最终交付不是“若干能前向的网络”，而是以下闭合证据：

1. 每个候选的论文、官方仓库、固定 commit、许可证和原任务均可核验；
2. 清楚记录保留了什么架构、替换了什么接口，以及哪些部分只是本项目的工程假设；
3. 所有候选通过同一惯性输入、目标、训练和评价契约，避免把数据权限或调参差异当成架构收益；
4. 结果同时报告轨迹精度、三 seed 波动、参数量、MACs、显存、吞吐和延迟；
5. 表现好坏都可追溯到源码、配置、checkpoint、split、原始逐序列结果和内容哈希；
6. 形成一组可复现的 architecture baselines 和一项受控跨领域迁移研究，而不是扩大未经验证的方法清单。

## 2. 与当前正式工作的隔离边界

- 当前正式身份保持 **53 profiles / 13 sources / 3 seeds / 2,067 cells**，不得回填或静默修改。
- 当前代码库口径是 **23 个论文方法 / 33 个注册架构 / 57 个配置 / 53 个正式矩阵配置**。
- 一个来源论文至多增加一个“独立论文方法”；其 XY/XYZ、宽度、深度、训练头或预训练变体只增加配置，不能重复计数。
- 惯性领域原生论文方法与跨领域来源论文分别统计；跨域移植不意味着该论文原本研究惯导，也不证明本项目是该架构首次用于惯导。
- 新候选先放在 `research/staging/cross_domain_baselines/`，不注册 model、不加入公开配置、不触发 GPU 或正式训练。
- 只有新 source manifest、coverage、capacity policy、seed identity 和 Owner 批准全部建立后，才能进入下一版矩阵。
- 当前冻结训练、结果闸门、CTIN 失败处置、统计、轨迹缩短和许可清单均按原计划继续，本研究线不替代它们。

## 3. IMUNet 给出的可复用先例

[IMUNet](https://arxiv.org/html/2208.00068v1) 将 MobileNet、MobileNetV2、MnasNet 和
EfficientNetB0 的视觉架构蓝图改成一维惯性回归网络：保留 stage、深度可分离卷积、倒残差、
通道和激活等主体设计，把二维算子、图像通道、分类池化与分类头替换为一维算子、六轴 IMU、
时域池化和二维速度回归头，并从头训练。官方参考仓库固定为
[`BehnamZeinali/IMUNet@c57f14d`](https://github.com/BehnamZeinali/IMUNet/commit/c57f14d0f4f5bbb8e29d836dabacc8718ca825e1)。

本计划继承这个“保留归纳偏置、替换任务接口”的原则，但加强四点：严格分离 validation/test、
固定三 seed、同时评价精度和效率、在进入结果表前做身份与内容哈希验收。IMUNet 仓库没有 LICENSE，
所以它只能作为公开行为规格参考；不得由论文的 CC 许可推导出代码再分发权。

## 4. 候选分层

### 4.1 第一批：三个正交归纳偏置

| 候选 | 原领域与来源 | 官方代码与许可 | 保留的核心 | 惯性化替换 | 当前判断 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ModernTCN** | 通用时间序列；ICLR 2024 Spotlight | [`luodhhh/ModernTCN@56a9a2c`](https://github.com/luodhhh/ModernTCN/tree/56a9a2c018385cd5acef015378cae7f084d1b11c)，MIT | patch embedding、大核 depthwise temporal conv、cross-variable FFN、残差 | 原任务头 → 统一 XYZ 窗口平均速度头 | 第一优先；卷积路线，中等算力。200 点短窗可能限制大感受野收益，仍是应报告的可证伪假设 |
| **PatchTST** | 通用时间序列预测；ICLR 2023 | [`yuqinie98/PatchTST@204c21e`](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9)，Apache-2.0 | patch token、共享且 channel-independent 的 Transformer 编码器 | per-channel forecast head → 保留六通道顺序的线性速度任务头 | 第一优先；注意力路线，中等算力。编码器保持 CI；有序输出头是显式惯性适配，编码器内跨轴 mixer 另列消融 |
| **S4D** | 长序列建模；[NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html) | [`state-spaces/s4@e757cef`](https://github.com/state-spaces/s4/tree/e757cef57d89e448c413de7325ed5601aceaac13)，Apache-2.0 | 对角结构化状态空间层、残差堆叠和序列状态 | sequence task head → 统一 XYZ 窗口平均速度头 | 第一优先；状态空间路线。先做纯 PyTorch S4D，不把完整 S4 专用 kernel 或 M2EIT 的 selective-SSM 混为同一实现 |

首批只比较这三种架构偏置。它们均默认从头监督训练；不得加载其他数据域的预训练权重。

PatchTST 第一版草案曾对六个通道的共享速度候选取均值，导致输出对任意通道置换不变。
这会抹去 gyro/acc 的单位差异与 XYZ 轴身份，不能作为主惯性 baseline。
当前采用通道独立编码器和按固定 `C,D,N` 顺序展平的线性速度头；测试同时证明编码器独立性及最终输出的轴敏感性。
BatchNorm 在训练模式下仍会跨折叠的 `B×C` 汇总统计，所以 CI 仅表示无跨通道注意力/特征混合路径，不能宣称训练统计完全独立。

### 4.2 第二批：回答不同问题，不能混入首批主比较

| 候选 | 来源与许可 | 研究价值 | 为什么后置 |
| :--- | :--- | :--- | :--- |
| **TS-TCC** | IJCAI 2021；[`emadeldeen24/TS-TCC@15bdfd6`](https://github.com/emadeldeen24/TS-TCC/tree/15bdfd6eced6aeb6c638839828effe1b4b2ea391)，MIT | 检验 weak/strong view 的时序与上下文对比预训练 | 它主要改变训练协议而不是骨干；必须建立单独 pretrain → finetune 身份，并证明增强不破坏 IMU 物理语义 |
| **EEGNet** | Journal of Neural Engineering 2018；[`vlawhern/arl-eegmodels@4a512e5`](https://github.com/vlawhern/arl-eegmodels/tree/4a512e503198db2010848813ead9afbf8cd54c97)，CC0/Apache-2.0 边界待逐文件确认 | 极轻量 temporal conv + 跨通道 depthwise filter，可检验六轴显式混合和端侧效率 | EEG 电极空间不等于 IMU 轴；必须将六轴表述为传感通道，不能虚构空间拓扑 |
| **TimesNet** | ICLR 2023；[`thuml/TimesNet@2b39e14`](https://github.com/thuml/TimesNet/tree/2b39e14cd16448e0ed94e4df81c4f5dc86f658d9) 与可运行 [`Time-Series-Library@4e938a1`](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e)，MIT | FFT 周期发现与二维 temporal variation 是另一类频域偏置 | 与首批相比结构、显存和适配复杂度更高；先证明 200 点 IMU 窗口有稳定可辨识周期，再决定是否投入 |
| **Mamba** | COLM 2024；[`state-spaces/mamba@e9594ce`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)，Apache-2.0 | 检验完整 selective-SSM backbone，而不是单个时间专家 | 第二批末位；与 M2EIT 重叠最高，正式高效路径依赖 CUDA 扩展，且没有官方 IMU/通用时序监督 recipe |

Mamba 虽列入第二批候选，但在未完成 M2EIT 与 S4D 的组件重叠审计前暂缓实施；不能为了达到某个方法数量而强行纳入。
普通 LSTM、TCN、ResNet、MobileNet/EfficientNet 已被当前仓库覆盖，不再以换名称的方式重复迁移。

## 5. 统一适配契约

首批候选只有满足下列共同契约，结果才可横向比较：

| 项 | 冻结规则 |
| :--- | :--- |
| 输入 | `B × 6 × 200`，200 Hz；通道顺序固定为 `[gyro_xyz, acc_xyz]`；不得暗中增加姿态、位置、标签或未来窗 |
| 坐标/权限 | 共享 `gravity_world` 输入视图和同一种 orientation 权限；deployable 与 reference/oracle 必须分 track，不得混表 |
| 主输出 | `B × 3` 的窗口平均 XYZ 速度；Z 必须真实学习；同预测 XY 只取 `[:, :2]`，不得另训二维网络冒充 |
| 任务头 | 首批统一使用确定性线性速度头和同一 MSE；`logstd`、分箱或门控头只能作为后续独立 head ablation |
| 初始化 | 从头训练；任何自监督或跨数据预训练必须建立单独 profile、父 checkpoint provenance 和无 test 泄漏证明 |
| 数据与选模 | 共享 train/val/test split、窗口和增强；只允许 train 拟合、val 调参和选模，test 在冻结后一次评估 |
| 预算 | 相同 seeds、epoch/step 上限、batch 语义、early-stop 规则和超参搜索额度；OOM 降级须进入 profile identity |
| 输出报告 | ATE/RTE/速度误差与失败率，加 params、MACs、峰值显存、训练吞吐、同硬件/软件/精度下的延迟 |
| 因果性 | 每个模型明确 `window_noncausal` 或 `stream_causal`；窗口内部双向建模不等于使用窗口外未来信息 |
| 命名 | 使用 `architecture_port` 和来源限定名；不得把适配版表述成原论文在惯导上的官方方法 |

需要同时保存一张 keep/replace 表：原始输入和任务、保留的 block/stage/normalization/activation、
改变的维度、tokenization、pooling、head、loss，以及所有无法从论文或代码确定的假设。

同为线性头不意味着读出或参数量一致：当前 XYZ 头部 ModernTCN / PatchTST / S4D 为
1,155 / 57,603 / 387 参数，PatchTST 保留 `C,D,N` 展平，另两者先做时间均值。
因此首批比较只能对这些完整适配配置作结论；必须报告 backbone/head 成本，并预注册
PatchTST 时间 GAP 的独立 matched-pooling 消融，不能将差异直接归因于三种骨干偏置。

## 6. 分阶段验收与子 Agent 编排

| Gate | 子任务 | 通过条件 | 失败后的处理 |
| :--- | :--- | :--- | :--- |
| G0 来源与法律 | 论文、官方 repo、commit、逐文件 SHA、LICENSE、依赖和预训练权重权限 | primary source 与许可字段齐全；能区分论文许可和代码/权重许可 | 无许可时只做 clean-room 规格，不复制或分发源码/权重 |
| G1 隔离结构 PoC | source card、provenance、未注册 model/config、科学契约测试 | CPU forward/loss/backward 有限；真实 Z 梯度；结构特性和形状测试通过；不触及 active registry | 保留失败证据，修正必须更新假设或 identity |
| G2 标准生命周期 | 原窗口/默认宽度接入 NIO → Trainer → Validator → Predictor → checkpoint reload | 真实 train/val/test 小型 fixture；参数更新；reload 等价；结果非空且维度/单位正确 | 不得用随机张量 smoke 宣称可训练或可复现 |
| G3 容量与依赖 | 当前目标硬件上的 native DataLoader、增强、forward、loss、backward、optimizer 和推理 | 峰值显存/吞吐/延迟收据齐全；无隐藏 kernel 降级；common batch 或显式降级身份可执行 | 不自动占用正式训练 GPU；先做宽度/批次决策 |
| G4 预注册 pilot | 在看 test 前冻结候选、数据、seed、调参空间、停止规则和统计脚本 | 三 seed；train/val 选择；blind test 一次；结果走 matrix report；失败与 N/A 同样入账 | 不按中间排名替换数据集、seed 或方法 |
| G5 下一版准入 | 独立复审 fidelity、许可、结果链、成本和科学增益 | Owner 批准新的 source/coverage/capacity/seed/version；当前 2,067 行保持不变 | 未通过者仍可作为负结果或 staging，不计正式方法 |

当前任务分工：

- Evidence Agent：核验 IMUNet 先例、候选论文、官方仓库、commit、许可和重叠边界；
- ModernTCN Agent：只在隔离 staging 完成 G0/G1，不注册、不启动 GPU；
- PatchTST Agent：只在隔离 staging 完成 G0/G1，并以测试证明编码器 channel independence 和有序任务头的轴敏感性；
- S4D Agent：只在隔离 staging 完成 G0/G1，核对 ZOH/FFT 数值与梯度；
- Root：维护本计划、审查三个实现的一致性、运行服务器交叉测试和补齐 G2 生命周期；
- 后续 Independent Reviewer：在任何 active registration 或下一版冻结前做反例审查。

## 7. 预注册比较与决策规则

首轮 pilot 的目标不是挑出“看起来最好”的网络，而是判断三种归纳偏置是否值得进入完整下一版：

1. 在读取 test 结果前选定一个行人源和一个机器人源，并固定其 manifest/split hash；
2. 每个候选只使用同额调参额度，搜索范围在运行前写入机器清单；
3. 以 validation 可训练性、数值稳定性、资源上限和预先定义的最低有效改进/非劣阈值决定是否继续，
   test 只用于冻结后的无偏报告；
4. 一个模型不因 test 排名差而从记录中删除；有效的负结果也是交付；
5. 不只与当前最佳方法比较，还与参数量和计算量接近的卷积、Transformer、SSM 基线配对比较；
6. 若模型只在更长 history 下有效，必须新建 history profile 并与所有可支持对照共享该输入，不能只给候选额外上下文。

进入完整下一版矩阵的必要条件是证据链完整、许可可接受、生命周期稳定且提供与现有架构不同的信息；
“在一个 seed/一个数据集上超过当前第一名”既不是必要条件，也不是充分条件。

## 8. 当前状态

- IMUNet 先例、四个视觉骨干的保留/替换关系、官方 commit 和无 LICENSE 风险已核清。
- 第一批三个候选及固定官方提交、许可、结构互补性已经形成可执行决策。
- 三个子 Agent 已完成隔离 model、XY/XYZ 配置、source card、provenance 和科学测试，并交叉独立复审；
  模型及配置仍不在 active registry / 公开配置目录。
- **2026-10-14 17:36 CST**：AutoDL 独立 CPU snapshot 第二轮验收为 **108 passed / 0 failed / 0 skipped**。
  包括 102 个科学节点和三架构各 XY/XYZ 共 6 个 NIO → Trainer → Validator → Predictor → checkpoint reload。
  复用三个独立 RIDI 工程片段，每项实际两次 optimizer update；不解释为收敛或精度结果。
- 独立复审的梯度 None 与同预测 XY 证据缺口已补齐，并使用新的 source manifest 重验；正式 source/policy/coverage/seed/native guards 不变。
- 精确源码/原始收据哈希、资源限制、参数分解和复验入口见
  [CPU 验收记录](../research/staging/cross_domain_baselines/CPU_HANDOFF.md)。
- 尚无候选完成 active registration、CUDA capacity、完整数据训练评估、三 seed pilot 或下一版准入；因此新增正式方法数仍为 **0**。
- 下一步依赖见[阶段交接](../research/staging/cross_domain_baselines/NEXT_STAGE_HANDOFF.md)；不能因为 CPU 工程通过就直接占用当前正式 GPU 或扩大矩阵。

