# CVPR 2026 十方法：惯性速度架构适配与末尾执行

> 用户范围更新：2026-10-14。十个独立 CVPR 2026 **主会**论文来源；全部新模型执行排在现有实验及此前已授权排队实验之后。
> 当前阶段：来源证据已闭合，隔离适配源码准备中。**0 个运行验收、0 个训练启动、0 个正式方法新增**。

最终交付是十个可追溯、可复现的惯性速度参考 baseline，以及同条件精度、稳定性和效率比较，
不是复现它们的视觉/遥感任务分数，也不是把十个论文名填入结果表。
每项必须保留来源特有的算法核心，明确任务接口替换，经独立源码审查、服务器生命周期与容量验收、
三 seed 训练和结果证据闭合后，才可称为可用 baseline。

## 十个来源及核心

下表锁定来源和拟议迁移核心，不是已通过测试的模型清单。
正文依据为 CVF 主会页面、完整原论文与作者固定 commit 的实际核心代码；
许可证、PDF 和关键源码 SHA256 见三份机器证据文件。Workshops、Findings、仅 README、未核代码/许可者不计数。

| # | 来源论文 / 作者代码 | 保留到惯性任务的辨识核心 | 许可 / 证据 |
| :--- | :--- | :--- | :--- |
| 1 | [ViT³](https://openaccess.thecvf.com/content/CVPR2026/html/Han_ViT3_Unlocking_Test-Time_Training_in_Vision_CVPR_2026_paper.html) · [ViTTT@36eb23b](https://github.com/LeapLabTHU/ViTTT/tree/36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f) | 本窗 K/V 的 negative-dot 快权重更新，SwiGLU 与局部卷积分支 | MIT · [来源卡](evidence_backbones.md) |
| 2 | [Edge-RecViT](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.html) · [code@9d0d34a](https://github.com/RicardoLee510520/Edge-RecViT/tree/9d0d34a8b3029d62955c9a1cf54ff7e4409ff934) | token 级递归深度路由、首尾独立与共享中段权重 | Apache-2.0 · [来源卡](evidence_backbones.md) |
| 3 | [PFGNet](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.html) · [code@9db0247](https://github.com/fhjdqaq/PFGNet/tree/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa) | 多尺度外围卷积、中心抑制、局部导数/方差驱动尺度门控 | Apache-2.0 · [来源卡](evidence_backbones.md) |
| 4 | [BinaryAttention](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html) · [code@a8e0268](https://github.com/EdwardChasel/BinaryAttention/tree/a8e0268ffc63ed4bb26bceac5fa015d358732055) | 缩放 sign-Q/K、补偿与来源 STE；浮点 fake-quant 参考路径 | Apache-2.0 · [来源卡](evidence_backbones.md) |
| 5 | [UCAN](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_UCAN_Unified_Convolutional_Attention_Network_for_Expansive_Receptive_Fields_in_CVPR_2026_paper.html) · [code@b812c0a](https://github.com/hokiyoshi/UCAN/tree/b812c0adb8ce70d442973db5c87bab7d1e8c3333) | 局部窗口注意力、Hedgehog 全局/通道双融合、sharing/receiving、大核精炼 | Apache-2.0 · [来源卡](evidence_backbones.md) |
| 6 | [MS-Temba](https://openaccess.thecvf.com/content/CVPR2026/html/Sinha_MS-Temba_Multi-Scale_Temporal_Mamba_for_Understanding_Long_Untrimmed_Videos_CVPR_2026_paper.html) · [code@d42b460](https://github.com/thearkaprava/MS-Temba/tree/d42b4606d03884bfe2591d0bed1ed7a1dd01ab22) | dilation 1/2/3 独立相位选择性 SSM、精确交织还原、多尺度 SSM 融合 | Apache-2.0 · [来源卡](evidence_temporal.md) |
| 7 | [Spectral Scalpel](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Spectral_Scalpel_Amplifying_Adjacent_Action_Discrepancy_via_Frequency-Selective_Filtering_for_CVPR_2026_paper.html) · [code@01adb1e](https://github.com/HaoyuJi/SpecScalpel/tree/01adb1ed708d410023b8482c52270cac358c8525) | MASF 多尺度自适应频域滤波与 FACM 频谱通道混合 | MIT · [来源卡](evidence_temporal.md) |
| 8 | [TESSERA v1](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_TESSERA_Temporal_Embeddings_of_Surface_Spectra_for_Earth_Representation_and_CVPR_2026_paper.html) · [code@e368539](https://github.com/ucam-eo/tessera/tree/e368539f7c4711e5f16fdd09801573830e34baf9) | 双流 Transformer、GRU 条件时域注意力池化、有序拼接与 reducer；**仅架构，SSL omitted** | MIT · [来源卡](evidence_temporal.md) |
| 9 | [CSaN](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Selection-as-Nonlinearity_Bridging_Attention_and_Activation_via_a_Joint_Game-Decision_Lens_CVPR_2026_paper.html) · [code@a07175f](https://github.com/SudongCAI/CSaN/tree/a07175ffb9fa82ec58eca397e80aa63027c5d73f) | SoftGShift、行温度/列幅值校准、public/private 注意力预算补偿 | Apache-2.0 · [来源卡](evidence_structured.md) |
| 10 | [MixerCSeg](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.html) · [code@7ad3fd6](https://github.com/spiderforest/MixerCSeg/tree/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d) | TransMixer：每 token 的 delta 通道排序、global/local gather、两路处理与 scatter | MIT · [来源卡](evidence_structured.md) |

序号只是清单顺序，不是 GPU 调度优先级。三个来源证据快照分别为
[backbones](candidate_backbones.json)、[temporal](candidate_temporal.json)、[structured](candidate_structured.json)；
其 `not_implemented` / `actions=false` 描述证据采集时点，当前静态适配状态另由 [tail queue](tail_queue.json) 记录。
论文来源不等于作者提供了官方惯导实现；十项均需本项目适配。不声称首次用于惯导或保证更优。

## 共同任务契约与不可混淆的变更

- 输入严格为 `B×6×200`、200 Hz，有序 `[gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z]`；
  `gravity_world`、`orientation=reference`、保留重力，结果标为 `reference_oracle`，不混作 deployable。
- 无图像、文本、骨架、地理信息、初速度、额外传感器或窗外未来；独立窗口零初态，不跨窗/跨 split 携带快权重。
- 输出是真实学习的 `B×3` 窗口平均速度，单位 m/s；XY 仅为同一次 XYZ 预测的切片，Z 不补零。
- 全部从头监督训练，主架构身份共享 MSE、同划分、三个既有 seed 和预注册预算。
  替换六轴 stem、1D 时间算子、位置、池化/线性任务头不隐藏；不同 readout 必须显式标识。
- 论文专用训练损失不自动混入 MSE 表：MS-Temba C-alignment/辅助头、TESSERA BT/随机视图/mix、
  原分类/动作边界/重建/分割/蒸馏目标若以后研究，须建立独立 recipe 与预算，不多算独立论文。

TESSERA 主选身份为 `tessera_v1_dualstream_architecture_velocity`。
temporal 证据中的原 `tessera_v1_temporal_dualstream_velocity_xyz` 是包含 SSL 的条件提案，
**不能直接继承其 keep/loss 表为本批默认**。本批仅 gyro3/acc3 两条真实数据流、encoder、
GRU-conditioned attention pooling、有序 concat/reducer；时间位置由既有 T200/200Hz 内部派生。
保留官方 v1 与论文在层数、pool/fusion、mix 设置上的差异；不称完整 TESSERA/SSL 复现。

ViT³ 的 inner update 只读输入 K/V、不读速度目标、每窗从基参数重置；不是拿 test 标签训练。
BinaryAttention 目前范围是浮点参考，而非 bit-packed CUDA 内核；不移用论文的低比特加速结论。
PFGNet 的频率引导来自局部梯度/Laplacian/方差，不替换为泛用 FFT。
CSaN 锁官方文件最后生效的 Attention/SoftGShift，无早期 tanh 变体偷换。
MixerCSeg 明确只移植 TransMixer 时域核心，二维 HOG/SRF/分割网络不冒称完整保留。

## 编排与责任

| 工作包 | 执行者 | 当前/退出条件 |
| :--- | :--- | :--- |
| 来源、原理和 keep/replace | 三个来源子 agent；Root 复核 | 十篇主会、完整 PDF、作者代码、固定 commit、许可与核心哈希已记录；依赖 notice 后续逐文件核对 |
| ViT³ / Edge-RecViT / PFGNet / BinaryAttention / UCAN 静态适配 | `modern_tcn_port` | 隔离 `model.py`、3D 草案、provenance、许可、待执行科学测试；不 import 模型 |
| MS-Temba / Spectral Scalpel / TESSERA 架构静态适配 | `patchtst_review` | 核心忠实、SSL/aux 损失身份明确；不能实现的核心报 HOLD |
| CSaN / MixerCSeg 静态适配 | `s4d_port` | 校准变体、逐 token 路由与 1D scan 的边界可核对 |
| 统筹、独立审查、现任务监控 | Root | 保持冻结身份；静态完整性不冒充运行通过；汇报实际阶段与阻塞 |
| CPU 生命周期、CUDA 容量、训练/评测 | 原任务终结后编排 | tail barrier、新源码/策略/coverage/seed、资源准入与运行收据缺一不可 |

实现静态可并行；**所有本批模型 import、CPU 前后向/测试/生命周期、GPU 容量、训练和 val/test 都后置**。
本机继续只做文档、SSH 编排和静态解析，框架执行只能在 AutoDL。
源码草案不自动注册公开 registry，不追加冻结矩阵，不热部署运行 snapshot。

仅元数据/语法检查可在本地用 `python3 research/staging/cvpr2026_baselines/check_static_manifest.py`；
它只解析 JSON/AST，不导入任何模型，不是科学或容量验收。
[科学单测草案](test_native_lifecycle.py) 之外，各目录将保留核心数学/输入权限测试。
公共 [collection guard](conftest.py) 要求 AutoDL 私有 `IPB_CVPR2026_TAIL_READINESS` 收据，
绑定原队列终态和资源交接证据及本清单哈希；没有该收据时拒绝收集模型测试。
这个守卫本身尚未执行验收，`pytest --collect-only` 也不能绕过后置约束。

## 末尾执行与验收顺序

1. 现有实验及此前已授权排队实验全部有终态处置；失败/N/A 留原因，没有未解释的 running。
2. 释放并重新取得共享 CPU/GPU 资源租约；形成 tail readiness 收据。瞬时空闲显存不是准入。
3. 十项独立 source/fidelity 审查 → 服务器 CPU 科学与原生生命周期 → CUDA 容量。
   依赖不可用或失败如实 HOLD；不偷偷用通用骨干、改 batch/宽深或换 loss。
4. 独立冻结新 profile、coverage、capacity/policy、source、数据与 seed 身份，以及 control、停止规则和选模规则。
5. 拟议小规模 pilot 为十方法 × RIDI/EuRoC 两来源 × 三 seed = **60 个候选单元**；
   对照、预算和来源适用性仍待冻结，不是已注册的 60 个训练，也不自动展开到 13 来源。
6. 通过 checkpoint、val/test、小结果哈希、同预测 XY 和三 seed 统计验收；再决定正式扩展与论文表。

[机器编排](tail_queue.json) 明确 `execution_enabled=false`、`active_queue_enrolled=false`、
`tail_barrier.satisfied=false`、`readiness_receipt=null`，它是准备/依赖清单，不是部署的自动调度器。
不得停止、重启、抢占原任务或自动重试 CTIN。

当前正式仍为 **53 profiles × 13 sources × 3 seeds = 2,067 cells**；425 文件冻结来源不变。
ModernTCN/PatchTST/S4D 的旧隔离源码和 108 项 AutoDL CPU 证据保留，**不计入这十项，也不沿用旧三方法 pilot**。
参见[原交付计划](../../../docs/DELIVERY_EXECUTION_PLAN.md)与[跨域范围更新](../../../docs/CROSS_DOMAIN_BASELINES_PLAN.md#当前范围与执行顺序)。

许可核验只是固定源码使用条件的证据，不代表 Owner/Legal 已批准整个仓库、派生代码、权重或数据再分发。
