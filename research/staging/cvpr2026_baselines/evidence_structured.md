# CVPR 2026 主会：注意力补偿与动态通道路由候选证据

记录时间：2026-10-14 15:14 UTC。仅 primary-source 阅读、原字节哈希与静态源码检查；
没有导入模型、运行 CPU/GPU 模型测试、训练、下载权重或改变 active registry / 正式 2,067 rows。
Root 已将本组范围从最初的“四个结构候选”缩为补齐去重清单中的**第 9、10 个论文来源**。
PFGNet、ViTTT 等已有 owner 的八项不在本组重复计数。

## 交付目的与排队边界

交付不是复现视觉任务分数，而是两个来源限定的架构参考：在同输入权限、监督、划分和预算下，
检验“注意力预算补偿”和“由 selective-SSM 的输入依赖 delta 驱动的 global/local 路由”是否适合惯性速度回归。
它们是待实现候选，不是原论文的官方惯导方法，也不声称首次迁移到惯导。

统一拟议契约为有序 `[gyro_xyz, acc_xyz]` 的 `B×6×200`、200 Hz、共享 gravity-world/
orientation track；从零训练、无外部模态/预训练/窗外未来/跨窗状态；一个线性 XYZ 窗口平均速度头和同一 MSE。
Z 真实受监督，XY 仅截取同一 XYZ 预测。动态计算、读出、位置编码及 adapter 的成本都进入完整 profile。
**所有模型导入、测试、容量和训练必须在原有任务终结、tail barrier 和新批次准入后执行**；
当前证据不授权抢占资源或提前运行。参见 [tail queue](tail_queue.json)。

## 1. CSaN：锁定 ViT-family 的实际生效补偿路径

来源为 Sudong Cai 等的
[Selection-as-Nonlinearity: Bridging Attention and Activation via a Joint Game-Decision Lens for Interpretable, Discriminative Visual Representations](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Selection-as-Nonlinearity_Bridging_Attention_and_Activation_via_a_Joint_Game-Decision_Lens_CVPR_2026_paper.html)，
CVPR 2026 主会，pp. 11621–11631；
[CVF 原 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cai_Selection-as-Nonlinearity_Bridging_Attention_and_Activation_via_a_Joint_Game-Decision_Lens_CVPR_2026_paper.pdf)
完整 1,254,501 bytes / 11 pages / SHA256 `b7cf0d1340db0aabc942c65211a1807224392d8f75b239c44c9ed5979d027d7e`。
PDF 第一页直接指定官方项目；原任务是视觉 Transformer 表征与分类，核心是层次预算校准和 public/private value 协作。

官方来源固定为
[`SudongCAI/CSaN@a07175ffb9fa82ec58eca397e80aa63027c5d73f`](https://github.com/SudongCAI/CSaN/tree/a07175ffb9fa82ec58eca397e80aa63027c5d73f)。
[`LICENSE`](https://github.com/SudongCAI/CSaN/blob/a07175ffb9fa82ec58eca397e80aa63027c5d73f/LICENSE)
为 Apache-2.0，11,357 bytes / SHA256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`。
固定的真实模型代码
[`MODELS/vit_csan.py`](https://github.com/SudongCAI/CSaN/blob/a07175ffb9fa82ec58eca397e80aa63027c5d73f/MODELS/vit_csan.py)
为 128,775 bytes / SHA256 `db47f5d9026cd433758ed2751765776765653e42a832516776a0dba338472459`；
`Attention`、`SoftGShift`、`Block`、`VisionTransformer` 均有实现，不是仅 README。
源码还含 timm 的模型工厂、权重加载与其他家族配置；它们不是拟议 adapter 的运行入口，不加载任何预训练权重。

重要版本消歧：该文件有两个同名 `Attention` 定义，后一个（297–390 行）覆盖前一个。
本候选锁**后一个、SoftGShift 路径**，不把前段 GELU+tanh 路径或 Swin/Hiera 版本混在一起。
它从 QK 分数的 quasi-linear row statistic 和输入投影生成补偿，经
`SoftGShift(x)=x*sigmoid(Wx)`、降维/展开，形成 row temperature、column magnifier、public gain 和 private gain。
校准后读出包括 public attention-value 路径及同 token 的 private value 路径。
后段代码的 gain/temperature 输出是线性值（部分加 1），**没有前段的 tanh 约束**；实现时不得悄悄改成另一个变体。

| 项 | 保留的来源结构 | 惯性适配提案 / 不能声称 |
| :--- | :--- | :--- |
| 核心 | QKV、quasi-linear importance、SoftGShift、row/column 补偿、public/private readout | 完整保留，不退化成普通 attention 或另加一个泛用 gate |
| block | pre-LayerNorm、两条残差、GELU MLP、显式 DropPath/LayerScale 配置 | 选定普通 `Block`，不调用平行 block 或不同 family 混出未声明变体 |
| 输入/位置 | 图像 patch embedding、视觉位置/分类 token | 有序六轴到 latent token 的学习投影；一维窗内位置；移除分类 token，显式声明 readout 改动 |
| 读出/监督 | 分类读出与交叉熵 | temporal mean + 一个 linear XYZ 头 + window-average-velocity MSE；无 teacher/图像/语义标签 |
| 规模 | ViT-Tiny 工厂给 `D=192,depth=12,heads=3` | 可作首个宽深参考，时间 token 化另行冻结；缩宽/改深必须进入 profile identity |

风险：校准仍计算 QK，不能声称线性复杂度；未约束的校准系数需要有限值/梯度验收，不能静默 clamp。
窗口内全局交互为 `window_noncausal`，但不访问窗外。位置、pooling 和规模是本项目假设，分类收益不等于惯导收益。
作者 README 明确 CSaN 是 model-specific compensation，不是任意骨干上的现成 plug-in；因此只做上述 ViT 路径。
后续需在 tail 后做 public/private 两路数值与梯度、版本消歧、真实 Z、轴敏感性和原生命周期验收；当前尚未执行。

## 2. MixerCSeg：保留 TransMixer 的输入依赖路由，不冒称完整二维裂缝网络

来源为 Zilong Zhao 等的
[MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.html)，
CVPR 2026 主会，pp. 17969–17978；
[CVF 原 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.pdf)
完整 7,852,095 bytes / 10 pages / SHA256 `a9c196236f73923e60e402d3b6d4d903a058c923de2086b113090344b04fba4f`。
PDF/主会页面均直接指定官方仓库。原任务为图像裂缝分割；核心将 SSM 的隐式注意力按 delta 分解为不同全局/局部通道处理。

官方来源固定为
[`spiderforest/MixerCSeg@7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d`](https://github.com/spiderforest/MixerCSeg/tree/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d)。
[`LICENSE`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/LICENSE)
为 MIT（2026 spiderforest），1,069 bytes / SHA256 `e7d62576226fe965fd57f8fb192c4b1163be9bb92e17304a7f3f6cbaa326aada`。
这是项目 root license；不据此抹去 VMamba/Mamba 等借用代码的第三方归属和 notices，逐依赖/复制文件复审仍须在实现时保留。

已读取固定完整源文件：

| 真实代码 | bytes | SHA256 | 确認用途 |
| :--- | ---: | :--- | :--- |
| [`VMamba/models/vmamba.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/VMamba/models/vmamba.py) | 62,669 | `3a259ceacbd05f8521292c5e0340cef0f02f22912208d4229614fc0347770fcc` | `SS2Dv2` 内实际路由与 `TransMixer` residual block |
| [`VMamba/models/utils.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/VMamba/models/utils.py) | 9,723 | `a935b81d78a1a43d91536e2dff1cfa750dfc76322179216a3b891d393e5c0927` | `get_global_local_index`、`Attention`、`LocalModule` |
| [`VMamba/models/csms6s.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/VMamba/models/csms6s.py) | 10,627 | `0a73a10dc8328c697fa155b312105f9accce5e4511db5d5f7b7a3fac71b90b36` | CUDA 扫描接口和 `selective_scan_torch` 明确递推参考 |
| [`models/encoder/vss_block.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/models/encoder/vss_block.py) | 4,933 | `d00aad8f6d6d34443aabf09c7d185204aef1c4d8bcd5e391f2c37205eb5bf587` | 层次 stage 与 TransMixer/HoG 模块接线 |
| [`models/layers/hog_edge.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/models/layers/hog_edge.py) | 5,036 | `fc9fd40ae70f87046134989329b12da7f527bc0072a019c1f7ae5b879777c6d8` | 二维 Sobel/角度 histogram、DEGConv 对应路径 |
| [`models/decoder/SRF.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/models/decoder/SRF.py) | 3,561 | `c6798ec41d6434fbfed9a53551be40485dd68f2125c9a0d48053cd206692edb7` | SRF 多层 refinement/fusion 和分割读出 |
| [`models/segmentor/MixerCSeg.py`](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/models/segmentor/MixerCSeg.py) | 1,869 | `511f9dec06917ee988e09dc975f4df9d5146e641e2ee5f20f698ba6fd56024f7` | 实际四 stage 配置与 BCE/Dice 来源任务头 |

`SS2Dv2` 真正启用 `self.mixer=True`。路由 helper 将 bias 加到 delta 后做 softplus，
**对每个样本、每个 token 在 channel 维 descending sort**，按 `g_ratio=.5` 选择 global/local channel indices。
它不是时间 token pruning，也不是每个窗口固定一次 channel 二分。
扫描输出按这些 indices gather，global 分支做 self-attention，local 分支做 GroupNorm/max-pool/sigmoid refinement，
再按原 indices scatter 回去后 merge。简单并排 Mamba+attention 不等于这个核心。

| 项 | 保留的来源核心 | 显式替换/适配范围 |
| :--- | :--- | :--- |
| SSM | 输入依赖 delta/B/C、稳定负 A、D skip、门控输出与 residual/MLP | 四向二维扫描 → 独立窗内正/反向一维扫描；零初态；必须验数值与梯度，不能混成 S4D |
| TransMixer | 每 token 的 delta-ranked global/local indices；两条不同分支；scatter 回原通道 | local 2D pooling/卷积 → 1D temporal pooling/卷积；不删除 routing 来省事 |
| 多层骨架 | 来源 builder 的四 stage，depth `[1,1,1,1]`、width `[16,32,64,128]`、state `[8,8,16,16]` 可作参考 | stem/降采样/长度对齐适配 200 点；参数规模未冻结、未测容量 |
| DEG/SRF | 源码确有二维 HOG 和分割 fusion | 本候选限定为 **TransMixer temporal-core 1D port**；二维 HOG/SRF/分割头不冒称完整复现，不能伪造 IMU 的人体/空间方向标签 |
| 读出/目标 | 来源为 BCE+Dice 的像素输出 | temporal mean + 一个 linear XYZ 头 + 同一 MSE；无预训练/裂缝标签/骨架/其他传感器 |

风险：二维到时域、四向到双向、省去 HOG/SRF 是实质架构边界，名称必须来源限定。
global 分支仍有二次 attention，不能把整个 adapter 宣称线性；hard sort 是分段可微路由，tie、gather/scatter 梯度与排序敏感性要验。
原代码 in-place 写回不能未经梯度核验照搬；functional scatter 如数学等价也要记录。
来源 attention 的 fused dropout 不按 eval 自动置零，拟议 eval 必须确定性并记录任何语义修正。
无外部 pose/图像/人体拓扑权限；只从本窗 latent 特征求 delta。
这比一般 selective SSM 多了明确的纸面/源码专属路由，但不复现裂缝分割性能，也不提供惯性动力学保证。

## 3. 非 selected / 排除证据（不凑 licensed 十项）

| 发现 | primary evidence | 本轮处置 |
| :--- | :--- | :--- |
| Gamba | [CVPR 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Gamba_Mamba-based_graph_convolutional_network_with_dynamic_graph_topology_learning_CVPR_2026_paper.html) 指向 [official@8947e3b](https://github.com/RCEricZhou/Gamba/tree/8947e3b71dede5ac67ee4f0181d1f77791d9b169)；实际 tree 仅 `.gitignore`/README，无模型/许可 | 排除当前代码未释放；不能用主会接收代替真实 open-source core |
| CAFE | [Ke 的 CVPR2026 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Ke_Content-Aware_Frequency_Encoding_for_Implicit_Neural_Representations_with_Fourier-Chebyshev_Features_CVPR_2026_paper.html) 与 [official@e4bfb33](https://github.com/JunboKe0619/CAFE/tree/e4bfb3309f81f7400a6dc18f8e38d464867c9fba) 有真实 `model.py` | HOLD：root 无 LICENSE；本轮不计 open-source10。若以后有许可，只可将频率/特征乘积作用于 learned IMU token，不允许 test-time 真值拟合或只用时间坐标预测 |
| RS-SSM | [CVPR 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_RS-SSM_Refining_Forgotten_Specifics_in_State_Space_Model_for_Video_CVPR_2026_paper.html) 与 [official@7822631](https://github.com/zhoujiahuan1991/CVPR2026-RS-SSM/tree/7822631600c01ffd4eb01bbc3e28a7eddce2c74f)；root 无 LICENSE，实际模型目录位于 `utils/models`，非 root | HOLD：本轮未把 CwAP/FGIR 的完整 license/core 链核成合格；**不声称模型一定未发布**。third-party MMCV 的许可不自动授权全部新增核心 |
| MambaLiteUNet | [CVPR 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html) 与 [真实模型@09d6f25](https://github.com/maklachur/MambaLiteUNet/blob/09d6f25b9a1c733c5615ebc0c7748a3f1cc4a950/models/MambaLiteUNet.py) | HOLD：无 root LICENSE；默认 LGFM 要求平方 token 数，T200 不能直接照搬 |
| PMRNet | [CVPR 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html) 与 [真实 PMRNet.py@5be809c](https://github.com/KangBoce/PMRNet/blob/5be809cd0df183ff4f139119b2aedc6742815aeb/PMRNet.py) | HOLD：repo 只有模型文件、无 LICENSE。feature-space 的 Hamiltonian/RG/entropy-inspired 命名不是 IMU 真实动力学 |
| SAT | [官方仓库](https://github.com/PhuTran1005/SAT) 明确自标 CVPR 2026 Findings | 排除 Findings，不因 MIT 或实际代码而放宽主会限制 |
| Ga4DPF | [CVPR 主会页](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_4D_Local_Modeling_Toward_Dynamic_Global_Perception_for_Ambiguity-free_Rotation-Invariant_CVPR_2026_paper.html) 与 [MIT 实际 RINet@8170d48](https://github.com/jiaxunguo/ga4dpf/blob/8170d48848ec2a26dc1f9cfb07047de7edec3825/models/rinet.py) | 不 selected：源码依赖点坐标/法线/rotation metadata/Bingham 目标；硬当 gyro/acc 几何或只用 RI 输出会丢失世界系速度方向，不能靠额外 pose/监督补洞 |

TMambaDet/FoSS 主会论文已发现，但本轮没有核到完整 author-code/commit/license 链，不 selected；
未把二手索引或同名 GitHub 搜索结果当官方来源。ShiftLUT 的 root LICENSE 未核到，SLiM/GeoSemba 同样未核成完整许可链，均不计数。

## 4. 静态结论与待执行项

本组贡献 **2 个独立、licensed、真实 paper-specific core 的来源候选**：CSaN ViT 路径和 MixerCSeg TransMixer 路径。
接受页、完整原 PDF、author-repo 关系、固定 HEAD、许可和上述模型文件均已读/哈希。
CVF 直接浏览曾返回 403；最终从相同 publisher URL 加 `?download=1` 读到完整 HTML/PDF，未把截断 PDF 当完整。
原始源码只在内存中做文档/AST 检查，没有 vendor 到仓库；两个新文件是证据与接口草案。

下一步由 Root 做独立 source/fidelity 复审并冻结 adapter recipe；所有 G1/G2、容量、三 seed 训练与新矩阵准入均未开始。
本轮新增 accepted 方法和正式矩阵 rows 都为 **0**。
