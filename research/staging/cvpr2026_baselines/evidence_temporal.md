# CVPR 2026：时序方向 primary-source 候选证据

状态：只读来源与适配审查，未移植、未注册、未测试或训练。以下均不是已验收方法；
冻结 425 source / 53 profiles / 13 sources / 2,067 cells 不变。用户明确要求新增工作在
全部原任务之后执行；本次查证不授予提前启动 runtime 的权限。

本组给出两项主候选、一项有条件非 Mamba 候补。初始“四项时序”的目标后来由 Root
缩小为“再补1–2项实际 licensed 非 Mamba core”；不拿无代码/无许可/失去核心机制的方法凑第四项。
精确书目、publisher PDF/HTML 和 source/license SHA 见 [candidate_temporal.json](candidate_temporal.json)。
没有 vendor 上游源码或下载权重。

## 统一适配契约

输入固定六轴顺序 `gyro_x/y/z, acc_x/y/z`，200 Hz、T200、reference orientation、
gravity-world、保留重力、history0、无extra_inputs。输出真实学习的 XYZ whole-window
平均速度（m/s），XY 从同一 XYZ 输出切片。拒绝图像/文本/关节姿态/动作边界标签、
初速度及额外传感器。采用 observed-window noncausal 声明，不宣称零 lookahead。
所有适配判断都是推断/待实现方案，不能代替 CPU lifecycle、容量或三 seed 结果。

## 1. MS-Temba：主候选

主会与原始方法见 [CVF页面](https://openaccess.thecvf.com/content/CVPR2026/html/Sinha_MS-Temba_Multi-Scale_Temporal_Mamba_for_Understanding_Long_Untrimmed_Videos_CVPR_2026_paper.html)
及其 [publisher PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sinha_MS-Temba_Multi-Scale_Temporal_Mamba_for_Understanding_Long_Untrimmed_Videos_CVPR_2026_paper.pdf)。
官方 [固定commit](https://github.com/thearkaprava/MS-Temba/tree/d42b4606d03884bfe2591d0bed1ed7a1dd01ab22)：
`d42b4606d03884bfe2591d0bed1ed7a1dd01ab22`，根与 `vim/LICENSE` 为 Apache-2.0，
Mamba/Vim/Facebook/timm 来源需保留各自归属。

实际核心是 `vim/models_MSTemba.py` 的 MSTemba/VisionMamba，以及
`mamba-1p1p1/mamba_ssm/modules/mamba_simple_getC.py` 的双向 selective SSM。
保留 dilation1/2/3、独立 phase 参数、重交织、跨尺度投影与 SSM fuser；
I3D/CLIP前端换为有序六通道投影，末端换 temporal pool + Linear→3。
T200在第三尺度需要按源代码补齐至3倍数并恢复，不得混入补齐位置。

论文 Eq9 的 `1-cos(C_i,C_j)` 是一致性/对齐损失，虽代码函数名叫 diversity；
不能反向“纠正”为排斥损失。BCE换平均速度MSE；如保留各尺度辅助监督，
必须先 whole-window pooling，不能把平均速度伪称逐时刻速度标签。
loss权重、定制getC扫描/CUDA扩展和Triton兼容性全部待独立验收。不是 generic Mamba 换头的重命名。

## 2. Spectral Scalpel：主候选，但仅 novel spectral core-port

[CVF主会页面](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Spectral_Scalpel_Amplifying_Adjacent_Action_Discrepancy_via_Frequency-Selective_Filtering_for_CVPR_2026_paper.html)、
[publisher PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_Spectral_Scalpel_Amplifying_Adjacent_Action_Discrepancy_via_Frequency-Selective_Filtering_for_CVPR_2026_paper.pdf)
与官方 [固定commit](https://github.com/HaoyuJi/SpecScalpel/tree/01adb1ed708d410023b8482c52270cac358c8525)
`01adb1ed708d410023b8482c52270cac358c8525` 一致；MIT许可字节已查证。

迁移 `libs/models/SpecScalpel.py` 的 FFT_filter（MASF，117–171行）、
FFT_conv_filter（FACM，65–96行）和可选 linear temporal attention。
保留多尺度频谱滤波、动态/静态融合、real/imag共享点卷积、rFFT/irFFT与残差。
有序六轴投影产生latent序列；MASF的V采用明确singleton feature location，
不伪造人体关节或单位混合的骨架。池化后输出真实XYZ。

原joint graph/text stem、action/boundary分类与refinement、AADL和BERT损失不适用；
首版明确使用MSE，不制造动作段标注。**因此不是完整STAS算法复现**，
应命名“SpectralScalpel MASF/FACM inertial core-port”。2019 MS-TCN helper不是2026新方法，
不能把它单独当新baseline。singleton V、频率插值、mask/tail和完整XYZ梯度均需后续实测。

## 3. TESSERA v1：有条件非 Mamba 候补

[CVF主会页面](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_TESSERA_Temporal_Embeddings_of_Surface_Spectra_for_Earth_Representation_and_CVPR_2026_paper.html)
和 [完整14页 publisher PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Feng_TESSERA_Temporal_Embeddings_of_Surface_Spectra_for_Earth_Representation_and_CVPR_2026_paper.pdf)
已查证；原始PDF17,447,236B，早期16MB截断读取不作完整证据。
只用官方 [ucam-eo/tessera固定commit](https://github.com/ucam-eo/tessera/tree/e368539f7c4711e5f16fdd09801573830e34baf9)
`e368539f7c4711e5f16fdd09801573830e34baf9` 的 `tessera_infer/` v1 MIT核心，
不采用第三方fork、v2或QAT身份。

实际源有两条Transformer encoder、GRU-conditioned attention pooling、
有序concat/reducer与BT projector。可让gyro3和acc3分别进入真实sensor stream，
时间位置仅由既有200Hz窗口内部派生，不增加日历/地理/图像输入。
保留train-only随机采样双view、Barlow Twins、global train-window shuffling，
并将mix-based正则作为明确预注册的候选；不能访问target/test或改变200点target区间。
canonical inference固定完整T200。然后接平均速度XYZ监督头。

**必须解决 paper↔code 差异**：论文写4-layer Transformer、GRU pooling、2-layer融合MLP；
固定v1 builder实际8层，GRU后query attention pool，concat后单linear reducer，
且配置 `apply_mixup=False`。已有encoder/projector是真实代码，但不等于已有适配后的SSL训练loop。
完整SSL身份需要独立预算/recipe与fidelity桥；若删掉所有SSL仅留MSE，只能叫
dual-stream architecture baseline，不得宣称完整TESSERA。默认512宽/8层/FF4096容量尚未测量，
不可静默缩窄或降batch后记GO。

## 严格排除/暂缓

| 项目 | 原因 | 结论 |
| :--- | :--- | :--- |
| HAL | 有实际hierarchical Transformer，但根inventory未见license；不是“public即可复用” | 许可HOLD，不计数 |
| AMR | 固定commit仅LICENSE+82B README，无模型core | 拒绝，不计数 |
| Frame2Freq | 有models_adapter.py，根license未核；完整框架依赖CLIP语境 | 许可HOLD，不计数 |
| LaDy | 核心广义关节坐标、力与energy一致性需要真pose | 拒绝用六轴伪造骨架；不计数 |
| EMMA物理参数方法 | 有MIT实际代码，但去掉ODE/参数objective只剩通用LTC | 不强计完整EMMA |
| Gamba/TMambaDet | Root已分给S4D组 | 不重复 |
| WACV/CVPRW/CVPR2026 Findings | 不满足2026主会 | 不计数 |

## 交给 Root 的后续依赖

先固定独立paper/core-port identity及保留/替换/损失表，完成许可归属与paper↔code复审；
**原任务结束/授权终止处置明确后**，才可安排隔离实现→AutoDL科学CPU/native lifecycle→
真实B128/stride200容量→Owner冻结独立pilot预算→既有三seed pilot。
当前训练、容量、采样、收敛、排名与速度均没有新实测结论。
