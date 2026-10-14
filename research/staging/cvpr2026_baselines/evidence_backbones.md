# CVPR 2026 主会跨域候选：视觉骨干、attention 与卷积

状态：2026-10-14 的来源核验与适配设计；**不是已实现、已准入或已训练 baseline**。
本组交付五篇独立论文候选。只阅读官方论文和作者代码，不导入模型、不跑任何本地/远端
CPU/GPU 模型测试，不注册模型，不改变原 2,067 cells。新增候选的执行必须排在原任务之后。
机器化身份、原字节文件 SHA256 与许可记录见 [candidate_backbones.json](candidate_backbones.json)。

## 核验方式与统一适配边界

已直接读取 [CVF 2026 主会 accepted index](https://openaccess.thecvf.com/CVPR2026?day=all)
并核对下列每篇 `/content/CVPR2026/` 页面、主会 PDF、作者仓库和实际核心源码；
不以 CVPR2026W、CVPR2026F、旧年论文或第三方解读替代。部分 `web.open` 返回 403，
使用官方服务器 HTTP 原字节及已有 PDF 文本读取库继续核验；未安装依赖。
仓库 commit 由只读 `git ls-remote HEAD` 固定，源码从该 commit 的官方 raw URL 读取、哈希与静态分析；
没有下载数据、权重或整个仓库 archive。公开核心与许可存在，不等于已验证可运行或已完成法律发布审查。

以下均是**我们的新适配设计**：固定有序 `B×6×200` 输入，200 Hz，一秒 IMU 窗口，
通道 `[gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z]`；输入 frame/orientation/gravity 权限必须与对照组一致并显式标识。
输出为 `B×3` 的真实 XYZ 平均速度（m/s），一次预测派生 XY；不补 Z，不添加真值速度、位置、未来窗口或测试标签。
共享外层监督 MSE、从零初始化和预算；不调用视觉预训练或教师模型。若保留论文专用辅助 loss，
必须单列 recipe，不能继续称作相同 MSE-only 对照。这里只迁移论文的可辨识核心，
**没有一项可称为原作者官方惯导实现或论文结果复现**。
窗口内全上下文是 noncausal；不得默默升级为跨窗口有状态在线算法。

| 候选 | 可保留的独立核心 | 许可 | 主要适配风险 |
| :--- | :--- | :--- | :--- |
| ViT³ | 输入内 key/value 驱动的快权重更新，非线性与局部卷积双分支 | MIT | 2D 卷积与位置定义需要 1D 化；inner loss 不等于外层任务 loss |
| Edge-RecViT | token 递归深度路由、head/shared-middle/tail 参数共享 | Apache-2.0 | 发布代码与论文的 ranker/loss 不同；视觉“边缘”不能直接视作 IMU 运动事件 |
| PFGNet | 多尺度 peripheral 卷积、局部频率线索 gating、中心抑制 | Apache-2.0 | 空间滤波改为时间滤波；不能宣称学到的任意核必然是 band-pass |
| BinaryAttention | 缩放 sign-Q/K、learnable bias、可选 P/V fake quantization | Apache-2.0 | 发布 reference 用浮点 matmul；二值信息损失与从零训练未验证 |
| UCAN | 局部窗口 attention、Hedgehog/global-channel 双融合、同窗口跨层共享 | Apache-2.0 | attention map 共享不是全部权重绑定；2D 窗口/大核的 1D 化须明确 |

## 1. ViT³: Unlocking Test-Time Training in Vision

作者：Dongchen Han, Yining Li, Tianyu Li, Zixuan Cao, Ziming Wang, Jun Song, Yu Cheng,
Bo Zheng, Gao Huang。CVPR 2026 主会 pp. 51–61：[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Han_ViT3_Unlocking_Test-Time_Training_in_Vision_CVPR_2026_paper.html)、
[完整论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_ViT3_Unlocking_Test-Time_Training_in_Vision_CVPR_2026_paper.pdf)。
论文将上下文压缩为一个输入内更新的 inner model；最终核心采用 full-token 单次 inner 更新，
结合 simplified SwiGLU 与 depthwise convolution。不要把它解释为使用测试标签做目标任务微调。

作者仓库 [LeapLabTHU/ViTTT@36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f](https://github.com/LeapLabTHU/ViTTT/tree/36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f)，
[MIT LICENSE](https://github.com/LeapLabTHU/ViTTT/blob/36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f/LICENSE)
保留 Microsoft Corporation 的上游 copyright，不能改写成自有授权。
实际最小核心为 [ttt_block.py](https://github.com/LeapLabTHU/ViTTT/blob/36eb23bfe2eca5cd4ad9ce01ef93bf8a89c6020f/ttt_block.py)：
`TTT`, `inner_train_simplified_swiglu`, `inner_train_3x3dwc`, `forward`。

- **Keep**：Q/K/V 投影、逐样本/逐 head 的局部快权重、closed-form negative-dot-product 梯度、
  梯度归一化/裁剪、lr=1 的一步更新、更新权重作用于 Q、两个 inner 分支及输出投影。
- **Replace / omit**：RGB patch stem → 六轴时间 patch stem；3×3 DWConv → 三 tap temporal DWConv；
  2D RoPE → 明确的时间位置规则；图像分类/检测/生成 head → temporal pooling + 一个 XYZ linear head。
  原 scale 以九个空间邻域位置定义；1D 三邻域 scale 必须作为有身份的改动，不隐藏为同一模型。
- **已核实的隔离**：`forward` 的 k/v 全由当前 x 计算；inner methods 返回局部 `w1/w2/w3`，
  不对 `self.w1/self.w2/self.w3` 原地赋值，不用外部 optimizer，不接收 velocity target。
  每次 forward 从 base parameters 重新生成，conv 分支使用 `groups=B*d` 隔离样本；没有跨窗口 carry。
  这是当前源码的静态事实，后续仍须验证顺序不变性、batch 分解等价和参数不变性。
- **科学风险**：TTT 的 input-only inner 更新不是监督泄漏，但须单列计算成本；它是窗口内非因果机制，
  不能读后一窗口。外层速度 MSE 与 input-only negative-dot / fast-weight inner objective 是两个层次，不能混称“测试集训练”。

## 2. Edge-RecViT: Efficient Vision Transformer via Semantic-Refined Dynamic Recursion

作者：YiZhou Li, Jinyi Xu, Mingyu Yin, Xianyi Zhao。CVPR 2026 主会 pp. 12987–12996：
[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.html)、
[完整论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.pdf)。
论文的独立机制是 token-specific 递归深度与 head/shared-middle/tail 的协同，而不是普通 DeiT 换头。

作者仓库 [RicardoLee510520/Edge-RecViT@9d0d34a8b3029d62955c9a1cf54ff7e4409ff934](https://github.com/RicardoLee510520/Edge-RecViT/tree/9d0d34a8b3029d62955c9a1cf54ff7e4409ff934)，
[Apache-2.0 LICENSE](https://github.com/RicardoLee510520/Edge-RecViT/blob/9d0d34a8b3029d62955c9a1cf54ff7e4409ff934/LICENSE)。
实际核心为 [modeling_mor_vit.py](https://github.com/RicardoLee510520/Edge-RecViT/blob/9d0d34a8b3029d62955c9a1cf54ff7e4409ff934/optimum/mor_vit/modeling_mor_vit.py)
和 [token_choice_router.py](https://github.com/RicardoLee510520/Edge-RecViT/blob/9d0d34a8b3029d62955c9a1cf54ff7e4409ff934/optimum/mor_vit/token_choice_router.py)，
并非仅训练脚本或空项目页。

- **Keep**：一个有序 temporal token 流、独立 head/tail 和真正共享的 middle block、token-wise depth router、
  CLS 全深度路径和 eval argmax；训练态须显式保留源代码的 differentiable routing 机制。
- **Replace / omit**：视觉 patch/位置 → 六轴 1D patch/时间位置；classifier → 一层真实 XYZ velocity head；
  禁止自动下载 DeiT 权重，关闭非必要 DoRA/warmup/pretrained wrappers。
  专用分类 confidence/loss 和 ranker regularizers不并入共享 MSE；需要它们时另列 auxiliary-recipe。
- **必须公开的 paper/code 差异**：论文 §3.1 的 EARR 是两层 MLP，发布 router 是三层（GELU + ReLU）；
  主模型训练路径另接 Gumbel-Softmax straight-through，而 router 自身训练分支含 multinomial；
  发布 balancing loss 是 variance hinge，不是论文 §3.3 的 expected/realized DU 公式。
  因此应先称“官方发布代码核心的跨域 port”，不声称精确重现论文训练配方。
- **科学风险**：没有视觉监督时 temporal event routing 是否有益未知；路由塌缩需监测。
  参数共享减少参数不保证减少实际运算；未测得真正稀疏执行/速度，不能沿用论文 FLOPs 或 latency。
  CLS 与其他候选 GAP 的读出差异是混淆因素，需要 matched-pooling 消融。

## 3. PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning

作者：Xinyong Cai, Changbin Sun, Yong Wang, Hongyu Yang, Yuankai Wu。CVPR 2026 主会 pp. 38848–38858：
[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.html)、
[完整论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cai_PFGNet_A_Fully_Convolutional_Frequency-Guided_Peripheral_Gating_Network_for_Efficient_CVPR_2026_paper.pdf)。
原任务是视频/网格的未来帧预测；频率线索来自局部 derivative/Laplacian/variance，**不是 FFT 模型**。

作者仓库 [fhjdqaq/PFGNet@9db02476d14bfc457558ebd7fd0b2d68f47dd6fa](https://github.com/fhjdqaq/PFGNet/tree/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa)，
[Apache-2.0 LICENSE](https://github.com/fhjdqaq/PFGNet/blob/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa/LICENSE)。
实际核心：[pfg_model.py](https://github.com/fhjdqaq/PFGNet/blob/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa/openstl/models/pfg_model.py)、
[pfg_modules.py](https://github.com/fhjdqaq/PFGNet/blob/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa/openstl/modules/pfg_modules.py)、
[layers/pfg.py](https://github.com/fhjdqaq/PFGNet/blob/9db02476d14bfc457558ebd7fd0b2d68f47dd6fa/openstl/modules/layers/pfg.py)。

- **Keep**：MSInit 多尺度分支；PFG/PFGA 的 K={9,15,31} peripheral response、
  learnable tanh(beta) 中心抑制、三线索驱动的逐位置 softmax scale fusion、GLU channel mixing、
  源 GRN/GroupNorm/LayerScale 与两段 residual。
- **Replace / omit**：空间 Sobel x/y → 一个时间一阶差分，Laplacian → 时间二阶差分，
  spatial local variance → 三 tap temporal variance；DWConv2d → DWConv1d，明确原双空间轴不复用为六轴邻接。
  移除 SimVP 的 T*C packing 和 frame reconstruction decoder，以 `B,D,T` temporal backbone + GAP/XYZ head 替换。
  原 spatial-to-temporal 转译以及 pooling 都是新 adapter，不是 PFG_Model 原任务等价变换。
- **科学风险**：中心抑制可能削弱惯性低频偏置/重力线索，差分可能放大传感器噪声；必须保留对照与实测频率响应。
  任意学习卷积核不天然满足论文条件性 band-pass 论证，不能据模块命名宣称已消除噪声或轨迹缩短。
  源 GRN 是 `x + gamma*x/(L2_spatial(x)+eps) + beta`，不能替换成另一种 ConvNeXt GRN 却声称无改动。

## 4. BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers

作者：Chaodong Xiao, Zhengqiang Zhang, Lei Zhang。CVPR 2026 主会 pp. 12106–12117：
[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html)、
[完整论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.pdf)。
其独立核心是 Q/K 的 scaled sign representation 与 bias compensation，而不是一个新命名的 vanilla ViT。

作者仓库 [EdwardChasel/BinaryAttention@a8e0268ffc63ed4bb26bceac5fa015d358732055](https://github.com/EdwardChasel/BinaryAttention/tree/a8e0268ffc63ed4bb26bceac5fa015d358732055)，
[Apache-2.0 LICENSE](https://github.com/EdwardChasel/BinaryAttention/blob/a8e0268ffc63ed4bb26bceac5fa015d358732055/LICENSE)。
真实模型为 [models.py](https://github.com/EdwardChasel/BinaryAttention/blob/a8e0268ffc63ed4bb26bceac5fa015d358732055/models.py)
的 `Attention`/`Block`，quantizer 在 [utils.py](https://github.com/EdwardChasel/BinaryAttention/blob/a8e0268ffc63ed4bb26bceac5fa015d358732055/utils.py)。

- **Keep**：Q/K per-head token/channel abs-mean scale、sign 与源 straight-through rule、softmax、bias enhancement、
  P/V fake quantization 的独立显式开关、residual Transformer 骨干。
- **Replace / omit**：图像 patch stem → 有序 1D temporal patch；2D relative bias → 1D 时间距离 bias（含 CLS 三类关系）；
  分类/teacher distillation → 一层 XYZ 速度 head 与监督 MSE，预训练/teacher 均不使用。
  不把 gyro/acc 原始物理输入直接二值化；量化发生在已投影 Q/K。
- **科学风险**：官方公开 `Attention.forward` 用 dequantized tensors 的浮点 `@`，不是论文 bit-packed CUDA kernel。
  应命名 reference/fake-quant architecture profile，不能引用论文二倍 kernel speed 或 BOP/64 作为本 adapter 实测 FLOPs。
  论文主要微调/蒸馏结果不证明无教师、从零速度回归效果；幅值信息主要经 V/残差保留，是否足够须测。
  `SymQuantizer` 的 forward 量化与 backward clipping 语义要原样记录，不能用简单 `sign` 零梯度代替。

## 5. UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution

作者：Cao Thien Tan, Phan Thi Thu Trang, Do Nghiem Duc, Ho Ngoc Anh, Hanyang Zhuang, Nguyen Duc Dung。
CVPR 2026 主会 pp. 23409–23418：[CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_UCAN_Unified_Convolutional_Attention_Network_for_Expansive_Receptive_Fields_in_CVPR_2026_paper.html)、
[完整论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_UCAN_Unified_Convolutional_Attention_Network_for_Expansive_Receptive_Fields_in_CVPR_2026_paper.pdf)。
原任务为图像超分；独特组合是局部窗口、global/channel 双融合与 sharing/receiving 组，而不是仅复用旧 Hedgehog 模块。

作者仓库 [hokiyoshi/UCAN@b812c0adb8ce70d442973db5c87bab7d1e8c3333](https://github.com/hokiyoshi/UCAN/tree/b812c0adb8ce70d442973db5c87bab7d1e8c3333)，
[Apache-2.0 LICENSE](https://github.com/hokiyoshi/UCAN/blob/b812c0adb8ce70d442973db5c87bab7d1e8c3333/LICENSE)。
实际核心为 [basicsr/archs/ucan_arch.py](https://github.com/hokiyoshi/UCAN/blob/b812c0adb8ce70d442973db5c87bab7d1e8c3333/basicsr/archs/ucan_arch.py)：
`HedgehogFeatureMap`, `SDFL`, `DFRL`, `WindowsAttention`, `SharedWindowsAttention`, `HybridBlock`, `LKSA`。

- **Keep**：local window attention；Hedgehog 正负对称 feature map + Fourier position embedding 的 global branch；
  channel attention branch；同窗口内 sharing/receiving 的 attention map/QK 复用；large-kernel selective channel processing。
  对照 source 的 feature-map softmax、normalizer、residual/LayerScale，不能随意换成 vanilla linear attention。
- **Replace / omit**：2D patch/windows/卷积 → 1D 时间 patch/windows/卷积，显式定义 padding、mask 和 temporal dilation；
  RGB mean、SR PixelShuffle/reconstruction、L1/LDL/wavelet loss → 六轴 stem、GAP、一层 XYZ head、MSE。
  不把六个异质传感器轴伪装成图像宽度。共享 map/QK 仅允许本次窗口的层间传递，不跨 forward 保存。
- **科学风险**：源码的 sharing/receiving 主要复用 map/QK，并不是全层参数统一绑定；不要与 Edge 的 middle-weight tying 混称。
  原 2D FlashAttention 速度和 SR 图像质量不能搬到 T200；大核可能在一秒窗口中过度平滑。
  Fourier position embedding 不等于对 IMU 施加 FFT 去噪。

## 排除与后续判据

- [Ego 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_Convolutional_Neural_Networks_Driven_by_Content_Similarity_CVPR_2026_paper.pdf)
  指向的 `https://github.com/essenceoftheworld/ego` 在本次只读访问和 `git ls-remote` 中不可取得（repository not found）；
  未确认公开核心与许可，**不计入这五项**，不据第三方笔记认定开源成立。
- [FCDM 作者仓库](https://github.com/star-kwon/FCDM) 有模型代码但采用 CC-BY-NC；本轮不优先于五个 permissive 候选，
  其视觉 generative/pretrained VAE 路径也不构成本任务可直接使用的监督 baseline。

五项后续共同判据：独立 adapter 来源卡 + 每一改动的 rationale、严格 InputSpec、单一真实 XYZ head、
共享 loss/seed/window/数据曝光预算、源码不注册、source/active guards 不变。模型容量、读出策略与 norm 的差异
需要单独报告；MSE-only 或原方法辅助 loss 必须预先分 track。先做静态结构审查，原任务完成后才安排有收据的
CPU lifecycle/科学反例和 CUDA 容量，最后才可批准 train/val pilot、三 seed 与独立矩阵版本。
此文件不提供任何精度、稳定性、加速、惯性可观测性或正式准入 GO。
