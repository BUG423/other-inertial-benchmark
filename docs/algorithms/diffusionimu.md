# DiffusionIMU（`diffusionimu` / `diffusionimu_3d`）

> 公共集成状态：独立论文重实现，两配置已在 b639 / 443 文件 source 完成 CPU 与原生导出验收；CUDA / GPU batch128 容量待验。运行中的 common-benchmark-v1 仍使用旧31架构/53配置，未加入本方法。

## 一手来源与忠实性

[Teng et al., *DiffusionIMU: Diffusion-Based Inertial Navigation with Iterative Motion Refinement*, IJCAI 2025 main](https://www.ijcai.org/proceedings/2025/977)，[完整论文](https://www.ijcai.org/proceedings/2025/0977.pdf)，DOI `10.24963/ijcai.2025/977`。作者为 Xiaoqiang Teng、Chenyang Li、Shibiao Xu、Zhihao Hao、Deke Guo、Jingyuan Li、Haisheng Li、Weiliang Meng、Xiaopeng Zhang，未命中仓库 Shanshan Zhang 排除身份。

没有核验到作者网络代码。本实现是依据正文式 12–22 的 `paper-only` 独立重建，不以第三方实现冒充官方源码，也不把背景 DDPM 的噪声预测目标替代方法自己的多任务目标。[原 staging 规格卡](../../research/staging/algorithm_ports/diffusionimu/source-card.md)按当时原文保存，新包验收不会改写它的历史状态。

## 输入、输出与 XYZ

两个配置采用 IPB 200 Hz、200 样本、`gravity_world`、窗口级 `avg_velocity`，输入外部 `[gyro,acc]`、内部交换为 `[acc,gyro]`，加计保留重力。当前通过 task view 消费参考姿态，属于 privileged / oracle 权限；正文式 10 提及姿态，却没有完整定义单独姿态特征编码。

论文没有唯一明确窗口目标、最终输出轴数及全部 temporal pooling。`diffusionimu` 是本仓库声明的二维参考；`diffusionimu_3d` 的输出头真实学习 XYZ，标记 `3d_adapted`，不能称作者原始三维模型。原始速度是 `(B,D)`、m/s，不增加输出尺度校正。XYZ 的 XY 辅助指标来自同一个预测的 `[:, :2]`；独立二维配置不重复计为另一篇论文。

## 方法与损失

encoder 使用 local/global attention 改造的八个 bottleneck。噪声、对齐及迭代步骤对应式 12–20：

$$
z\sim\mathcal N(0,I),\qquad f_{NK}=\alpha h+(1-\alpha)z,\qquad
L_{align}=\|Wh-f_{NK}\|_2^2,
$$

$$
t\sim U\{0,\ldots,99\},\quad f^{(0)}=\operatorname{ReLU}(W_e f_{NK}+E(t)),\quad
f^{(k)}=\operatorname{ReLU}(W_{d,k}f^{(k-1)}+b_k),\quad k=1,2,3.
$$

最终 temporal mean pooling 后得到速度 $\hat v$ 和正值头 $u=\operatorname{Softplus}(W_u f)$。式 21–22 以 `diffusionimu_multitask` 实现：

$$
L=\operatorname{mean}_{\text{valid samples}}
\left[\|\hat v-v\|_2^2+\tfrac12\|u\|_2^2+\|Wh-f_{NK}\|_2^2\right].
$$

速度按每样本矢量平方范数求均值；对齐是每样本未按 feature/time 大小归一化的平方范数。改成逐分量平均会改变多任务权重，改为 Gaussian NLL 或 DDPM noise loss 会改变科学算法，必须另命名消融，不能当作实现补全。

## u 与随机推理的科学边界

`u` 保存为 `uncertainty_regularizer_head`，不是 `logstd`、σ 或协方差。当前 $u^2/2$ 惩罚没有 residual 耦合，能把辅助输出推向零；论文没有给出可用于 Gaussian NLL / NEES / coverage 的概率模型或单位，因此这些不确定性指标保持不可用。不能因为 Softplus 为正就标作已校准不确定度。

训练每窗采样 timestep；当前评估固定 `evaluation_timestep=0`，仍执行 fresh Gaussian `randn`，没有把噪声改成零来制造确定性。评估 timestep、$\alpha=\operatorname{sigmoid}(a)$ 且初始 0.5 是公开重建选择。复现一次预测需同 checkpoint、RNG 状态/seed、draw 次序、batch 排列、timestep 和 refinement 步数；保存重载/原生导出比较必须重放相同 RNG。相同权重在不同噪声 draw 下的数值差异不能自动判为 checkpoint 失败。

Native export 保留整个科学预测合同；TorchScript/ONNX 当前没有这个方法的专用审查 adapter，不从 RoNIN/IMUNet 的格式通过记录继承支持。

## 原配方、统一配方与假设

论文报告 Adam、batch 128、lr $3\times10^{-4}$、dropout 0.2、hidden 128、三次 refinement、100 个时间 embedding；总 epoch、scheduler、早停策略没有充分说明。当前配置只提供 `recipes.unified`；[公开 common-benchmark-v1](../protocols/common-benchmark-v1.json)是旧 53 配置正在执行的统一预算，不是本方法已经开始的实验记录。下一版本必须另做 source / taskview / 容量 / 正式 protocol 准入。

未由论文唯一决定的选择公开如下：八个 attention bottleneck 与 64/128/256/512 宽度、四头/radius-three local attention、stem/stride、temporal mean pooling、200 样本平均速度目标、姿态 task view、sigmoid 混合系数初值和 eval timestep。源码能运行也不消除这些拓扑及语义不确定性。

## 验收边界

新公共包 `diffusionimu` / `diffusionimu_3d` 已在独立443文件source通过实际CPU验收。两方法共同31个科学节点全部passed/0failed/0skipped；四配置各使用默认完整宽度、200@200Hz、B2，在三份不同已准入RIDI截段上走标准NIO train/val/predict，真实两次AdamW更新、checkpoint和原生export/import/predict重载。原始输出、轨迹和指标精确重放；DiffusionIMU保留噪声并重放同RNG。批次70.083s，source SHA `b639c69773b92ea11c6e999e9d7c5f88702c5bd9df140adc5ce67f95779f6b79`，scope及完整parent/JUnit/分配置SHA见[验收记录](../DEV_V1_INTEGRATION.md#新增两方法实际验收与旧正式范围)。

旧53配置与新4配置的CPU证据各自绑定其源码，合计57个当前配置具备对应准入；不能宣称同一批57tests。历史staging数学节点及两方法CPU生命周期保留原15文件字节，既不覆盖新批也不改历史证据。旧49神经CUDA证明不包括新4；本方法CUDA/GPU batch128容量及新正式训练仍pending。当前2,067正式单元的source/recipe/身份保持不变，2,223仅未来inventory。

工程合同通过不能消除论文未公开细节、declared adaptation或原论文精度复现边界。正值u仍不是σ/协方差，随机推理仍须同RNG重放；TorchScript/ONNX没有审查adapter。
