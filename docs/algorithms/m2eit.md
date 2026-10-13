# M2EIT（`m2eit` / `m2eit_3d`）

> 公共集成状态：独立论文重实现，两配置已在 b639 / 443 文件 source 完成 CPU 与原生导出验收；CUDA / GPU batch128 容量待验。运行中的 common-benchmark-v1 仍使用旧31架构/53配置，未加入本方法。

## 一手来源与忠实性

[Li et al., *M2EIT: Multi-Domain Mixture of Experts for Robust Neural Inertial Tracking*, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_M2EIT_Multi-Domain_Mixture_of_Experts_for_Robust_Neural_Inertial_Tracking_ICCV_2025_paper.html)；[完整论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_M2EIT_Multi-Domain_Mixture_of_Experts_for_Robust_Neural_Inertial_Tracking_ICCV_2025_paper.pdf)及[补充材料](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Li_M2EIT_Multi-Domain_Mixture_ICCV_2025_supplemental.pdf)。作者为 Yan Li、Yang Xu、Changhao Chen、Zhongchen Shi、Wei Chen、Liang Xie、Hongbo Chen、Erwei Yin，均未命中仓库的 Shanshan Zhang 排除身份。

没有核验到作者发布的 M2EIT 网络源码。本实现依据论文和补充材料独立重建，fidelity 为 `paper-only`，不能称为完整官方代码迁移或原论文精度复现。Mamba 是标准数学部件的来源，不能当作 M2EIT 作者源码。原始 [staging 规格卡](../../research/staging/algorithm_ports/m2eit/source-card.md)保留历史原文，其“尚未运行”只描述当时草稿；历史 staging 测试与新公共包验收分别记录，不覆盖旧证据。

## 输入、输出与三维适配

| 项目 | 论文与当前实现边界 |
|---|---|
| 窗口 | 六轴 IMU，200 样本；统一存储 200 Hz；没有把多专家结构换成单个通用 ResNet |
| 坐标与通道 | IPB `gravity_world`、外部 `[gyro,acc]`；模型内部交换为 `[acc,gyro]`，raw 不改写 |
| 姿态 | 当前 `orientation=reference`，因此是 privileged / oracle 输入。论文 RIDI/IMUNet 用估计姿态；RoNIN train/val 采用 20° 末端误差选择器、test 用估计姿态。当前未复现这个选择器，不能宣称可部署姿态等价 |
| 任务 | 窗口级平均速度，原始输出 `(B,D)`、单位 m/s、无输出尺度补偿；窗口内目标定义属于显式 IPB 映射 |
| 维度 | `m2eit` 是二维参考配置；`m2eit_3d` 的第三输出与真实 Z 标签学习，标记 `3d_adapted`，不是二维补零或原论文三维结果 |
| XY 辅助 | XYZ run 的 `xy_*` 来自同一预测 `[:, :2]`；专门二维配置是另一个声明的 run |
| 不确定性 | router 权重和 expert features 是诊断量；没有 `logstd`、协方差或可校准 σ，Gaussian NLL / NEES / coverage 不可用 |

## 方法核心与目标函数

空间专家包含六个 residual BasicBlock；时间专家使用两层双向 selective SSM；频率专家使用两层 Haar 残差分解。时间与频率分支分别以空间特征作 key/value，进行四头 cross-domain attention。三个专家经时间均值池化得到 $e_0,e_1,e_2$，式 16–18 对应

$$
\delta=\operatorname{softmax}(W_r[e_0;e_1;e_2]+b_r),\qquad
e=\sum_{j=0}^{2}\delta_j e_j,\qquad
\hat v=W_3\operatorname{ReLU}(W_2e+b_2)+b_3.
$$

式 19 的矢量平方误差用 `mse_sum`：对有效样本的 $\|\hat v-v\|_2^2$ 求平均，保留各轴相加，而不是悄悄按输出维数重新缩放。缺少作者训练代码时，不把 reduction 解释宣称为逐字节原实现等价。

selective recurrence 保留输入依赖的 $\Delta,B,C$ 和负对角 $A$，独立参考实现为

$$
s_t=\exp(\Delta_t A)\odot s_{t-1}+\Delta_t B_t x_t,\qquad
y_t=C_t s_t+D x_t.
$$

该显式循环保留 autograd；它是数学参考，不证明官方 compiled Mamba kernel 的精度或吞吐等价。前向、反向两支独立扫描，再恢复反向时间次序。

## 原配方与 unified 配方

论文报告 Adam、batch 128、lr $10^{-4}$、最多 300 epochs、patience 10、200 样本窗口；正文写每 5 epochs 衰减 10%，但作者 scheduler 实现未核验。论文在 RIDI、RoNIN 和 IMUNet 初版上评价，划分和姿态权限并非本仓库统一分组协议。

当前两个配置只提供 `recipes.unified`，不提供可宣称作者等价的 `official` recipe。[公开统一协议](../protocols/common-benchmark-v1.json)记录旧正式矩阵的 AdamW、lr $10^{-3}$、30/min15/patience10、train stride 200 / eval stride 10 等参数；新方法采用同类统一预算前仍需下一版本来源、任务视图、容量及正式 protocol 独立准入。旧协议已运行不等于本方法已正式训练。

## 公开重建假设

- 空间宽度 64/128/256、stage stride 与投影；独立 256 通道序列 embedding；回归隐藏宽度 128。论文未充分给出这些形状，作者参数量等价未核实。
- Haar 使用标准 approximation=`(even+odd)/sqrt(2)`、detail=`(even-odd)/sqrt(2)`。正文滤波器高低频标签与该标准命名有歧义；残差 band 沿时间拼接恢复长度是当前声明选择，原文没有明确 concat axis。
- 时间分支 state size 16、卷积核 2、两层及 attention 四头来自补充材料；Mamba expansion 2、dt 初始化和普通可微 selective scan 是公开实现假设。AMP/compiled kernel 性能另行验收。
- 正文的 S 符号存在输入/空间特征混用；当前各专家有独立 embedding 并通过 attention 对齐，未伪称唯一作者拓扑。

## 验收边界

新公共包 `m2eit` / `m2eit_3d` 已在独立443文件source通过实际CPU验收。两方法共同31个科学节点全部passed/0failed/0skipped；四配置各使用默认完整宽度、200@200Hz、B2，在三份不同已准入RIDI截段上走标准NIO train/val/predict，真实两次AdamW更新、checkpoint和原生export/import/predict重载。原始输出、轨迹和指标精确重放；DiffusionIMU保留噪声并重放同RNG。批次70.083s，source SHA `b639c69773b92ea11c6e999e9d7c5f88702c5bd9df140adc5ce67f95779f6b79`，scope及完整parent/JUnit/分配置SHA见[验收记录](../DEV_V1_INTEGRATION.md#新增两方法实际验收与旧正式范围)。

旧53配置与新4配置的CPU证据各自绑定其源码，合计57个当前配置具备对应准入；不能宣称同一批57tests。历史staging数学节点及两方法CPU生命周期保留原15文件字节，既不覆盖新批也不改历史证据。旧49神经CUDA证明不包括新4；本方法CUDA/GPU batch128容量及新正式训练仍pending。当前2,067正式单元的source/recipe/身份保持不变，2,223仅未来inventory。

工程合同通过不能消除论文未公开细节、declared adaptation或原论文精度复现边界。Haar拼接、宽度、embedding与compiled scan差异仍保持上述假设。
