# IPB v1 算法总表 / Algorithm Catalogue

> 当前实现和迁移后工程验收：2026-10-14；§2 的原始规格表保留 2026-09-17 历史口径。
> 本文件是 `docs/algorithms/*.md` 规格卡的索引，历史规格机读版本为
> [`docs/algorithms/index.yaml`](algorithms/index.yaml)，当前方法、完整作者、配置和源码来源见
> [`docs/models/provenance.json`](models/provenance.json)。形状夹具锁定已核对的原配置；
> 自主重实现的声明假设、三维适配、工程验收和原论文精度复现分别记录。

当前接入以 `dev/v1` 的 [模型接口](../src/inertial_benchmark/nn/base.py)、[注册器](../src/inertial_benchmark/nn/registry.py)和 [模型配置](../src/inertial_benchmark/cfg/models/)为准；现有实现与本轮服务器验收见 [当前状态](DEV_V1_INTEGRATION.md)。下表的“可移植”等标签保留规格基线含义，算法实现、形状夹具核对与论文精度复现分别记录。

当前实现为 **33 个活跃注册架构、57 份活跃模型配置、23 个独立论文方法与 2 个仓库经典定义**。论文方法中 **21 个已发表、2 个仅预印本**（GNIO、VeloBins）；RoNIN 的骨干深度、IMUNet 的骨干变体、EqNIO/NIO 的两个网络分支及二维/三维配置均不重复增加论文方法数。因此不能把33个架构宣称为33篇独立论文。

57 份配置包括 **53 份神经网络配置、2 份 RIDI 非梯度标定配置、2 份经典序列配置**。CPU准入覆盖旧53与新4各自 source 的真实工程收据；新4完整宽度200样本、B2、两次真实更新、验证、预测、checkpoint与原生导出已通过，不能称为同一批57测试。旧49神经配置完成CUDA工程验收，新4CUDA和GPU batch128容量未测。科学合同、接口通过与充分训练/论文精度分别记录。所有新增补丁需按实际源码独立验收，见[操作说明](../tools/ops_v1/README.md)。

主 benchmark 使用真实三维预测，二维辅助指标从同一 XYZ 预测的 XY 派生；原文二维配置和参数夹具作为独立参考轨保留。当前33份XYZ配置含32份神经网络配置和 `ridi_3d`；24份二维配置含21份神经网络配置、`ridi` 和两个经典参考。运行中旧正式源仍为31XYZ/22XY。原生 XYZ、显式 `3d_adapted`、二维参考以及姿态/初始速度输入权限分别报告，不能以补零 Z 混入三维主榜。

IONext 已按仓库拥有者指定的作者排除规则从注册器、活跃配置和目录移除；一手作者证据及历史源码/夹具留存见 [排除记录](models/excluded-methods.json)。TinyIO、CKANIO、StarIO 同样命中 Shanshan Zhang 提出方法的排除规则，不作为活跃候选。23 个当前论文方法均已逐项核对一手完整作者；不把 MEIO 的 Xuanxuan Zhang 或 RNIN 的 Guofeng Zhang 当作同一作者。

下表列出三维语义的代表，完整主配置覆盖上述 33 份，并不只交付四个模型：

| 三维主协议配置示例 | 已有架构 / 目标 | 忠实性边界 |
|---|---|---|
| `ronin_resnet18_3d` | `ronin_resnet` / XYZ 平均速度 | `3d_adapted`：原二维输出头扩为可学习三维，窗口和 MSE 保留 |
| `ronin_lstm_3d` | `ronin_lstm` / 逐帧 XYZ 速度 | `3d_adapted`：两个输出线性层随三维扩展，GlobalPosLoss 保留 |
| `imunet_3d` | `imunet` / XYZ 平均速度 | `3d_adapted`：噪声层与骨干不变，输出头新增可学习 Z |
| `tlio` | `tlio` / XYZ 位移与 logstd | 原生三维网络；不包含论文 EKF |
| `ionet_3d`、`lionet_3d` | 各自极坐标预测 / XYZ 平均速度 | 水平速度/航向损失保留，额外学习垂直速度；L-IONet 架构细节存在论文未明确的假设 |
| `eqnio_ronin_3d` | O(2) 规范化 / XYZ 平均速度 | XY 保持 O(2) 等变，Z 是独立学习的不变标量 |
| `ridi_3d` | SVM/SVR 与偏置修正双积分 | 垂直 SVR 用真实 XYZ 标签拟合；离线更新日程和参考姿态协议是显式适配 |
| `meio` | 多出口网络 / XYZ 位移和各向同性 logstd | 官方网络文件缺失，自主重建声明假设；不能称完整官方网络复现 |

三维适配配置只提供统一配方，不能以 `recipe=official` 宣称原文二维复现；旧二维 checkpoint 也不能直接改 `dims` 重载。IONet 与 EqNIO-RoNIN 的三维适配已纳入上述真实 CPU/CUDA 工程验收。PDR、`mean_speed_heading` 的 `constant` / `gated` 变体拒绝 `dims=3`；显式 `zero_velocity` 可定义三维零向量，`train_mean_velocity` 可定义训练集 XYZ 均值，但这两种常量基线不属于航向估计三维扩展。它们在完整矩阵的二维参考轨保留，三维主排序不纳入缺少预测 Z 的配置。

### 本轮新增独立论文方法

| 方法 / 一手来源 | 当前实现 | 忠实性与输入边界 |
|---|---|---|
| [L-IONet，IoT Journal 2020](https://arxiv.org/abs/2001.04061) | [`lionet` / `lionet_3d`](algorithms/lionet.md)：8 层因果门控 WaveNet，扩张率 1–128，水平极坐标头与损失 | 无可核验官方网络源码；卷积核、池化等声明假设。统一绝对速度任务改变原相对步进语义，XYZ 为显式适配 |
| [RIDI，ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.html) | [`ridi` / `ridi_3d`](algorithms/ridi.md)：train-only placement RBF SVM、逐轴 SVR、设备系加计偏置最小二乘与双积分 | 官方 [MIT 源码](https://github.com/higerra/ridi_imu)已在服务器固定版本取得；独立实现的离线块更新、窗口拟合上限与参考姿态协议须记录，不宣称原预训练精度 |
| [MEIO-Net，IoT Journal 2026](https://ieeexplore.ieee.org/document/11251007) | [`meio`](algorithms/meio.md)：多出口 XYZ 位移、各向同性不确定性、联合分阶段损失和实际早退 | 官方 [GPLv3 仓库](https://github.com/SunXuehang/MEIO-Net)有训练包装但缺 `NeuralNetwork/EarlyExit/EarlyExitDense.py`；当前是 `paper-core partial`，不是完整官方网络迁移 |

参考姿态或真值初始速度须与设备姿态/纯传感器输入分轨。TLIO/DIVE 的纯网络预测不包含其原 EKF；NIO-LieEvents 当前采用每窗参考初速度，未复现 RoNIN 原有状态递归推理；RIO 的 A-TTT 未作为完整原测试协议复现；VeloBins 是去执行器支路的 IMU-only 结构适配。统一接口和 CPU/CUDA 通过不消除这些科学差别。

## 新增两方法：已完成 CPU / 原生导出验收

M2EIT（ICCV2025）与 DiffusionIMU（IJCAI2025）增加两个独立已发表论文方法，各自 XY/XYZ 配置不增加论文数。公共包四配置已在独立 b639/443 文件 source 通过31科学节点和默认完整宽度 CPU/原生 export/import/predict；新四 CUDA、GPU batch128 容量和正式训练仍 pending。

| 方法 / 一手来源 | 当前配置与输出 | fidelity 与验收边界 |
|---|---|---|
| [M2EIT](algorithms/m2eit.md) | `m2eit` / `m2eit_3d`：三专家、两 cross-domain attention、soft router，平均速度 XY / XYZ | `paper-only`；Haar、widths、selective scan、姿态选择器假设保留；两配置 CPU/native export passed，GPU pending |
| [DiffusionIMU](algorithms/diffusionimu.md) | `diffusionimu` / `diffusionimu_3d`：attention encoder、feature noise、alignment、三步 refinement，平均速度 XY / XYZ | `paper-only`；输出维数/eval policy假设公开；u非σ，NLL/NEES/coverage不可用；同RNG重载 passed，GPU pending |

CPU57由旧53与新4的 exact-source 独立收据组成，不伪称同一批57。新4测试 parent receipt为 `ee3f064e0f8377e9d5299b8da081b8064b6206b83bb8fedb7e667a698e10f66b`；完整source、JUnit与分配置收据见[状态](DEV_V1_INTEGRATION.md#新增两方法实际验收与旧正式范围)。历史staging15源码原字节保留。

运行中 common-benchmark-v1仍为source1897、31架构/53配置/2,067正式单元、XYZ31/XY22，不由本轮新包替换。57×13×3=2,223是未来inventory，新正式协议尚未冻结或启动。旧49CUDA、旧53native export及新4native export是不同收据范围，TorchScript/ONNX不自动支持新方法。

## 1. 约定

- **fidelity**
  - `official`：实例化官方仓库模型（CPU、随机权重、固定种子）导出夹具，卡片内容以官方代码为准，与论文不一致处逐条注明。
  - `paper-only`：没有可用的官方源码，只依据论文写重实现规格。论文未给出的细节在卡片中标为假设，并给出推荐默认值；参数量是按卡片规格推导的，不能作为忠实性证据。
  - `定义`：经典非学习基线，按 IPB 自己的定义实现，只有少量标定标量。
- **参数量**：“官方”指官方发布的配置；“IPB”指按卡片 §6 适配到 IPB 协议（200 Hz、IPB 窗口、必要时改输出维度）后的配置，由夹具的 `total_params` 锁定。两者相同时只写一个数。
- **依赖外部姿态**：指官方**推理**时网络输入或轨迹重建是否需要外部姿态（设备 game RV、VIO 或滤波器姿态）。
- **状态**：`可移植` 表示卡片足以直接实现；`许可风险` 表示仓库无 LICENSE 或为强 copyleft，只能照卡独立实现，不得复制代码或再分发权重；`部分` 表示论文或代码缺少关键细节，卡中有待核实的假设。
- **夹具格式**：必需键为 `name, source_repo, commit, input_shape, output_shape, total_params, trainable_params, param_shapes, notes`。`param_shapes` 按 `named_parameters()` 的注册顺序排列（Keras 模型按 `model.weights` 顺序）。`input_shape`/`output_shape` 在单张量时为列表，多输入/多输出时为“名称 → 形状”的字典。各夹具另带 `param_names`、`buffer_shapes`、`official_config`、`benchmark_config` 等补充键；参数名仅作核对参考，移植实现不必沿用。
- **官方代码来源**：本轮在新 AutoDL 的 ignored `.local/external/reference-code/<repo>/<commit>` 固定版本保存 16 个官方参考仓库的网络/损失/预处理/训练和许可文档，记录 HEAD、逐文件 SHA 和体积；不拉数据集、权重包或完整历史。CTIN/iMoT 仓库只有文档，MEIO 缺核心网络。来源下载不自动提升实现忠实性；TinyOdom 的生成 C 文件可内嵌模型参数，作为隔离参考证据保存，不用于模型初始化或公开再分发。§5 的 `third_party` 路径是原规格审计历史路径。

## 2. 历史原始规格总表

以下按 P0（行人、官方代码可用）→ P1（跨平台或需接口扩展）→ paper-only 重实现 → 经典基线排序，记录原论文/发布代码的规格基线。它不是当前活跃配置或本轮测试状态表；例如 `pedestrian_diffusion` 只有规格卡、未注册，DIVE 旧二维适配留作 `dive_2d`，当前主 `dive` 已恢复可学习 XYZ 输出。L-IONet/RIDI/MEIO 的当前增量见上表。不得从历史“official / 可移植”推出已完整复现原系统。

| 名称 | 优先级 | fidelity | 许可 | 输入（官方） | 输出（官方） | 参数量（官方 / IPB） | 依赖外部姿态 | 官方数据集 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| [`ronin_resnet18`](algorithms/ronin_resnet18.md) | P0 | official | GPL-3.0 | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 4,634,882 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`ronin_lstm`](algorithms/ronin_lstm.md) | P0 | official | GPL-3.0 | 200 Hz×400 序列（batch-first）；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度 | 216,620 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`ronin_tcn`](algorithms/ronin_tcn.md) | P0 | official | GPL-3.0 | 200 Hz×400 序列（batch-first）；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度 | 540,488 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`tlio`](algorithms/tlio.md) | P0 | official | BSD-3 | 200 Hz×200；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd | 5,424,646 | 是 | TLIO | 可移植 |
| [`rnin`](algorithms/rnin.md) | P0 | official | Apache-2.0 | 100 Hz，10×100 子窗；`gravity_world`、去重力；`[gyro,acc]` | 10 步 3D 位移＋logstd | 5,358,342 | 是 | IDOL, RNIN | 可移植 |
| [`imunet`](algorithms/imunet.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 3,661,618 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mobilenet`](algorithms/imunet_mobilenet.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 3,178,978 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mobilenetv2`](algorithms/imunet_mobilenetv2.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 2,183,330 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mnasnet`](algorithms/imunet_mnasnet.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 2,979,114 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_efficientnetb0`](algorithms/imunet_efficientnetb0.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 3,231,906 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`eqnio_ronin`](algorithms/eqnio_ronin.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]`（内部 O(2) 规范化） | 2D 速度（＋规范帧） | 5,230,466 | 是 | RoNIN, RIDI, OxIOD | 可移植（许可风险） |
| [`eqnio_tlio`](algorithms/eqnio_tlio.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_yaw_local`；`[gyro,acc]`（内部 O(2) 规范化） | 3D 位移＋logstd（规范帧内） | 6,020,230 | 是 | TLIO, Aria Everyday | 可移植（许可风险） |
| [`llio`](algorithms/llio.md) | P0 | official | GPL-3.0 | 100 Hz×100（README 注释）；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd | 7,191,654 | 是 | unknown | 部分（假设待核实） |
| [`tinyodom`](algorithms/tinyodom.md) | P0 | official | BSD-3 | 200 Hz×400；机体系原始读数；`[acc,gyro,mag,step_mask]` | 2 s 位移 (x, y) | 102,008 / 100,481 | 否（IPB 适配后仅用于目标与积分） | RoNIN, OxIOD, AQUALOC, EuRoC MAV, GunDog | 可移植 |
| [`deepils`](algorithms/deepils.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 2,290,226 | 是 | RoNIN, RIDI, OxIOD, IMUNet, KIOD, INAIOD | 可移植（许可风险） |
| [`tartanimu`](algorithms/tartanimu.md) | P0 | official | Apache-2.0 | 200 Hz，10×1 s 子窗各抽到 40 点；`body`；`[gyro,acc]`＋平台头 | 逐子窗机体系 3D 速度＋logstd | 5,698,346 | 网络否；积分需要 | SubT-MRS, IDOL, Blackbird, UZH-FPV, TartanDrive 等 | 可移植 |
| [`dive`](algorithms/dive.md) | P1 | official | MIT | 400 Hz×1400；局部重力系、去重力；`[Log(C), acc−g]`＋窗末姿态 | 窗末 3D 速度＋logstd | 12,367,110 / 7,516,420 | 是 | DIDO, Blackbird | 可移植 |
| [`airio`](algorithms/airio.md) | P1 | official | BSD-3 | 200 Hz×1000；`body`＋逐样本姿态编码；`[acc,gyro]` | 逐帧机体系 3D 速度＋方差 | 387,014 | 是 | EuRoC, Blackbird, Pegasus | 可移植 |
| [`nio_lie_events_ronin`](algorithms/nio_lie_events.md) | P1 | official | 无 LICENSE ⚠ | 200 Hz×200 → 12 通道 SE(3) 李事件；需逐样本姿态、v0 | 2D 速度 | 4,637,570 | 是 | RoNIN, RIDI, OxIOD | 可移植（许可风险） |
| [`nio_lie_events_tlio`](algorithms/nio_lie_events.md) | P1 | official | 无 LICENSE ⚠ | 同上，`gravity_yaw_local` | 3D 位移＋logstd | 5,427,334 | 是 | TLIO, Aria Everyday | 可移植（许可风险） |
| [`pedestrian_diffusion`](algorithms/pedestrian_diffusion.md) | P1 | official | AGPL-3.0 ⚠ | 100 Hz×100 → STFT；`gravity_world`；`[acc,gyro]`＋CLIP 文本 | 逐帧 3D 速度与角速度 | 37,259,798 | 是 | RoNIN, RIDI, OxIOD, TLIO | 可移植（许可风险） |
| [`ionet`](algorithms/ionet.md) | paper-only | paper-only | 无代码 | 100 Hz×200；机体系；`[acc,gyro]` | 极坐标 (Δl, Δψ) | 论文未报告 / 推导 302,978 | 否（官方） | IONet 自采 Vicon 集、OxIOD | 可移植 |
| [`ctin`](algorithms/ctin.md) | paper-only | paper-only | 占位仓库（MIT，无源码） | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度＋logstd | 论文 0.557 M / 推导 477,572 | 是 | RIDI, OxIOD, RoNIN, IDOL, CTIN | 部分（假设待核实） |
| [`imot`](algorithms/imot.md) | paper-only | paper-only | 占位仓库（MIT，无源码） | 200 Hz×200；世界系（论文未明示）；`[acc,gyro]` | 2D 速度 | 论文 14.49 M / 推导 7,137,482 | 是（来源未明示） | RIDI, RoNIN, OxIOD, IDOL | 部分（假设待核实） |
| [`rio`](algorithms/rio.md) | paper-only | paper-only | 无代码 | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 速度 | 论文未报告 / 推导 4,634,882 | 是 | RoNIN, OxIOD, RIDI, IPS | 可移植 |
| [`gnio`](algorithms/gnio.md) | paper-only | paper-only | 无代码 | 200 Hz×200（1 s 窗、0.1 s 步长）；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd（门控头） | 论文 4.90 M / 推导 4,934,153 | 是 | OxIOD, RIDI, RoNIN, IDOL, TLIO | 可移植 |
| [`velobins`](algorithms/velobins.md) | paper-only | paper-only | 无代码 | 100 Hz×100；`body`；`[acc,gyro]`＋逐旋翼转速/PWM | 逐轴 512 箱分布 → 机体系 3D 速度＋方差 | 论文 ≈76.6 k / 推导 69,968（无转速支路） | 否（网络）；积分需要 | AI-IO, NanoBench, TII-RATM, NeuroBEM | 部分（假设待核实） |
| [`pdr`](algorithms/pdr.md) | baseline | 定义 | — | 200 Hz 连续序列；加计＋姿态 | 2D 平均速度 | 0（标定标量） | 是 | — | 可移植 |
| [`mean_speed_heading`](algorithms/mean_speed_heading.md) | baseline | 定义 | — | 姿态（航向）；门控变体用加计 | 2D 平均速度 | 0（标定标量） | 是 | — | 可移植 |

`nio_lie_events_ronin` 与 `nio_lie_events_tlio` 共用一张卡 [`nio_lie_events.md`](algorithms/nio_lie_events.md)：两者的事件生成完全相同，只是骨干不同；夹具仍按注册名分开。
IMUNet 的四个变体指仓库中的一维 MobileNet / MobileNetV2 / MnasNet / EfficientNetB0 实现，并非标准视觉版本（差异见各卡 §10）。

## 3. 协议扩展（状态）

DESIGN §3/§4 原本规定模型只接收 `(B, 6, T)` 并返回窗口级 `vel`。下列扩展已在 DESIGN §3/§4/§5 中落地
（**已落地**一列给出配置键与文档位置）；尚未落地的扩展，对应卡片仍给出降级方案，并要求在结果中标注。

| 扩展 | 用途 | 涉及算法 | 状态 |
|---|---|---|---|
| 逐帧目标 `target=frame_velocity`，输出布局 `(B,T,D)` | seq2seq 专用损失（RoNIN 的 latent velocity loss、CTIN 的 IVL/CNL、AirIO 的逐帧机体系速度） | `ronin_lstm`、`ronin_tcn`、`ctin`、`airio`、`pedestrian_diffusion` | **已落地**：DESIGN §3.1；`target=frame_velocity`，损失按 `mask` 跳过无效帧 |
| 多步目标 `target=multi_displacement` + `output_steps`，输出布局 `(B,H,D)` | 一次预测多步位移（RNIN 的 10 步） | `rnin` | **已落地**：DESIGN §3.1；窗口等分为 `output_steps` 段，各段按自己的跨度换算速度 |
| 重叠预测合并 `overlap` | 逐帧/多步输出在滑窗下重叠，需要确定的合并规则 | 同上 | **已落地**：DESIGN §5 第 1 条；`overlap=mean`（默认）/ `center` |
| 历史上下文 `history` / `history_stride`：输入 `(B,H_in,6,T)` | 多窗口序列输入（10 个 1 s 子窗） | `rnin`（`history=10, history_stride=100`）、`tartanimu`（方案 A） | **已落地**：DESIGN §3.2；有效性按整个输入跨度判断 |
| 额外输入 `extra_inputs`：逐样本姿态、重力、初始速度 v0 | 输入不能只由 `[gyro, acc]` 构造 | `dive`、`airio`、`nio_lie_events_*`、`pdr`、`mean_speed_heading` | **已落地**：DESIGN §3.3；`orientation` / `gravity` / `init_velocity`（**特权输入**，结果中标记并单列） |
| 序列级 / 有状态预测器钩子 | 整序列流式推理、上一窗口预测作为下一窗口输入、测试时训练 | `ronin_lstm@stream`、`ronin_tcn@stream`、`nio_lie_events_ronin`、`ionet`（官方协议模式）、`rio`（A-TTT）、`pdr` | **已落地**：DESIGN §4 的 `SequenceModel.predict_sequence(seq, view, starts)` 与 `calibrate(views, split)` |
| 时间戳 / `body_frame` 等序列级元信息 | PDR 的前向轴查表 | `pdr`、`mean_speed_heading` | **已落地**：序列级模型直接拿到 `Sequence` 与 `SequenceView`（含 `attrs`） |
| 训练姿态回退 `orientation_fallback`（设备姿态末端对齐误差 > 20° 时改用参考姿态） | RoNIN 系官方训练姿态规则 | `ronin_*`、`eqnio_ronin`、`nio_lie_events_ronin` | 未落地（卡片给降级方案） |
| 窗口速度上限过滤（例如 3 m/s、4 m/s） | 官方训练样本筛选 | `ronin_lstm`、`ronin_tcn`、`rnin` | 未落地（卡片给降级方案） |
| `dims=3` 与 `gravity_yaw_local` 的锚点 | TLIO 系官方以**窗口起点**偏航建局部系，IPB 以末端建系；非等变骨干在偏航增强下近似等价 | `tlio`、`llio`、`eqnio_tlio`（严格等变，可直接用 `gravity_world`）、`nio_lie_events_tlio`、`dive` | 未落地（仍用末端锚点；航向定义见 DESIGN §3） |
| 可复用输出头：`gated`（Softplus 幅值 × Tanh 门控）、`bins`（逐轴分箱分类 + 期望/argmax 解码） | 在任意骨干上做“回归 vs 门控 / 回归 vs 分箱”的对照消融 | `gnio`、`velobins` | 未落地（卡片给降级方案） |
| 分箱范围 `R` 的训练期统计与记录（逐轴 `max|v|×1.1`，只用训练划分），以及解码方式开关 | 分箱输出头的语义依赖数据集相关的箱网格 | `velobins` | 未落地（卡片给降级方案） |

两个经典基线已实现并注册：[`pdr`](algorithms/pdr.md)（`cfg/models/pdr.yaml`，
`src/inertial_benchmark/models/pdr/`）与 [`mean_speed_heading`](algorithms/mean_speed_heading.md)
（`cfg/models/mean_speed_heading.yaml`，`src/inertial_benchmark/models/mean_speed_heading/`）。
两者都是 `SequenceModel`，标定标量（`pdr` 的 `K`、`δ`；`mean_speed_heading` 的 `s̄`、`s̄_move`、`v̄`、`δ`）
只在 train 划分上拟合（传入其他划分直接报错），并随 checkpoint 与 `metrics.json` 的 `calibration` 一起发布。
测试见 `tests/test_baselines.py`（合成走路信号上的步数、振幅、距离误差、航向约定与标定恢复）。

## 4. 跨算法的官方实现问题（摘要）

逐条证据（文件与行号）见各卡 §10。

1. **窗口起点时间戳 / 按分量平均的 ATE**：RoNIN、IMUNet、DeepILS 都把窗口速度的时间戳记在窗口起点，积分轨迹因此滞后半个窗口；ATE/RTE 对 N×2 个坐标分量求均值，数值为欧氏定义的 1/√2（PedestrianDiffusion 为 3D 分量平均，是 1/√3）——**ATE 与 RTE 两者都已核实**（`ronin/source/metric.py`，`compute_absolute_trajectory_error` 与 `compute_relative_trajectory_error` 都是 `np.sqrt(np.mean(err ** 2))`，`err` 形状 `(·, 2)`），数值核对见 `docs/METRICS.md` §2.2。TinyOdom 的 ATE 用的是平均误差而非 RMSE。IPB 按 DESIGN §5/§6 统一计算，与论文数值对照时须换算。
2. **阶段切换或调度器实际不生效**
   - TLIO：前 9 个 epoch 用的是“logstd 被 detach 的 NLL”，并非论文所说的 MSE；`ReduceLROnPlateau` 从未 `step()`。
   - EqNIO-TLIO、NIO-TLIO：调度器同样从未调用；EqNIO 的规范帧架构从不调用 `zero_grad`，梯度一直累积。
   - RNIN：`start_cov_epochs=2000` 大于 `epochs=201`，协方差头从未训练，测试时 σ 恒为 1。
   - TartanIMU：机体系分支的损失恒为 20·L1，协方差头拿不到梯度。
   - EqNIO-RoNIN：发布的训练循环只跑 1 个 epoch。
   - GNIO（论文）：正文自称“两阶段 MSE→NLL”，式 13 却是固定权重的静态加和（λ_MSE=1e2、λ_NLL=1e-4），没有任何按 epoch 的切换。
   - VeloBins（论文）：**完全不优化 NLL**，不确定度只由误差条件高斯标签的 KL 监督学到——移植时不要“顺手补一个 NLL 损失”。
3. **通道语义**
   - RoNIN 的 `lstm_bi` 指 bilinear 层，LSTM 本身是**单向**的。
   - EqNIO 在网络内部把骨干输入重排为 `[a', s·ω']`（O(2)）；陀螺必须按**赝矢量**参与反射，否则等变性不成立。夹具中给出了实测误差：正确变换时 ≤1e-12，把陀螺当普通矢量时约 0.1–0.9。
   - TinyOdom 的官方输入是 10 通道 `[acc, gyro, mag, step_mask]`；AirIO、PedestrianDiffusion、iMoT、IONext 的通道顺序为 `[acc, gyro]`；DIVE 的输入是 `[Log(C), acc−g]`，不是原始 `[gyro, acc]`。
4. **测试数据参与选模或训练**
   - IMUNet（RIDI、自采数据集、OxIOD）、DeepILS（RIDI/IMUNet/KIOD/INAIOD）、AirIO（EuRoC）用测试集做验证或选模。
   - TinyOdom 的 RoNIN 笔记本因链式赋值，实际在 seen 测试集上训练。
   - RoNIN 的预训练模型是在整个数据集（含测试序列）上训练的。
5. **论文与代码不一致**：RoNIN-TCN 通道（论文 16… 对代码 32…）；IMUNet 的结构表与代码不符；TartanIMU（论文去重力、MSE+NLL，代码保留重力、20·L1）；EqNIO 规范帧向量的顺序与符号；LLIO 回归 MLP 的 dropout 写死为 0.5。一律以**官方代码实际行为**为准，论文版本作为变体记录。
6. **无法直接运行的官方脚本**：`np.int`/`np.math`（NumPy ≥ 1.24/2.0）、`ReduceLROnPlateau(verbose=True)`（torch 2.11）、缺失模块导入、调用签名错误（NIO 训练期事件堆叠）、键名错误（TinyOdom `dil_list`、RoNIN `'feature_sigma,'`）。

## 5. 官方仓库、提交与许可

此表保留原规格审计提交与历史路径；本轮服务器参考来源的实际路径、HEAD、文件 SHA 和许可扫描以 [provenance](models/provenance.json) 为准。16 个仓库已经取得文件不等于 16 个完整可运行官方算法：CTIN/iMoT 是文档占位，MEIO 是缺网络的训练包装。

| 仓库 | 提交 | 许可 | 本地路径 / 说明 |
|---|---|---|---|
| <https://github.com/Sachini/ronin> | `805b7f0f28bb164ce89ada9ac05a9470dbe3d715` | GPL-3.0 | `third_party/ronin` |
| <https://github.com/CathIAS/TLIO> | `d7051c4ff14834b93931984b75b40f894e39654d` | BSD-3-Clause（数据 CC BY-NC） | `third_party/TLIO` |
| <https://github.com/zju3dv/rnin-vio> | `b030ecc94f151159973a9086c8ba76e22bdbc56e` | Apache-2.0（`ronin_3d/` 为 GPL-3） | `third_party/rnin-vio` |
| <https://github.com/BehnamZeinali/IMUNet> | `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` | **无 LICENSE** | `third_party/IMUNet` |
| <https://github.com/RoyinaJayanth/EqNIO> | `97b7a60a5e5cc8f0132bba343c83de114211f827` | **无 LICENSE** | `third_party/EqNIO` |
| <https://github.com/RoyinaJayanth/NIO_Lie_Events> | `1e87470f6479c198875c5f1a32b392a46eb5af74` | **无 LICENSE** | `third_party/NIO_Lie_Events` |
| <https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer> | `e225ab8f3080c35c92481673b6af1d68692941f6` | GPL-3.0 | `third_party/LightweightLearnedInertialOdometer` |
| <https://github.com/nesl/tinyodom> | `7958781baeddf1b99a4f365fb54fb7bac5e96845` | BSD-3-Clause | `third_party/tinyodom`（Keras；导出用 keras-tcn 3.3.0 重建并与 TFLite 对拍） |
| <https://github.com/OmerTariq-KAIST/DeepILS-IoT-Journal-2025> | `4fd65aa9564075810dab494d43610857dc3e1d01` | **无 LICENSE**（文件头 All Rights Reserved） | `third_party/DeepILS-IoT-Journal-2025` |
| <https://github.com/superxslam/TartanIMU> | `8a8025000f85d2c6b0692b88e15c2c91e761f7d7` | Apache-2.0（HF 权重标签 MIT，但正文写仅限研究用途） | `third_party/TartanIMU` |
| <https://github.com/decargroup/DIVE> | `e982ddd0643c9d831de77a95467b282b07cceec1` | MIT | `third_party/DIVE` |
| <https://github.com/Air-IO/Air-IO> | `7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73` | BSD-3-Clause | `third_party/Air-IO` |
| <https://github.com/jacklu333333/PedestrianDiffusion> | `0b6cc2d019397e390ce757335e01712304339c9b` | **AGPL-3.0**（内含 GPL 代码副本） | `third_party/PedestrianDiffusion` |
| <https://github.com/bingrao/ctin> | `1441c726811ac87283903471562023fd429e3a08` | MIT（README 徽章写 Apache-2.0） | 当前服务器已取得 README/LICENSE；没有网络代码 |
| <https://github.com/Minh-Son-Nguyen/iMoT> | `7f275702bd9b0dbdb60bb4d5dae802f8e9ab9dac` | MIT | 当前服务器已取得占位文档；没有网络代码 |
| <https://github.com/higerra/ridi_imu> | `ce5d772a1ef8ce590121f0f0b254910bc07a2949` | MIT | 当前服务器已取得官方 SVM/SVR 与预处理/双积分来源 |
| <https://github.com/SunXuehang/MEIO-Net> | `e93e5b82ef9cdb9920e13ab0c056f6525c5d4d2e` | GPL-3.0 | 当前服务器已取得训练包装/配置；官方核心网络缺失 |

IONet、L-IONet、RIO、GNIO、VeloBins 没有核实到可运行官方网络源码（RIO 的 CVPR 论文链接到华为云 AI Gallery 页面，源码未核实；VeloBins 摘要称“录用后发布”；GNIO 未提供代码链接）。这五项加仅占位的 CTIN/iMoT、未核验作者网络代码的 M2EIT/DiffusionIMU 共九个 paper-only 方法。MEIO 单列 partial，不把包装文件等同完整网络；已排除的 IONext 不再作为待迁移算法。

**许可策略**：
- GPL/AGPL 仓库：只依据规格卡独立实现，不得复制代码，否则衍生文件受 copyleft 约束。
- 无 LICENSE 的仓库：默认保留全部权利；本仓库只保存规格卡与形状夹具（不含代码与权重），权重不得再分发。
- 夹具中的 `pretrained_golden`/`checkpoint_check` 只记录官方权重的 sha256、严格加载结果和少量推理输出，用于本地核对，不包含权重本身。

## 6. 排除项

仓库拥有者要求排除 **Shanshan Zhang 发表论文中自己提出的方法**。当前一手记录如下；命中按完整作者身份判断，历史文件保留审计，排除项不进入活跃 registry/config/catalog 或正式队列：

| 方法 | 一手作者来源 | 活跃状态 |
|---|---|---|
| IONext | [arXiv:2507.17089](https://arxiv.org/abs/2507.17089)，含 Shanshan Zhang | 原注册与二维/三维配置已移除，源码和夹具归档留存 |
| TinyIO | [arXiv:2507.15293](https://arxiv.org/abs/2507.15293)，含 Shanshan Zhang | 不接入候选 |
| CKANIO | [arXiv:2507.16865](https://arxiv.org/abs/2507.16865)，含 Shanshan Zhang | 不接入候选 |
| StarIO | [arXiv:2507.16121](https://arxiv.org/abs/2507.16121)，含 Shanshan Zhang | 不接入候选 |

其他领域/输入限制与作者排除是不同原因。下表保留原行人优先规格调研范围；对于跨域新方法可以在具备真实所需输入后另轨接入，不能直接删除执行器/LiDAR/磁力计支路冒充完整原方法。

| 项目 | 排除理由 |
|---|---|
| RepViT | 视觉骨干（面向图像的移动端 CNN/ViT），没有惯性里程计模型、任务或协议可复现 |
| CPUBone | 面向 CPU 推理的视觉骨干，不是惯性定位方法 |
| MoRPINet | 面向轮式/蛇形运动平台，运动模型不适用于行人 IPB 数据集 |
| TALOS-NIO | 没有论文、没有许可，既无可引用的规格来源，也不能合法再分发 |
| AI-IO、IMO、DIDO | 需要旋翼转速或推力等执行器输入，行人数据集没有这类信号 |

`velobins` 的官方配置同样需要旋翼转速/PWM，本来符合上面的排除理由；我们仍然登记它，是因为它的**分箱输出头**是我们研究线的直接竞品，
可以在行人骨干上作为对照实现。登记的是明确标注的 **IMU-only 变体**（去掉执行器支路，属于改变结构），论文的百分比收益不适用于该变体。

## 7. 未完成与待核实

- **论文全文不可得**：LLIO、DeepILS、DIVE 的论文全文在 IEEE 付费墙后，卡中的训练配方与报告数值标为“未核实”。其中 LLIO 状态为“部分”，拿到 PDF 后应补齐 §5、§7。
- **paper-only 的参数量与论文不符**：历史 CTIN 推导值比论文少 14%，iMoT 只到论文值的约 49%；网络可执行和 GPU 通过不能证明这些声明假设与原论文未公开结构一致。IONext 的版本差别只留在已排除历史记录，不是活跃待办。
- **未下载核对的权重**：RoNIN（FRDR 记录当前无文件）、IMUNet、RNIN、PedestrianDiffusion 的预训练权重，以及 NIO 其余 3 个 TLIO 权重。
- **TinyOdom**：官方未公布 NAS 最终结构，卡片固定仓库 `model.cc` 中的 RoNIN 候选结构（未训练权重，已与 TFLite 输出对拍）。
- **DIVE**：当前主 `dive` 恢复原 XYZ 头，200 @ 200 Hz 下为 7,517,446 参数；旧二维变体为 `dive_2d`、7,516,420 参数。两者与官方 3.5 s @ 400 Hz 的 12,367,110 参数仍有采样/窗口适配差别，均不包含原 EKF 融合。
- **GNIO**：注意力维度/头数与特征池化方式未给出，推导参数量 4,934,153（论文 4.90 M）依赖 §4.3 的假设；增强、权重衰减、划分未写明，按 TLIO 补齐并标注。
- **VeloBins**：编码器沿用 AI-IO 且未复述细节，只有 6,848 参数的分箱解码器可精确复现；箱中心是否均匀、速度目标取窗口哪个时刻、越界速度如何处理均未规定。
- **MEIO**：官方核心网络缺失，早退规则、头位置和布局仍有声明假设。后阶段 NLL 的 head 训练证据、实际选中出口及原始任务不确定性必须单列，不能把 epoch-one smoke 当作全部科学阶段验证。
- **范围继续扩充**：当前23个论文方法仍未达30个独立论文方法；新增方法先核完整作者、发表状态、官方源码与真实输入契约，再按现有 dev/v1 接口独立实现。已有数据集名字（例如 IDOL）不代表对应算法已实现。
