# 🧭 本轮状态 / Project Status

> 更新：2026-10-13，本轮 IMUNet 全量接入与真实 XYZ CPU 闭环。当前执行来源 `dev/v1@fc6ce90`；历史测试保留各自源码和数据指纹。CPU 验收仅在 AutoDL 执行；GPU 尚未接入，容量实测、正式预算与训练队列仍待冻结。

## 交付与执行阶段

最终交付是统一数据/转换契约、保留算法语义的接口、可比评价和可审计实验。一切按 dev/v1 的 NIO/ipb/IPB v1 与现有 model/converter API 接入；旧 main 的框架增量仅归档，有效能力后续按 v1 重新实现，不迁入其训练器、IB API 或同名指标定义。

| 阶段 | 现状与证据 |
| :--- | :--- |
| v1 代码 | 29 个注册架构、34 份模型配置（含三份三维适配）、21 个算法包；20 个官方代码/规格迁移、7 个论文重实现、2 个经典基线；八个核心转换器与 train/val/predict/report 主干，导出 API 尚缺 |
| 历史结果 | HANDOFF 的 705 tests、旧机器成绩和 AMP/dropout 观察为 2026-09-24 历史，不等于当前 AutoDL 复验 |
| 当前 AutoDL | 七个人体核心数据集 1,268 条、107.31900694 小时、790 条 train，full/hash 零 errors、全部 train_ready=true；加 EuRoC/Blackbird 后九个非空正式原生数据集 1,299 条、107.90444861 小时、795 条 train，八个可训练。首批四个 XYZ 配置的真实 CPU 生命周期通过；历史 144/719/81 项测试按各自范围独立记录 |
| 旧候选评价 | 另一接口的服务器检查与代表预测，不计入 v1 当前测试或排行榜 |
| GPU 阶段 | CPU 准入、预算/配方冻结后开展充分训练，目前不发布正式算法排名 |

## 三维主协议增量

主 benchmark 使用真实预测 XYZ，以三维指标为主，并从同一预测的 XY 派生二维辅助指标。首批为 `ronin_resnet18_3d`、`ronin_lstm_3d`、`imunet_3d`（均标 `3d_adapted`）和原生三维 `tlio`；前三者共用原架构，保留必要的目标布局和损失，扩展可学习三维输出，不宣称原文二维 checkpoint 或论文精度等价。原二维配置/夹具作为独立参考保留。

三维请求不得静默退成 XY，也不能以真值或零填充替代未预测的 Z；PDR 和常速航向的平面变体拒绝 `dims=3`，IONet/EqNIO-RoNIN 当前仍为二维专用。四个首批 XYZ 配置 × 七个人体核心集形成 **28 项配置/metadata 准入候选**，尚未充分训练；私有矩阵需绑定当前源码、完整数据指纹、输入权限和 GPU 实测预算后才能 armed。EuRoC 按跨域 UAV 单独评价，Blackbird 仅 test，不进入人体训练矩阵。旧混合二维计划已退役，CTIN 独立资源准入。TLIO 当前实现为网络，不含论文 EKF。

历史 `2c1f57f` 的三维头/目标/损失/重载、严格评价、XY 派生与容量准入目标测试已在 AutoDL CPU 完成：**144 unique passed / 0 skipped / 0 failed**，绑定 source manifest `5549c535e0e32b4240c2c4353440a7ef622b22d17165f445a60e1b4abaf4a761`。该计数不能声明为 `fc6ce90` 或后续文档提交重跑，也不与历史 719/81 或本次 IMUNet 14 项简单相加。GPU 阶段先测量容量和默认整批 gap，再冻结预算；尚无 GPU 训练或三维精度复现结论。

## 数据接入

| 数据 / 资产 | 本轮事实 | 下一步 |
| :--- | :--- | :--- |
| 八个核心转换器 | RoNIN、RIDI、OxIOD、TLIO、IDOL、RNIN、IMUNet、PedLocData 已有代码；前七个人体核心集完整数据已验收 | 进入 full-core GPU 容量候选；来源指纹、划分和警告见下表 |
| [EuRoC](datasets/euroc.md) | 11 条录制转换为 11 份原生 v1 HDF5，full/hash check 零 errors；correction 未应用 | 整录制 5 train / 1 val / 5 test，reference/oracle；部署输入权限仍需分 track |
| [Blackbird](datasets/blackbird.md) | 20 个发布片段转换为 20 份原生 v1 HDF5，full/hash check 零 errors | route 保守同组，仅 test 审计目录，不等于独立 20 次飞行或训练/验证划分 |
| [Pegasus](datasets/pegasus.md) / [ADVIO](datasets/advio.md) | 实际识别并拒收 7 / 3 个来源资产，两者均 0 canonical | 输出审计拒收报告；待来源姿态/帧契约证实后再接入 |
| [TartanIMU 2025 来源审计](datasets/tartanimu_2025.md) | 新增审计卡，无本轮 converter 注册 | 不把来源卡计为可训练数据或转换器 |
| [TLIO](datasets/tlio.md) | 本轮 raw 全包与服务器哈希已验收，354 条原生序列 full/hash 零 errors，train_ready=true | VIO 参考、分组重叠警告保留，进入容量测量候选 |
| [OxIOD](datasets/oxiod.md) | 本轮 raw 全包与服务器哈希已验收，128 个候选接收 126 / 拒收 2，五个主划分非空 | Vicon 参考；Tango、syn 和独立 test 目录暂不接入 |
| [RNIN](datasets/rnin.md) | 本轮 raw 全包与服务器哈希已验收，301 个候选接收 300 / 拒收 1，train_ready=true | 混合参考：75 条外部系统、225 条 BVIO，test 9 条全为 VIO；与 IDOL raw 分开记账 |
| [IMUNet](datasets/imunet.md) | 用户上传完整 ZIP 已做来源哈希与全成员 CRC 校验；126 个 processed CSV 接收 113 / 拒收 13，full/hash 零 errors、2 warnings | 65 train / 16 val / 32 test；同设备 Tango/ARCore VIO，参考姿态输入属 reference/oracle |
| [PedLocData](datasets/pedlocdata.md) | 部分 raw/demo 和未完成文件已取得，完整包尚未验收；不可训练 | 等待完整来源校验与 native 准入 |

### 七个人体核心数据集全量转换验收

| 数据 | 接收 / 拒收 | 实际划分（条） | 原生输出字节 / 检查 warnings |
| :--- | :--- | :--- | :--- |
| [RIDI](datasets/ridi.md) | 94 / 0 | train 45、val 4、官方 test 23，另 test_unseen_subject 11、test_unlisted_seen_subject 11；五项合计 94 | 117,002,894 / 3 |
| [RoNIN](datasets/ronin.md) | 151 / 1 | train 71、val 16、test 64（test_seen 32 + test_unseen 32）；另 test_ood_device_id 为 test 的诊断子集 | 964,361,898 / 7 |
| [IDOL](datasets/idol.md) | 130 / 0 | train 38、val 5、test 87（test_known 51 + test_unknown 36）；building 子集与这些集合有交集，不重复累加 | 833,504,808 / 34 |
| [RNIN](datasets/rnin.md) | 300 / 1 | train 240、val 51、test 9 | 357,298,253 / 15 |
| [TLIO](datasets/tlio.md) | 354 / 0 | train 283、val 35、test 36 | 1,283,873,945 / 3 |
| [OxIOD](datasets/oxiod.md) | 126 / 2 | train 48、val 13、test 9、test_unseen_subject 30、test_unseen_device 26 | 571,135,975 / 3 |
| [IMUNet](datasets/imunet.md) | 113 / 13 | train 65、val 16、test 32；val 从官方 train 按 subject2 生成 | 327,224,823 / 2 |

此前 RIDI/RoNIN/IDOL 批次为 **375 份 HDF5、1,914,869,600 字节**；RNIN/TLIO/OxIOD 增量为 **780 条、53.695518 小时**；本次 IMUNet 新增 **113 条、7.91155833 小时、65 条 train**。七个人体核心集总计 **1,268 条、107.31900694 小时、790 条 train**，均 full/hash 零 errors、`train_ready=true`。上表前三集保留此前 HDF5 文件字节证据，后四集为 receipt 的原生输出文件字节总和（含 metadata）；磁盘分配字节另行清点。作者官方 known/seen 划分允许 subject/group 共享，所有 warnings 保留。RoNIN `a011_3` 使用官方 gyro_integration 姿态后，重力平均倾角 4.1°，明确拒收。数据准入不证明充分训练或论文精度。

服务器仓库根目录的相对位置为 `datasets/ipb-v1/full-core/{ridi,ronin,idol,rnin,tlio,oxiod,imunet}`。使用 `export IPB_DATASETS="$PWD/datasets/ipb-v1/full-core"` 后，七个同名 `data=` 配置直接指向完整数据；`ridi_smoke` / `ronin_smoke`、pilot 和工程截段不进入正式训练数据统计。

| 数据 | 实际 dataset fingerprint |
| :--- | :--- |
| RIDI | `b4f38fa9a5fef00f0c5f91e73436524a16a3e36843f9dce9868dc5dba8801aa2` |
| RoNIN | `b2b88e29ccad818bf2dc7de3e93fcc44c8f59982e8aa66affd04e4d74f72968c` |
| IDOL | `895b5618a295eae3b34d445929c23f679fbedc8b8ec092f9e6a186f8bba39e5c` |
| RNIN | `3b0898a8cf2a6c2a74965945aa9798dc36754f0627f1c7e20db320b7fbc361af` |
| TLIO | `c2d0ddbe9d6ea11fa3e3ecd57783bc7a2abd4c2ea35550bc7c07922917d29638` |
| OxIOD | `15551c06363ecfb6ca0651ed81dd5a1d24bda7f82028bdeb4fbe6de4c93a59e7` |
| IMUNet | `3fe0f633264ab36a82814ea75211b359e91143e085d60ea8c26c7c31bdaca0a1` |

本轮 RNIN/TLIO/OxIOD 全量转换与独立 CPU 数据验收绑定 `2c1f57f`，部署清单 `source_manifest.json` 的 SHA256 为 `5549c535e0e32b4240c2c4353440a7ef622b22d17165f445a60e1b4abaf4a761`，清单与其中逐个核验的 365 个受管源码文件在执行前后均不变；各新增数据发布时，此前已就绪核心数据的 manifest/指纹也保持不变。三集分别完成四个首批模型任务视图 × train/val/test 的 12 个真实 B=2 XYZ 批次与四项 CLI dry-run；XY 与同一 XYZ 目标截取一致。这里只验数据与接口，不运行 GPU 训练，不建立精度排名。

本次 IMUNet 沿用 dev/v1 已有 `imunet@1.0` 转换器，无另写 converter。用户 ZIP **2,592,746,775 B**，SHA256 `d873476439b115aee77cbde6b40efcafdcfe62a736c2c3d5492d9079062275cb`；263 个文件 / 516 个 ZIP 成员 CRC 全部通过，解压 raw 文件共 **5,828,310,200 B**，包含 126 个 `processed/data.csv`。实际 raw 为 `/root/autodl-tmp/other-inertial-benchmark-v1/datasets/raw-incoming/imunet/d873476439b115ae/extracted/IMUNet_dataset`。原生输出为 `datasets/ipb-v1/full-core/imunet`；接收 113 / 拒收 13，共 **7.911558333333334 h、30,803.358194 m**，输出文件 **327,224,823 B**。13 条拒收为 11 条小米加速度比例异常、1 条参考世界系倾斜、1 条 train/test 逐字节重复；原始数据与拒收报告保留，不静默校准。原始四设备经拒收后剩 S10 38 / S21 26 / Tango 49，三个设备类型。官方 train/test 为 90/36；拒收后 81/32，官方 train 再按 subject2 抽出 16 条 val，正式 65/16/32；full/hash 的 test/train 三组、test/val 一组受试者重叠作为两条 warnings 保留。参考为同设备 Tango/ARCore VIO，参考姿态输入属 `reference/oracle`，相机到 IMU 杆臂未知；不称为独立外部真值。

IMUNet 本次转换、独立 XYZ 数据验收与真实 CPU 生命周期均绑定 `dev/v1@fc6ce90`、source manifest SHA256 `681b09b6150ef54aa773ae4a9eb6b3bdf20aa70c9353d281a0c279da64f47cac`；逐个核验的 **365 个受管源码文件**在执行前后未变，既有正式 native 的 manifest/指纹/划分哈希保持不变。四首批模型 × train/val/test 共 **12 个真实 B=2 XYZ 批次**通过，Z 来自 native 实测参考位置的数值推导，XY 与同一 XYZ 目标截取一致；**4 项 CLI dry-run**通过。已有 converter/raw 两模块的 **14 项测试通过，0 skipped / 0 failed**，包含全 raw 重复扫描与真实 ARCore 共轭负对照；计数不与历史 suite 简单相加。

RNIN 位置标签为混合参考：75 条使用 VICON 或其他外部定位系统，225 条使用 BVIO；其 9 条 test 全为 VIO，不将这部分称为独立外部真值，也不将 75 条全部标为 VICON。TLIO 使用 headset MSCKF/VIO 参考；OxIOD 保留 Vicon 实测 XYZ，但标记点与 IMU 的 lever arm 未知。OxIOD 的 val 从官方 train 按会话生成，与 train 会话不重叠；官方 test 和额外 subject/device 集不混入 train。iPhone 的 gravity/user_acc 按 G 恢复完整加速度，Android 使用对应 SI 列；八条 Nexus5 序列缺设备姿态，仅设备姿态依赖方法不能使用。RNIN 的 15、TLIO 的 3、OxIOD 的 3 条 full/hash warning 保留，包含官方分组重叠和边缘重力诊断。OxIOD 的 `handbag_data2_seq3` 与 `multi_users_user5_seq5` 因时间戳精度损坏拒收；Tango、syn 与独立 test 目录是未接入范围，不计入两条拒收。

此前 RIDI/RoNIN/IDOL 三集全量转换执行绑定源码 `0ae5ea3`，manifest SHA256 `7573a39dbe62a4fdeda0ac6eb6a67aee9c6d4da19c426f0cf211ec2d0b9689f3` 前后不变。执行 receipt SHA256 `52238c2b45f28a2a6a2f486f297dd8cb266db5021a4ab2fdf7bcf04a707eff80`；`full-core-completion-summary.json` SHA256 `f96229f385e46dd9bd9dcaf2297f719c50bfb4ee97f90810840ae3312a25d877` 汇总了 raw 来源哈希、conversion reports、full/hash reports、划分与拒收原因。此数据验收与下面的旧科学 suite、81 项目标测试分别记录。

`cc6554b` 对此前 16 个 converted-data 节点另作诊断，结果为 3 split-policy passed / 10 missing-dataset skipped / 3 historical-release fingerprint failed。后三项的 converter、200 Hz、序列数、时长和距离断言已通过，但新生成数据的 fingerprint 未匹配历史 release 前缀；新 HDF5 的 full/hash 检查通过不等于历史发布物逐字节复现。独立 receipt 为 `cc6554b-fresh-release-receipt.json`，SHA256 `10fc43218031702316b83171cbb1370f1d3c9fc3598556bb4101aa120642e4d4`，不改变下面旧验收计数，也不据此宣称 CI 全绿。

新增共四个 converter 注册入口：EuRoC/Blackbird 两个原生转换器、Pegasus/ADVIO 两个审计拒收入口；TartanIMU 2025 只有来源卡。逐数据集细节见 [datasets/](datasets/)。本轮新增合计 31 份原生 v1 文件、24,158,461 字节：EuRoC 15,790,930 字节 / 1,348.34 s；Blackbird 8,367,531 字节 / 759.25 s，保守划为 10 个 route groups。转换器针对四个文件的服务器测试为 10 passed（5.60 s），ruff 通过；两数据集 full/hash 检查均零 errors；保留 EuRoC 两条原官方集合重叠 warning，以及 Blackbird 缺少 train/val 和三个 egg 片段重力 marginal 共五条 warning，不隐去边界信息。验收记录为 `new-datasets-v1.json`，SHA256 `1b5f9d393804ef5fe898cb9ab9fa064ad724f9fbc4915735a9f49a35c066d4a5`；科学 suite 与目标验收范围分别记录如下。

## 当前数据磁盘清点

**2026-10-13 22:28（北京时间）** 的最终只读清点覆盖当前正式 v1、raw、归档、历史副本、pilot/staging 和 partial，按 `(st_dev, st_ino)` 去重，不跟随 symlink 别名；15 个数据集有占盘资产，HelmetPoser raw=0。实际文件系统分配字节共 **80,096,964,608 B = 80.096964608 GB = 74.5961113 GiB**，可用空间 **122.4149132 GiB**；GB=10⁹ B，GiB=2³⁰ B。归档与解压 raw 均实际保留，各占一层空间；逻辑文件大小 **80,084,460,718 B** 与分配字节分开报告，分配字节包含目录开销。IMUNet 所有保存层合计 **8.778166272 GB**。本次存储清点不重算内容哈希。

| 保存层 | 分配字节（B） | GB |
| :--- | ---: | ---: |
| 原始归档 | 24,864,436,224 | 24.864436224 |
| 解压 raw | 45,724,356,608 | 45.724356608 |
| 当前七人体核心原生输出 | 4,458,782,720 | 4.458782720 |
| EuRoC/Blackbird 等原生扩展（含空拒收目录 metadata） | 24,379,392 | 0.024379392 |
| 历史原生副本 | 4,620,910,592 | 4.620910592 |
| pilot / 调试 / staging | 287,596,544 | 0.287596544 |
| 失败转换 staging（保留审计） | 12,533,760 | 0.012533760 |
| 真实数据工程截段夹具 | 737,280 | 0.000737280 |
| 未完成 partial | 3,551,232 | 0.003551232 |
| 来源及审计 metadata | 4,947,968 | 0.004947968 |
| 未归类资产 | 94,732,288 | 0.094732288 |

九个非空正式原生数据集合计 **1,299 条、107.90444861 小时、795 条 train 序列**；八个可训练，Blackbird 仅 test。七个人体核心集为 **1,268 条、107.31900694 小时、790 条 train**；EuRoC 11 条（5/1/5）为跨域 UAV，Blackbird 20 个片段仅 test / 10 route groups。当前 native 保存层分配字节为 **4.483162112 GB**，含空拒收目录 metadata。序列统计不加入 pilot、legacy、历史副本、工程夹具或辅助 test 子集，存储统计包含实际保存副本。

| 数据集 | 所有服务器保存层 GB | 当前范围 |
| :--- | ---: | :--- |
| ridi | 1.104723968 | 94 条人体核心原生序列；保留 raw/历史副本/工程夹具 |
| ronin | 38.592077824 | 151 条人体核心原生序列；保留四个官方归档与 raw/历史副本 |
| idol | 5.142503424 | 130 条人体核心原生序列；保留 building 归档与 raw/历史副本 |
| rnin | 6.658166784 | 300 条人体核心原生序列；归档、raw 与 pilot |
| tlio | 14.476201984 | 354 条人体核心原生序列；归档、raw 与 pilot |
| oxiod | 4.485308416 | 126 条人体核心原生序列；额外 Tango/syn/test 未接入 |
| imunet | 8.778166272 | 113 条人体核心原生序列；完整 ZIP/raw、pilot、失败 staging 与工程夹具 |
| euroc | 0.172244992 | 11 条跨域 UAV 原生录制，5 train / 1 val / 5 test |
| blackbird | 0.119189504 | 20 个发布片段，仅 test / 10 route groups |
| pegasus | 0.062492672 | 来源归档与拒收报告；0 条正式原生序列 |
| advio | 0.324440064 | 三个来源归档与拒收报告；0 条正式原生序列 |
| tartanimu | 0.149123072 | 公开 raw 样例子集；未接入完整数据集 |
| pedlocdata | 0.002502656 | 本次服务器仅未完成 partial；不可训练 |
| grandtour | 0.028655616 | 一个小 mission 的 9 个来源文件；无正式 native |
| aqualoc | 0.001130496 | reference/calibration 与 1 MiB IMU partial；无正式 native |
| helmetposer | 0.000000000 | 目录为空，raw=0；无正式 native |
| shared_or_unclassified | 0.000036864 | 共享 / 未归类 metadata；不归入特定数据集 |

GrandTour 已取得一个小 mission 的 9 个文件、约 28.63 MB，逐文件 raw SHA 已核验，尚未转换为 native；AQUALOC 已取得 69,958 B 的 updated reference/calibration，IMU 包仅 1 MiB partial；HelmetPoser 尚无 raw。PedLocData 为部分 raw/demo 与未完成包；Pegasus/ADVIO 已保存来源 raw 和拒收报告但无正式 native。取得来源文件不等于完整公开全集或训练准入。本机 RNIN/TLIO/OxIOD/IMUNet 四个原始归档共 **8,465,048,858 B（8.465048858 GB）**，IMUNet 的 hardlink 别名不重复计数；本地归档不计入服务器总量，不作两端相加。

### 跨输入重复与划分审计

2026-10-13 22:28（北京时间）审计读取九个正式 native 的 manifest 与实际 split 文件，1,299 条序列的非空有效 `imu_sha256` 全部不同：1,299 个已知哈希、0 unknown、0 exact IMU hash matches，跨集与集内 train/test 精确输入重复候选均为 0。该检查为 metadata-only，未重新读取 HDF5 或重算输入内容哈希；九个 manifest 与存储清点时一致。无精确 IMU 哈希相等不能证明位置标签、整条轨迹、主体或路线相互独立；Blackbird 的重复 route 和重叠片段仍按相关样本处理，官方 subject/group overlap warnings 不因此删除。数据和划分未修改。

存储审计 receipt SHA256 `3dfb9d82b06313bd14bb46db33685ff0b02073cf746aa71c75588ee5b4bac1fc`，跨输入审计 receipt SHA256 `d384fc6ea581e6c9e1bff4875fec89aee91bdb142c12b3e2e28c29b7e35eb85f`；最终 metadata audit 绑定 source manifest `681b09b6150ef54aa773ae4a9eb6b3bdf20aa70c9353d281a0c279da64f47cac`，未导入模型、读取 HDF5 或修改受管 source/native。

## 当前服务器 CPU 闭环证据

当前 CPU 执行绑定 `fc6ce90` 与 manifest `681b09b6150ef54aa773ae4a9eb6b3bdf20aa70c9353d281a0c279da64f47cac`，所有训练与测试均在 AutoDL，CUDA 未初始化。历史 `2c1f57f` 的 **144 unique passed / 0 skipped / 0 failed**绑定上文原 manifest；本次 IMUNet 14 项不简单相加，文档改动不冒称测试重跑。

首批真实 XYZ CPU 生命周期实际通过如下（均 standard NIO/Trainer，1 epoch、B=2、2 次真实 optimizer updates，workers=0、cache=false；纯工程截段）：

| 模型 / 真实数据 | AutoDL CPU 用时 | 闭环结果 |
| :--- | ---: | :--- |
| `ronin_resnet18_3d` / RIDI | 25.856 s | train/val/predict、XYZ 输出与 checkpoint 重载一致 |
| `ronin_lstm_3d` / RIDI | 18.372 s | train/val/predict、XYZ 输出与 checkpoint 重载一致 |
| `tlio` / RIDI | 21.753 s | train/val/predict、XYZ 输出与 checkpoint 重载一致；网络范围，不含 EKF |
| `imunet_3d` / IMUNet | 22.283 s | 98 个可训练参数张量真实更新且 optimizer state step=2，loss=4.0785136 且 finite；XYZ 位置和实际窗口 vel_pred/target 重载前后严格相等 |

IMUNet 使用真实 train/val/test 230/210/200 样本截段，传感器、完整 XYZ 参考、相对时钟间隔与有效掩码均保留，仅工程时间原点归零。默认 `train_eval_gap` 实际执行 2 个 val 窗口、eval/dropout_only/bn_only/train 四模式，保留 MC=8，无 monkeypatch；新训练仅两次更新的速度比 2.223 产生 warning，不能推导精度、轨迹收缩原因或充分训练下的 BN 结论。四闭环保持全部九个正式 native 的 metadata/指纹/划分哈希不变。它们证明工程流程，不证明全数据默认 1,024 窗口整批 gap 的 GPU 容量或正式预算。

GPU 私有容量探测、默认整批 gap 与真实 tiny 生命周期检查已准备，实际 GPU 尚未接入；CPU 与私有准备已验收，可安排 4090D 接卡；接卡后实测显存、吞吐与默认整批 gap，再冻结正式预算。实测前不冻结 batch/lr/epoch/早停，不 armed 正式队列。广覆盖使用 seed=0；关键科学结论和消融最多 seeds 0/1/2。

此前 2026-10-13，pristine `faee13c` 核心加独立 CPU 验收 helper，实际完成：

| 项目 | 实测范围与结果 |
| :--- | :--- |
| raw → 原生 v1 | RIDI 三条、RoNIN 三条录制各 3 接收 / 0 拒绝；保留六条原生文件 |
| 短训练数据 | 每个数据集取不同 train/val/test 录制，各截 800 样本（200 Hz）；只用于 engineering_only 闭环 |
| 模型与训练 | NIO ronin_resnet18，4,634,882 参数；每数据集 4 training windows，batch=2，1 epoch，2 optimizer updates；workers=0、cache=false |
| 重载与结果 | checkpoint 重载后的 val/predict 为严格 array_equal，maxabs=0；预测 NPZ 与 PNG 保存 |
| 运行环境 | PyTorch 2.12.1+cu130，线程数 1，CUDA unavailable；当前无卡 CPU 模式约 0.5 CPU / 2 GiB |
| 模式守卫 | 未充分训练的 dropout delta 有 warning/error，记录诊断，不推导算法排名或收缩机制 |
| 原冻结 suite | 719 unique 节点：665 passed / 52 skipped / 2 resource_unverified；首次单进程 SIGKILL 未生成 JUnit，随后逐文件隔离覆盖与原参数补测；不是全部通过 |
| 新绘图补丁 | `6d44573` 独立目标验收 81 unique passed / 0 skipped / 0 failed / 0 error；含绘图契约 5、报表 7、文档 2，计数不并入原冻结 719 节点 |

闭环用截段与极少更新，验证接口、保存和重载一致性，不能证明充分训练精度，也不是完整 benchmark 数据发布。

原科学验收冻结 source manifest SHA256 为 `89736fbb634a023bcbf7a47fb7e04a26c6de2fdb2d170b08005441c45c669c40`。61 个文件分片先得到 635 passed / 69 skipped；随后 13 项 CTIN 常规测试通过，RIDI 8 / RoNIN 6 / IDOL 3 项真实 raw 测试通过并替换原跳过项，按 nodeid 去重后得到 665 / 52 / 2，不重复累加。该历史 suite 当时的 52 个跳过项分别为：16 项需要已转换数据、9 项 IMUNet 与 11 项 OxIOD 使用不可用的旧 raw 路径、4 项 PedLocData / 7 项 RNIN / 5 项 TLIO 缺 raw。

两项未验证都是 CTIN：原参数 mean 训练/验证闭环测试进程 SIGKILL（-9），center 未独立执行。最后一次 hook 位于首次训练 `B=16, N=200` 的 decoder/MHA，尚无顶层 forward_end/batch_end，也未到验证或 gap；没有证据据此确定 OOM 原因。唯一节点覆盖记录为 `integrated-v1-final-unique-coverage.json`，SHA256 `ac1b9a2f93e2b1ea64097d232d5c2265533a20b73b1c14e5d6d579e41e8f51da`。

`6d44573` 的目标验收覆盖 10 个测试模块，81 个 unique 节点全部通过；源码 manifest SHA256 `31272a9b5e0fe5e3613419903a9657eb8ef1efeba2204b1a608b25b8f12a473e` 在执行前后保持一致。全 src/tests Ruff、ops 显式 Ruff、compileall、独立 venv 的导入路径校验、`ipb info` / `ipb cfg` / main 矩阵 24 项 dry-run，以及 ops 七项 guard 和拒绝覆盖 stage（RC=2）均通过。测试 receipt 为 `final6d44573-affected-checks.json`，SHA256 `67b8733844714d1ec9637e69b39f9ce7ed6d6c3d79cc6ce54f1f9ee609ac3e1b`；操作 receipt 为 `final6d44573-operations.json`，SHA256 `8ebc4f47c2722d07414c1c6eb2732f5501bc88fad6ac152d329ad815c59c60aa`。目标验收包含新增与重复受影响测试，不是新一轮 719 节点全量通过声明。

## 从交付倒推的五个阶段门槛

| 阶段 | 工作与准入门槛 | 交付证据 |
| :--- | :--- | :--- |
| 1. 新数据契约 | 按现有 v1 converter API 审计字段、时间、单位、重力、参考姿态、mask 与 group；不满足条件明确拒收 | 数据卡、conversion_report、dataset.json/指纹、接收/拒收数与物理检查 |
| 2. 当前 CPU 链路 | 有来源数据上验 NIO train/val/predict、checkpoint、保存结果；真实运行新服务器 suite | 环境和源码指纹、测试记录、闭环 receipt；工程结果独立标记 |
| 3. AutoDL GPU pilot | 获 GPU 后选择代表算法与数据，实测训练、验证和模式诊断的显存/吞吐；冻结 epoch/lr/batch/早停/窗口/权限/评价预算 | 配方、硬件实测、可运行 pilot 与队列；搜索与选模只用 train/val；另测 train_eval_gap 整批资源门槛 |
| 4. 单 seed 广覆盖 | 先 seed=0 扩展已验收数据/算法，充分训练；按输入权限、评价维度和来源协议独立比较 | 主矩阵、逐序列结果、覆盖/NA、统计与图；测试集固定不调参 |
| 5. 三 seed 核心主张 | 对最终主结果与关键消融最多三个 seed，分别检验 oracle/长度支持、幅值方向、模式/精度与泛化 | 收缩机制证据、受控对比、局限、版本化发布与论文材料 |

阶段可并行开展来源审计和增量实现，但后一个结果依赖前一个准入证据。研究目标为 ICRA / IEEE RA-L，不设未经核实的会议截止日期或录用承诺。

GPU pilot 必须单独验 [Trainer.train_eval_gap](../src/inertial_benchmark/engine/trainer.py) 的资源开销：当前默认最多将 1,024 个窗口作为一个整批，未按 `val_batch` 限制；训练模式和 BatchNorm 的统计依赖批组成，直接分块会改变诊断语义，不能当作无损内存优化。先测量其可行预算与语义再冻结配方。它不是本次 CTIN SIGKILL 已证实的根因，实测 hooks 表明进程尚未完成首次训练前向，未到验证或该诊断。

## 已确认的评价增量待办

| 优先级 | 待办 | 审计与验收 |
| :--- | :--- | :--- |
| P0 | 输入权限 track / 报告隔离 | data/views 可用逐时刻 reference attitude，特权列表只含 init_velocity，报表只检查评价键；区分 sensor/device/reference attitude/initial state 并防混表 |
| 本轮目标验收通过 | 有效维度图与指标一致 | 时间误差/CDF按 metric_dims 和实际预测维数绘制并标注；3–4–5 样例与实际 2D fallback 已验收，水平/垂直分解与3D轨迹图仍待扩展 |
| 本轮目标验收通过 | 超过八模型的分页 | 每 dataset/split 分页保留所有模型箱线/CDF/PLR，全局 Pareto 集合保持完整；17 模型全覆盖与不超过八模型的旧文件名兼容已验收 |
| P1 | NA、覆盖与定义目录 | 记录缺失能力原因、单位、DOF、support、长度分辨率、aggregation 和评价源码指纹 |
| P1 | 严格 gap-aware profile | 现 RTE/D-RTE 允许有效端点跨 gap，短 RTE 会换算；新连续支持/不外推诊断独立版本化，保留旧定义 |
| P1 | 六面板 / HTML / 离线重算 | 复用 SequenceResult.load 和当前绘图/报告；新输出 receipt，不覆盖旧结果 |
| P2 | 姿态 / 概率诊断 | 仅在能力足够时启用；预测/参考与 covariance 的 variable、frame、units、DOF 必须匹配，不把 GT 当预测 |

单独 group 的不确定性、seed/序列覆盖不齐的权重和多模型统计需要明确。aligned ATE、oracle 与新严格诊断保持独立，不对主排名用 GT scale 校准。

## 收缩研究与预算

先验证 oracle/积分与长度分辨率，再检验幅值/方向、训练和推理模式、AMP/dropout 与泛化。复用历史 mode guard，但以当前数据和充分训练复验。先一个 seed 调通配方与预算，最终主结果和关键消融最多三个 seed。CPU 阶段降低准备成本，GPU 按冻结队列执行。
