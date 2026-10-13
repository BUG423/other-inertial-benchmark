# 🧭 本轮状态 / Project Status

> 更新：2026-10-14（北京时间）。主线为 `dev/v1`；新 AutoDL 已有 RTX 4090D 24 GiB、16 CPU 和约 62 GiB RAM。当前 13 个独立来源完成已取得版本/子集的原生准入，当前57份配置具有对应CPU准入（旧53与新4独立源码批），旧49神经配置完成GPU生命周期，新4GPU待验。运行中的旧正式覆盖仍为2,067个单元，统一正式配方已冻结，GPU 实验正在运行，已有单元完成 train/best/test；完整矩阵尚未完成。

## 🎯 交付与执行状态

交付目标是可扩展的惯性定位 benchmark：每个数据源保留真实单位、时间、参考与分组，转换为 IPB v1；算法按 NIO/ipb 统一运行但保留目标、窗口、布局与损失；结果记录数据/源码/配方/输入权限/硬件/随机种子/预测与指标。路径变短的机制研究在这个统一平台上开展。

| 层次 | 当前事实 | 证据边界 |
| :--- | :--- | :--- |
| 数据 | 13 独立来源，1,376 条、114.30638333 h；9 个有 train/val，4 个仅 test | 已取得版本或代表子集；不是公开全集全部取得 |
| 算法 | 33 架构、57 活跃配置；23 论文方法（21 已发表、2 预印本）及 2 经典定义 | 骨干/维度变体不增加论文数；实现与精度复现分别记录 |
| 官方参考源码 | 16 官方仓库固定提交，CTIN/iMoT/MEIO 为 `source_partial` | 源码存在不证明训练、许可或忠实性问题全部解决 |
| CPU 工程 | 57份配置各有CPU证明：旧53+新4独立source收据 | 神经模型真实短训练；CPU-only 方法按其真实标定/回归逻辑，不伪造 optimizer updates |
| GPU 工程 | 旧49/49神经配置实际通过；新4CUDA/GPU batch128待验 | 4090D 上真实 train/val/predict/checkpoint 重载；工程截段不构成论文主结果 |
| 冻结正式矩阵 | 旧53 配置 × 13 来源 × 3 种子 = 2,067 覆盖单元 | 能力不适用、待 checkpoint、pending/failed 都保留，不等于已完成 2,067 次训练 |
| 评价增量 | 默认 56 个 XYZ 轨迹/速度输出键、同预测 XY 和原生任务不确定性 | 原生任务不确定性 25 节点通过；框架 49 unique 节点最终通过；当前冻结源码锁/身份/dispatcher 8 定向节点通过 |
| 正式训练 | 统一正式配方已冻结、GPU 单元运行中 | 真实 train/best/test 已有完成单元；不是全矩阵完成，也不是论文主结果全部复现 |

按仓库拥有者的作者排除规则，张珊珊（Shanshan Zhang）相关被排除方法不进入活跃目录和队列；[IONext 排除记录](models/excluded-methods.json)保留一手作者依据和历史源码。算法规格与适配见 [ALGORITHMS.md](ALGORITHMS.md) 和 [模型卡](models/)。

## 📦 13 个已准入数据来源

各集完整性/full/hash 检查通过；原有九集迁移后逐个核验 **1,299 个 H5 的完整文件 SHA256**，零缺失、零差异，并复用迁移前已通过的科学数组检查。新 GrandTour/SubT-MRS/UZH-FPV/PedLocData Demo 实际转换、完整数组/hash 和真实 XYZ/XY 任务视图检查通过。参考性质、缺失设备姿态和所有 warnings 保留。

| 数据 / 已取得范围 | 接收 / 拒收 | 当前划分（条） | 时长 h | 参考与主要边界 |
| :--- | :--- | :--- | ---: | :--- |
| [RIDI](datasets/ridi.md) 发布 v2 | 94 / 0 | 45 train / 4 val / 23 test；另 11 unseen、11 unlisted-seen | 2.71128333 | 同设备 Tango VIO；3 warnings；官方列表缺项保留 |
| [RoNIN](datasets/ronin.md) 公开四包 | 151 / 1 | 71 / 16 / 64；test_seen/test_unseen 各 32 | 23.10657917 | 胸前 Tango VIO 与另一手机 IMU；7 warnings；官方列表缺 `a007_3`，`a011_3` 物理倾角拒收 |
| [IDOL](datasets/idol.md) 已取得 building 发布 | 130 / 0 | 38 / 5 / 87 | 19.89406806 | 发布处理参考/SLAM；34 warnings；building/known/unknown 辅助集不重复计数 |
| [RNIN](datasets/rnin.md) SenseINS 发布 | 300 / 1 | 240 / 51 / 9 | 8.49352917 | 75 外部系统参考、225 BVIO；test 全 BVIO；15 warnings；另 179 重复 train 输入拒收 |
| [TLIO](datasets/tlio.md) golden v1.5 | 354 / 0 | 283 / 35 / 36 | 31.60380278 | Headset MSCKF/VIO；发布输入已做 bias 补偿；3 warnings |
| [OxIOD](datasets/oxiod.md) v2.0 Vicon 配对部分 | 126 / 2 | 48 / 13 / 9；另 30 subject、26 device | 13.59818611 | Vicon 标记点参考，IMU 杆臂未知；8 Nexus5 缺设备姿态；3 warnings |
| [IMUNet](datasets/imunet.md) 126 processed CSV 候选 | 113 / 13 | 65 / 16 / 32 | 7.91155833 | 同设备 Tango/ARCore VIO；11 加速度比例异常、1 倾斜、1 跨 split 重复拒收；2 warnings |
| [EuRoC](datasets/euroc.md) AirIO 精简发布 | 11 / 0 | 5 / 1 / 5 | 0.37453889 | Leica/Vicon 与 IMU 批量状态估计；不是全部姿态独立 mocap；2 官方集合重叠 warnings |
| [Blackbird](datasets/blackbird.md) AirIO 20 片段 | 20 / 0 | test 20；10 route groups | 0.21090278 | Vicon body→IMU proxy、作者零杆臂假设；仅 test；5 warnings；route/相交段仍相关 |
| [GrandTour](datasets/grandtour.md) 一条固定 mission | 1 / 0 | test 1 | 0.09253750 | CPT7 GNSS/INS 估计参考，经 TF 杆臂转 ADIS 原点；无 device attitude；2 缺 train/val warnings |
| [SubT-MRS](datasets/subt_mrs.md) AirIMU 28 段、5 event-robot groups | 28 / 0 | test 28；三个 OOD 清单是同 28 段视图 | 2.86852222 | 地图约束视觉/LiDAR/IMU 融合相关参考；无 device attitude；1.0.1 placement 规范化；2 warnings |
| [UZH-FPV](datasets/uzh_fpv.md) v3 indoor_forward_6 Snapdragon | 1 / 0 | test 1 | 0.00838750 | vision full-pose、Leica 验证位置；只保留 IMU/reference 的 30.198 s 交集；2 warnings |
| [PedLocData](datasets/pedlocdata.md) SimpleDemo 官方子集 | 47 / 0 | grouped 32 / 9 / 6；录制组 9 / 2 / 2 | 3.43248750 | 未公开位置/姿态参考来源；无 device attitude；3 官方 split 泄漏 warnings 默认分组无泄漏 |

合计 **1,376 条、114.306383333333 h**。八个人体来源与 EuRoC 共九个可训练来源、**827 条 train**；四个 test-only 来源不造 train/val。RIDI 和 OxIOD 的额外测试序列计入正式序列数，重叠的辅助 test 视图不重复累加。TartanIMU 为混合来源集合，不作为新的独立采集来源增加计数。

### 原生路径与指纹

七个人体来源位于 `datasets/ipb-v1/full-core/{ridi,ronin,idol,rnin,tlio,oxiod,imunet}`；EuRoC、Blackbird、GrandTour、SubT-MRS、UZH-FPV、PedLocData 位于 `datasets/ipb-v1/{euroc,blackbird,grandtour,subt_mrs,uzh_fpv,pedlocdata}`。机器准入表为服务器 `.local/reports/formal-native-registry.json`，每行记录绝对目录、manifest/split SHA、数据指纹、参考、来源和验收 receipt。CLI 可直接使用实际数据目录，不假设十三集都位于同一个 `IPB_DATASETS` 层。

| 数据 | 实际 fingerprint |
| :--- | :--- |
| RIDI | `b4f38fa9a5fef00f0c5f91e73436524a16a3e36843f9dce9868dc5dba8801aa2` |
| RoNIN | `b2b88e29ccad818bf2dc7de3e93fcc44c8f59982e8aa66affd04e4d74f72968c` |
| IDOL | `895b5618a295eae3b34d445929c23f679fbedc8b8ec092f9e6a186f8bba39e5c` |
| RNIN | `3b0898a8cf2a6c2a74965945aa9798dc36754f0627f1c7e20db320b7fbc361af` |
| TLIO | `c2d0ddbe9d6ea11fa3e3ecd57783bc7a2abd4c2ea35550bc7c07922917d29638` |
| OxIOD | `15551c06363ecfb6ca0651ed81dd5a1d24bda7f82028bdeb4fbe6de4c93a59e7` |
| IMUNet | `3fe0f633264ab36a82814ea75211b359e91143e085d60ea8c26c7c31bdaca0a1` |
| EuRoC | `74ba02c5f72ace2795b8124a2497d5295c868a8a7f84b288d6e0658ebaf976b4` |
| Blackbird | `8e6bbf05a74e86189907fb91d2dccdbed130c3202d6ecd461ab2ee54eb88bb83` |
| GrandTour | `b6b9a6fbb2614ae92fbf683bc793b48ebde1e2f05382488fb6a2f7be05d03781` |
| SubT-MRS 1.0.1 | `11b721de4c4c8cdc7bca7958ff481fe09acc385525c6aa9192413eca80592a77` |
| UZH-FPV | `cf3a730796c11e958e7ea9e9dcb8ea4bbb77fbd6e7adb0727e15f687da84d560` |
| PedLocData Demo | `6001c5dcf9af0974ade1c20ea0d3be7aeb04706133ef87b5b41a2d2ea09a6cea` |

SubT-MRS 的 `1.0.0 → 1.0.1` 只把 placement 改为统一 `body`，另保留 `source_placement=robot_body` 与 `source_platform=ugv`。28 条的全部数值数组、mask、dtype/shape 与 split 文件逐 SHA 前后相同；旧版保存为审计副本，不重采样、不重复算正式数据。GrandTour、SubT-MRS、UZH-FPV 的真实科学测试分别为 9、6、8 项通过，来源及旧 native guards 不变。

PedLocData Demo 于 **2026-10-13 17:33:31 UTC** 完成 89.85 s 的 CPU 验收；现有 converter 数值实现不变，7 科学测试和全部 47 条真实 B2 XYZ/XY、位置导数/mask 检查通过。官方 MD5 与完整 SHA 验证后转换，默认 grouped 录制隔离全通过，原 12 来源 manifest/split SHA 前后相等。作者列表另存 `official_*`，三条泄漏 warnings 保留；Demo 的参考来源未公开，不声明 mocap 真值。主包未完成，Demo/full 仍一个来源。详情与完整摘要见[数据卡](datasets/pedlocdata.md)。

### Test-only checkpoint 预注册

**2026-10-13 16:44:30 UTC**、正式目标评价前，固定以下 primary checkpoint 来源。每个目标只能使用相同 profile、架构、科学 InputSpec、input track、随机种子和冻结配方的正式 source checkpoint，并校验训练来源证明及 checkpoint SHA；工程 smoke checkpoint 不能冒充。

| Test-only 目标 | checkpoint_training_source | 解释与比较范围 |
| :--- | :--- | :--- |
| Blackbird | EuRoC | 独立 UAV→UAV 迁移，保留 Blackbird route 相关性 |
| UZH-FPV | EuRoC | 独立 UAV→UAV 迁移；当前只评价取得的公共参考交集 |
| GrandTour | RIDI | 人体→腿式机器人跨域诊断，不进入人体同域训练榜 |
| SubT-MRS | RIDI | 人体→轮式机器人跨域诊断，不进入人体同域训练榜 |

这是预注册选择，不声明目标本域训练。额外 checkpoint 来源属于后续独立协议，不增加当前主单元数量。预注册 receipt SHA256 为 `bc3ab673c0af355f2da671392010edad09313ff88c1fc910fe38bbde86130b31`。

## 新增两方法实际验收与旧正式范围

M2EIT/DiffusionIMU公共包四配置于2026-10-13完成70.083s的服务器CPU批：实际collect31，JUnit31passed/0failed/0skipped；每配置使用原默认完整宽度、200@200Hz、B2，标准NIO进行两次真实AdamW更新、train/val/test三份不同已准入RIDI截段、验证/预测/checkpoint重载及原生export/import/predict。raw输出、轨迹和指标精确重放；DiffusionIMU保留评估噪声并重放相同RNG。没有CUDA模型/吞吐/显存测试，也没有重训旧53配置。

| 口径 | 当前库/已验收范围 | 正式实验边界 |
|---|---|---|
| 架构/配置/论文 | 33架构、57配置、23论文（21发表+2预印本）+2经典 | 新增2方法各XY/XYZ，不重复算论文；CPU57=旧53收据+新4独立source收据 |
| 维度 | XYZ33=32neural+RIDI3D；XY24=21neural+RIDI2D+2classics | 旧source1897正式范围仍XYZ31/XY22 |
| GPU | 旧49神经配置GPU生命周期/容量已验 | 新4CUDA/GPU batch128仍pending，不写53神经GPU全通过 |
| 导出 | 旧53native与新4native均有独立实际export/import/predict通过记录 | RoNIN/IMUNet张量adapter范围不变；新方法TorchScript/ONNX未审查支持 |
| 正式矩阵 | source1897/425files、31arch/53profiles/2,067单元继续运行 | 当前库57×13×3=2,223仅futureinventory；新formal未冻结/未启动，不中途替换旧protocol/recipe/checkpoint身份 |

新批source manifest SHA256：`b639c69773b92ea11c6e999e9d7c5f88702c5bd9df140adc5ce67f95779f6b79`（443managedfiles）。Parent receipt：`ee3f064e0f8377e9d5299b8da081b8064b6206b83bb8fedb7e667a698e10f66b`；真实JUnit：`ff4d30cc48cfca2d31c9aed74c9155562f57b213b7a891058148e92e8b05e293`。工程收据scope为 `.local/reports/model-integration-cpu-b639/`，原科学与runtime字节保留，新文档不改受测13源码、8export文件或历史staging15。

| 新profile | dims | 真updates | 分配置receipt SHA256 |
|---|---:|---:|---|
| m2eit | 2 | 2 | `2f0827a4f188d7c0d8ec190e159d88986a7604beb21a4320c1b45d16c914ac2d` |
| m2eit_3d | 3 | 2 | `fe68751c954241f096960be451a22b5db9afd11b50d76fd7fb1a14d7aaa12abd` |
| diffusionimu | 2 | 2 | `fbd3bec6a836b68c5f20be414d8936dc3cf88f44cc83eff1a5aac1819a3d1087` |
| diffusionimu_3d | 3 | 2 | `091aa57b9a6e81bbedb884304f7d67baaabad8ace9a09edb99f2c189ac4ae050` |

[M2EIT卡](algorithms/m2eit.md)和[DiffusionIMU卡](algorithms/diffusionimu.md)公开原配方、统一配方差异、姿态权限、Haar/拓扑假设、真实Z适配、随机推理以及u非σ。CPU/原生导出通过不证明原论文精度或未公开网络拓扑已恢复。

存储、重叠和初期长度快照各保自身时间点。19:02UTC存储与19:13UTC138-completed长度证据是历史快照，不是这次模型验收后的实时进展；不能用早期混域PLR归纳人类baseline普遍缩短或因果。

## 🚧 其他资产与阻碍项

| 数据源 / 资产 | 当前事实 | 可执行下一步与 gate |
| :--- | :--- | :--- |
| [PedLocData](datasets/pedlocdata.md) 主包扩展 | Demo 554,702,552 B 已验收并加入同一来源；主包 15,138,253,808 B 仍有限分块续传，未完整验收 | 主包官方 MD5/SHA 完整后另作版本扩展验收；保留已就绪 Demo 的指纹和结果，不新增独立来源计数 |
| [ADVIO](datasets/advio.md) | 3 官方 ZIP 已取得，审计入口 0 canonical | 需作者证实参考原点、camera→IMU 外参及时间/轴契约；不猜姿态、不伪造 GT |
| [Pegasus](datasets/pegasus.md) | 7 作者发布资产已取得，审计入口 0 canonical | 官方 loader 的 FRD 转换已定位；reference body/origin/exporter 契约仍待证实后转换 |
| [TartanIMU](datasets/tartanimu_2025.md) | 18 NPZ、四域混合集合代表部分 | 核各原始来源和与 IDOL/Blackbird/SubT/UZH 的重叠；不通过重新包装增加独立来源数 |
| AQUALOC | 5 reference/calibration 文件 69,958 B；IMU 归档只 1 MiB partial | 缺完整 IMU 和图像 ID→秒时间映射；447 行 COLMAP image ID 不当时间戳 |
| HelmetPoser | raw=0，已保存失败下载证据 | 人工可访问链接后服务器下载；没有文件不报 ready |
| Aria / HeadLoc | 前者 access/terms，后者独立来源身份未确认 | 先核官方身份与许可、字段和可下载入口；不以近似名称替代来源 |

完整/代表子集的选择遵服务器容量，下载仅在 AutoDL；已由用户取得的本机原包可按授权补传，但不新下载本地大包。失败 partial 与审计记录保留。无法取得完整 raw 时提供真实 converter/拒收契约，缺参考或姿态的能力明确标 unavailable/mask，不造标签。

## 💾 新服务器实际存储

快照开始于 **2026-10-13 16:52:38 UTC / 2026-10-14 00:52:38 北京时间**。只读 inode metadata walk 覆盖明确的数据根、历史层、失败 staging 和有 provenance 的工程数据夹具，按 `(st_dev, st_ino)` 全局去重、不跟随 symlink、不重新 hash 全部 H5。扫描耗时 74.76 s，0 errors。后台 raw partial 和 GPU 工程夹具在有界扫描期间可能增长，此后新下载不追溯算入这个快照。

| 项目 | 分配字节 B | GB / GiB |
| :--- | ---: | :--- |
| 全部数据保存层 | 82,519,162,880 | 82.519162880 GB / 76.852 GiB |
| 12 正式 native（含其 metadata） | 4,601,798,656 | 4.601798656 GB |
| 文件系统总量 | 268,435,456,000 | 250 GiB |
| 文件系统已用 | 93,535,883,264 | 93.535883264 GB |
| 当时可用 | 174,899,572,736 | 174.899572736 GB / 162.888 GiB |
| 已用中不属于本次数据审计范围 | 11,016,720,384 | 11.016720384 GB，含环境、源码、模型和 run 等 |

| 保存层 | GB |
| :--- | ---: |
| 原始归档 | 25.732469 |
| 解压 raw | 46.516679 |
| 当前人体核心 native | 4.458779 |
| 扩展 native（含空审计 metadata） | 0.143045 |
| legacy native | 4.620915 |
| SubT 旧版审计 native 副本 | 0.115118 |
| pilot/debug/staging | 0.287597 |
| 失败转换 staging | 0.012534 |
| 有来源的真实工程数据夹具 | 0.025076 |
| partial/staging | 0.507273 |
| 来源 metadata 与未归类保存资产 | 0.099680 |

| 数据集 | 所有服务器保存层 GB | 当前准入范围 |
| :--- | ---: | :--- |
| RIDI | 1.129062 | 94 native；raw、历史副本与工程夹具另计存储 |
| RoNIN | 38.592004 | 151 native；四包、raw、历史副本 |
| IDOL | 5.142503 | 130 native；building 发布及保存副本 |
| RNIN | 6.658167 | 300 native；raw/归档/pilot |
| TLIO | 14.476202 | 354 native；raw/归档/pilot |
| OxIOD | 4.485308 | 126 native；Tango/syn/standalone test 未接入 |
| IMUNet | 8.778166 | 113 native；完整 ZIP/raw、pilot 和失败 staging |
| EuRoC | 0.172245 | 11 native；AirIO IMU/reference/calibration，不含原始图像 |
| Blackbird | 0.119190 | 20 native；AirIO 片段仅 test，不是完整原始影像发布 |
| GrandTour | 0.031830 | 1 native；单 mission 9 个官方文件 |
| SubT-MRS | 1.284379 | 28 native；ZIP/CSV、1.0.0 审计副本 |
| UZH-FPV | 0.606654 | 1 native；一个官方 v3 Snapdragon ZIP |
| PedLocData | 0.506229 | 当时未完成 partial，不可训练 |
| ADVIO | 0.324440 | 3 来源归档，0 canonical |
| Pegasus | 0.062493 | 7 来源资产，0 canonical |
| TartanIMU | 0.149123 | 混合来源 NPZ 部分，不新增独立来源 |
| AQUALOC | 0.001130 | reference/calibration 与 1 MiB IMU partial |
| HelmetPoser | 0 | raw=0 |

服务器数据总量的逻辑文件大小为 **82,488,375,828 B**，分配大小包含目录开销。本机既有 RNIN/TLIO/OxIOD/IMUNet 原包共 **8,465,048,858 B**，IMUNet hardlink 不重复计数；它们不加入服务器占盘。正式序列统计排除 pilot、legacy、历史/审计副本、工程夹具、空拒收目录和重叠辅助视图。

存储 receipt：服务器 `.local/reports/all-dataset-storage-audit-newserver-final.json`，SHA256 `c39c0e4df640966760f66b101eb2cf51512828ee7256e2f294f3f8f8e3571cac`。后台下载会变化占盘，运行队列仍需逐次核验至少 30 GiB 保留空间并预算 checkpoint、optimizer、预测与图表。

### 跨输入重复质量审计

**2026-10-13 17:11:37 UTC** 的只读 manifest/split 审计覆盖 12 来源、1,329 条；所有有效非空 `imu_sha256` 均存在，**1,329 个 distinct / 0 unknown / 0 exact matches**，跨集与集内 primary train/test 的匹配候选均 0。它只比较 canonical gyro/acc 数组的既有摘要，不覆盖完整时间戳、mask、姿态、位置/标签或完整 task view；没有摘要匹配不证明路线、主体或轨迹独立。Blackbird 相交片段和重复路线仍相关，原有分组 overlap warnings 不删除，数据/split 未修改。

receipt：`.local/reports/cross-dataset-input-overlap-newserver.json`，SHA256 `60adecb98bce1a17de33e802d82964c7d87bdb4662d709355ea67651caa66169`；绑定 immutable source manifest `c832d069f4d6c20d3b8edc7d0a03dfde28ab69acb0cd6f17d7b90b9311fab889`。

## 🧪 全配置准备与正式实验边界

所有测试、转换和模型工程运行均在 AutoDL。原53份配置与新增4份分别在各自source完成CPU工程流程，共57份配置有对应证明；旧49神经配置在4090D完成GPU标准生命周期，新4CUDA/GPU batch128未测。各自 receipt 保存实际 source manifest、配置、输入和过程；短截段只用于工程，不加入正式数据统计或精度榜。

当前默认 `train_eval_gap` 按统一配置实测；历史 1,024 窗口整批诊断与有限工程预算分别记录，不能把改变 batch 组成的分块称为无损替代。本轮统一正式配方已按 49 配置实际容量与批准的源码桥冻结：30 epochs / min 15 / patience 10 / val interval 5 / batch 与 val batch 128 / stride 200 / eval stride 10 / AdamW lr 1e-3。cosine、weight decay 1e-4、AMP=false、cache=true、workers=2、gap_windows=128 等完整参数见[公开协议](protocols/common-benchmark-v1.json)；保留每模型窗口、history、损失与训练阶段，非梯度经典算法不伪造 AdamW 训练。其他数据来源计时仍以测得的 RIDI pipeline 投影，不把预算预测称为整矩阵实测。

新增位置分位数/分轴、连续段漂移、时间 RPE 和原生任务高斯不确定性已有代码；其中原生任务不确定性的 25 个节点已在服务器实际全部通过。无姿态预测时 rotation RPE 明确不可用；不把参考姿态当预测。只有原生 task 的 target/prediction/covariance variable、frame、units、DOF 一致才计算不确定性；缺 covariance/logstd 不假设单位方差。主 XYZ 与同预测 XY、对齐诊断、oracle、reference/oracle 输入、CPU-only 校准、同域/跨域和 XY 原文参考协议分别标识。

绑定 `2efd` source manifest 前缀 `607ebacf` 的框架目标 49 unique 节点最终全部通过：首次 45 passed / 4 fixture failed，修正测试夹具后独立复跑 4/4 passed；保留首次失败证据，不重跑或重复计数此前 45 项。25 个不确定性节点不与 49 项简单相加。四个 test-only transfer 流程实际执行 train-only 经典方法 → val/test/保存 → public shared state → target test-only；错误 seed、来源和篡改 checkpoint 的负对照明确拒绝。当前正式 immutable source manifest `1897c9ccfa959199ef3f8dbf8c7863dd40d5472d4da9b5551154366458650949` 的锁、来源/seed 身份与 dispatcher 8 定向节点实际通过；CPU/GPU 历史科学源码桥分别批准。正式 GPU 实验已运行，2026-10-13 17:59 UTC 已有单元完成 train/best/test，覆盖尚未完成。

历史 719/144/81/14 项验收各自保留当时的源码、资源和覆盖证据；本轮定向增量不与历史计数相加，也不把过去 skipped/resource_unverified 改写为全部通过。当前准备通过不证明论文 accuracy，尚无正式排行榜或收缩机制定论。

### 完整 13 来源采样准入

2026-10-13 **17:59:39 UTC**，425 个文件的冻结源码完成 714.028 s、RC0 的服务器只读采样资格链：**2,067 行全部核验，2,019 passed / 48 unsupported / 0 pending，377 个唯一 taskview identity**。采样均为 train stride 200 / eval stride 10；每个合法 split 记录实际 valid_windows、原生目标/输入形状、额外输入及 XYZ→XY 精确截取，三种子共享同一科学采样身份。

全部 **1,323 个神经训练行与 588 个神经 test-only 行通过**。48 个不支持行来自 `mean_speed_heading` 和 `pdr`，每个配置 24 行；EuRoC、Blackbird、OxIOD、TLIO、GrandTour、SubT-MRS、UZH-FPV、PedLocData 各 6 行，原因都是声明的 primary/extra split 缺少设备 `imu/orientation`。OxIOD 此限制来自 Nexus5 子集。所有行仍在完整矩阵，不以 reference quaternion 暗补 device input。

13 来源的 manifest 与全部 split 摘要、425 个冻结源码文件逐 SHA 前后完全相同，检查不执行模型或 GPU。server-original qualification receipt SHA256 为 `8753d8a6d35b0ada587abcc3cd365e957dc392ee150e185be6a3e9f4e2da5bb5`；完整 qualified coverage SHA256 为 `a271ba4ceef4eb6780acbbb7cfc5c92b9caa702f68c676676d4de6726acb044c`。本地 JSON 重排镜像的字节摘要与原服务器 receipt 不同，精确原字节镜像与语义等价证明另存；公开协议引用原服务器摘要。正式 coverage 由执行 owner 合并资格，逐单元 run-state 独立保存，不重置在运行或已完成的单元。

## 🛣️ 完整矩阵与研究路径

三个种子 **1735231569 / 843628571 / 951150512** 由操作系统随机源一次抽取，所有合格配置/来源共享，记录摘要和逐单元 override；工程测试自己的 seed 不冒充正式三 seed 实验。运行中的旧冻结源动态枚举53×13×3的2,067个覆盖单元，能力不适用和 checkpoint 依赖完整保留；不以一次 dry-run 或一轮 GPU smoke 宣称 coverage completed。

| 顺序 | 工作 | 必需验收 |
| :--- | :--- | :--- |
| 1 | 全来源/全配置科学 taskview 和资格 | 实际窗口、单位/参考权限、XYZ/XY、训练与目标 split、输入能力、独立来源及 warnings |
| 2 | 已冻结公平采样与正式预算 | 统一配方与实际容量绑定；完整任务视图已验收，持续核算真实执行时间/存储，train/val 选模，不用 test 调参 |
| 3 | 正在执行全矩阵与 checkpoint 依赖 | 同三 seed、immutable source/recipe、单元状态与原子结果保存；test-only 来源核验预注册 origin，未完成单元保留真实状态 |
| 4 | 统一评价与收缩诊断 | XYZ 主结果、同预测 XY、oracle/长度支持、幅值方向/模式/精度/泛化；报告有效覆盖和 NA |
| 5 | 研究发布 | 足够训练的比较和受控消融、可复现数据/算法矩阵、代码/配置/来源与局限；目标 ICRA / IEEE RA-L |

数据与代码增量可以并行，但正式结果必须基于真实准入和冻结协议。会议日期与录用不作未经核实的承诺。
