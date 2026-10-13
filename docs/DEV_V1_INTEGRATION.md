# 🧭 本轮状态 / Project Status

> 更新：2026-10-13，19:57（北京时间）只读复核。当前验收来源 `dev/v1@2c1f57f`；区分代码存在、当前服务器验收、历史结果与待办。正式 GPU benchmark 尚未完成，实际 GPU 探测尚未执行，训练队列未 armed。

## 交付与执行阶段

最终交付是统一数据/转换契约、保留算法语义的接口、可比评价和可审计实验。一切按 dev/v1 的 NIO/ipb/IPB v1 与现有 model/converter API 接入；旧 main 的框架增量仅归档，有效能力后续按 v1 重新实现，不迁入其训练器、IB API 或同名指标定义。

| 阶段 | 现状与证据 |
| :--- | :--- |
| v1 代码 | 29 个注册架构、34 份模型配置（含三份三维适配）、21 个算法包、八个核心转换器与 train/val/predict/report 主干；导出 API 尚缺 |
| 历史结果 | HANDOFF 的 705 tests、旧机器成绩和 AMP/dropout 观察为 2026-09-24 历史，不等于当前 AutoDL 复验 |
| 当前 AutoDL | 六个核心数据集完整转换得到 1,155 份原生 HDF5、99.40744861 小时、725 条 train 序列，full/hash 零 errors，全部 train_ready=true；新增三维目标验收 144 passed / 0 skipped / 0 failed。此前短 CPU NIO 闭环及原 719 节点、`6d44573` 的 81 项验收按历史范围独立记录 |
| 旧候选评价 | 另一接口的服务器检查与代表预测，不计入 v1 当前测试或排行榜 |
| GPU 阶段 | CPU 准入、预算/配方冻结后开展充分训练，目前不发布正式算法排名 |

## 三维主协议增量

主 benchmark 使用真实预测 XYZ，以三维指标为主，并从同一预测的 XY 派生二维辅助指标。首批为 `ronin_resnet18_3d`、`ronin_lstm_3d`、`imunet_3d`（均标 `3d_adapted`）和原生三维 `tlio`；前三者共用原架构，保留必要的目标布局和损失，扩展可学习三维输出，不宣称原文二维 checkpoint 或论文精度等价。原二维配置/夹具作为独立参考保留。

三维请求不得静默退成 XY，也不能以真值或零填充替代未预测的 Z；PDR 和常速航向的平面变体拒绝 `dims=3`，IONet/EqNIO-RoNIN 当前仍为二维专用。此前私有候选矩阵为四模型 × RIDI/RoNIN/IDOL × seed 0，仅验证集；当前六个 full-core 数据集均通过数据准入，扩展矩阵仍需重新绑定源码、数据指纹与容量预算，队列未 armed。旧混合二维计划已退役。CTIN 独立资源准入，不进入首批队列。

`2c1f57f` 的三维头/目标/损失/重载、严格评价、XY 派生与容量准入目标测试已在 AutoDL CPU 完成：144 unique passed / 0 skipped / 0 failed。此计数仅覆盖本次新增范围，不与下列历史 719/81 项相加，也不表示旧 suite 已全部通过。GPU 阶段先执行有时间上限的单模型容量测量，再冻结训练预算；实际 GPU 探测尚未执行，尚无 GPU 训练或三维精度复现结论。

## 数据接入

| 数据 / 资产 | 本轮事实 | 下一步 |
| :--- | :--- | :--- |
| 八个核心转换器 | RoNIN、RIDI、OxIOD、TLIO、IDOL、RNIN、IMUNet、PedLocData 已有代码；RIDI/RoNIN/IDOL/RNIN/TLIO/OxIOD 六集完整数据已验收 | 用 full-core 原生数据开展 GPU pilot，来源指纹、划分和警告见下表 |
| [EuRoC](datasets/euroc.md) | 11 条录制转换为 11 份原生 v1 HDF5，full/hash check 零 errors；correction 未应用 | 整录制 5 train / 1 val / 5 test，reference/oracle；部署输入权限仍需分 track |
| [Blackbird](datasets/blackbird.md) | 20 个发布片段转换为 20 份原生 v1 HDF5，full/hash check 零 errors | route 保守同组，仅 test 审计目录，不等于独立 20 次飞行或训练/验证划分 |
| [Pegasus](datasets/pegasus.md) / [ADVIO](datasets/advio.md) | 实际识别并拒收 7 / 3 个来源资产，两者均 0 canonical | 输出审计拒收报告；待来源姿态/帧契约证实后再接入 |
| [TartanIMU 2025 来源审计](datasets/tartanimu_2025.md) | 新增审计卡，无本轮 converter 注册 | 不把来源卡计为可训练数据或转换器 |
| [TLIO](datasets/tlio.md) | 本轮 raw 全包与服务器哈希已验收，354 条原生序列 full/hash 零 errors，train_ready=true | VIO 参考、分组重叠警告保留，进入容量测量候选 |
| [OxIOD](datasets/oxiod.md) | 本轮 raw 全包与服务器哈希已验收，128 个候选接收 126 / 拒收 2，五个主划分非空 | Vicon 参考；Tango、syn 和独立 test 目录暂不接入 |
| [RNIN](datasets/rnin.md) | 本轮 raw 全包与服务器哈希已验收，301 个候选接收 300 / 拒收 1，train_ready=true | 混合参考：75 条外部系统、225 条 BVIO，test 9 条全为 VIO；与 IDOL raw 分开记账 |
| [IMUNet](datasets/imunet.md) | 匿名下载需登录，尚未实收和验收完整 raw | 等待用户登录浏览器下载后上传；不读取浏览器凭据 |
| [PedLocData](datasets/pedlocdata.md) | 已确认公开媒体入口，但有界慢速尝试仅留下未验收的部分文件，尚不可训练 | 暂缓下载，等待更快链路后断点续传并完整校验 |

### 六个核心数据集全量转换验收

| 数据 | 接收 / 拒收 | 实际划分（条） | 原生输出字节 / 检查 warnings |
| :--- | :--- | :--- | :--- |
| [RIDI](datasets/ridi.md) | 94 / 0 | train 45、val 4、官方 test 23，另 test_unseen_subject 11、test_unlisted_seen_subject 11；五项合计 94 | 117,002,894 / 3 |
| [RoNIN](datasets/ronin.md) | 151 / 1 | train 71、val 16、test 64（test_seen 32 + test_unseen 32）；另 test_ood_device_id 为 test 的诊断子集 | 964,361,898 / 7 |
| [IDOL](datasets/idol.md) | 130 / 0 | train 38、val 5、test 87（test_known 51 + test_unknown 36）；building 子集与这些集合有交集，不重复累加 | 833,504,808 / 34 |
| [RNIN](datasets/rnin.md) | 300 / 1 | train 240、val 51、test 9 | 357,298,253 / 15 |
| [TLIO](datasets/tlio.md) | 354 / 0 | train 283、val 35、test 36 | 1,283,873,945 / 3 |
| [OxIOD](datasets/oxiod.md) | 126 / 2 | train 48、val 13、test 9、test_unseen_subject 30、test_unseen_device 26 | 571,135,975 / 3 |

此前 RIDI/RoNIN/IDOL 批次共 **375 份 HDF5、1,914,869,600 字节**；本轮新增 RNIN/TLIO/OxIOD 共 **780 条、53.695518 小时**。六个核心数据集总计 **1,155 条、99.40744861 小时、725 条 train 序列**，全部 full/hash 检查零 errors，`train_ready=true`；上表前三集沿用此前 HDF5 文件字节证据，新增三集为发布 receipt 的原生输出文件字节总和（含 metadata），不是纯 HDF5 大小；当前磁盘占用另按下表去重统计。这证明数据契约和划分可进入训练，不是已完成充分训练或论文精度验证。作者官方 known/seen 划分允许 subject/group 共享，RIDI/RoNIN/IDOL 的 3/7/34 条 warning 均保存；没有序列重复不等于所有 train/test 主体独立。RoNIN `a011_3` 使用官方选择的 gyro_integration 姿态后，重力平均倾角 4.1° 导致物理检查失败，明确拒收。

服务器仓库根目录的相对位置为 `datasets/ipb-v1/full-core/{ridi,ronin,idol,rnin,tlio,oxiod}`。使用 `export IPB_DATASETS="$PWD/datasets/ipb-v1/full-core"` 后，六个同名 `data=` 配置直接指向本轮完整数据；不再把 `ridi_smoke` / `ronin_smoke` 当正式训练数据。

| 数据 | 实际 dataset fingerprint |
| :--- | :--- |
| RIDI | `b4f38fa9a5fef00f0c5f91e73436524a16a3e36843f9dce9868dc5dba8801aa2` |
| RoNIN | `b2b88e29ccad818bf2dc7de3e93fcc44c8f59982e8aa66affd04e4d74f72968c` |
| IDOL | `895b5618a295eae3b34d445929c23f679fbedc8b8ec092f9e6a186f8bba39e5c` |
| RNIN | `3b0898a8cf2a6c2a74965945aa9798dc36754f0627f1c7e20db320b7fbc361af` |
| TLIO | `c2d0ddbe9d6ea11fa3e3ecd57783bc7a2abd4c2ea35550bc7c07922917d29638` |
| OxIOD | `15551c06363ecfb6ca0651ed81dd5a1d24bda7f82028bdeb4fbe6de4c93a59e7` |

本轮 RNIN/TLIO/OxIOD 全量转换与独立 CPU 数据验收绑定 `2c1f57f`，部署清单 `source_manifest.json` 的 SHA256 为 `5549c535e0e32b4240c2c4353440a7ef622b22d17165f445a60e1b4abaf4a761`，清单与其中逐个核验的 365 个受管源码文件在执行前后均不变；各新增数据发布时，此前已就绪核心数据的 manifest/指纹也保持不变。三集分别完成四个首批模型任务视图 × train/val/test 的 12 个真实 B=2 XYZ 批次与四项 CLI dry-run；XY 与同一 XYZ 目标截取一致。这里只验数据与接口，不运行 GPU 训练，不建立精度排名。

RNIN 位置标签为混合参考：75 条使用 VICON 或其他外部定位系统，225 条使用 BVIO；其 9 条 test 全为 VIO，不将这部分称为独立外部真值，也不将 75 条全部标为 VICON。TLIO 使用 headset MSCKF/VIO 参考；OxIOD 保留 Vicon 实测 XYZ，但标记点与 IMU 的 lever arm 未知。OxIOD 的 val 从官方 train 按会话生成，与 train 会话不重叠；官方 test 和额外 subject/device 集不混入 train。iPhone 的 gravity/user_acc 按 G 恢复完整加速度，Android 使用对应 SI 列；八条 Nexus5 序列缺设备姿态，仅设备姿态依赖方法不能使用。RNIN 的 15、TLIO 的 3、OxIOD 的 3 条 full/hash warning 保留，包含官方分组重叠和边缘重力诊断。OxIOD 的 `handbag_data2_seq3` 与 `multi_users_user5_seq5` 因时间戳精度损坏拒收；Tango、syn 与独立 test 目录是未接入范围，不计入两条拒收。

此前 RIDI/RoNIN/IDOL 三集全量转换执行绑定源码 `0ae5ea3`，manifest SHA256 `7573a39dbe62a4fdeda0ac6eb6a67aee9c6d4da19c426f0cf211ec2d0b9689f3` 前后不变。执行 receipt SHA256 `52238c2b45f28a2a6a2f486f297dd8cb266db5021a4ab2fdf7bcf04a707eff80`；`full-core-completion-summary.json` SHA256 `f96229f385e46dd9bd9dcaf2297f719c50bfb4ee97f90810840ae3312a25d877` 汇总了 raw 来源哈希、conversion reports、full/hash reports、划分与拒收原因。此数据验收与下面的旧科学 suite、81 项目标测试分别记录。

`cc6554b` 对此前 16 个 converted-data 节点另作诊断，结果为 3 split-policy passed / 10 missing-dataset skipped / 3 historical-release fingerprint failed。后三项的 converter、200 Hz、序列数、时长和距离断言已通过，但新生成数据的 fingerprint 未匹配历史 release 前缀；新 HDF5 的 full/hash 检查通过不等于历史发布物逐字节复现。独立 receipt 为 `cc6554b-fresh-release-receipt.json`，SHA256 `10fc43218031702316b83171cbb1370f1d3c9fc3598556bb4101aa120642e4d4`，不改变下面旧验收计数，也不据此宣称 CI 全绿。

新增共四个 converter 注册入口：EuRoC/Blackbird 两个原生转换器、Pegasus/ADVIO 两个审计拒收入口；TartanIMU 2025 只有来源卡。逐数据集细节见 [datasets/](datasets/)。本轮新增合计 31 份原生 v1 文件、24,158,461 字节：EuRoC 15,790,930 字节 / 1,348.34 s；Blackbird 8,367,531 字节 / 759.25 s，保守划为 10 个 route groups。转换器针对四个文件的服务器测试为 10 passed（5.60 s），ruff 通过；两数据集 full/hash 检查均零 errors；保留 EuRoC 两条原官方集合重叠 warning，以及 Blackbird 缺少 train/val 和三个 egg 片段重力 marginal 共五条 warning，不隐去边界信息。验收记录为 `new-datasets-v1.json`，SHA256 `1b5f9d393804ef5fe898cb9ab9fa064ad724f9fbc4915735a9f49a35c066d4a5`；科学 suite 与目标验收范围分别记录如下。

## 当前数据磁盘清点

2026-10-13 20:02（北京时间）只读统计覆盖服务器当前 v1 与历史数据目录，按 `(st_dev, st_ino)` 去重，不跟随候选目录的 symlink 别名。12 个数据集有文件（含 PedLocData partial）；实际占用 **71,288,442,880 B = 71.28844288 GB = 66.3925 GiB**。这里使用文件系统分配字节（含目录开销），GB 为 10⁹ B、GiB 为 2³⁰ B；当时可用空间约 **130.83 GiB**，可用空间不等于数据体积。真实保留的归档、解压数据与历史副本各自占盘，均计入总数。

| 保存层 | 分配字节（B） | GB |
| :--- | ---: | ---: |
| 原始归档 | 22,243,061,760 | 22.243062 |
| 解压 raw | 39,895,445,504 | 39.895446 |
| 当前六核心原生输出 | 4,131,315,712 | 4.131316 |
| 历史原生副本 | 4,620,910,592 | 4.620911 |
| EuRoC/Blackbird 等原生扩展 | 24,379,392 | 0.024379 |
| pilot / 调试 / staging | 271,200,256 | 0.271200 |
| 未完成 partial | 2,502,656 | 0.002503 |
| 元数据及未归类资产 | 99,627,008 | 0.099627 |

当前六核心加 EuRoC/Blackbird 的原生格式合计约 **4.15 GB**，区别于所有保存副本的 **71.29 GB**。八个非空当前原生数据集合计 **1,186 条、99.99289 小时、730 条 train 序列**；Blackbird 仅 test，不把八个非空 manifest 等同于八个可训练数据集。此清点读取现有 metadata，未重新运行 full/hash；数据准入依据上文已完成的独立验收，不重复计数历史或 pilot，也不累加辅助 test 子集。

| 数据集 | 所有服务器保存层 GB | 当前范围 |
| :--- | ---: | :--- |
| ridi | 1.104163 | 完整当前核心原生数据；保留 raw/历史副本 |
| ronin | 38.592078 | 完整当前核心原生数据；保留四个官方归档及 raw/历史副本 |
| idol | 5.142503 | 完整当前核心原生数据；保留 building 归档与 raw/历史副本 |
| rnin | 6.658167 | 完整当前核心原生数据；含归档、raw 与 pilot |
| tlio | 14.476202 | 完整当前核心原生数据；含归档、raw 与 pilot |
| oxiod | 4.485308 | 126 条当前核心原生数据；额外 Tango/syn/test 未接入 |
| euroc | 0.172245 | 11 条原生录制，5 train / 1 val / 5 test；取得的发布版本 |
| blackbird | 0.119190 | 20 个发布片段，仅 test / 10 route groups |
| pegasus | 0.062493 | 归档与拒收报告；0 条当前原生序列 |
| advio | 0.324440 | 三个来源归档与拒收报告；0 条当前原生序列 |
| tartanimu | 0.149123 | 公开 raw 样例子集，不是完整数据集 |
| pedlocdata | 0.002503 | 仅未完成 partial；不可训练 |
| 共享 / 未归类 | 0.000029 | 不归入特定数据集 |

各行包含该数据集的原始归档、解压数据、当前/历史原生输出及 pilot 等实际保存层，舍入后合计可能有末位差异；这些体积对应已取得的版本或子集，不代表所有公开全集。IMUNet、HeadLoc、Aria、AQUALOC、GrandTour 在本次限定目录清点中未发现文件，这不证明其来源或转换器已接入；待单独来源审计与下载验收。本地 RNIN/TLIO/OxIOD 三个原始归档共 **5,872,302,083 B（5.8723 GB）**，独立保留，不计入服务器总量。

## 当前服务器 CPU 闭环证据

当前 `2c1f57f` 的新增三维协议与容量准入目标测试为 **144 unique passed / 0 skipped / 0 failed**，独立于历史 suite。2026-10-13 19:57（北京时间）只读复核确认上述部署清单及逐文件哈希未变化、CPU/GPU 锁空闲，无活动下载/转换/训练 worker；实际 GPU 能力仍待探测，预算和队列未冻结/armed。

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
