# 🧭 本轮状态 / Project Status

> 更新：2026-10-13。主干审计来源 `dev/v1@faee13c`。区分代码存在、当前服务器验收、历史结果与待办；正式 GPU benchmark 尚未完成。

## 交付与执行阶段

最终交付是统一数据/转换契约、保留算法语义的接口、可比评价和可审计实验。一切按 dev/v1 的 NIO/ipb/IPB v1 与现有 model/converter API 接入；旧 main 的框架增量仅归档，有效能力后续按 v1 重新实现，不迁入其训练器、IB API 或同名指标定义。

| 阶段 | 现状与证据 |
| :--- | :--- |
| v1 代码 | 29 个注册架构、31 份模型配置、21 个算法包、八个核心转换器与 train/val/predict/report 主干；导出 API 尚缺 |
| 历史结果 | HANDOFF 的 705 tests、旧机器成绩和 AMP/dropout 观察为 2026-09-24 历史，不等于当前 AutoDL 复验 |
| 当前 AutoDL | RIDI / RoNIN 各三条录制已用原生转换器接收，均 0 拒绝；短 CPU NIO 闭环已完成；原冻结 719 节点覆盖为 665 passed / 52 skipped / 2 resource_unverified，新绘图补丁目标验收另记 |
| 旧候选评价 | 另一接口的服务器检查与代表预测，不计入 v1 当前测试或排行榜 |
| GPU 阶段 | CPU 准入、预算/配方冻结后开展充分训练，目前不发布正式算法排名 |

## 数据接入

| 数据 / 资产 | 本轮事实 | 下一步 |
| :--- | :--- | :--- |
| 八个核心转换器 | RoNIN、RIDI、OxIOD、TLIO、IDOL、RNIN、IMUNet、PedLocData 已有代码 | 当前 raw/canonical 存量与 receipt 分别核对，不冒认历史大规模转换数量 |
| [EuRoC](datasets/euroc.md) | 11 条录制转换为 11 份原生 v1 HDF5，full/hash check 零 errors；correction 未应用 | 整录制 5 train / 1 val / 5 test，reference/oracle；部署输入权限仍需分 track |
| [Blackbird](datasets/blackbird.md) | 20 个发布片段转换为 20 份原生 v1 HDF5，full/hash check 零 errors | route 保守同组，仅 test 审计目录，不等于独立 20 次飞行或训练/验证划分 |
| [Pegasus](datasets/pegasus.md) / [ADVIO](datasets/advio.md) | 实际识别并拒收 7 / 3 个来源资产，两者均 0 canonical | 输出审计拒收报告；待来源姿态/帧契约证实后再接入 |
| [TartanIMU 2025 来源审计](datasets/tartanimu_2025.md) | 新增审计卡，无本轮 converter 注册 | 不把来源卡计为可训练数据或转换器 |
| TLIO / IMUNet | 原生转换器已有，本轮 raw 未实收 | 服务器网络限制，继续核验 AutoDL 公开下载入口；实收后记录资产哈希并按 v1 转换/校验，已有数据与其他工作继续 |

新增共四个 converter 注册入口：EuRoC/Blackbird 两个原生转换器、Pegasus/ADVIO 两个审计拒收入口；TartanIMU 2025 只有来源卡。逐数据集细节见 [datasets/](datasets/)。本轮新增合计 31 份原生 v1 文件、24,158,461 字节：EuRoC 15,790,930 字节 / 1,348.34 s；Blackbird 8,367,531 字节 / 759.25 s，保守划为 10 个 route groups。转换器针对四个文件的服务器测试为 10 passed（5.60 s），ruff 通过；两数据集 full/hash 检查均零 errors；保留 EuRoC 两条原官方集合重叠 warning，以及 Blackbird 缺少 train/val 和三个 egg 片段重力 marginal 共五条 warning，不隐去边界信息。验收记录为 `new-datasets-v1.json`，SHA256 `1b5f9d393804ef5fe898cb9ab9fa064ad724f9fbc4915735a9f49a35c066d4a5`；完整 suite 计数随后单独记录。

## 当前服务器 CPU 闭环证据

2026-10-13，pristine `faee13c` 核心加独立 CPU 验收 helper，实际完成：

| 项目 | 实测范围与结果 |
| :--- | :--- |
| raw → 原生 v1 | RIDI 三条、RoNIN 三条录制各 3 接收 / 0 拒绝；保留六条原生文件 |
| 短训练数据 | 每个数据集取不同 train/val/test 录制，各截 800 样本（200 Hz）；只用于 engineering_only 闭环 |
| 模型与训练 | NIO ronin_resnet18，4,634,882 参数；每数据集 4 training windows，batch=2，1 epoch，2 optimizer updates；workers=0、cache=false |
| 重载与结果 | checkpoint 重载后的 val/predict 为严格 array_equal，maxabs=0；预测 NPZ 与 PNG 保存 |
| 运行环境 | PyTorch 2.12.1+cu130，线程数 1，CUDA unavailable；当前无卡 CPU 模式约 0.5 CPU / 2 GiB |
| 模式守卫 | 未充分训练的 dropout delta 有 warning/error，记录诊断，不推导算法排名或收缩机制 |
| 原冻结 suite | 719 unique 节点：665 passed / 52 skipped / 2 resource_unverified；首次单进程 SIGKILL 未生成 JUnit，随后逐文件隔离覆盖与原参数补测；不是全部通过 |
| 新绘图补丁 | 有效维度时间图/CDF与多模型分页的独立目标验收待服务器执行，不并入原冻结 719 节点计数 |

闭环用截段与极少更新，验证接口、保存和重载一致性，不能证明充分训练精度，也不是完整 benchmark 数据发布。

原科学验收冻结 source manifest SHA256 为 `89736fbb634a023bcbf7a47fb7e04a26c6de2fdb2d170b08005441c45c669c40`。61 个文件分片先得到 635 passed / 69 skipped；随后 13 项 CTIN 常规测试通过，RIDI 8 / RoNIN 6 / IDOL 3 项真实 raw 测试通过并替换原跳过项，按 nodeid 去重后得到 665 / 52 / 2，不重复累加。52 个跳过项分别为：16 项需要已转换数据、9 项 IMUNet 与 11 项 OxIOD 使用不可用的旧 raw 路径、4 项 PedLocData / 7 项 RNIN / 5 项 TLIO 缺 raw。

两项未验证都是 CTIN：原参数 mean 训练/验证闭环测试进程 SIGKILL（-9），center 未独立执行。最后一次 hook 位于首次训练 `B=16, N=200` 的 decoder/MHA，尚无顶层 forward_end/batch_end，也未到验证或 gap；没有证据据此确定 OOM 原因。唯一节点覆盖记录为 `integrated-v1-final-unique-coverage.json`，SHA256 `ac1b9a2f93e2b1ea64097d232d5c2265533a20b73b1c14e5d6d579e41e8f51da`。

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
| 本轮实现，待目标验收 | 有效维度图与指标一致 | 时间误差/CDF按 metric_dims 和实际预测维数绘制并标注；独立 server 目标测试待执行，水平/垂直分解与3D轨迹图仍待扩展 |
| 本轮实现，待目标验收 | 超过八模型的分页 | 每 dataset/split 分页保留所有模型箱线/CDF/PLR，全局 Pareto 集合保持完整；独立 server 目标测试待执行 |
| P1 | NA、覆盖与定义目录 | 记录缺失能力原因、单位、DOF、support、长度分辨率、aggregation 和评价源码指纹 |
| P1 | 严格 gap-aware profile | 现 RTE/D-RTE 允许有效端点跨 gap，短 RTE 会换算；新连续支持/不外推诊断独立版本化，保留旧定义 |
| P1 | 六面板 / HTML / 离线重算 | 复用 SequenceResult.load 和当前绘图/报告；新输出 receipt，不覆盖旧结果 |
| P2 | 姿态 / 概率诊断 | 仅在能力足够时启用；预测/参考与 covariance 的 variable、frame、units、DOF 必须匹配，不把 GT 当预测 |

单独 group 的不确定性、seed/序列覆盖不齐的权重和多模型统计需要明确。aligned ATE、oracle 与新严格诊断保持独立，不对主排名用 GT scale 校准。

## 收缩研究与预算

先验证 oracle/积分与长度分辨率，再检验幅值/方向、训练和推理模式、AMP/dropout 与泛化。复用历史 mode guard，但以当前数据和充分训练复验。先一个 seed 调通配方与预算，最终主结果和关键消融最多三个 seed。CPU 阶段降低准备成本，GPU 按冻结队列执行。
