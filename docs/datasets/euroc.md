# EuRoC MAV / AirIO 精简发布数据卡

转换器：[`euroc.py`](../../src/inertial_benchmark/data/converters/euroc.py)，`euroc@1.0.0`。
通过现有 `RawSequence → convert_dataset → IPB v1` 管线；不导入旧 v0.1 canonical。

来源：[ETH 官方数据页](https://projects.asl.ethz.ch/datasets/euroc-mav/)、
[原始发布 DOI](https://doi.org/10.3929/ethz-b-000690084)、
[AirIO 作者发布](https://github.com/Air-IO/Air-IO/releases/tag/datasets)。
当前取得的 AirIO 包只包含 IMU、参考状态和校准，不含原始双目图像，不称完整视觉数据集。
原数据许可条款未在此次证据中独立核实，不能把 AirIO 代码 BSD 许可当成数据许可。

| 实收文件 | 字节 | 观测 SHA256（上游未给 whole-file digest） |
|---|---:|---|
| `EuRoC-Dataset.zip` | 20,221,602 | `de09fdd03e9147461fd678aa6763454bdf8d1f07172753a9ec80073e9be38e4a` |
| `V101-state_groundtruth_estimate0.zip` | 185,158 | `2006e18061a30716879b1c7b0caad04fd6b2f8755d35859021c1dff3ee1e9c7c` |

包内 11 条 UAV 序列，均保留全段 CSV。第二包是 OpenVINS 重算的 V1_01 参考；本转换器默认**不应用**，
输出属性 `reference_variant=original_released_batch_state`，避免把不同参考协议混成一张评分表。
原始 ns 时间戳以 int64 读入，先减同一整数起点再换秒，不把 10¹⁸ ns 直接变 float 再相减。

| 来源字段 | 单位、坐标与 v1 去向 |
|---|---|
| `mav0/imu0/data.csv` | ns、机体系 gyro rad/s、比力 m/s²；原值进入 `RawSequence` |
| `state_groundtruth_estimate0/data.csv` | ns、`p_RS_R` m、`q_RS` wxyz、参考速度、gyro/acc bias |
| 两份 `sensor.yaml` | 必须同时存在且 `T_BS` 为单位阵；否则明确拒收，不能默认传感器 S=机体 B |
| 参考位置 / 姿态 | S 原点在 R 中的位置、S→R 姿态；identity `T_BS` 证实它们对应 IMU 原点和轴系 |
| 参考 bias | 保留在不可变源文件、由文件哈希追溯；不应用到网络输入 |

Machine Hall 参考状态由 Leica 位置与 IMU 批量估计，姿态依赖 IMU；Vicon Room 由 Vicon/IMU 批量估计。
不能把每条姿态都标成独立 Vicon 真值。所有来源 CSV/YAML SHA、header、校准文本写入序列属性。
解析器不重采样；现有 v1 管线统一 200 Hz、线性位置、SLERP、跨缺口无效 mask、首尾裁剪，全部进转换报告。

默认按整条录制划分，5 train / 1 val / 5 test：

- train：`MH_01_easy, MH_03_medium, MH_05_difficult, V1_02_medium, V2_01_easy`。
- val：`V2_03_difficult`，从 AirIO 六条训练录制中预先留出。
- test：`MH_02_easy, MH_04_difficult, V1_03_difficult, V2_02_medium, V1_01_easy`。

AirIO 实际发布 train/val 列表相同；原列表记为 `official_*`，不能用于正常选模。
`group_id=recording_id=完整序列名`；上述默认划分录制不交叉。

该集是单独 UAV 参考姿态赛道。未提供可部署设备姿态，`orientation=device` 必须拒绝；
`orientation=reference` 是使用参考姿态的特权/oracle 协议，不与手机部署姿态或纯惯性榜单混排。

```bash
ipb convert dataset=euroc source=/path/to/EuRoC-Dataset output=/path/to/ipb/euroc workers=1 overwrite=false
ipb check data=/path/to/ipb/euroc full=true hash=true
```

2026-10-13 AutoDL 实物验收：11 条全部接收、0 拒收；原生 200 Hz H5 合计 15,790,930 B，
有效录制总时长 1,348.34 s，默认 train/val/test 为 5/1/5。完整校验与哈希校验均为 0 errors；
两条 warnings 均是已声明的 `official_train/official_val` 重叠，不影响默认无交叉划分。
数据指纹为 `74ba02c5f72ace2795b8124a2497d5295c868a8a7f84b288d6e0658ebaf976b4`；
`conversion_report.json` SHA256 为
`6186ef667fc7d3909d18490306847a5847b848583ce70be1769123ec6cb5e6bc`。

