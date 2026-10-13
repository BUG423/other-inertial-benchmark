# UZH-FPV（官方 Snapdragon v3 文本版）

UZH-FPV 是独立采集的无人机竞速数据来源。本版直接读取官方公开参考的 Snapdragon
ZIP，仅解析 `imu.txt` 和 `groundtruth.txt`，不用影像作模型输入，也不展开或执行归档代码。
同来源的 AirIMU、Tartan 等再次打包版本不增加独立来源数。

## 官方来源与版本

- [官方主页与参考轨迹生成论文](https://fpv.ifi.uzh.ch/)
- [公开序列、下载和许可证](https://fpv.ifi.uzh.ch/datasets/)
- [位姿格式：IMU 原点、米、秒、xyzw、IMU→world](https://fpv.ifi.uzh.ch/2019-2020-submission-format/)
- [作者参考轨迹质量验证代码](https://github.com/uzh-rpg/uzh_fpv_open)
- [本次 indoor_forward_6 官方 v3 包](https://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_6_snapdragon_with_gt.zip)

本次服务器完整包 606,282,258 B，3,945 个成员，CRC 校验通过。
SHA256：`5da4d454c35b9b20d3566460d6f25f93b446d316eec05c828c861bf901c4f538`。
作者未提供已核实的预期 MD5/SHA；这里是取得文件的身份与传输完整性证据。
已在服务器通过 8 个科学测试，并完成真实转换、全数组/哈希检查及 B2 XYZ／XY
任务视图验收：1 条接收、0 条拒收，30.198 s、194.6 m，0 errors；仅缺少 train/val
的 2 条 warnings。IMU 的完整 69.628 s 中只有 30.198 s 有公开参考，规范化数据
只保留真实时间交集，未为无参考时段补造位姿。当前为独立来源的 test-only 子集。

## 坐标、时间和参考合同

官方发行的 IMU 数据行实际为 sample index、Unix 秒、gyro XYZ、accel XYZ；其注释标题
省略了 sample index，解析以真实八列结构为准。角速度 rad/s、加速度 SI 比力（含重力）。
十进制 epoch 先减去同一 IMU 时间原点，再转浮点秒；检查严格递增的时钟和连续整数索引。

参考行为时间戳、位置 XYZ、四元数 xyzw。按官方位姿约定，它是 IMU 在参考世界系中的
位姿；转换器不再次附加相机或 Leica 棱镜杆臂。输出 canonical XYZ、wxyz/body_to_world，
重力 z-up 必须通过真实物理检查，而非仅靠格式命名。

2022 更新的 v3 全位姿来自基于视觉的参考轨迹生成，并用 Leica 外部位置作质量验证。
姿态不能称为独立运动捕捉真值；参考姿态属于 privileged 模式。当前没有设备姿态输入，
也不以参考估计偏置标定输入 IMU。原始 Leica 距离/位置与更新全位姿是不同来源版本。

FPV 飞行速度高于人体步行，速度硬阈值使用数据域专用的 median 25 m/s、p95 35 m/s；
官方公开 Snapdragon 序列 vmax 可达 20.73 m/s。单位、重力、陀螺动态一致性和时间重叠
仍执行原科学检查，不能通过整体放宽所有阈值制造接收。

## 范围与划分

当前包是一条完整公开参考飞行，初始接入为 benchmark 派生的 test-only 目录，不假造
train/val，不把轨迹切片当独立飞行。扩充到更多官方公开 GT 录制后再设计组级训练协议；
官方 withheld GT 的比赛序列不伪造真值。

```bash
ipb convert dataset=uzh_fpv source=/path/to/official_zip_directory output=/path/to/ipb-v1/uzh_fpv workers=1
ipb check data=/path/to/ipb-v1/uzh_fpv full=true hash=true
```

数据许可证为 CC-BY-NC-SA-3.0。源码发行许可证不改变数据的非商业和署名/相同方式共享条款。
