# SubT-MRS（AirIMU CSV 版本）

独立数据来源为 SubT-MRS；这里接入的是 AirIMU 作者分段、调整后发布的 UGV CSV 版本，不是原始多模态全集。TartanIMU 等再次打包的集合若引用同一来源，不增加独立数据集计数。

## 来源与范围

- [SubT-MRS 官方主页](https://superodometry.com/datasets)
- [数据集论文](https://arxiv.org/abs/2307.07607)
- [AirIMU 版本说明](https://github.com/haleqiu/AirIMU/blob/main/SubT-MRS-DATASET.md)
- [作者原始读取代码](https://github.com/haleqiu/AirIMU/blob/main/datasets/SubTdataset.py)
- [作者下载包](https://github.com/sleepycan/AirIMU/releases/download/subt_dataset/SubT-MRS-Dataset.zip)

本次下载源完整包为 261,816,716 字节，服务器 SHA256 为
`1219531c248fbe8797106ba00ce4e4937c3527ebfead503d56bfeee4177de441`。
已接入 Final/Urban Challenge 的五个 event–robot 录制组，实际 28 条接收、0 条拒收，
合计 10,326.680 s、2,774.425 m。6 个科学测试、全数组/哈希检查和 28×真实 B2
XYZ／XY 任务视图通过；这是作者派生版本，不是原始多模态全集。

已在服务器完成 1.0.1 的元数据升级，安装位置统一为 `body`，另存
`source_placement=robot_body`、`source_platform=ugv`。28 条序列的全部数值数组、掩码
逐项 SHA256 相同，划分文件也完全相同，未重新采样；原 1.0.0 目录保留作审计。
升级后的 6 个科学测试、全数组/哈希检查和 28×真实 B2 XYZ／XY 任务视图通过，
0 errors，仅保留缺少 train/val 的 2 条 warnings。规范化版本变化更新了 manifest
fingerprint，原其他十个数据源的 manifest 与划分未变。

## 规范化合同

转换器保留整数纳秒时钟，先以整数减去同一 IMU 起点，再换算秒，避免绝对 Unix 时间的浮点损失。加速度保留比力（含重力，m/s²），角速度为 rad/s，位置为米，参考姿态从 xyzw 转为 canonical wxyz/body_to_world；统一流水线负责重采样。

参考轨迹来自高精地图约束的视觉、激光与 IMU 融合，包含输入 IMU 信息，并非独立运动捕捉真值。AirIMU 的 `ground_truth_imu.csv` 给出 IMU 原点的参考位姿和偏置；转换器不使用参考偏置校准输入。参考姿态属于 privileged 模式。原 IMU CSV 的姿态来源未明确，不直接宣称可部署的 device orientation。

转换必须通过重力方向、陀螺相对姿态、时间重叠和单位检查；未知或失败合同明确拒收，不靠调换四元数方向或放宽阈值制造可用数据。原速度字段保存，但统一窗口目标仍遵循 benchmark 的位置差分定义。

## 划分与评估

当前未核实作者训练/测试列表，初始版本明确为 benchmark 派生的 test-only 目录。一个 event–robot 的所有分段共享同一 recording group，不能把分段当成独立录制随机拆开。后续训练协议需要完整组别、组间重叠审计和公开说明；本版不伪造 train/val。

## 使用

```bash
ipb convert --converter subt_mrs --source /path/to/SubT-MRS --output /path/to/subt_mrs
ipb check /path/to/subt_mrs --full --hash
```

自动生成的 `test_ood_*` 辅助列表与主 test 指向相同的 28 条序列，不增加条数；
缺少训练集合时，它们不构成有训练域对照的泛化实验证据。
本版没有 train/val，不能把转换验收完成解释为本域训练就绪。
数据许可证需同时核实原始来源与派生发行条款，源码许可证不自动覆盖数据再发布。
