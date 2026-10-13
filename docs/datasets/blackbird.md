# Blackbird / AirIO 精简发布数据卡

转换器：[`blackbird.py`](../../src/inertial_benchmark/data/converters/blackbird.py)，`blackbird@1.0.0`。
该入口交付原生 IPB v1 转换和测试目录，**未启用 train/val，不称训练 ready**。

来源：[Blackbird 作者数据介绍](https://arxiv.org/abs/1810.01987)、
[AirIO 公开发布](https://github.com/Air-IO/Air-IO/releases/tag/datasets)。
`Blackbird.zip` 实收 16,708,895 B，166 个成员完整 CRC 通过；观测 SHA256 为
`f7ff9cabf5bf86b96053391fec9e3a22fdf07002f09a14ce216f0603e8b05d1b`。
原始数据许可条款另行核实，AirIO BSD 代码许可不能重授权数据。

包内 train5/test5/eval10 共 20 个 IMU 目录，对应 10 个不同的几何 route。
`clover/egg/halfMoon/star/winter` 各出现在 train/test/eval 三分区，共 15 段；
`ampersand/bentDice/oval/sid/sphinx` 各仅出现在 eval，共 5 段。实物时间边界表明这五条跨分区 route 的
片段在录制时钟中相邻；20 个目录和 10 个 route group 均不代表已核实的独立飞行次数。
相对路径为 `partition/geometry/yaw/maxSpeed`，`group_id=route_<geometry>` 只取 geometry，
因此跨分区、yaw 和速度保守同组。当前包每个 geometry 只有一个 maxSpeed，未观察到同路线多速度样本。
当前统一转换只写全部接收序列的 `test` 审计清单，不造训练/验证分割；
这个 `test` 是转换目录，不能宣称全是未见飞行或拿它选模。

| 原始字段 | 实际处理 |
|---|---|
| `imu_data.csv` | 秒时间；列 1–3 为 gyro rad/s，4–6 为比力 m/s²，原轴系保留 |
| IMU CSV header | 写成 acc→gyro，与作者 loader 及实物量级矛盾；按 pinned loader 读 gyro→acc，报告保留原 header |
| `groundTruthPoses.csv` | 整数微秒、NED position、body→NED 四元数 wxyz |
| 世界坐标 | `R_world_NED=diag(1,-1,-1)`，位置和姿态共同变换到 z-up |
| body / IMU 外参 | `q_IMU_world=Rx(π)·q_body_NED·Rz(π/2)`，与作者右乘变换一致 |
| lever arm | 作者取 0；位置仍为 Vicon body 的 IMU 原点 proxy，不称精确标定 |
| thrust / source velocity | thrust 不作输入；不注入作者五点平滑的差分速度，目标由统一 v1 位置视图计算 |

精确证据：[AirIO `BlackBirddataset.py` 固定提交](https://github.com/Air-IO/Air-IO/blob/7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73/datasets/BlackBirddataset.py)，
54–88 行坐标变换、125–126 行 gyro/acc 布局。作者文件 SHA256
`cbd6b7f7a46488b56ba419b8820b37ed4d8fb2f41578037b7c81e5eb66ed7295`，Git blob
`b918359e5502ad6d675a0cd9c1f4993f95b1887e`。转换器独立实现上述格式规则，没有复制作者训练实现。

参考是 Vicon body 转到 IMU 轴系的姿态标签，未提供可部署设备姿态；reference/oracle 赛道需单独标注，
`orientation=device` 拒绝。现有 v1 管线执行 200 Hz 重采样与 mask，无效来源不作为监督。

```bash
ipb convert dataset=blackbird source=/path/to/Blackbird output=/path/to/ipb/blackbird workers=1 overwrite=false
ipb check data=/path/to/ipb/blackbird full=true hash=true
```

2026-10-13 AutoDL 实物验收：20 段全部接收、0 拒收；原生 200 Hz H5 合计 8,367,531 B，
总时长 759.25 s、10 个 route group，只有 `test=20` 的转换审计清单。
完整校验与哈希校验均为 0 errors；五条 warnings 包括预期缺失的
train/val，以及同一 egg route 三段的边界重力诊断（|g| 误差约 −0.46 至 −0.48 m/s²），均保留在报告。
数据指纹为 `8e6bbf05a74e86189907fb91d2dccdbed130c3202d6ecd461ab2ee54eb88bb83`；
`conversion_report.json` SHA256 为
`659ad54598a0727267b9e0ef1805ea1c85c9e07e0e5e12cf71aa8a390a4cbac1`。

