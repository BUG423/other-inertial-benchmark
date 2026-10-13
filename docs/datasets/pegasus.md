# Pegasus / AirIO 仿真来源卡

审计转换入口：[`pegasus.py`](../../src/inertial_benchmark/data/converters/pegasus.py)，`pegasus@1.0.0-audit`。
沿用已有转换接口，但输出 `RawSequence.rejected`；不创建带伪姿态的 v1 canonical。

来源：[AirIO 作者发布](https://github.com/Air-IO/Air-IO/releases/tag/datasets)。
`PegasusDataset.zip` 实收 62,472,509 B、41 成员完整 CRC 通过，观测 SHA256：
`5cbb0c4e6a3c64caf844f7c2c68fa83c444279af1c6e4457546d7ee8ceafc8cc`。
数据许可未单独核实，不从代码 BSD 许可推断。

四个 TRAIN / 三个 TEST 仿真目录，源 IMU 为 time、acc3、gyro3；作者将 FRD IMU 用 `Rx(π)` 转为 FLU。
GT 为 time、xyzw q4、world velocity3、body velocity3、position3。
**GT 四元数的机体系到底是 FRD 还是 FLU 尚未确认，position 原点/lever arm 也未确认**。
不能只把 IMU 翻轴后就把未知 GT q 当成 FLU→world。恢复真实转换需要：

- 作者仿真导出源码或标定定义确认每个 GT 量的 frame、单位和原点。
- 陀螺/姿态动态一致性、静止重力及位置二阶差分核对。
- 原始完整录制 ID 及不交叉 train/val/test；仿真单独赛道。

```bash
ipb convert dataset=pegasus source=/path/to/raw-archives output=/path/to/ipb-audit/pegasus workers=1
```

入口可直接读取 ZIP 目录，不展开图像或完整 raw。每条候选进入 `conversion_report.json`，明确拒收原因与源归档 SHA；
接收数为 0，没有训练分割。这是审计入口，不是已完成的训练数据集。

2026-10-13 AutoDL 实物验收发现 7 个候选，全部以以上 frame/lever arm 原因拒收：0 H5、0 分割。
对空输出运行完整数据校验会返回 `no split files under splits/`，这是不能训练的明确结果，
不计为训练数据验收通过。实际 `conversion_report.json` SHA256 为
`c05aef79512f4b1c1108ab54a11b938783c281d30844b4708ebf4eddbdc2ec79`。

