# ADVIO 2018 来源卡

审计转换入口：[`advio.py`](../../src/inertial_benchmark/data/converters/advio.py)，`advio@1.0.0-audit`。
公开源：[Zenodo 1476931](https://zenodo.org/records/1476931)、
[作者固定提交](https://github.com/AaltoVision/ADVIO/tree/346e5968b49d361b87142f66c5c665cc6a8e8dfe)。
实收官方 API 的 `metadata.license.id=cc-by-nc-4.0`，数据许可为 CC BY-NC 4.0。

2026-10-13 服务器直下载三条完整代表归档，官方 MD5 和完整 ZIP CRC 均通过：

| 序列 | 场景 | 归档字节 | 官方 MD5 |
|---|---|---:|---|
| `advio-15` | 办公室 | 54,845,329 | `f5febcd087acd90531aea98efff71c7c` |
| `advio-07` | 电梯 | 74,009,013 | `ec22ead3bdc9a153f2423b9a60e854ee` |
| `advio-20` | 室外 | 195,538,352 | `2f7a99c24960acf0be73e9ec235e8e24` |
| 合计 | | 324,392,694 | |

全 23 条未下载完成，不把预分配文件大小当成实收。保留官方 API 文件表、逐归档 hash/CRC 与完成状态。
实际 ZIP 成员是 `iphone/gyro.csv`、`iphone/accelerometer.csv`、`ground-truth/pose.csv`、`fixpoints.csv`；
泛用 README 中 gyroscope/poses 的拼法与实物不同。

iPhone 加速度导出值已是 SI；作者 rosbag 脚本没有再乘 g。不要根据 CoreMotion 原生 API 的 g 单位
再给导出数据乘 9.8，也不隐式应用作者近似 bias。陀螺/加速度为独立时钟，应在未来转换时仅在覆盖区和允许缺口内同步。

参考轨迹由 **IMU + 人工 fixpoints** 重建，不是独立 Vicon；`pose.csv` 的 q 为相机/代理姿态，作者可视化按 wxyz 读。
相机→IMU 外参的表述方向与部分 YAML 命名存在待审计差异，lever arm 亦未核实。
当前不能把这个 q 宣布为合法 IMU body→world 姿态，也不能伪造 identity q 为有效标签。

```bash
ipb convert dataset=advio source=/path/to/advio-zip-directory output=/path/to/ipb-audit/advio workers=1
```

入口只审计 ZIP 目录，按现有转换报告明确拒收；三条来源对应 0 个 v1 canonical、无 train/val/test。
未来实装必须先核 IMU/相机 frames、外参和同步，再进入单独的相关参考质量赛道。

2026-10-13 AutoDL 实物验收：`advio-07/15/20` 三个候选均以相机→IMU 外参未核实明确拒收，
0 H5、0 分割。对空输出运行完整数据校验会返回 `no split files under splits/`，
不能计为训练数据验收通过。实际 `conversion_report.json` SHA256 为
`a8c165674dd3c2b627abb7572aaff9ddfc25a784df720c062cd54c50b506a36d`。

