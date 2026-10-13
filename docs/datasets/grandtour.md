# GrandTour 来源与转换契约

[作者数据主页](https://grand-tour.leggedrobotics.com/dataset)与
[Hugging Face 发布](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset)
提供独立采集的 ANYmal 足式机器人来源。当前已取得的代表任务为
`2024-10-01-11-29-55`，固定数据版本
`daf91c5e2b6db564822a35cc58222d73f8e75bab`，9 个文件共 28,629,696 B。
该代表任务已在迁移后的 AutoDL 通过转换验收：1 条接收、0 条拒收，
333.135 s、197.370 m，`test=1`。这是一个任务子集，不能代表作者全部公开数据。

9 个科学转换测试通过，全部源文件 SHA 与已取得版本相同，canonical 全数组与
哈希校验为 0 errors；两条 warnings 为预期的 train/val 不存在。
真实 B2 窗口通过 XYZ 位置差分和 XY 精确截取校验，缺失设备姿态明确拒绝；
接入前九个数据集的 manifest 与 split 指纹均保持不变。

`grandtour@1.0.0` 直接读取作者 Zarr-v2 TAR 中的数值数组，拒绝路径穿越、链接、
对象数组及未知压缩器。可选依赖为 `pip install '.[grandtour]'` 的 `numcodecs`。
不会展开影像、点云或执行归档中的代码。

ADIS16475-2 原始输入为 200 Hz 比力（m/s²）、角速度（rad/s）与 Unix 秒。
`orien` 全零表示没有设备姿态，因此 canonical 不提供 `device_orientation`。
参考采用 CPT7 Inertial Explorer 紧耦合 GNSS/INS 位姿，是估计器参考；
独立 Leica 棱镜 XYZ 保留在 raw 中，其坐标系和物理原点不能直接代替 IMU 原点。

[作者 ROS TF 约定](https://github.com/leggedrobotics/grand_tour_dataset/blob/main/examples_ros1/grand_tour_ros1/README.md)
把 `T_A_B` 定义为 B 的原点与姿态在 A 中的表达，映射 B 坐标到 A。
[作者 Zarr 数组代码](https://github.com/leggedrobotics/grand_tour_dataset/blob/main/examples_hugging_face/src/grand_tour/zarr_transforms.py)
使用 ROS `xyzw` 四元数。转换器使用同任务的 `box_base→cpt7_imu` 和
`box_base→adis16475_imu` 外参，计算
`T_world_ADIS = T_world_CPT7 · inv(T_box_CPT7) · T_box_ADIS`，同时修正姿态和随机器人旋转的杆臂。
输出为 ADIS 原点的 XYZ、wxyz body→world、ENU 重力对齐 z-up。
作者代码中部分变量名与其 TF 描述易混淆，所以真实数据验收还必须记录
陀螺积分与参考姿态动态一致性，不能只凭静态重力选择外参方向。

每条完整 mission 是一个 `group_id`。当前代表任务以 `test` 目录接入，
没有 train/val，不将同一轨迹切片伪装成独立训练与测试录制。
参考姿态是特权标签；缺少部署姿态的任务视图必须明确报 capability 不支持。
本域训练暂未启用；使用其他来源的同模型 checkpoint 评估时，必须明示训练来源、
输入规格、种子及跨域协议，不能宣称是在 GrandTour 上训练的同域精度。
原始数据、参考来源和已有九个数据集不会因接入该来源而被重写。

```bash
ipb convert dataset=grandtour source=/path/to/grandtour-small-mission output=/path/to/ipb-v1/grandtour workers=1
ipb check data=/path/to/ipb-v1/grandtour full=true hash=true
```
