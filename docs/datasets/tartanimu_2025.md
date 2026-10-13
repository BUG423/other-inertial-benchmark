# TartanIMU CVPR2025 公开 samples 来源卡

当前为 audit-source catalogue，**没有注册可产 v1 的 converter**。
来源：[作者 HF 固定版本](https://huggingface.co/datasets/raphael-blanchard/TartanIMU/tree/3577944d7fcac2882d60c03fc2bfc83718a5b60e/data/samples)、
[作者仓库](https://github.com/superxslam/TartanIMU)、
[CVPR2025 正文](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.pdf)。

此前服务器已实收 18 NPZ，149,020,866 B，上游 LFS SHA256 与 NPZ/ZIP CRC 全通过；
car/dog/drone/human 四域，**所有公开 samples 均位于作者 test**。
这是 CVPR2025 发布，不是 IROS2026 隐藏标签竞赛；不从这批样本派生 train/val。

实物 schema 审计：14 文件为标准形状，4 个 human 时间为 `(N,1)`。
两个 car adapt 文件存在 374 种重复联合 IMU/position/orientation 64 行块，长文件另有三个零 dt。
18 个文件不表示 18 个独立完整原始轨迹，必须先恢复原录制 ID 去重。

后续按已有八 converter 的模板接入，至少需要核对：

- 每个 NPZ 导出版本的 timestamp 单位、IMU rad/s 和 m/s²，以及是否已经重采样/retarget。
- 四元数顺序、body→world 方向、z-up 世界系、参考原点和 IMU 外参。
- 有效 mask、原始录制/连续时间范围、重复片段的统一 group_id。
- 姿态属于参考或设备可部署估计，参考系统是否同 IMU；原 test 只作外域测试。
- 权利和许可从数据发布核实，不从算法代码推断。

这些条件未通过前只保留不可变资产及审计证据，不生成看似有效的 v1 pose、不执行训练。
