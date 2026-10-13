# 推理导出 / Inference export

`NIO.export()` 与 `ipb export` 使用同一实现。`native` 返回推理包目录；`torchscript` / `onnx`
返回该包中的后端文件。`path` 始终指定一个新的**目录**。现有目录拒绝覆盖。

```bash
ipb export model=runs/train/exp/weights/best.pt format=native path=exports/native
ipb predict model=exports/native source=/data/ipb/ridi/sequences/example.h5 device=cpu
```

推理包包含 `model.pt`、`input-spec.json`、`model-config.json`、`training-phase.json`、
`provenance.json` 与 `manifest.json`。TorchScript/ONNX 包另包含对应的 `model.torchscript` 或
`model.onnx`。原生状态只保存数字 tensor，包括经典方法的已学习标定量、支持向量、系数与索引；
不序列化 sklearn 对象，不复制数据集。包保留实际 InputSpec、输出单位/坐标系、科学损失与阶段、
训练证据及来源哈希。哈希用于检测文件损坏，使用者仍需信任包及其清单的来源。

通过目录、同包 `model.pt` 或后端文件重载，都会核验清单中所有文件。普通历史 `.pt` 保留原加载
语义，不受其他格式的同目录清单影响。导出检查当前 state 能严格重建，包括 dtype；不能完整重载
的状态拒绝发布。仅有 epoch 不证明 sigma 已训练：旧 checkpoint 缺实际优化器更新证据、提前阶段、
损失或 kwargs 覆盖、未经训练记录的权重变动都不能重新认证其 uncertainty。

包不含优化器、调度器和 scaler。`resume=true`、`resume=exports/native/model.pt` 拒绝把推理包当作
续训 checkpoint；真正续训使用训练目录的 `last.pt`。可通过 `pretrained` 将导出权重用于新训练，
新训练的 epoch、优化器与 uncertainty 训练证据重新建立。

## 能力与边界

| 模型/格式 | Native | TorchScript / ONNX |
| --- | --- | --- |
| 可严格重建的神经模型数字状态 | 保留科学输入及原生 Predictor | 仅显式审查的 RoNIN ResNet、IMUNet 原类 |
| PDR、mean-speed-heading、RIDI XY/XYZ | 保留标定与原生序列物理管线 | 明确 unsupported |
| LieEvents、其他数据相关/随机/额外输入模型 | 保留所需 InputSpec 与数字状态 | 未审查时明确 unsupported |

这张表描述实现路径，不代表全部 53 个 profile 的全部格式已有实验验证。Native 的导出时检查证明
状态与科学配置重建一致；**完整轨迹一致**需要针对真实记录运行 Predictor 另行验收。

张量后端仅接受 float32 `(B,C,T)`：`C`、`T` 及通道顺序由原 InputSpec 固定，只有 batch 动态。
目前输出为原模型的 raw `vel`，未做世界坐标转换、滑窗合并、尺度换算和积分；输出单位与 target
必须读取清单。不能因为名字是 `vel` 就把位移任务输出当作 m/s。Exporter 使用 `torch.jit.script`
保留控制流，不允许未经审查的模型用 trace 冻结分支。

每次导出执行三组实际输入和后端，记录 batch、最大绝对差与容差，失败时不发布。
默认 `rtol=1e-5`、`atol=1e-6`，允许范围 `[0,1e-3]`；参数不会改变原模型科学输入。
`batch` 在 `[1,64]`，ONNX `opset` 在 `[13,20]`。不支持的模型或缺少后端依赖明确报错。

## 两种重载方式

**原生科学管线**：即使参数指向张量文件，NIO 仍加载经过验证的同包 `model.pt`，并执行 IPB
原生 Predictor；它并不运行 ONNX/TorchScript runtime。

```python
from inertial_benchmark import NIO

model = NIO("runs/train/exp/weights/best.pt")
native = model.export(format="native", path="exports/native")
restored = NIO(native)
trajectory = restored.predict("/data/ipb/ridi/sequences/example.h5", device="cpu")
```

**独立张量后端**：先通过 IPB task view 按清单生成同一 frame、rate、channel 和 scientific window
的 float32 输入。下列 `imu` 是已经完成这一处理的数组，不是随意截取的原始传感器数据。

```python
import numpy as np
import torch
from inertial_benchmark.cfg.export import verify_export_bundle

verify_export_bundle("exports/script")
backend = torch.jit.load("exports/script/model.torchscript", map_location="cpu").eval()
with torch.no_grad():
    raw_task_output = backend(torch.from_numpy(imu.astype(np.float32, copy=False)))
```

```python
import onnxruntime as ort
from inertial_benchmark.cfg.export import verify_export_bundle

verify_export_bundle("exports/onnx")
options = ort.SessionOptions()
options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
backend = ort.InferenceSession("exports/onnx/model.onnx", sess_options=options,
                              providers=["CPUExecutionProvider"])
raw_task_output = backend.run(["vel"], {"imu": imu.astype(np.float32, copy=False)})[0]
```

Native/TorchScript 需要 PyTorch。ONNX 需要 `onnx` 和 `onnxruntime`，普通安装可使用
`pip install 'other-inertial-benchmark[export]'`。正在训练的服务器应把可选依赖安装到独立
`--target` 目录，保留原 Torch/numpy 环境与源快照。所有本项目 framework 验收只在服务器执行。

## 本批验收状态

2026-10-13 UTC 的新 AutoDL CPU 验收实际收集并通过 **45 项定向测试**，0 failure/error/skip。
覆盖原生 XYZ/XY 轨迹与 RNG、四个真实标定模型状态、损失阶段/证据负例、文件损坏/路径、
推理包续训拒绝、CLI，以及三个已有普通 checkpoint 续训/阶段回归节点。
RoNIN XYZ 工程缩小结构与原 IMUNet XYZ 两个适配器均实际执行 TorchScript 和 ONNX Runtime，
batch 1/2/5 比对通过；这不代表所有 RoNIN backbone 或全部 53 配置的张量格式已测。
ONNX 1.23.2 / ONNX Runtime 1.31.0 安装在服务器私有 target，原训练 Torch/numpy 未改动。

另一个有限 CPU 批次复用已有 **53 个 profile 的真实工程 checkpoint** 与原 RIDI 短 excerpt，
不重训，完成 **53/53 native 导出 → NIO 重载 → Predictor**：数字状态、InputSpec、损失/阶段证据、
实际 raw 输出、速度/目标/位置/辅助数组、指标与下一次 RNG 完全一致；XYZ 的 XY 指标来自相同
预测切片。RIDI XYZ 的真实标定 checkpoint 还通过 CLI export → CLI predict → NPZ 的逐值对照。
该批实际耗时 70.16 s。短 excerpt 的序列化验收不代表充分训练的精度或全数据集性能。

两批使用独立两线程 CPU 预算，Torch 2 / interop 1，ONNX intra 2 / inter 1，未初始化 CUDA。
429 个新快照源文件、425 个正式训练源文件、13 个 native 来源元数据，以及 477 个原 checkpoint /
proof / fixture 文件前后哈希不变；正在运行的正式矩阵保持原身份。

验收绑定源 manifest：`d58706ac0feb287685a5f48eb4184d9855ff12c64e76efaa24c759cb8ce0f484`。
45 项 receipt SHA256：`cb9f9600b795839fb59ff291e2aae113b8d154b9257af43b598e7cc72d3b598f`；
53 native receipt SHA256：`0b02b27e2773776571fcf6bdefbdd1bc074e71627f3150f7e0dda1fa6ee0697b`。
私有操作记录位于 `.local/reports/export-supplemental-next-d587/` 和
`.local/reports/native53-export-d587/`。后续文档更新不改变受测快照的字节身份。
