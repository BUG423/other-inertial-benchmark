# Benchmark 工具 / Benchmark Tools

[中文](#中文) | [English](#english)

---

## 中文

### Inertial Positioning Lab

[Inertial Positioning Lab](https://github.com/BUG423/inertial-lab) 是面向本基准的独立 Android 端数据采集与真机评测工程（代码仓库已解耦并独立维护于 [BUG423/inertial-lab](https://github.com/BUG423/inertial-lab)）。它将移动设备上的数据入口、canonical 数据契约和端侧模型验证连接为同一条可复现流程：

```text
手机 IMU + ARCore Android sensor pose
    -> .iplab（无损传输归档）
    -> convert_dataset.py（由 inertial-lab 提供）
    -> canonical HDF5 v0.1
    -> CanonicalSequence / WindowDataset
```

主要能力与特性：

- 以 200 Hz 为规范目标采集加速度计和陀螺仪，硬件不足时明确记录实际采样率；
- 可选记录 ARCore VIO 局部位置、姿态、差分速度和逐样本有效掩码；
- 批量导入 `.iplab` 数据包与 `.iplmodel` LiteRT/TFLite 模型；
- 在 Android 真机上执行实时联测或离线“模型 × 序列”批量测试；
- 输出精度、延迟分位数（p50/p90/p99）、吞吐、CPU、内存、热状态与可用能量计数据。

### 数据边界与契约规范

ARCore 是融合相机与设备 IMU 的 VIO 参考源，并非独立的 Vicon/RTK 真值。采集界面会对显示轨迹执行短时平滑、静止锁定、跳点拒绝和跟踪恢复连续化；这些处理不会写回 `.iplab`。归档内保存的是原始 ARCore 位姿、来源字段和有效掩码，便于审计和重新处理。

相机只由 ARCore 内部用于 VIO，工具不展示或保存相机图像。中国大陆版 Android 设备可能需要从应用商店安装并启用 Google Play Services for AR（`com.google.ar.core`）。

### 转换为 Benchmark Canonical 格式

通过独立仓库 [inertial-lab](https://github.com/BUG423/inertial-lab) 中的转换工具可将 `.iplab` 转换为 Benchmark canonical HDF5：

```bash
# 在 inertial-lab 仓库中
python -m pip install -r requirements-dev.txt
python tools/convert_dataset.py recording.iplab -o datasets/android
```

转换结果位于 `datasets/android/sequences/`，可以直接由本基准项目的 `CanonicalSequence.from_hdf5()` 加载。转换流程会校验归档路径、文件大小、清单、有限数值、时间单调性、有效掩码和四元数范数。

### 源码与构建

Android 端工程源码、构建脚本与 Release 安装包统一移至独立仓库：
- 仓库地址：[https://github.com/BUG423/inertial-lab](https://github.com/BUG423/inertial-lab)
- 环境要求：JDK 17、Android SDK 36、Gradle 8.13

---

## English

### Inertial Positioning Lab

[Inertial Positioning Lab](https://github.com/BUG423/inertial-lab) is the dedicated Android data capture and on-device evaluation tool for this benchmark, independently maintained at [BUG423/inertial-lab](https://github.com/BUG423/inertial-lab). It connects mobile sensor collection, canonical benchmark data contracts, and edge model validation in a single reproducible workflow:

```text
Phone IMU + ARCore Android sensor pose
    -> .iplab lossless transport archive
    -> convert_dataset.py (provided by inertial-lab)
    -> canonical HDF5 v0.1
    -> CanonicalSequence / WindowDataset
```

Core capabilities:
- Targets 200 Hz IMU acquisition, recording actual sampling rates if hardware is constrained;
- Records optional ARCore VIO local position, orientation, differential velocity, and sample validity masks;
- Imports `.iplab` datasets and `.iplmodel` LiteRT/TFLite model packages;
- Runs live on-device testing or offline "model × sequence" batch benchmarking on Android;
- Measures accuracy, latency quantiles (p50/p90/p99), throughput, CPU, memory, thermal state, and power counters.

### Data Boundaries and Contract

ARCore is a vision-and-IMU fused reference, distinct from external motion capture (Vicon/RTK). Stabilization techniques applied on the mobile screen do not alter raw archived samples in `.iplab`. Archived records retain raw ARCore sensor poses, origin flags, and validity masks for traceable evaluation.

The camera is consumed exclusively by ARCore internally; raw camera frames are neither rendered nor stored.

### Converting to Benchmark Canonical Format

Using the conversion utility in the [inertial-lab](https://github.com/BUG423/inertial-lab) repository:

```bash
# In the inertial-lab repository
python -m pip install -r requirements-dev.txt
python tools/convert_dataset.py recording.iplab -o datasets/android
```

Converted sequences reside in `datasets/android/sequences/` and can be loaded directly with `CanonicalSequence.from_hdf5()`.

### Source and Build

The Android tool source code, workflows, and preview releases are hosted in the standalone repository:
- Repository: [https://github.com/BUG423/inertial-lab](https://github.com/BUG423/inertial-lab)
- Requirements: JDK 17, Android SDK 36, Gradle 8.13
