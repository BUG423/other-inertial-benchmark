# IPB 评测指标 / Metrics

> 状态：v1 开发基线。实现见 `src/inertial_benchmark/metrics/`，解析解测试见 `tests/test_metrics.py`。
> 本文与 [DESIGN.md](DESIGN.md) 第 5–6 节配套；两者冲突时先修正文档再改代码。

## 1. 约定

| 符号 | 含义 |
|---|---|
| `t_i`, `i = 0…N−1` | 序列统一时间轴，`t_i = i / r`，`r = 200 Hz` |
| `p_i` | 参考位置（`pose/position`），世界系 |
| `p̂_i` | 预测轨迹（DESIGN 第 5 节重建：窗口速度 → 首尾常值外推 → 梯形积分 → 锚定首个有效参考位置） |
| `p̃_i` | oracle 轨迹：把窗口**目标**当作预测、用同一流程积分得到 |
| `m_i` | 参考位姿有效掩码 `valid/pose`（轨迹指标只在 `m_i = 1` 处取样） |
| `v̂_k`, `v_k` | 第 `k` 个窗口的预测 / 目标速度，已换算到世界系 |
| `w_k` | 窗口有效掩码：窗口内全部样本 `valid/imu ∧ valid/pose` |
| `D` | 计算维度：统一三维主协议 `metric_dims = 3`（XYZ）；二维参考协议 `metric_dims = 2`（XY） |

- 所有指标**逐序列**计算，再在序列间聚合（第 5 节）。
- 无法定义时返回 `NaN`（例如没有有效样本），聚合时剔除并报告有效序列数 `n`。
- 距离单位米，速度 m/s，角度度（°）。

### 1.1 三维主结果与 XY 辅助结果

请求三维评价时，预测与参考位置必须提供 XYZ；非空的窗口速度和 oracle 轨迹也必须具备对应三维。轴数不足直接报错，不把请求降成二维，也不把该录制标记为 `skipped`。只有 XY 输出的方法应使用独立的二维参考协议；不补零或复制参考 Z 来进入三维主榜。来源没有三维参考时同样不能据此声明三维评价能力。

`compute_metrics(dims=3)` 保留原指标键作为 XYZ 主结果，并从同一份预测、参考、oracle 和速度的 `[:, :2]` 自动计算 `xy_` 辅助键，例如 `xy_ate`、`xy_rte`、`xy_plr`、`xy_vel_rmse`。不另训模型、不另推理、不重新积分；基础时间轴、有效掩码和指标参数相同，XY 的运动阈值筛选、参考长度和按距离的窗口仍按 XY 定义计算。训练损失没有伪造的 `xy_loss`。窗口速度为空时，相应速度指标保持 `NaN`。

成功评价后，NPZ 的 `meta`、`sequences.csv` 和 run 的 `metrics.json` 保存 `metric_dims_requested`、`metric_dims_effective` 以及 `predicted_axes`。两个维度必须一致；JSON/NPZ 的预测轴为数组，CSV 为 `x,y` 或 `x,y,z`。`distance_m` 和长度比图的参考长度轴使用实际评价维度；未评价和旧结果保持历史水平长度行为。原数据集 manifest 中的历史水平距离统计不变。

旧 NPZ 仍可加载，缺少维度证据时保持未知。历史二维报告兼容；历史三维报告可能经过旧版静默降维，必须重新评价并提供维度证据后再汇总。新的报告同时核对 run JSON 与每条未跳过的 CSV 记录，拒绝缺失或矛盾的维度证据。维度信息属于元数据，不参与指标均值或置信区间计算。

## 2. 轨迹级指标

### 2.1 ATE（绝对轨迹误差）

```
ATE = sqrt( mean_{i: m_i} ‖p̂_i − p_i‖² )
```

不做任何对齐（起点已锚定）。`ATE_aligned`（键 `ate_aligned`）先对 `p̂` 做最小二乘刚体对齐：
2D 时为 SE(2)（绕 z 旋转 + 平移），3D 时为 4 自由度（绕 z 旋转 + 三维平移），旋转角闭式解
`θ = atan2(Σ (a_x b_y − a_y b_x), Σ a·b)`，`a`、`b` 为去均值后的预测与参考水平坐标。有效样本少于 2 个时为 `NaN`。

> 与 RoNIN 官方代码的关系：官方 `compute_absolute_trajectory_error` 对 `N×2` 个**坐标分量**求均值，
> 即 `ATE_ronin_code = ATE / √2`（2D）。IPB 采用逐点欧氏距离（TUM 定义），与论文表格对比时需注意该因子。

### 2.2 RTE（RoNIN 定义）与 T-RTE@τ

```
Δ = round(τ · r)
RTE(τ) = sqrt( mean_{i: m_i ∧ m_{i+Δ}} ‖(p̂_{i+Δ} − p̂_i) − (p_{i+Δ} − p_i)‖² )
```

- `RTE`（键 `rte`）取 `τ = rte_delta = 60 s`，起点为每一个样本（步长 1）。
- **统一规则按有效跨度决定是否换算**（`valid_span = (i1 − i0)/r`，`i0`/`i1` 为首、末有效样本，
  中间的缺口计入跨度）：
  1. 只要存在至少一对两端均有效的 `(i, i+Δ)`，就用上式对**全部**这样的对求 RMSE，`rte_scaled = 0`；
  2. 否则（有效跨度不足 τ，或缺口把序列切成的片段都短于 τ）只取 `(i0, i1)` 这一对并线性换算：
     `RTE = ‖e(i0, i1)‖ · τ / valid_span`，`rte_scaled = 1`；
  3. 有效样本少于 2 个或 `valid_span = 0` 时为 `NaN`。
- 指标输出额外包含 `rte_scaled`（0/1，聚合后即“被换算的序列比例”）、`valid_span_s`（有效跨度，秒）
  与每个 τ 的 `t_rte_<τ>s_scaled`。
- 早期实现按**样本数**（`N − 1 < Δ`）判断短序列，于是 70 s 序列若有效跨度不足 60 s 就得到 `NaN`、
  40 s 序列却按比例放大给出数值：长序列中最难的那些被悄悄剔除，聚合结果存在选择偏差。现按有效跨度统一处理。
- RoNIN 官方代码使用 `(0, N−1)` 这一对并乘以 `12000 / N`，与按实际跨度换算相差 `N/(N−1)` 倍（可忽略）。
- `T-RTE@τ`（键 `t_rte_1s`、`t_rte_10s`，由 `t_rte` 配置）使用同一公式与同一换算规则。

> 与 RoNIN 官方代码的关系（**已核实**，`source/metric.py`）：`compute_relative_trajectory_error`
> 第 43–44 行先算 `err = est[Δ:] + gt[:-Δ] − est[:-Δ] − gt[Δ:]`（与上式的窗口位移差相同），
> 再取 `np.sqrt(np.mean(err ** 2))`——`err` 的形状是 `(N−Δ, 2)`，`np.mean` 同样对 `(N−Δ)×2` 个
> **坐标分量**求均值。因此 RTE 与 ATE 的口径完全一致：
> `RTE_ronin_code = RTE / √2`（2D）。合成轨迹上的数值核对（`N = 30 000`、`r = 200 Hz`、`Δ = 12 000`）：
> ATE 0.374670 / 0.264932、RTE 0.417241 / 0.295034，两个比值都等于 1.414213562。
> 换算路径（`compute_ate_rte` 在 `N < 12000` 时走 `delta = N−1` 并乘 `12000 / N`）同样是分量均值，
> 只是缩放基准用样本数而非跨度，故比值为 `√2 · (N−1)/N`（实测 1.41439，`N = 8 000`）。
> 结论：把 IPB 的 ATE 与 RTE 与 RoNIN 系论文表格对照时，**两个指标都要除以 √2**（3D 为 √3）。
> 同理，任何沿用这段官方 metric 的复现（IMUNet、DeepILS 等，见 `docs/ALGORITHMS.md` §4 第 1 条）
> 报出的 ATE/RTE 都是欧氏定义的 `1/√2`。

### 2.3 D-RTE@d（按距离的相对误差）

1. 参考累计距离 `s_i`：在 1 s 分辨率折线上累计（只累计两端均有效的线段），再线性插值回每个样本；
2. 起点沿参考路径每 `1 m` 取一个：`i_k = min{ i ≥ i_first : s_i ≥ k·1m }`（避免静止段被重复计数）。
   `i_first` 为首个有效样本：`s_i` 在它之前恒为 0，若从样本 0 开始搜索，序列开头无效时 `k = 0`
   的起点会落在无效样本上而被整段丢掉（早期实现的缺陷）；
3. 终点 `j_k = min{ j : s_j ≥ s_{i_k} + d }`；两端都有效才计入；
4. `D-RTE@d = sqrt( mean_k ‖(p̂_{j_k} − p̂_{i_k}) − (p_{j_k} − p_{i_k})‖² )`。

默认 `d = 10 m`（键 `d_rte_10m`，由 `d_rte` 配置）。参考总距离小于 `d` 时为 `NaN`。

### 2.4 PDE（终点漂移）

```
PDE = ‖p̂_e − p_e‖ / L_ref × 100      (%)
```

`e` 为最后一个有效样本；`L_ref` 为 1 s 分辨率的参考路径长度（见 2.5）。`L_ref < 1 m` 时为 `NaN`。

### 2.5 路径长度与 PLR

路径长度 `L_res(x)`：从首个有效样本起每 `res` 秒取一个样本（并补上末个有效样本），对相邻采样点的折线段求和，
**只累计两端均有效的线段**。同一序列的所有长度使用**同一组采样点**（由参考掩码决定）。

| 键 | 定义 |
|---|---|
| `plr` | `L_1s(p̂) / L_1s(p)` |
| `plr_dense` | `L_1s(p̂) / L_dense(p)`，`L_dense` 为 200 Hz 稠密折线长度 |
| `plr_oracle` | `L_1s(p̂) / L_1s(p̃)` |
| `oracle_ratio` | `L_1s(p̃) / L_dense(p)` |
| `ate_oracle` | oracle 轨迹的 ATE（协议本身能达到的下限参考） |

`ate_oracle` 属于默认报表列（`MAIN_METRICS`），必须与 `ate` 一起解读。含多秒 `valid` 缺口的序列上，
缺口内的窗口速度是插值来的，而参考位置在缺口两端可能不连续，于是 oracle 轨迹自身就偏离参考，
`ate_oracle` 可以超过模型的 `ate`（真实数据实例：RIDI `huayi_bag2` 有 4.42 s 的 IMU/位姿缺口，
oracle ATE 5.37 m，而模型 ATE 3.31 m）。这类序列上的 `ate`/`pde` 度量的是缺口而非模型，报告时应单独列出。

恒等式 `plr_dense = oracle_ratio × plr_oracle` 精确成立（测试锁定）。`oracle_ratio < 1` 反映窗口平均速度
与 1 s 采样对路径的平滑程度（协议效应），`plr_oracle` 反映模型效应。参考长度为 0 时对应比值为 `NaN`。

### 2.6 位置误差分布与连续有效段

新增指标直接进入同一份序列结果。XYZ 主结果和 `xy_` 辅助结果使用同一预测的轴截取；不重新积分。
设 `e_i = ‖p̂_i − p_i‖`，只取参考有效样本。预测或参考在有效样本处含非有限值时直接失败，
不能通过丢掉失败预测获得更好的指标；参考无效处可含 `NaN`。

| 键 | 定义 / 单位 |
|---|---|
| `position_mae` | `mean(e_i)`，米；这里是平均欧氏误差，区别于按坐标分量平均的 MAE |
| `position_rmse` | `sqrt(mean(e_i²))`，米；与未对齐 `ate` 同值 |
| `position_p50` / `position_p90` / `position_p95` | 欧氏误差的相应分位数，米；使用 NumPy 默认线性分位插值 |
| `position_max` / `position_std` | 最大欧氏误差、欧氏误差的总体标准差（`ddof=0`），米 |
| `position_{x,y,z}_mae` / `_rmse` | 各轴的绝对误差均值 / RMSE，米；二维结果不生成 Z 键 |
| `endpoint_error` | 最后一个有效样本的绝对位置误差，米；不是首尾相对位移误差 |
| `valid_position_samples` / `valid_segments` | 有效样本数 / 连续有效片段数；片段至少包含一个样本 |
| `valid_duration_s` | 每个片段 `(末索引−首索引)/r` 的和；不计无效缺口 |
| `reference_length_contiguous_m` / `predicted_length_contiguous_m` | 各片段分别每 1 s 采样并补末点的路径长度之和，米；不跨缺口连接 |
| `reference_length_contiguous_dense_m` / `predicted_length_contiguous_dense_m` | 相同片段逐相邻样本的稠密路径长度之和，米 |
| `plr_contiguous` | 连续段的 1 s 预测长度 / 1 s 参考长度 |
| `plr_contiguous_dense` | 连续段的稠密预测长度 / 稠密参考长度；与历史 `plr_dense` 的分子定义不同 |
| `segment_endpoint_error_sum` | 每个至少两样本片段的首尾相对位移误差范数之和，米 |
| `normalized_drift` | 上述相对位移误差之和 / `reference_length_contiguous_m × 100`，百分比；参考长度不足 1 m 时为 `NaN` |

`normalized_drift` 消除每个片段起点的常量偏移。它与历史 `pde` 的终点绝对误差定义不同，不更名替换。
无有效样本的位置统计为 `NaN`，计数及长度为零；静止片段的位置误差仍可算，长度比和归一化漂移为 `NaN`。
`position_std` 仅描述误差分布，不能用于选模：恒定但很大的偏差也可能有很小的标准差。

### 2.7 连续段相对误差与姿态可用性

`rpe_translation_<τ>s_{rmse,mae,p50,p90,p95,max,std,pairs}` 对每个连续有效片段内所有
`(i, i + round(τr))` 计算世界坐标系相对位移误差：

```
e_translation(i, j) = ‖(p̂_j − p̂_i) − (p_j − p_i)‖
```

默认间隔与 `t_rte`、`rte_delta` 相同，即 1、10、60 s。整个 `[i,j]` 都须参考有效，不能跨缺口。
不足该间隔的片段不外推、不缩放；没有配对时统计量为 `NaN`、`pairs = 0`。
这与历史 RTE 仅检查端点、无配对时按跨度换算的公开规则不同，两组键并存。

只提供位置时还可计算 `rpe_point_distance_<τ>s_rmse`，其每对误差为
`abs(‖p̂_j−p̂_i‖ − ‖p_j−p_i‖)`，对应 [evo 的 point-distance 关系](https://github.com/MichaelGrupp/evo/blob/master/evo/core/metrics.py)。
此指标对整段刚体旋转不敏感，应与相对位移向量误差一起阅读。

上述 world-displacement 指标不是完整 SE(3) 的 TUM relative-pose-error；
[evo 的完整 RPE](https://github.com/MichaelGrupp/evo/blob/master/evo/core/metrics.py) 比较相对位姿变换，需要模型预测姿态。
当前位置预测协议没有这项输出，因此 `rpe_rotation_rmse = NaN`、`rpe_rotation_available = 0`。
不能把参考姿态、速度方向或初始方向作为预测姿态来伪造旋转 RPE。计数、可用性、长度及比值均不参加选模。

## 3. 窗口级速度指标

只统计 `w_k = 1` 的窗口，取前 `D` 维。

| 键 | 定义 |
|---|---|
| `vel_rmse` | `sqrt( mean_k ‖v̂_k − v_k‖² )` |
| `speed_bias` | `mean_k (‖v̂_k‖ − ‖v_k‖)` |
| `speed_ratio` | `Σ_k ‖v̂_k‖ / Σ_k ‖v_k‖`（比值的聚合形式，避免低速窗口的除零放大；分母为 0 时 `NaN`） |
| `dir_err_mean` / `dir_err_median` | 夹角 `atan2(‖v̂_k × v_k‖, v̂_k · v_k)` 的均值 / 中位数，只统计 `‖v_k‖ > min_speed`（默认 0.2 m/s）；预测速度为零时记为 90° |
| `along_bias` / `along_rmse` | 误差在目标方向 `u_k = v_k/‖v_k‖` 上的分量 `a_k = (v̂_k − v_k)·u_k` 的均值 / RMSE（同样只统计运动窗口） |
| `cross_rmse` | 垂直分量 `‖(v̂_k − v_k) − a_k u_k‖` 的 RMSE |
| `num_windows` / `num_moving_windows` | 参与统计的窗口数 |

纯旋转误差 `θ` 时：`dir_err = θ`，`along_bias = mean‖v‖(cos θ − 1)`，`cross_rmse = sin θ · rms‖v‖`；
纯缩放 `(1+s)` 时：`speed_ratio = 1+s`，`cross_rmse = 0`（均有测试）。

窗口速度的来源：`avg_velocity` 与 `velocity_at_end` 目标直接是速度；`displacement` 目标除以窗口跨度
`(T−1)/r` 换算；`gravity_yaw_local` 与 `body` 坐标系先用窗口末端姿态旋回世界系。

### 3.1 条件不确定性接口

`metrics.gaussian_uncertainty_metrics(mean, target, logstd, valid)` 提供高斯不确定性的计算接口。
三项输入必须属于**同一原始任务、坐标系、单位及输出行布局**，形状为 `N×D`；
显式各向同性标量头允许 `logstd` 为 `N×1`。只选择有效行，不拟合校准参数、不裁剪模型给出的方差。
缺少 `logstd` 时返回 `uncertainty_task_available = 0` 和 `NaN`，不能假设单位方差。
有头但无有效行时 `available = 1`、`num_rows = 0`，各统计量仍为 `NaN`。
非有限有效预测、不合法正标准差或数值溢出直接报错。

令 `σ = exp(logstd)`、`z = (mean − target)/σ`，采用
[标准高斯密度](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) 定义：

| 键（统一前缀 `uncertainty_task_`） | 定义 |
|---|---|
| `nll` | `mean_n Σ_d [logσ_nd + z_nd²/2 + log(2π)/2]`，nats；包含归一化常数，按轴求和 |
| `nees` | `mean_n Σ_d z_nd²`；对声明的对角高斯模型，期望为 `D` |
| `nees_per_dim` | `nees/D`，理想期望为 1；不能只按越小越好排序 |
| `standardized_rmse` | `sqrt(mean_nd z_nd²)` |
| `sigma_mean` | 标准差的分量均值，单位与原始目标相同 |
| `coverage_1sigma` / `coverage_2sigma` | `mean_nd (abs(z_nd) ≤ 1/2)`，分量覆盖率；标准高斯期望约 0.6827 / 0.9545 |
| `box_coverage_1sigma` / `box_coverage_2sigma` | 每行**所有**分量同时落入相应区间的比例；不是一维覆盖率或联合椭球覆盖率 |
| `coverage_ece` | 中央区间覆盖校准误差：对各名义覆盖率 `c`，计算经验 `abs(z) ≤ Φ⁻¹((1+c)/2)` 的覆盖率，与 `c` 的绝对差，再取平均 |
| `available` / `num_rows` / `dims` / `isotropic` | 头是否存在 / 有效任务行数 / 计算轴数 / 是否显式各向同性标量头 |

覆盖率与 ECE 是 `[0,1]` 的比例。默认名义覆盖率为 0.1、0.2、…、0.9、0.95；
`Φ⁻¹` 是 [标准正态分位函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)。
这是回归中央区间覆盖 ECE，不能当作分类概率 ECE。多个输出步骤的有效掩码应按原始输出行展开。
没有协方差交叉项的方法只声明对角近似，不据此称完整滤波状态一致性或轨迹协方差 NEES。

该接口必须在世界系变换、重叠窗口合并、无效输出填补之前使用匹配的原始均值与标签。
当前 `SequenceResult.logstd` 是原始窗口布局，而 `vel_pred/vel_target` 已变换及合并，直接配对违反约定。
在预测器保存完整原始任务三元组和合同元数据之前，此接口仅为计算组件，不能宣称所有正式结果都已包含这些字段。
从三维原始分布取 XY 时必须同时截取匹配的均值、标签和方差分量；各向同性标量保持不变。

原始速度与位移的 NLL 不能直接混排；仅改变单位 `x→ax` 就会使 NLL 增加 `D·log(a)`。
正式报告需记录目标类型、坐标系、单位、分量数、覆盖水平以及对角/各向同性假设。
本组字段不自动用于主协议选模，训练科学损失与 `loss` 键保持原定义。

## 4. 效率指标

| 键 | 定义 |
|---|---|
| `params` | 全部参数个数 |
| `flops` | batch=1、单窗口 `(1, 6, T)` 前向的乘加次数（MACs）；`flops_backend` 记录计数后端 |
| `latency_ms_cpu` / `latency_ms_cuda` | batch=1 单窗口前向延迟：预热 10 次后 30 次的中位数（CUDA 用事件计时并同步）；CPU 线程数 ≤ 8 |

FLOPs 后端优先 `thop`，其次 `fvcore`，都不可用时使用内置钩子：
Conv `= 输出元素数 × (C_in/groups × Πk) (+ 输出元素数，若有偏置)`，Linear `= 行数 × (in × out + out)`，
BN/LayerNorm `= 2 × 输出元素数`，LSTM/GRU `= 步数 × 方向数 × Σ_层 [门数 × (in·h + h² + 2h) + 逐元素项]`，
MultiheadAttention `= 4·L·E² + L²·E`。不同后端约定略有差异（ResNet18@200 帧：thop 38.25 M，钩子 38.11 M），
比较时请使用同一后端。

## 5. 聚合与统计（`utils/reports.py`）

- **跨序列**：均值、中位数、标准差（`ddof=1`），跳过 `NaN`，同时报告 `n`。
- **分组 bootstrap 95% 置信区间**：按 `group_id` 有放回地重采样组（默认 2000 次，种子 0），
  每次对被抽中组内全部序列的指标求均值，取 2.5% / 97.5% 分位数。只有一个组时区间退化为点。
- **多种子**：先对每个种子求跨序列均值，再报告种子间的 `均值 ± 标准差`（`ddof=1`）。
- **配对 Wilcoxon 符号秩检验**：模型 A、B 在同一数据集同一划分的相同序列上配对，逐序列值先在种子间取均值；
  双侧检验（`scipy.stats.wilcoxon`，`zero_method="wilcox"`），报告 `n`、统计量、p 值与差值中位数。
  配对少于 5 对时不给出 p 值。

- **评测协议一致性**：汇总前核对各 run 的 `metrics.json` → `protocol` 块，评测协议键
  （`eval_stride`、`metric_dims`、`rte_delta`、`t_rte`、`d_rte`、`min_speed`）必须完全一致，
  否则报错（见 [DESIGN.md](DESIGN.md) 第 6 节）。
- **特权输入单列**：使用特权输入（`metrics.json` 的 `privileged_inputs`，例如来自参考真值的
  `init_velocity`）的方法在 `summary.csv` 中带 `privileged` 列、在 `summary.md` 里单独一张表，
  LaTeX 主表只含纯 IMU 方法——两者不可直接比较。

## 6. 诚实协议

训练期间只在 `val` 上计算上述指标并据此选模型（指标名 `fitness`，默认 `ate`，越小越好，配置解析时
即校验；跨序列聚合统计量是另一个键 `fitness_stat`，默认 `mean`，可取 `median`）。`fitness_stat=median`
取逐序列该指标的中位数，供 val 划分小而偏斜的数据集使用（RIDI 自动生成的 val 只有 1 名受试者、
含 2 条 train 中不存在的携带方式，均值型选模信号很噪）；缺省不变，改用时须在结果中注明，
详见 `docs/CLI.md` §4.3。
`test` 及其官方子集只在最终评测（`ipb val split=test` 或 `ipb benchmark`）中运行。
序列级/标定型方法（`pdr`、`mean_speed_heading`）的标定标量只在 `train` 划分上拟合，
结果写入 `metrics.json` 的 `calibration` 块随结果一起发布。

## 7. 可视化范围

逐帧误差时间图和 CDF 严格使用 `metric_dims`，与 `compute_metrics()` 的规则一致；轴标签注明 2D 或 3D。请求 3D 而预测只有 2D 时直接报错，须显式选择独立的二维参考协议。无效参考位姿仍为 NaN，CDF 只使用有限误差样本。轨迹俯视图仍显示 x/y。

多模型报表按每个 dataset/split 的模型名排序，箱线图、CDF 和 PLR 每页至多八个模型，超过八个时输出 `_page01`、`_page02` 等文件；八个以内保留原文件名。页内各图使用相同模型顺序；跨页按模型名识别。Pareto 图始终使用该 dataset/split 的完整模型集合计算前沿，不以分页子集代替全局集合。分页只影响图，CSV、统计和指标定义保持原有协议。
