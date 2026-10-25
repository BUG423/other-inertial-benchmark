# 三个跨域架构的 AutoDL CPU 验收

## 验收结论和边界

2026-10-14 **17:36 CST**，第二轮独立源码验收完成：
**108 passed / 0 failed / 0 skipped / 0 pending**。
其中 ModernTCN / PatchTST / S4D 的科学节点分别为 31 / 38 / 33，
另有三架构各 XY/XYZ 的 6 项标准 NIO 生命周期。

科学检查包含结构特性、默认宽度 forward/loss/backward、真实 Z 梯度、参数更新、
输入权限拒绝；S4D 还独立核对 ZOH 递推、FFT 卷积及梯度；PatchTST 检查编码器 CI
和最终速度头的轴/传感器身份；ModernTCN 检查分组、排列和残差。

每项生命周期使用默认 **T200 和原宽度**，三个不同 RIDI 工程序列分别用于 train/val/test，
4 个训练窗口、2 个验证窗口、batch 2、seed 1043，实际执行两次 optimizer update。
不是随机张量替代的 Trainer smoke，也不是完整数据训练、收敛或精度实验。

已直接断言：

- 每次更新的全部 `requires_grad` 参数均有有限梯度，至少一个参数改变；
- checkpoint 保存真实更新次数、输入维度和 MSE 配置；
- 验证/预测非空、输出有限，XYZ 标签含真实非零垂直速度；
- checkpoint 重载后，验证及预测的速度/目标/位置数组逐元素完全一致；
- XYZ 的 `xy_*` 指标与同一次预测数组的 `[:, :2]` 重新计算结果精确一致，
  计算指标不改变原预测；短片段不能定义的时间/距离指标仍保留 NaN/N/A，而不是伪造数值；
- 三个临时进程内注册项被移除，原 registry 恢复。

## 不可变身份与原始证据

本次测试的是独立 snapshot，不是运行中的正式 source，也不是提交后完整仓库字节的笼统验收。
后续新增文档和状态元数据不改变本表模型及测试文件；任何模型/测试变更均须新收据。

| 项 | 精确身份 |
| :--- | :--- |
| 测试源基线 | `dev/v1@8c1d818` 加隔离 staging，247 个受管文件 |
| source manifest SHA256 | `4c482569ef8b3a90fb22d67f44c1fa18b3aa648af37dd7fc9e47d257aae8b0b0` |
| request SHA256 | `6e6e11de0abcf7e8d5ca58c828807b573fb908bf40d120ae59939222a2f69cb7` |
| bounded runner SHA256 | `e8f17bb46d889f43c8c3b85a7b37e0e54a3256476926019a9d568edaedb23cae` |
| 原始 receipt SHA256 | `e45cae4fe1b53cf6456ed784c6e3aa4bbf28e01cfbf74c300cacb0f44f6e47de` |
| nodes.json SHA256 | `66d5e68931625b5ee223f7173bce34af4b4d089a542fb4d2ecd61f59141fa92d` |
| junit.xml SHA256 | `0f06fa70cebeab68ecd3b47e3135cea83d0bef09009a1d65ce32606b095ba4e0` |
| pytest.log SHA256 | `d2bb2e6fe5f74e03a60600fa2eed2a39cbc350a2d9efba00b883d75444d4c5af` |
| 工程 fixture manifest SHA256 | `3d8a676f45ec7b93a0365eadddee45ec1bad7025deb212a6dcecddc922bd8c53` |
| ModernTCN model.py SHA256 | `6f0b495718703b83d4c224a959ba8e171116eb22b6c342a32607952452489b23` |
| PatchTST model.py SHA256 | `795735ed95b1f949d9c4a7c4d4d23513e1ec0858ffb60afb2773adcf4968a4c4` |
| S4D model.py SHA256 | `c283b341243f9074c969aec21ca8bd9e07c77874d57774203dc14627c72ad470` |
| 共享 lifecycle test SHA256 | `722d6b196135dba1196d6f681ac96b3c9c9265edd24bc7b08a9568116c38082e` |

服务器原始证据根目录：
`<agent-project-root>/.local/reports/cross-domain-cpu-4c482569ef8b-20261014T093620Z/`。
包含 `receipt.json`、`nodes.json`、`junit.xml`、`pytest.log` 和六份独立生命周期 receipt；
前四份已按原字节镜像到本仓库同名 `.local/reports/` 子目录。
源码 snapshot 为
`<agent-project-root>/.local/source-snapshots/cross-domain-4c482569ef8b3a90-20261014T093620Z/repo/`；
对应 request、manifest 和 runner 保存在服务器 `.local/research/cross-domain-cpu/4c482569ef8b-20261014T093620Z/`。
凭据、checkpoint、fixture 数据和私有运行目录不加入 Git。

首轮 snapshot `a30ff66bd6363b32bb204d67f36ac630d347abb4eccc2f1dc12caa8387c53553`
亦为 108 passed。独立复审指出其 XY 声明只检查了字段存在、梯度检查跳过 None；
第二轮补齐这两项，收据身份另立，第一轮保留为历史证据，不覆盖。

## 隔离和资源

- CUDA 不可见且未初始化；Torch / interop / OMP 单线程，nice 10。
- 独占 supplementary CPU lease；不占正式 CPU/GPU 训练 lease。
- runner 上限 420 秒、child RSS 2 GiB；实测 wall 27.11 秒，child 峰值 RSS 959,946,752 bytes。
  这是 CPU 工程运行成本，不能称为 GPU 显存、标准吞吐或模型推理延迟。
- 前后逐 SHA 核对 425 个正式 source 文件、policy、capacity、seed、coverage 和 13 个 native 元数据/split，
  以及工程 fixture；全部不变。测试 snapshot 的全部文件亦不变。
- 收据最终 `protected_unchanged_on_exit=true`、`lease_released=true`。
  正式 matrix increment 为 0。

## 参数量与科学解释

以下总量经服务器原生实例核验；head/backbone 分解经独立源码解析复审。

| XYZ 适配 | Backbone | 线性头 | 总参数 | 当前读出 |
| :--- | ---: | ---: | ---: | :--- |
| ModernTCN | 88,096 | 1,155 | 89,251 | 时间均值 → 有序变量/特征 |
| PatchTST | 402,816 | 57,603 | 460,419 | 有序 `C,D,N` 展平 |
| S4D | 200,576 | 387 | 200,963 | 时间均值 |

对应 XY 总量为 88,866 / 441,218 / 200,834。
“同为线性头”不表示容量或读出算子一致；当前 PatchTST 的头部成本显著更高。
pilot 必须报告该分解，且在归因于 conv/attention/SSM 前进行独立 matched-pooling 消融。
消融建议：只对 PatchTST 增加时间 GAP → 有序特征 → Linear 的新 profile，保持编码器不变；
解析预期为 backbone 402,816 + XYZ head 2,307 = 405,123（尚未实现/运行，不是已验收模型）。
不同骨干宽度仍然存在，匹配读出不能被解释为纯归纳偏置的因果证明。

## 复验入口

仅在 AutoDL 的独立源码目录、通过资源预算和 supplementary lease 后运行。
先核对源码和合法工程 fixture 的哈希，设置 `IPB_CROSS_DOMAIN_FIXTURE`；
fixture 必须有 `engineering_only=true` 和三个独立 split，不能伪造或混入正式数据统计。
推荐使用现有单线程 wrapper 和逐节点证据，示意命令如下；外层仍需限时/限内存及正式 guards：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src IPB_TEST_NODE_REPORT=<fresh-output>/nodes.json \
IPB_CROSS_DOMAIN_FIXTURE=<accepted-engineering-fixture> \
<server-venv>/bin/python tools/ops_v1/cpu_pytest.py \
  -p no:cacheprovider --import-mode=importlib -q \
  research/staging/cross_domain_baselines/moderntcn/test_scientific_contract.py \
  research/staging/cross_domain_baselines/patchtst/test_scientific_contract.py \
  research/staging/cross_domain_baselines/s4d/test_scientific_contract.py \
  research/staging/cross_domain_baselines/test_native_lifecycle.py \
  --basetemp=<fresh-output>/pytest-tmp --junitxml=<fresh-output>/junit.xml
```

G3 容量与 G4 三 seed pilot 的准入、结果协议及 next-version identity 仍须独立建立；
此收据不是 `full_matrix.cpu_proofs()` 自动接纳的按 profile receipt，不能直接改字段后冒充准入。
