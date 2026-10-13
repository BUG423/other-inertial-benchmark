# L-IONet

[Chen et al., *Deep Learning based Pedestrian Inertial Navigation: Methods, Dataset and On-Device Inference*, IEEE Internet of Things Journal 7(5), 4431–4441 (2020)](https://arxiv.org/html/2001.04061v1), DOI 10.1109/JIOT.2020.2966773. Authors: Changhao Chen, Peijun Zhao, Chris Xiaoxuan Lu, Wei Wang, Andrew Markham, Niki Trigoni.

This independent implementation follows §IV-C, equations (5)–(6): eight causal dilated WaveNet layers, dilations 1–128, 32 filters, tanh/sigmoid gates, residual and skip connections, pooling and a polar regression head. It is a distinct paper method from the recurrent IONet implementation. No released parameter fixture is available; paper reproduction accuracy has not been established.

`lionet.yaml` preserves a planar polar reference. `lionet_3d.yaml` adds a learned vertical velocity head and vertical loss while preserving the horizontal polar loss. It is an explicitly adapted XYZ method, not an original-paper 3D result. Both use 400 native samples at 200 Hz, averaged in pairs to reproduce the paper's 200 samples over two seconds at 100 Hz. The common gravity-world task replaces the paper's relative step heading with absolute horizontal velocity and heading. Reference orientation is a privileged input and must be reported separately from device-orientation evaluation.

Kernel size 2, pointwise residual/skip projections, mean pooling and the input normalization choice are declared implementation assumptions: the paper does not fully specify them. The paper's original Adam learning rate 1e-5, batch 256 and attachment-specific experiments are source recipes, not claims about the unified benchmark recipe.

Tests cover causal dependence, horizontal polar semantics, real vertical gradients and exact state reload. Source/configuration readiness is separate from server CPU acceptance and formal accuracy results.
