# M2EIT staging draft

Status: unregistered, no server runtime proof, no benchmark result. This contributes **zero** accepted methods. Two dimension profiles represent one independent paper method.

Primary source: [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_M2EIT_Multi-Domain_Mixture_of_Experts_for_Robust_Neural_Inertial_Tracking_ICCV_2025_paper.pdf) and [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Li_M2EIT_Multi-Domain_Mixture_ICCV_2025_supplemental.pdf). Authors: Yan Li, Yang Xu, Changhao Chen, Zhongchen Shi, Wei Chen, Liang Xie, Hongbo Chen, Erwei Yin. No author matches the excluded Shanshan Zhang. Publisher and author publication page were checked; no official M2EIT network repository was verified. This is a mathematical reconstruction, not copied network source.

The paper specifies a 200-sample six-axis navigation-frame window and planar velocity regression. Its distinct components are six spatial BasicBlocks, bidirectional selective-state temporal modeling, a Haar frequency expert, two cross-domain attention translations, and a learned three-expert router. The supplement fixes two temporal/frequency layers, four attention heads, convolution width two and SSM state size 16. The training objective is squared vector error. Reported training uses Adam, batch 128, learning rate 1e-4, at most 300 epochs and patience ten; those are historical paper settings, not a frozen common benchmark recipe.

The draft accepts IPB `[gyro, acc]`, internally reorders channels, and returns raw `(B,D)` velocity in m/s with no output scaling. XYZ learns the third output; XY metrics must slice the same XYZ prediction. The original planar profile is retained. View orientation is reference-based and therefore a separate privileged-orientation track, not the paper's estimated-orientation selector reproduction.

Explicit reconstruction assumptions:

- Six residual blocks use widths 64/128/256, two blocks per stage; stride and projection details were not fully specified.
- Temporal and frequency experts use the independent 256-channel embedding. The paper overloads `S` in equations 6/9/12; the spatial output has a different temporal length.
- Haar residual bands concatenate along time to restore length. The concat axis and residual shape reconciliation are missing in equation 11. Standard approximation=sum/detail=difference is used; the printed low/high filter labels are reversed. This choice requires scientific review before faithful-method admission.
- Mamba expansion two and dt initialization follow the [original Mamba module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py). An independent explicit selective recurrence is used; no compiled kernel equivalence or throughput claim is made. AMP precision parity remains pending.
- Regressor hidden width 128, convolution padding and batch normalization follow stated draft choices. No parameter-count match with the authors is claimed.

Admission needs server scalar scan value/gradient tests, all-expert/router and real-Z gradients, original 200-window/default-width forward/loss/backward, raw target/XY checks, standard Trainer/validation/predict/checkpoint reload, CPU/GPU capacity and review of the Haar/embedding ambiguities. Do not register merely because reduced-width unit tests pass.
