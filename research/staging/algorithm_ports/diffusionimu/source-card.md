# DiffusionIMU staging draft

Status: unregistered, no server runtime proof, no benchmark result. This contributes **zero** accepted methods. Two dimension profiles represent one independent paper method.

Primary source: [IJCAI 2025 main-track article](https://www.ijcai.org/proceedings/2025/977), [full paper](https://www.ijcai.org/proceedings/2025/0977.pdf), DOI 10.24963/ijcai.2025/977. Authors: Xiaoqiang Teng, Chenyang Li, Shibiao Xu, Zhihao Hao, Deke Guo, Jingyuan Li, Haisheng Li, Weiliang Meng, Xiaopeng Zhang. No author matches excluded Shanshan Zhang. No author network repository was verified through publisher PDF and bounded searches. No third-party reconstruction is presented as official source.

The method-specific equations specify feature noise mixing, a learned alignment transform, 100 timestep embeddings, iterative linear/ReLU refinement, velocity regression and a positive Softplus auxiliary head. Experiments set latent dimension 128, three refinement steps and dropout 0.2. The loss adds velocity MSE, half the auxiliary head squared norm and alignment squared error. It **does not** define Gaussian NLL or noise-prediction training; background DDPM equations do not replace its method-specific loss. Paper training uses Adam, batch 128 and learning rate 3e-4. Full epoch/early-stop scheduling was not specified.

Draft assumptions and boundaries:

- The encoder's “ResNet-18 bottleneck” attention replacement is not specified completely. Eight local/global attention bottlenecks, four widths 64/128/256/512, radius-three local attention and four heads are a declared reconstruction, not a recovered official architecture.
- A 200-sample window, average-velocity target, final temporal mean pooling and orientation consumed by the IPB gravity-world view are adaptations. Equation 10 lists orientation without defining a separate orientation feature encoding. Original output dimension is ambiguous; XYZ is labeled adapted.
- A sigmoid-constrained scalar mixture initialized at 0.5 is assumed. Equation 21 is interpreted as a batch mean of squared vector norms; alignment uses each sample's unnormalized latent squared norm as equation 13 writes it. Component-mean scaling would change relative task weights and must be a named ablation.
- Training samples timestep per window. Evaluation fixes timestep zero while retaining fresh Gaussian latent noise. This inference policy is not supplied by the paper; record seed, timestep, decoder steps, draws and batching in prediction identity. Reload comparisons replay RNG exactly, not silently set noise to zero.
- Softplus `u` is a positive regularized head with undefined covariance units. Its penalty can drive it towards zero independently of residuals. It is saved as `uncertainty_regularizer_head`, never `logstd`; NLL/NEES/coverage are unavailable until a scientifically different calibrated model is defined.

Input remains IPB gyro-first; internal acceleration-first reordering does not alter raw files. Output is `(B,D)` world-frame velocity in m/s, and Z is learned. Admission requires equation-22/mask tests, all three refinement and real-Z gradients, stochastic inference RNG/reload tests, full default-profile CPU lifecycle and actual capacity. The global model/loss registries are untouched; copying this draft into active code requires explicit review and later server acceptance.
