# DiffusionIMU implementation contract

Independent reconstruction of [DiffusionIMU, IJCAI 2025](https://www.ijcai.org/proceedings/2025/977), based on the [full paper](https://www.ijcai.org/proceedings/2025/0977.pdf), equations 12–22. No official method network source was verified. This package does not replace the method with its introductory DDPM objective or claim original-paper accuracy.

`diffusionimu` is the declared planar reference; `diffusionimu_3d` learns XYZ. Both retain the 200-sample gravity-world average-velocity task, eight local/global attention bottlenecks, 128-dimensional latent features, 100 timestep embeddings, three iterative linear/ReLU refinement steps and dropout 0.2. Native gyro-first channels are reordered internally.

The dedicated `diffusionimu_multitask` objective is the masked sample mean of vector squared velocity error plus half the positive auxiliary-head squared norm plus latent alignment squared error. The auxiliary `u` is saved as `uncertainty_regularizer_head`. It is neither `logstd` nor calibrated covariance; Gaussian NLL, NEES and coverage remain unavailable. Scientific loss replacement requires a separately named ablation.

Declared reconstruction choices: attention topology, radius three and four heads; temporal mean pooling and average-velocity target; reference orientation as privileged view input; sigmoid noise mixture initialized at 0.5; interpretation of unnormalized alignment norms. Original output dimension is ambiguous. Training samples a timestep per window; evaluation fixes timestep zero and retains fresh Gaussian latent noise. Exact replay requires the same RNG, batching, timestep and decoder policy. Export and checkpoint validation must replay the RNG instead of suppressing noise.

Historical mathematical and runtime evidence is retained under `research/staging/algorithm_ports/diffusionimu`. Registration does not add cells to an existing frozen formal matrix. Tensor-format export remains unsupported until a dedicated adapter is reviewed and actually verified; native numeric-state export retains the stochastic prediction contract.
