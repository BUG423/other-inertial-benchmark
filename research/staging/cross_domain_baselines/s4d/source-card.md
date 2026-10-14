# S4D-Lin/ZOH velocity architecture port

Status: isolated, unregistered staging PoC. No formal matrix rows, source
registry, training data or weights are changed. No framework tests have run
locally; source-bound server CPU scientific and native lifecycle acceptance has
passed, as recorded in the [shared receipt handoff](../CPU_HANDOFF.md).

## Source and scope

The source paper is Albert Gu, Karan Goel, Ankit Gupta and Christopher Ré,
[On the Parameterization and Initialization of Diagonal State Space Models](https://papers.neurips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html),
NeurIPS 2022. It studies diagonal state-space parameterization and initialization
for long-sequence image, audio and medical/time-series tasks. This does not
provide a published inertial-odometry recipe or results for six-axis IMU input.

The official source is
[`state-spaces/s4@e757cef57d89e448c413de7325ed5601aceaac13`](https://github.com/state-spaces/s4/tree/e757cef57d89e448c413de7325ed5601aceaac13).
The exact reference is the pedagogical
[`models/s4/s4d.py`](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4d.py)
kernel/layer plus the residual stack in
[`example.py`](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/example.py).
The paper contains multiple initialization and discretization variants; this
port specifically retains the minimal source's **S4D-Lin and ZOH** variant.
It does not implement full S4, a HiPPO low-rank kernel, selective Mamba, or every
S4D variant evaluated in the paper.

Upstream code is Apache-2.0. The exact upstream license is retained as
`LICENSE.upstream`; hashes of the reference files appear in `provenance.json`.
The implementation independently expresses the published equations and the
source architecture. No upstream Python file or pretrained weight is copied
into this staging directory, and runtime dependencies are PyTorch and the
existing repository BaseModel/InputSpec only.

## Keep/replace contract

| Component | Source architecture | Velocity adapter |
| :--- | :--- | :--- |
| Input | Sequential CIFAR/MNIST examples, `(B,T,C)` | Ordered `[gyro_xyz,acc_xyz]`, `(B,6,200)`, 200 Hz, shared gravity-world view |
| Input encoder | Linear channel projection | Same linear projection, `6 → 128` |
| SSM | Complex diagonal conjugate pairs, fixed `B=1`, learnable `A,C,dt`, ZOH | Preserved; state dimension 64 represents 32 complex states and conjugates per feature |
| Initialization | `Re(A)=-0.5`, `Im(A_n)=πn`, log-uniform `dt∈[0.001,0.1]` | Preserved S4D-Lin initialization; learned `dt` is not the physical IMU sample interval |
| Sequence operator | Vandermonde kernel + FFT causal convolution | Preserved in pure PyTorch; `expm1(dt*A)` replaces equivalent `exp(dt*A)-1` for cancellation accuracy |
| Skip/block | Direct `D*u`, GELU, tied feature dropout, 1×1 Conv → GLU | Preserved; `Dropout1d` has the tied `(B,H,1)` dropout semantics |
| Stack | 4 residual layers, post-LayerNorm by default, optional prenorm | Preserved; default `H=128,N=64,layers=4,dropout=.1` follows example CLI defaults |
| Pool/head | Temporal mean and one linear classifier | Temporal mean and one linear D-dimensional window-average-velocity head |
| Target/loss | Class labels and cross entropy in example | XYZ/XY velocity in m/s and the shared MSE |
| Optimizer | Example gives selected SSM parameters lower LR/zero decay | Unified benchmark optimizer; source-specific groups are not silently installed |

The XY YAML is a separate engineering draft. Main XYZ experiments learn a real
Z output; XY metrics take the first two components of the same XYZ prediction.
No Z padding or privileged extra input is permitted. Each independent window
starts from zero state. The SSM layer is causal within the window, but temporal
pooling uses the full window; declare the model `window_noncausal`. A streaming
state interface is not implemented or claimed.

The source-specific optimization policy is a material fidelity boundary. The
official example attaches learning-rate/weight-decay hints to SSM parameters;
this port does not attach inert hints that the shared Trainer would ignore.
Any later source-faithful optimizer-group experiment requires a distinct
recipe and audited shared Trainer support. Architecture fidelity does not
establish reproduction of the source paper's training results.

## Scientific checks and handoff

`test_scientific_contract.py` checks the kernel against an independently written
ZOH impulse recurrence, then compares FFT convolution with full state recurrence,
explicit causal sums and grouped direct convolution. It compares gradients for
the input, D skip, `log_dt`, `log_A_real`, `A_imag` and the complex readout's real
and imaginary storage. It also checks no circular wraparound, causal prefixes,
source initialization, post/prenorm paths, genuine Z-only supervision, finite
default-width T200 backward and an actual optimizer update, deterministic state
reset/reload, rejection of malformed scientific inputs and no active registration.

Run framework acceptance **on AutoDL**, from a separately deployed staging
bundle, without deploying to or editing the immutable formal source:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  research/staging/cross_domain_baselines/s4d/test_scientific_contract.py
```

The root coordinator also owns the shared native-data lifecycle test. It must
exercise default-width T200 input, two actual updates, independent native
train/val/test fixtures, Predictor and checkpoint reload before G2 is claimed.
Passing tensor-based scientific tests proves G1 only. G3 CUDA capacity,
three-seed blind pilot and next-version admission remain pending. The full
Vandermonde tensor is `(H,N/2,T)` per layer; capacity must be measured instead of
inferring full-S4 or long-sequence efficiency from this short-window PoC.
