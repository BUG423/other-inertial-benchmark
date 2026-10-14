"""Unregistered MixerCSeg TransMixer temporal-core architecture port.

Primary source: spiderforest/MixerCSeg@7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d,
VMamba/models/{vmamba,utils,csms6s}.py (MIT root; see attribution in provenance).
This is NOT the full crack segmentation model: HOG/DEG, SRF, image hierarchy,
and decoder are omitted. Two 1D scan directions and Conv/Pool1d replace 2D
operators; width/depth are unmeasured draft hypotheses. Real per-token dynamic
delta-ranked channel routing, global attention, and local refinement are kept.
Pure PyTorch mathematical recurrence, no CUDA/vendor kernels or registry writes.
Only server-after-tail scientific tests are authorized; no runtime acceptance.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.data.views import CHANNELS
from inertial_benchmark.nn.base import BaseModel, InputSpec


def _canonical_spec(spec: InputSpec) -> None:
    if (tuple(spec.channels) != CHANNELS or spec.window != 200 or spec.rate != 200
            or spec.dims != 3 or spec.frame != "gravity_world"
            or spec.orientation != "reference" or spec.remove_gravity
            or spec.target != "avg_velocity" or spec.output_layout != "window"
            or spec.output_steps or spec.overlap != "mean" or spec.history or spec.history_stride
            or spec.extra_inputs):
        raise ValueError("requires canonical ordered six-axis T200@200Hz reference "
                         "gravity_world retained-gravity XYZ avg_velocity, no extras/history")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dropout(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("dropout must be finite in [0, 1)")


def positive_delta(delta: torch.Tensor, delta_bias: torch.Tensor | None) -> torch.Tensor:
    """Source softplus(input-conditioned delta + learned v0 bias)."""
    return F.softplus(delta if delta_bias is None else delta + delta_bias[None, :, None])


def get_global_local_index(delta: torch.Tensor, delta_bias: torch.Tensor | None,
                           ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank CHANNELS at each sample/token, never time tokens or static halves.

    Equal deltas retain the source torch.sort default tie semantics; no scientific
    tie-order invariance claim. Hard selection is discrete, as in the source.
    Returned indices are B,T,G and B,T,L, including the descending rank order.
    """
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("ratio must be finite in (0, 1)")
    channels = delta.shape[1]
    global_channels = int(channels * ratio)
    if global_channels < 1 or global_channels >= channels:
        raise ValueError("routing needs two nonempty channel branches")
    indices = positive_delta(delta, delta_bias).sort(dim=1, descending=True).indices
    global_indices, local_indices = indices.split((global_channels, channels-global_channels), dim=1)
    return global_indices.transpose(1, 2), local_indices.transpose(1, 2)


def selective_scan_reference(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                             B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                             delta_bias: torch.Tensor) -> torch.Tensor:
    """Source selective_scan_torch equations with per-call zero initial state.

    u/delta: B,K*channels,T; B/C: B,K,state,T; A: K*channels,state.
    h_t = exp(delta_t*A)*h_(t-1) + delta_t*B_t*u_t;
    y_t = sum(C_t*h_t) + D*u_t. NOT exact-ZOH S4D or an LTI convolution.
    Normal float32 math follows source; float64 is retained for verification.
    Lower-precision input uses float32 recurrence then returns input dtype.
    """
    batch, groups, state_dim, steps = B.shape
    mixed_channels = u.shape[1]
    if (mixed_channels % groups or u.shape != (batch, mixed_channels, steps)
            or delta.shape != u.shape or C.shape != B.shape
            or A.shape != (mixed_channels, state_dim)
            or D.shape != (mixed_channels,) or delta_bias.shape != (mixed_channels,)):
        raise ValueError("inconsistent grouped selective recurrence shapes")
    work_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    source_dtype = u.dtype
    u, delta, A, B, C, D, delta_bias = (tensor.to(work_dtype)
                                      for tensor in (u, delta, A, B, C, D, delta_bias))
    dt = positive_delta(delta, delta_bias)
    channels = mixed_channels // groups
    # Broadcast group-specific B/C, preserving shared coefficients across channels.
    grouped_u = u.reshape(batch, groups, channels, steps)
    grouped_dt = dt.reshape(batch, groups, channels, steps)
    grouped_A = A.reshape(groups, channels, state_dim)
    state = u.new_zeros(batch, groups, channels, state_dim)
    outputs = []
    for step in range(steps):
        dt_step = grouped_dt[..., step, None]
        state = (torch.exp(dt_step * grouped_A[None]) * state
                 + dt_step * B[:, :, None, :, step] * grouped_u[..., step, None])
        outputs.append((state * C[:, :, None, :, step]).sum(dim=-1))
    y = torch.stack(outputs, dim=-1).reshape(batch, mixed_channels, steps)
    return (y + D[None, :, None] * u).to(source_dtype)


class GlobalAttention(nn.Module):
    """Source GroupNorm/QKV branch; explicit standard dropout respects eval.

    Source fused path passes nonzero dropout_p in eval. We explicitly correct
    this semantics with nn.Dropout; at draft dropout=0 there is no difference.
    This branch remains quadratic in window tokens, not a linear-time claim.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim < 8 or dim % 8 or dim % num_heads:
            raise ValueError("global width must be divisible by 8 and num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.in_norm = nn.GroupNorm(dim // 8, dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        qkv = self.qkv(x).reshape(batch, steps, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        y = (self.attn_drop(attention) @ v).transpose(1, 2).reshape(batch, steps, dim)
        return self.proj_drop(self.proj(y))


class LocalRefiner1D(nn.Module):
    """Source local GroupNorm/maxpool/sigmoid gate/projection, 2D -> 1D."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 8 or dim % 8:
            raise ValueError("local width must be positive and divisible by 8")
        self.in_norm = nn.GroupNorm(dim // 8, dim)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(dim, 1, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x.transpose(1, 2))
        gate = torch.sigmoid(self.conv1(self.maxpool(x)))
        return self.conv2(gate * x).transpose(1, 2)


def route_channels(y: torch.Tensor, delta: torch.Tensor, delta_bias: torch.Tensor,
                   ratio: float, global_module: nn.Module, local_module: nn.Module) -> torch.Tensor:
    """Source gather/branch/scatter, functional instead of in-place for autograd."""
    global_indices, local_indices = get_global_local_index(delta, delta_bias, ratio)
    tokens = y.transpose(1, 2)
    global_y = global_module(tokens.gather(2, global_indices))
    local_y = local_module(tokens.gather(2, local_indices))
    indices = torch.cat((global_indices, local_indices), dim=2)
    values = torch.cat((global_y, local_y), dim=2)
    return torch.zeros_like(tokens).scatter(2, indices, values).transpose(1, 2)


class DynamicMixer1D(nn.Module):
    """SS2Dv2 mixer-enabled v0 core adapted to two independent 1D directions."""

    def __init__(self, d_model: int, d_state: int, expansion: int = 2,
                 num_heads: int = 8, global_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("d_state", d_state),
                            ("expansion", expansion), ("num_heads", num_heads)):
            _positive_int(name, value)
        _dropout(dropout)
        if isinstance(global_ratio, bool) or not math.isfinite(global_ratio) or not 0 < global_ratio < 1:
            raise ValueError("global_ratio must be finite in (0, 1)")
        self.k_group = 2  # source four 2D traversals -> explicit forward/backward 1D
        self.d_inner = d_model * expansion
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.global_ratio = global_ratio
        mixed_channels = self.k_group * self.d_inner
        global_dim = int(mixed_channels * global_ratio)
        local_dim = mixed_channels - global_dim
        if (global_dim < 8 or local_dim < 8 or global_dim % 8 or local_dim % 8
                or global_dim % num_heads):
            raise ValueError("routed widths must be divisible by 8; global also by num_heads")
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=True)
        projections = [nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
                       for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in projections]))
        # Source v0 dt initialization: uniform weights and inverse-softplus
        # log-uniform biases, softplus(bias) in [0.001, 0.1].
        dt_projections = [nn.Linear(self.dt_rank, self.d_inner, bias=True)
                          for _ in range(self.k_group)]
        for proj in dt_projections:
            nn.init.uniform_(proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                           + math.log(0.001)).clamp(min=1e-4)
            with torch.no_grad():
                proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_projs_weight = nn.Parameter(torch.stack([proj.weight for proj in dt_projections]))
        self.dt_projs_bias = nn.Parameter(torch.stack([proj.bias for proj in dt_projections]))
        self.A_logs = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32)
                                    .log().repeat(mixed_channels, 1))
        self.Ds = nn.Parameter(torch.ones(mixed_channels))
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True
        self.global_module = GlobalAttention(global_dim, num_heads, dropout)
        self.local_module = LocalRefiner1D(local_dim)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        z = F.silu(z)
        x = F.silu(self.conv1d(x.transpose(1, 2)))
        scans = torch.stack((x, x.flip(-1)), dim=1)
        coefficients = torch.einsum("bkdt,knd->bknt", scans, self.x_proj_weight)
        dt_low_rank, B, C = coefficients.split((self.dt_rank, self.d_state, self.d_state), dim=2)
        delta = torch.einsum("bkrt,kdr->bkdt", dt_low_rank, self.dt_projs_weight)
        u = scans.reshape(batch, self.k_group * self.d_inner, steps)
        delta = delta.reshape_as(u)
        bias = self.dt_projs_bias.reshape(-1)
        A = -self.A_logs.exp()
        y = selective_scan_reference(u, delta, A, B, C, self.Ds, bias)
        y = route_channels(y, delta, bias, self.global_ratio,
                           self.global_module, self.local_module)
        y = y.reshape(batch, self.k_group, self.d_inner, steps)
        # Return reverse traversal to original time order only AFTER dynamic routing.
        y = (y[:, 0] + y[:, 1].flip(-1)).transpose(1, 2)
        return self.dropout(self.out_proj(self.out_norm(y) * z))


class TransMixerBlock1D(nn.Module):
    """Source pre-LN TransMixer residual with GELU MLP, no DropPath/checkpoint."""

    def __init__(self, d_model: int, d_state: int, expansion: int, num_heads: int,
                 global_ratio: float, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.op = DynamicMixer1D(d_model, d_state, expansion, num_heads, global_ratio, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.op(self.norm(x))
        return x + self.mlp(self.norm2(x))


class TransMixerVelocity(BaseModel):
    """Narrow source-qualified TransMixer1D, not full MixerCSeg; real XYZ head."""

    default_loss = "mse"
    training_initialization = "scratch_supervised"
    implementation_kind = "architecture_core_port"
    source_architecture = "mixercseg_transmixer_delta_ranked_two_branch_bidi1d"
    inference_contract = "window_noncausal"

    def __init__(self, input_spec: InputSpec, d_model: int = 32, n_layers: int = 3,
                 d_state: int = 16, expansion: int = 2, num_heads: int = 8,
                 global_ratio: float = 0.5, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, pretrained: bool = False) -> None:
        _canonical_spec(input_spec)
        if pretrained is not False:
            raise ValueError("only pretrained=False (supervised from scratch) is permitted")
        for name, value in (("d_model", d_model), ("n_layers", n_layers),
                            ("d_state", d_state), ("expansion", expansion),
                            ("num_heads", num_heads)):
            _positive_int(name, value)
        _dropout(dropout)
        if (isinstance(mlp_ratio, bool) or not math.isfinite(mlp_ratio)
                or mlp_ratio <= 0 or int(d_model * mlp_ratio) < 1):
            raise ValueError("mlp_ratio must give a positive finite hidden width")
        super().__init__(input_spec)
        self.model_cfg = dict(d_model=d_model, n_layers=n_layers, d_state=d_state,
                              expansion=expansion, num_heads=num_heads, global_ratio=global_ratio,
                              mlp_ratio=mlp_ratio, dropout=dropout, pretrained=False)
        self.token_projection = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList(TransMixerBlock1D(d_model, d_state, expansion, num_heads,
                                                     global_ratio, mlp_ratio, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.velocity_head = nn.Linear(d_model, 3)

    def set_loss(self, name: str, **kwargs: object) -> None:
        if name != "mse" or kwargs:
            raise ValueError("this staged scientific profile permits plain MSE only")
        super().set_loss(name)

    def encode_window(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        x = self.token_projection(imu.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.velocity_head(self.encode_window(imu))}
