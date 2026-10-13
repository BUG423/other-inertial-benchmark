"""Paper-only sequence-GAN draft, retaining both domains and every objective.

The paper's body-frame, 100 Hz, 200-frame polar-displacement core is separate
from the declared gravity-world average-velocity/XYZ benchmark adaptation.
Unpublished recurrent widths and GAN optimizer scheduling are assumptions.
This module registers neither a model nor a global loss. See source-card.md.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import chain
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from inertial_benchmark.nn.base import BaseModel, InputSpec


PAPER_STEPS = 200
PAPER_RATE_HZ = 100.0
PAPER_LOSS_WEIGHTS = (0.01, 100.0, 0.1, 1.0)


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance per sample, then expectation over samples.

    Reduction over sequence/time is an explicit implementation assumption:
    the paper supplies norm objectives, not a runnable reduction convention.
    """
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("distance operands need identical batched shapes")
    return (a - b).square().flatten(1).sum(1).mean()


def absolute_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("distance operands need identical batched shapes")
    return (a - b).abs().flatten(1).sum(1).mean()


class SequenceMap(nn.Module):
    """Time-aligned LSTM map, with no autoregressive teacher forcing."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int,
                 layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.readout = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.rnn(x)
        return self.readout(sequence)


class WindowMap(nn.Module):
    """A recurrent window predictor/discriminator; last hidden readout is assumed."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int,
                 layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.readout = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.rnn(x)
        return self.readout(hidden[-1])


class MotionTransformerCore(nn.Module):
    """Equations 2--13 with labelled source and unpaired, unlabelled target.

    Input is (B,200,6), ordered [acc_xyz, gyro_xyz], as in the paper's (a,w).
    Domain IDs 0/1 select fixed one-hot domain vectors; their exact encoding is
    unreported in the paper. Recurrent widths/layers are configurable assumptions.
    For paper labels use label_dims=2: (delta_length_m, delta_heading_rad).
    label_dims=3 is exclusively a declared adaptation, never original 3D output.
    """

    def __init__(self, hidden: int = 128, layers: int = 1,
                 label_dims: int = 2, noise_std: float = 1.0) -> None:
        super().__init__()
        if hidden < 1 or layers < 1 or label_dims not in (2, 3):
            raise ValueError("positive recurrent dimensions and label_dims=2/3 required")
        if noise_std != 1.0:
            raise ValueError("Eq.5 uses unit normal latent noise; do not silently remove it")
        self.hidden, self.layers = int(hidden), int(layers)
        self.label_dims = int(label_dims)
        self.encoder = SequenceMap(8, hidden, hidden, layers)
        self.generators = nn.ModuleList([SequenceMap(hidden, hidden, 6, layers)
                                         for _ in range(2)])
        self.decoder = SequenceMap(hidden + 2, hidden, 6, layers)
        self.predictor = WindowMap(hidden, hidden, label_dims, layers)
        self.discriminators = nn.ModuleList([WindowMap(6, hidden, 1, layers)
                                            for _ in range(2)])

    @staticmethod
    def check_sequence(x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (PAPER_STEPS, 6) or x.shape[0] < 1:
            raise ValueError("paper sequence must be nonempty (B,200,6)")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("paper sequence must contain finite floating measurements")

    @staticmethod
    def domain_vector(x: torch.Tensor, domain: int) -> torch.Tensor:
        if domain not in (0, 1):
            raise ValueError("domain must be source=0 or target=1")
        theta = x.new_zeros(x.shape[0], x.shape[1], 2)
        theta[..., domain] = 1.0
        return theta

    def encode(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        self.check_sequence(x)
        return self.encoder(torch.cat([x, self.domain_vector(x, domain)], -1))

    def translate(self, x: torch.Tensor, source_domain: int,
                  target_domain: int) -> torch.Tensor:
        if target_domain not in (0, 1) or source_domain == target_domain:
            raise ValueError("translation requires two distinct source/target domains")
        return self.generators[target_domain](self.encode(x, source_domain))

    def predict(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        return self.predictor(self.encode(x, domain))

    def reconstruct(self, latent: torch.Tensor, domain: int) -> torch.Tensor:
        # The noise is on latent representations, not input measurements (Eq.5).
        noisy = latent + torch.randn_like(latent)
        return self.decoder(torch.cat([noisy, self.domain_vector(latent, domain)], -1))

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        return chain(self.encoder.parameters(), self.generators.parameters(),
                     self.decoder.parameters(), self.predictor.parameters())

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminators.parameters()

    def discriminator_loss(self, source: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """Maximizing Eq.8+9 is minimizing BCE for the two discriminators.

        Generated measurements are detached, so this stage cannot train G/E.
        The unpaired domains may have different numbers of samples.
        """
        self.check_sequence(source)
        self.check_sequence(target)
        with torch.no_grad():
            fake_target = self.translate(source, 0, 1)
            fake_source = self.translate(target, 1, 0)
        loss = source.new_zeros(())
        for domain, real, fake in ((0, source, fake_source), (1, target, fake_target)):
            real_logit, fake_logit = self.discriminators[domain](real), self.discriminators[domain](fake)
            loss = loss + F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
            loss = loss + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        return loss

    def generator_objective(self, source: torch.Tensor, labels: torch.Tensor,
                            target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """All five Eq.7 objectives; there is no target label argument.

        Uses the printed minimax log(1-D(G(x))) objective (negative BCE against
        zero), rather than silently substituting the non-saturating GAN loss.
        During optimizer updates the discriminator parameters must be frozen.
        """
        self.check_sequence(source)
        self.check_sequence(target)
        if labels.shape != (source.shape[0], self.label_dims) or not torch.isfinite(labels).all():
            raise ValueError("source labels must be finite (source_batch,label_dims)")
        zs, zt = self.encode(source, 0), self.encode(target, 1)
        fake_target, fake_source = self.generators[1](zs), self.generators[0](zt)
        encoded_target, encoded_source = self.encode(fake_target, 1), self.encode(fake_source, 0)
        fake_target_logit = self.discriminators[1](fake_target)
        fake_source_logit = self.discriminators[0](fake_source)
        gan = -F.binary_cross_entropy_with_logits(fake_target_logit, torch.zeros_like(fake_target_logit))
        gan = gan - F.binary_cross_entropy_with_logits(fake_source_logit, torch.zeros_like(fake_source_logit))
        autoencoder = (squared_distance(self.reconstruct(zs, 0), source)
                       + squared_distance(self.reconstruct(zt, 1), target))
        prediction = (squared_distance(self.predictor(zs), labels)
                      + squared_distance(self.predictor(encoded_target), labels))
        cycle = (absolute_distance(self.generators[0](encoded_target), source)
                 + absolute_distance(self.generators[1](encoded_source), target))
        perceptual = squared_distance(zs, encoded_target) + squared_distance(zt, encoded_source)
        w_ae, w_pred, w_cycle, w_perceptual = PAPER_LOSS_WEIGHTS
        total = gan + w_ae * autoencoder + w_pred * prediction + w_cycle * cycle + w_perceptual * perceptual
        terms = {"gan": gan, "reconstruction": autoencoder, "prediction": prediction,
                 "cycle": cycle, "perceptual": perceptual}
        return total, terms


@contextmanager
def frozen_discriminators(core: MotionTransformerCore):
    params = list(core.discriminator_parameters())
    previous = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, old in zip(params, previous):
            p.requires_grad_(old)


def _optimizer_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def _finite_gradients(parameters: Iterator[nn.Parameter]) -> None:
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("nonfinite domain-stage gradients; optimizer step refused")


def domain_step(core: MotionTransformerCore, source: torch.Tensor, labels: torch.Tensor,
                target: torch.Tensor, generator_optimizer: torch.optim.Optimizer,
                discriminator_optimizer: torch.optim.Optimizer) -> dict:
    """One D step then one joint E/G/decoder/predictor step, an explicit assumption.

    Caller supplies optimizers/recipe. This is a method-specific stage contract,
    not a replacement Trainer. Both unpaired training domains are required.
    Parameter ownership is checked before either update to prevent training the
    discriminator with the generator objective or discarding an entire branch.
    """
    generator_ids = {id(p) for p in core.generator_parameters()}
    discriminator_ids = {id(p) for p in core.discriminator_parameters()}
    if (_optimizer_ids(generator_optimizer) != generator_ids
            or _optimizer_ids(discriminator_optimizer) != discriminator_ids):
        raise ValueError("optimizers must own exactly the disjoint generator/discriminator partitions")
    core.check_sequence(source)
    core.check_sequence(target)
    if labels.shape != (len(source), core.label_dims) or not torch.isfinite(labels).all():
        raise ValueError("finite labelled source is required before any optimizer update")
    if (source.device != target.device or source.device != labels.device
            or source.dtype != target.dtype or source.dtype != labels.dtype):
        raise ValueError("source/target/labels must share floating dtype and device")
    core.train()
    generator_optimizer.zero_grad(set_to_none=True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    d_loss = core.discriminator_loss(source, target)
    if not torch.isfinite(d_loss):
        raise FloatingPointError("nonfinite discriminator objective")
    d_loss.backward()
    _finite_gradients(core.discriminator_parameters())
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)
    with frozen_discriminators(core):
        g_loss, terms = core.generator_objective(source, labels, target)
        if not torch.isfinite(g_loss):
            raise FloatingPointError("nonfinite joint generator objective")
        g_loss.backward()
    _finite_gradients(core.generator_parameters())
    generator_optimizer.step()
    return {"discriminator": d_loss.detach(), "total": g_loss.detach(),
            **{name: value.detach() for name, value in terms.items()}}


def integrate_polar(labels: torch.Tensor, initial_position: torch.Tensor,
                    initial_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper Eq.14: concatenate nonoverlapping (delta_length,delta_heading) windows.

    The returned trajectory includes the initial position. No GT heading is
    injected for later windows. This is exclusively planar paper inference.
    """
    if labels.ndim != 3 or labels.shape[-1] != 2 or labels.shape[1] < 1:
        raise ValueError("polar trajectory needs (B,number_of_windows,2)")
    if initial_position.shape != (labels.shape[0], 2) or initial_heading.shape != (labels.shape[0],):
        raise ValueError("initial position/heading shapes disagree with polar windows")
    headings = initial_heading[:, None] + labels[..., 1].cumsum(1)
    delta_xy = labels[..., :1] * torch.stack([headings.cos(), headings.sin()], -1)
    positions = initial_position[:, None, :] + delta_xy.cumsum(1)
    return torch.cat([initial_position[:, None, :], positions], 1), headings


class MotionTransformerAdaptation(BaseModel):
    """Explicit IPB gravity-world velocity adaptation, including learned Z.

    Original paper delta-length/delta-heading and recurrent pose chaining are
    not average world velocity. This adapter replaces their label semantics,
    preserves all domain-GAN branches, and refuses single-domain training.
    A standard NIO Trainer domain-batch/optimizer-stage hook is still required.
    """

    default_loss = "motiontransformer_five_objectives_staging"
    requires_unpaired_domains = True
    saved_outputs = ("speed", "heading", "vertical")

    def __init__(self, input_spec: InputSpec, hidden: int = 128, layers: int = 1,
                 decimation: int = 2, inference_domain: int = 1) -> None:
        super().__init__(input_spec)
        if (input_spec.frame != "gravity_world" or input_spec.target != "avg_velocity"
                or input_spec.history or input_spec.extra_inputs
                or input_spec.channels != ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")):
            raise ValueError("adaptation requires six-axis gravity_world window avg_velocity without extras/history")
        if decimation < 1 or decimation != int(decimation) or input_spec.window != PAPER_STEPS * decimation:
            raise ValueError("input window/decimation must preserve all 200 paper time steps")
        if input_spec.rate != PAPER_RATE_HZ * decimation:
            raise ValueError("input rate/decimation must preserve the paper 100 Hz, two-second span")
        if inference_domain not in (0, 1):
            raise ValueError("inference domain must be 0/1")
        self.core = MotionTransformerCore(hidden, layers, input_spec.dims)
        self.decimation, self.inference_domain = int(decimation), int(inference_domain)

    def prepare(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        if self.decimation > 1:
            imu = F.avg_pool1d(imu, self.decimation, self.decimation)
        # IPB gyro/acc -> paper acc/gyro; raw units are retained.
        return imu[:, [3, 4, 5, 0, 1, 2]].transpose(1, 2)

    def forward(self, imu: torch.Tensor) -> dict:
        polar = self.core.predict(self.prepare(imu), self.inference_domain)
        speed, heading = polar[:, 0], polar[:, 1]
        vel = speed[:, None] * torch.stack([heading.cos(), heading.sin()], -1)
        result = {"speed": speed, "heading": heading}
        if self.input_spec.dims == 3:
            # A learned vertical velocity, not zero padding or a magnitude projection.
            vel = torch.cat([vel, polar[:, 2:3]], -1)
            result["vertical"] = polar[:, 2]
        return {"vel": vel, **result}

    def set_loss(self, name: str, **kwargs) -> None:
        if name != self.default_loss or kwargs:
            raise ValueError("five-objective domain training cannot be replaced by a single-domain MSE")
        self.loss_name, self.loss_kwargs = name, {}

    def source_labels(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 2 or target.shape[-1] != self.input_spec.dims:
            raise ValueError("adapted labels require (B,dims) average world velocity")
        labels = torch.stack([target[:, :2].norm(dim=-1),
                              torch.atan2(target[:, 1], target[:, 0])], -1)
        return torch.cat([labels, target[:, 2:3]], -1) if self.input_spec.dims == 3 else labels

    def loss(self, out: dict, batch: dict, epoch: int = 0) -> tuple:
        if "target_domain_imu" not in batch:
            raise ValueError("MotionTransformer requires unpaired unlabelled target_domain_imu; standard single-domain Trainer is pending")
        if batch.get("mask") is not None and not batch["mask"].all():
            raise ValueError("domain coordinator must select valid training windows before constructing a batch")
        source, target = self.prepare(batch["imu"]), self.prepare(batch["target_domain_imu"])
        with frozen_discriminators(self.core):
            total, terms = self.core.generator_objective(source, self.source_labels(batch["target"]), target)
        return total, {key: value.detach() for key, value in terms.items()}
