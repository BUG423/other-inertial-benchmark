"""Independent mathematics of the released MEIO-Net multi-exit phase loss.

Exit weights are normalized powers. NLL weighting multiplies only log-sigma,
not the Mahalanobis residual, as in the official EarlyExitPDRModule specification.
The final reduction is a mean over exits (not a weighted sum). Zero warmup denotes
an instantaneous NLL switch, resolving the released wrapper's division by zero.
"""

from __future__ import annotations

import torch

from ...nn.losses import expand_mask, masked_mean, mse, register_loss


@register_loss("meio_multiexit")
class MEIOMultiExitLoss:
    name = "meio_multiexit"

    def __init__(self, switch_epoch: int = 10, warmup_epochs: int = 0,
                 mse_power: float = 1., covariance_power: float = .5) -> None:
        if switch_epoch < 0 or warmup_epochs < 0:
            raise ValueError("MEIO phase epochs must be nonnegative")
        if mse_power < 0 or covariance_power < 0:
            raise ValueError("MEIO exit weight powers must be nonnegative")
        self.switch_epoch, self.warmup_epochs = int(switch_epoch), int(warmup_epochs)
        self.mse_power, self.covariance_power = float(mse_power), float(covariance_power)

    @staticmethod
    def _weights(count: int, power: float, like: torch.Tensor) -> torch.Tensor:
        value = (torch.arange(1, count + 1, device=like.device, dtype=like.dtype) / count) ** power
        return value / value.sum()

    def phase(self, epoch: int) -> str:
        if epoch < self.switch_epoch:
            return "weighted_multiexit_mse"
        if self.warmup_epochs and epoch < self.switch_epoch + self.warmup_epochs:
            return "mse_nll_blend"
        return "weighted_multiexit_nll"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0, mask=None) -> tuple:
        if "all_vel" not in out:
            # Common validation reports the actually selected inference exit; the original
            # wrapper instead logs full-depth errors. No original validation equality claim.
            value = mse(out["vel"], target, mask)
            return value, {"mse": value.detach()}
        means, sigmas = out["all_vel"], out["all_logstd"]
        count = means.shape[1]
        mse_weights = self._weights(count, self.mse_power, means)
        covariance_weights = self._weights(count, self.covariance_power, means)
        errors, likelihoods = [], []
        for i in range(count):
            pred, sigma = means[:, i], sigmas[:, i]
            errors.append(mse(pred, target, mask))
            if epoch < self.switch_epoch:
                sigma = sigma.detach()
            terms = .5 * (pred - target).square() * torch.exp(-2 * sigma)
            terms = terms + covariance_weights[i] * sigma
            likelihoods.append(masked_mean(terms, expand_mask(mask, terms)))
        weighted_mse = (torch.stack(errors) * mse_weights).mean()
        weighted_nll = torch.stack(likelihoods).mean()
        if epoch < self.switch_epoch:
            value = weighted_mse
        else:
            alpha = (max(0., min(1., 1 - (epoch - self.switch_epoch) / self.warmup_epochs))
                     if self.warmup_epochs else 0.)
            value = alpha * weighted_mse + (1 - alpha) * weighted_nll
        return value, {"mse": weighted_mse.detach(), "nll": weighted_nll.detach()}
