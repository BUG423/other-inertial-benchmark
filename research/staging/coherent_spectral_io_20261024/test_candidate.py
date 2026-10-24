"""Run on the Agent only. Mathematical/gradient contracts, not accuracy tests."""
import io
import torch
from model import CoherentSpectralIO, DCPreservingSpectralGate


def test_spectral_dc_preservation_even_and_odd_and_identity():
    for length in (199, 200):
        gate = DCPreservingSpectralGate(8)
        x = torch.randn(4, 8, length)
        torch.testing.assert_close(gate(x), x, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            gate.static.normal_()
            gate.condition.weight.normal_(std=.1)
        torch.testing.assert_close(gate(x).mean(-1), x.mean(-1), atol=1e-6, rtol=1e-5)
        constant = torch.full_like(x, 3.25)
        torch.testing.assert_close(gate(constant), constant, atol=1e-6, rtol=1e-5)


def test_initial_controls_identical_and_every_axis_learns():
    model = CoherentSpectralIO()
    imu = torch.randn(4, 6, 200)
    initial = model(imu)['vel']
    model.spectral_enabled = model.cross_enabled = model.moments_enabled = False
    torch.testing.assert_close(model(imu)['vel'], initial, atol=1e-6, rtol=1e-5)
    model.spectral_enabled = model.cross_enabled = model.moments_enabled = True
    before = model.readout.base.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(4):
        opt.zero_grad()
        loss = (model(imu)['vel'] - torch.randn(4, 3)).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert (model.readout.base.weight - before).abs().sum(dim=1).gt(0).all()
    assert model.cross.project.weight.abs().sum() > 0
    assert model.readout.residual[-1].weight.abs().sum() > 0
    assert model.spectral.static.abs().sum() > 0


def test_window_independence_checkpoint_and_no_target_argument():
    model = CoherentSpectralIO().eval()
    a, b = torch.randn(2, 6, 200), torch.randn(3, 6, 200)
    expected = model(a)['vel']
    model(b)
    torch.testing.assert_close(model(a)['vel'], expected)
    torch.testing.assert_close(model(torch.cat((a, b)))['vel'][:2], expected, atol=1e-6, rtol=1e-5)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded = CoherentSpectralIO().eval()
    loaded.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(loaded(a)['vel'], expected, atol=0, rtol=0)
    assert expected.shape == (2, 3) and torch.isfinite(expected).all()


def test_spectral_only_has_a_learning_path_through_nonlinear_trunk():
    model = CoherentSpectralIO(cross_sensor=False, moments=False)
    model(torch.randn(4, 6, 200))['vel'].square().mean().backward()
    assert model.spectral.static.grad.abs().sum() > 0


def test_mean_mlp_control_matches_parameters_and_initial_prediction():
    model = CoherentSpectralIO()
    control = CoherentSpectralIO(temporal_moments=False)
    control.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in control.parameters())
    x = torch.randn(2, 6, 200)
    torch.testing.assert_close(model(x)['vel'], control(x)['vel'], atol=0, rtol=0)


def test_ambiguity_can_shorten_bayes_optimal_mse_prediction():
    # Opposite equally probable motions share the same observation. Norm(mean)
    # is zero although mean(norm) is one; forced expansion worsens vector MSE.
    target = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
    prediction = target.mean(0)
    assert prediction.norm() == 0 and target.norm(dim=-1).mean() == 1
    assert ((prediction - target) ** 2).mean() < ((torch.tensor([1., 0., 0.]) - target) ** 2).mean()
