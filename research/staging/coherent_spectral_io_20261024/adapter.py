"""Private NIO bridge; importing this file never registers a public model."""
from inertial_benchmark.nn.base import BaseModel
from inertial_benchmark.data.views import CHANNELS
from model import CoherentSpectralIO


class CandidateAdapter(BaseModel):
    causality = 'window_noncausal'
    default_loss = 'mse'

    def __init__(self, input_spec, width=48, depth=3, spectral=True, cross_sensor=True, moments=True, temporal_moments=True):
        super().__init__(input_spec)
        s = input_spec
        if (tuple(s.channels) != CHANNELS or s.window != 200 or s.rate != 200 or s.dims != 3
                or s.frame != 'gravity_world' or s.orientation != 'reference' or s.remove_gravity
                or s.target != 'avg_velocity' or s.output_layout != 'window' or s.output_steps
                or s.overlap != 'mean' or s.history or s.history_stride or s.extra_inputs):
            raise ValueError('Requires T200/200Hz XYZ average velocity, reference_oracle, no extra inputs')
        self.network = CoherentSpectralIO(width, depth, spectral, cross_sensor, moments, temporal_moments)

    def forward(self, imu):
        self.check_input(imu)
        return self.network(imu)

    def set_loss(self, name, **kwargs):
        if name != 'mse' or kwargs:
            raise ValueError('This ablation identity uses plain MSE')
        super().set_loss(name)
