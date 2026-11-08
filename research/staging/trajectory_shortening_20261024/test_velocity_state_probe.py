"""Analytical physics checks; run with the Agent's framework, not locally."""
import unittest

import numpy as np

from inertial_benchmark.data.format import Sequence
from inertial_benchmark.data.views import SequenceView, ViewConfig
from inertial_benchmark.utils.geometry import GRAVITY

from velocity_state_probe import average_velocity_increment, boosted_view


class VelocityStateTests(unittest.TestCase):
    def test_average_increment_matches_constant_acceleration_closed_form(self):
        acceleration = np.broadcast_to([2., -1., .5], (2, 200, 3))
        expected = np.array([.995, -.4975, .24875])
        np.testing.assert_allclose(average_velocity_increment(acceleration, .005),
                                   np.broadcast_to(expected, (2, 3)), atol=1e-12)

    def test_zero_acceleration_keeps_the_entire_initial_velocity(self):
        increment = average_velocity_increment(np.zeros((1, 200, 3)), .005)
        np.testing.assert_array_equal(increment, np.zeros((1, 3)))

    def test_galilean_boost_preserves_input_and_changes_average_velocity(self):
        t = np.arange(400) * .005
        a, v0 = np.array([.2, -.1, 0.]), np.array([1., 2., -.5])
        seq = Sequence(timestamp=t, gyroscope=np.zeros((400, 3)),
                       accelerometer=np.tile(a + [0., 0., GRAVITY], (400, 1)),
                       orientation=np.tile([1., 0., 0., 0.], (400, 1)),
                       position=t[:, None] * v0 + .5 * t[:, None] ** 2 * a,
                       valid_imu=np.ones(400, dtype=bool), valid_pose=np.ones(400, dtype=bool))
        cfg, starts, boost = ViewConfig(dims=3), np.array([0, 200]), np.array([.5, -.3, .2])
        original, changed = SequenceView(seq, cfg), boosted_view(seq, cfg, boost)
        np.testing.assert_array_equal(original.imu_windows(starts), changed.imu_windows(starts))
        np.testing.assert_allclose(changed.targets(starts) - original.targets(starts),
                                   np.broadcast_to(boost, (2, 3)), atol=3e-7)
        expected = v0 + a * original.target_times(starts)[:, None]
        np.testing.assert_allclose(original.targets(starts), expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
