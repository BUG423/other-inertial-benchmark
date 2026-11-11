"""Exact mask enumeration and analytical ridge controls for expected Dropout."""
import itertools
import unittest

import torch

from dropout_head_probe import expected_objective, fit_expected_head
from head_intervention_probe import fit_head


class ExpectedDropoutTests(unittest.TestCase):
    def test_expected_loss_and_gradient_match_all_two_feature_masks(self):
        x = torch.tensor([[1., -2.], [-.5, 3.]], dtype=torch.float64)
        y = torch.tensor([[2., 1., -.5], [-1., .4, 2.]], dtype=torch.float64)
        w = torch.tensor([[.2, .5], [-.3, .1], [.8, -.4]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
        expected = expected_objective(x, y, w, b, .5)
        enumerated = sum(((x * torch.tensor(mask) @ w.T + b - y).square().mean()) / 4
                         for mask in itertools.product((0., 2.), repeat=2))
        torch.testing.assert_close(expected, enumerated)
        for a, c in zip(torch.autograd.grad(expected, (w, b), retain_graph=True),
                        torch.autograd.grad(enumerated, (w, b))):
            torch.testing.assert_close(a, c)

    def test_centered_one_feature_solution_shrinks_by_keep_probability(self):
        initial = torch.nn.Linear(1, 3)
        state = {k: v.clone() for k, v in initial.state_dict().items()}
        x = torch.tensor([[-2.], [-1.], [1.], [2.]])
        y = x * torch.tensor([[2., 0., 0.]])
        plain, _ = fit_expected_head(initial, x, y, 'no_dropout')
        regularized, detail = fit_expected_head(initial, x, y, 'dropout_half')
        torch.testing.assert_close(plain(x), y, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(regularized(x), y*.5, atol=1e-6, rtol=1e-6)
        self.assertLess(detail['relative_normal_residual'], 1e-10)
        for key, value in initial.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))

    def test_bias_is_not_regularized(self):
        initial = torch.nn.Linear(1, 3)
        x, y = torch.zeros(4, 1), torch.tensor([[2., -1., .5]]).repeat(4, 1)
        fitted, _ = fit_expected_head(initial, x, y, 'dropout_half')
        torch.testing.assert_close(fitted(x), y, atol=1e-6, rtol=1e-6)

    def test_zero_dropout_matches_previous_least_squares_control(self):
        initial = torch.nn.Linear(2, 3)
        x = torch.tensor([[-1., 0.], [0., 1.], [1., 0.], [1., 2.]])
        y = x @ torch.tensor([[2., -1., .5], [-.5, 1., 2.]]) + .3
        previous, _ = fit_head(initial, x, y, 'least_squares')
        current, _ = fit_expected_head(initial, x, y, 'no_dropout')
        torch.testing.assert_close(previous(x), current(x), atol=1e-6, rtol=1e-6)

    def test_invalid_probabilities_and_arms_are_rejected(self):
        initial = torch.nn.Linear(1, 3)
        x, y = torch.ones(2, 1), torch.ones(2, 3)
        for probability in (-.1, 1., float('nan')):
            with self.assertRaises(ValueError):
                expected_objective(x, y, initial.weight, initial.bias, probability)
        with self.assertRaises(ValueError):
            fit_expected_head(initial, x, y, 'unknown')


if __name__ == '__main__':
    unittest.main()
