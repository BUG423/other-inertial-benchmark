"""Analytical controls for objective, fit, ambiguity and immutable initialization."""
import unittest

import torch

from head_intervention_probe import fit_head, objective


class HeadInterventionTests(unittest.TestCase):
    def test_magnitude_term_changes_radial_gradient_with_fixed_direction(self):
        p = torch.tensor([[0., .5, 0.]], requires_grad=True)
        y = torch.tensor([[1., 0., 0.]])
        vector = torch.autograd.grad(objective(p, y), p, retain_graph=True)[0]
        combined = torch.autograd.grad(objective(p, y, magnitude=True), p)[0]
        self.assertAlmostEqual(float(vector[0, 1]), 1/3, places=6)
        self.assertAlmostEqual(float(combined[0, 1]), 0., places=6)
        self.assertLess(float(combined[0, 0]), 0.)

    def test_least_squares_recovers_affine_mapping_without_mutating_initial(self):
        initial = torch.nn.Linear(2, 3)
        state = {k: v.clone() for k, v in initial.state_dict().items()}
        x = torch.tensor([[-1., 0.], [0., 1.], [1., 0.], [1., 2.]])
        y = x @ torch.tensor([[2., -1., .5], [-.5, 1., 2.]]) + torch.tensor([.2, -.3, .1])
        fitted, detail = fit_head(initial, x, y, 'least_squares')
        torch.testing.assert_close(fitted(x), y, atol=1e-6, rtol=1e-6)
        self.assertEqual(detail['retained_rank'], 3)
        for key, value in initial.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))

    def test_identical_features_with_opposite_targets_have_zero_mse_solution(self):
        initial = torch.nn.Linear(2, 3)
        x = torch.zeros(2, 2)
        y = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
        fitted, _ = fit_head(initial, x, y, 'least_squares')
        torch.testing.assert_close(fitted(x), torch.zeros(2, 3), atol=1e-6, rtol=0.)
        self.assertAlmostEqual(float(objective(fitted(x), y)), 1/3, places=6)

    def test_optimizer_arms_use_fixed_budget_and_preserve_original_head(self):
        torch.manual_seed(7)
        initial = torch.nn.Linear(2, 3)
        state = {k: v.clone() for k, v in initial.state_dict().items()}
        x = torch.tensor([[-1., 0.], [0., 1.], [1., 0.], [1., 2.]])
        y = torch.ones(4, 3)
        for arm in ('vector_mse', 'magnitude_aux'):
            fitted, detail = fit_head(initial, x, y, arm)
            self.assertEqual(detail['updates'], 256)
            self.assertTrue(torch.isfinite(fitted(x)).all())
            self.assertLess(float(objective(fitted(x), y)), float(objective(initial(x), y)))
            for key, value in initial.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]))


if __name__ == '__main__':
    unittest.main()
