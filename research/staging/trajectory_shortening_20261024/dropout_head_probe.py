"""Expected final-feature Dropout control, conditional on frozen representations."""
import copy

import numpy as np
import torch

from head_intervention_probe import main


def expected_objective(features, target, weight, bias, probability):
    if not 0 <= probability < 1:
        raise ValueError('Dropout probability must be in [0,1)')
    error = (features @ weight.T + bias - target).square().mean()
    penalty = (features.square().mean(0) * weight.square().sum(0)).sum()
    return error + probability / (1-probability) * penalty / target.shape[1]


def fit_expected_head(initial, features, target, arm):
    probabilities = dict(no_dropout=0., dropout_half=.5)
    if arm not in probabilities:
        raise ValueError('unknown expected Dropout arm')
    probability = probabilities[arm]
    x, y = features.numpy().astype(np.float64), target.numpy().astype(np.float64)
    energy = np.mean(x*x, axis=0)
    design = np.column_stack((x, np.ones(len(x)))) / np.sqrt(len(x))
    labels = y / np.sqrt(len(x))
    if probability:
        regularizer = np.column_stack((np.diag(np.sqrt(probability/(1-probability)*energy)),
                                       np.zeros(x.shape[1])))
        design = np.vstack((design, regularizer))
        labels = np.vstack((labels, np.zeros((x.shape[1], y.shape[1]))))
    original = np.vstack((initial.weight.detach().numpy().T, initial.bias.detach().numpy()))
    delta, _, rank, _ = np.linalg.lstsq(design, labels-design@original, rcond=1e-6)
    fitted = original + delta
    head = copy.deepcopy(initial).requires_grad_(False)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(fitted[:-1].T))
        head.bias.copy_(torch.from_numpy(fitted[-1]))
        objective = float(expected_objective(features, target, head.weight, head.bias, probability))
    normal = design.T @ (design @ fitted-labels)
    residual = float(np.linalg.norm(normal) / max(np.linalg.norm(design.T @ labels), 1e-12))
    return head, dict(updates=0, dropout_probability=probability, rcond=1e-6,
                      retained_rank=int(rank), design_columns=design.shape[1],
                      expected_objective=objective, relative_normal_residual=residual)


if __name__ == '__main__':
    main(plan_name='dropout_head_plan.json', receipt_name='dropout-head-receipt.json',
         schema='ib.expected-head-dropout.v1', fitter=fit_expected_head)
