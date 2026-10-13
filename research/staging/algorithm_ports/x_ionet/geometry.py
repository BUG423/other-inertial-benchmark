"""Float64 SO(3) primitives for an explicit engineering left-error convention."""

from __future__ import annotations

import math

import numpy as np


def vector3(values, name: str = "vector") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return values.copy()


def skew(values) -> np.ndarray:
    x, y, z = vector3(values)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (3, 3) or not np.isfinite(values).all()
            or not np.allclose(values.T @ values, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(values), 1.0, rtol=0, atol=1e-8)):
        raise ValueError("rotation must already be a finite proper SO(3) matrix")
    return values.copy()


def so3_exp(theta) -> np.ndarray:
    theta = vector3(theta, "rotation vector")
    angle2 = float(theta @ theta)
    matrix = skew(theta)
    if angle2 < 1e-8:
        a = 1 - angle2 / 6 + angle2 ** 2 / 120
        b = 0.5 - angle2 / 24 + angle2 ** 2 / 720
    else:
        angle = math.sqrt(angle2)
        a = math.sin(angle) / angle
        b = (1 - math.cos(angle)) / angle2
    return np.eye(3) + a * matrix + b * (matrix @ matrix)


def so3_log(matrix) -> np.ndarray:
    """Principal Log, with a canonical axis at exactly pi (axis sign is ambiguous)."""
    matrix = rotation(matrix)
    cosine = float(np.clip((np.trace(matrix) - 1) / 2, -1, 1))
    antisymmetric = np.array([matrix[2, 1] - matrix[1, 2],
                             matrix[0, 2] - matrix[2, 0],
                             matrix[1, 0] - matrix[0, 1]])
    # atan2 retains the small skew part near pi, where acos(trace) loses precision.
    angle = math.atan2(0.5 * float(np.linalg.norm(antisymmetric)), cosine)
    if angle < 1e-4:
        scale = 0.5 * (1 + angle ** 2 / 6 + 7 * angle ** 4 / 360)
        return scale * antisymmetric
    if math.pi - angle < 1e-5:
        # At/near pi the ordinary division by sin(angle) is ill-conditioned.
        outer = (matrix + matrix.T + 2 * np.eye(3)) / 4
        axis_index = int(np.argmax(np.diag(outer)))
        axis = np.zeros(3)
        axis[axis_index] = math.sqrt(max(float(outer[axis_index, axis_index]), 0))
        if axis[axis_index] <= 1e-12:
            raise ValueError("cannot recover the principal near-pi rotation axis")
        for i in range(3):
            if i != axis_index:
                axis[i] = outer[i, axis_index] / axis[axis_index]
        axis /= np.linalg.norm(axis)
        # Exactly pi has no signed principal axis. Preserve the positive largest
        # component rather than selecting a sign from floating sin(pi) residue.
        if angle < math.pi and float(axis @ antisymmetric) < 0:
            axis = -axis
        return angle * axis
    return (angle / (2 * math.sin(angle))) * antisymmetric


def left_jacobian(theta) -> np.ndarray:
    """Derivative of Log(Exp(theta+eta) Exp(-theta)) at eta=0.

    This is I + 0.5*hat(theta) + ... near zero, not the right-Jacobian sign.
    It is the reset block for additive old left-error tangent coordinates.
    """
    theta = vector3(theta, "left tangent")
    angle2 = float(theta @ theta)
    matrix = skew(theta)
    if angle2 < 1e-8:
        b = 0.5 - angle2 / 24 + angle2 ** 2 / 720
        c = 1 / 6 - angle2 / 120 + angle2 ** 2 / 5040
    else:
        angle = math.sqrt(angle2)
        b = (1 - math.cos(angle)) / angle2
        c = (angle - math.sin(angle)) / (angle2 * angle)
    return np.eye(3) + b * matrix + c * (matrix @ matrix)
