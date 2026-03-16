from __future__ import annotations

from collections.abc import Sequence

import numpy as np


BezierSegment = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def fit_curve(points: Sequence[Sequence[float]] | np.ndarray, error: float = 0.5) -> list[BezierSegment]:
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("fit_curve expects an (n, 2) point array")
    if len(coords) < 2:
        return []
    if len(coords) == 2:
        return [_line_as_cubic(coords[0], coords[1])]

    left_tangent = _estimate_tangent(coords, 0, forward=True)
    right_tangent = _estimate_tangent(coords, len(coords) - 1, forward=False)
    if _is_zero_vector(left_tangent) or _is_zero_vector(right_tangent):
        return [_line_as_cubic(coords[0], coords[-1])]

    return _fit_cubic(coords, left_tangent, right_tangent, max(1e-6, float(error)) ** 2)


def _fit_cubic(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    error_sq: float,
) -> list[BezierSegment]:
    if len(points) == 2:
        return [_line_as_cubic(points[0], points[1])]

    parameters = _chord_length_parameterize(points)
    bezier = _generate_bezier(points, parameters, left_tangent, right_tangent)
    max_error, split_index = _compute_max_error(points, bezier, parameters)
    if max_error <= error_sq:
        return [bezier]

    if max_error <= error_sq * 4.0:
        for _ in range(8):
            refined = _reparameterize(points, bezier, parameters)
            if refined is None:
                break
            bezier = _generate_bezier(points, refined, left_tangent, right_tangent)
            max_error, split_index = _compute_max_error(points, bezier, refined)
            if max_error <= error_sq:
                return [bezier]
            parameters = refined

    split_index = min(max(1, split_index), len(points) - 2)
    center_tangent = _estimate_center_tangent(points, split_index)
    if _is_zero_vector(center_tangent):
        center_tangent = _estimate_tangent(points, split_index, forward=True)
    if _is_zero_vector(center_tangent):
        center_tangent = _estimate_tangent(points, split_index, forward=False)

    left = _fit_cubic(points[: split_index + 1], left_tangent, center_tangent, error_sq)
    right = _fit_cubic(points[split_index:], -center_tangent, right_tangent, error_sq)
    return left + right


def _generate_bezier(
    points: np.ndarray,
    parameters: list[float],
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> BezierSegment:
    p0 = points[0]
    p3 = points[-1]
    matrix_c = np.zeros((2, 2), dtype=np.float64)
    vector_x = np.zeros(2, dtype=np.float64)

    for point, u in zip(points, parameters):
        b0, b1, b2, b3 = _bernstein_basis(u)
        a1 = left_tangent * b1
        a2 = right_tangent * b2
        matrix_c[0, 0] += np.dot(a1, a1)
        matrix_c[0, 1] += np.dot(a1, a2)
        matrix_c[1, 0] += np.dot(a1, a2)
        matrix_c[1, 1] += np.dot(a2, a2)

        baseline = (p0 * (b0 + b1)) + (p3 * (b2 + b3))
        delta = point - baseline
        vector_x[0] += np.dot(a1, delta)
        vector_x[1] += np.dot(a2, delta)

    alpha_left = 0.0
    alpha_right = 0.0
    determinant = _determinant(matrix_c)
    if abs(determinant) > 1e-12:
        alpha_left = _determinant(np.array([[vector_x[0], matrix_c[0, 1]], [vector_x[1], matrix_c[1, 1]]], dtype=np.float64)) / determinant
        alpha_right = _determinant(np.array([[matrix_c[0, 0], vector_x[0]], [matrix_c[1, 0], vector_x[1]]], dtype=np.float64)) / determinant

    segment_length = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * segment_length
    if alpha_left < epsilon or alpha_right < epsilon:
        distance = segment_length / 3.0
        return _line_as_cubic(p0, p3, left_tangent=left_tangent, right_tangent=right_tangent, distance=distance)

    p1 = p0 + left_tangent * alpha_left
    p2 = p3 + right_tangent * alpha_right
    return p0.copy(), p1, p2, p3.copy()


def _reparameterize(points: np.ndarray, bezier: BezierSegment, parameters: list[float]) -> list[float] | None:
    refined = [0.0]
    for point, parameter in zip(points[1:-1], parameters[1:-1]):
        value = _newton_raphson_root_find(bezier, point, parameter)
        if not np.isfinite(value):
            return None
        value = max(refined[-1] + 1e-4, min(1.0, value))
        refined.append(value)
    refined.append(1.0)
    if any(refined[index] >= refined[index + 1] for index in range(len(refined) - 1)):
        return None
    return refined


def _newton_raphson_root_find(bezier: BezierSegment, point: np.ndarray, parameter: float) -> float:
    q = _evaluate_cubic(bezier, parameter)
    q1 = _evaluate_quadratic(_first_derivative(bezier), parameter)
    q2 = _evaluate_linear(_second_derivative(bezier), parameter)
    diff = q - point
    numerator = np.dot(diff, q1)
    denominator = np.dot(q1, q1) + np.dot(diff, q2)
    if abs(denominator) <= 1e-12:
        return parameter
    return parameter - (numerator / denominator)


def _compute_max_error(points: np.ndarray, bezier: BezierSegment, parameters: list[float]) -> tuple[float, int]:
    max_error = 0.0
    split_index = len(points) // 2
    for index in range(1, len(points) - 1):
        diff = _evaluate_cubic(bezier, parameters[index]) - points[index]
        error = float(np.dot(diff, diff))
        if error > max_error:
            max_error = error
            split_index = index
    return max_error, split_index


def _chord_length_parameterize(points: np.ndarray) -> list[float]:
    distances = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        total += float(np.linalg.norm(points[index] - points[index - 1]))
        distances.append(total)
    if total <= 0.0:
        return [0.0 for _ in range(len(points) - 1)] + [1.0]
    return [distance / total for distance in distances]


def _bernstein_basis(parameter: float) -> tuple[float, float, float, float]:
    omt = 1.0 - parameter
    return (
        omt * omt * omt,
        3.0 * omt * omt * parameter,
        3.0 * omt * parameter * parameter,
        parameter * parameter * parameter,
    )


def _evaluate_cubic(bezier: BezierSegment, parameter: float) -> np.ndarray:
    b0, b1, b2, b3 = _bernstein_basis(parameter)
    return (bezier[0] * b0) + (bezier[1] * b1) + (bezier[2] * b2) + (bezier[3] * b3)


def _evaluate_quadratic(control_points: tuple[np.ndarray, np.ndarray, np.ndarray], parameter: float) -> np.ndarray:
    omt = 1.0 - parameter
    return (
        control_points[0] * omt * omt
        + control_points[1] * 2.0 * omt * parameter
        + control_points[2] * parameter * parameter
    )


def _evaluate_linear(control_points: tuple[np.ndarray, np.ndarray], parameter: float) -> np.ndarray:
    return (control_points[0] * (1.0 - parameter)) + (control_points[1] * parameter)


def _first_derivative(bezier: BezierSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (bezier[1] - bezier[0]) * 3.0,
        (bezier[2] - bezier[1]) * 3.0,
        (bezier[3] - bezier[2]) * 3.0,
    )


def _second_derivative(bezier: BezierSegment) -> tuple[np.ndarray, np.ndarray]:
    first = _first_derivative(bezier)
    return (
        (first[1] - first[0]) * 2.0,
        (first[2] - first[1]) * 2.0,
    )


def _estimate_center_tangent(points: np.ndarray, split_index: int) -> np.ndarray:
    if 0 < split_index < len(points) - 1:
        tangent = points[split_index - 1] - points[split_index + 1]
        if not _is_zero_vector(tangent):
            return _normalize(tangent)
    return np.zeros(2, dtype=np.float64)


def _estimate_tangent(points: np.ndarray, index: int, forward: bool) -> np.ndarray:
    step = 1 if forward else -1
    anchor = points[index]
    cursor = index + step
    while 0 <= cursor < len(points):
        delta = points[cursor] - anchor if forward else points[cursor] - anchor
        if not _is_zero_vector(delta):
            return _normalize(delta)
        cursor += step
    return np.zeros(2, dtype=np.float64)


def _line_as_cubic(
    start: np.ndarray,
    end: np.ndarray,
    left_tangent: np.ndarray | None = None,
    right_tangent: np.ndarray | None = None,
    distance: float | None = None,
) -> BezierSegment:
    delta = end - start
    segment_length = float(np.linalg.norm(delta))
    distance = segment_length / 3.0 if distance is None else distance
    if left_tangent is None or _is_zero_vector(left_tangent):
        left_tangent = _normalize(delta)
    if right_tangent is None or _is_zero_vector(right_tangent):
        right_tangent = _normalize(start - end)
    if _is_zero_vector(left_tangent) or _is_zero_vector(right_tangent):
        left_tangent = np.array([1.0, 0.0], dtype=np.float64)
        right_tangent = np.array([-1.0, 0.0], dtype=np.float64)
    return (
        start.copy(),
        start + left_tangent * distance,
        end + right_tangent * distance,
        end.copy(),
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _is_zero_vector(vector: np.ndarray) -> bool:
    return float(np.linalg.norm(vector)) <= 1e-12


def _determinant(matrix: np.ndarray) -> float:
    return float(matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1])
