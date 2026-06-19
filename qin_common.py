"""Shared optimizer and reporting utilities for QIN experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


Array = np.ndarray


@dataclass
class BenchmarkResult:
    name: str
    values: list[float]
    errors: list[float]
    scores: list[float]
    elapsed: float
    iterations: int
    best_raw: float
    best_reported: float
    abs_error: float
    rel_error: float
    score: float


def init_normal(shape: tuple[int, ...], seed: int = 42, scale: float = math.pi) -> Array:
    np.random.seed(seed)
    return np.array(np.random.normal(0.0, scale, shape), requires_grad=True)


def flatten(x: Array) -> Array:
    return np.reshape(x, (-1,))


def unflatten(x: Array, shape: tuple[int, ...]) -> Array:
    return np.reshape(x, shape)


def metric_matrix(metric_raw, n_params: int) -> Array:
    if isinstance(metric_raw, (tuple, list)) and len(metric_raw) == 1:
        metric_raw = metric_raw[0]
    metric = np.asarray(metric_raw)
    if metric.ndim != 2:
        metric = np.reshape(metric, (n_params, n_params))
    return np.asarray(metric, dtype=float)


def diagonal_hessian(
    objective: Callable[[Array], float],
    params: Array,
    shape: tuple[int, ...],
    eps: float = 1e-4,
    max_params: int | None = None,
) -> Array:
    theta = np.array(params, requires_grad=False)
    flat = flatten(theta)
    base_value = float(objective(theta))
    hdiag = np.zeros_like(flat)
    if max_params is None or max_params >= len(flat):
        indices = range(len(flat))
    else:
        indices = np.linspace(0, len(flat) - 1, max_params, dtype=int)

    for idx in indices:
        plus = np.array(flat, requires_grad=False)
        minus = np.array(flat, requires_grad=False)
        plus[idx] = plus[idx] + eps
        minus[idx] = minus[idx] - eps
        f_plus = float(objective(unflatten(plus, shape)))
        f_minus = float(objective(unflatten(minus, shape)))
        hdiag[idx] = (f_plus - 2.0 * base_value + f_minus) / (eps**2)

    return np.asarray(hdiag, dtype=float)

def golden_search_alpha(
    objective,
    params,
    shape,
    grad_flat,
    metric,
    curvature_diag,
    damping,
    lr,
    tol=1e-3,
    max_iter=20,
):

    a = 0.05
    b = 0.95

    phi_const = (np.sqrt(5.0) - 1.0) / 2.0

    def lookahead(alpha):

        M = (
            alpha * metric
            + (1.0 - alpha) * np.diag(curvature_diag)
            + damping * np.eye(len(grad_flat))
        )

        direction = solve_direction(
            M,
            grad_flat
        )

        candidate = np.array(
            params - lr * unflatten(direction, shape),
            requires_grad=True
        )

        return float(objective(candidate))

    c = b - phi_const * (b - a)
    d = a + phi_const * (b - a)

    fc = lookahead(c)
    fd = lookahead(d)

    for _ in range(max_iter):

        if abs(b - a) < tol:
            break

        if fc < fd:

            b = d
            d = c
            fd = fc

            c = b - phi_const * (b - a)
            fc = lookahead(c)

        else:

            a = c
            c = d
            fc = fd

            d = a + phi_const * (b - a)
            fd = lookahead(d)

    return 0.5 * (a + b)

def solve_direction(matrix: Array, grad_flat: Array) -> Array:
    try:
        return np.linalg.solve(matrix, grad_flat)
    except Exception:
        return np.linalg.pinv(matrix) @ grad_flat


def lbfgs_direction(
    grad_flat: Array,
    s_history: list[Array],
    y_history: list[Array],
    rho_history: list[float],
) -> Array:
    """Return the positive L-BFGS preconditioned gradient H_k g_k."""

    q = np.array(grad_flat, requires_grad=False)
    alphas = []
    for s_vec, y_vec, rho in reversed(list(zip(s_history, y_history, rho_history))):
        alpha_i = rho * float(np.dot(s_vec, q))
        alphas.append(alpha_i)
        q = q - alpha_i * y_vec

    if s_history:
        sy = float(np.dot(s_history[-1], y_history[-1]))
        yy = float(np.dot(y_history[-1], y_history[-1]))
        scale = sy / yy if yy > 1e-14 else 1.0
    else:
        scale = 1.0
    r = scale * q

    for (s_vec, y_vec, rho), alpha_i in zip(
        zip(s_history, y_history, rho_history), reversed(alphas)
    ):
        beta_i = rho * float(np.dot(y_vec, r))
        r = r + s_vec * (alpha_i - beta_i)
    return np.asarray(r, dtype=float)


def line_search(
    objective: Callable[[Array], float],
    params: Array,
    shape: tuple[int, ...],
    direction_flat: Array,
    grad_flat: Array,
    value: float,
    initial_lr: float,
    c: float = 1e-4,
    beta: float = 0.5,
    max_steps: int = 6,
) -> tuple[Array, float, float]:
    descent = float(np.dot(grad_flat, direction_flat))
    if not np.isfinite(descent) or descent <= 0:
        direction_flat = grad_flat
        descent = float(np.dot(grad_flat, grad_flat))

    lr = initial_lr
    best_params = params
    best_value = value

    for _ in range(max_steps):
        candidate = np.array(params - lr * unflatten(direction_flat, shape), requires_grad=True)
        candidate_value = float(objective(candidate))
        if candidate_value < best_value:
            best_params = candidate
            best_value = candidate_value
        if candidate_value <= value - c * lr * descent:
            return candidate, candidate_value, lr
        lr *= beta

    return np.array(best_params, requires_grad=True), best_value, lr

def make_energy_metrics(
    exact: float | None,
    reference: float | None,
    minimize: bool = True,
):
    def collect(values: list[float]):

        best_raw = min(values) if minimize else max(values)

        target = exact if exact is not None else best_raw

        best_reported = best_raw

        abs_error = abs(best_reported - target)

        rel_error = abs_error / max(abs(target), 1e-12)

        #
        # Correlation / recovery score
        #
        if (
            reference is not None
            and exact is not None
            and abs(reference - exact) > 1e-12
        ):

            denom = reference - exact

            score = (
                (reference - best_reported)
                / denom
                * 100.0
            )

            scores = [
                (
                    (reference - v)
                    / denom
                    * 100.0
                )
                for v in values
            ]

        else:

            score = 0.0

            scores = [
                0.0
                for _ in values
            ]

        errors = [
            abs(v - target)
            for v in values
        ]

        return (
            float(best_raw),
            float(best_reported),
            float(abs_error),
            float(rel_error),
            float(score),
            errors,
            scores,
        )

    return collect

def make_energy_metrics_chemistry(
    exact: float | None,
    reference: float | None,
    minimize: bool = True,
):
    def collect(values: list[float]):

        best_raw = min(values) if minimize else max(values)

        target = exact if exact is not None else best_raw

        best_reported = best_raw

        abs_error = abs(best_reported - target)

        rel_error = abs_error / max(abs(target), 1e-12)

        use_recovery_score = (
            reference is not None
            and exact is not None
            and abs(reference - exact) > 1e-2
        )

        if use_recovery_score:

            denom = reference - exact

            score = (
                (reference - best_reported)
                / denom
                * 100.0
            )

            scores = [
                (
                    (reference - v)
                    / denom
                    * 100.0
                )
                for v in values
            ]

        else:

            initial_error = max(
                abs(values[0] - target),
                1e-12
            )

            scores = [
                100.0
                * (
                    1.0
                    - abs(v - target)
                    / initial_error
                )
                for v in values
            ]

            score = scores[
                np.argmin(
                    [abs(v - target) for v in values]
                )
            ]

        errors = [
            abs(v - target)
            for v in values
        ]

        return (
            float(best_raw),
            float(best_reported),
            float(abs_error),
            float(rel_error),
            float(score),
            errors,
            scores,
        )

    return collect


def collect_result(
    name: str,
    values: list[float],
    elapsed: float,
    metrics: Callable[[list[float]], tuple[float, float, float, float, float, list[float], list[float]]],
) -> BenchmarkResult:
    best_raw, best_reported, abs_error, rel_error, score, errors, scores = metrics(values)
    return BenchmarkResult(
        name=name,
        values=values,
        errors=errors,
        scores=scores,
        elapsed=elapsed,
        iterations=len(values) - 1,
        best_raw=best_raw,
        best_reported=best_reported,
        abs_error=abs_error,
        rel_error=rel_error,
        score=score,
    )


def run_optimizer_suite(
    objective: Callable[[Array], float],
    shape: tuple[int, ...],
    metric_fn: Callable[[Array], Array] | None,
    metrics: Callable[[list[float]], tuple[float, float, float, float, float, list[float], list[float]]],
    max_iter: int = 100,
    tol: float = 1e-8,
    switch_iter: int = 30,
    alpha: float = 0.5,
    damping: float = 1e-4,
    hess_eps: float = 1e-4,
    lr_gd: float = 0.2,
    lr_adam: float = 0.05,
    lr_natural: float = 0.2,
    lr_sgd: float = 0.02,
    lr_lbfgs: float = 1.0,
    lbfgs_history_size: int = 10,
    hessian_max_params: int | None = None,
    verbose: bool = True,
    progress_interval: int = 1,
) -> list[BenchmarkResult]:
    n_params = math.prod(shape)
    grad_fn = qml.grad(objective)

    def natural_metric(params: Array) -> Array:
        if metric_fn is None:
            return np.eye(n_params)
        return metric_matrix(metric_fn(params), n_params)

    def report_progress(name: str, step: int, values: list[float], start: float) -> None:
        if verbose and progress_interval and (
            step == 1 or step % progress_interval == 0 or step == max_iter
        ):
            print(
                f"  {name} iter {step}/{max_iter}: current = {values[-1]:.8f}, "
                f"best = {min(values):.8f}, elapsed = {time.perf_counter() - start:.2f} s",
                flush=True,
            )

    def gradient_descent() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(1, max_iter + 1):
            grad = grad_fn(params)
            grad_flat = flatten(grad)
            params, value, _ = line_search(objective, params, shape, grad_flat, grad_flat, values[-1], lr_gd)
            values.append(float(value))
            report_progress("Gradient Descent", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("Gradient Descent", values, time.perf_counter() - start, metrics)

    def sgd() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(1, max_iter + 1):
            grad = grad_fn(params)
            params = np.array(params - lr_sgd * grad, requires_grad=True)
            values.append(float(objective(params)))
            report_progress("SGD", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("SGD", values, time.perf_counter() - start, metrics)

    def adam() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros(shape)
        v = np.zeros(shape)
        for step in range(1, max_iter + 1):
            grad = grad_fn(params)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad**2)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            params = np.array(params - lr_adam * m_hat / (np.sqrt(v_hat) + eps), requires_grad=True)
            values.append(float(objective(params)))
            report_progress("Adam", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("Adam", values, time.perf_counter() - start, metrics)
    '''
    def lbfgs() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        s_history: list[Array] = []
        y_history: list[Array] = []
        rho_history: list[float] = []
        grad_flat = flatten(grad_fn(params))

        for step in range(1, max_iter + 1):
            old_flat = flatten(params)
            direction = lbfgs_direction(grad_flat, s_history, y_history, rho_history)
            params, value, _ = line_search(
                objective, params, shape, direction, grad_flat, values[-1], lr_lbfgs
            )
            values.append(float(value))
            new_grad_flat = flatten(grad_fn(params))
            s_vec = flatten(params) - old_flat
            y_vec = new_grad_flat - grad_flat
            curvature = float(np.dot(s_vec, y_vec))
            if curvature > 1e-10:
                s_history.append(np.asarray(s_vec, dtype=float))
                y_history.append(np.asarray(y_vec, dtype=float))
                rho_history.append(1.0 / curvature)
                if len(s_history) > lbfgs_history_size:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
            grad_flat = new_grad_flat
            report_progress("L-BFGS", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("L-BFGS", values, time.perf_counter() - start, metrics)
    '''
    def qng() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(1, max_iter + 1):
            grad_flat = flatten(grad_fn(params))
            metric = natural_metric(params)
            matrix = metric + damping * np.eye(n_params)
            direction = solve_direction(matrix, grad_flat)
            params, value, _ = line_search(objective, params, shape, direction, grad_flat, values[-1], lr_natural)
            values.append(float(value))
            report_progress("QNG", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("QNG", values, time.perf_counter() - start, metrics)
    def qin() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(1, max_iter + 1):
            grad_flat = flatten(grad_fn(params))
            metric = natural_metric(params)
            hdiag = diagonal_hessian(objective, params, shape, eps=hess_eps, max_params=hessian_max_params)
            matrix = alpha * metric + (1.0 - alpha) * np.diag(hdiag) + damping * np.eye(n_params)
            direction = solve_direction(matrix, grad_flat)
            params, value, _ = line_search(objective, params, shape, direction, grad_flat, values[-1], lr_natural)
            values.append(float(value))
            report_progress("QIN", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("QIN", values, time.perf_counter() - start, metrics)

    def aqin() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(1, max_iter + 1):
            grad_flat = flatten(grad_fn(params))
            metric = natural_metric(params)
            hdiag = diagonal_hessian(objective, params, shape, eps=hess_eps, max_params=hessian_max_params)
            curvature_magnitude = hdiag**2
            matrix = alpha * metric + (1.0 - alpha) * np.diag(curvature_magnitude) + damping * np.eye(n_params)
            direction = solve_direction(matrix, grad_flat)
            params, value, _ = line_search(objective, params, shape, direction, grad_flat, values[-1], lr_natural)
            values.append(float(value))
            report_progress("AQIN", step, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("AQIN", values, time.perf_counter() - start, metrics)
     
    def adaptive_aqin_golden() -> BenchmarkResult:

      params = init_normal(shape)

      values = [float(objective(params))]

      start = time.perf_counter()

      alpha_history = []

      for step in range(1, max_iter + 1):

        grad_flat = flatten(
            grad_fn(params)
        )

        metric = natural_metric(
            params
        )

        hdiag = diagonal_hessian(
            objective,
            params,
            shape,
            eps=hess_eps,
            max_params=hessian_max_params
        )

        curvature = hdiag**2

        alpha_k = golden_search_alpha(
            objective=objective,
            params=params,
            shape=shape,
            grad_flat=grad_flat,
            metric=metric,
            curvature_diag=curvature,
            damping=damping,
            lr=lr_natural,
        )

        alpha_history.append(alpha_k)

        M = (
            alpha_k * metric
            +
            (1.0 - alpha_k)
            * np.diag(curvature)
            +
            damping*np.eye(n_params)
        )

        direction = solve_direction(
            M,
            grad_flat
        )

        params, value, _ = line_search(
            objective,
            params,
            shape,
            direction,
            grad_flat,
            values[-1],
            lr_natural
        )

        values.append(float(value))

        if (
            verbose
            and progress_interval
            and step % progress_interval == 0
        ):
            print(
                f"  Adaptive AQIN-Golden iter {step}: "
                f"best={min(values):.8f} "
                f"alpha={alpha_k:.4f}",
                flush=True,
            )

        if abs(values[-2] - values[-1]) < tol:
            break

      return collect_result(
        "Adaptive AQIN-Golden",
        values,
        time.perf_counter()-start,
        metrics
    )

    
    def qlbfgs() -> BenchmarkResult:

     params = init_normal(shape)

     values = [float(objective(params))]

     start = time.perf_counter()

     s_history = []
     y_history = []

     memory = 10

     for step in range(1, max_iter + 1):

        grad_flat = flatten(
            grad_fn(params)
        )

        n = len(grad_flat)

        #
        # L-BFGS inverse Hessian approximation
        #
        B = np.eye(n)

        for s, y in zip(s_history, y_history):

            ys = np.dot(y, s)

            if abs(ys) < 1e-12:
                continue

            rho = 1.0 / ys

            I = np.eye(n)

            B = (
                (I - rho * np.outer(s, y))
                @ B
                @ (I - rho * np.outer(y, s))
                + rho * np.outer(s, s)
            )

        metric = natural_metric(params)

        #
        # Quantum Geometry + L-BFGS
        #
        M = (
            alpha * metric
            +
            (1.0 - alpha) * B
            +
            damping * np.eye(n)
        )

        direction = solve_direction(
            M,
            grad_flat
        )

        params_new, value, _ = line_search(
            objective,
            params,
            shape,
            direction,
            grad_flat,
            values[-1],
            lr_natural
        )

        grad_new = flatten(
            grad_fn(params_new)
        )

        s = (
            flatten(params_new)
            -
            flatten(params)
        )

        y = (
            grad_new
            -
            grad_flat
        )

        if np.dot(s, y) > 1e-10:

            s_history.append(s)
            y_history.append(y)

            if len(s_history) > memory:

                s_history.pop(0)
                y_history.pop(0)

        params = params_new

        values.append(
            float(value)
        )

        if (
            verbose
            and progress_interval
            and step % progress_interval == 0
        ):
            print(
                f"  QLBFGS iter {step}: "
                f"best = {min(values):.8f}",
                flush=True,
            )

        if abs(
            values[-2]
            -
            values[-1]
        ) < tol:
            break

     return collect_result(
        "QNG+LBFGS",
        values,
        time.perf_counter() - start,
        metrics,
    )
    
    def adaptive_qlbfgs() -> BenchmarkResult:

       params = init_normal(shape)

       values = [float(objective(params))]

       start = time.perf_counter()

       s_history = []
       y_history = []
       rho_history = []

       memory = lbfgs_history_size

       for step in range(1, max_iter + 1):

         grad_flat = flatten(
            grad_fn(params)
         )

         grad_norm = np.linalg.norm(
            grad_flat
         )

         alpha_k = grad_norm / (
            grad_norm + 1.0
        )

         n = len(grad_flat)

         B = np.eye(n)

         for s, y, rho in zip(
            s_history,
            y_history,
            rho_history
         ):

            I = np.eye(n)

            B = (
                (I - rho*np.outer(s,y))
                @ B
                @ (I - rho*np.outer(y,s))
                +
                rho*np.outer(s,s)
            )

         metric = natural_metric(
            params
        )

         M = (
        alpha_k * metric
            +
        (1.0-alpha_k) * B
            +
        damping*np.eye(n)
        )

         direction = solve_direction(
            M,
            grad_flat
        )

         params_new, value, _ = line_search(
            objective,
            params,
            shape,
            direction,
            grad_flat,
            values[-1],
            lr_natural
        )

         grad_new = flatten(
            grad_fn(params_new)
        )

         s = (
            flatten(params_new)
            -
            flatten(params)
        )

         y = (
            grad_new
            -
            grad_flat
        )

         curvature = np.dot(
            s,
            y
        )

         if curvature > 1e-10:

            s_history.append(s)

            y_history.append(y)

            rho_history.append(
                1.0 / curvature
            )

            if len(s_history) > memory:

                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

         params = params_new

         values.append(
            float(value)
         )

         if (
            verbose
            and progress_interval
            and step % progress_interval == 0
        ):
            print(
                f"  Adaptive QLBFGS iter {step}: "
                f"best={min(values):.8f} "
                f"alpha={alpha_k:.4f}",
                flush=True
            )

         if abs(
            values[-2]
            -
            values[-1]
        ) < tol:
            break

       return collect_result(
        "Adaptive QNG+LBFGS",
        values,
        time.perf_counter()-start,
        metrics
    )


    def hybrid() -> BenchmarkResult:
        params = init_normal(shape)
        values = [float(objective(params))]
        start = time.perf_counter()
        for step in range(max_iter):
            grad_flat = flatten(grad_fn(params))
            metric = natural_metric(params)
            if step < switch_iter:
                matrix = metric + damping * np.eye(n_params)
            else:
                hdiag = diagonal_hessian(objective, params, shape, eps=hess_eps, max_params=hessian_max_params)
                matrix = alpha * metric + (1.0 - alpha) * np.diag(hdiag) + damping * np.eye(n_params)
            direction = solve_direction(matrix, grad_flat)
            params, value, _ = line_search(objective, params, shape, direction, grad_flat, values[-1], lr_natural)
            values.append(float(value))
            report_progress("Hybrid QNG->QIN", step + 1, values, start)
            if abs(values[-2] - values[-1]) < tol:
                break
        return collect_result("Hybrid QNG->QIN", values, time.perf_counter() - start, metrics)

    runners = [
    ("SGD", sgd),
    ("Gradient Descent", gradient_descent),
    ("Adam", adam),
    #("L-BFGS", lbfgs),
    ("QNG", qng),
    ("QIN", qin),
    ("AQIN", aqin),
    ("Adaptive AQIN", adaptive_aqin_golden),
    ("QNG+LBFGS", qlbfgs),
    ("Adaptive QNG+LBFGS", adaptive_qlbfgs),
    ("Hybrid QNG->QIN", hybrid),
   ]
    results = []
    for display_name, runner in runners:
        if verbose:
            print(f"Starting {display_name}...", flush=True)
        result = runner()
        results.append(result)
        if verbose:
            print(
                f"Completed {result.name}: best = {result.best_reported:.8f}, "
                f"time = {result.elapsed:.2f} s, iterations = {result.iterations}",
                flush=True,
            )
    return results


def exact_ground_energy(hamiltonian, n_qubits: int) -> float:
    matrix = qml.matrix(hamiltonian, wire_order=range(n_qubits))
    return float(np.min(np.linalg.eigvalsh(matrix)))


def hartree_fock_energy(hamiltonian, hf_state: Array, n_qubits: int) -> float:
    matrix = qml.matrix(hamiltonian, wire_order=range(n_qubits))
    state = np.asarray(hf_state)
    bits = [int(round(float(np.real(bit)))) for bit in state]
    index = int("".join(str(bit) for bit in bits), 2)
    basis = np.zeros(2**n_qubits, dtype=complex)
    basis[index] = 1.0
    return float(np.real(np.vdot(basis, matrix @ basis)))


def print_summary(title: str, results: list[BenchmarkResult], score_label: str = "Score %") -> None:
    print(f"\n{title}")
    header = (
        f"{'Optimizer':<18} {'Best':>14} {'Abs err':>12} {'Rel err':>12} "
        f"{score_label:>10} {'Time (s)':>10} {'Iters':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<18} {r.best_reported:>14.8f} {r.abs_error:>12.3e} "
            f"{r.rel_error:>12.3e} {r.score:>10.2f} {r.elapsed:>10.2f} {r.iterations:>8d}"
        )


def plot_results(
    title: str,
    results: list[BenchmarkResult],
    outpath: Path,
    ylabel: str = "Objective",
    score_label: str = "Correlation recovered (%)",
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    for r in results:
        ax.plot(r.values, label=r.name, linewidth=2)
    ax.set_title("Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r in results:
        ax.semilogy(np.maximum(r.errors, 1e-12), label=r.name, linewidth=2)
    ax.set_title("Absolute Error")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")

    ax = axes[1, 0]
    for r in results:
        ax.plot(r.scores, label=r.name, linewidth=2)
    ax.set_title(score_label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(score_label)

    ax = axes[1, 1]
    ax.bar([r.name for r in results], [r.elapsed for r in results])
    ax.set_title("Wall-Clock Time")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle(title, fontsize=14)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def print_recommendations(results: list[BenchmarkResult], problem: str) -> None:
    best = min(results, key=lambda r: (r.abs_error, r.elapsed))
    fastest = min(results, key=lambda r: r.elapsed)
    print("\nConclusions and recommendations")
    print(
        f"For {problem}, {best.name} delivered the strongest final objective "
        f"({best.best_reported:.8f}) with absolute error {best.abs_error:.3e}."
    )
    print(
        f"{fastest.name} was fastest at {fastest.elapsed:.2f} s, so it is the best baseline "
        "when quick screening matters."
    )
    print(
        "QIN, AQIN, and Hybrid QNG->QIN are most useful as refinement methods. QIN adds signed "
        "finite-difference curvature; AQIN uses squared curvature magnitude for a more stable "
        "positive-semidefinite curvature contribution."
    )
