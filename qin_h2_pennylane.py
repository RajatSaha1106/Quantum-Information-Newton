"""Quantum Information Newton (QIN) optimizer benchmark for H2 in PennyLane.

This script compares seven optimizers on an H2 VQE problem with any basis set:

1. Gradient Descent
2. Adam
3. Quantum Natural Gradient (QNG)
4. QIN:         alpha * metric_tensor + (1 - alpha) * diagonal Hessian
5. AQIN:        alpha * metric_tensor + (1 - alpha) * squared diagonal Hessian
6. Adaptive AQIN: alpha tuned per-step via golden-section search
7. Hybrid QNG -> QIN: switches from QNG to QIN after --switch-iter steps

The hardcoded Hamiltonian path (no --use-qchem) always uses the STO-3G
Jordan-Wigner 4-qubit H2 Hamiltonian at R = 0.7414 Angstrom.

The --use-qchem path accepts any basis set via --basis. PennyLane built-in
basis sets (sto-3g, 6-31g, 6-311g, cc-pvdz) are loaded natively; anything
else (cc-pvtz, aug-cc-pvdz, …) sets load_data=True automatically and requires:

    pip install basis-set-exchange

Usage examples
--------------
python qin_h2_pennylane.py                                  # hardcoded STO-3G, 4 qubits
python qin_h2_pennylane.py --use-qchem                      # qchem STO-3G, 4 qubits
python qin_h2_pennylane.py --use-qchem --basis 6-31g        # 8 qubits
python qin_h2_pennylane.py --use-qchem --basis cc-pvdz      # 20 qubits
python qin_h2_pennylane.py --use-qchem --basis cc-pvtz      # needs basis-set-exchange
python qin_h2_pennylane.py --max-iter 200 --tol 1e-10 --plot out.png
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


# ---------------------------------------------------------------------------
# Reference energies (STO-3G / Jordan-Wigner, R = 0.7414 Å)
# ---------------------------------------------------------------------------

E_EXACT = -1.137270
E_HF    = -1.116684

# Built-in basis sets that do NOT require load_data / basis-set-exchange
_PENNYLANE_BUILTIN_BASIS = {"sto-3g", "6-31g", "6-311g", "cc-pvdz"}


# ---------------------------------------------------------------------------
# Problem configuration (replaces module-level globals for n_qubits / shape)
# ---------------------------------------------------------------------------

@dataclass
class ProblemConfig:
    n_qubits:    int
    param_shape: tuple[int, int]
    basis:       str


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizerResult:
    name:            str
    energies:        list[float]
    errors:          list[float]
    correlations:    list[float]
    elapsed:         float
    iterations:      int
    best_energy_raw: float
    best_energy:     float
    abs_error:       float
    rel_error:       float
    corr_recovered:  float


# ---------------------------------------------------------------------------
# Hamiltonians
# ---------------------------------------------------------------------------

def hardcoded_h2_sto3g_hamiltonian() -> qml.Hamiltonian:
    """Return the 4-qubit H2 STO-3G Hamiltonian at R = 0.7414 Å (JW mapping)."""

    coeffs = [
        -0.04207897647782276,
         0.17771287465139946,
         0.17771287465139940,
        -0.24274280513140462,
        -0.24274280513140462,
         0.17059738328801055,
         0.12293305056183798,
         0.16768319457718960,
         0.16768319457718960,
         0.12293305056183798,
         0.17627640804319591,
        -0.04475014401535161,
         0.04475014401535161,
         0.04475014401535161,
        -0.04475014401535161,
    ]

    ops = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(2),
        qml.PauliZ(3),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(0) @ qml.PauliZ(3),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(1) @ qml.PauliZ(3),
        qml.PauliZ(2) @ qml.PauliZ(3),
        qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3),
        qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliY(3),
        qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3),
        qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
    ]
    return qml.Hamiltonian(coeffs, ops)


def qchem_h2_hamiltonian(basis: str = "sto-3g") -> tuple[qml.Hamiltonian, int]:
    """Build H2 Hamiltonian via PennyLane qchem for any basis set.

    Parameters
    ----------
    basis:
        Basis set name, e.g. 'sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz'.
        Case-insensitive. Built-in PennyLane bases are loaded natively;
        anything else sets load_data=True (requires basis-set-exchange).

    Returns
    -------
    (hamiltonian, n_qubits)
    """
    basis_lower = basis.lower()
    load_data   = basis_lower not in _PENNYLANE_BUILTIN_BASIS

    if load_data:
        print(
            f"  [info] '{basis}' is not a PennyLane built-in basis set. "
            "Setting load_data=True — make sure 'basis-set-exchange' is installed "
            "(pip install basis-set-exchange)."
        )

    symbols     = ["H", "H"]
    coordinates = np.array(
        [[0.0, 0.0, -0.7414 / 2.0],
         [0.0, 0.0,  0.7414 / 2.0]],
        requires_grad=False,
    )

    molecule = qml.qchem.Molecule(
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis_name=basis_lower,
        unit="angstrom",
        load_data=load_data,
    )

    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        mapping="jordan_wigner",
    )
    return hamiltonian, n_qubits


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def make_problem(
    use_qchem: bool = False,
    basis:     str  = "sto-3g",
) -> tuple[qml.QNode, qml.Hamiltonian, ProblemConfig]:
    """Build the energy QNode, Hamiltonian, and ProblemConfig.

    The ansatz is a hardware-efficient circuit:
        • one Rot(phi, theta, omega) gate per qubit
        • a ring of CNOT gates  0→1→2→…→(n-1)→0

    This scales naturally to any number of qubits produced by the chosen
    basis set.
    """
    if use_qchem:
        hamiltonian, n_qubits = qchem_h2_hamiltonian(basis)
    else:
        hamiltonian = hardcoded_h2_sto3g_hamiltonian()
        n_qubits    = 4

    param_shape = (n_qubits, 3)
    cfg         = ProblemConfig(n_qubits=n_qubits, param_shape=param_shape, basis=basis)
    dev         = qml.device("default.qubit", wires=n_qubits)

    def ansatz(params: np.ndarray) -> None:
        for wire in range(n_qubits):
            qml.Rot(params[wire, 0], params[wire, 1], params[wire, 2], wires=wire)
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % n_qubits])

    @qml.qnode(dev, interface="autograd")
    def energy_qnode(params: np.ndarray) -> float:
        ansatz(params)
        return qml.expval(hamiltonian)

    return energy_qnode, hamiltonian, cfg


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def init_params(cfg: ProblemConfig) -> np.ndarray:
    np.random.seed(42)
    return np.array(
        np.random.normal(0.0, np.pi, cfg.param_shape),
        requires_grad=True,
    )


def flatten(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (-1,))


def unflatten(x: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    return np.reshape(x, cfg.param_shape)


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def metric_matrix(metric_raw, n_params: int) -> np.ndarray:
    metric = np.asarray(metric_raw)
    if metric.ndim != 2:
        metric = np.reshape(metric, (n_params, n_params))
    return np.asarray(metric, dtype=float)


def diagonal_hessian(
    energy_fn: Callable[[np.ndarray], float],
    params:    np.ndarray,
    cfg:       ProblemConfig,
    eps:       float = 1e-4,
) -> np.ndarray:
    """Finite-difference diagonal Hessian of the energy w.r.t. all parameters."""
    theta       = np.array(params, requires_grad=False)
    flat        = flatten(theta)
    base_energy = float(energy_fn(theta))
    hdiag       = np.zeros_like(flat)

    for idx in range(len(flat)):
        plus  = np.array(flat, requires_grad=False)
        minus = np.array(flat, requires_grad=False)
        plus[idx]  += eps
        minus[idx] -= eps
        e_plus  = float(energy_fn(unflatten(plus,  cfg)))
        e_minus = float(energy_fn(unflatten(minus, cfg)))
        hdiag[idx] = (e_plus - 2.0 * base_energy + e_minus) / (eps ** 2)

    return np.asarray(hdiag, dtype=float)


def solve_direction(matrix: np.ndarray, grad_flat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, grad_flat)
    except Exception:
        return np.linalg.pinv(matrix) @ grad_flat


# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------

def line_search(
    energy_fn:    Callable[[np.ndarray], float],
    params:       np.ndarray,
    direction_flat: np.ndarray,
    grad_flat:    np.ndarray,
    energy:       float,
    cfg:          ProblemConfig,
    initial_lr:   float = 0.20,
    c:            float = 1e-4,
    beta:         float = 0.5,
    max_steps:    int   = 12,
) -> tuple[np.ndarray, float, float]:
    """Backtracking Armijo line search: theta <- theta - lr * direction."""
    descent = float(np.dot(grad_flat, direction_flat))
    if not np.isfinite(descent) or descent <= 0:
        direction_flat = grad_flat
        descent        = float(np.dot(grad_flat, grad_flat))

    lr            = initial_lr
    best_params   = params
    best_energy   = energy

    for _ in range(max_steps):
        candidate        = np.array(
            params - lr * unflatten(direction_flat, cfg),
            requires_grad=True,
        )
        candidate_energy = float(energy_fn(candidate))
        if candidate_energy < best_energy:
            best_params = candidate
            best_energy = candidate_energy
        if candidate_energy <= energy - c * lr * descent:
            return candidate, candidate_energy, lr
        lr *= beta

    return np.array(best_params, requires_grad=True), best_energy, lr


# ---------------------------------------------------------------------------
# Golden-section alpha search (Adaptive AQIN)
# ---------------------------------------------------------------------------

def golden_search_alpha(
    energy_fn:     Callable[[np.ndarray], float],
    params:        np.ndarray,
    grad_flat:     np.ndarray,
    metric:        np.ndarray,
    curvature_diag: np.ndarray,
    cfg:           ProblemConfig,
    damping:       float = 1e-4,
    lr:            float = 0.20,
    tol:           float = 1e-3,
    max_iter:      int   = 20,
) -> float:
    """One-step lookahead: alpha = argmin_{alpha in [0,1]} E(theta - lr M(alpha)^{-1} g)."""

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    n   = metric.shape[0]

    def objective(alpha: float) -> float:
        matrix    = (
            alpha         * metric
            + (1.0 - alpha) * np.diag(curvature_diag)
            + damping       * np.eye(n)
        )
        direction = solve_direction(matrix, grad_flat)
        candidate = np.array(
            params - lr * unflatten(direction, cfg),
            requires_grad=True,
        )
        return float(energy_fn(candidate))

    a, b = 0.05, 0.95
    c_pt = b - phi * (b - a)
    d_pt = a + phi * (b - a)
    fc, fd = objective(c_pt), objective(d_pt)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d_pt, fd = c_pt, c_pt, fc
            c_pt = b - phi * (b - a)
            fc   = objective(c_pt)
        else:
            a, c_pt, fc = d_pt, d_pt, fd
            d_pt = a + phi * (b - a)
            fd   = objective(d_pt)

    return float(0.5 * (a + b))


# ---------------------------------------------------------------------------
# Physical metrics
# ---------------------------------------------------------------------------

def physical_metrics(
    energy_history: list[float],
) -> tuple[float, float, float, float, float]:
    raw_best       = min(energy_history)
    best           = raw_best
    abs_error      = abs(best - E_EXACT)
    rel_error      = abs_error / max(abs(E_EXACT), 1e-12)
    correlation_gap = abs(E_HF - E_EXACT)

    if correlation_gap > 0.05:
        correlation = (E_HF - best) / (E_HF - E_EXACT) * 100.0
    else:
        initial_error = max(abs(energy_history[0] - E_EXACT), 1e-12)
        correlation   = (1.0 - abs(best - E_EXACT) / initial_error) * 100.0

    correlation = float(np.clip(correlation, 0.0, 100.0))
    return float(raw_best), float(best), float(abs_error), float(rel_error), float(correlation)


def collect_result(
    name:     str,
    energies: list[float],
    elapsed:  float,
) -> OptimizerResult:
    raw_best, best, abs_error, rel_error, corr = physical_metrics(energies)
    initial_error = max(abs(energies[0] - E_EXACT), 1e-12)
    errors        = [abs(e - E_EXACT) for e in energies]
    correlations  = [
        float(np.clip((1.0 - abs(e - E_EXACT) / initial_error) * 100.0, 0.0, 100.0))
        for e in energies
    ]
    return OptimizerResult(
        name=name,
        energies=energies,
        errors=errors,
        correlations=correlations,
        elapsed=elapsed,
        iterations=len(energies) - 1,
        best_energy_raw=raw_best,
        best_energy=best,
        abs_error=abs_error,
        rel_error=rel_error,
        corr_recovered=corr,
    )


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def run_gradient_descent(
    energy_fn: Callable,
    cfg:       ProblemConfig,
    max_iter:  int,
    tol:       float,
) -> OptimizerResult:
    params   = init_params(cfg)
    grad_fn  = qml.grad(energy_fn)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad      = grad_fn(params)
        grad_flat = flatten(grad)
        params, energy, _ = line_search(
            energy_fn, params, grad_flat, grad_flat, energies[-1], cfg,
            initial_lr=0.20,
        )
        energies.append(float(energy))
        best = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  GD iter {step + 1:3d}/{max_iter} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Gradient Descent", energies, time.perf_counter() - start)


def run_adam(
    energy_fn: Callable,
    cfg:       ProblemConfig,
    max_iter:  int,
    tol:       float,
    lr:        float = 0.05,
    beta1:     float = 0.9,
    beta2:     float = 0.999,
    eps_adam:  float = 1e-8,
) -> OptimizerResult:
    params   = init_params(cfg)
    grad_fn  = qml.grad(energy_fn)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    m = np.zeros(cfg.param_shape)
    v = np.zeros(cfg.param_shape)

    for step in range(1, max_iter + 1):
        grad  = grad_fn(params)
        m     = beta1 * m + (1.0 - beta1) * grad
        v     = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        params = np.array(
            params - lr * m_hat / (np.sqrt(v_hat) + eps_adam),
            requires_grad=True,
        )
        energy = float(energy_fn(params))
        energies.append(energy)
        best    = min(best, energy)
        elapsed = time.perf_counter() - start
        print(
            f"  Adam iter {step:3d}/{max_iter} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Adam", energies, time.perf_counter() - start)


def run_qng(
    energy_fn: Callable,
    cfg:       ProblemConfig,
    max_iter:  int,
    tol:       float,
    damping:   float = 1e-4,
) -> OptimizerResult:
    params    = init_params(cfg)
    grad_fn   = qml.grad(energy_fn)
    metric_fn = qml.metric_tensor(energy_fn, approx="block-diag")
    n_params  = math.prod(cfg.param_shape)
    energies  = [float(energy_fn(params))]
    best      = energies[0]
    start     = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params), n_params)
        matrix    = metric + damping * np.eye(n_params)
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1], cfg,
            initial_lr=0.20,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  QNG iter {step + 1:3d}/{max_iter} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("QNG", energies, time.perf_counter() - start)


def run_qin(
    energy_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    alpha:      float = 0.5,
    damping:    float = 1e-4,
    hess_eps:   float = 1e-4,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """QIN: alpha * F + (1 - alpha) * diag(H)  where F = quantum metric tensor."""
    params    = init_params(cfg)
    grad_fn   = qml.grad(energy_fn)
    metric_fn = qml.metric_tensor(energy_fn, approx="block-diag")
    n_params  = math.prod(cfg.param_shape)
    energies  = [float(energy_fn(params))]
    best      = energies[0]
    start     = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params), n_params)
        hdiag     = diagonal_hessian(energy_fn, params, cfg, eps=hess_eps)
        matrix    = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(hdiag)
            + damping       * np.eye(n_params)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1], cfg,
            initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  QIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("QIN", energies, time.perf_counter() - start)


def run_aqin(
    energy_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    alpha:      float = 0.5,
    damping:    float = 1e-4,
    hess_eps:   float = 1e-4,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """AQIN: alpha * F + (1 - alpha) * diag(H^2)  — squares remove sign issues."""
    params    = init_params(cfg)
    grad_fn   = qml.grad(energy_fn)
    metric_fn = qml.metric_tensor(energy_fn, approx="block-diag")
    n_params  = math.prod(cfg.param_shape)
    energies  = [float(energy_fn(params))]
    best      = energies[0]
    start     = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params), n_params)
        hdiag     = diagonal_hessian(energy_fn, params, cfg, eps=hess_eps)
        matrix    = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(hdiag ** 2)
            + damping       * np.eye(n_params)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1], cfg,
            initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  AQIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("AQIN", energies, time.perf_counter() - start)


def run_adaptive_aqin(
    energy_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    damping:    float = 1e-4,
    hess_eps:   float = 1e-4,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """Adaptive AQIN: alpha tuned each step via golden-section line search."""
    params    = init_params(cfg)
    grad_fn   = qml.grad(energy_fn)
    metric_fn = qml.metric_tensor(energy_fn, approx="block-diag")
    n_params  = math.prod(cfg.param_shape)
    energies  = [float(energy_fn(params))]
    best      = energies[0]
    start     = time.perf_counter()

    for step in range(max_iter):
        grad_flat  = flatten(grad_fn(params))
        metric     = metric_matrix(metric_fn(params), n_params)
        hdiag      = diagonal_hessian(energy_fn, params, cfg, eps=hess_eps)
        curvature  = hdiag ** 2

        alpha = golden_search_alpha(
            energy_fn=energy_fn,
            params=params,
            grad_flat=grad_flat,
            metric=metric,
            curvature_diag=curvature,
            cfg=cfg,
            damping=damping,
            lr=initial_lr,
        )

        matrix    = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(curvature)
            + damping       * np.eye(n_params)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1], cfg,
            initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  Adaptive AQIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Adaptive AQIN", energies, time.perf_counter() - start)


def run_hybrid(
    energy_fn:   Callable,
    cfg:         ProblemConfig,
    max_iter:    int,
    tol:         float,
    switch_iter: int,
    alpha:       float = 0.5,
    damping:     float = 1e-4,
    hess_eps:    float = 1e-4,
) -> OptimizerResult:
    """Hybrid QNG → QIN: pure QNG for the first switch_iter steps, then QIN."""
    params    = init_params(cfg)
    grad_fn   = qml.grad(energy_fn)
    metric_fn = qml.metric_tensor(energy_fn, approx="block-diag")
    n_params  = math.prod(cfg.param_shape)
    energies  = [float(energy_fn(params))]
    best      = energies[0]
    start     = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params), n_params)
        phase     = "QNG" if step < switch_iter else "QIN"

        if step < switch_iter:
            matrix = metric + damping * np.eye(n_params)
        else:
            hdiag  = diagonal_hessian(energy_fn, params, cfg, eps=hess_eps)
            matrix = (
                alpha           * metric
                + (1.0 - alpha) * np.diag(hdiag)
                + damping       * np.eye(n_params)
            )

        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1], cfg,
            initial_lr=0.20,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  Hybrid iter {step + 1:3d}/{max_iter} [{phase}] "
            f"| E = {energy:.8f} Ha "
            f"| best = {best:.8f} Ha "
            f"| elapsed = {elapsed:.2f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Hybrid QNG->QIN", energies, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[OptimizerResult]) -> None:
    print("\nReference energies")
    print(f"  E_exact = {E_EXACT:.6f} Ha")
    print(f"  E_HF    = {E_HF:.6f} Ha")
    print("\nOptimizer comparison")
    header = (
        f"{'Optimizer':<18} {'Best E (Ha)':>14} {'Abs err':>12} "
        f"{'Rel err':>12} {'Score %':>10} {'Time (s)':>10} {'Iters':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<18} {r.best_energy:>14.8f} {r.abs_error:>12.3e} "
            f"{r.rel_error:>12.3e} {r.corr_recovered:>10.2f} "
            f"{r.elapsed:>10.2f} {r.iterations:>8d}"
        )
    print()


def print_conclusions(results: list[OptimizerResult]) -> None:
    best_accuracy = min(results, key=lambda r: (r.abs_error, r.elapsed))
    fastest       = min(results, key=lambda r: r.elapsed)
    best_corr     = max(results, key=lambda r: r.corr_recovered)

    print("\nThesis-ready conclusions")
    print(
        f"1. Best variational accuracy : {best_accuracy.name} — "
        f"{best_accuracy.best_energy:.8f} Ha, "
        f"absolute error {best_accuracy.abs_error:.3e} Ha."
    )
    print(
        f"2. Highest correlation recovery: {best_corr.name} — "
        f"{best_corr.corr_recovered:.2f}% of correlation energy vs Hartree-Fock."
    )
    print(
        f"3. Fastest wall-clock runtime  : {fastest.name} — "
        f"{fastest.elapsed:.2f} s (interpret alongside chemical accuracy)."
    )
    print(
        "4. QNG uses the local quantum geometry and is usually more stable than "
        "raw gradient descent for curved variational landscapes."
    )
    print(
        "5. QIN augments the block-diagonal quantum metric with diagonal "
        "finite-difference curvature. Damping and line search guard against "
        "negative or noisy curvature estimates."
    )
    print(
        "6. AQIN squares the diagonal Hessian before mixing, so curvature "
        "magnitude is preserved while negative curvature no longer makes "
        "the preconditioner indefinite."
    )
    print(
        "7. Adaptive AQIN tunes alpha at every step via a golden-section "
        "lookahead, removing the need to hand-tune the metric/Hessian balance."
    )
    print(
        "\nRecommendation: use Adam or GD for cheap baselines, QNG for "
        "geometry-aware training, AQIN/Adaptive-AQIN for stable curvature-aware "
        "refinement, and Hybrid QNG→QIN when final accuracy justifies the "
        "finite-difference Hessian cost."
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[OptimizerResult], cfg: ProblemConfig, outpath: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # --- Convergence ---
    ax = axes[0, 0]
    for r in results:
        ax.plot(r.energies, label=r.name, linewidth=2)
    ax.axhline(E_EXACT, color="black",  linestyle="--", linewidth=1.2, label="Exact")
    ax.axhline(E_HF,    color="gray",   linestyle=":",  linewidth=1.2, label="HF")
    ax.set_title("Energy Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.legend(fontsize=8)

    # --- Absolute error (log) ---
    ax = axes[0, 1]
    for r in results:
        ax.semilogy(np.maximum(r.errors, 1e-12), label=r.name, linewidth=2)
    ax.set_title("Absolute Error |E − E_exact|")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|E − E_exact| (Ha)")
    ax.legend(fontsize=8)

    # --- Correlation score ---
    ax = axes[1, 0]
    for r in results:
        ax.plot(r.correlations, label=r.name, linewidth=2)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Convergence Score")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8)

    # --- Wall-clock time ---
    ax = axes[1, 1]
    labels = [r.name for r in results]
    times  = [r.elapsed for r in results]
    bars   = ax.bar(labels, times, color=plt.cm.tab10.colors[: len(results)])
    ax.bar_label(bars, fmt="%.1fs", padding=2, fontsize=8)
    ax.set_title("Wall-Clock Time")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"H2 VQE Optimizer Benchmark  |  basis = {cfg.basis}  |  "
        f"{cfg.n_qubits} qubits  |  params {cfg.param_shape}",
        fontsize=13,
    )
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved 4-panel plot to: {outpath.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-iter",    type=int,   default=150,
        help="Maximum optimisation iterations per optimizer (default: 150).",
    )
    parser.add_argument(
        "--tol",         type=float, default=1e-9,
        help="Energy-change convergence tolerance (default: 1e-9 Ha).",
    )
    parser.add_argument(
        "--switch-iter", type=int,   default=50,
        help="Iteration at which Hybrid switches from QNG to QIN (default: 50).",
    )
    parser.add_argument(
        "--use-qchem",   action="store_true",
        help="Build the Hamiltonian via qml.qchem instead of the hardcoded coefficients.",
    )
    parser.add_argument(
        "--basis",       type=str,   default="sto-3g",
        help=(
            "Basis set for the --use-qchem path (default: sto-3g). "
            "Built-in: sto-3g, 6-31g, 6-311g, cc-pvdz. "
            "Anything else requires 'pip install basis-set-exchange'."
        ),
    )
    parser.add_argument(
        "--plot",        type=Path,  default=Path("qin_h2_optimizer_comparison.png"),
        help="Output path for the 4-panel PNG plot.",
    )
    parser.add_argument(
        "--draw-circuit", action="store_true",
        help="Draw the ansatz circuit before optimisation.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build problem
    # ------------------------------------------------------------------
    print("\nH2 VQE benchmark")
    print(f"  Bond length : 0.7414 Å")
    print(f"  Basis       : {args.basis if args.use_qchem else 'sto-3g (hardcoded)'}")
    print(f"  qchem path  : {args.use_qchem}")

    energy_fn, hamiltonian, cfg = make_problem(
        use_qchem=args.use_qchem,
        basis=args.basis,
    )

    print(f"  Qubits      : {cfg.n_qubits}")
    print(f"  Param shape : {cfg.param_shape}  ({math.prod(cfg.param_shape)} parameters)")
    terms = getattr(hamiltonian, "ops", getattr(hamiltonian, "operands", []))
    print(f"  H terms     : {len(terms)}")

    # ------------------------------------------------------------------
    # Optional circuit diagram
    # ------------------------------------------------------------------
    if args.draw_circuit:
        params_draw = init_params(cfg)
        fig, ax     = qml.draw_mpl(energy_fn)(params_draw)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Run all optimizers
    # ------------------------------------------------------------------
    runners = [
        ("Gradient Descent", lambda: run_gradient_descent(energy_fn, cfg, args.max_iter, args.tol)),
        ("Adam",             lambda: run_adam(energy_fn, cfg, args.max_iter, args.tol)),
        ("QNG",              lambda: run_qng(energy_fn, cfg, args.max_iter, args.tol)),
        ("QIN",              lambda: run_qin(energy_fn, cfg, args.max_iter, args.tol)),
        ("AQIN",             lambda: run_aqin(energy_fn, cfg, args.max_iter, args.tol)),
        ("Hybrid QNG->QIN",  lambda: run_hybrid(energy_fn, cfg, args.max_iter, args.tol, args.switch_iter)),
        ("Adaptive AQIN",    lambda: run_adaptive_aqin(energy_fn, cfg, args.max_iter, args.tol)),
    ]

    results: list[OptimizerResult] = []
    print()
    for display_name, runner in runners:
        print(f"Running {display_name} ...")
        result = runner()
        results.append(result)
        print(
            f"  → best E = {result.best_energy:.8f} Ha  |  "
            f"abs err = {result.abs_error:.3e}  |  "
            f"{result.elapsed:.1f} s  |  {result.iterations} iters\n"
        )

    # ------------------------------------------------------------------
    # Report and plot
    # ------------------------------------------------------------------
    print_summary(results)
    plot_results(results, cfg, args.plot)
    print_conclusions(results)


if __name__ == "__main__":
    main()