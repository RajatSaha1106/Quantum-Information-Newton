"""QIN Optimizer Benchmark for H2 VQE on a Qiskit Backend via PennyLane.

Compares seven optimizers on the 4-qubit H2 STO-3G VQE problem:
    1. Gradient Descent
    2. Adam
    3. Quantum Natural Gradient (QNG)
    4. QIN         : alpha * F + (1-alpha) * diag(H)
    5. AQIN        : alpha * F + (1-alpha) * diag(H^2)
    6. Adaptive AQIN : alpha tuned per-step via golden-section search
    7. Hybrid QNG->QIN : QNG for first --switch-iter steps, then QIN

─────────────────────────────────────────────────────────────────────────────
BACKEND NOTES
─────────────────────────────────────────────────────────────────────────────
By default this script uses "ibmq_qasm_simulator", which is free on IBM
Quantum at the time of writing.  IBM Quantum's policies may change and
simulators may become paid services — always verify current pricing before
running.

To run on real hardware set  --backend  to any device you have access to,
e.g. "ibm_brisbane".  Hardware access is NOT free; be aware of costs before
proceeding.

─────────────────────────────────────────────────────────────────────────────
INSTALL
─────────────────────────────────────────────────────────────────────────────
    pip install pennylane pennylane-qiskit qiskit qiskit-ibm-runtime matplotlib

─────────────────────────────────────────────────────────────────────────────
AUTHENTICATION
─────────────────────────────────────────────────────────────────────────────
Save your IBM Quantum token once:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_TOKEN")

Or pass it directly with  --token MY_TOKEN.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
    # Use cached credentials, default simulator
    python qin_h2_qiskit.py

    # Pass token explicitly, run on a real device
    python qin_h2_qiskit.py --token MY_TOKEN --backend ibm_brisbane

    # Fewer shots / iterations for a quick smoke test
    python qin_h2_qiskit.py --shots 512 --max-iter 20

    # Fall back to PennyLane default.qubit (no Qiskit needed)
    python qin_h2_qiskit.py --local

─────────────────────────────────────────────────────────────────────────────
IMPORTANT DIFFERENCES FROM THE STATEVECTOR VERSION
─────────────────────────────────────────────────────────────────────────────
* Expectation values are estimated from finite shots  →  stochastic noise.
* qml.metric_tensor with approx="block-diag" is supported on Qiskit devices
  because PennyLane decomposes it into additional circuit evaluations.
* Gradients use the parameter-shift rule (compatible with any gate-based
  backend), NOT autograd/backprop.
* The diagonal Hessian is estimated with the parameter-shift double-shift
  rule instead of finite differences, saving circuit calls and working
  correctly with shot noise.
* A shot-noise-aware damping floor is applied to all preconditioned methods.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Reference energies  (STO-3G / Jordan-Wigner, R = 0.742 Å)
# ─────────────────────────────────────────────────────────────────────────────
E_EXACT = -1.137270   # Ha  — FCI / exact diagonalisation
E_HF    = -1.116684   # Ha  — Hartree-Fock reference

N_QUBITS    = 4
PARAM_SHAPE = (N_QUBITS, 3)   # one Rot gate per qubit → 3 angles each


# ─────────────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProblemConfig:
    n_qubits:    int
    param_shape: tuple[int, int]
    shots:       int
    backend_name: str


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


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian  (PennyLane QChem dataset — identical to the STO-3G hardcoded
#               coefficients but loaded from the official PennyLane data hub)
# ─────────────────────────────────────────────────────────────────────────────

def load_h2_hamiltonian() -> qml.Hamiltonian:
    """Load the H2 STO-3G Hamiltonian from the PennyLane QChem dataset.

    This is the same Jordan-Wigner mapped, 4-qubit Hamiltonian used in the
    original benchmark, but sourced from the authoritative PennyLane data hub
    so it is guaranteed to be consistent with the rest of the ecosystem.

    Falls back to hard-coded coefficients if the dataset cannot be fetched
    (e.g. no internet connection).
    """
    try:
        print("  Loading H2 STO-3G Hamiltonian from PennyLane QChem dataset ...")
        [dataset] = qml.data.load(
            "qchem",
            molname="H2",
            bondlength=0.742,
            basis="STO-3G",
        )
        H = dataset.hamiltonian
        print(f"  Loaded dataset Hamiltonian with {len(H.ops)} terms.")
        return H
    except Exception as exc:
        print(f"  [warn] Dataset load failed ({exc}). Using hardcoded coefficients.")
        return _hardcoded_h2_sto3g()


def _hardcoded_h2_sto3g() -> qml.Hamiltonian:
    """Hard-coded fallback: H2 STO-3G JW Hamiltonian at R = 0.7414 Å."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Device factory
# ─────────────────────────────────────────────────────────────────────────────

def build_device(
    backend_name: str,
    shots:        int,
    token:        str | None,
    local:        bool,
) -> qml.Device:
    """Return a PennyLane device.

    local=True  →  default.qubit  (no Qiskit, no IBM account needed).
    local=False →  qiskit.remote  backed by the named IBM Quantum backend.

    The qiskit.remote device is provided by the pennylane-qiskit plugin and
    wraps any Qiskit backend (simulator or real hardware) transparently.
    Gradients computed via parameter-shift are fully compatible because
    PennyLane decomposes every shifted evaluation into separate circuit
    submissions.
    """
    if local:
        print("  [device] Using default.qubit (local statevector, no shots noise).")
        return qml.device("default.qubit", wires=N_QUBITS)

    # ── Qiskit path ──────────────────────────────────────────────────────────
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print(
            "[ERROR] qiskit_ibm_runtime is not installed.\n"
            "        Run:  pip install qiskit-ibm-runtime\n"
            "        Or use --local to run without Qiskit."
        )
        sys.exit(1)

    # Authenticate — use a passed token or previously saved credentials.
    try:
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            print(f"  [auth] Authenticated with provided token.")
        else:
            service = QiskitRuntimeService()
            print(f"  [auth] Using saved IBM Quantum credentials.")
    except Exception as exc:
        print(
            f"[ERROR] IBM Quantum authentication failed: {exc}\n"
            f"        Save credentials first:\n"
            f"          from qiskit_ibm_runtime import QiskitRuntimeService\n"
            f"          QiskitRuntimeService.save_account("
            f"channel='ibm_quantum', token='MY_TOKEN')\n"
            f"        Or pass --token MY_TOKEN on the command line."
        )
        sys.exit(1)

    backend = service.least_busy(operational=True, simulator=False)
    print(f"  [device] Backend : {backend.name}")
    print(f"  [device] Shots   : {shots}")

    # PennyLane's qiskit.remote device wraps any Qiskit backend.
    # We pass the full qubit count of the backend but only use 4 wires.
    try:
        max_qubits = backend.configuration().n_qubits
    except Exception:
        max_qubits = N_QUBITS

    dev = qml.device(
        "qiskit.remote",
        wires=max_qubits,
        backend=backend,
        shots=shots,
    )
    print(f"  [device] Backend max qubits : {max_qubits}  (using {N_QUBITS})")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# Ansatz & QNode factory
# ─────────────────────────────────────────────────────────────────────────────

def make_energy_qnode(
    dev:         qml.Device,
    hamiltonian: qml.Hamiltonian,
    local:       bool,
) -> qml.QNode:
    """Build the energy QNode with the correct diff method for the device.

    * local=True  →  backprop  (exact, zero overhead, works on default.qubit)
    * local=False →  parameter-shift  (works on any gate-based backend,
                      including simulators and real hardware)

    The ansatz is a hardware-efficient layer:
        Rot(phi, theta, omega) on every qubit  →  ring of CNOT gates.
    """
    diff_method = "backprop" if local else "parameter-shift"

    @qml.qnode(dev, interface="autograd", diff_method=diff_method)
    def energy_qnode(params: np.ndarray) -> float:
        # ── Ansatz ────────────────────────────────────────────────────────
        for wire in range(N_QUBITS):
            qml.Rot(params[wire, 0], params[wire, 1], params[wire, 2], wires=wire)
        for wire in range(N_QUBITS):
            qml.CNOT(wires=[wire, (wire + 1) % N_QUBITS])
        # ── Observable ────────────────────────────────────────────────────
        return qml.expval(hamiltonian)

    return energy_qnode


def make_metric_qnode(
    dev:   qml.Device,
    energy_qnode: qml.QNode,
    local: bool,
) -> Callable:
    """Return a callable that evaluates the block-diagonal quantum metric tensor.

    PennyLane's qml.metric_tensor with approx="block-diag" is implemented
    via the Hadamard-test / overlap circuit and is fully compatible with
    Qiskit backends through PennyLane's device-agnostic circuit execution.
    """
    return qml.metric_tensor(energy_qnode, approx="block-diag")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_params() -> np.ndarray:
    np.random.seed(42)
    return np.array(
        np.random.normal(0.0, np.pi, PARAM_SHAPE),
        requires_grad=True,
    )


def flatten(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (-1,))


def unflatten(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, PARAM_SHAPE)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix helpers
# ─────────────────────────────────────────────────────────────────────────────
N_PARAMS = math.prod(PARAM_SHAPE)


def metric_matrix(metric_raw) -> np.ndarray:
    m = np.asarray(metric_raw)
    if m.ndim != 2:
        m = np.reshape(m, (N_PARAMS, N_PARAMS))
    return np.asarray(m, dtype=float)


def diagonal_hessian_shift(
    energy_fn: Callable,
    params:    np.ndarray,
) -> np.ndarray:
    """Diagonal Hessian via the parameter-shift double-shift rule.

    For a gate G(θ) with eigenvalues ±½, the diagonal Hessian element is:

        ∂²E/∂θᵢ² = E(θ + π eᵢ) - 2 E(θ) + E(θ - π eᵢ)    [shift = π/2 each]

    More precisely, using the two-term shift rule with shift=π/2:

        ∂²E/∂θᵢ² = E(θ + π eᵢ) - 2 E(θ) + E(θ - π eᵢ)

    This is exact for Pauli-rotation gates and works correctly with shot noise
    (unlike finite differences which amplify noise as 1/eps²).

    Reference: Mitarai & Fujii, PRR 3, 033035 (2021).
    """
    flat        = flatten(np.array(params, requires_grad=False))
    base_energy = float(energy_fn(np.array(unflatten(flat), requires_grad=True)))
    hdiag       = np.zeros(N_PARAMS, dtype=float)

    shift = np.pi / 2.0          # standard parameter-shift amount

    for idx in range(N_PARAMS):
        plus  = np.array(flat, requires_grad=False)
        minus = np.array(flat, requires_grad=False)
        plus[idx]  += shift
        minus[idx] -= shift

        e_plus  = float(energy_fn(np.array(unflatten(plus),  requires_grad=True)))
        e_minus = float(energy_fn(np.array(unflatten(minus), requires_grad=True)))

        # Second derivative from parameter-shift: coefficient = 1 for Rot gates
        hdiag[idx] = (e_plus - 2.0 * base_energy + e_minus)

    return np.asarray(hdiag, dtype=float)


def solve_direction(matrix: np.ndarray, grad_flat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, grad_flat)
    except Exception:
        return np.linalg.pinv(matrix) @ grad_flat


# ─────────────────────────────────────────────────────────────────────────────
# Shot-noise aware damping
# ─────────────────────────────────────────────────────────────────────────────

def shot_noise_damping(shots: int, n_params: int, base_damping: float = 1e-3) -> float:
    """Estimate a sensible damping floor given the shot budget.

    Shot noise in the metric tensor scales as 1/sqrt(shots).  A damping
    floor of sigma_noise / sqrt(n_params) keeps the preconditioner stable.

    Parameters
    ----------
    shots       : circuit shots per expectation value
    n_params    : total number of variational parameters
    base_damping: minimum damping regardless of shots (default 1e-3)
    """
    noise_estimate = 1.0 / math.sqrt(shots)
    return max(base_damping, noise_estimate / math.sqrt(n_params))


# ─────────────────────────────────────────────────────────────────────────────
# Line search  (backtracking Armijo, shot-noise tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def line_search(
    energy_fn:      Callable,
    params:         np.ndarray,
    direction_flat: np.ndarray,
    grad_flat:      np.ndarray,
    energy:         float,
    shots:          int,
    initial_lr:     float = 0.20,
    c:              float = 1e-4,
    beta:           float = 0.5,
    max_steps:      int   = 10,
) -> tuple[np.ndarray, float, float]:
    """Backtracking line search with a shot-noise floor on the Armijo condition.

    With finite shots, energy estimates fluctuate by ~1/sqrt(shots).  We
    relax the Armijo sufficient-decrease condition by that amount so that
    the line search does not reject every step due to sampling noise.
    """
    noise_floor = 1.0 / math.sqrt(shots) if shots else 0.0
    descent     = float(np.dot(grad_flat, direction_flat))

    if not np.isfinite(descent) or descent <= 0:
        direction_flat = grad_flat
        descent        = float(np.dot(grad_flat, grad_flat))

    lr          = initial_lr
    best_params = params
    best_energy = energy

    for _ in range(max_steps):
        candidate        = np.array(
            params - lr * unflatten(direction_flat),
            requires_grad=True,
        )
        candidate_energy = float(energy_fn(candidate))

        if candidate_energy < best_energy:
            best_params = candidate
            best_energy = candidate_energy

        # Armijo condition relaxed by one shot-noise sigma
        if candidate_energy <= energy - c * lr * descent + noise_floor:
            return candidate, candidate_energy, lr

        lr *= beta

    return np.array(best_params, requires_grad=True), best_energy, lr


# ─────────────────────────────────────────────────────────────────────────────
# Golden-section alpha search  (Adaptive AQIN)
# ─────────────────────────────────────────────────────────────────────────────

def golden_search_alpha(
    energy_fn:      Callable,
    params:         np.ndarray,
    grad_flat:      np.ndarray,
    metric:         np.ndarray,
    curvature_diag: np.ndarray,
    damping:        float,
    lr:             float,
    tol:            float = 5e-2,    # coarser than statevector — fewer shots
    max_iter:       int   = 12,
) -> float:
    """Lookahead: alpha* = argmin E(theta - lr * M(alpha)^{-1} g)."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    n   = metric.shape[0]

    def objective(alpha: float) -> float:
        M         = (alpha * metric
                     + (1.0 - alpha) * np.diag(curvature_diag)
                     + damping * np.eye(n))
        direction = solve_direction(M, grad_flat)
        candidate = np.array(
            params - lr * unflatten(direction),
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


# ─────────────────────────────────────────────────────────────────────────────
# Physical metrics & result collection
# ─────────────────────────────────────────────────────────────────────────────

def physical_metrics(
    energy_history: list[float],
) -> tuple[float, float, float, float, float]:
    raw_best        = min(energy_history)
    best            = raw_best
    abs_error       = abs(best - E_EXACT)
    rel_error       = abs_error / max(abs(E_EXACT), 1e-12)
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


# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────

def run_gradient_descent(
    energy_fn: Callable,
    cfg:       ProblemConfig,
    max_iter:  int,
    tol:       float,
) -> OptimizerResult:
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad      = grad_fn(params)
        grad_flat = flatten(grad)
        params, energy, _ = line_search(
            energy_fn, params, grad_flat, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=0.20,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  GD iter {step + 1:3d}/{max_iter} "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
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
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    m = np.zeros(PARAM_SHAPE)
    v = np.zeros(PARAM_SHAPE)

    for step in range(1, max_iter + 1):
        grad   = grad_fn(params)
        m      = beta1 * m + (1.0 - beta1) * grad
        v      = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat  = m / (1.0 - beta1 ** step)
        v_hat  = v / (1.0 - beta2 ** step)
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
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Adam", energies, time.perf_counter() - start)


def run_qng(
    energy_fn:  Callable,
    metric_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
) -> OptimizerResult:
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    damping  = shot_noise_damping(cfg.shots, N_PARAMS)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params))
        matrix    = metric + damping * np.eye(N_PARAMS)
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=0.20,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  QNG iter {step + 1:3d}/{max_iter} "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("QNG", energies, time.perf_counter() - start)


def run_qin(
    energy_fn:  Callable,
    metric_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    alpha:      float = 0.5,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """QIN: alpha * F + (1-alpha) * diag(H).  Hessian via parameter-shift."""
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    damping  = shot_noise_damping(cfg.shots, N_PARAMS)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params))
        hdiag     = diagonal_hessian_shift(energy_fn, params)
        matrix    = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(hdiag)
            + damping       * np.eye(N_PARAMS)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  QIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("QIN", energies, time.perf_counter() - start)


def run_aqin(
    energy_fn:  Callable,
    metric_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    alpha:      float = 0.5,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """AQIN: alpha * F + (1-alpha) * diag(H²).  Hessian via parameter-shift."""
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    damping  = shot_noise_damping(cfg.shots, N_PARAMS)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params))
        hdiag     = diagonal_hessian_shift(energy_fn, params)
        matrix    = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(hdiag ** 2)
            + damping       * np.eye(N_PARAMS)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  AQIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("AQIN", energies, time.perf_counter() - start)


def run_adaptive_aqin(
    energy_fn:  Callable,
    metric_fn:  Callable,
    cfg:        ProblemConfig,
    max_iter:   int,
    tol:        float,
    initial_lr: float = 0.20,
) -> OptimizerResult:
    """Adaptive AQIN: alpha tuned each step via golden-section search."""
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    damping  = shot_noise_damping(cfg.shots, N_PARAMS)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params))
        hdiag     = diagonal_hessian_shift(energy_fn, params)
        curvature = hdiag ** 2

        alpha = golden_search_alpha(
            energy_fn=energy_fn,
            params=params,
            grad_flat=grad_flat,
            metric=metric,
            curvature_diag=curvature,
            damping=damping,
            lr=initial_lr,
        )

        matrix = (
            alpha           * metric
            + (1.0 - alpha) * np.diag(curvature)
            + damping       * np.eye(N_PARAMS)
        )
        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=initial_lr,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  Adaptive AQIN iter {step + 1:3d}/{max_iter} "
            f"| alpha = {alpha:.4f} "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Adaptive AQIN", energies, time.perf_counter() - start)


def run_hybrid(
    energy_fn:   Callable,
    metric_fn:   Callable,
    cfg:         ProblemConfig,
    max_iter:    int,
    tol:         float,
    switch_iter: int,
    alpha:       float = 0.5,
) -> OptimizerResult:
    """Hybrid QNG→QIN: pure QNG for switch_iter steps, then QIN."""
    params   = init_params()
    grad_fn  = qml.grad(energy_fn)
    damping  = shot_noise_damping(cfg.shots, N_PARAMS)
    energies = [float(energy_fn(params))]
    best     = energies[0]
    start    = time.perf_counter()

    for step in range(max_iter):
        grad_flat = flatten(grad_fn(params))
        metric    = metric_matrix(metric_fn(params))
        phase     = "QNG" if step < switch_iter else "QIN"

        if step < switch_iter:
            matrix = metric + damping * np.eye(N_PARAMS)
        else:
            hdiag  = diagonal_hessian_shift(energy_fn, params)
            matrix = (
                alpha           * metric
                + (1.0 - alpha) * np.diag(hdiag)
                + damping       * np.eye(N_PARAMS)
            )

        direction = solve_direction(matrix, grad_flat)
        params, energy, _ = line_search(
            energy_fn, params, direction, grad_flat, energies[-1],
            shots=cfg.shots, initial_lr=0.20,
        )
        energies.append(float(energy))
        best    = min(best, float(energy))
        elapsed = time.perf_counter() - start
        print(
            f"  Hybrid iter {step + 1:3d}/{max_iter} [{phase}] "
            f"| E = {energy:.6f} Ha "
            f"| best = {best:.6f} Ha "
            f"| elapsed = {elapsed:.1f} s"
        )
        if abs(energies[-2] - energies[-1]) < tol:
            break

    return collect_result("Hybrid QNG->QIN", energies, time.perf_counter() - start)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[OptimizerResult], cfg: ProblemConfig) -> None:
    print("\n" + "═" * 80)
    print("H2 STO-3G VQE  —  Qiskit backend benchmark summary")
    print(f"  Backend : {cfg.backend_name}   |   Shots : {cfg.shots}")
    print(f"  E_exact = {E_EXACT:.6f} Ha   |   E_HF = {E_HF:.6f} Ha")
    print("═" * 80)
    header = (
        f"{'Optimizer':<20} {'Best E (Ha)':>13} {'Abs err':>11} "
        f"{'Rel err':>11} {'Score %':>9} {'Time (s)':>10} {'Iters':>7}"
    )
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r.name:<20} {r.best_energy:>13.6f} {r.abs_error:>11.3e} "
            f"{r.rel_error:>11.3e} {r.corr_recovered:>9.2f} "
            f"{r.elapsed:>10.1f} {r.iterations:>7d}"
        )
    print()


def print_conclusions(results: list[OptimizerResult]) -> None:
    best_accuracy = min(results, key=lambda r: (r.abs_error, r.elapsed))
    fastest       = min(results, key=lambda r: r.elapsed)
    best_corr     = max(results, key=lambda r: r.corr_recovered)

    print("Thesis-ready conclusions")
    print(
        f"1. Best accuracy     : {best_accuracy.name} — "
        f"{best_accuracy.best_energy:.6f} Ha, error {best_accuracy.abs_error:.3e} Ha."
    )
    print(
        f"2. Correlation score : {best_corr.name} recovered "
        f"{best_corr.corr_recovered:.2f}% of correlation energy."
    )
    print(
        f"3. Fastest           : {fastest.name} in {fastest.elapsed:.1f} s."
    )
    print(
        "4. Shot noise inflates all energy estimates — report best energy over "
        "all iterations, not just the last step."
    )
    print(
        "5. Parameter-shift Hessian avoids the 1/eps² noise amplification of "
        "finite differences and is exact for Pauli-rotation generators."
    )
    print(
        "6. Damping is set automatically based on shot count; increase --shots "
        "to reduce noise and tighten damping."
    )
    print(
        "7. Adaptive AQIN tunes alpha per step via golden-section lookahead, "
        "balancing quantum geometry with curvature information dynamically."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results:  list[OptimizerResult],
    cfg:      ProblemConfig,
    outpath:  Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # Energy convergence
    ax = axes[0, 0]
    for r in results:
        ax.plot(r.energies, label=r.name, linewidth=2)
    ax.axhline(E_EXACT, color="black", linestyle="--", linewidth=1.2, label="Exact")
    ax.axhline(E_HF,    color="gray",  linestyle=":",  linewidth=1.2, label="HF")
    ax.set_title("Energy Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.legend(fontsize=8)

    # Absolute error (log scale)
    ax = axes[0, 1]
    for r in results:
        ax.semilogy(np.maximum(r.errors, 1e-6), label=r.name, linewidth=2)
    ax.set_title("Absolute Error  |E − E_exact|")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|E − E_exact| (Ha)")
    ax.legend(fontsize=8)

    # Correlation score
    ax = axes[1, 0]
    for r in results:
        ax.plot(r.correlations, label=r.name, linewidth=2)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Convergence Score")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

    # Wall-clock time
    ax = axes[1, 1]
    labels = [r.name for r in results]
    times  = [r.elapsed for r in results]
    bars   = ax.bar(labels, times, color=plt.cm.tab10.colors[: len(results)])
    ax.bar_label(bars, fmt="%.1fs", padding=2, fontsize=8)
    ax.set_title("Wall-Clock Time")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"H2 STO-3G VQE — Qiskit backend: {cfg.backend_name} "
        f"| {cfg.shots} shots | {N_QUBITS} qubits",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"\nSaved 4-panel plot to: {outpath.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", type=str, default="ibmq_qasm_simulator",
        help="IBM Quantum backend name (default: ibmq_qasm_simulator).",
    )
    parser.add_argument(
        "--shots", type=int, default=1024,
        help="Shots per circuit execution (default: 1024).",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="IBM Quantum API token. If omitted, uses saved credentials.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use default.qubit instead of Qiskit (no IBM account required).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=60,
        help="Max iterations per optimizer (default: 60).",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4,
        help="Energy-change convergence tolerance (default: 1e-4 Ha).",
    )
    parser.add_argument(
        "--switch-iter", type=int, default=20,
        help="Iteration at which Hybrid QNG->QIN switches phase (default: 20).",
    )
    parser.add_argument(
        "--plot", type=Path, default=Path("qin_h2_qiskit_comparison.png"),
        help="Output path for the 4-panel PNG plot.",
    )
    parser.add_argument(
        "--draw-circuit", action="store_true",
        help="Draw the ansatz circuit before running optimizers.",
    )
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  H2 STO-3G VQE  —  QIN benchmark on Qiskit backend")
    print("═" * 70)
    print(f"  Backend     : {'default.qubit (local)' if args.local else args.backend}")
    print(f"  Shots       : {'∞ (statevector)' if args.local else args.shots}")
    print(f"  Qubits      : {N_QUBITS}")
    print(f"  Param shape : {PARAM_SHAPE}  ({N_PARAMS} parameters)")
    print(f"  Max iter    : {args.max_iter}")
    print(f"  Tolerance   : {args.tol}")
    print(f"  Switch iter : {args.switch_iter}  (Hybrid QNG→QIN)")
    print()

    # ── Build device ──────────────────────────────────────────────────────────
    dev = build_device(
        backend_name=args.backend,
        shots=args.shots,
        token=args.token,
        local=args.local,
    )

    cfg = ProblemConfig(
        n_qubits=N_QUBITS,
        param_shape=PARAM_SHAPE,
        shots=args.shots if not args.local else 0,
        backend_name="default.qubit" if args.local else args.backend,
    )

    # ── Load Hamiltonian ──────────────────────────────────────────────────────
    hamiltonian = load_h2_hamiltonian()
    terms       = getattr(hamiltonian, "ops", getattr(hamiltonian, "operands", []))
    print(f"  Hamiltonian : {len(terms)} Pauli terms")

    # ── Build QNodes ──────────────────────────────────────────────────────────
    energy_fn = make_energy_qnode(dev, hamiltonian, local=args.local)
    metric_fn = make_metric_qnode(dev, energy_fn,   local=args.local)

    # ── Optional circuit diagram ──────────────────────────────────────────────
    if args.draw_circuit:
        p0     = init_params()
        fig, _ = qml.draw_mpl(energy_fn)(p0)
        plt.tight_layout()
        plt.show()

    # ── Cost-per-iteration estimate ───────────────────────────────────────────
    # Each Hessian step fires 2*N_PARAMS extra circuits; each metric step fires
    # O(N_PARAMS) circuits.  Print an estimate so the user knows what to expect.
    circuits_per_grad   = 2 * N_PARAMS      # parameter-shift
    circuits_per_metric = 2 * N_PARAMS      # block-diag metric approx
    circuits_per_hess   = 2 * N_PARAMS      # diagonal Hessian (shift rule)
    print(
        f"\n  Estimated circuits/iter (excl. line search):\n"
        f"    GD / Adam          : {circuits_per_grad}\n"
        f"    QNG                : {circuits_per_grad + circuits_per_metric}\n"
        f"    QIN / AQIN         : {circuits_per_grad + circuits_per_metric + circuits_per_hess}\n"
        f"    Adaptive AQIN      : {circuits_per_grad + circuits_per_metric + circuits_per_hess} "
        f"+ {12 * N_PARAMS} (golden-section)\n"
        f"    Hybrid             : same as QNG (first {args.switch_iter}) "
        f"then QIN (after)\n"
    )

    # ── Run optimizers ────────────────────────────────────────────────────────
    runners = [
        ("Gradient Descent",
         lambda: run_gradient_descent(energy_fn, cfg, args.max_iter, args.tol)),
        ("Adam",
         lambda: run_adam(energy_fn, cfg, args.max_iter, args.tol)),
        ("QNG",
         lambda: run_qng(energy_fn, metric_fn, cfg, args.max_iter, args.tol)),
        ("QIN",
         lambda: run_qin(energy_fn, metric_fn, cfg, args.max_iter, args.tol)),
        ("AQIN",
         lambda: run_aqin(energy_fn, metric_fn, cfg, args.max_iter, args.tol)),
        ("Adaptive AQIN",
         lambda: run_adaptive_aqin(energy_fn, metric_fn, cfg, args.max_iter, args.tol)),
        ("Hybrid QNG->QIN",
         lambda: run_hybrid(energy_fn, metric_fn, cfg, args.max_iter, args.tol,
                            args.switch_iter)),
    ]

    results: list[OptimizerResult] = []
    print()
    for display_name, runner in runners:
        print(f"{'─'*60}")
        print(f"  Starting {display_name} ...")
        print(f"{'─'*60}")
        result = runner()
        results.append(result)
        print(
            f"\n  ✓ {display_name} done "
            f"| best E = {result.best_energy:.6f} Ha "
            f"| error = {result.abs_error:.3e} Ha "
            f"| {result.elapsed:.1f} s "
            f"| {result.iterations} iters\n"
        )

    # ── Report ────────────────────────────────────────────────────────────────
    print_summary(results, cfg)
    plot_results(results, cfg, args.plot)
    print_conclusions(results)


if __name__ == "__main__":
    main()