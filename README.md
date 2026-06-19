# QIN — Quantum Information Newton

Hybrid geometry–curvature optimizers for Variational Quantum Algorithms (VQAs), built on [PennyLane](https://pennylane.ai/).

Quantum Natural Gradient (QNG) preconditions parameter updates with the **Fubini–Study metric tensor**, respecting the geometry of the quantum state manifold. This repo asks a follow-up question: what happens if you *also* give the optimizer local curvature information about the cost landscape?

**Quantum Information Newton (QIN)** answers that by blending the quantum geometric tensor with a finite-difference estimate of the cost Hessian's diagonal:

```
M(α) = α · G + (1 − α) · diag(H)
θ_{t+1} = θ_t − η · M(α)⁻¹ · ∇C(θ_t)
```

where `G` is the (block-diagonal approximation to the) quantum geometric tensor and `diag(H)` is a finite-difference diagonal Hessian estimate.

## Why this exists

QNG alone is excellent at global navigation of variational landscapes but ignores local curvature. Pure curvature-based Newton steps are unstable far from the optimum — Hessian estimates are noisy and unreliable in flat or highly non-convex regions early in training. Benchmarks in this repo show that a **two-stage hybrid schedule** — QNG for global exploration, then a geometry–curvature blend for local refinement — recovers the robustness of QNG while gaining the sharper convergence of curvature-aware steps near the optimum.

## Optimizers implemented

| Name | Description |
|---|---|
| `SGD` | Plain stochastic gradient descent |
| `Gradient Descent` | Full-batch gradient descent with backtracking line search |
| `Adam` | Standard Adam |
| `QNG` | Quantum Natural Gradient (Fubini–Study metric preconditioning) |
| `QIN` | `α·G + (1−α)·diag(H)` — signed finite-difference curvature |
| `AQIN` | `α·G + (1−α)·diag(H²)` — squared curvature for a positive semi-definite correction |
| `Adaptive AQIN` | AQIN with `α` re-tuned at every step via golden-section search |
| `QNG+LBFGS` | Quantum geometry blended with an L-BFGS inverse-Hessian approximation |
| `Adaptive QNG+LBFGS` | Same, with a gradient-norm-dependent adaptive mixing weight |
| `Hybrid QNG→QIN` | QNG for the first `--switch-iter` steps, then switches to QIN |

All preconditioned methods use a damped linear solve (`M + λI`) with a `pinv` fallback for numerical safety, and a backtracking Armijo line search.

## Repository structure

```
qin_common.py                      Shared optimizer suite, metrics, plotting, reporting
qin_h2_real_pennylane.py           H2 STO-3G VQE on IBM Quantum backends (Qiskit + shot noise)
qin_hardware_efficient_pennylane.py  Hardware-efficient ansatz on TFIM / XXZ / random Heisenberg models
qin_lih_pennylane.py               LiH VQE with active-space selection
```

## Benchmark problems

- **H₂** — STO-3G basis, Jordan–Wigner mapping, 4 qubits, 12-parameter hardware-efficient ansatz. Run on `default.qubit` or real IBM Quantum hardware/simulators via the `pennylane-qiskit` plugin.
- **LiH** — STO-3G basis, configurable active electrons/orbitals, Hartree–Fock reference energy included.
- **Spin models** — Transverse-Field Ising (TFIM), XXZ, and random Heisenberg Hamiltonians on a hardware-efficient ansatz with a ring-CNOT entangling layer, scaled to 20–100 trainable parameters.

## Installation

```bash
pip install pennylane matplotlib
# For the IBM Quantum / real-hardware script:
pip install pennylane-qiskit qiskit qiskit-ibm-runtime
```

## Usage

**LiH VQE:**
```bash
python qin_lih_pennylane.py --layers 2 --active-electrons 2 --active-orbitals 4
```

**Hardware-efficient ansatz on a spin model:**
```bash
python qin_hardware_efficient_pennylane.py --hamiltonian tfim --qubits 6 --layers 4
```

**H₂ on a local simulator (no IBM account needed):**
```bash
python qin_h2_real_pennylane.py --local
```

**H₂ on real IBM Quantum hardware:**
```bash
python qin_h2_real_pennylane.py --token YOUR_IBM_TOKEN --backend ibm_brisbane --shots 1024
```

Each script prints a comparison table (best energy, absolute/relative error, ground-state recovery %, wall-clock time, iterations) and saves a 4-panel convergence plot.

## Key finding

On the H₂ benchmark, QNG reliably reaches chemical accuracy. Pure QIN is unstable in early iterations because diagonal Hessian estimates are unreliable far from the optimum. The **Hybrid QNG→QIN** schedule resolves this: quantum geometric information dominates during global exploration, and curvature information is introduced only once the optimizer is already close to the basin of the true minimum — recovering both the stability of QNG and the sharper local convergence that curvature-aware steps provide.

## Notes on the real-hardware script

The IBM Quantum script differs from the statevector version in several ways: gradients use the parameter-shift rule rather than autograd/backprop, the diagonal Hessian uses a parameter-shift double-shift rule instead of finite differences (fewer circuit evaluations, robust to shot noise), and a shot-noise-aware damping floor is applied to all preconditioned methods. IBM Quantum's free-tier access and pricing policies change over time — verify current terms before running on real backends, and note that hardware access is **not free**.

## License

Add your preferred license here (e.g. MIT).
