"""Hardware-efficient ansatz benchmark with 20-100 trainable parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np

from qin_common import (
    exact_ground_energy,
    make_energy_metrics,
    plot_results,
    print_recommendations,
    print_summary,
    run_optimizer_suite,
)


def build_ising_hamiltonian(n_qubits: int):
    coeffs = []
    ops = []
    for wire in range(n_qubits):
        coeffs.append(-0.7 + 0.08 * wire)
        ops.append(qml.PauliZ(wire))
        coeffs.append(0.35)
        ops.append(qml.PauliX(wire))
    for wire in range(n_qubits - 1):
        coeffs.append(0.55)
        ops.append(qml.PauliZ(wire) @ qml.PauliZ(wire + 1))
    coeffs.append(0.40)
    ops.append(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0))
    return qml.Hamiltonian(coeffs, ops)

def build_tfim_hamiltonian(
    n_qubits: int,
    J: float = 0.55,
    g: float = 0.35
):

    coeffs = []
    ops = []

    for i in range(n_qubits):

        coeffs.append(g)
        ops.append(qml.PauliX(i))

    for i in range(n_qubits):

        coeffs.append(J)
        ops.append(
            qml.PauliZ(i)
            @
            qml.PauliZ((i+1)%n_qubits)
        )

    return qml.Hamiltonian(coeffs, ops)

def build_xxz_hamiltonian(
    n_qubits: int,
    Jxy: float = 1.0,
    Jz: float = 0.5
):

    coeffs = []
    ops = []

    for i in range(n_qubits):

        j = (i+1)%n_qubits

        coeffs.extend([
            Jxy,
            Jxy,
            Jz
        ])

        ops.extend([
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j),
            qml.PauliZ(i) @ qml.PauliZ(j),
        ])

    return qml.Hamiltonian(coeffs, ops)

def build_random_heisenberg(
    n_qubits: int,
    seed: int = 42
):

    rng = np.random.default_rng(seed)

    coeffs = []
    ops = []

    for i in range(n_qubits):

        j = (i+1)%n_qubits

        Jx = rng.uniform(-1.0,1.0)
        Jy = rng.uniform(-1.0,1.0)
        Jz = rng.uniform(-1.0,1.0)

        coeffs.extend([
            Jx,
            Jy,
            Jz
        ])

        ops.extend([
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j),
            qml.PauliZ(i) @ qml.PauliZ(j),
        ])

    return qml.Hamiltonian(coeffs, ops)

def hardware_efficient_layer(params, n_qubits: int):
    for wire in range(n_qubits):
        qml.Rot(params[wire, 0], params[wire, 1], params[wire, 2], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])


def make_problem(
    n_qubits: int,
    layers: int,
    model: str,
    seed: int = 42,
):

    if model == "tfim":

        hamiltonian = build_tfim_hamiltonian(
            n_qubits
        )

    elif model == "xxz":

        hamiltonian = build_xxz_hamiltonian(
            n_qubits
        )

    elif model == "heisenberg":

        hamiltonian = build_random_heisenberg(
            n_qubits,
            seed=seed,
        )

    else:

        raise ValueError(
            f"Unknown Hamiltonian model: {model}"
        )

    shape = (layers, n_qubits, 3)

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    @qml.qnode(dev, interface="autograd")
    def objective(params):

        for block in range(layers):

            hardware_efficient_layer(
                params[block],
                n_qubits,
            )

        return qml.expval(
            hamiltonian
        )

    metric_fn = qml.metric_tensor(
        objective,
        approx="block-diag",
    )

    exact_energy = exact_ground_energy(
        hamiltonian,
        n_qubits,
    )

    return (
        objective,
        metric_fn,
        shape,
        exact_energy,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4, help="Default is 6*4*3 = 72 parameters.")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--switch-iter", type=int, default=8)
    parser.add_argument("--hessian-max-params", type=int, default=16)
    parser.add_argument("--plot", type=Path, default=Path("qin_hardware_efficient_comparison.png"))
    parser.add_argument(
    "--hamiltonian",
    choices=["tfim", "xxz", "heisenberg"],
    default="tfim",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    objective, metric_fn, shape, exact = make_problem(
    args.qubits,
    args.layers,
    args.hamiltonian
    )
    import matplotlib.pyplot as plt

    params = np.random.randn(*shape)

    fig, ax = qml.draw_mpl(
      objective,
      expansion_strategy="device"
    )(params)

    plt.show()
    n_params = int(np.prod(shape))
    if not 20 <= n_params <= 100:
        raise ValueError(f"This benchmark is constrained to 20-100 parameters; got {n_params}.")

    print(f"  Hamiltonian: {args.hamiltonian}")
    print(f"  Qubits: {args.qubits}")
    print(f"  Layers: {args.layers}")
    print(f"  Parameters: {n_params}")
    print(f"  Exact Ising ground energy: {exact:.8f}")

    results = run_optimizer_suite(
        objective,
        shape,
        metric_fn,
        make_energy_metrics(exact, reference=0.0),
        max_iter=args.max_iter,
        tol=args.tol,
        switch_iter=args.switch_iter,
        hessian_max_params=args.hessian_max_params,
    )
    plot_path = (
    args.plot
    if args.plot is not None
    else Path(
        f"qin_{args.hamiltonian}_{args.qubits}q_{args.layers}l.png"
    )
    )

    print_summary(
    "Hardware-efficient optimizer comparison",
    results,
    score_label="Recover %"
    )

    plot_results(
    "Hardware-Efficient Ansatz: GD, Adam, QNG, QIN, AQIN, Hybrid",
    results,
    args.plot,        # <-- WRONG
    ylabel="Energy",
    score_label="Ground-state recovery (%)",
   )
    print(f"\nSaved plot to: {args.plot.resolve()}")
    print_recommendations(results, "the hardware-efficient Ising benchmark")


if __name__ == "__main__":
    main()
