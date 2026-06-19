"""BeH2 VQE benchmark with GD, Adam, QNG, QIN, AQIN, and Hybrid QNG->QIN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np

from qin_common import (
    exact_ground_energy,
    hartree_fock_energy,
    make_energy_metrics_chemistry,
    plot_results,
    print_recommendations,
    print_summary,
    run_optimizer_suite,
)


def build_beh2(active_electrons: int = 4, active_orbitals: int = 4):
    symbols = ["H", "Be", "H"]
    bond = 1.3264
    coordinates = np.array(
        [[0.0, 0.0, -bond], [0.0, 0.0, 0.0], [0.0, 0.0, bond]],
        requires_grad=False,
    )
    molecule = qml.qchem.Molecule(
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis_name="sto-3g",
        unit="angstrom",
    )
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping="jordan_wigner",
    )
    hf_state = qml.qchem.hf_state(active_electrons, n_qubits)
    return hamiltonian, n_qubits, hf_state


def layer(params, n_qubits: int):
    for wire in range(n_qubits):
        qml.Rot(params[wire, 0], params[wire, 1], params[wire, 2], wires=wire)
    for wire in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[wire, wire + 1])
    for wire in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[wire, wire + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])


def make_problem(layers: int, active_electrons: int, active_orbitals: int, compute_exact: bool):
    hamiltonian, n_qubits, hf_state = build_beh2(active_electrons, active_orbitals)
    shape = (layers, n_qubits, 3)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def energy(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        for block in range(layers):
            layer(params[block], n_qubits)
        return qml.expval(hamiltonian)

    metric_fn = qml.metric_tensor(energy, approx="block-diag")
    exact = exact_ground_energy(hamiltonian, n_qubits) if compute_exact else None
    hf_energy = hartree_fock_energy(hamiltonian, hf_state, n_qubits) if compute_exact else None
    return energy, metric_fn, shape, n_qubits, exact, hf_energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=3, help="3 layers gives 72 parameters for 8 active qubits.")
    parser.add_argument("--active-electrons", type=int, default=4)
    parser.add_argument("--active-orbitals", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--switch-iter", type=int, default=4)
    parser.add_argument("--hessian-max-params", type=int, default=12)
    parser.add_argument("--no-exact", action="store_true", help="Skip exact diagonalization.")
    parser.add_argument("--plot", type=Path, default=Path("qin_beh2_optimizer_comparison.png"))
    args = parser.parse_args()

    objective, metric_fn, shape, n_qubits, exact, hf_energy = make_problem(
        args.layers, args.active_electrons, args.active_orbitals, not args.no_exact
    )
    import matplotlib.pyplot as plt

    params = np.random.randn(*shape)

    fig, ax = qml.draw_mpl(
      objective,
      expansion_strategy="device"
    )(params)

    plt.show()
    print("BeH2 VQE benchmark")
    print("  Linear geometry: H-Be-H, Be-H = 1.3264 Angstrom, STO-3G")
    print(f"  Active space: electrons={args.active_electrons}, orbitals={args.active_orbitals}")
    print(f"  Qubits: {n_qubits}, parameters: {np.prod(shape)}")
    if exact is not None:
        print(f"  Exact active-space energy: {exact:.8f} Ha")
        print(f"  Hartree-Fock active-space energy: {hf_energy:.8f} Ha")
    
    results = run_optimizer_suite(
        objective,
        shape,
        metric_fn,
        make_energy_metrics_chemistry(exact, hf_energy),
        max_iter=args.max_iter,
        tol=args.tol,
        switch_iter=args.switch_iter,
        hessian_max_params=args.hessian_max_params,
    )
    print_summary("BeH2 optimizer comparison", results, score_label="Corr %")
    plot_results("BeH2 VQE: GD, Adam, QNG, QIN, AQIN, Hybrid", results, args.plot, ylabel="Energy (Ha)")
    print(f"\nSaved plot to: {args.plot.resolve()}")
    print_recommendations(results, "BeH2 active-space VQE")


if __name__ == "__main__":
    main()
