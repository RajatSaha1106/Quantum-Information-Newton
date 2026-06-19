"""QAOA MaxCut benchmark with GD, Adam, QNG, QIN, AQIN, and Hybrid QNG->QIN."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np

from qin_common import make_energy_metrics, plot_results, print_recommendations, print_summary, run_optimizer_suite


EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]


def maxcut_value(bits, edges=EDGES) -> int:
    return sum(1 for i, j in edges if bits[i] != bits[j])


def exact_maxcut(n_nodes: int, edges=EDGES) -> int:
    return max(maxcut_value(bits, edges) for bits in itertools.product([0, 1], repeat=n_nodes))


def cut_hamiltonian(n_nodes: int, edges=EDGES):
    coeffs = []
    ops = []
    for i, j in edges:
        coeffs.append(0.5)
        ops.append(qml.Identity(0))
        coeffs.append(-0.5)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def qaoa_layer(gamma, beta, edges, n_nodes: int):
    for i, j in edges:
        qml.CNOT(wires=[i, j])
        qml.RZ(-2.0 * gamma, wires=j)
        qml.CNOT(wires=[i, j])
    for wire in range(n_nodes):
        qml.RX(2.0 * beta, wires=wire)


def make_problem(depth: int):
    n_nodes = 6
    hamiltonian = cut_hamiltonian(n_nodes)
    shape = (2, depth)
    dev = qml.device("default.qubit", wires=n_nodes)

    @qml.qnode(dev, interface="autograd")
    def expected_cut(params):
        gammas = params[0]
        betas = params[1]
        for wire in range(n_nodes):
            qml.Hadamard(wires=wire)
        for layer in range(depth):
            qaoa_layer(gammas[layer], betas[layer], EDGES, n_nodes)
        return qml.expval(hamiltonian)

    def objective(params):
        return -expected_cut(params)

    return objective, qml.metric_tensor(expected_cut, approx="block-diag"), shape, n_nodes, exact_maxcut(n_nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--switch-iter", type=int, default=20)
    parser.add_argument("--hessian-max-params", type=int, default=None)
    parser.add_argument("--plot", type=Path, default=Path("qin_qaoa_maxcut_comparison.png"))
    args = parser.parse_args()

    objective, metric_fn, shape, n_nodes, optimum = make_problem(args.depth)

    import matplotlib.pyplot as plt

    params = np.random.randn(*shape)

    fig, ax = qml.draw_mpl(
      objective,
      expansion_strategy="device"
    )(params)

    plt.show()
    
    print("QAOA MaxCut benchmark")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {len(EDGES)}")
    print(f"  QAOA depth: {args.depth}")
    print(f"  Parameters: {np.prod(shape)}")
    print(f"  Exact MaxCut value: {optimum}")

    results = run_optimizer_suite(
        objective,
        shape,
        metric_fn,
        make_energy_metrics(exact=-float(optimum), reference=0.0),
        max_iter=args.max_iter,
        tol=args.tol,
        switch_iter=args.switch_iter,
        lr_gd=0.1,
        lr_adam=0.03,
        lr_natural=0.1,
        hessian_max_params=args.hessian_max_params,
    )
    print_summary("QAOA MaxCut optimizer comparison", results, score_label="Opt %")
    plot_results(
        "QAOA MaxCut: GD, Adam, QNG, QIN, AQIN, Hybrid",
        results,
        args.plot,
        ylabel="Negative expected cut",
        score_label="Approximation ratio (%)",
    )
    print(f"\nSaved plot to: {args.plot.resolve()}")
    print_recommendations(results, "QAOA MaxCut")


if __name__ == "__main__":
    main()
