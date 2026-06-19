"""Quantum classifier benchmark with GD, Adam, QNG, QIN, AQIN, and Hybrid QNG->QIN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np

from qin_common import make_energy_metrics, plot_results, print_recommendations, print_summary, run_optimizer_suite


def make_dataset():
    features = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-0.6, -0.8],
            [-0.8, 0.6],
            [0.7, -0.7],
            [0.8, 0.8],
        ],
        requires_grad=False,
    )
    labels = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0], requires_grad=False)
    return features, labels


def classifier_layer(params, n_qubits: int):
    for wire in range(n_qubits):
        qml.Rot(params[wire, 0], params[wire, 1], params[wire, 2], wires=wire)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


def make_problem(layers: int):
    features, labels = make_dataset()
    n_qubits = 4
    shape = (layers, n_qubits, 3)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def model(x, params):
        qml.AngleEmbedding([x[0], x[1], x[0] * x[1], x[0] ** 2 + x[1] ** 2], wires=range(n_qubits))
        for block in range(layers):
            classifier_layer(params[block], n_qubits)
        return qml.expval(qml.PauliZ(0))

    def loss(params):
        predictions = [model(x, params) for x in features]
        return np.mean((np.stack(predictions) - labels) ** 2)

    n_trainable = int(np.prod(shape))
    trainable_tape_indices = list(range(n_qubits, n_qubits + n_trainable))
    single_metric = qml.metric_tensor(model, approx="block-diag", argnum=trainable_tape_indices)

    def metric_fn(params):
        total = None
        for x in features:
            metric = single_metric(x, params)
            total = metric if total is None else total + metric
        return total / len(features)

    return loss, metric_fn, shape, len(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=3, help="Default is 4*3*3 = 36 parameters.")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--switch-iter", type=int, default=10)
    parser.add_argument("--hessian-max-params", type=int, default=12)
    parser.add_argument("--plot", type=Path, default=Path("qin_quantum_classifier_comparison.png"))
    args = parser.parse_args()

    objective, metric_fn, shape, n_samples = make_problem(args.layers)

    import matplotlib.pyplot as plt

    params = np.random.randn(*shape)

    fig, ax = qml.draw_mpl(
      objective,
      expansion_strategy="device"
    )(params)

    plt.show()

    
    print("Quantum classifier benchmark")
    print("  Dataset: small XOR-style binary classification set")
    print(f"  Samples: {n_samples}")
    print("  Feature map: 4-qubit AngleEmbedding with nonlinear lifted features")
    print(f"  Parameters: {np.prod(shape)}")

    results = run_optimizer_suite(
        objective,
        shape,
        metric_fn,
        make_energy_metrics(exact=0.0, reference=1.0),
        max_iter=args.max_iter,
        tol=args.tol,
        switch_iter=args.switch_iter,
        lr_gd=0.1,
        lr_adam=0.03,
        lr_natural=0.1,
        hessian_max_params=args.hessian_max_params,
    )
    print_summary("Quantum classifier optimizer comparison", results, score_label="Loss red %")
    plot_results(
        "Quantum Classifier: GD, Adam, QNG, QIN, AQIN, Hybrid",
        results,
        args.plot,
        ylabel="Mean squared error",
        score_label="Loss reduction proxy (%)",
    )
    print(f"\nSaved plot to: {args.plot.resolve()}")
    print_recommendations(results, "the quantum classifier")


if __name__ == "__main__":
    main()
