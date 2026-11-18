"""Quantum fidelity kernel with a bandwidth.

Circuit: ZZ feature map (Havlicek et al., 2019) with linear entanglement and 2 repetitions.
Every feature x_i is first multiplied by the bandwidth c; one qubit per feature
(if more qubits than features are requested, features are repeated cyclically).

    get_states            fast NumPy statevectors, used by every experiment
    get_states_pennylane  the same circuit built gate by gate in PennyLane (reference only)
    kernel_from_states    fidelity |<a|b>|^2 for every pair of rows
    off_diagonal_stats    mean and variance of the off-diagonal similarities (concentration)
"""
import numpy as np
import pandas as pd
from paths import PROCESSED


def load_ready(name):
    """Prepared data of one dataset: X_train, y_train, X_test, y_test."""
    train = pd.read_csv(PROCESSED / f"{name}_train_ready.csv")
    test = pd.read_csv(PROCESSED / f"{name}_test_ready.csv")
    return (train.drop(columns="target").values, train["target"].values,
            test.drop(columns="target").values, test["target"].values)


# ---------------- reference: the circuit gate by gate (PennyLane) ----------------
def encode(x, n_qubits, c=1.0, reps=2):
    import pennylane as qml
    x = c * np.asarray(x)[np.arange(n_qubits) % len(x)]
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


def get_states_pennylane(X, n_qubits, c=1.0, reps=2, device="lightning.qubit"):
    """Reference statevectors, one row at a time (slow)."""
    import pennylane as qml
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        encode(x, n_qubits, c, reps)
        return qml.state()

    return np.array([circuit(x) for x in X])


# ---------------- fast version: the same circuit as matrix algebra ----------------
# RZ(2x) multiplies basis state |z> by exp(-i x s) with s = +1 (bit 0) or -1 (bit 1), and
# CNOT-RZ(phi)-CNOT multiplies it by exp(-i phi/2 s_i s_(i+1)). Both are diagonal, so one
# repetition is "Hadamard on every qubit, then one phase per basis state".
# Qubit 0 is the most significant bit, as in PennyLane.

def _hadamard_matrix(k):
    """Unnormalised Hadamard matrix of size 2^k."""
    H = np.array([[1.0]])
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H


def _hadamard_all(state, n_qubits):
    """Hadamard on every qubit of a batch of states, as two small products H1 @ S @ H2."""
    n1 = n_qubits // 2
    n2 = n_qubits - n1
    H1, H2 = _hadamard_matrix(n1).astype(complex), _hadamard_matrix(n2).astype(complex)
    S = state.reshape(state.shape[0], 2 ** n1, 2 ** n2)
    return (H1 @ S @ H2).reshape(state.shape[0], -1) / np.sqrt(2 ** n_qubits)


def get_states(X, n_qubits, c=1.0, reps=2, batch=256):
    """Statevectors of the feature map for all rows (rows x 2^n_qubits)."""
    X = c * np.asarray(X, dtype=float)[:, np.arange(n_qubits) % np.asarray(X).shape[1]]
    N = 2 ** n_qubits
    bits = (np.arange(N)[:, None] >> (n_qubits - 1 - np.arange(n_qubits))) & 1
    s = 1 - 2 * bits
    ss = s[:, :-1] * s[:, 1:]
    out = np.empty((X.shape[0], N), dtype=complex)
    for start in range(0, X.shape[0], batch):          # batches keep memory small at 14 qubits
        xb = X[start:start + batch]
        P = np.pi - xb
        phase = np.exp(-1j * (xb @ s.T + (P[:, :-1] * P[:, 1:]) @ ss.T))
        state = np.full((xb.shape[0], N), 1 / np.sqrt(N), dtype=complex) * phase
        for _ in range(reps - 1):
            state = _hadamard_all(state, n_qubits) * phase
        out[start:start + batch] = state
    return out


def kernel_from_states(S_a, S_b):
    """Fidelity |<a|b>|^2 for every pair (rows of S_a against rows of S_b)."""
    return np.abs(S_a @ S_b.conj().T) ** 2


def off_diagonal_stats(K):
    """Mean and variance of the off-diagonal similarities: the concentration meter."""
    off = K[np.triu_indices_from(K, k=1)]
    return off.mean(), off.var()
