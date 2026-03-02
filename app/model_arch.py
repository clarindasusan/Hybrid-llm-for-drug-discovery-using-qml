import torch
import torch.nn as nn


class HybridQMLModel(nn.Module):
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, feature_dim: int = 64, quantum_layer=None):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim

        # Classical encoder: PCA-reduced features → qubit angles
        self.pre = nn.Sequential(
            nn.Linear(feature_dim, 64),   # pre.0
            nn.ReLU(),                     # pre.1  (no weights, not in state_dict)
            nn.Linear(64, n_qubits),       # pre.2
            nn.Tanh(),                     # pre.3  (no weights, not in state_dict)
        )

        # Quantum variational weights: (n_layers, n_qubits, 3) for StronglyEntanglingLayers
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        # Quantum layer (PennyLane QNode — only used during training)
        self.quantum_layer = quantum_layer

        # Classical decoder: n_qubits expectation values → 1 logit
        self.post = nn.Linear(n_qubits, 1)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)

        # Classical pre-processing → qubit rotation angles in (-π, π)
        x_encoded = self.pre(x) * torch.pi   # (batch_size, n_qubits)

        if self.quantum_layer is not None:
            # ── Training path: real quantum circuit ──────────────────────
            quantum_outputs = []
            for i in range(x_encoded.shape[0]):
                q_out = self.quantum_layer(self.q_weights, x_encoded[i])
                quantum_outputs.append(torch.stack(q_out))   # (n_qubits,)
            q_batch = torch.stack(quantum_outputs).float()    # (batch_size, n_qubits)
        else:
            # ── Inference path: mock StronglyEntanglingLayers ────────────
            # Simulates data re-uploading across n_layers with entanglement
            batch_size = x_encoded.shape[0]

            # Initialise qubit state as input angles
            state = x_encoded                                 # (batch_size, n_qubits)

            for layer in range(self.n_layers):
                w = self.q_weights[layer]                     # (n_qubits, 3)
                # Simulate Rot gates: RZ(w2) RY(w1) RZ(w0)
                rx = torch.sin(state * w[:, 0] + w[:, 1])
                ry = torch.cos(state * w[:, 1] + w[:, 2])
                rz = torch.tanh(state * w[:, 2] + w[:, 0])
                rotated = (rx + ry + rz) / 3.0               # (batch_size, n_qubits)
                # Simulate CNOT entanglement: each qubit mixes with neighbour
                shifted = torch.roll(rotated, 1, dims=1)
                state = torch.tanh(rotated + 0.5 * shifted)  # (batch_size, n_qubits)

            q_batch = state                                   # (batch_size, n_qubits)

        # Classical post-processing → logit
        out = self.post(q_batch)    # (batch_size, 1)
        return out