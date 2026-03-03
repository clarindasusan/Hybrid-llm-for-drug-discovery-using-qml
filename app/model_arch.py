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
        self.post = nn.Linear(1, 1)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)

        # Classical pre-processing → qubit rotation angles in (-π, π)
        x_encoded = self.pre(x) * torch.pi   # (batch_size, n_qubits)

        if self.quantum_layer is not None:
            quantum_outputs = []
            for i in range(x_encoded.shape[0]):
                q_out = self.quantum_layer(self.q_weights, x_encoded[i])
                quantum_outputs.append(q_out)
            q_batch = torch.stack(quantum_outputs).float()  # (batch,)
        else:
            # Mock: produce single scalar per sample
            q_batch = torch.mean(x_encoded, dim=1)  # (batch,)                                   # (batch_size, n_qubits)

        # Classical post-processing → logit
        out = self.post(q_batch.unsqueeze(1))    # (batch_size, 1)
        return out