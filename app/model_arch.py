import torch
import torch.nn as nn
import pennylane as qml

class HybridQMLModel(nn.Module):
    def __init__(self, n_features: int, n_qubits: int, quantum_layer=None):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Classical → Quantum embedding
        self.classical_layer_in = nn.Linear(n_features, n_qubits)
        
        # Quantum layer weights (for Rot gates: 3 parameters per qubit)
        self.q_weights = nn.Parameter(torch.randn(n_qubits, 3))
        
        # Quantum → Classical output (1 expectation value → 1 prediction)
        self.classical_layer_out = nn.Linear(1, 1)
        
        # Store quantum layer if provided (for training)
        # For inference, we'll use a mock version
        self.quantum_layer = quantum_layer

    def forward(self, x):
        # Classical preprocessing
        x_classical = torch.relu(self.classical_layer_in(x))  # Shape: (batch_size, n_qubits)
        
        # Quantum processing - process each sample in the batch
        quantum_outputs = []
        for i in range(x_classical.shape[0]):
            sample_features = x_classical[i, :]
            
            if self.quantum_layer is not None:
                # Use actual quantum circuit (training)
                q_output = self.quantum_layer(self.q_weights, sample_features)
            else:
                # Mock quantum layer for inference without PennyLane
                # This simulates the quantum circuit behavior
                q_output = torch.tanh(torch.sum(self.q_weights * sample_features.unsqueeze(1)))
            
            quantum_outputs.append(q_output)
        
        # Stack quantum outputs
        quantum_output_batch = torch.stack(quantum_outputs)  # Shape: (batch_size,)
        quantum_output_batch = quantum_output_batch.to(torch.float32)
        
        # Classical post-processing
        output = self.classical_layer_out(quantum_output_batch.unsqueeze(1))  # Shape: (batch_size, 1)
        return output