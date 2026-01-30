import torch
import torch.nn as nn

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
        self.quantum_layer = quantum_layer

    def forward(self, x):
        # Classical preprocessing
        x_classical = torch.relu(self.classical_layer_in(x))  # Shape: (batch_size, n_qubits)
        
        if self.quantum_layer is not None:
            # Use actual quantum circuit (training)
            quantum_outputs = []
            for i in range(x_classical.shape[0]):
                sample_features = x_classical[i, :]
                q_output = self.quantum_layer(self.q_weights, sample_features)
                quantum_outputs.append(q_output)
            quantum_output_batch = torch.stack(quantum_outputs)
        else:
            # IMPROVED VECTORIZED mock quantum layer for inference
            # This better preserves input variability
            
            # Expand weights for broadcasting: (1, n_qubits, 3)
            weights_expanded = self.q_weights.unsqueeze(0)
            
            # Apply non-linear transformation per qubit
            # Simulate rotation gates with input-dependent angles
            # Shape: (batch_size, n_qubits)
            
            # Multiple rotation components (simulating Rot gates)
            theta1 = weights_expanded[:, :, 0]  # (1, n_qubits)
            theta2 = weights_expanded[:, :, 1]
            theta3 = weights_expanded[:, :, 2]
            
            # Create input-dependent rotations
            # This creates much more variation based on input features
            angle1 = theta1 * x_classical + theta2
            angle2 = theta2 * torch.sin(x_classical * theta3)
            angle3 = theta3 * torch.cos(x_classical * theta1)
            
            # Combine rotations (simulating quantum gate operations)
            rotated = (torch.sin(angle1) * torch.cos(angle2) + 
                      torch.cos(angle3) * torch.tanh(x_classical))
            
            # Simulate entanglement and measurement (expectation value)
            # Use weighted mean instead of simple mean to preserve more information
            qubit_weights = torch.softmax(self.q_weights[:, 0], dim=0)
            quantum_output_batch = torch.tanh(
                (rotated * qubit_weights.unsqueeze(0)).sum(dim=1)
            )  # (batch_size,)
        
        quantum_output_batch = quantum_output_batch.to(torch.float32)
        
        # Classical post-processing
        output = self.classical_layer_out(quantum_output_batch.unsqueeze(1))  # Shape: (batch_size, 1)
        return output