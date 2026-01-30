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
        
        # Quantum processing - process each sample in the batch
        quantum_outputs = []
        for i in range(x_classical.shape[0]):
            sample_features = x_classical[i, :]
            
            if self.quantum_layer is not None:
                # Use actual quantum circuit (training)
                q_output = self.quantum_layer(self.q_weights, sample_features)
            else:
                # IMPROVED mock quantum layer for inference
                # Better simulation that preserves input variability
                
                # Apply rotation-like transformations per qubit
                rotated = torch.zeros(self.n_qubits)
                for q in range(self.n_qubits):
                    # Simulate Rot gate: Rz(θ₁) Ry(θ₂) Rz(θ₃)
                    theta = self.q_weights[q, :]  # 3 rotation angles
                    feature = sample_features[q]
                    
                    # Combine rotation parameters with input feature
                    # This creates a non-linear transformation that varies with input
                    angle = theta[0] * feature + theta[1] * torch.sin(feature) + theta[2]
                    rotated[q] = torch.cos(angle) * torch.sin(theta[1] * feature)
                
                # Simulate entanglement by using mean (mimics expectation value)
                q_output = torch.tanh(rotated.mean())
            
            quantum_outputs.append(q_output)
        
        # Stack quantum outputs
        quantum_output_batch = torch.stack(quantum_outputs)  # Shape: (batch_size,)
        quantum_output_batch = quantum_output_batch.to(torch.float32)
        
        # Classical post-processing
        output = self.classical_layer_out(quantum_output_batch.unsqueeze(1))  # Shape: (batch_size, 1)
        return output