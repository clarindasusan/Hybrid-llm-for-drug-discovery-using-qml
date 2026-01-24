import torch
class ModelInference:
    def __init__(self):
        # Device
        self.qml_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load trained QML model
        self.qml_model = self._load_model()
        self.qml_model.eval()

        # 🔹 Feature dimension (CRITICAL)
        self.expected_feature_dim = 2053
