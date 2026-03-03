import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path
import logging
import numpy as np

# =========================
# PATHS (HF Spaces SAFE)
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "biogpt-lora-finetuned"
QML_MODEL_PATH = BASE_DIR / "models" / "qml_model.pth"

# =========================
# INTERNAL IMPORTS
# =========================

from app.model_arch import HybridQMLModel
from app.utils import smiles_to_features, repair_smiles

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self):
        """Initialize all models on startup (CPU-only, HF Spaces safe)"""

        self.llm_device = torch.device("cpu")
        self.qml_device = torch.device("cpu")

        logger.info("🖥️ Running on CPU (Hugging Face Spaces)")

        self._load_llm()
        self._load_qml()

    # =========================
    # LLM LOADING
    # =========================
    def _load_llm(self):
        try:
            model_path = MODEL_DIR.resolve()
            base_model_name = "microsoft/biogpt"

            if not model_path.exists():
                raise FileNotFoundError(f"LLM path not found: {model_path}")

            logger.info(f"Loading BioGPT LoRA from {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.llm_device)

            self.llm_model = PeftModel.from_pretrained(
                base_model,
                model_path,
                local_files_only=True
            ).to(self.llm_device)

            self.llm_model.eval()
            logger.info("✅ BioGPT + LoRA loaded")

        except Exception:
            logger.error("❌ Failed to load LLM", exc_info=True)
            raise

    # =========================
    # QML MODEL LOADING
    # =========================
    def _load_qml(self):
        try:
            if not QML_MODEL_PATH.exists():
                raise FileNotFoundError(f"QML model not found: {QML_MODEL_PATH}")

            # This .pth contains numpy arrays (pca_components, scaler_mean etc.)
            # saved alongside the model weights. PyTorch 2.6 changed weights_only
            # default to True, which blocks numpy deserialization.
            # weights_only=False is safe here — this is our own trusted checkpoint.
            checkpoint = torch.load(QML_MODEL_PATH, map_location="cpu", weights_only=False)

            # ── Read architecture metadata saved inside the .pth ──────────
            n_qubits    = checkpoint.get("n_qubits",    8)
            n_layers    = checkpoint.get("n_layers",    3)
            feature_dim = checkpoint.get("feature_dim", 64)

            # ── Read preprocessing objects saved inside the .pth ──────────
            self.pca_components   = checkpoint.get("pca_components",   None)  # np.ndarray (n_components, n_raw_features)
            self.pca_mean         = checkpoint.get("pca_mean",         None)  # np.ndarray (n_raw_features,)
            self.scaler_mean      = checkpoint.get("scaler_mean",      None)  # np.ndarray (n_raw_features,)
            self.scaler_scale     = checkpoint.get("scaler_scale",     None)  # np.ndarray (n_raw_features,)
            self.fingerprint_bits = checkpoint.get("fingerprint_bits", 2048)

            logger.info(
                f"QML config — n_qubits={n_qubits}, n_layers={n_layers}, "
                f"feature_dim={feature_dim}, fingerprint_bits={self.fingerprint_bits}"
            )

            # ── Build model and load weights ──────────────────────────────
            self.qml_model = HybridQMLModel(
                n_qubits=n_qubits,
                n_layers=n_layers,
                feature_dim=feature_dim,
                quantum_layer=None   # inference-only: uses mock quantum path
            )

            self.qml_model.load_state_dict(checkpoint["model_state_dict"])
            self.qml_model.to(self.qml_device)
            self.qml_model.eval()

            logger.info("✅ QML model loaded")
            logger.info(f"Scaler mean shape: {self.scaler_mean.shape}")
            logger.info(f"PCA components shape: {self.pca_components.shape}")

        except Exception:
            logger.error("❌ Failed to load QML model", exc_info=True)
            raise

    # =========================
    # INTERNAL: FEATURE PIPELINE
    # =========================
    def _prepare_features(self, smiles: str) -> np.ndarray:
        """
        Full preprocessing pipeline matching training:
          1. smiles_to_features → raw features (Morgan fingerprint + descriptors)
          2. StandardScaler (scaler_mean / scaler_scale)
          3. PCA (pca_mean / pca_components) → feature_dim floats
        Returns a float32 array of shape (feature_dim,).
        """
        # Step 1: Raw features
        raw = smiles_to_features(smiles, n_bits=self.fingerprint_bits)

        if raw is None:
            logger.warning(f"smiles_to_features returned None for: {smiles} — using zeros")
            feature_dim = (
                self.pca_components.shape[0]
                if self.pca_components is not None
                else 64
            )
            return np.zeros(feature_dim, dtype=np.float32)

        raw = np.array(raw, dtype=np.float32)

        # Step 2: StandardScaler
        if self.scaler_mean is not None and self.scaler_scale is not None:
            scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
            raw = (raw - self.scaler_mean) / scale

        # Step 3: PCA
        if self.pca_components is not None and self.pca_mean is not None:
            raw = raw - self.pca_mean
            raw = raw @ self.pca_components.T   # (n_components,)

        return raw.astype(np.float32)

    # =========================
    # MOLECULE GENERATION
    # =========================
    def generate_molecules(self, disease: str, num_candidates: int = 3) -> list:
        """
        Generate raw SMILES strings for a disease.
        Returns exactly what the LLM generates — no validation or repair.
        """
        if not disease or not disease.strip():
            raise ValueError("Disease name cannot be empty")

        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "30"))
        temperature    = float(os.getenv("TEMPERATURE",    "0.9"))

        prompt = f"Disease: {disease} -> SMILES:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.llm_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        raw_smiles = []
        for output in outputs:
            text   = self.tokenizer.decode(output, skip_special_tokens=True)
            smiles = text.split("SMILES:", 1)[-1].strip().replace(" ", "")
            if smiles:
                raw_smiles.append(smiles)

        return list(dict.fromkeys(raw_smiles))[:num_candidates]

    # =========================
    # DRUG POTENTIAL PREDICTION
    # =========================
    def predict_drug_potential(self, smiles: str) -> dict:
        """
        Predict drug-likeness for a SMILES string.
        Validates, repairs, extracts features, then runs QML inference.
        """
        try:
            # Step 1: Validate and repair SMILES
            fixed_smiles = repair_smiles(smiles)

            if fixed_smiles is None:
                return {
                    "prediction":      "invalid",
                    "probability":     0.0,
                    "score":           0.0,
                    "is_promising":    False,
                    "confidence":      "low",
                    "error":           "Invalid SMILES — could not be repaired",
                    "original_smiles": smiles,
                    "repaired_smiles": None
                }

            # Step 2: Feature extraction + preprocessing (scaler → PCA)
            features = self._prepare_features(fixed_smiles)   # (feature_dim,)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, feature_dim)

            # Step 3: QML inference
            with torch.no_grad():
                logit       = self.qml_model(features_tensor)
                probability = torch.sigmoid(logit).item()

            # Step 4: Decision thresholds
            is_promising     = probability >= 0.5
            confidence_score = abs(probability - 0.5)
            confidence = (
                "high"   if confidence_score > 0.30 else
                "medium" if confidence_score > 0.15 else
                "low"
            )

            return {
                "prediction":      "drug" if is_promising else "not drug",
                "probability":     round(probability, 4),
                "score":           round(probability, 4),
                "is_promising":    is_promising,
                "confidence":      confidence,
                "original_smiles": smiles,
                "repaired_smiles": fixed_smiles if fixed_smiles != smiles else None
            }

        except Exception as e:
            logger.error("❌ Prediction failure", exc_info=True)
            return {
                "prediction":      "unknown",
                "probability":     0.5,
                "score":           0.5,
                "is_promising":    False,
                "confidence":      "low",
                "error":           str(e),
                "original_smiles": smiles,
                "repaired_smiles": None
            }