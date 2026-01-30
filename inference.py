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
from app.utils import smiles_to_features
from app.utils import repair_smiles

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self):
        """Initialize all models on startup (CPU-only, HF Spaces safe)"""

        self.llm_device = torch.device("cpu")
        self.qml_device = torch.device("cpu")
        self.expected_feature_dim = 2053

        logger.info("🖥️ Running on CPU (Hugging Face Spaces)")
        logger.info(f"LLM device: {self.llm_device}")
        logger.info(f"QML device: {self.qml_device}")

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

            checkpoint = torch.load(QML_MODEL_PATH, map_location="cpu")

            n_features = checkpoint.get("n_features", self.expected_feature_dim)
            n_qubits = checkpoint.get("n_qubits", 4)

            self.qml_model = HybridQMLModel(
                n_features=n_features,
                n_qubits=n_qubits,
                quantum_layer=None  # inference-only
            )

            self.qml_model.load_state_dict(checkpoint["model_state_dict"])
            self.qml_model.to(self.qml_device)
            self.qml_model.eval()

            logger.info("✅ QML model loaded")

        except Exception:
            logger.error("❌ Failed to load QML model", exc_info=True)
            raise

    # =========================
    # MOLECULE GENERATION
    # =========================
    def generate_molecules(self, disease: str, num_candidates: int = 3) -> list:
        if not disease or not disease.strip():
            raise ValueError("Disease name cannot be empty")

        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "30"))
        temperature = float(os.getenv("TEMPERATURE", "0.9"))

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

        repaired_smiles = []

        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            raw_smiles = text.split("SMILES:", 1)[-1].strip().replace(" ", "")

            if not raw_smiles:
                continue

            fixed = repair_smiles(raw_smiles)

            if fixed:
                repaired_smiles.append(fixed)
            else:
                logger.warning(f"❌ Discarded invalid SMILES: {raw_smiles}")

        # Deduplicate + limit
        return list(dict.fromkeys(repaired_smiles))[:num_candidates]

    # =========================
    # DRUG POTENTIAL PREDICTION
    # =========================
    def predict_drug_potential(self, smiles: str) -> dict:
        try:
            # 🔒 Final SMILES safety check
            fixed_smiles = repair_smiles(smiles)

            if fixed_smiles is None:
                return {
                    "prediction": "invalid",
                    "probability": 0.0,
                    "score": 0.0,
                    "is_promising": False,
                    "confidence": "low",
                    "error": "Invalid SMILES after repair"
                }

            # Step 1: Feature extraction
            features = smiles_to_features(fixed_smiles)
            feature_status = "ok"

            if features is None:
                features = np.zeros(self.expected_feature_dim, dtype=np.float32)
                feature_status = "fallback"

            features_tensor = torch.tensor(
                features, dtype=torch.float32
            ).unsqueeze(0)

            # Step 2: QML inference
            with torch.no_grad():
                logit = self.qml_model(features_tensor)
                probability = torch.sigmoid(logit).item()

            # Step 3: Decision
            is_promising = probability >= 0.5
            confidence_score = abs(probability - 0.5)

            confidence = (
                "high" if confidence_score > 0.30
                else "medium" if confidence_score > 0.15
                else "low"
            )

            return {
                "prediction": "drug" if is_promising else "not drug",
                "probability": round(probability, 4),
                "score": round(probability, 4),
                "is_promising": is_promising,
                "confidence": confidence,
                "feature_status": feature_status
            }

        except Exception as e:
            logger.error("❌ Prediction failure", exc_info=True)
            return {
                "prediction": "unknown",
                "probability": 0.5,
                "score": 0.5,
                "is_promising": False,
                "confidence": "low",
                "error": str(e)
            }
