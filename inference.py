import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path
import logging

# =========================
# PATHS (HF Spaces SAFE)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "biogpt-lora-finetuned"
QML_MODEL_PATH = BASE_DIR / "models" / "qml_model.pth"

# Import the model architecture
from app.model_arch import HybridQMLModel
from app.utils import smiles_to_features

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self):
        """Initialize all models on startup - CPU only (HF Spaces compatible)"""

        # HF Spaces = CPU by default
        self.llm_device = torch.device("cpu")
        self.qml_device = torch.device("cpu")

        logger.info("🖥️ Running on CPU (Hugging Face Spaces)")
        logger.info(f"LLM device: {self.llm_device}")
        logger.info(f"QML device: {self.qml_device}")

        # Load models
        self._load_llm()
        self._load_qml()

    # =========================
    # LLM LOADING
    # =========================
    def _load_llm(self):
        try:
            model_path = MODEL_DIR.resolve()
            base_model_name = "microsoft/biogpt"

            logger.info(f"Loading BioGPT LoRA from: {model_path}")

            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Base model (HF cache handled automatically)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.llm_device)

            # Load LoRA adapter
            self.llm_model = PeftModel.from_pretrained(
                base_model,
                model_path,
                local_files_only=True
            ).to(self.llm_device)

            self.llm_model.eval()
            logger.info("✅ BioGPT + LoRA loaded successfully")

        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            raise

    # =========================
    # QML MODEL LOADING
    # =========================
    def _load_qml(self):
        try:
            qml_path = QML_MODEL_PATH.resolve()
            n_qubits = int(os.getenv("N_QUBITS", "4"))

            logger.info(f"Loading QML model from {qml_path}")

            if not qml_path.exists():
                raise FileNotFoundError(f"QML model not found: {qml_path}")

            checkpoint = torch.load(qml_path, map_location=self.qml_device)

            n_features = checkpoint.get("n_features", 2053)

            self.qml_model = HybridQMLModel(
                n_features=n_features,
                n_qubits=n_qubits,
                quantum_layer=None  # inference-only
            )

            self.qml_model.load_state_dict(checkpoint["model_state_dict"])
            self.qml_model.to(self.qml_device)
            self.qml_model.eval()

            logger.info("✅ QML model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading QML model: {str(e)}")
            raise

    # =========================
    # MOLECULE GENERATION
    # =========================
    def generate_molecules(self, disease: str, num_candidates: int = 3) -> list:

        if not disease or not disease.strip():
            raise ValueError("Disease name cannot be empty")

        if not (1 <= num_candidates <= 10):
            raise ValueError("num_candidates must be between 1 and 10")

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
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        generated = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            smiles = text.split("SMILES:", 1)[-1].strip().replace(" ", "")
            if smiles:
                generated.append(smiles)

        # Deduplicate
        return list(dict.fromkeys(generated))[:num_candidates]

    # =========================
    # DRUG POTENTIAL PREDICTION
    # =========================
    def predict_drug_potential(self, smiles: str) -> dict:
        try:
            features = smiles_to_features(smiles)

            if features is None:
                return {
                    "error": "Could not generate features for SMILES",
                    "prediction": "error",
                    "probability": 0.0,
                    "score": 0.0,
                    "is_promising": False,
                    "confidence": "low"
                }

            features_tensor = torch.tensor(
                features, dtype=torch.float32
            ).unsqueeze(0).to(self.qml_device)

            with torch.no_grad():
                output_logit = self.qml_model(features_tensor)
                probability = torch.sigmoid(output_logit).item()

            is_promising = probability >= 0.5
            confidence_score = abs(probability - 0.5)

            confidence = (
                "high" if confidence_score > 0.3
                else "medium" if confidence_score > 0.15
                else "low"
            )

            return {
                "prediction": "drug" if is_promising else "not drug",
                "probability": round(probability, 4),
                "score": round(probability, 4),
                "is_promising": is_promising,
                "confidence": confidence
            }

        except Exception as e:
            logger.error("Prediction error", exc_info=True)
            return {
                "error": str(e),
                "prediction": "error",
                "probability": 0.0,
                "score": 0.0,
                "is_promising": False,
                "confidence": "low"
            }