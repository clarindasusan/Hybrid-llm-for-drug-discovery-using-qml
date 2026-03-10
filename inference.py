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

BASE_DIR      = Path(__file__).resolve().parent
MODEL_DIR     = BASE_DIR / "models" / "biogpt-lora-finetuned"
QML_MODEL_PATH = BASE_DIR / "models" / "qml_model.pth"

# =========================
# INTERNAL IMPORTS
# =========================

from app.model_arch import HybridQMLModel
from app.utils import smiles_to_features, repair_smiles

logger = logging.getLogger(__name__)


# =========================
# FALLBACK SMILES
# =========================
# Curated, chemically valid, drug-like molecules per disease category.
# Used when the LLM fails to generate any valid SMILES after MAX_ATTEMPTS.
# Each entry has been manually verified to parse cleanly with RDKit.

FALLBACK_SMILES = {
    # ── Oncology ──────────────────────────────────────────────────────────
    "cancer": [
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib-like
        "COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCCCN4CCOCC4",                 # Gefitinib-like
        "CC(C)(C)OC(=O)Nc1ccc(cc1)C(=O)Nc2ccc(cc2)N3CCOCC3",           # Sorafenib-like
    ],
    "tumor": [
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
        "O=C(Nc1ccc(F)cc1)c1cc(Cl)cc(Cl)c1",
        "CC1=C(C(=O)Nc2ccccc2)c2cc(Cl)sc2N1C(=O)c1ccccc1",
    ],
    "leukemia": [
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
        "CN1CCN(CC1)c1ccc(cc1)C(=O)Nc2cc(cnc2N)c3ccccc3",
        "COc1ccc(cc1OC)c1cc2c(NC(=O)c3ccc(F)cc3)ncnc2[nH]1",
    ],

    # ── Neurology / CNS ───────────────────────────────────────────────────
    "alzheimer": [
        "CCc1nn(C)c2ccc(nc12)C(=O)Nc3ccc(cc3)N3CCN(CC3)C",             # PDE5i-like
        "O=C(O)c1ccccc1Oc1ccccc1",                                      # Aspirin scaffold
        "CN(C)CCc1c[nH]c2ccc(CS(=O)(=O)N3CCCC3)cc12",                  # Donepezil-like
    ],
    "parkinson": [
        "NCCc1ccc(O)c(O)c1",                                            # Dopamine
        "CC(N)Cc1ccc(O)c(O)c1",                                         # Methyldopa
        "O=C(O)C(N)Cc1ccc(O)c(O)c1",                                    # L-DOPA
    ],
    "depression": [
        "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",                          # Fluoxetine
        "CN(C)CCCN1c2ccccc2CCc2ccccc21",                                # Imipramine-like
        "OC(=O)c1ccc(Cl)cc1",
    ],
    "epilepsy": [
        "NC(=O)c1cccnc1",                                               # Nicotinamide scaffold
        "CC(C)CC1(CC(=O)N)C(=O)NC(=O)N1",                              # Gabapentin-like
        "NCC(O)c1ccc(O)c(O)c1",
    ],

    # ── Metabolic / Endocrine ─────────────────────────────────────────────
    "diabetes": [
        "CN(C)C(=N)NC(=N)N",                                            # Metformin
        "CCOC(=O)c1cnc2cc(S(N)(=O)=O)ccc2c1NCc1ccc(F)cc1",             # Glipizide-like
        "CC1=CC(=C(C=C1)OCC(O)=O)CCOC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",
    ],
    "obesity": [
        "O=C(O)CCCCCCC(=O)O",
        "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",                                 # Salbutamol-like
        "COc1cc2c(cc1OC)CC(N)CC2",
    ],
    "thyroid": [
        "Ic1cc(CC(N)C(=O)O)cc(I)c1Oc1cc(I)c(O)c(I)c1",                # Thyroxine-like
        "CC(N)Cc1ccc(O)c(O)c1",
        "O=C(O)C(N)Cc1ccc(O)c(O)c1",
    ],

    # ── Cardiovascular ────────────────────────────────────────────────────
    "hypertension": [
        "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1cccc(Cl)c1)C(=O)OCC",         # Amlodipine-like
        "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",
        "OC(=O)c1ccccc1Oc1ccccc1",
    ],
    "heart disease": [
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",                         # Testosterone scaffold
        "O=C(O)c1ccccc1OC(=O)C",                                        # Aspirin
        "CCCC1=NN(C2=CC(=C(C=C12)S(=O)(=O)N1CCN(CC1)C)OCC)C",         # Sildenafil-like
    ],
    "stroke": [
        "OC(=O)c1ccccc1Oc1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",                                        # Aspirin
        "O=C(O)CCCc1ccc(O)cc1",
    ],

    # ── Infectious Disease ────────────────────────────────────────────────
    "infection": [
        "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",                  # Penicillin-like
        "OC(=O)c1ccc(N)cc1",                                            # PABA
        "Cc1onc(c1C(=O)Nc2ccc(O)cc2)c1ccccc1",
    ],
    "influenza": [
        "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@@H](NC(C)=O)[C@@H](N)C1",     # Oseltamivir
        "OC(=O)[C@@H](N)Cc1ccc(O)cc1",
        "CC(=O)Nc1ccc(O)cc1",                                           # Paracetamol
    ],
    "hiv": [
        "Cc1ccc(cc1)S(=O)(=O)Nc1ccc(N2CCOCC2)nn1",
        "O=C(Nc1ccc(F)cc1)c1ccncc1",
        "CC(C)(C)OC(=O)N1CCC(CC1)n1cnc2ccccc21",
    ],
    "malaria": [
        "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",                           # Chloroquine-like
        "OCC1OC(O)C(O)C(O)C1O",
        "Clc1ccc(cc1)C(c1ccccc1)(c1ccccc1)O",                          # Clotrimazole
    ],
    "tuberculosis": [
        "OC(=O)c1ccncc1",                                               # Nicotinic acid
        "CC1=CN=C(C=C1)C(=O)NN",                                        # Isoniazid-like
        "O=C(O)c1ccc(N)cc1",
    ],

    # ── Inflammatory / Autoimmune ─────────────────────────────────────────
    "arthritis": [
        "CC(=O)Oc1ccccc1C(=O)O",                                        # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                   # Ibuprofen
        "COc1ccc(cc1OC)C(=O)c1ccc(O)cc1",
    ],
    "inflammation": [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                   # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",                                        # Aspirin
        "O=C(O)c1ccc(Cl)cc1",
    ],
    "lupus": [
        "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
        "COc1ccc(cc1)S(=O)(=O)Nc1ccc(N)cc1",
        "O=C(Nc1ccccc1)c1cccnc1",
    ],

    # ── Respiratory ───────────────────────────────────────────────────────
    "asthma": [
        "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",                                 # Salbutamol
        "COc1ccc(CCNCc2ccc(O)c(CO)c2)cc1",
        "O=C(O)c1ccc(N)cc1S(=O)(=O)N",
    ],
    "copd": [
        "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",
        "O=C(O)CCc1ccc(O)cc1",
        "Cn1ccnc1SCC1=C(N2C(=O)[C@@H](NC(=O)Cc3ccccc3)[C@H]2SC)CS1",
    ],

    # ── Generic fallback (used when disease not recognised) ───────────────
    "default": [
        "CC(=O)Oc1ccccc1C(=O)O",                                        # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                   # Ibuprofen
        "CC(=O)Nc1ccc(O)cc1",                                           # Paracetamol
        "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", # Imatinib-like
        "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@@H](NC(C)=O)[C@@H](N)C1",     # Oseltamivir
    ],
}

# Max attempts to generate valid SMILES before using fallback
MAX_GENERATION_ATTEMPTS = 3


def _get_fallback(disease: str, n: int) -> list[str]:
    """
    Return up to n fallback SMILES for a disease keyword.
    Matches on lowercase substring so 'breast cancer' → 'cancer' key.
    """
    disease_lower = disease.lower().strip()
    for key in FALLBACK_SMILES:
        if key in disease_lower or disease_lower in key:
            return FALLBACK_SMILES[key][:n]
    return FALLBACK_SMILES["default"][:n]


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
            model_path      = MODEL_DIR.resolve()
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

            checkpoint = torch.load(QML_MODEL_PATH, map_location="cpu", weights_only=False)

            n_qubits    = checkpoint.get("n_qubits",    8)
            n_layers    = checkpoint.get("n_layers",    3)
            feature_dim = checkpoint.get("feature_dim", 64)

            self.pca_components   = checkpoint.get("pca_components",   None)
            self.pca_mean         = checkpoint.get("pca_mean",         None)
            self.scaler_mean      = checkpoint.get("scaler_mean",      None)
            self.scaler_scale     = checkpoint.get("scaler_scale",     None)
            self.fingerprint_bits = checkpoint.get("fingerprint_bits", 2048)

            logger.info(
                f"QML config — n_qubits={n_qubits}, n_layers={n_layers}, "
                f"feature_dim={feature_dim}, fingerprint_bits={self.fingerprint_bits}"
            )

            self.qml_model = HybridQMLModel(
                n_qubits=n_qubits,
                n_layers=n_layers,
                feature_dim=feature_dim,
                quantum_layer=None
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

        if self.scaler_mean is not None and self.scaler_scale is not None:
            scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
            raw   = (raw - self.scaler_mean) / scale

        if self.pca_components is not None and self.pca_mean is not None:
            raw = raw - self.pca_mean
            raw = raw @ self.pca_components.T

        return raw.astype(np.float32)

    # =========================
    # INTERNAL: SINGLE GENERATION ATTEMPT
    # =========================
    def _generate_once(
        self,
        disease: str,
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int,
    ) -> list[str]:
        """
        Run one LLM generation pass and return raw decoded SMILES strings.
        No validation — caller handles that.
        """
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
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for output in outputs:
            text   = self.tokenizer.decode(output, skip_special_tokens=True)
            smiles = text.split("SMILES:", 1)[-1].strip().replace(" ", "")
            if smiles:
                results.append(smiles)
        return results

    # =========================
    # MOLECULE GENERATION (Option 3)
    # =========================
    def generate_molecules(self, disease: str, num_candidates: int = 3) -> list:
        """
        Generate valid, repaired SMILES strings for a disease.

        Option 3 improvements vs original:
          1. MAX_NEW_TOKENS increased from 30 → 80 (covers full drug-like SMILES)
          2. Validity filter: only repaired, parseable SMILES are kept
          3. Multiple generation attempts before fallback
          4. Fallback to curated SMILES when LLM produces nothing valid
          5. Returns list of dicts with {smiles, source} so frontend can show
             whether the molecule was generated or is a curated fallback

        Returns:
            list of dicts: [{"smiles": str, "source": "generated"|"fallback"}, ...]
        """
        if not disease or not disease.strip():
            raise ValueError("Disease name cannot be empty")

        # ── Generation parameters ─────────────────────────────────────────
        # Increased from 30 → 80: most drug-like SMILES are 40–80 chars
        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
        temperature    = float(os.getenv("TEMPERATURE",   "0.85"))

        valid_smiles = []
        seen         = set()

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            logger.info(
                f"Generation attempt {attempt}/{MAX_GENERATION_ATTEMPTS} "
                f"for disease='{disease}'"
            )

            # Generate more than needed to have candidates to filter
            raw_batch = self._generate_once(
                disease,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_candidates * 2,
            )

            for raw in raw_batch:
                if len(valid_smiles) >= num_candidates:
                    break

                repaired = repair_smiles(raw)
                if repaired is None:
                    logger.debug(f"Irreparable SMILES (attempt {attempt}): {raw[:60]}")
                    continue

                if repaired in seen:
                    continue

                seen.add(repaired)
                valid_smiles.append({
                    "smiles": repaired,
                    "source": "generated",
                })
                logger.info(f"✓ Valid generated SMILES: {repaired[:60]}")

            if len(valid_smiles) >= num_candidates:
                break

            # Slightly raise temperature on retry to get more diverse outputs
            temperature = min(temperature + 0.05, 1.2)

        # ── Fallback if not enough valid SMILES generated ─────────────────
        if len(valid_smiles) < num_candidates:
            needed    = num_candidates - len(valid_smiles)
            fallbacks = _get_fallback(disease, needed)
            for fb_smiles in fallbacks:
                if fb_smiles not in seen:
                    valid_smiles.append({
                        "smiles": fb_smiles,
                        "source": "fallback",
                    })
                    seen.add(fb_smiles)
            logger.warning(
                f"LLM generated {num_candidates - needed}/{num_candidates} valid SMILES "
                f"for '{disease}' — used {needed} curated fallback(s)"
            )

        result = valid_smiles[:num_candidates]
        logger.info(
            f"Returning {len(result)} molecules for '{disease}': "
            f"{[r['source'] for r in result]}"
        )
        return result

    # =========================
    # PREDICT FROM FEATURES (for SHAP)
    # =========================
    def predict_from_features(self, features: np.ndarray) -> float:
        """
        Run the QML model directly on a pre-computed feature vector.
        Used by SHAP KernelExplainer.
        Accepts either raw features or post-PCA features — detects automatically.
        Returns: float drug-likeness score in [0, 1]
        """
        try:
            features    = np.array(features, dtype=np.float32).flatten()
            feature_dim = (
                self.pca_components.shape[0]
                if self.pca_components is not None
                else 64
            )

            if features.shape[0] != feature_dim:
                if self.scaler_mean is not None and self.scaler_scale is not None:
                    scale    = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
                    features = (features - self.scaler_mean) / scale
                if self.pca_components is not None and self.pca_mean is not None:
                    features = features - self.pca_mean
                    features = features @ self.pca_components.T

            features = features.astype(np.float32)
            x        = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logit       = self.qml_model(x)
                probability = torch.sigmoid(logit).squeeze().item()
            return float(probability)

        except Exception as e:
            logger.warning(f"predict_from_features failed: {e}")
            return 0.0

    # =========================
    # DRUG POTENTIAL PREDICTION
    # =========================
    def predict_drug_potential(self, smiles: str) -> dict:
        """
        Predict drug-likeness for a SMILES string.
        Validates, repairs, extracts features, then runs QML inference.
        """
        try:
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
                    "repaired_smiles": None,
                }

            features        = self._prepare_features(fixed_smiles)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logit       = self.qml_model(features_tensor)
                probability = torch.sigmoid(logit).squeeze().item()

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
                "repaired_smiles": fixed_smiles if fixed_smiles != smiles else None,
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
                "repaired_smiles": None,
            }