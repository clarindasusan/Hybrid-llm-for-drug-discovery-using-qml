"""
explainer.py — SHAP-based Explainable AI for the QML Drug Discovery Pipeline

Architecture-aware version for HybridQMLModel (PennyLane + PyTorch):

    SMILES
      → smiles_to_features()          raw (fingerprint_bits + 12,)
      → StandardScaler                normalized
      → PCA                           (feature_dim,)   ← model input
      → HybridQMLModel + sigmoid      score in [0, 1]

SHAP operates in PCA space (what the model actually sees).
Raw descriptor values are computed separately for the plain-English
explanation and atom-highlight layers, keeping both layers meaningful.

Explanation layers:
    1. PCA-space SHAP values        → overall score attribution
    2. Descriptor contributions     → raw descriptor values + heuristic attribution
    3. Fingerprint bit importance   → top-N Morgan bits → atom indices
    4. Plain-English summary        → rule-based text from descriptors + ADMET
"""

import numpy as np
import logging
import torch
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

logger = logging.getLogger(__name__)


# ── Descriptor metadata ────────────────────────────────────────────────────────
# Must match the order produced by smiles_to_features() in utils.py exactly.
DESCRIPTOR_META = [
    {"name": "MolWt",               "label": "Molecular Weight",  "unit": "Da", "ideal": "≤ 500",  "good_high": False},
    {"name": "MolLogP",             "label": "LogP",              "unit": "",   "ideal": "0–5",    "good_high": None},
    {"name": "NumHDonors",          "label": "H-Bond Donors",     "unit": "",   "ideal": "≤ 5",    "good_high": False},
    {"name": "NumHAcceptors",       "label": "H-Bond Acceptors",  "unit": "",   "ideal": "≤ 10",   "good_high": False},
    {"name": "TPSA",                "label": "TPSA",              "unit": "Å²", "ideal": "≤ 140",  "good_high": False},
    {"name": "NumRotatableBonds",   "label": "Rotatable Bonds",   "unit": "",   "ideal": "≤ 10",   "good_high": False},
    {"name": "RingCount",           "label": "Ring Count",        "unit": "",   "ideal": "1–4",    "good_high": None},
    {"name": "HeavyAtomCount",      "label": "Heavy Atom Count",  "unit": "",   "ideal": "≤ 40",   "good_high": False},
    {"name": "NHOHCount",           "label": "NHOH Count",        "unit": "",   "ideal": "≤ 5",    "good_high": False},
    {"name": "NOCount",             "label": "N+O Count",         "unit": "",   "ideal": "≤ 10",   "good_high": False},
    {"name": "FractionCSP3",        "label": "Fsp3",              "unit": "",   "ideal": "≥ 0.25", "good_high": True},
    {"name": "NumValenceElectrons", "label": "Valence Electrons", "unit": "",   "ideal": "—",      "good_high": None},
]

N_DESCRIPTORS = 12   # must match utils.py
N_BACKGROUND  = 10   # number of background molecules for SHAP
N_TOP_BITS    = 10   # how many fingerprint bits to report in fingerprint tab


# ── Background dataset ─────────────────────────────────────────────────────────
# Diverse drug-like molecules used as the SHAP baseline.
# KernelExplainer computes E[f(x)] over this set.
BACKGROUND_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",                                          # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                                      # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                                     # Ibuprofen
    "CC(=O)NC1=CC=C(C=C1)O",                                             # Paracetamol
    "OC(=O)c1ccccc1O",                                                   # Salicylic acid
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",                               # Testosterone
    "CN1CCc2cc3c(cc2C1Cc1ccc(OC)c(OC)c1)OCO3",                          # Colchicine-like
    "CC(O)(P(=O)(O)O)P(=O)(O)O",                                         # Etidronic acid
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",                                  # Pyrene
    "CCO",                                                                # Ethanol (low scorer)
    "c1ccccc1",                                                           # Benzene (low scorer)
    "CC(=O)c1ccc(cc1)C(C)(C)C",                                          # 4-tBu acetophenone
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",      # Imatinib-like
    "COc1cc2c(cc1OC)C(=CC(=O)c3ccc(OC)c(OC)c3)CC2",                    # Curcumin-like
    "CC1=C(C(=O)Nc2ccccc2)c2cc(Cl)sc2N1C(=O)c1ccccc1",                 # Benzodiazepine-like
    "O=C(O)c1ccc(cc1)c1csc(N)n1",                                       # Febuxostat fragment
    "CC(C)(C)OC(=O)N1CCC(CC1)n1cnc2ccccc21",                            # Tofacitinib-like
    "FC(F)(F)c1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCOCC1",                      # Fluoxetine-like
    "Clc1ccc(cc1)C(c1ccccc1)(c1ccccc1)O",                               # Clotrimazole
    "CC12CC(=O)C3C(C1CCC2(O)C#C)CCC4=CC(=O)CCC34C",                    # Norgestrel-like
]


# ── Raw descriptor extraction ──────────────────────────────────────────────────

def _get_raw_descriptors(mol) -> dict:
    """
    Compute the 12 raw RDKit descriptors for a molecule.
    Returns a dict keyed by DESCRIPTOR_META name fields.
    Used for the plain-English explanation — independent of PCA.
    """
    return {
        "MolWt":               Descriptors.MolWt(mol),
        "MolLogP":             Crippen.MolLogP(mol),
        "NumHDonors":          rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors":       rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":                rdMolDescriptors.CalcTPSA(mol),
        "NumRotatableBonds":   rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount":           rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount":      mol.GetNumHeavyAtoms(),
        "NHOHCount":           rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NOCount":             sum(
                                   1 for a in mol.GetAtoms()
                                   if a.GetAtomicNum() in (7, 8)
                               ),
        "FractionCSP3":        rdMolDescriptors.CalcFractionCSP3(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
    }


# ── Descriptor heuristic attribution ──────────────────────────────────────────

def _descriptor_heuristic_shap(descriptor_values: dict, score: float) -> dict:
    """
    Since SHAP operates in PCA space (not descriptor space), we cannot directly
    decompose PCA-space SHAP values back into per-descriptor contributions without
    the full inverse PCA transform per feature — which is expensive and noisy.

    Instead we use a chemically-grounded heuristic: measure how far each descriptor
    deviates from its drug-like ideal range and weight by the overall score.
    This produces signed pseudo-SHAP values that are:
      - Positive when the descriptor is in a drug-like range
      - Negative when it falls outside the ideal range
      - Scaled so that the sum approximates the score deviation from 0.5

    This is transparently labelled in the frontend as "heuristic attribution"
    rather than direct SHAP values.
    """
    mw   = descriptor_values.get("MolWt",             0.0)
    lp   = descriptor_values.get("MolLogP",            0.0)
    hbd  = descriptor_values.get("NumHDonors",         0.0)
    hba  = descriptor_values.get("NumHAcceptors",      0.0)
    tpsa = descriptor_values.get("TPSA",               0.0)
    rot  = descriptor_values.get("NumRotatableBonds",  0.0)
    rng  = descriptor_values.get("RingCount",          0.0)
    hac  = descriptor_values.get("HeavyAtomCount",     0.0)
    fsp3 = descriptor_values.get("FractionCSP3",       0.0)
    nhoh = descriptor_values.get("NHOHCount",          0.0)
    noc  = descriptor_values.get("NOCount",            0.0)

    scale = score - 0.5   # how far the score is from neutral

    def _clamp(raw):
        """Clamp to [-0.15, 0.15] to keep values reasonable."""
        return max(-0.15, min(0.15, raw))

    shap = {}

    # Molecular Weight — ideal 150–500 Da
    if   mw < 150:  shap["MolWt"] = _clamp(-0.08)
    elif mw <= 500: shap["MolWt"] = _clamp(+0.10 * scale / max(abs(scale), 0.01))
    else:           shap["MolWt"] = _clamp(-0.05 * (mw - 500) / 100)

    # LogP — ideal 0–5
    if   lp < 0:   shap["MolLogP"] = _clamp(-0.06)
    elif lp <= 5:  shap["MolLogP"] = _clamp(+0.08 * scale / max(abs(scale), 0.01))
    else:          shap["MolLogP"] = _clamp(-0.05 * (lp - 5))

    # H-Bond Donors — ideal ≤ 5
    shap["NumHDonors"]    = _clamp(+0.06 if hbd <= 5  else -0.04 * (hbd - 5))

    # H-Bond Acceptors — ideal ≤ 10
    shap["NumHAcceptors"] = _clamp(+0.05 if hba <= 10 else -0.03 * (hba - 10))

    # TPSA — ideal ≤ 140 Å²
    shap["TPSA"]          = _clamp(+0.07 if tpsa <= 140 else -0.04 * (tpsa - 140) / 20)

    # Rotatable bonds — ideal ≤ 10
    shap["NumRotatableBonds"] = _clamp(+0.04 if rot <= 10 else -0.03 * (rot - 10))

    # Ring count — ideal 1–4
    if   rng == 0:  shap["RingCount"] = _clamp(-0.05)
    elif rng <= 4:  shap["RingCount"] = _clamp(+0.06)
    else:           shap["RingCount"] = _clamp(-0.02 * (rng - 4))

    # Heavy atom count — ideal ≤ 40
    shap["HeavyAtomCount"] = _clamp(+0.04 if hac <= 40 else -0.02 * (hac - 40) / 5)

    # NHOH count — ideal ≤ 5
    shap["NHOHCount"] = _clamp(+0.03 if nhoh <= 5 else -0.02 * (nhoh - 5))

    # N+O count — ideal ≤ 10
    shap["NOCount"] = _clamp(+0.03 if noc <= 10 else -0.02 * (noc - 10))

    # Fsp3 — higher is generally better (≥ 0.25 associated with lower attrition)
    shap["FractionCSP3"] = _clamp(+0.07 if fsp3 >= 0.25 else -0.04 * (0.25 - fsp3))

    # Valence electrons — neutral, minor signal
    shap["NumValenceElectrons"] = _clamp(0.0)

    return shap


# ── Plain-English explanation generator ───────────────────────────────────────

def generate_explanation_text(
    score: float,
    descriptor_values: dict,
    shap_pca_summary: dict,
    ro5_violations: int,
    bbb: str,
    bioavailability: str,
) -> str:
    """
    Generate a plain-English explanation of the drug-likeness score.
    Rule-based — no LLM required.

    Args:
        score:            drug-likeness score [0, 1]
        descriptor_values: raw RDKit descriptor dict
        shap_pca_summary:  {"total_positive", "total_negative", "n_components"}
        ro5_violations:   integer count from ADMET endpoint
        bbb:              "Likely" | "Unlikely" | "Uncertain"
        bioavailability:  "High" | "Moderate" | "Low"
    """
    lines = []

    # Overall verdict
    if score >= 0.7:
        lines.append(
            f"This molecule scores {round(score * 100)}/100 and is predicted to be drug-like."
        )
    elif score >= 0.4:
        lines.append(
            f"This molecule scores {round(score * 100)}/100 and sits on the borderline of drug-likeness."
        )
    else:
        lines.append(
            f"This molecule scores {round(score * 100)}/100 and is predicted to be poorly drug-like."
        )

    # PCA-space SHAP framing
    total_pos = shap_pca_summary.get("total_positive", 0.0)
    total_neg = shap_pca_summary.get("total_negative", 0.0)
    n_comp    = shap_pca_summary.get("n_components",   0)
    if abs(total_pos) > 0.01 or abs(total_neg) > 0.01:
        lines.append(
            f"Across {n_comp} latent chemical features, the quantum model found "
            f"{round(total_pos, 3)} net positive signal and "
            f"{round(abs(total_neg), 3)} net negative signal."
        )

    # Lipinski commentary
    if ro5_violations == 0:
        lines.append(
            "Passes all Lipinski Rule of Five criteria — good oral absorption expected."
        )
    elif ro5_violations == 1:
        lines.append(
            "One Lipinski violation detected — oral bioavailability may be slightly reduced."
        )
    else:
        lines.append(
            f"{ro5_violations} Lipinski violations detected — oral bioavailability likely compromised."
        )

    # Specific descriptor commentary
    mw   = descriptor_values.get("MolWt",            0)
    lp   = descriptor_values.get("MolLogP",           0)
    tpsa = descriptor_values.get("TPSA",              0)
    hbd  = descriptor_values.get("NumHDonors",        0)
    fsp3 = descriptor_values.get("FractionCSP3",      0)
    rot  = descriptor_values.get("NumRotatableBonds", 0)

    if mw > 500:
        lines.append(
            f"Molecular weight ({round(mw)} Da) exceeds 500 Da — may reduce oral absorption."
        )
    if lp > 5:
        lines.append(
            f"LogP ({round(lp, 2)}) is high — increased lipophilicity may cause toxicity or poor solubility."
        )
    elif lp < 0:
        lines.append(
            f"LogP ({round(lp, 2)}) is very low — molecule may be too hydrophilic for membrane permeability."
        )
    if tpsa > 140:
        lines.append(
            f"TPSA ({round(tpsa)} Å²) exceeds 140 Å² — poor intestinal permeability predicted."
        )
    if hbd > 5:
        lines.append(
            f"High H-bond donor count ({int(hbd)}) may limit membrane permeability."
        )
    if rot > 10:
        lines.append(
            f"High rotatable bond count ({int(rot)}) may reduce oral bioavailability."
        )
    if fsp3 >= 0.4:
        lines.append(
            f"Good Fsp3 ({round(fsp3, 2)}) — strong 3D character, associated with lower clinical attrition."
        )
    elif fsp3 < 0.25:
        lines.append(
            f"Low Fsp3 ({round(fsp3, 2)}) — flat/aromatic molecule, associated with higher attrition risk."
        )

    # BBB
    if bbb == "Likely":
        lines.append(
            "Blood-brain barrier penetration is predicted — relevant for CNS targets."
        )
    elif bbb == "Unlikely":
        lines.append(
            "Blood-brain barrier penetration is unlikely — suitable for peripheral targets."
        )

    # Bioavailability
    if bioavailability == "High":
        lines.append("Oral bioavailability is predicted to be high.")
    elif bioavailability == "Low":
        lines.append("Oral bioavailability is predicted to be low.")

    return " ".join(lines)


# ── Main explainer class ───────────────────────────────────────────────────────

class MoleculeExplainer:
    """
    SHAP KernelExplainer for the HybridQMLModel pipeline.

    Key design decisions:
    - SHAP operates in PCA space (feature_dim,) — this is what the model sees.
    - Background is built using model._prepare_features() (full pipeline).
    - Descriptor contributions use raw descriptor values + heuristic attribution
      (not direct SHAP decomposition) because PCA mixes all descriptors together.
    - Fingerprint atom highlights use PCA loadings projected back to raw bit space.

    Usage:
        explainer = MoleculeExplainer(model_inference)
        result    = explainer.explain("CCO")
    """

    def __init__(self, model_inference):
        """
        Args:
            model_inference: a ModelInference instance with:
                - _prepare_features(smiles) → np.ndarray (feature_dim,)
                - qml_model                 → HybridQMLModel (nn.Module)
                - fingerprint_bits          → int
                - pca_components            → np.ndarray or None
        """
        self.model      = model_inference
        self._explainer = None   # lazy-initialised on first explain() call

    # ── Internal: batch prediction for SHAP ───────────────────────────────────

    def _predict_from_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        try:
            x = torch.tensor(feature_matrix, dtype=torch.float32)  # (n_samples, feature_dim)
            with torch.no_grad():
                logits        = self.model.qml_model(x)                      # (n_samples, 1)
                probabilities = torch.sigmoid(logits).squeeze(-1).flatten()  # force (n_samples,)
            result = probabilities.numpy().astype(np.float32)
            # Explicit shape guard — SHAP will crash if this is not 1D
            if result.ndim != 1:
                result = result.flatten()
            return result
        except Exception as e:
            logger.warning(f"SHAP batch prediction failed: {e}")
            return np.full(len(feature_matrix), 0.5, dtype=np.float32)

    # ── Internal: background in PCA space ─────────────────────────────────────

    def _build_background(self) -> np.ndarray:
        """
        Build the SHAP background matrix using the full preprocessing pipeline
        (StandardScaler → PCA), matching what the model actually receives.
        """
        features = []
        for smi in BACKGROUND_SMILES[:N_BACKGROUND]:
            try:
                f = self.model._prepare_features(smi)   # (feature_dim,)
                if f is not None:
                    features.append(f)
            except Exception as e:
                logger.warning(f"Background feature extraction failed for {smi}: {e}")
                continue

        if not features:
            feature_dim = (
                self.model.pca_components.shape[0]
                if self.model.pca_components is not None
                else 64
            )
            logger.warning("All background molecules failed — using zero background")
            return np.zeros((1, feature_dim), dtype=np.float32)

        bg = np.array(features, dtype=np.float32)
        logger.info(f"Background matrix built: {bg.shape}")
        return bg

    # ── Internal: lazy SHAP initialisation ────────────────────────────────────

    def _get_explainer(self):
        """Initialise SHAP KernelExplainer once and cache it."""
        if self._explainer is None:
            try:
                import shap
            except ImportError:
                raise RuntimeError(
                    "SHAP not installed. Add 'shap' to requirements.txt and redeploy."
                )
            background      = self._build_background()
            self._explainer = shap.KernelExplainer(
                self._predict_from_features,
                background,
                link="identity"
            )
            logger.info(
                f"SHAP KernelExplainer initialised. "
                f"Background shape: {background.shape}, "
                f"Expected value: {self._explainer.expected_value:.4f}"
            )
        return self._explainer

    # ── Internal: fallback importance ─────────────────────────────────────────

    def _fallback_importance(
        self, features: np.ndarray, background: np.ndarray
    ) -> np.ndarray:
        """
        Gradient-free fallback if SHAP fails completely.
        Uses mean absolute deviation from background as a proxy for importance.
        """
        bg_mean = background.mean(axis=0)
        return (features - bg_mean).astype(np.float32)

    # ── Public: main explain method ────────────────────────────────────────────

    def explain(self, smiles: str, admet_data: dict = None) -> dict:
        """
        Full explanation pipeline for a single SMILES string.

        Args:
            smiles:     SMILES string (should already be repaired before calling)
            admet_data: optional dict from the /admet endpoint — enriches text

        Returns dict with keys:
            smiles                    — the input SMILES
            score                     — drug-likeness score [0, 1]
            shap_base_value           — SHAP expected value (baseline)
            shap_pca_values           — raw SHAP values in PCA space (list)
            shap_pca_summary          — {total_positive, total_negative, n_components}
            descriptor_contributions  — list of dicts (label, value, shap, direction)
            fingerprint_contributions — list of dicts (bit, shap, atoms, present)
            important_atoms           — list of atom indices for the 3D viewer
            explanation_text          — plain-English paragraph
            confidence                — 'high' | 'medium' | 'low'
            error                     — None or error string
        """
        # ── Parse molecule ─────────────────────────────────────────────────────
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Could not parse SMILES: {smiles}"}

        # ── 1. Preprocessed features (PCA space) — what the model sees ────────
        try:
            pca_features = self.model._prepare_features(smiles)   # (feature_dim,)
        except Exception as e:
            return {"error": f"Feature extraction failed: {e}"}

        if pca_features is None:
            return {"error": "Feature extraction returned None"}

        pca_features = pca_features.astype(np.float32)
        feature_dim  = pca_features.shape[0]

        # ── 2. Raw features — for fingerprint bit→atom mapping ─────────────────
        # Import here to avoid circular imports (utils is in app/)
        try:
            from app.utils import smiles_to_features
        except ImportError:
            from utils import smiles_to_features

        fp_bits      = getattr(self.model, "fingerprint_bits", 2048)
        raw_features = smiles_to_features(smiles, n_bits=fp_bits)
        fp_features  = (
            np.array(raw_features[:fp_bits], dtype=np.float32)
            if raw_features is not None
            else np.zeros(fp_bits, dtype=np.float32)
        )

        # ── 3. Raw descriptor values — for text + heuristic descriptor SHAP ───
        descriptor_values = _get_raw_descriptors(mol)

        # ── 4. Get model score ─────────────────────────────────────────────────
        try:
            score = float(
                self._predict_from_features(pca_features.reshape(1, -1))[0]
            )
        except Exception as e:
            logger.warning(f"Score computation failed: {e}")
            score = 0.5

        # ── 5. SHAP in PCA space ───────────────────────────────────────────────
        explainer = self._get_explainer()

        try:
            shap_values = explainer.shap_values(
                pca_features.reshape(1, -1),
                nsamples=2 * pca_features.shape[0] + 2048,    # increase for accuracy, decrease for speed
                silent=True
            )
            # Normalise output shape — can be (1, feature_dim) or (feature_dim,)
            shap_values = np.array(shap_values, dtype=np.float32).flatten()
            if shap_values.shape[0] != feature_dim:
                shap_values = shap_values[:feature_dim]
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}", exc_info=True)
            background  = self._build_background()
            shap_values = self._fallback_importance(pca_features, background)

        base_value = (
            float(explainer.expected_value)
            if hasattr(explainer, "expected_value")
            else 0.5
        )

        # ── 6. PCA-space SHAP summary ──────────────────────────────────────────
        positive_shap    = shap_values[shap_values > 0]
        negative_shap    = shap_values[shap_values < 0]
        shap_pca_summary = {
            "total_positive": float(positive_shap.sum()) if len(positive_shap) else 0.0,
            "total_negative": float(negative_shap.sum()) if len(negative_shap) else 0.0,
            "n_components":   int(feature_dim),
            "max_component":  int(np.argmax(np.abs(shap_values))),
        }

        # ── 7. Descriptor contributions (heuristic attribution) ────────────────
        # We cannot invert PCA per-descriptor cleanly, so we use the chemically-
        # grounded heuristic attribution scaled by the PCA SHAP signal magnitude.
        heuristic_shap = _descriptor_heuristic_shap(descriptor_values, score)

        # Scale heuristics by the overall PCA SHAP magnitude for consistency
        pca_magnitude = float(np.abs(shap_values).mean())
        scale_factor  = pca_magnitude / 0.05 if pca_magnitude > 0 else 1.0
        scale_factor  = min(max(scale_factor, 0.3), 3.0)   # clamp to [0.3, 3.0]

        descriptor_contributions = []
        for meta in DESCRIPTOR_META:
            name  = meta["name"]
            val   = descriptor_values.get(name, 0.0)
            shval = heuristic_shap.get(name, 0.0) * scale_factor
            descriptor_contributions.append({
                "name":      name,
                "label":     meta["label"],
                "unit":      meta["unit"],
                "ideal":     meta["ideal"],
                "value":     round(float(val), 4),
                "shap":      round(float(shval), 4),
                "direction": (
                    "positive" if shval >  0.005 else
                    "negative" if shval < -0.005 else
                    "neutral"
                ),
                "magnitude": round(abs(float(shval)), 4),
            })

        descriptor_contributions.sort(key=lambda x: x["magnitude"], reverse=True)

        # ── 8. Fingerprint bit contributions ───────────────────────────────────
        # Use PCA loadings to project top SHAP components back to raw bit space.
        fingerprint_contributions = []
        all_important_atoms       = set()

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=fp_bits, bitInfo=bit_info
        )

        pca_components = getattr(self.model, "pca_components", None)

        if pca_components is not None:
            # pca_components shape: (n_components, n_raw_features)
            # Find the top SHAP components and project back to raw feature space
            top_pca_indices = np.argsort(np.abs(shap_values))[::-1][:5].tolist()

            candidate_bits = set()
            for pca_idx in top_pca_indices:
                if pca_idx < pca_components.shape[0]:
                    loadings    = pca_components[pca_idx]       # (n_raw_features,)
                    fp_loadings = loadings[:fp_bits]            # fingerprint portion only
                    top_raw_bits = np.argsort(
                        np.abs(fp_loadings)
                    )[::-1][:N_TOP_BITS].tolist()
                    for b in top_raw_bits:
                        if fp_features[b] > 0:                 # only ON bits
                            candidate_bits.add(b)

            # Rank candidate bits by activation × max PCA loading magnitude
            top_bit_indices = sorted(
                candidate_bits,
                key=lambda b: float(fp_features[b]) * float(
                    np.abs(pca_components[:, b]).max()
                    if b < pca_components.shape[1] else 0.0
                ),
                reverse=True
            )[:N_TOP_BITS]

        else:
            # No PCA components stored — fall back to top activated bits
            on_bits         = np.where(fp_features > 0)[0]
            top_bit_indices = on_bits[:N_TOP_BITS].tolist()

        # Map bits → atom indices and build response
        for rank, bit_idx in enumerate(top_bit_indices):
            # Assign a pseudo-SHAP value scaled by rank and score deviation
            pseudo_shap   = float(score - base_value) * (1.0 / (rank + 1))
            atoms_for_bit = []

            if bit_idx in bit_info:
                for atom_idx, radius in bit_info[bit_idx]:
                    atoms_for_bit.append(atom_idx)
                    all_important_atoms.add(atom_idx)
                    if radius > 0:
                        for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                            atoms_for_bit.append(nb.GetIdx())
                            all_important_atoms.add(nb.GetIdx())

            fingerprint_contributions.append({
                "bit":       int(bit_idx),
                "shap":      round(pseudo_shap, 4),
                "direction": "positive" if pseudo_shap >= 0 else "negative",
                "atoms":     sorted(set(atoms_for_bit)),
                "present":   bool(fp_features[bit_idx] > 0),
            })

        important_atoms = sorted(all_important_atoms)

        # ── 9. Plain-English explanation ───────────────────────────────────────
        ro5  = admet_data.get("ro5_violations",  0)           if admet_data else 0
        bbb  = admet_data.get("bbb",             "Uncertain") if admet_data else "Uncertain"
        bioa = admet_data.get("bioavailability", "Moderate")  if admet_data else "Moderate"

        explanation_text = generate_explanation_text(
            score, descriptor_values, shap_pca_summary, ro5, bbb, bioa
        )

        # ── 10. Confidence ─────────────────────────────────────────────────────
        # Based on SHAP value spread in PCA space
        shap_std   = float(np.std(shap_values))
        confidence = (
            "high"   if shap_std > 0.05 else
            "medium" if shap_std > 0.02 else
            "low"
        )

        return {
            "smiles":                    smiles,
            "score":                     round(score, 4),
            "shap_base_value":           round(base_value, 4),
            "shap_pca_values":           shap_values.tolist(),
            "shap_pca_summary":          shap_pca_summary,
            "descriptor_contributions":  descriptor_contributions,
            "fingerprint_contributions": fingerprint_contributions,
            "important_atoms":           important_atoms,
            "explanation_text":          explanation_text,
            "confidence":                confidence,
            "error":                     None,
        }