"""
explainer.py — SHAP-based Explainable AI for the QML Drug Discovery Pipeline

Treats the QML model as a pure black box:
    features (1036-d vector) → score (float)

Explanation layers:
    1. Descriptor-level SHAP values  (12 RDKit descriptors)
    2. Fingerprint bit importance     (top-N Morgan bits)
    3. Atom-level highlights          (bits → atom indices via bitInfo)
    4. Plain-English summary          (rule-based from SHAP + ADMET)
"""

import numpy as np
import logging
from typing import Optional
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# ── Descriptor metadata ────────────────────────────────────────────────────────
# Must match the order in smiles_to_features() in utils.py exactly
DESCRIPTOR_META = [
    {"name": "MolWt",               "label": "Molecular Weight",      "unit": "Da",  "ideal": "≤ 500",    "good_high": False},
    {"name": "MolLogP",             "label": "LogP",                  "unit": "",    "ideal": "0–5",      "good_high": None},
    {"name": "NumHDonors",          "label": "H-Bond Donors",         "unit": "",    "ideal": "≤ 5",      "good_high": False},
    {"name": "NumHAcceptors",       "label": "H-Bond Acceptors",      "unit": "",    "ideal": "≤ 10",     "good_high": False},
    {"name": "TPSA",                "label": "TPSA",                  "unit": "Å²",  "ideal": "≤ 140",    "good_high": False},
    {"name": "NumRotatableBonds",   "label": "Rotatable Bonds",       "unit": "",    "ideal": "≤ 10",     "good_high": False},
    {"name": "RingCount",           "label": "Ring Count",            "unit": "",    "ideal": "1–4",      "good_high": None},
    {"name": "HeavyAtomCount",      "label": "Heavy Atom Count",      "unit": "",    "ideal": "≤ 40",     "good_high": False},
    {"name": "NHOHCount",           "label": "NHOH Count",            "unit": "",    "ideal": "≤ 5",      "good_high": False},
    {"name": "NOCount",             "label": "N+O Count",             "unit": "",    "ideal": "≤ 10",     "good_high": False},
    {"name": "FractionCSP3",        "label": "Fsp3",                  "unit": "",    "ideal": "≥ 0.25",   "good_high": True},
    {"name": "NumValenceElectrons", "label": "Valence Electrons",     "unit": "",    "ideal": "—",        "good_high": None},
]

N_BITS        = 1024   # must match utils.py
N_DESCRIPTORS = 12     # must match utils.py
N_BACKGROUND  = 20     # SHAP background samples — higher = slower but more accurate
N_TOP_BITS    = 10     # how many fingerprint bits to report


# ── Background dataset ─────────────────────────────────────────────────────────
# A diverse set of known drug-like molecules used as SHAP baseline.
# KernelExplainer computes E[f(x)] over this background.
BACKGROUND_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",           # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",      # Ibuprofen
    "CC(=O)NC1=CC=C(C=C1)O",              # Paracetamol
    "OC(=O)c1ccccc1O",                    # Salicylic acid
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",# Testosterone
    "CN1CCc2cc3c(cc2C1Cc1ccc(OC)c(OC)c1)OCO3", # Colchicine-like
    "CC(O)(P(=O)(O)O)P(=O)(O)O",          # Etidronic acid
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",   # Pyrene
    "CCO",                                 # Ethanol (low scorer)
    "c1ccccc1",                            # Benzene (low scorer)
    "CC(=O)c1ccc(cc1)C(C)(C)C",           # 4-tBu acetophenone
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", # Imatinib-like
    "COc1cc2c(cc1OC)C(=CC(=O)c3ccc(OC)c(OC)c3)CC2", # Curcumin-like
    "CC1=C(C(=O)Nc2ccccc2)c2cc(Cl)sc2N1C(=O)c1ccccc1", # Benzodiazepine-like
    "O=C(O)c1ccc(cc1)c1csc(N)n1",         # Febuxostat fragment
    "CC(C)(C)OC(=O)N1CCC(CC1)n1cnc2ccccc21", # Tofacitinib-like
    "FC(F)(F)c1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCOCC1", # Fluoxetine-like
    "Clc1ccc(cc1)C(c1ccccc1)(c1ccccc1)O", # Clotrimazole
    "CC12CC(=O)C3C(C1CCC2(O)C#C)CCC4=CC(=O)CCC34C", # Norgestrel-like
]


def _get_background_features(smiles_to_features_fn) -> np.ndarray:
    """Build background feature matrix from diverse drug-like molecules."""
    features = []
    for smi in BACKGROUND_SMILES:
        f = smiles_to_features_fn(smi)
        if f is not None:
            features.append(f)
    if not features:
        # Fallback: zero background
        return np.zeros((1, N_BITS + N_DESCRIPTORS), dtype=np.float32)
    return np.array(features, dtype=np.float32)


# ── Atom highlight computation ─────────────────────────────────────────────────

def get_important_atoms(mol, important_bit_indices: list[int]) -> list[int]:
    """
    Map important Morgan fingerprint bit indices back to atom indices.

    Morgan bits encode circular atom environments. The bitInfo dict maps
    bit_index → list of (center_atom_idx, radius) tuples.
    """
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=N_BITS, bitInfo=bit_info
    )
    important_atoms = set()
    for bit_idx in important_bit_indices:
        if bit_idx in bit_info:
            for atom_idx, radius in bit_info[bit_idx]:
                important_atoms.add(atom_idx)
                # Also include neighbors within the radius
                if radius > 0:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        important_atoms.add(neighbor.GetIdx())
    return sorted(important_atoms)


# ── Plain-English explanation generator ───────────────────────────────────────

def generate_explanation_text(
    score: float,
    descriptor_shap: dict,   # {descriptor_name: shap_value}
    descriptor_values: dict, # {descriptor_name: actual_value}
    ro5_violations: int,
    bbb: str,
    bioavailability: str,
) -> str:
    """
    Generate a plain-English explanation of the drug-likeness score.
    Rule-based — no LLM required.
    """
    lines = []

    # Overall verdict
    if score >= 0.7:
        lines.append(f"This molecule scores {round(score * 100)}/100 and is predicted to be drug-like.")
    elif score >= 0.4:
        lines.append(f"This molecule scores {round(score * 100)}/100 and sits on the borderline of drug-likeness.")
    else:
        lines.append(f"This molecule scores {round(score * 100)}/100 and is predicted to be poorly drug-like.")

    # Top positive contributors
    positive = sorted(
        [(k, v) for k, v in descriptor_shap.items() if v > 0.01],
        key=lambda x: x[1], reverse=True
    )[:3]
    if positive:
        pos_labels = []
        for name, val in positive:
            meta = next((m for m in DESCRIPTOR_META if m["name"] == name), None)
            label = meta["label"] if meta else name
            pos_labels.append(label)
        lines.append(f"Score driven up by: {', '.join(pos_labels)}.")

    # Top negative contributors
    negative = sorted(
        [(k, v) for k, v in descriptor_shap.items() if v < -0.01],
        key=lambda x: x[1]
    )[:3]
    if negative:
        neg_labels = []
        for name, val in negative:
            meta = next((m for m in DESCRIPTOR_META if m["name"] == name), None)
            label = meta["label"] if meta else name
            neg_labels.append(label)
        lines.append(f"Score penalized by: {', '.join(neg_labels)}.")

    # Lipinski commentary
    if ro5_violations == 0:
        lines.append("Passes all Lipinski Rule of Five criteria — good oral absorption expected.")
    elif ro5_violations == 1:
        lines.append("One Lipinski violation detected — oral bioavailability may be slightly reduced.")
    else:
        lines.append(f"{ro5_violations} Lipinski violations detected — oral bioavailability likely compromised.")

    # Specific descriptor commentary
    mw  = descriptor_values.get("MolWt", 0)
    lp  = descriptor_values.get("MolLogP", 0)
    tpsa = descriptor_values.get("TPSA", 0)
    hbd = descriptor_values.get("NumHDonors", 0)
    fsp3 = descriptor_values.get("FractionCSP3", 0)

    if mw > 500:
        lines.append(f"Molecular weight ({round(mw)} Da) exceeds 500 Da — may reduce oral absorption.")
    if lp > 5:
        lines.append(f"LogP ({round(lp, 2)}) is high — increased lipophilicity may cause toxicity.")
    elif lp < 0:
        lines.append(f"LogP ({round(lp, 2)}) is very low — molecule may be too hydrophilic for membrane permeability.")
    if tpsa > 140:
        lines.append(f"TPSA ({round(tpsa)} Å²) exceeds 140 Å² — poor intestinal permeability predicted.")
    if hbd > 5:
        lines.append(f"High H-bond donor count ({int(hbd)}) may limit membrane permeability.")
    if fsp3 >= 0.4:
        lines.append(f"Good Fsp3 ({round(fsp3, 2)}) — molecule has good 3D character, associated with lower attrition.")

    # BBB
    if bbb == "Likely":
        lines.append("Blood-brain barrier penetration is predicted — relevant for CNS targets.")
    elif bbb == "Unlikely":
        lines.append("Blood-brain barrier penetration is unlikely — suitable for peripheral targets.")

    return " ".join(lines)


# ── Main explainer class ───────────────────────────────────────────────────────

class MoleculeExplainer:
    """
    SHAP KernelExplainer wrapper for the QML drug discovery model.

    Usage:
        explainer = MoleculeExplainer(model_inference, smiles_to_features)
        result = explainer.explain("CCO")
    """

    def __init__(self, model_inference, smiles_to_features_fn):
        self.model     = model_inference
        self.feat_fn   = smiles_to_features_fn
        self._explainer = None   # lazy-initialised on first call

    def _predict_from_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Wrapper so SHAP can call the QML model on a batch of feature vectors.
        Returns array of scores, shape (n_samples,).
        """
        scores = []
        for features in feature_matrix:
            try:
                # Reconstruct a minimal SMILES-independent prediction path.
                # We call the model's internal predict method directly on features.
                # Adjust this call to match your ModelInference API.
                score = self.model.predict_from_features(features)
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Prediction failed for a SHAP sample: {e}")
                scores.append(0.0)
        return np.array(scores, dtype=np.float32)

    def _get_explainer(self, background: np.ndarray):
        """Lazy-initialise SHAP KernelExplainer (slow first time, cached after)."""
        if self._explainer is None:
            try:
                import shap
                # Use only descriptor portion for efficiency if fingerprint is large
                # Full vector explanation is also possible but slower
                self._explainer = shap.KernelExplainer(
                    self._predict_from_features,
                    background,
                    link="identity"
                )
                logger.info("SHAP KernelExplainer initialised.")
            except ImportError:
                raise RuntimeError("SHAP not installed. Run: pip install shap")
        return self._explainer

    def explain(self, smiles: str, admet_data: dict = None) -> dict:
        """
        Full explanation pipeline for a single SMILES string.

        Args:
            smiles:     canonical SMILES string (should already be repaired)
            admet_data: optional dict from /admet endpoint for richer text

        Returns:
            dict with keys:
                descriptor_contributions  — list of {name, label, value, shap, direction}
                fingerprint_contributions — list of {bit, shap, atoms}
                important_atoms           — list of atom indices to highlight
                explanation_text          — plain-English summary string
                shap_base_value           — SHAP expected value (baseline score)
                confidence                — 'high' | 'medium' | 'low'
        """
        import shap

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Could not parse SMILES: {smiles}"}

        # ── 1. Compute features ──────────────────────────────────────────────
        features = self.feat_fn(smiles)
        if features is None:
            return {"error": "Feature extraction failed"}

        features = features.astype(np.float32)
        fp_features   = features[:N_BITS]
        desc_features = features[N_BITS:]

        # ── 2. Build background and get SHAP values ──────────────────────────
        background = _get_background_features(self.feat_fn)
        explainer  = self._get_explainer(background)

        try:
            # nsamples: more = more accurate but slower
            # 100 is a good balance for production
            shap_values = explainer.shap_values(
                features.reshape(1, -1),
                nsamples=100,
                silent=True
            )
            # shap_values shape: (1, n_features) or (n_features,)
            if hasattr(shap_values, '__len__') and len(shap_values) == 1:
                shap_values = shap_values[0]
            shap_values = np.array(shap_values, dtype=np.float32).flatten()
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}", exc_info=True)
            # Fall back to gradient-free approximation
            shap_values = self._fallback_importance(features, background)

        shap_fp   = shap_values[:N_BITS]
        shap_desc = shap_values[N_BITS:]

        base_value = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.5

        # ── 3. Descriptor contributions ──────────────────────────────────────
        descriptor_contributions = []
        descriptor_values_dict   = {}

        for i, meta in enumerate(DESCRIPTOR_META):
            val   = float(desc_features[i])
            shval = float(shap_desc[i]) if i < len(shap_desc) else 0.0
            descriptor_values_dict[meta["name"]] = val
            descriptor_contributions.append({
                "name":      meta["name"],
                "label":     meta["label"],
                "unit":      meta["unit"],
                "ideal":     meta["ideal"],
                "value":     round(val, 4),
                "shap":      round(shval, 4),
                "direction": "positive" if shval > 0.005 else "negative" if shval < -0.005 else "neutral",
                "magnitude": round(abs(shval), 4),
            })

        # Sort by absolute SHAP magnitude
        descriptor_contributions.sort(key=lambda x: x["magnitude"], reverse=True)

        # ── 4. Fingerprint bit contributions ─────────────────────────────────
        # Get top-N most impactful bits
        top_bit_indices = np.argsort(np.abs(shap_fp))[::-1][:N_TOP_BITS].tolist()

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=N_BITS, bitInfo=bit_info
        )

        fingerprint_contributions = []
        all_important_atoms       = set()

        for bit_idx in top_bit_indices:
            shval = float(shap_fp[bit_idx])
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
                "shap":      round(shval, 4),
                "direction": "positive" if shval > 0 else "negative",
                "atoms":     sorted(set(atoms_for_bit)),
                "present":   bool(fp_features[bit_idx] > 0),
            })

        important_atoms = sorted(all_important_atoms)

        # ── 5. Plain-English explanation ─────────────────────────────────────
        desc_shap_dict = {
            m["name"]: float(shap_desc[i])
            for i, m in enumerate(DESCRIPTOR_META)
            if i < len(shap_desc)
        }

        # Get score for the molecule
        try:
            score = float(self._predict_from_features(features.reshape(1, -1))[0])
        except Exception:
            score = 0.5

        # Pull ADMET fields if provided
        ro5  = admet_data.get("ro5_violations", 0) if admet_data else 0
        bbb  = admet_data.get("bbb", "Uncertain") if admet_data else "Uncertain"
        bioa = admet_data.get("bioavailability", "Moderate") if admet_data else "Moderate"

        explanation_text = generate_explanation_text(
            score, desc_shap_dict, descriptor_values_dict, ro5, bbb, bioa
        )

        # ── 6. Confidence ─────────────────────────────────────────────────────
        # Based on how spread the SHAP values are — concentrated = confident
        shap_std = float(np.std(shap_desc))
        confidence = "high" if shap_std > 0.05 else "medium" if shap_std > 0.02 else "low"

        return {
            "smiles":                    smiles,
            "score":                     round(score, 4),
            "shap_base_value":           round(base_value, 4),
            "descriptor_contributions":  descriptor_contributions,
            "fingerprint_contributions": fingerprint_contributions,
            "important_atoms":           important_atoms,
            "explanation_text":          explanation_text,
            "confidence":                confidence,
        }

    def _fallback_importance(
        self, features: np.ndarray, background: np.ndarray
    ) -> np.ndarray:
        """
        Gradient-free fallback importance if SHAP fails.
        Uses mean absolute difference from background as a proxy.
        """
        bg_mean = background.mean(axis=0)
        return (features - bg_mean).astype(np.float32)