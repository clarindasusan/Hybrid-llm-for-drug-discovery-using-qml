import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import logging

import os
from dotenv import load_dotenv

# Load .env immediately
load_dotenv()

# FORCE Hugging Face cache locations (before transformers import)
os.environ["HF_HOME"] = "D:/hf-cache"
os.environ["HF_HUB_CACHE"] = "D:/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf-cache"

logger = logging.getLogger(__name__)
def smiles_to_features(smiles: str, n_features: int = 2053) -> np.ndarray | None:

    """
    Convert SMILES string to molecular feature vector.

    Only returns None if SMILES is truly unparsable.
    Never blocks inference due to chemistry edge cases.
    """
    try:

        smiles = smiles.strip()
        if not smiles:

            logger.warning("Empty SMILES string")
            return None  # true invalid input

        # --- STEP 1: Parse SMILES (ONLY hard failure allowed) ---
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning(f"Unparsable SMILES: {smiles}")
            return None

        # Try sanitization, but DO NOT fail if it breaks
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Sanitization failed, continuing anyway: {e}")

        # --- STEP 2: Fingerprint (fallback to zeros) ---
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )
            fp_array = np.array(fingerprint, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Fingerprint failed, using zeros: {e}")
            fp_array = np.zeros(2048, dtype=np.float32)

        # --- STEP 3: Descriptors (robust, never fail) ---
        def safe_descriptor(fn, default=0.0):
            try:
                val = fn(mol)
                if val is None or np.isnan(val) or np.isinf(val):
                    return default
                return float(val)
            except Exception:
                return default

        descriptors = np.array([
            safe_descriptor(Descriptors.MolWt),
            safe_descriptor(Descriptors.MolLogP),
            safe_descriptor(Descriptors.NumHDonors),
            safe_descriptor(Descriptors.NumHAcceptors),
            safe_descriptor(Descriptors.TPSA),
        ], dtype=np.float32)

        # --- STEP 4: Normalize descriptors ---
        descriptor_ranges = np.array([
            [0, 1000],    # MolWt
            [-5, 10],     # LogP
            [0, 10],      # H donors
            [0, 15],      # H acceptors
            [0, 200],     # TPSA
        ], dtype=np.float32)

        normalized = np.zeros_like(descriptors)
        for i in range(len(descriptors)):
            min_val, max_val = descriptor_ranges[i]
            normalized[i] = (descriptors[i] - min_val) / (max_val - min_val)
            normalized[i] = np.clip(normalized[i], 0.0, 1.0)

        # --- STEP 5: Combine features ---
        features = np.concatenate([fp_array, normalized])

        # --- STEP 6: Enforce fixed size ---
        if features.shape[0] < n_features:
            
            features = np.pad(
                features,
                (0, n_features - features.shape[0]),
                mode="constant"
            )
        elif features.shape[0] > n_features:
            features = features[:n_features]

        return features.astype(np.float32)

    except Exception:
        logger.exception("Unexpected SMILES feature failure")
        return None


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is chemically valid
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def get_molecular_properties(smiles: str) -> dict:
    """
    Get basic molecular properties from SMILES
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular properties
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "h_bond_donors": Descriptors.NumHDonors(mol),
            "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol)
        }
        
    except Exception as e:
        logger.error(f"Error getting molecular properties: {str(e)}")
        return {"error": str(e)}