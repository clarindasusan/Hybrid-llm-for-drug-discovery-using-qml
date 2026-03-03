import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import logging
import os
from dotenv import load_dotenv

# Load .env immediately
load_dotenv()

# Force Hugging Face cache locations
os.environ["HF_HOME"] = "D:/hf-cache"
os.environ["HF_HUB_CACHE"] = "D:/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf-cache"

logger = logging.getLogger(__name__)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def smiles_to_features(
    smiles: str,
    n_bits: int = 1024
) -> np.ndarray | None:
    """
    Convert SMILES string to molecular feature vector.
    Returns None only if SMILES is truly unparsable.
    """

    try:
        smiles = smiles.strip()
        if not smiles:
            logger.warning("Empty SMILES string")
            return None

        # STEP 1: Parse SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.warning(f"Unparsable SMILES: {smiles}")
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Sanitization failed, continuing anyway: {e}")

        # STEP 2: Morgan Fingerprint
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=n_bits
            )
            fp_array = np.array(fingerprint, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Fingerprint failed, using zeros: {e}")
            fp_array = np.zeros(n_bits, dtype=np.float32)

        # STEP 3: Safe descriptor extraction
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
            safe_descriptor(Descriptors.NumRotatableBonds),
            safe_descriptor(Descriptors.RingCount),
            safe_descriptor(Descriptors.HeavyAtomCount),
            safe_descriptor(Descriptors.NHOHCount),
            safe_descriptor(Descriptors.NOCount),
            safe_descriptor(Descriptors.FractionCSP3),
            safe_descriptor(Descriptors.NumValenceElectrons),
        ], dtype=np.float32)

        # STEP 4: Normalize descriptors
        '''
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

        '''
        # STEP 5: Combine fingerprint + descriptors
        features = np.concatenate([fp_array, descriptors])

        return features.astype(np.float32)

    except Exception:
        logger.exception("Unexpected SMILES feature failure")
        return None


# ==========================================================
# VALIDATION
# ==========================================================

def validate_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


# ==========================================================
# STRUCTURAL FIX HELPERS
# ==========================================================

def fix_parentheses(smiles: str) -> str:
    open_count = smiles.count('(')
    close_count = smiles.count(')')

    if open_count == close_count:
        return smiles

    if open_count > close_count:
        return smiles + ')' * (open_count - close_count)

    if close_count > open_count:
        diff = close_count - open_count
        result = smiles.rstrip(')')
        expected_close = close_count - diff
        return result + ')' * expected_close

    return smiles


def fix_brackets(smiles: str) -> str:
    open_count = smiles.count('[')
    close_count = smiles.count(']')

    if open_count == close_count:
        return smiles

    if open_count > close_count:
        return smiles + ']' * (open_count - close_count)

    if close_count > open_count:
        diff = close_count - open_count
        result = smiles.rstrip(']')
        expected_close = close_count - diff
        return result + ']' * expected_close

    return smiles


# ==========================================================
# SMILES REPAIR PIPELINE
# ==========================================================

def repair_smiles(smiles: str, verbose: bool = False):
    if not smiles or not smiles.strip():
        return None

    smiles = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    try:
        fixed = fix_parentheses(smiles)
        fixed = fix_brackets(fixed)
        mol = Chem.MolFromSmiles(fixed)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                Chem.SanitizeMol(mol, catchErrors=True)
                return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    return None


# ==========================================================
# MOLECULAR PROPERTIES
# ==========================================================

def get_molecular_properties(smiles: str) -> dict:
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