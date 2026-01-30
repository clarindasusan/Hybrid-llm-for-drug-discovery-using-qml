import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import logging
import re

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


def fix_parentheses(smiles: str) -> str:
    """
    Attempt to fix mismatched parentheses in SMILES string.
    
    Args:
        smiles: SMILES string with potential parentheses issues
        
    Returns:
        SMILES string with balanced parentheses
    """
    # Count parentheses
    open_count = smiles.count('(')
    close_count = smiles.count(')')
    
    if open_count == close_count:
        return smiles
    
    # More opening than closing - add closing parentheses
    if open_count > close_count:
        return smiles + ')' * (open_count - close_count)
    
    # More closing than opening - try to remove extra closing or add opening
    # This is trickier, try removing trailing closing parens
    if close_count > open_count:
        diff = close_count - open_count
        # Remove extra closing parentheses from the end
        result = smiles.rstrip(')')
        expected_close = close_count - diff
        result = result + ')' * expected_close
        return result
    
    return smiles


def fix_brackets(smiles: str) -> str:
    """
    Attempt to fix mismatched brackets in SMILES string.
    
    Args:
        smiles: SMILES string with potential bracket issues
        
    Returns:
        SMILES string with balanced brackets
    """
    open_count = smiles.count('[')
    close_count = smiles.count(']')
    
    if open_count == close_count:
        return smiles
    
    if open_count > close_count:
        return smiles + ']' * (open_count - close_count)
    
    if close_count > open_count:
        # Remove extra closing brackets from the end
        diff = close_count - open_count
        result = smiles.rstrip(']')
        expected_close = close_count - diff
        result = result + ']' * expected_close
        return result
    
    return smiles


def truncate_invalid_suffix(smiles: str) -> str:
    """
    If SMILES has a clearly incomplete suffix, try to truncate it.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Truncated SMILES if applicable
    """
    # If it ends with an opening parenthesis or bracket, remove incomplete part
    if smiles.endswith('('):
        return smiles[:-1]
    if smiles.endswith('['):
        return smiles[:-1]
    
    # Find the last valid ring closure or complete structure
    # This is a heuristic - look for common patterns
    for i in range(len(smiles) - 1, 0, -1):
        test_smiles = smiles[:i]
        try:
            mol = Chem.MolFromSmiles(test_smiles, sanitize=False)
            if mol is not None:
                return test_smiles
        except:
            continue
    
    return smiles


def repair_smiles(smiles: str, verbose: bool = False):
    """
    Attempt to repair and validate SMILES strings with multiple fallback strategies.
    
    This function tries multiple strategies including structural repair:
    1. Direct parsing (most SMILES are already valid)
    2. Fix parentheses/brackets
    3. Remove stereochemistry
    4. Truncate incomplete suffixes
    5. Parse without sanitization
    6. InChI round-trip
    
    Args:
        smiles: Input SMILES string
        verbose: If True, log repair attempts
        
    Returns:
        Canonical SMILES string if valid/repairable, None otherwise
    """
    if not smiles or not smiles.strip():
        return None
    
    smiles = smiles.strip()
    
    # Strategy 1: Try direct parsing (most SMILES are already valid)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if verbose:
                logger.info(f"✓ SMILES valid as-is: {smiles[:50]}")
            return canonical
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Fix structural issues (parentheses, brackets)
    try:
        # Fix parentheses
        fixed = fix_parentheses(smiles)
        fixed = fix_brackets(fixed)
        
        if fixed != smiles:
            mol = Chem.MolFromSmiles(fixed)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if verbose:
                    logger.info(f"✓ Repaired by fixing parentheses/brackets: {smiles[:50]}")
                return canonical
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Parse without sanitization, then sanitize carefully
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            # Try full sanitization
            try:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if verbose:
                    logger.info(f"✓ Repaired with sanitization: {smiles[:50]}")
                return canonical
            except Exception:
                # If full sanitization fails, try partial
                try:
                    Chem.SanitizeMol(mol, catchErrors=True)
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    if verbose:
                        logger.info(f"✓ Repaired with partial sanitization: {smiles[:50]}")
                    return canonical
                except Exception:
                    pass
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Try removing stereochemistry markers
    if '/' in smiles or '\\' in smiles or '@' in smiles:
        try:
            cleaned = smiles.replace('/', '').replace('\\', '').replace('@', '')
            # Also try fixing parentheses on cleaned version
            cleaned = fix_parentheses(cleaned)
            cleaned = fix_brackets(cleaned)
            
            mol = Chem.MolFromSmiles(cleaned)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if verbose:
                    logger.info(f"✓ Repaired by removing stereochemistry: {smiles[:50]}")
                return canonical
        except Exception as e:
            if verbose:
                logger.debug(f"Strategy 4 failed: {e}")
    
    # Strategy 5: Try truncating incomplete suffix
    try:
        truncated = truncate_invalid_suffix(smiles)
        if truncated != smiles:
            # Also fix parentheses on truncated version
            truncated = fix_parentheses(truncated)
            truncated = fix_brackets(truncated)
            
            mol = Chem.MolFromSmiles(truncated)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if verbose:
                    logger.info(f"✓ Repaired by truncation: {smiles[:50]} -> {truncated[:50]}")
                return canonical
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 5 failed: {e}")
    
    # Strategy 6: Try fixing common notation issues
    try:
        # Remove spaces and fix double equals
        cleaned = smiles.replace(' ', '').replace('(=O)(=O)', '(=O)')
        cleaned = fix_parentheses(cleaned)
        cleaned = fix_brackets(cleaned)
        
        mol = Chem.MolFromSmiles(cleaned, sanitize=False)
        if mol is not None:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                mol = Chem.RemoveHs(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if verbose:
                    logger.info(f"✓ Repaired with kekulization: {smiles[:50]}")
                return canonical
            except Exception:
                try:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    if canonical:
                        if verbose:
                            logger.info(f"✓ Repaired without kekulization: {smiles[:50]}")
                        return canonical
                except Exception:
                    pass
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 6 failed: {e}")
    
    # Strategy 7: Try InChI round-trip (last resort)
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            inchi = Chem.MolToInchi(mol)
            if inchi:
                mol_from_inchi = Chem.MolFromInchi(inchi)
                if mol_from_inchi is not None:
                    canonical = Chem.MolToSmiles(mol_from_inchi, canonical=True)
                    if verbose:
                        logger.info(f"✓ Repaired via InChI: {smiles[:50]}")
                    return canonical
    except Exception as e:
        if verbose:
            logger.debug(f"Strategy 7 failed: {e}")
    
    # All strategies failed
    if verbose:
        logger.warning(f"✗ Could not repair SMILES: {smiles[:50]}")
    return None


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


def get_smiles_info(smiles: str) -> dict:
    """
    Get diagnostic information about a SMILES string.
    Useful for debugging validation issues.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with validation info and repair status
    """
    result = {
        "original": smiles,
        "is_valid": False,
        "can_parse": False,
        "can_sanitize": False,
        "has_paren_issues": False,
        "has_bracket_issues": False,
        "repaired": None,
        "error": None
    }
    
    try:
        # Check for structural issues
        result["has_paren_issues"] = smiles.count('(') != smiles.count(')')
        result["has_bracket_issues"] = smiles.count('[') != smiles.count(']')
        
        # Can we parse it?
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        result["can_parse"] = mol is not None
        
        if mol is not None:
            # Can we sanitize it?
            try:
                Chem.SanitizeMol(mol)
                result["can_sanitize"] = True
                result["is_valid"] = True
            except Exception as e:
                result["error"] = str(e)
        
        # Try repair
        repaired = repair_smiles(smiles, verbose=True)
        result["repaired"] = repaired
        
    except Exception as e:
        result["error"] = str(e)
    
    return result