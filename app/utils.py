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

def smiles_to_features(smiles: str, n_features: int = 2053) -> np.ndarray:
    """
    Convert SMILES string to molecular feature vector
    
    Args:
        smiles: SMILES string
        n_features: Expected number of features (must match training)
        
    Returns:
        Feature vector as numpy array, or None if invalid
    """
    try:
        # Strip whitespace
        smiles = smiles.strip()
        
        if not smiles:
            logger.warning("Empty SMILES string provided")
            return None
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            logger.warning(f"Invalid SMILES (RDKit parsing failed): {smiles}")
            return None
        
        # Sanitize molecule (catches some edge cases)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Molecule sanitization failed for {smiles}: {e}")
            return None
        
        # Generate Morgan fingerprint (2048 bits by default)
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.array(fingerprint)
        except Exception as e:
            logger.error(f"Fingerprint generation failed for {smiles}: {e}")
            return None
        
        # Calculate molecular descriptors (5 descriptors)
        try:
            descriptors = np.array([
                Descriptors.MolWt(mol),              # Molecular weight
                Descriptors.MolLogP(mol),            # LogP (lipophilicity)
                Descriptors.NumHDonors(mol),         # Hydrogen bond donors
                Descriptors.NumHAcceptors(mol),      # Hydrogen bond acceptors
                Descriptors.TPSA(mol)                # Topological polar surface area
            ])
        except Exception as e:
            logger.error(f"Descriptor calculation failed for {smiles}: {e}")
            return None
        
        # Normalize descriptors (simple min-max scaling)
        # These ranges are typical for drug-like molecules
        descriptor_ranges = np.array([
            [0, 1000],    # MolWt range
            [-5, 10],     # LogP range
            [0, 10],      # HDonors range
            [0, 15],      # HAcceptors range
            [0, 200]      # TPSA range
        ])
        
        normalized_descriptors = np.zeros(5)
        for i in range(5):
            min_val, max_val = descriptor_ranges[i]
            normalized_descriptors[i] = (descriptors[i] - min_val) / (max_val - min_val)
            # Clip to [0, 1]
            normalized_descriptors[i] = np.clip(normalized_descriptors[i], 0, 1)
        
        # Combine fingerprint and descriptors
        features = np.concatenate([fp_array, normalized_descriptors])
        
        logger.info(f"Generated features of length {len(features)} for SMILES: {smiles}")
        
        # Ensure correct length (pad or truncate if needed)
        if len(features) < n_features:
            # Pad with zeros
            features = np.pad(features, (0, n_features - len(features)), mode='constant')
        elif len(features) > n_features:
            # Truncate
            features = features[:n_features]
        
        return features.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error converting SMILES to features: {str(e)}", exc_info=True)
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