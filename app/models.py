from pydantic import BaseModel, Field
from typing import List, Optional

import os
from dotenv import load_dotenv

# Load .env immediately
load_dotenv()

# FORCE Hugging Face cache locations (before transformers import)
os.environ["HF_HOME"] = "D:/hf-cache"
os.environ["HF_HUB_CACHE"] = "D:/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf-cache"

class GenerateRequest(BaseModel):
    """Request model for molecule generation"""
    disease: str = Field(..., description="Name of the disease to target")
    num_candidates: int = Field(default=3, ge=1, le=10, description="Number of molecules to generate")

class GenerateResponse(BaseModel):
    """Response model for molecule generation"""
    disease: str
    generated_molecules: List[str]
    count: int

class PredictRequest(BaseModel):
    """Request model for drug potential prediction"""
    smiles: str = Field(..., description="SMILES string of the molecule")

class PredictResponse(BaseModel):
    """Response model for drug potential prediction"""
    smiles: str
    score: float
    is_promising: bool
    confidence: str
    error: Optional[str] = None
    

class BatchGenerateRequest(BaseModel):
    """Request model for batch molecule generation"""
    diseases: List[str] = Field(..., description="List of diseases to target")
    num_candidates_per_disease: int = Field(default=3, ge=1, le=10, description="Number of molecules per disease")

class BatchGenerateResponse(BaseModel):
    """Response model for batch generation"""
    results: List[GenerateResponse]
    total_molecules: int

class MolecularProperties(BaseModel):
    """Molecular properties response"""
    smiles: str
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    h_bond_donors: Optional[int] = None
    h_bond_acceptors: Optional[int] = None
    tpsa: Optional[float] = None
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    num_rings: Optional[int] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    llm_loaded: bool
    qml_loaded: bool
    device: str