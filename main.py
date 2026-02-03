from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


# Add app directory to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your model
from inference import ModelInference

# FastAPI app
app = FastAPI(
    title="Drug Predictor API",
    description="Predict drug-likeness from SMILES strings using Quantum ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# ========================================
# REQUEST/RESPONSE MODELS
# ========================================

class GenerateRequest(BaseModel):
    disease: str = Field(..., description="Name of the disease")
    num_candidates: int = Field(3, description="Number of molecules to generate", ge=1, le=10)

class GenerateResponse(BaseModel):
    disease: str
    molecules: List[str]
    note: str = "Raw SMILES from generation - not validated"

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule", example="CCO")

class PredictResponse(BaseModel):
    smiles: str
    score: float
    is_promising: bool
    confidence: str
    original_smiles: Optional[str] = None
    repaired_smiles: Optional[str] = None
    sdf: Optional[str] = None
    error: Optional[str] = None
    

# Global model instance
model_inference = None

# ========================================
# LIFECYCLE EVENTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model_inference
    try:
        logger.info("Starting up... Loading models...")
        model_inference = ModelInference()
        logger.info("✓ Models loaded successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}", exc_info=True)
        # Don't raise - let the app start but return 503 for predictions

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    executor.shutdown(wait=True)

# ========================================
# HEALTH ENDPOINTS
# ========================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information"""
    return {
        "status": "ok",
        "message": "Drug Predictor API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "generate": "/generate - Generate SMILES for a disease (no validation)",
            "predict": "/predict - Predict drug potential (with validation)",
            "examples": "/examples"
        }
    }

@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model_inference is not None else "unhealthy",
        "models_loaded": model_inference is not None,
        "timestamp": datetime.now().isoformat()
    }

# ========================================
# GENERATION ENDPOINT (NO VALIDATION)
# ========================================

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_molecules_api(request: GenerateRequest):
    """
    Generate candidate molecules for a given disease.
    
    **Note**: Returns raw SMILES strings without validation or repair.
    Use the /predict endpoint to validate and score molecules.
    
    - **disease**: Name of the disease/condition
    - **num_candidates**: Number of molecules to generate (1-10)
    """
    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again in a moment."
        )

    try:
        loop = asyncio.get_event_loop()
        molecules = await loop.run_in_executor(
            executor,
            model_inference.generate_molecules,
            request.disease,
            request.num_candidates
        )
        
        return GenerateResponse(
            disease=request.disease,
            molecules=molecules,
            note="Raw SMILES from generation - not validated. Use /predict to validate and score."
        )
    except Exception as e:
        logger.error(f"Error generating molecules: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def smiles_to_3d_sdf(smiles: str) -> str | None:
    try:
        # 1️⃣ Parse WITHOUT sanitization
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # 2️⃣ Try sanitization in controlled way
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_AROMATICITY
            )
            Chem.Kekulize(mol, clearAromaticFlags=True)

        except Exception as e:
            logger.warning(f"Sanitization issue (continuing): {e}")

        # 3️⃣ Add hydrogens
        mol = Chem.AddHs(mol)

        # 4️⃣ Embed 3D (robust settings)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True

        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            logger.warning("ETKDG failed, retrying with basic embedding")
            result = AllChem.EmbedMolecule(mol)
        
        if result != 0:
            logger.error("3D embedding failed completely")
            return None


        # 5️⃣ Optimize geometry
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        # 6️⃣ Return SDF
        return Chem.MolToMolBlock(mol)

    except Exception as e:
        logger.error(f"3D generation failed completely: {e}", exc_info=True)
        return None


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_drug_potential(request: PredictRequest):

    if model_inference is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again in a moment."
        )

    try:
        logger.info(f"Predicting for SMILES: {request.smiles}")

        loop = asyncio.get_event_loop()

        # 🔥 RUN MODEL (THIS WAS MISSING)
        prediction = await loop.run_in_executor(
            executor,
            model_inference.predict_drug_potential,
            request.smiles
        )

        # 🔥 HANDLE ERROR CASE FIRST
        if "error" in prediction:
            return PredictResponse(
                smiles=prediction.get("original_smiles", request.smiles),
                score=prediction.get("score", 0.0),
                is_promising=prediction.get("is_promising", False),
                confidence=prediction.get("confidence", "low"),
                original_smiles=prediction.get("original_smiles"),
                repaired_smiles=prediction.get("repaired_smiles"),
                error=prediction.get("error"),
                sdf=None
            )

        # 🔥 CHOOSE FINAL SMILES FOR 3D
        render_smiles = (
            prediction.get("repaired_smiles")
            or prediction.get("original_smiles")
            or request.smiles
        )

        # 🔥 GENERATE 3D STRUCTURE
        sdf_3d = smiles_to_3d_sdf(render_smiles)

        if sdf_3d is None:
            logger.warning(f"3D generation failed for SMILES: {render_smiles}")

        return PredictResponse(
            smiles=render_smiles,
            score=prediction["score"],
            is_promising=prediction["is_promising"],
            confidence=prediction["confidence"],
            original_smiles=prediction.get("original_smiles"),
            repaired_smiles=prediction.get("repaired_smiles"),
            sdf=sdf_3d
        )
     


    except Exception as e:
        logger.error(f"Error in predict_drug_potential: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# EXAMPLES ENDPOINT
# ========================================

@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Get example molecules to test the API"""
    return {
        "generation_examples": [
            {"disease": "diabetes", "expected": "Molecules related to insulin/glucose regulation"},
            {"disease": "hypertension", "expected": "Molecules related to blood pressure"},
            {"disease": "cancer", "expected": "Potential anti-cancer compounds"},
        ],
        "prediction_examples": [
            {
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "description": "Common pain reliever and anti-inflammatory"
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "description": "Stimulant found in coffee and tea"
            },
            {
                "name": "Ibuprofen",
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "description": "Non-steroidal anti-inflammatory drug (NSAID)"
            },
            {
                "name": "Paracetamol (Acetaminophen)",
                "smiles": "CC(=O)NC1=CC=C(C=C1)O",
                "description": "Pain reliever and fever reducer"
            },
            {
                "name": "Ethanol",
                "smiles": "CCO",
                "description": "Simple alcohol - not a drug (should score low)"
            },
            {
                "name": "Benzene",
                "smiles": "c1ccccc1",
                "description": "Simple aromatic compound (should score low)"
            },
            {
                "name": "Invalid SMILES (will be repaired or rejected)",
                "smiles": "CCOHC=NC",
                "description": "Example of potentially invalid SMILES that will be validated"
            }
        ]
    }