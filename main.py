from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys
from pathlib import Path

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

# Pydantic models
class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule", example="CCO")

class PredictResponse(BaseModel):
    smiles: str
    score: float
    is_promising: bool
    confidence: str

# Global model instance
model_inference = None

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
            "predict": "/predict",
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
from typing import List

class GenerateRequest(BaseModel):
    disease: str = Field(..., description="Name of the disease")
    num_candidates: int = Field(3, description="Number of molecules to generate", ge=1, le=10)

class GenerateResponse(BaseModel):
    disease: str
    molecules: List[str]

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_molecules_api(request: GenerateRequest):
    """Generate candidate molecules for a given disease"""
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
        return GenerateResponse(disease=request.disease, molecules=molecules)
    except Exception as e:
        logger.error(f"Error generating molecules: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_drug_potential(request: PredictRequest):
    """
    Predict drug potential for a given molecule
    
    - **smiles**: SMILES string representation of the molecule
    
    Returns probability score, classification, and confidence level.
    """
    if model_inference is None:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please try again in a moment."
        )
    
    try:
        logger.info(f"Predicting for SMILES: {request.smiles}")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            executor,
            model_inference.predict_drug_potential,
            request.smiles
        )
        
        # Check for errors
        if "error" in prediction:
            if "Could not generate features" in prediction.get("error", ""):
                raise HTTPException(status_code=400, detail=prediction["error"])
            else:
                raise HTTPException(status_code=500, detail=prediction["error"])
        
        return PredictResponse(
            smiles=request.smiles,
            score=prediction["score"],
            is_promising=prediction["is_promising"],
            confidence=prediction["confidence"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_drug_potential: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Get example molecules to test the API"""
    return {
        "examples": [
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
            }
        ]
    }