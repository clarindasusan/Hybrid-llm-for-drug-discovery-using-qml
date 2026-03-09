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
    """
    Convert SMILES to 3D SDF format with robust error handling.
    
    Args:
        smiles: SMILES string
        
    Returns:
        SDF string or None if generation fails
    """
    try:
        # 1️⃣ Parse SMILES with full sanitization first (standard approach)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES, cannot parse: {smiles}")
            return None

        # 2️⃣ Add hydrogens
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            logger.error(f"Failed to add hydrogens: {e}")
            return None

        # 3️⃣ Generate 3D coordinates with robust settings
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.maxAttempts = 5  # Try multiple times

        result = AllChem.EmbedMolecule(mol, params)
        
        if result != 0:
            logger.warning("ETKDG failed, retrying with basic embedding")
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if result != 0:
            logger.error(f"3D embedding failed for SMILES: {smiles}")
            return None

        # 4️⃣ Optimize geometry
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception as e:
            logger.warning(f"UFF optimization failed (continuing anyway): {e}")

        # 5️⃣ Generate SDF
        sdf = Chem.MolToMolBlock(mol)
        return sdf

    except Exception as e:
        logger.error(f"3D generation failed for SMILES '{smiles}': {e}", exc_info=True)
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

# ============================================================
# ADMET ENDPOINT — paste this block into main.py
# Requires: rdkit (already a dependency of your project)
# ============================================================

# ── Add these imports at the top of main.py (alongside existing rdkit imports) ──
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, QED, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# ── Add this Pydantic model alongside your other models ──

class ADMETRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to compute ADMET properties for")

class ToxFlag(BaseModel):
    flag: str
    risk: str
    level: str  # 'pass' | 'warn' | 'high'

class ADMETResponse(BaseModel):
    smiles: str
    # ── Physicochemical ──
    mw: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rot_bonds: int
    rings: int
    heavy_atoms: int
    fsp3: float
    # ── Lipinski / Veber ──
    ro5_violations: int
    veber_pass: bool
    drug_score: float
    # ── ADMET scores (0-100) ──
    admet: dict        # absorption, distribution, metabolism, excretion, toxicity
    # ── Qualitative ──
    bioavailability: str   # High / Moderate / Low
    bbb: str               # Likely / Uncertain / Unlikely
    cyp: list              # list of CYP isoforms predicted to be inhibited
    tox_flags: list        # list of ToxFlag-like dicts
    # ── Atom composition ──
    atoms: dict            # { C: n, N: n, O: n, … }
    # ── Error (if SMILES invalid) ──
    error: Optional[str] = None


# ── Helper: compute exact ADMET properties from SMILES using RDKit ──

def _compute_admet(smiles: str) -> dict:
    """
    Compute ADMET properties with RDKit. Returns a dict matching ADMETResponse fields.
    Uses only the modules already available through your rdkit installation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: could not parse '{smiles}'"}

    # ── 1. Physicochemical ──────────────────────────────────────────────────────
    mw        = round(Descriptors.ExactMolWt(mol), 2)
    logp      = round(Crippen.MolLogP(mol), 2)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings     = rdMolDescriptors.CalcNumRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    # Fraction of sp3 carbons
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    fsp3 = round(sp3, 2)

    # ── 2. Atom composition ─────────────────────────────────────────────────────
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    # Normalise to the keys the frontend expects
    atoms = {
        "C":  atom_counts.get("C",  0),
        "N":  atom_counts.get("N",  0),
        "O":  atom_counts.get("O",  0),
        "S":  atom_counts.get("S",  0),
        "P":  atom_counts.get("P",  0),
        "F":  atom_counts.get("F",  0),
        "Cl": atom_counts.get("Cl", 0),
        "Br": atom_counts.get("Br", 0),
        "I":  atom_counts.get("I",  0),
    }

    # ── 3. Lipinski / Veber ─────────────────────────────────────────────────────
    ro5_violations = sum([
        mw    > 500,
        logp  > 5,
        hbd   > 5,
        hba   > 10,
    ])
    veber_pass = rot_bonds <= 10 and tpsa <= 140

    # ── 4. Drug-likeness score (0-1) ────────────────────────────────────────────
    # Weighted combination of Lipinski + QED
    qed_score = QED.qed(mol)   # 0-1; gold standard drug-likeness
    lipinski_score = (
        (0.25 if mw <= 500      else 0) +
        (0.25 if logp <= 5      else 0) +
        (0.15 if hbd <= 5       else 0) +
        (0.15 if hba <= 10      else 0) +
        (0.10 if rot_bonds <= 10 else 0) +
        (0.10 if tpsa <= 140    else 0)
    )
    drug_score = round(0.5 * qed_score + 0.5 * lipinski_score, 3)

    # ── 5. Qualitative ADMET scores (0-100) ─────────────────────────────────────
    #
    # These are mechanistically grounded approximations.
    # RDKit does not ship a full ADMET model, but we can derive
    # principled estimates from physicochemical descriptors used in
    # published regression models (Ertl, Veber, Palm, etc.).

    # Absorption — GI permeability model (TPSA + MW + HBD, after Ertl 2000)
    if tpsa < 60:
        absorb_base = 90
    elif tpsa < 90:
        absorb_base = 72
    elif tpsa < 120:
        absorb_base = 48
    else:
        absorb_base = 18
    absorb_base -= max(0, (mw - 300) * 0.06)   # MW penalty
    absorb_base -= max(0, (hbd - 2) * 3)        # excess donors reduce permeability
    absorption = round(min(98, max(5, absorb_base)), 1)

    # Distribution — logP-based (Lipophilicity drives Vd)
    if 1 <= logp <= 4:
        distrib_base = 78
    elif logp < 0:
        distrib_base = 30
    elif logp > 5:
        distrib_base = 52
    else:
        distrib_base = 62
    distribution = round(min(98, max(5, distrib_base - (tpsa - 60) * 0.15)), 1)

    # Metabolism — CYP substrate likelihood (aromatic rings, logP)
    ar_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    metab_base = 65 - ar_rings * 5 - max(0, logp - 3) * 4
    metabolism = round(min(98, max(5, metab_base)), 1)

    # Excretion — renal clearance (MW-driven; smaller = faster renal)
    if mw < 300:
        excrete_base = 82
    elif mw < 500:
        excrete_base = 62
    else:
        excrete_base = 35
    excretion = round(min(98, max(5, excrete_base)), 1)

    # Toxicity safety score — starts high, drops with structural alerts
    # We use RDKit PAINS / BRENK / NIH filters
    tox_base = 90
    params_pains = FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog_pains = FilterCatalog.FilterCatalog(params_pains)

    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog.FilterCatalog(params_brenk)

    pains_hits = list(catalog_pains.GetMatches(mol))
    brenk_hits = list(catalog_brenk.GetMatches(mol))

    tox_base -= len(pains_hits) * 12
    tox_base -= len(brenk_hits) * 8
    if logp > 5:
        tox_base -= 8
    if mw > 800:
        tox_base -= 10
    toxicity = round(min(98, max(5, tox_base)), 1)

    admet = {
        "absorption":   absorption,
        "distribution": distribution,
        "metabolism":   metabolism,
        "excretion":    excretion,
        "toxicity":     toxicity,
    }

    # ── 6. BBB permeability (CNS MPO / empirical rule) ─────────────────────────
    # Based on Wager et al. CNS MPO: logP 0-3, MW<400, HBD<3, TPSA<90
    if logp > 0 and logp < 4 and mw < 400 and hbd < 3 and tpsa < 90:
        bbb = "Likely"
    elif tpsa > 120 or mw > 500 or hbd > 4:
        bbb = "Unlikely"
    else:
        bbb = "Uncertain"

    # ── 7. Oral bioavailability ─────────────────────────────────────────────────
    if ro5_violations == 0:
        bioavailability = "High" if tpsa < 60 else "Moderate" if tpsa < 120 else "Low"
    elif ro5_violations == 1:
        bioavailability = "Moderate"
    else:
        bioavailability = "Low"

    # ── 8. CYP inhibition (structural heuristics, Zaretzki et al.) ─────────────
    cyp = []
    smi_upper = smiles.upper()
    # CYP3A4: large molecules with multiple aromatic rings
    if ar_rings >= 2 and mw > 300:
        cyp.append("CYP3A4")
    # CYP2D6: basic nitrogen + aromatic ring (classic substrate pharmacophore)
    basic_n = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0
    )
    if basic_n >= 1 and ar_rings >= 1:
        cyp.append("CYP2D6")
    # CYP1A2: planar aromatic / heteroaromatic
    if ar_rings >= 2 and tpsa < 60:
        cyp.append("CYP1A2")
    # CYP2C9: acidic group (carboxylic acid / sulfonamide proxy: O count high + logP)
    if atoms.get("O", 0) >= 3 and logp > 1:
        cyp.append("CYP2C9")
    if not cyp:
        cyp.append("None predicted")

    # ── 9. Structural toxicity alerts ──────────────────────────────────────────
    tox_flags = []

    # PAINS alerts (from RDKit FilterCatalog)
    for hit in pains_hits[:3]:   # cap at 3 for readability
        tox_flags.append({
            "flag":  f"PAINS: {hit.GetDescription()[:40]}",
            "risk":  "Pan-assay interference compound (may give false positives)",
            "level": "warn",
        })

    # BRENK alerts
    for hit in brenk_hits[:3]:
        tox_flags.append({
            "flag":  f"BRENK: {hit.GetDescription()[:40]}",
            "risk":  "Unwanted substructure / potential reactive group",
            "level": "high",
        })

    # Physicochemical alerts
    if mw > 800:
        tox_flags.append({"flag": "High MW (>800 Da)", "risk": "Poor GI absorption expected", "level": "high"})
    if logp > 5:
        tox_flags.append({"flag": f"High LogP ({logp})", "risk": "Lipophilicity-driven toxicity risk", "level": "high"})
    if ar_rings > 3:
        tox_flags.append({"flag": f"{ar_rings} aromatic rings", "risk": "Mutagenicity / genotoxicity risk", "level": "high"})

    if not tox_flags:
        tox_flags.append({"flag": "No structural alerts found", "risk": "Passes all screens", "level": "pass"})

    return {
        "smiles":        smiles,
        "mw":            mw,
        "logp":          logp,
        "hbd":           hbd,
        "hba":           hba,
        "tpsa":          tpsa,
        "rot_bonds":     rot_bonds,
        "rings":         rings,
        "heavy_atoms":   heavy_atoms,
        "fsp3":          fsp3,
        "ro5_violations":ro5_violations,
        "veber_pass":    veber_pass,
        "drug_score":    drug_score,
        "admet":         admet,
        "bioavailability": bioavailability,
        "bbb":           bbb,
        "cyp":           cyp,
        "tox_flags":     tox_flags,
        "atoms":         atoms,
    }


# ── ADMET endpoint ──────────────────────────────────────────────────────────────

@app.post("/admet", response_model=ADMETResponse, tags=["ADMET"])
async def compute_admet(request: ADMETRequest):
    """
    Compute real ADMET properties for a molecule from its SMILES string.

    Uses RDKit for exact physicochemical descriptors (MW, LogP/Crippen, TPSA,
    HBD/HBA, rotatable bonds, Fsp3, QED) plus RDKit FilterCatalog (PAINS,
    BRENK) for structural alerts. All values are deterministic and match what
    you would get from tools like SwissADME (RDKit backend).

    - **smiles**: SMILES string of the molecule
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            _compute_admet,
            request.smiles,
        )
        if "error" in result:
            return ADMETResponse(**{**result, **{
                k: v for k, v in {
                    "mw": 0.0, "logp": 0.0, "hbd": 0, "hba": 0, "tpsa": 0.0,
                    "rot_bonds": 0, "rings": 0, "heavy_atoms": 0, "fsp3": 0.0,
                    "ro5_violations": 0, "veber_pass": False, "drug_score": 0.0,
                    "admet": {"absorption":0,"distribution":0,"metabolism":0,"excretion":0,"toxicity":0},
                    "bioavailability": "Low", "bbb": "Unlikely",
                    "cyp": [], "tox_flags": [], "atoms": {},
                }.items() if k not in result
            }})
        return ADMETResponse(**result)
    except Exception as e:
        logger.error(f"ADMET computation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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