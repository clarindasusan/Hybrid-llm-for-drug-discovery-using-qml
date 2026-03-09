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
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from app.utils import repair_smiles 

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from inference import ModelInference

app = FastAPI(
    title="Drug Predictor API",
    description="Predict drug-likeness from SMILES strings using Quantum ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ── Request / Response Models ─────────────────────────────────────────────────

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

class ADMETRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to compute ADMET properties for")

# ── Global model ──────────────────────────────────────────────────────────────

model_inference = None

@app.on_event("startup")
async def startup_event():
    global model_inference
    try:
        logger.info("Loading models...")
        model_inference = ModelInference()
        logger.info("✓ Models loaded successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    executor.shutdown(wait=True)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "Drug Predictor API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {"/generate": "Generate SMILES", "/predict": "Predict + validate", "/admet": "ADMET properties"}
    }

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy" if model_inference is not None else "unhealthy",
        "models_loaded": model_inference is not None,
        "timestamp": datetime.now().isoformat()
    }

# ── Generate ──────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_molecules_api(request: GenerateRequest):
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again in a moment.")
    try:
        loop = asyncio.get_event_loop()
        molecules = await loop.run_in_executor(executor, model_inference.generate_molecules, request.disease, request.num_candidates)
        return GenerateResponse(disease=request.disease, molecules=molecules, note="Raw SMILES - use /predict to validate.")
    except Exception as e:
        logger.error(f"Error generating molecules: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ── 3D SDF helper ─────────────────────────────────────────────────────────────

def smiles_to_3d_sdf(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        params.maxAttempts = 5
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        logger.error(f"3D generation failed: {e}")
        return None

# ── Predict ───────────────────────────────────────────────────────────────────



@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_drug_potential(request: PredictRequest):
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    try:
        original_smiles = request.smiles

        # ── Repair SMILES before doing anything else ──
        repaired = repair_smiles(original_smiles)
        if repaired is None:
            return PredictResponse(
                smiles=original_smiles,
                score=0.0,
                is_promising=False,
                confidence="low",
                original_smiles=original_smiles,
                repaired_smiles=None,
                error="SMILES could not be parsed or repaired",
                sdf=None
            )

        # Use the repaired canonical SMILES from here on
        smiles_to_use = repaired

        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            executor,
            model_inference.predict_drug_potential,
            smiles_to_use
        )

        if "error" in prediction:
            return PredictResponse(
                smiles=original_smiles,
                score=prediction.get("score", 0.0),
                is_promising=False,
                confidence="low",
                original_smiles=original_smiles,
                repaired_smiles=repaired if repaired != original_smiles else None,
                error=prediction["error"],
                sdf=None
            )

        render_smiles = repaired
        sdf_3d = smiles_to_3d_sdf(render_smiles)

        return PredictResponse(
            smiles=render_smiles,
            score=prediction["score"],
            is_promising=prediction["is_promising"],
            confidence=prediction["confidence"],
            original_smiles=original_smiles,
            repaired_smiles=repaired if repaired != original_smiles else None,
            sdf=sdf_3d
        )

    except Exception as e:
        logger.error(f"Error in predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ── ADMET ─────────────────────────────────────────────────────────────────────
# NOTE: Does NOT require model_inference — pure RDKit, always works

def _compute_admet(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: could not parse '{smiles}'"}

    mw         = round(Descriptors.ExactMolWt(mol), 2)
    logp       = round(Crippen.MolLogP(mol), 2)
    hbd        = rdMolDescriptors.CalcNumHBD(mol)
    hba        = rdMolDescriptors.CalcNumHBA(mol)
    tpsa       = round(rdMolDescriptors.CalcTPSA(mol), 1)
    rot_bonds  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings      = rdMolDescriptors.CalcNumRings(mol)
    heavy_atoms= mol.GetNumHeavyAtoms()
    fsp3       = round(rdMolDescriptors.CalcFractionCSP3(mol), 2)
    qed_score  = QED.qed(mol)
    ar_rings   = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Atom composition
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    atoms = {k: atom_counts.get(k, 0) for k in ["C","N","O","S","P","F","Cl","Br","I"]}

    # Lipinski / Veber
    ro5_violations = sum([mw>500, logp>5, hbd>5, hba>10])
    veber_pass     = rot_bonds <= 10 and tpsa <= 140

    # Drug score
    lipinski_score = (
        (0.25 if mw<=500 else 0)+(0.25 if logp<=5 else 0)+
        (0.15 if hbd<=5 else 0)+(0.15 if hba<=10 else 0)+
        (0.10 if rot_bonds<=10 else 0)+(0.10 if tpsa<=140 else 0)
    )
    drug_score = round(0.5*qed_score + 0.5*lipinski_score, 3)

    # ADMET scores
    if tpsa<60:   absorb_base=90
    elif tpsa<90: absorb_base=72
    elif tpsa<120:absorb_base=48
    else:         absorb_base=18
    absorb_base -= max(0,(mw-300)*0.06) + max(0,(hbd-2)*3)
    absorption = round(min(98,max(5,absorb_base)),1)

    if 1<=logp<=4:   distrib_base=78
    elif logp<0:     distrib_base=30
    elif logp>5:     distrib_base=52
    else:            distrib_base=62
    distribution = round(min(98,max(5,distrib_base-(tpsa-60)*0.15)),1)

    metab_base  = 65 - ar_rings*5 - max(0,logp-3)*4
    metabolism  = round(min(98,max(5,metab_base)),1)

    if mw<300:   excrete_base=82
    elif mw<500: excrete_base=62
    else:        excrete_base=35
    excretion = round(min(98,max(5,excrete_base)),1)

    # PAINS + BRENK for toxicity
    params_p = FilterCatalogParams(); params_p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params_b = FilterCatalogParams(); params_b.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    cat_p = FilterCatalog.FilterCatalog(params_p)
    cat_b = FilterCatalog.FilterCatalog(params_b)
    pains_hits = list(cat_p.GetMatches(mol))
    brenk_hits = list(cat_b.GetMatches(mol))

    tox_base = 90 - len(pains_hits)*12 - len(brenk_hits)*8
    if logp>5: tox_base-=8
    if mw>800: tox_base-=10
    toxicity = round(min(98,max(5,tox_base)),1)

    # BBB
    if logp>0 and logp<4 and mw<400 and hbd<3 and tpsa<90: bbb="Likely"
    elif tpsa>120 or mw>500 or hbd>4: bbb="Unlikely"
    else: bbb="Uncertain"

    # Bioavailability
    if ro5_violations==0:
        bioavailability = "High" if tpsa<60 else "Moderate" if tpsa<120 else "Low"
    elif ro5_violations==1: bioavailability="Moderate"
    else: bioavailability="Low"

    # CYP
    cyp=[]
    basic_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==7 and a.GetTotalNumHs()>0)
    if ar_rings>=2 and mw>300:  cyp.append("CYP3A4")
    if basic_n>=1 and ar_rings>=1: cyp.append("CYP2D6")
    if ar_rings>=2 and tpsa<60: cyp.append("CYP1A2")
    if atoms.get("O",0)>=3 and logp>1: cyp.append("CYP2C9")
    if not cyp: cyp.append("None predicted")

    # Tox flags
    tox_flags=[]
    for h in pains_hits[:3]:
        tox_flags.append({"flag":f"PAINS: {h.GetDescription()[:40]}","risk":"Pan-assay interference compound","level":"warn"})
    for h in brenk_hits[:3]:
        tox_flags.append({"flag":f"BRENK: {h.GetDescription()[:40]}","risk":"Unwanted substructure / potential reactive group","level":"high"})
    if mw>800:   tox_flags.append({"flag":"High MW (>800 Da)","risk":"Poor GI absorption expected","level":"high"})
    if logp>5:   tox_flags.append({"flag":f"High LogP ({logp})","risk":"Lipophilicity-driven toxicity risk","level":"high"})
    if ar_rings>3: tox_flags.append({"flag":f"{ar_rings} aromatic rings","risk":"Mutagenicity / genotoxicity risk","level":"high"})
    if not tox_flags:
        tox_flags.append({"flag":"No structural alerts found","risk":"Passes all screens","level":"pass"})

    return {
        "smiles":smiles,"mw":mw,"logp":logp,"hbd":hbd,"hba":hba,
        "tpsa":tpsa,"rot_bonds":rot_bonds,"rings":rings,
        "heavy_atoms":heavy_atoms,"fsp3":fsp3,
        "ro5_violations":ro5_violations,"veber_pass":veber_pass,
        "drug_score":drug_score,
        "admet":{"absorption":absorption,"distribution":distribution,
                 "metabolism":metabolism,"excretion":excretion,"toxicity":toxicity},
        "bioavailability":bioavailability,"bbb":bbb,"cyp":cyp,
        "tox_flags":tox_flags,"atoms":atoms,"error":None,
    }


@app.post("/admet", tags=["ADMET"])
async def compute_admet(request: ADMETRequest):
    """
    Compute ADMET properties using RDKit — does NOT require ML model to be loaded.
    Works immediately on startup.
    """
    # ✅ NO model_inference check — ADMET is pure RDKit, always available
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _compute_admet, request.smiles)
        return result
    except Exception as e:
        logger.error(f"ADMET error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Examples ──────────────────────────────────────────────────────────────────

@app.get("/examples", tags=["Examples"])
async def get_examples():
    return {
        "prediction_examples": [
            {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"name":"Caffeine","smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
            {"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
            {"name":"Paracetamol","smiles":"CC(=O)NC1=CC=C(C=C1)O"},
        ]
    }