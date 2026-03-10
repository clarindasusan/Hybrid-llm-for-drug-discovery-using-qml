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
from explainer import MoleculeExplainer


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
molecule_explainer = None


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

# ════════════════════════════════════════════════════════════════════════════════
# Replace your existing /generate endpoint in main.py with this version.
# The only change is handling the new list-of-dicts format from generate_molecules()
# and passing the "source" field through to the frontend.
# ════════════════════════════════════════════════════════════════════════════════

# ── Updated Pydantic models ───────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    disease: str = Field(..., description="Disease name to generate candidates for")
    num_candidates: int = Field(default=3, ge=1, le=10)

class MoleculeCandidate(BaseModel):
    smiles:          str
    score:           float
    is_promising:    bool
    confidence:      str
    prediction:      str
    source:          str            # "generated" | "fallback"
    original_smiles: Optional[str] = None
    repaired_smiles: Optional[str] = None
    error:           Optional[str] = None

class GenerateResponse(BaseModel):
    disease:    str
    candidates: List[MoleculeCandidate]
    generated:  int                 # count of LLM-generated (not fallback)
    fallback:   int                 # count of curated fallbacks used
    timestamp:  str


# ── Updated /generate endpoint ────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_drug_candidates(request: GenerateRequest):
    """
    Generate drug candidate SMILES for a disease using BioGPT + LoRA,
    then score each with the QML model.

    Each candidate includes a `source` field:
    - **generated**: produced by the LLM and validated
    - **fallback**: from the curated library (used when LLM output is invalid)
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        loop = asyncio.get_event_loop()

        # generate_molecules now returns list of {"smiles": str, "source": str}
        molecule_dicts = await loop.run_in_executor(
            executor,
            lambda: model_inference.generate_molecules(
                request.disease,
                request.num_candidates
            )
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # ── Score each candidate ──────────────────────────────────────────────
    candidates = []
    for item in molecule_dicts:
        smiles = item["smiles"]
        source = item["source"]
        try:
            prediction = await loop.run_in_executor(
                executor,
                model_inference.predict_drug_potential,
                smiles
            )
            candidates.append(MoleculeCandidate(
                smiles=smiles,
                score=prediction.get("score",        0.0),
                is_promising=prediction.get("is_promising", False),
                confidence=prediction.get("confidence",   "low"),
                prediction=prediction.get("prediction",   "unknown"),
                source=source,
                original_smiles=prediction.get("original_smiles"),
                repaired_smiles=prediction.get("repaired_smiles"),
                error=prediction.get("error"),
            ))
        except Exception as e:
            logger.error(f"Scoring failed for {smiles}: {e}")
            candidates.append(MoleculeCandidate(
                smiles=smiles,
                score=0.0,
                is_promising=False,
                confidence="low",
                prediction="unknown",
                source=source,
                error=str(e),
            ))

    n_generated = sum(1 for c in candidates if c.source == "generated")
    n_fallback  = sum(1 for c in candidates if c.source == "fallback")

    return GenerateResponse(
        disease=request.disease,
        candidates=candidates,
        generated=n_generated,
        fallback=n_fallback,
        timestamp=datetime.utcnow().isoformat(),
    )

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
    cat_p = FilterCatalog(params_p)
    cat_b = FilterCatalog(params_b)
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






# ── 2. New global variable (alongside model_inference = None) ─────────────────


# ── 3. Update startup_event to initialise the explainer ──────────────────────
@app.on_event("startup")
async def startup_event():
    global model_inference, molecule_explainer
    try:
        logger.info("Loading models...")
        model_inference = ModelInference()
        logger.info("✓ Models loaded successfully!")

        # Initialise explainer — background computation happens lazily
        # on first /explain call, not here, to keep startup fast
        molecule_explainer = MoleculeExplainer(model_inference)
        logger.info("✓ Explainer ready!")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}", exc_info=True)


# ── 4. New Pydantic models ────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to explain")
    include_admet: bool = Field(
        True, description="Compute ADMET data to enrich the explanation text"
    )

class DescriptorContribution(BaseModel):
    name:      str
    label:     str
    unit:      str
    ideal:     str
    value:     float
    shap:      float
    direction: str   # 'positive' | 'negative' | 'neutral'
    magnitude: float

class FingerprintContribution(BaseModel):
    bit:       int
    shap:      float
    direction: str
    atoms:     List[int]
    present:   bool

class ExplainResponse(BaseModel):
    smiles:                    str
    original_smiles:           Optional[str]
    repaired_smiles:           Optional[str]
    score:                     float
    shap_base_value:           float
    descriptor_contributions:  List[DescriptorContribution]
    fingerprint_contributions: List[FingerprintContribution]
    important_atoms:           List[int]
    explanation_text:          str
    confidence:                str
    error:                     Optional[str] = None


# ── 5. The /explain endpoint ──────────────────────────────────────────────────

@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain_prediction(request: ExplainRequest):
    """
    Explain a drug-likeness prediction using SHAP (SHapley Additive exPlanations).

    Returns:
    - **descriptor_contributions**: how each RDKit descriptor pushed the score up/down
    - **fingerprint_contributions**: which Morgan fingerprint bits mattered most
    - **important_atoms**: atom indices to highlight in the 3D viewer
    - **explanation_text**: plain-English summary of the prediction
    - **confidence**: high / medium / low based on SHAP value spread

    The QML model is treated as a black box — SHAP perturbs the input features
    and measures how the output changes, giving model-agnostic explanations.
    """
    if molecule_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explainer not initialised. Models may still be loading."
        )

    original_smiles = request.smiles

    # ── Repair SMILES ─────────────────────────────────────────────────────────
    repaired = repair_smiles(original_smiles)
    if repaired is None:
        return ExplainResponse(
            smiles=original_smiles,
            original_smiles=original_smiles,
            repaired_smiles=None,
            score=0.0,
            shap_base_value=0.0,
            descriptor_contributions=[],
            fingerprint_contributions=[],
            important_atoms=[],
            explanation_text="Could not parse or repair the SMILES string.",
            confidence="low",
            error="SMILES could not be parsed or repaired",
        )

    smiles_to_use   = repaired
    repaired_field  = repaired if repaired != original_smiles else None

    # ── Optionally compute ADMET for richer explanation text ─────────────────
    admet_data = None
    if request.include_admet:
        try:
            admet_data = _compute_admet(smiles_to_use)
            if admet_data.get("error"):
                admet_data = None
        except Exception:
            admet_data = None

    # ── Run SHAP explanation (CPU-bound → thread pool) ────────────────────────
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: molecule_explainer.explain(smiles_to_use, admet_data)
        )
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result and result["error"]:
        return ExplainResponse(
            smiles=smiles_to_use,
            original_smiles=original_smiles,
            repaired_smiles=repaired_field,
            score=0.0,
            shap_base_value=0.0,
            descriptor_contributions=[],
            fingerprint_contributions=[],
            important_atoms=[],
            explanation_text=result["error"],
            confidence="low",
            error=result["error"],
        )

    return ExplainResponse(
        smiles=smiles_to_use,
        original_smiles=original_smiles,
        repaired_smiles=repaired_field,
        score=result["score"],
        shap_base_value=result["shap_base_value"],
        descriptor_contributions=[
            DescriptorContribution(**d) for d in result["descriptor_contributions"]
        ],
        fingerprint_contributions=[
            FingerprintContribution(**f) for f in result["fingerprint_contributions"]
        ],
        important_atoms=result["important_atoms"],
        explanation_text=result["explanation_text"],
        confidence=result["confidence"],
    )
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