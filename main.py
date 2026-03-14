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
#from rdkit.Chem import Draw, rdChemReactions

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

class SynthesisRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the drug-like molecule")
    score:  float = Field(0.0, description="QML drug-likeness score (0–1)")
 
class ReagentItem(BaseModel):
    name:         str
    role:         str        # e.g. "Solvent", "Base", "Catalyst", "Coupling agent"
    hazard:       str        # e.g. "Flammable", "Corrosive", "Low hazard"
    hazard_level: str        # "low" | "medium" | "high"
    cas:          Optional[str] = None
 
class SynthesisStep(BaseModel):
    step:        int
    reaction:    str         # reaction type name
    description: str         # what happens chemically
    reagents:    List[str]   # names of reagents for this step
    conditions:  str         # temperature, time, atmosphere
    yield_est:   str         # e.g. "70–85%"
    difficulty:  str         # "Easy" | "Moderate" | "Challenging"
    notes:       str         # practical tips
 
class SafetyNote(BaseModel):
    category: str            # "PPE", "Waste", "Storage", "Emergency"
    detail:   str
 
class SynthesisResponse(BaseModel):
    smiles:            str
    iupac_hint:        str   # best-guess name from substructure
    complexity:        str   # "Simple" | "Moderate" | "Complex"
    total_steps:       int
    overall_yield_est: str
    overall_difficulty: str
    steps:             List[SynthesisStep]
    all_reagents:      List[ReagentItem]
    safety_notes:      List[SafetyNote]
    retrosynthesis_summary: str
    error:             Optional[str] = None


# SYNTHESIS CALCULATION

REACTION_TEMPLATES = [
 
    # ── Amide bond (most common in drugs — peptide/amide coupling) ────────────
    {
        "smarts":    "[NX3;H1,H2][CX3](=[OX1])",
        "reaction":  "Amide Coupling",
        "description": (
            "Form the amide bond by coupling a carboxylic acid with an amine "
            "using a peptide coupling reagent. Activate the acid in situ, then "
            "add the amine component."
        ),
        "reagents": ["HATU", "DIPEA", "DMF", "Carboxylic acid precursor", "Amine precursor"],
        "conditions": "0 °C → RT, 2–12 h, N₂ atmosphere, anhydrous DMF",
        "yield_est": "65–90%",
        "difficulty": "Moderate",
        "notes": (
            "Keep reaction anhydrous. HATU can be replaced with EDC/HOBt for "
            "larger scale. Monitor by TLC (ninhydrin stain). "
            "Purify by column chromatography (EtOAc/hexane gradient)."
        ),
    },
 
    # ── Ester bond ────────────────────────────────────────────────────────────
    {
        "smarts":    "[OX2][CX3](=[OX1])[#6]",
        "reaction":  "Fischer Esterification / Acyl Chloride Route",
        "description": (
            "Form the ester by reacting the carboxylic acid with the alcohol "
            "under acid catalysis (Fischer), or via the acyl chloride for "
            "higher reactivity and yield."
        ),
        "reagents": ["Carboxylic acid precursor", "Alcohol precursor",
                     "Thionyl chloride (SOCl₂)", "Triethylamine (TEA)", "DCM"],
        "conditions": "Acyl chloride route: 0 °C → RT, 1–3 h, N₂; Fischer: reflux, H₂SO₄ cat.",
        "yield_est": "70–92%",
        "difficulty": "Easy",
        "notes": (
            "Acyl chloride route preferred for sensitive substrates. "
            "SOCl₂ must be handled in a fume hood — highly corrosive. "
            "Quench with sat. NaHCO₃ to remove acid traces."
        ),
    },
 
    # ── Aromatic amine (aniline) — reductive amination or nitro reduction ─────
    {
        "smarts":    "[NX3;H1,H2][c]",
        "reaction":  "Nitro Reduction → Aromatic Amine",
        "description": (
            "Introduce the aromatic amine via catalytic hydrogenation of a "
            "nitro precursor (easily prepared by nitration). "
            "Fe/AcOH (Baeyer–Villiger) is an alternative for acid-sensitive substrates."
        ),
        "reagents": ["Nitro-arene precursor", "Pd/C (10%)", "H₂ (1 atm)",
                     "EtOH", "Fe powder", "Acetic acid"],
        "conditions": "H₂/Pd-C: RT, 1–4 h, H₂ balloon; Fe/AcOH: 80 °C, 2 h",
        "yield_est": "80–95%",
        "difficulty": "Easy",
        "notes": (
            "Filter Pd/C through Celite under N₂ — pyrophoric when wet. "
            "Confirm complete reduction by TLC (Rf shift) and MS. "
            "Fe/AcOH route avoids H₂ but requires aqueous workup."
        ),
    },
 
    # ── Reductive amination (secondary/tertiary amine from ketone+amine) ──────
    {
        "smarts":    "[NX3;H0]([#6])[#6]",
        "reaction":  "Reductive Amination",
        "description": (
            "Condense the ketone/aldehyde with the amine to form an imine "
            "intermediate, then reduce with NaBH₃CN or NaBH(OAc)₃ to give "
            "the secondary/tertiary amine."
        ),
        "reagents": ["Aldehyde/ketone precursor", "Amine precursor",
                     "NaBH(OAc)₃", "AcOH (cat.)", "DCE or MeOH"],
        "conditions": "RT, 12–24 h, open to air tolerated; AcOH 1 equiv. as activator",
        "yield_est": "60–85%",
        "difficulty": "Moderate",
        "notes": (
            "NaBH(OAc)₃ preferred over NaBH₃CN (less toxic). "
            "Molecular sieves (4Å) improve yield by removing water. "
            "Monitor imine formation at 30 min by LC-MS before adding reductant."
        ),
    },
 
    # ── Suzuki–Miyaura coupling (biaryl / aryl-heteroaryl) ───────────────────
    {
        "smarts":    "[c][c]",
        "reaction":  "Suzuki–Miyaura Cross-Coupling",
        "description": (
            "Couple an aryl/heteroaryl halide with an arylboronic acid using "
            "Pd(0) catalysis to form the biaryl bond. "
            "Key step for connecting aromatic fragments."
        ),
        "reagents": ["Aryl halide precursor (Br or I preferred)",
                     "Arylboronic acid", "Pd(PPh₃)₄ or Pd(dppf)Cl₂",
                     "K₂CO₃ or Cs₂CO₃", "DMF/H₂O (4:1)"],
        "conditions": "80–100 °C, 4–16 h, N₂ atmosphere, sealed vial",
        "yield_est": "60–90%",
        "difficulty": "Moderate",
        "notes": (
            "Aryl iodides react faster than bromides. "
            "Rigorously degas solvent before use (freeze-pump-thaw ×3). "
            "Pd catalyst loading: 2–5 mol%. "
            "Filter through silica to remove Pd black before column."
        ),
    },
 
    # ── Sulfonamide ───────────────────────────────────────────────────────────
    {
        "smarts":    "[NX3][SX4](=[OX1])(=[OX1])",
        "reaction":  "Sulfonamide Formation",
        "description": (
            "React a sulfonyl chloride with an amine (primary or secondary) "
            "in the presence of a base. "
            "Sulfonyl chlorides are easily prepared from sulfonic acids via SOCl₂."
        ),
        "reagents": ["Sulfonyl chloride precursor", "Amine precursor",
                     "Pyridine or TEA", "DCM"],
        "conditions": "0 °C → RT, 1–6 h, N₂; base (2 equiv.) to neutralise HCl",
        "yield_est": "70–90%",
        "difficulty": "Easy",
        "notes": (
            "Pyridine acts as both base and catalyst. "
            "Sulfonyl chlorides are lachrymatory — handle in fume hood. "
            "Wash with 1M HCl then sat. NaHCO₃ to remove pyridine."
        ),
    },
 
    # ── Piperazine / cyclic amine ─────────────────────────────────────────────
    {
        "smarts":    "[NX3]1CC[NX3]CC1",
        "reaction":  "N-Alkylation of Piperazine",
        "description": (
            "Mono-alkylate piperazine at one nitrogen using an alkyl halide "
            "or via reductive amination with an aldehyde. "
            "Protect the second nitrogen as Boc if selectivity is needed."
        ),
        "reagents": ["Piperazine", "Alkyl halide or aldehyde",
                     "K₂CO₃ or NaBH(OAc)₃", "MeCN or DCE",
                     "Boc₂O (if protection needed)"],
        "conditions": "60–80 °C, 4–12 h (alkylation); RT 12 h (reductive amination)",
        "yield_est": "55–80%",
        "difficulty": "Moderate",
        "notes": (
            "Excess piperazine (2–3 equiv.) minimises bis-alkylation. "
            "Boc protection/deprotection adds 2 steps but improves selectivity. "
            "Purify by SCX cartridge (catch-and-release) before column."
        ),
    },
 
    # ── Morpholine ring ───────────────────────────────────────────────────────
    {
        "smarts":    "[NX3]1CCOCC1",
        "reaction":  "N-Functionalization of Morpholine",
        "description": (
            "Alkylate or acylate the morpholine nitrogen. "
            "Morpholine is commercially available — typically used as a "
            "nucleophile to install the ring onto an electrophilic fragment."
        ),
        "reagents": ["Morpholine", "Electrophile (acid chloride, alkyl halide, or aldehyde)",
                     "TEA or DIPEA", "DCM or THF"],
        "conditions": "RT → 60 °C, 1–8 h",
        "yield_est": "65–88%",
        "difficulty": "Easy",
        "notes": (
            "Morpholine is a weak nucleophile — use activated electrophiles. "
            "For acylation, 1.1 equiv. acid chloride + 1.5 equiv. TEA in DCM at 0 °C."
        ),
    },
 
    # ── Hydroxyl group (phenol or alcohol) ────────────────────────────────────
    {
        "smarts":    "[OX2H]",
        "reaction":  "Phenol/Alcohol Introduction via Demethylation or Reduction",
        "description": (
            "Introduce the hydroxyl via BBr₃-mediated O-demethylation of the "
            "methyl ether precursor, or via reduction of a ketone with NaBH₄. "
            "Alternatively, use directed ortho-lithiation for phenols."
        ),
        "reagents": ["Methyl ether precursor", "BBr₃ (1M in DCM)",
                     "DCM", "MeOH (quench)", "NaBH₄ (if ketone route)"],
        "conditions": "BBr₃ route: −78 °C → RT, 2–4 h, N₂ anhydrous; NaBH₄: 0 °C → RT, 1 h",
        "yield_est": "70–90%",
        "difficulty": "Moderate",
        "notes": (
            "BBr₃ is highly moisture-sensitive — use Schlenk technique. "
            "Quench carefully with MeOH at 0 °C (vigorous gas evolution). "
            "Protect other sensitive groups (esters hydrolyse under BBr₃)."
        ),
    },
 
    # ── Heterocycle: imidazole / triazole ─────────────────────────────────────
    {
        "smarts":    "c1cnc[nH]1",
        "reaction":  "Imidazole Ring Formation (van Leusen / Debus–Radziszewski)",
        "description": (
            "Construct the imidazole ring via the Debus–Radziszewski condensation "
            "of an aldehyde, an α-diketone, and ammonia; or use van Leusen "
            "TosMIC reagent for substituted imidazoles."
        ),
        "reagents": ["Aldehyde", "TosMIC (tosylmethyl isocyanide)",
                     "K₂CO₃", "MeOH", "NH₄OAc (Debus route)"],
        "conditions": "van Leusen: RT→60 °C, 4 h, K₂CO₃/MeOH; Debus: AcOH, 100 °C, 6 h",
        "yield_est": "50–75%",
        "difficulty": "Challenging",
        "notes": (
            "TosMIC is moisture-sensitive — store at 4 °C. "
            "van Leusen gives 1,5-disubstituted product; "
            "control substitution pattern carefully. "
            "Purify by reverse-phase HPLC if regioisomers form."
        ),
    },
 
    # ── Fluorine introduction ─────────────────────────────────────────────────
    {
        "smarts":    "[F][c]",
        "reaction":  "Aromatic Fluorination (Balz–Schiemann / Halex)",
        "description": (
            "Introduce aryl fluorine via Balz–Schiemann reaction "
            "(diazotisation of aniline → fluoroborate salt → thermolysis) "
            "or nucleophilic Halex exchange on activated aryl chlorides."
        ),
        "reagents": ["Aniline precursor", "NaNO₂", "HBF₄",
                     "KF (Halex)", "DMSO (Halex solvent)"],
        "conditions": "Balz–Schiemann: 0 °C diazotisation, then 150 °C thermolysis; Halex: 160 °C, 12 h",
        "yield_est": "40–70%",
        "difficulty": "Challenging",
        "notes": (
            "Balz–Schiemann thermolysis can be exothermic — scale up carefully. "
            "Halex requires electron-withdrawing groups ortho/para to the halide. "
            "Consider purchasing fluorinated building blocks to avoid this step."
        ),
    },
 
    # ── Carbamate (urethane) ──────────────────────────────────────────────────
    {
        "smarts":    "[NX3][CX3](=[OX1])[OX2]",
        "reaction":  "Carbamate Formation",
        "description": (
            "React an amine with a chloroformate or CDI-activated alcohol "
            "to form the carbamate (urethane) linkage. "
            "Boc protection/deprotection is a special case of this reaction."
        ),
        "reagents": ["Amine precursor", "Chloroformate or Boc₂O",
                     "TEA or DIPEA", "DCM", "DMAP (cat.)"],
        "conditions": "0 °C → RT, 1–4 h, N₂",
        "yield_est": "75–92%",
        "difficulty": "Easy",
        "notes": (
            "Chloroformates are lachrymatory and moisture-sensitive. "
            "DMAP (0.1 equiv.) dramatically accelerates reaction. "
            "Boc deprotection: 4M HCl in dioxane or TFA/DCM (1:1), RT, 1 h."
        ),
    },
]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# REAGENT LIBRARY
# Maps reagent name → ReagentItem metadata
# ══════════════════════════════════════════════════════════════════════════════
 
REAGENT_LIBRARY = {
    "HATU": ReagentItem(
        name="HATU", role="Coupling agent",
        hazard="Irritant — avoid inhalation", hazard_level="medium",
        cas="148893-10-1"
    ),
    "DIPEA": ReagentItem(
        name="DIPEA (Hünig's base)", role="Base",
        hazard="Flammable, corrosive vapour", hazard_level="medium",
        cas="7087-68-5"
    ),
    "DMF": ReagentItem(
        name="DMF (N,N-Dimethylformamide)", role="Solvent",
        hazard="Reproductive toxin (Cat. 1B) — minimise exposure", hazard_level="high",
        cas="68-12-2"
    ),
    "DCM": ReagentItem(
        name="DCM (Dichloromethane)", role="Solvent",
        hazard="Suspected carcinogen — use in fume hood", hazard_level="medium",
        cas="75-09-2"
    ),
    "Pd/C (10%)": ReagentItem(
        name="Palladium on Carbon 10%", role="Heterogeneous catalyst",
        hazard="Pyrophoric when wet — handle under N₂, dispose as heavy metal waste",
        hazard_level="high", cas="7440-05-3"
    ),
    "NaBH(OAc)₃": ReagentItem(
        name="Sodium triacetoxyborohydride", role="Mild reducing agent",
        hazard="Moisture-sensitive, irritant", hazard_level="low",
        cas="56553-60-7"
    ),
    "Pd(PPh₃)₄": ReagentItem(
        name="Tetrakis(triphenylphosphine)palladium(0)", role="Pd(0) catalyst",
        hazard="Air/moisture sensitive — store under N₂ at −20 °C", hazard_level="medium",
        cas="14221-01-3"
    ),
    "Pd(dppf)Cl₂": ReagentItem(
        name="[1,1′-Bis(diphenylphosphino)ferrocene]palladium(II) dichloride",
        role="Pd(II) precatalyst",
        hazard="Irritant, store under N₂", hazard_level="medium",
        cas="72287-26-4"
    ),
    "K₂CO₃": ReagentItem(
        name="Potassium carbonate", role="Base",
        hazard="Low hazard — mild irritant", hazard_level="low",
        cas="584-08-7"
    ),
    "Cs₂CO₃": ReagentItem(
        name="Caesium carbonate", role="Strong base",
        hazard="Irritant, expensive — use K₂CO₃ when possible", hazard_level="low",
        cas="534-17-8"
    ),
    "Thionyl chloride (SOCl₂)": ReagentItem(
        name="Thionyl chloride", role="Activating agent",
        hazard="Highly corrosive, reacts violently with water — fume hood essential",
        hazard_level="high", cas="7719-09-7"
    ),
    "TEA": ReagentItem(
        name="Triethylamine (TEA)", role="Base",
        hazard="Flammable, irritant", hazard_level="low",
        cas="121-44-8"
    ),
    "Pyridine": ReagentItem(
        name="Pyridine", role="Base / catalyst",
        hazard="Flammable, foul odour, possible carcinogen — use in fume hood",
        hazard_level="medium", cas="110-86-1"
    ),
    "BBr₃ (1M in DCM)": ReagentItem(
        name="Boron tribromide (1M in DCM)", role="Lewis acid / demethylating agent",
        hazard="Highly corrosive, moisture-sensitive — Schlenk technique required",
        hazard_level="high", cas="10294-33-4"
    ),
    "TosMIC (tosylmethyl isocyanide)": ReagentItem(
        name="TosMIC", role="Heterocycle building block",
        hazard="Irritant, moisture-sensitive", hazard_level="medium",
        cas="36635-61-7"
    ),
    "NaNO₂": ReagentItem(
        name="Sodium nitrite", role="Diazotising agent",
        hazard="Oxidiser, toxic — avoid mixing with organics outside reaction",
        hazard_level="high", cas="7632-00-0"
    ),
    "HBF₄": ReagentItem(
        name="Tetrafluoroboric acid", role="Fluoride source",
        hazard="Corrosive, fluoride toxicity risk", hazard_level="high",
        cas="16872-11-0"
    ),
    "EtOH": ReagentItem(
        name="Ethanol", role="Solvent",
        hazard="Flammable — keep away from ignition sources", hazard_level="low",
        cas="64-17-5"
    ),
    "MeOH": ReagentItem(
        name="Methanol", role="Solvent / quench",
        hazard="Flammable, toxic if ingested", hazard_level="medium",
        cas="67-56-1"
    ),
    "THF": ReagentItem(
        name="Tetrahydrofuran (THF)", role="Solvent",
        hazard="Highly flammable, forms explosive peroxides on storage — test for peroxides before use",
        hazard_level="medium", cas="109-99-9"
    ),
    "MeCN": ReagentItem(
        name="Acetonitrile (MeCN)", role="Solvent",
        hazard="Flammable, irritant", hazard_level="low",
        cas="75-05-8"
    ),
    "DMSO": ReagentItem(
        name="Dimethyl sulfoxide (DMSO)", role="High-boiling polar solvent",
        hazard="Low acute toxicity but penetrates skin — carry dissolved reagents through skin",
        hazard_level="medium", cas="67-68-5"
    ),
    "Boc₂O": ReagentItem(
        name="Di-tert-butyl dicarbonate (Boc₂O)", role="Protecting group reagent",
        hazard="Irritant, releases CO₂ on reaction", hazard_level="low",
        cas="24424-99-5"
    ),
    "DMAP (cat.)": ReagentItem(
        name="4-Dimethylaminopyridine (DMAP)", role="Nucleophilic catalyst",
        hazard="Toxic — handle with care, avoid skin contact", hazard_level="medium",
        cas="1122-58-3"
    ),
    "NaBH₄": ReagentItem(
        name="Sodium borohydride", role="Mild reducing agent",
        hazard="Moisture-sensitive, flammable H₂ release — no protic solvents until cool",
        hazard_level="medium", cas="16940-66-2"
    ),
    "AcOH (cat.)": ReagentItem(
        name="Acetic acid (glacial)", role="Catalyst / pH modifier",
        hazard="Corrosive vapour — fume hood", hazard_level="low",
        cas="64-19-7"
    ),
    "Fe powder": ReagentItem(
        name="Iron powder", role="Reductant (Baeyer–Villiger reduction)",
        hazard="Flammable solid — keep dry", hazard_level="low",
        cas="7439-89-6"
    ),
    "Acetic acid": ReagentItem(
        name="Acetic acid", role="Solvent / proton source",
        hazard="Corrosive at high concentration", hazard_level="low",
        cas="64-19-7"
    ),
    "Morpholine": ReagentItem(
        name="Morpholine", role="Amine building block",
        hazard="Flammable, corrosive vapour", hazard_level="medium",
        cas="110-91-8"
    ),
    "Piperazine": ReagentItem(
        name="Piperazine", role="Diamine building block",
        hazard="Irritant, sensitiser — use gloves", hazard_level="low",
        cas="110-85-0"
    ),
    "K₂CO₃ or Cs₂CO₃": ReagentItem(
        name="K₂CO₃ or Cs₂CO₃", role="Base",
        hazard="Low hazard", hazard_level="low", cas="584-08-7"
    ),
    "DMF/H₂O (4:1)": ReagentItem(
        name="DMF/Water mixture", role="Solvent system",
        hazard="DMF is a reproductive toxin — minimise skin contact", hazard_level="high",
        cas=None
    ),
}
 
# Default reagent for anything not in the library
def _reagent_info(name: str) -> ReagentItem:
    if name in REAGENT_LIBRARY:
        return REAGENT_LIBRARY[name]
    return ReagentItem(
        name=name, role="Reagent",
        hazard="Consult SDS before use", hazard_level="medium"
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY SCORER
# ══════════════════════════════════════════════════════════════════════════════
 
def _score_complexity(mol) -> tuple[str, str]:
    """
    Returns (complexity_label, overall_difficulty) based on molecular features.
    """
    mw        = Descriptors.ExactMolWt(mol)
    n_rings   = rdMolDescriptors.CalcNumRings(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_stereo  = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_heavy   = mol.GetNumHeavyAtoms()
 
    score = 0
    if mw > 500:      score += 2
    elif mw > 350:    score += 1
    if n_rings > 3:   score += 2
    elif n_rings > 1: score += 1
    if n_stereo > 1:  score += 2
    elif n_stereo > 0: score += 1
    if rot_bonds > 8: score += 1
    if n_heavy > 35:  score += 1
 
    if score <= 2:
        return "Simple",   "Easy"
    elif score <= 5:
        return "Moderate", "Moderate"
    else:
        return "Complex",  "Challenging"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# YIELD ESTIMATOR
# Combines per-step yield estimates into an overall yield
# ══════════════════════════════════════════════════════════════════════════════
 
def _overall_yield(steps: list[SynthesisStep]) -> str:
    """
    Multiply step yield midpoints to get overall yield estimate.
    """
    if not steps:
        return "N/A"
    product = 1.0
    for step in steps:
        # Parse "X–Y%" → midpoint
        try:
            parts = step.yield_est.replace('%', '').split('–')
            lo, hi = float(parts[0]), float(parts[1])
            product *= ((lo + hi) / 2) / 100.0
        except Exception:
            product *= 0.75
    pct = round(product * 100, 1)
    # Express as a range ±10%
    lo = max(5, pct - 10)
    hi = min(95, pct + 10)
    return f"{lo:.0f}–{hi:.0f}%"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SAFETY NOTES GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
 
def _build_safety_notes(matched_templates: list[dict], mol) -> list[SafetyNote]:
    notes = []
 
    # Always-present baseline notes
    notes.append(SafetyNote(
        category="PPE",
        detail="Lab coat, nitrile gloves, and safety glasses required at all times. "
               "Use a fume hood for all liquid transfers and reactions."
    ))
    notes.append(SafetyNote(
        category="Waste",
        detail="Segregate halogenated (DCM, CHCl₃) from non-halogenated waste. "
               "Heavy metal waste (Pd, Rh) must be collected separately for specialist disposal."
    ))
 
    # Conditional notes based on reagents used
    reagent_names = {r for t in matched_templates for r in t.get("reagents", [])}
 
    if any(r in reagent_names for r in ["Pd/C (10%)", "Pd(PPh₃)₄", "Pd(dppf)Cl₂"]):
        notes.append(SafetyNote(
            category="Storage",
            detail="Palladium catalysts: store under N₂ at −20 °C. "
                   "Wet Pd/C is pyrophoric — never allow to dry on filter paper."
        ))
    if "BBr₃ (1M in DCM)" in reagent_names:
        notes.append(SafetyNote(
            category="Emergency",
            detail="BBr₃ contact: flush with large volumes of water for 15 min, "
                   "seek medical attention immediately. Keep neutralising solution (NaHCO₃) nearby."
        ))
    if "DMF" in reagent_names or "DMF/H₂O (4:1)" in reagent_names:
        notes.append(SafetyNote(
            category="PPE",
            detail="DMF is a reproductive toxin — use double gloves, minimise skin exposure. "
                   "Do not use DMF near open flames (flash point 58 °C)."
        ))
    if any(r in reagent_names for r in ["Thionyl chloride (SOCl₂)", "NaNO₂", "HBF₄"]):
        notes.append(SafetyNote(
            category="Emergency",
            detail="Highly reactive reagents in use. Ensure eyewash station and "
                   "safety shower are accessible. Work with a partner."
        ))
 
    # Stereocentre warning
    stereocentres = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if stereocentres:
        notes.append(SafetyNote(
            category="Storage",
            detail=f"Molecule has {len(stereocentres)} stereocentre(s). "
                   "Confirm absolute configuration by chiral HPLC and optical rotation. "
                   "Store enantiopure material at −20 °C under N₂."
        ))
 
    notes.append(SafetyNote(
        category="Waste",
        detail="All intermediates and final compound should be characterised "
               "by ¹H NMR, ¹³C NMR, and HRMS before biological testing."
    ))
 
    return notes
 
 
# ══════════════════════════════════════════════════════════════════════════════
# RETROSYNTHESIS SUMMARY TEXT
# ══════════════════════════════════════════════════════════════════════════════
 
def _build_summary(mol, matched_templates: list[dict], complexity: str) -> str:
    mw      = round(Descriptors.ExactMolWt(mol), 1)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    rxn_names = [t["reaction"] for t in matched_templates]
 
    summary = (
        f"Retrosynthetic analysis identifies {len(matched_templates)} key bond "
        f"disconnection(s) for this {complexity.lower()} molecule (MW {mw} Da, "
        f"{n_rings} ring(s)"
    )
    if n_stereo:
        summary += f", {n_stereo} stereocentre(s)"
    summary += "). "
 
    if rxn_names:
        summary += f"Primary synthetic steps: {', '.join(rxn_names)}. "
 
    summary += (
        "The proposed route uses commercially available starting materials "
        "and standard laboratory equipment. "
    )
 
    if complexity == "Simple":
        summary += (
            "This molecule is accessible in a straightforward linear sequence "
            "suitable for a well-equipped undergraduate lab."
        )
    elif complexity == "Moderate":
        summary += (
            "A postgraduate-level synthesis is recommended. "
            "Allow 1–2 weeks for synthesis and purification."
        )
    else:
        summary += (
            "This is a challenging target requiring experienced synthetic chemists, "
            "Schlenk/glovebox technique, and chiral resolution or asymmetric synthesis. "
            "Allow 3–6 weeks for a skilled medicinal chemistry team."
        )
 
    return summary
 
 
# ══════════════════════════════════════════════════════════════════════════════
# IUPAC HINT (best-effort name from known fragments)
# ══════════════════════════════════════════════════════════════════════════════
 
def _iupac_hint(mol) -> str:
    """
    Heuristic name hint — not a full IUPAC name, but informative for the scientist.
    Uses ring counts, MW, and heteroatom composition.
    """
    mw = round(Descriptors.ExactMolWt(mol), 1)
    n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
 
    parts = []
    if n_ar >= 2:
        parts.append("biaryl")
    elif n_ar == 1:
        parts.append("monoaryl")
    if n_rings > n_ar:
        parts.append(f"{n_rings - n_ar} aliphatic ring(s)")
    parts.append(f"MW {mw}")
    parts.append(f"HBD {hbd} / HBA {hba}")
 
    return f"Drug-like compound — {', '.join(parts)}. Use RDKit or ChemDraw for full IUPAC name."
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN SYNTHESIS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
 
def _compute_synthesis(smiles: str, score: float) -> dict:
    """
    Core synthesis route computation.
    Returns a dict matching SynthesisResponse fields.
    """
    # ── Parse molecule ────────────────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "smiles": smiles, "iupac_hint": "N/A", "complexity": "N/A",
            "total_steps": 0, "overall_yield_est": "N/A",
            "overall_difficulty": "N/A", "steps": [], "all_reagents": [],
            "safety_notes": [], "retrosynthesis_summary": "",
            "error": f"Could not parse SMILES: {smiles}",
        }
 
    # ── Match reaction templates ──────────────────────────────────────────────
    matched = []
    seen_reactions = set()
 
    for template in REACTION_TEMPLATES:
        patt = Chem.MolFromSmarts(template["smarts"])
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            if template["reaction"] not in seen_reactions:
                matched.append(template)
                seen_reactions.add(template["reaction"])
 
    # Always add a final purification step
    purification_step = {
        "reaction":    "Purification & Characterisation",
        "description": (
            "Purify the crude product by silica gel column chromatography "
            "(or preparative HPLC for polar/complex molecules). "
            "Characterise by ¹H NMR, ¹³C NMR, HRMS, and HPLC purity (>95%)."
        ),
        "reagents":   ["Silica gel", "EtOAc", "Hexane", "MeOH", "CDCl₃ (NMR)"],
        "conditions": "RT; gradient elution; lyophilise if aqueous HPLC used",
        "yield_est":  "85–98%",
        "difficulty": "Easy",
        "notes": (
            "Record ¹H NMR in CDCl₃ or DMSO-d₆. "
            "HRMS (ESI+) to confirm molecular formula. "
            "Check HPLC purity at 214 nm and 254 nm. "
            "Store final compound at −20 °C as DMSO stock (10 mM)."
        ),
    }
 
    # If no templates matched, add a generic linear synthesis step
    if not matched:
        matched = [{
            "reaction":    "Linear Convergent Synthesis",
            "description": (
                "No specific reaction motif detected automatically. "
                "Propose a convergent synthesis by breaking the molecule at "
                "the most complex bond disconnection (largest fragment split). "
                "Consult SciFinder or Reaxys for literature precedent."
            ),
            "reagents":   ["Starting material A", "Starting material B",
                           "Appropriate coupling reagent", "Suitable solvent"],
            "conditions": "Optimise based on functional groups present",
            "yield_est":  "50–75%",
            "difficulty": "Moderate",
            "notes": (
                "Use RDKit retrosynthesis or consult ASKCOS/Chematica "
                "for automated route planning. "
                "Purchase the closest commercially available analogue as "
                "a reference standard."
            ),
        }]
 
    # ── Build step objects ────────────────────────────────────────────────────
    steps = []
    all_template_steps = matched + [purification_step]
    for i, t in enumerate(all_template_steps, start=1):
        steps.append(SynthesisStep(
            step=i,
            reaction=t["reaction"],
            description=t["description"],
            reagents=t["reagents"],
            conditions=t["conditions"],
            yield_est=t["yield_est"],
            difficulty=t["difficulty"],
            notes=t["notes"],
        ))
 
    # ── Collect all unique reagents ───────────────────────────────────────────
    seen_reagents = set()
    all_reagents  = []
    for t in all_template_steps:
        for r_name in t.get("reagents", []):
            if r_name not in seen_reagents:
                seen_reagents.add(r_name)
                all_reagents.append(_reagent_info(r_name))
 
    # ── Complexity + yield ────────────────────────────────────────────────────
    complexity, overall_difficulty = _score_complexity(mol)
    overall_yield = _overall_yield(steps[:-1])  # exclude purification from yield calc
 
    # ── Safety ────────────────────────────────────────────────────────────────
    safety_notes = _build_safety_notes(matched, mol)
 
    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_summary(mol, matched, complexity)
 
    return {
        "smiles":               smiles,
        "iupac_hint":           _iupac_hint(mol),
        "complexity":           complexity,
        "total_steps":          len(steps),
        "overall_yield_est":    overall_yield,
        "overall_difficulty":   overall_difficulty,
        "steps":                steps,
        "all_reagents":         all_reagents,
        "safety_notes":         safety_notes,
        "retrosynthesis_summary": summary,
        "error":                None,
    }
 
 
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

@app.post("/synthesis", response_model=SynthesisResponse, tags=["Lab Synthesis"])
async def lab_synthesis(request: SynthesisRequest):
    """
    Generate a wet-lab synthesis route for a drug-like molecule.
 
    Uses RDKit substructure matching against a library of named reaction
    templates to identify key bond disconnections, then returns:
    - Step-by-step synthesis instructions
    - Reagents with hazard classifications
    - Safety notes and PPE requirements
    - Estimated yields and overall difficulty
 
    Only meaningful for molecules with score >= 0.4 (medium/high drug-likeness).
    """
    if request.score < 0.4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Drug-likeness score {request.score:.3f} is too low for synthesis planning. "
                "Lab synthesis is only suggested for molecules with score ≥ 0.4 "
                "(medium or high drug-likeness)."
            )
        )
 
    repaired = repair_smiles(request.smiles)
    if repaired is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse or repair SMILES: {request.smiles}"
        )
 
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: _compute_synthesis(repaired, request.score)
        )
    except Exception as e:
        logger.error(f"Synthesis computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
 
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
 
    return SynthesisResponse(**result)
 
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