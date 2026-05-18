import React, { useState, useRef, useEffect } from 'react'
import './GeneratePage.css'
import ADMETDashboard from './Admetdashboard'
import VRMolViewer from './Vrmolviewer'
import ExplanationPanel from './Explanationpanel'
import LabSynthesis from './LabSynthesis'
import RankingPanel from './RankingPanel'


const API = 'https://clarindasusan-drug-predictor-api.hf.space'

const SUGGESTIONS = [
  'Cystic Fibrosis',
  'Gaucher Disease',
  "Wilson's Disease",
  'Fabry Disease',
]

// ── 3D Molecule Viewer ────────────────────────────────────────────────────────
function MolViewer({ sdf }) {
  const containerRef = useRef(null)
  const viewerRef    = useRef(null)
  const [selectedAtom, setSelectedAtom] = useState(null)

  useEffect(() => {
    if (!sdf || !containerRef.current) return

    const init = () => {
      if (viewerRef.current) {
        try { viewerRef.current.spin(false) } catch(e) {}
      }
      containerRef.current.innerHTML = ''
      const viewer = window.$3Dmol.createViewer(containerRef.current, {
        backgroundColor: '#0d0618',
      })
      viewerRef.current = viewer
      viewer.addModel(sdf, 'sdf')

      const cpkColors = {
        C: '#909090', H: '#ffffff', N: '#3050f8', O: '#ff0d0d',
        S: '#ffff30', F: '#90e050', Cl: '#1ff01f', Br: '#a62929',
        P: '#ff8000', I: '#940094', Fe: '#e06633', Ca: '#3dff00',
      }
      viewer.setStyle({}, {
        stick:  { radius: 0.12, color: '#909090' },
        sphere: { scale: 0.26, color: '#909090' },
      })
      Object.entries(cpkColors).forEach(([elem, color]) => {
        viewer.setStyle({ elem }, {
          stick:  { radius: 0.12, colorscheme: 'Jmol' },
          sphere: { scale: 0.26, color },
        })
      })

      viewer.setClickable({}, true, (atom) => {
        setSelectedAtom({
          element: atom.elem,
          index:   atom.index,
          x: atom.x?.toFixed(2) ?? '—',
          y: atom.y?.toFixed(2) ?? '—',
          z: atom.z?.toFixed(2) ?? '—',
        })
      })

      viewer.zoomTo()
      viewer.render()
      setTimeout(() => {
        try { viewer.resize(); viewer.render() } catch(e) {}
      }, 100)
      viewer.spin('y', 0.5)
    }

    if (window.$3Dmol) {
      init()
    } else {
      const script = document.createElement('script')
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js'
      script.onload = init
      document.head.appendChild(script)
    }

    return () => {
      if (viewerRef.current) try { viewerRef.current.spin(false) } catch(e) {}
    }
  }, [sdf])

  return (
    <div className="mol-viewer-wrap">
      <div className="mol-viewer-label">
        <span className="mol-viewer-dot" />
        3D Molecular Structure · CPK Coloring · Click atom for info
      </div>
      <div ref={containerRef} className="mol-viewer" />
      <div className="mol-legend">
        {[
          { el: 'C',  color: '#909090' },
          { el: 'H',  color: '#e8e8e8' },
          { el: 'N',  color: '#4169e1' },
          { el: 'O',  color: '#e13030' },
          { el: 'S',  color: '#d4c400' },
          { el: 'F',  color: '#40d0d0' },
          { el: 'Cl', color: '#30d030' },
          { el: 'Br', color: '#a05010' },
          { el: 'P',  color: '#e08000' },
        ].map(({ el, color }) => (
          <div key={el} className="mol-legend-item">
            <span className="mol-legend-dot" style={{ background: color }} />
            <span>{el}</span>
          </div>
        ))}
      </div>
      {selectedAtom && (
        <div className="mol-atom-info">
          <div className="mol-atom-title">Selected Atom</div>
          <div className="mol-atom-row"><span>Element:</span>   <span>{selectedAtom.element}</span></div>
          <div className="mol-atom-row"><span>Atom Index:</span><span>{selectedAtom.index}</span></div>
          <div className="mol-atom-row"><span>Position:</span> <span>{selectedAtom.x}, {selectedAtom.y}, {selectedAtom.z}</span></div>
        </div>
      )}
    </div>
  )
}

// ── Predict Panel ─────────────────────────────────────────────────────────────
function PredictPanel({ smiles, onOpenADMET, onOpenVR, onOpenSynthesis, onOpenRank }) {
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)

  const handlePredict = async () => {
    if (!smiles.trim()) return
    setLoading(true); setError(null); setResult(null)

    const controller = new AbortController()
    const timeoutId  = setTimeout(() => controller.abort(), 120000)

    try {
      const res = await fetch(`${API}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ smiles: smiles.trim() }),
        signal:  controller.signal,
      })
      clearTimeout(timeoutId)

      const text = await res.text()
      if (!text || !text.trim()) throw new Error('Empty response — Space may be waking up. Retry in 30 seconds.')
      let data
      try { data = JSON.parse(text) } catch { throw new Error(`Non-JSON response: ${text.slice(0, 200)}`) }
      if (!res.ok) throw new Error(data?.detail || `API error ${res.status}`)
      setResult(data)
    } catch (e) {
      clearTimeout(timeoutId)
      setError(
        e.name === 'AbortError'
          ? 'Prediction timed out — retry in a moment.'
          : e.message
      )
    } finally {
      setLoading(false)
    }
  }

  const scoreColor = result
    ? result.score >= 0.7 ? '#7ecba0'
    : result.score >= 0.4 ? '#d4af72'
    : '#f87171'
    : '#a855f7'

  return (
    <div className="predict-panel">
      <button
        className="predict-btn"
        onClick={handlePredict}
        disabled={loading || !smiles.trim()}
      >
        {loading
          ? <span className="gp-spinner-wrap"><span className="gp-spinner" />Running quantum circuit…</span>
          : '⚛ Predict Drug-likeness'}
      </button>

      {error && (
        <div className="predict-error">
          {error}
          <button className="predict-retry-btn" onClick={handlePredict} disabled={loading}>↺ Retry</button>
        </div>
      )}

      {result && (
        <div className="predict-result">
          <div className="predict-result-title">3. Prediction Result</div>

          <div className="predict-cards">
            <div className="predict-card">
              <div className="predict-card-label">Score</div>
              <div className="predict-card-value" style={{ color: scoreColor }}>
                {result.score.toFixed(4)}
              </div>
            </div>
            <div className="predict-card">
              <div className="predict-card-label">Confidence</div>
              <div className="predict-card-value">{result.confidence}</div>
            </div>
            <div className="predict-card">
              <div className="predict-card-label">Promising</div>
              <div className="predict-card-value" style={{ color: result.is_promising ? '#7ecba0' : '#f87171' }}>
                {result.is_promising ? 'Yes' : 'No'}
              </div>
            </div>
          </div>

          {result.repaired_smiles && result.repaired_smiles !== smiles && (
            <div className="predict-repaired">
              <span className="predict-repaired-label">Repaired SMILES</span>
              <span className="predict-repaired-val">{result.repaired_smiles}</span>
            </div>
          )}

          {result.sdf
            ? <MolViewer sdf={result.sdf} />
            : <div className="mol-no-3d">3D structure unavailable for this molecule</div>
          }

          <button
            className="admet-launch-btn"
            onClick={() => onOpenADMET({ smiles: result.repaired_smiles || smiles, sdf: result.sdf || null })}
          >
            <span className="admet-launch-icon">⬡</span>
            View ADMET Property Dashboard
            <span className="admet-launch-arrow">→</span>
          </button>

          <button
            className={`vr-launch-btn${!result.sdf ? ' vr-launch-btn--disabled' : ''}`}
            onClick={() => result.sdf && onOpenVR({ smiles: result.repaired_smiles || smiles, sdf: result.sdf })}
            disabled={!result.sdf}
            title={!result.sdf ? '3D structure unavailable' : 'Open WebXR 3D molecule viewer'}
          >
            <span className="vr-launch-icon">◉</span>
            Enable VR
            <span className="vr-launch-arrow">→</span>
          </button>

          {/* ── Lab Synthesis Button ── */}
          {result.score >= 0.4 ? (
            <button
              className="synthesis-launch-btn"
              onClick={() => onOpenSynthesis({ smiles: result.repaired_smiles || smiles, score: result.score })}
            >
              <span className="synthesis-launch-icon">⚗</span>
              Lab Synthesis Route
              <span className="synthesis-launch-arrow">→</span>
            </button>
          ) : (
            <div className="synthesis-low-score">
              ⚗ Lab Synthesis unavailable — score too low (min 0.4)
            </div>
          )}

          {/* ── Rank Candidates Button ── */}
          <button
            className="rank-launch-btn"
            onClick={() => onOpenRank({ smiles: result.repaired_smiles || smiles, qml_score: result.score })}
          >
            <span className="rank-launch-icon">⬡</span>
            Rank by Symptom Relevance
            <span className="rank-launch-arrow">→</span>
          </button>
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function GeneratePage({ onBack }) {
  const [disease,         setDisease]         = useState('')
  const [count,           setCount]           = useState(1)
  const [loading,         setLoading]         = useState(false)
  const [result,          setResult]          = useState(null)
  const [error,           setError]           = useState(null)
  const [editedMolecules, setEditedMolecules] = useState([])
  const [admetData,       setAdmetData]       = useState(null)
  const [vrData,          setVrData]          = useState(null)
  const [xaiData,         setXaiData]         = useState(null)
  const [synthesisData,   setSynthesisData]   = useState(null)
  const [rankData,        setRankData]        = useState(null)   // { smiles, qml_score, disease }

  // ── Page routing — synthesis first so it doesn't fight with others ─────────
  if (synthesisData) {
    return (
      <LabSynthesis
        smiles={synthesisData.smiles}
        score={synthesisData.score}
        apiBase={API}
        onClose={() => setSynthesisData(null)}
      />
    )
  }

  if (rankData) {
    return (
      <RankingPanel
        smiles={rankData.smiles}
        qmlScore={rankData.qml_score}
        disease={disease}
        apiBase={API}
        onClose={() => setRankData(null)}
      />
    )
  }

  if (xaiData) {
    return (
      <div className="xai-page">
        <div className="xai-topbar">
          <button className="admet-back-btn" onClick={() => setXaiData(null)}>← Back to ADMET</button>
        </div>
        <ExplanationPanel
          smiles={xaiData.smiles}
          apiBase={API}
          totalAtoms={xaiData.totalAtoms || 0}
        />
      </div>
    )
  }

  if (admetData) {
    return (
      <ADMETDashboard
        smiles={admetData.smiles}
        sdf={admetData.sdf}
        apiBase={API}
        onClose={() => { setAdmetData(null); setXaiData(null) }}
        onOpenXAI={setXaiData}
      />
    )
  }

  if (vrData) {
    return (
      <VRMolViewer
        key={vrData.smiles}
        smiles={vrData.smiles}
        sdf={vrData.sdf}
        onClose={() => setVrData(null)}
      />
    )
  }

  // ── Generation handler ─────────────────────────────────────────────────────
  const handleGenerate = async () => {
    if (!disease.trim()) return
    setLoading(true); setError(null); setResult(null); setEditedMolecules([])

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 180000)

    try {
      const res = await fetch(`${API}/generate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ disease: disease.trim(), num_candidates: count }),
        signal:  controller.signal,
      })
      clearTimeout(timeoutId)

      const text = await res.text()
      if (!text || !text.trim()) {
        throw new Error('Server returned an empty response — Space may be waking up. Wait 30 seconds and retry.')
      }
      let data
      try { data = JSON.parse(text) }
      catch { throw new Error(`Server returned non-JSON: ${text.slice(0, 200)}`) }
      if (!res.ok) throw new Error(data?.detail || `API error ${res.status}`)

      setResult(data)
      setEditedMolecules(data.candidates.map(c => c.smiles))

    } catch (e) {
      clearTimeout(timeoutId)
      setError(
        e.name === 'AbortError'
          ? 'Generation timed out (3 min). Try a common disease like "Cancer" first to wake the Space, then retry.'
          : e.message
      )
    } finally {
      setLoading(false)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="gp-root">
      <button className="gp-back" onClick={onBack}>← Back to Home</button>

      <div className="gp-container">

        <div className="gp-header">
          <div className="gp-label">API · /generate + /predict</div>
          <h1 className="gp-title">Molecule <span>Generation & Prediction</span></h1>
          <p className="gp-subtitle">
            Select a rare disease, generate SMILES candidates using the LLM,
            then predict drug-likeness with the quantum classifier.
          </p>
        </div>

        <div className="gp-step-label">1. Select Disease &amp; Count</div>
        <div className="gp-card">
          <div className="gp-field">
            <label>Target Disease</label>
            <input
              type="text"
              className="gp-input"
              placeholder="Select below or type a disease…"
              value={disease}
              onChange={e => setDisease(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleGenerate()}
            />
            <div className="gp-suggestions">
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  className={`gp-chip ${disease === s ? 'active' : ''}`}
                  onClick={() => setDisease(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="gp-field">
            <label>Number of Molecules</label>
            <div className="gp-stepper">
              <button className="gp-step-btn" onClick={() => setCount(c => Math.max(1, c - 1))}>−</button>
              <span className="gp-step-val">{count}</span>
              <button className="gp-step-btn" onClick={() => setCount(c => Math.min(10, c + 1))}>+</button>
              <span className="gp-step-hint">max 10</span>
            </div>
          </div>

          <button
            className="gp-submit"
            onClick={handleGenerate}
            disabled={loading || !disease.trim()}
          >
            {loading
              ? <span className="gp-spinner-wrap"><span className="gp-spinner" />Generating…</span>
              : 'Generate Molecules'}
          </button>
        </div>

        {loading && (
          <div className="gp-card gp-loading-card">
            <div className="gp-rings">
              {[50, 38, 26].map((r, i) => (
                <svg key={i} width="100" height="100" viewBox="0 0 100 100"
                  style={{
                    position: 'absolute', inset: 0,
                    animation: `gpSpin ${2 + i * 0.8}s linear infinite ${i % 2 ? 'reverse' : ''}`,
                  }}>
                  <circle cx="50" cy="50" r={r} fill="none"
                    stroke={['#a855f7', '#7c3aed', 'rgba(212,175,114,0.5)'][i]}
                    strokeWidth="1.5"
                    strokeDasharray={`${r * 0.65} ${r * 5}`}
                    strokeLinecap="round"
                  />
                </svg>
              ))}
              <span className="gp-ring-label">LLM</span>
            </div>
            <p className="gp-loading-text">
              Synthesising candidates for <em>{disease}</em>…
            </p>
            <p className="gp-loading-subtext">
              Rare disease generation may take up to 3 minutes — please wait
            </p>
          </div>
        )}

        {error && (
          <div className="gp-card gp-error-card">
            <div className="gp-error-badge">Error</div>
            <p className="gp-error-msg">{error}</p>
            <p className="gp-error-hint">API: <code>{API}</code></p>
            <button className="gp-retry-btn" onClick={handleGenerate} disabled={loading}>↺ Retry</button>
          </div>
        )}

        {result && editedMolecules.map((smiles, i) => {
          const candidate = result.candidates[i]
          return (
            <div key={i} className="gp-candidate-block">
              <div className="gp-step-label">
                {editedMolecules.length > 1
                  ? `2. Generated SMILES — Candidate ${i + 1}`
                  : '2. Generated SMILES'}
                <span className={`gp-source-badge gp-source-badge--${candidate?.source}`}>
                  {candidate?.source === 'Generated' ? '⚗ Generated' : '✦ Generated'}
                </span>
              </div>

              <div className="gp-smiles-row">
                <input
                  className="gp-input gp-smiles-input"
                  value={smiles}
                  onChange={e => {
                    const updated = [...editedMolecules]
                    updated[i] = e.target.value
                    setEditedMolecules(updated)
                  }}
                  placeholder="SMILES string…"
                />
                <button
                  className="gp-copy-btn"
                  onClick={() => navigator.clipboard.writeText(smiles)}
                >
                  Copy
                </button>
              </div>

              <PredictPanel
                smiles={smiles}
                onOpenADMET={setAdmetData}
                onOpenVR={setVrData}
                onOpenSynthesis={setSynthesisData}
                onOpenRank={setRankData}
              />
            </div>
          )
        })}

        {result && (
          <div className="gp-note">
            {result.generated > 0 && `✦ ${result.generated} LLM-generated`}
            {result.generated > 0 && result.fallback > 0 && ' · '}
            {result.fallback > 0 && `⚗ ${result.fallback} Generated`}
          </div>
        )}

      </div>
    </div>
  )
}