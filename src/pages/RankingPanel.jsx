import React, { useState } from 'react'
import './RankingPanel.css'


const SCORE_COLOR = s => s >= 0.7 ? '#6ee7b7' : s >= 0.4 ? '#d4af72' : '#f87171'

// ── Signal bar ─────────────────────────────────────────────────────────────────
function SignalBar({ label, value, color }) {
  return (
    <div className="rp-signal-row">
      <div className="rp-signal-label">{label}</div>
      <div className="rp-signal-track">
        <div className="rp-signal-fill" style={{ width: `${Math.min(100, value * 100)}%`, background: color }} />
      </div>
      <div className="rp-signal-val" style={{ color }}>{(value * 100).toFixed(0)}%</div>
    </div>
  )
}

// ── Ranked card ────────────────────────────────────────────────────────────────
function RankedCard({ candidate, isTop }) {
  const [expanded, setExpanded] = useState(isTop)
  const sc = SCORE_COLOR(candidate.final_score)

  return (
    <div className={`rp-card${isTop ? ' rp-card--top' : ''}`}>
      <button className="rp-card-header" onClick={() => setExpanded(e => !e)}>
        <div className="rp-rank-badge" style={{ color: isTop ? '#fbbf24' : '#94a3b8' }}>
          #{candidate.rank}
        </div>
        <div className="rp-card-main">
          <div className="rp-card-label">
            {candidate.label || candidate.smiles.slice(0, 28) + (candidate.smiles.length > 28 ? '…' : '')}
          </div>
          <div className="rp-card-smiles">{candidate.smiles.slice(0, 44)}{candidate.smiles.length > 44 ? '…' : ''}</div>
        </div>
        <div className="rp-card-scores">
          <div className="rp-score-chip" style={{ color: sc, borderColor: sc + '44' }}>
            {candidate.final_score.toFixed(3)}
          </div>
          <div className="rp-score-sub">final</div>
        </div>
        <span className="rp-chevron">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="rp-card-body">
          {/* Score breakdown row */}
          <div className="rp-score-row">
            <div className="rp-score-box">
              <div className="rp-score-box-label">QML Score</div>
              <div className="rp-score-box-val" style={{ color: SCORE_COLOR(candidate.qml_score) }}>
                {candidate.qml_score.toFixed(4)}
              </div>
            </div>
            <div className="rp-score-box">
              <div className="rp-score-box-label">Symptom Relevance</div>
              <div className="rp-score-box-val" style={{ color: SCORE_COLOR(candidate.symptom_relevance) }}>
                {candidate.symptom_relevance.toFixed(4)}
              </div>
            </div>
            <div className="rp-score-box rp-score-box--final">
              <div className="rp-score-box-label">Final Score</div>
              <div className="rp-score-box-val" style={{ color: sc }}>
                {candidate.final_score.toFixed(4)}
              </div>
            </div>
          </div>

          {/* Signal bars */}
          <div className="rp-signals">
            <div className="rp-signals-title">Signal Breakdown</div>
            <SignalBar label="Fingerprint Similarity" value={candidate.signals.fingerprint_similarity} color="#a855f7" />
            <SignalBar label="Physicochemical Fit"    value={candidate.signals.physchem_fit}           color="#60a5fa" />
            <SignalBar label="LLM Relevance"          value={candidate.signals.llm_score}              color={candidate.signals.llm_used ? '#6ee7b7' : '#475569'} />
            {!candidate.signals.llm_used && (
              <div className="rp-llm-note"></div>
            )}
          </div>

          {/* Explanation */}
          <div className="rp-explanation">{candidate.explanation}</div>
        </div>
      )}
    </div>
  )
}

// ── Main RankingPanel ──────────────────────────────────────────────────────────
export default function RankingPanel({ smiles, qmlScore, disease, apiBase, onClose }) {
  const [symptoms,   setSymptoms]   = useState(disease || '')
  const [weightQml,  setWeightQml]  = useState(0.6)
  const [useLlm,     setUseLlm]     = useState(false)
  const [loading,    setLoading]    = useState(false)
  const [result,     setResult]     = useState(null)
  const [error,      setError]      = useState(null)

  const runRank = async () => {
    if (!symptoms.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 60000)
      const res = await fetch(`${apiBase}/rank_candidates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidates: [{ smiles, qml_score: qmlScore, label: 'Current molecule' }],
          symptoms: symptoms.trim(),
          weight_qml: weightQml,
          use_llm: useLlm,
        }),
        signal: controller.signal,
      })
      clearTimeout(timeout)
      const text = await res.text()
      if (!text.trim()) throw new Error('Empty response — server may be waking up. Try again in 30s.')
      let json
      try { json = JSON.parse(text) } catch { throw new Error(`Server error: ${text.slice(0, 120)}`) }
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`)
      setResult(json)
    } catch (err) {
      setError(err.name === 'AbortError' ? 'Timed out — server may be sleeping. Wait 30s and retry.' : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rp-page">
      <div className="rp-topbar">
        <button className="vr-back-btn" onClick={onClose}>← Back to Results</button>
        <div className="rp-topbar-center">
          <span className="rp-eyebrow">⬡ Symptom-Based Ranking</span>
          <span className="rp-smiles-tag" title={smiles}>
            {smiles.length > 50 ? smiles.slice(0, 50) + '…' : smiles}
          </span>
        </div>
        <div style={{ width: '160px' }} />
      </div>

      <div className="rp-body">

        {/* Config card */}
        <div className="rp-config-card">
          <div className="rp-config-title">Rank this molecule against your target indication</div>

          <div className="rp-config-field">
            <label className="rp-config-label">Symptoms</label>
            <input
              className="rp-config-input"
              placeholder="e.g. inflammation and joint pain, arthritis"
              value={symptoms}
              onChange={e => setSymptoms(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && runRank()}
            />
            <div className="rp-config-hint">
              Supported: diabetes · hypertension · cancer · depression · alzheimer · infection · inflammation
            </div>
          </div>

          <div className="rp-config-row">
            <div className="rp-config-field rp-config-field--half">
              <label className="rp-config-label">QML Weight ({(weightQml * 100).toFixed(0)}%)</label>
              <input
                type="range" min="0" max="1" step="0.1"
                value={weightQml}
                onChange={e => setWeightQml(parseFloat(e.target.value))}
                className="rp-slider"
              />
              <div className="rp-slider-labels">
                <span>Symptom-focused</span>
                <span>QML-focused</span>
              </div>
            </div>
          </div>

          <button
            className={`rp-run-btn${loading ? ' loading' : ''}`}
            onClick={runRank}
            disabled={loading || !symptoms.trim()}
          >
            {loading
              ? <><span className="rp-spinner" />Analysing…</>
              : <>⬡ Rank by Symptom Relevance</>}
          </button>
        </div>

        {error && (
          <div className="rp-error">
            <span>⚠</span> {error}
            <button className="rp-retry-btn" onClick={runRank}>↺ Retry</button>
          </div>
        )}

        {result && (
          <div className="rp-results">
            {/* Summary header */}
            <div className="rp-results-header">
              <div className="rp-results-meta">
                <span className="rp-matched-badge">
                  {result.disease_matched === 'general' ? 'General' : result.disease_matched.charAt(0).toUpperCase() + result.disease_matched.slice(1)} matched
                </span>
                <span className="rp-weights-tag">QML {(result.weight_qml * 100).toFixed(0)}% · Relevance {(result.weight_symptom * 100).toFixed(0)}%</span>
              </div>
              <div className="rp-targets">
                <span className="rp-targets-label">Known targets:</span>
                {result.targets.slice(0, 4).map((t, i) => (
                  <span key={i} className="rp-target-chip">{t}</span>
                ))}
              </div>
            </div>

            {/* Ranked cards */}
            <div className="rp-cards-list">
              {result.ranked.map((candidate, i) => (
                <RankedCard key={i} candidate={candidate} isTop={i === 0} />
              ))}
            </div>
          </div>
        )}

        {!result && !loading && !error && (
          <div className="rp-empty-hint">
            Enter your target disease or symptoms above and click Rank to see how well this molecule matches the therapeutic context.
          </div>
        )}

      </div>
    </div>
  )
}