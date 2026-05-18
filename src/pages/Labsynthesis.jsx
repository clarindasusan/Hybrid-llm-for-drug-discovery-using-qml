import React, { useState, useEffect, useRef } from 'react'
import './LabSynthesis.css'


const HAZARD_COLOR = { low: '#6ee7b7', medium: '#fbbf24', high: '#f87171' }
const HAZARD_BG    = { low: 'rgba(110,231,183,.1)', medium: 'rgba(251,191,36,.08)', high: 'rgba(248,113,113,.1)' }
const DIFF_COLOR   = { Easy: '#6ee7b7', Moderate: '#fbbf24', Challenging: '#f87171' }
const CATEGORY_ICON = { PPE: '🥽', Waste: '♻', Storage: '🧊', Emergency: '🚨' }

// ── Lab Conditions Strip ───────────────────────────────────────────────────────
function LabConditionsStrip({ apiBase }) {
  const [conditions, setConditions] = useState(null)
  const [status, setStatus]         = useState('loading') // 'loading' | 'ok' | 'error' | 'stale'
  const [lastUpdated, setLastUpdated] = useState(null)
  const intervalRef = useRef(null)

  const fetchConditions = () => {
    fetch(`${apiBase}/lab-conditions/latest`)
      .then(async res => {
        const json = await res.json()
        if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`)
        if (json.message) {
          // "No readings received yet"
          setStatus('stale')
          setConditions(null)
          return
        }
        setConditions(json)
        setLastUpdated(new Date())
        setStatus('ok')
      })
      .catch(() => setStatus('error'))
  }

  useEffect(() => {
    fetchConditions()
    intervalRef.current = setInterval(fetchConditions, 60_000)
    return () => clearInterval(intervalRef.current)
  }, [apiBase])

  // ── No data yet ──
  if (status === 'loading') {
    return (
      <div className="lc-strip lc-strip--loading">
        <span className="lc-dot lc-dot--pulse" />
        <span className="lc-label">Fetching lab conditions…</span>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="lc-strip lc-strip--error">
        <span className="lc-icon">⚠</span>
        <span className="lc-label">Lab monitor unreachable — sensor data unavailable</span>
        <button className="lc-retry" onClick={fetchConditions}>Retry</button>
      </div>
    )
  }

  if (status === 'stale' || !conditions) {
    return (
      <div className="lc-strip lc-strip--stale">
        <span className="lc-icon">📡</span>
        <span className="lc-label">Awaiting first reading from Raspberry Pi sensor…</span>
      </div>
    )
  }

  // ── Use warnings[] from backend to determine out-of-range params ──
  const warnedParams = new Set((conditions.warnings || []).map(w => w.parameter))
  const isOk = param => !warnedParams.has(param)

  // Find warning detail for tooltip
  const warnDetail = param => {
    const w = (conditions.warnings || []).find(w => w.parameter === param)
    return w ? `${w.status}: ${w.value} (range ${w.range})` : ''
  }

  const metrics = [
    conditions.temperature != null && {
      icon: '🌡', label: 'Temp',     value: `${Number(conditions.temperature).toFixed(1)} °C`,
      ok: isOk('temperature'), warn: warnDetail('temperature') },
    conditions.humidity    != null && {
      icon: '💧', label: 'Humidity', value: `${Number(conditions.humidity).toFixed(1)} %`,
      ok: isOk('humidity'),    warn: warnDetail('humidity') },
    conditions.pressure    != null && {
      icon: '🔵', label: 'Pressure', value: `${Number(conditions.pressure).toFixed(1)} hPa`,
      ok: isOk('pressure'),    warn: warnDetail('pressure') },
    conditions.co2_ppm     != null && {
      icon: '🫧', label: 'CO₂',      value: `${conditions.co2_ppm} ppm`,
      ok: isOk('co2_ppm'),     warn: warnDetail('co2_ppm') },
    conditions.voc_ppb     != null && {
      icon: '🧪', label: 'VOC',      value: `${conditions.voc_ppb} ppb`,
      ok: isOk('voc_ppb'),     warn: warnDetail('voc_ppb') },
    conditions.light_lux   != null && {
      icon: '💡', label: 'Light',    value: `${conditions.light_lux} lux`,
      ok: isOk('light_lux'),   warn: warnDetail('light_lux') },
  ].filter(Boolean)

  const labReady   = conditions.lab_ready
  const readyColor = labReady ? '#6ee7b7' : '#f87171'
  const readyBg    = labReady ? 'rgba(110,231,183,.12)' : 'rgba(248,113,113,.12)'

  return (
    <div className={`lc-strip lc-strip--live ${labReady ? 'lc-strip--ready' : 'lc-strip--not-ready'}`}>
      {/* Ready badge */}
      <div className="lc-ready-badge" style={{ color: readyColor, background: readyBg, borderColor: readyColor + '44' }}>
        <span className="lc-dot" style={{ background: readyColor }} />
        {labReady ? 'Lab Ready' : 'Lab Not Ready'}
      </div>

      {/* Metrics */}
      <div className="lc-metrics">
        {metrics.map((m, i) => (
          <div key={i} className={`lc-metric${m.ok ? '' : ' lc-metric--warn'}`} title={m.warn || undefined}>
            <span className="lc-metric-icon">{m.icon}</span>
            <div className="lc-metric-body">
              <span className="lc-metric-label">{m.label}</span>
              <span className="lc-metric-val" style={{ color: m.ok ? '#6ee7b7' : '#f87171' }}>
                {m.value}
              </span>
            </div>
            {!m.ok && <span className="lc-metric-warn-icon">⚠</span>}
          </div>
        ))}
      </div>

      {/* Inline warnings */}
      {conditions.warnings && conditions.warnings.length > 0 && (
        <div className="lc-warnings">
          {conditions.warnings.map((w, i) => (
            <span key={i} className="lc-warning-pill">
              ⚠ {w.parameter}: {w.value} — {w.status} (range {w.range})
            </span>
          ))}
        </div>
      )}

      {/* Timestamp */}
      <div className="lc-right">
        {conditions.notes && (
          <span className="lc-notes" title={conditions.notes}>
            {conditions.notes.length > 40 ? conditions.notes.slice(0, 40) + '…' : conditions.notes}
          </span>
        )}
        {(conditions.timestamp || lastUpdated) && (
          <span className="lc-timestamp">
            {conditions.timestamp
              ? `Sensor: ${new Date(conditions.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
              : `Polled: ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`}
          </span>
        )}
      </div>
    </div>
  )
}

// ── Step card ──────────────────────────────────────────────────────────────────
function StepCard({ step, expanded, onToggle }) {
  return (
    <div className={`ls-step-card${expanded ? ' expanded' : ''}`}>
      <button className="ls-step-header" onClick={onToggle}>
        <div className="ls-step-num">Step {step.step}</div>
        <div className="ls-step-summary">
          <div className="ls-step-reaction">{step.reaction}</div>
          <div className="ls-step-meta">
            <span className="ls-step-yield">{step.yield_est} yield</span>
            <span className="ls-step-diff" style={{ color: DIFF_COLOR[step.difficulty] || '#94a3b8' }}>
              {step.difficulty}
            </span>
          </div>
        </div>
        <span className="ls-step-chevron">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="ls-step-body">
          <p className="ls-step-desc">{step.description}</p>

          <div className="ls-step-section-label">Reagents</div>
          <div className="ls-step-reagents">
            {step.reagents.map((r, i) => (
              <span key={i} className="ls-reagent-chip">{r}</span>
            ))}
          </div>

          <div className="ls-step-row">
            <div className="ls-step-detail">
              <div className="ls-step-section-label">Conditions</div>
              <div className="ls-step-conditions">{step.conditions}</div>
            </div>
          </div>

          <div className="ls-step-section-label">Practical Notes</div>
          <div className="ls-step-notes">{step.notes}</div>
        </div>
      )}
    </div>
  )
}

// ── Reagent row ────────────────────────────────────────────────────────────────
function ReagentRow({ reagent }) {
  const hl = reagent.hazard_level || 'medium'
  return (
    <div className="ls-reagent-row" style={{ borderLeftColor: HAZARD_COLOR[hl] }}>
      <div className="ls-reagent-name">{reagent.name}</div>
      <div className="ls-reagent-role">{reagent.role}</div>
      <div className="ls-reagent-right">
        {reagent.cas && <span className="ls-reagent-cas">CAS {reagent.cas}</span>}
        <span className="ls-hazard-tag"
          style={{ color: HAZARD_COLOR[hl], background: HAZARD_BG[hl], borderColor: HAZARD_COLOR[hl] + '44' }}>
          {hl.toUpperCase()}
        </span>
      </div>
      <div className="ls-reagent-hazard">{reagent.hazard}</div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function LabSynthesis({ smiles, score, apiBase, onClose }) {
  const [data,       setData]       = useState(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [activeTab,  setActiveTab]  = useState('route')   // 'route' | 'reagents' | 'safety'
  const [expandedStep, setExpandedStep] = useState(0)     // index of expanded step (0 = first)

  useEffect(() => {
    if (!smiles) return
    setLoading(true); setError(null); setData(null)

    const controller = new AbortController()
    const timeout    = setTimeout(() => controller.abort(), 60000)

    fetch(`${apiBase}/synthesis`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ smiles, score: score || 0 }),
      signal:  controller.signal,
    })
      .then(async res => {
        clearTimeout(timeout)
        const text = await res.text()
        if (!text.trim()) throw new Error('Empty response — server may be waking up, try again.')
        let json
        try { json = JSON.parse(text) } catch { throw new Error(`Server error: ${text.slice(0, 120)}`) }
        if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`)
        if (json.error) throw new Error(json.error)
        setData(json)
      })
      .catch(err => {
        clearTimeout(timeout)
        setError(err.name === 'AbortError'
          ? 'Request timed out — server may be sleeping. Try again in 30s.'
          : err.message || String(err))
      })
      .finally(() => setLoading(false))

    return () => { clearTimeout(timeout); controller.abort() }
  }, [smiles, apiBase, score])

  const retry = () => {
    setError(null); setData(null); setLoading(true)
    fetch(`${apiBase}/synthesis`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, score: score || 0 }),
    })
      .then(async res => {
        const json = await res.json()
        if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`)
        if (json.error) throw new Error(json.error)
        setData(json)
      })
      .catch(err => setError(err.message || String(err)))
      .finally(() => setLoading(false))
  }

  const complexityColor = { Simple: '#6ee7b7', Moderate: '#fbbf24', Complex: '#f87171' }

  return (
    <div className="ls-page">
      {/* ── Top bar ── */}
      <div className="ls-topbar">
        <button className="ls-back-btn" onClick={onClose}>← Back to Results</button>
        <div className="ls-topbar-center">
          <span className="ls-eyebrow">⚗ Lab Synthesis Route</span>
          <span className="ls-smiles-tag" title={smiles}>
            {smiles.length > 50 ? smiles.slice(0, 50) + '…' : smiles}
          </span>
        </div>
        <div className="ls-topbar-right">
          {data && (
            <div className="ls-complexity-badge"
              style={{ color: complexityColor[data.complexity] || '#94a3b8',
                       borderColor: (complexityColor[data.complexity] || '#94a3b8') + '44' }}>
              {data.complexity} · {data.overall_difficulty}
            </div>
          )}
        </div>
      </div>

      {/* ── Lab Conditions Strip ── */}
      <LabConditionsStrip apiBase={apiBase} />

      {/* ── Loading ── */}
      {loading && (
        <div className="ls-loading">
          <div className="ls-loading-ring" />
          <div className="ls-loading-text">Analysing molecule and planning synthesis route…</div>
          <div className="ls-loading-sub">Running retrosynthetic analysis with RDKit</div>
        </div>
      )}

      {/* ── Error ── */}
      {error && !loading && !data && (
        <div className="ls-error-wrap">
          <div className="ls-error-icon">⚠</div>
          <div className="ls-error-msg">{error}</div>
          <button className="ls-retry-btn" onClick={retry}>↺ Retry</button>
        </div>
      )}

      {/* ── Content ── */}
      {data && !loading && (
        <div className="ls-content">

          {/* Summary bar */}
          <div className="ls-summary-bar">
            <div className="ls-summary-card">
              <div className="ls-summary-label">Total Steps</div>
              <div className="ls-summary-val">{data.total_steps}</div>
            </div>
            <div className="ls-summary-card">
              <div className="ls-summary-label">Overall Yield</div>
              <div className="ls-summary-val">{data.overall_yield_est}</div>
            </div>
            <div className="ls-summary-card">
              <div className="ls-summary-label">Difficulty</div>
              <div className="ls-summary-val"
                style={{ color: DIFF_COLOR[data.overall_difficulty] || '#94a3b8' }}>
                {data.overall_difficulty}
              </div>
            </div>
            <div className="ls-summary-card">
              <div className="ls-summary-label">Reagents</div>
              <div className="ls-summary-val">{data.all_reagents.length}</div>
            </div>
          </div>

          {/* IUPAC hint */}
          <div className="ls-iupac-box">
            <span className="ls-iupac-label">Structure hint</span>
            <span className="ls-iupac-val">{data.iupac_hint}</span>
          </div>

          {/* Retrosynthesis summary */}
          <div className="ls-retro-box">
            <div className="ls-retro-label">Retrosynthetic Summary</div>
            <p className="ls-retro-text">{data.retrosynthesis_summary}</p>
          </div>

          {/* Tab bar */}
          <div className="ls-tabs">
            {[
              { id: 'route',    label: `Synthesis Route (${data.steps.length} steps)` },
              { id: 'reagents', label: `Reagents (${data.all_reagents.length})` },
              { id: 'safety',   label: `Safety (${data.safety_notes.length})` },
            ].map(t => (
              <button key={t.id}
                className={`ls-tab${activeTab === t.id ? ' active' : ''}`}
                onClick={() => setActiveTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>

          {/* ── ROUTE TAB ── */}
          {activeTab === 'route' && (
            <div className="ls-steps-list">
              {data.steps.map((step, i) => (
                <StepCard
                  key={step.step}
                  step={step}
                  expanded={expandedStep === i}
                  onToggle={() => setExpandedStep(expandedStep === i ? -1 : i)}
                />
              ))}
            </div>
          )}

          {/* ── REAGENTS TAB ── */}
          {activeTab === 'reagents' && (
            <div className="ls-reagents-list">
              {/* Hazard legend */}
              <div className="ls-hazard-legend">
                {['low', 'medium', 'high'].map(hl => (
                  <span key={hl} className="ls-hazard-legend-item">
                    <span className="ls-hazard-dot" style={{ background: HAZARD_COLOR[hl] }} />
                    {hl.charAt(0).toUpperCase() + hl.slice(1)} hazard
                  </span>
                ))}
              </div>
              {data.all_reagents.map((r, i) => (
                <ReagentRow key={i} reagent={r} />
              ))}
            </div>
          )}

          {/* ── SAFETY TAB ── */}
          {activeTab === 'safety' && (
            <div className="ls-safety-list">
              {data.safety_notes.map((note, i) => (
                <div key={i} className={`ls-safety-card ls-safety-${note.category.toLowerCase()}`}>
                  <div className="ls-safety-header">
                    <span className="ls-safety-icon">
                      {CATEGORY_ICON[note.category] || '⚠'}
                    </span>
                    <span className="ls-safety-category">{note.category}</span>
                  </div>
                  <p className="ls-safety-detail">{note.detail}</p>
                </div>
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  )
}