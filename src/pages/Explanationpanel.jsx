import React, { useState, useEffect } from 'react'
import './ExplanationPanel.css'


// ─── SHAP waterfall bar ───────────────────────────────────────────────────────
function SHAPBar({ label, value, unit, shap, ideal, direction }) {
  const [anim, setAnim] = useState(0)
  useEffect(() => {
    const t = setTimeout(() => setAnim(Math.abs(shap)), 80)
    return () => clearTimeout(t)
  }, [shap])

  const maxShap   = 0.15
  const pct       = Math.min((anim / maxShap) * 100, 100)
  const color     = direction === 'positive' ? '#6ee7b7'
                  : direction === 'negative' ? '#f87171'
                  : '#64748b'
  const arrow     = direction === 'positive' ? '▲' : direction === 'negative' ? '▼' : '●'
  const shapLabel = shap > 0 ? `+${shap.toFixed(4)}` : shap.toFixed(4)

  return (
    <div className="xai-shap-row">
      <div className="xai-shap-meta">
        <span className="xai-shap-label">{label}</span>
        <div className="xai-shap-right">
          <span className="xai-shap-value">
            {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(3)) : value}
            {unit}
          </span>
          <span className="xai-shap-ideal">{ideal}</span>
          <span className="xai-shap-score" style={{ color }}>
            {arrow} {shapLabel}
          </span>
        </div>
      </div>
      <div className="xai-shap-track">
        <div className="xai-shap-fill" style={{
          width:      `${pct}%`,
          background: color,
          boxShadow:  `0 0 8px ${color}55`,
          transition: 'width 0.9s cubic-bezier(.4,0,.2,1)',
          marginLeft: direction === 'negative' ? 'auto' : '0',
        }} />
      </div>
    </div>
  )
}

// ─── PCA summary bar ──────────────────────────────────────────────────────────
function PCASummary({ summary }) {
  if (!summary) return null
  const { total_positive, total_negative, n_components, max_component } = summary
  const total = Math.abs(total_positive) + Math.abs(total_negative)
  const posPct = total > 0 ? (Math.abs(total_positive) / total) * 100 : 50

  return (
    <div className="xai-pca-summary">
      <div className="xai-pca-title">
        Quantum Latent Space · {n_components} PCA components
      </div>
      <div className="xai-pca-bar-wrap">
        <div className="xai-pca-bar">
          <div className="xai-pca-pos" style={{ width: `${posPct}%` }} />
          <div className="xai-pca-neg" style={{ width: `${100 - posPct}%` }} />
        </div>
        <div className="xai-pca-labels">
          <span style={{ color: '#6ee7b7' }}>+{total_positive.toFixed(3)} positive</span>
          <span style={{ color: '#f87171' }}>{total_negative.toFixed(3)} negative</span>
        </div>
      </div>
      <div className="xai-pca-note">
        Most influential latent dimension: PC-{max_component}.
        SHAP values computed in PCA space — descriptor contributions use chemically-grounded heuristic attribution.
      </div>
    </div>
  )
}

// ─── Confidence badge ─────────────────────────────────────────────────────────
function ConfidenceBadge({ confidence }) {
  const config = {
    high:   { color: '#6ee7b7', bg: 'rgba(110,231,183,0.1)', border: 'rgba(110,231,183,0.3)', label: 'High Confidence' },
    medium: { color: '#d4af72', bg: 'rgba(212,175,114,0.1)', border: 'rgba(212,175,114,0.3)', label: 'Medium Confidence' },
    low:    { color: '#f87171', bg: 'rgba(248,113,113,0.1)', border: 'rgba(248,113,113,0.3)', label: 'Low Confidence' },
  }
  const c = config[confidence] || config.medium
  return (
    <span className="xai-confidence-badge" style={{ color: c.color, background: c.bg, border: `1px solid ${c.border}` }}>
      ◈ {c.label}
    </span>
  )
}

// ─── Atom highlight legend ────────────────────────────────────────────────────
function AtomHighlightLegend({ atoms, totalAtoms }) {
  if (!atoms || atoms.length === 0) return null
  const tot = totalAtoms > 0 ? totalAtoms : 30
  const pct = Math.min(Math.round((atoms.length / tot) * 100), 100)
  return (
    <div className="xai-atom-legend">
      <div className="xai-atom-legend-bar">
        <div className="xai-atom-legend-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="xai-atom-legend-label">
        {atoms.length} atoms drive this prediction
        <em> (indices: {atoms.slice(0, 8).join(', ')}{atoms.length > 8 ? '…' : ''})</em>
      </span>
    </div>
  )
}

// ─── Loading state ────────────────────────────────────────────────────────────
function ExplainLoading({ wakeMsg }) {
  return (
    <div className="xai-loading">
      <div className="xai-loading-ring" />
      <div className="xai-loading-text">
        {wakeMsg ? 'Waking up server…' : 'Running SHAP Analysis…'}
      </div>
      <div className="xai-loading-sub">
        {wakeMsg
          ? wakeMsg
          : 'Computing Shapley values across 10 background molecules · may take up to 60s'}
      </div>
      {wakeMsg && (
        <div className="xai-wake-hint">
          ☕ Your Hugging Face Space was sleeping — waking it up now (60–90s).
          This only happens on first use. Subsequent visits are instant.
        </div>
      )}
    </div>
  )
}

// ─── Main ExplanationPanel ────────────────────────────────────────────────────
export default function ExplanationPanel({ smiles, apiBase = '', totalAtoms = 0 }) {
  const [data,        setData]        = useState(null)
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)
  const [tab,         setTab]         = useState('descriptors')
  const [retry,       setRetry]       = useState(0)
  const [autoRetries, setAutoRetries] = useState(0)
  const [wakeMsg,     setWakeMsg]     = useState(null)  // shown during auto-retry

  const MAX_AUTO_RETRIES = 3  // try up to 3 times automatically before showing error

  useEffect(() => {
    if (!smiles) return
    setLoading(true); setError(null); setData(null)

    let cancelled = false

    const run = async () => {
      // ── Step 1: ping /health to wake the Space (up to 120s) ──────────────
      setWakeMsg('Waking up server… please wait')
      try {
        const hc = new AbortController()
        const hTimeout = setTimeout(() => hc.abort(), 120000)
        const hr = await fetch(`${apiBase}/health`, { signal: hc.signal })
        clearTimeout(hTimeout)
        if (!hr.ok) throw new Error('health check failed')
      } catch(e) {
        // health ping failed — Space may be truly down, but still try /explain
      }
      if (cancelled) return

      // ── Step 2: call /explain now that the Space is awake ────────────────
      setWakeMsg(null)
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 180000) // 3 min

      try {
        const r = await fetch(`${apiBase}/explain`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ smiles, include_admet: true }),
          signal:  controller.signal,
        })
        clearTimeout(timeout)
        if (cancelled) return

        const text = await r.text()
        if (!text.trim()) throw new Error(`Empty response (HTTP ${r.status})`)
        let json
        try { json = JSON.parse(text) }
        catch { throw new Error(`Non-JSON response: ${text.slice(0, 200)}`) }
        if (!r.ok) throw new Error(json?.detail || `HTTP ${r.status}`)
        if (json.error) throw new Error(json.error)

        if (!cancelled) { setData(json); setLoading(false) }
      } catch(err) {
        clearTimeout(timeout)
        if (cancelled) return
        const isTimeout = err.name === 'AbortError'
        if (isTimeout && autoRetries < MAX_AUTO_RETRIES) {
          setWakeMsg(`Still computing… retrying (${autoRetries + 1}/${MAX_AUTO_RETRIES})`)
          setAutoRetries(c => c + 1)
          setTimeout(() => { if (!cancelled) setRetry(r => r + 1) }, 2000)
        } else {
          setWakeMsg(null)
          setError(isTimeout
            ? 'SHAP analysis timed out. Make sure the /explain endpoint is added to main.py on your Hugging Face Space.'
            : err.message || String(err))
          setLoading(false)
        }
      }
    }

    run()
    return () => { cancelled = true }
  }, [smiles, apiBase, retry])

  if (loading) return <ExplainLoading wakeMsg={wakeMsg} />

  if (error || !data) return (
    <div className="xai-error">
      <div className="xai-error-icon">⚠</div>
      <div className="xai-error-msg">{error || 'Unknown error'}</div>
      <div className="xai-error-hint">
        Make sure <code>explainer.py</code> is in your Space and the <code>/explain</code> endpoint is added to <code>main.py</code>.
      </div>
      <button className="xai-retry-btn"
        onClick={() => { setError(null); setLoading(true); setRetry(r => r + 1) }}>
        ↺ Retry
      </button>
    </div>
  )

  // ── Destructure exactly what explainer.py returns ─────────────────────────
  const {
    score,
    shap_base_value,
    shap_pca_summary,
    descriptor_contributions,   // [{name, label, unit, ideal, value, shap, direction, magnitude}]
    fingerprint_contributions,  // [{bit, shap, direction, atoms, present}]
    important_atoms,            // [int, ...]
    explanation_text,
    confidence,
  } = data

  const positiveDesc = descriptor_contributions.filter(d => d.direction === 'positive')
  const negativeDesc = descriptor_contributions.filter(d => d.direction === 'negative')
  const neutralDesc  = descriptor_contributions.filter(d => d.direction === 'neutral')

  return (
    <div className="xai-panel">

      {/* ── Header ── */}
      <div className="xai-header">
        <div className="xai-header-left">
          <div className="xai-eyebrow">
            <span className="xai-eyebrow-dot" />
            Explainable AI · KernelSHAP · QML Model
          </div>
          <h3 className="xai-title">
            Why did this molecule score {Math.round(score * 100)}/100?
          </h3>
          <div className="xai-header-meta">
            <ConfidenceBadge confidence={confidence} />
            <span className="xai-baseline">
              Baseline (E[f(x)]): {Math.round(shap_base_value * 100)}/100
            </span>
          </div>
        </div>

        {/* Score delta */}
        <div className="xai-score-delta">
          <div className="xai-delta-base">
            <span className="xai-delta-num">{Math.round(shap_base_value * 100)}</span>
            <span className="xai-delta-label">baseline</span>
          </div>
          <div className="xai-delta-arrow">
            {score >= shap_base_value ? '→ ▲' : '→ ▼'}
          </div>
          <div className={`xai-delta-final ${score >= shap_base_value ? 'up' : 'down'}`}>
            <span className="xai-delta-num">{Math.round(score * 100)}</span>
            <span className="xai-delta-label">final score</span>
          </div>
        </div>
      </div>

      {/* ── PCA summary ── */}
      <PCASummary summary={shap_pca_summary} />

      {/* ── Important atoms banner ── */}
      {important_atoms && important_atoms.length > 0 && (
        <div className="xai-atom-banner">
          <span className="xai-atom-banner-icon">⬡</span>
          <span>
            <strong>{important_atoms.length} atoms</strong> drive this prediction · indices:{' '}
            {important_atoms.slice(0, 10).join(', ')}{important_atoms.length > 10 ? '…' : ''}
          </span>
          <AtomHighlightLegend atoms={important_atoms} totalAtoms={totalAtoms} />
        </div>
      )}

      {/* ── Tabs ── */}
      <div className="xai-tabs">
        {[
          { id: 'descriptors',  label: 'Descriptors' },
          { id: 'fingerprints', label: 'Fingerprints' },
          { id: 'text',         label: 'Explanation' },
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`xai-tab ${tab === t.id ? 'active' : ''}`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── DESCRIPTORS TAB ── */}
      {tab === 'descriptors' && (
        <div className="xai-pane xai-fade">
          <div className="xai-heuristic-note">
            ◈ Descriptor contributions use chemically-grounded heuristic attribution
            scaled by KernelSHAP signal magnitude in PCA space.
          </div>
          <div className="xai-legend-row">
            <span className="xai-legend-item pos">▲ Positive — pushed score up</span>
            <span className="xai-legend-item neg">▼ Negative — pushed score down</span>
            <span className="xai-legend-item neu">● Neutral — little impact</span>
          </div>

          {positiveDesc.length > 0 && (
            <div className="xai-group">
              <div className="xai-group-title">Positive Contributors</div>
              {positiveDesc.map(d => (
                <SHAPBar key={d.name}
                  label={d.label} value={d.value} unit={d.unit}
                  shap={d.shap} ideal={d.ideal} direction="positive" />
              ))}
            </div>
          )}

          {negativeDesc.length > 0 && (
            <div className="xai-group">
              <div className="xai-group-title">Negative Contributors</div>
              {negativeDesc.map(d => (
                <SHAPBar key={d.name}
                  label={d.label} value={d.value} unit={d.unit}
                  shap={d.shap} ideal={d.ideal} direction="negative" />
              ))}
            </div>
          )}

          {neutralDesc.length > 0 && (
            <div className="xai-group xai-group-neutral">
              <div className="xai-group-title">Neutral</div>
              {neutralDesc.map(d => (
                <SHAPBar key={d.name}
                  label={d.label} value={d.value} unit={d.unit}
                  shap={d.shap} ideal={d.ideal} direction="neutral" />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── FINGERPRINTS TAB ── */}
      {tab === 'fingerprints' && (
        <div className="xai-pane xai-fade">
          <div className="xai-fp-note">
            Top {fingerprint_contributions.length} Morgan fingerprint bits (radius 2) ranked
            by PCA loading magnitude. Bits are mapped back to generating atom environments.
          </div>
          <div className="xai-fp-list">
            {fingerprint_contributions.map((fp, i) => (
              <div key={fp.bit} className={`xai-fp-row ${fp.direction}`}>
                <div className="xai-fp-rank">#{i + 1}</div>
                <div className="xai-fp-bit">
                  <span className="xai-fp-bit-label">Bit {fp.bit}</span>
                  <span className={`xai-fp-present ${fp.present ? 'on' : 'off'}`}>
                    {fp.present ? '● ON' : '○ OFF'}
                  </span>
                </div>
                <div className="xai-fp-shap" style={{
                  color: fp.direction === 'positive' ? '#6ee7b7' : '#f87171'
                }}>
                  {fp.shap > 0 ? '+' : ''}{fp.shap.toFixed(4)}
                </div>
                <div className="xai-fp-atoms">
                  {fp.atoms && fp.atoms.length > 0
                    ? `Atoms: ${fp.atoms.slice(0, 5).join(', ')}${fp.atoms.length > 5 ? '…' : ''}`
                    : 'No atom mapping'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── EXPLANATION TEXT TAB ── */}
      {tab === 'text' && (
        <div className="xai-pane xai-fade">
          <div className="xai-text-card">
            <div className="xai-text-icon">◈</div>
            <p className="xai-text-body">{explanation_text}</p>
          </div>
          <div className="xai-method-note">
            <div className="xai-method-title">Methodology</div>
            <div className="xai-method-body">
              KernelSHAP treats the HybridQMLModel as a black box and computes
              Shapley values in PCA-compressed feature space using 10 background
              drug-like molecules. The expected value E[f(x)] = {Math.round(shap_base_value * 100)}/100
              represents the average prediction across the background set.
              Descriptor contributions use chemically-grounded heuristic attribution
              scaled by the PCA SHAP signal — direct PCA inversion per descriptor
              is avoided due to feature mixing. Atom highlights come from mapping
              the top Morgan fingerprint bits (by PCA loading) back to their
              generating atom environments.
            </div>
          </div>
        </div>
      )}

    </div>
  )
}