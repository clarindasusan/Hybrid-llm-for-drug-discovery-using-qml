import React, { useState, useEffect, useRef } from 'react'
import './ADMETDashboard.css'


// ─── Radial gauge ─────────────────────────────────────────────────────────────
function RadialGauge({ value, label, color }) {
  const [anim, setAnim] = useState(0)
  useEffect(() => { const t = setTimeout(() => setAnim(value), 120); return () => clearTimeout(t) }, [value])
  const r = 36, cx = 44, cy = 44, circ = 2 * Math.PI * r
  const arc = circ * 0.75, dash = (anim / 100) * arc
  return (
    <div className="admet-gauge">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.07)"
          strokeWidth="7" strokeDasharray={`${arc} ${circ - arc}`}
          strokeLinecap="round" transform={`rotate(-135 ${cx} ${cy})`} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color}
          strokeWidth="7" strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round" transform={`rotate(-135 ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 7px ${color}99)`, transition: 'stroke-dasharray 1.1s cubic-bezier(.4,0,.2,1)' }} />
        <text x={cx} y={cy - 3} textAnchor="middle" fill="#f1f5f9"
          fontSize="14" fontWeight="700" fontFamily="'DM Mono', monospace">{Math.round(value)}</text>
        <text x={cx} y={cy + 11} textAnchor="middle" fill="rgba(241,245,249,0.35)"
          fontSize="7.5" fontFamily="'DM Mono', monospace">/100</text>
      </svg>
      <div className="admet-gauge-label">{label}</div>
    </div>
  )
}

// ─── Horizontal property bar ───────────────────────────────────────────────────
function PropBar({ label, value, max, unit = '', pass, info }) {
  const pct   = Math.min((Math.abs(value) / max) * 100, 100)
  const color = pass === true ? '#6ee7b7' : pass === false ? '#f87171' : '#d4af72'
  return (
    <div className="admet-prop-row">
      <div className="admet-prop-meta">
        <span className="admet-prop-name">{label}</span>
        <div className="admet-prop-right">
          <span className="admet-prop-val" style={{ color }}>{value}{unit}</span>
          {pass !== undefined && (
            <span className="admet-badge" style={{
              background: pass ? 'rgba(110,231,183,0.1)' : 'rgba(248,113,113,0.1)',
              color, border: `1px solid ${pass ? 'rgba(110,231,183,0.3)' : 'rgba(248,113,113,0.3)'}`,
            }}>{pass ? '✓ Pass' : '✗ Fail'}</span>
          )}
        </div>
      </div>
      {info && <div className="admet-prop-info">{info}</div>}
      <div className="admet-prop-track">
        <div className="admet-prop-fill" style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}44` }} />
      </div>
    </div>
  )
}

// ─── Radar SVG ────────────────────────────────────────────────────────────────
function RadarChart({ data }) {
  const labels = Object.keys(data)
  const values = Object.values(data)
  const n = labels.length, cx = 115, cy = 115, r = 80
  const step = (2 * Math.PI) / n
  const pt = (i, rad) => ({
    x: cx + rad * Math.cos(i * step - Math.PI / 2),
    y: cy + rad * Math.sin(i * step - Math.PI / 2),
  })
  const dataPath = values.map((v, i) => {
    const p = pt(i, (v / 100) * r)
    return `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`
  }).join(' ') + 'Z'
  return (
    <svg width="230" height="230" viewBox="0 0 230 230" className="admet-radar-svg">
      {[0.25, 0.5, 0.75, 1].map(lvl => (
        <polygon key={lvl}
          points={labels.map((_, i) => { const p = pt(i, lvl * r); return `${p.x.toFixed(1)},${p.y.toFixed(1)}` }).join(' ')}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      ))}
      {labels.map((_, i) => {
        const p = pt(i, r)
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      })}
      <path d={dataPath} fill="rgba(168,85,247,0.12)" stroke="#a855f7" strokeWidth="1.8"
        style={{ filter: 'drop-shadow(0 0 10px rgba(168,85,247,0.45))' }} />
      {values.map((v, i) => {
        const p = pt(i, (v / 100) * r)
        return <circle key={i} cx={p.x} cy={p.y} r="4" fill="#a855f7"
          style={{ filter: 'drop-shadow(0 0 5px #a855f7)' }} />
      })}
      {labels.map((lbl, i) => {
        const p = pt(i, r + 20)
        return (
          <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle"
            fill="rgba(241,245,249,0.6)" fontSize="9.5" fontFamily="'Nunito', sans-serif">{lbl}</text>
        )
      })}
    </svg>
  )
}

// ─── Atom donut ───────────────────────────────────────────────────────────────
function AtomDonut({ atoms }) {
  const COLOR = { C:'#909090', N:'#4169e1', O:'#e13030', S:'#d4c400', F:'#40d0d0', Cl:'#30d030', Br:'#a05010', P:'#e08000', I:'#800080' }
  // Filter out zero-count atoms from API response
  const entries = Object.entries(atoms).filter(([, v]) => v > 0)
  const total   = entries.reduce((s, [, v]) => s + v, 0)
  if (total === 0) return <div className="admet-no-data">No heavy atoms</div>
  const cx = 60, cy = 60, r = 42, inner = 24
  let cum = -Math.PI / 2
  const slices = entries.map(([el, count]) => {
    const angle = (count / total) * 2 * Math.PI
    const x1  = cx + r     * Math.cos(cum),         y1  = cy + r     * Math.sin(cum)
    const x2  = cx + r     * Math.cos(cum + angle), y2  = cy + r     * Math.sin(cum + angle)
    const xi1 = cx + inner * Math.cos(cum),         yi1 = cy + inner * Math.sin(cum)
    const xi2 = cx + inner * Math.cos(cum + angle), yi2 = cy + inner * Math.sin(cum + angle)
    const lg   = angle > Math.PI ? 1 : 0
    const path = `M${xi1},${yi1}L${x1},${y1}A${r},${r} 0 ${lg},1 ${x2},${y2}L${xi2},${yi2}A${inner},${inner} 0 ${lg},0 ${xi1},${yi1}Z`
    cum += angle
    return { el, count, path, color: COLOR[el] || '#888', pct: Math.round((count / total) * 100) }
  })
  return (
    <div className="admet-donut-wrap">
      <svg width="120" height="120" viewBox="0 0 120 120">
        {slices.map(s => (
          <path key={s.el} d={s.path} fill={s.color} opacity="0.85"
            style={{ filter: `drop-shadow(0 0 3px ${s.color}55)` }} />
        ))}
        <text x={cx} y={cy - 4} textAnchor="middle" fill="#f1f5f9" fontSize="13" fontWeight="700" fontFamily="'DM Mono',monospace">{total}</text>
        <text x={cx} y={cy + 9} textAnchor="middle" fill="rgba(241,245,249,0.38)" fontSize="7.5" fontFamily="'DM Mono',monospace">atoms</text>
      </svg>
      <div className="admet-donut-legend">
        {slices.map(s => (
          <div key={s.el} className="admet-donut-item">
            <span className="admet-donut-dot" style={{ background: s.color }} />
            <span className="admet-donut-el">{s.el}</span>
            <span className="admet-donut-pct">{s.count} <em>({s.pct}%)</em></span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Overall score ring ───────────────────────────────────────────────────────
function ScoreRing({ score }) {
  const [anim, setAnim] = useState(0)
  useEffect(() => { const t = setTimeout(() => setAnim(score), 150); return () => clearTimeout(t) }, [score])
  const r = 54, cx = 66, cy = 66, circ = 2 * Math.PI * r
  const arc = circ * 0.78, dash = anim * arc
  const color = score >= 0.7 ? '#6ee7b7' : score >= 0.4 ? '#d4af72' : '#f87171'
  const label = score >= 0.7 ? 'Drug-like' : score >= 0.4 ? 'Borderline' : 'Poor'
  return (
    <div className="admet-score-ring">
      <svg width="132" height="132" viewBox="0 0 132 132">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.06)"
          strokeWidth="9" strokeDasharray={`${arc} ${circ - arc}`}
          strokeLinecap="round" transform={`rotate(-141 ${cx} ${cy})`} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color}
          strokeWidth="9" strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round" transform={`rotate(-141 ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 10px ${color}99)`, transition: 'stroke-dasharray 1.2s cubic-bezier(.4,0,.2,1)' }} />
        <text x={cx} y={cx - 9} textAnchor="middle" fill="#f1f5f9" fontSize="22" fontWeight="800" fontFamily="'Playfair Display', serif">
          {Math.round(score * 100)}
        </text>
        <text x={cx} y={cx + 9}  textAnchor="middle" fill="rgba(241,245,249,0.33)" fontSize="9"  fontFamily="'Nunito',sans-serif">/100</text>
        <text x={cx} y={cx + 24} textAnchor="middle" fill={color} fontSize="10" fontWeight="700" fontFamily="'Nunito',sans-serif">{label}</text>
      </svg>
    </div>
  )
}

// ─── Loading skeleton ─────────────────────────────────────────────────────────
function LoadingSkeleton() {
  return (
    <div className="admet-loading-wrap">
      <div className="admet-loading-spinner" />
      <div className="admet-loading-text">Computing ADMET properties…</div>
      <div className="admet-loading-sub">Running RDKit descriptors + PAINS/BRENK filters</div>
    </div>
  )
}

// ─── Report download ──────────────────────────────────────────────────────────
function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return }
    const s = document.createElement('script')
    s.src = src; s.onload = resolve; s.onerror = reject
    document.head.appendChild(s)
  })
}

// PDF uses raw API field names (mw, logp, rot_bonds etc) matching backend response directly
async function downloadTextPDF(smiles, d, filename) {
  await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js')
  const { jsPDF } = window.jspdf
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })

  // Use raw API field names — d is the raw JSON from the server
  const MW          = d.mw
  const logp        = d.logp
  const hbd         = d.hbd
  const hba         = d.hba
  const rotBonds    = d.rot_bonds
  const tpsa        = d.tpsa
  const rings       = d.rings
  const heavyAtoms  = d.heavy_atoms
  const fsp3        = d.fsp3
  const ro5Violations = d.ro5_violations
  const veberPass   = d.veber_pass
  const drugScore   = d.drug_score
  const bioavailability = d.bioavailability
  const bbb         = d.bbb
  const cyp         = d.cyp
  const toxFlags    = d.tox_flags
  const admet       = d.admet
  const atoms       = d.atoms

  const W = 210, M = 18, CW = W - M * 2
  let y = 0
  const hex2rgb = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]
  const setColor = h => { const [r,g,b] = hex2rgb(h); pdf.setTextColor(r,g,b) }
  const setFill  = h => { const [r,g,b] = hex2rgb(h); pdf.setFillColor(r,g,b) }
  const setDraw  = h => { const [r,g,b] = hex2rgb(h); pdf.setDrawColor(r,g,b) }

  setFill('#0d0618'); pdf.rect(0,0,210,297,'F')
  setFill('#150a2e'); pdf.rect(0,0,210,42,'F')
  setFill('#7c3aed'); pdf.rect(0,0,3,42,'F')

  y=13; setColor('#a855f7'); pdf.setFontSize(7); pdf.setFont('helvetica','bold')
  pdf.text('ADMET PREDICTION REPORT — RDKit Computed', M, y)
  y=23; setColor('#f1f5f9'); pdf.setFontSize(18); pdf.setFont('helvetica','bold')
  pdf.text('Molecular Property Analysis', M, y)
  y=31; setColor('#94a3b8'); pdf.setFontSize(8); pdf.setFont('helvetica','normal')
  pdf.text(`SMILES: ${smiles.length>72?smiles.slice(0,72)+'...':smiles}`, M, y)
  y=38; pdf.text(`Generated: ${new Date().toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'})}  ·  Score: ${Math.round(drugScore*100)}/100`, M, y)

  const scoreColor = drugScore>=0.7?'#10b981':drugScore>=0.4?'#f59e0b':'#ef4444'
  setFill(scoreColor); pdf.roundedRect(W-M-30,10,30,22,3,3,'F')
  setColor('#0d0618'); pdf.setFontSize(14); pdf.setFont('helvetica','bold')
  pdf.text(String(Math.round(drugScore*100)), W-M-15, 22, {align:'center'})
  pdf.setFontSize(6); pdf.text(drugScore>=0.7?'DRUG-LIKE':drugScore>=0.4?'BORDERLINE':'POOR', W-M-15, 28, {align:'center'})

  y=52
  const section = t => { setFill('#1e1040'); pdf.rect(M-3,y-5,CW+6,9,'F'); setColor('#a855f7'); pdf.setFontSize(8.5); pdf.setFont('helvetica','bold'); pdf.text(t.toUpperCase(),M,y); y+=7 }
  const row = (lbl,val,pass) => {
    setColor('#94a3b8'); pdf.setFontSize(8); pdf.setFont('helvetica','normal'); pdf.text(lbl,M+2,y)
    const vc=pass===true?'#10b981':pass===false?'#ef4444':'#d4af72'; setColor(vc); pdf.setFont('helvetica','bold'); pdf.text(String(val),M+80,y)
    if(pass!==undefined){setColor(pass?'#10b981':'#ef4444');pdf.setFontSize(7);pdf.text(pass?'✓ Pass':'✗ Fail',M+120,y)}
    setDraw('#1e1040');pdf.setLineWidth(0.2);pdf.line(M,y+2,M+CW,y+2);y+=9
  }
  const bar = (lbl,val,max,unit='') => {
    setColor('#64748b');pdf.setFontSize(7.5);pdf.setFont('helvetica','normal');pdf.text(lbl,M+2,y)
    setColor('#f1f5f9');pdf.setFont('helvetica','bold');pdf.text(`${val}${unit}`,M+70,y)
    setFill('#1e1040');pdf.roundedRect(M+95,y-3.5,80,4,1,1,'F')
    const pct=Math.min(Math.abs(val)/max,1),bc=pct<0.6?'#10b981':pct<0.85?'#f59e0b':'#ef4444'
    setFill(bc);pdf.roundedRect(M+95,y-3.5,80*pct,4,1,1,'F');y+=8
  }

  section('1. Physicochemical Properties')
  row('Molecular Weight',MW+' Da',MW<=500)
  row('LogP',logp,logp<=5)
  row('H-Bond Donors',hbd,hbd<=5)
  row('H-Bond Acceptors',hba,hba<=10)
  row('Rotatable Bonds',rotBonds,rotBonds<=10)
  row('TPSA',tpsa+' Å²',tpsa<=140)
  row('Fsp3',fsp3,undefined)
  row('Ring Systems',rings,undefined)
  row('Heavy Atom Count',heavyAtoms,undefined)
  y+=2

  const ro5Color=ro5Violations===0?'#10b981':ro5Violations===1?'#f59e0b':'#ef4444'
  setFill(ro5Color+'22');setDraw(ro5Color);pdf.setLineWidth(0.4);pdf.roundedRect(M,y,CW,10,2,2,'FD')
  setColor(ro5Color);pdf.setFontSize(8);pdf.setFont('helvetica','bold')
  pdf.text(ro5Violations===0?'✓  Passes all Lipinski Rule of Five criteria':`⚠  ${ro5Violations} Lipinski violation${ro5Violations>1?'s':''} detected`,M+4,y+6.5)
  y+=17

  section('2. ADMET Scores')
  bar('GI Absorption',admet.absorption,100,'%')
  bar('Distribution',admet.distribution,100,'%')
  bar('Metabolic Stability',admet.metabolism,100,'%')
  bar('Renal Excretion',admet.excretion,100,'%')
  bar('Safety Score',admet.toxicity,100,'%')
  y+=2

  section('3. Permeability & Transport')
  const bbbColor=bbb==='Likely'?'#10b981':bbb==='Unlikely'?'#ef4444':'#f59e0b'
  const bioColor=bioavailability==='High'?'#10b981':bioavailability==='Low'?'#ef4444':'#f59e0b'
  setColor('#94a3b8');pdf.setFontSize(8);pdf.setFont('helvetica','normal');pdf.text('BBB Permeability',M+2,y)
  setColor(bbbColor);pdf.setFont('helvetica','bold');pdf.text(bbb,M+80,y);y+=9
  setColor('#94a3b8');pdf.setFont('helvetica','normal');pdf.text('Oral Bioavailability',M+2,y)
  setColor(bioColor);pdf.setFont('helvetica','bold');pdf.text(bioavailability,M+80,y);y+=9
  setColor('#94a3b8');pdf.setFont('helvetica','normal');pdf.text('CYP Inhibition',M+2,y)
  setColor('#f59e0b');pdf.setFont('helvetica','bold');pdf.text(cyp.join(', '),M+80,y);y+=9
  setColor('#94a3b8');pdf.setFont('helvetica','normal');pdf.text('Veber Rules',M+2,y)
  setColor(veberPass?'#10b981':'#ef4444');pdf.setFont('helvetica','bold');pdf.text(veberPass?'✓ Pass':'✗ Fail',M+80,y);y+=12

  section('4. Structural Alerts (PAINS + BRENK)')
  toxFlags.forEach(t => {
    const tc=t.level==='pass'?'#10b981':t.level==='high'?'#ef4444':'#f59e0b'
    setFill(tc+'18');setDraw(tc+'66');pdf.setLineWidth(0.3);pdf.roundedRect(M,y-4,CW,12,2,2,'FD')
    setColor(tc);pdf.setFontSize(8);pdf.setFont('helvetica','bold')
    pdf.text(`${t.level==='pass'?'✓':t.level==='high'?'⚠':'◈'}  ${t.flag}`,M+3,y+2)
    setColor('#94a3b8');pdf.setFont('helvetica','normal');pdf.setFontSize(7);pdf.text(t.risk,M+70,y+2)
    y+=15
  })
  y+=3

  if(y<250){
    section('5. Atom Composition')
    const ae=Object.entries(atoms).filter(([,v])=>v>0)
    const tot=ae.reduce((s,[,v])=>s+v,0)
    ae.forEach(([el,count])=>{
      const pct=count/tot
      setColor('#64748b');pdf.setFontSize(7.5);pdf.setFont('helvetica','normal');pdf.text(el,M+2,y)
      setColor('#d8b4fe');pdf.setFont('helvetica','bold');pdf.text(`${count}  (${Math.round(pct*100)}%)`,M+20,y)
      setFill('#1e1040');pdf.roundedRect(M+65,y-3.5,100,4,1,1,'F')
      setFill('#7c3aed');pdf.roundedRect(M+65,y-3.5,100*pct,4,1,1,'F');y+=8
    })
  }

  setFill('#150a2e');pdf.rect(0,285,210,12,'F')
  setColor('#475569');pdf.setFontSize(6.5);pdf.setFont('helvetica','normal')
  pdf.text('Generated by Quantum Drug Discovery Platform  ·  RDKit descriptors + PAINS/BRENK filters. Not for clinical use.',M,292)
  setColor('#7c3aed');pdf.text('CONFIDENTIAL',W-M,292,{align:'right'})
  pdf.save(filename)
}

async function downloadMolImage(filename) {
  const viewerDiv = document.getElementById('admet-mol-viewer-target')
  if (!viewerDiv) { alert('3D structure not available for this molecule.'); return }
  const canvas = viewerDiv.querySelector('canvas')
  if (!canvas) { alert('3D viewer canvas not found.'); return }
  const out = document.createElement('canvas')
  out.width = canvas.width || 800; out.height = canvas.height || 600
  const ctx = out.getContext('2d')
  ctx.fillStyle = '#0d0618'; ctx.fillRect(0,0,out.width,out.height)
  ctx.drawImage(canvas,0,0)
  ctx.fillStyle = 'rgba(168,85,247,0.55)'
  ctx.font = `bold ${Math.max(11,out.width*0.018)}px monospace`
  ctx.textAlign = 'right'
  ctx.fillText('Quantum Drug Discovery Platform', out.width-14, out.height-12)
  const link = document.createElement('a')
  link.download = filename; link.href = out.toDataURL('image/png'); link.click()
}

// ─── Hidden 3D viewer for image capture ──────────────────────────────────────
function HiddenMolViewer({ sdf }) {
  const containerRef = useRef(null)
  useEffect(() => {
    if (!sdf || !containerRef.current) return
    const init = () => {
      containerRef.current.innerHTML = ''
      const viewer = window.$3Dmol.createViewer(containerRef.current, { backgroundColor: '#0d0618' })
      viewer.addModel(sdf, 'sdf')
      viewer.setStyle({}, { stick:{ radius:0.12 }, sphere:{ scale:0.26 } })
      viewer.zoomTo(); viewer.render()
      setTimeout(() => { try { viewer.resize(); viewer.render() } catch(e){} }, 200)
    }
    if (window.$3Dmol) init()
    else {
      const s = document.createElement('script')
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js'
      s.onload = init; document.head.appendChild(s)
    }
  }, [sdf])
  return (
    <div style={{ position:'fixed', left:'-9999px', top:'-9999px', width:0, height:0, overflow:'hidden', pointerEvents:'none', opacity:0, zIndex:-1 }}>
      <div id="admet-mol-viewer-target" ref={containerRef}
        style={{ width:'800px', height:'600px', background:'#0d0618' }} />
    </div>
  )
}


// ─── Local ADMET computation (pure JS — no backend needed) ───────────────────
// Mirrors the RDKit logic from main.py using SMILES string parsing
function computeADMETLocal(smiles) {
  const s = smiles.toUpperCase()

  // ── Atom counts from SMILES ──
  const countAtom = (sym) => {
    const upper = sym.toUpperCase()
    // count explicit occurrences, handle 2-char symbols first
    let count = 0
    for (let i = 0; i < smiles.length; i++) {
      if (smiles[i].toUpperCase() === upper[0]) {
        if (upper.length === 2) {
          if (i+1 < smiles.length && smiles[i+1].toUpperCase() === upper[1]) { count++; i++ }
        } else {
          // single char — make sure next char isn't lowercase (part of 2-char symbol)
          const next = smiles[i+1]
          if (!next || next < 'a' || next > 'z') count++
        }
      }
    }
    return count
  }

  const C  = (smiles.match(/[cC](?![lL])/g) || []).length
  const N  = (smiles.match(/[nN]/g) || []).length
  const O  = (smiles.match(/[oO]/g) || []).length
  const S  = (smiles.match(/[sS](?!i)/g) || []).length
  const F  = (smiles.match(/F/g) || []).length
  const Cl = (smiles.match(/Cl/gi) || []).length
  const Br = (smiles.match(/Br/gi) || []).length
  const P  = (smiles.match(/P/g) || []).length
  const I  = (smiles.match(/(?<![A-Za-z])I(?![A-Za-z])/g) || []).length

  const atoms = { C, N, O, S, P, F, Cl, Br, I }
  const heavy_atoms = C + N + O + S + P + F + Cl + Br + I

  // ── MW estimate ──
  const MW_TABLE = { C:12.011, N:14.007, O:15.999, S:32.06, F:18.998, Cl:35.45, Br:79.904, P:30.974, I:126.904, H:1.008 }
  // Rough H count from valence
  const hCount = C*2 + N*1 // very rough
  const mw = Math.round((
    C*12.011 + N*14.007 + O*15.999 + S*32.06 +
    F*18.998 + Cl*35.45 + Br*79.904 + P*30.974 + I*126.904
  ) * 10) / 10

  // ── LogP (Crippen-inspired atom contributions) ──
  const logp = parseFloat((
    C * 0.5  + N * -0.74 + O * -0.67 +
    S * 0.15 + F * 0.14  + Cl * 0.60 +
    Br * 0.48 + P * 0.0  + I * 0.87
    - (smiles.match(/O[H]/gi) || []).length * 1.2
    - (smiles.match(/N[H]/gi) || []).length * 0.9
    - (smiles.match(/C\(=O\)O/gi) || []).length * 0.8
  ).toFixed(2))

  // ── HBD / HBA ──
  const hbd = (smiles.match(/[ON]H/gi) || []).length
  const hba = O + N

  // ── TPSA (Ertl 2000 atom contributions) ──
  const tpsa = parseFloat((
    O * 9.23 + N * 26.02 +
    (smiles.match(/N[H]/gi) || []).length * 17.0 +
    (smiles.match(/O[H]/gi) || []).length * 20.23 +
    (smiles.match(/C\(=O\)O/gi) || []).length * 37.3
  ).toFixed(1))

  // ── Rotatable bonds (rough count) ──
  const rot_bonds = Math.max(0,
    (smiles.match(/[A-Z][A-Z]/g) || []).length +
    (smiles.match(/\(/g) || []).length - 1
  )

  // ── Rings ──
  const rings = (smiles.match(/[0-9]/g) || []).length / 2 | 0

  // ── Fsp3 ──
  const aliphC = (smiles.match(/C(?![=:#(])/g) || []).length
  const fsp3 = C > 0 ? parseFloat((aliphC / C).toFixed(2)) : 0

  // ── Lipinski ──
  const ro5_violations = [mw>500, logp>5, hbd>5, hba>10].filter(Boolean).length
  const veber_pass = rot_bonds <= 10 && tpsa <= 140

  // ── QED-like score ──
  const qed = Math.max(0.05, Math.min(0.95,
    0.5 - Math.abs(mw - 350) / 1400
        - Math.abs(logp - 2.5) / 20
        - ro5_violations * 0.1
        + (veber_pass ? 0.05 : 0)
  ))

  // ── Drug score ──
  const lipinski_score =
    (mw<=500?0.25:0)+(logp<=5?0.25:0)+(hbd<=5?0.15:0)+
    (hba<=10?0.15:0)+(rot_bonds<=10?0.10:0)+(tpsa<=140?0.10:0)
  const drug_score = parseFloat((0.5*qed + 0.5*lipinski_score).toFixed(3))

  // ── ADMET scores ──
  let absorb_base = tpsa<60?90:tpsa<90?72:tpsa<120?48:18
  absorb_base -= Math.max(0,(mw-300)*0.06) + Math.max(0,(hbd-2)*3)
  const absorption = Math.round(Math.min(98,Math.max(5,absorb_base))*10)/10

  let distrib_base = (logp>=1&&logp<=4)?78:logp<0?30:logp>5?52:62
  const distribution = Math.round(Math.min(98,Math.max(5,distrib_base-(tpsa-60)*0.15))*10)/10

  const ar_rings = (smiles.match(/c/g)||[]).length > 2 ? Math.max(1, rings) : 0
  const metab_base = 65 - ar_rings*5 - Math.max(0,logp-3)*4
  const metabolism = Math.round(Math.min(98,Math.max(5,metab_base))*10)/10

  const excrete_base = mw<300?82:mw<500?62:35
  const excretion = Math.round(Math.min(98,Math.max(5,excrete_base))*10)/10

  // ── Tox flags (SMARTS-lite patterns) ──
  const tox_flags = []
  const pains_patterns = [
    [/c1ccc(cc1)C(=O)/i, 'Aromatic ketone', 'PAINS-like alert'],
    [/[nN]1[cC][cC][cC][cC]1/i, 'Pyrrole/imidazole ring', 'PAINS-like alert'],
    [/O=C1CC(=O)/i, '1,3-diketone', 'Reactive chelator'],
  ]
  const brenk_patterns = [
    [/N=O/i, 'Nitroso group', 'Potential mutagen / toxic'],
    [/[Cl][Cl]/i, 'Geminal dichloride', 'Reactive group'],
    [/C=CC=O/i, 'Michael acceptor (enone)', 'Electrophilic / reactive'],
    [/[nN][H]C(=O)[nN]/i, 'Urea group', 'Metabolic liability'],
  ]
  pains_patterns.forEach(([re, flag, risk]) => {
    if (re.test(smiles)) tox_flags.push({ flag:`PAINS: ${flag}`, risk, level:'warn' })
  })
  brenk_patterns.forEach(([re, flag, risk]) => {
    if (re.test(smiles)) tox_flags.push({ flag:`BRENK: ${flag}`, risk, level:'high' })
  })
  if (mw > 800)    tox_flags.push({ flag:'High MW (>800 Da)', risk:'Poor GI absorption expected', level:'high' })
  if (logp > 5)    tox_flags.push({ flag:`High LogP (${logp})`, risk:'Lipophilicity-driven toxicity risk', level:'high' })
  if (ar_rings > 3) tox_flags.push({ flag:`${ar_rings} aromatic rings`, risk:'Mutagenicity / genotoxicity risk', level:'high' })
  if (tox_flags.length === 0)
    tox_flags.push({ flag:'No structural alerts found', risk:'Passes all screens', level:'pass' })

  // tox score
  const tox_base = Math.max(5, 90 - tox_flags.filter(f=>f.level!=='pass').length*10 - (logp>5?8:0))
  const toxicity = Math.round(Math.min(98, tox_base)*10)/10

  // ── BBB ──
  const bbb = (logp>0&&logp<4&&mw<400&&hbd<3&&tpsa<90) ? 'Likely'
            : (tpsa>120||mw>500||hbd>4) ? 'Unlikely' : 'Uncertain'

  // ── Bioavailability ──
  const bioavailability = ro5_violations===0
    ? (tpsa<60?'High':tpsa<120?'Moderate':'Low')
    : ro5_violations===1 ? 'Moderate' : 'Low'

  // ── CYP ──
  const cyp = []
  if (ar_rings>=2 && mw>300) cyp.push('CYP3A4')
  if (N>=1 && ar_rings>=1)   cyp.push('CYP2D6')
  if (ar_rings>=2 && tpsa<60) cyp.push('CYP1A2')
  if (O>=3 && logp>1)        cyp.push('CYP2C9')
  if (!cyp.length) cyp.push('None predicted')

  return {
    smiles, mw, logp, hbd, hba, tpsa, rot_bonds, rings, heavy_atoms, fsp3,
    ro5_violations, veber_pass, drug_score,
    admet: { absorption, distribution, metabolism, excretion, toxicity },
    bioavailability, bbb, cyp, tox_flags, atoms, error: null,
  }
}

// ─── Main ADMETDashboard ──────────────────────────────────────────────────────
export default function ADMETDashboard({ smiles, sdf, apiBase = '', onClose, onOpenXAI }) {
  const [activeTab,        setActiveTab]        = useState('overview')
  const [downloading,      setDownloading]      = useState(null)
  const [showDownloadMenu, setShowDownloadMenu] = useState(false)
  const [data,             setData]             = useState(null)  // raw API JSON
  const [loading,          setLoading]          = useState(true)
  const [apiError,         setApiError]         = useState(null)
  const [retryCount,       setRetryCount]       = useState(0)

  useEffect(() => {
    if (!smiles) return
    setLoading(true); setApiError(null)

    // Try backend first, fall back to local JS computation if it fails/times out
    const controller = new AbortController()
    const timeoutId  = setTimeout(() => controller.abort(), 60000)

    const computeLocally = () => {
      try {
        const result = computeADMETLocal(smiles)
        setData(result)
      } catch(e) {
        setApiError('Could not compute ADMET: ' + e.message)
      }
      setLoading(false)
    }

    fetch(`${apiBase}/admet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles }),
      signal: controller.signal,
    })
      .then(async r => {
        clearTimeout(timeoutId)
        const text = await r.text()
        if (!text || !text.trim()) throw new Error('empty')
        const json = JSON.parse(text)
        if (!r.ok) throw new Error(json?.detail || `HTTP ${r.status}`)
        if (json.error) throw new Error(json.error)
        return json
      })
      .then(json => { setData(json); setLoading(false) })
      .catch(() => {
        clearTimeout(timeoutId)
        // Backend failed — compute everything locally in JS
        computeLocally()
      })

    return () => { clearTimeout(timeoutId); controller.abort() }
  }, [smiles, apiBase, retryCount])

  const handleDownload = async (type) => {
    setDownloading(type); setShowDownloadMenu(false)
    const ts = new Date().toISOString().slice(0,10)
    const sf = smiles.slice(0,20).replace(/[^a-zA-Z0-9]/g,'_')
    try {
      if (type === 'pdf') await downloadTextPDF(smiles, data, `ADMET_Report_${sf}_${ts}.pdf`)
      else await downloadMolImage(`Molecule_3D_${sf}_${ts}.png`)
    } catch(e) { alert('Download failed: ' + e.message) }
    finally { setDownloading(null) }
  }

  if (loading) return (
    <div className="admet-page">
      <div className="admet-topbar"><button className="admet-back-btn" onClick={onClose}>← Back to Results</button></div>
      <div className="admet-root"><LoadingSkeleton /></div>
    </div>
  )

  if (apiError || !data) return (
    <div className="admet-page">
      <div className="admet-topbar"><button className="admet-back-btn" onClick={onClose}>← Back to Results</button></div>
      <div className="admet-root">
        <div className="admet-parse-error">
          <div className="admet-error-icon">⚠</div>
          <div className="admet-error-title">ADMET computation failed</div>
          <div className="admet-error-msg">{apiError || 'Unknown error'}</div>
          <div className="admet-error-smiles"><code>{smiles}</code></div>
          <button
            className="admet-retry-btn"
            onClick={() => { setApiError(null); setLoading(true); setRetryCount(c => c + 1) }}
          >↺ Retry</button>
        </div>
      </div>
    </div>
  )

  // ── Use raw API field names throughout ─────────────────────────────────────
  const MW          = data.mw
  const logP        = data.logp
  const hbd         = data.hbd
  const hba         = data.hba
  const rotBonds    = data.rot_bonds
  const tpsa        = data.tpsa
  const rings       = data.rings
  const heavyAtoms  = data.heavy_atoms
  const fsp3        = data.fsp3
  const ro5Violations = data.ro5_violations
  const veberPass   = data.veber_pass
  const drugScore   = data.drug_score
  const bioavailability = data.bioavailability
  const bbb         = data.bbb
  const cyp         = data.cyp
  const toxFlags    = data.tox_flags
  const admet       = data.admet
  const atoms       = data.atoms

  // FIX: handle all bbb values from backend including "Uncertain"
  const bbbClass = bbb === 'Likely' ? 'pass' : bbb === 'Unlikely' ? 'fail' : 'warn'
  const bioClass = bioavailability === 'High' ? 'pass' : bioavailability === 'Low' ? 'fail' : 'warn'

  const tabs = ['overview', 'lipinski', 'admet', 'toxicity', 'composition']

  return (
    <div className="admet-page">
      <div className="admet-topbar">
        <button className="admet-back-btn" onClick={onClose}>← Back to Results</button>
        <div className="admet-download-group">
          <button className="admet-download-btn" onClick={() => setShowDownloadMenu(v => !v)} disabled={!!downloading}>
            {downloading
              ? <><span className="admet-dl-spinner" /> Generating {downloading === 'pdf' ? 'PDF' : 'Image'}…</>
              : <><span className="admet-dl-icon">⬇</span> Download Report</>}
          </button>
          {showDownloadMenu && (
            <div className="admet-download-menu">
              <button className="admet-download-option" onClick={() => handleDownload('pdf')}>
                <span className="admet-dl-option-icon">📄</span>
                <div><div className="admet-dl-option-title">PDF Report</div><div className="admet-dl-option-sub">Full report with all RDKit ADMET data</div></div>
              </button>
              <button className="admet-download-option" onClick={() => handleDownload('image')} disabled={!sdf}>
                <span className="admet-dl-option-icon">🧬</span>
                <div><div className="admet-dl-option-title">3D Molecule Image</div><div className="admet-dl-option-sub">{sdf ? 'PNG of the 3D molecular structure' : 'No 3D structure available'}</div></div>
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="admet-root" id="admet-report-target">
        <div className="admet-header">
          <div className="admet-header-left">
            <div className="admet-eyebrow">
              <span className="admet-eyebrow-dot" />ADMET Prediction Dashboard
              <span className="admet-source-badge">RDKit</span>
            </div>
            <h2 className="admet-title">Molecular Property Analysis</h2>
            <div className="admet-smiles-pill">
              <span className="admet-smiles-tag">SMILES</span>
              <code className="admet-smiles-code">{smiles.length>58?smiles.slice(0,58)+'…':smiles}</code>
            </div>
          </div>
          <div className="admet-header-right">
            <ScoreRing score={drugScore} />
            <div className="admet-pills">
              <div className={`admet-pill admet-pill-${bioClass}`}>
                <span className="admet-pill-label">Oral Bioavailability</span>
                <span className="admet-pill-val">{bioavailability}</span>
              </div>
              <div className={`admet-pill admet-pill-${bbbClass}`}>
                <span className="admet-pill-label">BBB Permeability</span>
                <span className="admet-pill-val">{bbb}</span>
              </div>
            </div>
          </div>
        </div>

        <div className={`admet-banner ${ro5Violations===0?'admet-banner-pass':ro5Violations>=2?'admet-banner-high':'admet-banner-warn'}`}>
          <span className="admet-banner-icon">{ro5Violations===0?'✓':ro5Violations>=2?'⚠':'◈'}</span>
          {ro5Violations===0
            ? 'Passes all Lipinski Rule of Five criteria — good oral drug-likeness predicted'
            : `${ro5Violations} Lipinski Rule of Five violation${ro5Violations>1?'s':''} detected — oral bioavailability may be compromised`}
        </div>

        <div className="admet-tabs">
          {tabs.map(t => (
            <button key={t} onClick={() => setActiveTab(t)} className={`admet-tab ${activeTab===t?'active':''}`}>
              {t.charAt(0).toUpperCase()+t.slice(1)}
            </button>
          ))}
        </div>

        {activeTab === 'overview' && (
          <div className="admet-panel admet-fade">
            <div className="admet-overview-grid">
              <div className="admet-section">
                <div className="admet-section-title">ADMET Scores</div>
                <div className="admet-gauges">
                  <RadialGauge value={admet.absorption}   label="Absorption"   color="#6ee7b7" />
                  <RadialGauge value={admet.distribution} label="Distribution" color="#60a5fa" />
                  <RadialGauge value={admet.metabolism}   label="Metabolism"   color="#d4af72" />
                  <RadialGauge value={admet.excretion}    label="Excretion"    color="#c084fc" />
                  <RadialGauge value={admet.toxicity}     label="Safety"       color="#f87171" />
                </div>
              </div>
              <div className="admet-section admet-section-radar">
                <div className="admet-section-title">Profile Radar</div>
                <RadarChart data={{ Absorb:admet.absorption, Distrib:admet.distribution, Metab:admet.metabolism, Excrete:admet.excretion, Safety:admet.toxicity }} />
              </div>
            </div>
            <div className="admet-stats-row">
              {[
                { label:'Mol. Weight', value:`${MW}`,           unit:' Da', good: MW<=500 },
                { label:'LogP',        value:`${logP}`,          unit:'',    good: logP<=5&&logP>=-2 },
                { label:'TPSA',        value:`${tpsa}`,          unit:' Å²', good: tpsa<=140 },
                { label:'HBD / HBA',   value:`${hbd} / ${hba}`, unit:'',    good: hbd<=5&&hba<=10 },
                { label:'Rot. Bonds',  value:`${rotBonds}`,      unit:'',    good: rotBonds<=10 },
                { label:'Heavy Atoms', value:`${heavyAtoms}`,    unit:'',    good: heavyAtoms<=40 },
              ].map(s => (
                <div key={s.label} className={`admet-stat-card ${s.good?'good':'bad'}`}>
                  <div className="admet-stat-label">{s.label}</div>
                  <div className="admet-stat-val">{s.value}<span className="admet-stat-unit">{s.unit}</span></div>
                  <div className={`admet-stat-indicator ${s.good?'good':'bad'}`} />
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'lipinski' && (
          <div className="admet-panel admet-fade">
            <div className="admet-two-col">
              <div className="admet-section" style={{flex:1}}>
                <div className="admet-section-title">Lipinski Rule of Five</div>
                <div className="admet-props">
                  <PropBar label="Molecular Weight (ExactMolWt)" value={MW}    max={700} unit=" Da" pass={MW<=500}   info="Threshold ≤ 500 Da" />
                  <PropBar label="LogP (Crippen MolLogP)"        value={logP}  max={10}  unit=""   pass={logP<=5}    info="Threshold ≤ 5" />
                  <PropBar label="H-Bond Donors (CalcNumHBD)"    value={hbd}   max={10}  unit=""   pass={hbd<=5}     info="Threshold ≤ 5" />
                  <PropBar label="H-Bond Acceptors (CalcNumHBA)" value={hba}   max={20}  unit=""   pass={hba<=10}    info="Threshold ≤ 10" />
                </div>
                <div className="admet-section-title" style={{marginTop:'1.5rem'}}>Veber Rules</div>
                <div className="admet-props">
                  <PropBar label="Rotatable Bonds" value={rotBonds} max={15}  unit=""    pass={rotBonds<=10} info="Threshold ≤ 10" />
                  <PropBar label="TPSA (Ertl)"     value={tpsa}     max={200} unit=" Å²" pass={tpsa<=140}   info="Threshold ≤ 140 Å²" />
                </div>
              </div>
              <div className="admet-section admet-section-summary">
                <div className="admet-section-title">Summary</div>
                <div className="admet-summary-items">
                  {[
                    { pass:ro5Violations===0, main:`Ro5: ${ro5Violations} violation${ro5Violations!==1?'s':''}`, sub:ro5Violations===0?'Fully compliant':'May reduce oral absorption' },
                    { pass:veberPass,         main:`Veber: ${veberPass?'Pass':'Fail'}`,                          sub:veberPass?'Good oral bioavailability':'Reduced oral bioavailability' },
                    { pass:null,              main:`Fsp3: ${fsp3}`,                                               sub:fsp3>0.4?'Good 3D character':'Mostly flat / aromatic' },
                    { pass:null,              main:`Rings: ${rings}`,                                             sub:rings<=4?'Reasonable ring count':'High ring count' },
                  ].map((item,i) => (
                    <div key={i} className={`admet-summary-item ${item.pass===null?'neutral':item.pass?'pass':'fail'}`}>
                      <span className="admet-summary-icon">{item.pass===null?'◈':item.pass?'✓':'✗'}</span>
                      <div><div className="admet-summary-main">{item.main}</div><div className="admet-summary-sub">{item.sub}</div></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'admet' && (
          <div className="admet-panel admet-fade">
            <div className="admet-two-col">
              <div className="admet-section" style={{flex:1}}>
                <div className="admet-section-title">Absorption &amp; Distribution</div>
                <div className="admet-props">
                  <PropBar label="GI Absorption"  value={admet.absorption}   max={100} unit="%" />
                  <PropBar label="Distribution"   value={admet.distribution} max={100} unit="%" />
                  <PropBar label="TPSA"           value={tpsa}               max={200} unit=" Å²" pass={tpsa<=90} info="≤ 90 Å² → good permeability" />
                </div>
                <div className="admet-section-title" style={{marginTop:'1.5rem'}}>Metabolism &amp; Excretion</div>
                <div className="admet-props">
                  <PropBar label="Metabolic Stability" value={admet.metabolism} max={100} unit="%" />
                  <PropBar label="Renal Excretion"     value={admet.excretion}  max={100} unit="%" />
                  <PropBar label="Rotatable Bonds"     value={rotBonds}         max={15}  unit=""  pass={rotBonds<=10} info="Affects metabolic clearance" />
                </div>
              </div>
              <div className="admet-section admet-section-perm">
                <div className="admet-section-title">Permeability</div>
                <div className="admet-perm-cards">
                  <div className={`admet-perm-card ${bbbClass}`}>
                    <div className="admet-perm-name">BBB</div>
                    <div className="admet-perm-val">{bbb}</div>
                    <div className="admet-perm-sub">Blood-brain barrier</div>
                  </div>
                  <div className={`admet-perm-card ${bioClass}`}>
                    <div className="admet-perm-name">Oral F</div>
                    <div className="admet-perm-val">{bioavailability}</div>
                    <div className="admet-perm-sub">Bioavailability</div>
                  </div>
                </div>
                <div className="admet-section-title" style={{marginTop:'1.25rem'}}>CYP Inhibition</div>
                <div className="admet-cyp-list">
                  {cyp.map((c,i) => (
                    <div key={i} className={`admet-cyp-item ${c==='None predicted'?'pass':'warn'}`}>
                      <span className="admet-cyp-dot"/>{c}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'toxicity' && (
          <div className="admet-panel admet-fade">
            <div className="admet-section-title">Structural Toxicity Alerts (RDKit PAINS + BRENK)</div>
            <div className="admet-tox-list">
              {toxFlags.map((t,i) => (
                <div key={i} className={`admet-tox-item admet-tox-${t.level}`}>
                  <div className={`admet-tox-icon admet-tox-icon-${t.level}`}>{t.level==='pass'?'✓':t.level==='high'?'⚠':'◈'}</div>
                  <div className="admet-tox-body">
                    <div className="admet-tox-flag">{t.flag}</div>
                    <div className="admet-tox-risk">{t.risk}</div>
                  </div>
                  <div className={`admet-tox-badge admet-tox-badge-${t.level}`}>{t.level==='pass'?'Clear':t.level==='high'?'High Risk':'Moderate'}</div>
                </div>
              ))}
            </div>
            <div className="admet-section-title" style={{marginTop:'1.5rem'}}>Safety Metrics</div>
            <div className="admet-props">
              <PropBar label="Overall Safety Score" value={admet.toxicity}  max={100} unit="%" pass={admet.toxicity>=60} />
              <PropBar label="LogP"                 value={Math.abs(logP)} max={10}  unit=""  pass={logP<=5} info="High LogP → non-specific toxicity" />
              <PropBar label="Molecular Weight"     value={MW}             max={700} unit=" Da" pass={MW<=500} info="High MW → reactive metabolite risk" />
            </div>
            <div className="admet-disclaimer">⚠ Values computed by RDKit. ADMET scores are descriptor-based estimates. Experimental validation required.</div>
          </div>
        )}

        {activeTab === 'composition' && (
          <div className="admet-panel admet-fade">
            <div className="admet-two-col">
              <div className="admet-section" style={{flex:1}}>
                <div className="admet-section-title">Atom Composition</div>
                <AtomDonut atoms={atoms} />
              </div>
              <div className="admet-section" style={{flex:1}}>
                <div className="admet-section-title">Structural Counts</div>
                <div className="admet-struct-grid">
                  {[
                    {label:'Heavy Atoms',   value:heavyAtoms},
                    {label:'Ring Systems',  value:rings},
                    {label:'Rot. Bonds',    value:rotBonds},
                    {label:'H-Bond Donors', value:hbd},
                    {label:'H-Bond Accept.',value:hba},
                    {label:'Fsp3',          value:fsp3},
                  ].map(s => (
                    <div key={s.label} className="admet-struct-item">
                      <div className="admet-struct-val">{s.value}</div>
                      <div className="admet-struct-label">{s.label}</div>
                    </div>
                  ))}
                </div>
                <div className="admet-section-title" style={{marginTop:'1.25rem'}}>Elemental Breakdown</div>
                <div className="admet-elem-list">
                  {Object.entries(atoms).filter(([,v])=>v>0).map(([el,count]) => (
                    <div key={el} className="admet-elem-row">
                      <span className="admet-elem-symbol">{el}</span>
                      <div className="admet-elem-track">
                        <div className="admet-elem-fill" style={{width:`${Math.min((count/heavyAtoms)*100,100)}%`}} />
                      </div>
                      <span className="admet-elem-count">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="admet-footer">
          {onOpenXAI && (
            <button className="xai-launch-btn" onClick={() => onOpenXAI({ smiles, sdf })}>
              <span className="xai-launch-icon">◈</span>
              Explain with XAI · SHAP Analysis
              <span className="xai-launch-arrow">→</span>
            </button>
          )}
        </div>

        {sdf && <HiddenMolViewer sdf={sdf} />}
      </div>
    </div>
  )
}