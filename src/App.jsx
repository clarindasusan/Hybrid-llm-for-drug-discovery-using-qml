import React, { useEffect, useState, useRef, useCallback } from 'react'
import './App.css'
import logoImg from './assets/logo.png'
import architectureImg from './assets/architecture.png'
import LoginPage from './pages/LoginPage'
import GeneratePage from './pages/Generatepage'
import RareDiseasePage from './pages/Rarediseasepage'
import { onAuthChange, fetchUserProfile, logoutUser } from './firebase/Auth'

// ── THEME CONTEXT ─────────────────────────────────────────────────────────────
export const ThemeContext = React.createContext({ theme: 'dark', toggleTheme: () => {} })

// ── MOLECULAR SPHERE CANVAS ────────────────────────────────────────────────────
function SphereCanvas({ theme }) {
  const ref = useRef(null)
  const animRef = useRef(null)
  const mouseRef = useRef({ x: -9999, y: -9999, hovering: false })

  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas.getContext('2d')
    let W = canvas.width = canvas.offsetWidth
    let H = canvas.height = canvas.offsetHeight

    const onResize = () => {
      W = canvas.width = canvas.offsetWidth
      H = canvas.height = canvas.offsetHeight
    }
    window.addEventListener('resize', onResize)

    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect()
      mouseRef.current.x = e.clientX - rect.left
      mouseRef.current.y = e.clientY - rect.top
    }
    const onMouseLeave = () => {
      mouseRef.current.x = -9999
      mouseRef.current.y = -9999
      mouseRef.current.hovering = false
    }
    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('mouseleave', onMouseLeave)

    const N = 280
    const nodes = []
    const golden = Math.PI * (3 - Math.sqrt(5))

    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2
      const r = Math.sqrt(1 - y * y)
      const theta = golden * i
      nodes.push({
        ox: Math.cos(theta) * r,
        oy: y,
        oz: Math.sin(theta) * r,
        size: Math.random() * 2 + 0.8,
        hue: Math.random() > 0.82
          ? [212, 175, 114]
          : Math.random() > 0.5
          ? [168, 85, 247]
          : [224, 64, 251],
        brightness: Math.random() * 0.5 + 0.5,
        phase: Math.random() * Math.PI * 2,
      })
    }

    const bonds = []
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = nodes[i].ox - nodes[j].ox
        const dy = nodes[i].oy - nodes[j].oy
        const dz = nodes[i].oz - nodes[j].oz
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
        if (d < 0.28) bonds.push([i, j, d])
      }
    }

    let rotY = 0, rotX = 0.2, time = 0
    let glowIntensity = 0

    const draw = () => {
      const isLight = canvas.dataset.theme === 'light'
      const CX = W * 0.68
      const CY = H * 0.5
      const R = Math.min(W, H) * 0.42

      const mx = mouseRef.current.x
      const my = mouseRef.current.y
      const dist = Math.sqrt((mx - CX) ** 2 + (my - CY) ** 2)
      const isHov = dist < R * 1.05
      mouseRef.current.hovering = isHov

      glowIntensity += isHov
        ? (1 - glowIntensity) * 0.07
        : (0 - glowIntensity) * 0.05
      const g = glowIntensity

      rotY += 0.003 + g * 0.005
      rotX = 0.18 + Math.sin(time * 0.0004) * 0.08
      time++

      if (isLight) {
        ctx.fillStyle = `rgba(245,240,255,${0.55 - g * 0.25})`
      } else {
        ctx.fillStyle = `rgba(13,6,24,${0.55 - g * 0.25})`
      }
      ctx.fillRect(0, 0, W, H)

      const grd = ctx.createRadialGradient(CX, CY, 0, CX, CY, R * 1.2)
      if (isLight) {
        grd.addColorStop(0, `rgba(139,92,246,${0.06 + g * 0.15})`)
        grd.addColorStop(0.5, `rgba(168,85,247,${0.02 + g * 0.08})`)
      } else {
        grd.addColorStop(0, `rgba(107,33,168,${0.08 + g * 0.2})`)
        grd.addColorStop(0.5, `rgba(168,85,247,${0.03 + g * 0.1})`)
      }
      grd.addColorStop(1, 'transparent')
      ctx.fillStyle = grd
      ctx.fillRect(0, 0, W, H)

      if (g > 0.05) {
        const outerGlow = ctx.createRadialGradient(CX, CY, R * 0.85, CX, CY, R * 1.25)
        outerGlow.addColorStop(0, 'transparent')
        outerGlow.addColorStop(0.4, `rgba(168,85,247,${g * 0.18})`)
        outerGlow.addColorStop(0.7, `rgba(224,64,251,${g * 0.1})`)
        outerGlow.addColorStop(1, 'transparent')
        ctx.fillStyle = outerGlow
        ctx.fillRect(0, 0, W, H)
      }

      const project = (x, y, z) => {
        const cosY = Math.cos(rotY), sinY = Math.sin(rotY)
        const x1 = x * cosY - z * sinY
        const z1 = x * sinY + z * cosY
        const cosX = Math.cos(rotX), sinX = Math.sin(rotX)
        const y2 = y * cosX - z1 * sinX
        const z2 = y * sinX + z1 * cosX
        const fov = 2.2
        const scale = fov / (fov + z2 + 0.1)
        return { px: CX + x1 * R * scale, py: CY + y2 * R * scale, scale, z: z2 }
      }

      const proj = nodes.map(n => ({ ...n, ...project(n.ox, n.oy, n.oz) }))

      bonds.forEach(([i, j, dist]) => {
        const a = proj[i], b = proj[j]
        if (a.z < -0.6 && b.z < -0.6) return
        const midZ = (a.z + b.z) / 2
        const visibility = Math.max(0, (midZ + 1) / 2)
        const alpha = visibility * (1 - dist / 0.28) * (isLight ? 0.15 + g * 0.15 : 0.2 + g * 0.2)
        if (alpha < 0.01) return
        ctx.beginPath()
        ctx.moveTo(a.px, a.py)
        ctx.lineTo(b.px, b.py)
        ctx.strokeStyle = `rgba(${a.hue[0]},${a.hue[1]},${a.hue[2]},${alpha})`
        ctx.lineWidth = 0.5 + g * 0.3
        ctx.stroke()
      })

      proj.sort((a, b) => a.z - b.z)

      proj.forEach(n => {
        if (n.z < -0.8) return
        const visibility = Math.max(0, (n.z + 1) / 2)
        const pulse = g > 0.1 ? 0.5 + 0.5 * Math.sin(time * 0.05 + n.phase) : 1
        const alpha = visibility * n.brightness * pulse * (isLight ? 0.6 + g * 0.2 : 0.75 + g * 0.2)
        const sz = n.size * n.scale * (1 + g * 0.4)

        const glowR = ctx.createRadialGradient(n.px, n.py, 0, n.px, n.py, sz * (4 + g * 3))
        glowR.addColorStop(0, `rgba(${n.hue[0]},${n.hue[1]},${n.hue[2]},${alpha * (0.4 + g * 0.3)})`)
        glowR.addColorStop(1, 'transparent')
        ctx.fillStyle = glowR
        ctx.beginPath()
        ctx.arc(n.px, n.py, sz * (4 + g * 3), 0, Math.PI * 2)
        ctx.fill()

        ctx.beginPath()
        ctx.arc(n.px, n.py, sz, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${n.hue[0]},${n.hue[1]},${n.hue[2]},${alpha})`
        ctx.fill()
      })

      animRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(animRef.current)
      window.removeEventListener('resize', onResize)
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('mouseleave', onMouseLeave)
    }
  }, [])

  // Update canvas dataset when theme changes so draw() picks it up
  useEffect(() => {
    if (ref.current) ref.current.dataset.theme = theme
  }, [theme])

  return (
    <canvas
      ref={ref}
      data-theme={theme}
      style={{
        position: 'absolute',
        top: '64px', left: 0, right: 0, bottom: 0,
        width: '100%',
        height: 'calc(100% - 64px)',
        cursor: 'crosshair',
      }}
    />
  )
}

// ── SUPPORTING COMPONENTS ──────────────────────────────────────────────────────
function MoleculeTag({ formula, delay }) {
  return <span className="mol-tag" style={{ animationDelay: `${delay}s` }}>{formula}</span>
}
function StatCard({ number, label, icon }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-number">{number}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}
function FeatureCard({ title, desc, delay }) {
  return (
    <div className="feature-card" style={{ animationDelay: `${delay}s` }}>
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  )
}
function PipelineStep({ number, title, desc, active }) {
  return (
    <div className={`pipeline-step ${active ? 'active' : ''}`}>
      <div className="step-number">{number}</div>
      <div className="step-content">
        <h4>{title}</h4>
        <p>{desc}</p>
      </div>
    </div>
  )
}

// ── THEME TOGGLE BUTTON ────────────────────────────────────────────────────────
function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className="theme-toggle"
      title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      aria-label="Toggle theme"
    >
      <span className="theme-toggle-track">
        <span className="theme-toggle-thumb" />
      </span>
      <span className="theme-toggle-icon">
        {theme === 'dark' ? '🌙' : '☀️'}
      </span>
    </button>
  )
}

// ── PROFILE DROPDOWN ──────────────────────────────────────────────────────────
function ProfileDropdown({ profile, onLogout }) {
  const [open, setOpen] = useState(false)
  const ref = useRef()

  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const avatar = profile?.name?.[0]?.toUpperCase() || '?'

  return (
    <div className="profile-wrapper" ref={ref}>
      <button className="profile-btn" onClick={() => setOpen(o => !o)}>
        <div className="avatar">{avatar}</div>
        <div className="profile-info">
          <span className="profile-name">{profile?.name || 'Researcher'}</span>
          <span className="profile-id">{profile?.scientistId || '...'}</span>
        </div>
        <span className="chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="profile-dropdown">
          <div className="dropdown-header">
            <div className="avatar avatar-lg">{avatar}</div>
            <div>
              <div className="dropdown-name">{profile?.name}</div>
              <div className="dropdown-email">{profile?.email}</div>
              <div className="dropdown-sci-id">{profile?.scientistId}</div>
            </div>
          </div>
          <div className="dropdown-divider" />
          <div className="dropdown-info">
            <div className="info-row"><span>🎓 Designation</span><span>{profile?.designation}</span></div>
            <div className="info-row"><span>🏛️ Institution</span><span>{profile?.institution}</span></div>
            {profile?.department && <div className="info-row"><span>🔬 Department</span><span>{profile?.department}</span></div>}
            <div className="info-row"><span>🌍 Country</span><span>{profile?.country}</span></div>
            <div className="info-row"><span>📅 Joined</span><span>{profile?.createdAt ? new Date(profile.createdAt).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : '—'}</span></div>
            <div className="info-row"><span>⚛️ Access</span><span>{profile?.accessLevel}</span></div>
          </div>
          <div className="dropdown-divider" />
          <button className="dropdown-logout" onClick={onLogout}>Sign Out</button>
        </div>
      )}
    </div>
  )
}

const molecules = ['C₁₂H₂₂O₁₁', 'C₉H₈O₄', 'C₁₇H₁₉NO₃', 'C₈H₉NO₂', 'C₁₆H₁₃ClN₂O']

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function App() {
  const [activeStep, setActiveStep] = useState(0)
  const [authState, setAuthState] = useState('loading')
  const [profile, setProfile] = useState(null)
  const [currentPage, setCurrentPage] = useState('home')

  // ── THEME STATE ────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(() => localStorage.getItem('qureai-theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('qureai-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  useEffect(() => {
    const unsub = onAuthChange(async (user) => {
      if (user) {
        const data = await fetchUserProfile(user.uid)
        setProfile(data)
        setAuthState('loggedIn')
      } else {
        setProfile(null)
        setAuthState('loggedOut')
      }
    })
    return () => unsub()
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setActiveStep(s => (s + 1) % 5), 1800)
    return () => clearInterval(interval)
  }, [])

  const themeCtx = { theme, toggleTheme }

  // Loading screen
  if (authState === 'loading') {
    return (
      <ThemeContext.Provider value={themeCtx}>
        <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1.5rem' }}>
          <img src={logoImg} alt="QureAI" style={{ width: 64, filter: 'drop-shadow(0 0 16px rgba(220,100,200,0.9))' }} />
          <div style={{ width: 40, height: 40, border: '3px solid rgba(168,85,247,0.3)', borderTopColor: '#a855f7', borderRadius: '50%', animation: 'spinLoad 0.8s linear infinite' }} />
          <style>{`@keyframes spinLoad { to { transform: rotate(360deg); } }`}</style>
        </div>
      </ThemeContext.Provider>
    )
  }

  if (authState === 'loggedOut') {
    return (
      <ThemeContext.Provider value={themeCtx}>
        <LoginPage onAuthSuccess={() => {}} theme={theme} />
      </ThemeContext.Provider>
    )
  }

  if (currentPage === 'generate') {
    return (
      <ThemeContext.Provider value={themeCtx}>
        <GeneratePage onBack={() => setCurrentPage('home')} theme={theme} />
      </ThemeContext.Provider>
    )
  }

  if (currentPage === 'raredisease') {
    return (
      <ThemeContext.Provider value={themeCtx}>
        <RareDiseasePage onBack={() => setCurrentPage('home')} theme={theme} />
      </ThemeContext.Provider>
    )
  }

  return (
    <ThemeContext.Provider value={themeCtx}>
      <div className={`app theme-${theme}`}>
        {/* NAV */}
        <nav className="nav">
          <div className="nav-logo">
            <img src={logoImg} alt="QureAI Logo" className="logo-img" />
            <span>Qure<b>AI</b></span>
          </div>
          <div className="nav-links">
            <a href="#about">About</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#features">Technology</a>
            <a href="#raredisease" onClick={e => { e.preventDefault(); setCurrentPage('raredisease') }} style={{ cursor: 'pointer' }}>Rare Disease</a>
            <a href="#results">Contact Us</a>
          </div>
          <div className="nav-right">
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
            <ProfileDropdown profile={profile} onLogout={async () => { await logoutUser() }} />
          </div>
        </nav>

        {/* ── HERO with Molecular Sphere ── */}
        <section className="hero" style={{ position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', minHeight: '100vh', padding: '0', paddingTop: '64px' }}>
          <SphereCanvas theme={theme} />

          {/* Vignette overlays */}
          <div style={{ position: 'absolute', inset: 0, zIndex: 2, pointerEvents: 'none', background: 'radial-gradient(ellipse 80% 80% at 68% 50%, transparent 35%, var(--hero-vignette) 100%)' }} />
          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '30%', zIndex: 2, pointerEvents: 'none', background: 'linear-gradient(to top, var(--hero-vignette-bottom), transparent)' }} />
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '20%', zIndex: 2, pointerEvents: 'none', background: 'linear-gradient(to bottom, var(--hero-vignette-top), transparent)' }} />

          {/* Hero content */}
          <div className="hero-content" style={{ position: 'relative', zIndex: 10, padding: '0 5%', maxWidth: '52%' }}>
            <h1 className="hero-title">
              Drug Discovery<br />
              <span className="gradient-text">with LLM</span><br />
              and QML
            </h1>
            <p className="hero-sub">
              Bridging computational linguistics and quantum learning to engineer smarter, faster molecular discovery for the diseases that need it most. By unifying knowledge extraction with advanced optimization strategies, we create a scalable pathway from insight to viable drug candidates.
            </p>
            <div className="hero-actions">
              <button className="btn-primary" onClick={() => setCurrentPage('generate')}>Explore Research ↓</button>
            </div>
          </div>
        </section>

        {/* ABOUT */}
        <section id="about" className="about">
          <div className="section-label">The Problem</div>
          <h2 className="section-title">Traditional Drug Discovery<br /><span className="gradient-text">is Time-Consuming</span></h2>
          <div className="about-grid">
            <div className="about-text">
              <p>The traditional drug discovery pipeline is plagued by years of trial, billions in cost,
                and catastrophic failure rates — especially for rare diseases where patient populations
                are small and clinical data is nearly impossible to gather.</p>
              <p>Our research proposes a paradigm shift: combining the linguistic reasoning power of
                large language models with the computational supremacy of quantum machine learning
                to generate, rank, and validate drug candidates at unprecedented speed.</p>
              <div className="about-quote">
                "Combining LLM reasoning with QML stability for minimal-data drug discovery."
              </div>
            </div>
            <div className="about-diagram">
              <div className="arch-img-wrap">
                <img src={architectureImg} alt="QureAI Architecture Diagram" className="arch-img" />
              </div>
            </div>
          </div>
        </section>

        {/* PIPELINE */}
        <section id="pipeline" className="pipeline">
          <div className="section-label">How It Works</div>
          <h2 className="section-title">The <span className="gradient-text">Discovery Pipeline</span></h2>
          <div className="pipeline-container">
            <div className="pipeline-steps">
              <PipelineStep number="01" title="Disease Input" desc="Rare disease profile and known biomarkers are fed into the system as structured input." active={activeStep === 0} />
              <PipelineStep number="02" title="LLM Extraction" desc="BioGPT extracts relevant molecular features and generates candidate molecular structures." active={activeStep === 1} />
              <PipelineStep number="03" title="Quantum Encoding" desc="Molecular representations are encoded into quantum states for QML processing." active={activeStep === 2} />
              <PipelineStep number="04" title="Druggability Scoring" desc="QML evaluates binding affinity and chemical viability for each candidate molecule." active={activeStep === 3} />
              <PipelineStep number="05" title="Ranked Output" desc="A final ranked list of optimized molecular candidates is generated for researchers." active={activeStep === 4} />
            </div>
            <div className="pipeline-visual">
              <div className="pipeline-orb">
                <div className="pipeline-orb-text">Step {activeStep + 1} / 5</div>
              </div>
            </div>
          </div>
        </section>

        {/* FEATURES */}
        <section id="features" className="features">
          <div className="section-label">Core Technology</div>
          <h2 className="section-title">Built using <span className="gradient-text">the following Technologies</span></h2>
          <div className="features-grid">
            <FeatureCard title="Knowledge-Guided Molecular Generation" desc="Biomedical literature is encoded using a fine-tuned LLM to generate target-aware molecular structures grounded in biological context and mechanistic relevance." delay={0} />
            <FeatureCard title="Quantum Feature Encoding" desc="Molecular descriptors are mapped into quantum states using parameterized feature embeddings, enabling richer representation in high-dimensional chemical space." delay={0.1} />
            <FeatureCard title="Variational Quantum Evaluation" desc="Hybrid quantum–classical circuits assess binding affinity and druggability through optimized parameterized layers trained on sparse bioactivity data." delay={0.2} />
            <FeatureCard title="Multi-Objective Classification" desc="Molecules are classified across affinity tiers, drug-likeness thresholds, and ADMET safety profiles using hybrid LLM–QML predictive models." delay={0.3} />
            <FeatureCard title="Predictive Property Modeling" desc="Integrated regression modules estimate IC50, toxicity risk, and pharmacokinetic profiles to prioritize clinically viable compounds." delay={0.4} />
            <FeatureCard title="Iterative Feedback Loop" desc="High-scoring molecules are reintroduced into the generation stage, enabling closed-loop refinement and progressive enhancement of therapeutic potential." delay={0.5} />
          </div>
        </section>

        {/* CONTACT US */}
        <section id="results" className="contact">
          <div className="section-label">Our Team</div>
          <h2 className="section-title">Meet the <span className="gradient-text">Researchers</span></h2>
          <p className="contact-sub">The minds behind QureAI's quantum drug discovery pipeline. Reach out to collaborate or learn more.</p>
          <div className="team-grid">
            {[
              { name: 'yyy', designation: 'xxx', img: null },
              { name: 'yyy', designation: 'xxx', img: null },
              { name: 'yyy', designation: 'xxx', img: null },
            ].map((member, i) => (
              <div className="team-card" key={i}>
                <div className="team-avatar-wrap">
                  <div className="team-avatar-ring" />
                  <div className="team-avatar">
                    {member.img
                      ? <img src={member.img} alt={member.name} />
                      : <span>{member.name[0]}</span>
                    }
                  </div>
                </div>
                <h3 className="team-name">{member.name}</h3>
                <div className="team-designation">{member.designation}</div>
                <p className="team-desc">
                  Advancing the frontiers of quantum-assisted drug discovery through computational research and precision medicine innovation.
                </p>
                <div className="team-links">
                  <a href="mailto:contact@qureai.org" className="team-link">✉ Email</a>
                  <a href="#" className="team-link">in LinkedIn</a>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* FOOTER */}
        <footer className="footer">
          <div className="footer-top">
            <div className="nav-logo">
              <img src={logoImg} alt="QureAI Logo" className="logo-img" />
              <span>Qure<b>AI</b></span>
            </div>
            <p>Advancing rare disease treatment through Quantum-AI drug discovery.</p>
          </div>
          <div className="footer-bottom">
            <span>© 2025 QureAI Research</span>
            <span>LLM + QML for Precision Medicine</span>
          </div>
        </footer>
      </div>
    </ThemeContext.Provider>
  )
}