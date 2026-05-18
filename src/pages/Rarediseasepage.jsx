import React, { useState } from 'react'
import './RareDiseasePage.css'


const RARE_DISEASES = [
  { name: 'Cystic Fibrosis', category: 'Genetic', prevalence: '1 in 3,500', affected: '~70,000 worldwide' },
  { name: 'Gaucher Disease', category: 'Lysosomal Storage', prevalence: '1 in 40,000', affected: '~10,000 in US' },
  { name: "Wilson's Disease", category: 'Metabolic', prevalence: '1 in 30,000', affected: '~10,000 in US' },
  { name: 'Fabry Disease', category: 'Lysosomal Storage', prevalence: '1 in 40,000–60,000', affected: '~50,000 worldwide' },
  { name: 'Huntington\'s Disease', category: 'Neurological', prevalence: '1 in 10,000', affected: '~30,000 in US' },
  { name: 'Phenylketonuria (PKU)', category: 'Metabolic', prevalence: '1 in 10,000–15,000', affected: '~16,000 in US' },
  { name: 'Marfan Syndrome', category: 'Connective Tissue', prevalence: '1 in 5,000', affected: '~200,000 in US' },
  { name: 'Ehlers-Danlos Syndrome', category: 'Connective Tissue', prevalence: '1 in 5,000', affected: '~200,000 in US' },
  { name: 'Amyotrophic Lateral Sclerosis (ALS)', category: 'Neurological', prevalence: '2–3 in 100,000', affected: '~30,000 in US' },
  { name: 'Spinal Muscular Atrophy (SMA)', category: 'Neurological', prevalence: '1 in 10,000', affected: '~10,000 in US' },
  { name: 'Duchenne Muscular Dystrophy', category: 'Neuromuscular', prevalence: '1 in 3,500 males', affected: '~15,000 in US' },
  { name: 'Pompe Disease', category: 'Lysosomal Storage', prevalence: '1 in 40,000', affected: '~10,000 in US' },
  { name: 'Niemann-Pick Disease', category: 'Lysosomal Storage', prevalence: '1 in 150,000', affected: '~1,800 in US' },
  { name: 'Tay-Sachs Disease', category: 'Lysosomal Storage', prevalence: '1 in 320,000', affected: 'Very rare' },
  { name: 'Maple Syrup Urine Disease', category: 'Metabolic', prevalence: '1 in 185,000', affected: '~2,000 in US' },
  { name: 'Alpha-1 Antitrypsin Deficiency', category: 'Genetic', prevalence: '1 in 2,500', affected: '~100,000 in US' },
  { name: 'Sickle Cell Disease', category: 'Haematological', prevalence: '1 in 365 (African-American)', affected: '~100,000 in US' },
  { name: 'Thalassemia', category: 'Haematological', prevalence: '1 in 10,000', affected: '~1,000 in US' },
  { name: 'Von Hippel-Lindau Disease', category: 'Genetic', prevalence: '1 in 36,000', affected: '~10,000 worldwide' },
  { name: 'Neurofibromatosis Type 1', category: 'Genetic', prevalence: '1 in 3,500', affected: '~100,000 in US' },
  { name: 'Tuberous Sclerosis Complex', category: 'Genetic', prevalence: '1 in 6,000', affected: '~50,000 in US' },
  { name: 'Rett Syndrome', category: 'Neurological', prevalence: '1 in 10,000–15,000 females', affected: '~9,000 in US' },
  { name: 'Angelman Syndrome', category: 'Genetic', prevalence: '1 in 12,000–20,000', affected: '~500,000 worldwide' },
  { name: 'Prader-Willi Syndrome', category: 'Genetic', prevalence: '1 in 15,000', affected: '~400,000 worldwide' },
  { name: 'Williams Syndrome', category: 'Genetic', prevalence: '1 in 7,500–10,000', affected: '~30,000 in US' },
  { name: 'Fragile X Syndrome', category: 'Genetic', prevalence: '1 in 4,000 males', affected: '~200,000 in US' },
  { name: 'CHARGE Syndrome', category: 'Genetic', prevalence: '1 in 8,500–10,000', affected: '~10,000 in US' },
  { name: 'Kabuki Syndrome', category: 'Genetic', prevalence: '1 in 32,000', affected: 'Very rare' },
  { name: 'FOXG1 Syndrome', category: 'Neurological', prevalence: '1 in 40,000+', affected: 'Very rare' },
  { name: 'Stiff Person Syndrome', category: 'Autoimmune', prevalence: '1 in 1,000,000', affected: '~5,000 in US' },
  { name: 'Systemic Mastocytosis', category: 'Haematological', prevalence: '1 in 10,000', affected: 'Rare' },
  { name: 'Erdheim-Chester Disease', category: 'Histiocytic', prevalence: 'Very rare', affected: '~1,500 worldwide' },
  { name: 'Castleman Disease', category: 'Lymphatic', prevalence: '1 in 100,000', affected: '~6,500/year in US' },
  { name: 'Pheochromocytoma', category: 'Endocrine', prevalence: '2–8 in 1,000,000', affected: 'Rare' },
  { name: 'Acromegaly', category: 'Endocrine', prevalence: '3–4 in 1,000,000', affected: '~25,000 in US' },
  { name: 'Primary Hyperaldosteronism', category: 'Endocrine', prevalence: '5–10% of hypertensives', affected: 'Underdiagnosed' },
  { name: 'Bartter Syndrome', category: 'Renal', prevalence: '1 in 1,000,000', affected: 'Very rare' },
  { name: 'Alport Syndrome', category: 'Renal', prevalence: '1 in 50,000', affected: '~30,000 in US' },
  { name: 'Autosomal Dominant PKD', category: 'Renal', prevalence: '1 in 500–1,000', affected: '~600,000 in US' },
  { name: 'Hereditary Haemochromatosis', category: 'Metabolic', prevalence: '1 in 200–500', affected: 'Common in northern Europe' },
  { name: 'Pulmonary Arterial Hypertension', category: 'Cardiovascular', prevalence: '15–50 in 1,000,000', affected: '~200,000 in US' },
  { name: 'Hereditary Haemorrhagic Telangiectasia', category: 'Vascular', prevalence: '1 in 5,000–8,000', affected: '~100,000 in US' },
  { name: 'Cavernous Angioma', category: 'Vascular', prevalence: '1 in 200', affected: '~1.5M in US' },
  { name: 'Mastocytosis', category: 'Haematological', prevalence: '1 in 10,000', affected: 'Rare' },
  { name: 'Paroxysmal Nocturnal Haemoglobinuria', category: 'Haematological', prevalence: '1–5 in 1,000,000', affected: '~4,000 in US' },
  { name: 'Cold Agglutinin Disease', category: 'Haematological', prevalence: '1 in 1,000,000', affected: 'Very rare' },
  { name: 'Atypical HUS', category: 'Renal', prevalence: '1–9 in 1,000,000', affected: 'Very rare' },
  { name: 'Epidermolysis Bullosa', category: 'Dermatological', prevalence: '1 in 50,000', affected: '~30,000 in US' },
  { name: 'Ichthyosis', category: 'Dermatological', prevalence: '1 in 300,000 (severe)', affected: '~16,000 in US' },
  { name: 'X-Linked Adrenoleukodystrophy', category: 'Metabolic', prevalence: '1 in 17,000', affected: '~15,000 in US' },
]

const CATEGORIES = ['All', ...Array.from(new Set(RARE_DISEASES.map(d => d.category))).sort()]

const CATEGORY_COLORS = {
  'Genetic': '#a855f7',
  'Lysosomal Storage': '#e040fb',
  'Metabolic': '#d4af72',
  'Neurological': '#60a5fa',
  'Neuromuscular': '#34d399',
  'Connective Tissue': '#f472b6',
  'Haematological': '#f87171',
  'Autoimmune': '#fbbf24',
  'Endocrine': '#a78bfa',
  'Renal': '#38bdf8',
  'Cardiovascular': '#fb923c',
  'Vascular': '#e879f9',
  'Dermatological': '#86efac',
  'Lymphatic': '#c084fc',
  'Histiocytic': '#fcd34d',
}

export default function RareDiseasePage({ onBack }) {
  const [search, setSearch] = useState('')
  const [activeCategory, setActiveCategory] = useState('All')
  const [expanded, setExpanded] = useState(null)

  const filtered = RARE_DISEASES.filter(d => {
    const matchSearch = d.name.toLowerCase().includes(search.toLowerCase())
    const matchCat = activeCategory === 'All' || d.category === activeCategory
    return matchSearch && matchCat
  })

  return (
    <div className="rd-root">
      <button className="rd-back" onClick={onBack}>← Back to Home</button>

      <div className="rd-container">
        {/* Header */}
        <div className="rd-header">
          <div className="rd-label">Database · 50 Conditions</div>
          <h1 className="rd-title">Rare Disease <span>Reference</span></h1>
          <p className="rd-subtitle">
            A curated index of rare diseases targeted by QureAI's quantum drug discovery pipeline. Each condition represents an unmet medical need where our LLM–QML approach can make a meaningful impact.
          </p>
        </div>

        {/* Search */}
        <div className="rd-search-wrap">
          <span className="rd-search-icon">🔍</span>
          <input
            className="rd-search"
            type="text"
            placeholder="Search diseases…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          {search && (
            <button className="rd-search-clear" onClick={() => setSearch('')}>✕</button>
          )}
        </div>

        {/* Category filters */}
        <div className="rd-filters">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              className={`rd-filter-btn ${activeCategory === cat ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat)}
              style={activeCategory === cat && cat !== 'All'
                ? { borderColor: CATEGORY_COLORS[cat], color: CATEGORY_COLORS[cat], background: `${CATEGORY_COLORS[cat]}18` }
                : {}
              }
            >{cat}</button>
          ))}
        </div>

        {/* Count */}
        <div className="rd-count">
          Showing <span>{filtered.length}</span> of {RARE_DISEASES.length} conditions
        </div>

        {/* Disease grid */}
        <div className="rd-grid">
          {filtered.map((disease, i) => {
            const color = CATEGORY_COLORS[disease.category] || '#a855f7'
            const isOpen = expanded === i
            return (
              <div
                key={i}
                className={`rd-card ${isOpen ? 'open' : ''}`}
                onClick={() => setExpanded(isOpen ? null : i)}
                style={{ '--accent': color }}
              >
                <div className="rd-card-top">
                  <div className="rd-card-left">
                    <span className="rd-card-num">{String(i + 1).padStart(2, '0')}</span>
                    <div>
                      <div className="rd-card-name">{disease.name}</div>
                      <div className="rd-card-cat" style={{ color }}>{disease.category}</div>
                    </div>
                  </div>
                  <span className="rd-card-chevron">{isOpen ? '▲' : '▼'}</span>
                </div>

                {isOpen && (
                  <div className="rd-card-details">
                    <div className="rd-detail-row">
                      <span>Prevalence</span>
                      <span>{disease.prevalence}</span>
                    </div>
                    <div className="rd-detail-row">
                      <span>Affected Population</span>
                      <span>{disease.affected}</span>
                    </div>
                    <div className="rd-detail-row">
                      <span>Classification</span>
                      <span style={{ color }}>{disease.category}</span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {filtered.length === 0 && (
          <div className="rd-empty">No diseases found matching "<em>{search}</em>"</div>
        )}
      </div>
    </div>
  )
}