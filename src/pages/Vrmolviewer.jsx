import React, { useState, useEffect, useRef } from 'react'
import './VRMolViewer.css'


const VR_API = 'https://clarindasusan-drug-predictor-api.hf.space'

// ─── SDF Parser ───────────────────────────────────────────────────────────────
function parseSDF(sdf) {
  if (!sdf) return { atoms: [], bonds: [] }
  const lines = sdf.split('\n')
  const numAtoms = parseInt((lines[3]||'').slice(0,3).trim(),10) || 0
  const numBonds = parseInt((lines[3]||'').slice(3,6).trim(),10) || 0
  const atoms = []
  for (let i = 0; i < numAtoms; i++) {
    const line = lines[4+i] || ''
    atoms.push({ id:i, visible:true,
      x:parseFloat(line.slice(0,10).trim()),
      y:parseFloat(line.slice(10,20).trim()),
      z:parseFloat(line.slice(20,30).trim()),
      elem:line.slice(31,34).trim() })
  }
  const bonds = []
  for (let i = 0; i < numBonds; i++) {
    const line = lines[4+numAtoms+i] || ''
    const a1=parseInt(line.slice(0,3).trim(),10)-1
    const a2=parseInt(line.slice(3,6).trim(),10)-1
    const order=parseInt(line.slice(6,9).trim(),10)||1
    if (a1>=0&&a2>=0&&a1<numAtoms&&a2<numAtoms) bonds.push({a1,a2,order})
  }
  return { atoms, bonds }
}

const CPK_HEX={C:0x909090,H:0xffffff,N:0x3050f8,O:0xff0d0d,S:0xffff30,F:0x90e050,Cl:0x1ff01f,Br:0xa62929,P:0xff8000,I:0x940094}
const CPK_STR={C:'#909090',H:'#e8e8e8',N:'#4169e1',O:'#e13030',S:'#d4c400',F:'#40d0d0',Cl:'#30d030',Br:'#a05010',P:'#e08000',I:'#800080'}
const RADII={H:0.28,C:0.38,N:0.34,O:0.32,S:0.48,F:0.30,Cl:0.42,Br:0.47,P:0.44,I:0.52}
const getColor =e=>CPK_HEX[e]??0x888888
const getRadius=e=>RADII[e]??0.36

// ═══════════════════════════════════════════════════════════════════════════════
// ─── CHEMISTRY ENGINE  (VSEPR geometry + clash detection) ─────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

// Standard valence (max bonds each element can form)
const VALENCE = {
  H:1, C:4, N:3, O:2, F:1, P:5, S:6, Cl:1, Br:1, I:1,
  Na:1, Mg:2, Al:3, Si:4, K:1, Ca:2, Fe:3, Cu:2, Zn:2, Se:2,
}

// Covalent radius (Å) — used for both bond-length estimation and clash detection
const COV_RADIUS = {
  H:0.31, C:0.76, N:0.71, O:0.66, F:0.57, P:1.07, S:1.05,
  Cl:1.02, Br:1.20, I:1.39, Na:1.66, Mg:1.41, Al:1.21, Si:1.11,
  K:2.03, Ca:1.76, Fe:1.32, Cu:1.32, Zn:1.22, Se:1.20,
}

// Bond length = sum of covalent radii × slight scaling factor
function bondLength(e1, e2) {
  return ((COV_RADIUS[e1]||0.77) + (COV_RADIUS[e2]||0.77)) * 1.15
}

// Minimum allowed distance between any two atoms (clash = below this)
function clashDist(e1, e2) {
  return ((COV_RADIUS[e1]||0.77) + (COV_RADIUS[e2]||0.77)) * 0.85
}

// Sum of bond orders on an atom
function usedValence(atomIdx, bonds) {
  return bonds
    .filter(b => b.a1 === atomIdx || b.a2 === atomIdx)
    .reduce((s, b) => s + (b.order||1), 0)
}

// Free valence slots remaining
function freeValence(atomIdx, elem, bonds) {
  return Math.max(0, (VALENCE[elem] ?? 1) - usedValence(atomIdx, bonds))
}

// All visible atoms with at least one free valence slot
function getOpenAtoms(atoms, bonds) {
  return atoms
    .filter(a => a.visible && freeValence(a.id, a.elem, bonds) > 0)
    .map(a => ({ ...a, free: freeValence(a.id, a.elem, bonds) }))
    .sort((a, b) => b.free - a.free)
}

// ── Vector helpers ─────────────────────────────────────────────────────────────
const v3 = (x,y,z) => ({x,y,z})
const vadd = (a,b) => v3(a.x+b.x, a.y+b.y, a.z+b.z)
const vsub = (a,b) => v3(a.x-b.x, a.y-b.y, a.z-b.z)
const vscale = (a,s) => v3(a.x*s, a.y*s, a.z*s)
const vlen = a => Math.sqrt(a.x*a.x + a.y*a.y + a.z*a.z)
const vnorm = a => { const l=vlen(a)||1; return v3(a.x/l,a.y/l,a.z/l) }
const vdot = (a,b) => a.x*b.x + a.y*b.y + a.z*b.z
const vcross = (a,b) => v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x)

// Rotate vector `v` around unit axis `k` by angle θ (Rodrigues' formula)
function vrotate(v, k, theta) {
  const c = Math.cos(theta), s = Math.sin(theta)
  return vadd(vadd(vscale(v, c), vscale(vcross(k,v), s)), vscale(k, vdot(k,v)*(1-c)))
}

// Find a unit vector perpendicular to `d`
function anyPerp(d) {
  const dn = vnorm(d)
  let perp = Math.abs(dn.x) < 0.9 ? v3(1,0,0) : v3(0,1,0)
  // Gram-Schmidt
  perp = vsub(perp, vscale(dn, vdot(dn, perp)))
  return vnorm(perp)
}

// ── VSEPR placement ────────────────────────────────────────────────────────────
// Returns the ideal unit direction for a new bond on `host` given its existing
// bond directions, based on electron-pair geometry (VSEPR).
function vsepDirection(existingDirs) {
  const n = existingDirs.length

  if (n === 0) {
    // Isolated atom — place along +X
    return v3(1, 0, 0)
  }

  if (n === 1) {
    // 1 existing bond → tetrahedral angle 109.47° away, in a random perp plane
    const d = existingDirs[0]
    const perp = anyPerp(d)
    const cosT = -1/3           // cos(109.47°)
    const sinT = Math.sqrt(8/9) // sin(109.47°)
    return vnorm(vadd(vscale(d, cosT), vscale(perp, sinT)))
  }

  if (n === 2) {
    // 2 existing bonds → place in the plane of the two, bisecting the open side
    // then rotate out-of-plane by tetrahedral angle
    const d0 = existingDirs[0], d1 = existingDirs[1]
    const bisect = vnorm(vadd(d0, d1))   // points INTO existing bonds
    const outPlane = vnorm(vcross(d0, d1))
    // New direction: opposite bisector, tilted out of plane
    const neg = vscale(bisect, -1)
    // Tetrahedral: 109.5° from each existing bond means ~109.5° from both
    // Tilt 54.75° out of the plane from -bisector  (half of 109.47°)
    return vnorm(vrotate(neg, outPlane, Math.PI * 0.35))
  }

  if (n === 3) {
    // 3 existing bonds (sp3) → new bond opposite to sum of existing
    const sum = existingDirs.reduce((acc, d) => vadd(acc, d), v3(0,0,0))
    return vnorm(vscale(sum, -1))
  }

  // 4+ existing bonds (e.g. P with 5 bonds) — place opposite to net sum,
  // with slight out-of-plane tilt to avoid exact collinearity
  const sum = existingDirs.reduce((acc, d) => vadd(acc, d), v3(0,0,0))
  const neg = vnorm(vscale(sum, -1))
  const perp = anyPerp(neg)
  return vnorm(vadd(neg, vscale(perp, 0.1)))
}

// ── Clash check ────────────────────────────────────────────────────────────────
// Returns true if `pos` is too close to any existing visible atom (except host)
function hasClash(pos, newElem, atoms, hostIdx) {
  for (const a of atoms) {
    if (!a.visible || a.id === hostIdx) continue
    const dx = pos.x-a.x, dy = pos.y-a.y, dz = pos.z-a.z
    const dist = Math.sqrt(dx*dx+dy*dy+dz*dz)
    if (dist < clashDist(newElem, a.elem)) return true
  }
  return false
}

// ── Main placement function ────────────────────────────────────────────────────
// Tries up to 8 rotational variants to find a clash-free position.
function placeAtom(newElem, hostAtom, hostIdx, atoms, bonds) {
  // Gather unit vectors of existing bonds FROM the host
  const existingDirs = bonds
    .filter(b => b.a1 === hostIdx || b.a2 === hostIdx)
    .map(b => {
      const other = atoms[b.a1 === hostIdx ? b.a2 : b.a1]
      if (!other) return null
      return vnorm(vsub(other, hostAtom))
    })
    .filter(Boolean)

  const bl = bondLength(hostAtom.elem, newElem)
  const idealDir = vsepDirection(existingDirs)
  const idealPos = vadd(hostAtom, vscale(idealDir, bl))

  if (!hasClash(idealPos, newElem, atoms, hostIdx)) return idealPos

  // Try rotating the ideal direction around the net-bond axis in 8 steps
  const axis = existingDirs.length > 0
    ? vnorm(existingDirs.reduce((a,b) => vadd(a,b), v3(0,0,0)))
    : anyPerp(idealDir)

  for (let i = 1; i <= 8; i++) {
    const rotDir = vnorm(vrotate(idealDir, axis, (Math.PI * 2 * i) / 8))
    const candidate = vadd(hostAtom, vscale(rotDir, bl))
    if (!hasClash(candidate, newElem, atoms, hostIdx)) return candidate
  }

  // Last resort: extend bond length slightly to escape clash
  return vadd(hostAtom, vscale(idealDir, bl * 1.3))
}

// ── Public API ─────────────────────────────────────────────────────────────────
// Returns { newAtoms, newBonds, newAtomId, hostAtomId, bondOrder, error }
function chemAddAtom(newElem, atoms, bonds, preferHostId = null) {
  const newElemValence = VALENCE[newElem] ?? 1
  const openAtoms = getOpenAtoms(atoms, bonds)

  if (openAtoms.length === 0) {
    return { error: 'All atoms are fully bonded — no free valence slots available.' }
  }

  // Select host: explicit preference → heavy atom with most free slots → anything
  let host = null
  if (preferHostId != null) host = openAtoms.find(a => a.id === preferHostId) || null
  if (!host) host = openAtoms.find(a => a.elem !== 'H') || openAtoms[0]

  // Validate: new element must also have free valence
  if (newElemValence < 1) {
    return { error: `${newElem} cannot form any bonds (valence 0).` }
  }

  // Bond order = 1 (single bond) for all manual additions
  const bondOrder = 1

  // Compute VSEPR-correct, clash-free 3D position
  const pos = placeAtom(newElem, host, host.id, atoms, bonds)

  const newAtomId = atoms.length
  const newAtoms = [
    ...atoms,
    { id: newAtomId, visible: true, added: true, elem: newElem,
      x: pos.x, y: pos.y, z: pos.z }
  ]
  const newBonds = [
    ...bonds,
    { a1: host.id, a2: newAtomId, order: bondOrder, added: true }
  ]

  return { newAtoms, newBonds, newAtomId, hostAtomId: host.id, bondOrder, error: null }
}

// ─── Molecule → display label (atom formula) ────────────────────────────────
// We no longer build SMILES in JS (too error-prone for ring systems).
// Instead we export a V2000 SDF and send it to /predict as smiles_from_sdf.
// The backend parses it with RDKit and returns the canonical SMILES + score.
function buildFormulaFromAtomsBonds(atoms, bonds) {
  // Just count visible atoms by element — used for display only
  const vis = atoms.filter(a => a.visible)
  const counts = {}
  vis.forEach(a => { counts[a.elem] = (counts[a.elem]||0)+1 })
  // Hill order: C first, H second, rest alphabetical
  const order = ['C','H',...Object.keys(counts).filter(e=>e!=='C'&&e!=='H').sort()]
  return order.filter(e=>counts[e]).map(e=>counts[e]>1?`${e}${counts[e]}`:e).join('')
}

// ─── Element library ──────────────────────────────────────────────────────────
const ELEMENT_LIBRARY = [
  { elem:'H',  name:'Hydrogen',   mass:1.008,   group:'Nonmetal'  },
  { elem:'C',  name:'Carbon',     mass:12.011,  group:'Nonmetal'  },
  { elem:'N',  name:'Nitrogen',   mass:14.007,  group:'Nonmetal'  },
  { elem:'O',  name:'Oxygen',     mass:15.999,  group:'Nonmetal'  },
  { elem:'F',  name:'Fluorine',   mass:18.998,  group:'Halogen'   },
  { elem:'P',  name:'Phosphorus', mass:30.974,  group:'Nonmetal'  },
  { elem:'S',  name:'Sulfur',     mass:32.06,   group:'Nonmetal'  },
  { elem:'Cl', name:'Chlorine',   mass:35.45,   group:'Halogen'   },
  { elem:'Br', name:'Bromine',    mass:79.904,  group:'Halogen'   },
  { elem:'I',  name:'Iodine',     mass:126.904, group:'Halogen'   },
  { elem:'Na', name:'Sodium',     mass:22.990,  group:'Metal'     },
  { elem:'Mg', name:'Magnesium',  mass:24.305,  group:'Metal'     },
  { elem:'Al', name:'Aluminium',  mass:26.982,  group:'Metal'     },
  { elem:'Si', name:'Silicon',    mass:28.085,  group:'Metalloid' },
  { elem:'K',  name:'Potassium',  mass:39.098,  group:'Metal'     },
  { elem:'Ca', name:'Calcium',    mass:40.078,  group:'Metal'     },
  { elem:'Fe', name:'Iron',       mass:55.845,  group:'Metal'     },
  { elem:'Cu', name:'Copper',     mass:63.546,  group:'Metal'     },
  { elem:'Zn', name:'Zinc',       mass:65.38,   group:'Metal'     },
  { elem:'Se', name:'Selenium',   mass:78.971,  group:'Nonmetal'  },
]

const GROUP_COLORS = {
  Nonmetal: 'rgba(110,231,183,0.08)', Halogen:  'rgba(248,113,113,0.08)',
  Metal:    'rgba(212,175,114,0.08)', Metalloid:'rgba(168,85,247,0.08)',
}
const GROUP_BORDER = {
  Nonmetal: 'rgba(110,231,183,0.22)', Halogen:  'rgba(248,113,113,0.22)',
  Metal:    'rgba(212,175,114,0.22)', Metalloid:'rgba(168,85,247,0.22)',
}
const GROUP_TEXT = {
  Nonmetal: '#6ee7b7', Halogen: '#f87171', Metal: '#d4af72', Metalloid: '#c084fc',
}

// ─── rebuildScene ─────────────────────────────────────────────────────────────
function rebuildScene(THREE,scene,groupRef,atomMeshesRef,atoms,bonds,mode){
  if(groupRef.current){
    scene.remove(groupRef.current)
    groupRef.current.traverse(o=>{
      o.geometry?.dispose()
      Array.isArray(o.material)?o.material.forEach(m=>m.dispose()):o.material?.dispose()
    })
  }
  atomMeshesRef.current=[]
  const group=new THREE.Group()
  groupRef.current=group
  scene.add(group)
  const vis=atoms.filter(a=>a.visible)
  if(!vis.length) return
  const cx=vis.reduce((s,a)=>s+a.x,0)/vis.length
  const cy=vis.reduce((s,a)=>s+a.y,0)/vis.length
  const cz=vis.reduce((s,a)=>s+a.z,0)/vis.length
  atoms.forEach((atom,idx)=>{
    if(!atom.visible){atomMeshesRef.current[idx]=null;return}
    const r=mode==='spacefill'?getRadius(atom.elem)*2.2:mode==='stick'?getRadius(atom.elem)*0.28:getRadius(atom.elem)
    const mesh=new THREE.Mesh(
      new THREE.SphereGeometry(r,24,24),
      new THREE.MeshPhongMaterial({
        color: getColor(atom.elem), shininess:90, specular:0x555555,
        emissive: atom.added ? 0x220033 : 0x000000  // subtle glow for added atoms
      })
    )
    mesh.position.set(atom.x-cx,atom.y-cy,atom.z-cz)
    mesh.userData={atomIdx:idx,atom}
    group.add(mesh)
    atomMeshesRef.current[idx]=mesh
  })
  if(mode!=='spacefill'){
    const up=new THREE.Vector3(0,1,0)
    bonds.forEach(bond=>{
      const a1=atoms[bond.a1],a2=atoms[bond.a2]
      if(!a1?.visible||!a2?.visible) return
      const p1=new THREE.Vector3(a1.x-cx,a1.y-cy,a1.z-cz)
      const p2=new THREE.Vector3(a2.x-cx,a2.y-cy,a2.z-cz)
      const dir=new THREE.Vector3().subVectors(p2,p1)
      const len=dir.length()
      const mid=p1.clone().addScaledVector(dir.clone().normalize(),len/2)
      const bR=mode==='stick'?0.13:0.07
      const mkCyl=(r,off,isNew)=>{
        const c=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,12),
          new THREE.MeshPhongMaterial({
            color: isNew ? 0x9333ea : 0x888888,
            shininess:40,
            emissive: isNew ? 0x1a0033 : 0x000000,
          }))
        c.position.copy(mid);if(off)c.position.add(off)
        c.quaternion.setFromUnitVectors(up,dir.clone().normalize())
        group.add(c)
      }
      const isNewBond = bond.added === true
      if(bond.order>=2){
        const perp=new THREE.Vector3().crossVectors(dir.normalize(),new THREE.Vector3(0,0,1)).normalize().multiplyScalar(0.13)
        mkCyl(bR*0.75,perp,isNewBond);mkCyl(bR*0.75,perp.clone().negate(),isNewBond)
      } else mkCyl(bR,null,isNewBond)
    })
  }
}

// ─── Atom Library Panel ───────────────────────────────────────────────────────
function AtomLibrary({ onAdd, addedLog, openAtoms, selectedHostId, onSelectHost }) {
  const [search,      setSearch]      = useState('')
  const [filterGroup, setFilterGroup] = useState('All')
  const [flashElem,   setFlashElem]   = useState(null)
  const [chemError,   setChemError]   = useState(null)

  const groups = ['All', 'Nonmetal', 'Halogen', 'Metal', 'Metalloid']

  const filtered = ELEMENT_LIBRARY.filter(el => {
    const matchSearch = el.elem.toLowerCase().includes(search.toLowerCase()) ||
                        el.name.toLowerCase().includes(search.toLowerCase())
    const matchGroup  = filterGroup === 'All' || el.group === filterGroup
    return matchSearch && matchGroup
  })

  const handleAdd = (elem) => {
    const result = onAdd(elem)
    if (result?.error) {
      setChemError(result.error)
      setTimeout(() => setChemError(null), 3500)
    } else {
      setFlashElem(elem)
      setChemError(null)
      setTimeout(() => setFlashElem(null), 600)
    }
  }

  return (
    <div className="vr-library">
      {/* Host atom selector */}
      <div className="vr-host-selector">
        <div className="vr-host-title">Bond to atom:</div>
        {openAtoms.length === 0 ? (
          <div className="vr-host-none">All atoms fully bonded — no free valence slots</div>
        ) : (
          <div className="vr-host-list">
            <button
              className={`vr-host-btn${selectedHostId === null ? ' active' : ''}`}
              onClick={() => onSelectHost(null)}
            >Auto</button>
            {openAtoms.slice(0, 8).map(a => (
              <button
                key={a.id}
                className={`vr-host-btn${selectedHostId === a.id ? ' active' : ''}`}
                style={{ borderColor: CPK_STR[a.elem] || '#888' }}
                onClick={() => onSelectHost(a.id)}
                title={`${a.elem} #${a.id+1} — ${a.free} free slot${a.free>1?'s':''}`}
              >
                <span style={{ color: CPK_STR[a.elem] || '#aaa' }}>{a.elem}</span>
                <span className="vr-host-id">#{a.id+1}</span>
                <span className="vr-host-free">{a.free}v</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {chemError && (
        <div className="vr-chem-error">⚠ {chemError}</div>
      )}

      <input
        className="vr-library-search"
        placeholder="🔍  Search element or name…"
        value={search}
        onChange={e => setSearch(e.target.value)}
      />

      <div className="vr-library-filters">
        {groups.map(g => (
          <button key={g}
            className={`vr-filter-btn${filterGroup===g?' active':''}`}
            onClick={() => setFilterGroup(g)}>{g}
          </button>
        ))}
      </div>

      <div className="vr-library-grid">
        {filtered.map(el => {
          const addedCount = addedLog.filter(a => a === el.elem).length
          const isFlashing = flashElem === el.elem
          const color = CPK_STR[el.elem] || GROUP_TEXT[el.group]
          const noSlots = openAtoms.length === 0
          return (
            <div key={el.elem}
              className={`vr-lib-card${isFlashing?' flash':''}${noSlots?' disabled':''}`}
              style={{ background:GROUP_COLORS[el.group], borderColor: isFlashing ? color : GROUP_BORDER[el.group] }}
              onClick={() => !noSlots && handleAdd(el.elem)}
              title={noSlots ? 'No free valence slots available' : `Add ${el.name} (valence ${VALENCE[el.elem]??1})`}
            >
              {addedCount > 0 && <div className="vr-lib-badge">{addedCount}</div>}
              <div className="vr-lib-symbol" style={{ color }}>{el.elem}</div>
              <div className="vr-lib-name">{el.name}</div>
              <div className="vr-lib-mass">{el.mass} u</div>
              <div className="vr-lib-valence">val. {VALENCE[el.elem]??'?'}</div>
            </div>
          )
        })}
      </div>

      {addedLog.length > 0 && (
        <div className="vr-added-log">
          <div className="vr-added-log-title">Added ({addedLog.length})</div>
          <div className="vr-added-log-list">
            {[...addedLog].reverse().slice(0,8).map((elem,i) => (
              <span key={i} className="vr-added-chip"
                style={{ color:CPK_STR[elem]||'#888', borderColor:(CPK_STR[elem]||'#888')+'55' }}>
                {elem}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main VRMolViewer ─────────────────────────────────────────────────────────
export default function VRMolViewer({ sdf, smiles, onClose, apiBase = VR_API }) {
  const [atoms,        setAtoms]        = useState(() => parseSDF(sdf).atoms.map(a=>({...a})))
  const [bonds,        setBonds]        = useState(() => parseSDF(sdf).bonds)
  const [mode,         setMode]         = useState('ball-stick')
  const [addedLog,     setAddedLog]     = useState([])
  const [activeTab,    setActiveTab]    = useState('atoms')
  const [selectedHostId, setSelectedHostId] = useState(null) // preferred host for next add
  const [lastAddInfo,  setLastAddInfo]  = useState(null)     // { elem, hostElem, hostId }

  const mountRef      = useRef(null)
  const sceneRef      = useRef(null)
  const rendererRef   = useRef(null)
  const cameraRef     = useRef(null)
  const groupRef      = useRef(null)
  const atomMeshesRef = useRef([])
  const raycasterRef  = useRef(null)
  const dragRef       = useRef({active:false,lastX:0,lastY:0})
  const xrSessionRef  = useRef(null)
  const atomsSnapRef  = useRef(atoms)
  const modeSnapRef   = useRef(mode)
  const bondsSnapRef  = useRef(bonds)

  useEffect(()=>{ atomsSnapRef.current = atoms },[atoms])
  useEffect(()=>{ modeSnapRef.current  = mode  },[mode])
  useEffect(()=>{ bondsSnapRef.current = bonds },[bonds])

  const [selectedAtom, setSelectedAtom] = useState(null)
  const [isLoading,    setIsLoading]    = useState(true)
  const [vrSupported,  setVrSupported]  = useState(false)
  const [vrActive,     setVrActive]     = useState(false)
  const [vrError,      setVrError]      = useState(null)
  const [showGuide,    setShowGuide]    = useState(false)
  const [vrKey,        setVrKey]        = useState(0)  // increments to force Three.js remount

  useEffect(()=>{
    if(!sceneRef.current||!window.THREE) return
    rebuildScene(window.THREE,sceneRef.current,groupRef,atomMeshesRef,atoms,bonds,mode)
  },[atoms,mode,bonds])

  useEffect(()=>{
    navigator.xr?.isSessionSupported('immersive-vr').then(s=>setVrSupported(s)).catch(()=>{})
  },[])

  // ── Chemistry-aware atom addition ─────────────────────────────────────────
  const addAtomFromLibrary = (elem) => {
    const result = chemAddAtom(elem, atoms, bonds, selectedHostId)
    if (result.error) return result  // bubble error up to AtomLibrary
    const { newAtoms, newBonds, newAtomId, hostAtomId, bondOrder } = result

    // Mark the new bond as "added" for purple highlight in viewer
    newBonds[newBonds.length - 1].added = true

    setAtoms(newAtoms)
    setBonds(newBonds)
    setAddedLog(prev => [...prev, elem])
    setLastAddInfo({ elem, hostElem: atoms[hostAtomId]?.elem, hostId: hostAtomId, bondOrder })
    // Auto-select the new atom in the panel
    setSelectedAtom({ ...newAtoms[newAtomId] })
    setActiveTab('atoms')
    return { error: null }
  }

  // Compute open atoms for the library panel
  const openAtoms = getOpenAtoms(atoms, bonds)

  // ── Prediction state ──────────────────────────────────────────────────────
  const [predLoading,  setPredLoading]  = useState(false)
  const [predResult,   setPredResult]   = useState(null)   // { score, confidence, is_promising, repaired_smiles, sdf }
  const [predError,    setPredError]    = useState(null)
  const [molChanged,   setMolChanged]   = useState(false)  // dirty flag

  // Mark molecule as changed whenever atoms/bonds change after mount
  const mountedRef = useRef(false)
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return }
    setMolChanged(true)
    setPredResult(null)
  }, [atoms, bonds])

  // ── atoms+bonds → SMILES (via SDF export → send to /predict) ─────────────
  // We export the current state as a minimal SDF string, then the /predict
  // endpoint runs RDKit on it and returns the canonical SMILES + score.
  // This avoids re-implementing a full SMILES encoder in JS.
  function exportAsSDF() {
    const vis = atoms.filter(a => a.visible)
    const visIdx = {}
    vis.forEach((a, i) => { visIdx[a.id] = i + 1 })
    const visBonds = bonds.filter(b => vis.find(a=>a.id===b.a1) && vis.find(a=>a.id===b.a2))

    const header = [
      '',
      ' VRMolViewer',
      '',
      `${String(vis.length).padStart(3)}${String(visBonds.length).padStart(3)}  0  0  0  0  0  0  0  0999 V2000`,
    ]
    const atomLines = vis.map(a =>
      `${a.x.toFixed(4).padStart(10)}${a.y.toFixed(4).padStart(10)}${a.z.toFixed(4).padStart(10)} ${a.elem.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`
    )
    const bondLines = visBonds.map(b =>
      `${String(visIdx[b.a1]).padStart(3)}${String(visIdx[b.a2]).padStart(3)}${String(b.order||1).padStart(3)}  0`
    )
    return [...header, ...atomLines, ...bondLines, 'M  END', '$$$$'].join('\n')
  }

  // ── Build SMILES from current visible atoms+bonds ────────────────────────
  // Uses DFS with ring-closure detection to generate a valid SMILES string
  // that reflects the user's additions/removals — this is what gets predicted.
  function getCurrentSmiles() {
    const vis = atoms.filter(a => a.visible)
    if (vis.length === 0) return smiles  // fallback to original

    const visSet = new Set(vis.map(a => a.id))
    const visiBonds = bonds.filter(b => visSet.has(b.a1) && visSet.has(b.a2))

    // If molecule is unmodified, return original smiles (most accurate)
    const hasChanges = atoms.some(a => a.added) ||
                       atoms.some(a => !a.visible && !a.added) ||
                       bonds.some(b => b.added)
    if (!hasChanges) return smiles

    // Build adjacency list
    const adj = {}
    vis.forEach(a => { adj[a.id] = [] })
    visiBonds.forEach(b => {
      adj[b.a1].push({ to: b.a2, order: b.order || 1 })
      adj[b.a2].push({ to: b.a1, order: b.order || 1 })
    })

    const bondSym = o => o === 2 ? '=' : o === 3 ? '#' : ''
    const ORGANIC = new Set(['B','C','N','O','P','S','F','Cl','Br','I'])
    const atomSym = e => ORGANIC.has(e) ? e : `[${e}]`

    // First pass: find ring closure edges
    const visited = new Set()
    const ringEdges = new Set()
    const dfsFind = (id, parentId) => {
      visited.add(id)
      for (const { to } of (adj[id] || [])) {
        if (to === parentId) continue
        const key = [Math.min(id,to), Math.max(id,to)].join('-')
        if (visited.has(to)) { ringEdges.add(key) }
        else dfsFind(to, id)
      }
    }
    dfsFind(vis[0].id, -1)
    visited.clear()

    const ringOpens = {}
    let ringCounter = 1
    let result = ''

    const dfs = (id, parentId, parentBondOrder) => {
      visited.add(id)
      const atom = atoms.find(a => a.id === id)
      if (!atom) return
      result += (parentId === -1 ? '' : bondSym(parentBondOrder)) + atomSym(atom.elem)

      // Ring closures on this atom
      for (const { to, order } of (adj[id] || [])) {
        if (to === parentId) continue
        const key = [Math.min(id,to), Math.max(id,to)].join('-')
        if (ringEdges.has(key)) {
          if (ringOpens[key] === undefined) {
            const digit = ringCounter++
            ringOpens[key] = digit
            result += bondSym(order) + digit
          } else {
            result += bondSym(order) + ringOpens[key]
          }
        }
      }

      // Children
      const children = (adj[id] || []).filter(({ to }) => {
        if (to === parentId) return false
        const key = [Math.min(id,to), Math.max(id,to)].join('-')
        return !ringEdges.has(key) && !visited.has(to)
      })
      children.forEach(({ to, order }, i) => {
        if (i < children.length - 1) { result += '('; dfs(to, id, order); result += ')' }
        else dfs(to, id, order)
      })
    }

    dfs(vis[0].id, -1, 1)
    return result || smiles
  }

  const runPrediction = async () => {
    setPredLoading(true); setPredError(null); setPredResult(null)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 90000)
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: getCurrentSmiles() }),
        signal: controller.signal,
      })
      clearTimeout(timeout)
      const text = await res.text()
      if (!text.trim()) throw new Error('Empty response — server may be waking up, try again in 30s')
      let json
      try { json = JSON.parse(text) } catch { throw new Error(`Server error: ${text.slice(0,120)}`) }
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`)
      if (json.error) throw new Error(json.error)
      setPredResult(json)
      setMolChanged(false)
    } catch (err) {
      const isTimeout = err.name === 'AbortError'
      setPredError(isTimeout ? 'Timed out — server may be sleeping. Wait 30s and retry.' : (err.message || String(err)))
    } finally {
      setPredLoading(false)
    }
  }

  // ── Three.js bootstrap ────────────────────────────────────────────────────
  useEffect(()=>{
    let mounted = true
    const cleanupHolder = { fn: () => {} }

    const boot = THREE => {
      if (!mounted || !mountRef.current) return
      Array.from(mountRef.current.querySelectorAll('canvas')).forEach(c => c.remove())
      let W = mountRef.current.clientWidth, H = mountRef.current.clientHeight
      if (!W || !H) {
        mountRef.current.style.width='100%'; mountRef.current.style.height='600px'
        W = mountRef.current.offsetWidth||window.innerWidth; H = mountRef.current.offsetHeight||600
      }
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0d0618)
      sceneRef.current = scene
      const renderer = new THREE.WebGLRenderer({ antialias:true })
      renderer.setPixelRatio(window.devicePixelRatio); renderer.setSize(W,H)
      renderer.xr.enabled = true
      mountRef.current.appendChild(renderer.domElement)
      rendererRef.current = renderer
      const camera = new THREE.PerspectiveCamera(55,W/H,0.01,1000)
      camera.position.set(0,0,9); cameraRef.current = camera
      scene.add(new THREE.AmbientLight(0x404060,0.9))
      const sun=new THREE.DirectionalLight(0xffffff,1.3); sun.position.set(5,10,5); scene.add(sun)
      const rim=new THREE.DirectionalLight(0xa855f7,0.4); rim.position.set(-5,-3,-5); scene.add(rim)
      const fill=new THREE.DirectionalLight(0x60a5fa,0.3); fill.position.set(0,-8,3); scene.add(fill)
      raycasterRef.current = new THREE.Raycaster()
      const sp=[]
      for(let i=0;i<900;i++) sp.push((Math.random()-.5)*200,(Math.random()-.5)*200,(Math.random()-.5)*200)
      const sg=new THREE.BufferGeometry()
      sg.setAttribute('position',new THREE.Float32BufferAttribute(sp,3))
      scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xffffff,size:0.15,transparent:true,opacity:0.5})))
      rebuildScene(THREE,scene,groupRef,atomMeshesRef,atomsSnapRef.current,bondsSnapRef.current,modeSnapRef.current)
      setIsLoading(false)
      const clock=new THREE.Clock()
      renderer.setAnimationLoop(()=>{
        const dt=clock.getDelta()
        if(groupRef.current&&!dragRef.current.active&&!xrSessionRef.current)
          groupRef.current.rotation.y+=dt*0.35
        renderer.render(scene,camera)
      })
      const cv=renderer.domElement
      const onDown=e=>{const s=e.touches?.[0]??e;dragRef.current={active:true,lastX:s.clientX,lastY:s.clientY};e.preventDefault()}
      const onUp=()=>{dragRef.current.active=false}
      const onMove=e=>{
        if(!dragRef.current.active||!groupRef.current) return
        const s=e.touches?.[0]??e
        groupRef.current.rotation.y+=(s.clientX-dragRef.current.lastX)*0.013
        groupRef.current.rotation.x+=(s.clientY-dragRef.current.lastY)*0.013
        dragRef.current.lastX=s.clientX;dragRef.current.lastY=s.clientY;e.preventDefault()
      }
      const onWheel=e=>{camera.position.z=Math.max(2,Math.min(22,camera.position.z+e.deltaY*0.022));e.preventDefault()}
      const onClick=e=>{
        if(!raycasterRef.current||!groupRef.current) return
        const rect=cv.getBoundingClientRect()
        const mouse=new THREE.Vector2(((e.clientX-rect.left)/rect.width)*2-1,((e.clientY-rect.top)/rect.height)*-2+1)
        raycasterRef.current.setFromCamera(mouse,camera)
        const meshes=atomMeshesRef.current.filter(Boolean)
        const hits=raycasterRef.current.intersectObjects(meshes)
        meshes.forEach(m=>m.material.emissive?.setHex(0x000000))
        if(hits.length>0){
          hits[0].object.material.emissive=new THREE.Color(0x6622bb)
          setSelectedAtom({...hits[0].object.userData.atom})
        } else setSelectedAtom(null)
      }
      cv.addEventListener('mousedown',onDown,{passive:false})
      cv.addEventListener('touchstart',onDown,{passive:false})
      cv.addEventListener('mousemove',onMove,{passive:false})
      cv.addEventListener('touchmove',onMove,{passive:false})
      cv.addEventListener('mouseup',onUp)
      cv.addEventListener('touchend',onUp)
      cv.addEventListener('wheel',onWheel,{passive:false})
      cv.addEventListener('click',onClick)
      const onResize=()=>{if(!mountRef.current) return;const w=mountRef.current.clientWidth,h=mountRef.current.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h)}
      window.addEventListener('resize',onResize)
      cleanupHolder.fn=()=>{
        mounted=false;renderer.setAnimationLoop(null)
        cv.removeEventListener('mousedown',onDown);cv.removeEventListener('touchstart',onDown)
        cv.removeEventListener('mousemove',onMove);cv.removeEventListener('touchmove',onMove)
        cv.removeEventListener('mouseup',onUp);cv.removeEventListener('touchend',onUp)
        cv.removeEventListener('wheel',onWheel);cv.removeEventListener('click',onClick)
        window.removeEventListener('resize',onResize)
        if(groupRef.current) groupRef.current.traverse(o=>{o.geometry?.dispose();Array.isArray(o.material)?o.material.forEach(m=>m.dispose()):o.material?.dispose()})
        renderer.dispose();try{renderer.forceContextLoss()}catch(e){}
        if(renderer.domElement.parentNode===mountRef.current) mountRef.current?.removeChild(renderer.domElement)
        sceneRef.current=null;rendererRef.current=null;cameraRef.current=null;groupRef.current=null;atomMeshesRef.current=[];raycasterRef.current=null
      }
    }
    const runBoot=()=>{
      if(window.THREE) boot(window.THREE)
      else{
        const ex=document.querySelector('script[data-three]')
        if(ex) ex.addEventListener('load',()=>boot(window.THREE))
        else{const s=document.createElement('script');s.src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';s.dataset.three='1';s.onload=()=>boot(window.THREE);document.head.appendChild(s)}
      }
    }
    const raf1=requestAnimationFrame(()=>{const raf2=requestAnimationFrame(runBoot);cleanupHolder.fn_raf=()=>cancelAnimationFrame(raf2)})
    const orig=cleanupHolder.fn
    cleanupHolder.fn=()=>{cancelAnimationFrame(raf1);cleanupHolder.fn_raf?.();orig()}
    return ()=>cleanupHolder.fn()
  },[]) // eslint-disable-line

  const hideAtom   = idx=>{setAtoms(p=>{const n=[...p];n[idx]={...n[idx],visible:false};return n});setSelectedAtom(p=>p?.id===idx?{...p,visible:false}:p)}
  const showAtom   = idx=>{setAtoms(p=>{const n=[...p];n[idx]={...n[idx],visible:true};return n});setSelectedAtom(p=>p?.id===idx?{...p,visible:true}:p)}
  // removeAtom: hide atom AND delete all bonds touching it so valence is freed on neighbours
  const removeAtom = idx=>{
    setAtoms(p=>{const n=[...p];n[idx]={...n[idx],visible:false};return n})
    setBonds(p=>p.filter(b=>b.a1!==idx && b.a2!==idx))
    setSelectedAtom(null)
  }
  const restoreAll = ()=>{setAtoms(p=>p.map(a=>({...a,visible:true})));setSelectedAtom(null)}

  const enterVR = async()=>{
    setVrError(null)
    // If navigator.xr is missing entirely the browser has no WebXR support at all
    if(!navigator.xr){
      setVrError('WebXR not available in this browser. Open this page inside your Meta Quest Browser.')
      return
    }
    if(!rendererRef.current) return
    try{
      const session=await navigator.xr.requestSession('immersive-vr',{requiredFeatures:['local'],optionalFeatures:['hand-tracking']})
      xrSessionRef.current=session;rendererRef.current.xr.setSession(session);setVrActive(true)
      session.addEventListener('end',()=>{setVrActive(false);xrSessionRef.current=null})
    }catch(err){
      setVrError(err.message||'Could not start VR session.')
    }
  }
  const exitVR=()=>xrSessionRef.current?.end()

  const hiddenCount = atoms.filter(a=>!a.visible).length
  const atomCounts  = atoms.reduce((acc,a)=>{if(a.visible)acc[a.elem]=(acc[a.elem]||0)+1;return acc},{})

  // ── Separate guide page — Three.js re-inits via key when returning ────────
  if (showGuide) {
    return (
      <div className="vr-guide-page">
        <div className="vr-guide-overlay-topbar">
          <button className="vr-back-btn" onClick={()=>setShowGuide(false)}>← Back to Viewer</button>
          <span className="vr-guide-page-title">◉ How to View in Oculus / Meta Quest</span>
          <div style={{width:'160px'}}/>
        </div>
        <div className="vr-guide-page-body">
          <div className="vr-guide-hero">
            <div className="vr-guide-hero-icon">◉</div>
            <div className="vr-guide-hero-text">
              <h2>Enter Immersive VR with your Oculus</h2>
              <p>WebXR only works when the page is opened <strong>inside the headset itself</strong>. Your PC browser cannot launch VR — follow these steps.</p>
            </div>
          </div>
          <div className="vr-guide-page-steps">
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">1</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Open a second terminal on your PC</div>
                <div className="vr-guide-page-step-desc">Keep your app running (<code>npm run dev</code>) and open a new terminal window.</div>
              </div>
            </div>
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">2</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Run ngrok to get a public HTTPS URL</div>
                <div className="vr-guide-page-step-desc">Type this in the new terminal:</div>
                <div className="vr-guide-page-code">ngrok http 5173</div>
                <div className="vr-guide-page-step-desc">
                  You will see a line like:<br/>
                  <code>Forwarding &nbsp;&nbsp; https://abc123.ngrok-free.app → localhost:5173</code><br/><br/>
                  <strong>Copy that https:// link.</strong>
                </div>
              </div>
            </div>
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">3</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Put on your Oculus headset</div>
                <div className="vr-guide-page-step-desc">Make sure your Quest is connected to the same Wi-Fi as your computer.</div>
              </div>
            </div>
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">4</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Open Meta Quest Browser inside the headset</div>
                <div className="vr-guide-page-step-desc">
                  Press the <strong>Meta/Oculus button</strong> on your right controller →
                  tap <strong>App Library</strong> →
                  find and open <strong>Browser</strong>.
                </div>
              </div>
            </div>
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">5</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Type the ngrok URL in the Quest Browser</div>
                <div className="vr-guide-page-step-desc">
                  In the address bar inside the headset, type the <code>https://abc123.ngrok-free.app</code> URL.<br/>
                  If a ngrok warning appears — click <strong>"Visit Site"</strong>.
                </div>
              </div>
            </div>
            <div className="vr-guide-page-step">
              <div className="vr-guide-page-num">6</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Navigate to the molecule viewer</div>
                <div className="vr-guide-page-step-desc">
                  Use the app as on PC — Generate → Predict → <strong>Enable VR</strong>.
                </div>
              </div>
            </div>
            <div className="vr-guide-page-step vr-guide-page-step--final">
              <div className="vr-guide-page-num vr-guide-page-num--final">7</div>
              <div className="vr-guide-page-step-body">
                <div className="vr-guide-page-step-title">Click "Enter Immersive VR" inside the headset</div>
                <div className="vr-guide-page-step-desc">
                  The button will be active in the Quest Browser. Click it — the molecule fills your view in full 3D.<br/><br/>
                  🎮 <strong>Right trigger</strong> — select an atom<br/>
                  🤚 <strong>Hand tracking</strong> — Quest 2 / 3<br/>
                  🔄 <strong>Move your head</strong> — orbit the molecule
                </div>
              </div>
            </div>
          </div>
          <div className="vr-guide-page-reminder">
            ⚠ Keep both terminals running — <code>npm run dev</code> and <code>ngrok http 5173</code>. Closing ngrok stops the URL.
          </div>
          <div className="vr-guide-page-supported">
            Supported: Meta Quest 2 · Quest 3 · Quest Pro · Any WebXR-capable headset
          </div>
          <button className="vr-guide-page-back-btn" onClick={()=>{ setShowGuide(false); setVrKey(k=>k+1) }}>
            ← Back to Molecule Viewer
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="vr-page" key={vrKey}>
      <div className="vr-topbar">
        <button className="vr-back-btn" onClick={onClose}>← Back to Results</button>
        <div className="vr-topbar-center">
          <span className="vr-eyebrow">⬡ WebXR Molecular Viewer</span>
          <span className="vr-smiles-tag" title={smiles}>{smiles.length>42?smiles.slice(0,42)+'…':smiles}</span>
        </div>
        <div className="vr-topbar-right">
          {vrActive
            ? <button className="vr-xr-btn vr-xr-exit" onClick={exitVR}>⬡ Exit VR</button>
            : <button className="vr-xr-btn" onClick={vrSupported ? enterVR : ()=>setShowGuide(true)}>
                <span className="vr-xr-icon">◉</span>
                {vrSupported ? 'Enter Immersive VR' : 'Enter Immersive VR'}
              </button>
          }
        </div>
      </div>

      {vrError && <div className="vr-error-banner">⚠ {vrError}</div>}



      {/* Last-add info toast */}
      {lastAddInfo && (
        <div className="vr-add-toast">
          ✓ Added <strong style={{color:CPK_STR[lastAddInfo.elem]||'#fff'}}>{lastAddInfo.elem}</strong> bonded to
          <strong style={{color:CPK_STR[lastAddInfo.hostElem]||'#fff'}}> {lastAddInfo.hostElem}</strong>
          #{lastAddInfo.hostId+1} (single bond)
        </div>
      )}

      <div className="vr-layout">
        <div className="vr-canvas-wrap">
          <div ref={mountRef} className="vr-canvas"/>
          {isLoading && <div className="vr-loading"><div className="vr-loading-ring"/><span>Loading 3D Structure…</span></div>}
          <div className="vr-canvas-hint">Drag to rotate · Scroll to zoom · Click atom to select</div>
          <div className="vr-mode-toggle">
            {[{id:'ball-stick',label:'⬡ Ball & Stick'},{id:'spacefill',label:'● Spacefill'},{id:'stick',label:'— Stick'}].map(m=>(
              <button key={m.id} className={`vr-mode-btn${mode===m.id?' active':''}`} onClick={()=>setMode(m.id)}>{m.label}</button>
            ))}
          </div>
        </div>

        <div className="vr-panel">
          <div className="vr-panel-tabs">
            <button className={`vr-panel-tab${activeTab==='atoms'?' active':''}`} onClick={()=>setActiveTab('atoms')}>Atoms</button>
            <button className={`vr-panel-tab${activeTab==='library'?' active':''}`} onClick={()=>setActiveTab('library')}>＋ Add</button>
            <button className={`vr-panel-tab${activeTab==='predict'?' active':''}`}
              onClick={()=>setActiveTab('predict')}
              style={{position:'relative'}}>
              Predict
              {molChanged && <span className="vr-tab-dot"/>}
            </button>
          </div>

          {activeTab==='atoms' && (<>
            {selectedAtom ? (
              <div className="vr-atom-card">
                <div className="vr-atom-card-header">
                  <div className="vr-atom-elem-badge" style={{background:CPK_STR[selectedAtom.elem]||'#888',color:'#0d0618'}}>{selectedAtom.elem}</div>
                  <div>
                    <div className="vr-atom-card-title">
                      Atom #{selectedAtom.id+1}
                      {selectedAtom.added && <span className="vr-added-tag">Added</span>}
                    </div>
                    <div className="vr-atom-card-sub">
                      {selectedAtom.elem} · {freeValence(selectedAtom.id, selectedAtom.elem, bonds)} free slot{freeValence(selectedAtom.id,selectedAtom.elem,bonds)!==1?'s':''}
                    </div>
                  </div>
                  <button className="vr-atom-close" onClick={()=>setSelectedAtom(null)}>✕</button>
                </div>
                <div className="vr-atom-coords">
                  <div className="vr-coord-row"><span>X</span><span>{selectedAtom.x?.toFixed(3)}</span></div>
                  <div className="vr-coord-row"><span>Y</span><span>{selectedAtom.y?.toFixed(3)}</span></div>
                  <div className="vr-coord-row"><span>Z</span><span>{selectedAtom.z?.toFixed(3)}</span></div>
                </div>
                <div className="vr-atom-actions">
                  {atoms[selectedAtom.id]?.visible
                    ? <button className="vr-action-btn vr-action-hide" onClick={()=>hideAtom(selectedAtom.id)}>◌ Hide</button>
                    : <button className="vr-action-btn vr-action-show" onClick={()=>showAtom(selectedAtom.id)}>◉ Show</button>}
                  <button className="vr-action-btn vr-action-remove" onClick={()=>removeAtom(selectedAtom.id)}>✕ Remove</button>
                  {freeValence(selectedAtom.id, selectedAtom.elem, bonds) > 0 && (
                    <button className="vr-action-btn vr-action-bond"
                      onClick={()=>{setSelectedHostId(selectedAtom.id);setActiveTab('library')}}
                      title="Bond a new atom here">
                      ⬡ Bond here
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="vr-select-hint">
                <div className="vr-select-hint-icon">◎</div>
                <div>Click any atom in the viewer to select it</div>
              </div>
            )}

            <div className="vr-section-title">Atoms ({atoms.filter(a=>a.visible).length} / {atoms.length} visible)</div>
            <div className="vr-atom-list">
              {atoms.map((atom,idx)=>(
                <div key={atom.id}
                  className={`vr-atom-row${!atom.visible?' hidden':''}${selectedAtom?.id===atom.id?' selected':''}${atom.added?' added':''}`}
                  onClick={()=>setSelectedAtom({...atom})}>
                  <span className="vr-atom-dot" style={{background:CPK_STR[atom.elem]||'#888'}}/>
                  <span className="vr-atom-label">{atom.elem}</span>
                  <span className="vr-atom-idx">#{atom.id+1}</span>
                  {atom.added && <span className="vr-atom-added-dot" title="Manually added">✦</span>}
                  <span className="vr-atom-valence-tag"
                    style={{color: freeValence(idx,atom.elem,bonds)>0 ? '#6ee7b7' : '#475569'}}>
                    {freeValence(idx,atom.elem,bonds)}v
                  </span>
                  <button className="vr-toggle-btn"
                    onClick={e=>{e.stopPropagation();atom.visible?hideAtom(idx):showAtom(idx)}}
                    title={atom.visible?'Hide':'Show'}>
                    {atom.visible?'◌':'◉'}
                  </button>
                </div>
              ))}
            </div>

            <div className="vr-section-title">Composition</div>
            <div className="vr-composition">
              {Object.entries(atomCounts).map(([el,count])=>(
                <div key={el} className="vr-comp-row">
                  <span className="vr-comp-dot" style={{background:CPK_STR[el]||'#888'}}/>
                  <span className="vr-comp-el">{el}</span>
                  <span className="vr-comp-count">{count}</span>
                </div>
              ))}
            </div>

            {hiddenCount>0 && <button className="vr-restore-btn" onClick={restoreAll}>↺ Restore All ({hiddenCount} hidden)</button>}
            <div className="vr-instructions">
              <div className="vr-instructions-title">In VR Headset</div>
              <div className="vr-instructions-item">🎮 Right trigger — select an atom</div>
              <div className="vr-instructions-item">🤚 Hand tracking — Quest 2 / 3</div>
              <div className="vr-instructions-item">🔄 Move your head to orbit</div>
              <button className="vr-guide-inline-btn" onClick={()=>setShowGuide(true)}>
                ◉ How to connect Oculus — Step by step guide
              </button>
            </div>
          </>)}

          {activeTab==='library' && (
            <AtomLibrary
              onAdd={addAtomFromLibrary}
              addedLog={addedLog}
              openAtoms={openAtoms}
              selectedHostId={selectedHostId}
              onSelectHost={setSelectedHostId}
            />
          )}

          {/* ── PREDICT TAB ── */}
          {activeTab==='predict' && (
            <div className="vr-predict-tab">
              {/* Current molecule SMILES preview */}
              <div className="vr-predict-smiles-box">
                <div className="vr-predict-smiles-label">Current Molecule</div>
                <div className="vr-predict-smiles-val">
                  {getCurrentSmiles()}
                </div>
                <div className="vr-predict-formula">
                  Formula: <strong>{buildFormulaFromAtomsBonds(atoms, bonds)}</strong>
                </div>
                <div className="vr-predict-atom-count">
                  {atoms.filter(a=>a.visible).length} atoms · {bonds.filter(b=>atoms[b.a1]?.visible&&atoms[b.a2]?.visible).length} bonds
                </div>
              </div>

              {molChanged && predResult && (
                <div className="vr-predict-stale">⚠ Molecule changed — re-run prediction</div>
              )}

              <button
                className={`vr-predict-btn${predLoading?' loading':''}`}
                onClick={runPrediction}
                disabled={predLoading || atoms.filter(a=>a.visible).length===0}
              >
                {predLoading
                  ? <><span className="vr-pred-spinner"/>Predicting…</>
                  : <><span>◈</span> Predict Drug-Likeness</>}
              </button>

              {predError && (
                <div className="vr-predict-error">
                  <span>⚠</span> {predError}
                  <button className="vr-predict-retry" onClick={runPrediction}>Retry</button>
                </div>
              )}

              {predResult && !predLoading && (
                <div className="vr-predict-result">
                  {/* Score bar */}
                  <div className="vr-pred-score-wrap">
                    <div className="vr-pred-score-label">Drug-Likeness Score</div>
                    <div className="vr-pred-score-num"
                      style={{color: predResult.score>=0.7?'#6ee7b7':predResult.score>=0.4?'#d4af72':'#f87171'}}>
                      {predResult.score.toFixed(4)}
                    </div>
                    <div className="vr-pred-bar-track">
                      <div className="vr-pred-bar-fill"
                        style={{
                          width:`${Math.min(100,predResult.score*100)}%`,
                          background: predResult.score>=0.7?'#6ee7b7':predResult.score>=0.4?'#d4af72':'#f87171'
                        }}/>
                    </div>
                  </div>

                  {/* Cards row */}
                  <div className="vr-pred-cards">
                    <div className="vr-pred-card">
                      <div className="vr-pred-card-label">Confidence</div>
                      <div className="vr-pred-card-val"
                        style={{color: predResult.confidence==='high'?'#6ee7b7':predResult.confidence==='medium'?'#d4af72':'#94a3b8'}}>
                        {predResult.confidence}
                      </div>
                    </div>
                    <div className="vr-pred-card">
                      <div className="vr-pred-card-label">Promising</div>
                      <div className="vr-pred-card-val"
                        style={{color: predResult.is_promising?'#6ee7b7':'#f87171'}}>
                        {predResult.is_promising ? '✓ Yes' : '✗ No'}
                      </div>
                    </div>
                  </div>

                  {/* Verdict banner */}
                  <div className={`vr-pred-verdict ${predResult.is_promising?'good':'bad'}`}>
                    {predResult.is_promising
                      ? '✦ This molecule shows drug-like potential'
                      : '✗ Low drug-like potential — try modifying the structure'}
                  </div>

                  {/* Canonical SMILES */}
                  {predResult.repaired_smiles && (
                    <div className="vr-pred-canon">
                      <div className="vr-pred-canon-label">Canonical SMILES</div>
                      <div className="vr-pred-canon-val">{predResult.repaired_smiles}</div>
                    </div>
                  )}

                  {/* Comparison to original */}
                  <div className="vr-pred-compare">
                    <div className="vr-pred-compare-label">Original molecule</div>
                    <div className="vr-pred-compare-val">{smiles.length>44?smiles.slice(0,44)+'…':smiles}</div>
                  </div>
                </div>
              )}

              {!predResult && !predLoading && !predError && (
                <div className="vr-predict-hint">
                  Modify the molecule by adding or removing atoms, then click Predict to score the new structure using the quantum drug-likeness model.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}