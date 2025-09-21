import React, { useMemo, useState, Suspense, useEffect, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Line, useTexture } from '@react-three/drei'
import axios from 'axios'
import create from 'zustand'
import * as THREE from 'three'

type Vec3 = [number, number, number]
type Trajectory = {
  name: string
  states: { t: number, r: Vec3, v: Vec3 }[]
  color: string
  source?: string
  epochMs?: number
  noradId?: number
  category?: string
  shape?: 'sphere' | 'box' | 'cone' | 'tetra' | 'octa'
  line1?: string
  line2?: string
  elements?: {
    apogee_km: number
    perigee_km: number
    inclination_deg: number
    period_min: number
  }
  objectType?: 'Payload' | 'Debris' | 'Rocket Body' | 'Unknown'
}
type Store = { t: number; setT: (v: number | ((prev: number) => number)) => void }
const useStore = create<Store>((set) => ({
  t: 0,
  setT: (v) =>
    set(typeof v === 'function'
      ? (s) => ({ t: (v as (p: number) => number)(s.t) })
      : { t: v }),
}))

type Preset = { name: string; norad: number }
const PRESETS: Preset[] = [
  { name: 'ISS (ZARYA)', norad: 25544 },
  { name: 'Hubble Space Telescope', norad: 20580 },
  { name: 'NOAA-20', norad: 43013 },
  { name: 'Landsat 8', norad: 39084 },
  { name: 'Sentinel-1A', norad: 39634 },
  { name: 'TIANGONG', norad: 48274 },
]

const EARTH_RADIUS_KM = 6371
const SCALE = 1 / EARTH_RADIUS_KM

function categorizeObject(name: string, noradId?: number) {
  const n = (name || '').toLowerCase()
  if (n.includes('iss') || n.includes('zarya') || n.includes('tiangong') || n.includes('station')) {
    return { category: 'station', color: '#ff6b6b', shape: 'octa' as const }
  }
  if (n.includes('hubble') || n.includes('hst') || n.includes('telescope')) {
    return { category: 'telescope', color: '#ffd166', shape: 'cone' as const }
  }
  if (n.includes('noaa') || n.includes('meteo') || n.includes('weather')) {
    return { category: 'weather', color: '#06d6a0', shape: 'sphere' as const }
  }
  if (n.includes('landsat') || n.includes('sentinel') || n.includes('terra') || n.includes('aqua')) {
    return { category: 'earth_obs', color: '#4cc9f0', shape: 'sphere' as const }
  }
  if (n.includes('navstar') || n.includes('gps') || n.includes('glonass') || n.includes('galileo')) {
    return { category: 'navigation', color: '#b089f0', shape: 'box' as const }
  }
  if (n.includes('deb') || n.includes('debris')) {
    return { category: 'debris', color: '#f72585', shape: 'tetra' as const }
  }
  if (n.includes('rocket') || n.includes('rb') || n.includes('cz-') || n.includes('soyuz')) {
    return { category: 'rocket', color: '#ffa07a', shape: 'cone' as const }
  }
  return { category: 'unknown', color: '#cbd5e1', shape: 'sphere' as const }
}

class ErrorBoundary extends React.Component<{ fallback: React.ReactNode; children?: React.ReactNode }, { hasError: boolean }> {
  state = { hasError: false }
  static getDerivedStateFromError() { return { hasError: true } }
  componentDidCatch(err: any) { console.error('Earth textures failed:', err) }
  render() { return this.state.hasError ? this.props.fallback : this.props.children }
}

function BasicEarth() {
  return (
    <mesh>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial color="#2441A2" roughness={0.9} metalness={0.0} emissive="#0a1433" emissiveIntensity={0.25}/>
    </mesh>
  )
}

function TexturedEarth() {
  const base = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets'
  const earthTex = useTexture({
    map: `${base}/earth_atmos_2048.jpg`,
    normalMap: `${base}/earth_normal_2048.jpg`,
    specularMap: `${base}/earth_specular_2048.jpg`,
    emissiveMap: `${base}/earth_lights_2048.png`,
  })
  const cloudsMap = useTexture(`${base}/earth_clouds_2048.png`)

  if (earthTex.map) earthTex.map.colorSpace = THREE.SRGBColorSpace
  if (earthTex.emissiveMap) earthTex.emissiveMap.colorSpace = THREE.SRGBColorSpace

  return (
    <group>
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial
          map={earthTex.map}
          normalMap={earthTex.normalMap}
          specularMap={earthTex.specularMap}
          specular={new THREE.Color('#334455')}
          shininess={8}
          emissive={new THREE.Color('#ffffff')}
          emissiveMap={earthTex.emissiveMap}
          emissiveIntensity={0.25}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.003, 64, 64]} />
        <meshStandardMaterial
          alphaMap={cloudsMap}
          transparent
          opacity={0.9}
          depthWrite={false}
          color="#ffffff"
          roughness={1}
          metalness={0}
        />
      </mesh>
    </group>
  )
}

function Earth() {
  return (
    <ErrorBoundary fallback={<BasicEarth />}>
      <Suspense fallback={<BasicEarth />}>
        <TexturedEarth />
      </Suspense>
    </ErrorBoundary>
  )
}

// function OrbitPath({ traj, scale=SCALE }: { traj: Trajectory, scale?: number }) {
//   return null
// }

function FullOrbit({ traj, visible, scale=SCALE }: { traj: Trajectory, visible: boolean, scale?: number }) {
  const points = useMemo(() => traj.states.map(s => new THREE.Vector3(s.r[0]*scale, s.r[1]*scale, s.r[2]*scale)), [traj, scale])
  if (!visible) return null
  return <Line points={points} color={traj.color} lineWidth={0.5} transparent opacity={0.22} />
}

function Trail({
  traj, idx, trailMinutes, sampleSec=60, scale=SCALE
}: { traj: Trajectory, idx: number, trailMinutes: number, sampleSec?: number, scale?: number }) {
  const total = traj.states.length
  if (total < 2 || idx <= 0) return null
  const steps = Math.max(1, Math.round(trailMinutes * 60 / sampleSec))
  const start = Math.max(1, idx - steps + 1)
  const segs = []
  const n = idx - start + 1
  for (let i = start; i <= idx; i++) {
    const a = traj.states[i - 1].r, b = traj.states[i].r
    const p1 = new THREE.Vector3(a[0]*scale, a[1]*scale, a[2]*scale)
    const p2 = new THREE.Vector3(b[0]*scale, b[1]*scale, b[2]*scale)
    const f = (i - start + 1) / (n + 1)     // 0..1
    const opacity = Math.max(0.06, Math.pow(f, 1.5))  // fade in
    segs.push(<Line key={i} points={[p1, p2]} color={traj.color} lineWidth={1.6} transparent opacity={opacity} />)
  }
  return <>{segs}</>
}

function DecayGhost({
  traj, idx, trailMinutes, exaggeration=1.0, sampleSec=60, scale=SCALE
}: { traj: Trajectory, idx: number, trailMinutes: number, exaggeration?: number, sampleSec?: number, scale?: number }) {
  if (idx <= 0 || exaggeration <= 0) return null
  const r0 = new THREE.Vector3(...traj.states[0].r)
  const h0 = r0.length() - EARTH_RADIUS_KM
  if (h0 > 1200) return null
  const steps = Math.max(1, Math.round(trailMinutes * 60 / sampleSec))
  const start = Math.max(1, idx - steps + 1)
  const segs = []
  const k = exaggeration * 1e-6
  for (let i = start; i <= idx; i++) {
    const sA = traj.states[i - 1], sB = traj.states[i]
    const tMid = 0.5 * (sA.t + sB.t)
    const shrink = Math.max(0.97, 1.0 - Math.min(0.03, k * tMid))
    const a = new THREE.Vector3(sA.r[0], sA.r[1], sA.r[2]).multiplyScalar(shrink * scale)
    const b = new THREE.Vector3(sB.r[0], sB.r[1], sB.r[2]).multiplyScalar(shrink * scale)
    const f = (i - start + 1) / (idx - start + 2)
    const opacity = 0.12 * f
    segs.push(<Line key={`dg-${i}`} points={[a, b]} color="#ffffff" lineWidth={0.8} dashed dashSize={0.04} gapSize={0.02} transparent opacity={opacity} />)
  }
  return <>{segs}</>
}

function Marker({ traj, index, selected, onSelect, scale=SCALE }: {
  traj: Trajectory, index: number, selected: boolean, onSelect: (i:number)=>void, scale?: number
}) {
  const t = useStore(s=>s.t)
  const idx = Math.min(traj.states.length-1, Math.max(0, Math.round(t)))
  const p = traj.states[idx].r
  const category = traj.category || 'unknown'
  const shape = traj.shape || 'sphere'
  const baseSize =
    category === 'station' ? 0.03 :
    category === 'debris' ? 0.018 :
    category === 'rocket' ? 0.024 :
    0.022
  const size = selected ? baseSize * 1.4 : baseSize
  return (
    <mesh
      position={[p[0]*scale, p[1]*scale, p[2]*scale]}
      onClick={(e)=>{ e.stopPropagation(); onSelect(index) }}
    >
      {shape === 'sphere' && <sphereGeometry args={[size, 16, 16]} />}
      {shape === 'box' && <boxGeometry args={[size*1.4, size*1.0, size*1.0]} />}
      {shape === 'cone' && <coneGeometry args={[size*0.9, size*2.0, 12]} />}
      {shape === 'tetra' && <tetrahedronGeometry args={[size*1.4, 0]} />}
      {shape === 'octa' && <octahedronGeometry args={[size*1.2, 0]} />}
      <meshStandardMaterial
        color={traj.color}
        emissive={traj.color}
        emissiveIntensity={selected ? 1.0 : 0.65}
      />
    </mesh>
  )
}

function classifyObjectType(name: string): 'Payload' | 'Debris' | 'Rocket Body' | 'Unknown' {
  const n = (name || '').toUpperCase()
  if (/\b(DEB|DEBRIS)\b/.test(n)) return 'Debris'
  if (/\b(R\/B|ROCKET|BOOSTER|RB|CZ-|SL-|SOYUZ|FALCON|ATLAS|ARIANE)\b/.test(n)) return 'Rocket Body'
  return 'Payload'
}

function deriveElementsFromTLE(line1?: string, line2?: string) {
  try {
    if (!line2) return undefined
    const incDeg = parseFloat(line2.slice(8, 16))
    const eccStr = line2.slice(26, 33).trim()
    const nRevsPerDay = parseFloat(line2.slice(52, 63))
    const e = (eccStr ? parseInt(eccStr, 10) : 0) / 1e7
    if (!isFinite(incDeg) || !isFinite(nRevsPerDay)) return undefined

    const MU = 398600.4418
    const R = 6378.1363
    const nRadS = nRevsPerDay * 2 * Math.PI / 86400.0
    const a = Math.cbrt(MU / (nRadS * nRadS))
    const ra = a * (1 + e)
    const rp = a * (1 - e)
    const apogee_km = ra - R
    const perigee_km = rp - R
    const period_min = 1440.0 / nRevsPerDay

    return {
      apogee_km,
      perigee_km,
      inclination_deg: incDeg,
      period_min,
    }
  } catch {
    return undefined
  }
}

function parseTLEEpochMs(line1: string): number | undefined {
  try {
    const yy = parseInt(line1.slice(18, 20), 10)
    const dayOfYear = parseFloat(line1.slice(20, 32))
    const year = yy < 57 ? 2000 + yy : 1900 + yy
    const jan1 = Date.UTC(year, 0, 1, 0, 0, 0, 0)
    return jan1 + (dayOfYear - 1) * 86400 * 1000
  } catch {
    return undefined
  }
}

function fmtUTC(ms: number): string {
  const d = new Date(ms)
  const pad = (n: number, l = 2) => `${n}`.padStart(l, '0')
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`
}

function FollowTarget({
  enabled,
  selectedIndex,
  trajectories,
  controlsRef,
}: {
  enabled: boolean
  selectedIndex: number | null
  trajectories: Trajectory[]
  controlsRef: React.RefObject<any>
}) {
  const { camera } = useThree()
  const targetRef = useRef(new THREE.Vector3())
  const distanceRef = useRef<number | null>(null)

  useEffect(() => {
    if (!enabled || selectedIndex == null) return
    const tr = trajectories[selectedIndex]
    if (!tr) return
    const idx = Math.min(tr.states.length - 1, Math.max(0, Math.round(useStore.getState().t)))
    const p = tr.states[idx].r
    targetRef.current.set(p[0] * SCALE, p[1] * SCALE, p[2] * SCALE)
    const currentDist = camera.position.distanceTo(targetRef.current)
    const minD = controlsRef.current?.minDistance ?? 1.2
    const maxD = controlsRef.current?.maxDistance ?? 8
    distanceRef.current = Math.min(Math.max(currentDist, minD), maxD)
    if (controlsRef.current?.target) {
      controlsRef.current.target.copy(targetRef.current)
      controlsRef.current.update?.()
    } else {
      camera.lookAt(targetRef.current)
    }
  }, [enabled, selectedIndex])

  useFrame(() => {
    if (!enabled || selectedIndex == null) return
    const tr = trajectories[selectedIndex]
    if (!tr) return

    const idx = Math.min(tr.states.length - 1, Math.max(0, Math.round(useStore.getState().t)))
    const pr = tr.states[idx].r
    const pv = tr.states[idx].v

    targetRef.current.set(pr[0] * SCALE, pr[1] * SCALE, pr[2] * SCALE)
    const minD = controlsRef.current?.minDistance ?? 1.2
    const maxD = controlsRef.current?.maxDistance ?? 8
    const camTarget = controlsRef.current?.target ?? targetRef.current

    const currentDist = camera.position.distanceTo(camTarget)
    const prevDist = distanceRef.current ?? currentDist
    distanceRef.current = prevDist + 0.1 * (Math.min(Math.max(currentDist, minD), maxD) - prevDist)

    const rHat = targetRef.current.clone().normalize()
    const vVec = new THREE.Vector3(pv[0], pv[1], pv[2])
    let vHat = vVec.lengthSq() > 1e-10 ? vVec.clone().normalize() : new THREE.Vector3().crossVectors(rHat, new THREE.Vector3(0, 1, 0)).normalize()
    if (!isFinite(vHat.length())) vHat = new THREE.Vector3(1, 0, 0)

    const right = new THREE.Vector3().crossVectors(vHat, rHat).normalize()
    const up = new THREE.Vector3().crossVectors(right, vHat).normalize()

    const dir = new THREE.Vector3()
      .addScaledVector(vHat, -0.75)
      .addScaledVector(up, 0.65)
      .normalize()

    const dist = Math.min(Math.max(distanceRef.current ?? 1.6, minD), maxD)
    const desiredPos = targetRef.current.clone().add(dir.multiplyScalar(dist))

    camera.position.lerp(desiredPos, 0.2)
    if (controlsRef.current?.target) {
      controlsRef.current.target.lerp(targetRef.current, 0.3)
      controlsRef.current.update?.()
    } else {
      camera.lookAt(targetRef.current)
    }
  })
  return null
}

export default function App(){
  const [debrisIds, setDebrisIds] = useState<Set<number>>(new Set());

  async function addDebrisFromCatalog(name: string) {
    const { data } = await axios.get("/api/debris/catalog", { params: { name, limit: 20 } });
    const ids: number[] = (data.records ?? [])
      .map((r: any) => r?.norad_id)
      .filter((id: number | null) => typeof id === "number");

    setNorads(prev => {
      const cur = new Set(prev.split(",").map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n)));
      ids.forEach(id => cur.add(id));
      return Array.from(cur).join(", ");
    });

    setDebrisIds(prev => {
      const next = new Set(prev);
      ids.forEach(id => next.add(id));
      return next;
    });
  }
  const [norads, setNorads] = useState('25544')
  const [minutes, setMinutes] = useState(360)
  const [trajectories, setTrajectories] = useState<Trajectory[]>([])
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [follow, setFollow] = useState<boolean>(false)
  const [busy, setBusy] = useState(false)
  const [risk, setRisk] = useState<any[]>([])
  const [collapsed, setCollapsed] = useState(false)
  const [query, setQuery] = useState('')
  const matches = useMemo(()=> {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return PRESETS.filter(p=> p.name.toLowerCase().includes(q)).slice(0,6)
  }, [query])
  const addNorad = (id:number) => {
    setNorads(prev => {
      const ids = prev.split(',').map(s=>parseInt(s.trim())).filter(Boolean)
      if (!ids.includes(id)) ids.push(id)
      return ids.join(', ')
    })
    setQuery('')
  }

  const t = useStore(s=>s.t); const setT = useStore(s=>s.setT)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState<number>(100)
  const [trailMinutes, setTrailMinutes] = useState<number>(60)
  const [showFullOrbit, setShowFullOrbit] = useState<boolean>(false)
  const [decayExaggeration, setDecayExaggeration] = useState<number>(1.0)
  const rafLast = useRef<number | null>(null)

  const fetchAndSim = async () => {
    try {
      setBusy(true)
      const ids = norads.split(',').map(s=>parseInt(s.trim())).filter(Boolean)
      const tleRecords = await Promise.all(ids.map(async (id) => {
        const r = await axios.get(`/api/tle/${id}`)
        return r.data.records[0]
      }))

      const trajs: Trajectory[] = []
      for (const rec of tleRecords) {
        const r = await axios.post('/api/predict', { line1: rec.line1, line2: rec.line2, minutes })
        const meta = categorizeObject(rec.name, rec.norad_id)
        const epochMs = parseTLEEpochMs(rec.line1)
        const elements = deriveElementsFromTLE(rec.line1, rec.line2)
        const objectType = classifyObjectType(rec.name)
        const isDebris = debrisIds.has(rec.norad_id);
        trajs.push({
          name: rec.name,
          states: r.data.states,
          color: isDebris ? "#f72585" : meta.color,
          source: r.data.source,
          epochMs,
          noradId: rec.norad_id,
          category: isDebris ? "debris" : meta.category,
          shape: isDebris ? "tetra" : meta.shape,
          line1: rec.line1,
          line2: rec.line2,
          elements,
          objectType: isDebris ? "Debris" : objectType,
        });
      }
      setTrajectories(trajs)

      if (trajs.length >= 2) {
        const debris = { name: trajs[0].name, states: trajs[0].states }
        const targets = trajs.slice(1).map(t=>({ name: t.name, states: t.states }))
        const rr = await axios.post("/api/risk", { debris, targets, threshold_km: 6500 })
        setRisk(rr.data.approaches)
      } else {
        setRisk([])
      }
      setT(0)
      setPlaying(false)
      setSelectedIndex(null)
      setFollow(false)
    } catch (e:any) {
      console.error(e)
      alert('Fetch failed. Is the FastAPI server running on 8000?')
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (!playing) { rafLast.current = null; return }
    const len0 = trajectories[0]?.states?.length ?? 0
    if (len0 === 0) { setPlaying(false); return }

    let raf = 0
    const step = (ts: number) => {
      if (rafLast.current == null) rafLast.current = ts
      const dt = (ts - rafLast.current) / 1000
      rafLast.current = ts
      const sampleSec = 60
      setT(prev => {
        const maxIdx = Math.max(0, ((trajectories[0]?.states?.length ?? 1) - 1))
        if (maxIdx === 0) return 0
        let next = prev + (dt * speed) / sampleSec
        if (next >= maxIdx) {
          next = maxIdx
          setPlaying(false)
        }
        return next
      })
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, trajectories, setT])

  const currentTimeSec = trajectories.length > 0
    ? trajectories[0].states[Math.min(trajectories[0].states.length-1, Math.max(0, Math.round(t)))]?.t ?? 0
    : 0

  const simEpochMs = trajectories[0]?.epochMs
  const simTimeStr = simEpochMs != null ? fmtUTC(simEpochMs + currentTimeSec * 1000) : '—'
  const totalSamples = trajectories[0]?.states?.length ?? 0
  const maxIdx = Math.max(0, totalSamples - 1)
  const totalTimeSec = maxIdx * 60

  const controlsRef = useRef<any>(null)

  const selected = selectedIndex != null ? trajectories[selectedIndex] : null

  return (
    <div style={{height:'100%'}}>
      <div className={`sidebar ${collapsed? 'collapsed': ''}`}>
        <div className="sidebar-header">
          <h2 className="title" style={{marginTop:0, marginBottom:0}}>Orbital Debris (PINN)</h2>
          <button
            className="collapse-btn"
            onClick={()=>setCollapsed(c=>!c)}
            aria-label={collapsed ? 'Expand panel' : 'Collapse panel'}
            title={collapsed ? 'Expand' : 'Collapse'}
          >
            {collapsed ? '▶' : '◀'}
          </button>
        </div>

        {!collapsed && (
          <div className="hide-when-collapsed">
            <div className="metric">Backend: <code>http://127.0.0.1:8000</code></div>
            <label style={{marginTop:8}}>Search object</label>
            <input
              className="input"
              value={query}
              onChange={e=>setQuery(e.target.value)}
              placeholder='Try "Hubble" or "ISS"'
            />
            {matches.length>0 && (
              <ul className="search-list">
                {matches.map(m=>(
                  <li key={m.norad} onClick={()=>addNorad(m.norad)}>
                    <span>{m.name}</span>
                    <span className="pill">NORAD {m.norad}</span>
                  </li>
                ))}
              </ul>
            )}
            <div className="quick-row">
              {PRESETS.slice(0,4).map(p=>(
                <button key={p.norad} className="chip" onClick={()=>addNorad(p.norad)}>{p.name}</button>
              ))}
            </div>
            <div className="section">
              <div className="section-title">Debris Catalogs</div>
              <div className="chips">
                <button className="chip chip--debris" onClick={() => addDebrisFromCatalog("fengyun1c")}>FY-1C Debris</button>
                <button className="chip chip--debris" onClick={() => addDebrisFromCatalog("cosmos1408")}>COSMOS-1408 Debris</button>
                <button className="chip chip--debris" onClick={() => addDebrisFromCatalog("iridium33")}>IRIDIUM-33 Debris</button>
              </div>
            </div>

            <label style={{marginTop:8}}>NORAD IDs (comma-separated)</label>
            <input className="input" value={norads} onChange={e=>setNorads(e.target.value)} placeholder="25544, 43013, ..." />

            <label htmlFor="minutes-range">Minutes to simulate: {minutes}</label>
            <input
              id="minutes-range"
              className="slider"
              type="range"
              min="30"
              max="1440"
              value={minutes}
              onChange={e => setMinutes(parseInt(e.target.value))}
              title="Minutes to simulate"
              placeholder={`${minutes}`}
              aria-label={`Minutes to simulate, ${minutes}`}
            />

            <div style={{display:'flex', gap:8, alignItems:'center', flexWrap:'wrap', marginTop:10}}>
              <button className="btn" onClick={fetchAndSim} disabled={busy}>
                {busy? 'Processing...' : 'Fetch & Simulate'}
              </button>
              <button
                className="btn"
                onClick={()=> setPlaying(p => !p)}
                disabled={busy || trajectories.length === 0}
                aria-pressed={playing}
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <div style={{display:'flex', gap:6, alignItems:'center'}}>
                {[1,10,100,1000].map(s=>(
                  <button
                    key={s}
                    className="chip"
                    onClick={()=> setSpeed(s)}
                    aria-pressed={speed===s}
                    style={{ borderColor: speed===s ? 'var(--accent)' : 'var(--panel-border)' }}
                  >
                    {s}x
                  </button>
                ))}
              </div>
              {busy && <span className="inline-spinner" aria-hidden />}
              <span className="pill tval mono">t = {currentTimeSec.toFixed(0)} s</span>
              <span className="pill time mono">Sim time: {simTimeStr}</span>
            </div>
            <label htmlFor="scrub-range" style={{marginTop:8}}>
              Scrub time ({Math.floor(currentTimeSec/60)} / {Math.max(0, Math.floor(totalTimeSec/60))} min)
            </label>
            <input
              id="scrub-range"
              className="slider"
              type="range"
              min={0}
              max={maxIdx}
              step={1}
              value={Math.min(maxIdx, Math.max(0, Math.round(t)))}
              disabled={totalSamples === 0}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10)
                setPlaying(false)
                setT(idx)
              }}
              aria-label="Scrub simulated time"
              title="Drag to scrub through time"
            />

            <label htmlFor="trail-range">Trail length (minutes): {trailMinutes}</label>
            <input
              id="trail-range"
              className="slider"
              type="range"
              min="5"
              max="360"
              step="5"
              value={trailMinutes}
              onChange={e => setTrailMinutes(parseInt(e.target.value))}
              aria-label="Trail length in minutes"
            />
            <div className="section">
              <div className="section-title">Display</div>
              <div className="control-row">
                <label className="switch" title="Show faint full orbit">
                  <input
                    type="checkbox"
                    checked={showFullOrbit}
                    onChange={(e)=>setShowFullOrbit(e.target.checked)}
                    aria-label="Show faint full orbit"
                  />
                  <span className="toggle" />
                </label>
                <div className="control-label">Show faint full orbit</div>
              </div>
              <div className="control-row">
                <div className="control-label">Decay exaggeration</div>
                <input
                  type="range"
                  className="slider slider-compact"
                  min="0"
                  max="3"
                  step="0.5"
                  value={decayExaggeration}
                  onChange={(e)=>setDecayExaggeration(parseFloat(e.target.value))}
                  aria-label="Decay exaggeration"
                />
                <span className="pill">{decayExaggeration.toFixed(1)}x</span>
              </div>
            </div>
            {selected && (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 12,
                  border: '1px solid var(--panel-border)',
                  background: 'rgba(255,255,255,0.05)'
                }}
              >
                <div style={{display:'flex', alignItems:'center', gap:8, marginBottom:8}}>
                  <span className="dot" style={{background: selected.color}} />
                  <div style={{fontWeight:600}}>{selected.name}</div>
                  {selected.source && <span className="tag">{selected.source.toUpperCase()}</span>}
                </div>
                <div className="metric mono">
                  <div>NORAD: <b>{selected.noradId ?? '—'}</b></div>
                  <div>Type: <b>{selected.objectType ?? 'Unknown'}</b></div>
                </div>
                {selected.elements && (
                  <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:8, marginTop:8}}>
                    <span className="pill">Apogee {(selected.elements.apogee_km).toFixed(0)} km</span>
                    <span className="pill">Perigee {(selected.elements.perigee_km).toFixed(0)} km</span>
                    <span className="pill">Incl {selected.elements.inclination_deg.toFixed(2)}°</span>
                    <span className="pill">Period {selected.elements.period_min.toFixed(1)} min</span>
                  </div>
                )}
                <div style={{display:'flex', gap:8, marginTop:10}}>
                  <button
                    className="btn"
                    onClick={()=> setFollow(f => !f)}
                    aria-pressed={follow}
                    title="Lock camera on this object"
                  >
                    {follow ? 'Unfollow' : 'Follow Object'}
                  </button>
                  <button
                    className="chip"
                    onClick={()=> { setSelectedIndex(null); setFollow(false) }}
                    title="Clear selection"
                  >
                    Deselect
                  </button>
                </div>
              </div>
            )}

            {trajectories.length>0 && (
              <div className="legend">
                <b>Trajectories</b>
                <ul>
                  {trajectories.map((tr, i)=>(
                    <li
                      key={tr.name}
                      onClick={()=>{ setSelectedIndex(i); }}
                      style={{cursor:'pointer', outline: selectedIndex===i ? '1px solid var(--accent)' : 'none'}}
                      title="Select to view details"
                    >
                      <span className="dot" style={{background: tr.color}} />
                      <span>{tr.name}</span>
                      {tr.source && <span className="tag">{tr.source.toUpperCase()}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {risk.length>0 && (
              <div style={{marginTop:12}}>
                <b>Potential Close Approaches</b>
                <ul>
                  {risk.map((r:any, i:number)=>(
                    <li key={i}>
                      <span className="pill">risk {(r.risk_score*100).toFixed(1)}%</span>
                      <span className="pill">dmin {r.min_distance_km.toFixed(2)} km</span>
                      with <b>{r.target}</b> at t={r.timestamp_min.toFixed(0)} s
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="metric" style={{marginTop:8}}>Tip: Add 2–4 IDs (e.g., <code>25544, 43013, 33591</code>)</div>
            <div className="metric" style={{marginTop:8}}>Uses NASA Open APIs (space weather) + CelesTrak TLEs.</div>
          </div>
        )}
      </div>
      {busy && (
        <div className="loading-overlay">
          <div className="spinner" />
          <div className="loading-text">Running PINN...</div>
        </div>
      )}

      <Canvas
        className="r3f-canvas"
        style={{ position: 'fixed', inset: 0, zIndex: 0 }}
        camera={{ position: [0, 0, 3] }}
        dpr={[1, 2]}
        onPointerMissed={(e)=>{ if (e.type === 'click') { setSelectedIndex(null); setFollow(false) } }}
      >
        <ambientLight intensity={0.9} />
        <hemisphereLight args={['#9ecbff', '#223', 0.6]} />
        <pointLight position={[10, 10, 10]} intensity={1.2} />
        <Earth />
        <Stars radius={100} depth={50} factor={2} />
        {trajectories.map((tr)=> <FullOrbit key={tr.name+'-full'} traj={tr} visible={showFullOrbit} />)}
        {trajectories.map((tr, i) => (
          <group key={tr.name+'-trail'}>
            <Trail
  traj={tr}
  idx={Math.min(tr.states.length-1, Math.max(0, Math.round(t)))}
  trailMinutes={trailMinutes}
/>
{tr.category === "debris" && (
  <Line
    points={tr.states.map(s => new THREE.Vector3(s.r[0]*SCALE, s.r[1]*SCALE, s.r[2]*SCALE))}
    color={tr.color}
    lineWidth={1}
    dashed
    dashSize={0.2}
    gapSize={0.2}
    transparent
    opacity={0.6}
/>
)}

            <DecayGhost traj={tr} idx={Math.min(tr.states.length-1, Math.max(0, Math.round(t)))} trailMinutes={trailMinutes} exaggeration={decayExaggeration} />
          </group>
        ))}
        {trajectories.map((tr, i)=> (
          <Marker
            key={tr.name+'-m'}
            traj={tr}
            index={i}
            selected={selectedIndex===i}
            onSelect={(idx)=> { setSelectedIndex(idx) }}
          />
        ))}
        <FollowTarget
          enabled={follow}
          selectedIndex={selectedIndex}
          trajectories={trajectories}
          controlsRef={controlsRef}
        />

        <OrbitControls
          ref={controlsRef}
          enablePan
          enableDamping
          dampingFactor={0.08}
          minDistance={1.2}
          maxDistance={8}
        />
      </Canvas>
    </div>
  )
}
