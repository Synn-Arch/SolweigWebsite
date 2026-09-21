import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import MapView from './MapView'
import { api, overlayUrl } from './api'
import type { JobStatus, Layer, ResultSummary, SceneInfo, Tree, Variable } from './types'

const HOURS = Array.from({ length: 24 }, (_, i) => i)

function fmt(v: number | undefined | null, digits = 1): string {
  return v == null || Number.isNaN(v) ? '–' : v.toFixed(digits)
}

export default function App() {
  const [token, setToken] = useState<string | null>(null)
  const [scene, setScene] = useState<SceneInfo | null>(null)
  const [baseline, setBaseline] = useState<ResultSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [variable, setVariable] = useState<Variable>('tmrt')
  const [layer, setLayer] = useState<Layer>('baseline')
  const [hour, setHour] = useState(14)
  const [opacity, setOpacity] = useState(0.65)
  const [playing, setPlaying] = useState(false)

  const [trees, setTrees] = useState<Tree[]>([])
  const [placing, setPlacing] = useState(false)
  const [height, setHeight] = useState(12)
  const [radius, setRadius] = useState(4)

  const [job, setJob] = useState<JobStatus | null>(null)
  const [scenario, setScenario] = useState<{ id: string; summary: ResultSummary; trees: Tree[] } | null>(null)
  const pollTimer = useRef<number | null>(null)

  // Initial load.
  useEffect(() => {
    Promise.all([api.config(), api.scene()])
      .then(async ([cfg, sc]) => {
        setToken(cfg.mapbox_token)
        setScene(sc)
        if (sc.baseline_ready) setBaseline(await api.baseline())
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  // Hour animation.
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => setHour((h) => (h + 1) % 24), 700)
    return () => window.clearInterval(id)
  }, [playing])

  // Poll a running job.
  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'failed') return
    const tick = async () => {
      try {
        const status = await api.job(job.id)
        setJob(status)
        if (status.status === 'done') {
          const summary = await api.result(status.id)
          setScenario({ id: status.id, summary, trees })
          setLayer('diff')
        }
      } catch (e) {
        setError((e as Error).message)
      }
    }
    pollTimer.current = window.setTimeout(tick, 2000)
    return () => {
      if (pollTimer.current) window.clearTimeout(pollTimer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job])

  const addTree = useCallback(
    (lon: number, lat: number) => {
      setTrees((list) => {
        if (scene && list.length >= scene.limits.max_trees) return list
        return [...list, { id: Math.random().toString(36).slice(2, 10), lon, lat, height, crown_radius: radius }]
      })
    },
    [height, radius, scene],
  )
  const moveTree = useCallback((id: string, lon: number, lat: number) => {
    setTrees((list) => list.map((t) => (t.id === id ? { ...t, lon, lat } : t)))
  }, [])
  const removeTree = useCallback((id: string) => setTrees((list) => list.filter((t) => t.id !== id)), [])
  const updateTree = (id: string, patch: Partial<Tree>) => setTrees((list) => list.map((t) => (t.id === id ? { ...t, ...patch } : t)))

  const run = async () => {
    setError(null)
    try {
      const status = await api.submit(trees)
      setJob(status)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const busy = job !== null && (job.status === 'queued' || job.status === 'running')
  const scenarioAvailable = scenario !== null
  const effectiveLayer: Layer = layer !== 'baseline' && !scenarioAvailable ? 'baseline' : layer

  const overlay = useMemo(() => {
    if (!scene || !scene.baseline_ready) return null
    if (effectiveLayer === 'baseline') return overlayUrl(scene.baseline_url, variable, hour, false)
    if (!scenario) return null
    return overlayUrl(`/results/jobs/${scenario.id}/`, variable, hour, effectiveLayer === 'diff')
  }, [scene, effectiveLayer, variable, hour, scenario])

  const stats = useMemo(() => {
    const base = baseline?.variables[variable].stats
    const scen = scenario?.summary.variables[variable]
    return {
      baseMean: base?.mean[hour],
      scenMean: scen?.stats.mean[hour],
      diffMin: scen?.diff_stats?.min[hour],
      diffMean: scen?.diff_stats?.mean[hour],
      changed: scen?.changed_pixels?.[hour],
    }
  }, [baseline, scenario, variable, hour])

  const progress = useMemo(() => {
    if (!job) return 0
    if (job.status === 'done') return 1
    if (job.phase === 'simulating') return 0.45 + 0.5 * (job.timesteps_done / job.timesteps_total)
    if (job.phase === 'rendering') return 0.97
    if (job.expected_seconds) return Math.min(0.42, (job.elapsed_seconds / job.expected_seconds) * 0.9)
    return 0.05
  }, [job])

  if (error && !scene) return <div className="fullscreen-message">Failed to load: {error}</div>
  if (!scene || token === null) return <div className="fullscreen-message">Loading scene…</div>
  if (!token) return <div className="fullscreen-message">Server has no Mapbox token. Set the MAPBOX_TOKEN secret and restart.</div>

  const varSpec = scene.variables[variable]
  const hourLabel = baseline?.hours[hour] ?? `${String(hour).padStart(2, '0')}:00`

  return (
    <div className="app">
      <MapView
        token={token}
        scene={scene}
        overlayUrl={overlay}
        opacity={opacity}
        trees={trees}
        placing={placing}
        onPlace={addTree}
        onMoveTree={moveTree}
        onRemoveTree={removeTree}
      />

      <aside className="panel">
        <header>
          <h1>SOLWEIG tree scenarios</h1>
          <p className="muted">
            {scene.cols}×{scene.rows} px at {scene.pixel_size_m} m · {scene.date}
            {!scene.baseline_ready && ' · baseline not computed yet'}
          </p>
        </header>

        <section>
          <h2>Overlay</h2>
          <div className="row">
            <label>
              Variable
              <select value={variable} onChange={(e) => setVariable(e.target.value as Variable)}>
                <option value="tmrt">TMRT</option>
                <option value="utci">UTCI</option>
              </select>
            </label>
            <label>
              Layer
              <select value={effectiveLayer} onChange={(e) => setLayer(e.target.value as Layer)}>
                <option value="baseline">Baseline</option>
                <option value="scenario" disabled={!scenarioAvailable}>
                  Scenario
                </option>
                <option value="diff" disabled={!scenarioAvailable}>
                  Difference
                </option>
              </select>
            </label>
          </div>
          <label>
            Opacity {Math.round(opacity * 100)}%
            <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          </label>
          <label>
            Hour · {hourLabel.replace('T', ' ')}
            <div className="row">
              <button className="icon" onClick={() => setPlaying((p) => !p)} title={playing ? 'Pause' : 'Play'}>
                {playing ? '❚❚' : '▶'}
              </button>
              <input type="range" min={0} max={23} step={1} value={hour} onChange={(e) => setHour(Number(e.target.value))} list="hours" />
            </div>
            <datalist id="hours">
              {HOURS.map((h) => (
                <option key={h} value={h} />
              ))}
            </datalist>
          </label>
          <Legend
            title={effectiveLayer === 'diff' ? `Δ ${varSpec.label} (scenario − baseline)` : varSpec.label}
            diverging={effectiveLayer === 'diff'}
            min={effectiveLayer === 'diff' ? -scene.diff_range : varSpec.vmin}
            max={effectiveLayer === 'diff' ? scene.diff_range : varSpec.vmax}
          />
          <dl className="stats">
            <dt>Baseline mean</dt>
            <dd>{fmt(stats.baseMean)} °C</dd>
            {scenarioAvailable && (
              <>
                <dt>Scenario mean</dt>
                <dd>{fmt(stats.scenMean)} °C</dd>
                <dt>Largest cooling</dt>
                <dd>{fmt(stats.diffMin)} °C</dd>
                <dt>Pixels changed (&gt;{scene.change_threshold_c} °C)</dt>
                <dd>{stats.changed ?? '–'}</dd>
              </>
            )}
          </dl>
        </section>

        <section>
          <h2>Trees ({trees.length})</h2>
          <div className="row">
            <label>
              Height (m)
              <input type="number" min={scene.limits.height[0]} max={scene.limits.height[1]} step={1} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
            </label>
            <label>
              Crown radius (m)
              <input type="number" min={scene.limits.crown_radius[0]} max={scene.limits.crown_radius[1]} step={0.5} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
            </label>
          </div>
          <div className="row">
            <button className={placing ? 'primary' : ''} onClick={() => setPlacing((p) => !p)} disabled={busy}>
              {placing ? 'Click map to place · done' : 'Add trees'}
            </button>
            <button onClick={() => setTrees([])} disabled={busy || trees.length === 0}>
              Clear
            </button>
          </div>
          {trees.length > 0 && (
            <ul className="tree-list">
              {trees.map((t, i) => (
                <li key={t.id}>
                  <span>#{i + 1}</span>
                  <input type="number" value={t.height} min={scene.limits.height[0]} max={scene.limits.height[1]} step={1} onChange={(e) => updateTree(t.id, { height: Number(e.target.value) })} title="Height (m)" />
                  <input type="number" value={t.crown_radius} min={scene.limits.crown_radius[0]} max={scene.limits.crown_radius[1]} step={0.5} onChange={(e) => updateTree(t.id, { crown_radius: Number(e.target.value) })} title="Crown radius (m)" />
                  <button className="icon" onClick={() => removeTree(t.id)} title="Remove">
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <p className="muted small">Drag a marker to move it, double-click to remove.</p>
        </section>

        <section>
          <button className="primary wide" onClick={run} disabled={busy || trees.length === 0 || !scene.baseline_ready}>
            {busy ? 'Running…' : 'Run analysis'}
          </button>
          {job && (
            <div className="job">
              <div className="progress">
                <div style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <p className="small">
                {job.status === 'queued' && `Queued (position ${job.queue_position + 1})`}
                {job.status === 'running' && `${job.phase} · ${Math.round(job.elapsed_seconds)}s`}
                {job.status === 'running' && job.phase === 'simulating' && ` · ${job.timesteps_done}/${job.timesteps_total} h`}
                {job.status === 'running' && job.expected_seconds && ` · ~${Math.round(job.expected_seconds / 60)} min expected`}
                {job.status === 'done' && `Done in ${Math.round(job.elapsed_seconds)}s · ${job.changed_pixels} canopy pixels changed`}
                {job.status === 'failed' && `Failed: ${job.error}`}
              </p>
            </div>
          )}
          {error && <p className="error small">{error}</p>}
          <p className="muted small">
            Each run recomputes sky view factors and 24 hourly timesteps for the whole scene
            {scene.baseline_model_seconds ? ` (baseline took ${Math.round(scene.baseline_model_seconds / 60)} min)` : ''}.
          </p>
        </section>
      </aside>
    </div>
  )
}

function Legend({ title, diverging, min, max }: { title: string; diverging: boolean; min: number; max: number }) {
  return (
    <div className="legend">
      <div className={`legend-bar ${diverging ? 'diverging' : 'sequential'}`} />
      <div className="legend-labels">
        <span>{min}</span>
        <span>{title}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}
