export type Variable = 'tmrt' | 'utci'
export type Layer = 'baseline' | 'scenario' | 'diff'

export interface Tree {
  id: string
  lon: number
  lat: number
  height: number
  crown_radius: number
}

export interface SceneInfo {
  rows: number
  cols: number
  pixel_size_m: number
  corners: [[number, number], [number, number], [number, number], [number, number]]
  center: [number, number]
  date: string
  variables: Record<Variable, { label: string; vmin: number; vmax: number }>
  diff_range: number
  change_threshold_c: number
  baseline_ready: boolean
  baseline_url: string
  baseline_model_seconds: number | null
  limits: { max_trees: number; height: [number, number]; crown_radius: [number, number] }
}

export interface HourlyStats {
  mean: number[]
  min: number[]
  max: number[]
}

export interface ResultSummary {
  hours: string[]
  variables: Record<
    Variable,
    {
      stats: HourlyStats
      baseline_stats?: HourlyStats
      diff_stats?: HourlyStats
      changed_pixels?: number[]
    }
  >
}

export interface JobStatus {
  id: string
  trees?: { lon: number; lat: number; height: number; crown_radius: number }[]
  status: 'queued' | 'running' | 'done' | 'failed'
  phase: string
  elapsed_seconds: number
  expected_seconds: number | null
  timesteps_done: number
  timesteps_total: number
  changed_pixels: number
  error: string | null
  result_url: string | null
  queue_position: number
}
