import type { JobStatus, ResultSummary, SceneInfo, Tree } from './types'

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  config: () => json<{ mapbox_token: string }>('/api/config'),
  scene: () => json<SceneInfo>('/api/scene'),
  baseline: () => json<ResultSummary>('/api/baseline'),
  submit: (trees: Tree[]) =>
    json<JobStatus>('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trees: trees.map(({ lon, lat, height, crown_radius }) => ({ lon, lat, height, crown_radius })),
      }),
    }),
  job: (id: string) => json<JobStatus>(`/api/jobs/${id}`),
  result: (id: string) => json<ResultSummary>(`/api/jobs/${id}/result`),
}

export function overlayUrl(base: string, variable: string, hour: number, diff: boolean): string {
  const name = diff ? `${variable}_diff_${String(hour).padStart(2, '0')}.png` : `${variable}_${String(hour).padStart(2, '0')}.png`
  return `${base}${name}`
}
