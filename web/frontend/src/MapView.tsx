import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import type { SceneInfo, Tree } from './types'

interface Props {
  token: string
  scene: SceneInfo
  overlayUrl: string | null
  opacity: number
  trees: Tree[]
  placing: boolean
  onPlace: (lon: number, lat: number) => void
  onMoveTree: (id: string, lon: number, lat: number) => void
  onRemoveTree: (id: string) => void
  onOverlayError: (message: string | null) => void
}

const SOURCE = 'solweig-overlay'
const LAYER = 'solweig-overlay-layer'
const MAX_CACHED_OVERLAYS = 150

// Download an overlay PNG ourselves (with retries) instead of letting Mapbox fetch
// it: Mapbox keeps showing the previous image when a load fails and only logs the
// error, which looks exactly like "the layer did not change".
async function fetchOverlay(url: string): Promise<Blob> {
  let last = ''
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    try {
      const response = await fetch(url)
      if (response.ok) return await response.blob()
      last = `HTTP ${response.status}`
    } catch (e) {
      last = (e as Error).message
    }
    await new Promise((resolve) => window.setTimeout(resolve, 750 * attempt))
  }
  throw new Error(`${url}: ${last}`)
}

export default function MapView({ token, scene, overlayUrl, opacity, trees, placing, onPlace, onMoveTree, onRemoveTree, onOverlayError }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const markers = useRef<Map<string, mapboxgl.Marker>>(new Map())
  const loaded = useRef(false)
  // Overlay URL -> object URL of the downloaded PNG (insertion order = age).
  const overlayCache = useRef<Map<string, string>>(new Map())
  const callbacks = useRef({ onPlace, onMoveTree, onRemoveTree, onOverlayError, placing })
  callbacks.current = { onPlace, onMoveTree, onRemoveTree, onOverlayError, placing }

  // Create the map once.
  useEffect(() => {
    if (!container.current || map.current) return
    mapboxgl.accessToken = token
    const m = new mapboxgl.Map({
      container: container.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: scene.center,
      zoom: 15.2,
      attributionControl: true,
    })
    m.addControl(new mapboxgl.NavigationControl(), 'top-right')
    m.on('load', () => {
      // Scene outline so users know where trees can be placed.
      m.addSource('scene-extent', {
        type: 'geojson',
        data: { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [[...scene.corners, scene.corners[0]]] } },
      })
      m.addLayer({ id: 'scene-extent-line', type: 'line', source: 'scene-extent', paint: { 'line-color': '#1f4e79', 'line-width': 2, 'line-dasharray': [2, 2] } })
      loaded.current = true
      m.fire('solweig-ready')
    })
    m.on('error', (e) => {
      if ((e as { sourceId?: string }).sourceId === SOURCE) callbacks.current.onOverlayError(`overlay: ${e.error?.message ?? 'failed to draw'}`)
    })
    m.on('click', (e) => {
      if (!callbacks.current.placing) return
      callbacks.current.onPlace(e.lngLat.lng, e.lngLat.lat)
    })
    map.current = m
    const cache = overlayCache.current
    return () => {
      m.remove()
      map.current = null
      loaded.current = false
      for (const objectUrl of cache.values()) URL.revokeObjectURL(objectUrl)
      cache.clear()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  // Cursor feedback for placement mode.
  useEffect(() => {
    const m = map.current
    if (m) m.getCanvas().style.cursor = placing ? 'crosshair' : ''
  }, [placing])

  // Swap the overlay image whenever the URL changes. The PNG is downloaded first
  // and handed to Mapbox as an object URL, so a failed download is reported and a
  // slow one can never overwrite a newer choice.
  useEffect(() => {
    const m = map.current
    if (!m) return
    let cancelled = false
    const show = (src: string | null) => {
      if (cancelled) return
      const existing = m.getSource(SOURCE) as mapboxgl.ImageSource | undefined
      if (!src) {
        if (m.getLayer(LAYER)) m.setLayoutProperty(LAYER, 'visibility', 'none')
        return
      }
      if (existing) {
        existing.updateImage({ url: src, coordinates: scene.corners })
        m.setLayoutProperty(LAYER, 'visibility', 'visible')
      } else {
        m.addSource(SOURCE, { type: 'image', url: src, coordinates: scene.corners })
        m.addLayer(
          { id: LAYER, type: 'raster', source: SOURCE, paint: { 'raster-opacity': opacity, 'raster-fade-duration': 0, 'raster-resampling': 'nearest' } },
          'scene-extent-line',
        )
      }
    }
    const whenReady = (src: string | null) => {
      if (loaded.current) show(src)
      else m.once('solweig-ready', () => show(src))
    }
    if (!overlayUrl) {
      whenReady(null)
      return
    }
    const cache = overlayCache.current
    const cached = cache.get(overlayUrl)
    if (cached) {
      callbacks.current.onOverlayError(null)
      whenReady(cached)
    } else {
      fetchOverlay(overlayUrl)
        .then((blob) => {
          const objectUrl = URL.createObjectURL(blob)
          cache.set(overlayUrl, objectUrl)
          while (cache.size > MAX_CACHED_OVERLAYS) {
            const [oldest, oldUrl] = cache.entries().next().value as [string, string]
            cache.delete(oldest)
            URL.revokeObjectURL(oldUrl)
          }
          if (cancelled) return
          callbacks.current.onOverlayError(null)
          whenReady(objectUrl)
        })
        .catch((e: Error) => {
          if (!cancelled) callbacks.current.onOverlayError(`could not load overlay ${e.message}`)
        })
    }
    return () => {
      cancelled = true
    }
  }, [overlayUrl, scene.corners]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const m = map.current
    if (m && m.getLayer(LAYER)) m.setPaintProperty(LAYER, 'raster-opacity', opacity)
  }, [opacity])

  // Keep one draggable marker per tree.
  useEffect(() => {
    const m = map.current
    if (!m) return
    const seen = new Set<string>()
    for (const tree of trees) {
      seen.add(tree.id)
      let marker = markers.current.get(tree.id)
      if (!marker) {
        const el = document.createElement('div')
        el.className = 'tree-marker'
        el.title = 'Drag to move, double-click to remove'
        el.addEventListener('dblclick', (ev) => {
          ev.stopPropagation()
          callbacks.current.onRemoveTree(tree.id)
        })
        marker = new mapboxgl.Marker({ element: el, draggable: true, anchor: 'center' }).setLngLat([tree.lon, tree.lat]).addTo(m)
        marker.on('dragend', () => {
          const { lng, lat } = marker!.getLngLat()
          callbacks.current.onMoveTree(tree.id, lng, lat)
        })
        markers.current.set(tree.id, marker)
      } else {
        const current = marker.getLngLat()
        if (current.lng !== tree.lon || current.lat !== tree.lat) marker.setLngLat([tree.lon, tree.lat])
      }
      const size = Math.max(14, Math.round(tree.crown_radius * 4))
      const el = marker.getElement()
      el.style.width = `${size}px`
      el.style.height = `${size}px`
    }
    for (const [id, marker] of markers.current) {
      if (!seen.has(id)) {
        marker.remove()
        markers.current.delete(id)
      }
    }
  }, [trees])

  return <div ref={container} className="map" />
}
