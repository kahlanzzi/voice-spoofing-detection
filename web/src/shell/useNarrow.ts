// True at the drawer widths (HANDOFF line 85, prototype line 403: ≤ 860 px). A media-query change event, not a
// timer. Without matchMedia (jsdom) the wide layout is assumed.
import { useSyncExternalStore } from 'react'

export const NARROW_QUERY = '(max-width: 860px)'

function query(): MediaQueryList | null {
  return typeof window.matchMedia === 'function' ? window.matchMedia(NARROW_QUERY) : null
}

function subscribe(listener: () => void): () => void {
  const list = query()
  list?.addEventListener('change', listener)
  return () => list?.removeEventListener('change', listener)
}

export function useNarrow(): boolean {
  return useSyncExternalStore(subscribe, () => query()?.matches ?? false)
}
