// The selected recording for React (Stage 18 PR A): the store's own subscribe and getSnapshot, no state of its own.
import { useSyncExternalStore } from 'react'
import { getSnapshot, subscribe } from './fileDraft'

export function useFileDraft(): File | null {
  return useSyncExternalStore(subscribe, getSnapshot)
}
