// The session state for React (Stage 18 PR 3c): the store's own subscribe and getSnapshot, no state of its own.
import { useSyncExternalStore } from 'react'
import { getSnapshot, subscribe, type SessionState } from './session'

export function useSession(): SessionState {
  return useSyncExternalStore(subscribe, getSnapshot)
}
