// The status for React (Stage 18 PR 7a): the store's own subscribe and getSnapshot, no state of its own.
import { useSyncExternalStore } from 'react'
import { getSnapshot, subscribe, type StatusState } from './status'

export function useStatus(): StatusState {
  return useSyncExternalStore(subscribe, getSnapshot)
}
