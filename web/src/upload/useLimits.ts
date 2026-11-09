// The limits for React (Stage 18 PR A): the store's own subscribe and getSnapshot, no state of its own.
import { useSyncExternalStore } from 'react'
import { getSnapshot, subscribe, type LimitsState } from './limits'

export function useLimits(): LimitsState {
  return useSyncExternalStore(subscribe, getSnapshot)
}
