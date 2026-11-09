// The analysis for React (Stage 18 PR B): the store's own subscribe and getSnapshot, no state of its own.
import { useSyncExternalStore } from 'react'
import { getSnapshot, subscribe, type AnalysisState } from './analysis'

export function useAnalysis(): AnalysisState {
  return useSyncExternalStore(subscribe, getSnapshot)
}
