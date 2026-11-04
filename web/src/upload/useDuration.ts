// The duration read for React (Stage 18 PR A). Only a file that can be audio is read (prototype line 1287). The state
// remembers which file it belongs to, so another file reads as 'reading' with no reset effect. A remount reads again:
// it is cheap, and the draft store stays the File alone.
import { useEffect, useState } from 'react'
import type { LimitsResponse } from '../api/types'
import { canBeAudio, type DurationInput } from './checks'
import { readDuration, type DurationRead } from './readDuration'

export function useDuration(file: File, limits: Pick<LimitsResponse, 'formats'>): DurationInput {
  const [read, setRead] = useState<{ file: File; duration: DurationRead } | null>(null)
  const readable = canBeAudio(file, limits)

  useEffect(() => {
    if (!readable) return
    return readDuration(file, (duration) => setRead({ file, duration }))
  }, [file, readable])

  if (!readable) return 'unreadable'
  return read !== null && read.file === file ? read.duration : 'reading'
}
