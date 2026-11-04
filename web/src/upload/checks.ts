// The pre-upload checks (Stage 18 PR 8b; prototype lines 898-913, HANDOFF line 177): pure, with no I/O and no Audio
// element (the duration read is readDuration.ts). Limits come from GET /api/limits in its own units, bytes and seconds
// (docs/api_design.md §4.2); nothing here is a guessed value. The MIME type is not an input: the server ignores it
// (§4.3).
//
// The client never accepts a file the server refuses for the same reason (api/intake.py). So there are two
// departures from the prototype: the extension follows the server (fileExtension), and a duration of 0 fails as
// empty, since the server answers empty_file for 0 frames (Entry 143 Q13). Anything else the browser cannot know,
// such as corrupt audio, stays with the server.
import type { LimitsResponse } from '../api/types'

const BYTES_PER_MB = 1048576 // prototype line 906

const EMPTY_TEXT = 'Empty file: it contains no data. Export the segment again and choose the new file.'

// 'reading': the read has not finished. 'unreadable': the browser could not tell; the service will check it.
export type DurationInput = number | 'reading' | 'unreadable'

// ok null: neither passed nor failed. wait: the check has not finished, so the file cannot be sent yet.
export interface FileCheck {
  readonly id: 'format' | 'size' | 'duration'
  readonly ok: boolean | null
  readonly wait: boolean
  readonly text: string
}

export interface CheckResult {
  readonly checks: readonly FileCheck[]
  readonly valid: boolean
}

type CheckLimits = Pick<LimitsResponse, 'formats' | 'max_size_bytes' | 'max_duration_s'>

// PurePath(name).suffix.lower()[1:] (api/intake.py:149), not the prototype's split('.').pop() (line 900), which
// accepts files named "wav" and ".wav" that the server refuses. "..wav" is refused too: Python 3.13 and 3.14 differ
// on it, and the stricter answer is right on both (Entry 143 Q12).
export function fileExtension(name: string): string {
  const stem = name.replace(/^\.+/, '')
  const dot = stem.lastIndexOf('.')
  return dot < 0 ? '' : stem.slice(dot + 1).toLowerCase()
}

// Prototype line 656.
export function formatSize(bytes: number): string {
  if (bytes >= BYTES_PER_MB) return `${(bytes / BYTES_PER_MB).toFixed(1)} MB`
  return bytes === 0 ? '0 bytes' : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

// Prototype line 657, except that it rounds before splitting: the prototype shows 119.6 s as "1 min 60 s" (Entry 143
// Q18).
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds * 10) / 10} s`
  const whole = Math.round(seconds)
  return `${Math.floor(whole / 60)} min ${whole % 60} s`
}

// "WAV, FLAC, MP3": the checks and S2's limits list, so they cannot disagree.
export function formatList(formats: readonly string[]): string {
  return formats.map((format) => format.toUpperCase()).join(', ')
}

// The size limit in MB, at most one decimal: the checks and S2's limits list.
export function limitInMb(bytes: number): number {
  return Number((bytes / BYTES_PER_MB).toFixed(1))
}

function formatCheck(extension: string, limits: CheckLimits): FileCheck {
  const formats = formatList(limits.formats)
  if (limits.formats.includes(extension)) {
    return { id: 'format', ok: true, wait: false, text: `Format ${extension.toUpperCase()} is accepted` }
  }
  const text =
    extension === ''
      ? `Unsupported format: this file has no extension. Use ${formats}.`
      : `Unsupported format: .${extension} files cannot be analyzed. Use ${formats}.`
  return { id: 'format', ok: false, wait: false, text }
}

function sizeCheck(size: number, limits: CheckLimits): FileCheck {
  if (size === 0) return { id: 'size', ok: false, wait: false, text: EMPTY_TEXT }
  const limit = limitInMb(limits.max_size_bytes)
  const shown = formatSize(size)
  if (size <= limits.max_size_bytes) {
    return { id: 'size', ok: true, wait: false, text: `Size ${shown} is within ${limit} MB` }
  }
  const text = `File too large: ${shown} is over the ${limit} MB limit. Export only the segment with the request.`
  return { id: 'size', ok: false, wait: false, text }
}

function durationCheck(duration: DurationInput, limits: CheckLimits): FileCheck {
  if (duration === 'reading') return { id: 'duration', ok: null, wait: true, text: 'Reading duration…' }
  // Prototype line 1292: a duration that is not finite is one the browser could not read.
  if (duration === 'unreadable' || !Number.isFinite(duration)) {
    const text = 'Duration could not be read in the browser; the service will check it'
    return { id: 'duration', ok: null, wait: false, text }
  }
  if (duration === 0) return { id: 'duration', ok: false, wait: false, text: EMPTY_TEXT }
  const limit = limits.max_duration_s
  const shown = formatDuration(duration)
  if (duration <= limit) {
    return { id: 'duration', ok: true, wait: false, text: `Duration ${shown} is within ${limit} s` }
  }
  const text = `Recording too long: ${shown} is over the ${limit} s limit. Upload only the segment with the request.`
  return { id: 'duration', ok: false, wait: false, text }
}

// Whether the duration is worth reading: an accepted format with data in it (prototype line 1287).
export function canBeAudio(file: Pick<File, 'name' | 'size'>, limits: Pick<CheckLimits, 'formats'>): boolean {
  return limits.formats.includes(fileExtension(file.name)) && file.size > 0
}

export function runChecks(file: Pick<File, 'name' | 'size'>, limits: CheckLimits, duration: DurationInput): CheckResult {
  const format = formatCheck(fileExtension(file.name), limits)
  const checks: readonly FileCheck[] = [
    format,
    sizeCheck(file.size, limits),
    !canBeAudio(file, limits)
      ? { id: 'duration', ok: null, wait: false, text: 'Duration not checked: the file cannot be read as audio' }
      : durationCheck(duration, limits),
  ]
  return { checks, valid: checks.every((check) => check.ok !== false && !check.wait) }
}
