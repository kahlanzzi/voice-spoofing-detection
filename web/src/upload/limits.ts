// Upload limits for the app (Stage 18 PR A), read by React through useSyncExternalStore(subscribe, getSnapshot): the
// one store behind S2 and S3. In memory only, the status store's pattern.
//
// GET /api/limits (docs/api_design.md §4.2). One request per signed-in session of the shell (Entry 145 Q20): the shell
// calls loadLimits() when it mounts, and it sends only from idle. Idle exists at page load and after the session ends,
// so StrictMode's second effect, navigation and a remount send nothing, and sign-out then sign-in sends one more. No
// polling: after that only S2's Retry, and only from failed (prototype line 1413). Check now does not reload the limits
// (prototype lines 1425-1428).
//
// A 500, no answer or a malformed 200 → failed: analysis stays off and nothing is guessed (HANDOFF line 207). A 401 is
// apiFetch's: it calls the session's expire(), the guard replaces the shell with S1, and the forget below drops the
// answer.
import { apiFetch } from '../api/client'
import { isHttpError } from '../api/errors'
import type { LimitsResponse } from '../api/types'
import { getSnapshot as getSession, subscribe as subscribeSession } from '../session/session'

export type LimitsState =
  | { readonly status: 'idle' }
  | { readonly status: 'loading' }
  | { readonly status: 'ready'; readonly limits: LimitsResponse }
  | { readonly status: 'failed' }

const IDLE: LimitsState = Object.freeze({ status: 'idle' })
const LOADING: LimitsState = Object.freeze({ status: 'loading' })
const FAILED: LimitsState = Object.freeze({ status: 'failed' })

let snapshot: LimitsState = IDLE
const listeners = new Set<() => void>()

// Bumped when the session ends, so a request that started before it is dropped.
let generation = 0
let inFlight: Promise<void> | null = null

export function getSnapshot(): LimitsState {
  return snapshot
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

// As in session.ts: a copy, so a listener that (un)subscribes does not change this round; a throwing listener does
// not stop the others, and its error is rethrown in a microtask. Not a timer.
function notify(): void {
  let failed = false
  let firstError: unknown
  for (const listener of [...listeners]) {
    try {
      listener()
    } catch (error) {
      if (!failed) {
        failed = true
        firstError = error
      }
    }
  }
  if (failed) {
    queueMicrotask(() => {
      throw firstError
    })
  }
}

function set(next: LimitsState): void {
  if (next === snapshot) return
  snapshot = next
  notify()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isPositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

// Lower case because checks.ts compares lower-case extensions. Anything else counts as no answer.
function isFormats(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((format) => typeof format === 'string' && format !== '' && format === format.toLowerCase())
  )
}

// The §4.2 body field by field; only its four fields are kept.
export function parseLimits(body: unknown): LimitsResponse | null {
  if (!isRecord(body)) return null
  const { formats, max_size_bytes: size, max_duration_s: duration, audio_deletion: deletion } = body
  if (!isFormats(formats) || !isPositive(size) || !isPositive(duration)) return null
  if (typeof deletion !== 'string' || deletion === '') return null
  return { formats: [...formats], max_size_bytes: size, max_duration_s: duration, audio_deletion: deletion }
}

async function runLoad(): Promise<void> {
  const started = generation
  let next: LimitsState
  try {
    const limits = parseLimits(await apiFetch<unknown>('/api/limits'))
    next = limits === null ? FAILED : { status: 'ready', limits }
  } catch (error) {
    // 401: apiFetch has already reported the expired session (and the forget below has normally run). Never failed.
    next = isHttpError(error) && error.status === 401 ? IDLE : FAILED
  }
  if (generation !== started) return
  set(next)
}

function start(): Promise<void> {
  set(LOADING)
  const load = runLoad().finally(() => {
    if (inFlight === load) inFlight = null
  })
  inFlight = load
  return load
}

// Only from idle; concurrent calls share one request. Never rejects.
export function loadLimits(): Promise<void> {
  if (inFlight) return inFlight
  return snapshot.status === 'idle' ? start() : Promise.resolve()
}

// Only from failed (S2's Retry). Never rejects.
export function retryLimits(): Promise<void> {
  if (inFlight) return inFlight
  return snapshot.status === 'failed' ? start() : Promise.resolve()
}

function forget(): void {
  generation += 1
  inFlight = null
  set(IDLE)
}

// Module level, like the client's handler slot: one subscription for the app's lifetime. No request on import.
subscribeSession(() => {
  if (getSession().status !== 'signedIn') forget()
})
