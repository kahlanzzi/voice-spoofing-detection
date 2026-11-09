// Service status for the app (Stage 18 PR 7a), read by React through useSyncExternalStore(subscribe, getSnapshot):
// the one store behind the rail status line and S10 (Entry 137 Q3). In memory only, the session store's pattern.
//
// GET /api/status (docs/api_design.md §4.1). Team decisions 2026-11-01 (project log Entry 127): a 500 or no answer
// → Unavailable; degraded → Ready with a notice naming what fails; no polling. So checkStatus() runs only when the
// shell mounts (after sign-in: /status needs a session, §4.8) and on S10's Check now (PR 7b): never on a timer,
// focus or visibility, and not on entering /status (prototype lines 1180 and 1425). The browser derives nothing:
// state and the failing parts come from the body as given; `degraded` is open decision 7 (HANDOFF §11 item 7).
//
// The status belongs to a signed-in session: once the session is anything else (expired, signed out, checking,
// unavailable), the store forgets its result and drops a check in flight, so the next sign-in starts from Checking
// with its own request. A 401 is apiFetch's: it calls the session's expire(), the guard replaces the shell with S1,
// and the forget above drops that answer.
import { apiFetch } from '../api/client'
import { isHttpError } from '../api/errors'
import type { StatusResponse, UtcTimestamp } from '../api/types'
import { getSnapshot as getSession, subscribe as subscribeSession } from '../session/session'

export type StatusPart = 'analysis' | 'history_and_stats'

export interface StatusDetails {
  readonly versions: StatusResponse['versions']
  readonly bundleManifestSha256: string | null
  readonly deferralOnFit: number
}

// timeSource 'api': checkedAt is the body's checked_at; 'browser': the browser's clock when the check failed, so
// S10 says "Last attempt", not "Last checked" (PR 7b).
export type StatusResult =
  | {
      readonly kind: 'ready'
      readonly notice: boolean // the API said degraded: Ready with a notice naming `failing` (a fallback if empty)
      readonly failing: readonly StatusPart[]
      readonly mock: boolean
      readonly checkedAt: UtcTimestamp
      readonly timeSource: 'api'
      readonly details: StatusDetails
    }
  | {
      readonly kind: 'unavailable'
      readonly source: 'service'
      readonly failing: readonly StatusPart[]
      readonly mock: boolean
      readonly checkedAt: UtcTimestamp
      readonly timeSource: 'api'
      readonly details: StatusDetails
    }
  | {
      readonly kind: 'unavailable'
      readonly source: 'noAnswer' // 500, 503, any other status, a malformed 200, a network failure
      readonly checkedAt: UtcTimestamp
      readonly timeSource: 'browser'
    }

// checking: a check is in flight. result: the last answer, kept while a new check runs (prototype line 1190).
export interface StatusState {
  readonly checking: boolean
  readonly result: StatusResult | null
}

const INITIAL: StatusState = Object.freeze({ checking: true, result: null })

let snapshot: StatusState = INITIAL
const listeners = new Set<() => void>()

// Bumped when the session ends, so a check that started before it is dropped.
let generation = 0
let inFlight: Promise<void> | null = null

export function getSnapshot(): StatusState {
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

function set(next: StatusState): void {
  if (next.checking === snapshot.checking && next.result === snapshot.result) return
  snapshot = next
  notify()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const STATES = ['ready', 'degraded', 'unavailable'] as const
const PARTS: readonly StatusPart[] = ['analysis', 'history_and_stats']

function isCheck(value: unknown): boolean {
  return value === 'available' || value === 'unavailable'
}

// The §4.1 body field by field; anything else counts as no answer.
export function parseStatus(body: unknown): StatusResponse | null {
  if (!isRecord(body) || !isRecord(body.checks) || !isRecord(body.versions)) return null
  const { state, checked_at: checkedAt, checks, versions, bundle_manifest_sha256: sha, deferral_on_fit: deferral, mock } = body
  const valid =
    STATES.includes(state as StatusResponse['state']) &&
    typeof checkedAt === 'string' &&
    PARTS.every((part) => isCheck(checks[part])) &&
    ['model_version', 'profile_version', 'policy_version'].every((key) => typeof versions[key] === 'string') &&
    (sha === null || typeof sha === 'string') &&
    typeof deferral === 'number' &&
    typeof mock === 'boolean'
  return valid ? (body as unknown as StatusResponse) : null
}

export function resultFrom(body: StatusResponse): StatusResult {
  const failing = PARTS.filter((part) => body.checks[part] === 'unavailable')
  const common = {
    failing,
    mock: body.mock,
    checkedAt: body.checked_at,
    timeSource: 'api',
    details: {
      versions: { ...body.versions },
      bundleManifestSha256: body.bundle_manifest_sha256,
      deferralOnFit: body.deferral_on_fit,
    },
  } as const
  if (body.state === 'unavailable') return { kind: 'unavailable', source: 'service', ...common }
  // The API's state wins over the checks (the browser derives nothing): ready names no failing part.
  const notice = body.state === 'degraded'
  return { kind: 'ready', ...common, notice, failing: notice ? failing : [] }
}

function noAnswer(): StatusResult {
  return { kind: 'unavailable', source: 'noAnswer', checkedAt: new Date().toISOString(), timeSource: 'browser' }
}

async function runCheck(): Promise<void> {
  const started = generation
  let result: StatusResult | null
  try {
    const body = parseStatus(await apiFetch<unknown>('/api/status'))
    result = body === null ? noAnswer() : resultFrom(body)
  } catch (error) {
    // 401: apiFetch has already reported the expired session (and the forget below has normally run); keep the result.
    result = isHttpError(error) && error.status === 401 ? null : noAnswer()
  }
  if (generation !== started) return
  set({ checking: false, result: result ?? snapshot.result })
}

// Concurrent calls share one request (StrictMode's second effect, a double click on Check now). The previous result
// stays on screen while it runs. Never rejects.
export function checkStatus(): Promise<void> {
  if (inFlight) return inFlight
  set({ checking: true, result: snapshot.result })
  const check = runCheck().finally(() => {
    if (inFlight === check) inFlight = null
  })
  inFlight = check
  return check
}

function forget(): void {
  generation += 1
  inFlight = null
  set(INITIAL)
}

// Module level, like the client's handler slot: one subscription for the app's lifetime. No request on import.
subscribeSession(() => {
  if (getSession().status !== 'signedIn') forget()
})
