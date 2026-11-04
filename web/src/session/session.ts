// Session state for the app (Stage 18 PR 3b), read by React through useSyncExternalStore(subscribe,
// getSnapshot). In memory only: the session itself is the server's HttpOnly cookie (docs/api_design.md
// §4.8), and the username is never stored or logged.
//
// States (HANDOFF §1 S1): checking until GET /api/session answers on load; signedIn; signedOut, with
// reason 'expired' for the S1 expired-session notice and 'signOut' after the rail's Sign out (PR 6b: S1 without
// next); unavailable for "sign-in service unavailable". A19: a failed check never signs the analyst out; only a 401
// does. Nothing here clears the file or note drafts (A23): their stores subscribe here (upload/fileDraft.ts, PR 8b),
// and nothing navigates (routing is PR 3c).
import { apiFetch, setSessionExpiredHandler } from '../api/client'
import { isHttpError } from '../api/errors'
import type { SessionInfo } from '../api/types'

export type SessionState =
  | { readonly status: 'checking' }
  | { readonly status: 'signedIn'; readonly username: string }
  | { readonly status: 'signedOut'; readonly reason: 'none' | 'expired' | 'signOut' }
  | { readonly status: 'unavailable' }

const CHECKING: SessionState = Object.freeze({ status: 'checking' })
const UNAVAILABLE: SessionState = Object.freeze({ status: 'unavailable' })
const SIGNED_OUT: SessionState = Object.freeze({ status: 'signedOut', reason: 'none' })
const EXPIRED: SessionState = Object.freeze({ status: 'signedOut', reason: 'expired' })
const SIGNED_OUT_BY_ANALYST: SessionState = Object.freeze({ status: 'signedOut', reason: 'signOut' })

let snapshot: SessionState = CHECKING
const listeners = new Set<() => void>()

// Bumped by every transition call, even one that changes nothing, so a check that started before it is
// dropped. Snapshot identity is not enough: expire() while already expired keeps the same object.
let generation = 0
let inFlight: Promise<void> | null = null
let signOutInFlight: Promise<boolean> | null = null

export function getSnapshot(): SessionState {
  return snapshot
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function sameState(a: SessionState, b: SessionState): boolean {
  if (a.status === 'signedIn' && b.status === 'signedIn') return a.username === b.username
  if (a.status === 'signedOut' && b.status === 'signedOut') return a.reason === b.reason
  return a.status === b.status
}

// A copy, so a listener that (un)subscribes does not change this round. A throwing listener does not
// stop the others or the caller (e.g. apiFetch's expiry call); its error is rethrown in a microtask so it
// still surfaces as uncaught. Not a timer.
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

function set(next: SessionState): void {
  if (sameState(snapshot, next)) return
  snapshot = next
  notify()
}

function transition(next: SessionState): void {
  generation += 1
  inFlight = null
  set(next)
}

// The handler for an expired session on any other route (client.ts), and S5 "Sign in again" (PR 10).
export function expire(): void {
  transition(EXPIRED)
}

// After POST /api/session answers 200 (PR 4).
export function markSignedIn(username: string): void {
  transition({ status: 'signedIn', username })
}

// After DELETE /api/session answers 204 or 401 (PR 6b).
export function markSignedOut(): void {
  transition(SIGNED_OUT_BY_ANALYST)
}

// Only a 401 signs out: expired if the analyst was signed in, else the plain sign-in screen. 403, 5xx,
// any other status, a non-JSON 2xx, a network error or anything else thrown → unavailable (A19).
function afterFailure(error: unknown): SessionState {
  if (isHttpError(error) && error.status === 401) return snapshot.status === 'signedIn' ? EXPIRED : SIGNED_OUT
  return UNAVAILABLE
}

async function runCheck(): Promise<void> {
  const started = generation
  let next: SessionState
  try {
    const info = await apiFetch<SessionInfo>('/api/session')
    next = { status: 'signedIn', username: info.username }
  } catch (error) {
    next = afterFailure(error)
  }
  // A transition since the call started wins; this answer is stale.
  if (generation === started) set(next)
}

// GET /api/session on load and for S1 Retry. Concurrent calls share one request; the state does not
// change until the answer (no 'checking' on Retry). Never rejects.
export function checkSession(): Promise<void> {
  if (inFlight) return inFlight
  const check = runCheck().finally(() => {
    if (inFlight === check) inFlight = null
  })
  inFlight = check
  return check
}

async function runSignOut(): Promise<boolean> {
  const started = generation
  let done: boolean
  try {
    await apiFetch<void>('/api/session', { method: 'DELETE' })
    done = true
  } catch (error) {
    // 401: no session left on the server, so the analyst is signed out anyway (§4.8). Anything else: still signed in.
    done = isHttpError(error) && error.status === 401
  }
  // A transition since the call started wins; this answer is stale.
  if (generation !== started) return snapshot.status === 'signedOut'
  if (done) markSignedOut()
  return done
}

// The rail's Sign out (PR 6b): DELETE /api/session. Concurrent calls share one request, so a double click sends one
// DELETE. True once signed out; false when the analyst is still signed in (503, 5xx, 403, network). Never rejects.
export function signOut(): Promise<boolean> {
  if (signOutInFlight) return signOutInFlight
  const call = runSignOut().finally(() => {
    if (signOutInFlight === call) signOutInFlight = null
  })
  signOutInFlight = call
  return call
}

// Once, at module level in main.tsx. The client has one handler slot, so a second call replaces the
// same function and cannot fire twice.
export function startSession(): Promise<void> {
  setSessionExpiredHandler(expire)
  return checkSession()
}
