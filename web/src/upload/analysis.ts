// One analysis for the app (Stage 18 PR B), read by React through useSyncExternalStore(subscribe, getSnapshot). A module
// store, so a POST outlives the page that started it: navigation does not abort or resend, and the result waits here
// until a new analysis starts (HANDOFF lines 275-277). In memory only.
//
// POST /api/analyses (docs/api_design.md §4.3). One part, `file`, named upload.<ext>: the server reads only the extension
// (api/intake.py:149), so the real file name stays in the browser (HANDOFF line 280). The browser adds Origin to the
// POST itself (client.ts). There is no timer: S4 ends when the service or nginx answers (project log Entry 146).
//
// One request per press: submitting is set before apiFetch runs, so a second click in the same tick finds it and returns,
// with no help from a render (prototype line 1312). A start is allowed only from idle or error.
//
// An analysis lives as long as its draft: draft gone (New analysis, sign-out, another analyst) means analysis forgotten,
// and an answer still in flight is dropped. Expiry keeps both, so the analyst returns to the same place (HANDOFF line
// 211, Entry 146 Q31), which is why this store does not forget on session end as limits.ts and status.ts do. The one
// exception is the session error: it has nothing left to say once the session has ended.
//
// A 401 here is this store's: apiFetch skips its expiry hook for this POST, and S5 offers Sign in again (HANDOFF S5).
import { apiFetch } from '../api/client'
import { isHttpError, isNetworkError } from '../api/errors'
import type { AnalysisResponse, ErrorCode, RequestId, RiskLevel, Verdict } from '../api/types'
import { REQUEST_ID } from '../routing/routes'
import { getSnapshot as getSession, subscribe as subscribeSession } from '../session/session'
import { fileExtension } from './checks'
import { getSnapshot as getDraft, subscribe as subscribeDraft } from './fileDraft'

// The six kinds of prototype lines 943-950 (HANDOFF line 278), and the four browser checks the service can repeat
// (§6; Entry 146 Q28).
export type AnalysisErrorKind =
  | 'network'
  | 'service'
  | 'timeout'
  | 'corrupt'
  | 'session'
  | 'server'
  | 'format'
  | 'size'
  | 'duration'
  | 'empty'

export type AnalysisState =
  | { readonly status: 'idle' }
  | { readonly status: 'submitting' }
  | { readonly status: 'result'; readonly analysis: AnalysisResponse }
  | { readonly status: 'error'; readonly kind: AnalysisErrorKind; readonly requestId: RequestId | null }

const IDLE: AnalysisState = Object.freeze({ status: 'idle' })
const SUBMITTING: AnalysisState = Object.freeze({ status: 'submitting' })

// Anything not listed is Unexpected service error: the app never sends what would cause it (§6).
const KIND_OF_CODE: Partial<Record<ErrorCode, AnalysisErrorKind>> = {
  session_expired: 'session',
  corrupt_audio: 'corrupt',
  unsupported_format: 'format',
  file_too_large: 'size',
  recording_too_long: 'duration',
  empty_file: 'empty',
  analysis_unavailable: 'service',
  analysis_timeout: 'timeout',
}

const VERDICTS: readonly unknown[] = ['synthetic', 'uncertain', 'authentic'] satisfies Verdict[]
const RISK_LEVELS: readonly unknown[] = ['LOW', 'MEDIUM', 'HIGH'] satisfies RiskLevel[]

let snapshot: AnalysisState = IDLE
const listeners = new Set<() => void>()

// Bumped when the analysis is forgotten, so a POST that started before it is dropped.
let generation = 0

export function getSnapshot(): AnalysisState {
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

function set(next: AnalysisState): void {
  if (next === snapshot) return
  snapshot = next
  notify()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isText(value: unknown): value is string {
  return typeof value === 'string' && value !== ''
}

function isRequestId(value: unknown): value is RequestId {
  return typeof value === 'string' && REQUEST_ID.test(value)
}

// The §11 body field by field (api/schemas.py); only its 8 fields are kept. No check across fields: the policy is the
// service's (HANDOFF line 270).
export function parseAnalysis(body: unknown): AnalysisResponse | null {
  if (!isRecord(body)) return null
  const { prediction, confidence, risk_level: risk, model_version: model, profile_version: profile } = body
  const { policy_version: policy, request_id: requestId, notice } = body
  if (!VERDICTS.includes(prediction) || !RISK_LEVELS.includes(risk)) return null
  if (typeof confidence !== 'number' || !(confidence >= 0 && confidence <= 1)) return null
  if (!isText(model) || !isText(profile) || !isText(policy) || !isText(notice) || !isRequestId(requestId)) return null
  return {
    prediction: prediction as Verdict,
    confidence,
    risk_level: risk as RiskLevel,
    model_version: model,
    profile_version: profile,
    policy_version: policy,
    request_id: requestId,
    notice,
  }
}

function uploadForm(file: File): FormData {
  const form = new FormData()
  form.append('file', file, `upload.${fileExtension(file.name)}`)
  return form
}

function failure(kind: AnalysisErrorKind, requestId: RequestId | null): AnalysisState {
  if (kind === 'session' && getSession().status !== 'signedIn') return IDLE
  return { status: 'error', kind, requestId }
}

function afterError(error: unknown): AnalysisState {
  if (isNetworkError(error)) return failure('network', null)
  if (isHttpError(error)) return failure(KIND_OF_CODE[error.code] ?? 'server', error.requestId)
  return failure('server', null)
}

async function runAnalysis(file: File): Promise<void> {
  const started = generation
  let next: AnalysisState
  try {
    const analysis = parseAnalysis(await apiFetch<unknown>('/api/analyses', { method: 'POST', body: uploadForm(file) }))
    next = analysis === null ? failure('server', null) : { status: 'result', analysis }
  } catch (error) {
    next = afterError(error)
  }
  if (generation !== started) return
  set(next)
}

// Analyze recording and Retry analysis. Only from idle or error. Never rejects.
export function startAnalysis(file: File): Promise<void> {
  if (snapshot.status !== 'idle' && snapshot.status !== 'error') return Promise.resolve()
  set(SUBMITTING)
  return runAnalysis(file)
}

// A pick on S5 replaces the file, so its error goes with it. Never while a POST is running.
export function resetAnalysis(): void {
  if (snapshot.status !== 'submitting') set(IDLE)
}

function forget(): void {
  generation += 1
  set(IDLE)
}

// Module level, like the draft's own subscription: one each for the app's lifetime. No request on import.
subscribeDraft(() => {
  if (getDraft() === null) forget()
})

subscribeSession(() => {
  if (snapshot.status === 'error' && snapshot.kind === 'session' && getSession().status !== 'signedIn') set(IDLE)
})
