// The list behind S7 (Stage 18 PR C): GET /api/analyses (docs/api_design.md §4.4), with no parameters, so the API's first
// page of 10, newest first (Entry 147 Q32). The state lives in pages/History.tsx, not in a module store (Q34): the list
// has one reader and must not outlive the page (HANDOFF line 68), so nothing is kept after leaving or sign-out.
//
// The page calls loadHistory() once per entry. Concurrent calls share one request, so StrictMode's second effect sends
// nothing. No polling: no timer, only an entry or Retry sends.
//
// One malformed record, or a malformed body, makes the whole answer failed: a list is never half rendered. A 401 is
// apiFetch's: it calls the session's expire() (client.ts), the guard goes to S1 with next, and the answer is dropped, so
// no failure flashes. An answer that started before the session ended is dropped too (the generation guard of
// upload/limits.ts): null, and the page keeps what it shows.
import { apiFetch } from '../api/client'
import { isHttpError } from '../api/errors'
import type { HistoryResponse, Note, StoredAnalysis } from '../api/types'
import { getSnapshot as getSession, subscribe as subscribeSession } from '../session/session'
import { parseAnalysis } from '../upload/analysis'

export type HistoryPage = Pick<HistoryResponse, 'items' | 'total'>

export type HistoryAnswer =
  | ({ readonly status: 'ready' } & Readonly<HistoryPage>)
  | { readonly status: 'failed' }

const FAILED: HistoryAnswer = Object.freeze({ status: 'failed' })

// As the API stores it (api/repository.py _UTC).
const UTC_SECOND = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/

// Bumped when the session ends, so a request that started before it is dropped.
let generation = 0
let inFlight: Promise<HistoryAnswer | null> | null = null

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isUtcTime(value: unknown): value is string {
  return typeof value === 'string' && UTC_SECOND.test(value) && !Number.isNaN(Date.parse(value))
}

function parseNote(value: unknown): Note | null | undefined {
  if (value === null) return null
  if (!isRecord(value)) return undefined
  const { text, updated_at: updatedAt, updated_by: updatedBy } = value
  if (typeof text !== 'string' || !isUtcTime(updatedAt)) return undefined
  if (updatedBy !== null && typeof updatedBy !== 'string') return undefined
  return { text, updated_at: updatedAt, updated_by: updatedBy }
}

function parseStored(value: unknown): StoredAnalysis | null {
  if (!isRecord(value)) return null
  const analysis = parseAnalysis(value.analysis)
  const note = parseNote(value.note)
  if (analysis === null || note === undefined || !isUtcTime(value.created_at)) return null
  return { analysis, created_at: value.created_at, note }
}

// The §4.4 fields the page uses, field by field. page, page_size and counts are not used, so neither checked nor kept.
export function parseHistory(body: unknown): HistoryPage | null {
  if (!isRecord(body) || !Array.isArray(body.items)) return null
  const { total } = body
  if (typeof total !== 'number' || !Number.isInteger(total) || total < 0) return null
  if (body.items.length > total || (body.items.length === 0 && total > 0)) return null
  const items = body.items.map(parseStored)
  if (items.includes(null)) return null
  return { items: items as StoredAnalysis[], total }
}

async function runLoad(): Promise<HistoryAnswer | null> {
  const started = generation
  let answer: HistoryAnswer | null
  try {
    const page = parseHistory(await apiFetch<unknown>('/api/analyses'))
    answer = page === null ? FAILED : { status: 'ready', ...page }
  } catch (error) {
    answer = isHttpError(error) && error.status === 401 ? null : FAILED
  }
  return generation === started ? answer : null
}

// Concurrent calls share one request. Never rejects.
export function loadHistory(): Promise<HistoryAnswer | null> {
  if (inFlight) return inFlight
  const load = runLoad().finally(() => {
    if (inFlight === load) inFlight = null
  })
  inFlight = load
  return load
}

// Module level, like the client's handler slot: one subscription for the app's lifetime. No request on import.
subscribeSession(() => {
  if (getSession().status !== 'signedIn') {
    generation += 1
    inFlight = null
  }
})
