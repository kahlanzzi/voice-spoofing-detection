import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules), as in limits.test.ts: the session store and the client come from the same
// fresh registry, so the subscription is fresh too.

type History = typeof import('./history')
type Session = typeof import('../session/session')
type Client = typeof import('../api/client')

let history: History
let session: Session
let client: Client
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const ID = `req_${'a'.repeat(32)}`

const ANALYSIS = {
  prediction: 'synthetic',
  confidence: 0.97,
  risk_level: 'HIGH',
  model_version: 'mock',
  profile_version: 'preprocessing-v1',
  policy_version: 'risk-v1',
  request_id: ID,
  notice: 'Decision support only.',
}

const NOTE = { text: 'Called back.', updated_at: '2026-11-03T10:00:00Z', updated_by: 'analyst01' }

const RECORD = { analysis: ANALYSIS, created_at: '2026-11-03T09:14:05Z', note: null }

const BODY = { items: [RECORD], total: 1, page: 1, page_size: 10, counts: { synthetic: 1, uncertain: 0, authentic: 0 } }

const READY = { status: 'ready', items: [RECORD], total: 1 }

type Body = Record<string, unknown>

// A copy of BODY with the first record's fields replaced.
const withRecord = (fields: Body): Body => ({ ...BODY, items: [{ ...RECORD, ...fields }] })
const withAnalysis = (fields: Body): Body => withRecord({ analysis: { ...ANALYSIS, ...fields } })
const without = (object: Body, key: string): Body => Object.fromEntries(Object.entries(object).filter(([name]) => name !== key))

function json(code: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status: code, headers: { 'Content-Type': 'application/json' } })
}

function apiError(code: number, name: string): Response {
  return json(code, { error: { code: name, message: 'x', request_id: null } })
}

function deferred(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

beforeEach(async () => {
  vi.resetModules()
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
  history = await import('./history')
  session = await import('../session/session')
  client = await import('../api/client')
  client.setSessionExpiredHandler(session.expire)
  session.markSignedIn('analyst01')
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('the request and the parser', () => {
  it('sends one GET /api/analyses with no query', async () => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await history.loadHistory()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/analyses')
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: 'GET', credentials: 'same-origin' })
  })

  it('keeps items and total of a contract body, and nothing else', () => {
    expect(history.parseHistory(BODY)).toEqual({ items: [RECORD], total: 1 })
  })

  it('accepts an empty list', () => {
    expect(history.parseHistory({ ...BODY, items: [], total: 0 })).toEqual({ items: [], total: 0 })
  })

  it('accepts a note, with an author or without one (notes saved before sign-in existed)', () => {
    const items = [{ ...RECORD, note: NOTE }, { ...RECORD, note: { ...NOTE, updated_by: null } }]
    expect(history.parseHistory({ ...BODY, items, total: 2 })?.items).toEqual(items)
  })

  it('one bad record rejects the whole body: never a half-rendered list', () => {
    const items = [RECORD, RECORD, { ...RECORD, created_at: 'yesterday' }]
    expect(history.parseHistory({ ...BODY, items, total: 3 })).toBeNull()
  })

  it.each<[string, unknown]>([
    ['body null', null],
    ['body an array', [RECORD]],
    ['body text', 'History'],
    ['no items', without(BODY, 'items')],
    ['items not an array', { ...BODY, items: {} }],
    ['no total', without(BODY, 'total')],
    ['total negative', { ...BODY, total: -1 }],
    ['total a fraction', { ...BODY, total: 1.5 }],
    ['total text', { ...BODY, total: '1' }],
    ['more items than total', { ...BODY, total: 0 }],
    ['no items with a total', { ...BODY, items: [] }],
    ['a record null', { ...BODY, items: [null] }],
    ['a record with no analysis', { ...BODY, items: [without(RECORD, 'analysis')] }],
    ['an unknown prediction', withAnalysis({ prediction: 'fake' })],
    ['an unknown risk_level', withAnalysis({ risk_level: 'SEVERE' })],
    ['a short request_id', withAnalysis({ request_id: 'req_8fa98c02' })],
    ['no model_version', { ...BODY, items: [{ ...RECORD, analysis: without(ANALYSIS, 'model_version') }] }],
    ['no created_at', { ...BODY, items: [without(RECORD, 'created_at')] }],
    ['created_at a number', withRecord({ created_at: 1 })],
    ['created_at with no Z', withRecord({ created_at: '2026-11-03T09:14:05' })],
    ['created_at with an offset', withRecord({ created_at: '2026-11-03T09:14:05+01:00' })],
    ['created_at not a date', withRecord({ created_at: '2026-13-45T09:14:05Z' })],
    ['no note field', { ...BODY, items: [without(RECORD, 'note')] }],
    ['note text alone', withRecord({ note: 'Called back.' })],
    ['a note with no text', withRecord({ note: without(NOTE, 'text') })],
    ['a note with no updated_at', withRecord({ note: without(NOTE, 'updated_at') })],
    ['a note with a numeric author', withRecord({ note: { ...NOTE, updated_by: 1 } })],
  ])('%s is malformed', (_label, body) => {
    expect(history.parseHistory(body)).toBeNull()
  })
})

describe('the fetch rule', () => {
  it('concurrent loads share one GET', async () => {
    const pending = deferred()
    fetchMock.mockReturnValueOnce(pending.promise)
    const first = history.loadHistory()
    const second = history.loadHistory()
    pending.resolve(json(200, BODY))
    expect(await Promise.all([first, second])).toEqual([READY, READY])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('a load after an answer sends another GET', async () => {
    fetchMock.mockImplementation(async () => json(200, BODY))
    await history.loadHistory()
    await history.loadHistory()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('a load after a failure sends another GET', async () => {
    fetchMock.mockResolvedValueOnce(apiError(500, 'internal_error')).mockResolvedValueOnce(json(200, BODY))
    expect(await history.loadHistory()).toEqual({ status: 'failed' })
    expect(await history.loadHistory()).toEqual(READY)
  })

  it('sets no timer: an hour later nothing more was sent', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(async () => json(200, BODY))
    await history.loadHistory()
    expect(vi.getTimerCount()).toBe(0)
    await vi.advanceTimersByTimeAsync(3_600_000)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('an answer after the session ended is dropped', async () => {
    const pending = deferred()
    fetchMock.mockReturnValueOnce(pending.promise)
    const load = history.loadHistory()
    session.markSignedOut()
    pending.resolve(json(200, BODY))
    expect(await load).toBeNull()
  })

  it('a new session does not share the old request', async () => {
    const pending = deferred()
    fetchMock.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(json(200, BODY))
    const old = history.loadHistory()
    session.markSignedOut()
    session.markSignedIn('analyst02')
    expect(await history.loadHistory()).toEqual(READY)
    pending.resolve(json(200, BODY))
    expect(await old).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('a 500 is failed', async () => {
    fetchMock.mockResolvedValueOnce(apiError(500, 'internal_error'))
    expect(await history.loadHistory()).toEqual({ status: 'failed' })
  })

  it('no answer is failed', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    expect(await history.loadHistory()).toEqual({ status: 'failed' })
  })

  it('a malformed 200 is failed', async () => {
    fetchMock.mockResolvedValueOnce(json(200, { items: 'none' }))
    expect(await history.loadHistory()).toEqual({ status: 'failed' })
  })

  it('a 403 is failed', async () => {
    fetchMock.mockResolvedValueOnce(apiError(403, 'access_denied'))
    expect(await history.loadHistory()).toEqual({ status: 'failed' })
  })

  it('a 401 expires the session and is never failed', async () => {
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    expect(await history.loadHistory()).toBeNull()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
  })
})
