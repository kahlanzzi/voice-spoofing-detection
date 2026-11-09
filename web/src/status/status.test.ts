import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules), as in session.test.ts: the store has no reset function. The session
// store and the client come from the same fresh registry, so the status store's session subscription is fresh too.

type Status = typeof import('./status')
type Session = typeof import('../session/session')
type Client = typeof import('../api/client')

let status: Status
let session: Session
let client: Client
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const BODY = {
  state: 'ready',
  checked_at: '2026-11-03T09:14:05Z',
  checks: { analysis: 'available', history_and_stats: 'available' },
  versions: { model_version: 'mock', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' },
  bundle_manifest_sha256: null,
  deferral_on_fit: 0.0993,
  mock: true,
} as const

const DETAILS = {
  versions: BODY.versions,
  bundleManifestSha256: null,
  deferralOnFit: 0.0993,
}

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
  status = await import('./status')
  session = await import('../session/session')
  client = await import('../api/client')
  session.markSignedIn('analyst01')
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('exports and the initial state', () => {
  it('are the store surface, with no reset', () => {
    expect(Object.keys(status).sort()).toEqual(['checkStatus', 'getSnapshot', 'parseStatus', 'resultFrom', 'subscribe'])
  })

  it('starts as checking with no result, and importing it sends no request', () => {
    expect(status.getSnapshot()).toEqual({ checking: true, result: null })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('checkStatus: a valid 200', () => {
  it('sends GET /api/status with same-origin credentials', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [path, init] = fetchMock.mock.calls[0]!
    expect(path).toBe('/api/status')
    expect(init).toMatchObject({ method: 'GET', credentials: 'same-origin' })
  })

  it('ready → ready, no notice, the API time and details', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    expect(status.getSnapshot()).toEqual({
      checking: false,
      result: {
        kind: 'ready',
        notice: false,
        failing: [],
        mock: true,
        checkedAt: '2026-11-03T09:14:05Z',
        timeSource: 'api',
        details: DETAILS,
      },
    })
  })

  it('mock false is kept', async () => {
    fetchMock.mockResolvedValue(json(200, { ...BODY, mock: false, bundle_manifest_sha256: 'ab'.repeat(32) }))
    await status.checkStatus()
    const { result } = status.getSnapshot()
    expect(result).toMatchObject({ kind: 'ready', mock: false, details: { bundleManifestSha256: 'ab'.repeat(32) } })
  })

  it.each([
    [{ analysis: 'available', history_and_stats: 'unavailable' }, ['history_and_stats']],
    [{ analysis: 'unavailable', history_and_stats: 'available' }, ['analysis']],
    [{ analysis: 'unavailable', history_and_stats: 'unavailable' }, ['analysis', 'history_and_stats']],
    [{ analysis: 'available', history_and_stats: 'available' }, []],
  ])('degraded with checks %o → ready with a notice naming %o', async (checks, failing) => {
    fetchMock.mockResolvedValue(json(200, { ...BODY, state: 'degraded', checks }))
    await status.checkStatus()
    expect(status.getSnapshot().result).toMatchObject({ kind: 'ready', notice: true, failing })
  })

  it('ready names no failing part even if a check says unavailable (the API state wins)', async () => {
    fetchMock.mockResolvedValue(json(200, { ...BODY, checks: { analysis: 'available', history_and_stats: 'unavailable' } }))
    await status.checkStatus()
    expect(status.getSnapshot().result).toMatchObject({ kind: 'ready', notice: false, failing: [] })
  })

  it('unavailable → unavailable from the service, with the failing checks and details', async () => {
    fetchMock.mockResolvedValue(json(200, { ...BODY, state: 'unavailable', checks: { analysis: 'available', history_and_stats: 'unavailable' } }))
    await status.checkStatus()
    expect(status.getSnapshot().result).toEqual({
      kind: 'unavailable',
      source: 'service',
      failing: ['history_and_stats'],
      mock: true,
      checkedAt: '2026-11-03T09:14:05Z',
      timeSource: 'api',
      details: DETAILS,
    })
  })
})

describe('checkStatus: no valid answer → unavailable with the browser time', () => {
  const NOW = new Date('2026-11-03T10:40:00Z')

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(NOW)
  })

  const NO_ANSWER = { kind: 'unavailable', source: 'noAnswer', checkedAt: NOW.toISOString(), timeSource: 'browser' }

  it.each([
    ['500 internal_error', () => apiError(500, 'internal_error')],
    ["503 analysis_unavailable (nginx, api down)", () => apiError(503, 'analysis_unavailable')],
    ['504', () => apiError(504, 'analysis_timeout')],
    ['418 with no body', () => new Response(null, { status: 418 })],
    ['403 access_denied', () => apiError(403, 'access_denied')],
    ['200 that is not JSON', () => new Response('<!doctype html>', { status: 200 })],
    ['200 with an unknown state', () => json(200, { ...BODY, state: 'up' })],
    ['200 without mock', () => json(200, { ...BODY, mock: undefined })],
    ['200 with a bad check value', () => json(200, { ...BODY, checks: { analysis: 'ok', history_and_stats: 'available' } })],
    ['200 without versions', () => json(200, { ...BODY, versions: null })],
    ['200 with a string deferral', () => json(200, { ...BODY, deferral_on_fit: '0.1' })],
    ['200 with an array', () => json(200, [])],
  ])('%s', async (_, response) => {
    fetchMock.mockResolvedValue(response())
    await status.checkStatus()
    expect(status.getSnapshot()).toEqual({ checking: false, result: NO_ANSWER })
  })

  it('a network failure', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    await status.checkStatus()
    expect(status.getSnapshot()).toEqual({ checking: false, result: NO_ANSWER })
  })

  it('anything else thrown', async () => {
    fetchMock.mockRejectedValue(new Error('odd'))
    await expect(status.checkStatus()).resolves.toBeUndefined()
    expect(status.getSnapshot().result).toEqual(NO_ANSWER)
  })
})

describe('one request at a time, and the previous result while checking', () => {
  it('concurrent calls share one request and one promise', async () => {
    const answer = deferred()
    fetchMock.mockReturnValue(answer.promise)
    const first = status.checkStatus()
    const second = status.checkStatus()
    expect(second).toBe(first)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    answer.resolve(json(200, BODY))
    await first
    expect(status.getSnapshot().checking).toBe(false)
  })

  it('a later call sends a new request', async () => {
    fetchMock.mockImplementation(async () => json(200, BODY))
    await status.checkStatus()
    await status.checkStatus()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('a re-check keeps the previous result, with checking true, until the answer', async () => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await status.checkStatus()
    const ready = status.getSnapshot().result
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const check = status.checkStatus()
    expect(status.getSnapshot()).toEqual({ checking: true, result: ready })
    expect(status.getSnapshot().result).toBe(ready)
    answer.resolve(apiError(503, 'analysis_unavailable'))
    await check
    expect(status.getSnapshot().result).toMatchObject({ kind: 'unavailable', source: 'noAnswer' })
  })
})

describe('the session', () => {
  it('401 session_expired: apiFetch expires the session; the store forgets and drops the answer', async () => {
    client.setSessionExpiredHandler(session.expire)
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await status.checkStatus()
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await status.checkStatus()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
    expect(status.getSnapshot()).toEqual({ checking: true, result: null })
  })

  it('401 with no expiry handler keeps the previous result (never Unavailable)', async () => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await status.checkStatus()
    const ready = status.getSnapshot().result
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await status.checkStatus()
    expect(status.getSnapshot()).toEqual({ checking: false, result: ready })
  })

  it.each([
    ['expire', () => session.expire()],
    ['markSignedOut', () => session.markSignedOut()],
  ])('%s while a check is in flight: its answer is dropped, and the next check sends its own request', async (_, end) => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const stale = status.checkStatus()
    end()
    expect(status.getSnapshot()).toEqual({ checking: true, result: null })
    session.markSignedIn('analyst02')
    fetchMock.mockResolvedValueOnce(json(200, { ...BODY, mock: false }))
    const fresh = status.checkStatus()
    expect(fresh).not.toBe(stale)
    answer.resolve(json(200, { ...BODY, state: 'unavailable' }))
    await Promise.all([stale, fresh])
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(status.getSnapshot().result).toMatchObject({ kind: 'ready', mock: false })
  })

  it('signing out forgets the last result', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    session.markSignedOut()
    expect(status.getSnapshot()).toEqual({ checking: true, result: null })
  })

  it('staying signed in (a repeat markSignedIn) keeps the result', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    const snapshot = status.getSnapshot()
    session.markSignedIn('analyst01')
    expect(status.getSnapshot()).toBe(snapshot)
  })
})

describe('subscribe and getSnapshot', () => {
  it('notifies on each change and stops after unsubscribe', async () => {
    fetchMock.mockImplementation(async () => json(200, BODY))
    const listener = vi.fn()
    const unsubscribe = status.subscribe(listener)
    await status.checkStatus()
    // checking stays true at the start (no change), then the answer.
    expect(listener).toHaveBeenCalledTimes(1)
    unsubscribe()
    await status.checkStatus()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('getSnapshot keeps the same object while unchanged', async () => {
    const first = status.getSnapshot()
    expect(status.getSnapshot()).toBe(first)
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    expect(status.getSnapshot()).not.toBe(first)
  })

  // The change is made synchronously (signing out forgets the result): with queueMicrotask stubbed, reading a fetch
  // body would never finish.
  it('a throwing listener does not stop the others; its error is rethrown in a microtask', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await status.checkStatus()
    const queued: (() => void)[] = []
    vi.spyOn(globalThis, 'queueMicrotask').mockImplementation((callback) => {
      queued.push(callback)
    })
    const boom = new Error('listener failed')
    const after = vi.fn()
    status.subscribe(() => {
      throw boom
    })
    status.subscribe(after)
    expect(() => session.markSignedOut()).not.toThrow()
    expect(after).toHaveBeenCalledTimes(1)
    expect(status.getSnapshot()).toEqual({ checking: true, result: null })
    expect(queued).toHaveLength(1)
    expect(() => queued[0]?.()).toThrow(boom)
  })
})
