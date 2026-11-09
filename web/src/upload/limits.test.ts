import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules), as in status.test.ts: the store has no reset function. The session store
// and the client come from the same fresh registry, so the limits store's session subscription is fresh too.

type Limits = typeof import('./limits')
type Session = typeof import('../session/session')
type Client = typeof import('../api/client')

let limits: Limits
let session: Session
let client: Client
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const BODY = {
  formats: ['wav', 'flac', 'mp3'],
  max_size_bytes: 20971520,
  max_duration_s: 60,
  audio_deletion: 'Deleted when processing finishes',
}

const READY = { status: 'ready', limits: BODY }

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

async function toFailed(): Promise<void> {
  fetchMock.mockResolvedValueOnce(apiError(500, 'internal_error'))
  await limits.loadLimits()
  expect(limits.getSnapshot()).toEqual({ status: 'failed' })
}

beforeEach(async () => {
  vi.resetModules()
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
  limits = await import('./limits')
  session = await import('../session/session')
  client = await import('../api/client')
  session.markSignedIn('analyst01')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('starts idle, and importing it sends no request', () => {
  expect(limits.getSnapshot()).toEqual({ status: 'idle' })
  expect(fetchMock).not.toHaveBeenCalled()
})

describe('loadLimits', () => {
  it('one GET /api/limits with same-origin credentials, then ready with the exact limits', async () => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await limits.loadLimits()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/limits')
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: 'GET', credentials: 'same-origin' })
    expect(limits.getSnapshot()).toEqual(READY)
  })

  it('is loading while it runs', async () => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const load = limits.loadLimits()
    expect(limits.getSnapshot()).toEqual({ status: 'loading' })
    answer.resolve(json(200, BODY))
    await load
    expect(limits.getSnapshot()).toEqual(READY)
  })

  it('concurrent calls share one request and one promise', async () => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const first = limits.loadLimits()
    expect(limits.loadLimits()).toBe(first)
    answer.resolve(json(200, BODY))
    await first
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('after ready it sends nothing', async () => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await limits.loadLimits()
    await limits.loadLimits()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(limits.getSnapshot()).toEqual(READY)
  })

  it('after failed it sends nothing, and it stays failed', async () => {
    await toFailed()
    await limits.loadLimits()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(limits.getSnapshot()).toEqual({ status: 'failed' })
  })

  it.each([
    ['500', () => Promise.resolve(apiError(500, 'internal_error'))],
    ['503', () => Promise.resolve(apiError(503, 'analysis_unavailable'))],
    ['a network failure', () => Promise.reject(new TypeError('network'))],
  ])('%s → failed, and it never rejects', async (_, answer) => {
    fetchMock.mockImplementationOnce(answer)
    await expect(limits.loadLimits()).resolves.toBeUndefined()
    expect(limits.getSnapshot()).toEqual({ status: 'failed' })
  })

  it.each([
    ['not JSON', new Response('<html>', { status: 200 })],
    ['an array', json(200, [])],
    ['no formats', json(200, { ...BODY, formats: undefined })],
    ['empty formats', json(200, { ...BODY, formats: [] })],
    ['upper-case formats', json(200, { ...BODY, formats: ['WAV'] })],
    ['an empty format', json(200, { ...BODY, formats: [''] })],
    ['a size of 0', json(200, { ...BODY, max_size_bytes: 0 })],
    ['a size as a string', json(200, { ...BODY, max_size_bytes: '20971520' })],
    ['a duration of 0', json(200, { ...BODY, max_duration_s: 0 })],
    ['no audio_deletion', json(200, { ...BODY, audio_deletion: undefined })],
    ['an empty audio_deletion', json(200, { ...BODY, audio_deletion: '' })],
  ])('a malformed 200, %s → failed', async (_, response) => {
    fetchMock.mockResolvedValueOnce(response)
    await limits.loadLimits()
    expect(limits.getSnapshot()).toEqual({ status: 'failed' })
  })

  it('extra fields are not kept', async () => {
    fetchMock.mockResolvedValueOnce(json(200, { ...BODY, mock: true }))
    await limits.loadLimits()
    expect(limits.getSnapshot()).toEqual(READY)
  })
})

describe('retryLimits', () => {
  it('from failed: loading, one new request, then ready', async () => {
    await toFailed()
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const retry = limits.retryLimits()
    expect(limits.getSnapshot()).toEqual({ status: 'loading' })
    answer.resolve(json(200, BODY))
    await retry
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(limits.getSnapshot()).toEqual(READY)
  })

  it('only from failed: idle and ready send nothing', async () => {
    await limits.retryLimits()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(limits.getSnapshot()).toEqual({ status: 'idle' })
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await limits.loadLimits()
    await limits.retryLimits()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('the session', () => {
  it('401 session_expired: apiFetch expires the session and the store forgets, never failed', async () => {
    client.setSessionExpiredHandler(session.expire)
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await limits.loadLimits()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
    expect(limits.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('401 with no expiry handler settles on idle, never stuck in loading', async () => {
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await limits.loadLimits()
    expect(limits.getSnapshot()).toEqual({ status: 'idle' })
  })

  it.each([
    ['sign-out', () => session.markSignedOut()],
    ['expiry', () => session.expire()],
    [
      'unavailable',
      async () => {
        fetchMock.mockRejectedValueOnce(new TypeError('network'))
        await session.checkSession()
        expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
      },
    ],
  ])('%s forgets the limits', async (_, end) => {
    fetchMock.mockResolvedValueOnce(json(200, BODY))
    await limits.loadLimits()
    await end()
    expect(limits.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('an answer after the session ended is dropped', async () => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const stale = limits.loadLimits()
    session.expire()
    answer.resolve(json(200, BODY))
    await stale
    expect(limits.getSnapshot()).toEqual({ status: 'idle' })
  })

  it('a new sign-in fetches again', async () => {
    fetchMock.mockResolvedValue(json(200, BODY))
    await limits.loadLimits()
    session.markSignedOut()
    session.markSignedIn('analyst01')
    fetchMock.mockResolvedValueOnce(json(200, { ...BODY, max_duration_s: 30 }))
    await limits.loadLimits()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(limits.getSnapshot()).toEqual({ status: 'ready', limits: { ...BODY, max_duration_s: 30 } })
  })
})
