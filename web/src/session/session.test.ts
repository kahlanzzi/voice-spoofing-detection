import { act, renderHook } from '@testing-library/react'
import { useSyncExternalStore } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules): the store has no reset function, so none can be reached from
// production code. The client is imported from the same fresh registry, so its handler slot is fresh too.

type Session = typeof import('./session')
type Client = typeof import('../api/client')

let session: Session
let client: Client
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const INFO = { username: 'analyst01', signed_in_at: '2026-11-01T09:00:00Z', expires_at: '2026-11-01T21:00:00Z' }

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function apiError(status: number, code: string): Response {
  return json(status, { error: { code, message: 'x', request_id: null } })
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
  session = await import('./session')
  client = await import('../api/client')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('exports', () => {
  it('are exactly the session surface, with no reset', () => {
    expect(Object.keys(session).sort()).toEqual(
      ['checkSession', 'expire', 'getSnapshot', 'markSignedIn', 'markSignedOut', 'startSession', 'subscribe'].sort(),
    )
  })

  it('the initial state is checking', () => {
    expect(session.getSnapshot()).toEqual({ status: 'checking' })
  })
})

describe('checkSession', () => {
  it('sends GET /api/session with same-origin credentials', async () => {
    fetchMock.mockResolvedValue(json(200, INFO))
    await session.checkSession()
    const [path, init] = fetchMock.mock.calls[0] ?? []
    expect(path).toBe('/api/session')
    expect(init?.method).toBe('GET')
    expect(init?.credentials).toBe('same-origin')
  })

  it('200 → signedIn with the username', async () => {
    fetchMock.mockResolvedValue(json(200, INFO))
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
  })

  it.each([
    ['session_expired', apiError(401, 'session_expired')],
    ['a malformed body', new Response('<html>', { status: 401 })],
  ])('401 with %s from checking → signedOut none', async (_, response) => {
    fetchMock.mockResolvedValue(response)
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'none' })
  })

  it('401 while signedIn → signedOut expired', async () => {
    session.markSignedIn('analyst01')
    fetchMock.mockResolvedValue(apiError(401, 'session_expired'))
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
  })

  it('401 from unavailable or signedOut expired → signedOut none', async () => {
    fetchMock.mockResolvedValueOnce(apiError(503, 'sign_in_unavailable'))
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'none' })

    session.expire()
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'none' })
  })

  it.each([
    ['403 access_denied', () => apiError(403, 'access_denied')],
    ['500 internal_error', () => apiError(500, 'internal_error')],
    ['503 sign_in_unavailable', () => apiError(503, 'sign_in_unavailable')],
    ['nginx 503 analysis_unavailable', () => apiError(503, 'analysis_unavailable')],
    ['400', () => apiError(400, 'invalid_request')],
    ['404', () => apiError(404, 'not_found')],
    ['429', () => apiError(429, 'too_many_requests')],
    ['502', () => new Response('Bad Gateway', { status: 502 })],
    ['504', () => new Response('', { status: 504 })],
    ['200 HTML', () => new Response('<!doctype html>', { status: 200 })],
  ])('%s → unavailable', async (_, response) => {
    fetchMock.mockResolvedValue(response())
    await session.checkSession()
    expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
  })

  it.each([
    ['a network TypeError', new TypeError('Failed to fetch')],
    ['an AbortError', new DOMException('Aborted', 'AbortError')],
  ])('fetch rejecting with %s → unavailable, and checkSession resolves', async (_, error) => {
    fetchMock.mockRejectedValue(error)
    await expect(session.checkSession()).resolves.toBeUndefined()
    expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
  })

  it('Retry from unavailable stays unavailable while it runs, with no notify', async () => {
    fetchMock.mockResolvedValueOnce(apiError(503, 'sign_in_unavailable'))
    await session.checkSession()
    const listener = vi.fn()
    session.subscribe(listener)
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const retry = session.checkSession()
    await Promise.resolve()
    expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
    expect(listener).not.toHaveBeenCalled()
    answer.resolve(json(200, INFO))
    await retry
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
    expect(listener).toHaveBeenCalledTimes(1)
  })
})

describe('dedupe', () => {
  it('two concurrent calls share one request; a later call sends another', async () => {
    fetchMock.mockImplementation(async () => json(200, INFO))
    const first = session.checkSession()
    const second = session.checkSession()
    expect(second).toBe(first)
    await Promise.all([first, second])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await session.checkSession()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})

describe('stale guard', () => {
  async function inFlightThen(change: () => Promise<void> | void): Promise<void> {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const check = session.checkSession()
    await change()
    answer.resolve(json(200, INFO))
    await check
  }

  it('an expiry on another route during a check wins', async () => {
    client.setSessionExpiredHandler(session.expire)
    await inFlightThen(async () => {
      fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
      await expect(client.apiFetch('/api/limits')).rejects.toThrow()
    })
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
  })

  it('markSignedOut during a check wins', async () => {
    await inFlightThen(() => session.markSignedOut())
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'none' })
  })

  it('an expire() that changes nothing still wins', async () => {
    session.expire()
    const before = session.getSnapshot()
    await inFlightThen(() => {
      session.expire()
      expect(session.getSnapshot()).toBe(before)
    })
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
  })

  it('a call after the expiry sends a new request', async () => {
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const stale = session.checkSession()
    session.expire()
    fetchMock.mockResolvedValueOnce(json(200, INFO))
    const fresh = session.checkSession()
    expect(fresh).not.toBe(stale)
    await fresh
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
    answer.resolve(apiError(401, 'session_expired'))
    await stale
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
  })
})

describe('expiry handler', () => {
  async function signedInWithHandler(): Promise<void> {
    fetchMock.mockResolvedValueOnce(json(200, INFO))
    await session.startSession()
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
  }

  it('a session_expired 401 on a normal route → signedOut expired', async () => {
    await signedInWithHandler()
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await expect(client.apiFetch('/api/limits')).rejects.toThrow()
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
  })

  it.each([
    ['GET', '/api/session'],
    ['POST', '/api/session'],
    ['DELETE', '/api/session'],
    ['POST', '/api/analyses'],
  ] as const)('a session_expired 401 on %s %s is not an expiry', async (method, path) => {
    await signedInWithHandler()
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await expect(client.apiFetch(path, { method })).rejects.toThrow()
    expect(session.getSnapshot()).toEqual({ status: 'signedIn', username: 'analyst01' })
  })

  it('startSession twice registers once: one transition and one notify per expiry', async () => {
    fetchMock.mockResolvedValue(json(200, INFO))
    await session.startSession()
    await session.startSession()
    const listener = vi.fn()
    session.subscribe(listener)
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await expect(client.apiFetch('/api/limits')).rejects.toThrow()
    expect(listener).toHaveBeenCalledTimes(1)
  })
})

describe('store', () => {
  it('notifies subscribers on a change, not after unsubscribe', () => {
    const listener = vi.fn()
    const unsubscribe = session.subscribe(listener)
    session.expire()
    expect(listener).toHaveBeenCalledTimes(1)
    unsubscribe()
    session.markSignedOut()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('an unsubscribe during notify does not change the current round', () => {
    const calls: string[] = []
    const offB = session.subscribe(() => calls.push('b'))
    const offA = session.subscribe(() => {
      calls.push('a')
      offA()
      offB()
    })
    session.subscribe(() => calls.push('c'))
    session.expire()
    expect(calls.sort()).toEqual(['a', 'b', 'c'])
    calls.length = 0
    session.markSignedOut()
    expect(calls).toEqual(['c'])
  })

  it('a throwing listener does not stop the others; its error is rethrown in a microtask', () => {
    const queued: (() => void)[] = []
    vi.spyOn(globalThis, 'queueMicrotask').mockImplementation((callback) => {
      queued.push(callback)
    })
    const boom = new Error('listener failed')
    const after = vi.fn()
    session.subscribe(() => {
      throw boom
    })
    session.subscribe(after)
    expect(() => session.expire()).not.toThrow()
    expect(after).toHaveBeenCalledTimes(1)
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
    expect(queued).toHaveLength(1)
    expect(() => queued[0]?.()).toThrow(boom)
  })

  it('getSnapshot keeps the same object while unchanged, a new one after a change', () => {
    session.markSignedIn('analyst01')
    const first = session.getSnapshot()
    expect(session.getSnapshot()).toBe(first)
    session.markSignedIn('analyst01')
    expect(session.getSnapshot()).toBe(first)
    session.expire()
    const expired = session.getSnapshot()
    expect(expired).not.toBe(first)
    session.expire()
    expect(session.getSnapshot()).toBe(expired)
  })

  it('works with useSyncExternalStore', () => {
    const { result } = renderHook(() => useSyncExternalStore(session.subscribe, session.getSnapshot))
    expect(result.current).toEqual({ status: 'checking' })
    act(() => session.markSignedIn('analyst01'))
    expect(result.current).toEqual({ status: 'signedIn', username: 'analyst01' })
    act(() => session.expire())
    expect(result.current).toEqual({ status: 'signedOut', reason: 'expired' })
  })
})

describe('memory only', () => {
  it('no storage, cookie or console call through every transition', async () => {
    const storage = (['setItem', 'getItem', 'removeItem', 'clear'] as const).map((name) =>
      vi.spyOn(Storage.prototype, name),
    )
    const cookie = [vi.spyOn(Document.prototype, 'cookie', 'get'), vi.spyOn(Document.prototype, 'cookie', 'set')]
    const consoleSpies = (['log', 'info', 'warn', 'error', 'debug'] as const).map((name) =>
      vi.spyOn(console, name).mockImplementation(() => undefined),
    )

    fetchMock.mockResolvedValueOnce(json(200, INFO))
    await session.startSession()
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await expect(client.apiFetch('/api/limits')).rejects.toThrow()
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await session.checkSession()
    session.markSignedIn('analyst01')
    fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
    await session.checkSession()
    session.markSignedOut()

    for (const spy of [...storage, ...cookie, ...consoleSpies]) expect(spy).not.toHaveBeenCalled()
  })
})
