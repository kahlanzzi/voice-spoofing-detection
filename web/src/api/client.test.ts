import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiFetch, checkApiPath, setSessionExpiredHandler } from './client'
import { ApiError } from './errors'

const ID = 'req_8fa98c02e4b14d0c9a1f3b7d6e5c2a10'

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function expired(): Response {
  return json(401, { error: { code: 'session_expired', message: 'No valid session.', request_id: null } })
}

let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  setSessionExpiredHandler(null)
})

function sent(): { path: string; init: RequestInit; headers: Record<string, string> } {
  const [path, init] = fetchMock.mock.calls[0]!
  return { path: path as string, init: init!, headers: init!.headers as Record<string, string> }
}

describe('checkApiPath', () => {
  it.each([
    '/api/status',
    '/api/session',
    `/api/analyses/${ID}`,
    `/api/analyses/${ID}/note`,
    '/api/analyses?verdict=synthetic&verdict=uncertain&page=2',
    '/api/stats?from=2026-10-24&to=2026-10-30',
  ])('accepts %s', (path) => {
    expect(() => checkApiPath(path)).not.toThrow()
  })

  it.each([
    'https://evil.example/api/status',
    '//evil.example/api/status',
    'api/status',
    '/api',
    '/apis/status',
    '/analyze',
    '/api//status',
    '/api/\\evil',
    '/api/../signin',
    '/api/./status',
    '/api/%2e%2e/signin',
    '/api/%2E./signin',
    '/api/status#x',
    '/api/status x',
    '/api/status\n',
    '/api/status\u0000',
    '',
  ])('refuses %j', (path) => {
    expect(() => checkApiPath(path)).toThrow(/not an \/api\/ path/)
  })

  it('apiFetch refuses before sending', async () => {
    await expect(apiFetch('https://evil.example/api/status')).rejects.toThrow(/not an \/api\/ path/)
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('request', () => {
  it('a GET sends Accept only, same-origin credentials', async () => {
    fetchMock.mockResolvedValue(json(200, { ok: true }))
    await apiFetch('/api/limits')
    const { path, init, headers } = sent()
    expect(path).toBe('/api/limits')
    expect(init.method).toBe('GET')
    expect(init.credentials).toBe('same-origin')
    expect(init.body).toBeUndefined()
    expect(headers).toEqual({ Accept: 'application/json' })
  })

  it('a JSON body sends Content-Type application/json and the body as typed', async () => {
    fetchMock.mockResolvedValue(json(200, {}))
    await apiFetch('/api/session', { method: 'POST', json: { username: ' Analyst01 ', password: ' p w ' } })
    const { init, headers } = sent()
    expect(headers).toEqual({ Accept: 'application/json', 'Content-Type': 'application/json' })
    expect(init.body).toBe('{"username":" Analyst01 ","password":" p w "}')
  })

  it('a FormData body is passed as is, with no Content-Type', async () => {
    fetchMock.mockResolvedValue(json(201, {}))
    const form = new FormData()
    form.append('file', new File(['RIFF'], 'call.wav'))
    await apiFetch('/api/analyses', { method: 'POST', body: form })
    const { init, headers } = sent()
    expect(init.body).toBe(form)
    expect(headers).toEqual({ Accept: 'application/json' })
  })

  it.each([
    ['GET', undefined],
    ['POST', { json: {} }],
    ['PUT', { json: { text: 'x' } }],
    ['DELETE', undefined],
  ] as const)('%s never sets Origin, Cookie or an auth header', async (method, extra) => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    await apiFetch('/api/session', { method, ...extra })
    const names = Object.keys(sent().headers).map((name) => name.toLowerCase())
    expect(names).not.toContain('origin')
    expect(names).not.toContain('cookie')
    expect(names).not.toContain('authorization')
  })

  it('refuses json and body together', async () => {
    await expect(apiFetch('/api/analyses', { method: 'POST', json: {}, body: new FormData() })).rejects.toThrow(
      /not both/,
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('response', () => {
  it('returns the parsed JSON', async () => {
    const session = { username: 'analyst01', signed_in_at: '2026-11-01T09:00:00Z', expires_at: '2026-11-01T21:00:00Z' }
    fetchMock.mockResolvedValue(json(200, session))
    await expect(apiFetch('/api/session')).resolves.toEqual(session)
  })

  it('a 204 returns undefined', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    await expect(apiFetch('/api/session', { method: 'DELETE' })).resolves.toBeUndefined()
  })

  it('a 2xx that is not JSON is an internal_error', async () => {
    fetchMock.mockResolvedValue(new Response('<!doctype html><title>app</title>', { status: 200 }))
    await expect(apiFetch('/api/limits')).rejects.toMatchObject({ kind: 'http', status: 200, code: 'internal_error' })
  })

  it('an error answer throws ApiError with the §6 code', async () => {
    fetchMock.mockResolvedValue(
      json(422, { error: { code: 'corrupt_audio', message: 'The recording could not be decoded.', request_id: ID } }),
    )
    const error = await apiFetch('/api/analyses', { method: 'POST', body: new FormData() }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ kind: 'http', status: 422, code: 'corrupt_audio', requestId: ID })
  })

  it('a fetch TypeError is a network error', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(apiFetch('/api/status')).rejects.toMatchObject({ kind: 'network', code: null, requestId: undefined })
  })

  it('any other fetch failure is rethrown unchanged', async () => {
    const abort = new DOMException('The operation was aborted.', 'AbortError')
    fetchMock.mockRejectedValue(abort)
    await expect(apiFetch('/api/status')).rejects.toBe(abort)
  })
})

describe('session expired handler', () => {
  it.each([
    ['GET', '/api/analyses?page=2'],
    ['GET', `/api/analyses/${ID}`],
    ['PUT', `/api/analyses/${ID}/note`],
    ['GET', '/api/status'],
    ['GET', '/api/limits'],
    ['GET', '/api/stats'],
  ] as const)('%s %s calls it on session_expired', async (method, path) => {
    const handler = vi.fn()
    setSessionExpiredHandler(handler)
    fetchMock.mockResolvedValue(expired())
    await expect(apiFetch(path, { method })).rejects.toMatchObject({ code: 'session_expired' })
    expect(handler).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['GET', '/api/session'],
    ['POST', '/api/session'],
    ['DELETE', '/api/session'],
    ['POST', '/api/analyses'],
  ] as const)('%s %s does not call it', async (method, path) => {
    const handler = vi.fn()
    setSessionExpiredHandler(handler)
    fetchMock.mockResolvedValue(expired())
    await expect(apiFetch(path, { method })).rejects.toMatchObject({ code: 'session_expired' })
    expect(handler).not.toHaveBeenCalled()
  })

  it('invalid_credentials does not call it', async () => {
    const handler = vi.fn()
    setSessionExpiredHandler(handler)
    fetchMock.mockResolvedValue(
      json(401, { error: { code: 'invalid_credentials', message: 'x', request_id: null } }),
    )
    await expect(apiFetch('/api/status')).rejects.toMatchObject({ code: 'invalid_credentials' })
    expect(handler).not.toHaveBeenCalled()
  })

  it('a 401 with a malformed body does not call it', async () => {
    const handler = vi.fn()
    setSessionExpiredHandler(handler)
    fetchMock.mockResolvedValue(new Response('Unauthorized', { status: 401 }))
    await expect(apiFetch('/api/status')).rejects.toMatchObject({ code: 'internal_error' })
    expect(handler).not.toHaveBeenCalled()
  })
})
