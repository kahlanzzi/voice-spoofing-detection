import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// A fresh module per test (vi.resetModules), as in session.test.ts: the store has no reset function. The session
// store comes from the same fresh registry, so the draft's session subscription is fresh too.

type Draft = typeof import('./fileDraft')
type Session = typeof import('../session/session')

let draft: Draft
let session: Session
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>

const FILE = new File(['audio'], 'call-with-a-customer.wav', { type: 'audio/wav' })
const OTHER_FILE = new File(['audio'], 'other.flac', { type: 'audio/flac' })
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

function withDraft(): void {
  session.markSignedIn('analyst01')
  draft.setFileDraft(FILE)
  expect(draft.getSnapshot()).toBe(FILE)
}

async function toUnavailable(): Promise<void> {
  fetchMock.mockRejectedValueOnce(new TypeError('network'))
  await session.checkSession()
  expect(session.getSnapshot()).toEqual({ status: 'unavailable' })
}

// A 401 while not signed in: the plain sign-in screen.
async function toSignedOutNone(): Promise<void> {
  session.expire()
  fetchMock.mockResolvedValueOnce(apiError(401, 'session_expired'))
  await session.checkSession()
  expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'none' })
}

beforeEach(async () => {
  vi.resetModules()
  fetchMock = vi.fn<typeof fetch>()
  vi.stubGlobal('fetch', fetchMock)
  session = await import('../session/session')
  draft = await import('./fileDraft')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  localStorage.clear()
  sessionStorage.clear()
})

describe('exports', () => {
  it('are exactly the store surface, with no reset', () => {
    expect(Object.keys(draft).sort()).toEqual(['clearFileDraft', 'getSnapshot', 'setFileDraft', 'subscribe'])
  })
})

describe('the store', () => {
  it('starts empty', () => {
    expect(draft.getSnapshot()).toBeNull()
  })

  it('keeps the same File object and notifies once', () => {
    session.markSignedIn('analyst01')
    const listener = vi.fn()
    draft.subscribe(listener)
    draft.setFileDraft(FILE)
    expect(draft.getSnapshot()).toBe(FILE)
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('setting the same File again does not notify', () => {
    withDraft()
    const listener = vi.fn()
    draft.subscribe(listener)
    draft.setFileDraft(FILE)
    expect(listener).not.toHaveBeenCalled()
  })

  it('another File replaces it', () => {
    withDraft()
    draft.setFileDraft(OTHER_FILE)
    expect(draft.getSnapshot()).toBe(OTHER_FILE)
  })

  it('clearing notifies; clearing nothing does not', () => {
    withDraft()
    const listener = vi.fn()
    draft.subscribe(listener)
    draft.clearFileDraft()
    expect(draft.getSnapshot()).toBeNull()
    expect(listener).toHaveBeenCalledTimes(1)
    draft.clearFileDraft()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('an unsubscribed listener is not called', () => {
    session.markSignedIn('analyst01')
    const listener = vi.fn()
    draft.subscribe(listener)()
    draft.setFileDraft(FILE)
    expect(listener).not.toHaveBeenCalled()
  })

  it('a throwing listener does not stop the others; its error is rethrown in a microtask', () => {
    const queued: (() => void)[] = []
    vi.spyOn(globalThis, 'queueMicrotask').mockImplementation((callback) => {
      queued.push(callback)
    })
    session.markSignedIn('analyst01')
    const boom = new Error('listener failed')
    const after = vi.fn()
    draft.subscribe(() => {
      throw boom
    })
    draft.subscribe(after)
    expect(() => draft.setFileDraft(FILE)).not.toThrow()
    expect(after).toHaveBeenCalledTimes(1)
    expect(draft.getSnapshot()).toBe(FILE)
    expect(queued).toHaveLength(1)
    expect(() => queued[0]?.()).toThrow(boom)
  })
})

describe('a File set while not signed in is not kept', () => {
  it.each<[string, () => void | Promise<void>]>([
    ['checking', () => undefined],
    ['expired', () => session.expire()],
    ['signed out', () => session.markSignedOut()],
    ['signed out, no reason', () => toSignedOutNone()],
    ['unavailable', () => toUnavailable()],
  ])('%s', async (_label, reach) => {
    await reach()
    draft.setFileDraft(FILE)
    expect(draft.getSnapshot()).toBeNull()
  })
})

describe('explicit sign-out clears it', () => {
  it('markSignedOut clears and notifies', () => {
    withDraft()
    const listener = vi.fn()
    draft.subscribe(listener)
    session.markSignedOut()
    expect(draft.getSnapshot()).toBeNull()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['204', () => new Response(null, { status: 204 })],
    ['401', () => apiError(401, 'session_expired')],
  ])('signOut() answered %s clears', async (_label, answer) => {
    withDraft()
    fetchMock.mockResolvedValueOnce(answer())
    expect(await session.signOut()).toBe(true)
    expect(draft.getSnapshot()).toBeNull()
  })

  it.each([
    ['503', () => Promise.resolve(apiError(503, 'service_unavailable'))],
    ['403', () => Promise.resolve(apiError(403, 'forbidden_origin'))],
    ['no answer', () => Promise.reject(new TypeError('network'))],
  ])('signOut() answered %s keeps it: still signed in', async (_label, answer) => {
    withDraft()
    fetchMock.mockImplementationOnce(answer)
    expect(await session.signOut()).toBe(false)
    expect(draft.getSnapshot()).toBe(FILE)
  })

  it('expiry, then sign-out from S1 state clears', () => {
    withDraft()
    session.expire()
    session.markSignedOut()
    expect(draft.getSnapshot()).toBeNull()
  })

  it('the next analyst starts empty, even the same one', () => {
    withDraft()
    session.markSignedOut()
    session.markSignedIn('analyst01')
    expect(draft.getSnapshot()).toBeNull()
  })

  // Entry 143 Q14: the rule is keyed on markSignedOut, which a stale DELETE answer does not reach.
  it('an expiry during the DELETE keeps the File: the expiry wins', async () => {
    withDraft()
    const answer = deferred()
    fetchMock.mockReturnValueOnce(answer.promise)
    const call = session.signOut()
    session.expire()
    answer.resolve(new Response(null, { status: 204 }))
    expect(await call).toBe(true)
    expect(session.getSnapshot()).toEqual({ status: 'signedOut', reason: 'expired' })
    expect(draft.getSnapshot()).toBe(FILE)
  })
})

describe('expiry keeps it, and the next sign-in compares usernames', () => {
  it('expiry keeps the File and does not notify', () => {
    withDraft()
    const listener = vi.fn()
    draft.subscribe(listener)
    session.expire()
    expect(draft.getSnapshot()).toBe(FILE)
    expect(listener).not.toHaveBeenCalled()
  })

  it('expiry then the same username keeps', () => {
    withDraft()
    session.expire()
    session.markSignedIn('analyst01')
    expect(draft.getSnapshot()).toBe(FILE)
  })

  it('expiry then a different username clears', () => {
    withDraft()
    session.expire()
    session.markSignedIn('analyst02')
    expect(draft.getSnapshot()).toBeNull()
  })

  it('expiry, a different username, then the first again stays cleared', () => {
    withDraft()
    session.expire()
    session.markSignedIn('analyst02')
    session.markSignedIn('analyst01')
    expect(draft.getSnapshot()).toBeNull()
  })

  it('a File chosen after a same-username return belongs to that analyst', () => {
    withDraft()
    session.expire()
    session.markSignedIn('analyst01')
    draft.setFileDraft(OTHER_FILE)
    session.expire()
    session.markSignedIn('analyst02')
    expect(draft.getSnapshot()).toBeNull()
  })

  it('a check that answers with another username clears', async () => {
    withDraft()
    session.expire()
    fetchMock.mockResolvedValueOnce(json(200, { ...INFO, username: 'analyst02' }))
    await session.checkSession()
    expect(draft.getSnapshot()).toBeNull()
  })

  it('a check that answers with the same username keeps', async () => {
    withDraft()
    session.expire()
    fetchMock.mockResolvedValueOnce(json(200, INFO))
    await session.checkSession()
    expect(draft.getSnapshot()).toBe(FILE)
  })
})

describe('a failed check never clears it (A19)', () => {
  it.each([
    ['the same username keeps', 'analyst01', FILE],
    ['a different username clears', 'analyst02', null],
  ])('unavailable keeps it; then %s', async (_label, username, expected) => {
    withDraft()
    await toUnavailable()
    expect(draft.getSnapshot()).toBe(FILE)
    session.markSignedIn(username)
    expect(draft.getSnapshot()).toBe(expected)
  })

  it.each([
    ['the same username keeps', 'analyst01', FILE],
    ['a different username clears', 'analyst02', null],
  ])('signed out with no reason keeps it; then %s', async (_label, username, expected) => {
    withDraft()
    await toSignedOutNone()
    expect(draft.getSnapshot()).toBe(FILE)
    session.markSignedIn(username)
    expect(draft.getSnapshot()).toBe(expected)
  })
})

describe('memory only', () => {
  it('a reload loses it', async () => {
    withDraft()
    vi.resetModules()
    const reloaded = await import('./fileDraft')
    expect(reloaded.getSnapshot()).toBeNull()
  })

  it('never touches storage or the console', () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    const logged = (['log', 'info', 'warn', 'error', 'debug'] as const).map((method) => vi.spyOn(console, method))
    withDraft()
    session.expire()
    session.markSignedIn('analyst02')
    expect(setItem).not.toHaveBeenCalled()
    for (const spy of logged) expect(spy).not.toHaveBeenCalled()
    expect(localStorage.length).toBe(0)
    expect(sessionStorage.length).toBe(0)
  })
})
