import { act, cleanup, render, screen } from '@testing-library/react'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SessionState } from '../session/session'
import * as session from '../session/session'
import { routes } from './router'

// The guard against a stand-in session store: the test sets the state; the store's transitions are spies
// that must never be called, because the guard only navigates.

const store = vi.hoisted(() => {
  let state: SessionState = { status: 'checking' }
  const listeners = new Set<() => void>()
  return {
    get: (): SessionState => state,
    set(next: SessionState): void {
      state = next
      for (const listener of [...listeners]) listener()
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
})

vi.mock('../session/session', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  checkSession: vi.fn(),
  expire: vi.fn(),
  markSignedIn: vi.fn(),
  markSignedOut: vi.fn(),
  startSession: vi.fn(),
}))

const ID = 'req_' + '0123456789abcdef'.repeat(2)
const CHECKING: SessionState = { status: 'checking' }
const SIGNED_IN: SessionState = { status: 'signedIn', username: 'analyst01' }
const SIGNED_OUT: SessionState = { status: 'signedOut', reason: 'none' }
const EXPIRED: SessionState = { status: 'signedOut', reason: 'expired' }
const UNAVAILABLE: SessionState = { status: 'unavailable' }

const FILTERS = '/history?verdict=a&verdict=b'
const FILTERS_NEXT = '?next=%2Fhistory%3Fverdict%3Da%26verdict%3Db'

function renderAt(entry: string, state: SessionState) {
  store.set(state)
  const router = createMemoryRouter(routes, { initialEntries: [entry] })
  render(<RouterProvider router={router} />)
  return router
}

function heading(): string | null {
  return screen.queryByRole('heading', { level: 1 })?.textContent ?? null
}

const S1_TITLE = 'Sign in to the analyst console'

// Which S1 variant is on screen (PR 4): Retry only when unavailable, the notice only when expired.
function s1Variant(): string {
  if (screen.queryByRole('button', { name: 'Retry' })) return 'sign-in service unavailable'
  if (screen.queryByText(/Your session expired/)) return 'expired session'
  return 'default'
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn<typeof fetch>())
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('a protected route', () => {
  it('checking: the checking line only, no S1, no page, URL unchanged', () => {
    const router = renderAt(FILTERS, CHECKING)
    expect(screen.getByRole('status').textContent).toBe('Checking your session…')
    expect(heading()).toBeNull()
    expect(router.state.location.pathname + router.state.location.search).toBe(FILTERS)
  })

  it('signedIn: the page', () => {
    renderAt(FILTERS, SIGNED_IN)
    expect(heading()).toBe('History')
  })

  it.each([
    ['signedOut none', SIGNED_OUT, 'default'],
    ['signedOut expired', EXPIRED, 'expired session'],
    ['unavailable', UNAVAILABLE, 'sign-in service unavailable'],
  ])('%s → /signin with next = path + query', async (_label, state, variant) => {
    const router = renderAt(FILTERS, state)
    expect(await screen.findByRole('heading', { level: 1, name: S1_TITLE })).toBeTruthy()
    expect(s1Variant()).toBe(variant)
    expect(router.state.location.pathname).toBe('/signin')
    expect(router.state.location.search).toBe(FILTERS_NEXT)
    expect(new URLSearchParams(router.state.location.search).get('next')).toBe(FILTERS)
    expect(router.state.historyAction).toBe('REPLACE')
  })

  it('expiry mid-page sends the analyst to sign-in with next', async () => {
    const router = renderAt(FILTERS, SIGNED_IN)
    expect(heading()).toBe('History')
    act(() => store.set(EXPIRED))
    expect(await screen.findByRole('heading', { level: 1, name: S1_TITLE })).toBeTruthy()
    expect(s1Variant()).toBe('expired session')
    expect(router.state.location.search).toBe(FILTERS_NEXT)
  })
})

describe('/signin', () => {
  it('checking: the checking line, no S1', () => {
    renderAt(`/signin${FILTERS_NEXT}`, CHECKING)
    expect(screen.getByRole('status').textContent).toBe('Checking your session…')
    expect(heading()).toBeNull()
  })

  it.each([
    ['signedOut none', SIGNED_OUT, 'default'],
    ['signedOut expired', EXPIRED, 'expired session'],
    ['unavailable', UNAVAILABLE, 'sign-in service unavailable'],
  ])('%s: S1 with its variant, next kept', (_label, state, variant) => {
    const router = renderAt(`/signin${FILTERS_NEXT}`, state)
    expect(heading()).toBe(S1_TITLE)
    expect(s1Variant()).toBe(variant)
    expect(router.state.location.search).toBe(FILTERS_NEXT)
  })

  it('signedIn with next=/history?verdict=a&verdict=b lands there, repeated params intact', async () => {
    const router = renderAt(`/signin${FILTERS_NEXT}`, SIGNED_IN)
    expect(await screen.findByRole('heading', { level: 1, name: 'History' })).toBeTruthy()
    expect(router.state.location.pathname).toBe('/history')
    expect(router.state.location.search).toBe('?verdict=a&verdict=b')
    expect(router.state.historyAction).toBe('REPLACE')
  })

  it.each([
    ['no next', '/signin'],
    ['empty next', '/signin?next='],
    ['scheme-relative next', '/signin?next=%2F%2Fevil.test'],
    ['absolute next', '/signin?next=https%3A%2F%2Fevil.test%2Fanalyze'],
    ['next to sign-in', '/signin?next=%2Fsignin'],
  ])('signedIn with %s → /analyze', async (_label, entry) => {
    const router = renderAt(entry, SIGNED_IN)
    expect(await screen.findByRole('heading', { level: 1, name: 'Analyze' })).toBeTruthy()
    expect(router.state.location.pathname).toBe('/analyze')
    expect(router.state.location.search).toBe('')
  })
})

describe('no sign-in flash while checking', () => {
  it.each(['/analyze', `/signin?next=%2Fhistory`])('%s shows no S1 until the check answers', async (entry) => {
    renderAt(entry, CHECKING)
    expect(screen.queryByRole('heading', { name: S1_TITLE })).toBeNull()
    expect(screen.queryByLabelText('Analyst ID')).toBeNull()
    act(() => store.set(SIGNED_OUT))
    expect(await screen.findByRole('heading', { level: 1, name: S1_TITLE })).toBeTruthy()
  })
})

describe('paths that are not app routes', () => {
  it.each([
    '/nope',
    '/analyze/',
    '/Analyze',
    '/SIGNIN',
    '/history/foo',
    `/history/${ID.slice(0, -1)}`,
    `/history/${ID.toUpperCase()}`,
    `/history/${ID}/`,
    '/index.html',
  ])('%s → not found, whatever the session', (entry) => {
    for (const state of [CHECKING, SIGNED_IN, SIGNED_OUT, EXPIRED, UNAVAILABLE]) {
      const router = renderAt(entry, state)
      expect(heading()).toBe('Page not found')
      expect(router.state.location.pathname).toBe(entry)
      cleanup()
    }
  })

  it('/ → /analyze, as nginx', async () => {
    const router = renderAt('/', SIGNED_IN)
    expect(await screen.findByRole('heading', { level: 1, name: 'Analyze' })).toBeTruthy()
    expect(router.state.location.pathname).toBe('/analyze')
  })
})

describe('pages', () => {
  it.each([
    ['/analyze', 'Analyze'],
    ['/history', 'History'],
    [`/history/${ID}`, 'Analysis details'],
    ['/overview', 'Overview'],
    ['/status', 'System status'],
  ])('%s has one h1, %s', (entry, title) => {
    renderAt(entry, SIGNED_IN)
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    expect(heading()).toBe(title)
  })

  it('details accepts a full request ID', () => {
    renderAt(`/history/${ID}`, SIGNED_IN)
    expect(screen.getByText(ID).tagName).toBe('CODE')
  })
})

describe('the guard has no side effects beyond navigation', () => {
  it('no session call, no fetch, no storage through every state and route', async () => {
    const storage = (['getItem', 'setItem', 'removeItem', 'clear'] as const).map((method) =>
      vi.spyOn(Storage.prototype, method),
    )
    for (const entry of [FILTERS, `/signin${FILTERS_NEXT}`, '/signin', '/nope', '/']) {
      for (const state of [CHECKING, SIGNED_IN, SIGNED_OUT, EXPIRED, UNAVAILABLE]) {
        renderAt(entry, state)
        await act(async () => undefined)
        cleanup()
      }
    }
    const router = renderAt(FILTERS, SIGNED_IN)
    act(() => store.set(EXPIRED))
    act(() => store.set(UNAVAILABLE))
    act(() => store.set(SIGNED_IN))
    expect(await screen.findByRole('heading', { level: 1, name: 'History' })).toBeTruthy()
    expect(router.state.location.pathname + router.state.location.search).toBe(FILTERS)

    for (const transition of [
      session.checkSession,
      session.expire,
      session.markSignedIn,
      session.markSignedOut,
      session.startSession,
    ]) {
      expect(transition).not.toHaveBeenCalled()
    }
    expect(fetch).not.toHaveBeenCalled()
    // The only storage access is the router's own: at creation it reads the view transitions it applied
    // (react-router lib/router/router.js restoreAppliedTransitions); it writes them only after a view
    // transition, which the app does not use. Nothing is written, and next never reaches storage.
    const [getItem, ...writes] = storage
    expect(new Set(getItem?.mock.calls.map(([key]) => key))).toEqual(new Set(['remix-router-transitions']))
    for (const spy of writes) expect(spy).not.toHaveBeenCalled()
  })
})
