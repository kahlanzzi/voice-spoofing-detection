import { act, cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import type { SessionState } from '../session/session'

// The page-change rules (HANDOFF lines 67-71, 174) through the real route tree, with a stand-in session store.

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
  signOut: vi.fn(),
  startSession: vi.fn(),
}))

const ID = 'req_' + '0123456789abcdef'.repeat(2)
const OTHER_ID = 'req_' + 'fedcba9876543210'.repeat(2)
const SIGNED_IN: SessionState = { status: 'signedIn', username: 'analyst01' }
const SIGNED_OUT: SessionState = { status: 'signedOut', reason: 'none' }
const EXPIRED: SessionState = { status: 'signedOut', reason: 'expired' }
const SIGNED_OUT_BY_ANALYST: SessionState = { status: 'signedOut', reason: 'signOut' }

let scrollTo: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  scrollTo = vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
  document.title = 'Voice Check analyst console'
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderAt(entries: string[], state: SessionState, strict = false) {
  store.set(state)
  const router = createMemoryRouter(routes, { initialEntries: entries, initialIndex: entries.length - 1 })
  const app = <RouterProvider router={router} />
  render(strict ? <StrictMode>{app}</StrictMode> : app)
  return router
}

const h1 = () => screen.getByRole('heading', { level: 1 })
const announced = () => document.getElementById('page-announcer')?.textContent ?? null
const announcedNode = () => document.getElementById('page-announcer')?.firstChild ?? null

async function go(router: ReturnType<typeof renderAt>, to: string | number) {
  await act(async () => {
    await (typeof to === 'number' ? router.navigate(to) : router.navigate(to))
  })
}

describe('first load', () => {
  it.each([false, true])('sets the title only: no focus, no announcement, no scroll (StrictMode %s)', (strict) => {
    renderAt(['/overview'], SIGNED_IN, strict)
    expect(document.title).toBe('Overview · Voice Check analyst console')
    expect(document.activeElement).toBe(document.body)
    expect(announced()).toBe('')
    expect(scrollTo).not.toHaveBeenCalled()
  })

  it('/ → /analyze on load is still the first page', () => {
    renderAt(['/'], SIGNED_IN)
    expect(h1().textContent).toBe('Analyze a recording')
    expect(document.title).toBe('Analyze a recording · Voice Check analyst console')
    expect(document.activeElement).toBe(document.body)
    expect(announced()).toBe('')
  })

  it('Checking, then the page at the same path: the first page, and later moves are page changes', async () => {
    const router = renderAt(['/history'], { status: 'checking' })
    expect(document.title).toBe('Voice Check analyst console')
    act(() => store.set(SIGNED_IN))
    expect(document.title).toBe('History · Voice Check analyst console')
    expect(document.activeElement).toBe(document.body)
    expect(announced()).toBe('')
    await go(router, '/overview')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Overview page')
  })

  it('a protected URL without a session lands on S1 as the first page', () => {
    renderAt(['/history'], SIGNED_OUT)
    expect(document.title).toBe('Sign in · Voice Check analyst console')
    expect(document.activeElement).toBe(document.body)
    expect(announced()).toBe('')
  })

  it('an unknown app path gets the Page not found title', () => {
    renderAt(['/history/foo'], SIGNED_IN)
    expect(document.title).toBe('Page not found · Voice Check analyst console')
  })
})

describe('in-app navigation', () => {
  it('a rail link: focus on the h1, announcement, title, scroll to the top', async () => {
    renderAt(['/analyze'], SIGNED_IN)
    await userEvent.click(screen.getByRole('link', { name: 'History' }))
    expect(h1().textContent).toBe('History')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('History page')
    expect(document.title).toBe('History · Voice Check analyst console')
    expect(scrollTo).toHaveBeenCalledWith(0, 0)
  })

  it.each([
    ['/history', 'History', 'History page'],
    [`/history/${ID}`, 'Analysis details', 'Analysis details page'],
    ['/overview', 'Overview', 'Overview page'],
    ['/status', 'System status', 'System status page'],
  ])('to %s', async (path, title, message) => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, path)
    expect(document.activeElement).toBe(h1())
    expect(h1().textContent).toBe(title)
    expect(announced()).toBe(message)
    expect(document.title).toBe(`${title} · Voice Check analyst console`)
  })

  it('details to other details: the same text is announced again in a new node', async () => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, `/history/${ID}`)
    const first = announcedNode()
    await go(router, `/history/${OTHER_ID}`)
    expect(announced()).toBe('Analysis details page')
    expect(announcedNode()).not.toBe(first)
    expect(document.activeElement).toBe(h1())
  })

  it('a query-only change is not a page change', async () => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, '/history')
    scrollTo.mockClear()
    const node = announcedNode()
    const button = screen.getAllByRole('button', { name: 'New analysis' })[0]!
    button.focus()
    await go(router, '/history?verdict=synthetic&page=2')
    expect(document.activeElement).toBe(button)
    expect(announcedNode()).toBe(node)
    expect(scrollTo).not.toHaveBeenCalled()
  })

  it('a link to the current page changes nothing', async () => {
    renderAt(['/overview'], SIGNED_IN)
    const link = screen.getByRole('link', { name: 'Overview' })
    await userEvent.click(link)
    expect(document.activeElement).toBe(link)
    expect(announced()).toBe('')
    expect(scrollTo).not.toHaveBeenCalled()
  })
})

describe('Back and Forward', () => {
  it('Back to another page: the same rules', async () => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, '/overview')
    scrollTo.mockClear()
    await go(router, -1)
    expect(router.state.historyAction).toBe('POP')
    expect(h1().textContent).toBe('Analyze a recording')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Analyze a recording page')
    expect(scrollTo).toHaveBeenCalledWith(0, 0)
    await go(router, 1)
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Overview page')
  })

  it('Back to History: focus and announcement, the scroll is left to PR 12', async () => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, '/history')
    await go(router, `/history/${ID}`)
    scrollTo.mockClear()
    await go(router, -1)
    expect(h1().textContent).toBe('History')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('History page')
    expect(scrollTo).not.toHaveBeenCalled()
  })

  it('a push to History still scrolls to the top', async () => {
    const router = renderAt(['/analyze'], SIGNED_IN)
    await go(router, '/history')
    expect(scrollTo).toHaveBeenCalledWith(0, 0)
  })
})

describe('sign-in and expiry', () => {
  it('S1 → the page after sign-in is a page change', async () => {
    renderAt(['/signin'], SIGNED_OUT)
    await act(async () => store.set(SIGNED_IN))
    expect(h1().textContent).toBe('Analyze a recording')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Analyze a recording page')
  })

  it('expiry mid-page: focus on the S1 h1 and the expiry message', async () => {
    renderAt(['/history'], SIGNED_IN)
    await act(async () => store.set(EXPIRED))
    expect(h1().textContent).toBe('Sign in to the analyst console')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Your session expired. Sign in again to continue.')
    expect(document.title).toBe('Sign in · Voice Check analyst console')
  })

  it('sign-out mid-page: focus on the S1 h1, "Signed out." (prototype line 1496), scroll to the top', async () => {
    renderAt(['/history'], SIGNED_IN)
    scrollTo.mockClear()
    await act(async () => store.set(SIGNED_OUT_BY_ANALYST))
    expect(h1().textContent).toBe('Sign in to the analyst console')
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('Signed out.')
    expect(document.title).toBe('Sign in · Voice Check analyst console')
    expect(scrollTo).toHaveBeenCalledWith(0, 0)
  })

  it('a first load of /signin after a sign-out says nothing', () => {
    renderAt(['/analyze'], SIGNED_OUT_BY_ANALYST)
    expect(h1().textContent).toBe('Sign in to the analyst console')
    expect(announced()).toBe('')
  })
})
