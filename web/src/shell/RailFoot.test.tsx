import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import type { SessionState } from '../session/session'
import { THEME_KEY } from '../theme'
import { SIGN_OUT_FAILED } from './RailFoot'

// The rail footer (Stage 18 PR 6b) in the real route tree, with a stand-in session store: signOut is a spy whose
// answer the test controls; a successful one moves the store to signedOut 'signOut', as session.ts does.

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

const signOut = vi.hoisted(() => vi.fn<() => Promise<boolean>>())

// The upload limits are a stand-in here, ready and inert: the shell's limits request is pinned in Shell.limits.test.tsx.
vi.mock('../upload/limits', () => {
  const ready = { status: 'ready', limits: { formats: ['wav'], max_size_bytes: 1048576, max_duration_s: 60, audio_deletion: 'Deleted' } }
  return { getSnapshot: () => ready, subscribe: () => () => undefined, loadLimits: vi.fn(), retryLimits: vi.fn() }
})

// History's GET is pinned in pages/History.test.tsx; here the page stays on its skeleton and sends nothing.
vi.mock('../history/history', () => ({ loadHistory: () => new Promise(() => undefined) }))

vi.mock('../session/session', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  checkSession: vi.fn(),
  expire: vi.fn(),
  markSignedIn: vi.fn(),
  markSignedOut: vi.fn(),
  signOut,
  startSession: vi.fn(),
}))

const SIGNED_IN: SessionState = { status: 'signedIn', username: 'analyst01' }

function deferred(): { promise: Promise<boolean>; resolve: (done: boolean) => void } {
  let resolve: (done: boolean) => void = () => undefined
  const promise = new Promise<boolean>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

beforeEach(() => {
  signOut.mockReset()
  store.set(SIGNED_IN)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
})

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return router
}

const foot = () => document.querySelector<HTMLElement>('.rail-foot')!
const themeSelect = () => within(foot()).getByRole<HTMLSelectElement>('combobox', { name: 'Theme' })
const signOutButton = () => within(foot()).getByRole('button', { name: 'Sign out' })

describe('identity and order', () => {
  it('the username, "Analyst ID", the theme, then Sign out (prototype lines 812-813)', () => {
    renderAt('/analyze')
    expect(foot().querySelector('.who-name')?.textContent).toBe('analyst01')
    expect(foot().querySelector('.who-role')?.textContent).toBe('Analyst ID')
    const order = [...foot().querySelectorAll('.who-name, select, button')].map((element) => element.tagName)
    expect(order).toEqual(['DIV', 'SELECT', 'BUTTON'])
    expect(screen.queryByRole('alert')).toBeNull()
  })
})

describe('theme', () => {
  it('a native select: Follow the system setting, Light, Dark (prototype line 485)', () => {
    renderAt('/analyze')
    expect([...themeSelect().options].map((option) => [option.value, option.textContent])).toEqual([
      ['system', 'Follow the system setting'],
      ['light', 'Light'],
      ['dark', 'Dark'],
    ])
    expect(themeSelect().value).toBe('system')
  })

  it('shows the stored choice', () => {
    window.localStorage.setItem(THEME_KEY, 'dark')
    renderAt('/analyze')
    expect(themeSelect().value).toBe('dark')
  })

  it('a choice is stored and applied; System removes data-theme', async () => {
    renderAt('/analyze')
    await userEvent.selectOptions(themeSelect(), 'dark')
    expect(window.localStorage.getItem(THEME_KEY)).toBe('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    await userEvent.selectOptions(themeSelect(), 'light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
    await userEvent.selectOptions(themeSelect(), 'system')
    expect(window.localStorage.getItem(THEME_KEY)).toBe('system')
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false)
    expect(themeSelect().value).toBe('system')
  })

  it('storage that throws: the choice still applies for this page view', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    renderAt('/analyze')
    await userEvent.selectOptions(themeSelect(), 'dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    expect(themeSelect().value).toBe('dark')
  })
})

describe('sign out', () => {
  it('busy: label kept, spinner, disabled, aria-busy; a double click calls signOut once', async () => {
    const answer = deferred()
    signOut.mockReturnValue(answer.promise)
    renderAt('/analyze')
    await userEvent.dblClick(signOutButton())
    expect(signOut).toHaveBeenCalledTimes(1)
    const button = signOutButton()
    expect(button.textContent).toBe('Sign out')
    expect(button.querySelector('.spin[aria-hidden="true"]')).not.toBeNull()
    expect((button as HTMLButtonElement).disabled).toBe(true)
    expect(button.getAttribute('aria-busy')).toBe('true')
    await act(async () => answer.resolve(false))
  })

  it('success → the guard sends the analyst to /signin, no next, no alert', async () => {
    signOut.mockImplementation(async () => {
      store.set({ status: 'signedOut', reason: 'signOut' })
      return true
    })
    const router = renderAt('/history?verdict=a')
    await userEvent.click(signOutButton())
    expect(router.state.location.pathname + router.state.location.search).toBe('/signin')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Sign in to the analyst console')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('failure: still in the shell, the alert, focus back on an enabled Sign out', async () => {
    signOut.mockResolvedValue(false)
    const router = renderAt('/history')
    await userEvent.click(signOutButton())
    expect(screen.getByRole('alert').textContent).toBe(SIGN_OUT_FAILED)
    expect(SIGN_OUT_FAILED).toBe('Sign-out did not complete. You are still signed in. Try again.')
    expect(router.state.location.pathname).toBe('/history')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('History')
    const button = signOutButton() as HTMLButtonElement
    expect(button.disabled).toBe(false)
    expect(button.hasAttribute('aria-busy')).toBe(false)
    expect(button.querySelector('.spin')).toBeNull()
    expect(document.activeElement).toBe(button)
  })

  it('a second failure mounts the alert again, so it is announced again', async () => {
    signOut.mockResolvedValue(false)
    renderAt('/history')
    await userEvent.click(signOutButton())
    const first = screen.getByRole('alert')
    await userEvent.click(signOutButton())
    const second = screen.getByRole('alert')
    expect(second).not.toBe(first)
    expect(second.textContent).toBe(SIGN_OUT_FAILED)
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(signOut).toHaveBeenCalledTimes(2)
  })
})
