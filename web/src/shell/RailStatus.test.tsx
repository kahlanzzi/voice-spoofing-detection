import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter, MemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import type { StatusResult, StatusState } from '../status/status'
import RailStatus from './RailStatus'

// The rail status line (Stage 18 PR 7a) with a stand-in status store whose state the test sets: the line only reads
// the store. checkStatus is a spy, so a test can see who starts a check; fetch is a spy that must never be called.

const store = vi.hoisted(() => {
  let state: StatusState = { checking: true, result: null }
  const listeners = new Set<() => void>()
  return {
    get: (): StatusState => state,
    set(next: StatusState): void {
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

const checkStatus = vi.hoisted(() => vi.fn<() => Promise<void>>())

vi.mock('../status/status', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  checkStatus,
}))

const SIGNED_IN = vi.hoisted(() => ({ status: 'signedIn', username: 'analyst01' }) as const)

vi.mock('../session/session', () => ({
  getSnapshot: () => SIGNED_IN,
  subscribe: () => () => undefined,
  checkSession: vi.fn(),
  expire: vi.fn(),
  markSignedIn: vi.fn(),
  markSignedOut: vi.fn(),
  signOut: vi.fn(),
  startSession: vi.fn(),
}))

const DETAILS = {
  versions: { model_version: 'mock', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' },
  bundleManifestSha256: null,
  deferralOnFit: 0.0993,
}

const READY: StatusResult = {
  kind: 'ready',
  notice: false,
  failing: [],
  mock: true,
  checkedAt: '2026-11-03T09:14:05Z',
  timeSource: 'api',
  details: DETAILS,
}

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  checkStatus.mockReset()
  checkStatus.mockResolvedValue(undefined)
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  store.set({ checking: true, result: null })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderShell(path = '/analyze') {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return router
}

const rail = () => screen.getByRole('complementary', { name: 'Main menu' })
const line = () => rail().querySelector<HTMLAnchorElement>('a.svc')!

describe('what it shows', () => {
  it('no result yet: Checking…, with the spinner', () => {
    renderShell()
    const link = within(rail()).getByRole('link', { name: 'Service status: Checking…' })
    expect(link.querySelector('.spin')?.getAttribute('aria-hidden')).toBe('true')
    expect(link.className).toBe('svc svc-checking')
  })

  it.each<[string, StatusResult, string, string]>([
    ['ready, mock', READY, 'Service status: Ready for analysis (mock)', 'svc-ready'],
    ['ready, not mock', { ...READY, mock: false }, 'Service status: Ready for analysis', 'svc-ready'],
    [
      'degraded (Ready with a notice), mock',
      { ...READY, notice: true, failing: ['history_and_stats'] },
      'Service status: Ready for analysis (mock), with a notice',
      'svc-notice',
    ],
    [
      'degraded with no failing part named',
      { ...READY, mock: false, notice: true, failing: [] },
      'Service status: Ready for analysis, with a notice',
      'svc-notice',
    ],
    [
      'unavailable from the service, mock',
      { ...READY, kind: 'unavailable', source: 'service', failing: ['analysis'] },
      'Service status: Unavailable (mock)',
      'svc-down',
    ],
    [
      'unavailable, no answer (the mock flag is unknown)',
      { kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-03T10:40:00.000Z', timeSource: 'browser' },
      'Service status: Unavailable',
      'svc-down',
    ],
  ])('%s', (_, result, name, kind) => {
    store.set({ checking: false, result })
    renderShell()
    const link = within(rail()).getByRole('link', { name })
    expect(link).toBe(line())
    expect(link.className).toBe(`svc ${kind}`)
    expect(link.getAttribute('aria-label')).toBeNull()
  })

  it.each<[StatusResult, string]>([
    [READY, 'Ready for analysis'],
    [{ ...READY, notice: true, failing: ['analysis'] }, 'Ready for analysis'],
    [{ kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-03T10:40:00.000Z', timeSource: 'browser' }, 'Unavailable'],
  ])('a different icon per state, hidden beside the visible text (%#)', (result, text) => {
    store.set({ checking: false, result })
    renderShell()
    const svg = line().querySelector('svg')!
    expect(svg.getAttribute('aria-hidden')).toBe('true')
    expect(line().textContent).toContain(text)
  })

  it('the three icons differ', () => {
    const shapes = ([READY, { ...READY, notice: true }, { ...READY, kind: 'unavailable', source: 'service' }] as StatusResult[]).map(
      (result) => {
        store.set({ checking: false, result })
        renderShell()
        const shape = line().querySelector('svg')!.innerHTML
        cleanup()
        return shape
      },
    )
    expect(new Set(shapes).size).toBe(3)
  })

  it('"(mock)" is visible text, not screen-reader only', () => {
    store.set({ checking: false, result: READY })
    renderShell()
    const mock = line().querySelector('.svc-mock')!
    expect(mock.textContent).toBe('(mock)')
    expect(line().textContent).toBe('Service status: Ready for analysis (mock)')
    expect(mock.className).not.toContain('sr')
  })

  it('a re-check keeps the last result on the line', () => {
    store.set({ checking: false, result: READY })
    renderShell()
    act(() => store.set({ checking: true, result: READY }))
    expect(within(rail()).getByRole('link', { name: 'Service status: Ready for analysis (mock)' })).toBeTruthy()
  })

  it('follows the store when the answer arrives', () => {
    renderShell()
    act(() => store.set({ checking: false, result: { ...READY, mock: false } }))
    expect(within(rail()).getByRole('link', { name: 'Service status: Ready for analysis' })).toBeTruthy()
  })

  it('is not a live region', () => {
    store.set({ checking: false, result: READY })
    renderShell()
    expect(line().closest('[aria-live], [role="status"], [role="alert"]')).toBeNull()
  })
})

describe('where it is and where it goes', () => {
  it('first in the rail footer, above the analyst', () => {
    renderShell()
    const foot = rail().querySelector('.rail-foot')!
    expect(foot.firstElementChild).toBe(line())
    expect(line().nextElementSibling?.className).toBe('who')
  })

  it('links to System status', async () => {
    store.set({ checking: false, result: READY })
    const router = renderShell('/history')
    expect(line().getAttribute('href')).toBe('/status')
    await userEvent.click(line())
    expect(router.state.location.pathname).toBe('/status')
    expect(screen.getByRole('heading', { level: 1, name: 'System status' })).toBeTruthy()
  })

  it('is not marked as the current page on /status (the nav link is)', () => {
    renderShell('/status')
    expect(line().getAttribute('aria-current')).toBeNull()
    expect(within(rail()).getByRole('link', { name: 'System status' }).getAttribute('aria-current')).toBe('page')
  })
})

describe('it never fetches', () => {
  it('rendered alone, it starts no check and sends no request', () => {
    render(
      <MemoryRouter>
        <RailStatus />
      </MemoryRouter>,
    )
    expect(screen.getByRole('link', { name: 'Service status: Checking…' })).toBeTruthy()
    expect(checkStatus).not.toHaveBeenCalled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('in the shell, the one check is the shell mount, not the line, and navigation adds none', async () => {
    const router = renderShell()
    expect(checkStatus).toHaveBeenCalledTimes(1)
    for (const path of ['/history', '/overview', '/status', '/analyze']) await act(() => router.navigate(path))
    expect(checkStatus).toHaveBeenCalledTimes(1)
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
