import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import { NARROW_QUERY } from './useNarrow'

// The shell (HANDOFF §2 line 40, §3 line 85, §7 lines 173-176) on a signed-in session. jsdom applies Shell.css but
// not its media queries, so the top bar is always display: none here and is queried with hidden: true; the
// narrow layout itself is checked in e2e/shell.spec.ts.

// One object: useSyncExternalStore needs a stable snapshot.
const SIGNED_IN = vi.hoisted(() => ({ status: 'signedIn', username: 'analyst01' }) as const)

vi.mock('../session/session', () => ({
  getSnapshot: () => SIGNED_IN,
  subscribe: () => () => undefined,
  checkSession: vi.fn(),
  expire: vi.fn(),
  markSignedIn: vi.fn(),
  markSignedOut: vi.fn(),
  startSession: vi.fn(),
}))

const ID = 'req_' + '0123456789abcdef'.repeat(2)

// A matchMedia stand-in whose (max-width: 860px) answer the test sets and changes.
const media = vi.hoisted(() => {
  let matches = false
  const listeners = new Set<() => void>()
  return {
    set(next: boolean): void {
      matches = next
      for (const listener of [...listeners]) listener()
    },
    install(query: string): void {
      window.matchMedia = ((asked: string) => ({
        get matches() {
          if (asked !== query) throw new Error(`unexpected media query ${asked}`)
          return matches
        },
        addEventListener: (_: string, listener: () => void) => listeners.add(listener),
        removeEventListener: (_: string, listener: () => void) => listeners.delete(listener),
      })) as unknown as typeof window.matchMedia
    },
  }
})

beforeEach(() => {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
  media.install(NARROW_QUERY)
  media.set(false)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return router
}

const rail = () => screen.getByRole('complementary', { name: 'Main menu' })
const nav = () => within(rail()).getByRole('navigation', { name: 'Pages' })
const menu = () => screen.getByRole('button', { name: 'Menu', hidden: true })
const main = () => screen.getByRole('main')

describe('layout and landmarks', () => {
  it('one main with the page, a rail aside with the Pages nav, a top-bar header', () => {
    renderAt('/history')
    expect(screen.getAllByRole('main')).toHaveLength(1)
    expect(main().id).toBe('main')
    expect(within(main()).getByRole('heading', { level: 1 }).textContent).toBe('History')
    expect(rail().id).toBe('rail')
    expect(document.querySelector('header.topbar')?.contains(menu())).toBe(true)
  })

  it('the rail: brand, New analysis, then Analyze, History, Overview, System status (HANDOFF line 40)', () => {
    renderAt('/analyze')
    expect(within(rail()).getByText('Voice Check')).toBeTruthy()
    expect(within(rail()).getByRole('button', { name: 'New analysis' })).toBeTruthy()
    const links = within(nav()).getAllByRole('link')
    expect(links.map((link) => link.textContent)).toEqual(['Analyze', 'History', 'Overview', 'System status'])
    expect(links.map((link) => link.getAttribute('href'))).toEqual(['/analyze', '/history', '/overview', '/status'])
  })

  it('the rail foot is empty until PR 6b and PR 7', () => {
    renderAt('/analyze')
    expect(rail().querySelector('.rail-foot')?.childElementCount).toBe(0)
  })
})

describe('the current page', () => {
  it.each([
    ['/analyze', 'Analyze'],
    ['/history', 'History'],
    [`/history/${ID}`, 'History'],
    ['/overview', 'Overview'],
    ['/status', 'System status'],
  ])('%s → %s has aria-current="page", and only it', (path, label) => {
    renderAt(path)
    const current = within(nav())
      .getAllByRole('link')
      .filter((link) => link.getAttribute('aria-current') === 'page')
    expect(current.map((link) => link.textContent)).toEqual([label])
  })

  it('follows navigation', async () => {
    renderAt('/analyze')
    await userEvent.click(within(nav()).getByRole('link', { name: 'Overview' }))
    expect(within(nav()).getByRole('link', { name: 'Overview' }).getAttribute('aria-current')).toBe('page')
    expect(within(nav()).getByRole('link', { name: 'Analyze' }).getAttribute('aria-current')).toBeNull()
  })
})

describe('skip link and keyboard order', () => {
  it('the skip link is the first Tab and moves focus to the h1', async () => {
    renderAt('/overview')
    await userEvent.tab()
    const skip = screen.getByRole('link', { name: 'Skip to main content' })
    expect(document.activeElement).toBe(skip)
    await userEvent.click(skip)
    expect(document.activeElement).toBe(within(main()).getByRole('heading', { level: 1 }))
  })

  it('Tab order (wide, top bar hidden): skip, New analysis, the four pages', async () => {
    renderAt('/overview')
    const order: (string | null)[] = []
    for (let i = 0; i < 6; i += 1) {
      await userEvent.tab()
      order.push(document.activeElement?.textContent ?? null)
    }
    expect(order).toEqual([
      'Skip to main content', 'New analysis', 'Analyze', 'History', 'Overview', 'System status',
    ])
  })
})

describe('New analysis', () => {
  it.each([0, 1])('button %i (top bar, rail) goes to /analyze', async (index) => {
    const router = renderAt('/history')
    const buttons = screen.getAllByRole('button', { name: 'New analysis', hidden: true })
    expect(buttons).toHaveLength(2)
    await userEvent.click(buttons[index]!)
    expect(router.state.location.pathname).toBe('/analyze')
  })
})

describe('the drawer (≤ 860 px)', () => {
  beforeEach(() => media.set(true))

  // Shell.css shows the top bar only under its ≤ 860 px media rule, which jsdom does not apply; an inline style
  // stands in for it so that Menu can take focus.
  function renderNarrow(path: string) {
    const router = renderAt(path)
    document.querySelector<HTMLElement>('header.topbar')!.style.display = 'flex'
    return router
  }

  it('closed: Menu not expanded, controls the rail, nothing inert', () => {
    renderNarrow('/analyze')
    expect(menu().getAttribute('aria-expanded')).toBe('false')
    expect(menu().getAttribute('aria-controls')).toBe('rail')
    expect(main().hasAttribute('inert')).toBe(false)
    expect(rail().classList.contains('open')).toBe(false)
  })

  it('Menu opens it: focus on the first page link, top bar and main inert', async () => {
    renderNarrow('/history')
    await userEvent.click(menu())
    expect(menu().getAttribute('aria-expanded')).toBe('true')
    expect(rail().classList.contains('open')).toBe(true)
    expect(document.activeElement).toBe(within(nav()).getByRole('link', { name: 'Analyze' }))
    expect(main().hasAttribute('inert')).toBe(true)
    expect(document.querySelector('header.topbar')?.hasAttribute('inert')).toBe(true)
    expect(document.querySelector('.scrim')).not.toBeNull()
  })

  it('Escape closes it and returns focus to Menu', async () => {
    renderNarrow('/history')
    await userEvent.click(menu())
    await userEvent.keyboard('{Escape}')
    expect(menu().getAttribute('aria-expanded')).toBe('false')
    expect(document.activeElement).toBe(menu())
    expect(main().hasAttribute('inert')).toBe(false)
  })

  it('the scrim closes it and returns focus to Menu', async () => {
    renderNarrow('/history')
    await userEvent.click(menu())
    await userEvent.click(document.querySelector('.scrim')!)
    expect(menu().getAttribute('aria-expanded')).toBe('false')
    expect(document.activeElement).toBe(menu())
  })

  it('a page link closes it; focus goes to the new h1', async () => {
    renderNarrow('/history')
    await userEvent.click(menu())
    await userEvent.click(within(nav()).getByRole('link', { name: 'Overview' }))
    expect(menu().getAttribute('aria-expanded')).toBe('false')
    expect(main().hasAttribute('inert')).toBe(false)
    expect(document.activeElement).toBe(within(main()).getByRole('heading', { level: 1, name: 'Overview' }))
  })

  it('Back closes it', async () => {
    const router = renderNarrow('/analyze')
    await act(async () => router.navigate('/history'))
    await userEvent.click(menu())
    await act(async () => router.navigate(-1))
    expect(menu().getAttribute('aria-expanded')).toBe('false')
  })

  it('widening the window closes it, so the page is never left inert', async () => {
    renderNarrow('/history')
    await userEvent.click(menu())
    act(() => media.set(false))
    expect(main().hasAttribute('inert')).toBe(false)
    expect(rail().classList.contains('open')).toBe(false)
    act(() => media.set(true))
    expect(menu().getAttribute('aria-expanded')).toBe('false')
  })

  it('without matchMedia the wide layout is assumed', async () => {
    window.matchMedia = undefined as unknown as typeof window.matchMedia
    renderNarrow('/history')
    await userEvent.click(menu())
    expect(main().hasAttribute('inert')).toBe(false)
  })
})
