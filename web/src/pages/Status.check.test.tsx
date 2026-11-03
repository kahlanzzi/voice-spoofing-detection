import { act, cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { markSignedIn, markSignedOut } from '../session/session'
import { checkStatus } from '../status/status'
import Status from './Status'

// S10's Check now on the real status store and a stubbed fetch (Stage 18 PR 7b). The store has no reset; it forgets
// its result and any check in flight when the session ends, so each test signs out and in again (vi.resetModules
// would load a second React beside Testing Library's). Each GET /api/status is held until the test answers it, so
// the busy state can be seen.

const BODY = {
  state: 'ready',
  checked_at: '2026-11-03T09:14:05Z',
  checks: { analysis: 'available', history_and_stats: 'available' },
  versions: { model_version: 'mock', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' },
  bundle_manifest_sha256: null,
  deferral_on_fit: 0.0993,
  mock: true,
}

const json = (code: number, body: unknown) => new Response(JSON.stringify(body), { status: code, headers: { 'Content-Type': 'application/json' } })
const UNAVAILABLE = () => json(503, { error: { code: 'analysis_unavailable', message: 'x', request_id: null } })

let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>
let pending: ((response: Response) => void)[]

beforeEach(() => {
  pending = []
  fetchMock = vi.fn<typeof fetch>(() => new Promise<Response>((resolve) => pending.push(resolve)))
  vi.stubGlobal('fetch', fetchMock)
  markSignedOut()
  markSignedIn('analyst01')
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// The shell's mount check, answered, so S10 starts on a Ready result.
async function startReady() {
  const first = checkStatus()
  await act(async () => {
    pending.shift()!(json(200, BODY))
    await first
  })
  expect(fetchMock).toHaveBeenCalledTimes(1)
}

const button = () => screen.getByRole('button', { name: 'Check now' })
const announcer = () => document.getElementById('status-announcer')!

async function answer(response: Response) {
  await act(async () => {
    pending.shift()!(response)
    await checkStatus()
  })
}

describe('entering /status', () => {
  it('sends no request, StrictMode included', () => {
    render(
      <StrictMode>
        <Status />
      </StrictMode>,
    )
    expect(fetchMock).not.toHaveBeenCalled()
    expect(announcer().childNodes).toHaveLength(0)
  })
})

describe('Check now', () => {
  it('busy: label kept, spinner, disabled, aria-busy; then Ready, enabled, focus back, one request', async () => {
    await startReady()
    render(<Status />)
    const user = userEvent.setup()
    await user.click(button())
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1]?.[0]).toBe('/api/status')
    expect(button().textContent).toBe('Check now')
    expect(button().hasAttribute('disabled')).toBe(true)
    expect(button().getAttribute('aria-busy')).toBe('true')
    expect(button().querySelector('.spin')?.getAttribute('aria-hidden')).toBe('true')
    expect(screen.getByRole('heading', { level: 2, name: 'Checking…' })).toBeTruthy()

    await answer(json(200, BODY))
    expect(button().hasAttribute('disabled')).toBe(false)
    expect(button().hasAttribute('aria-busy')).toBe(false)
    expect(button().querySelector('.spin')).toBeNull()
    expect(document.activeElement).toBe(button())
    expect(screen.getByRole('heading', { level: 2, name: 'Ready for analysis' })).toBeTruthy()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('a double click sends one request', async () => {
    await startReady()
    render(<Status />)
    await userEvent.setup().dblClick(button())
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await answer(json(200, BODY))
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('focus goes back to the button from body (a disabled button loses focus), not from elsewhere', async () => {
    await startReady()
    render(
      <>
        <Status />
        <a href="#elsewhere">elsewhere</a>
      </>,
    )
    const user = userEvent.setup()
    await user.click(button())
    const link = screen.getByRole('link', { name: 'elsewhere' })
    link.focus()
    link.blur()
    expect(document.activeElement).toBe(document.body)
    await answer(json(200, BODY))
    expect(document.activeElement).toBe(button())

    await user.click(button())
    screen.getByRole('link', { name: 'elsewhere' }).focus()
    await answer(json(200, BODY))
    expect(document.activeElement).toBe(screen.getByRole('link', { name: 'elsewhere' }))
  })

  it('no answer → Unavailable with "Last attempt", announced', async () => {
    await startReady()
    render(<Status />)
    await userEvent.setup().click(button())
    await answer(UNAVAILABLE())
    expect(screen.getByRole('heading', { level: 2, name: 'Unavailable' })).toBeTruthy()
    expect(screen.getByText(/^Last attempt /)).toBeTruthy()
    expect(screen.queryByText(/^Last checked /)).toBeNull()
    expect(announcer().textContent).toBe('Status checked: Unavailable.')
  })
})

describe('the announcement', () => {
  it('only after Check now finishes, a new node each time, never on mount or re-render', async () => {
    await startReady()
    const { rerender } = render(<Status />)
    rerender(<Status />)
    expect(announcer().childNodes).toHaveLength(0)

    const user = userEvent.setup()
    await user.click(button())
    expect(announcer().childNodes).toHaveLength(0)
    await answer(json(200, BODY))
    expect(announcer().textContent).toBe('Status checked: Ready for analysis.')
    const first = announcer().firstChild

    rerender(<Status />)
    expect(announcer().firstChild).toBe(first)

    await user.click(button())
    await answer(json(200, BODY))
    expect(announcer().textContent).toBe('Status checked: Ready for analysis.')
    expect(announcer().firstChild).not.toBe(first)
    expect(announcer().childNodes).toHaveLength(1)
  })

  it('a check started elsewhere does not announce on S10', async () => {
    await startReady()
    render(<Status />)
    const other = checkStatus()
    await act(async () => {
      pending.shift()!(UNAVAILABLE())
      await other
    })
    expect(screen.getByText(/^Last attempt /)).toBeTruthy()
    expect(announcer().childNodes).toHaveLength(0)
  })
})
