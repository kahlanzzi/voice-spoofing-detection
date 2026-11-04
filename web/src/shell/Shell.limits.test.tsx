import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { routes } from '../routing/router'
import { markSignedIn, markSignedOut } from '../session/session'
import { getSnapshot as getLimits } from '../upload/limits'

// The rule behind GET /api/limits (Entry 145 Q20): one per signed-in session of the shell, on the real session, limits
// and status stores. Shell.test.tsx cannot hold this: its session stand-in never changes, so the store never forgets.
// A real sign-out is what returns the store to idle between tests. vi.resetModules would give the routes a second React.

const STATUS = {
  state: 'ready',
  checked_at: '2026-11-03T09:14:05Z',
  checks: { analysis: 'available', history_and_stats: 'available' },
  versions: { model_version: 'mock', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' },
  bundle_manifest_sha256: null,
  deferral_on_fit: 0.0993,
  mock: true,
}

const LIMITS = { formats: ['wav'], max_size_bytes: 1048576, max_duration_s: 60, audio_deletion: 'Deleted' }

const fetchMock = vi.fn<typeof fetch>()
const calls = (path: string) => fetchMock.mock.calls.filter(([called]) => called === path)
let limitsStatus = 200

function renderAt(path: string, strict = false) {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  const app = <RouterProvider router={router} />
  render(strict ? <StrictMode>{app}</StrictMode> : app)
  return router
}

const settled = (status: string) => waitFor(() => expect(getLimits().status).toBe(status))

beforeEach(() => {
  limitsStatus = 200
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (path) =>
    path === '/api/limits'
      ? new Response(JSON.stringify(LIMITS), { status: limitsStatus })
      : new Response(JSON.stringify(STATUS), { status: 200 }),
  )
  vi.stubGlobal('fetch', fetchMock)
  window.matchMedia = (() => ({ matches: false, addEventListener: () => undefined, removeEventListener: () => undefined })) as unknown as typeof window.matchMedia
  markSignedIn('analyst01')
})

afterEach(() => {
  cleanup()
  markSignedOut()
  vi.unstubAllGlobals()
})

describe('GET /api/limits', () => {
  it("one GET /api/limits per signed-in session: StrictMode's double mount, navigation and a remount add none", async () => {
    const router = renderAt('/analyze', true)
    await settled('ready')
    expect(calls('/api/limits')).toHaveLength(1)
    for (const path of ['/history', '/overview', '/status', '/analyze']) await act(() => router.navigate(path))
    expect(calls('/api/limits')).toHaveLength(1)
    cleanup()
    renderAt('/history')
    await screen.findByRole('heading', { level: 1, name: 'History' })
    expect(getLimits().status).toBe('ready')
    expect(calls('/api/limits')).toHaveLength(1)
  })

  it('sign-out then sign-in sends one more', async () => {
    const router = renderAt('/analyze')
    await settled('ready')
    act(() => markSignedOut())
    expect(getLimits().status).toBe('idle')
    await screen.findByRole('heading', { level: 1, name: 'Sign in to the analyst console' })
    act(() => markSignedIn('analyst01'))
    await act(() => router.navigate('/analyze'))
    await settled('ready')
    expect(calls('/api/limits')).toHaveLength(2)
  })

  it('S10: Check now sends no /api/limits and offers no limits Retry', async () => {
    limitsStatus = 500
    renderAt('/status')
    await settled('failed')
    const checkNow = await screen.findByRole('button', { name: 'Check now' })
    await waitFor(() => expect(checkNow).toHaveProperty('disabled', false))
    await userEvent.click(checkNow)
    await waitFor(() => expect(calls('/api/status')).toHaveLength(2))
    await waitFor(() => expect(checkNow).toHaveProperty('disabled', false))
    expect(calls('/api/limits')).toHaveLength(1)
    expect(screen.queryByRole('button', { name: /retry/i })).toBeNull()
  })
})
