import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setSessionExpiredHandler } from '../api/client'
import type { RiskLevel, StoredAnalysis, Verdict } from '../api/types'
import { routes } from '../routing/router'
import { expire, markSignedIn, markSignedOut } from '../session/session'

// S7 in the real route tree, on the real session, status and limits stores; only fetch is a stand-in, so the rule "one
// GET per entry" is checked where it is made. A fixed zone that is not UTC, set before the formatters are built, so a
// hard-coded UTC fails on any machine (CI runs in UTC; see analyzedAt.test.ts).
vi.hoisted(() => {
  vi.stubEnv('TZ', 'Africa/Tunis')
})

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

const idOf = (digit: number | string) => `req_${String(digit).repeat(32)}` as `req_${string}`

interface RecordFields {
  id?: `req_${string}`
  prediction?: Verdict
  risk?: RiskLevel
  model?: string
  createdAt?: string
  note?: StoredAnalysis['note']
}

function record({ id = idOf('a'), prediction = 'synthetic', risk = 'HIGH', model = 'mock', createdAt = '2026-11-03T09:14:05Z', note = null }: RecordFields = {}): StoredAnalysis {
  return {
    analysis: {
      prediction,
      confidence: 0.97,
      risk_level: risk,
      model_version: model,
      profile_version: 'preprocessing-v1',
      policy_version: 'risk-v1',
      request_id: id,
      notice: 'Decision support only.',
    },
    created_at: createdAt,
    note,
  }
}

function listBody(items: StoredAnalysis[], total = items.length) {
  return { items, total, page: 1, page_size: 10, counts: { synthetic: 0, uncertain: 0, authentic: 0 } }
}

const json = (code: number, body: unknown) => new Response(JSON.stringify(body), { status: code })
const list = (items: StoredAnalysis[], total?: number) => json(200, listBody(items, total))
const apiError = (code: number, name: string) => json(code, { error: { code: name, message: 'x', request_id: null } })

function deferred(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const fetchMock = vi.fn<typeof fetch>()
let answerHistory: () => Response | Promise<Response>

const gets = () => fetchMock.mock.calls.filter(([path]) => String(path).startsWith('/api/analyses'))

function renderAt(path: string, strict = false) {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  const app = <RouterProvider router={router} />
  render(strict ? <StrictMode>{app}</StrictMode> : app)
  return router
}

const table = () => screen.findByRole('table', { name: 'Stored analysis results, newest first.' })
const bodyRows = async () => within(await table()).getAllByRole('row').slice(1)
const announced = () => document.getElementById('history-announcer')!.textContent
const h1 = () => screen.getByRole('heading', { level: 1 })
const FAILED_TEXT = 'History could not be loaded'

beforeEach(() => {
  answerHistory = () => list([record()])
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (path) => {
    if (path === '/api/limits') return json(200, LIMITS)
    if (String(path).startsWith('/api/analyses')) return answerHistory()
    return json(200, STATUS)
  })
  vi.stubGlobal('fetch', fetchMock)
  window.matchMedia = (() => ({ matches: false, addEventListener: () => undefined, removeEventListener: () => undefined })) as unknown as typeof window.matchMedia
  setSessionExpiredHandler(expire)
  markSignedIn('analyst01')
})

afterEach(() => {
  cleanup()
  markSignedOut()
  setSessionExpiredHandler(null)
  vi.unstubAllGlobals()
})

it('really runs in Africa/Tunis, so the times below mean something', () => {
  expect(Intl.DateTimeFormat().resolvedOptions().timeZone).toBe('Africa/Tunis')
  expect(new Date('2026-11-03T09:14:05Z').getTimezoneOffset()).toBe(-60)
})

describe('states', () => {
  it('loading: the skeleton of 8 lines, a status line, no alert and no table', async () => {
    answerHistory = () => deferred().promise
    renderAt('/history')
    expect((await screen.findByText('Loading history…')).getAttribute('role')).toBe('status')
    expect(document.querySelectorAll('.skel')).toHaveLength(8)
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.queryByRole('table')).toBeNull()
  })

  it('empty: the title, the sentence and New analysis, with no table', async () => {
    answerHistory = () => list([])
    renderAt('/history')
    expect(await screen.findByRole('heading', { level: 2, name: 'No analyses yet' })).toBeTruthy()
    expect(screen.getByText('Results appear here after the first analysis. Only results are stored: no audio and no original filenames.')).toBeTruthy()
    expect(within(screen.getByRole('main')).getByRole('button', { name: 'New analysis' })).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('empty: New analysis goes to /analyze', async () => {
    answerHistory = () => list([])
    const router = renderAt('/history')
    await screen.findByRole('heading', { level: 2, name: 'No analyses yet' })
    await userEvent.click(within(screen.getByRole('main')).getByRole('button', { name: 'New analysis' }))
    expect(router.state.location.pathname).toBe('/analyze')
  })

  it('failed: one alert with the title and the sentence, Retry outside it, no table', async () => {
    answerHistory = () => apiError(500, 'internal_error')
    renderAt('/history')
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toBe(`${FAILED_TEXT}The service did not return the stored results. No results were changed.`)
    expect(within(alert).queryByRole('button')).toBeNull()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
  })

  it('list: the caption, the four column headers and one row per record', async () => {
    answerHistory = () => list([record({ id: idOf('a') }), record({ id: idOf('b') })])
    renderAt('/history')
    const headers = within(await table()).getAllByRole('columnheader')
    expect(headers.map((header) => header.textContent)).toEqual(['Analyzed (GMT+1)', 'Request ID', 'Verdict · risk', 'Analyst note'])
    expect(headers.every((header) => header.getAttribute('scope') === 'col')).toBe(true)
    expect(await bodyRows()).toHaveLength(2)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('a malformed body is the failed state, with no row at all', async () => {
    answerHistory = () => json(200, listBody([record(), { ...record({ id: idOf('b') }), created_at: 'yesterday' }]))
    renderAt('/history')
    expect((await screen.findByRole('alert')).textContent).toContain(FAILED_TEXT)
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.queryByText(idOf('a'))).toBeNull()
  })

  it.each<[string, () => Response]>([
    ['list', () => list([record()])],
    ['empty', () => list([])],
    ['failed', () => apiError(500, 'internal_error')],
  ])('%s: the decision-support line and the storage line stay', async (_label, answer) => {
    answerHistory = answer
    renderAt('/history')
    await screen.findByText(/^History loaded|History could not be loaded/)
    expect(screen.getByText('Decision support only:')).toBeTruthy()
    expect(screen.getByText('Only results, request IDs and analyst notes are stored. No audio, no original filenames, no per-analyst figures.')).toBeTruthy()
  })
})

describe('the count wording: total for what is stored, the rows for what is on screen', () => {
  it('total 0: no count line and no Showing line', async () => {
    answerHistory = () => list([])
    renderAt('/history')
    await screen.findByRole('heading', { level: 2, name: 'No analyses yet' })
    expect(document.querySelector('.count-line')).toBeNull()
    expect(document.querySelector('.history-showing')).toBeNull()
    expect(announced()).toBe('History loaded. 0 results.')
  })

  it.each([
    [1, 1, '1 result', 'Showing 1–1 of 1', 'History loaded. 1 result.'],
    [10, 10, '10 results', 'Showing 1–10 of 10', 'History loaded. 10 results.'],
    [37, 10, '37 results', 'Showing 1–10 of 37', 'History loaded. 37 results.'],
  ])('total %i with %i rows', async (total, rows, count, showing, announcement) => {
    const items = Array.from({ length: rows }, (_, index) => record({ id: idOf(index.toString(16)) }))
    answerHistory = () => list(items, total)
    renderAt('/history')
    expect(await bodyRows()).toHaveLength(rows)
    expect(document.querySelector('.count-line strong')?.textContent).toBe(count)
    expect(document.querySelector('.count-line .muted')?.textContent).toBe('all stored results')
    expect(document.querySelector('.history-showing')?.textContent).toBe(showing)
    expect(announced()).toBe(announcement)
  })
})

describe('rows', () => {
  const firstRow = async (fields: RecordFields) => {
    answerHistory = () => list([record(fields)])
    renderAt('/history')
    return (await bodyRows())[0]!
  }

  it('the time is created_at in the browser zone; the header names the zone', async () => {
    const row = await firstRow({ createdAt: '2026-11-03T23:30:00Z' })
    expect(within(row).getAllByRole('cell')[0]?.textContent).toBe('4 Nov 2026, 00:30')
    expect(screen.getByRole('columnheader', { name: 'Analyzed (GMT+1)' })).toBeTruthy()
  })

  it('the request ID is plain text in full: no link, no button', async () => {
    const row = await firstRow({ id: idOf('c') })
    expect(within(row).getByText(idOf('c')).tagName).toBe('CODE')
    expect(within(row).queryByRole('link')).toBeNull()
    expect(within(row).queryByRole('button')).toBeNull()
  })

  it.each([
    ['synthetic', 'HIGH', 'Likely synthetic · HIGH'],
    ['uncertain', 'MEDIUM', 'Cannot decide · MEDIUM'],
    ['authentic', 'LOW', 'Likely authentic · LOW'],
  ] as const)('%s is marked with its words', async (prediction, risk, text) => {
    const row = await firstRow({ prediction, risk })
    expect(within(row).getByText(text)).toBeTruthy()
  })

  it('the risk comes from risk_level, not from the verdict', async () => {
    const row = await firstRow({ prediction: 'uncertain', risk: 'LOW' })
    expect(within(row).getByText('Cannot decide · LOW')).toBeTruthy()
  })

  it('model_version "mock" gets the Mock result tag', async () => {
    const row = await firstRow({ model: 'mock' })
    expect(within(row).getByText('Mock result').className).toBe('tag tag-demo')
  })

  it('another model_version gets no tag', async () => {
    const row = await firstRow({ model: 'cnn-v1' })
    expect(within(row).queryByText('Mock result')).toBeNull()
  })

  it('a record with a note says Added, and never the text', async () => {
    const row = await firstRow({ note: { text: 'Called back.', updated_at: '2026-11-03T10:00:00Z', updated_by: 'analyst01' } })
    expect(within(row).getAllByRole('cell')[3]?.textContent).toBe('Added')
    expect(document.body.textContent).not.toContain('Called back.')
  })

  it('a record with no note says None', async () => {
    const row = await firstRow({ note: null })
    expect(within(row).getAllByRole('cell')[3]?.textContent).toBe('None')
  })

  it('shows no score (HANDOFF lines 222, 300)', async () => {
    await firstRow({})
    const text = screen.getByRole('main').textContent
    expect(text).not.toContain('0.97')
    expect(text).not.toContain('%')
  })

  it('each cell carries its label for the card layout (HANDOFF line 182)', async () => {
    const row = await firstRow({})
    const labels = within(row).getAllByRole('cell').map((cell) => cell.getAttribute('data-label'))
    expect(labels).toEqual(['Analyzed', 'Request ID', 'Verdict · risk', 'Analyst note'])
  })

  it('keeps the order of the response: it is never sorted here', async () => {
    const older = record({ id: idOf('a'), createdAt: '2026-11-01T09:00:00Z' })
    const newer = record({ id: idOf('b'), createdAt: '2026-11-03T09:00:00Z' })
    answerHistory = () => list([older, newer])
    renderAt('/history')
    const rows = await bodyRows()
    expect(rows.map((row) => within(row).getByText(/^req_/).textContent)).toEqual([idOf('a'), idOf('b')])
  })
})

describe('?verdict= is ignored', () => {
  it('the GET has no query and every row is shown, as "all stored results"', async () => {
    answerHistory = () => list([record({ prediction: 'authentic', risk: 'LOW' })])
    renderAt('/history?verdict=synthetic')
    expect(await screen.findByText('Likely authentic · LOW')).toBeTruthy()
    expect(gets().map(([path]) => path)).toEqual(['/api/analyses'])
    expect(screen.getByText('all stored results')).toBeTruthy()
  })

  it('the query stays in the URL', async () => {
    const router = renderAt('/history?verdict=a&verdict=b')
    await table()
    expect(router.state.location.search).toBe('?verdict=a&verdict=b')
  })

  it('a query-only change sends no GET', async () => {
    const router = renderAt('/history')
    await table()
    await act(() => router.navigate('/history?verdict=uncertain'))
    await table()
    expect(gets()).toHaveLength(1)
  })
})

describe('one GET per entry', () => {
  it("StrictMode's double effect sends one", async () => {
    renderAt('/history', true)
    await table()
    expect(gets()).toHaveLength(1)
  })

  it('leaving and coming back sends one more', async () => {
    const router = renderAt('/history')
    await table()
    await act(() => router.navigate('/status'))
    expect(gets()).toHaveLength(1)
    await act(() => router.navigate('/history'))
    await table()
    expect(gets()).toHaveLength(2)
  })

  it('a new analysis appears on the next entry, with the skeleton, not the old rows, in between', async () => {
    const old = record({ id: idOf('a') })
    const fresh = record({ id: idOf('b') })
    answerHistory = () => list([old])
    const router = renderAt('/history')
    await table()
    await act(() => router.navigate('/status'))
    const pending = deferred()
    answerHistory = () => pending.promise
    await act(() => router.navigate('/history'))
    expect(document.querySelectorAll('.skel')).toHaveLength(8)
    expect(screen.queryByRole('table')).toBeNull()
    pending.resolve(list([fresh, old]))
    const rows = await bodyRows()
    expect(within(rows[0]!).getByText(idOf('b'))).toBeTruthy()
  })

  it('the rail link on /history sends none', async () => {
    renderAt('/history')
    await table()
    await userEvent.click(within(screen.getByRole('navigation', { name: 'Pages' })).getByRole('link', { name: 'History' }))
    await table()
    expect(gets()).toHaveLength(1)
  })

  it('an answer that lands after leaving is ignored, and the next entry starts on the skeleton', async () => {
    const late = deferred()
    answerHistory = () => late.promise
    const router = renderAt('/history')
    await waitFor(() => expect(gets()).toHaveLength(1))
    await act(() => router.navigate('/status'))
    await act(async () => late.resolve(list([record({ id: idOf('a') })])))
    answerHistory = () => deferred().promise
    await act(() => router.navigate('/history'))
    expect(document.querySelectorAll('.skel')).toHaveLength(8)
    expect(screen.queryByText(idOf('a'))).toBeNull()
    expect(gets()).toHaveLength(2)
  })
})

describe('focus and announcements', () => {
  it('a failure is one alert and no polite text', async () => {
    answerHistory = () => apiError(500, 'internal_error')
    renderAt('/history')
    await screen.findByRole('alert')
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    expect(announced()).toBe('')
  })

  it('the list arriving leaves focus on the h1', async () => {
    const router = renderAt('/status')
    await screen.findByRole('heading', { level: 1, name: 'System status' })
    await act(() => router.navigate('/history'))
    expect(document.activeElement).toBe(h1())
    await table()
    expect(document.activeElement).toBe(h1())
  })

  it('Retry moves focus to the h1, sends one GET and shows the list', async () => {
    answerHistory = () => apiError(500, 'internal_error')
    renderAt('/history')
    const retry = await screen.findByRole('button', { name: 'Retry' })
    const pending = deferred()
    answerHistory = () => pending.promise
    await userEvent.click(retry)
    expect(document.activeElement).toBe(h1())
    expect(document.querySelectorAll('.skel')).toHaveLength(8)
    expect(screen.queryByRole('alert')).toBeNull()
    pending.resolve(list([record()]))
    await table()
    expect(gets()).toHaveLength(2)
    expect(document.activeElement).toBe(h1())
    expect(announced()).toBe('History loaded. 1 result.')
  })

  it('Retry that fails again brings the alert back', async () => {
    answerHistory = () => apiError(500, 'internal_error')
    renderAt('/history')
    await userEvent.click(await screen.findByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(gets()).toHaveLength(2))
    expect((await screen.findByRole('alert')).textContent).toContain(FAILED_TEXT)
  })
})

describe('the session', () => {
  it('a 401 goes to S1 with next=/history, and shows no failure', async () => {
    answerHistory = () => apiError(401, 'session_expired')
    const router = renderAt('/history')
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'))
    expect(router.state.location.search).toBe('?next=%2Fhistory')
    expect(screen.queryByText(FAILED_TEXT)).toBeNull()
  })

  it('next keeps the query', async () => {
    answerHistory = () => apiError(401, 'session_expired')
    const router = renderAt('/history?verdict=synthetic')
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'))
    expect(router.state.location.search).toBe('?next=%2Fhistory%3Fverdict%3Dsynthetic')
  })

  it('an answer after sign-out is dropped', async () => {
    const late = deferred()
    answerHistory = () => late.promise
    const router = renderAt('/history')
    await waitFor(() => expect(gets()).toHaveLength(1))
    act(() => markSignedOut())
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'))
    await act(async () => late.resolve(list([record({ id: idOf('a') })])))
    expect(screen.queryByText(idOf('a'))).toBeNull()
    expect(screen.queryByText(FAILED_TEXT)).toBeNull()
  })
})
