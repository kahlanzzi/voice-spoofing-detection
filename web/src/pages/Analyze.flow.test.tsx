import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { markSignedIn } from '../session/session'
import type { StatusState } from '../status/status'
import { clearFileDraft, getSnapshot as getDraft, setFileDraft } from '../upload/fileDraft'
import type { LimitsState } from '../upload/limits'
import Analyze from './Analyze'

// S3 to the answer on the real analysis, draft and session stores, with fetch and Audio as stand-ins. Stand-ins for the
// limits and the status too: their tests are their own, and here they are only inputs. Analyze.test.tsx forbids fetch,
// so everything after the press is here.

const LIMITS = { formats: ['wav', 'flac', 'mp3'], max_size_bytes: 4096, max_duration_s: 60, audio_deletion: 'Deleted when processing finishes' }

const inputs = vi.hoisted(() => ({
  limits: { status: 'idle' } as LimitsState,
  status: { checking: false, result: null } as StatusState,
}))

vi.mock('../upload/limits', () => ({
  getSnapshot: () => inputs.limits,
  subscribe: () => () => undefined,
  loadLimits: vi.fn(),
  retryLimits: vi.fn(),
}))

vi.mock('../status/useStatus', () => ({ useStatus: () => inputs.status }))

class FakeAudio {
  static made: FakeAudio[] = []
  preload = ''
  src = ''
  duration = Number.NaN
  onloadedmetadata: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor() {
    FakeAudio.made.push(this)
  }
  removeAttribute(): void {
    this.src = ''
  }
}

const fetchMock = vi.fn<typeof fetch>()

const wav = (name = 'call.wav', size = 1024) => new File([new Uint8Array(size)], name)
const analyzeButton = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Analyze recording' })
const busyButton = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Analyzing…' })
const fileHeading = () => screen.getByRole('heading', { level: 2, name: /^Selected recording/ })
const announcer = () => document.getElementById('analyze-announcer')!
const warning = () => screen.queryByText('Analysis is currently unavailable. You can still try; the request may fail.')

function deferred(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function show() {
  const router = createMemoryRouter(
    [
      { path: '/analyze', Component: Analyze },
      { path: '/other', element: <h1>Other</h1> },
    ],
    { initialEntries: ['/analyze'] },
  )
  render(<RouterProvider router={router} />)
  return router
}

function readDone(duration: number | 'error') {
  const audio = FakeAudio.made.at(-1)!
  act(() => {
    if (duration === 'error') audio.onerror?.()
    else {
      audio.duration = duration
      audio.onloadedmetadata?.()
    }
  })
}

// S3 with the file picked through the real input.
async function picked(file = wav(), duration: number | 'error' | null = 2) {
  const router = show()
  await userEvent.upload(document.querySelector<HTMLInputElement>('input[type="file"]')!, file, { applyAccept: false })
  if (duration !== null) readDone(duration)
  return router
}

// S4 with the POST held.
async function waiting() {
  const answer = deferred()
  fetchMock.mockReturnValueOnce(answer.promise)
  const router = await picked()
  await userEvent.click(analyzeButton())
  return { answer, router }
}

beforeEach(() => {
  inputs.limits = { status: 'ready', limits: LIMITS }
  inputs.status = { checking: false, result: null }
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  FakeAudio.made = []
  vi.stubGlobal('Audio', FakeAudio)
  URL.createObjectURL = () => 'blob:local/1'
  URL.revokeObjectURL = () => undefined
  markSignedIn('analyst01')
})

afterEach(() => {
  cleanup()
  act(() => clearFileDraft())
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('the enable rule', () => {
  it('enabled when all three checks pass', async () => {
    await picked()
    expect(analyzeButton().disabled).toBe(false)
  })

  it('enabled when the duration is unreadable: the service will check it', async () => {
    await picked(wav(), 'error')
    expect(screen.getByText('Duration could not be read in the browser; the service will check it')).toBeTruthy()
    expect(analyzeButton().disabled).toBe(false)
  })

  it('disabled while the duration is reading', async () => {
    await picked(wav(), null)
    expect(analyzeButton().disabled).toBe(true)
  })

  it.each([
    ['format', wav('call.ogg'), null],
    ['size', wav('call.wav', 8192), 2],
    ['empty', wav('call.wav', 0), null],
    ['too long', wav(), 61],
  ] as const)('disabled when the %s check fails', async (_name, file, duration) => {
    await picked(file, duration)
    expect(analyzeButton().disabled).toBe(true)
  })

  it('no Analyze button while limits are not ready, and the file is kept', () => {
    const file = wav()
    setFileDraft(file)
    inputs.limits = { status: 'loading' }
    show()
    expect(screen.queryByRole('button', { name: 'Analyze recording' })).toBeNull()
    expect(getDraft()).toBe(file)
  })

  it('status Unavailable shows the warning and the button stays enabled', async () => {
    inputs.status = { checking: false, result: { kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-04T10:00:00Z', timeSource: 'browser' } }
    await picked()
    expect(warning()).not.toBeNull()
    expect(analyzeButton().disabled).toBe(false)
  })

  const details = { versions: { model_version: 'mock', profile_version: 'p', policy_version: 'r' }, bundleManifestSha256: null, deferralOnFit: 0.1 }
  const common = { mock: true, checkedAt: '2026-11-04T10:00:00Z', timeSource: 'api', details } as const

  it.each<[string, StatusState['result']]>([
    ['ready', { kind: 'ready', notice: false, failing: [], ...common }],
    ['degraded', { kind: 'ready', notice: true, failing: ['history_and_stats'], ...common }],
    ['no status yet', null],
  ])('no warning when %s', async (_name, result) => {
    inputs.status = { checking: false, result }
    await picked()
    expect(warning()).toBeNull()
  })
})

describe('S4', () => {
  it('the stage list marks Validating done and the wait as the current step', async () => {
    await waiting()
    const stages = within(screen.getByRole('list', { name: 'Progress' })).getAllByRole('listitem')
    expect(stages.map((stage) => stage.textContent)).toEqual(['Validating (done)', 'Uploading and analyzing'])
    expect(stages[1]!.getAttribute('aria-current')).toBe('step')
  })

  it('the info line, with no promise of a notification', async () => {
    await waiting()
    expect(document.getElementById('progress-text')?.textContent).toBe(
      'Uploading and analyzing… The request is sent once. You can safely open other pages; the result will be waiting here.',
    )
  })

  it('the button says Analyzing…, disabled and busy', async () => {
    await waiting()
    expect(busyButton().disabled).toBe(true)
    expect(busyButton().getAttribute('aria-busy')).toBe('true')
  })

  it('the section is busy', async () => {
    await waiting()
    expect(screen.getByRole('region', { name: /^Selected recording/ }).getAttribute('aria-busy')).toBe('true')
  })

  it('no Choose another file, Remove file or Cancel', async () => {
    await waiting()
    for (const name of ['Choose another file', 'Remove file', /cancel/i]) {
      expect(screen.queryByRole('button', { name })).toBeNull()
    }
  })

  it('the checks stay, and no percentage is shown', async () => {
    await waiting()
    expect(screen.getByRole('list', { name: 'Checks before upload' })).toBeTruthy()
    expect(document.body.textContent).not.toContain('%')
  })

  it('"Analysis started." once, politely', async () => {
    await waiting()
    expect(announcer().getAttribute('aria-live')).toBe('polite')
    expect(announcer().textContent).toBe('Analysis started.')
  })

  it('the warning stays during the wait', async () => {
    inputs.status = { checking: false, result: { kind: 'unavailable', source: 'noAnswer', checkedAt: '2026-11-04T10:00:00Z', timeSource: 'browser' } }
    await waiting()
    expect(warning()).not.toBeNull()
  })
})

describe('one POST per press', () => {
  it('a double click sends one POST', async () => {
    fetchMock.mockReturnValue(deferred().promise)
    await picked()
    await userEvent.dblClick(analyzeButton())
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('leaving and coming back resends nothing, and S4 is still there', async () => {
    const { router } = await waiting()
    await act(async () => router.navigate('/other'))
    await act(async () => router.navigate('/analyze'))
    expect(busyButton()).toBeTruthy()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('the name never reaches storage or console', async () => {
    const write = vi.spyOn(Storage.prototype, 'setItem')
    const logs = (['log', 'info', 'warn', 'error', 'debug'] as const).map((method) => vi.spyOn(console, method))
    await waiting()
    expect(write).not.toHaveBeenCalled()
    for (const log of logs) expect(log).not.toHaveBeenCalled()
  })
})

describe('focus', () => {
  it('the press moves focus to the file heading: the button is disabled', async () => {
    await waiting()
    expect(document.activeElement).toBe(fileHeading())
  })
})
