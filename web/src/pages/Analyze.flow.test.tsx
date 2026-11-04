import { act, cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getSnapshot as getSession, markSignedIn } from '../session/session'
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

const ANALYSIS = {
  prediction: 'authentic',
  confidence: 0.95,
  risk_level: 'LOW',
  model_version: 'mock',
  profile_version: 'preprocessing-v1',
  policy_version: 'risk-v1',
  request_id: `req_${'c'.repeat(32)}`,
  notice: 'Decision support only.',
}

const ID = `req_${'d'.repeat(32)}`

const fetchMock = vi.fn<typeof fetch>()

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

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

const button = (name: string) => screen.getByRole<HTMLButtonElement>('button', { name })

function apiError(status: number, code: string, requestId: string | null = ID): Response {
  return json(status, { error: { code, message: 'A server sentence.', request_id: requestId } })
}

// S5, through the press.
async function failed(answer: Response | 'network') {
  if (answer === 'network') fetchMock.mockRejectedValueOnce(new TypeError('network'))
  else fetchMock.mockResolvedValueOnce(answer)
  const router = await picked()
  await userEvent.click(analyzeButton())
  await screen.findByRole('alert')
  return router
}

async function finish(answer: ReturnType<typeof deferred>, response = json(201, ANALYSIS)) {
  await act(async () => answer.resolve(response))
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

  it('a result moves focus to its title', async () => {
    const { answer } = await waiting()
    await finish(answer)
    expect(document.activeElement).toBe(screen.getByRole('heading', { level: 2, name: 'Likely authentic voice' }))
  })

  it('a result is spoken assertively', async () => {
    const { answer } = await waiting()
    await finish(answer)
    const region = document.getElementById('analyze-assertive')!
    expect(region.getAttribute('aria-live')).toBe('assertive')
    expect(region.textContent).toBe(
      'Result ready. Verdict: Likely authentic voice. Low risk. Required action: continue your normal verification procedure.',
    )
  })

  it('a result is no alert (Q22)', async () => {
    const { answer } = await waiting()
    await finish(answer)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('Retry moves focus to the file heading too', async () => {
    await failed(apiError(500, 'internal_error'))
    fetchMock.mockReturnValueOnce(deferred().promise)
    await userEvent.click(button('Retry analysis'))
    expect(document.activeElement).toBe(fileHeading())
  })

  it('an error moves focus to its title', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(document.activeElement).toBe(screen.getByRole('heading', { level: 2, name: 'Something went wrong on the service' }))
  })

  it('an error is exactly one alert', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(screen.getAllByRole('alert')).toHaveLength(1)
  })

  it('an error is not also spoken politely (Q22)', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(announcer().textContent).toBe('Analysis started.')
    expect(document.getElementById('analyze-assertive')?.textContent).toBe('')
  })

  it('an answer while away moves no focus and speaks nothing', async () => {
    const { answer, router } = await waiting()
    await act(async () => router.navigate('/other'))
    await finish(answer)
    await act(async () => router.navigate('/analyze'))
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Analysis result')
    expect(document.activeElement).toBe(document.body)
    expect(document.getElementById('analyze-assertive')?.textContent).toBe('')
  })
})

describe('the result page', () => {
  it('the h1, the line under it and New analysis', async () => {
    const { answer } = await waiting()
    await finish(answer)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Analysis result')
    expect(document.querySelector('.page-sub')?.textContent).toBe('New result')
    expect(screen.getByRole('button', { name: 'New analysis' })).toBeTruthy()
  })

  it('no static notice: the card has the service\'s own', async () => {
    const { answer } = await waiting()
    await finish(answer)
    expect(document.querySelector('.notice')).toBeNull()
    expect(document.querySelector('.api-notice')?.textContent).toContain(ANALYSIS.notice)
  })

  it('the result stays after leaving and coming back', async () => {
    const { answer, router } = await waiting()
    await finish(answer)
    await act(async () => router.navigate('/other'))
    await act(async () => router.navigate('/analyze'))
    expect(screen.getByRole('heading', { level: 2, name: 'Likely authentic voice' })).toBeTruthy()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('the draft after a result', () => {
  it('a result keeps the draft: the card names the file', async () => {
    const { answer } = await waiting()
    await finish(answer)
    expect(getDraft()).not.toBeNull()
    expect(document.querySelector('.result-body')?.textContent).toContain('call.wav')
  })

  it("the result's New analysis clears it and gives S2", async () => {
    const { answer } = await waiting()
    await finish(answer)
    await userEvent.click(screen.getByRole('button', { name: 'New analysis' }))
    expect(getDraft()).toBeNull()
    expect(screen.getByRole('button', { name: 'Choose recording' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Analyze a recording')
  })
})

describe('S5', () => {
  const REFUSED = ['Choose another file', 'Remove file']
  const RETRY = ['Retry analysis', 'Choose another file']

  it.each<[string, () => Response | 'network', string, string, string, string[]]>([
    [
      'network', () => 'network', 'Network interruption', 'The connection to the service was interrupted',
      'Your browser could not reach the service, so the recording was not analyzed. Check your connection, then retry. Your file is still selected.',
      RETRY,
    ],
    [
      'service', () => apiError(503, 'analysis_unavailable'), 'Analysis service unavailable', 'The analysis service is unavailable',
      'The service received the request but cannot analyze recordings right now. Your file is still selected. Try again in a few minutes. If it keeps failing, report the request ID through your usual support channel.',
      ['Retry analysis', 'Check system status'],
    ],
    [
      'timeout', () => apiError(504, 'analysis_timeout'), 'Request timeout', 'The analysis did not finish in time',
      'The service did not answer before the console stopped waiting. Your file is still selected. Retrying sends a new request with a new request ID.',
      RETRY,
    ],
    [
      'corrupt', () => apiError(422, 'corrupt_audio'), 'Corrupt audio', 'The recording could not be decoded',
      'The file passed the checks in your browser, but the service could not decode the audio. It may be damaged, or not really in the format its name suggests. Export the segment again and choose the new file.',
      REFUSED,
    ],
    [
      'session', () => apiError(401, 'session_expired', null), 'Session expired', 'Your session has expired',
      'Sign in again to continue. Your selected file stays on this screen and nothing was analyzed.',
      ['Sign in again'],
    ],
    [
      'server', () => apiError(500, 'internal_error'), 'Unexpected service error', 'Something went wrong on the service',
      'The request reached the service but failed unexpectedly. Your file is still selected. Retry once. If it fails again, report the request ID through your usual support channel.',
      RETRY,
    ],
    [
      'format', () => apiError(415, 'unsupported_format'), 'Unsupported format', 'The service does not accept this format',
      'The file passed the checks in your browser, but the service refused its format. Choose a recording in an accepted format.',
      REFUSED,
    ],
    [
      'size', () => apiError(413, 'file_too_large'), 'File too large', 'The file is over the service’s size limit',
      'The file passed the checks in your browser, but the service refused its size. Export only the segment with the request.',
      REFUSED,
    ],
    [
      'duration', () => apiError(422, 'recording_too_long'), 'Recording too long', 'The recording is over the duration limit',
      'The service measured the recording and it is over the limit. Upload only the segment with the request.',
      REFUSED,
    ],
    [
      'empty', () => apiError(422, 'empty_file'), 'Empty file', 'The recording has no audio',
      'The service found no audio in this file. Export the segment again and choose the new file.',
      REFUSED,
    ],
  ])('%s: its badge, title, body, actions and alert', async (_kind, answer, badge, title, body, actions) => {
    await failed(answer())
    const panel = screen.getByRole('region', { name: title })
    expect(panel.querySelector('.alert-kind')?.textContent).toBe(badge)
    expect(panel.querySelector('.explain')?.textContent).toBe(body)
    expect([...panel.querySelectorAll('.actions > *')].map((action) => action.textContent)).toEqual(actions)
    expect(screen.getByRole('alert').textContent).toBe(`${title}. The analysis did not complete. No verdict was produced.`)
  })

  it('no verdict, risk or model output on any kind', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(screen.getByText('The analysis did not complete. No verdict was produced.')).toBeTruthy()
    expect(document.querySelector('.result')).toBeNull()
    for (const word of ['Likely', 'Cannot decide', 'RISK', 'Calibrated', 'Mock result']) {
      expect(document.body.textContent).not.toContain(word)
    }
  })

  it('the request ID from the body', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(screen.getByText(ID).tagName).toBe('CODE')
  })

  it('no request ID says None', async () => {
    await failed('network')
    expect(screen.getByText('None: the request was not accepted by the service')).toBeTruthy()
  })

  it('the server message is never shown', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(document.body.textContent).not.toContain('A server sentence.')
  })

  it('the timeout says nothing about a late result in History', async () => {
    await failed(apiError(504, 'analysis_timeout'))
    expect(document.body.textContent).not.toContain('History')
  })

  it('Retry analysis goes to S4', async () => {
    await failed(apiError(500, 'internal_error'))
    fetchMock.mockReturnValueOnce(deferred().promise)
    await userEvent.click(button('Retry analysis'))
    expect(busyButton().disabled).toBe(true)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('Check system status links /status', async () => {
    await failed(apiError(503, 'analysis_unavailable'))
    expect(screen.getByRole('link', { name: 'Check system status' }).getAttribute('href')).toBe('/status')
  })

  it('Choose another file opens the input, and a pick gives S3', async () => {
    await failed(apiError(422, 'corrupt_audio'))
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!
    const click = vi.spyOn(input, 'click')
    await userEvent.click(button('Choose another file'))
    expect(click).toHaveBeenCalledTimes(1)
    await userEvent.upload(input, wav('other.wav'), { applyAccept: false })
    expect(fileHeading()).toBeTruthy()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('Remove file gives S2', async () => {
    await failed(apiError(422, 'corrupt_audio'))
    await userEvent.click(button('Remove file'))
    expect(getDraft()).toBeNull()
    expect(document.activeElement).toBe(button('Choose recording'))
  })

  it('Sign in again ends the session, and S3 follows with the file', async () => {
    await failed(apiError(401, 'session_expired', null))
    await userEvent.click(button('Sign in again'))
    expect(getSession()).toEqual({ status: 'signedOut', reason: 'expired' })
    expect(getDraft()).not.toBeNull()
    expect(fileHeading()).toBeTruthy()
  })

  it('the file name row', async () => {
    await failed(apiError(500, 'internal_error'))
    expect(document.querySelector('.alert .kv dd')?.textContent).toBe('call.wav (this screen only)')
  })
})
