import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LimitsState } from '../upload/limits'
import { clearFileDraft, getSnapshot as getDraft } from '../upload/fileDraft'
import Analyze from './Analyze'

// S2 and S3 on a stand-in limits store whose state the test sets (the Status.test pattern), the real draft store and a
// fake Audio. fetch must never be called: the shell owns the one limits request.

const store = vi.hoisted(() => {
  let state: LimitsState = { status: 'idle' }
  const listeners = new Set<() => void>()
  return {
    get: (): LimitsState => state,
    set(next: LimitsState): void {
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

const retryLimits = vi.hoisted(() => vi.fn<() => Promise<void>>())

vi.mock('../upload/limits', () => ({
  getSnapshot: () => store.get(),
  subscribe: (listener: () => void) => store.subscribe(listener),
  loadLimits: vi.fn(),
  retryLimits,
}))

const sessionStore = vi.hoisted(() => {
  const signedIn = { status: 'signedIn', username: 'analyst01' } as const
  let state: { status: string; username?: string; reason?: string } = signedIn
  const listeners = new Set<() => void>()
  return {
    get: () => state,
    set(next: typeof state): void {
      state = next
      for (const listener of [...listeners]) listener()
    },
    reset: (): void => {
      state = signedIn
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
  getSnapshot: () => sessionStore.get(),
  subscribe: (listener: () => void) => sessionStore.subscribe(listener),
  expire: vi.fn(),
}))

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

const LIMITS = { formats: ['wav', 'flac', 'mp3'], max_size_bytes: 20971520, max_duration_s: 60, audio_deletion: 'Deleted when processing finishes' }
const READY: LimitsState = { status: 'ready', limits: LIMITS }

const fetchMock = vi.fn<typeof fetch>()

const chooseButton = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Choose recording' })
const requirements = () => screen.queryByLabelText('Upload requirements')
const fileInput = () => document.querySelector<HTMLInputElement>('input[type="file"]')

const wav = (name = 'call.wav', size = 1024) => new File([new Uint8Array(size)], name)
const rows = () => [...screen.getByRole('list', { name: 'Checks before upload' }).querySelectorAll('li')].map((row) => row.textContent)
const fileHeading = () => screen.getByRole('heading', { level: 2, name: /^Selected recording/ })
const announcer = () => document.getElementById('analyze-announcer')!
const analyzeButton = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Analyze recording' })

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
async function picked(file = wav()) {
  show(READY)
  await userEvent.upload(fileInput()!, file, { applyAccept: false })
}

function show(state: LimitsState) {
  store.set(state)
  return render(<Analyze />)
}

beforeEach(() => {
  store.set({ status: 'idle' })
  retryLimits.mockReset()
  retryLimits.mockResolvedValue(undefined)
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  sessionStore.reset()
  FakeAudio.made = []
  vi.stubGlobal('Audio', FakeAudio)
  URL.createObjectURL = () => 'blob:local/1'
  URL.revokeObjectURL = () => undefined
})

afterEach(() => {
  cleanup()
  act(() => clearFileDraft())
  expect(fetchMock).not.toHaveBeenCalled()
  vi.unstubAllGlobals()
})

describe('S2: the limits', () => {
  it('idle shows the loading line', () => {
    show({ status: 'idle' })
    expect(screen.getByRole('status').textContent).toBe('Loading upload limits from the service…')
  })

  it('loading has role status, no requirements, and the button is disabled', () => {
    show({ status: 'loading' })
    expect(screen.getByRole('status').textContent).toBe('Loading upload limits from the service…')
    expect(requirements()).toBeNull()
    expect(chooseButton().disabled).toBe(true)
  })

  it("failed has role alert with the prototype's text, no requirements, and the button is disabled", () => {
    show({ status: 'failed' })
    expect(screen.getByRole('alert').textContent).toBe(
      'Upload limits could not be loaded. Analysis stays off until they load, so the console never guesses them. Retry',
    )
    expect(screen.queryByRole('status')).toBeNull()
    expect(requirements()).toBeNull()
    expect(chooseButton().disabled).toBe(true)
  })

  it('Retry calls retryLimits and moves focus to the section heading', async () => {
    show({ status: 'failed' })
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retryLimits).toHaveBeenCalledTimes(1)
    act(() => store.set({ status: 'loading' }))
    expect(document.activeElement).toBe(screen.getByRole('heading', { level: 2, name: 'Choose the recorded segment' }))
  })

  it('ready shows the four items, formatted', () => {
    show(READY)
    const list = screen.getByLabelText('Upload requirements')
    expect(list.tagName).toBe('DL')
    expect([...list.querySelectorAll('dt')].map((term) => term.textContent)).toEqual([
      'Accepted formats',
      'Maximum size',
      'Maximum duration',
      'Audio after analysis',
    ])
    expect([...list.querySelectorAll('dd')].map((value) => value.textContent)).toEqual([
      'WAV, FLAC, MP3',
      '20 MB',
      '60 s',
      'Deleted when processing finishes',
    ])
  })

  it('the values follow the limits', () => {
    show({ status: 'ready', limits: { formats: ['ogg'], max_size_bytes: 20_000_000, max_duration_s: 2, audio_deletion: 'Kept for one hour' } })
    expect([...screen.getByLabelText('Upload requirements').querySelectorAll('dd')].map((value) => value.textContent)).toEqual([
      'OGG',
      '19.1 MB',
      '2 s',
      'Kept for one hour',
    ])
  })

  it.each<[LimitsState, number]>([
    [READY, 1],
    [{ status: 'loading' }, 0],
    [{ status: 'failed' }, 0],
  ])('the Mock tag is static and shown only when ready: %j', (state, count) => {
    show(state)
    expect(screen.queryAllByText('Mock values · real limits not decided')).toHaveLength(count)
  })
})

describe('S2: the page', () => {
  it('one primary button, enabled only when ready', () => {
    show(READY)
    expect(document.querySelectorAll('.btn-primary')).toHaveLength(1)
    expect(chooseButton().disabled).toBe(false)
  })

  it('the input is hidden, not focusable, and accepts what the limits say', () => {
    show(READY)
    const input = fileInput()!
    expect(input.className).toBe('sr')
    expect(input.tabIndex).toBe(-1)
    expect(input.getAttribute('aria-hidden')).toBe('true')
    expect(input.accept).toBe('.wav,.flac,.mp3,audio/*')
    expect(document.querySelectorAll('input[type="file"]')).toHaveLength(1)
  })

  it.each<[LimitsState]>([[{ status: 'idle' }], [{ status: 'loading' }], [{ status: 'failed' }], [READY]])(
    'the privacy text and the notice are on every state: %j',
    (state) => {
      show(state)
      expect(screen.getByText(/^The audio is used only for this analysis\./)).toBeTruthy()
      expect(screen.getByText('Decision support only.')).toBeTruthy()
    },
  )

  it.each<[LimitsState]>([[{ status: 'idle' }], [{ status: 'loading' }], [{ status: 'failed' }]])('no input until ready: %j', (state) => {
    show(state)
    expect(fileInput()).toBeNull()
  })

  it('no drop heading: there is no drop target', () => {
    show(READY)
    expect(screen.queryByText(/drop/i)).toBeNull()
    expect(within(screen.getByRole('region', { name: 'Choose the recorded segment' })).getByText('Browse for it on your computer.')).toBeTruthy()
  })

  it('Choose recording opens the input', async () => {
    show(READY)
    const click = vi.spyOn(fileInput()!, 'click')
    await userEvent.click(chooseButton())
    expect(click).toHaveBeenCalledTimes(1)
  })

  it('a pick stores the File', async () => {
    show(READY)
    const file = new File(['audio'], 'call.wav')
    await userEvent.upload(fileInput()!, file)
    expect(getDraft()).toBe(file)
  })
})

describe('S3: the file card', () => {
  it('a pick stores the File and focuses the file heading', async () => {
    const file = wav()
    show(READY)
    // fireEvent: userEvent.upload refocuses the input after its stand-in dialog, which a browser does not do.
    fireEvent.change(fileInput()!, { target: { files: [file] } })
    expect(getDraft()).toBe(file)
    expect(document.activeElement).toBe(fileHeading())
  })

  it('a change with no file keeps the draft', async () => {
    const file = wav()
    await picked(file)
    fireEvent.change(fileInput()!, { target: { files: [] } })
    expect(getDraft()).toBe(file)
  })

  it('name, size and the duration line', async () => {
    await picked(wav('call-with-a-customer.wav', 2048))
    expect(screen.getByText('call-with-a-customer.wav')).toBeTruthy()
    expect(screen.getByText('2 KB · reading duration…')).toBeTruthy()
    readDone(12.5)
    expect(screen.getByText('2 KB · duration 12.5 s')).toBeTruthy()
  })

  it('reading shows the waiting row, with no prefix', async () => {
    await picked()
    expect(rows()).toEqual(['Passed: Format WAV is accepted', 'Passed: Size 1 KB is within 20 MB', 'Reading duration…'])
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('all pass: three "Passed:" rows and no alert', async () => {
    await picked()
    readDone(12.5)
    expect(rows()).toEqual([
      'Passed: Format WAV is accepted',
      'Passed: Size 1 KB is within 20 MB',
      'Passed: Duration 12.5 s is within 60 s',
    ])
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it.each<[string, File, number | null, string]>([
    ['format', wav('call.ogg'), null, 'Failed: Unsupported format: .ogg files cannot be analyzed. Use WAV, FLAC, MP3.'],
    ['too large', wav('big.wav', 20971521), 12, 'Failed: File too large: 20.0 MB is over the 20 MB limit. Export only the segment with the request.'],
    ['too long', wav(), 61, 'Failed: Recording too long: 1 min 1 s is over the 60 s limit. Upload only the segment with the request.'],
    ['empty', wav('call.wav', 0), null, 'Failed: Empty file: it contains no data. Export the segment again and choose the new file.'],
  ])('%s fails: a "Failed:" row and the alert', async (_, file, duration, text) => {
    await picked(file)
    if (duration !== null) readDone(duration)
    expect(rows()).toContain(text)
    expect(screen.getByRole('alert').textContent).toBe(
      'Input error: this file cannot be analyzed. Choose another recording that passes every check. Nothing was sent and no verdict was made.',
    )
  })

  it('unreadable: no prefix and no alert', async () => {
    await picked()
    readDone('error')
    expect(rows()[2]).toBe('Duration could not be read in the browser; the service will check it')
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByText('1 KB · duration unknown')).toBeTruthy()
  })

  it.each([
    ['unsupported', wav('call.ogg')],
    ['empty', wav('call.wav', 0)],
  ])('%s creates no Audio', async (_, file) => {
    await picked(file)
    expect(FakeAudio.made).toHaveLength(0)
    expect(rows()[2]).toBe('Duration not checked: the file cannot be read as audio')
  })

  it('Analyze recording is disabled while the duration is read and when a check fails', async () => {
    await picked()
    expect(analyzeButton().disabled).toBe(true)
    readDone(61)
    expect(analyzeButton().disabled).toBe(true)
  })

  it('Choose another file opens the input; there is still only one', async () => {
    await picked()
    const click = vi.spyOn(fileInput()!, 'click')
    await userEvent.click(screen.getByRole('button', { name: 'Choose another file' }))
    expect(click).toHaveBeenCalledTimes(1)
    expect(document.querySelectorAll('input[type="file"]')).toHaveLength(1)
  })

  it('Remove file clears the draft, focuses Choose recording and announces', async () => {
    await picked()
    await userEvent.click(screen.getByRole('button', { name: 'Remove file' }))
    expect(getDraft()).toBeNull()
    expect(document.activeElement).toBe(chooseButton())
    expect(announcer().textContent).toBe('File removed.')
  })

  it('the announcement waits for the read, and says All checks passed only when all pass', async () => {
    await picked(wav('good.wav'))
    expect(announcer().textContent).toBe('')
    readDone(12.5)
    expect(announcer().textContent).toBe('File selected: good.wav. All checks passed.')
    await userEvent.upload(fileInput()!, wav('long.wav'))
    readDone(61)
    expect(announcer().textContent).toBe('File selected: long.wav.')
  })

  it('the announcement does not repeat on a remount', async () => {
    await picked()
    readDone(12.5)
    cleanup()
    render(<Analyze />)
    readDone(12.5)
    expect(announcer().textContent).toBe('')
  })

  it.each<[LimitsState]>([[{ status: 'loading' }], [{ status: 'failed' }]])('a kept draft with limits not ready shows only the limits block: %j', async (state) => {
    await picked()
    act(() => store.set(state))
    expect(screen.queryByRole('list', { name: 'Checks before upload' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Choose recording' })).toBeNull()
    expect(fileInput()).toBeNull()
    expect(document.querySelector('.limits-wait, .limits-failed')).not.toBeNull()
  })

  it('New analysis (the draft cleared) returns to S2', async () => {
    await picked()
    act(() => clearFileDraft())
    expect(chooseButton().disabled).toBe(false)
    expect(screen.queryByRole('list', { name: 'Checks before upload' })).toBeNull()
  })

  it('sign-out clears S3', async () => {
    await picked()
    act(() => sessionStore.set({ status: 'signedOut', reason: 'signOut' }))
    expect(getDraft()).toBeNull()
    expect(screen.queryByRole('list', { name: 'Checks before upload' })).toBeNull()
  })

  it('the name never reaches storage or console', async () => {
    const spies = [
      vi.spyOn(Storage.prototype, 'setItem'),
      ...(['log', 'info', 'warn', 'error', 'debug'] as const).map((method) => vi.spyOn(console, method)),
    ]
    await picked(wav('secret-name.wav'))
    readDone(12.5)
    for (const spy of spies) {
      expect(JSON.stringify(spy.mock.calls)).not.toContain('secret-name')
      spy.mockRestore()
    }
  })
})

it.each(['S2', 'S3'])('no "Demo scenarios" panel on %s', async (screenName) => {
  if (screenName === 'S2') show(READY)
  else await picked()
  expect(screen.queryByRole('heading', { name: 'Demo scenarios' })).toBeNull()
})
