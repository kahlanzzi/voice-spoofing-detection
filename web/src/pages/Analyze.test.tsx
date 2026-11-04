import { act, cleanup, render, screen, within } from '@testing-library/react'
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

const SIGNED_IN = vi.hoisted(() => ({ status: 'signedIn', username: 'analyst01' }) as const)

vi.mock('../session/session', () => ({
  getSnapshot: () => SIGNED_IN,
  subscribe: () => () => undefined,
}))

const LIMITS = { formats: ['wav', 'flac', 'mp3'], max_size_bytes: 20971520, max_duration_s: 60, audio_deletion: 'Deleted when processing finishes' }
const READY: LimitsState = { status: 'ready', limits: LIMITS }

const fetchMock = vi.fn<typeof fetch>()

const chooseButton = () => screen.getByRole<HTMLButtonElement>('button', { name: 'Choose recording' })
const requirements = () => screen.queryByLabelText('Upload requirements')
const fileInput = () => document.querySelector<HTMLInputElement>('input[type="file"]')

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
