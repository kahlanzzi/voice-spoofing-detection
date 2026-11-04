import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readDuration } from './readDuration'

// jsdom has no media playback and no object URLs, so a fake Audio and stubbed URL functions. What jsdom cannot show,
// the CSP allowing the read, is the e2e valid-WAV test.

class FakeAudio {
  static last: FakeAudio
  preload = ''
  src = ''
  duration = Number.NaN
  onloadedmetadata: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor() {
    FakeAudio.last = this
  }
  removeAttribute(): void {
    this.src = ''
  }
}

const FILE = new Blob(['audio'])
const onDone = vi.fn()
const revoke = vi.fn()

function loaded(duration: number, handler = FakeAudio.last.onloadedmetadata): void {
  FakeAudio.last.duration = duration
  handler?.()
}

beforeEach(() => {
  vi.useFakeTimers()
  onDone.mockReset()
  revoke.mockReset()
  vi.stubGlobal('Audio', FakeAudio)
  URL.createObjectURL = () => 'blob:local/1'
  URL.revokeObjectURL = revoke
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('readDuration', () => {
  it('preload is metadata and src is the object URL', () => {
    readDuration(FILE, onDone)
    expect(FakeAudio.last.preload).toBe('metadata')
    expect(FakeAudio.last.src).toBe('blob:local/1')
  })

  it('loadedmetadata gives the duration', () => {
    readDuration(FILE, onDone)
    loaded(12.5)
    expect(onDone.mock.calls).toEqual([[12.5]])
  })

  it('error → unreadable', () => {
    readDuration(FILE, onDone)
    FakeAudio.last.onerror?.()
    expect(onDone.mock.calls).toEqual([['unreadable']])
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('%d → unreadable', (duration) => {
    readDuration(FILE, onDone)
    loaded(duration)
    expect(onDone.mock.calls).toEqual([['unreadable']])
  })

  it('3 s with no event → unreadable, and not a moment before', () => {
    readDuration(FILE, onDone)
    vi.advanceTimersByTime(2999)
    expect(onDone).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onDone.mock.calls).toEqual([['unreadable']])
  })

  it('a late event after the timer is ignored', () => {
    readDuration(FILE, onDone)
    const late = FakeAudio.last.onloadedmetadata
    vi.advanceTimersByTime(3000)
    loaded(12.5, late)
    expect(onDone.mock.calls).toEqual([['unreadable']])
  })

  it('a late timer after an event is ignored', () => {
    readDuration(FILE, onDone)
    loaded(12.5)
    vi.advanceTimersByTime(3000)
    expect(onDone.mock.calls).toEqual([[12.5]])
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([
    ['an event', () => loaded(1)],
    ['an error', () => FakeAudio.last.onerror?.()],
    ['the timer', () => vi.advanceTimersByTime(3000)],
  ])('the URL is revoked exactly once after %s', (_, end) => {
    const cancel = readDuration(FILE, onDone)
    end()
    cancel()
    expect(revoke.mock.calls).toEqual([['blob:local/1']])
  })

  it('cancel stops the callback, clears the timer and revokes once', () => {
    const cancel = readDuration(FILE, onDone)
    const late = FakeAudio.last.onloadedmetadata
    cancel()
    cancel()
    loaded(12.5, late)
    expect(vi.getTimerCount()).toBe(0)
    expect(onDone).not.toHaveBeenCalled()
    expect(revoke).toHaveBeenCalledTimes(1)
  })
})
