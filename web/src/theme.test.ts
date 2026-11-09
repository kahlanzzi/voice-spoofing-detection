import { afterEach, describe, expect, it, vi } from 'vitest'
import { THEME_KEY, applyTheme, readTheme, writeTheme, type ThemeChoice } from './theme'

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
})

describe('applyTheme', () => {
  it.each<[ThemeChoice, string | null]>([
    ['light', 'light'],
    ['dark', 'dark'],
    ['system', null],
  ])('%s sets data-theme to %s', (choice, expected) => {
    document.documentElement.setAttribute('data-theme', 'dark')
    applyTheme(choice)
    expect(document.documentElement.getAttribute('data-theme')).toBe(expected)
  })
})

describe('readTheme / writeTheme', () => {
  it('round-trips each choice under vsd-theme', () => {
    for (const choice of ['system', 'light', 'dark'] as const) {
      writeTheme(choice)
      expect(window.localStorage.getItem(THEME_KEY)).toBe(choice)
      expect(readTheme()).toBe(choice)
    }
  })

  it('falls back to system when nothing or an invalid value is stored', () => {
    expect(readTheme()).toBe('system')
    window.localStorage.setItem(THEME_KEY, 'sepia')
    expect(readTheme()).toBe('system')
  })

  it('falls back to system when getItem throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    expect(readTheme()).toBe('system')
  })

  it('falls back to system when the localStorage getter throws', () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    expect(readTheme()).toBe('system')
  })

  it('does not throw when setItem throws', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('full', 'QuotaExceededError')
    })
    expect(() => writeTheme('dark')).not.toThrow()
  })
})
