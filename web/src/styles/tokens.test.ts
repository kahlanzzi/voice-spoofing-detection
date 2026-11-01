import { describe, expect, it } from 'vitest'
import css from './tokens.css?raw'

// Token coverage and WCAG 2.x contrast for tokens.css (Stage 18 PR 1b). Pairs are the ones the
// prototype uses (docs/design/analyst-console-prototype.html); thresholds 4.5:1 text, 3:1 non-text.

type Tokens = Record<string, string>

function block(pattern: RegExp): Tokens {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '')
  const body = pattern.exec(source)?.[1]
  if (body === undefined) throw new Error(`tokens.css: no block for ${pattern}`)
  return Object.fromEntries([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2]!.trim()]))
}

const light = block(/^:root \{([^}]*)\}/m)
const darkMedia = block(/:root:not\(\[data-theme="light"\]\) \{([^}]*)\}/)
const darkAttr = block(/^:root\[data-theme="dark"\] \{([^}]*)\}/m)
const themes = { light, dark: darkMedia } as const

// HANDOFF §4 colour tokens (lines 100-114) and the prototype-sourced ones (lines 14-69).
const THEMED = [
  'bg', 'surface', 'surface-2', 'line', 'line-strong', 'ink', 'ink-2', 'muted',
  'accent', 'accent-hover', 'accent-active', 'accent-soft', 'link', 'focus', 'danger', 'danger-soft',
  'nav', 'nav-2', 'nav-ink', 'nav-indicator',
  'syn-bg', 'syn-fg', 'syn-mark', 'unc-bg', 'unc-fg', 'unc-mark', 'auth-bg', 'auth-fg', 'auth-mark',
  'ok-bg', 'ok-fg', 'ok-mark', 'warn-bg', 'warn-fg', 'warn-mark', 'err-bg', 'err-fg', 'err-mark',
  'demo-bg', 'demo-fg', 'elev-1', 'elev-2',
  'accent-ink', 'nav-ink-strong', 'nav-muted', 'nav-line', 'scrim',
]
// HANDOFF lines 120-126 (and --r-pill from the prototype), plus the proposed --lh-* and --fs-input.
const STATIC = [
  'font-sans', 'font-mono', 'fs-xs', 'fs-sm', 'fs-base', 'fs-md', 'fs-lg', 'fs-xl', 'fs-2xl',
  'sp-1', 'sp-2', 'sp-3', 'sp-4', 'sp-5', 'sp-6', 'sp-7', 'sp-8', 'r-sm', 'r-md', 'r-lg', 'r-pill',
  'target', 'rail-w', 'content-max', 'dur', 'ease', 'lh-body', 'lh-title', 'fs-input',
]

const TEXT_PAIRS: [string, string][] = [
  ...['ink', 'ink-2', 'muted'].flatMap((fg) => ['bg', 'surface', 'surface-2'].map((bg): [string, string] => [fg, bg])),
  ['ink', 'accent-soft'],
  ['link', 'surface'], ['link', 'bg'], ['link', 'surface-2'], ['link', 'accent-soft'],
  ['accent-ink', 'accent'], ['accent-ink', 'accent-hover'], ['accent-ink', 'accent-active'],
  ['danger', 'surface'], ['danger', 'danger-soft'],
  ['bg', 'ink'],
  ['nav-ink', 'nav'], ['nav-ink', 'nav-2'], ['nav-ink-strong', 'nav'], ['nav-ink-strong', 'nav-2'],
  ['nav-muted', 'nav'], ['nav', 'nav-ink'],
  ...['syn', 'unc', 'auth', 'ok', 'warn', 'err'].map((k): [string, string] => [`${k}-fg`, `${k}-bg`]),
  ['err-fg', 'surface'],
  ['demo-fg', 'demo-bg'],
]

const NON_TEXT_PAIRS: [string, string][] = [
  ['line-strong', 'surface'], ['focus', 'surface'], ['focus', 'bg'],
  ['nav-indicator', 'nav'], ['nav-indicator', 'nav-2'],
  ...['syn', 'unc', 'auth', 'ok', 'warn', 'err'].map((k): [string, string] => [`${k}-mark`, 'surface']),
]

// Below 3:1, values unchanged on purpose: waiting on a team answer (Entry 128 open item).
const KNOWN_NON_TEXT_EXCEPTIONS = new Set([
  'light line-strong/surface', 'dark line-strong/surface', 'light unc-mark/surface', 'light warn-mark/surface',
])

function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex)
  if (!m) throw new Error(`not a 6-digit hex colour: ${hex}`)
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(m[1]!.slice(i, i + 2), 16) / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (hi + 0.05) / (lo + 0.05)
}

function ratio(theme: Tokens, fg: string, bg: string): number {
  return contrast(theme[`--${fg}`]!, theme[`--${bg}`]!)
}

describe('tokens.css coverage', () => {
  it.each(THEMED)('--%s is defined in light and both dark blocks', (name) => {
    for (const theme of [light, darkMedia, darkAttr]) expect(theme[`--${name}`]).toBeTruthy()
  })

  it.each(STATIC)('--%s is defined on :root', (name) => {
    expect(light[`--${name}`]).toBeTruthy()
  })

  it('has identical dark blocks for prefers-color-scheme and data-theme="dark"', () => {
    expect(darkAttr).toEqual(darkMedia)
  })
})

describe('contrast', () => {
  for (const [themeName, theme] of Object.entries(themes)) {
    it.each(TEXT_PAIRS)(`${themeName}: text --%s on --%s is at least 4.5:1`, (fg, bg) => {
      expect(ratio(theme, fg, bg)).toBeGreaterThanOrEqual(4.5)
    })

    it.each(NON_TEXT_PAIRS)(`${themeName}: non-text --%s on --%s is at least 3:1 or a known exception`, (fg, bg) => {
      const value = ratio(theme, fg, bg)
      if (KNOWN_NON_TEXT_EXCEPTIONS.has(`${themeName} ${fg}/${bg}`)) {
        // A fixed exception must leave the list, so the list stays exact.
        expect(value).toBeLessThan(3)
      } else {
        expect(value).toBeGreaterThanOrEqual(3)
      }
    })
  }

  it('matches the WCAG reference values', () => {
    expect(contrast('#000000', '#FFFFFF')).toBeCloseTo(21, 5)
    expect(contrast('#FFFFFF', '#FFFFFF')).toBe(1)
  })
})
