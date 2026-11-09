import { expect, it } from 'vitest'
import css from './Button.css?raw'

// Button.css (Stage 18 PR 8a). jsdom applies no cascade or media queries, so the rules are checked as text.

function mediaBlock(query: string): string {
  const start = css.indexOf(`@media (${query})`)
  expect(start).toBeGreaterThan(-1)
  let depth = 0
  for (let i = css.indexOf('{', start); i < css.length; i += 1) {
    if (css[i] === '{') depth += 1
    if (css[i] === '}' && --depth === 0) return css.slice(start, i + 1)
  }
  throw new Error(`unclosed @media (${query})`)
}

it('.btn comes before .btn-sm and .btn-quiet, so they win at the same specificity (Check now, Entry 141)', () => {
  const base = css.indexOf('.btn {')
  expect(base).toBeGreaterThan(-1)
  expect(base).toBeLessThan(css.indexOf('.btn-sm {'))
  expect(base).toBeLessThan(css.indexOf('.btn-quiet {'))
})

it('at ≤ 640 px .btn-sm and .btn-link are --target high (HANDOFF line 86)', () => {
  const rule = /\.btn-sm, \.btn-link\s*\{([^}]*)\}/.exec(mediaBlock('max-width: 640px'))?.[1] ?? ''
  expect(rule).toMatch(/min-height:\s*var\(--target\)/)
})

it('above 640 px .btn-sm is 36 px high (HANDOFF line 137)', () => {
  const outside = css.replace(mediaBlock('max-width: 640px'), '')
  expect(/\.btn-sm\s*\{([^}]*)\}/.exec(outside)?.[1]).toMatch(/min-height:\s*36px/)
})
