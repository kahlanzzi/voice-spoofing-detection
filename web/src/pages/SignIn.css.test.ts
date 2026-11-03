import { expect, it } from 'vitest'
import css from './SignIn.css?raw'

// HANDOFF line 86: at ≤ 640 px every target is at least 44 × 44 px, the Show/Hide toggle included (it is 36 px
// high above that, prototype line 277). jsdom applies no media queries, so the rule is checked as text.

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

it('the Show/Hide toggle is a 44 × 44 px target at ≤ 640 px', () => {
  const rule = /\.pw button\s*\{([^}]*)\}/.exec(mediaBlock('max-width: 640px'))?.[1] ?? ''
  expect(rule).toMatch(/(?:^|;)\s*height:\s*var\(--target\)/)
  expect(rule).toMatch(/min-width:\s*var\(--target\)/)
})

it('above 640 px the toggle keeps the prototype height', () => {
  const outside = css.replace(mediaBlock('max-width: 640px'), '')
  expect(/\.pw button\s*\{([^}]*)\}/.exec(outside)?.[1]).toMatch(/height:\s*36px/)
})
