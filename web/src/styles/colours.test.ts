import { expect, it } from 'vitest'

// HANDOFF.md line 93: raw colour values live in tokens.css only. Test files are left out of the
// scan so this file's own patterns do not match.
const files = import.meta.glob<string>(
  ['/src/**/*.{ts,tsx,css}', '!/src/styles/tokens.css', '!/src/**/*.test.{ts,tsx}'],
  { query: '?raw', import: 'default', eager: true },
)

const RAW_COLOUR = /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?)\(/i

it('scans the source files', () => {
  expect(Object.keys(files)).toEqual(
    expect.arrayContaining(['/src/App.tsx', '/src/main.tsx', '/src/theme.ts', '/src/styles/base.css']),
  )
  // CSS is read as text only with test.css enabled (vite.config.ts); an empty file would pass silently.
  expect(files['/src/styles/base.css']).toContain('var(--bg)')
})

it.each(Object.keys(files))('%s has no raw colour value', (path) => {
  expect(files[path]).not.toMatch(RAW_COLOUR)
})
