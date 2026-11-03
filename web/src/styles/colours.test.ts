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
    expect.arrayContaining([
      '/src/main.tsx',
      '/src/pages/SignIn.tsx',
      '/src/pages/SignIn.css',
      '/src/pages/signInOutcome.ts',
      '/src/components/Icon.tsx',
      '/src/routing/guard.tsx',
      '/src/theme.ts',
      '/src/styles/base.css',
      '/src/shell/Shell.tsx',
      '/src/shell/Shell.css',
      '/src/shell/PageChange.tsx',
      '/src/shell/RailFoot.tsx',
      '/src/shell/RailStatus.tsx',
      '/src/status/status.ts',
      '/src/status/useStatus.ts',
      '/src/status/checkedAt.ts',
      '/src/pages/Status.tsx',
      '/src/pages/Status.css',
      '/src/pages/statusView.ts',
    ]),
  )
  // CSS is read as text only with test.css enabled (vite.config.ts); an empty file would pass silently.
  expect(files['/src/styles/base.css']).toContain('var(--bg)')
  expect(files['/src/pages/SignIn.css']).toContain('var(--err-bg)')
  expect(files['/src/shell/Shell.css']).toContain('var(--nav)')
  expect(files['/src/shell/Shell.css']).toContain('var(--nav-ok)')
  expect(files['/src/pages/Status.css']).toContain('var(--ok-bg)')
})

it.each(Object.keys(files))('%s has no raw colour value', (path) => {
  expect(files[path]).not.toMatch(RAW_COLOUR)
})
