import { expect, it } from 'vitest'

// HANDOFF.md line 93: raw colour values live in tokens.css only. Test files are left out of the
// scan so this file's own patterns do not match.
// No type argument on this call: Vitest leaves a file out of its module cache only when it finds the glob call
// spelled without one (cli-api, generateCachePath). With a type argument the file list froze, so new files went
// unscanned and the test count depended on the cache (Entry 144).
const files = import.meta.glob(
  ['/src/**/*.{ts,tsx,css}', '!/src/styles/tokens.css', '!/src/**/*.test.{ts,tsx}'],
  { query: '?raw', import: 'default', eager: true },
) as Record<string, string>

const RAW_COLOUR = /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?)\(/i

// A fresh listing, so a frozen file list fails here. Not typed through @types/node: this file is in the app's
// tsconfig, which keeps Node's globals out of browser code. Relative to web/, where Vitest runs.
const NODE_FS: string = 'node:fs'

it('scans every source file on disk', async () => {
  const { readdirSync } = (await import(/* @vite-ignore */ NODE_FS)) as {
    readdirSync(path: string, options: { recursive: true }): string[]
  }
  const onDisk = readdirSync('src', { recursive: true })
    .filter((path) => /\.(ts|tsx|css)$/.test(path) && !/\.test\.tsx?$/.test(path) && path !== 'styles/tokens.css')
    .map((path) => `/src/${path}`)
  expect(Object.keys(files).sort()).toEqual(onDisk.sort())
})

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
      '/src/components/Button.tsx',
      '/src/components/Button.css',
      '/src/components/TextField.tsx',
      '/src/components/TextField.css',
      '/src/components/Banner.tsx',
      '/src/components/Banner.css',
      '/src/components/Spinner.tsx',
      '/src/components/Spinner.css',
    ]),
  )
  // CSS is read as text only with test.css enabled (vite.config.ts); an empty file would pass silently.
  expect(files['/src/styles/base.css']).toContain('var(--bg)')
  expect(files['/src/pages/SignIn.css']).toContain('var(--surface)')
  expect(files['/src/components/Banner.css']).toContain('var(--err-bg)')
  expect(files['/src/components/Button.css']).toContain('var(--accent)')
  expect(files['/src/shell/Shell.css']).toContain('var(--nav)')
  expect(files['/src/shell/Shell.css']).toContain('var(--nav-ok)')
  expect(files['/src/pages/Status.css']).toContain('var(--ok-bg)')
})

it.each(Object.keys(files))('%s has no raw colour value', (path) => {
  expect(files[path]).not.toMatch(RAW_COLOUR)
})
