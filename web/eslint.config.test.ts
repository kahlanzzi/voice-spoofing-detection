// @vitest-environment node
import { ESLint } from 'eslint'
import { beforeAll, describe, expect, it } from 'vitest'

// The no-timer rule in eslint.config.js: an error in src/api, not elsewhere.

const eslint = new ESLint()

// The first lint loads the config and its plugins, which can take longer than the 5 s default.
beforeAll(() => eslint.lintText('', { filePath: 'src/api/warm-up.ts' }), 60_000)

async function ruleIds(code: string, filePath: string): Promise<(string | null)[]> {
  const [result] = await eslint.lintText(code, { filePath })
  return result!.messages.map((message) => message.ruleId)
}

describe('no timers in src/api', () => {
  it.each([
    'setTimeout(() => undefined, 1)\n',
    'setInterval(() => undefined, 1)\n',
    'window.setTimeout(() => undefined, 1)\n',
    'globalThis.setInterval(() => undefined, 1)\n',
  ])('%s is an error in src/api', async (code) => {
    expect(await ruleIds(code, 'src/api/planted.ts')).toContain('no-restricted-syntax')
  })

  it('the same call outside src/api is allowed', async () => {
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/theme.ts')).not.toContain('no-restricted-syntax')
  })
})

// Session state (PR 3b): the timer rule also covers src/session, tests included; storage, cookie and
// console are errors there too, tests exempt. The two bans use different rules, so neither replaces the other.

describe('no timers, storage, cookie or console in src/session', () => {
  it.each([
    'setTimeout(() => undefined, 1)\n',
    'window.setTimeout(() => undefined, 1)\n',
    'globalThis.setInterval(() => undefined, 1)\n',
  ])('%s is a timer error in src/session', async (code) => {
    expect(await ruleIds(code, 'src/session/planted.ts')).toContain('no-restricted-syntax')
  })

  it.each([
    ["localStorage.setItem('k', 'v')\n", 'no-restricted-globals'],
    ["sessionStorage.getItem('k')\n", 'no-restricted-globals'],
    ["window.localStorage.setItem('k', 'v')\n", 'no-restricted-properties'],
    ["globalThis.sessionStorage.getItem('k')\n", 'no-restricted-properties'],
    ["self.localStorage.setItem('k', 'v')\n", 'no-restricted-properties'],
    ["document.cookie = 'x'\n", 'no-restricted-properties'],
    ["console.log('x')\n", 'no-restricted-globals'],
    ["window.console.log('x')\n", 'no-restricted-properties'],
    ["globalThis.console.error('x')\n", 'no-restricted-properties'],
  ])('%s is an error in src/session', async (code, rule) => {
    expect(await ruleIds(code, 'src/session/planted.ts')).toContain(rule)
  })

  it('a timer and storage in one file report both rules', async () => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('k', 'v')\n", 'src/session/planted.ts')
    expect(ids).toContain('no-restricted-syntax')
    expect(ids).toContain('no-restricted-globals')
  })

  it('console in a session test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/session/planted.test.ts')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/session/planted.test.ts')).toContain(
      'no-restricted-syntax',
    )
  })

  it('storage and console outside src/session are allowed', async () => {
    expect(await ruleIds("localStorage.setItem('k', 'v')\nconsole.log('x')\n", 'src/theme.ts')).toEqual([])
  })
})

// Routing (PR 3c): next and route state live only in the URL, so src/routing gets both bans; test files,
// .ts or .tsx, are exempt from the storage and console ban only.

describe('no timers, storage or console in src/routing', () => {
  it.each([
    ['setTimeout(() => undefined, 1)\n', 'no-restricted-syntax'],
    ['window.setInterval(() => undefined, 1)\n', 'no-restricted-syntax'],
    ["localStorage.setItem('next', '/analyze')\n", 'no-restricted-globals'],
    ["window.sessionStorage.setItem('next', '/analyze')\n", 'no-restricted-properties'],
    ["console.log('next')\n", 'no-restricted-globals'],
    ["globalThis.console.info('next')\n", 'no-restricted-properties'],
  ])('%s is an error in src/routing', async (code, rule) => {
    expect(await ruleIds(code, 'src/routing/planted.ts')).toContain(rule)
  })

  it('console in a .tsx test file is allowed, a timer is not', async () => {
    for (const filePath of ['src/routing/planted.test.tsx', 'src/session/planted.test.tsx']) {
      expect(await ruleIds("console.log('x')\n", filePath)).toEqual([])
      expect(await ruleIds('setTimeout(() => undefined, 1)\n', filePath)).toContain('no-restricted-syntax')
    }
  })
})

// The shell (PR 6a): no timers (page changes run in effects, not on a delay) and no storage or console (the theme
// is stored by theme.ts only).

describe('no timers, storage or console in src/shell', () => {
  it.each([
    ['setTimeout(() => undefined, 40)\n', 'no-restricted-syntax'],
    ['window.setInterval(() => undefined, 1)\n', 'no-restricted-syntax'],
    ["localStorage.setItem('vsd-theme', 'dark')\n", 'no-restricted-globals'],
    ["window.localStorage.getItem('vsd-theme')\n", 'no-restricted-properties'],
    ["console.log('x')\n", 'no-restricted-globals'],
  ])('%s is an error in src/shell', async (code, rule) => {
    expect(await ruleIds(code, 'src/shell/planted.tsx')).toContain(rule)
  })

  it('console in a shell test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/shell/planted.test.tsx')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/shell/planted.test.tsx')).toContain('no-restricted-syntax')
  })

  // PR 6b: the rail footer stores the theme through theme.ts only, and sign-out lives in session.ts.
  it.each(['src/shell/RailFoot.tsx', 'src/session/session.ts'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('vsd-theme', 'dark')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })
})

// The status store (PR 7a): no polling, so no timer (a relative "x minutes ago" would need one), and in memory only.

describe('no timers, storage or console in src/status', () => {
  it.each([
    ['setInterval(() => undefined, 60_000)\n', 'no-restricted-syntax'],
    ['globalThis.setTimeout(() => undefined, 1)\n', 'no-restricted-syntax'],
    ["sessionStorage.setItem('status', 'ready')\n", 'no-restricted-globals'],
    ["console.log('x')\n", 'no-restricted-globals'],
  ])('%s is an error in src/status', async (code, rule) => {
    expect(await ruleIds(code, 'src/status/planted.ts')).toContain(rule)
  })

  it('console in a status test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/status/planted.test.ts')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/status/planted.test.ts')).toContain('no-restricted-syntax')
  })

  it.each(['src/status/status.ts', 'src/status/checkedAt.ts', 'src/shell/RailStatus.tsx'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('k', 'v')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })
})

// S10 (PR 7b): no polling and no relative time, so no timer; it shows the store, never stores or logs it.

describe('no timers, storage or console in S10', () => {
  it.each(['src/pages/Status.tsx', 'src/pages/statusView.ts'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setInterval(() => undefined, 60_000)\nsessionStorage.setItem('status', 'x')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })

  it('console in an S10 test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/pages/Status.test.tsx')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/pages/Status.check.test.tsx')).toContain('no-restricted-syntax')
  })
})

// The shared components (PR 8a) render props only.

describe('no timers, storage or console in the shared components', () => {
  it.each(['src/components/Button.tsx', 'src/components/TextField.tsx'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('x', 'y')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })
})

// The upload store and checks (PR 8b): the File and its name stay in memory, never stored or logged.

describe('no timers, storage or console in src/upload', () => {
  it.each([
    ['setTimeout(() => undefined, 3000)\n', 'no-restricted-syntax'],
    ['globalThis.setInterval(() => undefined, 1)\n', 'no-restricted-syntax'],
    ["localStorage.setItem('file', 'name')\n", 'no-restricted-globals'],
    ["console.log('x')\n", 'no-restricted-globals'],
  ])('%s is an error in src/upload', async (code, rule) => {
    expect(await ruleIds(code, 'src/upload/planted.ts')).toContain(rule)
  })

  it.each(['src/upload/fileDraft.ts', 'src/upload/checks.ts', 'src/upload/analysis.ts'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('k', 'v')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })

  it('a second timer in src/upload/readDuration.ts is still an error: its one exemption is on one line', async () => {
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/upload/readDuration.ts')).toContain('no-restricted-syntax')
  })

  it('console in an upload test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/upload/planted.test.ts')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/upload/planted.test.ts')).toContain('no-restricted-syntax')
  })
})

// S7 (PR C): no polling, so no timer, and the list lives in the page's state only.

describe('no timers, storage or console in S7', () => {
  it.each([
    ['setInterval(() => undefined, 30000)\n', 'no-restricted-syntax'],
    ["localStorage.setItem('history', '[]')\n", 'no-restricted-globals'],
    ["console.log('x')\n", 'no-restricted-globals'],
  ])('%s is an error in src/history', async (code, rule) => {
    expect(await ruleIds(code, 'src/history/planted.ts')).toContain(rule)
  })

  it.each(['src/history/history.ts', 'src/pages/History.tsx'])('%s keeps the timer, storage and console bans', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nlocalStorage.setItem('k', 'v')\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })

  it('console in a History test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/history/planted.test.ts')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/history/planted.test.ts')).toContain('no-restricted-syntax')
  })
})

// The sign-in screen (PR 4): the password and Analyst ID live in its state only, and no lockout countdown, so
// SignIn and signInOutcome get both bans; other pages do not (later pages may need a timer, e.g. the toast).

describe('no timers, storage, cookie or console in the sign-in screen', () => {
  it.each([
    ['setTimeout(() => undefined, 1)\n', 'no-restricted-syntax'],
    ['window.setInterval(() => undefined, 1)\n', 'no-restricted-syntax'],
    ["localStorage.setItem('password', 'x')\n", 'no-restricted-globals'],
    ["window.sessionStorage.setItem('username', 'x')\n", 'no-restricted-properties'],
    ["document.cookie = 'x'\n", 'no-restricted-properties'],
    ["console.log('x')\n", 'no-restricted-globals'],
    ["globalThis.console.error('x')\n", 'no-restricted-properties'],
  ])('%s is an error in src/pages/SignIn.tsx and signInOutcome.ts', async (code, rule) => {
    expect(await ruleIds(code, 'src/pages/SignIn.tsx')).toContain(rule)
    expect(await ruleIds(code, 'src/pages/signInOutcome.ts')).toContain(rule)
  })

  it('console in the S1 test file is allowed, a timer is not', async () => {
    expect(await ruleIds("console.log('x')\n", 'src/pages/SignIn.test.tsx')).toEqual([])
    expect(await ruleIds('setTimeout(() => undefined, 1)\n', 'src/pages/SignIn.test.tsx')).toContain(
      'no-restricted-syntax',
    )
  })

  it('other pages are not banned', async () => {
    expect(await ruleIds("setTimeout(() => undefined, 1)\nconsole.log('x')\n", 'src/pages/Overview.tsx')).toEqual([])
  })

  it.each(['src/pages/Analyze.tsx', 'src/pages/AnalyzeUpload.tsx', 'src/pages/AnalyzeFile.tsx', 'src/pages/AnalyzeResult.tsx', 'src/pages/AnalyzeError.tsx'])('%s has the bans (PR A, B)', async (file) => {
    const ids = await ruleIds("setTimeout(() => undefined, 1)\nconsole.log('x')\n", file)
    expect(ids).toEqual(expect.arrayContaining(['no-restricted-syntax', 'no-restricted-globals']))
  })
})

// End-to-end tests (PR 5a): no fixed waits; secrets never typed with Playwright input APIs nor given to expect;
// Playwright's use() in a fixture is not a React hook. PR 5b: axe runs with WCAG_TAGS and no blanket exclusion; no
// bypassCSP.

describe('e2e bans', () => {
  it.each([
    'await page.waitForTimeout(100)\n',
    'setTimeout(() => undefined, 1)\n',
    'await field.fill(accountPassword())\n',
    'await field.pressSequentially(password)\n',
    'expect(accountPassword()).toBe(x)\n',
    'expect(cookieValue).toBe(x)\n',
    "new AxeBuilder({ page }).disableRules(['color-contrast'])\n",
    "new AxeBuilder({ page }).exclude('#si-user')\n",
    "new AxeBuilder({ page }).withTags(['wcag2a'])\n",
    'browser.newContext({ bypassCSP: true })\n',
  ])('%s is an error in e2e/', async (code) => {
    expect(await ruleIds(code, 'e2e/planted.spec.ts')).toContain('no-restricted-syntax')
  })

  it.each([
    "await field.fill('e2e-main')\n",
    'await field.fill(junk)\n',
    'expect(html.includes(accountPassword())).toBe(false)\n',
    'new AxeBuilder({ page }).withTags([...WCAG_TAGS])\n',
  ])('%s is allowed in e2e/', async (code) => {
    expect(await ruleIds(code, 'e2e/planted.spec.ts')).not.toContain('no-restricted-syntax')
  })

  it('use() in a fixture is not flagged as a React hook', async () => {
    const code = 'export const f = async ({ page }: { page: unknown }, use: (p: unknown) => Promise<void>) => { await use(page) }\n'
    expect(await ruleIds(code, 'e2e/support/planted.ts')).toEqual([])
  })
})
