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
