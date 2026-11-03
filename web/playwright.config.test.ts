// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { playwrightArgs } from './e2e/support/args.ts'
import { budgetProblem, countLoggedSignIns } from './e2e/support/budget.ts'
import config from './playwright.config.ts'

// The e2e settings that keep secrets out of artifacts and the sign-in POSTs inside nginx's burst (Stage 18 PR 5a).

describe('playwright.config.ts', () => {
  it('reports with list and json only (plain text for the secret scan), never html', () => {
    expect(config.reporter).toEqual([['list'], ['json', { outputFile: 'playwright-report/results.json' }]])
  })

  it('records no trace, video or screenshot', () => {
    expect(config.use).toMatchObject({ trace: 'off', video: 'off', screenshot: 'off' })
  })

  it('never retries and runs one worker, so the POST count is fixed', () => {
    expect(config).toMatchObject({ retries: 0, workers: 1, fullyParallel: false })
  })

  it('targets nginx on 8001, never the Vite dev server', () => {
    expect(config.use?.baseURL).toBe('http://localhost:8001')
  })

  it('has a setup project per browser that the browser project depends on', () => {
    expect(config.projects?.map(({ name, dependencies }) => [name, dependencies ?? []])).toEqual([
      ['chromium-setup', []],
      ['chromium', ['chromium-setup']],
      ['firefox-setup', []],
      ['firefox', ['firefox-setup']],
    ])
  })
})

// run.ts spawns Playwright without a shell, so the filter carries no quotes (a quote would match no project).

describe('playwrightArgs', () => {
  it.each([
    ['chromium', ['test', '--project=chromium*']],
    ['firefox', ['test', '--project=firefox*']],
  ])('%s → exactly %j', (browser, args) => {
    expect(playwrightArgs(browser)).toEqual(args)
    expect(playwrightArgs(browser).some((arg) => /["']/.test(arg))).toBe(false)
  })

  it('the filter matches only that browser’s projects', () => {
    const names = config.projects?.map((project) => project.name ?? '') ?? []
    const matching = (arg: string) => names.filter((name) => name.startsWith(arg.replace('--project=', '').replace('*', '')))
    expect(matching(playwrightArgs('chromium')[1]!)).toEqual(['chromium-setup', 'chromium'])
    expect(matching(playwrightArgs('firefox')[1]!)).toEqual(['firefox-setup', 'firefox'])
  })

  it('refuses any other browser', () => {
    expect(() => playwrightArgs('webkit')).toThrow('Unknown browser webkit')
  })
})

// The sign-in budget run.ts checks in nginx's log (e2e/support/budget.ts): lines in nginx's `vsd` log format.

describe('the sign-in POST budget', () => {
  const line = (method: string, path: string, status: number) =>
    `2026-11-03T10:00:00+00:00 "${method} ${path}" ${status} 98 0.012`
  const three = [line('POST', '/api/session', 200), line('POST', '/api/session', 401), line('POST', '/api/session', 403)]
  const noise = [line('GET', '/api/session', 401), line('GET', '/signin', 200), line('DELETE', '/api/session', 204), '']

  it('counts only POST /api/session lines and their 429s', () => {
    expect(countLoggedSignIns([...noise, ...three])).toEqual({ posts: 3, limited: 0 })
    expect(countLoggedSignIns([line('POST', '/api/session', 429), line('POST', '/api/analyses', 429)])).toEqual({ posts: 1, limited: 1 })
    expect(countLoggedSignIns([])).toEqual({ posts: 0, limited: 0 })
  })

  it('a green run needs exactly the planned POSTs and no 429', () => {
    expect(budgetProblem(countLoggedSignIns([...noise, ...three]), true)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(three.slice(0, 2)), true)).toMatch('must show 3')
    expect(budgetProblem(countLoggedSignIns([...three, line('POST', '/api/session', 200)]), true)).toMatch('must show 3')
    expect(budgetProblem(countLoggedSignIns([...three.slice(0, 2), line('POST', '/api/session', 429)]), true)).toMatch('429')
  })

  it('a green run with no log line fails: the check would be reading nothing', () => {
    expect(budgetProblem(countLoggedSignIns([]), true)).toMatch('must show 3 sign-in POSTs, nginx\'s log shows 0')
    expect(budgetProblem(countLoggedSignIns(noise), true)).toMatch('shows 0')
  })

  it('a failed run may show fewer, never more, and no 429', () => {
    expect(budgetProblem(countLoggedSignIns([]), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(three.slice(0, 1)), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(three), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns([...three, line('POST', '/api/session', 401)]), false)).toMatch('more than')
    expect(budgetProblem(countLoggedSignIns([line('POST', '/api/session', 429)]), false)).toMatch('429')
  })
})
