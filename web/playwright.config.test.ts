// @vitest-environment node
import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { playwrightArgs } from './e2e/support/args.ts'
import { applyExclusions, axeFindings, WCAG_TAGS } from './e2e/support/axe.ts'
import { budgetProblem, countLoggedSignIns, PLAN, SEGMENTS, ZONE_CAPACITY } from './e2e/support/budget.ts'
import { SECURITY_HEADERS_CONF, securityHeaders } from './e2e/support/nginx-headers.ts'
import config, { LAST } from './playwright.config.ts'

// The e2e settings that keep secrets out of artifacts and the sign-in POSTs inside nginx's burst (Stage 18 PR 5a, 5b).

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

  it('orders setup → main → last per browser through dependencies', () => {
    expect(config.projects?.map(({ name, dependencies }) => [name, dependencies ?? []])).toEqual([
      ['chromium-setup', []],
      ['chromium', ['chromium-setup']],
      ['chromium-last', ['chromium']],
      ['firefox-setup', []],
      ['firefox', ['firefox-setup']],
      ['firefox-last', ['firefox']],
    ])
  })

  it('runs every spec file in exactly one project per browser; lockout, outage and the failed sign-out last', () => {
    const specs = readdirSync(new URL('./e2e', import.meta.url)).filter((name) => name.endsWith('.spec.ts'))
    const list = (value: unknown) => (value === undefined ? undefined : [value].flat())
    const runs = (project: NonNullable<typeof config.projects>[number], file: string) =>
      (list(project.testMatch)?.includes(file) ?? true) && !(list(project.testIgnore)?.includes(file) ?? false)
    for (const browser of ['chromium', 'firefox']) {
      const projects = config.projects?.filter((project) => project.name?.startsWith(browser)) ?? []
      for (const file of specs) expect(projects.filter((project) => runs(project, file)).length, file).toBe(1)
      expect(specs.filter((file) => runs(projects.find((p) => p.name === `${browser}-last`)!, file))).toEqual(LAST)
    }
    expect(LAST).toEqual(['locked.spec.ts', 'outage.spec.ts', 'signout-failed.spec.ts'])
  })

  it('never bypasses the CSP (a CSP block of axe is reported)', () => {
    expect(config.use).not.toHaveProperty('bypassCSP')
    for (const project of config.projects ?? []) expect(project.use).not.toHaveProperty('bypassCSP')
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
    expect(matching(playwrightArgs('chromium')[1]!)).toEqual(['chromium-setup', 'chromium', 'chromium-last'])
    expect(matching(playwrightArgs('firefox')[1]!)).toEqual(['firefox-setup', 'firefox', 'firefox-last'])
  })

  it('refuses any other browser', () => {
    expect(() => playwrightArgs('webkit')).toThrow('Unknown browser webkit')
  })
})

// The sign-in budget run.ts checks in nginx's log (e2e/support/budget.ts): lines in nginx's `vsd` log format.

describe('the sign-in POST budget', () => {
  const line = (method: string, path: string, status: number) =>
    `2026-11-03T10:00:00+00:00 "${method} ${path}" ${status} 98 0.012`
  const post = (status: number) => line('POST', '/api/session', status)
  // A: setup 200, invalid 401, denied 403, http.spec 403 ×2, signout.spec 200. B: 401 ×5, then 429 account_locked.
  const green = [post(200), post(401), post(403), post(403), post(403), post(200), ...Array(5).fill(post(401)), post(429)]
  const noise = [line('GET', '/api/session', 401), line('GET', '/signin', 200), line('DELETE', '/api/session', 403), line('POST', '/api/analyses', 403), '']

  it('plans two segments, each within a fresh zone (both full), 12 POSTs and one 429 in all', () => {
    expect(SEGMENTS).toEqual({ A: 6, B: 6 })
    for (const posts of Object.values(SEGMENTS)) expect(posts).toBeLessThanOrEqual(ZONE_CAPACITY)
    expect(PLAN).toEqual({ posts: 12, limited: 1 })
  })

  it('counts only POST /api/session lines and their 429s', () => {
    expect(countLoggedSignIns([...noise, ...green])).toEqual({ posts: 12, limited: 1 })
    expect(countLoggedSignIns([post(429), line('POST', '/api/analyses', 429)])).toEqual({ posts: 1, limited: 1 })
    expect(countLoggedSignIns([])).toEqual({ posts: 0, limited: 0 })
  })

  it('a green run needs exactly 12 POSTs and exactly one 429', () => {
    expect(budgetProblem(countLoggedSignIns([...noise, ...green]), true)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(green.slice(1)), true)).toMatch('must show 12')
    expect(budgetProblem(countLoggedSignIns([...green, post(200)]), true)).toMatch('more than the planned 12')
    expect(budgetProblem(countLoggedSignIns([...green.slice(0, 11), post(401)]), true)).toMatch('must show 1 × 429')
    expect(budgetProblem(countLoggedSignIns([...green.slice(0, 10), post(429), post(429)]), true)).toMatch('2 sign-in POST(s) answered 429')
  })

  it('a green run with no log line fails: the check would be reading nothing', () => {
    expect(budgetProblem(countLoggedSignIns([]), true)).toMatch("must show 12 sign-in POSTs, nginx's log shows 0")
    expect(budgetProblem(countLoggedSignIns(noise), true)).toMatch('shows 0')
  })

  it('a failed run may show fewer, never more, and at most one 429', () => {
    expect(budgetProblem(countLoggedSignIns([]), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(green.slice(0, 3)), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns(green), false)).toBeNull()
    expect(budgetProblem(countLoggedSignIns([...green, post(401)]), false)).toMatch('more than')
    expect(budgetProblem(countLoggedSignIns([post(429), post(429)]), false)).toMatch('429')
  })
})

// The axe report keeps { id, impact, target } only: a node's HTML could hold typed text (e2e/support/axe.ts).

describe('axe findings', () => {
  const node = { target: ['#si-user'], html: '<input value="typed-secret">', failureSummary: 'Fix: typed-secret', impact: 'serious' }
  const results = { violations: [{ id: 'color-contrast', impact: 'serious', nodes: [node, { ...node, target: [['iframe', '#x']] }] }] }

  it('reports only the rule, the impact and the target, never the node HTML', () => {
    const findings = axeFindings(results as never)
    expect(findings).toEqual([
      { id: 'color-contrast', impact: 'serious', target: '#si-user' },
      { id: 'color-contrast', impact: 'serious', target: 'iframe >>> #x' },
    ])
    expect(JSON.stringify(findings)).not.toContain('typed-secret')
  })

  it('an exclusion removes only its exact rule and target; one that matches nothing is reported', () => {
    const findings = axeFindings(results as never)
    const exact = { rule: 'color-contrast', target: '#si-user', reason: 'r' }
    expect(applyExclusions(findings, [exact])).toEqual({ left: [findings[1]], unused: [] })
    expect(applyExclusions(findings, [{ ...exact, rule: 'label' }])).toEqual({ left: findings, unused: [{ rule: 'label', target: '#si-user' }] })
  })

  it('scans with the WCAG 2.0/2.1/2.2 A and AA tags (HANDOFF l.297)', () => {
    expect(WCAG_TAGS).toEqual(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'])
  })
})

// http.spec reads the expected security headers from the file nginx includes; the parser must see every line.

describe('securityHeaders', () => {
  it('reads every add_header line of deploy/nginx_security_headers.conf', () => {
    const lines = readFileSync(SECURITY_HEADERS_CONF, 'utf8').split('\n').filter((l) => l.trim().startsWith('add_header'))
    const headers = securityHeaders()
    expect(Object.keys(headers)).toHaveLength(lines.length)
    expect(Object.keys(headers)).toEqual(['referrer-policy', 'x-content-type-options', 'content-security-policy'])
  })

  it('refuses an add_header line it cannot read, rather than skipping it', () => {
    expect(() => securityHeaders('add_header X-Frame-Options DENY;\n')).toThrow('Unexpected add_header line')
  })
})
