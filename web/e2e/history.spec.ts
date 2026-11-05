// S7 in a real browser (Stage 18 PR C): the real GET through nginx against records this spec creates, and the states the
// real stack cannot give on demand through page.route. No sign-in POST: e2e-main's session from the setup project.
//
// The list is shared and grows with each green run (9 rows now, 6 from analyze-result.spec.ts), so nothing here assumes
// an empty or fixed list: records are found by the request ID of their own POST, and order is checked only among them.
// workers: 1 keeps them on the first page.
//
// The zone: signedInPage takes timezoneId from the project (fixtures.ts), so test.use({ timezoneId }) would not reach it.
// The config's Africa/Tunis applies on UTC runners too, and the first test proves it in the page.
import type { Page, Request, Response, Route } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'
import { wav } from './support/wav.ts'

const SHORT: Record<string, string> = {
  synthetic: 'Likely synthetic',
  uncertain: 'Cannot decide',
  authentic: 'Likely authentic',
}

const ROW_TIME = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  timeZone: 'Africa/Tunis',
})

const FAILURE = 'History could not be loaded'

const h1 = (page: Page) => page.getByRole('heading', { level: 1 })
const rail = (page: Page) => page.getByRole('complementary', { name: 'Main menu' })
const announcer = (page: Page) => page.locator('#history-announcer')

function isListGet(request: Request): boolean {
  return request.method() === 'GET' && new URL(request.url()).pathname === '/api/analyses'
}

function countGets(page: Page): Request[] {
  const gets: Request[] = []
  page.on('request', (request) => {
    if (isListGet(request)) gets.push(request)
  })
  return gets
}

// Only the GET: a POST goes on to the real API.
function answerGet(page: Page, status: number, body: unknown) {
  return page.route('**/api/analyses', (route: Route) =>
    route.request().method() === 'GET'
      ? route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
      : route.fallback(),
  )
}

async function analyze(page: Page, seconds: number) {
  await page.goto('/analyze')
  await expect(page.getByRole('button', { name: 'Choose recording' })).toBeEnabled()
  await page.locator('#file-input').setInputFiles({ name: 'call.wav', mimeType: 'application/octet-stream', buffer: wav(seconds) })
  await expect(page.getByRole('button', { name: 'Analyze recording' })).toBeEnabled()
  const answer = page.waitForResponse((response: Response) => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/analyses')
  await page.getByRole('button', { name: 'Analyze recording' }).click()
  const response = await answer
  expect(response.status()).toBe(201)
  await expect(page.locator('#res-h')).toBeFocused()
  return response.json()
}

function stored(id: string, prediction: string, risk: string) {
  return {
    analysis: {
      prediction,
      confidence: 0.97,
      risk_level: risk,
      model_version: 'mock',
      profile_version: 'preprocessing-v1',
      policy_version: 'risk-v1',
      request_id: `req_${id.repeat(32)}`,
      notice: 'Decision support only.',
    },
    created_at: '2026-11-03T09:14:05Z',
    note: null,
  }
}

test('the real stack: three analyses from this spec, each row with its own values, newest first; no axe violation', async ({ signedInPage: page }) => {
  const created = [await analyze(page, 1), await analyze(page, 2), await analyze(page, 7)]
  const answer = page.waitForResponse((response: Response) => isListGet(response.request()))
  await rail(page).getByRole('link', { name: 'History' }).click()
  await expect(h1(page)).toHaveText('History')
  await expect(h1(page)).toBeFocused()
  const { items, total } = await (await answer).json()
  await expect(page.locator('tbody tr').first()).toBeVisible()

  expect(await page.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone)).toBe('Africa/Tunis')
  await expect(page.getByRole('columnheader', { name: 'Analyzed (GMT+1)' })).toBeVisible()
  await expect(announcer(page)).toHaveText(`History loaded. ${total} results.`)
  await expect(page.locator('.count-line strong')).toHaveText(`${total} results`)
  await expect(page.locator('.history-showing')).toHaveText(`Showing 1–${items.length} of ${total}`)

  for (const body of created) {
    const row = page.locator('tbody tr').filter({ hasText: body.request_id })
    const record = items.find((item: { analysis: { request_id: string } }) => item.analysis.request_id === body.request_id)
    await expect(row).toHaveCount(1)
    await expect(row.locator('.pill')).toHaveText(`${SHORT[body.prediction]} · ${body.risk_level}`)
    await expect(row.getByText('Mock result')).toBeVisible()
    await expect(row.locator('td').first()).toHaveText(ROW_TIME.format(new Date(record.created_at)))
    await expect(row.getByRole('link')).toHaveCount(0)
  }
  const shown = await page.locator('tbody code').allTextContents()
  const places = created.map((body) => shown.indexOf(body.request_id))
  expect(places[2]!).toBeLessThan(places[1]!)
  expect(places[1]!).toBeLessThan(places[0]!)
  await expectNoAxeViolations(page, 'S7 list')
})

test('one GET per entry; an hour on the page sends no other', async ({ signedInPage: page }) => {
  await page.clock.install()
  const gets = countGets(page)
  await page.goto('/history')
  await expect(announcer(page)).toContainText('History loaded.')
  expect(gets).toHaveLength(1)
  await rail(page).getByRole('link', { name: 'System status' }).click()
  await expect(h1(page)).toHaveText('System status')
  expect(gets).toHaveLength(1)
  await rail(page).getByRole('link', { name: 'History' }).click()
  await expect(announcer(page)).toContainText('History loaded.')
  expect(gets).toHaveLength(2)
  await page.clock.fastForward('01:00:00')
  await expect(h1(page)).toHaveText('History')
  expect(gets).toHaveLength(2)
})

test('/history?verdict=synthetic: the GET has no query and the URL keeps it', async ({ signedInPage: page }) => {
  const gets = countGets(page)
  await page.goto('/history?verdict=synthetic')
  await expect(announcer(page)).toContainText('History loaded.')
  expect(gets.map((request) => new URL(request.url()).search)).toEqual([''])
  await expect(page).toHaveURL(/\/history\?verdict=synthetic$/)
})

test('no records: the empty state, New analysis goes to /analyze; no axe violation', async ({ signedInPage: page }) => {
  await answerGet(page, 200, { items: [], total: 0, page: 1, page_size: 10, counts: { synthetic: 0, uncertain: 0, authentic: 0 } })
  await page.goto('/history')
  await expect(page.getByRole('heading', { level: 2, name: 'No analyses yet' })).toBeVisible()
  await expect(announcer(page)).toHaveText('History loaded. 0 results.')
  await expect(page.getByRole('table')).toHaveCount(0)
  await expectNoAxeViolations(page, 'S7 empty')
  await page.getByRole('main').getByRole('button', { name: 'New analysis' }).click()
  await expect(page).toHaveURL(/\/analyze$/)
})

test('a 500: the failure, no axe violation; Retry gets the real list with focus on the h1', async ({ signedInPage: page }) => {
  await answerGet(page, 500, { error: { code: 'internal_error', message: 'x', request_id: null } })
  await page.goto('/history')
  await expect(page.getByRole('alert')).toContainText(FAILURE)
  await expect(announcer(page)).toHaveText('')
  await expectNoAxeViolations(page, 'S7 failed')
  await page.unroute('**/api/analyses')
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(announcer(page)).toContainText('History loaded.')
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(h1(page)).toBeFocused()
})

test('a malformed 200: the failure, and not one row', async ({ signedInPage: page }) => {
  await answerGet(page, 200, { items: [stored('a', 'synthetic', 'HIGH'), {}], total: 2 })
  await page.goto('/history')
  await expect(page.getByRole('alert')).toContainText(FAILURE)
  await expect(page.locator('tbody tr')).toHaveCount(0)
})

test('401 on the GET: S1 with next=/history, no failure', async ({ signedInPage: page }) => {
  await answerGet(page, 401, { error: { code: 'session_expired', message: 'x', request_id: null } })
  await page.goto('/history')
  await expect(page).toHaveURL(/\/signin\?next=%2Fhistory$/)
  await expect(page.getByText(FAILURE)).toHaveCount(0)
})

test('390 px: cards with their labels, the full ID, no sideways scroll; no axe violation', async ({ signedInPage: page }) => {
  const items = [stored('a', 'synthetic', 'HIGH'), stored('b', 'uncertain', 'MEDIUM'), stored('c', 'authentic', 'LOW')]
  await answerGet(page, 200, { items, total: 37, page: 1, page_size: 10, counts: { synthetic: 1, uncertain: 1, authentic: 1 } })
  await page.setViewportSize({ width: 390, height: 800 })
  await page.goto('/history')
  await expect(page.locator('tbody tr')).toHaveCount(3)
  await expect(page.locator('.history-showing')).toHaveText('Showing 1–3 of 37')
  await expect(announcer(page)).toHaveText('History loaded. 37 results.')
  await expect(page.locator('tbody tr').first().locator('td').first()).toHaveText('3 Nov 2026, 10:14')
  await expect(page.getByText(`req_${'a'.repeat(32)}`)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true)
  expect(await page.locator('.table-wrap').evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
  expect(await page.locator('tbody td').first().evaluate((node) => getComputedStyle(node, '::before').content)).toBe('"Analyzed"')
  await expectNoAxeViolations(page, 'S7 at 390 px')
})
