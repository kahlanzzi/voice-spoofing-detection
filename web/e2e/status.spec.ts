// The rail status line in a real browser (Stage 18 PR 7a; HANDOFF line 41, prototype line 803): one GET /api/status
// when the shell mounts, never on navigation (no polling, team decision 2026-11-01); the line follows the answer,
// with "(mock)" exactly when the API says mock. No sign-in POST: e2e-main's session from the setup project. On the
// real stack the API answers ready (derive_state never says degraded, api/routers/status.py), so Ready with a notice
// and Unavailable are covered by Vitest; the outage case is status-outage.spec.ts (last project).
//
// S10 (PR 7b): entering /status sends nothing; Check now sends one GET, a double click too; busy keeps the label
// (HANDOFF line 137); the time is the browser's zone (Africa/Tunis in playwright.config.ts) with its offset.
import type { Page, Request, Route } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const rail = (page: Page) => page.getByRole('complementary', { name: 'Main menu' })
const isStatus = (request: Request) => request.method() === 'GET' && new URL(request.url()).pathname === '/api/status'
const statusAnswer = (page: Page) => page.waitForResponse((response) => isStatus(response.request()))

function countStatusRequests(page: Page): () => number {
  let count = 0
  page.on('request', (request) => {
    if (isStatus(request)) count += 1
  })
  return () => count
}

test('the shell sends one GET /api/status; the line shows its answer, "(mock)" only when mock', async ({ signedInPage: page }) => {
  const requests = countStatusRequests(page)
  const answer = statusAnswer(page)
  await page.goto('/analyze')
  const response = await answer
  expect(response.status()).toBe(200)
  const body = await response.json()
  expect(body.state).toBe('ready')
  const name = `Service status: Ready for analysis${body.mock ? ' (mock)' : ''}`
  const line = rail(page).getByRole('link', { name, exact: true })
  await expect(line).toBeVisible()
  await expect(line).toHaveAttribute('href', '/status')
  await expect(line).toHaveText(name)
  // Visible text, not screen-reader only (Q2): what the eye sees includes "(mock)" when the API says so.
  expect(await line.locator('.svc-mock').count()).toBe(body.mock ? 1 : 0)
  if (body.mock) await expect(line.locator('.svc-mock')).toBeVisible()
  expect((await line.boundingBox())!.height).toBeGreaterThanOrEqual(44)
  expect(requests()).toBe(1)
})

test('navigation sends no status request; a reload sends one', async ({ signedInPage: page }) => {
  const requests = countStatusRequests(page)
  const answer = statusAnswer(page)
  await page.goto('/analyze')
  await answer
  await expect(rail(page).getByRole('link', { name: /^Service status: Ready for analysis/ })).toBeVisible()
  for (const label of ['History', 'Overview', 'System status', 'Analyze']) {
    await rail(page).getByRole('link', { name: label, exact: true }).click()
    await expect(page.getByRole('heading', { level: 1 })).toBeFocused()
  }
  await page.goBack()
  await page.goForward()
  await rail(page).getByRole('link', { name: /^Service status:/ }).click()
  await expect(page).toHaveURL(/\/status$/)
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('System status')
  await expect(page.getByRole('heading', { level: 1 })).toBeFocused()
  await expect(page.locator('#page-announcer')).toHaveText('System status page')
  expect(requests()).toBe(1)

  const again = statusAnswer(page)
  await page.reload()
  await again
  await expect(rail(page).getByRole('link', { name: /^Service status: Ready for analysis/ })).toBeVisible()
  expect(requests()).toBe(2)
})

test('axe on the rail with the line, wide and in the 390 px drawer, light and dark', async ({ signedInPage: page }) => {
  const answer = statusAnswer(page)
  await page.goto('/history')
  await answer
  await expect(rail(page).getByRole('link', { name: /^Service status: Ready for analysis/ })).toBeVisible()
  await expectNoAxeViolations(page, 'rail status line, wide')

  await page.setViewportSize({ width: 390, height: 800 })
  await page.getByRole('button', { name: 'Menu' }).click()
  const line = rail(page).getByRole('link', { name: /^Service status: Ready for analysis/ })
  await expect(line).toBeVisible()
  expect((await line.boundingBox())!.height).toBeGreaterThanOrEqual(44)
  await expectNoAxeViolations(page, 'rail status line, 390 px drawer open')
})

// S10 (PR 7b) -----------------------------------------------------------------------------------------------------

const TIME = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  timeZoneName: 'shortOffset',
  timeZone: 'Africa/Tunis',
})
const checkNow = (page: Page) => page.getByRole('button', { name: 'Check now' })
const summaryTitle = (page: Page) => page.locator('#sum-h')

// Holds every GET /api/status until release(), so the busy state can be seen; no fixed wait.
async function holdStatus(page: Page): Promise<() => void> {
  let release: () => void = () => undefined
  const gate = new Promise<void>((done) => (release = done))
  await page.route('**/api/status', async (route: Route) => {
    await gate
    await route.continue()
  })
  return release
}

test('S10 Ready: the mock tag, the three checks, "Last checked" in the browser zone, details closed; axe', async ({ signedInPage: page }) => {
  const requests = countStatusRequests(page)
  const answer = statusAnswer(page)
  await page.goto('/status')
  const body = await (await answer).json()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('System status')
  await expect(page.getByText('Whether recordings can be analyzed right now.')).toBeVisible()
  await expect(page.getByText('Mock status values', { exact: true })).toHaveCount(body.mock ? 1 : 0)
  await expect(summaryTitle(page)).toHaveText('Ready for analysis')
  await expect(page.locator('.check-item .state-pill')).toHaveText(['Connected', 'Available', 'Available'])
  await expect(page.locator('.checked-at')).toHaveText(`Last checked ${TIME.format(new Date(body.checked_at))}`)
  await expect(page.getByText(/^Last attempt/)).toHaveCount(0)

  const details = page.locator('details.more')
  await expect(details).not.toHaveAttribute('open')
  await expectNoAxeViolations(page, 'S10 Ready, details closed')
  await details.getByText('Technical details').click()
  await expect(details).toHaveAttribute('open')
  await expect(details.locator('dd')).toHaveText([
    body.versions.model_version,
    body.versions.profile_version,
    body.versions.policy_version,
    `deferral_on_fit = ${body.deferral_on_fit.toFixed(4)}`,
    body.bundle_manifest_sha256 ?? 'Not available until the bundle is frozen.',
  ])
  await expectNoAxeViolations(page, 'S10 Ready, details open')
  expect(requests()).toBe(1)
})

test('Check now: busy keeps the label, one GET, focus back, "Status checked" announced', async ({ signedInPage: page }) => {
  const requests = countStatusRequests(page)
  const first = statusAnswer(page)
  await page.goto('/status')
  await first
  await expect(checkNow(page)).toBeEnabled()
  await expect(page.locator('#status-announcer')).toHaveText('')

  const release = await holdStatus(page)
  await checkNow(page).click()
  await expect(checkNow(page)).toBeDisabled()
  await expect(checkNow(page)).toHaveAttribute('aria-busy', 'true')
  await expect(checkNow(page)).toHaveText('Check now')
  await expect(checkNow(page).locator('.spin')).toBeVisible()
  await expect(summaryTitle(page)).toHaveText('Checking…')
  const second = statusAnswer(page)
  release()
  expect((await second).status()).toBe(200)
  await expect(summaryTitle(page)).toHaveText('Ready for analysis')
  await expect(checkNow(page)).toBeEnabled()
  await expect(checkNow(page)).toBeFocused()
  await expect(page.locator('#status-announcer')).toHaveText('Status checked: Ready for analysis.')
  expect(requests()).toBe(2)
})

test('a double click on Check now sends one GET; ≥ 44 px at 390 px', async ({ signedInPage: page }) => {
  const requests = countStatusRequests(page)
  const first = statusAnswer(page)
  await page.goto('/status')
  await first
  const release = await holdStatus(page)
  await checkNow(page).dblclick()
  await expect(checkNow(page)).toBeDisabled()
  const second = statusAnswer(page)
  release()
  await second
  await expect(checkNow(page)).toBeEnabled()
  await expect(page.locator('#status-announcer')).toHaveText('Status checked: Ready for analysis.')
  expect(requests()).toBe(2)

  await page.setViewportSize({ width: 390, height: 800 })
  expect((await checkNow(page).boundingBox())!.height).toBeGreaterThanOrEqual(44)
  await expect(page.locator('details.more > summary')).toHaveCSS('min-height', '44px')
})
