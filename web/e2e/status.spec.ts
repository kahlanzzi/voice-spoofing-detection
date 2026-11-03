// The rail status line in a real browser (Stage 18 PR 7a; HANDOFF line 41, prototype line 803): one GET /api/status
// when the shell mounts, never on navigation (no polling, team decision 2026-11-01); the line follows the answer,
// with "(mock)" exactly when the API says mock. No sign-in POST: e2e-main's session from the setup project. On the
// real stack the API answers ready (derive_state never says degraded, api/routers/status.py), so Ready with a notice
// and Unavailable are covered by Vitest; the outage case comes with S10's Check now (PR 7b).
import type { Page, Request } from '@playwright/test'
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
