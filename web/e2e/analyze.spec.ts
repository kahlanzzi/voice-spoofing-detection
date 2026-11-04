// S2 and S3 in a real browser, on the real stack (Stage 18 PR A; HANDOFF §1): the limits from GET /api/limits, a held
// and a failed answer through page.route. No sign-in POST: e2e-main's session from the setup project.
import type { Page, Route } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const limitsAnswer = (page: Page) =>
  page.waitForResponse((r) => r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/limits')
const choose = (page: Page) => page.getByRole('button', { name: 'Choose recording' })
const requirements = (page: Page) => page.getByLabel('Upload requirements')

test('S2 loads the limits from the service; no axe violation', async ({ signedInPage: page }) => {
  const answer = limitsAnswer(page)
  await page.goto('/analyze')
  const response = await answer
  expect(response.status()).toBe(200)
  const body = await response.json()
  await expect(requirements(page).locator('dd')).toHaveText([
    body.formats.map((format: string) => format.toUpperCase()).join(', '),
    `${Number((body.max_size_bytes / 1048576).toFixed(1))} MB`,
    `${body.max_duration_s} s`,
    body.audio_deletion,
  ])
  await expect(page.getByText('Mock values · real limits not decided')).toBeVisible()
  await expect(choose(page)).toBeEnabled()
  await expectNoAxeViolations(page, 'S2 limits ready')
})

test('limits held: the loading line, Choose recording disabled; no axe violation', async ({ signedInPage: page }) => {
  let release: () => void = () => undefined
  const held = new Promise<void>((done) => {
    release = done
  })
  await page.route('**/api/limits', async (route: Route) => {
    await held
    await route.continue()
  })
  await page.goto('/analyze')
  await expect(page.getByText('Loading upload limits from the service…')).toBeVisible()
  await expect(choose(page)).toBeDisabled()
  await expectNoAxeViolations(page, 'S2 limits loading')
  release()
  await expect(requirements(page)).toBeVisible()
  await expect(choose(page)).toBeEnabled()
})

test('limits 500: the alert, analysis off, no axe violation; Retry loads them', async ({ signedInPage: page }) => {
  const body = JSON.stringify({ error: { code: 'internal_error', message: 'x', request_id: null } })
  await page.route('**/api/limits', (route: Route) => route.fulfill({ status: 500, contentType: 'application/json', body }))
  await page.goto('/analyze')
  await expect(page.getByRole('alert')).toContainText('Upload limits could not be loaded.')
  await expect(requirements(page)).toHaveCount(0)
  await expect(choose(page)).toBeDisabled()
  await expectNoAxeViolations(page, 'S2 limits failed')
  await page.unroute('**/api/limits')
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(requirements(page)).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('heading', { level: 2, name: 'Choose the recorded segment' })).toBeFocused()
})
