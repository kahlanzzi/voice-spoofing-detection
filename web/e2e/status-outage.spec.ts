// S10 with the API down behind nginx (Stage 18 PR 7b): Check now gets nginx's 503 analysis_unavailable, so S10 shows
// Unavailable with "Last attempt" (the browser's time; "Last checked" only after a valid 200) and the rail line says
// Unavailable too, from the same store; the analyst stays in the shell. Once the API is back, Check now shows Ready
// and "Last checked". In the last project, after outage.spec and signout-failed.spec, on e2e-main's saved session; no
// sign-in POST. Check now sends only GET /api/status, so nothing reaches the session while the API is down. The API
// is started again in `finally`, and by the global teardown. nginx answers only after proxy_connect_timeout (5 s,
// deploy/nginx.conf), so the test waits for that answer, not for a guessed time.
import type { Page, Response } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'
import { countSignInPosts } from './support/signin.ts'
import { BASE_URL, ensureApiUp, stopApi } from './support/stack.ts'

const isStatus = (response: Response) =>
  response.request().method() === 'GET' && new URL(response.url()).pathname === '/api/status'
const statusAnswer = (page: Page) => page.waitForResponse(isStatus)

test('api stopped → Check now → Unavailable, "Last attempt", the rail line too; api back → Ready, "Last checked"', async ({ signedInPage: page }) => {
  const posts = countSignInPosts(page)
  const checkNow = page.getByRole('button', { name: 'Check now' })
  const rail = page.getByRole('complementary', { name: 'Main menu' })
  const summaryTitle = page.locator('#sum-h')
  const time = page.locator('.checked-at')

  await ensureApiUp()
  const loaded = statusAnswer(page)
  await page.goto('/status')
  expect((await loaded).status()).toBe(200)
  await expect(summaryTitle).toHaveText('Ready for analysis')
  await expect(time).toHaveText(/^Last checked /)

  await stopApi()
  try {
    const down = statusAnswer(page)
    await checkNow.click()
    expect((await down).status()).toBe(503)
    expect((await (await down).json()).error.code).toBe('analysis_unavailable')
    await expect(summaryTitle).toHaveText('Unavailable')
    await expect(page.locator('.check-item .state-pill')).toHaveText(['Unreachable', 'Unavailable', 'Unavailable'])
    await expect(time).toHaveText(/^Last attempt \d{1,2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2} GMT\+1$/)
    await expect(page.getByText(/^Last checked/)).toHaveCount(0)
    await expect(page.getByText('Technical details')).toHaveCount(0)
    await expect(page.getByText('Mock status values', { exact: true })).toHaveCount(0)
    await expect(page.locator('#status-announcer')).toHaveText('Status checked: Unavailable.')
    await expect(checkNow).toBeFocused()
    await expect(rail.getByRole('link', { name: 'Service status: Unavailable', exact: true })).toBeVisible()
    await expect(page).toHaveURL(`${BASE_URL}/status`)
    await expect(rail).toBeVisible()
    await expect(page.getByRole('main')).toBeVisible()
    await expectNoAxeViolations(page, 'S10 Unavailable, no answer')
  } finally {
    await ensureApiUp()
  }

  const back = statusAnswer(page)
  await checkNow.click()
  expect((await back).status()).toBe(200)
  await expect(summaryTitle).toHaveText('Ready for analysis')
  await expect(time).toHaveText(/^Last checked /)
  await expect(rail.getByRole('link', { name: /^Service status: Ready for analysis/ })).toBeVisible()
  expect(posts()).toBe(0)
})
