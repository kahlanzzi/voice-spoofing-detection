// The API down behind nginx (Entry 134 live check): S1 shows the unavailable banner with Retry; once the API is back,
// Retry brings the default form. In the last project: nothing runs after the stop. No sign-in POST (GET
// /api/session is not rate-limited). The API is started again in `finally`, and by the global teardown. nginx answers
// 503 only after proxy_connect_timeout (5 s, deploy/nginx.conf): the stopped container's address does not refuse, it
// is silent. So the test waits for that answer, not for a guessed time.
import type { Page } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'
import { COPY, signInForm } from './support/signin.ts'
import { ensureApiUp, stopApi } from './support/stack.ts'

const sessionCheck = (page: Page) =>
  page.waitForResponse((r) => r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/session')

test('api stopped → unavailable + Retry; api back → Retry → the default form', async ({ page }) => {
  await stopApi()
  try {
    const down = sessionCheck(page)
    await page.goto('/signin')
    expect((await down).status()).toBe(503)
    expect((await (await down).json()).error.code).toBe('analysis_unavailable')
    const form = signInForm(page)
    await expect(form.alert).toHaveText(COPY.unavailable)
    await expect(form.retry).toBeVisible()
    await expectNoAxeViolations(page, 'S1 unavailable')

    await ensureApiUp()
    const check = sessionCheck(page)
    await form.retry.click()
    expect((await check).status()).toBe(401)
    await expect(form.alert).toHaveCount(0)
    await expect(form.retry).toHaveCount(0)
    await expect(form.username).toBeEnabled()
    await expect(form.submit).toBeEnabled()
  } finally {
    await ensureApiUp()
  }
})
