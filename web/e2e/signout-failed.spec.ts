// A sign-out that does not complete (Stage 18 PR 6b): the api is stopped behind nginx, so the DELETE gets nginx's
// 503 analysis_unavailable and the analyst stays signed in, with the alert and focus on Sign out. In the last project
// (it stops the api, like outage.spec), on e2e-main's saved session; no sign-in POST. The api is started again in
// `finally`, and by the global teardown. nginx answers only after proxy_connect_timeout (5 s, deploy/nginx.conf), so
// the test waits for that answer, not for a guessed time.
//
// Why e2e-main's session survives: nginx answered the DELETE itself, with no Set-Cookie, so the browser keeps the
// cookie; once the api is back, GET /api/session with that cookie answers 200, which it cannot do if the API had
// deleted the session row. That 200 is the proof the DELETE never reached the API.
import type { Page, Request } from '@playwright/test'
import { expect, test } from './support/fixtures.ts'
import { countSignInPosts } from './support/signin.ts'
import { BASE_URL, ensureApiUp, stopApi } from './support/stack.ts'

const SIGN_OUT_FAILED = 'Sign-out did not complete. You are still signed in. Try again.'

const isSignOut = (request: Request) => request.method() === 'DELETE' && new URL(request.url()).pathname === '/api/session'
const sessionCheck = (page: Page) =>
  page.waitForResponse((r) => r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/session')

test('api stopped → Sign out answers 503: still signed in, the alert, focus on Sign out; the session survives', async ({ signedInPage: page }) => {
  const posts = countSignInPosts(page)
  const signOut = page.getByRole('button', { name: 'Sign out' })
  const rail = page.getByRole('complementary', { name: 'Main menu' })

  await ensureApiUp()
  const loaded = sessionCheck(page)
  await page.goto('/analyze')
  expect((await loaded).status()).toBe(200)
  await expect(rail).toBeVisible()

  await stopApi()
  try {
    const response = page.waitForResponse((r) => isSignOut(r.request()))
    await signOut.click()
    expect((await response).status()).toBe(503)
    expect((await (await response).json()).error.code).toBe('analysis_unavailable')
    expect(await (await response).headerValue('set-cookie')).toBeNull()
    // In the rail: a limits GET still in flight when the api stopped would add S2's own alert (PR A).
    await expect(rail.getByRole('alert')).toHaveText(SIGN_OUT_FAILED)
    await expect(signOut).toBeEnabled()
    await expect(signOut).toBeFocused()
    await expect(page).toHaveURL(`${BASE_URL}/analyze`)
    await expect(rail).toBeVisible()
  } finally {
    await ensureApiUp()
  }

  const check = sessionCheck(page)
  await page.reload()
  expect((await check).status()).toBe(200)
  await expect(page).toHaveURL(`${BASE_URL}/analyze`)
  await expect(page.locator('.rail-foot .who-name')).toHaveText('e2e-main')
  expect(posts()).toBe(0)
})
