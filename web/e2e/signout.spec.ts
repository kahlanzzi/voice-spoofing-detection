// Sign-out from the rail footer on the real stack (Stage 18 PR 6b; docs/api_design.md §4.8 DELETE /session). Its own
// account, e2e-signout, in the test's own context, so e2e-main's saved session is never ended. One sign-in POST.
// The session cookie is kept in memory only, for the replay below, and is asserted only as booleans.
import { expect, test, type Request } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { ACCOUNTS, typeSecret } from './support/secret-guard.ts'
import { countSignInPosts, signInForm, signInResponse } from './support/signin.ts'
import { BASE_URL } from './support/stack.ts'

const isSignOut = (request: Request) => request.method() === 'DELETE' && new URL(request.url()).pathname === '/api/session'
const sessionCheck = (r: { request(): Request; url(): string }) =>
  r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/session'

test('Sign out: one DELETE with Origin, 204, S1 without next, the server session gone', async ({ page, context }) => {
  const posts = countSignInPosts(page)
  const form = signInForm(page)

  await test.step('sign in through S1', async () => {
    await page.goto('/analyze')
    const response = signInResponse(page)
    await form.username.fill(ACCOUNTS.signout)
    await typeSecret(form.password)
    await form.submit.click()
    expect((await response).status()).toBe(200)
    await expect(page).toHaveURL(`${BASE_URL}/analyze`)
    await expect(page.locator('.rail-foot .who-name')).toHaveText(ACCOUNTS.signout)
  })

  const saved = (await context.cookies()).find((cookie) => cookie.name === 'vsd_session')
  expect(saved !== undefined).toBe(true)

  await test.step('a double click sends one DELETE, with the page Origin, answered 204', async () => {
    let deletes = 0
    page.on('request', (request) => {
      if (isSignOut(request)) deletes += 1
    })
    const request = page.waitForRequest(isSignOut)
    const response = page.waitForResponse((r) => isSignOut(r.request()))
    await page.getByRole('button', { name: 'Sign out' }).dblclick()
    expect(await (await request).headerValue('origin')).toBe(BASE_URL)
    expect((await response).status()).toBe(204)
    await expect(form.heading).toBeVisible()
    expect(deletes).toBe(1)
  })

  await test.step('S1 at exactly /signin, h1 focused, "Signed out." announced, the cookie cleared', async () => {
    await expect(page).toHaveURL(`${BASE_URL}/signin`)
    await expect(form.heading).toBeFocused()
    await expect(page.locator('#page-announcer')).toHaveText('Signed out.')
    await expect(form.alert).toHaveCount(0)
    expect((await context.cookies()).some((cookie) => cookie.name === 'vsd_session')).toBe(false)
  })

  await test.step('a reload stays on S1 (GET /api/session 401)', async () => {
    const check = page.waitForResponse(sessionCheck)
    await page.reload()
    expect((await check).status()).toBe(401)
    await expect(page).toHaveURL(`${BASE_URL}/signin`)
    await expect(form.heading).toBeVisible()
    await expect(form.alert).toHaveCount(0)
  })

  await test.step('the old cookie added back is refused: the server session is gone', async () => {
    await context.addCookies([saved!])
    expect((await context.cookies()).some((cookie) => cookie.name === 'vsd_session')).toBe(true)
    expect((await context.request.get('/api/session')).status()).toBe(401)
  })

  await expectNoAxeViolations(page, 'S1 after sign-out')
  expect(posts()).toBe(1)
})
