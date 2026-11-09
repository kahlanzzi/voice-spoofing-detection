// The setup project of each browser: sign in through S1 as e2e-main, check what the browser does with the session
// (Origin, cookie, reload, storage), then save the state for the signed-in tests. Entry 133 and 134 live checks;
// docs/api_design.md §4.8 Cookie and CSRF. One sign-in POST.
import { expect, test } from '@playwright/test'
import { ACCOUNTS, accountPassword, cookieAttrs, stateFile, typeSecret, withoutValue } from './support/secret-guard.ts'
import { NEXT, SIGNIN_WITH_NEXT, signInForm, signInRequest, signInResponse } from './support/signin.ts'
import { BASE_URL } from './support/stack.ts'

test('sign in through the form, land on next, keep the session', async ({ page, context }, testInfo) => {
  await test.step('a protected URL goes to S1 with next', async () => {
    await page.goto(NEXT)
    await expect(page).toHaveURL(`${BASE_URL}${SIGNIN_WITH_NEXT}`)
  })

  const form = signInForm(page)
  const request = signInRequest(page)
  const response = signInResponse(page)
  await form.username.fill(ACCOUNTS.main)
  await typeSecret(form.password)
  await form.submit.click()

  await test.step('the POST carries the page Origin and answers 200', async () => {
    expect(await (await request).headerValue('origin')).toBe(BASE_URL)
    expect((await response).status()).toBe(200)
  })

  await test.step('Set-Cookie: HttpOnly, Secure, SameSite=Strict, Path=/api, no Domain or lifetime', async () => {
    const setCookie = await (await response).headerValue('set-cookie')
    expect(setCookie !== null).toBe(true)
    expect(cookieAttrs(setCookie ?? '')).toEqual({
      name: 'vsd_session', httpOnly: true, secure: true, sameSite: 'strict', path: '/api',
      domain: false, maxAge: false, expires: false,
    })
  })

  await test.step('lands on next', async () => {
    await expect(page).toHaveURL(`${BASE_URL}${NEXT}`)
    await expect(page.getByRole('heading', { level: 1, name: 'History' })).toBeVisible()
  })

  await test.step('the browser keeps the cookie as set (Secure accepted on http://localhost)', async () => {
    const jar = (await context.cookies()).map(withoutValue)
    expect(jar).toEqual([expect.objectContaining({
      name: 'vsd_session', domain: 'localhost', path: '/api', httpOnly: true, secure: true, sameSite: 'Strict', expires: -1,
    })])
  })

  await test.step('HttpOnly: absent from document.cookie inside Path=/api', async () => {
    // On /history the cookie's Path alone hides it, so the check runs on a page under /api.
    const answer = await page.goto('/api/session')
    expect(answer?.status()).toBe(200)
    expect(await page.evaluate(() => document.cookie.includes('vsd_session'))).toBe(false)
  })

  await test.step('a reload stays signed in', async () => {
    await page.goto(NEXT)
    await expect(page.getByRole('heading', { level: 1, name: 'History' })).toBeVisible()
    const check = page.waitForResponse((r) => r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/session')
    await page.reload()
    expect((await check).status()).toBe(200)
    await expect(page).toHaveURL(`${BASE_URL}${NEXT}`)
    await expect(page.getByRole('heading', { level: 1, name: 'History' })).toBeVisible()
  })

  await test.step('no Web Storage but the router key; the password is not in the page', async () => {
    const storage = await page.evaluate(() => ({ local: localStorage.length, session: Object.keys(sessionStorage) }))
    expect(storage.local).toBe(0)
    expect(storage.session.filter((key) => key !== 'remix-router-transitions')).toEqual([])
    expect((await page.content()).includes(accountPassword())).toBe(false)
  })

  await context.storageState({ path: stateFile(testInfo.project.name) })
})
