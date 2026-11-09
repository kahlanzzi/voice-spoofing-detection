// The lockout through S1 (docs/api_design.md §4.8 Lockout; Entry 134: the API answers the failure that reaches the
// threshold with 401 and sets the lock, so a fresh count gives 401 ×5, then 429). In the last project, after a
// second `restart web`: its six sign-in POSTs are segment B of the budget (support/budget.ts), a fresh zone's six.
import { randomBytes } from 'node:crypto'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'
import { ACCOUNTS, typeSecret } from './support/secret-guard.ts'
import { COPY, signInForm, signInResponse } from './support/signin.ts'
import { mustAdmin, restartWeb, webStartedAt } from './support/stack.ts'

// lockout_threshold in configs/api.yaml.
const THRESHOLD = 5

// Segment B needs its own zone; the start time proves the restart happened, whatever the timing.
test.beforeAll(async () => {
  const before = await webStartedAt()
  await restartWeb()
  expect(await webStartedAt(), 'the web container restarted before the lockout').toBeGreaterThan(before)
})

test('5 wrong passwords → invalid; the right one next → locked, no countdown, focus on the h1', async ({ page }) => {
  try {
    await page.goto('/signin')
    const form = signInForm(page)
    await form.username.fill(ACCOUNTS.locked)
    for (let attempt = 1; attempt <= THRESHOLD; attempt += 1) {
      await test.step(`wrong password ${attempt} → 401, the invalid banner`, async () => {
        const response = signInResponse(page)
        // A wrong password is not a secret: random junk, typed like any text.
        await form.password.fill(`wrong-${randomBytes(8).toString('hex')}`)
        await form.submit.click()
        expect((await response).status()).toBe(401)
        await expect(form.alert).toHaveText(COPY.invalid)
        await expect(form.password).toHaveValue('')
        await expect(form.username).toHaveValue(ACCOUNTS.locked)
      })
    }

    const response = signInResponse(page)
    await typeSecret(form.password)
    await form.submit.click()
    // nginx's own 429 (too_many_requests) would show the unavailable banner; this one must be the API's lock.
    expect((await response).status()).toBe(429)
    expect((await (await response).json()).error.code).toBe('account_locked')
    await expect(form.alert).toHaveText(COPY.locked)
    expect(await form.alert.textContent()).not.toMatch(/\d/)
    await expect(form.heading).toBeFocused()
    await expectNoAxeViolations(page, 'S1 locked')
  } finally {
    await mustAdmin('unlock', ACCOUNTS.locked)
  }
})
