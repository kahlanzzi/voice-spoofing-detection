// S1's error variants on the real stack (Entry 134 live check), each scanned by axe in light and dark. Two sign-in
// POSTs: invalid and denied, each on its own account so neither can lock or disable another test's account.
import { randomBytes } from 'node:crypto'
import { expect, test } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { ACCOUNTS, typeSecret } from './support/secret-guard.ts'
import { COPY, countSignInPosts, signInForm, signInResponse } from './support/signin.ts'
import { mustAdmin } from './support/stack.ts'

test('field errors: both messages, aria-invalid, focus on the Analyst ID, nothing sent', async ({ page }) => {
  const posts = countSignInPosts(page)
  await page.goto('/signin')
  const form = signInForm(page)
  await form.submit.click()
  await expect(page.getByText('Enter your analyst ID.')).toBeVisible()
  await expect(page.getByText('Enter your password.')).toBeVisible()
  await expect(form.username).toHaveAttribute('aria-invalid', 'true')
  await expect(form.password).toHaveAttribute('aria-invalid', 'true')
  await expect(form.username).toBeFocused()
  expect(posts()).toBe(0)
  await expectNoAxeViolations(page, 'S1 field errors')
})

test('invalid credentials: banner, password cleared, Analyst ID kept and focused', async ({ page }) => {
  await page.goto('/signin')
  const form = signInForm(page)
  const response = signInResponse(page)
  // A wrong password is not a secret: random junk, typed like any text.
  const junk = `wrong-${randomBytes(8).toString('hex')}`
  await form.username.fill(ACCOUNTS.invalid)
  await form.password.fill(junk)
  await form.submit.click()
  expect((await response).status()).toBe(401)
  await expect(form.alert).toHaveText(COPY.invalid)
  await expect(form.password).toHaveValue('')
  await expect(form.username).toHaveValue(ACCOUNTS.invalid)
  await expect(form.username).toBeFocused()
  await expectNoAxeViolations(page, 'S1 invalid')
})

test('access denied: a disabled account with the right password, focus on the h1', async ({ page }) => {
  await mustAdmin('disable', ACCOUNTS.denied)
  try {
    await page.goto('/signin')
    const form = signInForm(page)
    const response = signInResponse(page)
    await form.username.fill(ACCOUNTS.denied)
    await typeSecret(form.password)
    await form.submit.click()
    expect((await response).status()).toBe(403)
    await expect(form.alert).toHaveText(COPY.denied)
    // No toHaveValue('') here: on failure it would print the real password. The invalid test checks the clearing.
    expect(await form.password.evaluate((input) => (input as HTMLInputElement).value === '')).toBe(true)
    await expect(form.heading).toBeFocused()
    await expectNoAxeViolations(page, 'S1 denied')
  } finally {
    await mustAdmin('enable', ACCOUNTS.denied)
  }
})
