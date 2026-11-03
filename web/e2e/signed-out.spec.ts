// The route guard and `next` on the real stack (Entry 133 live check). No sign-in POST.
import { expect, test } from './support/fixtures.ts'
import { NEXT, SIGNIN_WITH_NEXT, signInForm } from './support/signin.ts'
import { BASE_URL } from './support/stack.ts'

test('signed out: a protected URL goes to S1 with the exact next, default variant', async ({ page }) => {
  await page.goto(NEXT)
  await expect(page).toHaveURL(`${BASE_URL}${SIGNIN_WITH_NEXT}`)
  const form = signInForm(page)
  await expect(form.heading).toBeVisible()
  await expect(form.username).toHaveValue('')
  await expect(form.password).toHaveValue('')
  await expect(form.alert).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Retry' })).toHaveCount(0)
})

test('signed in: /signin?next=… lands on next', async ({ signedInPage: page }) => {
  await page.goto(SIGNIN_WITH_NEXT)
  await expect(page).toHaveURL(`${BASE_URL}${NEXT}`)
  await expect(page.getByRole('heading', { level: 1, name: 'History' })).toBeVisible()
})
