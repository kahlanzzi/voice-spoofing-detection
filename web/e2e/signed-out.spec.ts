// The route guard and `next` on the real stack (Entry 133 live check); axe on S1's default variant. No sign-in POST.
import { expectNoAxeViolations } from './support/axe.ts'
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
  await expect(form.retry).toHaveCount(0)
  await expectNoAxeViolations(page, 'S1 default')
})

test('signed in: /signin?next=… lands on next', async ({ signedInPage: page }) => {
  await page.goto(SIGNIN_WITH_NEXT)
  await expect(page).toHaveURL(`${BASE_URL}${NEXT}`)
  await expect(page.getByRole('heading', { level: 1, name: 'History' })).toBeVisible()
})
