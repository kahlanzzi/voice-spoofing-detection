// The signed-in stub pages (Stage 18 PR 3c, src/routing/router.tsx): one h1 each, and axe in light and dark. No
// sign-in POST: the session comes from this browser's setup project.
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const PAGES = [
  { path: '/analyze', h1: 'Analyze' },
  { path: '/history', h1: 'History' },
  { path: `/history/req_${'0123456789abcdef'.repeat(2)}`, h1: 'Analysis details' },
  { path: '/overview', h1: 'Overview' },
  { path: '/status', h1: 'System status' },
]

for (const { path, h1 } of PAGES) {
  test(`${path}: one h1 "${h1}", no axe violation`, async ({ signedInPage: page }) => {
    await page.goto(path)
    await expect(page.getByRole('heading', { level: 1, name: h1 })).toBeVisible()
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1)
    await expectNoAxeViolations(page, path)
  })
}
