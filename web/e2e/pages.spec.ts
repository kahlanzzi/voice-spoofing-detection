// The signed-in stub pages inside the shell (Stage 18 PR 3c, 6a; src/routing/router.tsx): one h1 and one main each,
// the rail with the current item, the page title, and axe in light and dark. No sign-in POST: the session comes from
// this browser's setup project.
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const PAGES = [
  { path: '/analyze', h1: 'Analyze a recording', current: 'Analyze' },
  { path: '/history', h1: 'History', current: 'History' },
  { path: `/history/req_${'0123456789abcdef'.repeat(2)}`, h1: 'Analysis details', current: 'History' },
  { path: '/overview', h1: 'Overview', current: 'Overview' },
  { path: '/status', h1: 'System status', current: 'System status' },
]

for (const { path, h1, current } of PAGES) {
  test(`${path}: one h1 "${h1}" in the shell, ${current} current, no axe violation`, async ({ signedInPage: page }) => {
    await page.goto(path)
    await expect(page.getByRole('heading', { level: 1, name: h1 })).toBeVisible()
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1)
    await expect(page.getByRole('main')).toHaveCount(1)
    await expect(page).toHaveTitle(`${h1} · Voice Check analyst console`)
    const nav = page.getByRole('complementary', { name: 'Main menu' }).getByRole('navigation', { name: 'Pages' })
    await expect(nav.locator('[aria-current="page"]')).toHaveText([current])
    await expectNoAxeViolations(page, path)
  })
}
