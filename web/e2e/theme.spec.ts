// The rail footer's theme control in a real browser (Stage 18 PR 6b): stored across a reload, and "Follow the system
// setting" following the OS live through CSS, with no reload and no listener. No sign-in POST: the session comes
// from this browser's setup project, and each test has a fresh context, so the stored choice does not leak.
import type { Page } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const theme = (page: Page) => page.getByRole('combobox', { name: 'Theme' })
const dataTheme = (page: Page) => page.evaluate(() => document.documentElement.getAttribute('data-theme'))
const background = (page: Page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor)

test('Dark is kept after a reload', async ({ signedInPage: page }) => {
  await page.goto('/analyze')
  await theme(page).selectOption({ label: 'Dark' })
  expect(await dataTheme(page)).toBe('dark')
  await page.reload()
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
  expect(await dataTheme(page)).toBe('dark')
  await expect(theme(page)).toHaveValue('dark')
})

test('Follow the system setting: no data-theme, the colours follow prefers-color-scheme live', async ({ signedInPage: page }) => {
  await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' })
  await page.goto('/analyze')
  await theme(page).selectOption({ label: 'Dark' })
  await theme(page).selectOption({ label: 'Follow the system setting' })
  expect(await dataTheme(page)).toBeNull()
  const light = await background(page)

  await page.emulateMedia({ colorScheme: 'dark' })
  await expect.poll(() => background(page)).not.toBe(light)
  const dark = await background(page)
  await page.emulateMedia({ colorScheme: 'light' })
  await expect.poll(() => background(page)).toBe(light)
  await page.emulateMedia({ colorScheme: 'dark' })
  await expect.poll(() => background(page)).toBe(dark)
  await expect(theme(page)).toHaveValue('system')
})

test('axe on the shell with the rail footer, light and dark', async ({ signedInPage: page }) => {
  await page.goto('/analyze')
  await expect(page.locator('.rail-foot .who-name')).toHaveText('e2e-main')
  await expectNoAxeViolations(page, 'rail footer')
})
