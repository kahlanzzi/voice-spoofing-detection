// The app shell and the page-change rules in a real browser (Stage 18 PR 6a; HANDOFF lines 67-71, 85-89, 173-176).
// No sign-in POST: the session comes from this browser's setup project.
import type { Page } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

const TITLE = ' · Voice Check analyst console'
const announcer = (page: Page) => page.locator('#page-announcer')
const h1 = (page: Page) => page.getByRole('heading', { level: 1 })
const rail = (page: Page) => page.getByRole('complementary', { name: 'Main menu' })
const noSideScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)

test('first load: title only, no focus move, nothing announced; the skip link is the first Tab', async ({ signedInPage: page }) => {
  await page.goto('/overview')
  await expect(h1(page)).toHaveText('Overview')
  await expect(page).toHaveTitle(`Overview${TITLE}`)
  await expect(h1(page)).not.toBeFocused()
  await expect(announcer(page)).toHaveText('')
  await page.keyboard.press('Tab')
  const skip = page.getByRole('link', { name: 'Skip to main content' })
  await expect(skip).toBeFocused()
  await expect(skip).toBeVisible()
  await page.keyboard.press('Enter')
  await expect(h1(page)).toBeFocused()
})

test('a rail link and Back: focus on the h1, announcement, title, scroll to the top', async ({ signedInPage: page }) => {
  await page.setViewportSize({ width: 1280, height: 400 })
  await page.goto('/analyze')
  await page.evaluate(() => {
    document.body.style.minHeight = '3000px'
    window.scrollTo(0, 800)
  })
  await rail(page).getByRole('link', { name: 'History' }).click()
  await expect(h1(page)).toHaveText('History')
  await expect(h1(page)).toBeFocused()
  await expect(announcer(page)).toHaveText('History page')
  await expect(page).toHaveTitle(`History${TITLE}`)
  expect(await page.evaluate(() => window.scrollY)).toBe(0)

  await rail(page).getByRole('link', { name: 'System status' }).click()
  await expect(announcer(page)).toHaveText('System status page')
  await page.goBack()
  await expect(h1(page)).toHaveText('History')
  await expect(h1(page)).toBeFocused()
  await expect(announcer(page)).toHaveText('History page')
  await page.goForward()
  await expect(h1(page)).toHaveText('System status')
  await expect(h1(page)).toBeFocused()
  await expect(page).toHaveTitle(`System status${TITLE}`)
})

test('New analysis goes to Analyze', async ({ signedInPage: page }) => {
  await page.goto('/status')
  await rail(page).getByRole('button', { name: 'New analysis' }).click()
  await expect(page).toHaveURL(/\/analyze$/)
  await expect(h1(page)).toBeFocused()
})

for (const width of [1440, 1024, 768, 640, 390]) {
  test(`${width} px: no horizontal page scroll`, async ({ signedInPage: page }) => {
    await page.setViewportSize({ width, height: 800 })
    await page.goto(`/history/req_${'0123456789abcdef'.repeat(2)}`)
    await expect(h1(page)).toBeVisible()
    expect(await noSideScroll(page)).toBe(true)
  })
}

test('390 px: top bar, drawer with focus in, inert page, Escape back to Menu, axe', async ({ signedInPage: page }) => {
  await page.setViewportSize({ width: 390, height: 800 })
  await page.goto('/history')
  const menu = page.getByRole('button', { name: 'Menu' })
  await expect(menu).toBeVisible()
  await expect(page.getByRole('banner').getByRole('button', { name: 'New analysis' })).toBeVisible()
  await expect(rail(page)).toBeHidden()
  for (const target of [menu, page.getByRole('banner').getByRole('button', { name: 'New analysis' })]) {
    const box = await target.boundingBox()
    expect((box?.height ?? 0) >= 44 && (box?.width ?? 0) >= 44).toBe(true)
  }
  await expectNoAxeViolations(page, 'shell at 390 px, drawer closed')

  await menu.click()
  await expect(menu).toHaveAttribute('aria-expanded', 'true')
  await expect(rail(page)).toBeVisible()
  await expect(rail(page).getByRole('link', { name: 'Analyze' })).toBeFocused()
  await expect(page.locator('main')).toHaveAttribute('inert', '')
  await expectNoAxeViolations(page, 'shell at 390 px, drawer open')

  await page.keyboard.press('Escape')
  await expect(menu).toHaveAttribute('aria-expanded', 'false')
  await expect(menu).toBeFocused()
  await expect(page.locator('main')).not.toHaveAttribute('inert')

  await menu.click()
  await rail(page).getByRole('link', { name: 'Overview' }).click()
  await expect(h1(page)).toHaveText('Overview')
  await expect(h1(page)).toBeFocused()
  await expect(menu).toHaveAttribute('aria-expanded', 'false')
  await expect(rail(page)).toBeHidden()
})
