// S2 and S3 in a real browser, on the real stack (Stage 18 PR A; HANDOFF §1): the limits from GET /api/limits, a held
// and a failed answer through page.route. No sign-in POST: e2e-main's session from the setup project.
import type { Page, Route } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'

// A mono 8 kHz 16-bit WAV of silence, built here: no dataset file and nothing committed.
function wav(seconds: number): Buffer {
  const data = 8000 * 2 * seconds
  const header = Buffer.alloc(44)
  header.write('RIFF', 0, 'latin1')
  header.writeUInt32LE(36 + data, 4)
  header.write('WAVEfmt ', 8, 'latin1')
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(8000, 24)
  header.writeUInt32LE(16000, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36, 'latin1')
  header.writeUInt32LE(data, 40)
  return Buffer.concat([header, Buffer.alloc(data)])
}

async function pick(page: Page, name: string, buffer: Buffer) {
  await page.locator('#file-input').setInputFiles({ name, mimeType: 'application/octet-stream', buffer })
}

async function ready(page: Page) {
  await page.goto('/analyze')
  await expect(choose(page)).toBeEnabled()
}

const checks = (page: Page) => page.getByRole('list', { name: 'Checks before upload' }).getByRole('listitem')
const INPUT_ERROR = 'Input error: this file cannot be analyzed.'

const limitsAnswer = (page: Page) =>
  page.waitForResponse((r) => r.request().method() === 'GET' && new URL(r.url()).pathname === '/api/limits')
const choose = (page: Page) => page.getByRole('button', { name: 'Choose recording' })
const requirements = (page: Page) => page.getByLabel('Upload requirements')

test('S2 loads the limits from the service; no axe violation', async ({ signedInPage: page }) => {
  const answer = limitsAnswer(page)
  await page.goto('/analyze')
  const response = await answer
  expect(response.status()).toBe(200)
  const body = await response.json()
  await expect(requirements(page).locator('dd')).toHaveText([
    body.formats.map((format: string) => format.toUpperCase()).join(', '),
    `${Number((body.max_size_bytes / 1048576).toFixed(1))} MB`,
    `${body.max_duration_s} s`,
    body.audio_deletion,
  ])
  await expect(page.getByText('Mock values · real limits not decided')).toBeVisible()
  await expect(choose(page)).toBeEnabled()
  await expectNoAxeViolations(page, 'S2 limits ready')
})

test('limits held: the loading line, Choose recording disabled; no axe violation', async ({ signedInPage: page }) => {
  let release: () => void = () => undefined
  const held = new Promise<void>((done) => {
    release = done
  })
  await page.route('**/api/limits', async (route: Route) => {
    await held
    await route.continue()
  })
  await page.goto('/analyze')
  await expect(page.getByText('Loading upload limits from the service…')).toBeVisible()
  await expect(choose(page)).toBeDisabled()
  await expectNoAxeViolations(page, 'S2 limits loading')
  release()
  await expect(requirements(page)).toBeVisible()
  await expect(choose(page)).toBeEnabled()
})

test('limits 500: the alert, analysis off, no axe violation; Retry loads them', async ({ signedInPage: page }) => {
  const body = JSON.stringify({ error: { code: 'internal_error', message: 'x', request_id: null } })
  await page.route('**/api/limits', (route: Route) => route.fulfill({ status: 500, contentType: 'application/json', body }))
  await page.goto('/analyze')
  await expect(page.getByRole('alert')).toContainText('Upload limits could not be loaded.')
  await expect(requirements(page)).toHaveCount(0)
  await expect(choose(page)).toBeDisabled()
  await expectNoAxeViolations(page, 'S2 limits failed')
  await page.unroute('**/api/limits')
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(requirements(page)).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('heading', { level: 2, name: 'Choose the recorded segment' })).toBeFocused()
})

test('a valid WAV → S3: all checks passed, Analyze recording enabled; no axe violation', async ({ signedInPage: page }) => {
  await ready(page)
  await pick(page, 'call.wav', wav(1))
  // The duration row is the CSP guard: without media-src blob: it reads "could not be read" (Entry 145).
  await expect(checks(page)).toHaveText([/^Passed: Format WAV is accepted$/, /^Passed: Size 16 KB is within /, /^Passed: Duration 1 s is within /])
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Analyze recording' })).toBeEnabled()
  await expect(page.locator('#file-h')).toBeFocused()
  await expect(page.locator('#analyze-announcer')).toHaveText('File selected: call.wav. All checks passed.')
  await expectNoAxeViolations(page, 'S3 all checks passed')
})

test('a 61 s WAV → the recording is too long', async ({ signedInPage: page }) => {
  await ready(page)
  await pick(page, 'long.wav', wav(61))
  await expect(checks(page).nth(2)).toHaveText(/^Failed: Recording too long: 1 min 1 s is over the 60 s limit\./)
  await expect(page.getByRole('alert')).toContainText(INPUT_ERROR)
})

test('an unsupported file → the failure text; no axe violation', async ({ signedInPage: page }) => {
  await ready(page)
  await pick(page, 'call.ogg', Buffer.from('not audio'))
  await expect(checks(page).nth(0)).toHaveText(/^Failed: Unsupported format: \.ogg files cannot be analyzed\./)
  await expect(checks(page).nth(2)).toHaveText('Duration not checked: the file cannot be read as audio')
  await expect(page.getByRole('alert')).toContainText(INPUT_ERROR)
  await expectNoAxeViolations(page, 'S3 unsupported format')
})

test('Remove file and New analysis return to S2', async ({ signedInPage: page }) => {
  await ready(page)
  await pick(page, 'call.wav', wav(1))
  await page.getByRole('button', { name: 'Remove file' }).click()
  await expect(choose(page)).toBeFocused()
  await pick(page, 'call.wav', wav(1))
  await expect(checks(page)).toHaveCount(3)
  await page.getByRole('complementary', { name: 'Main menu' }).getByRole('button', { name: 'New analysis' }).click()
  await expect(choose(page)).toBeVisible()
  await expect(checks(page)).toHaveCount(0)
  await expect(page.getByRole('heading', { level: 1 })).toBeFocused()
})
