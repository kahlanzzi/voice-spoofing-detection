// Analyze recording to its answer in a real browser (Stage 18 PR B; HANDOFF §1 S4-S6): the real POST through nginx
// against MockAnalyzer, and S5 through page.route where the real stack cannot fail on demand. No sign-in POST: e2e-main's
// session from the setup project, and /api/analyses is outside nginx's sign-in zone. Each green run stores 6 rows, with
// no analyst on them (Entry 146).
//
// The hold is route.continue() with no overrides, so the browser sends its own body. Not route.fetch(): it re-sends
// request.postDataBuffer(), which may not hold a File. A lost body would show as the real API's 400, never as a pass.
import type { Page, Request, Response, Route } from '@playwright/test'
import { expectNoAxeViolations } from './support/axe.ts'
import { expect, test } from './support/fixtures.ts'
import { BASE_URL } from './support/stack.ts'
import { wav } from './support/wav.ts'

const ID = `req_${'e'.repeat(32)}`
const NO_VERDICT = 'The analysis did not complete. No verdict was produced.'

const VERDICTS: Record<string, string> = {
  synthetic: 'Likely synthetic voice',
  uncertain: 'Cannot decide',
  authentic: 'Likely authentic voice',
}

const analyzeButton = (page: Page) => page.getByRole('button', { name: 'Analyze recording' })
const waitingButton = (page: Page) => page.getByRole('button', { name: 'Analyzing…' })
const newAnalysis = (page: Page) => page.getByRole('complementary', { name: 'Main menu' }).getByRole('button', { name: 'New analysis' })
const resultTitle = (page: Page) => page.locator('#res-h')

const isPost = (request: Request) => request.method() === 'POST' && new URL(request.url()).pathname === '/api/analyses'
const postAnswer = (page: Page) => page.waitForResponse((response: Response) => isPost(response.request()))

function countPosts(page: Page): Request[] {
  const posts: Request[] = []
  page.on('request', (request) => {
    if (isPost(request)) posts.push(request)
  })
  return posts
}

async function holdPost(page: Page): Promise<() => void> {
  let release: () => void = () => undefined
  const held = new Promise<void>((done) => {
    release = done
  })
  await page.route('**/api/analyses', async (route: Route) => {
    await held
    await route.continue()
  })
  return release
}

function answerPost(page: Page, status: number, code: string, requestId: string | null) {
  const body = JSON.stringify({ error: { code, message: 'x', request_id: requestId } })
  return page.route('**/api/analyses', (route: Route) => route.fulfill({ status, contentType: 'application/json', body }))
}

// S3 with the file checked.
async function selected(page: Page, name: string, buffer: Buffer) {
  await page.goto('/analyze')
  await expect(page.getByRole('button', { name: 'Choose recording' })).toBeEnabled()
  await page.locator('#file-input').setInputFiles({ name, mimeType: 'application/octet-stream', buffer })
  await expect(analyzeButton(page)).toBeEnabled()
}

async function expectResult(page: Page, response: Response) {
  expect(response.status()).toBe(201)
  const body = await response.json()
  await expect(resultTitle(page)).toHaveText(VERDICTS[body.prediction]!)
  await expect(page.locator('.risk')).toHaveText(`${body.risk_level} RISK`)
  await expect(page.locator('.api-notice')).toHaveText(`Notice from the service: ${body.notice}`)
  await expect(page.locator('.result-body code')).toHaveText(body.request_id)
  await expect(page.getByText('Mock result')).toBeVisible()
  return body
}

test('the real stack: S4, then the result card with the values of the response; no axe violation', async ({ signedInPage: page }) => {
  // What the app hands to fetch: Playwright may not see the wire body of a File.
  await page.addInitScript(() => {
    const realFetch = window.fetch
    const names: string[] = []
    Object.assign(window, { uploadNames: names })
    window.fetch = (input, init) => {
      if (init?.body instanceof FormData) names.push(...[...init.body.values()].map((value) => (value instanceof File ? value.name : 'text')))
      return realFetch(input, init)
    }
  })
  const posts = countPosts(page)
  const release = await holdPost(page)
  await selected(page, 'the customer call.wav', wav(1))
  const answer = postAnswer(page)
  await analyzeButton(page).click()

  await expect(waitingButton(page)).toBeDisabled()
  await expect(page.getByRole('list', { name: 'Progress' }).getByRole('listitem')).toHaveText(['Validating (done)', 'Uploading and analyzing'])
  await expect(page.locator('#file-h')).toBeFocused()
  await expect(page.locator('#analyze-announcer')).toHaveText('Analysis started.')
  await expectNoAxeViolations(page, 'S4')
  release()

  const response = await answer
  const body = await expectResult(page, response)
  await expect(resultTitle(page)).toBeFocused()
  await expect(page.locator('#analyze-assertive')).toContainText(`Result ready. Verdict: ${VERDICTS[body.prediction]}.`)
  await expect(page).toHaveTitle(/^Analysis result · /)
  await page.getByText('Technical details').click()
  await expect(page.locator('.score-val')).toHaveText(body.confidence.toFixed(2))
  await expect(page.locator('details.more dd').first()).toHaveText(body.model_version)
  await expectNoAxeViolations(page, 'result')

  expect(posts).toHaveLength(1)
  expect((await response.request().allHeaders()).origin).toBe(BASE_URL)
  expect(await page.evaluate(() => (window as unknown as { uploadNames: string[] }).uploadNames)).toEqual(['upload.wav'])
})

test('the other two verdicts on the real stack; no axe violation', async ({ signedInPage: page }) => {
  for (const [seconds, prediction] of [[2, 'synthetic'], [7, 'uncertain']] as const) {
    await selected(page, 'call.wav', wav(seconds))
    const answer = postAnswer(page)
    await analyzeButton(page).click()
    const body = await expectResult(page, await answer)
    expect(body.prediction).toBe(prediction)
    await expectNoAxeViolations(page, `result ${prediction}`)
  }
})

test('a double click sends one POST', async ({ signedInPage: page }) => {
  const posts = countPosts(page)
  const release = await holdPost(page)
  await selected(page, 'call.wav', wav(1))
  const answer = postAnswer(page)
  await analyzeButton(page).dblclick()
  await expect(waitingButton(page)).toBeDisabled()
  release()
  await expectResult(page, await answer)
  expect(posts).toHaveLength(1)
})

test('New analysis during the wait: S4 stays, the result arrives, one POST', async ({ signedInPage: page }) => {
  const posts = countPosts(page)
  const release = await holdPost(page)
  await selected(page, 'call.wav', wav(1))
  const answer = postAnswer(page)
  await analyzeButton(page).click()
  await expect(waitingButton(page)).toBeDisabled()
  await newAnalysis(page).click()
  await expect(page.getByRole('heading', { level: 1 })).toBeFocused()
  await expect(waitingButton(page)).toBeDisabled()
  release()
  await expectResult(page, await answer)
  expect(posts).toHaveLength(1)
})

test('no answer: S5 with no request ID, no axe violation; Retry analysis gets the real result', async ({ signedInPage: page }) => {
  const posts = countPosts(page)
  await page.route('**/api/analyses', (route: Route) => route.abort())
  await selected(page, 'call.wav', wav(1))
  await analyzeButton(page).click()
  await expect(page.getByRole('alert')).toHaveText(`The connection to the service was interrupted. ${NO_VERDICT}`)
  await expect(page.locator('#err-h')).toBeFocused()
  await expect(page.getByText('None: the request was not accepted by the service')).toBeVisible()
  await expect(page.locator('.result')).toHaveCount(0)
  await expectNoAxeViolations(page, 'S5 network')
  await page.unroute('**/api/analyses')
  const answer = postAnswer(page)
  await page.getByRole('button', { name: 'Retry analysis' }).click()
  await expectResult(page, await answer)
  expect(posts).toHaveLength(2)
})

test('503: the service variant with the request ID and the status link; no axe violation', async ({ signedInPage: page }) => {
  await answerPost(page, 503, 'analysis_unavailable', ID)
  await selected(page, 'call.wav', wav(1))
  await analyzeButton(page).click()
  await expect(page.getByRole('alert')).toHaveText(`The analysis service is unavailable. ${NO_VERDICT}`)
  await expect(page.getByText(ID)).toBeVisible()
  await expect(page.getByRole('link', { name: 'Check system status' })).toHaveAttribute('href', '/status')
  await expectNoAxeViolations(page, 'S5 service')
})

test('bytes that are not audio: unreadable in the browser, Analyze enabled, the real 422 gives Corrupt audio', async ({ signedInPage: page }) => {
  await selected(page, 'bad.wav', Buffer.from('not audio'))
  await expect(page.getByText('Duration could not be read in the browser; the service will check it')).toBeVisible()
  const answer = postAnswer(page)
  await analyzeButton(page).click()
  const response = await answer
  expect(response.status()).toBe(422)
  const { error } = await response.json()
  expect(error.code).toBe('corrupt_audio')
  await expect(page.getByRole('alert')).toHaveText(`The recording could not be decoded. ${NO_VERDICT}`)
  await expect(page.getByText(error.request_id)).toBeVisible()
  await expectNoAxeViolations(page, 'S5 corrupt')
})

test('401 on the POST: the session variant; Sign in again goes to S1 with next=/analyze', async ({ signedInPage: page }) => {
  await answerPost(page, 401, 'session_expired', null)
  await selected(page, 'call.wav', wav(1))
  await analyzeButton(page).click()
  await expect(page.getByRole('alert')).toHaveText(`Your session has expired. ${NO_VERDICT}`)
  await page.getByRole('button', { name: 'Sign in again' }).click()
  await expect(page).toHaveURL(/\/signin\?next=%2Fanalyze$/)
})

test('status Unavailable: the warning on the file card, and Analyze stays enabled', async ({ signedInPage: page }) => {
  const body = JSON.stringify({ error: { code: 'analysis_unavailable', message: 'x', request_id: null } })
  await page.route('**/api/status', (route: Route) => route.fulfill({ status: 503, contentType: 'application/json', body }))
  await selected(page, 'call.wav', wav(1))
  await expect(page.getByText('Analysis is currently unavailable. You can still try; the request may fail.')).toBeVisible()
  await expect(analyzeButton(page)).toBeEnabled()
})
