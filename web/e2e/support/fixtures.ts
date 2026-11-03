// `signedInPage`: a page in a fresh context holding e2e-main's session, saved by this browser's setup project
// (signin.setup.spec.ts) in the run's private temp folder.
import { test as base, type Page } from '@playwright/test'
import { stateFile } from './secret-guard.ts'
import { BASE_URL } from './stack.ts'

export const test = base.extend<{ signedInPage: Page }>({
  signedInPage: async ({ browser }, use, testInfo) => {
    const context = await browser.newContext({
      baseURL: BASE_URL,
      timezoneId: testInfo.project.use.timezoneId,
      storageState: stateFile(`${testInfo.project.name}-setup`),
    })
    await use(await context.newPage())
    await context.close()
  },
})

export { expect } from '@playwright/test'
