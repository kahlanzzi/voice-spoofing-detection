// Before the run (docs/setup.md "End-to-end tests"): the stack must be up; nginx is restarted so the sign-in rate
// limit starts empty; the e2e accounts are created or given this run's password, enabled and unlocked.
import { ACCOUNTS, accountPassword, STATE_DIR_ENV } from './support/secret-guard.ts'
import { expect } from '@playwright/test'
import { apiHealthy, BASE_URL, restartWeb, seedAccount, webStartedAt } from './support/stack.ts'

export default async function globalSetup(): Promise<void> {
  const password = accountPassword()
  if (!process.env[STATE_DIR_ENV]) throw new Error(`${STATE_DIR_ENV} is not set: run npm run test:e2e.`)
  if (!(await apiHealthy())) throw new Error('The api container is not healthy: run docker compose up -d first.')
  try {
    await fetch(`${BASE_URL}/api/health`)
  } catch {
    throw new Error(`${BASE_URL} does not answer: run docker compose up -d first.`)
  }
  const before = await webStartedAt()
  await restartWeb()
  expect(await webStartedAt(), 'the web container restarted before the run').toBeGreaterThan(before)
  for (const username of Object.values(ACCOUNTS)) await seedAccount(username, password)
}
