import { defineConfig, devices } from '@playwright/test'

// End-to-end tests (Stage 18 PR 5a; docs/setup.md "End-to-end tests"): run with npm run test:e2e (chromium, also in
// CI) or test:e2e:firefox (local), which go through e2e/run.ts. Always the real stack, nginx on 8001, never the Vite
// dev server: its proxy rewrites Origin (A12), so it would hide the Origin and cookie bugs this suite exists to catch.
// No trace, video or screenshot: traces hold Set-Cookie headers, pictures can show the password field. Reporters
// list and json only, so the secret scan reads plain text. No retries and one worker: the sign-in POSTs of a run
// are counted against nginx's burst (the teardown checks nginx's log).
const browsers = [
  { name: 'chromium', use: devices['Desktop Chrome'] },
  { name: 'firefox', use: devices['Desktop Firefox'] },
] as const

export default defineConfig({
  testDir: './e2e',
  testMatch: '*.spec.ts',
  outputDir: 'test-results',
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  forbidOnly: !!process.env.CI,
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'playwright-report/results.json' }]],
  use: {
    baseURL: 'http://localhost:8001',
    timezoneId: 'Africa/Tunis',
    trace: 'off',
    video: 'off',
    screenshot: 'off',
  },
  projects: browsers.flatMap(({ name, use }) => [
    { name: `${name}-setup`, testMatch: 'signin.setup.spec.ts', use },
    { name, testIgnore: 'signin.setup.spec.ts', dependencies: [`${name}-setup`], use },
  ]),
})
