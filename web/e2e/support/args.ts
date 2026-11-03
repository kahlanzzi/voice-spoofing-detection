// The Playwright command line for one browser (run.ts). Spawned without a shell, so the project filter has no quotes:
// a quote would reach Playwright literally and match no project. One browser per run keeps the sign-in POSTs of a run
// inside nginx's burst (docs/setup.md "End-to-end tests").

export const BROWSERS = ['chromium', 'firefox'] as const
export type Browser = (typeof BROWSERS)[number]

export function isBrowser(value: string | undefined): value is Browser {
  return (BROWSERS as readonly (string | undefined)[]).includes(value)
}

export function playwrightArgs(browser: string): string[] {
  if (!isBrowser(browser)) throw new Error(`Unknown browser ${browser}; use ${BROWSERS.join(' or ')}.`)
  return ['test', `--project=${browser}*`]
}
