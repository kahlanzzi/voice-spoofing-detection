// nginx's sign-in rate limit (deploy/nginx.conf: 10/min, burst 5, nodelay) allows 6 POSTs at once from a fresh zone,
// and counts every POST to /api/session, whatever the API answers. A run has two segments, each opened by a
// `restart web` (a fresh zone), each at most ZONE_CAPACITY. run.ts reads nginx's access log after the run and checks
// the totals. Node only (no Playwright import), so run.ts can use it.
import { spawn } from 'node:child_process'
import { resolve } from 'node:path'

export const ZONE_CAPACITY = 6

// docs/setup.md "End-to-end tests". A (global setup's restart): signin.setup 1, signin-errors invalid 1 and denied 1,
// http.spec without Origin 1 and with a foreign one 1. B (locked.spec's restart): 5 wrong passwords and the right one.
export const SEGMENTS = { A: 5, B: 6 } as const

// The one planned 429 is the API's account_locked (locked.spec checks its code; nginx's own 429 would be a second).
export const PLAN = { posts: SEGMENTS.A + SEGMENTS.B, limited: 1 } as const

const SIGN_IN_POST = /"POST \/api\/session" (\d{3}) /

export interface SignInCount {
  posts: number
  limited: number
}

/** The sign-in POSTs in nginx's access-log lines (format: time "METHOD /path" status bytes duration), and the 429s. */
export function countLoggedSignIns(lines: string[]): SignInCount {
  const statuses = lines.flatMap((line) => SIGN_IN_POST.exec(line)?.[1] ?? [])
  return { posts: statuses.length, limited: statuses.filter((status) => status === '429').length }
}

/**
 * Why the count breaks the budget, or null. A green run must show exactly the planned POSTs and 429s: fewer POSTs
 * (0 included) means the check is not reading the log. A failed run may have stopped early, so it may show fewer,
 * never more. A 429 beyond the planned one means nginx limited the suite.
 */
export function budgetProblem(count: SignInCount, green: boolean, plan: SignInCount = PLAN): string | null {
  if (count.limited > plan.limited) return `${count.limited} sign-in POST(s) answered 429, more than the planned ${plan.limited}`
  if (count.posts > plan.posts) return `${count.posts} sign-in POSTs, more than the planned ${plan.posts}`
  if (green && count.posts !== plan.posts) return `a green run must show ${plan.posts} sign-in POSTs, nginx's log shows ${count.posts}`
  if (green && count.limited !== plan.limited) return `a green run must show ${plan.limited} × 429, nginx's log shows ${count.limited}`
  return null
}

/** nginx's log lines since the time, from `docker compose logs` run in the repo root. */
export function webLogSince(iso: string): Promise<string[]> {
  return new Promise((done, fail) => {
    const child = spawn('docker', ['compose', 'logs', '--no-log-prefix', '--since', iso, 'web'], {
      cwd: resolve(import.meta.dirname, '../../..'),
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (chunk: Buffer) => (stdout += chunk.toString()))
    child.stderr.on('data', (chunk: Buffer) => (stderr += chunk.toString()))
    child.on('error', fail)
    child.on('close', (code) => (code === 0 ? done(stdout.split('\n')) : fail(new Error(`docker compose logs web exited ${code}: ${stderr.trim()}`))))
  })
}
