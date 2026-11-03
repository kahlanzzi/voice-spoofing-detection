// nginx's sign-in rate limit (deploy/nginx.conf: 10/min, burst 5, nodelay) allows 6 POSTs at once from a fresh zone.
// A run makes PLANNED_SIGN_IN_POSTS; run.ts reads nginx's access log after the run and checks them. Node only (no
// Playwright import), so run.ts can use it.
import { spawn } from 'node:child_process'
import { resolve } from 'node:path'

// signin.setup (1), signin-errors invalid (1) and denied (1); docs/setup.md "End-to-end tests".
export const PLANNED_SIGN_IN_POSTS = 3

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
 * Why the count breaks the budget, or null. A green run must show exactly the planned POSTs: fewer (0 included)
 * means the check is not reading the log. A failed run may have stopped early, so it may show fewer, never more.
 * Any 429 means nginx limited the suite.
 */
export function budgetProblem(count: SignInCount, green: boolean, planned = PLANNED_SIGN_IN_POSTS): string | null {
  if (count.limited > 0) return `${count.limited} sign-in POST(s) answered 429`
  if (green && count.posts !== planned) return `a green run must show ${planned} sign-in POSTs, nginx's log shows ${count.posts}`
  if (count.posts > planned) return `${count.posts} sign-in POSTs, more than the planned ${planned}`
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
