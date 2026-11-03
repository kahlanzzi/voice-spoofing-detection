// The real stack (docker-compose.yml): nginx on http://localhost:8001 in front of the API. Commands run from the repo
// root, so Compose finds its file and .env (VSD_SIGNIN_HMAC_KEY, which it interpolates for every command). A
// password goes only on stdin, never in argv: api.admin reads it twice with getpass, which falls back to stdin
// without a TTY (`docker compose exec -T`, api/admin.py read_new_password). Output is kept and shown only on
// failure; the admin command prints prompts and warnings, never a password.
import { spawn } from 'node:child_process'
import { resolve } from 'node:path'
import { expect } from '@playwright/test'

export const BASE_URL = 'http://localhost:8001'
const REPO_ROOT = resolve(import.meta.dirname, '../../..')

export interface Result {
  code: number | null
  stdout: string
  stderr: string
}

export function compose(args: string[], stdin = ''): Promise<Result> {
  return new Promise((done, fail) => {
    const child = spawn('docker', ['compose', ...args], { cwd: REPO_ROOT, stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (chunk: Buffer) => (stdout += chunk.toString()))
    child.stderr.on('data', (chunk: Buffer) => (stderr += chunk.toString()))
    child.on('error', fail)
    child.on('close', (code) => done({ code, stdout, stderr }))
    child.stdin.end(stdin)
  })
}

async function mustCompose(args: string[], stdin = ''): Promise<Result> {
  const result = await compose(args, stdin)
  if (result.code !== 0) {
    throw new Error(`docker compose ${args.join(' ')} exited ${result.code}:\n${result.stderr.trim()}`)
  }
  return result
}

export type AdminCommand = 'create-account' | 'set-password' | 'disable' | 'enable' | 'unlock'

/** `python -m api.admin <command> <username>` in the api container; a password is typed twice on stdin. */
export function admin(command: AdminCommand, username: string, password?: string): Promise<Result> {
  const stdin = password === undefined ? '' : `${password}\n${password}\n`
  return compose(['exec', '-T', 'api', 'python', '-m', 'api.admin', command, username], stdin)
}

export async function mustAdmin(command: AdminCommand, username: string): Promise<void> {
  const result = await admin(command, username)
  if (result.code !== 0) throw new Error(`api.admin ${command} ${username} exited ${result.code}: ${result.stderr.trim()}`)
}

/** Create the account, or set its password if it exists; then enable and unlock it. */
export async function seedAccount(username: string, password: string): Promise<void> {
  const created = await admin('create-account', username, password)
  if (created.code !== 0) {
    if (!created.stderr.includes('already exists')) {
      throw new Error(`api.admin create-account ${username} exited ${created.code}: ${created.stderr.trim()}`)
    }
    const reset = await admin('set-password', username, password)
    if (reset.code !== 0) throw new Error(`api.admin set-password ${username} exited ${reset.code}: ${reset.stderr.trim()}`)
  }
  await mustAdmin('enable', username)
  await mustAdmin('unlock', username)
}

/** True when the api container reports healthy (its Docker health check, GET /api/health). */
export async function apiHealthy(): Promise<boolean> {
  const result = await compose(['ps', '--format', '{{.Service}} {{.Health}}', 'api'])
  return result.code === 0 && result.stdout.split('\n').some((line) => line.trim() === 'api healthy')
}

async function healthStatus(): Promise<number> {
  try {
    return (await fetch(`${BASE_URL}/api/health`)).status
  } catch {
    return 0
  }
}

/** Wait until GET /api/health answers 200 through nginx (Playwright's polling, no fixed sleep). */
export async function waitHealthy(): Promise<void> {
  await expect.poll(healthStatus, { message: `GET ${BASE_URL}/api/health through nginx`, timeout: 60_000 }).toBe(200)
}

/** Restart nginx: a new process gets a new shared-memory zone, so the sign-in rate limit starts empty. */
export async function restartWeb(): Promise<void> {
  await mustCompose(['restart', 'web'])
  await waitHealthy()
}
