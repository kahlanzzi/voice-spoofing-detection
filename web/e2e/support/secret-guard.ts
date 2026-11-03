// The run's secrets (the e2e accounts' password, the session cookie) and how they are kept out of every output.
// run.ts makes the password and passes it only in the child's env; nothing here prints, writes or logs a value.
// Playwright's step titles render typed text (`Fill "{value}"`, `Type "{text}"`, `Insert "{text}"`), and the JSON
// report keeps them, so the password is never typed with fill/type/insertText: typeSecret() edits through
// Evaluate, whose title renders no argument. Assertions take booleans, never a secret, because a failing expect
// prints what it was given.
import { randomBytes } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Cookie, Locator } from '@playwright/test'

export const PASSWORD_ENV = 'VSD_E2E_PASSWORD'
export const STATE_DIR_ENV = 'VSD_E2E_STATE_DIR'
export const ACCOUNTS = { main: 'e2e-main', invalid: 'e2e-invalid', denied: 'e2e-denied', locked: 'e2e-locked' } as const

// docs/api_design.md §4.8 and api/auth.py password_problem: 15–128 code points, no username, no service word.
const SERVICE_WORDS = ['vsd', 'voice', 'spoof']
const SESSION_COOKIE = /vsd_session=[A-Za-z0-9_-]{43}/

/** 48 hex characters (192 bits). Hex has no letter of the service words and no `-`, so it always passes §4.8. */
export function newPassword(): string {
  return randomBytes(24).toString('hex')
}

/** Why a password given in the env cannot be used, or null. */
export function passwordProblem(password: string): string | null {
  const length = [...password].length
  if (length < 15 || length > 128) return 'it must have 15 to 128 characters'
  const folded = password.toLowerCase()
  if (SERVICE_WORDS.some((word) => folded.includes(word))) return 'it must not contain vsd, voice or spoof'
  if (Object.values(ACCOUNTS).some((name) => folded.includes(name))) return 'it must not contain an e2e username'
  if (/\s/.test(password)) return 'it must not contain whitespace (it is sent on one stdin line)'
  return null
}

function requiredEnv(name: string): string {
  const value = process.env[name]
  if (!value) throw new Error(`${name} is not set: run the suite with npm run test:e2e (or test:e2e:firefox).`)
  return value
}

export function accountPassword(): string {
  return requiredEnv(PASSWORD_ENV)
}

export function stateFile(project: string): string {
  return join(requiredEnv(STATE_DIR_ENV), `${project}.json`)
}

/** Types the password into the field as editing (the paste path, which React's onChange sees), untitled. */
export async function typeSecret(field: Locator): Promise<void> {
  await field.evaluate((element, value) => {
    const input = element as HTMLInputElement
    input.focus()
    document.execCommand('insertText', false, value)
  }, accountPassword())
}

export interface CookieAttributes {
  name: string
  httpOnly: boolean
  secure: boolean
  sameSite: string | null
  path: string | null
  domain: boolean
  maxAge: boolean
  expires: boolean
}

/** The attributes of one Set-Cookie header, without its value. */
export function cookieAttrs(setCookie: string): CookieAttributes {
  const [pair = '', ...parts] = setCookie.split(';').map((part) => part.trim())
  const attributes = new Map(parts.map((part) => {
    const at = part.indexOf('=')
    return at === -1 ? [part.toLowerCase(), ''] : [part.slice(0, at).toLowerCase(), part.slice(at + 1)]
  }))
  return {
    name: pair.split('=')[0] ?? '',
    httpOnly: attributes.has('httponly'),
    secure: attributes.has('secure'),
    sameSite: attributes.get('samesite') ?? null,
    path: attributes.get('path') ?? null,
    domain: attributes.has('domain'),
    maxAge: attributes.has('max-age'),
    expires: attributes.has('expires'),
  }
}

export function withoutValue(cookie: Cookie): Omit<Cookie, 'value'> {
  return Object.fromEntries(Object.entries(cookie).filter(([key]) => key !== 'value')) as Omit<Cookie, 'value'>
}

/** The cookie values saved in the run's storage-state files (they are secrets too). */
export function stateCookieValues(stateDir: string): string[] {
  return readdirSync(stateDir)
    .filter((name) => name.endsWith('.json'))
    .flatMap((name) => {
      const state = JSON.parse(readFileSync(join(stateDir, name), 'utf8')) as { cookies?: { value: string }[] }
      return (state.cookies ?? []).map((cookie) => cookie.value).filter((value) => value.length > 0)
    })
}

/** Which kind of secret the text holds, or null. Never returns the value. */
export function secretKind(text: string, secrets: { password: string; cookieValues: string[] }): string | null {
  if (text.includes(secrets.password)) return 'password'
  if (secrets.cookieValues.some((value) => text.includes(value))) return 'session cookie value'
  if (SESSION_COOKIE.test(text)) return 'session cookie'
  return null
}

function filesUnder(dir: string): string[] {
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return []
  }
  return entries.flatMap((entry) => {
    const path = join(dir, entry.name)
    return entry.isDirectory() ? filesUnder(path) : entry.isFile() ? [path] : []
  })
}

/** A plain-text scan of every file under the folders (read as bytes, decoded as latin1 so no byte is lost). */
export function scanOutputs(dirs: string[], secrets: { password: string; cookieValues: string[] }) {
  const files = dirs.flatMap(filesUnder)
  const hits = files.flatMap((file) => {
    const kind = secretKind(readFileSync(file).toString('latin1'), secrets)
    return kind === null ? [] : [{ file, kind }]
  })
  return { files: files.length, hits }
}
