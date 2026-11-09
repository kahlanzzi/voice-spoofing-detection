// The only way to run the e2e suite (npm run test:e2e / test:e2e:firefox): node e2e/run.ts <chromium|firefox>.
// It owns the run's secrets from start to end. The e2e accounts' password (generated, or VSD_E2E_PASSWORD if set)
// and a private temp folder for the signed-in storage state go only into Playwright's env. Playwright's output is
// redacted line by line before it reaches the terminal or a CI log. After the run, every file in test-results/ and
// playwright-report/ is scanned as plain text for the password, the saved cookie values and any vsd_session=<id>.
// A hit exits 3 and names the file, never the value; the temp folder is deleted in every case. nginx's log then
// shows the run kept to its sign-in POST budget (support/budget.ts); a broken budget fails a green run.
import { spawn } from 'node:child_process'
import { appendFileSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { isBrowser, playwrightArgs } from './support/args.ts'
import { budgetProblem, countLoggedSignIns, PLAN, webLogSince } from './support/budget.ts'
import {
  newPassword, PASSWORD_ENV, passwordProblem, scanOutputs, secretKind, STATE_DIR_ENV, stateCookieValues,
} from './support/secret-guard.ts'

const WEB = resolve(import.meta.dirname, '..')
const OUTPUTS = [join(WEB, 'test-results'), join(WEB, 'playwright-report')]
const SECRET_FOUND = 3

const browser = process.argv[2]
if (!isBrowser(browser) || process.argv.length > 3) {
  console.error('usage: node e2e/run.ts <chromium|firefox>')
  process.exit(2)
}

const given = process.env[PASSWORD_ENV]
const problem = given === undefined ? null : passwordProblem(given)
if (problem !== null) {
  console.error(`${PASSWORD_ENV} cannot be used: ${problem}.`)
  process.exit(2)
}
const password = given ?? newPassword()
const stateDir = mkdtempSync(join(tmpdir(), 'vsd-e2e-'))
const cleanUp = () => rmSync(stateDir, { recursive: true, force: true })
for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    cleanUp()
    process.exit(130)
  })
}

let leaked = false
// Cookie values are only known once the setup test has saved the state; re-read them for each output line.
const secrets = () => ({ password, cookieValues: safeCookieValues() })
function safeCookieValues(): string[] {
  try {
    return stateCookieValues(stateDir)
  } catch {
    return []
  }
}

function relay(stream: NodeJS.ReadableStream, out: NodeJS.WriteStream): Promise<void> {
  return new Promise((done) => {
    const lines = createInterface({ input: stream, crlfDelay: Infinity })
    lines.on('line', (line) => {
      const kind = secretKind(line, secrets())
      if (kind !== null) leaked = true
      out.write(kind === null ? `${line}\n` : `[redacted: a line held a ${kind}]\n`)
    })
    lines.on('close', done)
  })
}

// nginx's log is read from here on (whole seconds, as Compose's --since); global-setup restarts nginx after it.
const startedAt = new Date(Math.floor(Date.now() / 1000) * 1000).toISOString()

let code: number | null
try {
  const child = spawn(join(WEB, 'node_modules', '.bin', 'playwright'), playwrightArgs(browser), {
    cwd: WEB,
    env: { ...process.env, [PASSWORD_ENV]: password, [STATE_DIR_ENV]: stateDir },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const relayed = Promise.all([relay(child.stdout, process.stdout), relay(child.stderr, process.stderr)])
  code = await new Promise<number | null>((done) => child.on('close', done))
  await relayed

  // The sign-in budget: exactly the planned POSTs and 429s on a green run, at most that on a failed one.
  let budget: string | null
  try {
    const count = countLoggedSignIns(await webLogSince(startedAt))
    console.log(`sign-in POSTs through nginx: ${count.posts} (planned ${PLAN.posts}), 429: ${count.limited} (planned ${PLAN.limited})`)
    budget = budgetProblem(count, code === 0)
  } catch (error) {
    budget = `nginx's log could not be read: ${(error as Error).message}`
  }
  if (budget !== null) {
    console.error(`sign-in budget: ${budget}`)
    if (code === 0) code = 1
  }

  const scan = scanOutputs(OUTPUTS, secrets())
  for (const hit of scan.hits) console.error(`secret scan: ${hit.kind} found in ${hit.file}`)
  if (leaked) console.error('secret scan: Playwright output held a secret (redacted above)')
  const clean = scan.hits.length === 0 && !leaked
  console.log(`secret scan: ${scan.hits.length} matches in ${scan.files} files${leaked ? ', 1+ in the console output' : ''}`)
  if (clean && process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, 'scan=clean\n')
  if (!clean) code = SECRET_FOUND
} finally {
  cleanUp()
}
process.exit(code ?? 1)
