// After the run, even after failures: every e2e account ends enabled-then-disabled and unlocked, so none keeps a
// session or a usable password between runs (disable ends its sessions). nginx's sign-in POST count is checked
// by run.ts, which knows whether the run passed (support/budget.ts).
import { ACCOUNTS } from './support/secret-guard.ts'
import { mustAdmin } from './support/stack.ts'

export default async function globalTeardown(): Promise<void> {
  for (const username of Object.values(ACCOUNTS)) {
    await mustAdmin('enable', username)
    await mustAdmin('unlock', username)
    await mustAdmin('disable', username)
  }
}
