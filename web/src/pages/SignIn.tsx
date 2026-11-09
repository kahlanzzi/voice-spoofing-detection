// S1 Sign in (HANDOFF §1 line 23; prototype docs/design/analyst-console-prototype.html lines 832-856, 1436-1447).
// POST /api/session (docs/api_design.md §4.8). On 200 it only calls markSignedIn: the guard (SignInGate) then
// sends the analyst to the checked `next`, so this screen never reads `next` or navigates.
//
// The password lives only in this component's state: sent in the POST body and nowhere else, never stored or
// logged (eslint.config.js bans storage, console and timers here). It is cleared after every failed answer
// (the prototype keeps only the Analyst ID, line 1444) and kept on a field error, when nothing was sent.
// One request at a time: a ref, set before any await, plus disabled buttons.
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { apiFetch } from '../api/client'
import type { SessionInfo, SignInRequest } from '../api/types'
import Banner from '../components/Banner'
import Button from '../components/Button'
import Icon from '../components/Icon'
import TextField from '../components/TextField'
import { checkSession, markSignedIn } from '../session/session'
import './SignIn.css'
import { SIGN_IN_COPY, signInOutcome, type SignInError } from './signInOutcome'

// From the guard: the session store's state.
export type SignInVariant = 'default' | 'expired' | 'unavailable'

type Busy = 'submit' | 'retry' | null
type FocusTarget = 'username' | 'password' | 'h1' | 'afterRetry'

const NO_FIELD_ERRORS = { username: false, password: false }

export default function SignIn({ variant }: { variant: SignInVariant }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [fieldErrors, setFieldErrors] = useState(NO_FIELD_ERRORS)
  const [error, setError] = useState<SignInError | null>(null)
  // A new key re-mounts the alert, so the same message is announced again.
  const [alertKey, setAlertKey] = useState(0)
  const [expiredSeen, setExpiredSeen] = useState(false)
  const [busy, setBusy] = useState<Busy>(null)
  const inFlight = useRef(false)
  const pendingFocus = useRef<FocusTarget | null>(null)
  const h1Ref = useRef<HTMLHeadingElement>(null)
  const usernameRef = useRef<HTMLInputElement>(null)
  const passwordRef = useRef<HTMLInputElement>(null)
  const retryRef = useRef<HTMLButtonElement>(null)

  // Focus after a failed submit or a Retry, once the new state is on screen (prototype lines 1439, 1444).
  useEffect(() => {
    const target = pendingFocus.current
    if (target === null) return
    pendingFocus.current = null
    const element = {
      username: usernameRef.current,
      password: passwordRef.current,
      h1: h1Ref.current,
      // Still unavailable: back to Retry. Signed out now: the form.
      afterRetry: retryRef.current ?? usernameRef.current,
    }[target]
    element?.focus()
  })

  function fail(kind: SignInError): void {
    pendingFocus.current = kind === 'invalid' ? 'username' : 'h1'
    setError(kind)
    setAlertKey((key) => key + 1)
    setPassword('')
    setShowPassword(false)
    setExpiredSeen(true)
  }

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    if (inFlight.current) return
    const typed = username.trim()
    const missing = { username: typed === '', password: password === '' }
    setError(null)
    setFieldErrors(missing)
    if (missing.username || missing.password) {
      pendingFocus.current = missing.username ? 'username' : 'password'
      return
    }
    inFlight.current = true
    setBusy('submit')
    try {
      const body: SignInRequest = { username: typed, password }
      const info = await apiFetch<SessionInfo>('/api/session', { method: 'POST', json: body })
      // The server's username (lower case, §4.8), not the typed text. A 200 without one is not a sign-in.
      if (typeof info?.username === 'string' && info.username !== '') markSignedIn(info.username)
      else fail('unavailable')
    } catch (failure) {
      fail(signInOutcome(failure))
    } finally {
      inFlight.current = false
      setBusy(null)
    }
  }

  // Only for the store's unavailable state (the check on load failed). The store stays unavailable until
  // the answer, so the button shows its own loading; a signed-in answer lets the guard navigate.
  async function retry(): Promise<void> {
    if (inFlight.current) return
    inFlight.current = true
    setBusy('retry')
    await checkSession()
    pendingFocus.current = 'afterRetry'
    inFlight.current = false
    setBusy(null)
    setAlertKey((key) => key + 1)
  }

  const banner = error ?? (variant === 'unavailable' ? 'unavailable' : null)
  const showRetry = error === null && variant === 'unavailable'
  const invalid = error === 'invalid'

  return (
    <main className="signin" id="main">
      <form className="signin-card" method="post" noValidate aria-labelledby="si-h" onSubmit={(event) => void submit(event)}>
        <div className="signin-brand">
          <div className="brand-mark">
            <Icon name="wave" size={20} />
          </div>
          <div>
            <div className="brand-name">Voice Check</div>
            <div className="signin-sub">Analyst console</div>
          </div>
        </div>
        <h1 id="si-h" tabIndex={-1} ref={h1Ref}>
          Sign in to the analyst console
        </h1>
        {variant === 'expired' && !expiredSeen && (
          <Banner tone="info" icon="clock" role="status">
            Your session expired. Sign in again to continue where you left off.
          </Banner>
        )}
        {banner !== null && (
          <Banner tone="err" icon={banner === 'locked' || banner === 'denied' ? 'lock' : 'alert'} role="alert" id="si-banner" key={alertKey}>
            {SIGN_IN_COPY[banner]}
          </Banner>
        )}
        {showRetry && (
          <Button variant="secondary" ref={retryRef} onClick={() => void retry()} disabled={busy !== null} busy={busy === 'retry'}>
            {busy === 'retry' ? 'Checking…' : 'Retry'}
          </Button>
        )}
        <TextField
          id="si-user"
          label="Analyst ID"
          name="username"
          autoComplete="username"
          ref={usernameRef}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          error={fieldErrors.username ? 'Enter your analyst ID.' : null}
          invalid={invalid}
          describedBy={invalid ? 'si-banner' : undefined}
        />
        <TextField
          id="si-pass"
          label="Password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          autoComplete="current-password"
          ref={passwordRef}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          error={fieldErrors.password ? 'Enter your password.' : null}
          invalid={invalid}
          describedBy={invalid ? 'si-banner' : undefined}
          action={
            <button
              type="button"
              aria-pressed={showPassword}
              aria-controls="si-pass"
              onClick={() => setShowPassword((shown) => !shown)}
            >
              {showPassword ? 'Hide' : 'Show'} <span className="sr">password</span>
            </button>
          }
        />
        <Button type="submit" variant="primary" disabled={busy !== null} busy={busy === 'submit'}>
          {busy === 'submit' ? 'Signing in…' : 'Sign in'}
        </Button>
        <p className="signin-help">Forgot your password? Ask your administrator.</p>
        <p className="sr" role="status">
          {busy === 'submit' ? 'Signing in…' : busy === 'retry' ? 'Checking…' : ''}
        </p>
      </form>
    </main>
  )
}
