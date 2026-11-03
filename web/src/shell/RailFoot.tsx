// The rail footer (Stage 18 PR 6b; HANDOFF line 41, prototype lines 144-149 and 812-813): the analyst, the theme
// control and Sign out. Inside the drawer on mobile, since the rail is the drawer. The service status line above it
// is RailStatus (PR 7a; prototype line 812).
//
// Theme: a native <select> (prototype line 485), stored and applied through theme.ts only. "Follow the system
// setting" removes data-theme, so the CSS follows the OS live, with no listener.
//
// Sign out: DELETE /api/session through signOut(), with no confirmation (docs/api_design.md §4.8; the unsaved-note
// dialog D1 is PR 11). On success the guard sends the analyst to S1 without next. On failure they stay signed in,
// the alert is mounted again (a new key, announced each time) and focus goes back to Sign out, which disabling
// took away. Busy keeps the label, adds the spinner, sets disabled and aria-busy (HANDOFF line 137).
import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { signOut } from '../session/session'
import { useSession } from '../session/useSession'
import { applyTheme, readTheme, writeTheme, type ThemeChoice } from '../theme'
import RailStatus from './RailStatus'

// Copy proposal (Entry 137 Q6), pending the team's review.
export const SIGN_OUT_FAILED = 'Sign-out did not complete. You are still signed in. Try again.'

const THEMES: readonly { value: ThemeChoice; label: string }[] = [
  { value: 'system', label: 'Follow the system setting' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

export default function RailFoot() {
  const session = useSession()
  const [theme, setTheme] = useState(readTheme)
  const [busy, setBusy] = useState(false)
  // 0: no failure yet. Each failure mounts the alert again.
  const [failures, setFailures] = useState(0)
  const inFlight = useRef(false)
  const refocus = useRef(false)
  const buttonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!refocus.current || busy) return
    refocus.current = false
    buttonRef.current?.focus()
  })

  function chooseTheme(event: ChangeEvent<HTMLSelectElement>): void {
    const choice = event.target.value as ThemeChoice
    writeTheme(choice)
    applyTheme(choice)
    setTheme(choice)
  }

  async function signOutClick(): Promise<void> {
    if (inFlight.current) return
    inFlight.current = true
    setBusy(true)
    const done = await signOut()
    inFlight.current = false
    // Signed out: the guard has already replaced the shell.
    if (done) return
    refocus.current = true
    setFailures((count) => count + 1)
    setBusy(false)
  }

  return (
    <div className="rail-foot">
      <RailStatus />
      <div className="who">
        <div className="who-name">{session.status === 'signedIn' ? session.username : ''}</div>
        <div className="who-role">Analyst ID</div>
      </div>
      <div className="theme">
        <label htmlFor="rail-theme">Theme</label>
        <select id="rail-theme" value={theme} onChange={chooseTheme}>
          {THEMES.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <button
        type="button"
        className="btn-out"
        ref={buttonRef}
        disabled={busy}
        aria-busy={busy || undefined}
        onClick={() => void signOutClick()}
      >
        {busy && <span className="spin" aria-hidden="true" />}
        Sign out
      </button>
      {failures > 0 && (
        <p className="banner-err" role="alert" key={failures}>
          {SIGN_OUT_FAILED}
        </p>
      )}
    </div>
  )
}
