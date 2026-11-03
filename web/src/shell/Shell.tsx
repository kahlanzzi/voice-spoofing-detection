// The app shell around the signed-in pages (Stage 18 PR 6a; HANDOFF §2 lines 40-41, §3 lines 82-86, §7 lines
// 173-176; prototype lines 796-815). A layout route under RequireSession, so /signin and the Checking view never
// render it and the guard stays the only access check.
//
// ≤ 860 px (useNarrow): a top bar with Menu and New analysis, and the rail becomes a drawer. Opening it moves focus
// to the first rail link; Escape or the scrim closes it and returns focus to Menu; the top bar and main are inert
// while it is open (prototype lines 804-807, 1402-1403, 1480-1483). Its open state is the location key it was
// opened at, so any navigation, Back included, closes it (prototype lines 757, 773) without an effect.
//
// New analysis goes to /analyze; clearing the current analysis (prototype line 762) comes with Analyze, PR 8. The
// rail foot holds the service status line (PR 7a), the analyst, theme and sign-out (RailFoot, PR 6b). The shell
// starts the one status check when it mounts: it mounts only once signed in, and GET /api/status needs a session
// (docs/api_design.md §4.8). No other check but S10's Check now (PR 7b); StrictMode's second run shares the request.
import { useEffect, useRef, useState, type MouseEvent } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router'
import Icon, { type IconName } from '../components/Icon'
import { checkStatus } from '../status/status'
import RailFoot from './RailFoot'
import './Shell.css'
import { useNarrow } from './useNarrow'

const NAV: readonly { to: string; label: string; icon: IconName }[] = [
  { to: '/analyze', label: 'Analyze', icon: 'upload' },
  { to: '/history', label: 'History', icon: 'history' },
  { to: '/overview', label: 'Overview', icon: 'grid' },
  { to: '/status', label: 'System status', icon: 'server' },
]

type FocusTarget = 'drawer' | 'menu'

export default function Shell() {
  const { key } = useLocation()
  const navigate = useNavigate()
  const narrow = useNarrow()
  const [openAt, setOpenAt] = useState<string | null>(null)
  // Widening the window closes the drawer for good, not just until it narrows again (state adjusted in render).
  if (!narrow && openAt !== null) setOpenAt(null)
  const open = narrow && openAt === key
  const pendingFocus = useRef<FocusTarget | null>(null)
  const menuRef = useRef<HTMLButtonElement>(null)
  const railRef = useRef<HTMLElement>(null)
  const mainRef = useRef<HTMLElement>(null)

  function close(): void {
    pendingFocus.current = 'menu'
    setOpenAt(null)
  }

  function toggle(): void {
    if (open) {
      close()
      return
    }
    pendingFocus.current = 'drawer'
    setOpenAt(key)
  }

  useEffect(() => {
    void checkStatus()
  }, [])

  // Focus once the drawer is on screen or gone (the SignIn.tsx pending-focus pattern).
  useEffect(() => {
    const target = pendingFocus.current
    if (target === null) return
    pendingFocus.current = null
    if (target === 'drawer') railRef.current?.querySelector<HTMLElement>('nav a')?.focus()
    else menuRef.current?.focus()
  })

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      pendingFocus.current = 'menu'
      setOpenAt(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open])

  // HANDOFF line 173: the skip link moves focus to the page h1.
  function skip(event: MouseEvent<HTMLAnchorElement>): void {
    event.preventDefault()
    mainRef.current?.querySelector<HTMLElement>('h1')?.focus()
  }

  const newAnalysis = (
    <button type="button" className="btn btn-primary" onClick={() => void navigate('/analyze')}>
      <Icon name="plus" size={18} />
      New analysis
    </button>
  )

  return (
    <>
      <a className="skip" href="#main" onClick={skip}>
        Skip to main content
      </a>
      <header className="topbar" inert={open}>
        <button
          type="button"
          className="menu"
          ref={menuRef}
          aria-expanded={open}
          aria-controls="rail"
          onClick={toggle}
        >
          <Icon name="menu" />
          Menu
        </button>
        {newAnalysis}
      </header>
      {open && <div className="scrim" onClick={close} />}
      <div className="shell">
        <aside className={open ? 'rail open' : 'rail'} id="rail" aria-label="Main menu" ref={railRef}>
          <div className="brand">
            <div className="brand-mark">
              <Icon name="wave" size={20} />
            </div>
            <div>
              <div className="brand-name">Voice Check</div>
              <div className="brand-sub">Analyst console</div>
            </div>
          </div>
          {newAnalysis}
          <nav className="nav" aria-label="Pages">
            <ul>
              {NAV.map(({ to, label, icon }) => (
                <li key={to}>
                  <NavLink to={to}>
                    <Icon name={icon} />
                    <span>{label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <RailFoot />
        </aside>
        <main className="main" id="main" ref={mainRef} inert={open}>
          <div className="wrap">
            <Outlet />
          </div>
        </main>
      </div>
    </>
  )
}
